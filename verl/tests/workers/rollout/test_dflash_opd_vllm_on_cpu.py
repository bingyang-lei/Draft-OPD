# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for the vLLM DFlash OPD helpers (no engine required)."""

import math

from verl.utils.vllm.dflash_opd_patch import DFlashOpdMetadataRegistry
from verl.workers.rollout.vllm_rollout.opd_utils import (
    build_dflash_extra_fields,
    iter_opd_draft_weights,
    map_opd_draft_weight_name,
)


def test_map_opd_draft_weight_name_strips_wrapper_prefixes():
    assert map_opd_draft_weight_name("draft_model.layers.0.self_attn.q_proj.weight") == (
        "layers.0.self_attn.q_proj.weight"
    )
    assert map_opd_draft_weight_name("_fsdp_wrapped_module.draft_model.fc.weight") == "fc.weight"
    assert map_opd_draft_weight_name("module._fsdp_wrapped_module.draft_model.hidden_norm.weight") == (
        "hidden_norm.weight"
    )


def test_map_opd_draft_weight_name_handles_nested_marker():
    assert map_opd_draft_weight_name("student.draft_model.layers.1.mlp.gate_proj.weight") == (
        "layers.1.mlp.gate_proj.weight"
    )


def test_map_opd_draft_weight_name_rejects_target_weights():
    assert map_opd_draft_weight_name("main_model.model.layers.0.self_attn.q_proj.weight") is None
    assert map_opd_draft_weight_name("model.layers.0.self_attn.q_proj.weight") is None
    assert map_opd_draft_weight_name("lm_head.weight") is None


def test_iter_opd_draft_weights_filters_and_renames():
    weights = [
        ("draft_model.fc.weight", "w_fc"),
        ("main_model.model.embed_tokens.weight", "w_embed"),
        ("draft_model.layers.0.self_attn.q_proj.weight", "w_q"),
    ]
    mapped = list(iter_opd_draft_weights(iter(weights)))
    assert mapped == [("fc.weight", "w_fc"), ("layers.0.self_attn.q_proj.weight", "w_q")]


def test_registry_pop_replays_response_positions():
    registry = DFlashOpdMetadataRegistry()
    # Step 1: draft block of 4, 2 accepted then a rejection at response pos 3.
    registry.record_step("req-1", 2, 4, rejected_token_id=111, teacher_logprob=-0.5)
    # Step 2: draft block of 4, all accepted (bonus token, no rejection).
    registry.record_step("req-1", 4, 4, rejected_token_id=-1, teacher_logprob=float("nan"))
    # Step 3: draft block of 4, 0 accepted, immediate rejection.
    registry.record_step("req-1", 0, 4, rejected_token_id=222, teacher_logprob=-1.5)

    metadata = registry.pop("req-1")
    assert metadata is not None
    # Response starts with the prefill token at position 0.
    # step1: positions 1,2 accepted; rejection at 3; recovered token at 3.
    # step2: positions 4-7 accepted; bonus at 8.
    # step3: rejection at 9.
    assert metadata["rejected_draft_anchor_indices"] == [3, 9]
    assert metadata["rejected_draft_offsets"] == [3, 1]
    assert metadata["rejected_draft_token_ids"] == [111, 222]
    assert metadata["rejected_draft_teacher_logprobs"] == [-0.5, -1.5]

    # Second pop returns None (entry was consumed).
    assert registry.pop("req-1") is None


def test_registry_pop_missing_request_returns_none():
    assert DFlashOpdMetadataRegistry().pop("nope") is None


def test_build_dflash_extra_fields_matches_sglang_contract():
    metadata = {
        "rejected_draft_anchor_indices": [3, 9],
        "rejected_draft_offsets": [3, 1],
        "rejected_draft_token_ids": [111, 222],
        "rejected_draft_teacher_logprobs": [-0.5, -1.5],
    }
    fields = build_dflash_extra_fields(metadata, response_len=10)
    assert fields["dflash_reject_token_indices"] == [3, 9]
    assert fields["dflash_rejected_draft_anchor_indices"] == [3, 9]
    assert fields["dflash_rejected_draft_offsets"] == [3, 1]
    assert fields["dflash_rejected_draft_token_ids"] == [111, 222]
    assert fields["dflash_rejected_draft_teacher_logprobs"] == [-0.5, -1.5]
    assert fields["dflash_reject_token_count"] == 2
    assert fields["dflash_non_reject_token_count"] == 8
    assert fields["dflash_empty_reject_token_indices"] == 0


def test_build_dflash_extra_fields_drops_out_of_range_anchors():
    metadata = {
        "rejected_draft_anchor_indices": [3, 12],
        "rejected_draft_offsets": [3, 1],
        "rejected_draft_token_ids": [111, 222],
        "rejected_draft_teacher_logprobs": [-0.5, -1.5],
    }
    fields = build_dflash_extra_fields(metadata, response_len=10)
    assert fields["dflash_reject_token_indices"] == [3]
    assert fields["dflash_rejected_draft_token_ids"] == [111]
    assert fields["dflash_rejected_draft_teacher_logprobs"] == [-0.5]


def test_build_dflash_extra_fields_none_and_empty():
    assert build_dflash_extra_fields(None, response_len=10) == {}
    empty = {
        "rejected_draft_anchor_indices": [],
        "rejected_draft_offsets": [],
        "rejected_draft_token_ids": [],
        "rejected_draft_teacher_logprobs": [],
    }
    fields = build_dflash_extra_fields(empty, response_len=5)
    assert fields["dflash_reject_token_indices"] == []
    assert fields["dflash_reject_token_count"] == 0
    assert fields["dflash_empty_reject_token_indices"] == 1


def test_build_dflash_extra_fields_length_mismatch_raises():
    bad = {
        "rejected_draft_anchor_indices": [3],
        "rejected_draft_offsets": [3, 1],
        "rejected_draft_token_ids": [111],
        "rejected_draft_teacher_logprobs": [-0.5],
    }
    try:
        build_dflash_extra_fields(bad, response_len=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError for inconsistent metadata lengths")


def test_registry_teacher_logprob_nan_passthrough():
    registry = DFlashOpdMetadataRegistry()
    registry.record_step("req-2", 1, 2, rejected_token_id=42, teacher_logprob=float("nan"))
    metadata = registry.pop("req-2")
    assert math.isnan(metadata["rejected_draft_teacher_logprobs"][0])


def test_empty_dflash_extra_fields_sglang_parity():
    from verl.workers.rollout.vllm_rollout.opd_utils import empty_dflash_extra_fields

    fields = empty_dflash_extra_fields(response_len=7)
    assert fields["dflash_reject_token_indices"] == []
    assert fields["dflash_rejected_draft_anchor_indices"] == []
    assert fields["dflash_rejected_draft_offsets"] == []
    assert fields["dflash_rejected_draft_token_ids"] == []
    assert fields["dflash_rejected_draft_teacher_logprobs"] == []
    assert fields["dflash_reject_token_count"] == 0
    assert fields["dflash_non_reject_token_count"] == 7
    assert fields["dflash_empty_reject_token_indices"] == 1

    zero = empty_dflash_extra_fields(response_len=0)
    assert zero["dflash_non_reject_token_count"] == 0
    assert zero["dflash_empty_reject_token_indices"] == 0
