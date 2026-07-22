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

import asyncio
import math
from types import SimpleNamespace

import pytest

from verl.utils.vllm.dflash_opd_patch import DFlashOpdMetadataRegistry, compute_dflash_num_accepted
from verl.workers.rollout.vllm_rollout.opd_utils import (
    build_dflash_extra_fields,
    iter_opd_draft_weights,
    map_opd_draft_weight_name,
)

# PLACEHOLDER_TOKEN_ID in vLLM v1 is -1; use a sentinel here to avoid importing vllm.
_PH = -1


def test_num_accepted_stock_placeholder_full_accept():
    # 4 drafts all accepted + bonus, rest padded -> num_accepted == n_draft.
    draft = [10, 11, 12, 13]
    sampled = [10, 11, 12, 13, 99, _PH, _PH]  # accepted prefix + bonus, then padding
    n_valid = 5  # 4 accepted + 1 bonus
    assert compute_dflash_num_accepted(n_valid=n_valid, n_draft=4, sampled_row=sampled, draft_row=draft) == 4


def test_num_accepted_stock_placeholder_rejection():
    # Reject at position 2: sampled[2] is the recovered target token, != draft[2].
    draft = [10, 11, 12, 13]
    sampled = [10, 11, 55, _PH, _PH, _PH, _PH]
    n_valid = 3  # 2 accepted + 1 recovered
    assert compute_dflash_num_accepted(n_valid=n_valid, n_draft=4, sampled_row=sampled, draft_row=draft) == 2


def test_num_accepted_ascend_no_placeholder_padding():
    # vllm-ascend style: row is fully filled (no PLACEHOLDER padding), so the
    # placeholder count reports full acceptance, but the matching prefix detects
    # the real rejection at position 2.
    draft = [10, 11, 12, 13]
    sampled = [10, 11, 55, 77, 88]  # no placeholders anywhere
    n_valid = 5  # placeholder count would say "all 4 accepted"
    assert compute_dflash_num_accepted(n_valid=n_valid, n_draft=4, sampled_row=sampled, draft_row=draft) == 2


def test_num_accepted_reject_at_first_position():
    draft = [10, 11, 12, 13]
    sampled = [99, 88, 77, 66, 55]
    assert compute_dflash_num_accepted(n_valid=5, n_draft=4, sampled_row=sampled, draft_row=draft) == 0


def test_num_accepted_no_speculation_is_zero():
    assert compute_dflash_num_accepted(n_valid=1, n_draft=0, sampled_row=[42], draft_row=[]) == 0


class _FakeEngine:
    """Records the sleep level requested via collective_rpc."""

    def __init__(self):
        self.sleep_level = None

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        if method == "sleep":
            self.sleep_level = (kwargs or {}).get("level")

    async def reset_encoder_cache(self):
        pass


def _make_hybrid_server(*, is_composed_dflash_student: bool, lora_rank: int = 0, lora_merge: bool = False):
    """Build a bare vLLMHttpServer for exercising ``_sleep_hybrid`` only.

    Bypasses ``__init__`` (which spins up config/ray plumbing) and injects the
    handful of attributes ``_sleep_hybrid`` reads.
    """
    vllm_async_server = pytest.importorskip("verl.workers.rollout.vllm_rollout.vllm_async_server")
    server = vllm_async_server.vLLMHttpServer.__new__(vllm_async_server.vLLMHttpServer)
    server.model_config = SimpleNamespace(lora_rank=lora_rank, lora={"rank": lora_rank, "merge": lora_merge})
    server.is_composed_dflash_student = is_composed_dflash_student
    server.engine = _FakeEngine()
    return server


def test_sleep_hybrid_uses_level1_for_composed_dflash_student():
    """The frozen target is never re-synced after wake, so it must survive sleep."""
    server = _make_hybrid_server(is_composed_dflash_student=True)
    asyncio.run(server._sleep_hybrid())
    assert server.engine.sleep_level == 1


def test_sleep_hybrid_uses_level1_for_lora_adapter():
    server = _make_hybrid_server(is_composed_dflash_student=False, lora_rank=8)
    asyncio.run(server._sleep_hybrid())
    assert server.engine.sleep_level == 1


def test_sleep_hybrid_uses_level2_for_plain_full_weight_engine():
    server = _make_hybrid_server(is_composed_dflash_student=False)
    asyncio.run(server._sleep_hybrid())
    assert server.engine.sleep_level == 2


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


def test_registry_pop_matches_engine_suffixed_request_id():
    # The engine records under input_batch.req_ids, which vLLM suffixes as
    # "<base>-<tag>"; the frontend pops with the base id it passed to generate().
    registry = DFlashOpdMetadataRegistry()
    base = "17edb0fe1b494949b81adda2a96ab123"
    engine_key = f"{base}-b50b54a4"
    registry.record_step(engine_key, 1, 7, rejected_token_id=1438, teacher_logprob=-0.5)

    metadata = registry.pop(base)
    assert metadata is not None
    assert metadata["rejected_draft_anchor_indices"] == [2]
    assert metadata["rejected_draft_token_ids"] == [1438]
    # Consumed.
    assert registry.pop(base) is None


def test_registry_pop_prefers_exact_key_over_suffix_match():
    registry = DFlashOpdMetadataRegistry()
    registry.record_step("abc", 0, 7, rejected_token_id=10, teacher_logprob=-0.1)
    registry.record_step("abc-xyz", 0, 7, rejected_token_id=20, teacher_logprob=-0.2)

    exact = registry.pop("abc")
    assert exact is not None
    assert exact["rejected_draft_token_ids"] == [10]
    # The suffixed entry is still retrievable afterwards.
    suffixed = registry.pop("abc")
    assert suffixed is not None
    assert suffixed["rejected_draft_token_ids"] == [20]


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
