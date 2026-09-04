import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from transformers import PretrainedConfig

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker
from verl.models.transformers.dflash_student import ComposedDFlashStudentForCausalLM, full_vocab_tv_distance
from verl.models.transformers.eagle3_student import ComposedEagle3StudentForCausalLM, Eagle3DraftModel
from verl.protocol import DataProto
from verl.trainer.distillation import requires_external_teacher, resolve_teacher_logprob_source
from verl.trainer.distillation.losses import (
    distillation_loss,
    get_effective_distillation_response_mask,
    get_rejected_draft_distillation_stream,
)
from verl.trainer.ppo.core_algos import kl_penalty
from verl.utils import tensordict_utils as tu
from verl.workers.config import DistillationConfig, DistillationLossConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.rollout.sglang_rollout.utils import align_dflash_reject_token_mask


def _single_sequence_logprobs(values):
    values = torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1)
    return torch.nested.as_nested_tensor([values], layout=torch.jagged)


def _object_array(*values):
    array = np.empty(len(values), dtype=object)
    array[:] = list(values)
    return array


def _local_forward_kl(student_log_probs, teacher_log_probs):
    student_log_probs = torch.as_tensor(student_log_probs, dtype=torch.float32)
    teacher_log_probs = torch.as_tensor(teacher_log_probs, dtype=torch.float32)
    student_probs = student_log_probs.exp()
    teacher_probs = teacher_log_probs.exp()
    return teacher_probs * (teacher_log_probs - student_log_probs) + (1.0 - teacher_probs) * (
        torch.log1p(-teacher_probs) - torch.log1p(-student_probs)
    )


def _teacher_source_config(
    *,
    teacher_path="/models/main",
    composed=True,
    source="auto",
    use_tv_loss=False,
    use_policy_gradient=False,
    loss_mode="k3",
):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {
                    "path": "/models/main",
                    "override_config": {
                        "verl_composed_dflash_student": composed,
                        "verl_dflash_main_model_path": "/models/main",
                    },
                }
            },
            "distillation": {
                "enabled": True,
                "teacher_logprob_source": source,
                "teacher_models": {"teacher_model": {"model_path": teacher_path}},
                "distillation_loss": {
                    "loss_mode": loss_mode,
                    "use_tv_loss": use_tv_loss,
                    "use_policy_gradient": use_policy_gradient,
                },
            },
        }
    )


def test_teacher_source_auto_uses_composed_main_only_for_matching_dflash_teacher():
    config = _teacher_source_config()

    assert resolve_teacher_logprob_source(config) == "composed_main"
    assert config.distillation.teacher_logprob_source == "composed_main"
    assert not requires_external_teacher(config)

    different_teacher = _teacher_source_config(teacher_path="/models/different")
    assert resolve_teacher_logprob_source(different_teacher) == "server"
    assert requires_external_teacher(different_teacher)

    non_composed = _teacher_source_config(composed=False)
    assert resolve_teacher_logprob_source(non_composed) == "server"
    assert requires_external_teacher(non_composed)


def test_composed_main_teacher_source_rejects_unsupported_training_modes():
    with pytest.raises(NotImplementedError, match="direct supervised"):
        resolve_teacher_logprob_source(_teacher_source_config(use_policy_gradient=True))

    with pytest.raises(NotImplementedError, match="scalar sampled"):
        resolve_teacher_logprob_source(_teacher_source_config(loss_mode="forward_kl_topk"))

    config = DistillationConfig(
        enabled=True,
        teacher_logprob_source="composed_main",
        distillation_loss=DistillationLossConfig(use_policy_gradient=False, loss_mode="k3"),
    )
    assert config.teacher_models == {}


def test_agent_loop_skips_post_rollout_teacher_scoring_for_composed_main():
    worker = object.__new__(AgentLoopWorker)
    worker.use_external_teacher = False
    output = SimpleNamespace(extra_fields={})

    asyncio.run(
        worker._compute_teacher_logprobs(
            output,
            prompt_ids=[1, 2],
            response_ids=[3, 4],
            validate=False,
        )
    )

    assert output.extra_fields == {}


def test_composed_main_hidden_forward_skips_causal_lm_head():
    class _Decoder:
        def __init__(self):
            self.call_count = 0

        def __call__(self, **kwargs):
            self.call_count += 1
            return SimpleNamespace(hidden_states=(kwargs["input_ids"].float().unsqueeze(-1),))

    class _CausalLM:
        def __init__(self):
            self.decoder = _Decoder()
            self.call_count = 0

        def get_decoder(self):
            return self.decoder

        def __call__(self, **kwargs):
            self.call_count += 1
            raise AssertionError("The trajectory-wide CausalLM head must not run.")

    student = object.__new__(ComposedDFlashStudentForCausalLM)
    student.main_model = _CausalLM()
    output = student._forward_main_hidden_only(input_ids=torch.tensor([[1, 2]]), output_hidden_states=True)

    assert output.hidden_states[-1].tolist() == [[[1.0], [2.0]]]
    assert student.main_model.decoder.call_count == 1
    assert student.main_model.call_count == 0


class _CaseLogTokenizer:
    _pieces = {
        1: "A",
        2: "B",
        3: "C",
        4: "D",
        5: "E",
        99: "<prompt>",
        101: "x",
        102: "<eos>",
    }

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._pieces[int(token_id)] for token_id in ids)


def test_align_dflash_reject_token_mask_adds_prefill_token():
    assert align_dflash_reject_token_mask([False, True, False], response_len=4) == [
        False,
        False,
        True,
        False,
    ]
    assert align_dflash_reject_token_mask([False, True, False, True], response_len=4) == [
        False,
        True,
        False,
        True,
    ]
    assert align_dflash_reject_token_mask([True], response_len=4) == [False, True, False, False]


def test_case_log_reconstructs_draft_blocks_with_accepted_prefixes():
    batch = DataProto(
        batch=TensorDict(
            {
                "prompts": torch.tensor([[99]], dtype=torch.long),
                "responses": torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long),
            },
            batch_size=[1],
        ),
        non_tensor_batch={
            "dflash_reject_token_indices": _object_array([2, 4]),
            "dflash_rejected_draft_anchor_indices": _object_array([0, 0]),
            "dflash_rejected_draft_offsets": _object_array([2, 3]),
            "dflash_rejected_draft_token_ids": _object_array([101, 102]),
        },
    )
    worker = object.__new__(AgentLoopWorker)
    worker.tokenizer = _CaseLogTokenizer()

    text = worker._build_draft_token_case_text(
        batch=batch,
        sample_idx=0,
        response_ids=[1, 2, 3, 4, 5],
    )
    record = worker._build_case_log_record(batch=batch, sample_idx=0)

    assert text == "AB[{B}x<eos>]CD[{D}]E"
    assert record == {"response_text_with_draft_token": "AB[{B}x<eos>]CD[{D}]E"}


def test_case_log_only_writes_on_update_accumulation_boundaries():
    assert AgentLoopManager._should_write_case_log_step(0, 25)
    assert not AgentLoopManager._should_write_case_log_step(24, 25)
    assert AgentLoopManager._should_write_case_log_step(25, 25)
    assert AgentLoopManager._should_write_case_log_step(torch.tensor(50), "25")
    assert not AgentLoopManager._should_write_case_log_step(None, 25)
    assert AgentLoopManager._should_write_case_log_step(3, 1)


def test_prepare_composed_dflash_inputs_preserves_true_lengths_after_no_padding():
    prompts = torch.tensor([[0, 0, 11, 12, 13], [21, 22, 23, 24, 25]], dtype=torch.long)
    responses = torch.tensor([[31, 32, 0, 0], [41, 42, 43, 0]], dtype=torch.long)
    padded_input_ids = torch.cat([prompts, responses], dim=1)
    attention_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.long
    )
    padded_position_ids = torch.arange(padded_input_ids.shape[1], dtype=torch.long).unsqueeze(0).expand_as(
        padded_input_ids
    )
    response_mask = attention_mask[:, prompts.shape[1] :]
    input_ids = torch.nested.as_nested_tensor(
        [padded_input_ids[i, attention_mask[i].bool()] for i in range(2)], layout=torch.jagged
    )
    position_ids = torch.nested.as_nested_tensor(
        [padded_position_ids[i, attention_mask[i].bool()] for i in range(2)], layout=torch.jagged
    )

    batch = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(
        batch,
        dflash_reject_token_indices=[[1], [2]],
        opd_use_composed_teacher_logprobs=True,
    )
    tu.assign_non_tensor(
        batch,
        dflash_rejected_draft_anchor_indices=[[0, 1], []],
        dflash_rejected_draft_offsets=[[2, 3], []],
        dflash_rejected_draft_token_ids=[[101, 102], []],
        dflash_rejected_draft_teacher_logprobs=[[-0.1, -0.2], []],
    )

    engine = object.__new__(FSDPEngineWithLMHead)
    model_inputs, output_args = engine._prepare_composed_dflash_inputs(batch)

    assert output_args == {"dflash_opd": True}
    assert model_inputs["dflash_prompt_lengths"].tolist() == [3, 5]
    assert model_inputs["dflash_response_lengths"].tolist() == [2, 3]
    assert model_inputs["input_ids"].tolist() == [
        [11, 12, 13, 31, 32, 0, 0, 0],
        [21, 22, 23, 24, 25, 41, 42, 43],
    ]
    assert model_inputs["attention_mask"].tolist() == [
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ]
    assert model_inputs["dflash_rejected_draft_anchor_indices"].tolist() == [[0, 1], [-2, -2]]
    assert model_inputs["dflash_rejected_draft_offsets"].tolist() == [[2, 3], [-1, -1]]
    assert model_inputs["dflash_rejected_draft_token_ids"].tolist() == [[101, 102], [-1, -1]]
    assert torch.allclose(
        model_inputs["dflash_rejected_draft_teacher_logprobs"],
        torch.tensor([[-0.1, -0.2], [0.0, 0.0]], dtype=torch.float32),
    )
    assert model_inputs["dflash_rejected_draft_mask"].tolist() == [[True, True], [False, False]]
    assert model_inputs["dflash_use_composed_teacher_logprobs"] is True


def test_dflash_anchor_plan_skips_empty_rejects_and_uses_block_size_as_total_length():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    input_ids = torch.arange(25, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    _, segment_lens, _, block_keep_mask, _, metrics = student._build_opd_anchor_plan(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=torch.tensor([5]),
        response_lengths=torch.tensor([20]),
        reject_token_indices=torch.tensor([[2]]),
        draft_block_size=16,
    )
    assert metrics["valid_anchor_count"] == 2
    assert int(segment_lens[block_keep_mask].max().item()) == 15

    _, _, _, empty_keep_mask, _, empty_metrics = student._build_opd_anchor_plan(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=torch.tensor([5]),
        response_lengths=torch.tensor([20]),
        reject_token_indices=torch.tensor([[-1]]),
        draft_block_size=16,
    )
    assert not bool(empty_keep_mask.any())
    assert empty_metrics["empty_reject_sample_count"] == 1
    assert empty_metrics["skipped_sample_count"] == 1


def test_dflash_anchor_plan_can_stride_response_anchors():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    input_ids = torch.arange(12, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    _, _, _, block_keep_mask, _, metrics = student._build_opd_anchor_plan(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=torch.tensor([2]),
        response_lengths=torch.tensor([10]),
        reject_token_indices=torch.tensor([[1, 2, 3, 4]]),
        draft_block_size=4,
        response_anchor_stride=2,
    )

    assert int(block_keep_mask.sum().item()) == 3
    assert metrics["valid_anchor_count"] == 3


def test_random_response_anchor_plan_preserves_count_and_is_seeded():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    input_ids = torch.arange(12, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    baseline_positions, baseline_segment_lens, _, baseline_keep_mask, _, baseline_metrics = (
        student._build_opd_anchor_plan(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=torch.tensor([2]),
            response_lengths=torch.tensor([10]),
            reject_token_indices=torch.tensor([[1, 3, 6]]),
            draft_block_size=6,
        )
    )
    random_positions_a, random_segment_lens_a, _, random_keep_mask_a, _, random_metrics_a = (
        student._build_opd_anchor_plan(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=torch.tensor([2]),
            response_lengths=torch.tensor([10]),
            reject_token_indices=torch.tensor([[1, 3, 6]]),
            draft_block_size=6,
            random_response_anchor_enabled=True,
            random_response_anchor_seed=7,
        )
    )
    random_positions_b, random_segment_lens_b, _, random_keep_mask_b, _, random_metrics_b = (
        student._build_opd_anchor_plan(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=torch.tensor([2]),
            response_lengths=torch.tensor([10]),
            reject_token_indices=torch.tensor([[1, 3, 6]]),
            draft_block_size=6,
            random_response_anchor_enabled=True,
            random_response_anchor_seed=7,
        )
    )

    assert int(random_keep_mask_a.sum().item()) == int(baseline_keep_mask.sum().item())
    assert random_metrics_a["valid_anchor_count"] == baseline_metrics["valid_anchor_count"]
    assert random_metrics_a["total_reject_count"] == baseline_metrics["total_reject_count"]
    assert torch.equal(random_positions_a, random_positions_b)
    assert torch.equal(random_segment_lens_a, random_segment_lens_b)
    assert torch.equal(random_keep_mask_a, random_keep_mask_b)
    assert random_metrics_a == random_metrics_b
    baseline_lens = baseline_segment_lens[baseline_keep_mask]
    random_lens = random_segment_lens_a[random_keep_mask_a]
    assert int(random_lens.sum().item()) == int(baseline_lens.sum().item())
    assert torch.equal(torch.sort(random_lens).values, torch.sort(baseline_lens).values)
    assert int(random_segment_lens_a[random_keep_mask_a].min().item()) > 0
    assert int(random_segment_lens_a[random_keep_mask_a].max().item()) <= 5
    covered_response_indices = set()
    for full_anchor, segment_len in zip(
        random_positions_a[random_keep_mask_a].tolist(),
        random_lens.tolist(),
        strict=False,
    ):
        for offset in range(1, segment_len + 1):
            covered_response_indices.add(full_anchor + offset - 2)
    assert len(covered_response_indices) == int(random_lens.sum().item())

    found_different_seed = False
    found_different_from_baseline = not torch.equal(
        random_positions_a[random_keep_mask_a],
        baseline_positions[baseline_keep_mask],
    )
    for seed in range(8, 32):
        random_positions_c, _, _, random_keep_mask_c, _, _ = student._build_opd_anchor_plan(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=torch.tensor([2]),
            response_lengths=torch.tensor([10]),
            reject_token_indices=torch.tensor([[1, 3, 6]]),
            draft_block_size=6,
            random_response_anchor_enabled=True,
            random_response_anchor_seed=seed,
        )
        if not torch.equal(
            random_positions_a[random_keep_mask_a],
            random_positions_c[random_keep_mask_c],
        ):
            found_different_seed = True
        if not torch.equal(
            random_positions_c[random_keep_mask_c],
            baseline_positions[baseline_keep_mask],
        ):
            found_different_from_baseline = True
        if found_different_seed and found_different_from_baseline:
            break
    assert found_different_seed
    assert found_different_from_baseline


def test_random_response_anchor_plan_keeps_rejected_draft_zero_length_anchor():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    input_ids = torch.arange(8, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    anchor_positions, segment_lens, _, block_keep_mask, _, metrics = student._build_opd_anchor_plan(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=torch.tensor([2]),
        response_lengths=torch.tensor([6]),
        reject_token_indices=torch.tensor([[-1]]),
        draft_block_size=6,
        random_response_anchor_enabled=True,
        random_response_anchor_seed=7,
        rejected_draft_anchor_indices=torch.tensor([[2]]),
        rejected_draft_offsets=torch.tensor([[3]]),
        rejected_draft_mask=torch.tensor([[True]]),
    )

    assert int(block_keep_mask.sum().item()) == 1
    assert anchor_positions[0, 0].item() == 4
    assert segment_lens[0, 0].item() == 0
    assert metrics["valid_anchor_count"] == 0


def test_random_response_anchor_plan_can_skip_rejected_draft_zero_length_anchor():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    input_ids = torch.arange(8, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    anchor_positions, segment_lens, _, block_keep_mask, _, metrics = student._build_opd_anchor_plan(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=torch.tensor([2]),
        response_lengths=torch.tensor([6]),
        reject_token_indices=torch.tensor([[-1]]),
        draft_block_size=6,
        random_response_anchor_enabled=True,
        random_response_anchor_seed=7,
        rejected_draft_anchor_indices=torch.tensor([[2]]),
        rejected_draft_offsets=torch.tensor([[3]]),
        rejected_draft_mask=torch.tensor([[True]]),
        include_rejected_draft_anchors=False,
    )

    assert not bool(block_keep_mask.any())
    assert anchor_positions.shape == (1, 1)
    assert segment_lens.shape == (1, 1)
    assert metrics["valid_anchor_count"] == 0


def test_rejected_draft_anchor_plan_deduplicates_and_respects_token_limit():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    input_ids = torch.arange(12, dtype=torch.long).unsqueeze(0)

    limited_positions, limited_keep_mask = student._build_rejected_draft_anchor_plan(
        input_ids=input_ids,
        prompt_lengths=torch.tensor([2]),
        response_lengths=torch.tensor([10]),
        draft_block_size=6,
        max_tokens_per_sample=2,
        rejected_draft_anchor_indices=torch.tensor([[2, 2, 3]]),
        rejected_draft_offsets=torch.tensor([[3, 4, 2]]),
        rejected_draft_token_ids=torch.tensor([[101, 102, 103]]),
        rejected_draft_mask=torch.tensor([[True, True, True]]),
    )
    full_positions, full_keep_mask = student._build_rejected_draft_anchor_plan(
        input_ids=input_ids,
        prompt_lengths=torch.tensor([2]),
        response_lengths=torch.tensor([10]),
        draft_block_size=6,
        max_tokens_per_sample=None,
        rejected_draft_anchor_indices=torch.tensor([[2, 2, 3]]),
        rejected_draft_offsets=torch.tensor([[3, 4, 2]]),
        rejected_draft_token_ids=torch.tensor([[101, 102, 103]]),
        rejected_draft_mask=torch.tensor([[True, True, True]]),
    )

    assert limited_positions[limited_keep_mask].tolist() == [4]
    assert full_positions[full_keep_mask].tolist() == [4, 5]


def test_rejected_draft_logits_are_selected_from_anchor_and_offset():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    draft_hidden = torch.zeros(1, 4, 10, dtype=torch.float32)
    draft_hidden[0, 2, 7] = 3.0
    output_embeddings = torch.nn.Linear(10, 10, bias=False)
    with torch.no_grad():
        output_embeddings.weight.copy_(torch.eye(10))

    student_log_probs, teacher_log_probs, mask = student._collect_rejected_draft_log_probs(
        draft_hidden=draft_hidden,
        output_embeddings=output_embeddings,
        prompt_lengths=torch.tensor([2]),
        response_lengths=torch.tensor([4]),
        anchor_positions=torch.tensor([[2]]),
        block_keep_mask=torch.tensor([[True]]),
        draft_block_size=4,
        lm_head_chunk_size=2,
        max_tokens_per_sample=None,
        rejected_draft_anchor_indices=torch.tensor([[0]]),
        rejected_draft_offsets=torch.tensor([[2]]),
        rejected_draft_token_ids=torch.tensor([[7]]),
        rejected_draft_teacher_logprobs=torch.tensor([[-1.25]]),
        rejected_draft_mask=torch.tensor([[True]]),
    )

    expected = torch.log_softmax(draft_hidden[0, 2], dim=-1)[7]
    assert torch.allclose(student_log_probs, expected.view(1, 1))
    assert torch.allclose(teacher_log_probs, torch.tensor([[-1.25]]))
    assert mask.tolist() == [[True]]


def test_empty_rejected_draft_metadata_returns_empty_stream():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    output_embeddings = torch.nn.Linear(10, 10, bias=False)

    student_log_probs, teacher_log_probs, mask = student._collect_rejected_draft_log_probs(
        draft_hidden=torch.zeros(1, 4, 10, dtype=torch.float32),
        output_embeddings=output_embeddings,
        prompt_lengths=torch.tensor([2]),
        response_lengths=torch.tensor([4]),
        anchor_positions=torch.tensor([[2]]),
        block_keep_mask=torch.tensor([[True]]),
        draft_block_size=4,
        lm_head_chunk_size=2,
        max_tokens_per_sample=None,
        rejected_draft_anchor_indices=torch.tensor([[-2]]),
        rejected_draft_offsets=torch.tensor([[-1]]),
        rejected_draft_token_ids=torch.tensor([[-1]]),
        rejected_draft_teacher_logprobs=torch.tensor([[0.0]]),
        rejected_draft_mask=torch.tensor([[False]]),
    )

    assert student_log_probs.shape == (1, 1)
    assert teacher_log_probs.shape == (1, 1)
    assert mask.tolist() == [[False]]


def test_rejected_draft_max_tokens_per_sample_limits_loss_tokens():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    draft_hidden = torch.zeros(1, 4, 10, dtype=torch.float32)
    draft_hidden[0, 1, 3] = 2.0
    draft_hidden[0, 2, 4] = 2.0
    output_embeddings = torch.nn.Linear(10, 10, bias=False)
    with torch.no_grad():
        output_embeddings.weight.copy_(torch.eye(10))

    student_log_probs, _, mask = student._collect_rejected_draft_log_probs(
        draft_hidden=draft_hidden,
        output_embeddings=output_embeddings,
        prompt_lengths=torch.tensor([2]),
        response_lengths=torch.tensor([4]),
        anchor_positions=torch.tensor([[2]]),
        block_keep_mask=torch.tensor([[True]]),
        draft_block_size=4,
        lm_head_chunk_size=2,
        max_tokens_per_sample=1,
        rejected_draft_anchor_indices=torch.tensor([[0, 0]]),
        rejected_draft_offsets=torch.tensor([[1, 2]]),
        rejected_draft_token_ids=torch.tensor([[3, 4]]),
        rejected_draft_teacher_logprobs=torch.tensor([[-1.0, -2.0]]),
        rejected_draft_mask=torch.tensor([[True, True]]),
    )

    assert mask.tolist() == [[True, False]]
    assert student_log_probs[0, 0].item() != 0
    assert student_log_probs[0, 1].item() == 0


def test_selected_lm_log_probs_match_full_logits_path():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    draft_hidden = torch.randn(2, 5, 8, dtype=torch.float32)
    output_embeddings = torch.nn.Linear(8, 11, bias=False)
    batch_indices = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    draft_indices = torch.tensor([1, 3, 0, 4], dtype=torch.long)
    token_ids = torch.tensor([2, 5, 7, 1], dtype=torch.long)

    selected, entropy = student._compute_selected_lm_log_probs(
        draft_hidden=draft_hidden,
        output_embeddings=output_embeddings,
        batch_indices=batch_indices,
        draft_indices=draft_indices,
        token_ids=token_ids,
        chunk_size=2,
        calculate_entropy=True,
    )

    full_logits = output_embeddings(draft_hidden)
    full_log_probs = torch.log_softmax(full_logits.float(), dim=-1)
    expected = full_log_probs[batch_indices, draft_indices, token_ids]
    selected_full_log_probs = full_log_probs[batch_indices, draft_indices]
    expected_entropy = -(selected_full_log_probs.exp() * selected_full_log_probs).sum(dim=-1)
    assert torch.allclose(selected, expected)
    assert entropy is not None
    assert torch.allclose(entropy, expected_entropy)


def test_eagle3_target_to_draft_mapping_masks_unsupported_tokens():
    config = PretrainedConfig(
        hidden_size=4,
        intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        max_position_embeddings=16,
        rope_theta=10000,
        vocab_size=10,
        draft_vocab_size=3,
        pad_token_id=0,
    )
    draft_model = Eagle3DraftModel(config=config)
    draft_model.d2t.copy_(torch.tensor([0, 2, 4], dtype=torch.long))
    draft_model.t2d.zero_()
    draft_model.t2d[torch.tensor([0, 3, 6])] = True
    draft_model.refresh_target_to_draft()

    student = object.__new__(ComposedEagle3StudentForCausalLM)
    torch.nn.Module.__init__(student)
    student.draft_model = draft_model

    draft_ids, supported = student._map_target_to_draft_ids(torch.tensor([0, 3, 6, 5, 11]))

    assert draft_ids.tolist() == [0, 1, 2, 0, 0]
    assert supported.tolist() == [True, True, True, False, False]


def test_eagle3_native_response_loss_uses_target_distribution_and_masks_unsupported_argmax():
    class _TinyDraft:
        def __init__(self):
            self.t2d = torch.tensor([True, False, True, False, True])
            self.target_to_draft = torch.tensor([0, -1, 1, -1, 2])

        def compute_logits(self, hidden_states):
            return hidden_states

    student = object.__new__(ComposedEagle3StudentForCausalLM)
    student.draft_model = _TinyDraft()
    draft_hidden = torch.tensor([[[2.0, 1.0, -1.0], [-1.0, 0.0, 1.0]]], dtype=torch.float32)
    target_logits = torch.tensor(
        [
            [
                [0.0, -5.0, 3.0, -1.0, 1.0],
                [0.0, 5.0, 1.0, -1.0, 2.0],
            ]
        ],
        dtype=torch.float32,
    )

    scalar_log_probs, entropy, scalar_supported, native_losses, native_supported, top1_match = (
        student._compute_eagle3_native_response_outputs(
            draft_hidden=draft_hidden,
            target_logits=target_logits,
            batch_indices=torch.tensor([0, 0], dtype=torch.long),
            draft_indices=torch.tensor([0, 1], dtype=torch.long),
            target_row_indices=torch.tensor([0, 1], dtype=torch.long),
            target_token_ids=torch.tensor([2, 3], dtype=torch.long),
            chunk_size=1,
            calculate_entropy=True,
        )
    )

    draft_log_probs = torch.log_softmax(draft_hidden[0, 0], dim=-1)
    target_p = torch.softmax(target_logits[0, 0, [0, 2, 4]], dim=-1)
    expected_native = -(target_p * draft_log_probs).sum()
    assert torch.allclose(scalar_log_probs[0], draft_log_probs[1])
    assert scalar_supported.tolist() == [True, False]
    assert torch.allclose(native_losses[0], expected_native)
    assert native_losses[1].item() == 0.0
    assert native_supported.tolist() == [True, False]
    assert top1_match.tolist() == [False, False]
    assert entropy is not None
    assert entropy[1].item() == 0.0


def test_eagle3_rejected_forced_inputs_use_previous_rejected_token():
    student = object.__new__(ComposedEagle3StudentForCausalLM)
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.long)
    forced = student._build_rejected_forced_input_token_ids(
        input_ids=input_ids,
        prompt_lengths=torch.tensor([2]),
        response_lengths=torch.tensor([4]),
        anchor_positions=torch.tensor([[3]]),
        block_keep_mask=torch.tensor([[True]]),
        draft_block_size=4,
        max_tokens_per_sample=None,
        rejected_draft_anchor_indices=torch.tensor([[1, 1]]),
        rejected_draft_offsets=torch.tensor([[1, 2]]),
        rejected_draft_token_ids=torch.tensor([[101, 102]]),
        rejected_draft_mask=torch.tensor([[True, True]]),
    )

    assert forced.tolist() == [[[-1, -1, 101, 102]]]


def test_eagle3_native_distillation_loss_uses_precomputed_ce_and_logs_scalar_k3():
    data = TensorDict(
        {
            "prompts": torch.tensor([[1]], dtype=torch.long),
            "responses": torch.tensor([[2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "teacher_logprobs": _single_sequence_logprobs([-0.5, -0.7, 0.0]),
        },
        batch_size=[1],
    )
    model_output = {
        "log_probs": torch.tensor([-0.4, -0.9, 0.0], dtype=torch.float32),
        "opd_loss_mask": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
        "eagle3_native_ce_losses": torch.tensor([1.25, 0.75, 0.0], dtype=torch.float32),
        "eagle3_selected_scalar_loss_mask": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
    }
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={})
    loss_config = SimpleNamespace(
        loss_mode="eagle3_native_target_distribution",
        loss_max_clamp=None,
        use_policy_gradient=False,
        response_stream_weight=1.0,
        rejected_draft_stream_weight=0.0,
    )
    distillation_config = SimpleNamespace(distillation_loss=loss_config)

    loss, metrics = distillation_loss(config, distillation_config, model_output, data)

    expected_scalar_k3 = kl_penalty(torch.tensor([-0.4, -0.9]), torch.tensor([-0.5, -0.7]), "k3").mean()
    assert torch.allclose(loss, torch.tensor(1.0))
    assert metrics["eagle3/native_ce_loss"].values == [1.0]
    assert torch.allclose(
        torch.as_tensor(metrics["eagle3/selected_scalar_k3_loss"].values[0]),
        expected_scalar_k3,
    )


def test_rejected_draft_stream_accepts_nested_postprocess_output():
    student = torch.nested.as_nested_tensor(
        [torch.tensor([-1.1, -0.8]), torch.tensor([0.0])], layout=torch.jagged
    )
    teacher = torch.nested.as_nested_tensor(
        [torch.tensor([-0.6, -1.2]), torch.tensor([0.0])], layout=torch.jagged
    )
    mask = torch.nested.as_nested_tensor(
        [torch.tensor([True, True]), torch.tensor([False])], layout=torch.jagged
    )

    stream = get_rejected_draft_distillation_stream(
        {
            "opd_rejected_draft_student_log_probs": student,
            "opd_rejected_draft_teacher_log_probs": teacher,
            "opd_rejected_draft_loss_mask": mask,
        }
    )

    assert stream is not None
    student_values, teacher_values, mask_values = stream
    assert torch.allclose(student_values, torch.tensor([-1.1, -0.8, 0.0]))
    assert torch.allclose(teacher_values, torch.tensor([-0.6, -1.2, 0.0]))
    assert mask_values.tolist() == [True, True, False]


def test_effective_distillation_response_mask_applies_opd_loss_mask():
    data = TensorDict(
        {
            "prompts": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "responses": torch.tensor([[4, 5, 6, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
        },
        batch_size=[1],
    )
    model_output = {
        "opd_loss_mask": torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
    }

    effective_mask = get_effective_distillation_response_mask(data=data, model_output=model_output)
    assert effective_mask.tolist() == [[True, True, False, False]]


def test_scalar_distillation_uses_composed_main_logprobs_without_rollout_teacher_output():
    data = TensorDict(
        {
            "prompts": torch.tensor([[1]], dtype=torch.long),
            "responses": torch.tensor([[2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
        },
        batch_size=[1],
    )
    model_output = {
        "log_probs": torch.tensor([-0.4, -0.9, 0.0], dtype=torch.float32),
        "opd_teacher_log_probs": torch.tensor([-0.5, -0.7, 0.0], dtype=torch.float32),
        "opd_loss_mask": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
    }
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={})
    loss_config = SimpleNamespace(
        loss_mode="k3",
        loss_max_clamp=None,
        use_policy_gradient=False,
        response_stream_weight=1.0,
        rejected_draft_stream_weight=0.0,
    )

    loss, _ = distillation_loss(
        config,
        SimpleNamespace(distillation_loss=loss_config),
        model_output,
        data,
    )

    expected = kl_penalty(torch.tensor([-0.4, -0.9]), torch.tensor([-0.5, -0.7]), "k3").mean()
    assert torch.allclose(loss, expected)


def test_combined_k3_loss_includes_rejected_draft_tokens():
    data = TensorDict(
        {
            "prompts": torch.tensor([[1]], dtype=torch.long),
            "responses": torch.tensor([[2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "teacher_logprobs": _single_sequence_logprobs([-0.5, -0.7, 0.0]),
        },
        batch_size=[1],
    )
    model_output = {
        "log_probs": torch.tensor([-0.4, -0.9, 0.0], dtype=torch.float32),
        "opd_loss_mask": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
        "opd_rejected_draft_student_log_probs": torch.tensor([[-1.1, -0.8]], dtype=torch.float32),
        "opd_rejected_draft_teacher_log_probs": torch.tensor([[-0.6, -1.2]], dtype=torch.float32),
        "opd_rejected_draft_loss_mask": torch.tensor([[True, True]]),
    }
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={})
    loss_config = SimpleNamespace(
        loss_mode="k3",
        loss_max_clamp=None,
        use_policy_gradient=False,
        response_stream_weight=1.0,
        rejected_draft_stream_weight=1.0,
    )
    distillation_config = SimpleNamespace(distillation_loss=loss_config)

    loss, metrics = distillation_loss(config, distillation_config, model_output, data)

    response_losses = kl_penalty(
        torch.tensor([-0.4, -0.9]),
        torch.tensor([-0.5, -0.7]),
        "k3",
    )
    rejected_losses = kl_penalty(
        torch.tensor([-1.1, -0.8]),
        torch.tensor([-0.6, -1.2]),
        "k3",
    )
    assert torch.allclose(loss, torch.cat([response_losses, rejected_losses]).mean())
    assert metrics["distillation/rejected_draft_token_count"].values == [2.0]


def test_forward_kl_weight_uses_local_forward_kl_for_response_and_rejected_draft_tokens():
    data = TensorDict(
        {
            "prompts": torch.tensor([[1]], dtype=torch.long),
            "responses": torch.tensor([[2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "teacher_logprobs": _single_sequence_logprobs([-0.5, -0.7, 0.0]),
        },
        batch_size=[1],
    )
    model_output = {
        "log_probs": torch.tensor([-0.4, -0.9, 0.0], dtype=torch.float32),
        "opd_loss_mask": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
        "opd_rejected_draft_student_log_probs": torch.tensor([[-1.1, -0.8]], dtype=torch.float32),
        "opd_rejected_draft_teacher_log_probs": torch.tensor([[-0.6, -1.2]], dtype=torch.float32),
        "opd_rejected_draft_loss_mask": torch.tensor([[True, True]]),
    }
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={})
    loss_config = SimpleNamespace(
        loss_mode="k3",
        loss_max_clamp=None,
        use_policy_gradient=False,
        reverse_kl_weight=0.0,
        forward_kl_weight=1.0,
        response_stream_weight=1.0,
        rejected_draft_stream_weight=1.0,
    )
    distillation_config = SimpleNamespace(distillation_loss=loss_config)

    loss, metrics = distillation_loss(config, distillation_config, model_output, data)

    response_losses = _local_forward_kl(torch.tensor([-0.4, -0.9]), torch.tensor([-0.5, -0.7]))
    rejected_losses = _local_forward_kl(torch.tensor([-1.1, -0.8]), torch.tensor([-0.6, -1.2]))
    assert torch.allclose(loss, torch.cat([response_losses, rejected_losses]).mean())
    assert torch.allclose(
        torch.as_tensor(metrics["distillation/forward_kl_loss"].values[0]),
        response_losses.mean(),
    )
    assert torch.allclose(
        torch.as_tensor(metrics["distillation/rejected_draft_forward_kl_loss"].values[0]),
        rejected_losses.mean(),
    )


def test_reverse_and_forward_kl_weights_are_combined_before_token_mean():
    data = TensorDict(
        {
            "prompts": torch.tensor([[1]], dtype=torch.long),
            "responses": torch.tensor([[2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "teacher_logprobs": _single_sequence_logprobs([-0.5, -0.7, 0.0]),
        },
        batch_size=[1],
    )
    model_output = {
        "log_probs": torch.tensor([-0.4, -0.9, 0.0], dtype=torch.float32),
        "opd_loss_mask": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
        "opd_rejected_draft_student_log_probs": torch.tensor([[-1.1, -0.8]], dtype=torch.float32),
        "opd_rejected_draft_teacher_log_probs": torch.tensor([[-0.6, -1.2]], dtype=torch.float32),
        "opd_rejected_draft_loss_mask": torch.tensor([[True, True]]),
    }
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={})
    loss_config = SimpleNamespace(
        loss_mode="k3",
        loss_max_clamp=None,
        use_policy_gradient=False,
        reverse_kl_weight=0.25,
        forward_kl_weight=0.75,
        response_stream_weight=1.0,
        rejected_draft_stream_weight=1.0,
    )
    distillation_config = SimpleNamespace(distillation_loss=loss_config)

    loss, _ = distillation_loss(config, distillation_config, model_output, data)

    response_student = torch.tensor([-0.4, -0.9])
    response_teacher = torch.tensor([-0.5, -0.7])
    rejected_student = torch.tensor([-1.1, -0.8])
    rejected_teacher = torch.tensor([-0.6, -1.2])
    response_losses = 0.25 * kl_penalty(response_student, response_teacher, "k3") + 0.75 * _local_forward_kl(
        response_student, response_teacher
    )
    rejected_losses = 0.25 * kl_penalty(rejected_student, rejected_teacher, "k3") + 0.75 * _local_forward_kl(
        rejected_student, rejected_teacher
    )
    assert torch.allclose(loss, torch.cat([response_losses, rejected_losses]).mean())


def test_rejected_draft_can_use_reverse_kl_while_response_uses_forward_kl():
    data = TensorDict(
        {
            "prompts": torch.tensor([[1]], dtype=torch.long),
            "responses": torch.tensor([[2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "teacher_logprobs": _single_sequence_logprobs([-0.5, -0.7, 0.0]),
        },
        batch_size=[1],
    )
    model_output = {
        "log_probs": torch.tensor([-0.4, -0.9, 0.0], dtype=torch.float32),
        "opd_loss_mask": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
        "opd_rejected_draft_student_log_probs": torch.tensor([[-1.1, -0.8]], dtype=torch.float32),
        "opd_rejected_draft_teacher_log_probs": torch.tensor([[-0.6, -1.2]], dtype=torch.float32),
        "opd_rejected_draft_loss_mask": torch.tensor([[True, True]]),
        "opd_rejected_draft_offsets": torch.tensor([[1, 2]]),
    }
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={})
    loss_config = SimpleNamespace(
        loss_mode="k3",
        loss_max_clamp=None,
        use_policy_gradient=False,
        reverse_kl_weight=0.0,
        forward_kl_weight=1.0,
        response_stream_weight=1.0,
        rejected_draft_stream_weight=2.0,
        rejected_draft_use_reverse_kl=True,
        rejected_draft_position_decay_enabled=True,
        rejected_draft_position_decay=0.5,
    )
    distillation_config = SimpleNamespace(distillation_loss=loss_config)

    loss, metrics = distillation_loss(config, distillation_config, model_output, data)

    response_student = torch.tensor([-0.4, -0.9])
    response_teacher = torch.tensor([-0.5, -0.7])
    rejected_student = torch.tensor([-1.1, -0.8])
    rejected_teacher = torch.tensor([-0.6, -1.2])
    response_losses = _local_forward_kl(response_student, response_teacher)
    rejected_losses = kl_penalty(rejected_student, rejected_teacher, "k3")
    rejected_weights = torch.tensor([1.0, 0.5])
    expected = (response_losses.sum() + 2.0 * (rejected_losses * rejected_weights).sum()) / (
        2.0 + 2.0 * rejected_weights.sum()
    )
    assert torch.allclose(loss, expected)
    assert torch.allclose(
        torch.as_tensor(metrics["distillation/forward_kl_loss"].values[0]),
        response_losses.mean(),
    )
    assert torch.allclose(
        torch.as_tensor(metrics["distillation/rejected_draft_reverse_kl_loss"].values[0]),
        rejected_losses.mean(),
    )
    assert "distillation/rejected_draft_forward_kl_loss" not in metrics


def test_rejected_draft_position_decay_weights_loss_by_draft_offset():
    data = TensorDict(
        {
            "prompts": torch.tensor([[1]], dtype=torch.long),
            "responses": torch.tensor([[2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "response_mask": torch.tensor([[1]], dtype=torch.long),
            "teacher_logprobs": _single_sequence_logprobs([-0.5, 0.0]),
        },
        batch_size=[1],
    )
    rejected_student = torch.tensor([-1.1, -0.8, -1.4], dtype=torch.float32)
    rejected_teacher = torch.tensor([-0.6, -1.2, -1.0], dtype=torch.float32)
    model_output = {
        "log_probs": torch.tensor([-0.4, 0.0], dtype=torch.float32),
        "opd_loss_mask": torch.tensor([1.0, 0.0], dtype=torch.float32),
        "opd_rejected_draft_student_log_probs": rejected_student.unsqueeze(0),
        "opd_rejected_draft_teacher_log_probs": rejected_teacher.unsqueeze(0),
        "opd_rejected_draft_loss_mask": torch.tensor([[True, True, True]]),
        "opd_rejected_draft_offsets": torch.tensor([[1, 2, 3]]),
    }
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={})
    loss_config = SimpleNamespace(
        loss_mode="k3",
        loss_max_clamp=None,
        use_policy_gradient=False,
        response_stream_weight=0.0,
        rejected_draft_stream_weight=1.0,
        rejected_draft_position_decay_enabled=True,
        rejected_draft_position_decay=0.5,
    )
    distillation_config = SimpleNamespace(distillation_loss=loss_config)

    loss, _ = distillation_loss(config, distillation_config, model_output, data)

    rejected_losses = kl_penalty(rejected_student, rejected_teacher, "k3")
    weights = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32)
    assert torch.allclose(loss, (rejected_losses * weights).sum() / weights.sum())


def test_accept_draft_position_decay_weights_response_by_draft_offset():
    data = TensorDict(
        {
            "prompts": torch.tensor([[1]], dtype=torch.long),
            "responses": torch.tensor([[2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "teacher_logprobs": _single_sequence_logprobs([-0.5, -0.7, 0.0]),
        },
        batch_size=[1],
    )
    response_student = torch.tensor([-0.4, -0.9], dtype=torch.float32)
    response_teacher = torch.tensor([-0.5, -0.7], dtype=torch.float32)
    model_output = {
        "log_probs": torch.tensor([-0.4, -0.9, 0.0], dtype=torch.float32),
        "opd_loss_mask": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
        "opd_response_offsets": torch.tensor([1, 2, 0], dtype=torch.long),
    }
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={})
    loss_config = SimpleNamespace(
        loss_mode="k3",
        loss_max_clamp=None,
        use_policy_gradient=False,
        accept_draft_position_decay_enabled=True,
        rejected_draft_position_decay=0.5,
    )
    distillation_config = SimpleNamespace(distillation_loss=loss_config)

    loss, _ = distillation_loss(config, distillation_config, model_output, data)

    response_losses = kl_penalty(response_student, response_teacher, "k3")
    weights = torch.tensor([1.0, 0.5], dtype=torch.float32)
    assert torch.allclose(loss, (response_losses * weights).sum() / weights.sum())


def test_accept_and_rejected_draft_position_decay_share_decay_value():
    data = TensorDict(
        {
            "prompts": torch.tensor([[1]], dtype=torch.long),
            "responses": torch.tensor([[2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "teacher_logprobs": _single_sequence_logprobs([-0.5, -0.7, 0.0]),
        },
        batch_size=[1],
    )
    response_student = torch.tensor([-0.4, -0.9], dtype=torch.float32)
    response_teacher = torch.tensor([-0.5, -0.7], dtype=torch.float32)
    rejected_student = torch.tensor([-1.1, -0.8], dtype=torch.float32)
    rejected_teacher = torch.tensor([-0.6, -1.2], dtype=torch.float32)
    model_output = {
        "log_probs": torch.tensor([-0.4, -0.9, 0.0], dtype=torch.float32),
        "opd_loss_mask": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
        "opd_response_offsets": torch.tensor([1, 2, 0], dtype=torch.long),
        "opd_rejected_draft_student_log_probs": rejected_student.unsqueeze(0),
        "opd_rejected_draft_teacher_log_probs": rejected_teacher.unsqueeze(0),
        "opd_rejected_draft_loss_mask": torch.tensor([[True, True]]),
        "opd_rejected_draft_offsets": torch.tensor([[1, 3]]),
    }
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={})
    loss_config = SimpleNamespace(
        loss_mode="k3",
        loss_max_clamp=None,
        use_policy_gradient=False,
        response_stream_weight=1.0,
        rejected_draft_stream_weight=2.0,
        accept_draft_position_decay_enabled=True,
        rejected_draft_position_decay_enabled=True,
        rejected_draft_position_decay=0.5,
    )
    distillation_config = SimpleNamespace(distillation_loss=loss_config)

    loss, _ = distillation_loss(config, distillation_config, model_output, data)

    response_losses = kl_penalty(response_student, response_teacher, "k3")
    rejected_losses = kl_penalty(rejected_student, rejected_teacher, "k3")
    response_weights = torch.tensor([1.0, 0.5], dtype=torch.float32)
    rejected_weights = torch.tensor([1.0, 0.25], dtype=torch.float32)
    expected = (
        (response_losses * response_weights).sum() + 2.0 * (rejected_losses * rejected_weights).sum()
    ) / (response_weights.sum() + 2.0 * rejected_weights.sum())
    assert torch.allclose(loss, expected)


def _single_sample_position_decay_combined_loss(
    response_student,
    response_teacher,
    rejected_student,
    rejected_teacher,
    offsets,
    global_info,
):
    response_student = torch.tensor(response_student, dtype=torch.float32)
    response_teacher = torch.tensor(response_teacher, dtype=torch.float32)
    rejected_student = torch.tensor(rejected_student, dtype=torch.float32)
    rejected_teacher = torch.tensor(rejected_teacher, dtype=torch.float32)
    offsets = torch.tensor(offsets, dtype=torch.long)
    data = TensorDict(
        {
            "prompts": torch.tensor([[1]], dtype=torch.long),
            "responses": torch.tensor([[2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "teacher_logprobs": _single_sequence_logprobs(torch.cat([response_teacher, torch.tensor([0.0])])),
        },
        batch_size=[1],
    )
    model_output = {
        "log_probs": torch.cat([response_student, torch.tensor([0.0])]),
        "opd_loss_mask": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
        "opd_rejected_draft_student_log_probs": rejected_student.unsqueeze(0),
        "opd_rejected_draft_teacher_log_probs": rejected_teacher.unsqueeze(0),
        "opd_rejected_draft_loss_mask": torch.ones((1, rejected_student.numel()), dtype=torch.bool),
        "opd_rejected_draft_offsets": offsets.unsqueeze(0),
    }
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info=global_info)
    loss_config = SimpleNamespace(
        loss_mode="k3",
        loss_max_clamp=None,
        use_policy_gradient=False,
        response_stream_weight=1.0,
        rejected_draft_stream_weight=2.0,
        rejected_draft_position_decay_enabled=True,
        rejected_draft_position_decay=0.5,
    )
    distillation_config = SimpleNamespace(distillation_loss=loss_config)
    loss, _ = distillation_loss(config, distillation_config, model_output, data)
    response_numerator = kl_penalty(response_student, response_teacher, "k3").sum()
    weights = torch.pow(torch.tensor(0.5), (offsets.to(torch.float32) - 1.0).clamp_min(0.0))
    rejected_numerator = (kl_penalty(rejected_student, rejected_teacher, "k3") * weights).sum()
    return loss, response_numerator, rejected_numerator


def test_position_decay_combined_k3_loss_is_invariant_to_micro_batch_splits():
    offsets = [1, 2, 3]
    weights = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32)
    global_info = {
        "dp_size": 1,
        "batch_num_tokens": 4,
        "opd_rejected_draft_batch_num_tokens": 6,
        "opd_rejected_draft_batch_effective_num_tokens": float((weights.sum() * 2).item()),
    }
    loss_a, response_a, rejected_a = _single_sample_position_decay_combined_loss(
        [-0.4, -0.9],
        [-0.5, -0.7],
        [-1.1, -0.8, -1.4],
        [-0.6, -1.2, -1.0],
        offsets,
        global_info,
    )
    loss_b, response_b, rejected_b = _single_sample_position_decay_combined_loss(
        [-0.3, -1.2],
        [-0.4, -0.9],
        [-0.7, -1.4, -1.0],
        [-0.8, -1.0, -0.5],
        offsets,
        global_info,
    )

    expected = (response_a + response_b + 2.0 * (rejected_a + rejected_b)) / (4.0 + 2.0 * weights.sum() * 2.0)
    assert torch.allclose(loss_a + loss_b, expected)


def _single_sample_combined_loss(response_student, response_teacher, rejected_student, rejected_teacher, global_info):
    response_student = torch.tensor(response_student, dtype=torch.float32)
    response_teacher = torch.tensor(response_teacher, dtype=torch.float32)
    rejected_student = torch.tensor(rejected_student, dtype=torch.float32)
    rejected_teacher = torch.tensor(rejected_teacher, dtype=torch.float32)
    data = TensorDict(
        {
            "prompts": torch.tensor([[1]], dtype=torch.long),
            "responses": torch.tensor([[2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "teacher_logprobs": _single_sequence_logprobs(torch.cat([response_teacher, torch.tensor([0.0])])),
        },
        batch_size=[1],
    )
    model_output = {
        "log_probs": torch.cat([response_student, torch.tensor([0.0])]),
        "opd_loss_mask": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
        "opd_rejected_draft_student_log_probs": rejected_student.unsqueeze(0),
        "opd_rejected_draft_teacher_log_probs": rejected_teacher.unsqueeze(0),
        "opd_rejected_draft_loss_mask": torch.tensor([[True, True]]),
    }
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info=global_info)
    loss_config = SimpleNamespace(
        loss_mode="k3",
        loss_max_clamp=None,
        use_policy_gradient=False,
        response_stream_weight=1.0,
        rejected_draft_stream_weight=1.0,
    )
    distillation_config = SimpleNamespace(distillation_loss=loss_config)
    loss, _ = distillation_loss(config, distillation_config, model_output, data)
    numerator = torch.cat(
        [
            kl_penalty(response_student, response_teacher, "k3"),
            kl_penalty(rejected_student, rejected_teacher, "k3"),
        ]
    ).sum()
    return loss, numerator


def test_combined_k3_loss_is_invariant_to_micro_batch_splits():
    global_info = {
        "dp_size": 1,
        "batch_num_tokens": 4,
        "opd_rejected_draft_batch_num_tokens": 4,
    }
    loss_a, numerator_a = _single_sample_combined_loss(
        [-0.4, -0.9], [-0.5, -0.7], [-1.1, -0.8], [-0.6, -1.2], global_info
    )
    loss_b, numerator_b = _single_sample_combined_loss(
        [-0.3, -1.2], [-0.4, -0.9], [-0.7, -1.4], [-0.8, -1.0], global_info
    )

    expected = (numerator_a + numerator_b) / 8
    assert torch.allclose(loss_a + loss_b, expected)


def test_combined_k3_loss_dp_scaled_average_matches_global_mean():
    global_info = {
        "dp_size": 2,
        "batch_num_tokens": 4,
        "opd_rejected_draft_batch_num_tokens": 4,
    }
    rank0_loss, numerator_a = _single_sample_combined_loss(
        [-0.4, -0.9], [-0.5, -0.7], [-1.1, -0.8], [-0.6, -1.2], global_info
    )
    rank1_loss, numerator_b = _single_sample_combined_loss(
        [-0.3, -1.2], [-0.4, -0.9], [-0.7, -1.4], [-0.8, -1.0], global_info
    )

    expected = (numerator_a + numerator_b) / 8
    assert torch.allclose((rank0_loss + rank1_loss) / 2, expected)


def test_count_dflash_rejected_draft_tokens_respects_empty_values_and_limit():
    assert (
        FSDPEngineWithLMHead._count_dflash_rejected_draft_tokens(
            [[1, 2], [], [3, -1]], batch_size=3, max_tokens_per_sample=None
        )
        == 3
    )
    assert (
        FSDPEngineWithLMHead._count_dflash_rejected_draft_tokens(
            [[1, 2], [], [3, -1]], batch_size=3, max_tokens_per_sample=1
        )
        == 2
    )
    assert (
        FSDPEngineWithLMHead._count_dflash_rejected_draft_tokens(
            None, batch_size=3, max_tokens_per_sample=None
        )
        == 0
    )


def test_count_dflash_rejected_draft_tokens_can_keep_first_reject_per_anchor():
    assert (
        FSDPEngineWithLMHead._count_dflash_rejected_draft_tokens(
            [[10, 11, 12, 13]],
            batch_size=1,
            max_tokens_per_sample=None,
            raw_anchor_indices=[[0, 0, 1, 1]],
            raw_offsets=[[4, 3, 2, 5]],
            first_token_only=True,
        )
        == 2
    )
    assert (
        FSDPEngineWithLMHead._count_dflash_rejected_draft_tokens(
            [[10, 11, 12, 13]],
            batch_size=1,
            max_tokens_per_sample=1,
            raw_anchor_indices=[[0, 0, 1, 1]],
            raw_offsets=[[4, 3, 2, 5]],
            first_token_only=True,
        )
        == 1
    )


def test_build_dflash_rejected_draft_tensors_filters_first_reject_per_anchor():
    batch = TensorDict({}, batch_size=[1])
    tu.assign_non_tensor(
        batch,
        dflash_rejected_draft_anchor_indices=[[0, 0, 1, 1]],
        dflash_rejected_draft_offsets=[[4, 3, 2, 5]],
        dflash_rejected_draft_token_ids=[[10, 11, 12, 13]],
        dflash_rejected_draft_teacher_logprobs=[[-0.4, -0.3, -0.2, -0.5]],
        opd_rejected_draft_first_token_only=True,
    )

    tensors = FSDPEngineWithLMHead._build_dflash_rejected_draft_tensors(
        micro_batch=batch,
        batch_size=1,
        device=torch.device("cpu"),
    )

    assert tensors["dflash_rejected_draft_anchor_indices"].tolist() == [[0, 1]]
    assert tensors["dflash_rejected_draft_offsets"].tolist() == [[3, 2]]
    assert tensors["dflash_rejected_draft_token_ids"].tolist() == [[11, 12]]
    assert torch.allclose(
        tensors["dflash_rejected_draft_teacher_logprobs"],
        torch.tensor([[-0.3, -0.2]], dtype=torch.float32),
    )
    assert tensors["dflash_rejected_draft_mask"].tolist() == [[True, True]]


def test_build_dflash_rejected_draft_tensors_keeps_full_suffix_for_tv_validation():
    batch = TensorDict({}, batch_size=[1])
    tu.assign_non_tensor(
        batch,
        dflash_rejected_draft_anchor_indices=[[0, 0]],
        dflash_rejected_draft_offsets=[[2, 3]],
        dflash_rejected_draft_token_ids=[[10, 11]],
        dflash_rejected_draft_teacher_logprobs=[[-0.2, -0.3]],
        opd_rejected_draft_first_token_only=True,
        opd_use_tv_loss=True,
    )

    tensors = FSDPEngineWithLMHead._build_dflash_rejected_draft_tensors(
        micro_batch=batch,
        batch_size=1,
        device=torch.device("cpu"),
    )

    assert tensors["dflash_rejected_draft_offsets"].tolist() == [[2, 3]]
    assert tensors["dflash_rejected_draft_mask"].tolist() == [[True, True]]


def test_count_dflash_tv_blocks_unifies_response_and_rejected_anchors():
    batch = TensorDict(
        {
            "responses": torch.tensor([[1, 2, 3, 4, 5]]),
            "response_mask": torch.ones((1, 5), dtype=torch.long),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(
        batch,
        dflash_reject_token_indices=[[2]],
        dflash_rejected_draft_anchor_indices=[[-1, 0]],
        dflash_rejected_draft_offsets=[[3, 2]],
        dflash_rejected_draft_token_ids=[[10, 11]],
    )

    # Response blocks have anchors -1 and 2; rejected anchor -1 merges with
    # the first response block, while rejected anchor 0 adds one block.
    assert FSDPEngineWithLMHead._count_dflash_tv_blocks(batch) == 3


def test_full_vocab_tv_matches_manual_value_and_only_backpropagates_to_draft():
    draft_logits = torch.tensor([[0.0, 1.0, -1.0]], requires_grad=True)
    target_logits = torch.tensor([[1.0, -0.5, 0.25]], requires_grad=True)

    tv = full_vocab_tv_distance(draft_logits, target_logits)
    expected = 0.5 * (
        torch.softmax(draft_logits.detach(), dim=-1) - torch.softmax(target_logits.detach(), dim=-1)
    ).abs().sum(dim=-1)
    assert torch.allclose(tv, expected)

    tv.sum().backward()
    assert draft_logits.grad is not None
    assert torch.count_nonzero(draft_logits.grad).item() > 0
    assert target_logits.grad is None


def test_e2e_tv_uses_cumulative_overlap_instead_of_mean_token_tv():
    tv = torch.tensor([0.2, 0.4], dtype=torch.float32, requires_grad=True)
    losses, expected_lengths = ComposedDFlashStudentForCausalLM._compute_e2e_tv_block_outputs(
        tv_distances=tv,
        block_indices=torch.tensor([0, 0]),
        block_count=1,
    )

    expected_length = torch.tensor(0.8 + 0.8 * 0.6)
    expected_loss = 1.0 - expected_length / 2.0
    assert torch.allclose(expected_lengths, expected_length.reshape(1))
    assert torch.allclose(losses, expected_loss.reshape(1))
    assert not torch.allclose(losses, tv.detach().mean().reshape(1))
    losses.sum().backward()
    assert tv.grad is not None


def _tv_position_plan(*, first_token_only: bool, offsets=(2, 3), token_ids=(101, 102)):
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    return student._build_tv_position_plan(
        prompt_lengths=torch.tensor([2]),
        response_lengths=torch.tensor([6]),
        anchor_positions=torch.tensor([[2]]),
        segment_lens=torch.tensor([[4]]),
        block_keep_mask=torch.tensor([[True]]),
        valid_seq_lens=torch.tensor([8]),
        draft_block_size=6,
        rejected_draft_anchor_indices=torch.tensor([[0] * len(offsets)]),
        rejected_draft_offsets=torch.tensor([list(offsets)]),
        rejected_draft_token_ids=torch.tensor([list(token_ids)]),
        rejected_draft_mask=torch.ones((1, len(offsets)), dtype=torch.bool),
        first_token_only=first_token_only,
    )


def test_tv_position_plan_merges_accept_and_first_reject_without_residual_duplication():
    plan = _tv_position_plan(first_token_only=True)

    # offset 1 is accepted; offset 2 is the first rejected draft. The response
    # residual at offset 2 and the later response offsets are not separate TV positions.
    assert plan["offsets"].tolist() == [1, 2]
    assert plan["draft_indices"].tolist() == [1, 2]
    assert plan["target_row_indices"].tolist() == [2, 3]
    assert plan["block_indices"].tolist() == [0, 0]
    assert plan["block_count"] == 1
    assert plan["branch_specs"] == []


def test_tv_position_plan_selects_full_rejected_suffix_and_reindexes_one_block():
    plan = _tv_position_plan(first_token_only=False)

    assert plan["offsets"].tolist() == [1, 2, 3]
    assert plan["block_indices"].tolist() == [0, 0, 0]
    assert plan["target_row_indices"].tolist() == [2, 3, -1]
    assert plan["branch_specs"][0]["flat_indices"] == [2]
    assert plan["branch_specs"][0]["suffix_indices"] == [1]


def test_tv_position_plan_rejects_conflicts_out_of_bounds_and_noncontiguous_suffixes():
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        _tv_position_plan(first_token_only=False, offsets=(2, 2), token_ids=(101, 102))
    with pytest.raises(ValueError, match="out of bounds"):
        _tv_position_plan(first_token_only=False, offsets=(6,), token_ids=(101,))
    with pytest.raises(ValueError, match="contiguous"):
        _tv_position_plan(first_token_only=False, offsets=(2, 4), token_ids=(101, 102))


def test_tv_position_plan_uses_all_response_offsets_without_rejected_metadata_and_handles_empty_blocks():
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    plan = student._build_tv_position_plan(
        prompt_lengths=torch.tensor([2]),
        response_lengths=torch.tensor([4]),
        anchor_positions=torch.tensor([[1, 0]]),
        segment_lens=torch.tensor([[3, 0]]),
        block_keep_mask=torch.tensor([[True, False]]),
        valid_seq_lens=torch.tensor([6]),
        draft_block_size=5,
        rejected_draft_anchor_indices=None,
        rejected_draft_offsets=None,
        rejected_draft_token_ids=None,
        rejected_draft_mask=None,
        first_token_only=True,
    )
    assert plan["offsets"].tolist() == [1, 2, 3]
    assert plan["block_count"] == 1

    empty = student._build_tv_position_plan(
        prompt_lengths=torch.tensor([2]),
        response_lengths=torch.tensor([0]),
        anchor_positions=torch.tensor([[0]]),
        segment_lens=torch.tensor([[0]]),
        block_keep_mask=torch.tensor([[False]]),
        valid_seq_lens=torch.tensor([2]),
        draft_block_size=5,
        rejected_draft_anchor_indices=None,
        rejected_draft_offsets=None,
        rejected_draft_token_ids=None,
        rejected_draft_mask=None,
        first_token_only=True,
    )
    assert empty["block_count"] == 0
    assert empty["offsets"].numel() == 0


def test_tv_later_rejected_position_uses_reconstructed_teacher_branch_context():
    class _BranchTeacher:
        def __call__(self, *, input_ids, **kwargs):
            hidden = input_ids.float().cumsum(dim=1).unsqueeze(-1)
            return SimpleNamespace(hidden_states=(hidden,))

    student = object.__new__(ComposedDFlashStudentForCausalLM)
    student.main_model = _BranchTeacher()
    plan = _tv_position_plan(first_token_only=False)
    target_lm_hidden = torch.zeros((1, 8, 1), dtype=torch.float32)
    selected, branch_count = student._compute_tv_teacher_branch_hidden(
        input_ids=torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]]),
        position_ids=torch.arange(8).unsqueeze(0),
        target_lm_hidden=target_lm_hidden,
        tv_plan=plan,
    )

    assert selected[:2].tolist() == [[0.0], [0.0]]
    # prefix through token 13 followed by the first rejected token 101.
    assert selected[2].item() == 10 + 11 + 12 + 13 + 101
    assert branch_count == 1


def _tv_loss_for_micro_batch(block_losses, *, global_block_count, dp_size=1):
    block_losses = torch.tensor(block_losses, dtype=torch.float32, requires_grad=True)
    block_count = float(block_losses.numel())
    position_count = block_count * 2.0
    model_output = {
        "opd_tv_block_losses": block_losses,
        "opd_tv_block_mask": torch.ones_like(block_losses, dtype=torch.bool),
        "opd_tv_distance_sum": torch.tensor(position_count * 0.25),
        "opd_tv_overlap_sum": torch.tensor(position_count * 0.75),
        "opd_tv_expected_accept_length_sum": torch.tensor(block_count * 1.5),
        "opd_tv_block_count": torch.tensor(block_count),
        "opd_tv_position_count": torch.tensor(position_count),
        "opd_tv_teacher_branch_position_count": torch.tensor(0.0),
    }
    config = SimpleNamespace(
        loss_agg_mode="token-mean",
        global_batch_info={"opd_tv_global_block_count": global_block_count, "dp_size": dp_size},
    )
    loss_config = SimpleNamespace(use_tv_loss=True, use_policy_gradient=False)
    loss, metrics = distillation_loss(
        config,
        SimpleNamespace(distillation_loss=loss_config),
        model_output,
        TensorDict({}, batch_size=[]),
    )
    return loss, metrics


def test_tv_block_mean_is_invariant_to_micro_batch_splits_and_dp_scaling():
    loss_a, metrics = _tv_loss_for_micro_batch([0.2, 0.4], global_block_count=4)
    loss_b, _ = _tv_loss_for_micro_batch([0.6, 0.8], global_block_count=4)
    assert torch.allclose(loss_a + loss_b, torch.tensor(0.5))
    assert metrics["distillation/tv_distance"].values == [0.25]
    assert metrics["distillation/tv_overlap"].values == [0.75]

    rank0, _ = _tv_loss_for_micro_batch([0.2, 0.4], global_block_count=4, dp_size=2)
    rank1, _ = _tv_loss_for_micro_batch([0.6, 0.8], global_block_count=4, dp_size=2)
    assert torch.allclose((rank0 + rank1) / 2.0, torch.tensor(0.5))


def test_tv_mode_rejects_policy_gradient_and_non_composed_output():
    with pytest.raises(NotImplementedError, match="direct supervised"):
        DistillationLossConfig(use_tv_loss=True, use_policy_gradient=True)

    ignored = DistillationLossConfig(
        use_tv_loss=True,
        use_policy_gradient=False,
        loss_mode="ignored-by-exact-tv",
        reverse_kl_weight=-1.0,
        forward_kl_weight=-2.0,
        loss_max_clamp=-3.0,
    )
    assert ignored.loss_settings.use_estimator

    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={})
    loss_config = SimpleNamespace(use_tv_loss=True, use_policy_gradient=False)
    with pytest.raises(NotImplementedError, match="composed DFLASH"):
        distillation_loss(
            config,
            SimpleNamespace(distillation_loss=loss_config),
            {},
            TensorDict({}, batch_size=[]),
        )
