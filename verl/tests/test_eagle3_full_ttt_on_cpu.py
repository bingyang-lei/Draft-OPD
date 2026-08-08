# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""CPU tests for the SpecForge-aligned EAGLE3 full TTT OPD forward.

The reference implementations in this file are direct transcriptions of SpecForge's
``OnlineEagle3Model.forward`` (specforge/core/eagle3.py) and the ``cache_hidden`` branch of
its draft ``LlamaAttention.forward`` (specforge/modeling/draft/llama3_eagle.py), operating
on the same module weights. They pin down the token/feature/rope/attention alignment that
the OPD training forward must reproduce.
"""

import copy
import math

import pytest
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM, PretrainedConfig

from verl.models.transformers.eagle3_student import (
    ComposedEagle3StudentForCausalLM,
    Eagle3DraftModel,
    _repeat_kv,
    _rotate_half,
    _shift_left_2d,
    build_eagle3_ttt_attention_mask,
)

VOCAB_SIZE = 64
DRAFT_VOCAB_SIZE = 32
HIDDEN_SIZE = 32
SEQ_LEN = 24
BATCH_SIZE = 2
TTT_LENGTH = 4
TTT_STEP_WEIGHT = 0.8
LM_HEAD_CHUNK_SIZE = 7  # deliberately small to exercise chunk boundaries


def _make_draft_model() -> Eagle3DraftModel:
    config = PretrainedConfig(
        vocab_size=VOCAB_SIZE,
        draft_vocab_size=DRAFT_VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=HIDDEN_SIZE * 2,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        rope_theta=10000.0,
        pad_token_id=0,
    )
    draft_model = Eagle3DraftModel(config=config)
    # Draft vocab covers the even target token ids: draft id i <-> target id 2 * i.
    t2d = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
    t2d[::2] = True
    d2t = torch.arange(DRAFT_VOCAB_SIZE, dtype=torch.int64)  # target_id - draft_id = i
    draft_model.t2d.copy_(t2d)
    draft_model.d2t.copy_(d2t)
    draft_model.refresh_target_to_draft()
    return draft_model


def _make_composed_model() -> ComposedEagle3StudentForCausalLM:
    torch.manual_seed(0)
    main_config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=HIDDEN_SIZE * 2,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=0,
    )
    main_model = LlamaForCausalLM(main_config)
    composed_config = copy.deepcopy(main_config)
    composed_config.verl_eagle3_replay_block_size = TTT_LENGTH + 1
    composed_config.verl_eagle3_ttt_length = TTT_LENGTH
    composed_config.verl_eagle3_ttt_step_weight = TTT_STEP_WEIGHT
    composed_config.verl_eagle3_response_stream_impl = "full_ttt"
    composed_config.verl_eagle3_response_loss_mode = "native_target_distribution"
    composed_config.verl_eagle3_lm_head_chunk_size = LM_HEAD_CHUNK_SIZE
    composed_config.verl_eagle3_response_anchor_stride = 1
    return ComposedEagle3StudentForCausalLM(
        config=composed_config,
        main_model=main_model,
        draft_model=_make_draft_model(),
    )


def _make_batch():
    torch.manual_seed(1)
    input_ids = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), dtype=torch.long)
    prompt_lengths = torch.tensor([5, 7], dtype=torch.long)
    response_lengths = torch.tensor([15, 9], dtype=torch.long)
    valid_lens = prompt_lengths + response_lengths
    attention_mask = (
        torch.arange(SEQ_LEN).unsqueeze(0) < valid_lens.unsqueeze(1)
    ).to(dtype=torch.long)
    input_ids = input_ids * attention_mask  # zero out padding positions
    return input_ids, attention_mask, prompt_lengths, response_lengths


def _specforge_padding_left_false(tensor: torch.Tensor) -> torch.Tensor:
    """SpecForge specforge/utils.py::padding with left=False."""
    zeropadding = torch.zeros_like(tensor[:, -1:])
    return torch.cat((tensor[:, 1:], zeropadding), dim=1)


def _specforge_reference_hiddens(draft_model, target_hidden, input_ids, attention_mask, ttt_length):
    """Transcription of SpecForge OnlineEagle3Model.forward returning per-step hidden states."""
    batch_size, seq_len, _ = target_hidden.shape
    dtype = target_hidden.dtype
    device = target_hidden.device

    hidden = draft_model.project_hidden_states(target_hidden)
    # generate_eagle3_data applies one left shift before the TTT loop.
    global_input_ids = _specforge_padding_left_false(input_ids)
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    # prepare_decoder_attention_mask: additive causal mask + expanded key-padding mask.
    min_value = torch.finfo(dtype).min
    causal = torch.full((seq_len, seq_len), min_value, dtype=dtype, device=device).triu(diagonal=1)
    causal = causal.view(1, 1, seq_len, seq_len).expand(batch_size, 1, seq_len, seq_len)
    key_padding = (1.0 - attention_mask.to(dtype)).view(batch_size, 1, 1, seq_len) * min_value
    additive_mask = (causal + key_padding).clamp_min(min_value)

    layer = draft_model.midlayer
    attn = layer.self_attn
    head_dim = attn.head_dim
    cache_k: list[torch.Tensor] = []
    cache_v: list[torch.Tensor] = []
    step_hiddens = []
    for _ in range(ttt_length):
        input_embeds = draft_model.embed_input_ids(global_input_ids).to(dtype)

        residual = hidden
        normed_hidden = layer.hidden_norm(hidden)
        normed_embeds = layer.input_layernorm(input_embeds)
        attn_input = torch.cat((normed_embeds, normed_hidden), dim=-1)

        query = attn.q_proj(attn_input).view(batch_size, seq_len, attn.num_heads, head_dim).transpose(1, 2)
        key = attn.k_proj(attn_input).view(batch_size, seq_len, attn.num_key_value_heads, head_dim).transpose(1, 2)
        value = attn.v_proj(attn_input).view(batch_size, seq_len, attn.num_key_value_heads, head_dim).transpose(1, 2)

        lck = len(cache_k)
        cos, sin = attn.rotary_emb(query, position_ids=position_ids + lck)
        query = (query * cos) + (_rotate_half(query) * sin)
        key = (key * cos) + (_rotate_half(key) * sin)
        key = _repeat_kv(key, attn.num_key_value_groups)
        value = _repeat_kv(value, attn.num_key_value_groups)

        cache_k.append(key)
        cache_v.append(value)

        attn_weights = torch.matmul(query, cache_k[0].transpose(2, 3)) / math.sqrt(head_dim)
        attn_weights = attn_weights + additive_mask
        for i in range(1, len(cache_k)):
            chain_logits = (query * cache_k[i]).sum(dim=-1) / math.sqrt(head_dim)
            attn_weights = torch.cat((attn_weights, chain_logits.unsqueeze(-1)), dim=-1)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)

        attn_output = torch.matmul(attn_weights[..., :seq_len], cache_v[0])
        for i in range(1, len(cache_v)):
            attn_output = attn_output + attn_weights[..., seq_len + i - 1].unsqueeze(-1) * cache_v[i]
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, attn.q_size)
        attn_output = attn.o_proj(attn_output)

        hidden = residual + attn_output
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))

        step_hiddens.append(hidden)
        global_input_ids = _specforge_padding_left_false(global_input_ids)
    return step_hiddens


def _run_main_model(composed, input_ids, attention_mask):
    with torch.no_grad():
        outputs = composed.main_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    target_hidden = composed._extract_target_hidden(outputs.hidden_states)
    return target_hidden, outputs.logits


def test_ttt_step_matches_specforge_reference():
    composed = _make_composed_model()
    input_ids, attention_mask, _, _ = _make_batch()
    target_hidden, _ = _run_main_model(composed, input_ids, attention_mask)

    reference_hiddens = _specforge_reference_hiddens(
        composed.draft_model, target_hidden, input_ids, attention_mask, TTT_LENGTH
    )

    hidden = composed.draft_model.project_hidden_states(target_hidden)
    base_position_ids = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0).expand(BATCH_SIZE, -1)
    additive_mask = build_eagle3_ttt_attention_mask(
        attention_mask,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        dtype=hidden.dtype,
        device=hidden.device,
    )
    past_keys: list[torch.Tensor] = []
    past_values: list[torch.Tensor] = []
    for step_idx in range(TTT_LENGTH):
        token_ids = _shift_left_2d(input_ids, step_idx + 1)
        input_embeds = composed.draft_model.embed_input_ids(token_ids).to(hidden.dtype)
        hidden, key_states, value_states = composed._eagle3_ttt_step(
            input_embeds,
            hidden,
            base_position_ids,
            additive_mask,
            step_idx,
            *past_keys,
            *past_values,
        )
        past_keys.append(key_states)
        past_values.append(value_states)
        torch.testing.assert_close(
            hidden,
            reference_hiddens[step_idx],
            atol=1e-5,
            rtol=1e-5,
            msg=f"TTT step {step_idx} hidden mismatch vs SpecForge reference",
        )


def test_full_ttt_forward_bookkeeping():
    composed = _make_composed_model()
    input_ids, attention_mask, prompt_lengths, response_lengths = _make_batch()
    target_hidden, target_logits = _run_main_model(composed, input_ids, attention_mask)

    result = composed._run_eagle3_full_ttt_forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=None,
        target_hidden=target_hidden,
        target_logits=target_logits,
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
        ttt_length=TTT_LENGTH,
        ttt_step_weight=TTT_STEP_WEIGHT,
        lm_head_chunk_size=LM_HEAD_CHUNK_SIZE,
        calculate_entropy=False,
        profile_enabled=False,
    )

    reference_hiddens = _specforge_reference_hiddens(
        composed.draft_model, target_hidden, input_ids, attention_mask, TTT_LENGTH
    )

    t2d = composed.draft_model.t2d.bool()
    valid_lens = prompt_lengths + response_lengths
    expected_ce = torch.zeros(BATCH_SIZE, SEQ_LEN)
    expected_loss_mask = torch.zeros(BATCH_SIZE, SEQ_LEN)
    expected_log_probs = torch.zeros(BATCH_SIZE, SEQ_LEN)
    expected_scalar_mask = torch.zeros(BATCH_SIZE, SEQ_LEN)
    expected_step0_supported = 0
    expected_step0_top1 = 0
    expected_step0_total = 0
    expected_step0_scalar = 0
    target_to_draft = composed.draft_model.target_to_draft

    with torch.no_grad():
        for step_idx, step_hidden in enumerate(reference_hiddens):
            weight = TTT_STEP_WEIGHT**step_idx
            for batch_idx in range(BATCH_SIZE):
                for pos in range(SEQ_LEN):
                    label_pos = pos + step_idx + 2
                    in_response = (
                        label_pos >= int(prompt_lengths[batch_idx])
                        and label_pos < int(valid_lens[batch_idx])
                    )
                    if not in_response:
                        continue
                    logits = composed.draft_model.compute_logits(step_hidden[batch_idx, pos])
                    log_probs = F.log_softmax(logits.float(), dim=-1)
                    teacher_row = target_logits[batch_idx, pos + step_idx + 1]
                    if step_idx == 0:
                        expected_step0_total += 1
                    if bool(t2d[teacher_row.argmax()]):
                        target_p = F.softmax(teacher_row[t2d].float(), dim=-1)
                        expected_ce[batch_idx, pos] += weight * float(-(target_p * log_probs).sum())
                        expected_loss_mask[batch_idx, pos] = 1.0
                        if step_idx == 0:
                            expected_step0_supported += 1
                            expected_step0_top1 += int(logits.argmax() == target_p.argmax())
                    if step_idx == 0:
                        label_token = int(input_ids[batch_idx, label_pos])
                        draft_label = int(target_to_draft[label_token])
                        if draft_label >= 0:
                            expected_log_probs[batch_idx, pos] = float(log_probs[draft_label])
                            expected_scalar_mask[batch_idx, pos] = 1.0
                            expected_step0_scalar += 1

    torch.testing.assert_close(result["native_ce_loss_by_seq"], expected_ce, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(result["loss_mask_by_seq"], expected_loss_mask)
    torch.testing.assert_close(result["log_probs_by_seq"], expected_log_probs, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(result["selected_scalar_loss_mask_by_seq"], expected_scalar_mask)
    assert result["native_total_count"] == expected_step0_total
    assert result["native_supported_count"] == expected_step0_supported
    assert result["native_top1_correct_count"] == expected_step0_top1
    assert result["response_lm_token_count"] == expected_step0_scalar
    assert result["ran_draft_forward"] is True


def test_forward_opd_full_ttt_end_to_end_backward():
    composed = _make_composed_model()
    input_ids, attention_mask, prompt_lengths, response_lengths = _make_batch()
    reject_indices = torch.tensor([[3, 8], [2, -1]], dtype=torch.long)

    output = composed(
        input_ids=input_ids,
        attention_mask=attention_mask,
        dflash_prompt_lengths=prompt_lengths,
        dflash_response_lengths=response_lengths,
        dflash_reject_token_indices=reject_indices,
    )

    assert output["eagle3_response_loss_mode_id"].item() == 1
    native_ce = output["eagle3_native_ce_losses"]
    loss_mask = output["dflash_loss_mask"]
    assert native_ce.shape == (BATCH_SIZE, SEQ_LEN)
    assert loss_mask.sum() > 0
    assert torch.all(native_ce[loss_mask.bool()] > 0)

    loss = (native_ce * loss_mask).sum() / loss_mask.sum()
    loss.backward()
    draft_grads = [p.grad for p in composed.draft_model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in draft_grads)
    assert all(p.grad is None for p in composed.main_model.parameters())


def test_forward_opd_anchor_block_fallback_still_runs():
    composed = _make_composed_model()
    composed.config.verl_eagle3_response_stream_impl = "anchor_block"
    composed.config.verl_eagle3_response_loss_mode = "scalar_k3"
    input_ids, attention_mask, prompt_lengths, response_lengths = _make_batch()
    reject_indices = torch.tensor([[3, 8], [2, -1]], dtype=torch.long)

    output = composed(
        input_ids=input_ids,
        attention_mask=attention_mask,
        dflash_prompt_lengths=prompt_lengths,
        dflash_response_lengths=response_lengths,
        dflash_reject_token_indices=reject_indices,
    )
    assert output["eagle3_response_loss_mode_id"].item() == 0
    assert output["dflash_log_probs"].shape == (BATCH_SIZE, SEQ_LEN)
    assert output["dflash_loss_mask"].sum() > 0


def test_full_ttt_requires_native_loss_mode():
    composed = _make_composed_model()
    composed.config.verl_eagle3_response_loss_mode = "scalar_k3"
    input_ids, attention_mask, prompt_lengths, response_lengths = _make_batch()
    with pytest.raises(RuntimeError, match="full_ttt requires"):
        composed(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dflash_prompt_lengths=prompt_lengths,
            dflash_response_lengths=response_lengths,
            dflash_reject_token_indices=torch.full((BATCH_SIZE, 1), -1, dtype=torch.long),
        )
