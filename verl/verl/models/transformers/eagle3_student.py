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
from __future__ import annotations

import copy
import logging
import math
import os
import time
from typing import Any, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from transformers.activations import ACT2FN

from verl.models.transformers.dflash_student import ComposedDFlashStudentForCausalLM

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

EAGLE3_ATTENTION_IMPL_IDS = {
    "sdpa_unrolled": 10,
    "eager_unrolled": 11,
    "ttt_full": 12,
}

EAGLE3_RESPONSE_LOSS_MODES = {
    "scalar_k3",
    "native_target_distribution",
}

EAGLE3_RESPONSE_STREAM_IMPLS = {
    "full_ttt",
    "anchor_block",
}


def _config_get(config: PretrainedConfig, attr_name: str, default: Any = None) -> Any:
    return getattr(config, attr_name, default)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def _shift_left_2d(tensor: torch.Tensor, shift: int) -> torch.Tensor:
    """SpecForge-style left shift along dim 1, padding the tail with zeros."""
    if shift <= 0:
        return tensor
    batch_size, seq_len = tensor.shape
    if shift >= seq_len:
        return torch.zeros_like(tensor)
    pad = tensor.new_zeros((batch_size, shift))
    return torch.cat((tensor[:, shift:], pad), dim=1)


def build_eagle3_ttt_attention_mask(
    attention_mask_2d: Optional[torch.Tensor],
    *,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Additive causal (+padding) mask used by the step-0 keys of the full TTT forward."""
    allowed = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).tril()
    allowed = allowed.view(1, 1, seq_len, seq_len)
    if attention_mask_2d is not None:
        key_mask = attention_mask_2d.to(dtype=torch.bool, device=device).view(batch_size, 1, 1, seq_len)
        allowed = allowed & key_mask
    else:
        allowed = allowed.expand(batch_size, 1, seq_len, seq_len)
    additive = torch.zeros(allowed.shape, dtype=dtype, device=device)
    return additive.masked_fill(~allowed, torch.finfo(dtype).min)


class Eagle3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Eagle3RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float):
        super().__init__()
        self.dim = int(dim)
        self.max_position_embeddings = int(max_position_embeddings)
        self.base = float(base)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=x.device)
        freqs = torch.einsum("bs,d->bsd", position_ids.to(dtype=inv_freq.dtype), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype).unsqueeze(1)
        sin = emb.sin().to(dtype=x.dtype).unsqueeze(1)
        return cos, sin


class Eagle3Attention(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.hidden_size = int(_config_get(config, "hidden_size"))
        self.num_heads = int(_config_get(config, "num_attention_heads"))
        self.num_key_value_heads = int(_config_get(config, "num_key_value_heads", self.num_heads))
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = int(_config_get(config, "head_dim", self.hidden_size // self.num_heads))
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim
        input_size = self.hidden_size * 2

        self.q_proj = nn.Linear(input_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(input_size, self.kv_size, bias=False)
        self.v_proj = nn.Linear(input_size, self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, self.hidden_size, bias=False)
        self.rotary_emb = Eagle3RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=int(_config_get(config, "max_position_embeddings", 40960)),
            base=float(_config_get(config, "rope_theta", 10000.0)),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cache_hidden: Optional[list[list[torch.Tensor]]],
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        batch_size, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(query_states, position_ids=position_ids)
        query_states = (query_states * cos) + (_rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (_rotate_half(key_states) * sin)

        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        if cache_hidden is not None:
            cache_hidden[0].append(key_states)
            cache_hidden[1].append(value_states)
            key_states = torch.cat(cache_hidden[0], dim=2)
            value_states = torch.cat(cache_hidden[1], dim=2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if cache_hidden is None and q_len > 1:
            key_len = key_states.shape[-2]
            causal_mask = torch.ones((q_len, key_len), dtype=torch.bool, device=attn_weights.device).tril()
            attn_weights = attn_weights.masked_fill(~causal_mask.view(1, 1, q_len, key_len), torch.finfo(attn_weights.dtype).min)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype=query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.q_size)
        return self.o_proj(attn_output)

    def ttt_forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.LongTensor,
        step_idx: int,
        attention_mask: torch.Tensor,
        past_keys: tuple[torch.Tensor, ...],
        past_values: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full-sequence TTT attention replicating SpecForge's draft attention.

        The query at sequence position ``p`` of TTT step ``step_idx`` uses rope position
        ``p + step_idx``, attends causally over the step-0 keys (with the additive
        ``attention_mask``), and attends diagonally (same position only) to the chain keys
        of steps ``1..step_idx`` with a joint softmax. Mirrors SpecForge
        ``llama3_eagle.py::LlamaAttention.forward`` cache_hidden branch.
        """
        batch_size, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(batch_size, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(batch_size, q_len, self.num_key_value_heads, self.head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = self.rotary_emb(query_states, position_ids=position_ids + step_idx)
        query_states = (query_states * cos) + (_rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (_rotate_half(key_states) * sin)

        # Cache (and return) pre-repeat kv to keep the cross-step saved tensors 4x smaller;
        # heads are expanded transiently at consume time.
        if step_idx == 0:
            context_keys = _repeat_kv(key_states, self.num_key_value_groups)
            context_values = _repeat_kv(value_states, self.num_key_value_groups)
            chain_keys: list[torch.Tensor] = []
            chain_values: list[torch.Tensor] = []
        else:
            context_keys = _repeat_kv(past_keys[0], self.num_key_value_groups)
            context_values = _repeat_kv(past_values[0], self.num_key_value_groups)
            chain_keys = [_repeat_kv(k, self.num_key_value_groups) for k in past_keys[1:]]
            chain_keys.append(_repeat_kv(key_states, self.num_key_value_groups))
            chain_values = [_repeat_kv(v, self.num_key_value_groups) for v in past_values[1:]]
            chain_values.append(_repeat_kv(value_states, self.num_key_value_groups))

        attn_weights = torch.matmul(query_states, context_keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        for chain_key in chain_keys:
            chain_logits = (query_states * chain_key).sum(dim=-1) / math.sqrt(self.head_dim)
            attn_weights = torch.cat((attn_weights, chain_logits.unsqueeze(-1)), dim=-1)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype=query_states.dtype)

        attn_output = torch.matmul(attn_weights[..., :q_len], context_values)
        for chain_idx, chain_value in enumerate(chain_values):
            attn_output = attn_output + attn_weights[..., q_len + chain_idx].unsqueeze(-1) * chain_value
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, q_len, self.q_size)
        return self.o_proj(attn_output), key_states, value_states


class Eagle3MLP(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.hidden_size = int(_config_get(config, "hidden_size"))
        self.intermediate_size = int(_config_get(config, "intermediate_size"))
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[str(_config_get(config, "hidden_act", "silu"))]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Eagle3DecoderLayer(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        hidden_size = int(_config_get(config, "hidden_size"))
        eps = float(_config_get(config, "rms_norm_eps", 1e-6))
        self.self_attn = Eagle3Attention(config=config)
        self.mlp = Eagle3MLP(config=config)
        self.hidden_norm = Eagle3RMSNorm(hidden_size, eps=eps)
        self.input_layernorm = Eagle3RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = Eagle3RMSNorm(hidden_size, eps=eps)

    def forward(
        self,
        *,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[list[list[torch.Tensor]]],
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)
        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)
        hidden_states = self.self_attn(
            hidden_states,
            cache_hidden=cache_hidden,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Eagle3DraftModel(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.hidden_size = int(_config_get(config, "hidden_size"))
        self.vocab_size = int(_config_get(config, "vocab_size"))
        self.draft_vocab_size = int(_config_get(config, "draft_vocab_size"))
        pad_token_id = _config_get(config, "pad_token_id")
        padding_idx = None if pad_token_id is None else int(pad_token_id)
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=padding_idx)
        self.midlayer = Eagle3DecoderLayer(config=config)
        self.fc = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=False)
        self.norm = Eagle3RMSNorm(self.hidden_size, eps=float(_config_get(config, "rms_norm_eps", 1e-6)))
        self.lm_head = nn.Linear(self.hidden_size, self.draft_vocab_size, bias=False)
        self.register_buffer("t2d", torch.ones(self.vocab_size, dtype=torch.bool))
        self.register_buffer("d2t", torch.zeros(self.draft_vocab_size, dtype=torch.int64))
        self.register_buffer("target_to_draft", torch.full((self.vocab_size,), -1, dtype=torch.long), persistent=False)

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc(hidden_states)

    def backbone(
        self,
        *,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[list[list[torch.Tensor]]],
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        return self.midlayer(
            input_emb=input_embeds,
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,
            position_ids=position_ids,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(hidden_states))

    def refresh_target_to_draft(self) -> None:
        target_to_draft = torch.full((self.vocab_size,), -1, dtype=torch.long, device=self.d2t.device)
        draft_ids = torch.arange(self.draft_vocab_size, dtype=torch.long, device=self.d2t.device)
        target_ids = draft_ids + self.d2t.to(dtype=torch.long, device=self.d2t.device)
        in_bounds = (target_ids >= 0) & (target_ids < self.vocab_size)
        target_to_draft[target_ids[in_bounds]] = draft_ids[in_bounds]
        if self.t2d.numel() == self.vocab_size:
            target_to_draft = torch.where(self.t2d.to(device=target_to_draft.device), target_to_draft, -1)
        self.target_to_draft = target_to_draft


class ComposedEagle3StudentForCausalLM(ComposedDFlashStudentForCausalLM):
    """Train-only composed model: frozen target model + trainable EAGLE3 draft."""

    def _configure_draft_attention(self) -> None:
        return None

    def __init__(
        self,
        config: PretrainedConfig,
        main_model: PreTrainedModel,
        draft_model: Eagle3DraftModel,
    ):
        super().__init__(config=config, main_model=main_model, draft_model=draft_model)
        num_layers = int(getattr(main_model.config, "num_hidden_layers"))
        self.target_layer_ids = [1, num_layers // 2 - 1, num_layers - 4]
        self.draft_model.set_input_embeddings(self.main_model.get_input_embeddings())
        self.draft_model.refresh_target_to_draft()
        self.freeze_main_model()

    def _get_block_size(self) -> int:
        value = getattr(self.config, "verl_eagle3_replay_block_size", None)
        if value is None:
            value = os.getenv("VERL_EAGLE3_REPLAY_BLOCK_SIZE", "4")
        value = int(value)
        if value <= 1:
            raise ValueError(f"verl_eagle3_replay_block_size must be > 1, got {value}.")
        return value

    def _get_lm_head_chunk_size(self) -> int:
        value = getattr(self.config, "verl_eagle3_lm_head_chunk_size", None)
        if value is None:
            value = os.getenv("VERL_EAGLE3_LM_HEAD_CHUNK_SIZE", "2048")
        value = int(value)
        if value <= 0:
            raise ValueError(f"verl_eagle3_lm_head_chunk_size must be positive, got {value}.")
        return value

    def _get_response_anchor_stride(self) -> int:
        value = getattr(self.config, "verl_eagle3_response_anchor_stride", None)
        if value is None:
            value = os.getenv("VERL_EAGLE3_RESPONSE_ANCHOR_STRIDE", "1")
        value = int(value)
        if value <= 0:
            raise ValueError(f"verl_eagle3_response_anchor_stride must be positive, got {value}.")
        return value

    def _get_random_response_anchor_enabled(self) -> bool:
        value = getattr(self.config, "verl_eagle3_random_response_anchor_enabled", None)
        if value is None:
            value = os.getenv("VERL_EAGLE3_RANDOM_RESPONSE_ANCHOR_ENABLED", "0")
        return str(value).lower() in {"1", "true", "yes", "on"}

    def _get_random_response_anchor_seed(self) -> int:
        value = getattr(self.config, "verl_eagle3_random_response_anchor_seed", None)
        if value is None:
            value = os.getenv("VERL_EAGLE3_RANDOM_RESPONSE_ANCHOR_SEED", "42")
        return int(value)

    def _get_response_loss_mode(self) -> str:
        value = getattr(self.config, "verl_eagle3_response_loss_mode", None)
        if value is None:
            value = os.getenv("VERL_EAGLE3_RESPONSE_LOSS_MODE", "scalar_k3")
        value = str(value)
        if value not in EAGLE3_RESPONSE_LOSS_MODES:
            raise ValueError(
                f"Unsupported verl_eagle3_response_loss_mode={value!r}. "
                f"Supported modes are: {sorted(EAGLE3_RESPONSE_LOSS_MODES)}."
            )
        return value

    def _get_response_stream_impl(self) -> str:
        value = getattr(self.config, "verl_eagle3_response_stream_impl", None)
        if value is None:
            value = os.getenv("VERL_EAGLE3_RESPONSE_STREAM_IMPL", "full_ttt")
        value = str(value)
        if value not in EAGLE3_RESPONSE_STREAM_IMPLS:
            raise ValueError(
                f"Unsupported verl_eagle3_response_stream_impl={value!r}. "
                f"Supported impls are: {sorted(EAGLE3_RESPONSE_STREAM_IMPLS)}."
            )
        return value

    def _get_ttt_length(self) -> int:
        value = getattr(self.config, "verl_eagle3_ttt_length", None)
        if value is None:
            value = os.getenv("VERL_EAGLE3_TTT_LENGTH")
        if value is None or str(value).lower() in {"", "none", "null"}:
            return self._get_block_size() - 1
        value = int(value)
        if value <= 0:
            raise ValueError(f"verl_eagle3_ttt_length must be positive, got {value}.")
        return value

    def _get_ttt_step_weight(self) -> float:
        value = getattr(self.config, "verl_eagle3_ttt_step_weight", None)
        if value is None:
            value = os.getenv("VERL_EAGLE3_TTT_STEP_WEIGHT", "0.8")
        value = float(value)
        if value <= 0.0:
            raise ValueError(f"verl_eagle3_ttt_step_weight must be positive, got {value}.")
        return value

    def _get_rejected_draft_max_tokens_per_sample(self) -> Optional[int]:
        value = getattr(self.config, "verl_eagle3_rejected_draft_max_tokens_per_sample", None)
        if value is None:
            value = os.getenv("VERL_EAGLE3_REJECTED_DRAFT_MAX_TOKENS_PER_SAMPLE")
        if value is None or str(value).lower() in {"", "none", "null"}:
            return None
        value = int(value)
        if value <= 0:
            raise ValueError(f"verl_eagle3_rejected_draft_max_tokens_per_sample must be positive, got {value}.")
        return value

    def _is_dflash_profiling_enabled(self) -> bool:
        return os.getenv("VERL_EAGLE3_PROFILE", "0").lower() in {"1", "true", "yes", "on"}

    def _map_target_to_draft_ids(self, target_token_ids: torch.LongTensor) -> tuple[torch.LongTensor, torch.Tensor]:
        target_to_draft = self.draft_model.target_to_draft.to(device=target_token_ids.device)
        in_bounds = (target_token_ids >= 0) & (target_token_ids < target_to_draft.numel())
        safe_target_ids = target_token_ids.clamp(0, target_to_draft.numel() - 1)
        draft_ids = target_to_draft[safe_target_ids]
        supported = in_bounds & (draft_ids >= 0)
        return draft_ids.clamp_min(0), supported

    def _compute_selected_eagle3_log_probs(
        self,
        *,
        draft_hidden: torch.Tensor,
        batch_indices: torch.LongTensor,
        draft_indices: torch.LongTensor,
        target_token_ids: torch.LongTensor,
        chunk_size: int,
        calculate_entropy: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        if batch_indices.numel() == 0:
            empty = draft_hidden.new_empty((0,), dtype=torch.float32)
            return empty, empty if calculate_entropy else None, torch.zeros((0,), dtype=torch.bool, device=draft_hidden.device)

        draft_token_ids, supported = self._map_target_to_draft_ids(target_token_ids.to(dtype=torch.long))
        if not bool(supported.any()):
            empty = draft_hidden.new_empty((0,), dtype=torch.float32)
            return empty, empty if calculate_entropy else None, supported

        selected_hidden = draft_hidden[batch_indices[supported], draft_indices[supported], :]
        selected_token_ids = draft_token_ids[supported]
        log_prob_chunks: list[torch.Tensor] = []
        entropy_chunks: list[torch.Tensor] = []
        for start in range(0, selected_hidden.shape[0], chunk_size):
            end = min(start + chunk_size, selected_hidden.shape[0])
            logits = self.draft_model.compute_logits(selected_hidden[start:end])
            log_probs = F.log_softmax(logits.float(), dim=-1)
            labels = selected_token_ids[start:end].to(device=log_probs.device)
            log_prob_chunks.append(log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1))
            if calculate_entropy:
                entropy_chunks.append(-(log_probs.exp() * log_probs).sum(dim=-1))

        selected_log_probs = torch.cat(log_prob_chunks, dim=0)
        selected_entropy = torch.cat(entropy_chunks, dim=0) if calculate_entropy else None
        return selected_log_probs, selected_entropy, supported

    def _compute_eagle3_native_response_outputs(
        self,
        *,
        draft_hidden: torch.Tensor,
        target_logits: torch.Tensor,
        batch_indices: torch.LongTensor,
        draft_indices: torch.LongTensor,
        target_row_indices: torch.LongTensor,
        target_token_ids: torch.LongTensor,
        chunk_size: int,
        calculate_entropy: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total_count = int(batch_indices.numel())
        empty = draft_hidden.new_empty((0,), dtype=torch.float32)
        empty_bool = torch.zeros((0,), dtype=torch.bool, device=draft_hidden.device)
        if total_count == 0:
            return empty, empty if calculate_entropy else None, empty_bool, empty, empty_bool, empty_bool

        t2d = self.draft_model.t2d.to(device=target_logits.device, dtype=torch.bool)
        if target_logits.shape[-1] != t2d.numel():
            raise RuntimeError(
                "EAGLE3 native response loss requires target logits vocab to match draft t2d mapping, got "
                f"target_logits={target_logits.shape[-1]} and t2d={t2d.numel()}."
            )
        draft_vocab_size = int(t2d.sum().item())

        scalar_log_probs = draft_hidden.new_zeros((total_count,), dtype=torch.float32)
        scalar_supported_parts: list[torch.Tensor] = []
        native_losses = draft_hidden.new_zeros((total_count,), dtype=torch.float32)
        native_supported_parts: list[torch.Tensor] = []
        top1_match_parts: list[torch.Tensor] = []
        entropy_values = draft_hidden.new_zeros((total_count,), dtype=torch.float32) if calculate_entropy else None

        draft_token_ids, scalar_supported = self._map_target_to_draft_ids(target_token_ids.to(dtype=torch.long))
        for start in range(0, total_count, chunk_size):
            end = min(start + chunk_size, total_count)
            selected_hidden = draft_hidden[batch_indices[start:end], draft_indices[start:end], :]
            logits = self.draft_model.compute_logits(selected_hidden)
            if logits.shape[-1] != draft_vocab_size:
                raise RuntimeError(
                    "EAGLE3 draft logits vocab does not match target t2d-supported vocab, got "
                    f"draft_logits={logits.shape[-1]} and t2d_supported={draft_vocab_size}."
                )
            log_probs = F.log_softmax(logits.float(), dim=-1)

            scalar_chunk_supported = scalar_supported[start:end].to(device=log_probs.device)
            if bool(scalar_chunk_supported.any()):
                scalar_labels = draft_token_ids[start:end].to(device=log_probs.device)
                chunk_positions = torch.arange(end - start, device=log_probs.device)
                scalar_output_positions = torch.arange(start, end, device=log_probs.device)[scalar_chunk_supported]
                scalar_log_probs[scalar_output_positions] = log_probs[
                    chunk_positions[scalar_chunk_supported], scalar_labels[scalar_chunk_supported]
                ]
            scalar_supported_parts.append(scalar_chunk_supported)

            target_chunk = target_logits[batch_indices[start:end], target_row_indices[start:end], :]
            target_argmax = target_chunk.argmax(dim=-1)
            native_chunk_supported = t2d[target_argmax].to(device=log_probs.device)
            target_p = F.softmax(target_chunk[:, t2d].float(), dim=-1).detach().to(device=log_probs.device)
            chunk_losses = -(target_p * log_probs).sum(dim=-1)
            native_output_positions = torch.arange(start, end, device=log_probs.device)[native_chunk_supported]
            native_losses[native_output_positions] = chunk_losses[native_chunk_supported]

            top1_match = torch.zeros((end - start,), dtype=torch.bool, device=log_probs.device)
            if bool(native_chunk_supported.any()):
                top1_match[native_chunk_supported] = (
                    logits.argmax(dim=-1)[native_chunk_supported] == target_p.argmax(dim=-1)[native_chunk_supported]
                )
                if entropy_values is not None:
                    chunk_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
                    entropy_values[native_output_positions] = chunk_entropy[native_chunk_supported]
            native_supported_parts.append(native_chunk_supported)
            top1_match_parts.append(top1_match)

        native_supported = torch.cat(native_supported_parts, dim=0)
        scalar_supported = torch.cat(scalar_supported_parts, dim=0)
        top1_match = torch.cat(top1_match_parts, dim=0)
        return scalar_log_probs, entropy_values, scalar_supported, native_losses, native_supported, top1_match

    def _eagle3_ttt_step(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        additive_attention_mask: torch.Tensor,
        step_idx: int,
        *past_kv_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One functional TTT step (decoder layer) so it can be gradient-checkpointed."""
        num_cached = len(past_kv_flat) // 2
        past_keys = past_kv_flat[:num_cached]
        past_values = past_kv_flat[num_cached:]
        layer = self.draft_model.midlayer

        residual = hidden_states
        normed_hidden = layer.hidden_norm(hidden_states)
        normed_embeds = layer.input_layernorm(input_embeds)
        attn_input = torch.cat((normed_embeds, normed_hidden), dim=-1)
        attn_output, key_states, value_states = layer.self_attn.ttt_forward(
            attn_input,
            position_ids=position_ids,
            step_idx=step_idx,
            attention_mask=additive_attention_mask,
            past_keys=past_keys,
            past_values=past_values,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))
        return hidden_states, key_states, value_states

    def _eagle3_loss_chunk(
        self,
        selected_hidden: torch.Tensor,
        target_logits: torch.Tensor,
        batch_indices: torch.Tensor,
        teacher_row_indices: torch.Tensor,
        draft_label_ids: torch.Tensor,
        t2d: torch.Tensor,
        compute_entropy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure per-chunk loss computation, gradient-checkpointed by the caller.

        The fp32 log-prob and teacher-distribution tensors dominate activation memory when
        kept alive across all TTT steps, so they are recomputed during backward instead.
        """
        logits = self.draft_model.compute_logits(selected_hidden)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        teacher_rows = target_logits[batch_indices, teacher_row_indices, :]
        native_supported = t2d[teacher_rows.argmax(dim=-1)]
        target_p = F.softmax(teacher_rows[:, t2d].float(), dim=-1).detach()
        ce_losses = -(target_p * log_probs).sum(dim=-1)
        scalar_log_probs = log_probs.gather(dim=-1, index=draft_label_ids.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        top1_match = logits.argmax(dim=-1) == target_p.argmax(dim=-1)
        if compute_entropy:
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        else:
            entropy = torch.zeros_like(ce_losses)
        return ce_losses, scalar_log_probs, native_supported, top1_match, entropy

    def _run_eagle3_full_ttt_forward(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        target_hidden: torch.Tensor,
        target_logits: torch.Tensor,
        prompt_lengths: torch.LongTensor,
        response_lengths: torch.LongTensor,
        ttt_length: int,
        ttt_step_weight: float,
        lm_head_chunk_size: int,
        calculate_entropy: bool,
        profile_enabled: bool,
    ) -> dict[str, Any]:
        """SpecForge-aligned full-sequence TTT forward for the response stream.

        Replicates SpecForge ``OnlineEagle3Model.forward``: at TTT step ``idx`` the query at
        sequence position ``p`` consumes the embedding of token ``t_{p+idx+1}`` together with
        the (recursively produced) hidden state at ``p``, and is trained against the target
        model's distribution at row ``p+idx+1`` (i.e. the distribution over ``t_{p+idx+2}``).
        Per-step CE losses are accumulated per position with ``ttt_step_weight ** idx``.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        t2d = self.draft_model.t2d.to(device=target_logits.device, dtype=torch.bool)
        if target_logits.shape[-1] != t2d.numel():
            raise RuntimeError(
                "EAGLE3 full TTT requires target logits vocab to match draft t2d mapping, got "
                f"target_logits={target_logits.shape[-1]} and t2d={t2d.numel()}."
            )

        if position_ids is not None and position_ids.dim() == 2:
            base_position_ids = position_ids
        else:
            base_position_ids = (
                torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)
            )

        if attention_mask is not None:
            valid_seq_lens = attention_mask.long().sum(dim=1)
        else:
            valid_seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)

        hidden = self.draft_model.project_hidden_states(target_hidden)
        additive_mask = build_eagle3_ttt_attention_mask(
            attention_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=hidden.dtype,
            device=device,
        )

        log_probs_by_seq = hidden.new_zeros((batch_size, seq_len), dtype=torch.float32)
        loss_mask_by_seq = hidden.new_zeros((batch_size, seq_len), dtype=torch.float32)
        selected_scalar_loss_mask_by_seq = hidden.new_zeros((batch_size, seq_len), dtype=torch.float32)
        native_ce_loss_by_seq = hidden.new_zeros((batch_size, seq_len), dtype=torch.float32)
        entropy_by_seq = hidden.new_zeros((batch_size, seq_len), dtype=torch.float32) if calculate_entropy else None

        positions = torch.arange(seq_len, device=device, dtype=torch.long)
        response_starts = prompt_lengths.to(device=device, dtype=torch.long)
        response_ends = response_starts + response_lengths.to(device=device, dtype=torch.long)

        past_keys: list[torch.Tensor] = []
        past_values: list[torch.Tensor] = []
        response_lm_token_count = 0
        native_total_count = 0
        native_supported_count = 0
        native_top1_correct_count = 0
        ran_any_step = False

        draft_start_time = time.perf_counter()
        for step_idx in range(ttt_length):
            label_positions = positions + step_idx + 2
            valid = (
                (label_positions.unsqueeze(0) >= response_starts.unsqueeze(1))
                & (label_positions.unsqueeze(0) < response_ends.unsqueeze(1))
                & (label_positions.unsqueeze(0) < valid_seq_lens.unsqueeze(1))
            )
            if step_idx > 0 and not bool(valid.any()):
                break

            token_ids = _shift_left_2d(input_ids, step_idx + 1)
            input_embeds = self.draft_model.embed_input_ids(token_ids).to(dtype=hidden.dtype)
            hidden, key_states, value_states = torch.utils.checkpoint.checkpoint(
                self._eagle3_ttt_step,
                input_embeds,
                hidden,
                base_position_ids,
                additive_mask,
                step_idx,
                *past_keys,
                *past_values,
                use_reentrant=False,
            )
            past_keys.append(key_states)
            past_values.append(value_states)
            ran_any_step = True

            if not bool(valid.any()):
                continue

            batch_indices, position_indices = torch.nonzero(valid, as_tuple=True)
            total_count = int(batch_indices.numel())
            step_weight = float(ttt_step_weight) ** step_idx
            label_token_ids = input_ids[batch_indices, position_indices + step_idx + 2]
            draft_label_ids, scalar_supported = self._map_target_to_draft_ids(label_token_ids.to(dtype=torch.long))

            for start in range(0, total_count, lm_head_chunk_size):
                end = min(start + lm_head_chunk_size, total_count)
                chunk_batch = batch_indices[start:end]
                chunk_positions = position_indices[start:end]
                ce_losses, scalar_log_probs, native_supported, top1_match, chunk_entropy = (
                    torch.utils.checkpoint.checkpoint(
                        self._eagle3_loss_chunk,
                        hidden[chunk_batch, chunk_positions, :],
                        target_logits,
                        chunk_batch,
                        chunk_positions + step_idx + 1,
                        draft_label_ids[start:end],
                        t2d,
                        calculate_entropy and step_idx == 0,
                        use_reentrant=False,
                    )
                )

                supported_batch = chunk_batch[native_supported]
                supported_positions = chunk_positions[native_supported]
                native_ce_loss_by_seq[supported_batch, supported_positions] += (
                    step_weight * ce_losses[native_supported]
                )
                loss_mask_by_seq[supported_batch, supported_positions] = 1.0

                if step_idx == 0:
                    chunk_scalar_supported = scalar_supported[start:end]
                    if bool(chunk_scalar_supported.any()):
                        scalar_batch = chunk_batch[chunk_scalar_supported]
                        scalar_positions = chunk_positions[chunk_scalar_supported]
                        log_probs_by_seq[scalar_batch, scalar_positions] = scalar_log_probs[chunk_scalar_supported]
                        selected_scalar_loss_mask_by_seq[scalar_batch, scalar_positions] = 1.0
                        response_lm_token_count += int(chunk_scalar_supported.sum().item())
                    native_total_count += end - start
                    native_supported_count += int(native_supported.sum().item())
                    native_top1_correct_count += int(top1_match[native_supported].sum().item())
                    if entropy_by_seq is not None:
                        entropy_by_seq[supported_batch, supported_positions] = chunk_entropy[native_supported]

        self._maybe_sync_for_profile(profile_enabled, device)
        draft_forward_ms = (time.perf_counter() - draft_start_time) * 1000.0

        if not bool(loss_mask_by_seq.any()):
            trainable_param = next((param for param in self.draft_model.parameters() if param.requires_grad), None)
            if trainable_param is None:
                raise RuntimeError("EAGLE3 OPD requires at least one trainable draft parameter.")
            grad_anchor = trainable_param.flatten()[0].float() * 0.0
            native_ce_loss_by_seq = native_ce_loss_by_seq + grad_anchor
            log_probs_by_seq = log_probs_by_seq + grad_anchor

        return {
            "log_probs_by_seq": log_probs_by_seq,
            "loss_mask_by_seq": loss_mask_by_seq,
            "selected_scalar_loss_mask_by_seq": selected_scalar_loss_mask_by_seq,
            "native_ce_loss_by_seq": native_ce_loss_by_seq,
            "entropy_by_seq": entropy_by_seq,
            "response_lm_token_count": response_lm_token_count,
            "native_total_count": native_total_count,
            "native_supported_count": native_supported_count,
            "native_top1_correct_count": native_top1_correct_count,
            "ran_draft_forward": ran_any_step,
            "draft_forward_ms": draft_forward_ms,
            "draft_q_token_count": batch_size * seq_len * ttt_length,
        }

    def _run_eagle3_draft_forward(
        self,
        *,
        input_ids: torch.LongTensor,
        target_hidden: torch.Tensor,
        anchor_positions: torch.LongTensor,
        block_keep_mask: torch.Tensor,
        draft_block_size: int,
        forced_input_token_ids: Optional[torch.LongTensor],
        profile_enabled: bool,
    ) -> tuple[torch.Tensor, str, float]:
        batch_size, seq_len = input_ids.shape
        num_blocks = int(anchor_positions.shape[1])
        hidden_size = int(self.draft_model.config.hidden_size)
        safe_anchors = anchor_positions.clamp(0, max(seq_len - 1, 0))
        hidden_gather_index = safe_anchors.unsqueeze(-1).expand(batch_size, num_blocks, target_hidden.shape[-1])
        anchor_hidden = torch.gather(target_hidden, dim=1, index=hidden_gather_index).reshape(-1, target_hidden.shape[-1])
        hidden = self.draft_model.project_hidden_states(anchor_hidden).view(-1, 1, hidden_size)
        cache_hidden: list[list[torch.Tensor]] = [[], []]
        draft_hidden = target_hidden.new_zeros((batch_size, num_blocks * draft_block_size, hidden_size))

        draft_start_time = time.perf_counter()
        for offset in range(1, draft_block_size):
            token_positions = (anchor_positions + offset - 1).clamp(0, max(seq_len - 1, 0))
            token_ids = torch.gather(input_ids, dim=1, index=token_positions)
            if forced_input_token_ids is not None:
                forced_ids = forced_input_token_ids[:, :, offset]
                token_ids = torch.where(forced_ids >= 0, forced_ids, token_ids)
            input_embeds = self.draft_model.embed_input_ids(token_ids.reshape(-1)).view(-1, 1, hidden_size)
            position_ids = (anchor_positions + offset - 1).clamp_min(0).reshape(-1, 1)
            hidden = self.draft_model.backbone(
                input_embeds=input_embeds,
                hidden_states=hidden,
                cache_hidden=cache_hidden,
                position_ids=cast(torch.LongTensor, position_ids),
            )
            hidden_by_block = hidden.view(batch_size, num_blocks, hidden_size)
            flat_indices = torch.arange(num_blocks, device=input_ids.device) * draft_block_size + offset
            draft_hidden[:, flat_indices, :] = hidden_by_block

        self._maybe_sync_for_profile(profile_enabled, input_ids.device)
        return draft_hidden, "eager_unrolled", (time.perf_counter() - draft_start_time) * 1000.0

    def _build_rejected_forced_input_token_ids(
        self,
        *,
        input_ids: torch.LongTensor,
        prompt_lengths: torch.LongTensor,
        response_lengths: torch.LongTensor,
        anchor_positions: torch.LongTensor,
        block_keep_mask: torch.Tensor,
        draft_block_size: int,
        max_tokens_per_sample: Optional[int],
        rejected_draft_anchor_indices: Optional[torch.LongTensor],
        rejected_draft_offsets: Optional[torch.LongTensor],
        rejected_draft_token_ids: Optional[torch.LongTensor],
        rejected_draft_mask: Optional[torch.Tensor],
    ) -> torch.LongTensor:
        batch_size, num_blocks = anchor_positions.shape
        forced = torch.full((batch_size, num_blocks, draft_block_size), -1, dtype=torch.long, device=input_ids.device)
        if (
            rejected_draft_anchor_indices is None
            or rejected_draft_offsets is None
            or rejected_draft_token_ids is None
            or rejected_draft_mask is None
            or not bool(rejected_draft_mask.any())
        ):
            return forced

        rejected_width = int(rejected_draft_anchor_indices.shape[1])
        for batch_idx in range(batch_size):
            prompt_len = int(prompt_lengths[batch_idx].item())
            response_len = int(response_lengths[batch_idx].item())
            valid_len = prompt_len + response_len
            selected_count = 0
            for item_idx in range(rejected_width):
                if not bool(rejected_draft_mask[batch_idx, item_idx].item()):
                    continue
                if max_tokens_per_sample is not None and selected_count >= max_tokens_per_sample:
                    continue
                offset = int(rejected_draft_offsets[batch_idx, item_idx].item())
                token_id = int(rejected_draft_token_ids[batch_idx, item_idx].item())
                if offset <= 0 or offset >= draft_block_size or token_id < 0:
                    continue
                anchor_resp = int(rejected_draft_anchor_indices[batch_idx, item_idx].item())
                full_anchor = prompt_len - 1 if anchor_resp < 0 else prompt_len + anchor_resp
                if full_anchor < 0 or full_anchor >= valid_len:
                    continue
                block_matches = (anchor_positions[batch_idx] == full_anchor) & block_keep_mask[batch_idx]
                if not bool(block_matches.any()):
                    continue
                block_idx = int(torch.nonzero(block_matches, as_tuple=False)[0, 0].item())
                if offset + 1 < draft_block_size:
                    forced[batch_idx, block_idx, offset + 1] = token_id
                selected_count += 1
        return forced

    def _collect_eagle3_rejected_draft_log_probs(
        self,
        *,
        draft_hidden: torch.Tensor,
        prompt_lengths: torch.LongTensor,
        response_lengths: torch.LongTensor,
        anchor_positions: torch.LongTensor,
        block_keep_mask: torch.Tensor,
        draft_block_size: int,
        lm_head_chunk_size: int,
        max_tokens_per_sample: Optional[int],
        rejected_draft_anchor_indices: Optional[torch.LongTensor],
        rejected_draft_offsets: Optional[torch.LongTensor],
        rejected_draft_token_ids: Optional[torch.LongTensor],
        rejected_draft_teacher_logprobs: Optional[torch.Tensor],
        rejected_draft_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(prompt_lengths.shape[0])
        rejected_width = 1
        if rejected_draft_anchor_indices is not None and rejected_draft_anchor_indices.dim() >= 2:
            rejected_width = max(1, int(rejected_draft_anchor_indices.shape[1]))

        student_tensor = draft_hidden.new_zeros((batch_size, rejected_width), dtype=torch.float32)
        teacher_tensor = draft_hidden.new_zeros((batch_size, rejected_width), dtype=torch.float32)
        mask_tensor = torch.zeros((batch_size, rejected_width), dtype=torch.bool, device=draft_hidden.device)
        if (
            rejected_draft_anchor_indices is None
            or rejected_draft_offsets is None
            or rejected_draft_token_ids is None
            or rejected_draft_teacher_logprobs is None
            or rejected_draft_mask is None
            or not bool(rejected_draft_mask.any())
        ):
            return student_tensor, teacher_tensor, mask_tensor

        selected_batch_indices: list[int] = []
        selected_draft_indices: list[int] = []
        selected_token_ids: list[int] = []
        selected_item_indices: list[tuple[int, int]] = []
        selected_counts = [0 for _ in range(batch_size)]
        for batch_idx in range(batch_size):
            prompt_len = int(prompt_lengths[batch_idx].item())
            response_len = int(response_lengths[batch_idx].item())
            valid_len = prompt_len + response_len
            for item_idx in range(rejected_width):
                if not bool(rejected_draft_mask[batch_idx, item_idx].item()):
                    continue
                if max_tokens_per_sample is not None and selected_counts[batch_idx] >= max_tokens_per_sample:
                    continue
                offset = int(rejected_draft_offsets[batch_idx, item_idx].item())
                token_id = int(rejected_draft_token_ids[batch_idx, item_idx].item())
                if offset <= 0 or offset >= draft_block_size or token_id < 0:
                    continue
                anchor_resp = int(rejected_draft_anchor_indices[batch_idx, item_idx].item())
                full_anchor = prompt_len - 1 if anchor_resp < 0 else prompt_len + anchor_resp
                if full_anchor < 0 or full_anchor >= valid_len:
                    continue
                block_matches = (anchor_positions[batch_idx] == full_anchor) & block_keep_mask[batch_idx]
                if not bool(block_matches.any()):
                    continue
                block_idx = int(torch.nonzero(block_matches, as_tuple=False)[0, 0].item())
                selected_batch_indices.append(batch_idx)
                selected_draft_indices.append(block_idx * draft_block_size + offset)
                selected_token_ids.append(token_id)
                selected_item_indices.append((batch_idx, item_idx))
                selected_counts[batch_idx] += 1

        if selected_batch_indices:
            selected_batch_tensor = torch.tensor(selected_batch_indices, dtype=torch.long, device=draft_hidden.device)
            selected_draft_tensor = torch.tensor(selected_draft_indices, dtype=torch.long, device=draft_hidden.device)
            selected_token_tensor = torch.tensor(selected_token_ids, dtype=torch.long, device=draft_hidden.device)
            selected_log_probs, _, supported = self._compute_selected_eagle3_log_probs(
                draft_hidden=draft_hidden,
                batch_indices=selected_batch_tensor,
                draft_indices=selected_draft_tensor,
                target_token_ids=selected_token_tensor,
                chunk_size=lm_head_chunk_size,
                calculate_entropy=False,
            )
            supported_item_indices = [item for item, keep in zip(selected_item_indices, supported.tolist(), strict=False) if keep]
            if supported_item_indices:
                item_batch_indices = torch.tensor(
                    [batch_idx for batch_idx, _ in supported_item_indices],
                    dtype=torch.long,
                    device=draft_hidden.device,
                )
                item_column_indices = torch.tensor(
                    [item_idx for _, item_idx in supported_item_indices],
                    dtype=torch.long,
                    device=draft_hidden.device,
                )
                student_tensor[item_batch_indices, item_column_indices] = selected_log_probs
                teacher_tensor[item_batch_indices, item_column_indices] = rejected_draft_teacher_logprobs[
                    item_batch_indices, item_column_indices
                ].to(device=draft_hidden.device, dtype=torch.float32)
                mask_tensor[item_batch_indices, item_column_indices] = True
        return student_tensor, teacher_tensor, mask_tensor

    def _forward_opd(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        prompt_lengths: torch.LongTensor,
        response_lengths: torch.LongTensor,
        reject_token_indices: torch.LongTensor,
        rejected_draft_anchor_indices: Optional[torch.LongTensor] = None,
        rejected_draft_offsets: Optional[torch.LongTensor] = None,
        rejected_draft_token_ids: Optional[torch.LongTensor] = None,
        rejected_draft_teacher_logprobs: Optional[torch.Tensor] = None,
        rejected_draft_mask: Optional[torch.Tensor] = None,
        calculate_entropy: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if input_ids.dim() != 2:
            raise ValueError(f"EAGLE3 OPD requires padded 2D input_ids, got shape={tuple(input_ids.shape)}.")

        profile_enabled = self._is_dflash_profiling_enabled()
        self._maybe_sync_for_profile(profile_enabled, input_ids.device)
        total_start_time = time.perf_counter()

        target_kwargs = dict(kwargs)
        for key in (
            "dflash_prompt_lengths",
            "dflash_response_lengths",
            "dflash_reject_token_indices",
            "dflash_rejected_draft_anchor_indices",
            "dflash_rejected_draft_offsets",
            "dflash_rejected_draft_token_ids",
            "dflash_rejected_draft_teacher_logprobs",
            "dflash_rejected_draft_mask",
            "dflash_calculate_entropy",
        ):
            target_kwargs.pop(key, None)

        teacher_start_time = time.perf_counter()
        with torch.no_grad():
            teacher_outputs = self.main_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
                **target_kwargs,
            )
            if teacher_outputs.hidden_states is None:
                raise RuntimeError("Teacher model did not return hidden states required by EAGLE3 draft.")
            target_hidden = self._extract_target_hidden(teacher_outputs.hidden_states)
            target_logits = getattr(teacher_outputs, "logits", None)
        self._maybe_sync_for_profile(profile_enabled, input_ids.device)
        teacher_forward_ms = (time.perf_counter() - teacher_start_time) * 1000.0

        draft_block_size = self._get_block_size()
        lm_head_chunk_size = self._get_lm_head_chunk_size()
        response_anchor_stride = self._get_response_anchor_stride()
        response_loss_mode = self._get_response_loss_mode()
        random_response_anchor_enabled = self._get_random_response_anchor_enabled()
        random_response_anchor_seed = self._get_random_response_anchor_seed()
        rejected_draft_max_tokens_per_sample = self._get_rejected_draft_max_tokens_per_sample()
        response_stream_impl = self._get_response_stream_impl()
        if response_stream_impl == "full_ttt" and response_loss_mode != "native_target_distribution":
            raise RuntimeError(
                "verl_eagle3_response_stream_impl=full_ttt requires "
                "verl_eagle3_response_loss_mode=native_target_distribution."
            )
        if response_loss_mode == "native_target_distribution" and target_logits is None:
            raise RuntimeError("EAGLE3 native response loss requires target model logits.")

        anchor_positions, segment_lens, row_starts, block_keep_mask, valid_seq_lens, opd_metrics = (
            self._build_opd_anchor_plan(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_lengths=prompt_lengths,
                response_lengths=response_lengths,
                reject_token_indices=reject_token_indices,
                draft_block_size=draft_block_size,
                response_anchor_stride=response_anchor_stride,
                random_response_anchor_enabled=random_response_anchor_enabled,
                random_response_anchor_seed=random_response_anchor_seed,
                rejected_draft_anchor_indices=rejected_draft_anchor_indices,
                rejected_draft_offsets=rejected_draft_offsets,
                rejected_draft_mask=rejected_draft_mask,
                include_rejected_draft_anchors=False,
            )
        )

        batch_size, seq_len = input_ids.shape
        rejected_anchor_positions, rejected_block_keep_mask = self._build_rejected_draft_anchor_plan(
            input_ids=input_ids,
            prompt_lengths=prompt_lengths,
            response_lengths=response_lengths,
            draft_block_size=draft_block_size,
            max_tokens_per_sample=rejected_draft_max_tokens_per_sample,
            rejected_draft_anchor_indices=rejected_draft_anchor_indices,
            rejected_draft_offsets=rejected_draft_offsets,
            rejected_draft_token_ids=rejected_draft_token_ids,
            rejected_draft_mask=rejected_draft_mask,
        )

        response_actual_block_count = block_keep_mask.sum()
        rejected_actual_block_count = rejected_block_keep_mask.sum()
        response_padded_block_count = batch_size * int(anchor_positions.shape[1]) if bool(block_keep_mask.any()) else 0
        rejected_padded_block_count = (
            batch_size * int(rejected_anchor_positions.shape[1]) if bool(rejected_block_keep_mask.any()) else 0
        )
        padded_block_count = response_padded_block_count + rejected_padded_block_count
        actual_block_count = response_actual_block_count + rejected_actual_block_count
        max_blocks_per_sample = (block_keep_mask.sum(dim=1) + rejected_block_keep_mask.sum(dim=1)).max().to(
            dtype=torch.float32
        )
        draft_q_token_count = padded_block_count * draft_block_size

        log_probs_by_seq = target_hidden.new_zeros((batch_size, seq_len), dtype=torch.float32)
        loss_mask_by_seq = target_hidden.new_zeros((batch_size, seq_len), dtype=torch.float32)
        selected_scalar_loss_mask_by_seq = target_hidden.new_zeros((batch_size, seq_len), dtype=torch.float32)
        native_ce_loss_by_seq = target_hidden.new_zeros((batch_size, seq_len), dtype=torch.float32)
        entropy_by_seq = (
            target_hidden.new_zeros((batch_size, seq_len), dtype=torch.float32) if calculate_entropy else None
        )

        rejected_width = 1
        if rejected_draft_anchor_indices is not None and rejected_draft_anchor_indices.dim() >= 2:
            rejected_width = max(1, int(rejected_draft_anchor_indices.shape[1]))
        rejected_student_log_probs = target_hidden.new_zeros((batch_size, rejected_width), dtype=torch.float32)
        rejected_teacher_log_probs = target_hidden.new_zeros((batch_size, rejected_width), dtype=torch.float32)
        rejected_loss_mask = torch.zeros((batch_size, rejected_width), dtype=torch.bool, device=input_ids.device)

        draft_forward_ms = 0.0
        response_lm_token_count = 0
        native_total_count = 0
        native_supported_count = 0
        native_top1_correct_count = 0
        attn_impl = "eager_unrolled"
        ran_draft_forward = False

        lm_head_start_time = time.perf_counter()
        if response_stream_impl == "full_ttt":
            ttt_result = self._run_eagle3_full_ttt_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                target_hidden=target_hidden,
                target_logits=target_logits,
                prompt_lengths=prompt_lengths,
                response_lengths=response_lengths,
                ttt_length=self._get_ttt_length(),
                ttt_step_weight=self._get_ttt_step_weight(),
                lm_head_chunk_size=lm_head_chunk_size,
                calculate_entropy=calculate_entropy,
                profile_enabled=profile_enabled,
            )
            log_probs_by_seq = ttt_result["log_probs_by_seq"]
            loss_mask_by_seq = ttt_result["loss_mask_by_seq"]
            selected_scalar_loss_mask_by_seq = ttt_result["selected_scalar_loss_mask_by_seq"]
            native_ce_loss_by_seq = ttt_result["native_ce_loss_by_seq"]
            if calculate_entropy and ttt_result["entropy_by_seq"] is not None:
                entropy_by_seq = ttt_result["entropy_by_seq"]
            response_lm_token_count = ttt_result["response_lm_token_count"]
            native_total_count = ttt_result["native_total_count"]
            native_supported_count = ttt_result["native_supported_count"]
            native_top1_correct_count = ttt_result["native_top1_correct_count"]
            draft_forward_ms += ttt_result["draft_forward_ms"]
            draft_q_token_count = ttt_result["draft_q_token_count"] + rejected_padded_block_count * draft_block_size
            ran_draft_forward = bool(ttt_result["ran_draft_forward"])
            attn_impl = "ttt_full"
        elif bool(block_keep_mask.any()):
            draft_hidden, attn_impl, response_draft_ms = self._run_eagle3_draft_forward(
                input_ids=input_ids,
                target_hidden=target_hidden,
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                draft_block_size=draft_block_size,
                forced_input_token_ids=None,
                profile_enabled=profile_enabled,
            )
            draft_forward_ms += response_draft_ms
            ran_draft_forward = True

            response_batch_indices: list[torch.Tensor] = []
            response_draft_indices: list[torch.Tensor] = []
            response_row_indices: list[torch.Tensor] = []
            response_labels: list[torch.Tensor] = []
            for block_offset in range(1, draft_block_size):
                active_blocks = block_keep_mask & (segment_lens >= block_offset)
                if not bool(active_blocks.any()):
                    continue
                block_indices = torch.nonzero(active_blocks, as_tuple=False)
                batch_indices = block_indices[:, 0]
                anchor_block_indices = block_indices[:, 1]
                row_indices = row_starts[batch_indices, anchor_block_indices] + (block_offset - 1)
                label_indices = row_indices + 1
                in_bounds = label_indices < valid_seq_lens[batch_indices]
                if not bool(in_bounds.any()):
                    continue
                batch_indices = batch_indices[in_bounds]
                anchor_block_indices = anchor_block_indices[in_bounds]
                row_indices = row_indices[in_bounds]
                label_indices = label_indices[in_bounds]
                flat_draft_indices = anchor_block_indices * draft_block_size + block_offset
                response_batch_indices.append(batch_indices)
                response_draft_indices.append(flat_draft_indices)
                response_row_indices.append(row_indices)
                response_labels.append(input_ids[batch_indices, label_indices])

            if response_batch_indices:
                response_batch_tensor = torch.cat(response_batch_indices, dim=0)
                response_draft_tensor = torch.cat(response_draft_indices, dim=0)
                response_row_tensor = torch.cat(response_row_indices, dim=0)
                response_label_tensor = torch.cat(response_labels, dim=0)
                if response_loss_mode == "native_target_distribution":
                    (
                        selected_log_probs,
                        selected_entropy,
                        scalar_supported,
                        native_losses,
                        native_supported,
                        native_top1_match,
                    ) = self._compute_eagle3_native_response_outputs(
                        draft_hidden=draft_hidden,
                        target_logits=target_logits,
                        batch_indices=response_batch_tensor,
                        draft_indices=response_draft_tensor,
                        target_row_indices=response_row_tensor,
                        target_token_ids=response_label_tensor,
                        chunk_size=lm_head_chunk_size,
                        calculate_entropy=calculate_entropy,
                    )
                    if bool(scalar_supported.any()):
                        scalar_supported_batch = response_batch_tensor[scalar_supported]
                        scalar_supported_rows = response_row_tensor[scalar_supported]
                        log_probs_by_seq[scalar_supported_batch, scalar_supported_rows] = selected_log_probs[
                            scalar_supported
                        ]
                        selected_scalar_loss_mask_by_seq[scalar_supported_batch, scalar_supported_rows] = 1.0
                    if bool(native_supported.any()):
                        native_supported_batch = response_batch_tensor[native_supported]
                        native_supported_rows = response_row_tensor[native_supported]
                        native_ce_loss_by_seq[native_supported_batch, native_supported_rows] = native_losses[
                            native_supported
                        ]
                        loss_mask_by_seq[native_supported_batch, native_supported_rows] = 1.0
                        if entropy_by_seq is not None and selected_entropy is not None:
                            entropy_by_seq[native_supported_batch, native_supported_rows] = selected_entropy[
                                native_supported
                            ]
                    response_lm_token_count = int(scalar_supported.sum().item())
                    native_total_count = int(native_supported.numel())
                    native_supported_count = int(native_supported.sum().item())
                    native_top1_correct_count = int(native_top1_match[native_supported].sum().item())
                else:
                    selected_log_probs, selected_entropy, supported = self._compute_selected_eagle3_log_probs(
                        draft_hidden=draft_hidden,
                        batch_indices=response_batch_tensor,
                        draft_indices=response_draft_tensor,
                        target_token_ids=response_label_tensor,
                        chunk_size=lm_head_chunk_size,
                        calculate_entropy=calculate_entropy,
                    )
                    if bool(supported.any()):
                        supported_batch = response_batch_tensor[supported]
                        supported_rows = response_row_tensor[supported]
                        log_probs_by_seq[supported_batch, supported_rows] = selected_log_probs
                        loss_mask_by_seq[supported_batch, supported_rows] = 1.0
                        selected_scalar_loss_mask_by_seq[supported_batch, supported_rows] = 1.0
                        if entropy_by_seq is not None and selected_entropy is not None:
                            entropy_by_seq[supported_batch, supported_rows] = selected_entropy
                        response_lm_token_count = supported_batch.numel()
            del draft_hidden

        # NOTE: the rejected-draft stream below (and the legacy anchor_block response stream)
        # keeps the original block-chain alignment, where the embedding of t_a is paired with
        # the target feature f_a. The SpecForge/SGLang EAGLE3 convention pairs e(t_{a+1}) with
        # f_a instead (see _run_eagle3_full_ttt_forward). Revisit this alignment before enabling
        # rejected_draft_stream_weight > 0 for EAGLE3.
        if bool(rejected_block_keep_mask.any()):
            forced_input_token_ids = self._build_rejected_forced_input_token_ids(
                input_ids=input_ids,
                prompt_lengths=prompt_lengths,
                response_lengths=response_lengths,
                anchor_positions=rejected_anchor_positions,
                block_keep_mask=rejected_block_keep_mask,
                draft_block_size=draft_block_size,
                max_tokens_per_sample=rejected_draft_max_tokens_per_sample,
                rejected_draft_anchor_indices=rejected_draft_anchor_indices,
                rejected_draft_offsets=rejected_draft_offsets,
                rejected_draft_token_ids=rejected_draft_token_ids,
                rejected_draft_mask=rejected_draft_mask,
            )
            rejected_draft_hidden, attn_impl, rejected_draft_ms = self._run_eagle3_draft_forward(
                input_ids=input_ids,
                target_hidden=target_hidden,
                anchor_positions=rejected_anchor_positions,
                block_keep_mask=rejected_block_keep_mask,
                draft_block_size=draft_block_size,
                forced_input_token_ids=forced_input_token_ids,
                profile_enabled=profile_enabled,
            )
            draft_forward_ms += rejected_draft_ms
            ran_draft_forward = True
            rejected_student_log_probs, rejected_teacher_log_probs, rejected_loss_mask = (
                self._collect_eagle3_rejected_draft_log_probs(
                    draft_hidden=rejected_draft_hidden,
                    prompt_lengths=prompt_lengths,
                    response_lengths=response_lengths,
                    anchor_positions=rejected_anchor_positions,
                    block_keep_mask=rejected_block_keep_mask,
                    draft_block_size=draft_block_size,
                    lm_head_chunk_size=lm_head_chunk_size,
                    max_tokens_per_sample=rejected_draft_max_tokens_per_sample,
                    rejected_draft_anchor_indices=rejected_draft_anchor_indices,
                    rejected_draft_offsets=rejected_draft_offsets,
                    rejected_draft_token_ids=rejected_draft_token_ids,
                    rejected_draft_teacher_logprobs=rejected_draft_teacher_logprobs,
                    rejected_draft_mask=rejected_draft_mask,
                )
            )
            del rejected_draft_hidden

        if not ran_draft_forward:
            trainable_param = next((param for param in self.draft_model.parameters() if param.requires_grad), None)
            if trainable_param is None:
                raise RuntimeError("EAGLE3 OPD requires at least one trainable draft parameter.")
            log_probs_by_seq = log_probs_by_seq + trainable_param.flatten()[0].float() * 0.0

        lm_head_ms = (time.perf_counter() - lm_head_start_time) * 1000.0
        rejected_lm_token_count = rejected_loss_mask.sum()
        selected_lm_token_count = loss_mask_by_seq.sum() + rejected_lm_token_count
        output = {
            "dflash_log_probs": log_probs_by_seq,
            "dflash_loss_mask": loss_mask_by_seq,
            "eagle3_response_loss_mode_id": log_probs_by_seq.new_tensor(
                1 if response_loss_mode == "native_target_distribution" else 0
            ),
            "dflash_opd_valid_anchor_count": log_probs_by_seq.new_tensor(opd_metrics["valid_anchor_count"]),
            "dflash_opd_skipped_sample_count": log_probs_by_seq.new_tensor(opd_metrics["skipped_sample_count"]),
            "dflash_opd_empty_reject_sample_count": log_probs_by_seq.new_tensor(
                opd_metrics["empty_reject_sample_count"]
            ),
            "dflash_opd_total_reject_count": log_probs_by_seq.new_tensor(opd_metrics["total_reject_count"]),
            "dflash_opd_sample_count": log_probs_by_seq.new_tensor(opd_metrics["sample_count"]),
            "dflash_opd_target_token_count": loss_mask_by_seq.sum(),
            "dflash_rejected_draft_student_log_probs": rejected_student_log_probs,
            "dflash_rejected_draft_teacher_log_probs": rejected_teacher_log_probs,
            "dflash_rejected_draft_loss_mask": rejected_loss_mask,
            "dflash_rejected_draft_offsets": rejected_draft_offsets
            if rejected_draft_offsets is not None
            else torch.zeros_like(rejected_loss_mask, dtype=torch.long),
            "dflash_opd_rejected_draft_token_count": rejected_lm_token_count,
            "dflash_opd_actual_block_count": actual_block_count.to(dtype=torch.float32),
            "dflash_opd_padded_block_count": log_probs_by_seq.new_tensor(padded_block_count),
            "dflash_opd_max_blocks_per_sample": max_blocks_per_sample,
            "dflash_opd_draft_q_token_count": log_probs_by_seq.new_tensor(draft_q_token_count),
            "dflash_opd_response_lm_token_count": log_probs_by_seq.new_tensor(response_lm_token_count),
            "dflash_opd_rejected_lm_token_count": rejected_lm_token_count,
            "dflash_opd_selected_lm_token_count": selected_lm_token_count,
            "dflash_opd_lm_head_chunk_size": log_probs_by_seq.new_tensor(lm_head_chunk_size),
            "dflash_opd_response_anchor_stride": log_probs_by_seq.new_tensor(response_anchor_stride),
            "dflash_opd_rejected_draft_max_tokens_per_sample": log_probs_by_seq.new_tensor(
                rejected_draft_max_tokens_per_sample or 0
            ),
            "dflash_opd_attention_impl_id": log_probs_by_seq.new_tensor(
                EAGLE3_ATTENTION_IMPL_IDS.get(attn_impl, -1)
            ),
        }
        if response_loss_mode == "native_target_distribution":
            output.update(
                {
                    "eagle3_native_ce_losses": native_ce_loss_by_seq,
                    "eagle3_selected_scalar_loss_mask": selected_scalar_loss_mask_by_seq,
                    "eagle3_target_argmax_total_count": log_probs_by_seq.new_tensor(native_total_count),
                    "eagle3_target_argmax_supported_count": log_probs_by_seq.new_tensor(native_supported_count),
                    "eagle3_draft_target_top1_correct_count": log_probs_by_seq.new_tensor(
                        native_top1_correct_count
                    ),
                }
            )
        if profile_enabled:
            output.update(
                {
                    "dflash_opd_profile_teacher_forward_ms": log_probs_by_seq.new_tensor(teacher_forward_ms),
                    "dflash_opd_profile_draft_forward_ms": log_probs_by_seq.new_tensor(draft_forward_ms),
                    "dflash_opd_profile_lm_head_ms": log_probs_by_seq.new_tensor(lm_head_ms),
                    "dflash_opd_profile_total_forward_ms": log_probs_by_seq.new_tensor(
                        (time.perf_counter() - total_start_time) * 1000.0
                    ),
                }
            )
        if entropy_by_seq is not None:
            output["dflash_entropy"] = entropy_by_seq
        return output

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        dflash_prompt_lengths: Optional[torch.LongTensor] = None,
        dflash_response_lengths: Optional[torch.LongTensor] = None,
        dflash_reject_token_indices: Optional[torch.LongTensor] = None,
        dflash_rejected_draft_anchor_indices: Optional[torch.LongTensor] = None,
        dflash_rejected_draft_offsets: Optional[torch.LongTensor] = None,
        dflash_rejected_draft_token_ids: Optional[torch.LongTensor] = None,
        dflash_rejected_draft_teacher_logprobs: Optional[torch.Tensor] = None,
        dflash_rejected_draft_mask: Optional[torch.Tensor] = None,
        dflash_calculate_entropy: bool = False,
        **kwargs,
    ):
        if input_ids is None and inputs_embeds is None:
            raise ValueError("ComposedEagle3StudentForCausalLM requires either input_ids or inputs_embeds.")

        if dflash_reject_token_indices is not None:
            if dflash_prompt_lengths is None or dflash_response_lengths is None:
                raise ValueError("EAGLE3 OPD requires prompt and response lengths with reject-token indices.")
            if input_ids is None:
                raise ValueError("EAGLE3 OPD path requires input_ids.")
            return self._forward_opd(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                prompt_lengths=dflash_prompt_lengths,
                response_lengths=dflash_response_lengths,
                reject_token_indices=dflash_reject_token_indices,
                rejected_draft_anchor_indices=dflash_rejected_draft_anchor_indices,
                rejected_draft_offsets=dflash_rejected_draft_offsets,
                rejected_draft_token_ids=dflash_rejected_draft_token_ids,
                rejected_draft_teacher_logprobs=dflash_rejected_draft_teacher_logprobs,
                rejected_draft_mask=dflash_rejected_draft_mask,
                calculate_entropy=bool(dflash_calculate_entropy),
                **kwargs,
            )

        with torch.no_grad():
            return self.main_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )


def build_composed_eagle3_student(
    *,
    main_model_path: str,
    draft_model_path: str,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
    config: PretrainedConfig,
) -> ComposedEagle3StudentForCausalLM:
    main_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=main_model_path,
        torch_dtype=torch_dtype,
        config=config,
        trust_remote_code=trust_remote_code,
    )
    draft_config = AutoConfig.from_pretrained(draft_model_path, trust_remote_code=True)
    draft_model = Eagle3DraftModel(config=draft_config)
    state_dict = load_safetensors_file(os.path.join(draft_model_path, "model.safetensors"))
    missing_keys, unexpected_keys = draft_model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        logger.warning("Unexpected EAGLE3 draft checkpoint keys: %s", unexpected_keys)
    missing_without_embedding = [key for key in missing_keys if not key.startswith("embed_tokens.")]
    if missing_without_embedding:
        logger.warning("Missing EAGLE3 draft checkpoint keys: %s", missing_without_embedding)

    model = ComposedEagle3StudentForCausalLM(
        config=copy.deepcopy(config),
        main_model=main_model,
        draft_model=draft_model,
    )
    return model
