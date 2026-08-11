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
import os
import random
import time
from contextlib import nullcontext
from typing import Any, Callable, Optional, cast

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DFLASH_ATTENTION_IMPL_IDS = {
    "eager": 0,
    "sdpa": 1,
    "flex_attention": 2,
}


def full_vocab_tv_distance(draft_logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    """Compute exact total-variation distance in FP32.

    The target distribution is deliberately detached so gradients can only flow
    through the draft distribution.
    """
    if draft_logits.shape != target_logits.shape:
        raise ValueError(
            "Draft and target logits must have identical shapes for full-vocabulary TV, got "
            f"draft={tuple(draft_logits.shape)}, target={tuple(target_logits.shape)}."
        )
    draft_probs = F.softmax(draft_logits.float(), dim=-1)
    target_probs = F.softmax(target_logits.detach().float(), dim=-1)
    return 0.5 * (draft_probs - target_probs).abs().sum(dim=-1)

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    BlockMask = None
    create_block_mask = None
    FLEX_ATTENTION_AVAILABLE = False


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [int(round(start + (i * span) / (num_draft_layers - 1))) for i in range(num_draft_layers)]


def resolve_target_layer_ids(main_model: PreTrainedModel, draft_model: PreTrainedModel) -> list[int]:
    if hasattr(draft_model, "target_layer_ids") and draft_model.target_layer_ids is not None:
        return [int(layer_id) for layer_id in draft_model.target_layer_ids]

    draft_config = getattr(draft_model, "config", None)
    dflash_config = getattr(draft_config, "dflash_config", None) if draft_config is not None else None
    if isinstance(dflash_config, dict) and dflash_config.get("target_layer_ids") is not None:
        return [int(layer_id) for layer_id in dflash_config["target_layer_ids"]]

    num_target_layers = int(main_model.config.num_hidden_layers)
    num_draft_layers = int(draft_model.config.num_hidden_layers)
    return build_target_layer_ids(num_target_layers=num_target_layers, num_draft_layers=num_draft_layers)


def create_dflash_sdpa_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    seq_len: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    batch_size, num_blocks = anchor_positions.shape
    q_len = num_blocks * block_size
    kv_len = seq_len + q_len

    q_indices = torch.arange(q_len, device=device).view(1, 1, -1, 1)
    kv_indices = torch.arange(kv_len, device=device).view(1, 1, 1, -1)
    q_block_ids = q_indices // block_size

    anchor_expanded = anchor_positions.view(batch_size, 1, num_blocks, 1).repeat_interleave(block_size, dim=2)
    mask_context = (kv_indices < seq_len) & (kv_indices < anchor_expanded)

    is_draft = kv_indices >= seq_len
    kv_block_ids = (kv_indices - seq_len) // block_size
    mask_draft = is_draft & (q_block_ids == kv_block_ids)

    valid_block = block_keep_mask.view(batch_size, 1, num_blocks, 1).repeat_interleave(block_size, dim=2)
    return (mask_context | mask_draft) & valid_block


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    seq_len: int,
    block_size: int,
    device: torch.device,
) -> BlockMask:
    if not FLEX_ATTENTION_AVAILABLE or create_block_mask is None:
        raise RuntimeError("flex_attention is not available for DFLASH block mask construction.")

    batch_size, num_blocks = anchor_positions.shape
    q_len = num_blocks * block_size
    kv_len = seq_len + q_len

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        safe_q_block_id = q_block_id.clamp(max=num_blocks - 1)
        anchor_pos = anchor_positions[b, safe_q_block_id]

        is_context = kv_idx < seq_len
        mask_context = is_context & (kv_idx < anchor_pos)

        is_draft = kv_idx >= seq_len
        kv_block_id = (kv_idx - seq_len) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)

        is_valid_block = block_keep_mask[b, safe_q_block_id]
        in_bounds = q_block_id < num_blocks
        return (mask_context | mask_draft) & is_valid_block & in_bounds

    return create_block_mask(
        dflash_mask_mod, B=batch_size, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device
    )


def _set_config_attn_implementation(config: Any, attn_impl: str) -> None:
    for attr_name in ("_attn_implementation", "_attn_implementation_internal", "attn_implementation"):
        if hasattr(config, attr_name):
            setattr(config, attr_name, attn_impl)


class ComposedDFlashStudentForCausalLM(PreTrainedModel):
    """Train-only composed model: frozen target model + trainable DFLASH draft."""

    config_class = PretrainedConfig
    base_model_prefix = "main_model"
    supports_gradient_checkpointing = True

    @staticmethod
    def _normalize_wrapper_attn_implementation(config: PretrainedConfig) -> PretrainedConfig:
        """Force wrapper-level attention impl to eager for HF init checks.

        The composed student wrapper itself has no native attention blocks, but the
        inherited PreTrainedModel init path still validates attn implementation and
        rejects flash_attention_2 for unknown architectures.
        """
        wrapper_config = copy.deepcopy(config)
        for attr_name in ("_attn_implementation", "_attn_implementation_internal", "attn_implementation"):
            if hasattr(wrapper_config, attr_name):
                setattr(wrapper_config, attr_name, "eager")
        return wrapper_config

    def __init__(
        self,
        config: PretrainedConfig,
        main_model: PreTrainedModel,
        draft_model: PreTrainedModel,
    ):
        super().__init__(self._normalize_wrapper_attn_implementation(config))
        self.main_model = main_model
        self.draft_model = draft_model
        self.target_layer_ids = resolve_target_layer_ids(main_model=main_model, draft_model=draft_model)
        self._configure_draft_attention()
        self.freeze_main_model()

    def _configure_draft_attention(self) -> None:
        requested_impl = (
            getattr(self.config, "verl_dflash_attention_impl", None)
            or os.getenv("VERL_DFLASH_ATTENTION_IMPL")
            or "flex_attention"
        )
        requested_impl = str(requested_impl).lower()
        if requested_impl == "auto":
            requested_impl = "flex_attention"

        if requested_impl == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            logger.warning("flex_attention is not available; falling back to sdpa for DFLASH draft attention.")
            requested_impl = "sdpa"

        if requested_impl not in DFLASH_ATTENTION_IMPL_IDS:
            logger.warning(
                "Unsupported DFLASH draft attention implementation %s. "
                "Arbitrary DFLASH block masks are only wired for flex_attention, sdpa, or eager; falling back to %s.",
                requested_impl,
                "flex_attention" if FLEX_ATTENTION_AVAILABLE else "sdpa",
            )
            requested_impl = "flex_attention" if FLEX_ATTENTION_AVAILABLE else "sdpa"

        draft_config = getattr(self.draft_model, "config", None)
        if draft_config is not None:
            _set_config_attn_implementation(draft_config, requested_impl)

    def freeze_main_model(self) -> None:
        for param in self.main_model.parameters():
            param.requires_grad = False
        self.main_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the frozen teacher in eval mode even while training the draft module.
        self.main_model.eval()
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for wrapped submodules."""
        super().gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        kwargs = gradient_checkpointing_kwargs or {}
        for module in (self.main_model, self.draft_model):
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for wrapped submodules."""
        super().gradient_checkpointing_disable()
        for module in (self.main_model, self.draft_model):
            if hasattr(module, "gradient_checkpointing_disable"):
                module.gradient_checkpointing_disable()

    def _set_gradient_checkpointing(
        self,
        enable: bool = True,
        gradient_checkpointing_func: Optional[Callable[..., Any]] = None,
    ):
        """HF hook for toggling gradient checkpointing support."""
        self.gradient_checkpointing = enable
        if hasattr(self, "config"):
            self.config.gradient_checkpointing = enable

        for child in (self.main_model, self.draft_model):
            if hasattr(child, "_set_gradient_checkpointing"):
                try:
                    if gradient_checkpointing_func is None:
                        child._set_gradient_checkpointing(enable=enable)
                    else:
                        child._set_gradient_checkpointing(
                            enable=enable,
                            gradient_checkpointing_func=gradient_checkpointing_func,
                        )
                except TypeError:
                    # Compatibility for older HF-style signatures used by some remote-code models.
                    child._set_gradient_checkpointing(enable)
            else:
                child.gradient_checkpointing = enable
                if hasattr(child, "config"):
                    child.config.gradient_checkpointing = enable

    def get_input_embeddings(self):
        return self.main_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.main_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.main_model.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.main_model.set_output_embeddings(value)

    def _extract_target_hidden(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        selected_states = []
        for layer_id in self.target_layer_ids:
            hidden_index = int(layer_id) + 1
            if hidden_index >= len(hidden_states):
                raise ValueError(
                    f"target_layer_id={layer_id} out of range for hidden_states size={len(hidden_states)}"
                )
            selected_states.append(hidden_states[hidden_index])
        return torch.cat(selected_states, dim=-1)

    def _resolve_position_ids(
        self,
        noise_embedding: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
    ) -> torch.LongTensor:
        if position_ids is not None:
            return position_ids
        batch_size, seq_len = noise_embedding.shape[:2]
        position_ids_tensor = (
            torch.arange(seq_len, device=noise_embedding.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        )
        return cast(torch.LongTensor, position_ids_tensor)

    def _get_mask_token_id(self) -> int:
        mask_token_id = getattr(self.draft_model, "mask_token_id", None)
        if mask_token_id is None:
            draft_config = getattr(self.draft_model, "config", None)
            dflash_config = getattr(draft_config, "dflash_config", None) if draft_config is not None else None
            if isinstance(dflash_config, dict):
                mask_token_id = dflash_config.get("mask_token_id")
        if mask_token_id is None:
            raise ValueError(
                "DFLASH OPD requires draft_model.mask_token_id or config.dflash_config['mask_token_id']."
            )
        return int(mask_token_id)

    def _get_block_size(self) -> int:
        block_size = getattr(self.draft_model, "block_size", None)
        if block_size is None:
            block_size = getattr(getattr(self.draft_model, "config", None), "block_size", None)
        if block_size is None:
            raise ValueError("DFLASH OPD requires draft_model.block_size or draft_model.config.block_size.")
        return int(block_size)

    def _get_lm_head_chunk_size(self) -> int:
        chunk_size = getattr(self.config, "verl_dflash_lm_head_chunk_size", None)
        if chunk_size is None:
            chunk_size = os.getenv("VERL_DFLASH_LM_HEAD_CHUNK_SIZE", "2048")
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError(f"verl_dflash_lm_head_chunk_size must be positive, got {chunk_size}.")
        return chunk_size

    def _get_response_anchor_stride(self) -> int:
        stride = getattr(self.config, "verl_dflash_response_anchor_stride", None)
        if stride is None:
            stride = os.getenv("VERL_DFLASH_RESPONSE_ANCHOR_STRIDE", "1")
        stride = int(stride)
        if stride <= 0:
            raise ValueError(f"verl_dflash_response_anchor_stride must be positive, got {stride}.")
        return stride

    def _get_random_response_anchor_enabled(self) -> bool:
        value = getattr(self.config, "verl_dflash_random_response_anchor_enabled", None)
        if value is None:
            value = os.getenv("VERL_DFLASH_RANDOM_RESPONSE_ANCHOR_ENABLED", "0")
        return str(value).lower() in {"1", "true", "yes", "on"}

    def _get_random_response_anchor_seed(self) -> int:
        value = getattr(self.config, "verl_dflash_random_response_anchor_seed", None)
        if value is None:
            value = os.getenv("VERL_DFLASH_RANDOM_RESPONSE_ANCHOR_SEED", "42")
        return int(value)

    def _get_rejected_draft_max_tokens_per_sample(self) -> Optional[int]:
        value = getattr(self.config, "verl_dflash_rejected_draft_max_tokens_per_sample", None)
        if value is None:
            value = os.getenv("VERL_DFLASH_REJECTED_DRAFT_MAX_TOKENS_PER_SAMPLE")
        if value is None or str(value).lower() in {"", "none", "null"}:
            return None
        value = int(value)
        if value <= 0:
            raise ValueError(f"verl_dflash_rejected_draft_max_tokens_per_sample must be positive, got {value}.")
        return value

    def _is_dflash_profiling_enabled(self) -> bool:
        return os.getenv("VERL_DFLASH_PROFILE", "0").lower() in {"1", "true", "yes", "on"}

    def _maybe_sync_for_profile(self, enabled: bool, device: torch.device) -> None:
        if enabled and device.type == "cuda":
            torch.cuda.synchronize(device)

    def _is_oom_error(self, exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return "out of memory" in message or "cuda error: out of memory" in message

    def _draft_sdpa_context(self):
        if torch.cuda.is_available():
            return torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=True,
                enable_mem_efficient=True,
                enable_cudnn=False,
            )
        return nullcontext()

    def _create_position_ids_for_anchors(self, anchor_positions: torch.Tensor, block_size: int) -> torch.Tensor:
        batch_size, num_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(block_size, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(batch_size, -1)

    def _create_noise_embedding_for_anchors(
        self,
        input_ids: torch.LongTensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        mask_token_id = self._get_mask_token_id()
        num_blocks = anchor_positions.shape[1]

        noise_ids = torch.full(
            (batch_size, num_blocks * block_size),
            mask_token_id,
            dtype=torch.long,
            device=device,
        )
        block_starts = torch.arange(num_blocks, device=device) * block_size
        block_starts = block_starts.unsqueeze(0).expand(batch_size, -1)

        safe_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, safe_anchor_positions)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, num_blocks)
        noise_ids[batch_indices, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(mask_token_id, dtype=torch.long, device=device),
        )
        return self.get_input_embeddings()(noise_ids)

    def _compute_selected_lm_log_probs(
        self,
        *,
        draft_hidden: torch.Tensor,
        output_embeddings: torch.nn.Module,
        batch_indices: torch.LongTensor,
        draft_indices: torch.LongTensor,
        token_ids: torch.LongTensor,
        chunk_size: int,
        calculate_entropy: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if batch_indices.numel() == 0:
            empty = draft_hidden.new_empty((0,), dtype=torch.float32)
            return empty, empty if calculate_entropy else None

        selected_hidden = draft_hidden[batch_indices, draft_indices, :]
        log_prob_chunks: list[torch.Tensor] = []
        entropy_chunks: list[torch.Tensor] = []
        for start in range(0, selected_hidden.shape[0], chunk_size):
            end = min(start + chunk_size, selected_hidden.shape[0])
            logits = output_embeddings(selected_hidden[start:end])
            log_probs = F.log_softmax(logits.float(), dim=-1)
            labels = token_ids[start:end].to(device=log_probs.device)
            log_prob_chunks.append(log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1))
            if calculate_entropy:
                entropy_chunks.append(-(log_probs.exp() * log_probs).sum(dim=-1))

        selected_log_probs = torch.cat(log_prob_chunks, dim=0)
        selected_entropy = torch.cat(entropy_chunks, dim=0) if calculate_entropy else None
        return selected_log_probs, selected_entropy

    def _compute_reverse_kl_sums(
        self,
        *,
        q_hidden: torch.Tensor,
        p_hidden: torch.Tensor,
        output_embeddings: torch.nn.Module,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q_hidden.numel() == 0:
            zero = p_hidden.new_tensor(0.0, dtype=torch.float32)
            return zero, zero

        sum_kl = p_hidden.new_tensor(0.0, dtype=torch.float32)
        num_states = p_hidden.new_tensor(0.0, dtype=torch.float32)
        for start in range(0, q_hidden.shape[0], chunk_size):
            end = min(start + chunk_size, q_hidden.shape[0])
            q_logits = output_embeddings(q_hidden[start:end])
            p_logits = output_embeddings(p_hidden[start:end])
            q_log_probs = F.log_softmax(q_logits.float(), dim=-1)
            p_log_probs = F.log_softmax(p_logits.float(), dim=-1)
            kl = (q_log_probs.exp() * (q_log_probs - p_log_probs)).sum(dim=-1)
            sum_kl = sum_kl + kl.sum()
            num_states = num_states + p_hidden.new_tensor(float(kl.numel()), dtype=torch.float32)
        return sum_kl, num_states

    def _build_replay_dis_anchor_plan(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        prompt_lengths: torch.LongTensor,
        response_lengths: torch.LongTensor,
        replay_block_anchor_indices: torch.LongTensor,
        replay_block_accepted_lengths: torch.LongTensor,
        replay_block_drafted_lengths: torch.LongTensor,
        replay_block_mask: torch.Tensor,
        draft_block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        width = int(replay_block_anchor_indices.shape[1])
        anchor_positions = torch.zeros((batch_size, width), dtype=torch.long, device=device)
        accepted_lengths = torch.zeros((batch_size, width), dtype=torch.long, device=device)
        drafted_lengths = torch.zeros((batch_size, width), dtype=torch.long, device=device)
        block_keep_mask = torch.zeros((batch_size, width), dtype=torch.bool, device=device)
        if attention_mask is not None:
            valid_seq_lens = attention_mask.long().sum(dim=1)
        else:
            valid_seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)

        for batch_idx in range(batch_size):
            prompt_len = int(prompt_lengths[batch_idx].item())
            response_len = int(response_lengths[batch_idx].item())
            valid_len = min(prompt_len + response_len, seq_len)
            for item_idx in range(width):
                if not bool(replay_block_mask[batch_idx, item_idx].item()):
                    continue
                accepted_len = int(replay_block_accepted_lengths[batch_idx, item_idx].item())
                drafted_len = int(replay_block_drafted_lengths[batch_idx, item_idx].item())
                if drafted_len <= 0:
                    continue
                drafted_len = min(drafted_len, draft_block_size - 1)
                accepted_len = max(0, min(accepted_len, drafted_len))
                anchor_resp = int(replay_block_anchor_indices[batch_idx, item_idx].item())
                full_anchor = prompt_len - 1 if anchor_resp < 0 else prompt_len + anchor_resp
                if full_anchor < 0 or full_anchor >= valid_len:
                    continue
                anchor_positions[batch_idx, item_idx] = full_anchor
                accepted_lengths[batch_idx, item_idx] = accepted_len
                drafted_lengths[batch_idx, item_idx] = drafted_len
                block_keep_mask[batch_idx, item_idx] = True

        return anchor_positions, accepted_lengths, drafted_lengths, block_keep_mask, valid_seq_lens, replay_block_mask

    def _compute_replay_dis_mismatch(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        target_hidden: torch.Tensor,
        target_lm_hidden: torch.Tensor,
        output_embeddings: torch.nn.Module,
        prompt_lengths: torch.LongTensor,
        response_lengths: torch.LongTensor,
        replay_block_anchor_indices: Optional[torch.LongTensor],
        replay_block_accepted_lengths: Optional[torch.LongTensor],
        replay_block_drafted_lengths: Optional[torch.LongTensor],
        replay_block_mask: Optional[torch.Tensor],
        draft_block_size: int,
        lm_head_chunk_size: int,
        profile_enabled: bool,
    ) -> dict[str, torch.Tensor]:
        groups = (
            "matched_target_trajectory",
            "accepted_replay",
            "first_rejected_replay",
            "post_rejection_suffix_replay",
        )
        zero = target_hidden.new_tensor(0.0, dtype=torch.float32)
        empty_metrics = {
            f"dflash_replay_dis_{group}_{suffix}": zero.clone()
            for group in groups
            for suffix in ("sum_kl", "num_states")
        }
        if (
            replay_block_anchor_indices is None
            or replay_block_accepted_lengths is None
            or replay_block_drafted_lengths is None
            or replay_block_mask is None
            or not bool(replay_block_mask.any())
        ):
            return empty_metrics

        (
            replay_anchor_positions,
            replay_accepted_lengths,
            replay_drafted_lengths,
            replay_keep_mask,
            valid_seq_lens,
            _,
        ) = self._build_replay_dis_anchor_plan(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
            response_lengths=response_lengths,
            replay_block_anchor_indices=replay_block_anchor_indices,
            replay_block_accepted_lengths=replay_block_accepted_lengths,
            replay_block_drafted_lengths=replay_block_drafted_lengths,
            replay_block_mask=replay_block_mask,
            draft_block_size=draft_block_size,
        )
        if not bool(replay_keep_mask.any()):
            return empty_metrics

        with torch.no_grad():
            replay_draft_hidden, _, _ = self._run_dflash_draft_forward(
                input_ids=input_ids,
                target_hidden=target_hidden,
                anchor_positions=replay_anchor_positions,
                block_keep_mask=replay_keep_mask,
                draft_block_size=draft_block_size,
                profile_enabled=profile_enabled,
                checkpoint_forward=False,
            )

            selected: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {group: [] for group in groups}
            for block_offset in range(1, draft_block_size):
                active_blocks = replay_keep_mask & (replay_drafted_lengths >= block_offset)
                if not bool(active_blocks.any()):
                    continue
                block_indices = torch.nonzero(active_blocks, as_tuple=False)
                batch_indices = block_indices[:, 0]
                block_ids = block_indices[:, 1]
                row_indices = replay_anchor_positions[batch_indices, block_ids] + (block_offset - 1)
                label_indices = row_indices + 1
                in_bounds = label_indices < valid_seq_lens[batch_indices]
                if not bool(in_bounds.any()):
                    continue

                batch_indices = batch_indices[in_bounds]
                block_ids = block_ids[in_bounds]
                row_indices = row_indices[in_bounds]
                flat_draft_indices = block_ids * draft_block_size + block_offset
                q_hidden = replay_draft_hidden[batch_indices, flat_draft_indices, :]
                p_hidden = target_lm_hidden[batch_indices, row_indices, :]
                selected["matched_target_trajectory"].append((q_hidden, p_hidden))

                accepted_lens = replay_accepted_lengths[batch_indices, block_ids]
                accepted_mask = block_offset <= accepted_lens
                first_rejected_mask = block_offset == (accepted_lens + 1)
                suffix_mask = block_offset > (accepted_lens + 1)
                for group, mask in (
                    ("accepted_replay", accepted_mask),
                    ("first_rejected_replay", first_rejected_mask),
                    ("post_rejection_suffix_replay", suffix_mask),
                ):
                    if bool(mask.any()):
                        selected[group].append((q_hidden[mask], p_hidden[mask]))

            metrics = dict(empty_metrics)
            for group, hidden_pairs in selected.items():
                if not hidden_pairs:
                    continue
                q_group_hidden = torch.cat([pair[0] for pair in hidden_pairs], dim=0)
                p_group_hidden = torch.cat([pair[1] for pair in hidden_pairs], dim=0)
                sum_kl, num_states = self._compute_reverse_kl_sums(
                    q_hidden=q_group_hidden,
                    p_hidden=p_group_hidden,
                    output_embeddings=output_embeddings,
                    chunk_size=lm_head_chunk_size,
                )
                metrics[f"dflash_replay_dis_{group}_sum_kl"] = sum_kl.detach()
                metrics[f"dflash_replay_dis_{group}_num_states"] = num_states.detach()
            return metrics

    def _build_random_response_anchor_plan(
        self,
        *,
        input_ids: torch.LongTensor,
        batch_idx: int,
        prompt_len: int,
        response_len: int,
        segment_lens: list[int],
        seed: int,
    ) -> tuple[list[int], list[int]]:
        if response_len <= 0 or not segment_lens:
            return [], []
        segment_lens = [int(segment_len) for segment_len in segment_lens if int(segment_len) > 0]
        if not segment_lens:
            return [], []
        token_count = sum(segment_lens)
        if token_count > response_len:
            raise ValueError(
                "Random DFLASH response anchors cannot preserve token count: "
                f"segment token count {token_count} exceeds response_len={response_len}."
            )
        valid_len = max(0, min(int(prompt_len + response_len), int(input_ids.shape[1])))
        if valid_len > 0:
            sample_ids = input_ids[batch_idx, :valid_len].to(dtype=torch.long)
            weights = torch.arange(1, valid_len + 1, dtype=torch.long, device=input_ids.device)
            token_hash = int((sample_ids * weights).sum().item())
        else:
            token_hash = 0
        rng = random.Random(int(seed) + batch_idx * 1000003 + token_hash)
        segment_lens = list(segment_lens)
        rng.shuffle(segment_lens)
        gaps = [0 for _ in range(len(segment_lens) + 1)]
        for _ in range(response_len - token_count):
            gaps[rng.randrange(len(gaps))] += 1

        anchors_resp: list[int] = []
        cursor = gaps[0]
        for segment_idx, segment_len in enumerate(segment_lens):
            anchors_resp.append(cursor - 1)
            cursor += segment_len + gaps[segment_idx + 1]
        return anchors_resp, segment_lens

    def _build_opd_anchor_plan(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        prompt_lengths: torch.LongTensor,
        response_lengths: torch.LongTensor,
        reject_token_indices: torch.LongTensor,
        draft_block_size: int,
        response_anchor_stride: int = 1,
        random_response_anchor_enabled: bool = False,
        random_response_anchor_seed: int = 42,
        rejected_draft_anchor_indices: Optional[torch.LongTensor] = None,
        rejected_draft_offsets: Optional[torch.LongTensor] = None,
        rejected_draft_mask: Optional[torch.Tensor] = None,
        include_rejected_draft_anchors: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        all_anchors: list[list[int]] = []
        all_segment_lens: list[list[int]] = []
        all_row_starts: list[list[int]] = []
        max_blocks = 0
        empty_reject_sample_count = 0
        skipped_sample_count = 0
        total_reject_count = 0
        response_anchor_count = 0
        for batch_idx in range(batch_size):
            prompt_len = int(prompt_lengths[batch_idx].item())
            response_len = int(response_lengths[batch_idx].item())
            if response_len <= 0:
                anchors_resp: list[int] = []
                boundaries_resp: list[int] = []
            else:
                raw_rejects = reject_token_indices[batch_idx]
                rejects = sorted(
                    {
                        int(idx.item())
                        for idx in raw_rejects
                        if int(idx.item()) >= 0 and int(idx.item()) < response_len
                    }
                )
                total_reject_count += len(rejects)
                if len(rejects) == 0:
                    empty_reject_sample_count += 1
                    anchors_resp = []
                    boundaries_resp = []
                elif rejects[-1] < response_len - 1:
                    rejects.append(response_len - 1)
                    anchors_resp = [-1] + rejects
                    boundaries_resp = rejects
                else:
                    anchors_resp = [-1] + rejects
                    boundaries_resp = rejects

            if response_anchor_stride > 1 and anchors_resp:
                anchor_boundary_pairs = list(zip(anchors_resp, boundaries_resp, strict=False))
                anchor_boundary_pairs = [
                    pair
                    for pair_idx, pair in enumerate(anchor_boundary_pairs)
                    if pair_idx % response_anchor_stride == 0 or pair_idx == len(anchor_boundary_pairs) - 1
                ]
                anchors_resp = [pair[0] for pair in anchor_boundary_pairs]
                boundaries_resp = [pair[1] for pair in anchor_boundary_pairs]
            sample_anchors: list[int] = []
            sample_segment_lens: list[int] = []
            sample_row_starts: list[int] = []
            response_anchors_resp: list[int] = []
            response_segment_lens: list[int] = []
            for anchor_resp, boundary_resp in zip(anchors_resp, boundaries_resp, strict=False):
                full_anchor = prompt_len - 1 if anchor_resp < 0 else prompt_len + anchor_resp
                segment_len = boundary_resp - anchor_resp
                if segment_len <= 0:
                    continue
                segment_len = min(segment_len, draft_block_size - 1)
                if full_anchor < 0 or full_anchor >= seq_len - 1:
                    continue
                response_anchors_resp.append(anchor_resp)
                response_segment_lens.append(segment_len)

            if random_response_anchor_enabled and response_len > 0 and response_segment_lens:
                response_anchors_resp, response_segment_lens = self._build_random_response_anchor_plan(
                    input_ids=input_ids,
                    batch_idx=batch_idx,
                    prompt_len=prompt_len,
                    response_len=response_len,
                    segment_lens=response_segment_lens,
                    seed=random_response_anchor_seed,
                )

            for anchor_resp, segment_len in zip(response_anchors_resp, response_segment_lens, strict=False):
                full_anchor = prompt_len - 1 if anchor_resp < 0 else prompt_len + anchor_resp
                if full_anchor < 0 or full_anchor >= seq_len - 1:
                    continue
                sample_anchors.append(full_anchor)
                sample_segment_lens.append(segment_len)
                sample_row_starts.append(full_anchor)

            if response_len > 0 and len(sample_anchors) == 0:
                skipped_sample_count += 1
            response_anchor_count += len(sample_anchors)

            if (
                include_rejected_draft_anchors
                and rejected_draft_anchor_indices is not None
                and rejected_draft_offsets is not None
                and rejected_draft_mask is not None
            ):
                valid_len = prompt_len + response_len
                rejected_count = rejected_draft_anchor_indices.shape[1]
                existing_anchors = set(sample_anchors)
                for item_idx in range(rejected_count):
                    if not bool(rejected_draft_mask[batch_idx, item_idx].item()):
                        continue
                    offset = int(rejected_draft_offsets[batch_idx, item_idx].item())
                    if offset <= 0 or offset >= draft_block_size:
                        continue
                    anchor_resp = int(rejected_draft_anchor_indices[batch_idx, item_idx].item())
                    full_anchor = prompt_len - 1 if anchor_resp < 0 else prompt_len + anchor_resp
                    if full_anchor < 0 or full_anchor >= min(valid_len, seq_len):
                        continue
                    if full_anchor in existing_anchors:
                        continue
                    sample_anchors.append(full_anchor)
                    sample_segment_lens.append(0)
                    sample_row_starts.append(full_anchor)
                    existing_anchors.add(full_anchor)

            all_anchors.append(sample_anchors)
            all_segment_lens.append(sample_segment_lens)
            all_row_starts.append(sample_row_starts)
            max_blocks = max(max_blocks, len(sample_anchors))

        max_blocks = max(max_blocks, 1)

        anchor_positions = torch.zeros((batch_size, max_blocks), dtype=torch.long, device=device)
        segment_lens = torch.zeros((batch_size, max_blocks), dtype=torch.long, device=device)
        row_starts = torch.zeros((batch_size, max_blocks), dtype=torch.long, device=device)
        block_keep_mask = torch.zeros((batch_size, max_blocks), dtype=torch.bool, device=device)

        for batch_idx, sample_anchors in enumerate(all_anchors):
            n_blocks = len(sample_anchors)
            if n_blocks == 0:
                continue
            anchor_positions[batch_idx, :n_blocks] = torch.tensor(sample_anchors, dtype=torch.long, device=device)
            segment_lens[batch_idx, :n_blocks] = torch.tensor(
                all_segment_lens[batch_idx], dtype=torch.long, device=device
            )
            row_starts[batch_idx, :n_blocks] = torch.tensor(all_row_starts[batch_idx], dtype=torch.long, device=device)
            block_keep_mask[batch_idx, :n_blocks] = True

        if attention_mask is not None:
            valid_seq_lens = attention_mask.long().sum(dim=1)
        else:
            valid_seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        opd_metrics = {
            "valid_anchor_count": response_anchor_count,
            "skipped_sample_count": skipped_sample_count,
            "empty_reject_sample_count": empty_reject_sample_count,
            "total_reject_count": total_reject_count,
            "sample_count": batch_size,
        }
        return anchor_positions, segment_lens, row_starts, block_keep_mask, valid_seq_lens, opd_metrics

    def _build_rejected_draft_anchor_plan(
        self,
        *,
        input_ids: torch.LongTensor,
        prompt_lengths: torch.LongTensor,
        response_lengths: torch.LongTensor,
        draft_block_size: int,
        max_tokens_per_sample: Optional[int],
        rejected_draft_anchor_indices: Optional[torch.LongTensor],
        rejected_draft_offsets: Optional[torch.LongTensor],
        rejected_draft_token_ids: Optional[torch.LongTensor],
        rejected_draft_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        if (
            rejected_draft_anchor_indices is None
            or rejected_draft_offsets is None
            or rejected_draft_token_ids is None
            or rejected_draft_mask is None
            or not bool(rejected_draft_mask.any())
        ):
            anchor_positions = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            block_keep_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
            return anchor_positions, block_keep_mask

        all_anchors: list[list[int]] = []
        max_blocks = 0
        rejected_width = int(rejected_draft_anchor_indices.shape[1])
        for batch_idx in range(batch_size):
            prompt_len = int(prompt_lengths[batch_idx].item())
            response_len = int(response_lengths[batch_idx].item())
            valid_len = min(prompt_len + response_len, seq_len)
            sample_anchors: list[int] = []
            existing_anchors: set[int] = set()
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
                if full_anchor not in existing_anchors:
                    sample_anchors.append(full_anchor)
                    existing_anchors.add(full_anchor)
                selected_count += 1

            all_anchors.append(sample_anchors)
            max_blocks = max(max_blocks, len(sample_anchors))

        max_blocks = max(max_blocks, 1)
        anchor_positions = torch.zeros((batch_size, max_blocks), dtype=torch.long, device=device)
        block_keep_mask = torch.zeros((batch_size, max_blocks), dtype=torch.bool, device=device)
        for batch_idx, sample_anchors in enumerate(all_anchors):
            n_blocks = len(sample_anchors)
            if n_blocks == 0:
                continue
            anchor_positions[batch_idx, :n_blocks] = torch.tensor(sample_anchors, dtype=torch.long, device=device)
            block_keep_mask[batch_idx, :n_blocks] = True
        return anchor_positions, block_keep_mask

    def _collect_rejected_draft_log_probs(
        self,
        *,
        draft_hidden: torch.Tensor,
        output_embeddings: torch.nn.Module,
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
                teacher_tensor[batch_idx, item_idx] = rejected_draft_teacher_logprobs[
                    batch_idx, item_idx
                ].to(device=draft_hidden.device, dtype=torch.float32)
                mask_tensor[batch_idx, item_idx] = True

        if selected_batch_indices:
            selected_item_batch_indices = torch.tensor(
                [batch_idx for batch_idx, _ in selected_item_indices],
                dtype=torch.long,
                device=draft_hidden.device,
            )
            selected_item_column_indices = torch.tensor(
                [item_idx for _, item_idx in selected_item_indices],
                dtype=torch.long,
                device=draft_hidden.device,
            )
            selected_log_probs, _ = self._compute_selected_lm_log_probs(
                draft_hidden=draft_hidden,
                output_embeddings=output_embeddings,
                batch_indices=torch.tensor(selected_batch_indices, dtype=torch.long, device=draft_hidden.device),
                draft_indices=torch.tensor(selected_draft_indices, dtype=torch.long, device=draft_hidden.device),
                token_ids=torch.tensor(selected_token_ids, dtype=torch.long, device=draft_hidden.device),
                chunk_size=lm_head_chunk_size,
                calculate_entropy=False,
            )
            student_tensor[selected_item_batch_indices, selected_item_column_indices] = selected_log_probs
        return student_tensor, teacher_tensor, mask_tensor

    def _build_tv_position_plan(
        self,
        *,
        prompt_lengths: torch.LongTensor,
        response_lengths: torch.LongTensor,
        anchor_positions: torch.LongTensor,
        segment_lens: torch.LongTensor,
        block_keep_mask: torch.Tensor,
        valid_seq_lens: torch.LongTensor,
        draft_block_size: int,
        rejected_draft_anchor_indices: Optional[torch.LongTensor],
        rejected_draft_offsets: Optional[torch.LongTensor],
        rejected_draft_token_ids: Optional[torch.LongTensor],
        rejected_draft_mask: Optional[torch.Tensor],
        first_token_only: bool,
    ) -> dict[str, Any]:
        """Merge accepted and rejected positions into exact speculative blocks.

        Positions are keyed by ``(sample, anchor, offset)``. Rejected metadata
        replaces the target-trajectory residual at the same offset, so every
        original speculative position appears at most once.
        """
        device = anchor_positions.device
        batch_size = int(prompt_lengths.shape[0])
        rejected_by_sample_anchor: list[dict[int, dict[int, int]]] = [dict() for _ in range(batch_size)]

        rejected_tensors = (
            rejected_draft_anchor_indices,
            rejected_draft_offsets,
            rejected_draft_token_ids,
            rejected_draft_mask,
        )
        present = [tensor is not None for tensor in rejected_tensors]
        if any(present) and not all(present):
            raise ValueError(
                "Exact DFLASH TV requires anchor, offset, token-id, and mask rejected metadata together."
            )
        if all(present):
            assert rejected_draft_anchor_indices is not None
            assert rejected_draft_offsets is not None
            assert rejected_draft_token_ids is not None
            assert rejected_draft_mask is not None
            expected_shape = rejected_draft_mask.shape
            for name, tensor in (
                ("anchor_indices", rejected_draft_anchor_indices),
                ("offsets", rejected_draft_offsets),
                ("token_ids", rejected_draft_token_ids),
            ):
                if tensor.shape != expected_shape:
                    raise ValueError(
                        f"Rejected DFLASH TV metadata shape mismatch for {name}: "
                        f"{tuple(tensor.shape)} != {tuple(expected_shape)}."
                    )
            for batch_idx in range(batch_size):
                prompt_len = int(prompt_lengths[batch_idx].item())
                valid_len = int(valid_seq_lens[batch_idx].item())
                for item_idx in range(int(rejected_draft_mask.shape[1])):
                    if not bool(rejected_draft_mask[batch_idx, item_idx].item()):
                        continue
                    anchor_resp = int(rejected_draft_anchor_indices[batch_idx, item_idx].item())
                    full_anchor = prompt_len - 1 if anchor_resp < 0 else prompt_len + anchor_resp
                    offset = int(rejected_draft_offsets[batch_idx, item_idx].item())
                    token_id = int(rejected_draft_token_ids[batch_idx, item_idx].item())
                    if full_anchor < 0 or full_anchor >= valid_len:
                        raise ValueError(
                            "Rejected DFLASH TV anchor is out of bounds: "
                            f"sample={batch_idx}, anchor={anchor_resp}, full_anchor={full_anchor}, "
                            f"valid_len={valid_len}."
                        )
                    if offset <= 0 or offset >= draft_block_size:
                        raise ValueError(
                            "Rejected DFLASH TV offset is out of bounds: "
                            f"sample={batch_idx}, anchor={anchor_resp}, offset={offset}, "
                            f"draft_block_size={draft_block_size}."
                        )
                    if token_id < 0:
                        raise ValueError(
                            "Rejected DFLASH TV token id must be non-negative: "
                            f"sample={batch_idx}, anchor={anchor_resp}, offset={offset}, token_id={token_id}."
                        )
                    by_offset = rejected_by_sample_anchor[batch_idx].setdefault(full_anchor, {})
                    previous_token_id = by_offset.get(offset)
                    if previous_token_id is not None and previous_token_id != token_id:
                        raise ValueError(
                            "Conflicting duplicate rejected DFLASH TV metadata at "
                            f"sample={batch_idx}, anchor={anchor_resp}, offset={offset}: "
                            f"token_ids={previous_token_id} and {token_id}."
                        )
                    by_offset[offset] = token_id

        for batch_idx, rejected_by_anchor in enumerate(rejected_by_sample_anchor):
            valid_len = int(valid_seq_lens[batch_idx].item())
            for full_anchor, by_offset in rejected_by_anchor.items():
                sorted_offsets = sorted(by_offset)
                expected_offsets = list(range(sorted_offsets[0], sorted_offsets[-1] + 1))
                if sorted_offsets != expected_offsets:
                    raise ValueError(
                        "Rejected DFLASH TV suffix offsets must be contiguous: "
                        f"sample={batch_idx}, full_anchor={full_anchor}, offsets={sorted_offsets}."
                    )
                first_offset = sorted_offsets[0]
                first_target_row = full_anchor + first_offset - 1
                if first_target_row >= valid_len:
                    raise ValueError(
                        "Rejected DFLASH TV first-reject context exceeds the target trajectory: "
                        f"sample={batch_idx}, full_anchor={full_anchor}, first_offset={first_offset}, "
                        f"target_row={first_target_row}, valid_len={valid_len}."
                    )

        batch_indices: list[int] = []
        draft_indices: list[int] = []
        target_row_indices: list[int] = []
        block_indices: list[int] = []
        offsets: list[int] = []
        branch_specs: list[dict[str, Any]] = []
        seen_anchors: set[tuple[int, int]] = set()
        effective_block_idx = 0

        for batch_idx in range(batch_size):
            for anchor_block_idx in range(int(anchor_positions.shape[1])):
                if not bool(block_keep_mask[batch_idx, anchor_block_idx].item()):
                    continue
                full_anchor = int(anchor_positions[batch_idx, anchor_block_idx].item())
                anchor_key = (batch_idx, full_anchor)
                if anchor_key in seen_anchors:
                    raise ValueError(
                        "Duplicate DFLASH TV anchor plan entry for "
                        f"sample={batch_idx}, full_anchor={full_anchor}."
                    )
                seen_anchors.add(anchor_key)
                rejected_by_offset = rejected_by_sample_anchor[batch_idx].get(full_anchor)
                branch_spec: Optional[dict[str, Any]] = None
                if rejected_by_offset:
                    rejected_offsets = sorted(rejected_by_offset)
                    first_rejected_offset = rejected_offsets[0]
                    selected_rejected_offsets = rejected_offsets[:1] if first_token_only else rejected_offsets
                    selected_offsets = list(range(1, first_rejected_offset)) + selected_rejected_offsets
                    if not first_token_only and len(rejected_offsets) > 1:
                        branch_spec = {
                            "sample_idx": batch_idx,
                            "full_anchor": full_anchor,
                            "first_offset": first_rejected_offset,
                            "token_ids": [rejected_by_offset[offset] for offset in rejected_offsets],
                            "flat_indices": [],
                            "suffix_indices": [],
                        }
                else:
                    segment_len = int(segment_lens[batch_idx, anchor_block_idx].item())
                    selected_offsets = list(range(1, segment_len + 1))

                if not selected_offsets:
                    continue
                for offset in selected_offsets:
                    target_row = full_anchor + offset - 1
                    is_later_reject = bool(
                        rejected_by_offset
                        and offset > min(rejected_by_offset)
                        and offset in rejected_by_offset
                    )
                    if not is_later_reject and target_row >= int(valid_seq_lens[batch_idx].item()):
                        raise ValueError(
                            "DFLASH TV target-trajectory position is out of bounds: "
                            f"sample={batch_idx}, full_anchor={full_anchor}, offset={offset}, "
                            f"target_row={target_row}, valid_len={int(valid_seq_lens[batch_idx].item())}."
                        )
                    flat_idx = len(batch_indices)
                    batch_indices.append(batch_idx)
                    draft_indices.append(anchor_block_idx * draft_block_size + offset)
                    target_row_indices.append(-1 if is_later_reject else target_row)
                    block_indices.append(effective_block_idx)
                    offsets.append(offset)
                    if is_later_reject:
                        assert branch_spec is not None
                        branch_spec["flat_indices"].append(flat_idx)
                        branch_spec["suffix_indices"].append(offset - int(branch_spec["first_offset"]))
                if branch_spec is not None:
                    branch_specs.append(branch_spec)
                effective_block_idx += 1

        missing_rejected_anchors = {
            (batch_idx, full_anchor)
            for batch_idx, by_anchor in enumerate(rejected_by_sample_anchor)
            for full_anchor in by_anchor
            if (batch_idx, full_anchor) not in seen_anchors
        }
        if missing_rejected_anchors:
            raise ValueError(
                "Rejected DFLASH TV metadata anchors are missing from the draft anchor plan: "
                f"{sorted(missing_rejected_anchors)}."
            )

        def _long_tensor(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.long, device=device)

        return {
            "batch_indices": _long_tensor(batch_indices),
            "draft_indices": _long_tensor(draft_indices),
            "target_row_indices": _long_tensor(target_row_indices),
            "block_indices": _long_tensor(block_indices),
            "offsets": _long_tensor(offsets),
            "branch_specs": branch_specs,
            "block_count": effective_block_idx,
        }

    def _compute_tv_teacher_branch_hidden(
        self,
        *,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor],
        target_lm_hidden: torch.Tensor,
        tv_plan: dict[str, Any],
    ) -> tuple[torch.Tensor, int]:
        target_rows = tv_plan["target_row_indices"]
        if target_rows.numel() == 0:
            return target_lm_hidden.new_empty((0, target_lm_hidden.shape[-1])), 0
        safe_rows = target_rows.clamp_min(0)
        selected = target_lm_hidden[tv_plan["batch_indices"], safe_rows].detach().clone()
        branch_position_count = 0
        for branch_spec in tv_plan["branch_specs"]:
            sample_idx = int(branch_spec["sample_idx"])
            full_anchor = int(branch_spec["full_anchor"])
            first_offset = int(branch_spec["first_offset"])
            prefix_len = full_anchor + first_offset
            branch_tokens = torch.tensor(
                branch_spec["token_ids"], dtype=input_ids.dtype, device=input_ids.device
            )
            branch_input_ids = torch.cat([input_ids[sample_idx, :prefix_len], branch_tokens], dim=0).unsqueeze(0)
            branch_attention_mask = torch.ones_like(branch_input_ids, dtype=torch.long)
            branch_position_ids = None
            if position_ids is not None:
                if position_ids.dim() != 2:
                    raise NotImplementedError(
                        "Exact DFLASH TV rejected-branch reconstruction currently supports 2D text position_ids only."
                    )
                branch_position_ids = torch.arange(
                    branch_input_ids.shape[1], dtype=position_ids.dtype, device=position_ids.device
                ).unsqueeze(0)
            with torch.no_grad():
                branch_outputs = self.main_model(
                    input_ids=branch_input_ids,
                    attention_mask=branch_attention_mask,
                    position_ids=branch_position_ids,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                if branch_outputs.hidden_states is None:
                    raise RuntimeError("Teacher branch forward did not return hidden states for exact DFLASH TV.")
                branch_hidden = branch_outputs.hidden_states[-1]
            flat_indices = torch.tensor(branch_spec["flat_indices"], dtype=torch.long, device=input_ids.device)
            suffix_indices = torch.tensor(branch_spec["suffix_indices"], dtype=torch.long, device=input_ids.device)
            branch_rows = prefix_len - 1 + suffix_indices
            selected[flat_indices] = branch_hidden[0, branch_rows].detach()
            branch_position_count += int(flat_indices.numel())
        return selected, branch_position_count

    def _compute_full_vocab_tv_from_hidden(
        self,
        *,
        draft_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        output_embeddings: torch.nn.Module,
        chunk_size: int,
    ) -> torch.Tensor:
        if draft_hidden.shape != target_hidden.shape:
            raise ValueError(
                "Draft and target hidden states must have identical shapes for exact DFLASH TV, got "
                f"draft={tuple(draft_hidden.shape)}, target={tuple(target_hidden.shape)}."
            )
        if draft_hidden.numel() == 0:
            return draft_hidden.new_empty((0,), dtype=torch.float32)

        def _linear_with_frozen_head(hidden: torch.Tensor) -> torch.Tensor:
            if isinstance(output_embeddings, torch.nn.Linear):
                bias = output_embeddings.bias.detach() if output_embeddings.bias is not None else None
                return F.linear(hidden, output_embeddings.weight.detach(), bias)
            if any(param.requires_grad for param in output_embeddings.parameters()):
                raise RuntimeError("Exact DFLASH TV requires a frozen target LM head.")
            return output_embeddings(hidden)

        tv_chunks: list[torch.Tensor] = []
        for start in range(0, int(draft_hidden.shape[0]), chunk_size):
            end = min(start + chunk_size, int(draft_hidden.shape[0]))
            draft_chunk = draft_hidden[start:end]
            target_chunk = target_hidden[start:end]

            def _tv_chunk(draft_arg: torch.Tensor, target_arg: torch.Tensor) -> torch.Tensor:
                draft_logits = _linear_with_frozen_head(draft_arg)
                with torch.no_grad():
                    target_logits = _linear_with_frozen_head(target_arg)
                return full_vocab_tv_distance(draft_logits, target_logits)

            if torch.is_grad_enabled() and draft_chunk.requires_grad:
                tv_chunk = torch_checkpoint.checkpoint(
                    _tv_chunk,
                    draft_chunk,
                    target_chunk,
                    use_reentrant=False,
                )
            else:
                tv_chunk = _tv_chunk(draft_chunk, target_chunk)
            tv_chunks.append(tv_chunk)
        return torch.cat(tv_chunks, dim=0)

    @staticmethod
    def _compute_e2e_tv_block_outputs(
        tv_distances: torch.Tensor,
        block_indices: torch.LongTensor,
        block_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_losses: list[torch.Tensor] = []
        expected_accept_lengths: list[torch.Tensor] = []
        for block_idx in range(block_count):
            block_tv = tv_distances[block_indices == block_idx]
            if block_tv.numel() == 0:
                raise RuntimeError(f"Exact DFLASH TV block {block_idx} has no positions.")
            cumulative_overlap = torch.cumprod((1.0 - block_tv).clamp(0.0, 1.0), dim=0)
            expected_accept_length = cumulative_overlap.sum()
            expected_accept_lengths.append(expected_accept_length)
            block_losses.append(1.0 - expected_accept_length / float(block_tv.numel()))
        if not block_losses:
            empty = tv_distances.new_empty((0,), dtype=torch.float32)
            return empty, empty
        return torch.stack(block_losses), torch.stack(expected_accept_lengths)

    def _run_dflash_draft_forward(
        self,
        *,
        input_ids: torch.LongTensor,
        target_hidden: torch.Tensor,
        anchor_positions: torch.LongTensor,
        block_keep_mask: torch.Tensor,
        draft_block_size: int,
        profile_enabled: bool,
        checkpoint_forward: bool = False,
    ) -> tuple[torch.Tensor, str, float]:
        batch_size, seq_len = input_ids.shape
        noise_embedding = self._create_noise_embedding_for_anchors(
            input_ids=input_ids,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            block_size=draft_block_size,
        )
        context_position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        draft_position_ids = self._create_position_ids_for_anchors(anchor_positions, block_size=draft_block_size)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        draft_config = getattr(self.draft_model, "config", None)
        attn_impl = str(getattr(draft_config, "_attn_implementation", "eager"))
        if attn_impl == "flex_attention":
            try:
                draft_attention_mask = create_dflash_block_mask(
                    anchor_positions=anchor_positions,
                    block_keep_mask=block_keep_mask,
                    seq_len=seq_len,
                    block_size=draft_block_size,
                    device=input_ids.device,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to create flex_attention BlockMask for DFLASH OPD (%s); falling back to sdpa.",
                    exc,
                )
                attn_impl = "sdpa"
                if draft_config is not None:
                    _set_config_attn_implementation(draft_config, attn_impl)
                draft_attention_mask = create_dflash_sdpa_mask(
                    anchor_positions=anchor_positions,
                    block_keep_mask=block_keep_mask,
                    seq_len=seq_len,
                    block_size=draft_block_size,
                    device=input_ids.device,
                )
        else:
            draft_attention_mask = create_dflash_sdpa_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                seq_len=seq_len,
                block_size=draft_block_size,
                device=input_ids.device,
            )

        def _draft_forward(noise_embedding_arg: torch.Tensor, target_hidden_arg: torch.Tensor) -> torch.Tensor:
            draft_outputs = self.draft_model(
                position_ids=full_position_ids,
                attention_mask=draft_attention_mask,
                noise_embedding=noise_embedding_arg,
                target_hidden=target_hidden_arg,
                use_cache=False,
            )
            draft_hidden = draft_outputs[0] if isinstance(draft_outputs, tuple) else draft_outputs
            if hasattr(draft_hidden, "last_hidden_state"):
                draft_hidden = draft_hidden.last_hidden_state
            return draft_hidden

        def _maybe_checkpoint_draft_forward() -> torch.Tensor:
            if checkpoint_forward and torch.is_grad_enabled():
                return torch_checkpoint.checkpoint(
                    _draft_forward,
                    noise_embedding,
                    target_hidden,
                    use_reentrant=False,
                )
            return _draft_forward(noise_embedding, target_hidden)

        draft_start_time = time.perf_counter()
        try:
            with self._draft_sdpa_context():
                draft_hidden = _maybe_checkpoint_draft_forward()
        except RuntimeError as exc:
            if attn_impl != "flex_attention" or self._is_oom_error(exc):
                raise
            logger.warning(
                "DFLASH draft flex_attention forward failed (%s); retrying this micro-batch with sdpa.",
                exc,
            )
            attn_impl = "sdpa"
            if draft_config is not None:
                _set_config_attn_implementation(draft_config, attn_impl)
            draft_attention_mask = create_dflash_sdpa_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                seq_len=seq_len,
                block_size=draft_block_size,
                device=input_ids.device,
            )
            with self._draft_sdpa_context():
                draft_hidden = _maybe_checkpoint_draft_forward()
        self._maybe_sync_for_profile(profile_enabled, input_ids.device)
        draft_forward_ms = (time.perf_counter() - draft_start_time) * 1000.0
        return draft_hidden, attn_impl, draft_forward_ms

    def _forward_opd_tv(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        prompt_lengths: torch.LongTensor,
        response_lengths: torch.LongTensor,
        reject_token_indices: torch.LongTensor,
        target_hidden: torch.Tensor,
        target_lm_hidden: torch.Tensor,
        rejected_draft_anchor_indices: Optional[torch.LongTensor],
        rejected_draft_offsets: Optional[torch.LongTensor],
        rejected_draft_token_ids: Optional[torch.LongTensor],
        rejected_draft_mask: Optional[torch.Tensor],
        first_token_only: bool,
        use_task_rewards: bool,
        calculate_entropy: bool,
        profile_enabled: bool,
    ) -> dict[str, torch.Tensor]:
        draft_block_size = self._get_block_size()
        lm_head_chunk_size = self._get_lm_head_chunk_size()
        (
            anchor_positions,
            segment_lens,
            _,
            block_keep_mask,
            valid_seq_lens,
            _,
        ) = self._build_opd_anchor_plan(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
            response_lengths=response_lengths,
            reject_token_indices=reject_token_indices,
            draft_block_size=draft_block_size,
            # Exact TV is defined on every rollout speculative block. Sampling
            # or striding anchors would change that objective.
            response_anchor_stride=1,
            random_response_anchor_enabled=False,
            rejected_draft_anchor_indices=rejected_draft_anchor_indices,
            rejected_draft_offsets=rejected_draft_offsets,
            rejected_draft_mask=rejected_draft_mask,
            include_rejected_draft_anchors=True,
        )
        tv_plan = self._build_tv_position_plan(
            prompt_lengths=prompt_lengths,
            response_lengths=response_lengths,
            anchor_positions=anchor_positions,
            segment_lens=segment_lens,
            block_keep_mask=block_keep_mask,
            valid_seq_lens=valid_seq_lens,
            draft_block_size=draft_block_size,
            rejected_draft_anchor_indices=rejected_draft_anchor_indices,
            rejected_draft_offsets=rejected_draft_offsets,
            rejected_draft_token_ids=rejected_draft_token_ids,
            rejected_draft_mask=rejected_draft_mask,
            first_token_only=first_token_only,
        )
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is None:
            raise RuntimeError("Main model output embeddings are required for exact DFLASH TV.")

        position_count = int(tv_plan["batch_indices"].numel())
        block_count = int(tv_plan["block_count"])
        if position_count == 0:
            trainable_param = next((param for param in self.draft_model.parameters() if param.requires_grad), None)
            if trainable_param is None:
                raise RuntimeError("DFLASH TV requires at least one trainable draft parameter.")
            zero_with_grad = trainable_param.flatten()[0].float() * 0.0
            output = {
                "dflash_tv_block_losses": zero_with_grad.reshape(1),
                "dflash_tv_block_mask": torch.zeros((1,), dtype=torch.bool, device=input_ids.device),
                "dflash_tv_distance_sum": zero_with_grad.detach(),
                "dflash_tv_overlap_sum": zero_with_grad.detach(),
                "dflash_tv_expected_accept_length_sum": zero_with_grad.detach(),
                "dflash_tv_block_count": zero_with_grad.detach(),
                "dflash_tv_position_count": zero_with_grad.detach(),
                "dflash_tv_teacher_branch_position_count": zero_with_grad.detach(),
            }
            if use_task_rewards:
                batch_size, seq_len = input_ids.shape
                output["dflash_log_probs"] = target_hidden.new_zeros(
                    (batch_size, seq_len), dtype=torch.float32
                ) + zero_with_grad
                output["dflash_loss_mask"] = target_hidden.new_zeros(
                    (batch_size, seq_len), dtype=torch.float32
                )
                if calculate_entropy:
                    output["dflash_entropy"] = target_hidden.new_zeros(
                        (batch_size, seq_len), dtype=torch.float32
                    )
            return output

        draft_hidden, _, _ = self._run_dflash_draft_forward(
            input_ids=input_ids,
            target_hidden=target_hidden,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            draft_block_size=draft_block_size,
            profile_enabled=profile_enabled,
            checkpoint_forward=True,
        )
        selected_draft_hidden = draft_hidden[tv_plan["batch_indices"], tv_plan["draft_indices"]]
        selected_target_hidden, branch_position_count = self._compute_tv_teacher_branch_hidden(
            input_ids=input_ids,
            position_ids=position_ids,
            target_lm_hidden=target_lm_hidden,
            tv_plan=tv_plan,
        )
        tv_distances = self._compute_full_vocab_tv_from_hidden(
            draft_hidden=selected_draft_hidden,
            target_hidden=selected_target_hidden,
            output_embeddings=output_embeddings,
            chunk_size=lm_head_chunk_size,
        )
        block_losses, expected_accept_lengths = self._compute_e2e_tv_block_outputs(
            tv_distances=tv_distances,
            block_indices=tv_plan["block_indices"],
            block_count=block_count,
        )
        output = {
            "dflash_tv_block_losses": block_losses,
            "dflash_tv_block_mask": torch.ones_like(block_losses, dtype=torch.bool),
            "dflash_tv_distance_sum": tv_distances.detach().sum(),
            "dflash_tv_overlap_sum": (1.0 - tv_distances.detach()).sum(),
            "dflash_tv_expected_accept_length_sum": expected_accept_lengths.detach().sum(),
            "dflash_tv_block_count": block_losses.new_tensor(float(block_count)).detach(),
            "dflash_tv_position_count": block_losses.new_tensor(float(position_count)).detach(),
            "dflash_tv_teacher_branch_position_count": block_losses.new_tensor(
                float(branch_position_count)
            ).detach(),
        }
        if use_task_rewards:
            batch_size, seq_len = input_ids.shape
            log_probs_by_seq = target_hidden.new_zeros((batch_size, seq_len), dtype=torch.float32)
            loss_mask_by_seq = target_hidden.new_zeros((batch_size, seq_len), dtype=torch.float32)
            entropy_by_seq = (
                target_hidden.new_zeros((batch_size, seq_len), dtype=torch.float32)
                if calculate_entropy
                else None
            )
            response_batch_indices: list[torch.Tensor] = []
            response_draft_indices: list[torch.Tensor] = []
            response_row_indices: list[torch.Tensor] = []
            response_labels: list[torch.Tensor] = []
            for block_offset in range(1, draft_block_size):
                active_blocks = block_keep_mask & (segment_lens >= block_offset)
                if not bool(active_blocks.any()):
                    continue
                active_indices = torch.nonzero(active_blocks, as_tuple=False)
                batch_indices = active_indices[:, 0]
                block_indices = active_indices[:, 1]
                row_indices = anchor_positions[batch_indices, block_indices] + block_offset - 1
                label_indices = row_indices + 1
                in_bounds = label_indices < valid_seq_lens[batch_indices]
                if not bool(in_bounds.any()):
                    continue
                batch_indices = batch_indices[in_bounds]
                block_indices = block_indices[in_bounds]
                row_indices = row_indices[in_bounds]
                response_batch_indices.append(batch_indices)
                response_draft_indices.append(block_indices * draft_block_size + block_offset)
                response_row_indices.append(row_indices)
                response_labels.append(input_ids[batch_indices, row_indices + 1])
            if response_batch_indices:
                batch_tensor = torch.cat(response_batch_indices)
                draft_tensor = torch.cat(response_draft_indices)
                row_tensor = torch.cat(response_row_indices)
                label_tensor = torch.cat(response_labels)
                selected_log_probs, selected_entropy = self._compute_selected_lm_log_probs(
                    draft_hidden=draft_hidden,
                    output_embeddings=output_embeddings,
                    batch_indices=batch_tensor,
                    draft_indices=draft_tensor,
                    token_ids=label_tensor,
                    chunk_size=lm_head_chunk_size,
                    calculate_entropy=calculate_entropy,
                )
                log_probs_by_seq[batch_tensor, row_tensor] = selected_log_probs
                loss_mask_by_seq[batch_tensor, row_tensor] = 1.0
                if entropy_by_seq is not None and selected_entropy is not None:
                    entropy_by_seq[batch_tensor, row_tensor] = selected_entropy
            output["dflash_log_probs"] = log_probs_by_seq
            output["dflash_loss_mask"] = loss_mask_by_seq
            if entropy_by_seq is not None:
                output["dflash_entropy"] = entropy_by_seq
        return output

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
        use_tv_loss: bool = False,
        rejected_draft_first_token_only: bool = False,
        use_task_rewards: bool = False,
        calculate_entropy: bool = False,
        use_replay_dis: bool = False,
        replay_block_anchor_indices: Optional[torch.LongTensor] = None,
        replay_block_accepted_lengths: Optional[torch.LongTensor] = None,
        replay_block_drafted_lengths: Optional[torch.LongTensor] = None,
        replay_block_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if input_ids.dim() != 2:
            raise ValueError(f"DFLASH OPD requires padded 2D input_ids, got shape={tuple(input_ids.shape)}.")

        profile_enabled = self._is_dflash_profiling_enabled()
        self._maybe_sync_for_profile(profile_enabled, input_ids.device)
        total_start_time = time.perf_counter()
        teacher_forward_ms = 0.0
        draft_forward_ms = 0.0
        lm_head_ms = 0.0

        target_kwargs = dict(kwargs)
        target_kwargs.pop("dflash_prompt_lengths", None)
        target_kwargs.pop("dflash_response_lengths", None)
        target_kwargs.pop("dflash_reject_token_indices", None)
        target_kwargs.pop("dflash_rejected_draft_anchor_indices", None)
        target_kwargs.pop("dflash_rejected_draft_offsets", None)
        target_kwargs.pop("dflash_rejected_draft_token_ids", None)
        target_kwargs.pop("dflash_rejected_draft_teacher_logprobs", None)
        target_kwargs.pop("dflash_rejected_draft_mask", None)
        target_kwargs.pop("dflash_use_tv_loss", None)
        target_kwargs.pop("dflash_rejected_draft_first_token_only", None)
        target_kwargs.pop("dflash_use_task_rewards", None)
        target_kwargs.pop("dflash_calculate_entropy", None)
        target_kwargs.pop("dflash_use_replay_dis", None)
        target_kwargs.pop("dflash_replay_block_anchor_indices", None)
        target_kwargs.pop("dflash_replay_block_accepted_lengths", None)
        target_kwargs.pop("dflash_replay_block_drafted_lengths", None)
        target_kwargs.pop("dflash_replay_block_mask", None)

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
                raise RuntimeError("Teacher model did not return hidden states required by DFLASH draft.")
            target_lm_hidden = teacher_outputs.hidden_states[-1]
            target_hidden = self._extract_target_hidden(teacher_outputs.hidden_states)
        self._maybe_sync_for_profile(profile_enabled, input_ids.device)
        teacher_forward_ms = (time.perf_counter() - teacher_start_time) * 1000.0

        if use_tv_loss:
            return self._forward_opd_tv(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                prompt_lengths=prompt_lengths,
                response_lengths=response_lengths,
                reject_token_indices=reject_token_indices,
                target_hidden=target_hidden,
                target_lm_hidden=target_lm_hidden,
                rejected_draft_anchor_indices=rejected_draft_anchor_indices,
                rejected_draft_offsets=rejected_draft_offsets,
                rejected_draft_token_ids=rejected_draft_token_ids,
                rejected_draft_mask=rejected_draft_mask,
                first_token_only=rejected_draft_first_token_only,
                use_task_rewards=use_task_rewards,
                calculate_entropy=calculate_entropy,
                profile_enabled=profile_enabled,
            )

        # SGLang DFLASH uses block_size as the total block length: one anchor
        # token plus block_size - 1 future-token predictions.
        draft_block_size = self._get_block_size()
        lm_head_chunk_size = self._get_lm_head_chunk_size()
        response_anchor_stride = self._get_response_anchor_stride()
        random_response_anchor_enabled = self._get_random_response_anchor_enabled()
        random_response_anchor_seed = self._get_random_response_anchor_seed()
        rejected_draft_max_tokens_per_sample = self._get_rejected_draft_max_tokens_per_sample()
        split_random_rejected_pass = random_response_anchor_enabled
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
                include_rejected_draft_anchors=not split_random_rejected_pass,
            )
        )

        batch_size, seq_len = input_ids.shape

        def _plan_block_stats(
            positions: torch.Tensor,
            keep_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, int, torch.Tensor]:
            actual = keep_mask.sum()
            if bool(keep_mask.any()):
                padded = batch_size * int(positions.shape[1])
                max_per_sample = keep_mask.sum(dim=1).max().to(dtype=torch.float32)
            else:
                padded = 0
                max_per_sample = actual.to(dtype=torch.float32)
            return actual, padded, max_per_sample

        rejected_anchor_positions = torch.zeros((batch_size, 1), dtype=torch.long, device=input_ids.device)
        rejected_block_keep_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=input_ids.device)
        if split_random_rejected_pass:
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

        response_actual_block_count, response_padded_block_count, response_max_blocks = _plan_block_stats(
            anchor_positions, block_keep_mask
        )
        rejected_actual_block_count, rejected_padded_block_count, rejected_max_blocks = _plan_block_stats(
            rejected_anchor_positions, rejected_block_keep_mask
        )
        actual_block_count = response_actual_block_count + rejected_actual_block_count
        padded_block_count = response_padded_block_count + rejected_padded_block_count
        max_blocks_per_sample = torch.maximum(response_max_blocks, rejected_max_blocks)
        if split_random_rejected_pass:
            max_blocks_per_sample = (block_keep_mask.sum(dim=1) + rejected_block_keep_mask.sum(dim=1)).max().to(
                dtype=torch.float32
            )
        draft_q_token_count = padded_block_count * draft_block_size

        output_embeddings = self.get_output_embeddings()
        if output_embeddings is None:
            raise RuntimeError("Main model output embeddings are required for composed DFLASH student logits.")

        lm_head_start_time = time.perf_counter()
        log_probs_by_seq = target_hidden.new_zeros((batch_size, seq_len), dtype=torch.float32)
        loss_mask_by_seq = target_hidden.new_zeros((batch_size, seq_len), dtype=torch.float32)
        response_offsets_by_seq = torch.zeros((batch_size, seq_len), dtype=torch.long, device=input_ids.device)
        entropy_by_seq = (
            target_hidden.new_zeros((batch_size, seq_len), dtype=torch.float32) if calculate_entropy else None
        )

        rejected_width = 1
        if rejected_draft_anchor_indices is not None and rejected_draft_anchor_indices.dim() >= 2:
            rejected_width = max(1, int(rejected_draft_anchor_indices.shape[1]))
        rejected_student_log_probs = target_hidden.new_zeros((batch_size, rejected_width), dtype=torch.float32)
        rejected_teacher_log_probs = target_hidden.new_zeros((batch_size, rejected_width), dtype=torch.float32)
        rejected_loss_mask = torch.zeros((batch_size, rejected_width), dtype=torch.bool, device=input_ids.device)
        response_lm_token_count = 0
        attn_impl = str(getattr(getattr(self.draft_model, "config", None), "_attn_implementation", "eager"))
        ran_draft_forward = False

        if bool(block_keep_mask.any()):
            draft_hidden, attn_impl, response_draft_forward_ms = self._run_dflash_draft_forward(
                input_ids=input_ids,
                target_hidden=target_hidden,
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                draft_block_size=draft_block_size,
                profile_enabled=profile_enabled,
                checkpoint_forward=split_random_rejected_pass,
            )
            draft_forward_ms += response_draft_forward_ms
            ran_draft_forward = True

            response_batch_indices: list[torch.Tensor] = []
            response_draft_indices: list[torch.Tensor] = []
            response_row_indices: list[torch.Tensor] = []
            response_labels: list[torch.Tensor] = []
            response_offsets: list[torch.Tensor] = []
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
                response_offsets.append(torch.full_like(row_indices, block_offset))

            if response_batch_indices:
                response_batch_tensor = torch.cat(response_batch_indices, dim=0)
                response_draft_tensor = torch.cat(response_draft_indices, dim=0)
                response_row_tensor = torch.cat(response_row_indices, dim=0)
                response_label_tensor = torch.cat(response_labels, dim=0)
                response_offset_tensor = torch.cat(response_offsets, dim=0)
                selected_log_probs, selected_entropy = self._compute_selected_lm_log_probs(
                    draft_hidden=draft_hidden,
                    output_embeddings=output_embeddings,
                    batch_indices=response_batch_tensor,
                    draft_indices=response_draft_tensor,
                    token_ids=response_label_tensor,
                    chunk_size=lm_head_chunk_size,
                    calculate_entropy=calculate_entropy,
                )
                log_probs_by_seq[response_batch_tensor, response_row_tensor] = selected_log_probs
                loss_mask_by_seq[response_batch_tensor, response_row_tensor] = 1.0
                response_offsets_by_seq[response_batch_tensor, response_row_tensor] = response_offset_tensor
                if entropy_by_seq is not None and selected_entropy is not None:
                    entropy_by_seq[response_batch_tensor, response_row_tensor] = selected_entropy
                response_lm_token_count = response_batch_tensor.numel()

            if not split_random_rejected_pass:
                rejected_student_log_probs, rejected_teacher_log_probs, rejected_loss_mask = (
                    self._collect_rejected_draft_log_probs(
                        draft_hidden=draft_hidden,
                        output_embeddings=output_embeddings,
                        prompt_lengths=prompt_lengths,
                        response_lengths=response_lengths,
                        anchor_positions=anchor_positions,
                        block_keep_mask=block_keep_mask,
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
            del draft_hidden

        if split_random_rejected_pass and bool(rejected_block_keep_mask.any()):
            rejected_draft_hidden, rejected_attn_impl, rejected_draft_forward_ms = self._run_dflash_draft_forward(
                input_ids=input_ids,
                target_hidden=target_hidden,
                anchor_positions=rejected_anchor_positions,
                block_keep_mask=rejected_block_keep_mask,
                draft_block_size=draft_block_size,
                profile_enabled=profile_enabled,
                checkpoint_forward=True,
            )
            draft_forward_ms += rejected_draft_forward_ms
            attn_impl = rejected_attn_impl
            ran_draft_forward = True
            rejected_student_log_probs, rejected_teacher_log_probs, rejected_loss_mask = (
                self._collect_rejected_draft_log_probs(
                    draft_hidden=rejected_draft_hidden,
                    output_embeddings=output_embeddings,
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
                raise RuntimeError("DFLASH OPD requires at least one trainable draft parameter.")
            zero_with_grad = trainable_param.flatten()[0].float() * 0.0
            log_probs_by_seq = log_probs_by_seq + zero_with_grad
        self._maybe_sync_for_profile(profile_enabled, input_ids.device)
        lm_head_ms = (time.perf_counter() - lm_head_start_time) * 1000.0
        rejected_lm_token_count = rejected_loss_mask.sum()
        selected_lm_token_count = loss_mask_by_seq.sum() + rejected_lm_token_count

        # Return a plain container so FSDP can discover tensors and attach
        # pre-backward hooks. A custom object can leave FSDP in IDLE at backward.
        output = {
            "dflash_log_probs": log_probs_by_seq,
            "dflash_loss_mask": loss_mask_by_seq,
            "dflash_response_offsets": response_offsets_by_seq,
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
                DFLASH_ATTENTION_IMPL_IDS.get(attn_impl, -1)
            ),
        }
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
        if use_replay_dis:
            output.update(
                self._compute_replay_dis_mismatch(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_hidden=target_hidden,
                    target_lm_hidden=target_lm_hidden,
                    output_embeddings=output_embeddings,
                    prompt_lengths=prompt_lengths,
                    response_lengths=response_lengths,
                    replay_block_anchor_indices=replay_block_anchor_indices,
                    replay_block_accepted_lengths=replay_block_accepted_lengths,
                    replay_block_drafted_lengths=replay_block_drafted_lengths,
                    replay_block_mask=replay_block_mask,
                    draft_block_size=draft_block_size,
                    lm_head_chunk_size=lm_head_chunk_size,
                    profile_enabled=profile_enabled,
                )
            )
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
        dflash_use_tv_loss: bool = False,
        dflash_rejected_draft_first_token_only: bool = False,
        dflash_use_task_rewards: bool = False,
        dflash_calculate_entropy: bool = False,
        dflash_use_replay_dis: bool = False,
        dflash_replay_block_anchor_indices: Optional[torch.LongTensor] = None,
        dflash_replay_block_accepted_lengths: Optional[torch.LongTensor] = None,
        dflash_replay_block_drafted_lengths: Optional[torch.LongTensor] = None,
        dflash_replay_block_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if input_ids is None and inputs_embeds is None:
            raise ValueError("ComposedDFlashStudentForCausalLM requires either input_ids or inputs_embeds.")

        if dflash_reject_token_indices is not None:
            if dflash_prompt_lengths is None or dflash_response_lengths is None:
                raise ValueError("DFLASH OPD requires prompt and response lengths with reject-token indices.")
            if input_ids is None:
                raise ValueError("DFLASH OPD path requires input_ids.")
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
                use_tv_loss=bool(dflash_use_tv_loss),
                rejected_draft_first_token_only=bool(dflash_rejected_draft_first_token_only),
                use_task_rewards=bool(dflash_use_task_rewards),
                calculate_entropy=bool(dflash_calculate_entropy),
                use_replay_dis=bool(dflash_use_replay_dis),
                replay_block_anchor_indices=dflash_replay_block_anchor_indices,
                replay_block_accepted_lengths=dflash_replay_block_accepted_lengths,
                replay_block_drafted_lengths=dflash_replay_block_drafted_lengths,
                replay_block_mask=dflash_replay_block_mask,
                **kwargs,
            )

        # Forward the teacher under no_grad to produce DFLASH context features.
        with torch.no_grad():
            teacher_outputs = self.main_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
            if teacher_outputs.hidden_states is None:
                raise RuntimeError("Teacher model did not return hidden states required by DFLASH draft.")

            if inputs_embeds is None:
                noise_embedding = self.get_input_embeddings()(input_ids)
            else:
                noise_embedding = inputs_embeds
            target_hidden = self._extract_target_hidden(teacher_outputs.hidden_states)

        draft_position_ids = self._resolve_position_ids(noise_embedding=noise_embedding, position_ids=position_ids)
        draft_outputs = self.draft_model(
            position_ids=draft_position_ids,
            attention_mask=None,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            use_cache=bool(use_cache),
        )

        if isinstance(draft_outputs, tuple):
            draft_hidden = draft_outputs[0]
        elif hasattr(draft_outputs, "last_hidden_state"):
            draft_hidden = draft_outputs.last_hidden_state
        else:
            draft_hidden = draft_outputs

        output_embeddings = self.get_output_embeddings()
        if output_embeddings is None:
            raise RuntimeError("Main model output embeddings are required for composed DFLASH student logits.")
        logits = output_embeddings(draft_hidden)
        model_hidden_states = (draft_hidden,) if output_hidden_states else None

        if not return_dict:
            output = (logits,)
            if model_hidden_states is not None:
                output = output + (model_hidden_states,)
            return output

        return CausalLMOutputWithPast(
            logits=logits,
            hidden_states=model_hidden_states,
        )


def build_composed_dflash_student(
    *,
    main_model_path: str,
    draft_model_path: str,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
    config: PretrainedConfig,
) -> ComposedDFlashStudentForCausalLM:
    main_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=main_model_path,
        torch_dtype=torch_dtype,
        config=config,
        trust_remote_code=trust_remote_code,
    )

    try:
        draft_model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=draft_model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    except OSError as exc:
        logger.warning(
            "Failed to load draft model weights from %s (%s). Falling back to config-only initialization.",
            draft_model_path,
            exc,
        )
        draft_config = AutoConfig.from_pretrained(draft_model_path, trust_remote_code=True)
        draft_model = AutoModel.from_config(draft_config, trust_remote_code=True)

    model = ComposedDFlashStudentForCausalLM(config=config, main_model=main_model, draft_model=draft_model)
    return model
