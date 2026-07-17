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
"""OPD (On-Policy Distillation) helpers for the vLLM rollout path.

This module mirrors the DFLASH OPD contract of the SGLang rollout path (see
``verl/workers/rollout/sglang_rollout/sglang_rollout.py`` and
``verl/workers/rollout/sglang_rollout/utils.py``) so that the training side —
the composed DFLASH student and the distillation losses — stays
engine-agnostic.

Two things live here:

1. Draft-only weight-sync helpers (:func:`map_opd_draft_weight_name` and
   :func:`iter_opd_draft_weights`) that strip FSDP/wrapper prefixes and keep
   only the ``draft_model.*`` subtree of the composed student, renamed to the
   draft-model-relative keys the inference engine expects.
2. ``extra_fields`` construction (:func:`build_dflash_extra_fields`) from the
   per-request reject metadata produced by the engine-side patch
   (:mod:`verl.utils.vllm.dflash_opd_patch`), matching the keys consumed by
   ``verl/experimental/agent_loop/agent_loop.py``.
"""

from __future__ import annotations

from typing import Any, Generator, Iterator, Optional

import torch

# Keys consumed by agent_loop.AgentLoopOutput.to_dict() and by the composed
# DFLASH student forward. Keep in sync with the SGLang rollout path.
DFLASH_REJECT_TOKEN_INDICES = "dflash_reject_token_indices"
DFLASH_REJECTED_DRAFT_ANCHOR_INDICES = "dflash_rejected_draft_anchor_indices"
DFLASH_REJECTED_DRAFT_OFFSETS = "dflash_rejected_draft_offsets"
DFLASH_REJECTED_DRAFT_TOKEN_IDS = "dflash_rejected_draft_token_ids"
DFLASH_REJECTED_DRAFT_TEACHER_LOGPROBS = "dflash_rejected_draft_teacher_logprobs"

# Engine-side metadata dict keys produced by the vLLM(-Ascend) patch.
META_REJECTED_ANCHOR_INDICES = "rejected_draft_anchor_indices"
META_REJECTED_OFFSETS = "rejected_draft_offsets"
META_REJECTED_TOKEN_IDS = "rejected_draft_token_ids"
META_REJECTED_TEACHER_LOGPROBS = "rejected_draft_teacher_logprobs"


def map_opd_draft_weight_name(weight_name: str) -> Optional[str]:
    """Map a composed-student weight name to the draft-model-relative name.

    Returns ``None`` for weights that do not belong to the draft subtree.

    Examples:
        ``draft_model.model.layers.0.self_attn.q_proj.weight``
            -> ``model.layers.0.self_attn.q_proj.weight``
        ``_fsdp_wrapped_module.draft_model.fc.weight`` -> ``fc.weight``
        ``main_model.model.layers.0...`` -> ``None``
    """
    normalized_name = weight_name
    # Remove common distributed wrappers first.
    while normalized_name.startswith(("_fsdp_wrapped_module.", "module.")):
        if normalized_name.startswith("_fsdp_wrapped_module."):
            normalized_name = normalized_name[len("_fsdp_wrapped_module.") :]
        elif normalized_name.startswith("module."):
            normalized_name = normalized_name[len("module.") :]

    if normalized_name.startswith("draft_model."):
        return normalized_name[len("draft_model.") :]

    draft_marker = ".draft_model."
    marker_idx = normalized_name.find(draft_marker)
    if marker_idx >= 0:
        return normalized_name[marker_idx + len(draft_marker) :]

    return None


def iter_opd_draft_weights(
    weights: Generator[tuple[str, torch.Tensor], None, None],
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Keep only draft-subtree weights, renamed to draft-relative keys."""
    for name, tensor in weights:
        mapped_name = map_opd_draft_weight_name(name)
        if mapped_name is None:
            continue
        yield mapped_name, tensor


def build_dflash_extra_fields(metadata: Optional[dict[str, Any]], response_len: int) -> dict[str, Any]:
    """Build OPD ``extra_fields`` from engine-side reject metadata.

    Args:
        metadata: Per-request metadata dict produced by the engine-side patch
            (see :mod:`verl.utils.vllm.dflash_opd_patch`), or ``None`` when the
            patch is not active / the request has no metadata. Expected keys::

                rejected_draft_anchor_indices: list[int]
                    response positions (0-based) where a draft token was
                    rejected and replaced by the target token.
                rejected_draft_offsets: list[int]
                    1-based offset of the rejected draft token inside its
                    draft block (used for position-decayed loss weights).
                rejected_draft_token_ids: list[int]
                    the rejected draft token ids.
                rejected_draft_teacher_logprobs: list[float]
                    target logprob of each rejected draft token.

        response_len: Number of tokens in the produced response.

    Returns:
        A dict suitable for ``TokenOutput.extra_fields``. When ``metadata`` is
        ``None``, an empty dict is returned (the request is treated as a
        non-OPD rollout and no DFLASH keys are emitted).
    """
    if metadata is None or response_len <= 0:
        return {}

    anchor_indices = [int(pos) for pos in metadata.get(META_REJECTED_ANCHOR_INDICES, [])]
    offsets = [int(off) for off in metadata.get(META_REJECTED_OFFSETS, [])]
    token_ids = [int(tid) for tid in metadata.get(META_REJECTED_TOKEN_IDS, [])]
    teacher_logprobs = [float(lp) for lp in metadata.get(META_REJECTED_TEACHER_LOGPROBS, [])]

    if not (len(anchor_indices) == len(offsets) == len(token_ids) == len(teacher_logprobs)):
        raise ValueError(
            "Inconsistent DFLASH reject metadata lengths: "
            f"anchors={len(anchor_indices)}, offsets={len(offsets)}, "
            f"token_ids={len(token_ids)}, teacher_logprobs={len(teacher_logprobs)}"
        )

    # Keep only anchors that fall inside the produced response. The final
    # verify step may reject draft tokens beyond the emitted response (e.g.
    # when generation stops on EOS/length), which cannot be supervised.
    in_range = [i for i, pos in enumerate(anchor_indices) if 0 <= pos < response_len]
    if len(in_range) != len(anchor_indices):
        anchor_indices = [anchor_indices[i] for i in in_range]
        offsets = [offsets[i] for i in in_range]
        token_ids = [token_ids[i] for i in in_range]
        teacher_logprobs = [teacher_logprobs[i] for i in in_range]

    extra_fields: dict[str, Any] = {
        DFLASH_REJECT_TOKEN_INDICES: anchor_indices,
        DFLASH_REJECTED_DRAFT_ANCHOR_INDICES: anchor_indices,
        DFLASH_REJECTED_DRAFT_OFFSETS: offsets,
        DFLASH_REJECTED_DRAFT_TOKEN_IDS: token_ids,
        DFLASH_REJECTED_DRAFT_TEACHER_LOGPROBS: teacher_logprobs,
    }

    # Debug/monitoring counters, mirroring the SGLang rollout path.
    extra_fields["dflash_reject_token_count"] = len(anchor_indices)
    extra_fields["dflash_non_reject_token_count"] = max(response_len - len(anchor_indices), 0)
    extra_fields["dflash_empty_reject_token_indices"] = int(response_len > 0 and len(anchor_indices) == 0)

    return extra_fields


def merge_dflash_metadata(parts: Iterator[Optional[dict[str, Any]]]) -> Optional[dict[str, Any]]:
    """Merge per-rank metadata parts (from collective_rpc over TP workers).

    The engine-side patch records reject metadata on every worker that runs
    the rejection sampler. With TP > 1 each worker holds identical values, so
    the first non-empty part wins; this helper exists mainly to tolerate
    ``None`` parts from ranks that did not see the request.
    """
    for part in parts:
        if part:
            return part
    return None
