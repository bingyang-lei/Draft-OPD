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
"""vLLM(-Ascend) engine-side patch: export per-request DFlash reject metadata.

OPD (On-Policy Distillation) training needs to know, for every rollout
request, *where* the target model rejected draft tokens during speculative
verification and what the target's logprob of the rejected draft token was.
The SGLang rollout path gets this from the sglang-dflash fork
(``dflash_reject_token_mask`` etc. in the response meta_info). Stock
vLLM/vllm-ascend does not expose it, so this module monkey-patches the
speculative-decode sampling seam to record it.

Design (vllm v1 + vllm-ascend, ``NPUModelRunner``):

1. ``NPUModelRunner._sample(logits, spec_decode_metadata)`` is wrapped. After
   the original call returns ``SamplerOutput``, the wrapper replays the
   verification per request on-device: a draft token at block offset ``j``
   was accepted iff ``sampled_token_ids[req, j] == draft_token_ids[req, j]``
   for every ``j`` before the first mismatch. The first mismatch position is
   the rejection point; its block offset and the rejected draft token id are
   recorded, together with the target logprob of the rejected draft token
   (gathered from the raw target logits, TP-aware).
2. Events accumulate in a per-request registry
   (:class:`DFlashOpdMetadataRegistry`). Response positions are tracked
   cumulatively: a request's response is assumed to start with the single
   prefill-sampled token (position 0), and every verify step appends
   ``num_accepted + 1`` tokens (accepted prefix + recovered/bonus token).
3. The vLLM rollout server (``vLLMHttpServer.generate``) retrieves the record
   after the request finishes via ``collective_rpc`` to the worker extension
   method :func:`get_dflash_opd_metadata`, and converts it into the
   ``dflash_*`` extra fields with
   ``verl.workers.rollout.vllm_rollout.opd_utils.build_dflash_extra_fields``.

Activation: set ``VERL_DFLASH_OPD=1`` in the server environment (done by
``vLLMHttpServer`` when the model is a composed DFlash student) and let the
worker extension call :func:`apply_dflash_opd_patches`. Both are no-ops when
the server runs without speculative decoding.

Limitations / assumptions:

- Requests contribute verify events only from decode steps; preemption or
  discarded steps are not rolled back (OPD rollouts run with enough KV head
  room; aborted requests are dropped by TTL cleanup).
- Teacher logprobs are computed from the raw target logits (optionally
  temperature-scaled), i.e. before top-k/top-p truncation.
- TP > 1: every rank computes identical sampler outputs, so
  :func:`get_dflash_opd_metadata` results from any rank are equivalent; the
  caller keeps the first non-empty one.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

ENV_FLAG = "VERL_DFLASH_OPD"

# Registry entries are dropped after this many seconds without being popped,
# to bound memory when requests are aborted mid-flight.
_REGISTRY_TTL_SECONDS = 3600.0


def opd_enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0") == "1"


class DFlashOpdMetadataRegistry:
    """Per-process registry of speculative-verification reject events.

    Events per request (append order == verify step order):

    .. code-block:: text

        (num_accepted, num_draft, rejected_token_id, teacher_logprob)

    ``rejected_token_id`` is ``-1`` when the whole draft block was accepted
    (bonus token appended, no rejection). ``teacher_logprob`` is the target
    distribution's logprob of the rejected draft token (``nan`` when not
    applicable).
    """

    def __init__(self):
        self._events: dict[str, list[tuple[int, int, int, float]]] = {}
        self._response_lens: dict[str, int] = {}
        self._timestamps: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Recording (model-runner side)
    # ------------------------------------------------------------------
    def record_step(
        self,
        req_id: str,
        num_accepted: int,
        num_draft: int,
        rejected_token_id: int,
        teacher_logprob: float,
    ) -> None:
        import time

        if req_id not in self._events:
            # First verify step for this request: the response currently holds
            # exactly the prefill-sampled token at position 0.
            self._events[req_id] = []
            self._response_lens[req_id] = 1
        self._events[req_id].append((num_accepted, num_draft, rejected_token_id, teacher_logprob))
        # Every verify step appends accepted prefix + one recovered/bonus token.
        self._response_lens[req_id] += num_accepted + 1
        self._timestamps[req_id] = time.monotonic()

    # ------------------------------------------------------------------
    # Retrieval (frontend side, via collective_rpc on the worker extension)
    # ------------------------------------------------------------------
    def pop(self, req_id: str) -> Optional[dict[str, Any]]:
        """Pop and materialize the OPD metadata dict for a finished request."""
        events = self._events.pop(req_id, None)
        self._response_lens.pop(req_id, None)
        self._timestamps.pop(req_id, None)
        if events is None:
            return None

        anchor_indices: list[int] = []
        offsets: list[int] = []
        token_ids: list[int] = []
        teacher_logprobs: list[float] = []

        # Replay the steps to recover response positions. The response starts
        # with the prefill token at position 0.
        pos = 1
        for num_accepted, _num_draft, rejected_token_id, teacher_logprob in events:
            pos += num_accepted
            if rejected_token_id >= 0:
                anchor_indices.append(pos)
                offsets.append(num_accepted + 1)
                token_ids.append(rejected_token_id)
                teacher_logprobs.append(teacher_logprob)
            pos += 1  # recovered/bonus token

        return {
            "rejected_draft_anchor_indices": anchor_indices,
            "rejected_draft_offsets": offsets,
            "rejected_draft_token_ids": token_ids,
            "rejected_draft_teacher_logprobs": teacher_logprobs,
            "_num_verify_steps": len(events),
        }

    def gc(self) -> int:
        """Drop stale entries (aborted requests). Returns the dropped count."""
        import time

        now = time.monotonic()
        stale = [rid for rid, ts in self._timestamps.items() if now - ts > _REGISTRY_TTL_SECONDS]
        for rid in stale:
            self._events.pop(rid, None)
            self._response_lens.pop(rid, None)
            self._timestamps.pop(rid, None)
        return len(stale)


# Process-local registry. The model runner and the worker extension live in
# the same process, so both share this object.
_REGISTRY = DFlashOpdMetadataRegistry()


def get_registry() -> DFlashOpdMetadataRegistry:
    return _REGISTRY


# ---------------------------------------------------------------------------
# Teacher logprob of arbitrary tokens under tensor parallelism.
# ---------------------------------------------------------------------------
def _gather_token_logprobs(
    target_logits: torch.Tensor,
    token_ids: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """log p_target(token_id) for each (row, token_id) pair, TP-aware.

    Args:
        target_logits: ``[num_rows, vocab_size // tp_size]`` raw logits for the
            verify positions of interest (pre top-k/top-p).
        token_ids: ``[num_rows]`` global token ids (the rejected draft tokens).
        temperature: sampling temperature to scale the logits with (1.0 = raw).

    Returns:
        ``[num_rows]`` float32 logprobs on the same device.
    """
    from vllm.distributed.parallel_state import get_tp_group

    if temperature and temperature != 1.0:
        target_logits = target_logits / temperature

    tp_group = get_tp_group()
    world = tp_group.world_size
    local_vocab = target_logits.shape[-1]

    # Local logsumexp, then combine across shards: lse = log(sum(exp(lse_i))).
    local_lse = target_logits.float().logsumexp(dim=-1)
    # Local token logit; -inf when the token lives on another shard.
    rank = tp_group.rank_in_group
    start = rank * local_vocab
    in_range = (token_ids >= start) & (token_ids < start + local_vocab)
    local_idx = (token_ids - start).clamp(min=0, max=local_vocab - 1)
    local_token_logit = target_logits.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1).float()
    local_token_logit = torch.where(in_range, local_token_logit, torch.full_like(local_token_logit, float("-inf")))

    if world == 1:
        return local_token_logit - local_lse

    gathered_lse = tp_group.all_gather(local_lse.unsqueeze(-1), dim=-1)  # [rows, world]
    global_lse = gathered_lse.logsumexp(dim=-1)
    gathered_token = tp_group.all_gather(local_token_logit.unsqueeze(-1), dim=-1)  # [rows, world]
    global_token_logit = gathered_token.max(dim=-1).values
    return global_token_logit - global_lse


# ---------------------------------------------------------------------------
# The sampler seam wrapper
# ---------------------------------------------------------------------------
def _record_verify_events(
    model_runner,
    spec_decode_metadata,
    logits: torch.Tensor,
    sampling_metadata,
    sampler_output,
) -> None:
    """Derive per-request reject events from one verify step and record them."""
    try:
        _record_verify_events_impl(model_runner, spec_decode_metadata, logits, sampling_metadata, sampler_output)
    except Exception:  # never break serving for observability
        logger.exception("dflash_opd_patch: failed to record verify events")


def _record_verify_events_impl(
    model_runner,
    spec_decode_metadata,
    logits: torch.Tensor,
    sampling_metadata,
    sampler_output,
) -> None:
    from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID

    input_batch = model_runner.input_batch
    req_ids = list(input_batch.req_ids)
    if not req_ids:
        return

    num_draft_per_req = list(spec_decode_metadata.num_draft_tokens)
    if len(num_draft_per_req) != len(req_ids):
        # Row order/length mismatch; skip rather than record garbage.
        logger.warning(
            "dflash_opd_patch: num_draft_tokens len %d != batch req count %d; skip step",
            len(num_draft_per_req),
            len(req_ids),
        )
        return

    # Per-request accepted counts. A verify row always contains exactly
    # ``num_accepted + 1`` real tokens (accepted prefix + recovered/bonus
    # token) followed by PLACEHOLDER padding, so counting non-placeholder
    # tokens gives num_accepted without comparing token values. This stays
    # correct for random rejection sampling, where the recovered token may
    # coincidentally equal the rejected draft token.
    sampled = sampler_output.sampled_token_ids  # [batch, max_spec_len + 1]
    if sampled is None or sampled.ndim != 2:
        return
    batch_size = sampled.shape[0]
    if batch_size != len(req_ids):
        return

    draft_token_ids = spec_decode_metadata.draft_token_ids  # [num_tokens]

    # Slice this step's draft rows and align raw target logits.
    raw_target_logits = logits[spec_decode_metadata.target_logits_indices]

    # Single device sync for the whole batch: per-row valid-token counts and
    # the draft ids (needed only for the rare rejected events).
    n_valid_per_row = (sampled != PLACEHOLDER_TOKEN_ID).sum(dim=1).tolist()
    draft_ids_list = draft_token_ids.tolist()

    rejected_events: list[tuple[int, int, int, int]] = []  # (row, num_accepted, rejected_token, flat_draft_row)
    accepted_counts: list[int] = []
    row_start = 0
    for row in range(batch_size):
        n_draft = int(num_draft_per_req[row])
        row_end = row_start + n_draft
        n_valid = int(n_valid_per_row[row])
        n_accepted = max(min(n_valid - 1, n_draft), 0)
        accepted_counts.append(n_accepted)
        if 0 < n_draft and n_accepted < n_draft:
            rejected_token = int(draft_ids_list[row_start + n_accepted])
            rejected_events.append((row, n_accepted, rejected_token, row_start + n_accepted))
        row_start = row_end

    # Debug: dump per-row acceptance stats for the first few verify steps,
    # including the request id (to tell real traffic from dummy/profile runs)
    # and the target logits magnitude (to spot degenerate forward passes).
    # Enable with OPD_PATCH_DEBUG=1 in the rollout server environment.
    if os.environ.get("OPD_PATCH_DEBUG") == "1":
        _dbg_cnt = getattr(_record_verify_events_impl, "_dbg_cnt", 0)
        if _dbg_cnt < 8:
            _record_verify_events_impl._dbg_cnt = _dbg_cnt + 1
            logger.info(
                "OPD_PATCH_DEBUG n=%d req0=%s num_draft=%s n_valid=%s sampled_row0=%s draft_row0=%s "
                "rejected_events=%s logits_absmax=%.4f",
                _dbg_cnt,
                req_ids[0],
                num_draft_per_req,
                n_valid_per_row,
                sampled[0].detach().cpu().tolist(),
                draft_ids_list[: int(num_draft_per_req[0]) if num_draft_per_req else 8],
                rejected_events[:4],
                float(raw_target_logits.abs().max().item()) if raw_target_logits.numel() else float("nan"),
            )

    teacher_lps: dict[int, float] = {}
    if rejected_events:
        rows = torch.tensor([e[3] for e in rejected_events], device=raw_target_logits.device, dtype=torch.long)
        toks = torch.tensor([e[2] for e in rejected_events], device=raw_target_logits.device, dtype=torch.long)
        temperature = 1.0
        temp_tensor = getattr(sampling_metadata, "temperature", None)
        if temp_tensor is not None:
            # Per-request temperature; use the max to stay simple and document
            # the approximation (OPD rollouts run a single temperature).
            try:
                temperature = float(temp_tensor.max().item())
                if temperature <= 0:
                    temperature = 1.0
            except Exception:
                temperature = 1.0
        lps = _gather_token_logprobs(raw_target_logits[rows], toks, temperature=temperature)
        for event_idx, lp in enumerate(lps.tolist()):
            teacher_lps[event_idx] = float(lp)

    registry = get_registry()
    event_idx_by_row = {e[0]: i for i, e in enumerate(rejected_events)}
    for row in range(batch_size):
        req_id = req_ids[row]
        n_draft = int(num_draft_per_req[row])
        event_idx = event_idx_by_row.get(row)
        if event_idx is None:
            # Includes n_draft == 0 rows (plain decode step): record them so
            # the response-position replay in pop() still advances by the one
            # sampled token and later anchors stay aligned.
            registry.record_step(req_id, accepted_counts[row], n_draft, -1, float("nan"))
        else:
            event = rejected_events[event_idx]
            registry.record_step(
                req_id,
                event[1],
                n_draft,
                event[2],
                teacher_lps.get(event_idx, float("nan")),
            )


def apply_dflash_opd_patches() -> bool:
    """Idempotently wrap ``NPUModelRunner._sample`` to record reject metadata.

    Returns True when the patch was (or already had been) applied. Never
    raises: any failure is logged and reported as False, so a version drift
    in vllm-ascend cannot kill the engine workers at startup (the rollout
    server's startup probe surfaces the broken state instead).
    """
    try:
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    except Exception as exc:  # noqa: BLE001
        logger.warning("dflash_opd_patch: vllm_ascend import failed (%r); patch NOT applied", exc)
        return False

    try:
        if getattr(NPUModelRunner, "_dflash_opd_patched", False):
            return True

        original_sample = getattr(NPUModelRunner, "_sample", None)
        if original_sample is None:
            logger.warning(
                "dflash_opd_patch: NPUModelRunner has no '_sample' attribute in this vllm-ascend "
                "version (checked %s); patch NOT applied",
                type(NPUModelRunner).__module__,
            )
            return False

        def _sample_with_opd_recording(self, logits, spec_decode_metadata):
            sampler_output = original_sample(self, logits, spec_decode_metadata)
            if spec_decode_metadata is not None and opd_enabled():
                _record_verify_events(self, spec_decode_metadata, logits, self.input_batch.sampling_metadata, sampler_output)
            return sampler_output

        NPUModelRunner._sample = _sample_with_opd_recording
        NPUModelRunner._dflash_opd_patched = True
        logger.info("dflash_opd_patch: NPUModelRunner._sample wrapped for OPD reject metadata recording")
        return True
    except Exception:  # noqa: BLE001
        logger.exception("dflash_opd_patch: failed to apply; patch NOT applied")
        return False


# ---------------------------------------------------------------------------
# Worker-extension entry points (invoked via collective_rpc from the frontend)
# ---------------------------------------------------------------------------
def get_dflash_opd_metadata(request_id: str) -> Optional[dict[str, Any]]:
    """Pop the recorded OPD metadata for a finished request."""
    return get_registry().pop(request_id)


def peek_dflash_opd_metadata(request_id: str) -> Optional[dict[str, Any]]:
    """Read without popping (debugging)."""
    registry = get_registry()
    events = registry._events.get(request_id)
    if events is None:
        return None
    # Materialize a copy via pop on a shadow registry.
    shadow = DFlashOpdMetadataRegistry()
    shadow._events[request_id] = list(events)
    shadow._response_lens[request_id] = registry._response_lens.get(request_id, 1)
    return shadow.pop(request_id)


def gc_dflash_opd_metadata() -> int:
    """Drop stale registry entries; returns the dropped count."""
    return get_registry().gc()
