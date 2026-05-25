"""DFLASH spec-v2 overlap scheduling data structures (WIP).

The spec-v2 path will mirror the scheduler integration used by Eagle v2:
- the worker returns `(next_token_ids, accept_lens, next_draft_input)`
- scheduler output processing (not the worker) mutates `req.output_ids`

This file is intentionally introduced early to keep spec-v2-specific state
isolated from the existing spec-v1 implementation in `dflash_info.py`.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.overlap_utils import FutureIndices
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

_OVERLAP_PLAN_STREAMS: dict[str, torch.cuda.Stream] = {}


def _get_overlap_plan_stream(
    device: torch.device | str,
) -> tuple[Optional[torch.cuda.Stream], contextlib.AbstractContextManager]:
    """Return an optional plan stream/context for overlap scheduling prep kernels."""
    if not envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        return None, contextlib.nullcontext()

    device_str = str(device)
    stream = _OVERLAP_PLAN_STREAMS.get(device_str)
    if stream is None:
        stream = torch.get_device_module(device_str).Stream()
        _OVERLAP_PLAN_STREAMS[device_str] = stream
    return stream, torch.get_device_module(device_str).stream(stream)


@dataclass
class DFlashDraftInputV2(SpecInput):
    """Draft-side state carried across overlap iterations (spec-v2)."""

    # Required by overlap FutureMap plumbing (match Eagle v2 field names).
    topk_p: torch.Tensor
    topk_index: torch.Tensor
    verified_id: torch.Tensor
    new_seq_lens: torch.Tensor
    hidden_states: torch.Tensor
    verify_done: Optional[torch.cuda.Event] = None
    max_top_k: int = 1
    uniform_top_k_value: Optional[int] = None

    # Filled by scheduler after dispatch.
    future_indices: Optional[FutureIndices] = None

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        # Spec v2 draft state itself does not change token accounting.
        return (1, 1)

    @classmethod
    def create_idle_input(cls, device: torch.device) -> "DFlashDraftInputV2":
        return cls(
            topk_p=torch.empty((0, 1), device=device, dtype=torch.float32),
            topk_index=torch.empty((0, 1), device=device, dtype=torch.int64),
            verified_id=torch.empty((0,), device=device, dtype=torch.int32),
            new_seq_lens=torch.empty((0,), device=device, dtype=torch.int32),
            hidden_states=torch.empty((0, 1), device=device, dtype=torch.float16),
            verify_done=None,
        )

    def prepare_for_decode(self, batch: ScheduleBatch):
        """Allocate headroom in the shared req_to_token pool for the next DFLASH step.

        DFLASH spec-v2 uses overlap scheduling's "over-allocation" approach: we reserve
        future KV slots ahead of time so the worker can gather `out_cache_loc` directly
        from `req_to_token` without allocator backup/restore.
        """
        plan_stream, plan_stream_ctx = _get_overlap_plan_stream(batch.device)
        if plan_stream is None:
            # Ensure previous forward is completed before mutating shared buffers.
            batch.maybe_wait_verify_done()

        bs = batch.batch_size()
        if bs == 0:
            return

        # For DFLASH, each decode step needs a fixed-size verify block.
        block_size = int(get_global_server_args().speculative_num_draft_tokens)
        if block_size <= 0:
            raise ValueError(
                f"DFLASH invalid speculative_num_draft_tokens={block_size}."
            )

        top_ks = [int(req.sampling_params.top_k) for req in batch.reqs]
        self.max_top_k = max(max(top_ks), 1) if top_ks else 1
        self.uniform_top_k_value = (
            top_ks[0]
            if top_ks and all(top_k == top_ks[0] for top_k in top_ks)
            else None
        )

        page_size = batch.token_to_kv_pool_allocator.page_size

        cur_kv_lens_cpu_t = torch.tensor(
            [req.kv_allocated_len for req in batch.reqs],
            dtype=torch.int32,
            device="cpu",
        )

        caller_stream = None
        if plan_stream is not None:
            caller_stream = torch.get_device_module(batch.device).current_stream()

        with plan_stream_ctx:
            if plan_stream is not None and caller_stream is not None:
                # `batch.seq_lens`, `batch.req_pool_indices`, and related tensors may
                # have just been rebuilt on the scheduler stream by filter/merge ops.
                # The plan stream must wait for those writes before reading them.
                plan_stream.wait_stream(caller_stream)

            if plan_stream is not None and self.verify_done is not None:
                plan_stream.wait_event(self.verify_done)

            cur_kv_lens = cur_kv_lens_cpu_t.to(device=batch.device)
            committed_kv_lens = batch.seq_lens.to(dtype=torch.int32)
            nxt_kv_lens = torch.maximum(cur_kv_lens, committed_kv_lens + 2 * block_size)
            nxt_kv_lens_cpu_t = nxt_kv_lens.to(device="cpu")
            num_needed_tokens = int(
                (nxt_kv_lens_cpu_t - cur_kv_lens_cpu_t).sum().item()
            )

            if num_needed_tokens > 0:
                if page_size == 1:
                    out_cache_loc = alloc_token_slots(
                        batch.tree_cache, num_needed_tokens
                    )
                else:
                    last_loc = get_last_loc(
                        batch.req_to_token_pool.req_to_token,
                        batch.req_pool_indices,
                        cur_kv_lens,
                    )
                    out_cache_loc = alloc_paged_token_slots_extend(
                        batch.tree_cache,
                        cur_kv_lens,
                        cur_kv_lens_cpu_t,
                        nxt_kv_lens,
                        nxt_kv_lens_cpu_t,
                        last_loc,
                        num_needed_tokens,
                    )

                # Updating req_to_token is a write to a shared tensor: it must not overlap
                # with the previous batch's forward, which also reads req_to_token.
                assign_req_to_token_pool_func(
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    cur_kv_lens,
                    nxt_kv_lens,
                    out_cache_loc,
                    bs,
                )
        if caller_stream is not None:
            # Enqueue the dependency on the caller's stream, not inside the
            # plan-stream context, so forward work cannot observe partially
            # prepared req_to_token / KV allocation state.
            caller_stream.wait_stream(plan_stream)

        nxt_kv_lens_cpu = nxt_kv_lens_cpu_t.tolist()
        for req, new_alloc_len in zip(batch.reqs, nxt_kv_lens_cpu, strict=True):
            req.kv_allocated_len = int(new_alloc_len)

        # NOTE: In overlap scheduling, per-request CPU state (e.g., `req.kv_committed_len`)
        # can lag behind `batch.seq_lens` by one iteration because result processing is
        # overlapped with the next forward. Avoid using lagging CPU state for buffer sizing,
        # and never force a GPU->CPU sync here.
        #
        # `seq_lens_sum` is used for allocation sizing in attention backends (e.g., FlashInfer
        # kv_indices buffers). Use allocated KV lengths as a safe upper bound.
        batch.seq_lens_cpu = nxt_kv_lens_cpu_t.to(dtype=torch.int64)
        batch.seq_lens_sum = int(nxt_kv_lens_cpu_t.sum().item())

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        if self.future_indices is not None:
            self.future_indices.indices = self.future_indices.indices[new_indices]
            return

        self.topk_p = self.topk_p[new_indices]
        self.topk_index = self.topk_index[new_indices]
        self.verified_id = self.verified_id[new_indices]
        self.new_seq_lens = self.new_seq_lens[new_indices]
        self.hidden_states = self.hidden_states[new_indices]

    def merge_batch(self, spec_info: "DFlashDraftInputV2"):
        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            self.future_indices = FutureIndices(
                indices=torch.cat(
                    [self.future_indices.indices, spec_info.future_indices.indices]
                )
            )
            return

        self.topk_p = torch.cat([self.topk_p, spec_info.topk_p], dim=0)
        self.topk_index = torch.cat([self.topk_index, spec_info.topk_index], dim=0)
        self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], dim=0)
        self.new_seq_lens = torch.cat(
            [self.new_seq_lens, spec_info.new_seq_lens], dim=0
        )
        self.hidden_states = torch.cat(
            [self.hidden_states, spec_info.hidden_states], dim=0
        )
