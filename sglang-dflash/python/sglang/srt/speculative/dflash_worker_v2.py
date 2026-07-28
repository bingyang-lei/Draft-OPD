import logging
from typing import Optional

import torch

from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    compute_position,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
    compute_dflash_accept_len_and_bonus,
    compute_dflash_candidate_logprobs,
    compute_dflash_sampling_accept_len_and_bonus,
    is_dflash_sampling_verify_available,
    make_dflash_verify_uniform_samples,
)
from sglang.srt.speculative.dflash_worker import DFlashWorker
from sglang.srt.speculative.eagle_info_v2 import assign_extend_cache_locs_func
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

logger = logging.getLogger(__name__)


class DFlashWorkerV2(DFlashWorker):
    """DFLASH speculative decoding worker (spec-v2 overlap scheduling).

    This is intentionally implemented as a *separate* worker from the existing
    spec-v1 `DFlashWorker` (non-overlap), to keep the v1 path stable and to
    minimize risk while bringing up overlap scheduling.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )

    def _validate_phase1_sampling_support(
        self, model_worker_batch: ModelWorkerBatch
    ) -> None:
        sampling_info = model_worker_batch.sampling_info
        if sampling_info is None or sampling_info.is_all_greedy:
            return

        if (
            not is_dflash_sampling_verify_available()
            and not self._warned_sampling_fallback
            and self.tp_rank == 0
        ):
            logger.warning(
                "DFLASH non-greedy verification is unavailable on this build/device; "
                "falling back to greedy argmax verification."
            )
            self._warned_sampling_fallback = True

    def _make_next_draft_input_prefill(
        self,
        *,
        verified_id: torch.Tensor,
        seq_lens: torch.Tensor,
        verify_done: Optional[torch.cuda.Event] = None,
    ) -> DFlashDraftInputV2:
        bs = int(seq_lens.numel())
        device = verified_id.device
        return DFlashDraftInputV2(
            topk_p=torch.ones((bs, 1), device=device, dtype=torch.float32),
            topk_index=torch.zeros((bs, 1), device=device, dtype=torch.int64),
            verified_id=verified_id.to(dtype=torch.int32),
            new_seq_lens=seq_lens.to(dtype=torch.int32),
            hidden_states=torch.empty((bs, 1), device=device, dtype=torch.float16),
            verify_done=verify_done,
        )

    def _make_next_draft_input_decode(
        self,
        *,
        verified_id: torch.Tensor,
        new_seq_lens: torch.Tensor,
        verify_done: Optional[torch.cuda.Event] = None,
    ) -> DFlashDraftInputV2:
        bs = int(new_seq_lens.numel())
        device = verified_id.device
        return DFlashDraftInputV2(
            topk_p=torch.ones((bs, 1), device=device, dtype=torch.float32),
            topk_index=torch.zeros((bs, 1), device=device, dtype=torch.int64),
            verified_id=verified_id.to(dtype=torch.int32),
            new_seq_lens=new_seq_lens.to(dtype=torch.int32),
            hidden_states=torch.empty((bs, 1), device=device, dtype=torch.float16),
            verify_done=verify_done,
        )

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        **kwargs,
    ) -> GenerationBatchResult:
        if getattr(model_worker_batch, "return_logprob", False):
            raise ValueError(
                "DFLASH speculative decoding does not support return_logprob yet."
            )
        self._validate_phase1_sampling_support(model_worker_batch)

        if (
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        ):
            # Target prefill: capture DFlash aux hidden states for prompt tokens.
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_output = self.target_worker.forward_batch_generation(
                model_worker_batch, **kwargs
            )

            logits_output, next_token_ids = (
                batch_output.logits_output,
                batch_output.next_token_ids,
            )

            if logits_output.hidden_states is None:
                raise RuntimeError(
                    "DFLASH requires target aux hidden capture for prefill, but got None. "
                    "Make sure the target model has DFlash layers-to-capture configured."
                )

            if (
                model_worker_batch.extend_seq_lens is None
                or model_worker_batch.extend_prefix_lens is None
            ):
                raise RuntimeError(
                    "DFLASH expected extend_seq_lens / extend_prefix_lens to be populated in extend mode, "
                    "but got None."
                )

            # Materialize prompt tokens into the draft KV cache immediately. This is required
            # for radix cache safety (the scheduler may update radix after prefill returns).
            device = next_token_ids.device
            ctx_lens = torch.tensor(
                model_worker_batch.extend_seq_lens, dtype=torch.int32, device=device
            )
            draft_seq_lens = torch.tensor(
                model_worker_batch.extend_prefix_lens, dtype=torch.int32, device=device
            )

            if model_worker_batch.out_cache_loc is None:
                raise RuntimeError(
                    "DFLASH prefill expected out_cache_loc, but got None."
                )
            positions, _ = compute_position(
                self.model_runner.server_args.attention_backend,
                draft_seq_lens,
                ctx_lens,
                int(sum(model_worker_batch.extend_seq_lens)),
            )
            self._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=logits_output.hidden_states,
                cache_loc=model_worker_batch.out_cache_loc,
                positions=positions,
            )

            # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
            logits_output.hidden_states = None

            batch_output.next_draft_input = self._make_next_draft_input_prefill(
                verified_id=next_token_ids,
                seq_lens=model_worker_batch.seq_lens,
            )
            verify_done = torch.get_device_module(device).Event()
            verify_done.record()
            batch_output.next_draft_input.verify_done = verify_done
            return batch_output

        # Decode / target-verify stage.
        if model_worker_batch.spec_info is None:
            model_worker_batch.spec_info = DFlashDraftInputV2.create_idle_input(
                device=self.device
            )

        draft_input = model_worker_batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError(
                "DFLASH spec-v2 expected DFlashDraftInputV2 state on the running batch."
            )

        if model_worker_batch.forward_mode.is_idle():
            empty_ids = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_lens = torch.empty((0,), dtype=torch.int32, device=self.device)
            next_draft_input = self._make_next_draft_input_decode(
                verified_id=torch.empty((0,), device=self.device, dtype=torch.int32),
                new_seq_lens=torch.empty((0,), device=self.device, dtype=torch.int32),
            )
            verify_done = torch.get_device_module(self.device).Event()
            verify_done.record()
            next_draft_input.verify_done = verify_done
            return GenerationBatchResult(
                logits_output=None,
                next_token_ids=empty_ids,
                accept_lens=empty_lens,
                next_draft_input=next_draft_input,
                can_run_cuda_graph=False,
            )

        # `seq_lens` is carried over from the previous overlap iteration and may have been
        # produced on another stream.
        model_worker_batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        bs = len(model_worker_batch.seq_lens)
        device = self.device

        # --- 1) Draft a non-causal block with the draft model.
        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if (
            lm_head is None
            or not hasattr(lm_head, "weight")
            or not hasattr(lm_head, "shard_indices")
        ):
            raise RuntimeError(
                "DFLASH requires the target model to expose a vocab-parallel `lm_head` with `weight` and "
                "`shard_indices` attributes."
            )

        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        block_ids = self._draft_block_ids_buf[:bs]
        block_ids.fill_(int(self._mask_token_id))
        block_ids[:, 0].copy_(draft_input.verified_id.to(torch.long))

        noise_embedding = embed_module(block_ids)
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        prefix_lens = model_worker_batch.seq_lens
        positions_2d = self._draft_block_positions_buf[:bs]
        torch.add(
            prefix_lens.to(torch.int64).unsqueeze(1),
            self._block_pos_offsets,
            out=positions_2d,
        )
        positions = positions_2d.reshape(-1)

        end_offset = prefix_lens + int(self.block_size)
        verify_out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=model_worker_batch.req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            start_offset=prefix_lens,
            end_offset=end_offset,
            batch_size=bs,
            draft_token_num=int(self.block_size),
            device=device,
        )

        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
        if self.use_compact_draft_cache:
            # Rebuild the draft-local sliding-window view from committed target state.
            draft_prefix_lens = self._compute_compact_draft_seq_lens(prefix_lens)
            seq_lens_cpu.copy_(draft_prefix_lens.to(device="cpu", dtype=torch.int32))

            suffix_start = prefix_lens.to(torch.int64) - draft_prefix_lens.to(
                torch.int64
            )
            suffix_cache_loc = self._gather_req_to_token_segments(
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                req_pool_indices=model_worker_batch.req_pool_indices,
                start=suffix_start,
                lengths=draft_prefix_lens,
            )
            assign_req_to_token_pool_func(
                model_worker_batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                torch.zeros_like(draft_prefix_lens),
                draft_prefix_lens,
                suffix_cache_loc,
                bs,
            )

            block_end = self._draft_block_end_buf[:bs]
            torch.add(draft_prefix_lens, int(self.block_size), out=block_end)
            assign_req_to_token_pool_func(
                model_worker_batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                draft_prefix_lens,
                block_end,
                verify_out_cache_loc,
                bs,
            )
            draft_seq_lens = draft_prefix_lens
        else:
            # Non-windowed path uses the shared overallocated mapping directly.
            draft_seq_lens = prefix_lens
            if model_worker_batch.seq_lens_cpu is not None:
                if model_worker_batch.seq_lens_cpu.dtype == torch.int32:
                    seq_lens_cpu.copy_(model_worker_batch.seq_lens_cpu)
                else:
                    seq_lens_cpu.copy_(model_worker_batch.seq_lens_cpu.to(torch.int32))
            else:
                seq_lens_cpu.copy_(prefix_lens.to("cpu", dtype=torch.int32))

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=block_ids.flatten(),
            req_pool_indices=model_worker_batch.req_pool_indices,
            seq_lens=draft_seq_lens,
            out_cache_loc=verify_out_cache_loc,
            seq_lens_sum=int(draft_seq_lens.sum().item()),
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            req_to_token_pool=self.draft_model_runner.req_to_token_pool,
            token_to_kv_pool=self.draft_model_runner.token_to_kv_pool,
            attn_backend=self.draft_model_runner.attn_backend,
            input_embeds=input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

        with torch.inference_mode():
            draft_logits_output = self.draft_model_runner.forward(
                forward_batch
            ).logits_output

        draft_hidden = draft_logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("DFLASH draft model returned no hidden states.")
        draft_hidden = draft_hidden.view(bs, int(self.block_size), -1)
        draft_next = self._greedy_sample_from_vocab_parallel_head(
            hidden_states=draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1]),
            lm_head=lm_head,
        ).view(bs, int(self.block_size) - 1)

        draft_tokens = self._draft_block_tokens_buf[:bs]
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        draft_tokens[:, 1:].copy_(draft_next)

        # --- 2) Target verify.
        # TARGET_VERIFY uses standard causal masking; custom masks are unnecessary here.
        custom_mask = None

        verify_input_ids = draft_tokens.reshape(-1)
        verify_input = DFlashVerifyInput(
            draft_token=verify_input_ids,
            positions=positions,
            draft_token_num=int(self.block_size),
            custom_mask=custom_mask,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )

        model_worker_batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not model_worker_batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        model_worker_batch.input_ids = verify_input_ids
        model_worker_batch.out_cache_loc = verify_out_cache_loc
        model_worker_batch.spec_info = verify_input
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL

        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = (
            model_worker_batch.seq_lens.clone() if need_mamba_verify_commit else None
        )

        target_out = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True, **kwargs
        )
        logits_output = target_out.logits_output
        can_run_cuda_graph = target_out.can_run_cuda_graph

        sampling_info = model_worker_batch.sampling_info
        if sampling_info is not None:
            apply_dflash_verify_logits_adjustments(
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=int(self.block_size),
            )

        candidates = draft_tokens
        use_sampling_distribution = (
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and is_dflash_sampling_verify_available()
        )
        if use_sampling_distribution:
            (
                uniform_samples,
                uniform_samples_for_final_sampling,
            ) = make_dflash_verify_uniform_samples(
                sampling_info=sampling_info,
                positions=verify_input.positions,
                draft_token_num=int(self.block_size),
            )
            accept_len, bonus = compute_dflash_sampling_accept_len_and_bonus(
                candidates=candidates,
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                max_top_k=draft_input.max_top_k,
                uniform_top_k_value=draft_input.uniform_top_k_value,
                uniform_samples=uniform_samples,
                uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
            )
        else:
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
                bs, int(self.block_size)
            )
            accept_len, bonus = compute_dflash_accept_len_and_bonus(
                candidates=candidates,
                target_predict=target_predict,
            )
        candidate_teacher_logprobs = compute_dflash_candidate_logprobs(
            candidates=candidates,
            next_token_logits=logits_output.next_token_logits,
            sampling_info=sampling_info,
            use_sampling_distribution=use_sampling_distribution,
        )
        commit_lens = accept_len.to(torch.int32) + 1  # [bs]
        max_acc = int(self.block_size) - 1
        suffix_metadata = {
            "offsets": [[] for _ in range(bs)],
            "token_ids": [[] for _ in range(bs)],
            "teacher_logprobs": [[] for _ in range(bs)],
        }
        if max_acc > 0:
            candidate_suffix_cpu = candidates[:, 1:].cpu()
            teacher_suffix_cpu = candidate_teacher_logprobs.cpu()
            accept_len_cpu = accept_len.cpu().tolist()
            for row_idx, acc_len_item in enumerate(accept_len_cpu):
                acc_len_item = int(acc_len_item)
                if acc_len_item >= max_acc:
                    continue
                suffix_metadata["offsets"][row_idx] = list(
                    range(acc_len_item + 1, int(self.block_size))
                )
                suffix_metadata["token_ids"][row_idx] = [
                    int(token_id)
                    for token_id in candidate_suffix_cpu[
                        row_idx, acc_len_item:max_acc
                    ].tolist()
                ]
                suffix_metadata["teacher_logprobs"][row_idx] = [
                    float(logprob)
                    for logprob in teacher_suffix_cpu[
                        row_idx, acc_len_item:max_acc
                    ].tolist()
                ]

        if need_mamba_verify_commit:
            assert seq_lens_pre_verify is not None
            self._update_target_mamba_state_after_verify(
                batch=model_worker_batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
            )

        out_tokens = torch.empty(
            (bs, int(self.block_size)), dtype=torch.int64, device=device
        )
        if int(self.block_size) > 1:
            out_tokens[:, : int(self.block_size) - 1].copy_(candidates[:, 1:])
        out_tokens[:, int(self.block_size) - 1].fill_(0)
        out_tokens.scatter_(1, accept_len.to(torch.int64)[:, None], bonus[:, None])

        # --- 3) Materialize committed verify-input tokens into draft KV cache.
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DFLASH verify requires target hidden states, but got None."
            )
        hidden = hidden.view(bs, int(self.block_size), -1)

        # Keep KV append dense to avoid boolean-index packing (which can introduce sync).
        offsets = self._block_pos_offsets  # [block_size]
        mask2d = offsets[None, :] < commit_lens.to(torch.int64)[:, None]  # [bs, block]
        mask_flat = mask2d.reshape(-1)

        loc2d = verify_out_cache_loc.view(bs, int(self.block_size))
        loc2d = torch.where(mask2d, loc2d, loc2d.new_zeros(()))
        loc_flat = loc2d.reshape(-1)

        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=loc_flat,
            positions=positions,
            mask_valid=mask_flat,
        )

        # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
        logits_output.hidden_states = None

        new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        next_draft_input = self._make_next_draft_input_decode(
            verified_id=bonus,
            new_seq_lens=new_seq_lens,
        )
        verify_done = torch.get_device_module(device).Event()
        verify_done.record()
        next_draft_input.verify_done = verify_done

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            dflash_rejected_draft_metadata=suffix_metadata,
        )
