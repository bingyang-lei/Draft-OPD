# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import logging
import multiprocessing as mp
import os
from dataclasses import asdict
from typing import Generator, Optional

import ray
import sglang.srt.entrypoints.engine
import torch
from peft import LoraConfig
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    MultiprocessingSerializer,
    assert_pkg_version,
    is_cuda,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from sglang.srt.weight_sync.utils import _preprocess_tensor_for_update_weights
from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter
from verl.workers.rollout.sglang_rollout.utils import (
    SGLANG_LORA_NAME,
    get_named_tensor_buckets,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


async def _update_weights_with_optional_draft_flag(
    *,
    engine,
    params_batch: list[tuple[str, torch.Tensor]],
    device_mesh_key: str,
    device_mesh: DeviceMesh,
    load_format: Optional[str] = None,
    disable_draft_model: Optional[bool] = None,
    draft_model_only: Optional[bool] = None,
):
    """SGLang weight update that can optionally target draft/target worker."""
    infer_tp_size = device_mesh[device_mesh_key].mesh.size()[0]
    infer_tp_rank = device_mesh[device_mesh_key].get_local_rank()
    from sglang.srt.model_executor.model_runner import LocalSerializedTensor
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

    monkey_patch_torch_reductions()

    named_tensors_batch = [
        (
            name,
            MultiprocessingSerializer.serialize(_preprocess_tensor_for_update_weights(tensor.detach())),
        )
        for name, tensor in params_batch
    ]

    if infer_tp_rank == 0:
        gathered_serialized_batches = [None for _ in range(infer_tp_size)]
    else:
        gathered_serialized_batches = None

    torch.distributed.gather_object(
        obj=named_tensors_batch,
        object_gather_list=gathered_serialized_batches,
        dst=device_mesh[device_mesh_key].mesh.tolist()[0],
        group=device_mesh[device_mesh_key].get_group(),
    )

    if infer_tp_rank == 0:
        logical_tensors = zip(*gathered_serialized_batches, strict=True)
        named_tensors = [
            (
                tensor_group[0][0],
                LocalSerializedTensor(values=[rank_part[1] for rank_part in tensor_group]),
            )
            for tensor_group in logical_tensors
        ]

        from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput

        request_kwargs = {
            "serialized_named_tensors": [MultiprocessingSerializer.serialize(named_tensors) for _ in range(infer_tp_size)],
            "load_format": load_format,
        }
        if (
            disable_draft_model is not None
            and "disable_draft_model" in getattr(UpdateWeightsFromTensorReqInput, "__dataclass_fields__", {})
        ):
            request_kwargs["disable_draft_model"] = disable_draft_model
        if (
            draft_model_only is not None
            and "draft_model_only" in getattr(UpdateWeightsFromTensorReqInput, "__dataclass_fields__", {})
        ):
            request_kwargs["draft_model_only"] = draft_model_only

        update_weights_request = UpdateWeightsFromTensorReqInput(**request_kwargs)
        return await engine.update_weights_from_tensor(update_weights_request)


# patch to avoid issue https://github.com/sgl-project/sglang/issues/6723
def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"
    # Enable faulthandler in subprocesses
    os.environ["PYTHONFAULTHANDLER"] = "1"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer_python",
            "0.2.5",
            "Please uninstall the old version and reinstall the latest version by following the instructions at https://docs.flashinfer.ai/installation.html.",
        )
    if is_cuda():
        assert_pkg_version(
            "sgl-kernel",
            "0.1.1",
            "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
        )

    # Set mp start method
    mp.set_start_method("spawn", force=True)


sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config


# because chatCompletion is an async method, it makes the whole ray actor be an async actor
# which can not call loop.run_until_complete. So we need to make the engine to be an async class
class ServerAdapter(BaseRollout):
    """SGLang server adapter used in native http server mode, serve as http client to request SGLang server
    to resume/release/update weights and kv_cache.

    - hybrid mode: reside in each hybrid worker to sync weights between training engine and SGLang server.
    - standalone/colocated mode: just a dummy placeholder to occupy the GPU to prevent ray scheduling new GPU actor.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
        replica_rank: int = -1,
    ):
        super().__init__(config, model_config, device_mesh)
        if self.config.get("quantization", None) == "fp8":
            import sglang
            from packaging import version

            assert version.parse(sglang.__version__) >= version.parse("0.5.5"), (
                "sglang>=0.5.5 is required for FP8 quantization"
            )
            FP8_BLOCK_QUANT_KWARGS = {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
            fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)
            self.model_config.hf_config.quantization_config = fp8_block_quant_kwargs
        self._engine: AsyncHttpServerAdapter = None

        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        rollout_world_size = self.config.tensor_model_parallel_size * self.config.data_parallel_size
        if replica_rank == -1:
            self.replica_rank = rank // rollout_world_size
        else:
            self.replica_rank = replica_rank
        self.rollout_rank = rank % rollout_world_size
        self.node_rank = self.rollout_rank // local_world_size
        self.local_rank = self.rollout_rank % local_world_size

        # sleep_level controls what gets released during sleep/release:
        #   2 (default) = release weights + kv_cache (full sleep, merge path)
        #   1 = release kv_cache only (keep base weights, adapter path)
        # Set by engine_workers.update_weights() when lora.merge=False.
        self.sleep_level = 2

    @staticmethod
    def _map_opd_draft_weight_name(weight_name: str) -> Optional[str]:
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

    @staticmethod
    def _map_dflash_draft_weight_name(weight_name: str) -> Optional[str]:
        return ServerAdapter._map_opd_draft_weight_name(weight_name)

    def _iter_opd_draft_weights(
        self, weights: Generator[tuple[str, torch.Tensor], None, None]
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        for name, tensor in weights:
            mapped_name = self._map_opd_draft_weight_name(name)
            if mapped_name is None:
                continue
            yield mapped_name, tensor

    def _iter_dflash_draft_weights(
        self, weights: Generator[tuple[str, torch.Tensor], None, None]
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        yield from self._iter_opd_draft_weights(weights)

    async def _init_server_adapter(self):
        if self._engine is not None:
            return

        # device_mesh is needed to gather cuda ipc handle to update weights
        if self.device_mesh is None:
            assert torch.distributed.is_initialized(), "torch distributed must be initialized"
            infer_tp = self.config.tensor_model_parallel_size * self.config.data_parallel_size
            infer_pp = self.config.pipeline_model_parallel_size
            infer_world_size = infer_tp * infer_pp
            dp = torch.distributed.get_world_size() // infer_world_size
            self.device_mesh = init_device_mesh(
                "cpu", mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
            )

        # Only init http server adapter in tp rank 0
        if self.device_mesh["infer_tp"].get_local_rank() != 0:
            return

        # Lazy init http server adapter because http server is launched after hybrid engine.
        self.server_actor = ray.get_actor(f"sglang_server_{self.replica_rank}_{self.node_rank}")
        server_address, server_port = await self.server_actor.get_server_address.remote()
        logger.debug(
            f"replica_rank={self.replica_rank} node_rank={self.node_rank}, "
            f"server address: {server_address}, port: {server_port}"
        )
        host = f"[{server_address}]" if is_valid_ipv6_address(server_address) else server_address
        self._engine = AsyncHttpServerAdapter(
            model_path=self.model_config.local_path,
            host=host,
            port=server_port,
            launch_server=False,
            trust_remote_code=self.model_config.trust_remote_code,
        )

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tag: weights or kv_cache.
        """
        await self._init_server_adapter()
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
            await self._engine.resume_memory_occupation(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory.

        When sleep_level=1 (LoRA adapter mode), only releases kv_cache
        to keep base weights alive across training iterations.
        When sleep_level=2 (default/merge mode), releases everything.
        """
        await self._init_server_adapter()
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
            if self.sleep_level == 1:
                tags = ["kv_cache"]
            else:
                tags = ["kv_cache", "weights"]
            await self._engine.release_memory_occupation(tags=tags)

    async def update_weights(
        self, weights: Generator[tuple[str, torch.Tensor], None, None], global_steps: int = None, **kwargs
    ):
        """
        Update model weights using tensor buckets, similar to THUDM/slime's implementation.

        Notes:
          - For the best performance of `rebuild_cuda_tensor`, it is recommended to:
              1. Enable `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`.
              2. Manually set `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
            when using Tensor Parallelism (TP >= 8).
          - See reference implementations in SLIME:
            - Main logic: https://github.com/THUDM/slime/blob/fb7605cc5fb09af0f9369d37f7192f12bddee577/slime/ray/ppo_actor.py#L452
            - runtime envs: https://github.com/THUDM/slime/blob/fb7605cc5fb09af0f9369d37f7192f12bddee577/slime/ray/ppo_actor.py#L39
        """
        await self._init_server_adapter()

        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        dflash_draft_only = bool(kwargs.get("dflash_draft_only", False))
        if dflash_draft_only and peft_config is not None:
            raise ValueError("dflash_draft_only sync is not compatible with LoRA adapter sync.")

        if peft_config and base_sync_done:
            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                # unload lora
                models_result = await self._engine.available_models()
                exists = any(item["id"] == SGLANG_LORA_NAME for item in models_result["data"])
                if exists:
                    await self._engine.unload_lora_adapter(SGLANG_LORA_NAME)

                # load lora by tensor
                serialize_peft_config, serialize_named_tensors = self.wrap_lora_params(peft_config, weights)
                from sglang.srt.managers.io_struct import LoadLoRAAdapterFromTensorsReqInput

                req = LoadLoRAAdapterFromTensorsReqInput(
                    lora_name=SGLANG_LORA_NAME,
                    config_dict=serialize_peft_config,
                    serialized_tensors=serialize_named_tensors,
                )
                # send http request
                await self._engine.load_lora_adapter_from_tensor(req)
        else:
            update_weights_bucket_bytes = int(self.config.checkpoint_engine.update_weights_bucket_megabytes) << 20
            if dflash_draft_only:
                weights = self._iter_opd_draft_weights(weights)

            if self.config.get("quantization", None) == "fp8":
                from verl.utils.sglang.sglang_fp8_utils import SGLangFP8QuantizerHelper

                logger.info("Convert bf16 weights to fp8 format before loading")
                fp8_quantizer_helper = SGLangFP8QuantizerHelper(self.model_config.hf_config.quantization_config)
                weights = fp8_quantizer_helper.quant_weights_by_name(
                    weights,
                    dtype=self.model_config.hf_config.dtype,
                )
            else:
                weights = weights

            updated_tensor_count = 0
            async for params_batch in get_named_tensor_buckets(weights, update_weights_bucket_bytes):
                updated_tensor_count += len(params_batch)
                if dflash_draft_only:
                    await _update_weights_with_optional_draft_flag(
                        engine=self._engine,
                        params_batch=params_batch,
                        device_mesh_key="infer_tp",
                        device_mesh=self.device_mesh,
                        disable_draft_model=False,
                        draft_model_only=True,
                    )
                else:
                    await sgl_update_weights(
                        engine=self._engine,
                        params_batch=params_batch,
                        device_mesh_key="infer_tp",
                        device_mesh=self.device_mesh,
                    )

            if dflash_draft_only and updated_tensor_count == 0:
                raise RuntimeError(
                    "dflash_draft_only sync found no draft_model tensors. "
                    "Check actor weight names and composed student configuration."
                )

        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            await self._engine.flush_cache()
            if global_steps is not None:
                await self.server_actor.set_global_steps.remote(global_steps)

        if dflash_draft_only:
            sync_stats = {
                "opd/weight_sync/draft_tensor_count": float(updated_tensor_count),
            }
            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                logger.warning("DFLASH draft weight sync tensor_count=%s", updated_tensor_count)
            return sync_stats

    def wrap_lora_params(self, peft_config: LoraConfig, weights: Generator[tuple[str, torch.Tensor]]):
        # peft config
        peft_config_json = asdict(peft_config)
        peft_config_json["task_type"] = peft_config_json["task_type"].value
        peft_config_json["peft_type"] = peft_config_json["peft_type"].value
        peft_config_json["target_modules"] = list(peft_config_json["target_modules"])

        # lora weights
        processed_weights: dict[str, torch.Tensor] = {
            name: _preprocess_tensor_for_update_weights(tensor.detach()) for name, tensor in weights
        }

        infer_tp_size = self.device_mesh["infer_tp"].mesh.size()[0]
        serialized_named_tensors = []
        for i in range(infer_tp_size):
            serialized_tensors = MultiprocessingSerializer.serialize(processed_weights, output_str=True)
            serialized_named_tensors.append(serialized_tensors)

        return peft_config_json, serialized_named_tensors
