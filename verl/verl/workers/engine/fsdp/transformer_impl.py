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
"""
The concrete Engine implementation using PyTorch FullyShardedDataParallel (FSDP)
"""

import gc
import logging
import os
import warnings
from contextlib import nullcontext
from typing import Callable, ContextManager, Optional

import torch
import torch.distributed
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    apply_fsdp2,
    collect_lora_params,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    merged_lora_context,
    normalize_peft_param_name,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    replace_lora_wrapper,
)
from verl.utils.model import convert_weight_keys, extract_multi_modal_inputs
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_group,
    set_ulysses_sequence_parallel_group,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import enable_full_determinism, postprocess_batch_func, prepare_micro_batches
from .utils import create_device_mesh, get_sharding_strategy

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class FSDPEngine(BaseEngine):
    """
    Concrete Engine implementation using PyTorch FullyShardedDataParallel (FSDP).

    Supports model sharding, activation/optimizer offloading, LoRA, and sequence parallelism.
    """

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        """
        Initialize the FSDPEngine.

        Sets up distributed device meshes, LoRA, and offload policies based on config.

        Args:
            config: Configuration object with FSDP and model settings.
        """
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        self.mode = None

        self.rank = torch.distributed.get_rank()

        # Apply NPU patches for FSDP backend
        from .utils import apply_npu_fsdp_patches

        apply_npu_fsdp_patches()

        # build device mesh for Ulysses Sequence Parallel

        self.use_remove_padding = self.model_config.use_remove_padding

        self._init_device_mesh()

        if self.engine_config.full_determinism:
            enable_full_determinism(seed=self.engine_config.seed)

        # set FSDP offload params
        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload
        self._is_lora = self.model_config.lora_rank > 0

        # QAT (Quantization-Aware Training)
        self._qat_config = getattr(self.engine_config, "qat", None)
        self._qat_enabled = self._qat_config is not None and getattr(self._qat_config, "enable", False)
        if self._qat_enabled:
            logger.info(f"QAT enabled: mode={self._qat_config.mode}, group_size={self._qat_config.group_size}")

        if self.engine_config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.engine_config.use_torch_compile  #  use torch compile by default
            else entropy_from_logits
        )

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def is_mp_src_rank_with_outputs(self):
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
        else:
            is_collect = True
        return is_collect

    def initialize(self):
        """
        Build the model, optimizer, and learning rate scheduler under FSDP.

        Applies device, dtype, and precision configurations, including mixed precision.
        Sets up checkpoint manager and FLOPs counter.
        """
        # This is used to import external_lib into the huggingface systems
        self._build_model_optimizer()

        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.module,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.model_config.get_processor(),
            checkpoint_config=self.checkpoint_config,
            trust_remote_code=self.model_config.trust_remote_code,
        )

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)

    def _init_device_mesh(self):
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.engine_config.fsdp_size

        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)
        self.ulysses_device_mesh = None
        self.ulysses_parallel_group = None
        self.ulysses_sequence_parallel_size = self.engine_config.ulysses_sequence_parallel_size
        dp_size = self.get_data_parallel_size()
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp_size, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )
            self.ulysses_parallel_group = self.ulysses_device_mesh["sp"].get_group()

        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

    def _is_composed_dflash_student_enabled(self) -> bool:
        return bool(getattr(self.model_config.hf_config, "verl_composed_dflash_student", False))

    def _is_composed_eagle3_student_enabled(self) -> bool:
        return bool(getattr(self.model_config.hf_config, "verl_composed_eagle3_student", False))

    def _is_composed_opd_draft_student_enabled(self) -> bool:
        dflash_enabled = self._is_composed_dflash_student_enabled()
        eagle3_enabled = self._is_composed_eagle3_student_enabled()
        if dflash_enabled and eagle3_enabled:
            raise ValueError("Only one composed OPD draft student can be enabled: DFLASH or EAGLE3.")
        return dflash_enabled or eagle3_enabled

    def _resolve_composed_dflash_paths(self) -> tuple[str, str]:
        main_model_path = getattr(self.model_config.hf_config, "verl_dflash_main_model_path", None)
        if main_model_path is None:
            main_model_path = self.model_config.local_path

        draft_model_path = getattr(self.model_config.hf_config, "verl_dflash_draft_model_path", None)
        if not draft_model_path:
            raise ValueError(
                "Composed DFLASH student requires actor_rollout_ref.model.override_config.verl_dflash_draft_model_path."
        )
        return main_model_path, draft_model_path

    def _resolve_composed_eagle3_paths(self) -> tuple[str, str]:
        main_model_path = getattr(self.model_config.hf_config, "verl_eagle3_main_model_path", None)
        if main_model_path is None:
            main_model_path = self.model_config.local_path

        draft_model_path = getattr(self.model_config.hf_config, "verl_eagle3_draft_model_path", None)
        if not draft_model_path:
            raise ValueError(
                "Composed EAGLE3 student requires actor_rollout_ref.model.override_config.verl_eagle3_draft_model_path."
            )
        return main_model_path, draft_model_path

    def _build_module(self):
        from verl.utils.model import get_hf_auto_model_class
        from verl.utils.torch_dtypes import PrecisionType

        torch_dtype = self.engine_config.model_dtype

        if torch_dtype is None:
            # if it is training, we force torch_dtype to fp32
            torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16

        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not self.model_config.hf_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.model_config.model_type == "language_model":
                if self._is_composed_dflash_student_enabled():
                    from verl.models.transformers.dflash_student import build_composed_dflash_student

                    main_model_path, draft_model_path = self._resolve_composed_dflash_paths()
                    module = build_composed_dflash_student(
                        main_model_path=main_model_path,
                        draft_model_path=draft_model_path,
                        torch_dtype=torch_dtype,
                        trust_remote_code=self.model_config.trust_remote_code,
                        config=self.model_config.hf_config,
                    )
                elif self._is_composed_eagle3_student_enabled():
                    from verl.models.transformers.eagle3_student import build_composed_eagle3_student

                    main_model_path, draft_model_path = self._resolve_composed_eagle3_paths()
                    module = build_composed_eagle3_student(
                        main_model_path=main_model_path,
                        draft_model_path=draft_model_path,
                        torch_dtype=torch_dtype,
                        trust_remote_code=self.model_config.trust_remote_code,
                        config=self.model_config.hf_config,
                    )
                else:
                    auto_class = get_hf_auto_model_class(hf_config=self.model_config.hf_config)
                    module = auto_class.from_pretrained(
                        pretrained_model_name_or_path=self.model_config.local_path,
                        torch_dtype=torch_dtype,
                        config=self.model_config.hf_config,
                        trust_remote_code=self.model_config.trust_remote_code,
                    )
            else:
                from verl.utils.model import load_valuehead_model

                assert self.model_config.model_type == "value_model", (
                    f"Unsupported model type: {self.model_config.model_type}"
                )
                self.model_config.hf_config.num_labels = 1
                self.model_config.hf_config.classifier_dropout = 0.0
                self.model_config.hf_config.hidden_dropout = "0"
                self.model_config.hf_config.summary_dropout_prob = 0.0
                module = load_valuehead_model(
                    local_path=self.model_config.local_path,
                    torch_dtype=torch_dtype,
                    model_config=self.model_config.hf_config,
                    trust_remote_code=self.model_config.trust_remote_code,
                )

            use_liger = self.model_config.use_liger
            # Apply Liger kernel; disable fused_linear_cross_entropy (conflicts with verl's forward patching)
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(
                    model=module,
                    fused_linear_cross_entropy=False,
                    swiglu=True,
                )

            fused_kernel_options = self.model_config.fused_kernel_options
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            use_fused_kernels = self.model_config.use_fused_kernels
            apply_monkey_patch(
                model=module,
                use_remove_padding=self.use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )

            # some parameters may not in torch_dtype
            module.to(torch_dtype)

            if self.model_config.enable_gradient_checkpointing:
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return module

    def _build_lora_module(self, module):
        module.enable_input_require_grads()

        lora_adapter_path = getattr(self.model_config, "lora_adapter_path", None)
        if lora_adapter_path is not None:
            from peft import PeftModel

            from verl.utils.fs import copy_to_local

            print(f"Loading pre-trained LoRA adapter to from: {lora_adapter_path}")
            # Copy adapter to local if needed
            local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.model_config.use_shm)

            module = PeftModel.from_pretrained(module, local_adapter_path, is_trainable=True)
            peft_config = module.peft_config["default"]
            # Ensure task_type is TaskType enum, not string
            if isinstance(peft_config.task_type, str):
                peft_config.task_type = TaskType.CAUSAL_LM
        else:
            # Convert config to regular Python types before creating PEFT model
            lora_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": self.model_config.lora_rank,
                "lora_alpha": self.model_config.lora_alpha,
                "target_modules": convert_to_regular_types(self.model_config.target_modules),
                "target_parameters": convert_to_regular_types(self.model_config.target_parameters),
                "exclude_modules": convert_to_regular_types(self.model_config.exclude_modules),
                "bias": "none",
            }
            module = get_peft_model(module, LoraConfig(**lora_config))

        return module

    def _build_fsdp_module(self, module):
        # TODO(ziheng): need to improve
        from torch.distributed.fsdp import CPUOffload, MixedPrecision

        from verl.utils.torch_dtypes import PrecisionType

        mixed_precision_config = self.engine_config.mixed_precision
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=module,
            config=self.engine_config.wrap_policy,
            is_lora=self.model_config.lora_rank > 0,
        )

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # Note: We force turn off CPUOffload because it causes incorrect results when using grad accumulation
        if self.engine_config.strategy == "fsdp":
            # cpu_offload:
            # - actor: None
            # - critic: None
            # - ref: CPUOffload(offload_params=True)

            # We force reference policy to use CPUOffload to save memory.
            # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
            cpu_offload = None
            if self.engine_config.forward_only:
                cpu_offload = CPUOffload(offload_params=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False

            use_orig_params = self.engine_config.use_orig_params
            if self._is_composed_opd_draft_student_enabled() and not use_orig_params:
                if self.rank == 0:
                    logger.info(
                        "Enable FSDP use_orig_params=True for composed OPD draft student "
                        "to support mixed frozen/trainable parameters."
                    )
                use_orig_params = True

            module = FSDP(
                module,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=self.engine_config.forward_prefetch,
                use_orig_params=use_orig_params,
                cpu_offload=cpu_offload,
            )
        elif self.engine_config.strategy == "fsdp2":
            # - actor: offload_policy
            # - critic: offload_policy
            # - ref: CPUOffloadPolicy(pin_memory=True)
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            offload_policy = None
            if self.engine_config.offload_policy or self.engine_config.forward_only:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": self.engine_config.reshard_after_forward,
            }
            full_state = module.state_dict()
            apply_fsdp2(module, fsdp_kwargs, self.engine_config)
            fsdp2_load_full_state_dict(module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {self.engine_config.strategy}")

        if self.model_config.enable_activation_offload:
            enable_gradient_checkpointing = self.model_config.enable_gradient_checkpointing
            enable_activation_offloading(module, self.engine_config.strategy, enable_gradient_checkpointing)

        if torch.distributed.get_world_size() == 1 and fsdp_version(module) == 1:
            FSDP.set_state_dict_type(
                module,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(module) == 1:
            FSDP.set_state_dict_type(
                module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        return module

    def _build_optimizer(self, module):
        from verl.workers.config.optimizer import build_optimizer

        params = list(module.parameters())
        trainable_params = [param for param in params if param.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found when constructing optimizer.")

        if self.rank == 0 and len(trainable_params) != len(params):
            logger.info(
                "Optimizer will update %d/%d parameters with requires_grad=True.",
                len(trainable_params),
                len(params),
            )

        optimizer = build_optimizer(trainable_params, self.optimizer_config)

        return optimizer

    def _build_lr_scheduler(self, optimizer):
        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        optim_config = self.optimizer_config

        total_steps = optim_config.total_training_steps
        num_warmup_steps = optim_config.lr_warmup_steps
        lr_scheduler_type = optim_config.lr_scheduler_type
        min_lr_ratio = optim_config.min_lr_ratio
        num_cycles = optim_config.num_cycles
        zero_indexed_step = optim_config.zero_indexed_step
        if num_warmup_steps <= 0:
            num_warmup_steps_ratio = optim_config.lr_warmup_steps_ratio
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        if lr_scheduler_type == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)
        elif lr_scheduler_type == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
                num_cycles=num_cycles,
                zero_indexed_step=zero_indexed_step,
            )
        else:
            raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")
        return lr_scheduler

    def _apply_qat(self, module):
        """Apply QAT transformations to the model before FSDP wrapping."""
        from verl.utils.qat.core import apply_qat, enable_qat_fuse

        module = apply_qat(
            module,
            {
                "enable": self._qat_config.enable,
                "mode": self._qat_config.mode,
                "group_size": self._qat_config.group_size,
                "ignore_patterns": list(self._qat_config.ignore_patterns),
                "activation_observer": self._qat_config.activation_observer,
            },
        )
        enable_qat_fuse(module)

        if self._qat_config.mode == "w4a4":
            self._restore_w4a4_input_scales(module, self.model_config.local_path)

        return module

    def _restore_w4a4_input_scales(self, model, model_path):
        """Restore input_global_scale and input_amax from checkpoint for W4A4 mode."""
        import glob

        from safetensors import safe_open

        safetensor_files = glob.glob(f"{model_path}/model*.safetensors")
        loaded_count = 0

        for sf_path in safetensor_files:
            with safe_open(sf_path, framework="pt") as f:
                for key in f.keys():
                    if "input_global_scale" in key:
                        module_path = key.replace(".input_global_scale", "")
                        amax_key = f"{module_path}.input_amax"

                        module = model
                        for part in module_path.split("."):
                            module = module[int(part)] if part.isdigit() else getattr(module, part)

                        scale_val = f.get_tensor(key)
                        val = scale_val.item() if scale_val.numel() == 1 else scale_val.max().item()
                        module.input_global_scale.fill_(val)

                        amax_val = f.get_tensor(amax_key)
                        amax = amax_val.item() if amax_val.numel() == 1 else amax_val.max().item()
                        module.input_amax.fill_(amax)
                        loaded_count += 1

        logger.info(f"[QAT W4A4] Restored {loaded_count} input_global_scale/input_amax from {model_path}")

    def _build_model_optimizer(self):
        from verl.utils.model import print_model_size

        # Load base model with specified configuration and dtype
        module = self._build_module()
        # Apply LoRA adapters if low-rank adaptation is enabled
        if self._is_lora:
            module = self._build_lora_module(module)

        # Apply QAT before FSDP wrapping (training only)
        if self._qat_enabled and not self.engine_config.forward_only:
            module = self._apply_qat(module)

        self._trainable_param_names = {name for name, param in module.named_parameters() if param.requires_grad}
        if not self.engine_config.forward_only and len(self._trainable_param_names) == 0:
            raise ValueError("No trainable parameters found after model initialization.")

        # Synchronize all distributed processes before proceeding
        torch.distributed.barrier()
        if self.rank == 0:
            print_model_size(module)
        log_gpu_memory_usage("After init model from HF AutoModel", logger=logger)

        # Wrap model with FSDP for distributed training (sharding, mixed precision, etc.)
        log_gpu_memory_usage("Before FSDP", logger=None)
        module = self._build_fsdp_module(module)
        log_gpu_memory_usage("After FSDP", logger=None)

        if not self.engine_config.forward_only:
            # Initialize optimizer with model parameters and config settings
            optimizer = self._build_optimizer(module)
            # Create learning rate scheduler with warmup and decay settings
            lr_scheduler = self._build_lr_scheduler(optimizer)
        else:
            optimizer = None
            lr_scheduler = None

        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def train_mode(self, **kwargs):
        """
        Return a context manager that switches to training mode with FSDP-specific handling.

        Includes parameter and optimizer offload entry/exit.
        """
        return EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        """
        Return a context manager that switches to evaluation mode with FSDP-specific handling.

        Includes activation offload entry/exit.
        """
        return EngineEvalModeCtx(self, **kwargs)

    def get_data_parallel_rank(self):
        if self.ulysses_device_mesh is not None:
            return self.ulysses_device_mesh["dp"].get_local_rank()
        else:
            return torch.distributed.get_rank()

    def get_data_parallel_size(self):
        return torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size

    def get_data_parallel_group(self):
        if self.ulysses_device_mesh is not None:
            return self.ulysses_device_mesh.get_group(mesh_dim="dp")
        else:
            return torch.distributed.group.WORLD

    def get_model_parallel_group(self):
        raise NotImplementedError

    def get_context_parallel_group(self):
        raise NotImplementedError

    def _compute_extra_global_batch_info(self, data: TensorDict) -> dict[str, float]:
        return {}

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> list[TensorDict]:
        # note that the global_batch_size should include data on all the dp
        tu.assign_non_tensor(data, sp_size=self.ulysses_sequence_parallel_size)

        # compute num_tokens in global batch for loss normalization
        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())
        extra_global_batch_info = self._compute_extra_global_batch_info(data)
        if extra_global_batch_info:
            tu.assign_non_tensor(data, **extra_global_batch_info)

        micro_batches, indices = prepare_micro_batches(
            data=data, dp_group=self.get_data_parallel_group(), same_micro_num_in_dp=True
        )

        output_lst = []

        ctx = torch.no_grad() if forward_only else nullcontext()

        for micro_batch in micro_batches:
            with ctx:
                loss, meta_info = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)

                if not forward_only:
                    loss.backward()

            output_lst.append(meta_info)

        # postprocess and return
        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def optimizer_zero_grad(self):
        """
        Zero gradients and enforce FSDP grad-clipping logic.
        """
        self.optimizer.zero_grad()

    def optimizer_step(self):
        """
        Clip gradients, skip update if non-finite, and step optimizer.

        Returns:
            grad_norm (float): Norm of gradients before clipping.
        """
        assert self.optimizer_config.clip_grad is not None

        if isinstance(self.module, FSDP):
            grad_norm = self.module.clip_grad_norm_(self.optimizer_config.clip_grad)
        elif isinstance(self.module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.module.parameters(), max_norm=self.optimizer_config.clip_grad)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.module.parameters(), max_norm=self.optimizer_config.clip_grad
            )

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        if self._qat_enabled:
            from verl.utils.qat.core import invalidate_all_scales

            invalidate_all_scales(self.module)

        return grad_norm.item()

    def lr_scheduler_step(self):
        """
        Advance FSDP scheduler and return updated learning rate.
        """
        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]  # only return the first group
        return lr

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """
        Move FSDP model and/or optimizer to CPU or GPU with offload support.
        Note that this function executes irrespective of offload config. It serves as manual control
        """
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

        if self.engine_config.forward_only:
            # force cpu_offload
            return

        device_name = get_device_name()

        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                load_fsdp_model_to_gpu(self.module)
            if optimizer and self.optimizer is not None:
                load_fsdp_optimizer(self.optimizer, device)
            gc.collect()
        elif device == "cpu":
            if model:
                offload_fsdp_model_to_cpu(self.module)
            if optimizer and self.optimizer is not None:
                offload_fsdp_optimizer(self.optimizer)
        else:
            raise ValueError(f"Invalid device type: {device}")

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Save FSDP checkpoint, handling parameter offload as needed.
        """
        origin_module_device = next(self.module.parameters()).device.type
        if self._is_offload_param or origin_module_device == "cpu":
            load_fsdp_model_to_gpu(self.module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: int = True, **kwargs
    ) -> None:
        """
        Load FSDP checkpoint, restoring parameters and optimizer state.
        """
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.module)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.optimizer)

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):
        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
        trainable_only = kwargs.pop("trainable_only", False)

        load_fsdp_model_to_gpu(self.module)

        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_config = None
        merge_lora = self.model_config.lora.get("merge", False)

        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if hasattr(peft_model, "peft_config"):  # LoRA
            if not merge_lora:
                peft_config = peft_model.peft_config.get("default", None)
                params = collect_lora_params(
                    module=self.module,
                    layered_summon=layered_summon,
                    base_sync_done=base_sync_done,
                )
                if not base_sync_done:
                    params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
            else:  # merge lora
                with merged_lora_context(self.module, backup_adapters=True):
                    params = self.module.state_dict()
                    params = normalize_peft_param_name(params)
        else:
            params = self.module.state_dict()

        if trainable_only:
            trainable_param_names = getattr(self, "_trainable_param_names", None)
            if trainable_param_names is None:
                raise RuntimeError("trainable_only=True requires initialized _trainable_param_names.")

            def _is_trainable_param_name(name: str) -> bool:
                if name in trainable_param_names:
                    return True
                if name.startswith("_fsdp_wrapped_module."):
                    return name[len("_fsdp_wrapped_module.") :] in trainable_param_names
                if name.startswith("module."):
                    return name[len("module.") :] in trainable_param_names
                return False

            params = {name: tensor for name, tensor in params.items() if _is_trainable_param_name(name)}
            if len(params) == 0:
                raise RuntimeError("No trainable parameters found for tensor sync.")

        params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        if peft_config is not None and base_sync_done:
            per_tensor_param = params.items()
        else:
            device = get_device_id()  # used when fsdp2 set cpu_offload_policy
            # TODO: cast fp32 to bf16 to reduce weight sync overhead, need more fine-grained control, e.g MoE gate
            per_tensor_param = (
                (
                    name,
                    param.to(device, non_blocking=True).full_tensor().to(torch.bfloat16, non_blocking=True)
                    if isinstance(param, DTensor)
                    else param,
                )
                for name, param in params.items()
            )

        if self._qat_enabled:
            from verl.utils.qat.quantizer import QATQuantizer
            from verl.utils.torch_dtypes import PrecisionType

            mixed_precision_config = self.engine_config.mixed_precision
            if mixed_precision_config is not None:
                param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            else:
                param_dtype = torch.bfloat16

            quantizer = QATQuantizer(
                mode=self._qat_config.mode,
                group_size=self._qat_config.group_size,
                ignore_patterns=list(self._qat_config.ignore_patterns),
                device=torch.device(get_device_id()),
                param_dtype=param_dtype,
            )
            per_tensor_param = quantizer.quantize_with_fusion(
                per_tensor_param,
                target_device=torch.device("cpu"),
            )

        peft_config_dict = peft_config.to_dict() if peft_config is not None else None
        return per_tensor_param, peft_config_dict

    def disable_adapter(self) -> ContextManager:
        return self.module.disable_adapter()


class EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: FSDPEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, FSDPEngine)
        super().__enter__()
        self.prev_sp_group = get_ulysses_sequence_parallel_group()
        set_ulysses_sequence_parallel_group(self.engine.ulysses_parallel_group)
        self.engine.module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, FSDPEngine)
        set_ulysses_sequence_parallel_group(self.prev_sp_group)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.engine.engine_config.fsdp_size > 1:
            if fsdp_version(self.engine.module) == 1:
                self.engine.module._handle.reshard(True)
            elif fsdp_version(self.engine.module) == 2:
                self.engine.module.reshard()

        super().__exit__(exc_type, exc_value, traceback)


class EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: FSDPEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, FSDPEngine)
        super().__enter__()
        self.prev_sp_group = get_ulysses_sequence_parallel_group()
        set_ulysses_sequence_parallel_group(self.engine.ulysses_parallel_group)
        self.engine.module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, FSDPEngine)
        set_ulysses_sequence_parallel_group(self.prev_sp_group)
        self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


@EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class FSDPEngineWithLMHead(FSDPEngine):
    def _is_composed_opd_draft_module(self) -> bool:
        module = getattr(self, "module", None)
        while module is not None and hasattr(module, "module"):
            module = module.module
        return module is not None and module.__class__.__name__ in {
            "ComposedDFlashStudentForCausalLM",
            "ComposedEagle3StudentForCausalLM",
        }

    def _is_composed_dflash_module(self) -> bool:
        return self._is_composed_opd_draft_module()

    def _is_exact_composed_dflash_module(self) -> bool:
        module = getattr(self, "module", None)
        while module is not None and hasattr(module, "module"):
            module = module.module
        return module is not None and module.__class__.__name__ == "ComposedDFlashStudentForCausalLM"

    def _get_dflash_rejected_draft_max_tokens_per_sample(self) -> Optional[int]:
        value = getattr(self.model_config.hf_config, "verl_dflash_rejected_draft_max_tokens_per_sample", None)
        if value is None:
            value = os.getenv("VERL_DFLASH_REJECTED_DRAFT_MAX_TOKENS_PER_SAMPLE")
        if value is None or str(value).lower() in {"", "none", "null"}:
            return None
        value = int(value)
        if value <= 0:
            raise ValueError(f"verl_dflash_rejected_draft_max_tokens_per_sample must be positive, got {value}.")
        return value

    @classmethod
    def _count_dflash_rejected_draft_tokens(
        cls,
        raw_token_ids,
        *,
        batch_size: int,
        max_tokens_per_sample: Optional[int],
        raw_anchor_indices=None,
        raw_offsets=None,
        first_token_only: bool = False,
    ) -> int:
        token_ids = cls._normalize_dflash_rejected_draft_values(raw_token_ids, batch_size=batch_size, cast_fn=int)
        if len(token_ids) != batch_size:
            raise ValueError(
                "DFLASH rejected draft metadata batch size mismatch: "
                f"token_ids has {len(token_ids)} entries for batch_size={batch_size}."
            )
        if first_token_only:
            anchors = cls._normalize_dflash_rejected_draft_values(
                raw_anchor_indices, batch_size=batch_size, cast_fn=int
            )
            offsets = cls._normalize_dflash_rejected_draft_values(raw_offsets, batch_size=batch_size, cast_fn=int)
            token_ids, _, _, _ = cls._filter_dflash_rejected_draft_first_tokens(
                anchors=anchors,
                offsets=offsets,
                token_ids=token_ids,
                teacher_logprobs=None,
            )
        count = 0
        for sample_token_ids in token_ids:
            valid_count = sum(1 for token_id in sample_token_ids if token_id >= 0)
            if max_tokens_per_sample is not None:
                valid_count = min(valid_count, max_tokens_per_sample)
            count += valid_count
        return count

    @classmethod
    def _sum_dflash_rejected_draft_position_weights(
        cls,
        raw_token_ids,
        raw_offsets,
        *,
        batch_size: int,
        max_tokens_per_sample: Optional[int],
        decay: float,
        raw_anchor_indices=None,
        first_token_only: bool = False,
    ) -> float:
        if decay <= 0.0 or decay > 1.0:
            raise ValueError(f"rejected_draft_position_decay must be in (0, 1], got {decay}.")
        token_ids = cls._normalize_dflash_rejected_draft_values(raw_token_ids, batch_size=batch_size, cast_fn=int)
        offsets = cls._normalize_dflash_rejected_draft_values(raw_offsets, batch_size=batch_size, cast_fn=int)
        if len(token_ids) != batch_size or len(offsets) != batch_size:
            raise ValueError(
                "DFLASH rejected draft metadata batch size mismatch: "
                f"token_ids has {len(token_ids)} entries and offsets has {len(offsets)} entries "
                f"for batch_size={batch_size}."
            )
        if first_token_only:
            anchors = cls._normalize_dflash_rejected_draft_values(
                raw_anchor_indices, batch_size=batch_size, cast_fn=int
            )
            token_ids, offsets, _, _ = cls._filter_dflash_rejected_draft_first_tokens(
                anchors=anchors,
                offsets=offsets,
                token_ids=token_ids,
                teacher_logprobs=None,
            )

        total_weight = 0.0
        for sample_idx, (sample_token_ids, sample_offsets) in enumerate(zip(token_ids, offsets, strict=True)):
            if len(sample_token_ids) != len(sample_offsets):
                raise ValueError(
                    "DFLASH rejected draft metadata length mismatch for sample "
                    f"{sample_idx}: token_ids={len(sample_token_ids)}, offsets={len(sample_offsets)}."
                )
            selected_count = 0
            for token_id, offset in zip(sample_token_ids, sample_offsets, strict=True):
                if token_id < 0 or offset <= 0:
                    continue
                if max_tokens_per_sample is not None and selected_count >= max_tokens_per_sample:
                    break
                total_weight += decay ** max(offset - 1, 0)
                selected_count += 1
        return total_weight

    @classmethod
    def _count_dflash_tv_blocks(cls, data: TensorDict) -> int:
        batch_size = int(data.batch_size[0])
        response_mask = data.get("response_mask")
        responses = data.get("responses")
        if response_mask is not None:
            if response_mask.is_nested:
                response_lengths = response_mask.offsets().diff().tolist()
            else:
                response_lengths = response_mask.long().sum(dim=1).tolist()
        elif responses is not None and responses.is_nested:
            response_lengths = responses.offsets().diff().tolist()
        elif responses is not None:
            response_lengths = [int(responses.shape[1])] * batch_size
        else:
            raise ValueError("Exact DFLASH TV global block counting requires responses or response_mask.")

        raw_rejects = tu.get_non_tensor_data(data=data, key="dflash_reject_token_indices", default=None)
        reject_indices = cls._normalize_dflash_reject_indices(raw_rejects, batch_size=batch_size)
        if not reject_indices:
            reject_indices = [[] for _ in range(batch_size)]
        raw_anchors = tu.get_non_tensor_data(data=data, key="dflash_rejected_draft_anchor_indices", default=None)
        raw_offsets = tu.get_non_tensor_data(data=data, key="dflash_rejected_draft_offsets", default=None)
        raw_token_ids = tu.get_non_tensor_data(data=data, key="dflash_rejected_draft_token_ids", default=None)
        rejected_anchors = cls._normalize_dflash_rejected_draft_values(
            raw_anchors, batch_size=batch_size, cast_fn=int
        )
        rejected_offsets = cls._normalize_dflash_rejected_draft_values(
            raw_offsets, batch_size=batch_size, cast_fn=int
        )
        rejected_token_ids = cls._normalize_dflash_rejected_draft_values(
            raw_token_ids, batch_size=batch_size, cast_fn=int
        )

        total = 0
        for sample_idx in range(batch_size):
            response_len = int(response_lengths[sample_idx])
            anchors: set[int] = set()
            rejects = sorted({idx for idx in reject_indices[sample_idx] if 0 <= idx < response_len})
            if rejects:
                boundaries = list(rejects)
                if boundaries[-1] < response_len - 1:
                    boundaries.append(response_len - 1)
                response_anchors = [-1] + boundaries
                for anchor, boundary in zip(response_anchors, boundaries, strict=False):
                    if boundary > anchor:
                        anchors.add(anchor)

            lengths = {
                len(rejected_anchors[sample_idx]),
                len(rejected_offsets[sample_idx]),
                len(rejected_token_ids[sample_idx]),
            }
            if len(lengths) != 1:
                raise ValueError(
                    "Rejected DFLASH TV metadata length mismatch while counting global blocks for "
                    f"sample {sample_idx}: anchors={len(rejected_anchors[sample_idx])}, "
                    f"offsets={len(rejected_offsets[sample_idx])}, "
                    f"token_ids={len(rejected_token_ids[sample_idx])}."
                )
            for anchor, offset, token_id in zip(
                rejected_anchors[sample_idx],
                rejected_offsets[sample_idx],
                rejected_token_ids[sample_idx],
                strict=True,
            ):
                if token_id >= 0 and offset > 0 and (-1 <= anchor < response_len):
                    anchors.add(anchor)
            total += len(anchors)
        return total

    def _compute_extra_global_batch_info(self, data: TensorDict) -> dict[str, float]:
        if not self._is_composed_dflash_module():
            return {}
        if bool(tu.get_non_tensor_data(data=data, key="opd_use_tv_loss", default=False)):
            if not self._is_exact_composed_dflash_module():
                raise NotImplementedError(
                    "use_tv_loss=True is only supported by ComposedDFlashStudentForCausalLM."
                )
            local_block_count = self._count_dflash_tv_blocks(data)
            global_block_count = torch.tensor(
                float(local_block_count), dtype=torch.float32, device=get_device_id()
            )
            torch.distributed.all_reduce(
                global_block_count, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
            )
            return {"opd_tv_global_block_count": float(global_block_count.item())}
        raw_anchors = tu.get_non_tensor_data(data=data, key="dflash_rejected_draft_anchor_indices", default=None)
        raw_token_ids = tu.get_non_tensor_data(data=data, key="dflash_rejected_draft_token_ids", default=None)
        raw_offsets = tu.get_non_tensor_data(data=data, key="dflash_rejected_draft_offsets", default=None)
        batch_size = int(data.batch_size[0])
        max_tokens_per_sample = self._get_dflash_rejected_draft_max_tokens_per_sample()
        first_token_only = bool(
            tu.get_non_tensor_data(data=data, key="opd_rejected_draft_first_token_only", default=False)
        )
        local_count = self._count_dflash_rejected_draft_tokens(
            raw_token_ids,
            batch_size=batch_size,
            max_tokens_per_sample=max_tokens_per_sample,
            raw_anchor_indices=raw_anchors,
            raw_offsets=raw_offsets,
            first_token_only=first_token_only,
        )
        global_count = torch.tensor(float(local_count), dtype=torch.float32, device=get_device_id())
        torch.distributed.all_reduce(
            global_count, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
        )
        batch_info = {"opd_rejected_draft_batch_num_tokens": float(global_count.item())}

        position_decay_enabled = bool(
            tu.get_non_tensor_data(data=data, key="opd_rejected_draft_position_decay_enabled", default=False)
        )
        if position_decay_enabled and not first_token_only and raw_offsets is not None:
            position_decay = float(
                tu.get_non_tensor_data(data=data, key="opd_rejected_draft_position_decay", default=0.9)
            )
            local_effective_count = self._sum_dflash_rejected_draft_position_weights(
                raw_token_ids,
                raw_offsets,
                batch_size=batch_size,
                max_tokens_per_sample=max_tokens_per_sample,
                decay=position_decay,
                raw_anchor_indices=raw_anchors,
                first_token_only=first_token_only,
            )
            global_effective_count = torch.tensor(
                float(local_effective_count), dtype=torch.float32, device=get_device_id()
            )
            torch.distributed.all_reduce(
                global_effective_count, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
            )
            batch_info["opd_rejected_draft_batch_effective_num_tokens"] = float(global_effective_count.item())
        return batch_info

    @staticmethod
    def _normalize_dflash_reject_indices(raw_indices, batch_size: int) -> list[list[int]]:
        if raw_indices is None:
            return []
        if hasattr(raw_indices, "tolist"):
            raw_indices = raw_indices.tolist()
        if batch_size == 1 and (not isinstance(raw_indices, (list, tuple)) or (
            raw_indices and all(not isinstance(item, (list, tuple)) for item in raw_indices)
        )):
            raw_indices = [raw_indices]
        normalized: list[list[int]] = []
        for item in raw_indices:
            if item is None:
                normalized.append([])
            elif hasattr(item, "tolist"):
                normalized.append([int(x) for x in item.tolist()])
            elif isinstance(item, (list, tuple)):
                normalized.append([int(x) for x in item])
            else:
                normalized.append([int(item)])
        return normalized

    @staticmethod
    def _normalize_dflash_rejected_draft_values(raw_values, batch_size: int, cast_fn) -> list[list]:
        def flatten_item(value) -> list:
            value = tu.unwrap_non_tensor_data(value)
            if value is None:
                return []
            if hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, (list, tuple)):
                flattened = []
                for sub_value in value:
                    flattened.extend(flatten_item(sub_value))
                return flattened
            return [cast_fn(value)]

        if raw_values is None:
            return [[] for _ in range(batch_size)]
        raw_values = tu.unwrap_non_tensor_data(raw_values)
        if hasattr(raw_values, "tolist"):
            raw_values = raw_values.tolist()
        elif not isinstance(raw_values, (list, tuple)):
            try:
                raw_values = list(raw_values)
            except TypeError:
                pass
        if isinstance(raw_values, (list, tuple)):
            raw_values = [tu.unwrap_non_tensor_data(item) for item in raw_values]
        if batch_size == 1:
            raw_values = [raw_values]

        normalized: list[list] = []
        for item in raw_values:
            normalized.append(flatten_item(item))
        return normalized

    @staticmethod
    def _filter_dflash_rejected_draft_first_tokens(
        *,
        anchors: list[list[int]],
        offsets: list[list[int]],
        token_ids: list[list[int]],
        teacher_logprobs: Optional[list[list[float]]],
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]], Optional[list[list[float]]]]:
        filtered_anchors: list[list[int]] = []
        filtered_offsets: list[list[int]] = []
        filtered_token_ids: list[list[int]] = []
        filtered_teacher_logprobs: Optional[list[list[float]]] = [] if teacher_logprobs is not None else None

        for sample_idx in range(len(token_ids)):
            sample_anchors = anchors[sample_idx] if sample_idx < len(anchors) else []
            sample_offsets = offsets[sample_idx] if sample_idx < len(offsets) else []
            sample_token_ids = token_ids[sample_idx]
            sample_teacher_logprobs = (
                teacher_logprobs[sample_idx]
                if teacher_logprobs is not None and sample_idx < len(teacher_logprobs)
                else []
            )
            lengths = {len(sample_anchors), len(sample_offsets), len(sample_token_ids)}
            if teacher_logprobs is not None:
                lengths.add(len(sample_teacher_logprobs))
            if len(lengths) != 1:
                raise ValueError(
                    "DFLASH rejected draft metadata length mismatch for first-token filtering "
                    f"sample {sample_idx}: anchors={len(sample_anchors)}, offsets={len(sample_offsets)}, "
                    f"token_ids={len(sample_token_ids)}"
                    + (
                        f", teacher_logprobs={len(sample_teacher_logprobs)}."
                        if teacher_logprobs is not None
                        else "."
                    )
                )

            selected_by_anchor: dict[int, tuple[int, int, int, int, Optional[float]]] = {}
            for item_idx, (anchor, offset, token_id) in enumerate(
                zip(sample_anchors, sample_offsets, sample_token_ids, strict=True)
            ):
                if token_id < 0 or offset <= 0:
                    continue
                teacher_value = sample_teacher_logprobs[item_idx] if teacher_logprobs is not None else None
                current = selected_by_anchor.get(anchor)
                if current is None or offset < current[2]:
                    selected_by_anchor[anchor] = (item_idx, anchor, offset, token_id, teacher_value)

            selected_items = sorted(selected_by_anchor.values(), key=lambda item: item[0])
            filtered_anchors.append([anchor for _, anchor, _, _, _ in selected_items])
            filtered_offsets.append([offset for _, _, offset, _, _ in selected_items])
            filtered_token_ids.append([token_id for _, _, _, token_id, _ in selected_items])
            if filtered_teacher_logprobs is not None:
                filtered_teacher_logprobs.append([float(teacher_value) for _, _, _, _, teacher_value in selected_items])

        return filtered_anchors, filtered_offsets, filtered_token_ids, filtered_teacher_logprobs

    @classmethod
    def _build_dflash_rejected_draft_tensors(
        cls,
        *,
        micro_batch: TensorDict,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        anchors = cls._normalize_dflash_rejected_draft_values(
            tu.get_non_tensor_data(
                data=micro_batch, key="dflash_rejected_draft_anchor_indices", default=None
            ),
            batch_size=batch_size,
            cast_fn=int,
        )
        offsets = cls._normalize_dflash_rejected_draft_values(
            tu.get_non_tensor_data(
                data=micro_batch, key="dflash_rejected_draft_offsets", default=None
            ),
            batch_size=batch_size,
            cast_fn=int,
        )
        token_ids = cls._normalize_dflash_rejected_draft_values(
            tu.get_non_tensor_data(
                data=micro_batch, key="dflash_rejected_draft_token_ids", default=None
            ),
            batch_size=batch_size,
            cast_fn=int,
        )
        teacher_logprobs = cls._normalize_dflash_rejected_draft_values(
            tu.get_non_tensor_data(
                data=micro_batch, key="dflash_rejected_draft_teacher_logprobs", default=None
            ),
            batch_size=batch_size,
            cast_fn=float,
        )
        first_token_only = bool(
            tu.get_non_tensor_data(data=micro_batch, key="opd_rejected_draft_first_token_only", default=False)
        )
        use_tv_loss = bool(tu.get_non_tensor_data(data=micro_batch, key="opd_use_tv_loss", default=False))
        # Exact TV validates and merges the complete suffix in the model. Keep the
        # legacy pre-filtering behavior unchanged for sampled-KL training.
        if first_token_only and not use_tv_loss:
            anchors, offsets, token_ids, teacher_logprobs = cls._filter_dflash_rejected_draft_first_tokens(
                anchors=anchors,
                offsets=offsets,
                token_ids=token_ids,
                teacher_logprobs=teacher_logprobs,
            )

        for key, values in (
            ("anchor_indices", anchors),
            ("offsets", offsets),
            ("token_ids", token_ids),
            ("teacher_logprobs", teacher_logprobs),
        ):
            if len(values) != batch_size:
                raise ValueError(
                    "DFLASH rejected draft metadata batch size mismatch: "
                    f"{key} has {len(values)} entries for batch_size={batch_size}."
                )

        for sample_idx in range(batch_size):
            lengths = {
                len(anchors[sample_idx]),
                len(offsets[sample_idx]),
                len(token_ids[sample_idx]),
                len(teacher_logprobs[sample_idx]),
            }
            if len(lengths) != 1:
                raise ValueError(
                    "DFLASH rejected draft metadata length mismatch for sample "
                    f"{sample_idx}: anchors={len(anchors[sample_idx])}, "
                    f"offsets={len(offsets[sample_idx])}, "
                    f"token_ids={len(token_ids[sample_idx])}, "
                    f"teacher_logprobs={len(teacher_logprobs[sample_idx])}."
                )

        max_items = max((len(item) for item in anchors), default=0)
        width = max(max_items, 1)
        anchor_tensor = torch.full((batch_size, width), -2, dtype=torch.long, device=device)
        offset_tensor = torch.full((batch_size, width), -1, dtype=torch.long, device=device)
        token_tensor = torch.full((batch_size, width), -1, dtype=torch.long, device=device)
        teacher_tensor = torch.zeros((batch_size, width), dtype=torch.float32, device=device)
        mask_tensor = torch.zeros((batch_size, width), dtype=torch.bool, device=device)

        for sample_idx in range(batch_size):
            count = len(anchors[sample_idx])
            if count == 0:
                continue
            anchor_tensor[sample_idx, :count] = torch.tensor(anchors[sample_idx], dtype=torch.long, device=device)
            offset_tensor[sample_idx, :count] = torch.tensor(offsets[sample_idx], dtype=torch.long, device=device)
            token_tensor[sample_idx, :count] = torch.tensor(token_ids[sample_idx], dtype=torch.long, device=device)
            teacher_tensor[sample_idx, :count] = torch.tensor(
                teacher_logprobs[sample_idx], dtype=torch.float32, device=device
            )
            mask_tensor[sample_idx, :count] = True

        return {
            "dflash_rejected_draft_anchor_indices": anchor_tensor,
            "dflash_rejected_draft_offsets": offset_tensor,
            "dflash_rejected_draft_token_ids": token_tensor,
            "dflash_rejected_draft_teacher_logprobs": teacher_tensor,
            "dflash_rejected_draft_mask": mask_tensor,
        }

    @classmethod
    def _build_dflash_replay_block_tensors(
        cls,
        *,
        micro_batch: TensorDict,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        anchors = cls._normalize_dflash_rejected_draft_values(
            tu.get_non_tensor_data(data=micro_batch, key="dflash_replay_block_anchor_indices", default=None),
            batch_size=batch_size,
            cast_fn=int,
        )
        accepted_lengths = cls._normalize_dflash_rejected_draft_values(
            tu.get_non_tensor_data(data=micro_batch, key="dflash_replay_block_accepted_lengths", default=None),
            batch_size=batch_size,
            cast_fn=int,
        )
        drafted_lengths = cls._normalize_dflash_rejected_draft_values(
            tu.get_non_tensor_data(data=micro_batch, key="dflash_replay_block_drafted_lengths", default=None),
            batch_size=batch_size,
            cast_fn=int,
        )
        for sample_idx in range(batch_size):
            lengths = {
                len(anchors[sample_idx]),
                len(accepted_lengths[sample_idx]),
                len(drafted_lengths[sample_idx]),
            }
            if len(lengths) != 1:
                raise ValueError(
                    "DFLASH replay block metadata length mismatch for sample "
                    f"{sample_idx}: anchors={len(anchors[sample_idx])}, "
                    f"accepted={len(accepted_lengths[sample_idx])}, drafted={len(drafted_lengths[sample_idx])}."
                )

        max_items = max((len(item) for item in anchors), default=0)
        width = max(max_items, 1)
        anchor_tensor = torch.full((batch_size, width), -2, dtype=torch.long, device=device)
        accepted_tensor = torch.zeros((batch_size, width), dtype=torch.long, device=device)
        drafted_tensor = torch.zeros((batch_size, width), dtype=torch.long, device=device)
        mask_tensor = torch.zeros((batch_size, width), dtype=torch.bool, device=device)

        for sample_idx in range(batch_size):
            count = len(anchors[sample_idx])
            if count == 0:
                continue
            anchor_tensor[sample_idx, :count] = torch.tensor(anchors[sample_idx], dtype=torch.long, device=device)
            accepted_tensor[sample_idx, :count] = torch.tensor(
                accepted_lengths[sample_idx], dtype=torch.long, device=device
            )
            drafted_tensor[sample_idx, :count] = torch.tensor(
                drafted_lengths[sample_idx], dtype=torch.long, device=device
            )
            mask_tensor[sample_idx, :count] = True

        return {
            "dflash_replay_block_anchor_indices": anchor_tensor,
            "dflash_replay_block_accepted_lengths": accepted_tensor,
            "dflash_replay_block_drafted_lengths": drafted_tensor,
            "dflash_replay_block_mask": mask_tensor,
        }

    @staticmethod
    def _nested_to_padded_tensor_no_kernel(nested_tensor: torch.Tensor, padding: int, output_size: tuple[int, ...]):
        """Pad jagged nested tensors without calling aten::to_padded_tensor.

        Some PyTorch builds do not provide the CUDA kernel for
        torch.nested.to_padded_tensor. The DFLASH OPD path needs padded tensors
        before calling the target model, so use unbind + copy instead.
        """
        if not nested_tensor.is_nested:
            slices = tuple(slice(0, min(size, nested_tensor.shape[dim])) for dim, size in enumerate(output_size))
            output = nested_tensor.new_full(output_size, padding)
            output[slices] = nested_tensor[slices]
            return output

        parts = list(nested_tensor.unbind())
        if len(parts) == 0:
            raise ValueError("Cannot pad an empty nested tensor.")

        output = parts[0].new_full(output_size, padding)
        for i, part in enumerate(parts):
            if part.dim() == 1:
                output[i, : part.shape[0]] = part
            elif part.dim() == 2:
                output[i, : part.shape[0], : part.shape[1]] = part
            else:
                raise NotImplementedError(f"Unsupported nested tensor part dim for DFLASH OPD: {part.dim()}.")
        return output

    @staticmethod
    def _compute_composed_dflash_lengths(
        *,
        attention_mask: torch.Tensor,
        prompts: torch.Tensor,
        responses: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        if prompts.is_nested:
            prompt_lengths = prompts.offsets().diff().to(device=device, dtype=torch.long)
        else:
            prompt_width = prompts.shape[1]
            prompt_lengths = attention_mask[:, :prompt_width].sum(dim=1).to(device=device, dtype=torch.long)

        if responses.is_nested:
            response_lengths = responses.offsets().diff().to(device=device, dtype=torch.long)
        else:
            prompt_width = prompts.shape[1] if not prompts.is_nested else int(prompt_lengths.max().item())
            response_width = responses.shape[1]
            response_lengths = (
                attention_mask[:, prompt_width : prompt_width + response_width]
                .sum(dim=1)
                .to(device=device, dtype=torch.long)
            )
        return prompt_lengths, response_lengths

    @staticmethod
    def _compact_padded_composed_dflash_inputs(
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        pad_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_lengths = attention_mask.long().sum(dim=1)
        max_seq_len = int(valid_lengths.max().item())
        batch_size = input_ids.shape[0]
        device = input_ids.device

        compact_input_ids = input_ids.new_full((batch_size, max_seq_len), pad_token_id)
        compact_attention_mask = (
            torch.arange(max_seq_len, device=device).unsqueeze(0) < valid_lengths.to(device=device).unsqueeze(1)
        ).to(dtype=attention_mask.dtype)

        if position_ids.dim() == 3:
            compact_position_ids = position_ids.new_zeros((batch_size, position_ids.shape[1], max_seq_len))
        else:
            compact_position_ids = position_ids.new_zeros((batch_size, max_seq_len))

        for batch_idx in range(batch_size):
            valid_mask = attention_mask[batch_idx].bool()
            valid_len = int(valid_lengths[batch_idx].item())
            compact_input_ids[batch_idx, :valid_len] = input_ids[batch_idx, valid_mask]
            if position_ids.dim() == 3:
                compact_position_ids[batch_idx, :, :valid_len] = position_ids[batch_idx, :, valid_mask]
            else:
                compact_position_ids[batch_idx, :valid_len] = position_ids[batch_idx, valid_mask]

        return compact_input_ids, compact_attention_mask, compact_position_ids

    def _prepare_composed_dflash_inputs(self, micro_batch: TensorDict):
        pad_token_id = tu.get_non_tensor_data(data=micro_batch, key="pad_token_id", default=0)
        input_ids = micro_batch["input_ids"]
        position_ids = micro_batch["position_ids"]
        attention_mask = micro_batch["attention_mask"]
        original_attention_mask = attention_mask
        prompts = micro_batch["prompts"]
        responses = micro_batch["responses"]

        prompt_lengths, response_lengths = self._compute_composed_dflash_lengths(
            attention_mask=original_attention_mask,
            prompts=prompts,
            responses=responses,
            device=input_ids.device,
        )

        if input_ids.is_nested:
            batch_size = micro_batch.batch_size[0]
            seq_lengths = input_ids.offsets().diff()
            max_seq_len = int(seq_lengths.max().item())
            device = input_ids.device
            input_ids = self._nested_to_padded_tensor_no_kernel(
                input_ids, padding=pad_token_id, output_size=(batch_size, max_seq_len)
            )
            attention_mask = torch.arange(max_seq_len, device=device).unsqueeze(0) < seq_lengths.unsqueeze(1)
            attention_mask = attention_mask.to(dtype=torch.long)
            if position_ids.dim() == 3:
                position_ids = self._nested_to_padded_tensor_no_kernel(
                    position_ids, padding=0, output_size=(batch_size, 4, max_seq_len)
                ).transpose(0, 1)
            else:
                position_ids = self._nested_to_padded_tensor_no_kernel(
                    position_ids, padding=0, output_size=(batch_size, max_seq_len)
                )
        else:
            batch_size = input_ids.shape[0]
            if not bool(original_attention_mask.all()):
                input_ids, attention_mask, position_ids = self._compact_padded_composed_dflash_inputs(
                    input_ids=input_ids,
                    attention_mask=original_attention_mask,
                    position_ids=position_ids,
                    pad_token_id=pad_token_id,
                )

        raw_reject_indices = tu.get_non_tensor_data(
            data=micro_batch, key="dflash_reject_token_indices", default=None
        )
        reject_indices = self._normalize_dflash_reject_indices(raw_reject_indices, batch_size=batch_size)
        if len(reject_indices) != batch_size:
            raise ValueError(
                "DFLASH OPD reject metadata batch size mismatch: "
                f"got {len(reject_indices)} entries for batch_size={batch_size}."
            )
        max_rejects = max((len(item) for item in reject_indices), default=0)
        reject_tensor = torch.full(
            (batch_size, max(max_rejects, 1)), -1, dtype=torch.long, device=input_ids.device
        )
        for i, sample_indices in enumerate(reject_indices):
            if sample_indices:
                reject_tensor[i, : len(sample_indices)] = torch.tensor(
                    sample_indices, dtype=torch.long, device=input_ids.device
                )

        use_replay_dis = bool(tu.get_non_tensor_data(data=micro_batch, key="opd_use_replay_dis", default=False))
        use_tv_loss = bool(tu.get_non_tensor_data(data=micro_batch, key="opd_use_tv_loss", default=False))
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "dflash_prompt_lengths": prompt_lengths,
            "dflash_response_lengths": response_lengths,
            "dflash_reject_token_indices": reject_tensor,
            **self._build_dflash_rejected_draft_tensors(
                micro_batch=micro_batch,
                batch_size=batch_size,
                device=input_ids.device,
            ),
            "dflash_calculate_entropy": tu.get_non_tensor_data(
                data=micro_batch, key="calculate_entropy", default=False
            ),
            "dflash_use_replay_dis": use_replay_dis,
            "dflash_use_tv_loss": use_tv_loss,
            "dflash_rejected_draft_first_token_only": bool(
                tu.get_non_tensor_data(
                    data=micro_batch, key="opd_rejected_draft_first_token_only", default=False
                )
            ),
            "dflash_use_task_rewards": bool(
                tu.get_non_tensor_data(data=micro_batch, key="opd_use_task_rewards", default=False)
            ),
        }
        if use_replay_dis:
            model_inputs.update(
                self._build_dflash_replay_block_tensors(
                    micro_batch=micro_batch,
                    batch_size=batch_size,
                    device=input_ids.device,
                )
            )
        return model_inputs, {"dflash_opd": True}

    def prepare_model_inputs(self, micro_batch: TensorDict):
        if self._is_composed_dflash_module():
            return self._prepare_composed_dflash_inputs(micro_batch=micro_batch)

        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        temperature = micro_batch["temperature"]
        temperature_item = temperature
        if use_fused_kernels:
            assert not isinstance(temperature, torch.Tensor), (
                "use_fused_kernels does not support per sample temperature yet"
            )
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        multi_modal_inputs = extract_multi_modal_inputs(micro_batch.get("multi_modal_inputs", []))
        input_ids = micro_batch["input_ids"]
        position_ids = micro_batch["position_ids"]

        if not isinstance(temperature, torch.Tensor):
            temperature = torch.tensor([temperature] * input_ids.shape[0], device=input_ids.device)

        temperature = temperature.to(torch.float32)
        assert temperature.shape[0] == input_ids.shape[0]

        # args used to get outputs
        output_args = {}

        if use_remove_padding:
            # support per sample temperature
            # temperature (bsz,)
            # input_ids (bsz, j1)
            temperature_rmpad = verl_F.expand_as_nested(temperature, input_ids).values()  # (total_nnz,)
            temperature_rmpad = temperature_rmpad.unsqueeze(0)  # (1, total_nnz)

            if pad_mode == DatasetPadMode.NO_PADDING:
                input_ids_rmpad = input_ids.values().unsqueeze(0)  # (1, total_nnz)
                if position_ids.dim() == 3:
                    position_ids_rmpad = position_ids.values().unsqueeze(1)  # (4, 1, total_nnz)
                else:
                    position_ids_rmpad = position_ids.values().unsqueeze(0)  # (1, total_nnz)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.use_ulysses_sp:
                is_vlm_model = hasattr(getattr(self.module, "module", self.module).config, "vision_config")
                if is_vlm_model:
                    # vlm model's inputs will be sliced after embedding
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                        input_ids_rmpad,
                        position_ids_rmpad=position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                else:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad,
                        position_ids_rmpad=position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size,
                        skip_position_ids_rmpad=True if self.__class__.__name__ == "VeOmniEngineWithLMHead" else False,
                    )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled,
                    position_ids_rmpad=None,
                    sp_size=self.ulysses_sequence_parallel_size,
                )

                temperature_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                    temperature_rmpad, position_ids_rmpad=None, sp_size=self.ulysses_sequence_parallel_size, pad_value=1
                )

                output_args["pad_size"] = pad_size

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)
            temperature_rmpad = temperature_rmpad.squeeze(0)
            output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled
            output_args["temperature_rmpad"] = temperature_rmpad

            # only pass input_ids and position_ids to enable flash_attn_varlen

            model_inputs = {
                "input_ids": input_ids_rmpad,
                "attention_mask": None,
                "position_ids": position_ids_rmpad,
            }

        else:
            if pad_mode == DatasetPadMode.NO_PADDING:
                input_ids = micro_batch["input_ids"]
                position_ids = micro_batch["position_ids"]
                loss_mask = micro_batch["loss_mask"]

                pad_token_id = tu.get_non_tensor_data(data=micro_batch, key="pad_token_id", default=0)
                batch_size = micro_batch.batch_size[0]
                seq_len_effective = input_ids.offsets().diff()
                max_seq_len = max(seq_len_effective)

                input_ids_rmpad_rolled = torch.roll(input_ids.values(), shifts=-1, dims=0)
                output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled
                # we store the per sample temperature
                output_args["temperature"] = temperature

                input_ids = torch.nested.to_padded_tensor(
                    input_ids, padding=pad_token_id, output_size=(batch_size, max_seq_len)
                )

                if position_ids.dim() == 3:
                    position_ids = torch.nested.to_padded_tensor(
                        position_ids, padding=0, output_size=(batch_size, 4, max_seq_len)
                    ).transpose(0, 1)  # (4, batch_size, max_seq_len)
                else:
                    position_ids = torch.nested.to_padded_tensor(
                        position_ids, padding=0, output_size=(batch_size, max_seq_len)
                    )

                attention_mask_list = [torch.ones_like(t, dtype=torch.int32) for t in loss_mask]
                attention_mask = torch.nested.as_nested_tensor(attention_mask_list, layout=torch.jagged)
                attention_mask = torch.nested.to_padded_tensor(
                    attention_mask, padding=0, output_size=(batch_size, max_seq_len)
                )

                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                }

            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        extra_args = {}
        if use_fused_kernels:
            extra_args["temperature"] = temperature_item
            extra_args["return_dict"] = True

        model_inputs.update(multi_modal_inputs)
        model_inputs.update(extra_args)

        return model_inputs, output_args

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict, logits_processor_func):
        if output_args.get("dflash_opd", False):
            tv_block_losses = (
                output.get("dflash_tv_block_losses")
                if isinstance(output, dict)
                else getattr(output, "dflash_tv_block_losses", None)
            )
            if tv_block_losses is not None:
                tv_keys = (
                    "block_losses",
                    "block_mask",
                    "distance_sum",
                    "overlap_sum",
                    "expected_accept_length_sum",
                    "block_count",
                    "position_count",
                    "teacher_branch_position_count",
                )
                model_output = {}
                for suffix in tv_keys:
                    source_key = f"dflash_tv_{suffix}"
                    value = output.get(source_key) if isinstance(output, dict) else getattr(output, source_key, None)
                    if value is None:
                        raise RuntimeError(f"Exact DFLASH TV forward did not return {source_key}.")
                    model_output[f"opd_tv_{suffix}"] = value
                dflash_log_probs = output.get("dflash_log_probs") if isinstance(output, dict) else getattr(
                    output, "dflash_log_probs", None
                )
                if dflash_log_probs is not None:
                    dflash_loss_mask = output.get("dflash_loss_mask") if isinstance(output, dict) else getattr(
                        output, "dflash_loss_mask", None
                    )
                    if dflash_loss_mask is None:
                        raise RuntimeError("Task-reward DFLASH TV forward did not return dflash_loss_mask.")

                    def _format_tv_sequence_output(sequence_output: torch.Tensor) -> torch.Tensor:
                        input_ids = micro_batch["input_ids"]
                        if input_ids.is_nested:
                            seq_lens = input_ids.offsets().diff()
                            values = torch.cat(
                                [
                                    sequence_output[i, : int(seq_lens[i].item())]
                                    for i in range(seq_lens.numel())
                                ],
                                dim=0,
                            )
                            return torch.nested.nested_tensor_from_jagged(values, input_ids.offsets())
                        attention_mask = micro_batch["attention_mask"]
                        seq_lens = attention_mask.sum(dim=1).to(torch.long)
                        return torch.cat(
                            [
                                sequence_output[i, : int(seq_lens[i].item())]
                                for i in range(seq_lens.numel())
                            ],
                            dim=0,
                        )

                    model_output["log_probs"] = _format_tv_sequence_output(dflash_log_probs)
                    model_output["opd_loss_mask"] = _format_tv_sequence_output(dflash_loss_mask)
                    calculate_entropy = tu.get_non_tensor_data(
                        data=micro_batch, key="calculate_entropy", default=False
                    )
                    if calculate_entropy:
                        dflash_entropy = output.get("dflash_entropy") if isinstance(output, dict) else getattr(
                            output, "dflash_entropy", None
                        )
                        if dflash_entropy is None:
                            raise RuntimeError("Task-reward DFLASH TV forward did not return dflash_entropy.")
                        model_output["entropy"] = _format_tv_sequence_output(dflash_entropy)
                return model_output

            if isinstance(output, dict):
                dflash_log_probs = output.get("dflash_log_probs")
                dflash_entropy = output.get("dflash_entropy")
                dflash_loss_mask = output.get("dflash_loss_mask")
                dflash_response_offsets = output.get("dflash_response_offsets")
                dflash_rejected_draft_student_log_probs = output.get("dflash_rejected_draft_student_log_probs")
                dflash_rejected_draft_teacher_log_probs = output.get("dflash_rejected_draft_teacher_log_probs")
                dflash_rejected_draft_loss_mask = output.get("dflash_rejected_draft_loss_mask")
                dflash_rejected_draft_offsets = output.get("dflash_rejected_draft_offsets")
                eagle3_native_ce_losses = output.get("eagle3_native_ce_losses")
                eagle3_selected_scalar_loss_mask = output.get("eagle3_selected_scalar_loss_mask")
            else:
                dflash_log_probs = getattr(output, "dflash_log_probs", None)
                dflash_entropy = getattr(output, "dflash_entropy", None)
                dflash_loss_mask = getattr(output, "dflash_loss_mask", None)
                dflash_response_offsets = getattr(output, "dflash_response_offsets", None)
                dflash_rejected_draft_student_log_probs = getattr(
                    output, "dflash_rejected_draft_student_log_probs", None
                )
                dflash_rejected_draft_teacher_log_probs = getattr(
                    output, "dflash_rejected_draft_teacher_log_probs", None
                )
                dflash_rejected_draft_loss_mask = getattr(output, "dflash_rejected_draft_loss_mask", None)
                dflash_rejected_draft_offsets = getattr(output, "dflash_rejected_draft_offsets", None)
                eagle3_native_ce_losses = getattr(output, "eagle3_native_ce_losses", None)
                eagle3_selected_scalar_loss_mask = getattr(output, "eagle3_selected_scalar_loss_mask", None)

            if dflash_log_probs is None:
                raise RuntimeError("Composed DFLASH OPD forward did not return dflash_log_probs.")
            if dflash_loss_mask is None:
                raise RuntimeError("Composed DFLASH OPD forward did not return dflash_loss_mask.")

            def format_dflash_sequence_output(sequence_output: torch.Tensor) -> torch.Tensor:
                input_ids = micro_batch["input_ids"]
                if input_ids.is_nested:
                    seq_lens = input_ids.offsets().diff()
                    values = torch.cat(
                        [sequence_output[i, : int(seq_lens[i].item())] for i in range(seq_lens.numel())], dim=0
                    )
                    return torch.nested.nested_tensor_from_jagged(values, input_ids.offsets())

                attention_mask = micro_batch["attention_mask"]
                seq_lens = attention_mask.sum(dim=1).to(torch.long)
                return torch.cat(
                    [sequence_output[i, : int(seq_lens[i].item())] for i in range(seq_lens.numel())], dim=0
                )

            model_output = {
                "log_probs": format_dflash_sequence_output(dflash_log_probs),
                "opd_loss_mask": format_dflash_sequence_output(dflash_loss_mask),
            }
            if dflash_response_offsets is not None:
                model_output["opd_response_offsets"] = format_dflash_sequence_output(dflash_response_offsets)
            if eagle3_native_ce_losses is not None:
                model_output["eagle3_native_ce_losses"] = format_dflash_sequence_output(eagle3_native_ce_losses)
            if eagle3_selected_scalar_loss_mask is not None:
                model_output["eagle3_selected_scalar_loss_mask"] = format_dflash_sequence_output(
                    eagle3_selected_scalar_loss_mask
                )
            if dflash_rejected_draft_student_log_probs is not None:
                model_output["opd_rejected_draft_student_log_probs"] = dflash_rejected_draft_student_log_probs
            if dflash_rejected_draft_teacher_log_probs is not None:
                model_output["opd_rejected_draft_teacher_log_probs"] = dflash_rejected_draft_teacher_log_probs
            if dflash_rejected_draft_loss_mask is not None:
                model_output["opd_rejected_draft_loss_mask"] = dflash_rejected_draft_loss_mask
            if dflash_rejected_draft_offsets is not None:
                model_output["opd_rejected_draft_offsets"] = dflash_rejected_draft_offsets
            for source_key, output_key in (
                ("dflash_opd_valid_anchor_count", "opd_valid_anchor_count"),
                ("dflash_opd_skipped_sample_count", "opd_skipped_sample_count"),
                ("dflash_opd_empty_reject_sample_count", "opd_empty_reject_sample_count"),
                ("dflash_opd_total_reject_count", "opd_total_reject_count"),
                ("dflash_opd_sample_count", "opd_sample_count"),
                ("dflash_opd_target_token_count", "opd_target_token_count"),
                ("dflash_opd_rejected_draft_token_count", "opd_rejected_draft_token_count"),
                ("dflash_opd_attention_impl_id", "opd_attention_impl_id"),
                ("dflash_opd_profile_teacher_forward_ms", "opd_profile_teacher_forward_ms"),
                ("dflash_opd_profile_draft_forward_ms", "opd_profile_draft_forward_ms"),
                ("dflash_opd_profile_lm_head_ms", "opd_profile_lm_head_ms"),
                ("dflash_opd_profile_total_forward_ms", "opd_profile_total_forward_ms"),
                ("eagle3_response_loss_mode_id", "eagle3_response_loss_mode_id"),
                ("eagle3_target_argmax_total_count", "eagle3_target_argmax_total_count"),
                ("eagle3_target_argmax_supported_count", "eagle3_target_argmax_supported_count"),
                ("eagle3_draft_target_top1_correct_count", "eagle3_draft_target_top1_correct_count"),
            ):
                value = output.get(source_key) if isinstance(output, dict) else getattr(output, source_key, None)
                if value is not None:
                    model_output[output_key] = value
            for state_group in (
                "matched_target_trajectory",
                "accepted_replay",
                "first_rejected_replay",
                "post_rejection_suffix_replay",
            ):
                for source_suffix, output_suffix in (("sum_kl", "sum_KL"), ("num_states", "num_states")):
                    source_key = f"dflash_replay_dis_{state_group}_{source_suffix}"
                    value = output.get(source_key) if isinstance(output, dict) else getattr(output, source_key, None)
                    if value is not None:
                        model_output[f"replay_dis/mismatch/{state_group}/{output_suffix}"] = value
            calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)
            if calculate_entropy:
                if dflash_entropy is None:
                    raise RuntimeError("Composed DFLASH OPD forward did not return dflash_entropy.")
                model_output["entropy"] = format_dflash_sequence_output(dflash_entropy)
            return model_output

        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)
        distillation_use_topk = tu.get_non_tensor_data(data=micro_batch, key="distillation_use_topk", default=False)

        model_output = {}

        input_ids = micro_batch["input_ids"]

        if use_remove_padding:
            input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]
            temperature_rmpad = output_args["temperature_rmpad"]

            if use_fused_kernels:
                # temperature is singleton
                log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
            else:
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                logits_rmpad.div_(temperature_rmpad.clamp(min=1e-8).unsqueeze(-1).to(logits_rmpad.dtype))

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(
                    logits=logits_rmpad,
                    labels=input_ids_rmpad_rolled,
                    inplace_backward=inplace_backward,
                )

                # compute entropy
                if calculate_entropy:
                    if not self.engine_config.entropy_checkpointing:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                    else:
                        entropy_rmpad = torch.utils.checkpoint.checkpoint(
                            self.compute_entropy_from_logits, logits_rmpad
                        )

                # logits_processor_func return tensors with shape (1, total_nnz/sp_size)
                if distillation_use_topk:
                    outputs = logits_processor_func(student_logits=logits_rmpad.unsqueeze(0), data=micro_batch)
                    cu_seqlens = input_ids.offsets()
                    for k, v in outputs.items():
                        v = v.squeeze(0)
                        assert v.shape == log_probs.shape, f"log_probs shape: {log_probs.shape}, {k} shape: {v.shape}"
                        if self.use_ulysses_sp:
                            pad_size = output_args["pad_size"]
                            v = gather_outputs_and_unpad(v, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                        model_output[k] = torch.nested.nested_tensor_from_jagged(v, cu_seqlens)

            # gather log_prob if sp > 1
            if self.use_ulysses_sp:
                pad_size = output_args["pad_size"]

                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(
                    log_probs,
                    gather_dim=0,
                    unpad_dim=0,
                    padding_size=pad_size,
                )
                if calculate_entropy:
                    entropy_rmpad = gather_outputs_and_unpad(
                        entropy_rmpad,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                # (bsz, j1), for each sample, is the length of each sample: [real_prompt length + real_response length]
                log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
                if calculate_entropy:
                    entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        else:  # not using rmpad and no ulysses sp
            response_length = tu.get_non_tensor_data(data=micro_batch, key="max_response_length", default=1024)
            if use_fused_kernels:
                log_probs = output.log_probs[:, -response_length - 1 : -1]
                entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:
                logits = output.logits  # (bsz, response_length, vocab_size)
                temperature = output_args["temperature"]  # (bsz,)
                temperature = temperature.unsqueeze(-1).unsqueeze(-1)
                logits.div_(temperature.clamp(min=1e-8).to(logits.dtype))

                if calculate_entropy:
                    if not self.engine_config.entropy_checkpointing:
                        entropy = verl_F.entropy_from_logits(logits)
                    else:
                        entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

                if pad_mode == DatasetPadMode.NO_PADDING:
                    cu_seqlens = input_ids.offsets()
                    seq_lengths = cu_seqlens.diff()
                    starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
                    logits = torch.nested.narrow(logits, 1, starts, seq_lengths, layout=torch.jagged)
                    logits_rmpad = torch.cat([t for t in logits.unbind()])
                    input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]
                    log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)
                    # (bsz, j1), for each sample, length of each sample: [real_prompt_length + real_response_length]
                    log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
                    if calculate_entropy:
                        entropy = torch.nested.narrow(entropy, 1, starts, seq_lengths, layout=torch.jagged)
                        entropy_rmpad = torch.cat([t for t in entropy.unbind()])
                        entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
                else:
                    raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        model_output["log_probs"] = log_probs
        if calculate_entropy:
            model_output["entropy"] = entropy

        return model_output

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        device_name = get_device_name()
        # actually, we should avoid assigning like this...
        micro_batch = micro_batch.to(get_device_id())
        model_inputs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)

        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            raw_output = self.module(
                **model_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating

            model_output = self.prepare_model_outputs(
                output=raw_output, output_args=output_args, micro_batch=micro_batch, logits_processor_func=loss_function
            )

            if loss_function is not None:
                loss, metrics = loss_function(
                    model_output=model_output, data=micro_batch, dp_group=self.get_data_parallel_group()
                )
            else:
                assert forward_only, "forward_only must be True when loss_function is None"
                loss = torch.tensor(1.0, device=device_name)
                metrics = {}

            for key, value in model_output.items():
                if key.startswith("replay_dis/") and isinstance(value, torch.Tensor) and value.dim() == 0:
                    metrics[key] = value.detach().float().item()

            def keep_postprocess_model_output(key: str, value: object) -> bool:
                if key.startswith("opd_tv_"):
                    return False
                if not isinstance(value, torch.Tensor):
                    return False
                # Scalar diagnostics are already represented in metrics. The engine
                # postprocessor only knows how to rebuild per-sample tensor outputs.
                return value.dim() > 0

            output = {
                "model_output": {
                    key: value
                    for key, value in model_output.items()
                    if keep_postprocess_model_output(key, value)
                },
                "loss": loss.detach().item(),
                "metrics": metrics,
            }

            return loss, output


@EngineRegistry.register(model_type="value_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class FSDPEngineWithValueHead(FSDPEngineWithLMHead):
    """
    The only difference between critic and actor is how the raw model output is processed
    """

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict, logits_processor_func):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)

        input_ids = micro_batch["input_ids"]
        if use_remove_padding:
            if hasattr(self.module, "v_head"):
                # For trl.AutoModelForCausalLMWithValueHead
                values_rmpad = output[2].squeeze(0)
            else:
                values_rmpad = output.logits
                values_rmpad = values_rmpad.squeeze(0)  # (total_nnz, 1)
                # critic model arch is like Qwen3ForTokenClassfication and num_labels=1
                # so we squeeze the last dimension here to get the value for each token
                values_rmpad = values_rmpad.squeeze(-1)

            # gather output if sp > 1
            if self.use_ulysses_sp:
                pad_size = output_args["pad_size"]
                values_rmpad = gather_outputs_and_unpad(values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                # (bsz, j1), for each sample, is the length of each sample: [real_prompt length + real_response length]
                values = torch.nested.nested_tensor_from_jagged(values_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        else:
            if hasattr(self.module, "v_head"):
                # For trl.AutoModelForCausalLMWithValueHead
                values = output[2]
            else:
                values = output.logits.squeeze(-1)

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                seq_lengths = cu_seqlens.diff()
                starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
                values = torch.nested.narrow(values, 1, starts, seq_lengths, layout=torch.jagged)
                values_rmpad = torch.cat([t for t in values.unbind()])
                # (bsz, j1), for each sample, length of each sample: [real_prompt_length + real_response_length]
                values = torch.nested.nested_tensor_from_jagged(values_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        return {"values": values}
