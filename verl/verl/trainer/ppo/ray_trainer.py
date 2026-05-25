# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import math
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import ListConfig, OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.distillation.losses import is_distillation_enabled
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_sglang_rollout_metrics,
    compute_sglang_spec_accept_length,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import (
    Role,
    WorkerType,
    need_critic,
    need_reference_policy,
    need_reward_model,
    need_teacher_policy,
)
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import DistillationConfig, EngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _safe_experiment_path(experiment_name: Optional[str]) -> Path:
    raw_name = experiment_name or os.environ.get("WANDB_NAME") or os.environ.get("EXP_NAME") or "unknown"
    raw_name = str(raw_name).strip().lstrip("/\\") or "unknown"
    parts = [part for part in Path(raw_name).parts if part not in ("", ".", "..")]
    return Path(*parts) if parts else Path("unknown")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # GDPO: pass raw data for per-dimension reward extraction
        if adv_estimator in (AdvantageEstimator.GDPO, "gdpo"):
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
            adv_kwargs["batch"] = data.batch
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # old_log_probs needed for path-variance proxy: w_t = 1 - 2*exp(old_log_probs) + sum_pi_squared
            adv_kwargs["old_log_probs"] = data.batch["old_log_probs"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_teacher_policy = need_teacher_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        self.rollout_speed_test_only = bool(self.config.trainer.get("rollout_speed_test_only", False))

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.checkpoint_manager = None

    @staticmethod
    def _as_path_list(paths) -> list[str]:
        if isinstance(paths, (list, tuple, ListConfig)):
            return [str(path) for path in paths]
        return [str(paths)]

    @staticmethod
    def _speed_test_name_from_path(path: str, used_names: set[str]) -> str:
        name = os.path.splitext(os.path.basename(str(path)))[0]
        for suffix in ("_user_prompt", "_128"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
        while "__" in safe_name:
            safe_name = safe_name.replace("__", "_")
        safe_name = safe_name or "benchmark"
        base_name = safe_name
        suffix_id = 2
        while safe_name in used_names:
            safe_name = f"{base_name}_{suffix_id}"
            suffix_id += 1
        used_names.add(safe_name)
        return safe_name

    def _create_rollout_speed_test_dataloaders(self, create_rl_dataset, collate_fn, num_workers: int):
        val_files = self._as_path_list(self.config.data.val_files)
        assert len(val_files) > 0, "rollout speed test requires at least one data.val_files entry."

        val_batch_size_config = self.config.data.val_batch_size
        used_names: set[str] = set()
        self.rollout_speed_test_dataloaders = []
        for path in val_files:
            dataset = create_rl_dataset(
                [path],
                self.config.data,
                self.tokenizer,
                self.processor,
                is_train=False,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
            assert len(dataset) > 0, f"Rollout speed test dataset is empty: {path}"
            val_batch_size = val_batch_size_config if val_batch_size_config is not None else len(dataset)
            dataloader = StatefulDataLoader(
                dataset=dataset,
                batch_size=val_batch_size,
                num_workers=num_workers,
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )
            name = self._speed_test_name_from_path(path, used_names)
            self.rollout_speed_test_dataloaders.append((name, path, dataset, dataloader))

        sizes = {name: len(dataset) for name, _, dataset, _ in self.rollout_speed_test_dataloaders}
        print(f"Rollout speed test dataloaders: {sizes}")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None and not self.rollout_speed_test_only:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        if self.rollout_speed_test_only:
            self.val_dataloader = None
            self._create_rollout_speed_test_dataloaders(create_rl_dataset, collate_fn, num_workers)
        else:
            val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
            if val_batch_size is None:
                val_batch_size = len(self.val_dataset)

            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=val_batch_size,
                num_workers=num_workers,
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        if not self.rollout_speed_test_only:
            assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        if self.rollout_speed_test_only:
            print(f"Size of train dataloader: {len(self.train_dataloader)}")
        else:
            print(
                f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
                f"{len(self.val_dataloader)}"
            )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")
        update_accumulation_steps = max(int(self.config.trainer.get("update_accumulation_steps", 1)), 1)
        total_update_steps = math.ceil(total_training_steps / update_accumulation_steps)
        if update_accumulation_steps > 1:
            print(
                f"Actor update accumulation: {update_accumulation_steps} raw rollout steps per optimizer step; "
                f"total optimizer steps: {total_update_steps}"
            )

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_update_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_update_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False, default=str))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights(self.global_steps)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _validate_rollout_speed(
        self,
        *,
        max_samples_per_benchmark: Optional[int] = None,
        worker_count: Optional[int] = None,
        force_one_sample_per_server: bool = False,
        clear_kv_cache_per_benchmark: bool = False,
    ):
        if not hasattr(self, "rollout_speed_test_dataloaders"):
            raise RuntimeError("rollout_speed_test_dataloaders is not initialized.")

        metric_dict: dict[str, float] = {}
        for benchmark_name, _, _, dataloader in self.rollout_speed_test_dataloaders:
            if clear_kv_cache_per_benchmark:
                self.async_rollout_manager.clear_kv_cache()

            accept_length_fields: dict[str, list[np.ndarray]] = defaultdict(list)
            remaining_samples = max_samples_per_benchmark

            for test_data in dataloader:
                test_batch = DataProto.from_single_dict(test_data)
                if remaining_samples is not None:
                    if remaining_samples <= 0:
                        break
                    if len(test_batch) > remaining_samples:
                        test_batch = test_batch[:remaining_samples]
                    remaining_samples -= len(test_batch)

                if "uid" not in test_batch.non_tensor_batch:
                    test_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                    )

                test_batch = test_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
                )
                test_gen_batch = self._get_gen_batch(test_batch)
                test_gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                    "validate": True,
                    "global_steps": self.global_steps,
                }

                test_outputs = []
                if force_one_sample_per_server:
                    if worker_count is None:
                        raise ValueError("worker_count is required when force_one_sample_per_server=True.")
                    if len(test_gen_batch) % worker_count != 0:
                        raise ValueError(
                            "force_one_sample_per_server rollout speed eval requires each batch size to be divisible "
                            f"by worker_count, got batch_size={len(test_gen_batch)} worker_count={worker_count}."
                        )
                    for wave_start in range(0, len(test_gen_batch), worker_count):
                        wave_batch = test_gen_batch[wave_start : wave_start + worker_count]
                        wave_batch.meta_info = dict(test_gen_batch.meta_info)
                        test_output = self.async_rollout_manager.generate_sequences(
                            wave_batch,
                            worker_count=worker_count,
                            force_one_sample_per_server=True,
                        )
                        test_output.meta_info.pop("timing", None)
                        test_outputs.append(test_output)
                else:
                    size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
                    test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
                    test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                    test_output_gen_batch_padded.meta_info.pop("timing", None)
                    test_outputs.append(unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size))

                for test_output in test_outputs:
                    for key in (
                        "sglang_completion_tokens",
                        "sglang_spec_verify_ct",
                        "sglang_spec_accept_token_num",
                    ):
                        value = test_output.non_tensor_batch.get(key)
                        if value is not None:
                            accept_length_fields[key].append(value)

            merged_output = DataProto(
                batch=None,
                non_tensor_batch={
                    key: np.concatenate(values, axis=0)
                    for key, values in accept_length_fields.items()
                    if values
                },
                meta_info={},
            )
            accept_length = compute_sglang_spec_accept_length(merged_output)
            if accept_length is not None:
                metric_dict[f"test_rollout/{benchmark_name}/sglang_spec_accept_length"] = accept_length

        return metric_dict

    def _run_validation(self):
        if self.rollout_speed_test_only:
            return self._run_rollout_speed_validation()
        return self._validate()

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                distillation_config=self.config.get("distillation"),
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            # convert critic_cfg into TrainingWorkerConfig for the unified model engine worker
            from verl.workers.engine_workers import TrainingWorkerConfig

            orig_critic_cfg = critic_cfg
            engine_config: EngineConfig = orig_critic_cfg.engine
            engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu

            critic_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=orig_critic_cfg.model,
                engine_config=engine_config,
                optimizer_config=orig_critic_cfg.optim,
                checkpoint_config=orig_critic_cfg.checkpoint,
            )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/verl-project/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            # assign critic loss
            from functools import partial

            from verl.workers.utils.losses import value_loss

            value_loss_ = partial(value_loss, config=orig_critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        from verl.experimental.reward_loop import RewardLoopManager

        # initalize reward loop manager
        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # initialize teacher loop manager
        if self.use_teacher_policy:
            from verl.experimental.teacher_loop import MultiTeacherModelManager

            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
            self.teacher_model_manager = MultiTeacherModelManager(
                config=self.config,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool.
        #
        # Distillation-only training may explicitly disable task rewards; in that case
        # the rollout data may not contain reward_model/ground_truth fields at all, so
        # reward loop should be skipped entirely.
        use_task_rewards = True
        if self.use_teacher_policy:
            use_task_rewards = self.config.distillation.distillation_loss.use_task_rewards
        enable_agent_reward_loop = (
            not self.use_rm or self.config.reward.reward_model.enable_resource_pool
        ) and use_task_rewards

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        # To stream teacher computation with actor rollout, we instead pass the full manager so that the
        # teacher loop workers can sleep/wake together with rollout workers
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
            reward_loop_worker_handles=reward_loop_worker_handles,
            teacher_model_manager=self.teacher_model_manager,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        # Support custom CheckpointEngineManager via config
        checkpoint_manager_class_fqn = self.config.actor_rollout_ref.rollout.get("checkpoint_manager_class")
        if checkpoint_manager_class_fqn:
            CheckpointEngineManager = load_class_from_fqn(checkpoint_manager_class_fqn, "CheckpointEngineManager")
        else:
            from verl.checkpoint_engine import CheckpointEngineManager
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            steps_per_epoch = len(self.train_dataloader)
            at_epoch_boundary = steps_per_epoch > 0 and self.global_steps % steps_per_epoch == 0
            if at_epoch_boundary:
                print(
                    f"Skipping dataloader state restore: global_steps={self.global_steps} "
                    f"is at an epoch boundary (steps_per_epoch={steps_per_epoch}). "
                    f"The saved state marks the dataloader as exhausted. "
                    f"Next epoch will iterate from scratch."
                )
            else:
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        tu.assign_non_tensor(batch_td, compute_loss=False)
        output = self.critic_wg.infer_batch(batch_td)
        output = output.get()
        values = tu.get(output, "values")
        values = no_padding_2_padding(values, batch_td)
        values = tu.get_tensordict({"values": values.float()})
        values = DataProto.from_tensordict(values)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        # step 1: convert dataproto to tensordict.
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        metadata = {"calculate_entropy": False, "compute_loss": False}
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        tu.assign_non_tensor(batch_td, **metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
        # gather output
        log_probs = tu.get(output, "log_probs")
        # step 4. No padding to padding
        log_probs = no_padding_2_padding(log_probs, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
        ref_log_prob = DataProto.from_tensordict(ref_log_prob)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
        # step 1: convert dataproto to tensordict.
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        # gather output
        entropy = tu.get(output, "entropy")
        log_probs = tu.get(output, "log_probs")
        routed_experts = tu.get(output, "routed_experts")

        old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
        # step 4. No padding to padding
        entropy = no_padding_2_padding(entropy, batch_td)
        log_probs = no_padding_2_padding(log_probs, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        if routed_experts is not None:
            old_log_prob = tu.get_tensordict(
                {"old_log_probs": log_probs.float(), "entropys": entropy.float(), "routed_experts": routed_experts}
            )
        else:
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
        old_log_prob = DataProto.from_tensordict(old_log_prob)
        return old_log_prob, old_log_prob_mfu

    def _is_direct_distillation_only(self) -> bool:
        if not is_distillation_enabled(self.config.get("distillation")) or self.distillation_config is None:
            return False
        loss_config = self.distillation_config.distillation_loss
        return (not loss_config.use_policy_gradient) and (not loss_config.use_task_rewards)

    def _update_actor(self, batch: DataProto, effective_mini_batch_size: Optional[int] = None) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        distillation_use_topk = (
            self.distillation_config.distillation_loss.loss_settings.use_topk
            if is_distillation_enabled(self.config.get("distillation"))
            else False
        )
        rejected_draft_position_decay_enabled = False
        rejected_draft_position_decay = 1.0
        if is_distillation_enabled(self.config.get("distillation")) and self.distillation_config is not None:
            loss_config = self.distillation_config.distillation_loss
            rejected_draft_position_decay_enabled = bool(
                getattr(loss_config, "rejected_draft_position_decay_enabled", True)
            )
            rejected_draft_position_decay = float(getattr(loss_config, "rejected_draft_position_decay", 0.9))
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        if effective_mini_batch_size is not None:
            ppo_mini_batch_size = effective_mini_batch_size
        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        seed = self.config.actor_rollout_ref.actor.data_loader_seed
        shuffle = self.config.actor_rollout_ref.actor.shuffle
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            distillation_use_topk=distillation_use_topk,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
            opd_rejected_draft_position_decay_enabled=rejected_draft_position_decay_enabled,
            opd_rejected_draft_position_decay=rejected_draft_position_decay,
            compute_loss=True,
        )
        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        # modify key name
        actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        ppo_epochs = self.config.critic.ppo_epochs
        seed = self.config.critic.data_loader_seed
        shuffle = self.config.critic.shuffle
        tu.assign_non_tensor(
            batch_td,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
        )

        output = self.critic_wg.train_mini_batch(batch_td)
        output = output.get()
        output = tu.get(output, "metrics")
        output = rename_dict(output, "critic/")
        # modify key name
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        return critic_output

    @staticmethod
    def _add_timing(dst: dict[str, float], src: dict[str, float]) -> None:
        for key, value in src.items():
            if isinstance(value, (int, float)):
                dst[key] = dst.get(key, 0.0) + float(value)

    @staticmethod
    def _flatten_update_results(value):
        if value is None:
            return
        if isinstance(value, dict):
            yield value
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                yield from RayPPOTrainer._flatten_update_results(item)

    @classmethod
    def _extract_weight_sync_metrics(cls, update_result) -> dict[str, Any]:
        stats = [
            item
            for item in cls._flatten_update_results(update_result)
            if "opd/weight_sync/draft_tensor_count" in item
        ]
        if not stats:
            return {}

        first = stats[0]
        for item in stats[1:]:
            key = "opd/weight_sync/draft_tensor_count"
            if item.get(key) != first.get(key):
                print(
                    "Warning: inconsistent DFLASH draft weight sync tensor_count across workers; "
                    "using first worker stats for logging."
                )
                break
        return {"opd/weight_sync/draft_tensor_count": first["opd/weight_sync/draft_tensor_count"]}

    @staticmethod
    def _split_string_metrics(metrics: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        string_metrics = {key: value for key, value in metrics.items() if isinstance(value, str)}
        numeric_metrics = {key: value for key, value in metrics.items() if not isinstance(value, str)}
        return numeric_metrics, string_metrics

    def _initial_val_repeat_times(self) -> int:
        raw_repeat_times = self.config.trainer.get("initial_val_repeat_times", 1)
        try:
            repeat_times = int(raw_repeat_times)
        except (TypeError, ValueError):
            raise ValueError(f"trainer.initial_val_repeat_times must be an integer, got {raw_repeat_times!r}") from None
        return max(repeat_times, 1)

    def _rollout_speed_test_max_samples_per_benchmark(self) -> Optional[int]:
        raw_max_samples = self.config.trainer.get("rollout_speed_test_max_samples_per_benchmark", None)
        if raw_max_samples is None:
            raw_max_samples = self.config.trainer.get("initial_val_max_samples_per_benchmark", None)
        if raw_max_samples is None:
            return None
        try:
            max_samples = int(raw_max_samples)
        except (TypeError, ValueError):
            raise ValueError(
                "trainer.rollout_speed_test_max_samples_per_benchmark must be an integer or null, "
                f"got {raw_max_samples!r}"
            ) from None
        return max_samples if max_samples > 0 else None

    def _rollout_speed_test_worker_count(self) -> Optional[int]:
        raw_worker_count = self.config.trainer.get("rollout_speed_test_worker_count", None)
        if raw_worker_count is None:
            raw_worker_count = self.config.trainer.get("initial_val_worker_count", None)
        if raw_worker_count is None:
            return None
        try:
            worker_count = int(raw_worker_count)
        except (TypeError, ValueError):
            raise ValueError(
                f"trainer.rollout_speed_test_worker_count must be an integer or null, got {raw_worker_count!r}"
            ) from None
        return worker_count if worker_count > 0 else None

    def _rollout_speed_test_kwargs(self) -> dict[str, Any]:
        max_samples_per_benchmark = self._rollout_speed_test_max_samples_per_benchmark()
        worker_count = self._rollout_speed_test_worker_count()
        force_one_sample_per_server = max_samples_per_benchmark is not None and worker_count is not None
        if (
            force_one_sample_per_server
            and (max_samples_per_benchmark < worker_count or max_samples_per_benchmark % worker_count != 0)
        ):
            raise ValueError(
                "deterministic rollout speed eval requires "
                "trainer.rollout_speed_test_max_samples_per_benchmark to be a positive multiple of "
                "trainer.rollout_speed_test_worker_count, "
                f"got {max_samples_per_benchmark=} {worker_count=}."
            )
        rollout_n = int(self.config.actor_rollout_ref.rollout.val_kwargs.n)
        if force_one_sample_per_server and rollout_n != 1:
            raise ValueError(f"deterministic rollout speed eval requires rollout.val_kwargs.n=1, got {rollout_n}.")
        clear_kv_cache = self.config.trainer.get("rollout_speed_test_clear_kv_cache", None)
        if clear_kv_cache is None:
            clear_kv_cache = self.config.trainer.get("initial_val_clear_kv_cache", False)
        return {
            "max_samples_per_benchmark": max_samples_per_benchmark,
            "worker_count": worker_count,
            "force_one_sample_per_server": force_one_sample_per_server,
            "clear_kv_cache_per_benchmark": bool(clear_kv_cache) and force_one_sample_per_server,
        }

    def _run_rollout_speed_validation(self) -> dict[str, Any]:
        return self._validate_rollout_speed(**self._rollout_speed_test_kwargs())

    def _initial_eval_log_path(self) -> Path:
        experiment_path = _safe_experiment_path(self.config.trainer.get("experiment_name", None))
        return _repo_root() / "logs" / experiment_path / "eval.log"

    @staticmethod
    def _filter_accept_metrics(metrics: dict[str, Any]) -> dict[str, float]:
        accept_metrics = {}
        for key, value in metrics.items():
            if "accept" not in key:
                continue
            try:
                accept_metrics[key] = float(value)
            except (TypeError, ValueError):
                continue
        return accept_metrics

    @staticmethod
    def _config_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        return value

    def _save_by_test_metric_enabled(self) -> bool:
        return self._config_bool(self.config.trainer.get("save_by_test_metric_enabled", False))

    def _should_save_by_test_metric(self, val_metrics: dict[str, Any]) -> tuple[bool, dict[str, float]]:
        metric_key = str(self.config.trainer.get("save_by_test_metric_key", "") or "").strip()
        if not metric_key:
            raise ValueError("trainer.save_by_test_metric_key must be set when save_by_test_metric_enabled=True.")

        raw_threshold = self.config.trainer.get("save_by_test_metric_threshold", None)
        if raw_threshold is None:
            raise ValueError("trainer.save_by_test_metric_threshold must be set when save_by_test_metric_enabled=True.")
        threshold = float(raw_threshold)

        metric_value = self._optional_float(val_metrics.get(metric_key))
        decision_metrics = {
            "checkpoint/save_by_test_metric_threshold": threshold,
            "checkpoint/save_by_test_metric_triggered": 0.0,
        }
        if metric_value is None:
            available_keys = ", ".join(sorted(str(key) for key in val_metrics))
            print(
                "Skip metric-gated checkpoint: "
                f"{metric_key} is missing or non-finite. Available validation keys: {available_keys}"
            )
            return False, decision_metrics

        decision_metrics["checkpoint/save_by_test_metric_value"] = metric_value
        should_save = metric_value > threshold
        decision_metrics["checkpoint/save_by_test_metric_triggered"] = float(should_save)
        if should_save:
            print(f"Saving checkpoint: {metric_key}={metric_value:.6g} > {threshold:.6g}.")
        else:
            print(f"Skip checkpoint: {metric_key}={metric_value:.6g} <= {threshold:.6g}.")
        return should_save, decision_metrics

    @staticmethod
    def _format_initial_eval_line(repeat_idx: int, repeat_times: int, metrics: dict[str, float]) -> str:
        metric_text = " - ".join(f"{key}:{value}" for key, value in sorted(metrics.items()))
        return f"initial_eval_repeat:{repeat_idx}/{repeat_times} - {metric_text}"

    def _write_initial_eval_log(self, lines: list[str]) -> None:
        log_path = self._initial_eval_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as fout:
            for line in lines:
                fout.write(line + "\n")

    def _run_initial_validation_repeats(self) -> dict[str, Any]:
        repeat_times = self._initial_val_repeat_times()
        clear_kv_cache = bool(self.config.trainer.get("initial_val_clear_kv_cache", False))
        eval_log_lines = []
        last_val_metrics = None
        for repeat_idx in range(1, repeat_times + 1):
            if self.rollout_speed_test_only:
                val_metrics = self._run_rollout_speed_validation()
            else:
                if clear_kv_cache:
                    self.async_rollout_manager.clear_kv_cache()
                val_metrics = self._run_validation()
            assert val_metrics, f"{val_metrics=}"
            last_val_metrics = val_metrics

            accept_metrics = self._filter_accept_metrics(val_metrics)
            eval_line = self._format_initial_eval_line(repeat_idx, repeat_times, accept_metrics)
            eval_log_lines.append(eval_line)
            print(eval_line)
            self._write_initial_eval_log(eval_log_lines)

        assert last_val_metrics is not None
        return last_val_metrics

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)

        # In rollout-speed-test mode, run speed validation only on test_freq boundaries.
        if self.config.trainer.get("val_before_train", True) and not self.rollout_speed_test_only:
            val_metrics = self._run_initial_validation_repeats()
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.skip.get("enable", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        update_accumulation_steps = max(int(self.config.trainer.get("update_accumulation_steps", 1)), 1)
        update_step = self.global_steps // update_accumulation_steps
        pending_batches: list[DataProto] = []
        pending_gen_outputs: list[DataProto] = []
        pending_timing_raw: dict[str, float] = {}

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}
                completed_batch = None
                completed_gen_output = None
                completed_raw_steps = 0
                reward_extra_infos_dict = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch.non_tensor_batch["experiment_name"] = np.array(
                    [self.config.trainer.experiment_name] * len(gen_batch), dtype=object
                )
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate one raw rollout batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.async_rollout_manager.start_profile()
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        if curr_step_profile:
                            self.async_rollout_manager.stop_profile()

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                batch_reward = self._compute_reward_colocate(batch)
                                batch = batch.union(batch_reward)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = batch.batch["rm_scores"].sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # global_steps is only needed by rollout tracing. Accumulated batches must have
                    # consistent non-metric meta_info for DataProto.concat.
                    batch.meta_info.pop("global_steps", None)
                    gen_batch_output.meta_info.pop("global_steps", None)
                    pending_batches.append(batch)
                    pending_gen_outputs.append(gen_batch_output)
                    should_update = len(pending_batches) >= update_accumulation_steps or is_last_step

                    if should_update:
                        # Only sleep rollout replicas when we are about to enter the trainer update/sync phase.
                        # Sleeping after every raw rollout would leave SGLang asleep during accumulation-only steps.
                        self.checkpoint_manager.sleep_replicas()
                        batch = pending_batches[0] if len(pending_batches) == 1 else DataProto.concat(pending_batches)
                        completed_gen_output = (
                            pending_gen_outputs[0]
                            if len(pending_gen_outputs) == 1
                            else DataProto.concat(pending_gen_outputs)
                        )
                        completed_raw_steps = len(pending_batches)

                        # Balance the number of valid tokens across DP ranks.
                        # NOTE: This usually changes the order of data in the `batch`,
                        # which won't affect the advantage calculation (since it's based on uid),
                        # but might affect the loss calculation (due to the change of mini-batching).
                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)

                        # compute global_valid tokens
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                        # get images_seqlens
                        images_seqlens_all = []
                        for multi_modal_input in batch.non_tensor_batch.get("multi_modal_inputs", []):
                            if "image_grid_thw" not in multi_modal_input.keys():
                                continue
                            images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                        batch.meta_info["images_seqlens"] = images_seqlens_all
                        with marked_timer("reward", timing_raw, color="yellow"):
                            # compute reward model score
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                batch_reward = self._compute_reward_colocate(batch)
                                batch = batch.union(batch_reward)

                            # extract reward_tensor and reward_extra_infos_dict for training
                            reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                        # Operating Mode Selection:
                        # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                        # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                        #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get(
                            "bypass_mode", False
                        )
                        direct_distillation_only = self._is_direct_distillation_only()
                        if direct_distillation_only:
                            pass
                        elif bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                            from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                            apply_bypass_mode(
                                batch=batch,
                                rollout_corr_config=rollout_corr_config,
                                policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                            )
                        else:  # Recompute old_log_probs
                            with marked_timer("old_log_prob", timing_raw, color="blue"):
                                old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                                entropys = old_log_prob.batch["entropys"]
                                response_masks = batch.batch["response_mask"]
                                actor_config = self.config.actor_rollout_ref.actor
                                entropy_agg = agg_loss(
                                    loss_mat=entropys,
                                    loss_mask=response_masks,
                                    loss_agg_mode=actor_config.loss_agg_mode,
                                    loss_scale_factor=actor_config.loss_scale_factor,
                                )
                                old_log_prob_metrics = {
                                    "actor/entropy": entropy_agg.detach().item(),
                                    "perf/mfu/actor_infer": old_log_prob_mfu,
                                }
                                metrics.update(old_log_prob_metrics)
                                old_log_prob.batch.pop("entropys")
                                if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                    raise ValueError(
                                        "Detected conflicting router replay configuration: "
                                        "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                        "cannot be enabled simultaneously. "
                                        "The enable_rollout_routing_replay option is only used in R3 mode; "
                                        "it should not be set when using R2 mode."
                                    )
                                batch = batch.union(old_log_prob)
                                if "rollout_log_probs" in batch.batch.keys():
                                    # TODO: we may want to add diff of probs too.
                                    from verl.utils.debug.metrics import calculate_debug_metrics

                                    metrics.update(calculate_debug_metrics(batch))

                        if not direct_distillation_only:
                            assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                                ref_log_prob = self._compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute values
                        if self.use_critic:
                            with marked_timer("values", timing_raw, color="cyan"):
                                values = self._compute_values(batch)
                                batch = batch.union(values)

                        with marked_timer("adv", timing_raw, color="brown"):
                            # we combine with rule-based rm
                            batch.batch["token_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update(
                                    {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                                )

                            if direct_distillation_only:
                                zero_token_values = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
                                batch.batch["token_level_rewards"] = zero_token_values
                                batch.batch["advantages"] = zero_token_values.clone()
                                batch.batch["returns"] = zero_token_values.clone()
                            # compute rewards. apply_kl_penalty if available
                            elif self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            # Compute rollout correction: IS weights, rejection sampling, and metrics
                            # Only runs in decoupled mode (computes once per batch using stable π_old)
                            # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                            if (
                                rollout_corr_config is not None
                                and "rollout_log_probs" in batch.batch
                                and not bypass_recomputing_logprobs  # Only in decoupled mode
                                and not direct_distillation_only
                            ):
                                from verl.trainer.ppo.rollout_corr_helper import (
                                    compute_rollout_correction_and_add_to_batch,
                                )

                                # Compute IS weights, apply rejection sampling, compute metrics
                                batch, is_metrics = compute_rollout_correction_and_add_to_batch(
                                    batch, rollout_corr_config
                                )
                                # IS and off-policy metrics already have rollout_corr/ prefix
                                metrics.update(is_metrics)

                            if not direct_distillation_only:
                                # compute advantages, executed on the driver process
                                norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                    "norm_adv_by_std_in_grpo", True
                                )  # GRPO adv normalization factor

                                batch = compute_advantage(
                                    batch,
                                    adv_estimator=self.config.algorithm.adv_estimator,
                                    gamma=self.config.algorithm.gamma,
                                    lam=self.config.algorithm.lam,
                                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                    config=self.config.algorithm,
                                )

                        # update critic
                        if self.use_critic:
                            with marked_timer("update_critic", timing_raw, color="pink"):
                                critic_output = self._update_critic(batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update(critic_output_metrics)

                        checkpoint_saved_this_step = False
                        save_by_test_metric_enabled = self._save_by_test_metric_enabled()

                        # implement critic warmup
                        if self.config.trainer.critic_warmup > self.global_steps:
                            # Still in critic warmup, only update weights to wake up rollout replicas.
                            weight_sync_result = self.checkpoint_manager.update_weights(self.global_steps)
                            metrics.update(self._extract_weight_sync_metrics(weight_sync_result))
                        else:
                            # update actor once for the whole accumulated rollout batch
                            with marked_timer("update_actor", timing_raw, color="red"):
                                actor_output = self._update_actor(batch, effective_mini_batch_size=len(batch))

                            # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                            esi_close_to_expiration = should_save_ckpt_esi(
                                max_steps_duration=self.max_steps_duration,
                                redundant_time=self.config.trainer.esi_redundant_time,
                            )
                            if save_by_test_metric_enabled:
                                if esi_close_to_expiration:
                                    print("Force saving checkpoint: ESI instance expiration approaching.")
                                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                                        self._save_checkpoint()
                                    checkpoint_saved_this_step = True
                            else:
                                save_freq = int(self.config.trainer.get("save_freq", -1))
                                save_start_step = max(int(self.config.trainer.get("save_start_step", 0)), 0)
                                periodic_save = (
                                    save_freq > 0
                                    and self.global_steps >= save_start_step
                                    and (self.global_steps - save_start_step) % save_freq == 0
                                )
                                # Check if the conditions for saving a checkpoint are met.
                                # The conditions include a mandatory condition (1) and
                                # one of the following optional conditions (2/3/4):
                                # 1. The save frequency is set to a positive value.
                                # 2. It's the last training step.
                                # 3. The current raw step is at/after save_start_step and matches save_freq.
                                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                                if save_freq > 0 and (is_last_step or periodic_save or esi_close_to_expiration):
                                    if esi_close_to_expiration:
                                        print("Force saving checkpoint: ESI instance expiration approaching.")
                                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                                        self._save_checkpoint()
                                    checkpoint_saved_this_step = True

                            # update weights from trainer to rollout once per accumulated update
                            with marked_timer("update_weights", timing_raw, color="red"):
                                weight_sync_result = self.checkpoint_manager.update_weights(self.global_steps)
                            metrics.update(self._extract_weight_sync_metrics(weight_sync_result))

                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                        # Log rollout generations if enabled
                        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                        if rollout_data_dir:
                            self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                        # validate
                        if self.config.trainer.test_freq > 0 and (
                            is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                        ):
                            with marked_timer("testing", timing_raw, color="green"):
                                val_metrics: dict = self._run_validation()
                                if is_last_step:
                                    last_val_metrics = val_metrics
                            metrics.update(val_metrics)
                            if save_by_test_metric_enabled:
                                should_save, checkpoint_metrics = self._should_save_by_test_metric(val_metrics)
                                metrics.update(checkpoint_metrics)
                                if should_save and not checkpoint_saved_this_step:
                                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                                        self._save_checkpoint()
                                    checkpoint_saved_this_step = True

                        completed_batch = batch

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                self._add_timing(pending_timing_raw, timing_raw)

                progress_bar.update(1)

                if completed_batch is not None:
                    update_step += 1
                    metrics.update(compute_sglang_rollout_metrics(completed_gen_output, pending_timing_raw))
                    # training metrics
                    metrics.update(
                        {
                            "training/global_step": self.global_steps,
                            "training/update_step": update_step,
                            "training/epoch": epoch,
                            "training/accumulated_raw_steps": completed_raw_steps,
                        }
                    )
                    # collect metrics
                    metrics.update(compute_data_metrics(batch=completed_batch, use_critic=self.use_critic))
                    # GDPO per-component reward metrics
                    gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                    if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                        for key in gdpo_reward_keys:
                            if key in completed_batch.non_tensor_batch:
                                vals = np.asarray(completed_batch.non_tensor_batch[key], dtype=np.float32)
                                metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                                metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                                metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                                metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                    metrics.update(compute_timing_metrics(batch=completed_batch, timing_raw=pending_timing_raw))
                    # TODO: implement actual tflpo and theoretical tflpo
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    metrics.update(
                        compute_throughout_metrics(batch=completed_batch, timing_raw=pending_timing_raw, n_gpus=n_gpus)
                    )
                    # compute variance proxy metrics
                    gradient_norm = metrics.get("actor/grad_norm", None)
                    metrics.update(compute_variance_proxy_metrics(batch=completed_batch, gradient_norm=gradient_norm))
                    # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                    # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                    if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                        self.train_dataloader.sampler.update(batch=completed_batch)

                    # TODO: make a canonical logger that supports various backend.
                    # TensorBoard/MLflow scalar loggers cannot accept strings, so keep tensor-name
                    # diagnostics on text-capable backends.
                    numeric_metrics, string_metrics = self._split_string_metrics(metrics)
                    logger.log(data=numeric_metrics, step=self.global_steps)
                    if string_metrics:
                        logger.log(
                            data=string_metrics,
                            step=self.global_steps,
                            backend=["wandb", "vemlp_wandb", "file"],
                        )

                    pending_batches = []
                    pending_gen_outputs = []
                    pending_timing_raw = {}

                    if (
                        hasattr(self.config.actor_rollout_ref.actor, "profiler")
                        and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                    ):
                        self.actor_rollout_wg.dump_memory_snapshot(
                            tag=f"post_update_step{self.global_steps + 1}", sub_dir=f"step{self.global_steps + 1}"
                        )

                self.global_steps += 1

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
