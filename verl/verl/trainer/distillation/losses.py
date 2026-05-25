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

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

_AGG_LOSS_GLOBAL_BATCH_INFO_KEYS = {"dp_size", "batch_num_tokens", "global_batch_size", "loss_scale_factor"}
EAGLE3_NATIVE_TARGET_DISTRIBUTION_LOSS = "eagle3_native_target_distribution"


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator]) != 1:
            raise ValueError(
                f"Expected only one of use_estimator, use_topk, but got {self.use_estimator=}, {self.use_topk=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def _flatten_nested_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if getattr(tensor, "is_nested", False):
        return tensor.values()
    return tensor


def _padded_response_mask(response_mask: torch.Tensor) -> torch.Tensor:
    if response_mask.is_nested:
        return response_mask.bool().to_padded_tensor(False)
    return response_mask.bool()


def get_effective_distillation_response_mask(data: TensorDict, model_output: Optional[dict] = None) -> torch.Tensor:
    response_mask = _padded_response_mask(data["response_mask"])
    if model_output is None or "opd_loss_mask" not in model_output:
        return response_mask

    opd_loss_mask = no_padding_2_padding(model_output["opd_loss_mask"], data).bool()
    if opd_loss_mask.shape != response_mask.shape:
        raise ValueError(
            f"OPD loss mask shape {tuple(opd_loss_mask.shape)} does not match "
            f"response mask shape {tuple(response_mask.shape)}."
        )
    return response_mask & opd_loss_mask


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    distillation_losses_response = distillation_losses[response_mask.bool()]
    if distillation_losses_response.numel() == 0:
        zero = distillation_losses.detach().new_tensor(0.0)
        return {
            "distillation/loss_min": Metric(AggregationType.MIN, zero),
            "distillation/loss_max": Metric(AggregationType.MAX, zero),
        }
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
    }


def compute_opd_distillation_metrics(
    *,
    model_output: dict,
    response_mask: torch.Tensor,
    effective_response_mask: torch.Tensor,
) -> dict[str, Metric]:
    metrics: dict[str, Metric] = {}
    if "opd_loss_mask" not in model_output:
        return metrics
    response_token_count = response_mask.sum().to(torch.float32)
    effective_token_count = effective_response_mask.sum().to(torch.float32)
    metrics["distillation/opd_effective_token_count"] = Metric(AggregationType.SUM, effective_token_count)
    if response_token_count.item() > 0:
        metrics["distillation/opd_effective_token_ratio"] = Metric(
            AggregationType.MEAN, effective_token_count / response_token_count
        )
    else:
        metrics["distillation/opd_effective_token_ratio"] = Metric(
            AggregationType.MEAN, response_token_count.new_tensor(0.0)
        )

    response_sample_mask = response_mask.sum(dim=-1) > 0
    skipped_sample_mask = response_sample_mask & (effective_response_mask.sum(dim=-1) == 0)
    if response_sample_mask.any():
        skipped_ratio = skipped_sample_mask.float().sum() / response_sample_mask.float().sum()
    else:
        skipped_ratio = response_token_count.new_tensor(0.0)
    metrics["distillation/opd_skipped_sample_ratio"] = Metric(AggregationType.MEAN, skipped_ratio)

    sum_metric_keys = {
        "opd_valid_anchor_count": "distillation/opd_valid_anchor_count",
        "opd_skipped_sample_count": "distillation/opd_skipped_sample_count",
        "opd_empty_reject_sample_count": "distillation/opd_empty_reject_sample_count",
        "opd_total_reject_count": "distillation/opd_total_reject_count",
        "opd_sample_count": "distillation/opd_sample_count",
        "opd_target_token_count": "distillation/opd_target_token_count",
        "opd_rejected_draft_token_count": "distillation/opd_rejected_draft_token_count",
    }
    for source_key, metric_key in sum_metric_keys.items():
        value = model_output.get(source_key)
        if value is not None:
            metrics[metric_key] = Metric(AggregationType.SUM, value)

    mean_metric_keys = {
        "opd_attention_impl_id": "distillation/opd_attention_impl_id",
        "opd_profile_teacher_forward_ms": "distillation/opd_profile_teacher_forward_ms",
        "opd_profile_draft_forward_ms": "distillation/opd_profile_draft_forward_ms",
        "opd_profile_lm_head_ms": "distillation/opd_profile_lm_head_ms",
        "opd_profile_total_forward_ms": "distillation/opd_profile_total_forward_ms",
        "eagle3_response_loss_mode_id": "eagle3/response_loss_mode_id",
    }
    for source_key, metric_key in mean_metric_keys.items():
        value = model_output.get(source_key)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            metric_value = value.detach().to(dtype=torch.float32)
        else:
            metric_value = effective_token_count.new_tensor(float(value), dtype=torch.float32)
        metrics[metric_key] = Metric(AggregationType.MEAN, metric_value)
    eagle3_total = model_output.get("eagle3_target_argmax_total_count")
    eagle3_supported = model_output.get("eagle3_target_argmax_supported_count")
    eagle3_top1 = model_output.get("eagle3_draft_target_top1_correct_count")
    if eagle3_total is not None and eagle3_supported is not None:
        eagle3_total = _flatten_nested_tensor(eagle3_total).detach().to(dtype=torch.float32)
        eagle3_supported = _flatten_nested_tensor(eagle3_supported).detach().to(dtype=torch.float32)
        zero = effective_token_count.new_tensor(0.0)
        supported_ratio = torch.where(eagle3_total > 0, eagle3_supported / eagle3_total.clamp_min(1.0), zero)
        metrics["eagle3/target_argmax_supported_ratio"] = Metric(AggregationType.MEAN, supported_ratio)
    if eagle3_supported is not None and eagle3_top1 is not None:
        eagle3_supported = _flatten_nested_tensor(eagle3_supported).detach().to(dtype=torch.float32)
        eagle3_top1 = _flatten_nested_tensor(eagle3_top1).detach().to(dtype=torch.float32)
        zero = effective_token_count.new_tensor(0.0)
        top1_acc = torch.where(eagle3_supported > 0, eagle3_top1 / eagle3_supported.clamp_min(1.0), zero)
        metrics["eagle3/draft_target_top1_acc"] = Metric(AggregationType.MEAN, top1_acc)
    return metrics


def get_rejected_draft_distillation_stream(
    model_output: dict,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    mask = model_output.get("opd_rejected_draft_loss_mask")
    if mask is None:
        return None

    student_log_probs = model_output.get("opd_rejected_draft_student_log_probs")
    teacher_log_probs = model_output.get("opd_rejected_draft_teacher_log_probs")
    if student_log_probs is None or teacher_log_probs is None:
        raise RuntimeError(
            "Rejected DFLASH draft loss mask is present, but student or teacher logprobs are missing."
        )

    mask = _flatten_nested_tensor(mask).bool()
    student_log_probs = _flatten_nested_tensor(student_log_probs)
    teacher_log_probs = _flatten_nested_tensor(teacher_log_probs)
    if student_log_probs.shape != teacher_log_probs.shape or student_log_probs.shape != mask.shape:
        raise ValueError(
            "Rejected DFLASH draft stream shape mismatch: "
            f"student={tuple(student_log_probs.shape)}, teacher={tuple(teacher_log_probs.shape)}, "
            f"mask={tuple(mask.shape)}."
        )
    if not bool(mask.any()):
        return None
    return student_log_probs, teacher_log_probs, mask


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    match config.strategy:
        # VeOmni uses FSDP2 internally, so its loss computation is identical to FSDP.
        case "fsdp" | "veomni":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
        case "megatron":
            import verl.trainer.distillation.megatron.losses as megatron_losses

            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    outputs = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"],
        teacher_topk_ids=data["teacher_ids"],
        config=distillation_config,
        data_format=data_format,
    )

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # Called as logits processor
    if student_logits is not None:
        if model_output is not None and get_rejected_draft_distillation_stream(model_output) is not None:
            raise NotImplementedError(
                "DFLASH rejected draft token training is not supported with forward_kl_topk."
            )
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    # Called as final policy loss
    distillation_loss_config = distillation_config.distillation_loss
    config.global_batch_info["dp_size"] = tu.get_non_tensor_data(data=data, key="dp_size", default=1)
    config.global_batch_info["batch_num_tokens"] = tu.get_non_tensor_data(
        data=data, key="batch_num_tokens", default=None
    )
    config.global_batch_info["global_batch_size"] = tu.get_non_tensor_data(
        data=data, key="global_batch_size", default=None
    )
    config.global_batch_info["loss_scale_factor"] = getattr(config, "loss_scale_factor", None)
    config.global_batch_info["opd_rejected_draft_batch_num_tokens"] = tu.get_non_tensor_data(
        data=data, key="opd_rejected_draft_batch_num_tokens", default=None
    )
    config.global_batch_info["opd_rejected_draft_batch_effective_num_tokens"] = tu.get_non_tensor_data(
        data=data, key="opd_rejected_draft_batch_effective_num_tokens", default=None
    )
    distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data, dp_group=dp_group)
    if not distillation_loss_config.use_task_rewards:
        distill_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)
        return distill_loss, distill_metrics

    policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)

    # Combine distillation with policy loss
    policy_metrics.update(distill_metrics)
    distillation_loss_coef = (
        distillation_loss_config.distillation_loss_coef if distillation_loss_config.use_task_rewards else 1.0
    )
    policy_loss += distill_loss * distillation_loss_coef
    policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)

    return policy_loss, policy_metrics


def _scalar_like(value: Any, like: torch.Tensor) -> torch.Tensor:
    value = tu.unwrap_non_tensor_data(value)
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=like.device, dtype=like.dtype)
    return like.new_tensor(float(value))


def _compute_loss_clamp_fraction(losses: torch.Tensor, mask: torch.Tensor, loss_max_clamp: float) -> torch.Tensor:
    valid_losses = losses[mask.bool()]
    if valid_losses.numel() == 0:
        return losses.detach().new_tensor(0.0)
    return (valid_losses.detach().abs() >= loss_max_clamp).float().mean()


def _global_sum(value: torch.Tensor, dp_group=None) -> torch.Tensor:
    if dp_group is None or not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return value
    global_value = value.detach().clone()
    torch.distributed.all_reduce(global_value, op=torch.distributed.ReduceOp.SUM, group=dp_group)
    return global_value


def _loss_weight(loss_config: DistillationLossConfig, name: str, default: float) -> float:
    weight = float(getattr(loss_config, name, default))
    if weight < 0:
        raise ValueError(f"{name} must be non-negative, got {weight}.")
    return weight


def _sampled_kl_loss_mode(loss_config: DistillationLossConfig) -> str:
    loss_mode = str(getattr(loss_config, "loss_mode", "k3"))
    if loss_mode == EAGLE3_NATIVE_TARGET_DISTRIBUTION_LOSS:
        return "k3"
    return loss_mode


def _local_bernoulli_forward_kl(
    *,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    loss_config: DistillationLossConfig,
) -> torch.Tensor:
    student_log_probs = student_log_probs.float()
    teacher_log_probs = teacher_log_probs.float()
    log_prob_min_clamp = getattr(loss_config, "log_prob_min_clamp", None)
    min_log_prob = -80.0 if log_prob_min_clamp is None else float(log_prob_min_clamp)
    student_log_probs = student_log_probs.clamp_min(min_log_prob)
    teacher_log_probs = teacher_log_probs.clamp_min(min_log_prob)

    eps = torch.finfo(student_log_probs.dtype).eps
    max_log_prob = math.log1p(-eps)
    student_log_probs = student_log_probs.clamp(max=max_log_prob)
    teacher_log_probs = teacher_log_probs.clamp(max=max_log_prob)
    student_probs = student_log_probs.exp()
    teacher_probs = teacher_log_probs.exp()
    return teacher_probs * (teacher_log_probs - student_log_probs) + (1.0 - teacher_probs) * (
        torch.log1p(-teacher_probs) - torch.log1p(-student_probs)
    )


def _valid_mean(losses: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_losses = losses[mask.bool()]
    if valid_losses.numel() == 0:
        return losses.detach().new_tensor(0.0)
    return valid_losses.mean()


def _combine_sampled_reverse_forward_losses(
    *,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    loss_config: DistillationLossConfig,
    mask: torch.Tensor,
    stream_name: str,
    force_reverse_kl: bool = False,
) -> tuple[torch.Tensor, dict[str, Metric]]:
    reverse_weight = _loss_weight(loss_config, "reverse_kl_weight", 1.0)
    forward_weight = _loss_weight(loss_config, "forward_kl_weight", 0.0)
    metric_prefix = "distillation/" if stream_name == "response" else f"distillation/{stream_name}_"
    sampled_loss_mode = _sampled_kl_loss_mode(loss_config)
    if force_reverse_kl:
        reverse_losses = kl_penalty(
            logprob=student_log_probs,
            ref_logprob=teacher_log_probs,
            kl_penalty=sampled_loss_mode,
        )
        return reverse_losses, {
            f"{metric_prefix}reverse_kl_loss": Metric(AggregationType.MEAN, _valid_mean(reverse_losses, mask))
        }
    if reverse_weight == 0 and forward_weight == 0:
        raise ValueError("At least one of reverse_kl_weight or forward_kl_weight must be positive.")
    if forward_weight == 0 and reverse_weight == 1.0:
        return (
            kl_penalty(
                logprob=student_log_probs,
                ref_logprob=teacher_log_probs,
                kl_penalty=sampled_loss_mode,
            ),
            {},
        )

    total_losses: Optional[torch.Tensor] = None
    metrics: dict[str, Metric] = {}
    reverse_losses = None
    forward_losses = None
    if reverse_weight > 0:
        reverse_losses = kl_penalty(
            logprob=student_log_probs,
            ref_logprob=teacher_log_probs,
            kl_penalty=sampled_loss_mode,
        )
        total_losses = reverse_losses * reverse_weight
    if forward_weight > 0:
        forward_losses = _local_bernoulli_forward_kl(
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            loss_config=loss_config,
        )
        if total_losses is None:
            total_losses = forward_losses * forward_weight
        else:
            total_losses = total_losses + forward_losses * forward_weight

    assert total_losses is not None
    if forward_weight > 0:
        if reverse_losses is not None:
            metrics[f"{metric_prefix}reverse_kl_loss"] = Metric(
                AggregationType.MEAN, _valid_mean(reverse_losses, mask)
            )
        if forward_losses is not None:
            metrics[f"{metric_prefix}forward_kl_loss"] = Metric(
                AggregationType.MEAN, _valid_mean(forward_losses, mask)
            )
    return total_losses, metrics


def _build_rejected_draft_position_weights(
    *,
    model_output: dict,
    rejected_draft_mask: torch.Tensor,
    loss_config: DistillationLossConfig,
) -> tuple[torch.Tensor, bool]:
    enabled = bool(getattr(loss_config, "rejected_draft_position_decay_enabled", True))
    if not enabled:
        return rejected_draft_mask.to(dtype=torch.float32), False

    decay = float(getattr(loss_config, "rejected_draft_position_decay", 0.9))
    if decay <= 0.0 or decay > 1.0:
        raise ValueError(f"rejected_draft_position_decay must be in (0, 1], got {decay}.")

    offsets = model_output.get("opd_rejected_draft_offsets")
    if offsets is None:
        return rejected_draft_mask.to(dtype=torch.float32), False

    offsets = _flatten_nested_tensor(offsets).to(device=rejected_draft_mask.device)
    if offsets.shape != rejected_draft_mask.shape:
        raise RuntimeError(
            "Rejected DFLASH draft offsets must match the rejected draft mask shape, got "
            f"offsets={tuple(offsets.shape)}, mask={tuple(rejected_draft_mask.shape)}."
        )
    exponents = (offsets.to(dtype=torch.float32) - 1.0).clamp_min(0.0)
    weights = torch.pow(offsets.new_tensor(decay, dtype=torch.float32), exponents)
    return weights * rejected_draft_mask.to(dtype=torch.float32), True


def _agg_loss_global_batch_info(config: ActorConfig) -> dict[str, Any]:
    global_batch_info = getattr(config, "global_batch_info", {}) or {}
    return {key: value for key, value in global_batch_info.items() if key in _AGG_LOSS_GLOBAL_BATCH_INFO_KEYS}


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
    dp_group=None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    rejected_draft_stream = get_rejected_draft_distillation_stream(model_output)
    if rejected_draft_stream is not None and loss_config.loss_mode == "forward_kl_topk":
        raise NotImplementedError("DFLASH rejected draft token training is not supported with forward_kl_topk.")
    if rejected_draft_stream is not None and loss_config.use_policy_gradient:
        raise NotImplementedError(
            "DFLASH rejected draft token training is not supported when use_policy_gradient=True."
        )
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = _padded_response_mask(data["response_mask"])
    effective_response_mask = get_effective_distillation_response_mask(data=data, model_output=model_output)
    loss_agg_mode = config.loss_agg_mode

    distillation_metrics.update(
        compute_distillation_loss_range(
            distillation_losses=distillation_losses, response_mask=effective_response_mask
        )
    )
    distillation_metrics.update(
        compute_opd_distillation_metrics(
            model_output=model_output,
            response_mask=response_mask,
            effective_response_mask=effective_response_mask,
        )
    )
    if loss_config.loss_max_clamp is not None:
        clamp_fraction = _compute_loss_clamp_fraction(
            distillation_losses, effective_response_mask, loss_config.loss_max_clamp
        )
        distillation_metrics["distillation/loss_clamp_fraction"] = Metric(
            AggregationType.MEAN, clamp_fraction
        )
        if rejected_draft_stream is not None:
            distillation_metrics["distillation/non_rejected_token_loss_clamp_fraction"] = Metric(
                AggregationType.MEAN, clamp_fraction
            )
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    if rejected_draft_stream is not None:
        response_valid_losses = distillation_losses[effective_response_mask.bool()]
        if response_valid_losses.numel() > 0:
            non_rejected_token_loss = response_valid_losses.mean()
        else:
            non_rejected_token_loss = distillation_losses.detach().new_tensor(0.0)
        distillation_metrics["distillation/non_rejected_token_loss"] = Metric(
            AggregationType.MEAN, non_rejected_token_loss
        )

    rejected_draft_losses = None
    rejected_draft_mask = None
    rejected_loss_weights = None
    rejected_position_decay_applied = False
    if rejected_draft_stream is not None:
        rejected_student_log_probs, rejected_teacher_log_probs, rejected_draft_mask = rejected_draft_stream
        rejected_loss_weights, rejected_position_decay_applied = _build_rejected_draft_position_weights(
            model_output=model_output,
            rejected_draft_mask=rejected_draft_mask,
            loss_config=loss_config,
        )
        rejected_draft_losses, rejected_component_metrics = _combine_sampled_reverse_forward_losses(
            student_log_probs=rejected_student_log_probs,
            teacher_log_probs=rejected_teacher_log_probs,
            loss_config=loss_config,
            mask=rejected_draft_mask,
            stream_name="rejected_draft",
            force_reverse_kl=bool(getattr(loss_config, "rejected_draft_use_reverse_kl", False)),
        )
        distillation_metrics.update(rejected_component_metrics)
        if loss_config.loss_max_clamp is not None:
            rejected_clamp_fraction = _compute_loss_clamp_fraction(
                rejected_draft_losses, rejected_draft_mask, loss_config.loss_max_clamp
            )
            distillation_metrics["distillation/rejected_draft_loss_clamp_fraction"] = Metric(
                AggregationType.MEAN, rejected_clamp_fraction
            )
            rejected_draft_losses = rejected_draft_losses.clamp(
                min=-loss_config.loss_max_clamp,
                max=loss_config.loss_max_clamp,
            )
        rejected_valid_losses = rejected_draft_losses[rejected_draft_mask]
        distillation_metrics["distillation/rejected_draft_token_count"] = Metric(
            AggregationType.SUM, rejected_draft_mask.sum().to(torch.float32)
        )
        distillation_metrics["distillation/rejected_draft_loss"] = Metric(
            AggregationType.MEAN, rejected_valid_losses.mean()
        )

    if not bool(effective_response_mask.any()) and rejected_draft_losses is None:
        if "log_probs" in model_output:
            zero_loss = no_padding_2_padding(model_output["log_probs"], data).sum() * 0.0
        else:
            zero_loss = distillation_losses.sum() * 0.0
        return zero_loss, distillation_metrics

    if loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        for k, v in _agg_loss_global_batch_info(config).items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = data["old_log_probs"].to_padded_tensor(0.0)
        rollout_is_weights = data.get("rollout_is_weights", None)
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=-distillation_losses.detach(),
            response_mask=effective_response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        if rejected_draft_losses is None:
            distillation_loss = agg_loss(
                loss_mat=distillation_losses,
                loss_mask=effective_response_mask,
                loss_agg_mode=loss_agg_mode,
                **_agg_loss_global_batch_info(config),
            )
        else:
            response_weight = float(getattr(loss_config, "response_stream_weight", 1.0))
            rejected_weight = float(getattr(loss_config, "rejected_draft_stream_weight", 1.0))
            global_batch_info = getattr(config, "global_batch_info", {}) or {}
            response_count = effective_response_mask.sum().to(dtype=distillation_losses.dtype)
            rejected_count = rejected_draft_mask.sum().to(dtype=distillation_losses.dtype)
            rejected_effective_count = rejected_loss_weights.sum().to(dtype=distillation_losses.dtype)
            response_sum = (distillation_losses * effective_response_mask).sum()
            rejected_sum = (rejected_draft_losses * rejected_loss_weights.to(dtype=rejected_draft_losses.dtype)).sum()
            global_response_count = global_batch_info.get("batch_num_tokens")
            if global_response_count is None:
                global_response_count = response_count
            global_rejected_count = global_batch_info.get("opd_rejected_draft_batch_num_tokens")
            if global_rejected_count is None:
                global_rejected_count = rejected_count
            global_rejected_effective_count = global_batch_info.get(
                "opd_rejected_draft_batch_effective_num_tokens"
            )
            if global_rejected_effective_count is None:
                global_rejected_effective_count = (
                    _global_sum(rejected_effective_count, dp_group)
                    if rejected_position_decay_applied
                    else _scalar_like(global_rejected_count, rejected_effective_count)
                )
            denom = (
                response_weight * _scalar_like(global_response_count, response_count)
                + rejected_weight * _scalar_like(global_rejected_effective_count, rejected_effective_count)
            )
            if denom.item() <= 0:
                distillation_loss = (response_sum + rejected_sum) * 0.0
            else:
                distillation_loss = (response_weight * response_sum + rejected_weight * rejected_sum) / denom
                distillation_loss = distillation_loss * _scalar_like(
                    global_batch_info.get("dp_size", 1), distillation_loss
                )
            distillation_metrics["distillation/combined_token_count"] = Metric(
                AggregationType.SUM, response_count.detach() + rejected_count.detach()
            )

    return distillation_loss, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["forward_kl_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    response_mask_bool = get_effective_distillation_response_mask(data=data, model_output=model_output)
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    if student_mass.numel() == 0:
        zero = distillation_losses.detach().new_tensor(0.0)
        distillation_metrics = {
            "distillation/student_mass": zero.item(),
            "distillation/student_mass_min": Metric(AggregationType.MIN, zero),
            "distillation/student_mass_max": Metric(AggregationType.MAX, zero),
            "distillation/teacher_mass": zero.item(),
            "distillation/teacher_mass_min": Metric(AggregationType.MIN, zero),
            "distillation/teacher_mass_max": Metric(AggregationType.MAX, zero),
        }
        return distillation_losses, distillation_metrics
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
    }

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)

    return distillation_losses, distillation_metrics


@register_distillation_loss(
    DistillationLossSettings(names=[EAGLE3_NATIVE_TARGET_DISTRIBUTION_LOSS], use_estimator=True)
)  # type: ignore[arg-type]
def compute_eagle3_native_target_distribution_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Use the EAGLE3 native target-distribution CE produced by the composed student.

    The model forward computes target_p = softmax(target_logits[..., t2d]) from the
    frozen target model and CE against draft-vocab logits. The sampled scalar logprob
    path remains in model_output["log_probs"] only for the diagnostic k3 metric.
    """
    if distillation_config.distillation_loss.use_policy_gradient:
        raise NotImplementedError(
            "EAGLE3 native target-distribution loss is a supervised CE and does not support "
            "use_policy_gradient=True."
        )
    if "eagle3_native_ce_losses" not in model_output:
        raise RuntimeError(
            "EAGLE3 native target-distribution loss requires model_output['eagle3_native_ce_losses']."
        )

    native_ce_losses = no_padding_2_padding(model_output["eagle3_native_ce_losses"], data)
    response_mask_bool = get_effective_distillation_response_mask(data=data, model_output=model_output)
    if native_ce_losses.shape != response_mask_bool.shape:
        raise ValueError(
            "EAGLE3 native CE loss shape mismatch: "
            f"losses={tuple(native_ce_losses.shape)}, mask={tuple(response_mask_bool.shape)}."
        )

    native_ce_mean = _valid_mean(native_ce_losses, response_mask_bool)
    metrics = {
        "eagle3/native_ce_loss": Metric(AggregationType.MEAN, native_ce_mean),
    }

    if "log_probs" in model_output and "teacher_logprobs" in data:
        scalar_log_probs = no_padding_2_padding(model_output["log_probs"], data)
        teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
        scalar_mask = model_output.get("eagle3_selected_scalar_loss_mask")
        if scalar_mask is None:
            scalar_mask_bool = response_mask_bool
        else:
            scalar_mask_bool = no_padding_2_padding(scalar_mask, data).bool()
        if scalar_log_probs.shape == teacher_log_probs.shape == scalar_mask_bool.shape:
            scalar_k3_losses = kl_penalty(
                logprob=scalar_log_probs,
                ref_logprob=teacher_log_probs,
                kl_penalty="k3",
            )
            metrics["eagle3/selected_scalar_k3_loss"] = Metric(
                AggregationType.MEAN, _valid_mean(scalar_k3_losses, scalar_mask_bool)
            )

    return native_ce_losses, metrics


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    response_mask_bool = get_effective_distillation_response_mask(data=data, model_output=model_output)
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_losses, component_metrics = _combine_sampled_reverse_forward_losses(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        loss_config=loss_config,
        mask=response_mask_bool,
        stream_name="response",
    )
    # Since k1 can be negative, log the mean absolute loss.
    if response_mask_bool.any():
        abs_loss = distillation_losses[response_mask_bool].abs().mean()
    else:
        abs_loss = distillation_losses.detach().new_tensor(0.0)
    metrics = {
        "distillation/abs_loss": Metric(AggregationType.MEAN, abs_loss),
    }
    metrics.update(component_metrics)
    return distillation_losses, metrics
