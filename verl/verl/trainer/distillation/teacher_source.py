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

import os
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf, open_dict

TEACHER_LOGPROB_SOURCE_AUTO = "auto"
TEACHER_LOGPROB_SOURCE_SERVER = "server"
TEACHER_LOGPROB_SOURCE_COMPOSED_MAIN = "composed_main"
TEACHER_LOGPROB_SOURCES = {
    TEACHER_LOGPROB_SOURCE_AUTO,
    TEACHER_LOGPROB_SOURCE_SERVER,
    TEACHER_LOGPROB_SOURCE_COMPOSED_MAIN,
}

_SCALAR_DISTILLATION_LOSSES = {"kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"}


def _select(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, DictConfig):
        return OmegaConf.select(config, key, default=default)
    current = config
    for part in key.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(part, default)
        else:
            current = getattr(current, part, default)
    return current


def _normalize_model_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    path = str(path)
    if os.path.exists(path):
        return os.path.realpath(os.path.expanduser(path))
    return path.rstrip("/")


def _configured_teacher_paths(config: Any) -> list[str]:
    teacher_models = _select(config, "distillation.teacher_models", default={}) or {}
    if isinstance(teacher_models, DictConfig):
        teacher_models = OmegaConf.to_container(teacher_models, resolve=True)
    paths = []
    for teacher_config in teacher_models.values():
        if isinstance(teacher_config, DictConfig):
            model_path = teacher_config.get("model_path")
        elif isinstance(teacher_config, dict):
            model_path = teacher_config.get("model_path")
        else:
            model_path = getattr(teacher_config, "model_path", None)
        if model_path is not None:
            paths.append(str(model_path))
    return paths


def _validate_composed_main_source(config: Any) -> None:
    if not bool(_select(config, "actor_rollout_ref.model.override_config.verl_composed_dflash_student", False)):
        raise ValueError(
            "distillation.teacher_logprob_source=composed_main requires "
            "actor_rollout_ref.model.override_config.verl_composed_dflash_student=true."
        )

    use_policy_gradient = bool(_select(config, "distillation.distillation_loss.use_policy_gradient", False))
    if use_policy_gradient:
        raise NotImplementedError(
            "Composed-main teacher scoring only supports direct supervised distillation "
            "(use_policy_gradient=False)."
        )

    use_tv_loss = bool(_select(config, "distillation.distillation_loss.use_tv_loss", False))
    loss_mode = str(_select(config, "distillation.distillation_loss.loss_mode", "k3"))
    if not use_tv_loss and loss_mode not in _SCALAR_DISTILLATION_LOSSES:
        raise NotImplementedError(
            "Composed-main teacher scoring supports exact TV or scalar sampled distillation losses only; "
            f"got loss_mode={loss_mode!r}."
        )


def resolve_teacher_logprob_source(config: DictConfig) -> str:
    """Resolve and persist the teacher-logprob source before resource construction."""
    distillation_config = config.get("distillation")
    if distillation_config is None or not bool(distillation_config.get("enabled", False)):
        return TEACHER_LOGPROB_SOURCE_SERVER

    requested = str(distillation_config.get("teacher_logprob_source", TEACHER_LOGPROB_SOURCE_AUTO))
    if requested not in TEACHER_LOGPROB_SOURCES:
        raise ValueError(
            f"Unsupported distillation.teacher_logprob_source={requested!r}; "
            f"expected one of {sorted(TEACHER_LOGPROB_SOURCES)}."
        )

    resolved = requested
    reason = "explicit configuration"
    if requested == TEACHER_LOGPROB_SOURCE_AUTO:
        composed_dflash = bool(
            _select(config, "actor_rollout_ref.model.override_config.verl_composed_dflash_student", False)
        )
        if not composed_dflash:
            resolved = TEACHER_LOGPROB_SOURCE_SERVER
            reason = "actor is not a composed DFLASH student"
        else:
            main_model_path = _select(
                config,
                "actor_rollout_ref.model.override_config.verl_dflash_main_model_path",
                _select(config, "actor_rollout_ref.model.path"),
            )
            teacher_paths = _configured_teacher_paths(config)
            same_single_teacher = len(teacher_paths) <= 1 and (
                not teacher_paths or _normalize_model_path(teacher_paths[0]) == _normalize_model_path(main_model_path)
            )
            if same_single_teacher:
                resolved = TEACHER_LOGPROB_SOURCE_COMPOSED_MAIN
                reason = "composed DFLASH main model matches the configured teacher"
            else:
                resolved = TEACHER_LOGPROB_SOURCE_SERVER
                reason = "configured teacher differs from the composed DFLASH main model or uses multi-teacher routing"

    if resolved == TEACHER_LOGPROB_SOURCE_COMPOSED_MAIN:
        _validate_composed_main_source(config)

    with open_dict(distillation_config):
        distillation_config.teacher_logprob_source = resolved
    print(f"[distillation] teacher_logprob_source={resolved} ({reason})")
    return resolved


def requires_external_teacher(config: Any) -> bool:
    distillation_config = _select(config, "distillation")
    if distillation_config is None or not bool(_select(config, "distillation.enabled", False)):
        return False
    source = str(
        _select(
            config,
            "distillation.teacher_logprob_source",
            TEACHER_LOGPROB_SOURCE_SERVER,
        )
    )
    # Unresolved `auto` is treated as the legacy server path. Normal launchers
    # call resolve_teacher_logprob_source before constructing any resources.
    return source != TEACHER_LOGPROB_SOURCE_COMPOSED_MAIN

