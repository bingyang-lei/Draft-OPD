import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from verl import DataProto


REPLAY_CATEGORIES = ("accepted", "first_rejected", "post_rejection_suffix")
MISMATCH_GROUPS = (
    "matched_target_trajectory",
    "accepted_replay",
    "first_rejected_replay",
    "post_rejection_suffix_replay",
)
ANCHOR_FULL_ACCEPT = "full_accept_after_block"


def reverse_kl_from_logits(q_logits: torch.Tensor, p_logits: torch.Tensor) -> torch.Tensor:
    """Exact KL(q || p) over the full vocabulary for each row."""
    q_log_probs = F.log_softmax(q_logits.float(), dim=-1)
    p_log_probs = F.log_softmax(p_logits.float(), dim=-1)
    return (q_log_probs.exp() * (q_log_probs - p_log_probs)).sum(dim=-1)


def replay_block_counts(accepted_length: int, drafted_length: int) -> dict[str, int]:
    """Return replay-token composition counts for one draft block."""
    r = max(int(accepted_length), 0)
    length = max(int(drafted_length), 0)
    if r > length:
        raise ValueError(f"accepted_length cannot exceed drafted_length: r={r}, L={length}.")
    first_rejected = 1 if r < length else 0
    return {
        "accepted": r,
        "first_rejected": first_rejected,
        "post_rejection_suffix": max(length - r - first_rejected, 0),
    }


def replay_anchor_bucket(accepted_length: int, drafted_length: int) -> str:
    r = max(int(accepted_length), 0)
    length = max(int(drafted_length), 0)
    if r > length:
        raise ValueError(f"accepted_length cannot exceed drafted_length: r={r}, L={length}.")
    if r == length:
        return ANCHOR_FULL_ACCEPT
    return str(r + 1)


def _to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _as_list(value: Any) -> list[Any]:
    value = _to_python(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _normalize_nested(values: Any, batch_size: int, *, cast_fn) -> list[list[Any]]:
    def flatten_item(value: Any) -> list[Any]:
        value = _to_python(value)
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            flattened: list[Any] = []
            for sub_value in value:
                flattened.extend(flatten_item(sub_value))
            return flattened
        return [cast_fn(value)]

    values = _to_python(values)
    if values is None:
        return [[] for _ in range(batch_size)]
    if not isinstance(values, list):
        values = [values]
    if batch_size == 1:
        values = [values]
    if len(values) != batch_size:
        raise ValueError(f"Replay distribution metadata has {len(values)} samples, expected {batch_size}.")
    normalized: list[list[Any]] = []
    for sample_values in values:
        normalized.append(flatten_item(sample_values))
    return normalized


def _safe_experiment_path(experiment_name: str) -> Path:
    raw_name = str(experiment_name or "unknown").strip().lstrip("/\\") or "unknown"
    parts = [part for part in Path(raw_name).parts if part not in ("", ".", "..")]
    return Path(*parts) if parts else Path("unknown")


def _sum_metric_value(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().sum().item())
    if isinstance(value, np.ndarray):
        return float(value.astype(float).sum())
    if isinstance(value, (list, tuple)):
        return sum(_sum_metric_value(item) for item in value)
    return float(value)


@dataclass
class ReplayDistributionTracker:
    total_training_steps: int
    max_anchor_position: int = 16
    composition_counts: dict[str, int] = field(
        default_factory=lambda: {category: 0 for category in REPLAY_CATEGORIES}
    )
    anchor_counts: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "first_half": {str(pos): 0 for pos in range(1, 17)} | {ANCHOR_FULL_ACCEPT: 0},
            "second_half": {str(pos): 0 for pos in range(1, 17)} | {ANCHOR_FULL_ACCEPT: 0},
        }
    )
    mismatch_sums: dict[str, float] = field(default_factory=lambda: {group: 0.0 for group in MISMATCH_GROUPS})
    mismatch_counts: dict[str, int] = field(default_factory=lambda: {group: 0 for group in MISMATCH_GROUPS})

    @property
    def total_replay_token_count(self) -> int:
        return sum(self.composition_counts.values())

    def _half_name(self, global_step: int) -> str:
        return "first_half" if float(global_step) <= float(self.total_training_steps) / 2.0 else "second_half"

    def update_rollout(self, gen_output: "DataProto", *, fallback_global_step: int) -> None:
        batch_size = len(gen_output)
        non_tensor = gen_output.non_tensor_batch
        accepted_lengths = _normalize_nested(
            non_tensor.get("dflash_replay_block_accepted_lengths"), batch_size, cast_fn=int
        )
        drafted_lengths = _normalize_nested(
            non_tensor.get("dflash_replay_block_drafted_lengths"), batch_size, cast_fn=int
        )
        if "dflash_replay_block_accepted_lengths" not in non_tensor or "dflash_replay_block_drafted_lengths" not in non_tensor:
            raise RuntimeError(
                "use_replay_dis=True requires DFLASH per-block metadata "
                "(dflash_replay_block_accepted_lengths and dflash_replay_block_drafted_lengths)."
            )
        global_steps = _as_list(non_tensor.get("global_steps"))
        if len(global_steps) != batch_size:
            global_steps = [fallback_global_step for _ in range(batch_size)]

        for sample_idx, (sample_r, sample_l) in enumerate(zip(accepted_lengths, drafted_lengths, strict=True)):
            if len(sample_r) != len(sample_l):
                raise ValueError(
                    "Replay block metadata length mismatch for sample "
                    f"{sample_idx}: accepted={len(sample_r)}, drafted={len(sample_l)}."
                )
            half_name = self._half_name(int(global_steps[sample_idx]))
            for accepted_length, drafted_length in zip(sample_r, sample_l, strict=True):
                if int(drafted_length) <= 0:
                    continue
                counts = replay_block_counts(accepted_length, drafted_length)
                for category, count in counts.items():
                    self.composition_counts[category] += count

                bucket = replay_anchor_bucket(accepted_length, drafted_length)
                if bucket != ANCHOR_FULL_ACCEPT and int(bucket) > self.max_anchor_position:
                    raise ValueError(
                        f"anchor_position={bucket} exceeds configured max_anchor_position={self.max_anchor_position}."
                    )
                self.anchor_counts[half_name][bucket] += 1

    def update_mismatch(self, metrics: dict[str, Any]) -> None:
        for group in MISMATCH_GROUPS:
            self.mismatch_sums[group] += _sum_metric_value(metrics.get(f"replay_dis/mismatch/{group}/sum_KL"))
            self.mismatch_counts[group] += int(
                round(_sum_metric_value(metrics.get(f"replay_dis/mismatch/{group}/num_states")))
            )

    def composition_rows(self) -> list[dict[str, Any]]:
        total = self.total_replay_token_count
        rows = [
            {
                "category": category,
                "count": self.composition_counts[category],
                "raw_share": self.composition_counts[category] / total if total > 0 else 0.0,
            }
            for category in REPLAY_CATEGORIES
        ]
        rows.append({"category": "total", "count": total, "raw_share": 1.0 if total > 0 else 0.0})
        return rows

    def mismatch_rows(self) -> list[dict[str, Any]]:
        rows = []
        for group in MISMATCH_GROUPS:
            count = self.mismatch_counts[group]
            sum_kl = self.mismatch_sums[group]
            rows.append(
                {
                    "state_group": group,
                    "num_states": count,
                    "sum_KL": sum_kl,
                    "mean_KL": sum_kl / count if count > 0 else 0.0,
                }
            )
        return rows

    def anchor_rows(self, half_name: str) -> list[dict[str, Any]]:
        counts = self.anchor_counts[half_name]
        total = sum(counts.values())
        rows = [
            {
                "anchor_position": str(pos),
                "count": counts[str(pos)],
                "frequency": counts[str(pos)] / total if total > 0 else 0.0,
            }
            for pos in range(1, self.max_anchor_position + 1)
        ]
        rows.append(
            {
                "anchor_position": ANCHOR_FULL_ACCEPT,
                "count": counts[ANCHOR_FULL_ACCEPT],
                "frequency": counts[ANCHOR_FULL_ACCEPT] / total if total > 0 else 0.0,
            }
        )
        rows.append({"anchor_position": "total", "count": total, "frequency": 1.0 if total > 0 else 0.0})
        return rows

    def save(self, *, experiment_name: str, root: str | Path) -> Path:
        output_dir = Path(root) / _safe_experiment_path(experiment_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        payloads = {
            "composition.json": {"rows": self.composition_rows()},
            "mismatch.json": {"rows": self.mismatch_rows()},
            "anchor_position_frequency_first_half.json": {"rows": self.anchor_rows("first_half")},
            "anchor_position_frequency_second_half.json": {"rows": self.anchor_rows("second_half")},
        }
        for filename, payload in payloads.items():
            with (output_dir / filename).open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.write("\n")
        return output_dir

    def format_composition_summary(self) -> str:
        lines = ["Replay-token composition", f"{'category':<28}{'count':>12}{'raw_share':>14}"]
        for row in self.composition_rows():
            lines.append(f"{row['category']:<28}{row['count']:>12d}{row['raw_share']:>14.4f}")
        return "\n".join(lines)

    def format_mismatch_summary(self) -> str:
        lines = ["Draft-target mismatch: reverse KL", f"{'state_group':<34}{'num_states':>12}{'mean_KL(q||p)':>18}"]
        for row in self.mismatch_rows():
            lines.append(f"{row['state_group']:<34}{row['num_states']:>12d}{row['mean_KL']:>18.6f}")
        return "\n".join(lines)

    def format_anchor_summary(self, half_name: str, title: Optional[str] = None) -> str:
        lines = [title or half_name, f"{'anchor_position':<28}{'count':>12}{'frequency':>14}"]
        for row in self.anchor_rows(half_name):
            lines.append(f"{row['anchor_position']:<28}{row['count']:>12d}{row['frequency']:>14.4f}")
        return "\n".join(lines)


def flatten_metric_dict(metrics: dict[str, Any], keys: Iterable[str]) -> dict[str, float]:
    return {key: _sum_metric_value(metrics.get(key)) for key in keys}
