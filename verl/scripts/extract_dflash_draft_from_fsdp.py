#!/usr/bin/env python3
"""Extract a DFLASH draft model from a composed-model FSDP actor checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file
from torch.distributed._tensor import Placement, Shard

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--actor-dir",
        required=True,
        type=Path,
        help="Path to a verl FSDP actor checkpoint directory containing model_world_size_*_rank_*.pt.",
    )
    parser.add_argument(
        "--reference-draft-dir",
        required=True,
        type=Path,
        help="Existing DFLASH draft HF directory used as the source for config.json and dflash.py.",
    )
    parser.add_argument(
        "--target-dir",
        required=True,
        type=Path,
        help="Output directory for the extracted draft model.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Dtype used when saving merged draft weights.",
    )
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def get_world_size(actor_dir: Path) -> int:
    config_path = actor_dir / "fsdp_config.json"
    with config_path.open("r", encoding="utf-8") as fin:
        config = json.load(fin)
    world_size = config.get("world_size")
    if world_size is None:
        raise ValueError(f"Missing world_size in {config_path}.")
    return int(world_size)


def merge_by_placement(tensors: list[torch.Tensor], placement: Placement) -> torch.Tensor:
    if placement.is_replicate():
        return tensors[0].contiguous()
    if placement.is_partial():
        raise NotImplementedError("Partial DTensor placement is not supported.")
    if placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    raise NotImplementedError(f"Unsupported DTensor placement: {placement}.")


def load_draft_shards(actor_dir: Path, world_size: int, dtype: torch.dtype) -> tuple[dict[str, list[torch.Tensor]], dict]:
    state_by_key: dict[str, list[torch.Tensor]] = {}
    placement_by_key: dict[str, tuple[Shard, ...]] = {}

    for rank in range(world_size):
        shard_path = actor_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing FSDP shard: {shard_path}")

        print(f"Loading draft weights from {shard_path}")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        for full_key, tensor in shard.items():
            if not full_key.startswith("draft_model."):
                continue

            key = full_key.removeprefix("draft_model.")
            if isinstance(tensor, DTensor):
                local_tensor = tensor._local_tensor.to(dtype)
                placements = tuple(tensor.placements)
                if key not in placement_by_key:
                    placement_by_key[key] = placements
                elif placement_by_key[key] != placements:
                    raise ValueError(f"Inconsistent DTensor placements for {key}: {placement_by_key[key]} vs {placements}")
            else:
                local_tensor = tensor.to(dtype)

            state_by_key.setdefault(key, []).append(local_tensor.cpu())

        del shard
        gc.collect()

    return state_by_key, placement_by_key


def merge_draft_state(state_by_key: dict[str, list[torch.Tensor]], placement_by_key: dict) -> dict[str, torch.Tensor]:
    merged = {}
    for key in sorted(state_by_key):
        shards = state_by_key[key]
        if key in placement_by_key:
            placements = placement_by_key[key]
            if len(placements) != 1:
                raise NotImplementedError(f"Only 1D FSDP sharding is supported, got placements={placements} for {key}.")
            merged[key] = merge_by_placement(shards, placements[0])
        else:
            merged[key] = torch.cat(shards, dim=0).contiguous()
    return merged


def copy_reference_files(reference_draft_dir: Path, target_dir: Path) -> None:
    for filename in ("config.json", "dflash.py"):
        src = reference_draft_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing required reference file: {src}")
        shutil.copy2(src, target_dir / filename)


def validate_against_reference_keys(reference_draft_dir: Path, draft_state: dict[str, torch.Tensor]) -> None:
    ref_path = reference_draft_dir / "model.safetensors"
    if not ref_path.exists():
        print(f"Reference safetensors not found at {ref_path}; skip key validation.")
        return

    with safe_open(ref_path, framework="pt", device="cpu") as fin:
        ref_keys = set(fin.keys())
    draft_keys = set(draft_state.keys())
    missing = sorted(ref_keys - draft_keys)
    extra = sorted(draft_keys - ref_keys)
    if missing or extra:
        raise ValueError(f"Draft key mismatch. Missing={missing}, extra={extra}")


def main() -> None:
    args = parse_args()
    actor_dir = args.actor_dir.resolve()
    reference_draft_dir = args.reference_draft_dir.resolve()
    target_dir = args.target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    dtype = get_dtype(args.dtype)
    world_size = get_world_size(actor_dir)
    state_by_key, placement_by_key = load_draft_shards(actor_dir, world_size, dtype)
    draft_state = merge_draft_state(state_by_key, placement_by_key)
    validate_against_reference_keys(reference_draft_dir, draft_state)

    save_path = target_dir / "model.safetensors"
    print(f"Saving {len(draft_state)} draft tensors to {save_path}")
    save_file(draft_state, save_path)
    copy_reference_files(reference_draft_dir, target_dir)
    print(f"Saved DFLASH draft checkpoint to {target_dir}")


if __name__ == "__main__":
    main()
