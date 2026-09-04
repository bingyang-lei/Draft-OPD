#!/usr/bin/env python3
"""Extract an EAGLE3 draft checkpoint from a composed-model FSDP actor checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open, save_file
from torch.distributed._tensor import Placement

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
        help="verl FSDP actor checkpoint directory containing model_world_size_*_rank_*.pt.",
    )
    parser.add_argument(
        "--reference-draft-dir",
        required=True,
        type=Path,
        help="Existing SpecForge EAGLE3 checkpoint directory used for config and expected safetensors keys.",
    )
    parser.add_argument(
        "--target-dir",
        required=True,
        type=Path,
        help="Output directory for the extracted EAGLE3 draft checkpoint.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Floating dtype used when saving merged draft weights.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty target directory.",
    )
    parser.add_argument(
        "--copy-reference-training-state",
        action="store_true",
        help=(
            "Copy reference training_state.pt into the target directory. This is useful only for tooling "
            "that expects the file to exist; it is not a valid optimizer/scheduler state for the converted weights."
        ),
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


def load_reference_metadata(reference_draft_dir: Path) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    ref_path = reference_draft_dir / "model.safetensors"
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing reference safetensors: {ref_path}")

    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, str] = {}
    with safe_open(ref_path, framework="pt", device="cpu") as fin:
        for key in fin.keys():
            tensor = fin.get_tensor(key)
            shapes[key] = tuple(tensor.shape)
            dtypes[key] = str(tensor.dtype)
    return shapes, dtypes


def maybe_cast(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if tensor.is_floating_point():
        return tensor.to(dtype)
    return tensor


def merge_by_placement(tensors: list[torch.Tensor], placement: Placement) -> torch.Tensor:
    if placement.is_replicate():
        return tensors[0].contiguous()
    if placement.is_partial():
        raise NotImplementedError("Partial DTensor placement is not supported.")
    if placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    raise NotImplementedError(f"Unsupported DTensor placement: {placement}.")


def load_draft_shards(
    actor_dir: Path,
    world_size: int,
    dtype: torch.dtype,
    expected_keys: set[str],
) -> tuple[dict[str, list[torch.Tensor]], dict[str, tuple[Any, ...]], list[str]]:
    state_by_key: dict[str, list[torch.Tensor]] = {}
    placement_by_key: dict[str, tuple[Any, ...]] = {}
    skipped_keys: set[str] = set()

    for rank in range(world_size):
        shard_path = actor_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing FSDP shard: {shard_path}")

        print(f"Loading EAGLE3 draft tensors from {shard_path}")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        for full_key, tensor in shard.items():
            if not full_key.startswith("draft_model."):
                continue

            key = full_key.removeprefix("draft_model.")
            if key not in expected_keys:
                skipped_keys.add(key)
                continue

            if isinstance(tensor, DTensor):
                local_tensor = maybe_cast(tensor.to_local(), dtype)
                placements = tuple(tensor.placements)
                if key not in placement_by_key:
                    placement_by_key[key] = placements
                elif placement_by_key[key] != placements:
                    raise ValueError(f"Inconsistent DTensor placements for {key}: {placement_by_key[key]} vs {placements}")
            else:
                local_tensor = maybe_cast(tensor, dtype)

            state_by_key.setdefault(key, []).append(local_tensor.cpu())

        del shard
        gc.collect()

    return state_by_key, placement_by_key, sorted(skipped_keys)


def merge_replicated_or_sharded_tensor(
    key: str,
    shards: list[torch.Tensor],
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    if not shards:
        raise ValueError(f"No shards found for {key}.")

    first = shards[0]
    if tuple(first.shape) == expected_shape:
        for shard_idx, shard in enumerate(shards[1:], start=1):
            if tuple(shard.shape) != expected_shape:
                raise ValueError(
                    f"Replicated tensor {key} has inconsistent shape at shard {shard_idx}: "
                    f"{tuple(shard.shape)} vs expected {expected_shape}."
                )
            if first.numel() <= 10_000_000 and not torch.equal(first, shard):
                raise ValueError(f"Replicated tensor {key} differs between rank 0 and rank {shard_idx}.")
        return first.contiguous()

    concatenated = torch.cat(shards, dim=0).contiguous()
    if tuple(concatenated.shape) == expected_shape:
        return concatenated

    raise ValueError(
        f"Cannot merge non-DTensor tensor {key}: first_shape={tuple(first.shape)}, "
        f"cat_shape={tuple(concatenated.shape)}, expected_shape={expected_shape}."
    )


def merge_draft_state(
    state_by_key: dict[str, list[torch.Tensor]],
    placement_by_key: dict[str, tuple[Any, ...]],
    reference_shapes: dict[str, tuple[int, ...]],
) -> dict[str, torch.Tensor]:
    missing = sorted(set(reference_shapes) - set(state_by_key))
    if missing:
        raise ValueError(f"Actor checkpoint is missing EAGLE3 draft tensors required by reference: {missing}")

    merged: dict[str, torch.Tensor] = {}
    for key in sorted(reference_shapes):
        shards = state_by_key[key]
        if key in placement_by_key:
            placements = placement_by_key[key]
            if len(placements) != 1:
                raise NotImplementedError(f"Only 1D FSDP sharding is supported, got placements={placements} for {key}.")
            tensor = merge_by_placement(shards, placements[0])
        else:
            tensor = merge_replicated_or_sharded_tensor(key, shards, reference_shapes[key])

        if tuple(tensor.shape) != reference_shapes[key]:
            raise ValueError(f"Shape mismatch for {key}: got {tuple(tensor.shape)}, expected {reference_shapes[key]}.")
        merged[key] = tensor
    return merged


def prepare_target_dir(target_dir: Path, overwrite: bool) -> None:
    if target_dir.exists() and any(target_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Target directory is not empty: {target_dir}. Pass --overwrite to reuse it.")
    target_dir.mkdir(parents=True, exist_ok=True)


def remove_stale_outputs(target_dir: Path, *, copy_training_state: bool) -> None:
    filenames = ["model.safetensors", "config.json", "conversion_metadata.json"]
    if not copy_training_state:
        filenames.append("training_state.pt")
    for filename in filenames:
        path = target_dir / filename
        if path.exists():
            path.unlink()


def copy_reference_files(reference_draft_dir: Path, target_dir: Path, copy_training_state: bool) -> list[str]:
    copied: list[str] = []
    for filename in ("config.json",):
        src = reference_draft_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing required reference file: {src}")
        shutil.copy2(src, target_dir / filename)
        copied.append(filename)

    if copy_training_state:
        src = reference_draft_dir / "training_state.pt"
        if not src.exists():
            raise FileNotFoundError(f"Missing reference training_state.pt: {src}")
        shutil.copy2(src, target_dir / "training_state.pt")
        copied.append("training_state.pt")
    return copied


def write_metadata(
    target_dir: Path,
    *,
    actor_dir: Path,
    reference_draft_dir: Path,
    world_size: int,
    dtype: str,
    tensor_count: int,
    skipped_keys: list[str],
    copied_files: list[str],
) -> None:
    metadata = {
        "format": "specforge_eagle3_draft_from_verl_fsdp_actor",
        "actor_dir": str(actor_dir),
        "reference_draft_dir": str(reference_draft_dir),
        "world_size": world_size,
        "dtype": dtype,
        "tensor_count": tensor_count,
        "skipped_draft_keys": skipped_keys,
        "copied_reference_files": copied_files,
    }
    with (target_dir / "conversion_metadata.json").open("w", encoding="utf-8") as fout:
        json.dump(metadata, fout, ensure_ascii=False, indent=2)
        fout.write("\n")


def main() -> None:
    args = parse_args()
    actor_dir = args.actor_dir.resolve()
    reference_draft_dir = args.reference_draft_dir.resolve()
    target_dir = args.target_dir.resolve()

    prepare_target_dir(target_dir, overwrite=args.overwrite)
    if args.overwrite:
        remove_stale_outputs(target_dir, copy_training_state=args.copy_reference_training_state)
    dtype = get_dtype(args.dtype)
    reference_shapes, _ = load_reference_metadata(reference_draft_dir)
    expected_keys = set(reference_shapes)
    world_size = get_world_size(actor_dir)

    state_by_key, placement_by_key, skipped_keys = load_draft_shards(
        actor_dir=actor_dir,
        world_size=world_size,
        dtype=dtype,
        expected_keys=expected_keys,
    )
    draft_state = merge_draft_state(
        state_by_key=state_by_key,
        placement_by_key=placement_by_key,
        reference_shapes=reference_shapes,
    )

    save_path = target_dir / "model.safetensors"
    print(f"Saving {len(draft_state)} EAGLE3 draft tensors to {save_path}")
    save_file(draft_state, save_path)
    copied_files = copy_reference_files(
        reference_draft_dir=reference_draft_dir,
        target_dir=target_dir,
        copy_training_state=args.copy_reference_training_state,
    )
    write_metadata(
        target_dir,
        actor_dir=actor_dir,
        reference_draft_dir=reference_draft_dir,
        world_size=world_size,
        dtype=args.dtype,
        tensor_count=len(draft_state),
        skipped_keys=skipped_keys,
        copied_files=copied_files,
    )
    print(f"Saved EAGLE3 draft checkpoint to {target_dir}")
    if skipped_keys:
        print(f"Skipped draft keys not present in the reference checkpoint: {skipped_keys}")


if __name__ == "__main__":
    main()
