from __future__ import annotations

from collections.abc import Iterable as IterableABC
from typing import Any, Dict, Iterable, List, Optional


DFLASH_REJECTED_DRAFT_META_KEYS = frozenset(
    {
        "dflash_rejected_draft_anchor_indices",
        "dflash_rejected_draft_offsets",
        "dflash_rejected_draft_token_ids",
        "dflash_rejected_draft_teacher_logprobs",
        "dflash_replay_block_anchor_indices",
        "dflash_replay_block_accepted_lengths",
        "dflash_replay_block_drafted_lengths",
    }
)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, IterableABC) and not isinstance(value, (str, bytes, bytearray, dict)):
        return list(value)
    return [value]


def _flatten_dflash_rejected_draft_chunk(chunk: Any) -> list[Any]:
    """Flatten token-aligned chunks into per-request metadata lists."""
    flat: list[Any] = []
    for item in _as_list(chunk):
        if item is None:
            continue
        flat.extend(_as_list(item))
    return flat


def extend_customized_info_chunk(
    customized_info: Dict[str, List[Any]], key: str, chunk: Any
) -> None:
    if chunk is None:
        return
    values = customized_info.setdefault(key, [])
    if key in DFLASH_REJECTED_DRAFT_META_KEYS:
        values.extend(_flatten_dflash_rejected_draft_chunk(chunk))
        return
    if isinstance(chunk, list):
        values.extend(chunk)
    elif isinstance(chunk, tuple):
        values.extend(chunk)
    elif hasattr(chunk, "tolist"):
        chunk = chunk.tolist()
        if isinstance(chunk, list):
            values.extend(chunk)
        else:
            values.append(chunk)
    else:
        values.append(chunk)


def update_customized_info_meta(
    *,
    meta_info: Dict[str, Any],
    state: Any,
    recv_customized_info: Optional[Dict[str, List[Any]]],
    recv_index: int,
    use_stream_output: bool,
    emit_accumulated: bool = True,
) -> None:
    if recv_customized_info:
        for key, value_by_req in recv_customized_info.items():
            chunk = value_by_req[recv_index]
            if use_stream_output:
                if key in DFLASH_REJECTED_DRAFT_META_KEYS:
                    meta_info[key] = _flatten_dflash_rejected_draft_chunk(chunk)
                else:
                    meta_info[key] = chunk
            else:
                extend_customized_info_chunk(state.customized_info, key, chunk)
    if not use_stream_output and emit_accumulated and state.customized_info:
        for key, values in state.customized_info.items():
            meta_info[key] = values


def append_dflash_reject_token_mask(
    req: Any,
    committed_token_count: int,
    *,
    output_ids_already_updated: bool,
) -> None:
    """Append DFLASH OPD anchor metadata in response-token coordinates."""
    committed_token_count = int(committed_token_count)
    if committed_token_count <= 0:
        return

    if req.customized_info is None:
        req.customized_info = {}
    reject_mask = req.customized_info.setdefault("dflash_reject_token_mask", [])

    prefix_len = len(req.output_ids)
    if output_ids_already_updated:
        prefix_len -= committed_token_count
    prefix_len = max(0, prefix_len)
    if len(reject_mask) < prefix_len:
        reject_mask.extend([False] * (prefix_len - len(reject_mask)))

    reject_mask.extend([False] * committed_token_count)
    reject_mask[-1] = True


def _pad_token_aligned_metadata(values: list[Any], prefix_len: int) -> None:
    if len(values) < prefix_len:
        values.extend([[] for _ in range(prefix_len - len(values))])


def append_dflash_rejected_draft_metadata(
    req: Any,
    committed_token_count: int,
    *,
    anchor_index: int,
    offsets: Iterable[int],
    token_ids: Iterable[int],
    teacher_logprobs: Iterable[float],
    output_ids_already_updated: bool,
) -> None:
    """Append rejected DFLASH draft suffix metadata in response-token coordinates.

    The scheduler slices customized info by output-token offset before sending it to
    the tokenizer process. To preserve that contract while still producing flat
    per-request metadata, each key is stored as a token-aligned list of chunks here
    and flattened by `extend_customized_info_chunk`.
    """

    offsets_list = [int(x) for x in _as_list(offsets)]
    token_ids_list = [int(x) for x in _as_list(token_ids)]
    teacher_logprobs_list = [float(x) for x in _as_list(teacher_logprobs)]
    if not offsets_list:
        return
    if not (len(offsets_list) == len(token_ids_list) == len(teacher_logprobs_list)):
        raise ValueError(
            "DFLASH rejected draft metadata length mismatch: "
            f"offsets={len(offsets_list)}, token_ids={len(token_ids_list)}, "
            f"teacher_logprobs={len(teacher_logprobs_list)}."
        )

    committed_token_count = int(committed_token_count)
    if committed_token_count <= 0:
        return

    if req.customized_info is None:
        req.customized_info = {}

    prefix_len = len(req.output_ids)
    if output_ids_already_updated:
        prefix_len -= committed_token_count
    prefix_len = max(0, prefix_len)

    chunks_by_key = {
        "dflash_rejected_draft_anchor_indices": [int(anchor_index)] * len(offsets_list),
        "dflash_rejected_draft_offsets": offsets_list,
        "dflash_rejected_draft_token_ids": token_ids_list,
        "dflash_rejected_draft_teacher_logprobs": teacher_logprobs_list,
    }
    for key, chunk in chunks_by_key.items():
        values = req.customized_info.setdefault(key, [])
        _pad_token_aligned_metadata(values, prefix_len)
        values.extend([[] for _ in range(committed_token_count)])
        values[-1] = chunk


def append_dflash_replay_block_metadata(
    req: Any,
    committed_token_count: int,
    *,
    anchor_index: int,
    accepted_length: int,
    drafted_length: int,
    output_ids_already_updated: bool,
) -> None:
    """Append one DFLASH replay block record in response-token coordinates."""

    committed_token_count = int(committed_token_count)
    accepted_length = int(accepted_length)
    drafted_length = int(drafted_length)
    if committed_token_count <= 0 or drafted_length <= 0:
        return
    if accepted_length < 0 or accepted_length > drafted_length:
        raise ValueError(
            "DFLASH replay block metadata length mismatch: "
            f"accepted_length={accepted_length}, drafted_length={drafted_length}."
        )

    if req.customized_info is None:
        req.customized_info = {}

    prefix_len = len(req.output_ids)
    if output_ids_already_updated:
        prefix_len -= committed_token_count
    prefix_len = max(0, prefix_len)

    chunks_by_key = {
        "dflash_replay_block_anchor_indices": [int(anchor_index)],
        "dflash_replay_block_accepted_lengths": [accepted_length],
        "dflash_replay_block_drafted_lengths": [drafted_length],
    }
    for key, chunk in chunks_by_key.items():
        values = req.customized_info.setdefault(key, [])
        _pad_token_aligned_metadata(values, prefix_len)
        values.extend([[] for _ in range(committed_token_count)])
        values[-1] = chunk
