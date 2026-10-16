# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared helpers for the disaggregation modules."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple


def intersect(a: Tuple[int, int], b: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Overlap of two half-open ``[lo, hi)`` ranges, or ``None`` if disjoint."""
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if lo < hi else None


def transfers_for_src(plan, src_rank):
    """Transfers in ``plan`` originating from ``src_rank`` (any KV/SSM
    reshard transfer -- both expose a ``src_rank`` field)."""
    return [t for t in plan if t.src_rank == src_rank]


def transfers_for_dst(plan, dst_rank):
    """Transfers in ``plan`` destined for ``dst_rank``."""
    return [t for t in plan if t.dst_rank == dst_rank]


def transfer_peer_records(peer_meta: Any, src_block_ids: List[int]) -> List[Tuple[dict, List[int]]]:
    """Normalize flat/TP/PP transfer metadata into peer/block records."""

    def append_metas(raw_metas: Any, default_blocks: List[int]) -> None:
        metas = raw_metas if isinstance(raw_metas, list) else [raw_metas]
        for meta in metas:
            if not isinstance(meta, dict):
                raise ValueError("transfer peer metadata entries must be dictionaries")
            blocks = meta.get("block_ids", default_blocks)
            records.append((meta, [int(block) for block in blocks]))

    records: List[Tuple[dict, List[int]]] = []
    if isinstance(peer_meta, dict) and "pp_metas" in peer_meta:
        for entry in peer_meta["pp_metas"]:
            raw_metas = entry.get("tp_metas", entry)
            blocks = [int(block) for block in entry.get("block_ids", [])]
            append_metas(raw_metas, blocks)
        return records

    if isinstance(peer_meta, dict) and "tp_metas" in peer_meta:
        peer_meta = peer_meta["tp_metas"]
    blocks = [int(block) for block in src_block_ids]
    append_metas(peer_meta, blocks)
    return records


def transfer_block_count(peer_meta: Any, src_block_ids: List[int]) -> int:
    """Return the sequence-block count represented by transfer metadata."""

    records = transfer_peer_records(peer_meta, src_block_ids)
    return len(records[0][1]) if records else 0


def drop_transfer_prefix_blocks(
    peer_meta: Any, src_block_ids: List[int], prefix_count: int
) -> Tuple[Any, List[int]]:
    """Return transfer metadata with a sequence prefix removed.

    Used when the decode engine reuses cached prefix blocks and only pulls the uncached suffix.
    """

    if prefix_count == 0:
        return peer_meta, list(src_block_ids)
    if prefix_count < 0:
        raise ValueError("transfer prefix count must be non-negative")

    def trim_metadata(value: Any) -> Any:
        if isinstance(value, list):
            return [trim_metadata(entry) for entry in value]
        if not isinstance(value, dict):
            return value

        trimmed = dict(value)
        if "block_ids" in trimmed:
            block_ids = list(trimmed["block_ids"])
            if prefix_count > len(block_ids):
                raise ValueError(
                    f"cannot remove {prefix_count} blocks from a {len(block_ids)}-block peer"
                )
            trimmed["block_ids"] = block_ids[prefix_count:]
        for key in ("pp_metas", "tp_metas"):
            if key in trimmed:
                trimmed[key] = trim_metadata(trimmed[key])
        return trimmed

    trimmed_src_blocks = list(src_block_ids[prefix_count:])
    return trim_metadata(peer_meta), trimmed_src_blocks
