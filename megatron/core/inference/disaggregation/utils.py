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
    """Return transfer metadata with a sequence prefix removed."""

    if prefix_count == 0:
        return peer_meta, list(src_block_ids)
    if prefix_count < 0:
        raise ValueError("transfer prefix count must be non-negative")

    trimmed_src_blocks = list(src_block_ids[prefix_count:])
    if not isinstance(peer_meta, dict) or "pp_metas" not in peer_meta:
        return peer_meta, trimmed_src_blocks

    trimmed_meta = dict(peer_meta)
    trimmed_stages = []
    for stage in peer_meta["pp_metas"]:
        stage = dict(stage)
        stage_blocks = list(stage.get("block_ids", []))
        if prefix_count > len(stage_blocks):
            raise ValueError(
                f"cannot remove {prefix_count} blocks from a {len(stage_blocks)}-block PP stage"
            )
        stage["block_ids"] = stage_blocks[prefix_count:]
        trimmed_stages.append(stage)
    trimmed_meta["pp_metas"] = trimmed_stages
    return trimmed_meta, trimmed_src_blocks
