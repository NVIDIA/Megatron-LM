# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared helpers for the disaggregation modules."""

from __future__ import annotations

from typing import Optional, Tuple


def intersect(a: Tuple[int, int], b: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Overlap of two half-open ``[lo, hi)`` ranges, or ``None`` if disjoint."""
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if lo < hi else None


def transfers_for_src(plan, src_rank):
    """Transfers in ``plan`` originating from ``src_rank`` (any KV/Mamba
    reshard transfer -- both expose a ``src_rank`` field)."""
    return [t for t in plan if t.src_rank == src_rank]


def transfers_for_dst(plan, dst_rank):
    """Transfers in ``plan`` destined for ``dst_rank``."""
    return [t for t in plan if t.dst_rank == dst_rank]
