"""Sequence parallel scatter/gather helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive.ops.sp_ops import (
    AllGatherDim0,
    AllGatherDim0ForNonSPConsumer,
    ScatterToSP,
)

if TYPE_CHECKING:
    from megatron.lite.primitive.parallel.state import ParallelState


def scatter_to_sequence_parallel(x: torch.Tensor, ps: ParallelState) -> torch.Tensor:
    """Scatter [S, B, H] → [S/tp, B, H] for sequence parallel. No-op when tp=1."""
    if ps.tp_size == 1:
        return x
    return ScatterToSP.apply(x, ps.tp_size, ps.tp_rank, ps.tp_group)


def gather_from_sequence_parallel(x: torch.Tensor, ps: ParallelState) -> torch.Tensor:
    """Gather [S/tp, B, H] → [S, B, H] from sequence parallel. No-op when tp=1."""
    if ps.tp_size == 1:
        return x
    return AllGatherDim0.apply(x, ps.tp_size, ps.tp_rank, ps.tp_group)


def gather_for_non_sp_head(x: torch.Tensor, ps: ParallelState) -> torch.Tensor:
    """AllGather for non-SP consumer (e.g. vocab parallel head)."""
    if ps.tp_size == 1:
        return x
    return AllGatherDim0ForNonSPConsumer.apply(x, ps.tp_size, ps.tp_rank, ps.tp_group)


__all__ = [
    "gather_for_non_sp_head",
    "gather_from_sequence_parallel",
    "scatter_to_sequence_parallel",
]
