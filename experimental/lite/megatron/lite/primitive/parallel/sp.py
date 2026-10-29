# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Sequence parallel scatter/gather helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    from megatron.lite.primitive.parallel.state import ParallelState


def _ag_dim0(x: torch.Tensor, tp_size: int, group: dist.ProcessGroup) -> torch.Tensor:
    out = torch.empty(tp_size * x.shape[0], x.shape[1], x.shape[2], dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(out, x.contiguous(), group=group)
    return out


def _rs_dim0(x: torch.Tensor, local_seq: int, group: dist.ProcessGroup) -> torch.Tensor:
    out = torch.empty(local_seq, x.shape[1], x.shape[2], dtype=x.dtype, device=x.device)
    dist.reduce_scatter_tensor(out, x.contiguous(), group=group)
    return out


class _AllGatherDim0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp_size, tp_rank, group):
        del tp_rank
        ctx.group = group
        ctx.local_seq = x.shape[0]
        return _ag_dim0(x, tp_size, group)

    @staticmethod
    def backward(ctx, grad):
        out = _rs_dim0(grad, ctx.local_seq, ctx.group)
        return out, None, None, None


class _AllGatherDim0ForNonSPConsumer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp_size, tp_rank, group):
        ctx.tp_rank = tp_rank
        ctx.local_seq = x.shape[0]
        return _ag_dim0(x, tp_size, group)

    @staticmethod
    def backward(ctx, grad):
        start = ctx.tp_rank * ctx.local_seq
        return grad[start : start + ctx.local_seq].contiguous(), None, None, None


class _ScatterToSP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp_size, tp_rank, group):
        ctx.tp_size = tp_size
        ctx.group = group
        local_seq = x.shape[0] // tp_size
        start = tp_rank * local_seq
        return x[start : start + local_seq, :, :].contiguous()

    @staticmethod
    def backward(ctx, grad):
        return _ag_dim0(grad, ctx.tp_size, ctx.group), None, None, None


def scatter_to_sequence_parallel(x: torch.Tensor, ps: ParallelState) -> torch.Tensor:
    """Scatter [S, B, H] → [S/tp, B, H] for sequence parallel. No-op when tp=1."""
    if ps.tp_size == 1:
        return x
    return _ScatterToSP.apply(x, ps.tp_size, ps.tp_rank, ps.tp_group)


def gather_from_sequence_parallel(x: torch.Tensor, ps: ParallelState) -> torch.Tensor:
    """Gather [S/tp, B, H] → [S, B, H] from sequence parallel. No-op when tp=1."""
    if ps.tp_size == 1:
        return x
    return _AllGatherDim0.apply(x, ps.tp_size, ps.tp_rank, ps.tp_group)


def gather_for_non_sp_head(x: torch.Tensor, ps: ParallelState) -> torch.Tensor:
    """AllGather for non-SP consumer (e.g. vocab parallel head)."""
    if ps.tp_size == 1:
        return x
    return _AllGatherDim0ForNonSPConsumer.apply(x, ps.tp_size, ps.tp_rank, ps.tp_group)


__all__ = [
    "gather_for_non_sp_head",
    "gather_from_sequence_parallel",
    "scatter_to_sequence_parallel",
]
