"""Sequence-parallel autograd primitives for non-TE layers (embedding, scatter/gather).

AllGather/ReduceScatter operate on the sequence dimension (dim 0) with layout
[S, B, H]. NCCL operates on dim 0 directly — no transpose needed.
TE-based layers use TE's native sequence_parallel=True instead.
"""

from __future__ import annotations

import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]


def _ag_dim0(x: torch.Tensor, tp_size: int, group: dist.ProcessGroup) -> torch.Tensor:
    """AllGather on dim 0: [S/tp, B, H] → [S, B, H]."""
    out = torch.empty(tp_size * x.shape[0], x.shape[1], x.shape[2], dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(out, x.contiguous(), group=group)
    return out


def _rs_dim0(x: torch.Tensor, local_seq: int, group: dist.ProcessGroup) -> torch.Tensor:
    """ReduceScatter on dim 0: [S, B, H] → [S/tp, B, H]."""
    out = torch.empty(local_seq, x.shape[1], x.shape[2], dtype=x.dtype, device=x.device)
    dist.reduce_scatter_tensor(out, x.contiguous(), group=group)
    return out


class AllGatherDim0(torch.autograd.Function):
    """AllGather [S/tp, B, H] → [S, B, H]. Backward: ReduceScatter."""

    @staticmethod
    def forward(ctx, x, tp_size, tp_rank, group):
        ctx.group = group
        ctx.local_seq = x.shape[0]
        return _ag_dim0(x, tp_size, group)

    @staticmethod
    def backward(ctx, grad):
        out = _rs_dim0(grad, ctx.local_seq, ctx.group)
        return out, None, None, None


class ReduceScatterDim0(torch.autograd.Function):
    """ReduceScatter [S, B, H] → [S/tp, B, H]. Backward: AllGather."""

    @staticmethod
    def forward(ctx, x, tp_size, tp_rank, group):
        ctx.tp_size = tp_size
        ctx.group = group
        local_seq = x.shape[0] // tp_size
        return _rs_dim0(x, local_seq, ctx.group)

    @staticmethod
    def backward(ctx, grad):
        return _ag_dim0(grad, ctx.tp_size, ctx.group), None, None, None


class AllGatherDim0ForNonSPConsumer(torch.autograd.Function):
    """AllGather [S/tp, B, H] → [S, B, H]. Backward: Scatter (no reduce)."""

    @staticmethod
    def forward(ctx, x, tp_size, tp_rank, group):
        ctx.tp_rank = tp_rank
        ctx.local_seq = x.shape[0]
        return _ag_dim0(x, tp_size, group)

    @staticmethod
    def backward(ctx, grad):
        start = ctx.tp_rank * ctx.local_seq
        return grad[start : start + ctx.local_seq].contiguous(), None, None, None


class ScatterToSP(torch.autograd.Function):
    """Scatter [S, B, H] → [S/tp, B, H] (no comm). Backward: AllGather."""

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


__all__ = ["AllGatherDim0", "AllGatherDim0ForNonSPConsumer", "ReduceScatterDim0", "ScatterToSP"]
