# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared execution path for Gated Delta Product context parallelism."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from megatron.core.ssm.context_parallel.chunkwise import (
    BackwardContextT,
    CPSavedContext,
    LinearAttentionCPBackend,
    LocalContextT,
    chunkwise_cp_backward,
    chunkwise_cp_forward,
)


@dataclass(frozen=True)
class GDPInputs:
    """Rank-local GDP operands passed to a chunkwise-CP backend."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    cu_seqlens: torch.Tensor | None
    num_householder: int
    scale: float


GDPInputGradients = tuple[
    torch.Tensor,  # dq
    torch.Tensor,  # dk
    torch.Tensor,  # dv
    torch.Tensor,  # dg
    torch.Tensor,  # dbeta
]


class GDPChunkwiseContextParallel(torch.autograd.Function):
    """Connect a GDP backend's saved context to PyTorch autograd."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        num_householder,
        scale,
        cp_group,
        backend,
        preceding_rank_start,
        following_rank_stop,
    ):
        """Run chunkwise-CP forward and save the backend context for backward."""
        inputs = GDPInputs(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            num_householder=num_householder,
            scale=scale,
        )
        cp_rank = cp_group.rank()
        result = chunkwise_cp_forward(
            backend=backend,
            inputs=inputs,
            cp_group=cp_group,
            preceding_slice=slice(preceding_rank_start, cp_rank),
        )
        ctx.save_for_backward(*result.saved_context.tensors)
        ctx.saved_context_metadata = result.saved_context.metadata
        ctx.cp_group = cp_group
        ctx.cp_rank = cp_rank
        ctx.following_rank_stop = following_rank_stop
        ctx.backend = backend
        return result.output

    @staticmethod
    def backward(ctx, output_grad):
        """Run chunkwise-CP backward using the saved backend context."""
        saved_context = CPSavedContext(
            tensors=ctx.saved_tensors, metadata=ctx.saved_context_metadata
        )
        dq, dk, dv, dg, dbeta = chunkwise_cp_backward(
            backend=ctx.backend,
            output_grad=output_grad,
            saved_context=saved_context,
            cp_group=ctx.cp_group,
            following_slice=slice(ctx.cp_rank + 1, ctx.following_rank_stop),
        )
        return dq, dk, dv, dg, dbeta, None, None, None, None, None, None, None


class GDPChunkwiseCPBackend(
    LinearAttentionCPBackend[GDPInputs, LocalContextT, BackwardContextT, GDPInputGradients],
    Protocol[LocalContextT, BackwardContextT],
):
    """GDP local-kernel and autograd-adapter contract."""

    autograd_function: type[GDPChunkwiseContextParallel]


@torch.compiler.disable
def gdp_chunkwise_context_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    num_householder: int,
    scale: float,
    cp_group: torch.distributed.ProcessGroup,
    backend: GDPChunkwiseCPBackend[LocalContextT, BackwardContextT],
    preceding_rank_start: int = 0,
    following_rank_stop: int | None = None,
) -> torch.Tensor:
    """Run GDP chunkwise CP through the selected local-kernel backend."""
    if following_rank_stop is None:
        following_rank_stop = cp_group.size()
    cp_rank = cp_group.rank()
    if not 0 <= preceding_rank_start <= cp_rank:
        raise ValueError(
            "preceding_rank_start must be in [0, cp_rank], got "
            f"{preceding_rank_start} for rank {cp_rank}"
        )
    if not cp_rank < following_rank_stop <= cp_group.size():
        raise ValueError(
            "following_rank_stop must be in (cp_rank, cp_size], got "
            f"{following_rank_stop} for rank {cp_rank} and size {cp_group.size()}"
        )
    return backend.autograd_function.apply(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        num_householder,
        scale,
        cp_group,
        backend,
        preceding_rank_start,
        following_rank_stop,
    )
