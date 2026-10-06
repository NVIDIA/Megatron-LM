# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Chunk-aligned GDN CP with recurrent boundary-state communication.

Only boundary state scans are ordered across CP ranks. WY preparation, output
formation, and local gradient algebra remain distributed. Unpacked inputs only.
Opt-in alternative to parallel affine summaries. State and adjoint recurrences
are ordered across ranks; local WY and output/gradient algebra are parallel.
"""

from typing import Any

import torch
import torch.distributed as dist
from fla.ops.common.chunk_delta_h import (
    chunk_gated_delta_rule_bwd_dhu,
    chunk_gated_delta_rule_fwd_h,
)
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from megatron.core.tensor_parallel.mappings import all_to_all


def overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Return the number of tokens shared by two half-open intervals."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def repartition(
    x: torch.Tensor,
    source_length: int,
    target_length: int,
    true_length: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Differentiable contiguous all-to-all, including empty receive ranges.

    Input/output are [B,T,...]. Only real tokens are communicated. Padding is
    appended after the true global end, never inside an algorithmic chunk.
    """
    if source_length == target_length:
        return x
    rank, size = group.rank(), group.size()
    src = (rank * source_length, min((rank + 1) * source_length, true_length))
    dst = (rank * target_length, min((rank + 1) * target_length, true_length))
    sends = [
        overlap(src, (r * target_length, min((r + 1) * target_length, true_length)))
        for r in range(size)
    ]
    recvs = [
        overlap(dst, (r * source_length, min((r + 1) * source_length, true_length)))
        for r in range(size)
    ]
    batch = x.shape[0]
    token_major = x.transpose(0, 1)[: sum(sends)].contiguous()
    received = all_to_all(group, token_major, output_split_sizes_=recvs, input_split_sizes=sends)
    if received.shape[0] < target_length:
        received = torch.cat(
            (received, received.new_zeros(target_length - received.shape[0], batch, *x.shape[2:])),
            dim=0,
        )
    return received.transpose(0, 1).contiguous()


class RecurrentBoundaryGDN(torch.autograd.Function):
    """Run local GDN kernels with ordered boundary-state and adjoint communication."""

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """Prepare local blocks, receive the actual state, and send its successor."""
        rank, size = group.rank(), group.size()
        gc = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2)
        w, u, A = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=gc, beta=beta, chunk_size=64)
        state = torch.zeros(
            (q.shape[0], v.shape[2], k.shape[3], v.shape[3]), device=q.device, dtype=torch.float32
        )
        if rank > 0:
            dist.recv(state, src=dist.get_global_rank(group, rank - 1), group=group)
        h, v_new, final = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=gc,
            initial_state=state,
            output_final_state=rank < size - 1,
            chunk_size=64,
        )
        if rank < size - 1:
            dist.send(final.contiguous(), dst=dist.get_global_rank(group, rank + 1), group=group)
        output = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=gc, scale=scale, chunk_size=64)
        ctx.save_for_backward(q, k, v, gc, beta, A, state)
        ctx.scale = scale
        ctx.group = group
        ctx.g_dtype = g.dtype
        return output

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx: Any, do: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """Propagate the boundary adjoint before completing local input gradients."""
        q, k, v, g, beta, A, state = ctx.saved_tensors
        group, scale = ctx.group, ctx.scale
        rank, size = group.rank(), group.size()
        # Recompute and prepare the local contribution before receiving dS.
        w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g)
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=g, initial_state=state, output_final_state=False, chunk_size=64
        )
        dv = chunk_bwd_dv_local(q=q, k=k, g=g, do=do, scale=scale, chunk_size=64)
        dht = None
        if rank < size - 1:
            dht = torch.empty_like(state)
            dist.recv(dht, src=dist.get_global_rank(group, rank + 1), group=group)
        dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
            q=q, k=k, w=w, g=g, h0=state, dht=dht, do=do, dv=dv, scale=scale, chunk_size=64
        )
        if rank > 0:
            dist.send(dh0.contiguous(), dst=dist.get_global_rank(group, rank - 1), group=group)
        dq, dk, dw, dg = chunk_bwd_dqkwg(
            q=q, k=k, v=v_new, w=w, g=g, h=h, dv=dv, do=do, dh=dh, scale=scale, chunk_size=64
        )
        dk2, dv, db, dg2 = prepare_wy_repr_bwd(k=k, v=v, beta=beta, g=g, A=A, dw=dw, du=dv)
        dk.add_(dk2)
        dg.add_(dg2)
        dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(ctx.g_dtype), db.to(beta), None, None


@torch.compiler.disable
def gdn_recurrent_context_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cp_group: dist.ProcessGroup,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply native GDN to contiguous sequence shards with recurrent state passing.

    Args:
        q: Queries of shape ``[B, T_local, H_q, K]``.
        k: Keys with the same shape as q.
        v: Values of shape ``[B, T_local, H_v, V]``.
        g: Log-decay gates of shape ``[B, T_local, H_v]``.
        beta: Write strengths with the same shape as g.
        scale: Query scaling factor, usually ``K**-0.5``.
        cp_group: Process group ordered by contiguous sequence shard.
        cu_seqlens: Reserved for packed inputs; must be None in this implementation.

    Returns:
        Local output of shape ``[B, T_local, H_v, V]`` in the original shard layout.

    Raises:
        ValueError: If packed inputs or an empty local shard are supplied.
    """
    if cu_seqlens is not None:
        raise ValueError("GDN recurrent CP currently supports unpacked inputs only")
    length = q.shape[1]
    if length <= 0:
        raise ValueError("GDN recurrent CP requires a nonempty input shard")
    full_length = length * cp_group.size()
    aligned = (length + 63) // 64 * 64
    # Preserve original global 64-token blocks. Padding carries beta=0, g=0,
    # q=k=v=0 and is placed after the real sequence only.
    inputs = [repartition(x, length, aligned, full_length, cp_group) for x in (q, k, v, g, beta)]
    output = RecurrentBoundaryGDN.apply(*inputs, scale, cp_group)
    return repartition(output, aligned, length, full_length, cp_group)
