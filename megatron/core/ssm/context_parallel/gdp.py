# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Adapted from the Gated Delta Product context-parallel implementation in
# flash-linear-attention (https://github.com/fla-org/flash-linear-attention).
#
# Licensed under the MIT license; see the LICENSE file in the repository root.

"""FLA backend for Gated Delta Product chunkwise context parallelism."""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    from fla.ops.common.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_dhu,
        chunk_gated_delta_rule_fwd_h,
    )
    from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local
    from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from fla.ops.cp.chunk_delta_h import (
        merge_fwd_bwd_kernel,
        pre_process_bwd_kernel_merged,
        pre_process_fwd_kernel_merged,
    )
    from fla.ops.gated_delta_product.chunk_deltaproduct_h import chunk_gated_delta_product_fwd_h
    from fla.ops.gated_delta_product.chunk_deltaproduct_o import chunk_gated_delta_product_fwd_o
    from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
    from fla.ops.utils import chunk_local_cumsum, solve_tril
    from fla.ops.utils.constant import RCP_LN2
    from fla.ops.utils.index import prepare_chunk_indices
    from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard, tensor_cache
except ImportError as exc:
    # The caller guards this optional backend; this marker handles direct imports in CI.
    raise ImportError("UnavailableError: FLA GDP chunkwise CP backend is unavailable") from exc

from megatron.core.ssm.context_parallel.chunkwise import (
    CPBackwardPackedSummary,
    CPBackwardSummary,
    CPForwardPackedSummary,
    CPForwardSummary,
    CPSavedContext,
    LinearAttentionCPBackend,
    chunkwise_cp_backward,
    chunkwise_cp_forward,
)


@tensor_cache
def _expand_cu_seqlens(cu_seqlens: torch.Tensor, num_householder: int) -> torch.Tensor:
    if num_householder == 1:
        return cu_seqlens
    return cu_seqlens * num_householder


@dataclass(frozen=True)
class GDPInputs:
    """Local FLA GDP operands passed to the CP backend."""

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


@dataclass(frozen=True)
class GDPSavedInputs:
    """GDP operands retained for the FLA backward kernels."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    beta: torch.Tensor
    cu_seqlens_dp: torch.Tensor | None
    num_householder: int
    scale: float
    g_dtype: torch.dtype


@dataclass(frozen=True)
class GDPLocalContext:
    """FLA GDP workspace passed from forward prepare to forward apply."""

    inputs: GDPInputs
    g_cumsum: torch.Tensor
    g_interleaved: torch.Tensor
    A: torch.Tensor
    w: torch.Tensor
    u: torch.Tensor
    cu_seqlens_dp: torch.Tensor | None
    chunk_indices: torch.Tensor | None
    chunk_indices_dp: torch.Tensor | None


@dataclass(frozen=True)
class GDPSavedMetadata:
    """Non-tensor GDP state saved for backward."""

    num_householder: int
    scale: float
    has_cu_seqlens: bool
    g_dtype: torch.dtype


@dataclass(frozen=True)
class GDPBackwardContext:
    """GDP state passed from backward prepare to backward apply."""

    inputs: GDPSavedInputs
    g_interleaved: torch.Tensor
    A: torch.Tensor
    initial_state: torch.Tensor
    chunk_indices_dp: torch.Tensor | None
    q_interleaved: torch.Tensor
    output_grad_interleaved: torch.Tensor
    w: torch.Tensor
    h: torch.Tensor
    v_new: torch.Tensor
    dv: torch.Tensor


class FLAGatedDeltaProductCPBackend:
    """FLA implementation of the linear-attention chunkwise-CP interface."""

    def cp_forward_prepare(self, inputs: GDPInputs) -> tuple[CPForwardSummary, GDPLocalContext]:
        """Compute the local state summary and reusable FLA forward workspace."""
        local_context = _prepare_fla_forward(inputs)
        packed_summary = _compute_fragment_summary(
            k=inputs.k,
            w=local_context.w,
            u=local_context.u,
            g=local_context.g_interleaved,
            cu_seqlens=local_context.cu_seqlens_dp,
        )
        return CPForwardPackedSummary(packed=packed_summary), local_context

    def cp_forward_apply(
        self, local_context: GDPLocalContext, preceding_summaries: CPForwardSummary
    ) -> tuple[torch.Tensor, CPSavedContext]:
        """Compose the incoming state and compute the local GDP output."""
        inputs = local_context.inputs
        initial_state = _build_initial_state(inputs, preceding_summaries)
        h, v_new, _ = chunk_gated_delta_product_fwd_h(
            k=inputs.k,
            w=local_context.w,
            u=local_context.u,
            g=local_context.g_interleaved,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=local_context.cu_seqlens_dp,
            num_householder=inputs.num_householder,
            chunk_indices=local_context.chunk_indices,
        )
        output = chunk_gated_delta_product_fwd_o(
            q=inputs.q,
            k=inputs.k,
            v=v_new,
            h=h,
            g=local_context.g_cumsum,
            scale=inputs.scale,
            cu_seqlens=inputs.cu_seqlens,
            num_householder=inputs.num_householder,
            chunk_indices=local_context.chunk_indices,
        )

        saved_tensors = [
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.beta,
            local_context.g_interleaved,
            local_context.A,
            initial_state,
        ]
        if inputs.cu_seqlens is not None:
            assert local_context.cu_seqlens_dp is not None
            assert local_context.chunk_indices_dp is not None
            saved_tensors.extend((local_context.cu_seqlens_dp, local_context.chunk_indices_dp))
        return output, CPSavedContext(
            tensors=tuple(saved_tensors),
            metadata=GDPSavedMetadata(
                num_householder=inputs.num_householder,
                scale=inputs.scale,
                has_cu_seqlens=inputs.cu_seqlens is not None,
                g_dtype=inputs.g.dtype,
            ),
        )

    def cp_backward_prepare(
        self, output_grad: torch.Tensor, saved_context: CPSavedContext
    ) -> tuple[CPBackwardSummary, GDPBackwardContext]:
        """Compute the local output contribution to the incoming-state gradient."""
        inputs, g_interleaved, A, initial_state, chunk_indices_dp = _restore_saved_context(
            saved_context
        )
        q_interleaved = _interleave_last_update(inputs.q, inputs.num_householder)
        output_grad_interleaved = _interleave_last_update(output_grad, inputs.num_householder)

        w, u = recompute_w_u_fwd(
            k=inputs.k,
            v=inputs.v,
            beta=inputs.beta,
            A=A,
            g=g_interleaved,
            cu_seqlens=inputs.cu_seqlens_dp,
            chunk_indices=chunk_indices_dp,
        )
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=inputs.k,
            w=w,
            u=u,
            g=g_interleaved,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=inputs.cu_seqlens_dp,
            chunk_indices=chunk_indices_dp,
        )
        dv = chunk_bwd_dv_local(
            q=q_interleaved,
            k=inputs.k,
            g=g_interleaved,
            do=output_grad_interleaved,
            scale=inputs.scale,
            cu_seqlens=inputs.cu_seqlens_dp,
            chunk_indices=chunk_indices_dp,
        )
        packed_summary = _compute_backward_fragment_summary(
            q=q_interleaved,
            k=inputs.k,
            w=w,
            g=g_interleaved,
            do=output_grad_interleaved,
            dv=dv,
            scale=inputs.scale,
            cu_seqlens=inputs.cu_seqlens_dp,
        )
        return CPBackwardPackedSummary(packed=packed_summary), GDPBackwardContext(
            inputs=inputs,
            g_interleaved=g_interleaved,
            A=A,
            initial_state=initial_state,
            chunk_indices_dp=chunk_indices_dp,
            q_interleaved=q_interleaved,
            output_grad_interleaved=output_grad_interleaved,
            w=w,
            h=h,
            v_new=v_new,
            dv=dv,
        )

    def cp_backward_apply(
        self, backward_context: GDPBackwardContext, following_summaries: CPBackwardSummary
    ) -> GDPInputGradients:
        """Compose the outgoing-state gradient and compute all local GDP gradients."""
        final_state_grad = _build_final_state_grad(backward_context.inputs, following_summaries)
        return _apply_fla_backward(backward_context, final_state_grad)


def _prepare_fla_forward(inputs: GDPInputs) -> GDPLocalContext:
    """Build the GDP WY representation used by both summary and output kernels."""
    cu_seqlens_dp = (
        _expand_cu_seqlens(inputs.cu_seqlens, inputs.num_householder)
        if inputs.cu_seqlens is not None
        else None
    )
    chunk_indices = (
        prepare_chunk_indices(inputs.cu_seqlens, 64) if inputs.cu_seqlens is not None else None
    )
    chunk_indices_dp = (
        prepare_chunk_indices(cu_seqlens_dp, 64) if cu_seqlens_dp is not None else None
    )

    g_interleaved = inputs.g.new_zeros(
        inputs.g.shape[0],
        inputs.g.shape[1],
        inputs.num_householder,
        inputs.g.shape[2],
        dtype=torch.float32,
    )
    g_interleaved[:, :, 0] = inputs.g
    g_interleaved = g_interleaved.flatten(1, 2).contiguous()
    g_cumsum = chunk_local_cumsum(
        inputs.g,
        chunk_size=64,
        scale=RCP_LN2,
        cu_seqlens=inputs.cu_seqlens,
        output_dtype=torch.float32,
        chunk_indices=chunk_indices,
    )
    g_interleaved = chunk_local_cumsum(
        g_interleaved,
        chunk_size=64,
        scale=RCP_LN2,
        cu_seqlens=cu_seqlens_dp,
        output_dtype=torch.float32,
        chunk_indices=chunk_indices_dp,
    )
    A = chunk_scaled_dot_kkt_fwd(
        k=inputs.k,
        g=g_interleaved,
        beta=inputs.beta,
        cu_seqlens=cu_seqlens_dp,
        output_dtype=torch.float32,
        chunk_indices=chunk_indices_dp,
    )
    A = solve_tril(
        A=A, cu_seqlens=cu_seqlens_dp, chunk_indices=chunk_indices_dp, output_dtype=inputs.k.dtype
    )
    w, u = recompute_w_u_fwd(
        k=inputs.k,
        v=inputs.v,
        beta=inputs.beta,
        A=A,
        g=g_interleaved,
        cu_seqlens=cu_seqlens_dp,
        chunk_indices=chunk_indices_dp,
    )
    return GDPLocalContext(
        inputs=inputs,
        g_cumsum=g_cumsum,
        g_interleaved=g_interleaved,
        A=A,
        w=w,
        u=u,
        cu_seqlens_dp=cu_seqlens_dp,
        chunk_indices=chunk_indices,
        chunk_indices_dp=chunk_indices_dp,
    )


def _compute_fragment_summary(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """Compute packed ``[delta_S | M]`` for the rank-local trailing boundary fragment."""
    batch, length, key_heads, key_dim = k.shape
    value_heads = u.shape[2]
    value_dim = u.shape[-1]
    if u.shape[:2] != (batch, length):
        raise ValueError("FLA GDP chunkwise CP requires key and value sequence shapes to match")
    if value_heads % key_heads != 0:
        raise ValueError(
            "FLA GDP chunkwise CP requires the number of value heads to be divisible by "
            f"the number of key heads, got {value_heads} and {key_heads}"
        )
    if key_dim > 256:
        raise ValueError(f"FLA GDP chunkwise CP supports key dimensions up to 256, got {key_dim}")

    block_size = 32 if key_dim <= 64 else 64
    grid_columns = triton.cdiv(value_dim, block_size) + triton.cdiv(key_dim, block_size)
    if cu_seqlens is None:
        summary = k.new_zeros(batch, value_heads, key_dim, value_dim + key_dim, dtype=torch.float32)
        grid = (grid_columns, value_heads, batch)
        boundary_cu_seqlens = None
        multiple_sequences = True
    else:
        boundary_cu_seqlens = cu_seqlens[-2:]
        summary = k.new_zeros(value_heads, key_dim, value_dim + key_dim, dtype=torch.float32)
        grid = (grid_columns, value_heads)
        multiple_sequences = False

    pre_process_fwd_kernel_merged[grid](
        k=k,
        v=u,
        w=w,
        g=g,
        gk=None,
        bg=None,
        u=u,
        hm=summary,
        cu_seqlens=boundary_cu_seqlens,
        T=length,
        H=key_heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BT=64,
        BK1=triton.next_power_of_2(key_dim),
        BLOCK_SIZE=block_size,
        MULTI_SEQS=multiple_sequences,
    )
    return summary


def _compute_backward_fragment_summary(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """Compute packed ``[gamma | M^T]`` for the rank-local leading boundary fragment."""
    batch, length, query_heads, key_dim = q.shape
    value_heads = do.shape[2]
    value_dim = do.shape[-1]
    if value_heads % query_heads != 0:
        raise ValueError(
            "FLA GDP chunkwise CP requires the number of value heads to be divisible by "
            f"the number of query heads, got {value_heads} and {query_heads}"
        )
    if key_dim > 256:
        raise ValueError(f"FLA GDP chunkwise CP supports key dimensions up to 256, got {key_dim}")

    block_size = 32 if key_dim <= 64 else 64
    grid = (triton.cdiv(value_dim, block_size) + triton.cdiv(key_dim, block_size), value_heads)

    def launch(
        q_fragment: torch.Tensor,
        k_fragment: torch.Tensor,
        w_fragment: torch.Tensor,
        g_fragment: torch.Tensor,
        do_fragment: torch.Tensor,
        dv_fragment: torch.Tensor,
        summary: torch.Tensor,
        boundary_cu_seqlens: torch.Tensor | None,
    ) -> None:
        pre_process_bwd_kernel_merged[grid](
            q=q_fragment,
            k=k_fragment,
            w=w_fragment,
            g=g_fragment,
            gk=None,
            do=do_fragment,
            dhm=summary,
            dv=dv_fragment,
            cu_seqlens=boundary_cu_seqlens,
            scale=scale,
            T=length,
            H=query_heads,
            HV=value_heads,
            K=key_dim,
            V=value_dim,
            BT=64,
            BK1=triton.next_power_of_2(key_dim),
            BLOCK_SIZE=block_size,
            USE_BG=False,
        )

    if cu_seqlens is None:
        merged_summary = q.new_zeros(
            batch, value_heads, key_dim, value_dim + key_dim, dtype=torch.float32
        )
        # Unlike the forward prepare kernel, FLA's backward prepare kernel has no MULTI_SEQS
        # mode and handles one sequence per launch. BLH treats every batch element as an
        # independent sequence, so launch the same FLA kernel for each element.
        for batch_index in range(batch):
            batch_slice = slice(batch_index, batch_index + 1)
            launch(
                q[batch_slice],
                k[batch_slice],
                w[batch_slice],
                g[batch_slice],
                do[batch_slice],
                dv[batch_slice],
                merged_summary[batch_index],
                None,
            )
    else:
        merged_summary = q.new_zeros(value_heads, key_dim, value_dim + key_dim, dtype=torch.float32)
        launch(q, k, w, g, do, dv, merged_summary, cu_seqlens[:2])

    return merged_summary


def _merge_affine_summaries(
    packed_summaries: torch.Tensor, output: torch.Tensor, forward: bool
) -> torch.Tensor:
    """Compose an already-selected summary slice from zero into ``output``."""
    summary_count = packed_summaries.shape[0]
    if summary_count == 0:
        output.zero_()
        return output

    heads, key_dim, value_dim = output.shape[-3:]
    merge_heads = heads
    output_state = output
    if output.ndim == 4:
        batch = output.shape[0]
        merge_heads = batch * heads
        packed_summaries = packed_summaries.flatten(1, 2)
        output_state = output.view(merge_heads, key_dim, value_dim)

    def grid(meta):
        return (triton.cdiv(value_dim, meta["BV"]), merge_heads)

    # The interface passes only the selected summary slice, so rebase FLA's CP merge indices onto
    # that slice. Forward visits [0, n); backward visits [n - 1, ..., 0].
    merge_rank = summary_count if forward else -1

    merge_fwd_bwd_kernel[grid](
        h=output_state,
        ag_hm=packed_summaries,
        pre_or_post_num_ranks=summary_count,
        rank=merge_rank,
        seq_offsets=None,
        init_offsets=None,
        h0_seq_ids=None,
        h0=None,
        HV=merge_heads,
        K=key_dim,
        V=value_dim,
        BK=triton.next_power_of_2(key_dim),
        FORWARD=forward,
        INTRACARD_MODE=False,
        NUM_SEQ_ENTRIES=0,
        STATE_V_FIRST=False,
    )
    return output


def _merge_forward_summaries(summaries: CPForwardSummary, output: torch.Tensor) -> torch.Tensor:
    if not isinstance(summaries, CPForwardPackedSummary):
        raise TypeError("The FLA GDP backend requires packed forward summaries")
    return _merge_affine_summaries(summaries.packed, output, forward=True)


def _merge_backward_summaries(
    summaries: CPBackwardSummary, output_grad: torch.Tensor
) -> torch.Tensor:
    if not isinstance(summaries, CPBackwardPackedSummary):
        raise TypeError("The FLA GDP backend requires packed backward summaries")
    return _merge_affine_summaries(summaries.packed, output_grad, forward=False)


def _state_shape(
    q: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor | None
) -> tuple[int, int, int, int]:
    sequence_count = q.shape[0] if cu_seqlens is None else cu_seqlens.numel() - 1
    return sequence_count, v.shape[2], q.shape[3], v.shape[3]


def _build_initial_state(inputs: GDPInputs, summaries: CPForwardSummary) -> torch.Tensor:
    initial_state = inputs.q.new_zeros(
        _state_shape(inputs.q, inputs.v, inputs.cu_seqlens), dtype=torch.float32
    )
    if inputs.cu_seqlens is None:
        return _merge_forward_summaries(summaries, initial_state)
    _merge_forward_summaries(summaries, initial_state[0])
    return initial_state


def _build_final_state_grad(inputs: GDPSavedInputs, summaries: CPBackwardSummary) -> torch.Tensor:
    final_state_grad = inputs.q.new_zeros(
        _state_shape(inputs.q, inputs.v, inputs.cu_seqlens_dp), dtype=torch.float32
    )
    if inputs.cu_seqlens_dp is None:
        return _merge_backward_summaries(summaries, final_state_grad)
    _merge_backward_summaries(summaries, final_state_grad[-1])
    return final_state_grad


def _restore_saved_context(
    saved_context: CPSavedContext,
) -> tuple[GDPSavedInputs, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    metadata = saved_context.metadata
    if not isinstance(metadata, GDPSavedMetadata):
        raise TypeError("FLAGatedDeltaProductCPBackend received incompatible saved metadata")
    expected_count = 7 + 2 * metadata.has_cu_seqlens
    if len(saved_context.tensors) != expected_count:
        raise ValueError("FLAGatedDeltaProductCPBackend received an incompatible tensor bundle")

    q, k, v, beta, g_interleaved, A, initial_state = saved_context.tensors[:7]
    cu_seqlens_dp = saved_context.tensors[7] if metadata.has_cu_seqlens else None
    chunk_indices_dp = saved_context.tensors[8] if metadata.has_cu_seqlens else None
    return (
        GDPSavedInputs(
            q=q,
            k=k,
            v=v,
            beta=beta,
            cu_seqlens_dp=cu_seqlens_dp,
            num_householder=metadata.num_householder,
            scale=metadata.scale,
            g_dtype=metadata.g_dtype,
        ),
        g_interleaved,
        A,
        initial_state,
        chunk_indices_dp,
    )


def _interleave_last_update(tensor: torch.Tensor, num_householder: int) -> torch.Tensor:
    interleaved = tensor.new_zeros(
        tensor.shape[0], tensor.shape[1], num_householder, tensor.shape[2], tensor.shape[3]
    )
    interleaved[:, :, -1] = tensor
    return interleaved.flatten(1, 2).contiguous()


def _apply_fla_backward(
    context: GDPBackwardContext, final_state_grad: torch.Tensor
) -> GDPInputGradients:
    """Apply FLA's GDP backward kernels using Megatron-composed boundary state gradients."""
    inputs = context.inputs

    dh, _, dv = chunk_gated_delta_rule_bwd_dhu(
        q=context.q_interleaved,
        k=inputs.k,
        w=context.w,
        g=context.g_interleaved,
        h0=None,
        dht=final_state_grad,
        do=context.output_grad_interleaved,
        dv=context.dv,
        scale=inputs.scale,
        cu_seqlens=inputs.cu_seqlens_dp,
        chunk_indices=context.chunk_indices_dp,
    )
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=context.q_interleaved,
        k=inputs.k,
        v=context.v_new,
        w=context.w,
        g=context.g_interleaved,
        h=context.h,
        dv=dv,
        do=context.output_grad_interleaved,
        dh=dh,
        scale=inputs.scale,
        cu_seqlens=inputs.cu_seqlens_dp,
        chunk_indices=context.chunk_indices_dp,
    )
    assert dw is not None
    assert dg is not None
    dk2, dv, db, dg2 = prepare_wy_repr_bwd(
        k=inputs.k,
        v=inputs.v,
        beta=inputs.beta,
        g=context.g_interleaved,
        A=context.A,
        dw=dw,
        du=dv,
        cu_seqlens=inputs.cu_seqlens_dp,
        chunk_indices=context.chunk_indices_dp,
    )
    assert dg2 is not None
    dk.add_(dk2)
    dg.add_(dg2)
    dg = chunk_local_cumsum(
        dg,
        chunk_size=64,
        reverse=True,
        cu_seqlens=inputs.cu_seqlens_dp,
        chunk_indices=context.chunk_indices_dp,
    )

    dq = dq.unflatten(1, (inputs.q.shape[1], inputs.num_householder))[:, :, -1].contiguous()
    dg = dg.unflatten(1, (inputs.q.shape[1], inputs.num_householder))[:, :, 0].contiguous()
    return (
        dq.to(inputs.q),
        dk.to(inputs.k),
        dv.to(inputs.v),
        dg.to(dtype=inputs.g_dtype),
        db.to(inputs.beta),
    )


class _GDPChunkwiseContextParallel(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
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
    @input_guard
    @autocast_custom_bwd
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
    backend: LinearAttentionCPBackend[
        GDPInputs, GDPLocalContext, GDPBackwardContext, GDPInputGradients
    ],
    preceding_rank_start: int = 0,
    following_rank_stop: int | None = None,
) -> torch.Tensor:
    """Run FLA GDP with Megatron-owned chunkwise-CP communication."""
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
    return _GDPChunkwiseContextParallel.apply(
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
