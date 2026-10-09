# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.
#
# The combined-scan forward/backward below adapts mamba_ssm's ssd_combined.py,
# and state passing adapts ssd_state_passing.py. The upstream portions are
# licensed under Apache-2.0; see the Mamba license in the repository LICENSE.

"""Differentiable SSD scan over chunk-aligned shared-prefix forests.

Only inter-chunk state passing changes topology. All intra-chunk operations
reuse the installed mamba_ssm kernels. The first segment of each root is its
prefix; its remaining segments are siblings, each initialized from the prefix
terminal state. No prefix recurrence is replayed and no sibling is padded to
another sibling's length.
"""

import torch
import triton
import triton.language as tl
from einops import rearrange
from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_bwd, _bmm_chunk_fwd
from mamba_ssm.ops.triton.ssd_chunk_scan import (
    _chunk_scan_bwd_dC,
    _chunk_scan_bwd_dcb,
    _chunk_scan_bwd_ddAcs_stable,
    _chunk_scan_bwd_dstates,
    _chunk_scan_fwd,
)
from mamba_ssm.ops.triton.ssd_chunk_state import (
    _chunk_cumsum_bwd,
    _chunk_cumsum_fwd,
    _chunk_state_bwd_db,
    _chunk_state_fwd,
)
from mamba_ssm.ops.triton.ssd_combined import _chunk_scan_chunk_state_bwd_dx
from torch import Tensor
from torch.autograd.function import once_differentiable

_STATE_BLOCK = 256


# Strides are runtime arguments so a new chunk count does not compile a new kernel; the dA
# head stride (nchunks * chunk_size) is not specialized at all. IDX is tl.int32, or tl.int64
# when a tensor reaches 2^31 elements (see _index_dtype).
@triton.jit(do_not_specialize=["stride_a_head"])
def _forest_state_fwd_kernel(
    chunk_states,
    dA,
    segment_chunks,
    root_segments,
    out,
    dim: tl.constexpr,
    stride_chunk,
    stride_head,
    stride_dim,
    stride_a_head,
    stride_a_chunk,
    stride_out_chunk,
    stride_out_head,
    BLOCK: tl.constexpr,
    IDX: tl.constexpr,
):
    tile = tl.program_id(0)
    root = tl.program_id(1)
    head = tl.program_id(2).to(IDX)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < dim
    state_base = chunk_states + head * stride_head + offsets * stride_dim
    out_base = out + head * stride_out_head + offsets
    a_base = dA + head * stride_a_head
    first_segment = tl.load(root_segments + root)
    stop_segment = tl.load(root_segments + root + 1)
    prefix_start = tl.load(segment_chunks + first_segment).to(IDX)
    prefix_stop = tl.load(segment_chunks + first_segment + 1).to(IDX)

    # Keep the FP32 accumulator live across chunks, including at the fork.
    # The stored entries have the same rounding as upstream state passing.
    prefix_state = tl.full((BLOCK,), 0.0, tl.float32)
    for chunk in range(prefix_start, prefix_stop):
        tl.store(out_base + chunk * stride_out_chunk, prefix_state, valid)
        delta = tl.load(state_base + chunk * stride_chunk, valid, 0.0).to(tl.float32)
        scale = tl.exp(tl.load(a_base + chunk * stride_a_chunk).to(tl.float32))
        prefix_state = scale * prefix_state + delta

    for segment in range(first_segment + 1, stop_segment):
        start = tl.load(segment_chunks + segment).to(IDX)
        stop = tl.load(segment_chunks + segment + 1).to(IDX)
        state = prefix_state
        for chunk in range(start, stop):
            tl.store(out_base + chunk * stride_out_chunk, state, valid)
            delta = tl.load(state_base + chunk * stride_chunk, valid, 0.0).to(tl.float32)
            scale = tl.exp(tl.load(a_base + chunk * stride_a_chunk).to(tl.float32))
            state = scale * state + delta


# As in the forward kernel; the dA_tiles head stride (nchunks * tiles) also varies per pack.
@triton.jit(do_not_specialize=["stride_a_head", "stride_da_head"])
def _forest_state_bwd_kernel(
    entries,
    direct_grads,
    dA,
    segment_chunks,
    root_segments,
    chunk_grads,
    dA_tiles,
    converted_entries,
    dim: tl.constexpr,
    stride_entry_chunk,
    stride_entry_head,
    stride_entry_dim,
    stride_direct_chunk,
    stride_direct_head,
    stride_direct_dim,
    stride_a_head,
    stride_a_chunk,
    stride_grad_chunk,
    stride_grad_head,
    stride_da_head,
    stride_da_chunk,
    stride_converted_chunk,
    stride_converted_head,
    CONVERT_ENTRIES: tl.constexpr,
    BLOCK: tl.constexpr,
    IDX: tl.constexpr,
):
    tile = tl.program_id(0)
    root = tl.program_id(1)
    head = tl.program_id(2).to(IDX)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < dim
    entry_base = entries + head * stride_entry_head + offsets * stride_entry_dim
    direct_base = direct_grads + head * stride_direct_head + offsets * stride_direct_dim
    grad_base = chunk_grads + head * stride_grad_head + offsets
    a_base = dA + head * stride_a_head
    da_base = dA_tiles + head * stride_da_head + tile
    if CONVERT_ENTRIES:
        converted_base = converted_entries + head * stride_converted_head + offsets
    first_segment = tl.load(root_segments + root)
    stop_segment = tl.load(root_segments + root + 1)
    prefix_start = tl.load(segment_chunks + first_segment).to(IDX)
    prefix_stop = tl.load(segment_chunks + first_segment + 1).to(IDX)

    # Sum siblings in a fixed reverse order in FP32. No atomic state updates,
    # no materialized per-branch initial-state gradients, and no host loops.
    prefix_adjoint = tl.full((BLOCK,), 0.0, tl.float32)
    for segment in range(stop_segment - 1, first_segment, -1):
        start = tl.load(segment_chunks + segment).to(IDX)
        stop = tl.load(segment_chunks + segment + 1).to(IDX)
        adjoint = tl.full((BLOCK,), 0.0, tl.float32)
        for chunk in range(stop - 1, start - 1, -1):
            tl.store(grad_base + chunk * stride_grad_chunk, adjoint, valid)
            entry = tl.load(entry_base + chunk * stride_entry_chunk, valid, 0.0).to(tl.float32)
            scale = tl.exp(tl.load(a_base + chunk * stride_a_chunk).to(tl.float32))
            # This term includes the first branch chunk: its initial state is
            # the prefix terminal, so dropping it would lose dA at the fork.
            tl.store(da_base + chunk * stride_da_chunk, tl.sum(entry * adjoint) * scale)
            direct = tl.load(direct_base + chunk * stride_direct_chunk, valid, 0.0).to(tl.float32)
            adjoint = scale * adjoint + direct
            if CONVERT_ENTRIES:
                tl.store(converted_base + chunk * stride_converted_chunk, entry, valid)
        prefix_adjoint += adjoint

    for chunk in range(prefix_stop - 1, prefix_start - 1, -1):
        tl.store(grad_base + chunk * stride_grad_chunk, prefix_adjoint, valid)
        entry = tl.load(entry_base + chunk * stride_entry_chunk, valid, 0.0).to(tl.float32)
        scale = tl.exp(tl.load(a_base + chunk * stride_a_chunk).to(tl.float32))
        tl.store(da_base + chunk * stride_da_chunk, tl.sum(entry * prefix_adjoint) * scale)
        direct = tl.load(direct_base + chunk * stride_direct_chunk, valid, 0.0).to(tl.float32)
        prefix_adjoint = scale * prefix_adjoint + direct
        if CONVERT_ENTRIES:
            tl.store(converted_base + chunk * stride_converted_chunk, entry, valid)


def _index_dtype(*tensors):
    """Return the Triton index dtype: int64 only when an element offset can reach 2^31."""
    span = max(
        sum((size - 1) * stride for size, stride in zip(t.shape, t.stride())) for t in tensors
    )
    return tl.int64 if span >= 2**31 else tl.int32


def _forest_state_fwd(states, dA, segment_chunks, root_segments, out_dtype):
    _, nchunks, nheads, dim = states.shape
    out = torch.empty((1, nchunks, nheads, dim), device=states.device, dtype=out_dtype)
    grid = (triton.cdiv(dim, _STATE_BLOCK), root_segments.numel() - 1, nheads)
    with torch.cuda.device(states.device.index):
        _forest_state_fwd_kernel[grid](
            states,
            dA,
            segment_chunks,
            root_segments,
            out,
            dim,
            states.stride(1),
            states.stride(2),
            states.stride(3),
            dA.stride(1),
            dA.stride(2),
            out.stride(1),
            out.stride(2),
            BLOCK=_STATE_BLOCK,
            IDX=_index_dtype(states, dA, out),
        )
    return out


def _forest_state_bwd(entries, dA, direct_grads, segment_chunks, root_segments, dtype):
    _, nchunks, nheads, dim = entries.shape
    chunk_grads = torch.empty((1, nchunks, nheads, dim), device=entries.device, dtype=dtype)
    converted = entries if entries.dtype == dtype else torch.empty_like(chunk_grads)
    tiles = triton.cdiv(dim, _STATE_BLOCK)
    dA_tiles = torch.empty((1, nheads, nchunks, tiles), device=dA.device, dtype=torch.float32)
    grid = (tiles, root_segments.numel() - 1, nheads)
    with torch.cuda.device(entries.device.index):
        _forest_state_bwd_kernel[grid](
            entries,
            direct_grads,
            dA,
            segment_chunks,
            root_segments,
            chunk_grads,
            dA_tiles,
            converted,
            dim,
            entries.stride(1),
            entries.stride(2),
            entries.stride(3),
            direct_grads.stride(1),
            direct_grads.stride(2),
            direct_grads.stride(3),
            dA.stride(1),
            dA.stride(2),
            chunk_grads.stride(1),
            chunk_grads.stride(2),
            dA_tiles.stride(1),
            dA_tiles.stride(2),
            converted.stride(1),
            converted.stride(2),
            CONVERT_ENTRIES=converted is not entries,
            BLOCK=_STATE_BLOCK,
            IDX=_index_dtype(entries, direct_grads, dA, chunk_grads, dA_tiles, converted),
        )
    return chunk_grads, dA_tiles.sum(dim=-1).to(dA.dtype), converted


def _forest_scan_fwd(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    segment_chunks,
    root_segments,
    D,
    dt_bias,
    dt_softplus,
    save_intermediates=False,
):
    dA_cumsum, dt_out = _chunk_cumsum_fwd(
        dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus
    )
    chunk_states = _chunk_state_fwd(B, x, dt_out, dA_cumsum, states_in_fp32=True)
    states = _forest_state_fwd(
        chunk_states.flatten(-2), dA_cumsum[..., -1], segment_chunks, root_segments, C.dtype
    ).unflatten(-1, x.shape[-1:] + B.shape[-1:])
    if not save_intermediates:
        del chunk_states
    CB = _bmm_chunk_fwd(C, B, chunk_size, output_dtype=torch.float32)
    out, _ = _chunk_scan_fwd(CB, x, dt_out, dA_cumsum, C, states, D=D, z=None)
    intermediates = (dA_cumsum, dt_out, CB, chunk_states) if save_intermediates else ()
    return out, intermediates


def _forest_scan_bwd(
    dout,
    x,
    dt_in,
    A,
    B,
    C,
    chunk_size,
    segment_chunks,
    root_segments,
    D,
    dt_bias,
    dt_softplus,
    intermediates=(),
):
    # The default recomputes all intermediates, as the pinned combined backward
    # does. Opt-in saved tensors are the exact FP32 forward results. In both
    # paths chunk-entry states are reconstructed in FP32 below; the rounded
    # forward entry states are never substituted for this backward recurrence.
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    dt_in = dt_in.clone()  # Preserve the upstream Triton device-context workaround.
    if intermediates:
        dA_cumsum, dt, CB, chunk_states = intermediates
    else:
        dA_cumsum, dt = _chunk_cumsum_fwd(
            dt_in, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus
        )
        CB = _bmm_chunk_fwd(C, B, chunk_size, output_dtype=torch.float32)
        chunk_states = _chunk_state_fwd(B, x, dt, dA_cumsum, states_in_fp32=True)
    states = _forest_state_fwd(
        chunk_states.flatten(-2), dA_cumsum[..., -1], segment_chunks, root_segments, torch.float32
    ).unflatten(-1, x.shape[-1:] + B.shape[-1:])
    del chunk_states
    direct = _chunk_scan_bwd_dstates(C, dA_cumsum, dout, dtype=states.dtype)
    dstates, ddA_chunk_cumsum, states = _forest_state_bwd(
        states.flatten(-2),
        dA_cumsum[..., -1],
        direct.flatten(-2),
        segment_chunks,
        root_segments,
        x.dtype,
    )
    del direct
    states = states.unflatten(-1, x.shape[-1:] + B.shape[-1:])
    dstates = dstates.unflatten(-1, x.shape[-1:] + B.shape[-1:])
    ngroups = B.shape[2]
    dx, ddt, dD = _chunk_scan_chunk_state_bwd_dx(x, dt, dA_cumsum, B, CB, dout, dstates, D=D)
    dB, ddA_next = _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, B=B, ngroups=ngroups)
    dC, ddA_cumsum_prev = _chunk_scan_bwd_dC(states, dA_cumsum, dout, C=C, ngroups=ngroups)
    dCB = _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, ngroups=ngroups).to(CB.dtype)
    dB_out, dC_out = torch.empty_like(B), torch.empty_like(C)
    _bmm_chunk_bwd(C, dCB, residual=dB, out=dB_out)
    _bmm_chunk_bwd(B, rearrange(dCB, "... l s -> ... s l"), residual=dC, out=dC_out)
    ddA_cumsum_prev[..., -1] += ddA_chunk_cumsum
    ddA_prev = ddA_cumsum_prev.flip([-1]).cumsum(dim=-1).flip([-1])
    ddA = _chunk_scan_bwd_ddAcs_stable(x, dt, dA_cumsum, dout, CB)
    ddA += ddA_next + ddA_prev
    ddt_out, dA, ddt_bias = _chunk_cumsum_bwd(
        ddA, ddt, dt_in, A, dt_bias=dt_bias, dt_softplus=dt_softplus
    )
    return dx, ddt_out, dA, dB_out, dC_out, dD, ddt_bias


class _MambaChunkScanForest(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        segment_chunks,
        root_segments,
        D,
        dt_bias,
        dt_softplus,
        save_intermediates,
    ):
        """Run the chunk-aligned forest scan and retain inputs needed by backward."""
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if x.stride(-1) != 1 and x.stride(1) != 1:
            x = x.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        out, intermediates = _forest_scan_fwd(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            segment_chunks,
            root_segments,
            D,
            dt_bias,
            dt_softplus,
            save_intermediates=save_intermediates,
        )
        ctx.save_for_backward(
            x, dt, A, B, C, segment_chunks, root_segments, D, dt_bias, *intermediates
        )
        ctx.chunk_size = chunk_size
        ctx.dt_softplus = dt_softplus
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        """Propagate output gradients through forked scan states and chunk parameters."""
        x, dt, A, B, C, segment_chunks, root_segments, D, dt_bias, *intermediates = (
            ctx.saved_tensors
        )
        dx, ddt, dA, dB, dC, dD, ddt_bias = _forest_scan_bwd(
            dout,
            x,
            dt,
            A,
            B,
            C,
            ctx.chunk_size,
            segment_chunks,
            root_segments,
            D,
            dt_bias,
            ctx.dt_softplus,
            intermediates=intermediates,
        )
        return dx, ddt, dA, dB, dC, None, None, None, dD, ddt_bias, None, None


def mamba_chunk_scan_forest(
    x: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    chunk_size: int,
    segment_chunks: Tensor,
    root_segments: Tensor,
    *,
    D: Tensor | None = None,
    dt_bias: Tensor | None = None,
    dt_softplus: bool = True,
    state_dtype: torch.dtype | None = None,
    save_intermediates: bool = False,
) -> Tensor:
    """Scan independent shared-prefix roots with differentiable state forks.

    Args:
        x: Input with shape ``[1, T, H, P]``; T must be chunk aligned.
        dt: Time steps with shape ``[1, T, H]``.
        A: Decay parameters with shape ``[H]``.
        B: Input state projections with shape ``[1, T, G, N]``.
        C: Output state projections with the same shape as B.
        chunk_size: Positive power-of-two chunk size supported by mamba_ssm.
        segment_chunks: Contiguous CUDA int32 cumulative chunk boundaries,
            starting at zero and ending at ``T // chunk_size``. Monotone
            nondecreasing boundaries allow empty segments.
        root_segments: Contiguous CUDA int32 cumulative segment boundaries,
            starting at zero and ending at ``segment_chunks.numel() - 1``.
            Boundaries must increase strictly. The first segment of each root
            is its prefix; all later segments are independent siblings.
        D: Optional skip weights with shape ``[H]`` or ``[H, P]``.
        dt_bias: Optional time-step bias with shape ``[H]``.
        dt_softplus: Apply softplus to biased time steps.
        state_dtype: Only the pinned upstream default (None or C.dtype) is
            supported. Inter-chunk accumulators always remain FP32.
        save_intermediates: Opt in to retaining the FP32 time-step/cumulative
            decay, C-B products and chunk states for backward. This removes
            three recomputation kernels at the cost of four saved tensors.
            Defaults to recomputation and is disabled when grad mode is off
            or no floating input needs a gradient. State passing is unchanged.

    Returns:
        Output with shape and dtype matching x, including supplied tail padding.

    Notes:
        The caller must validate the metadata values when constructing the
        cached layout. This function validates tensor metadata without device
        synchronization or per-layer GPU validation launches. All segments
        must start on chunk boundaries; convolution context is the caller's
        responsibility. No gate, external states, or second derivatives are
        provided. Padding must be excluded from downstream loss/output maps.
    """
    if x.ndim != 4 or x.shape[0] != 1 or x.shape[1] == 0:
        raise ValueError("Forest scan requires nonempty x with shape [1, T, H, P]")
    if chunk_size < 1 or chunk_size & (chunk_size - 1) or x.shape[1] % chunk_size:
        raise ValueError("Forest scan requires power-of-two chunk size and chunk-aligned T")
    _, length, heads, headdim = x.shape
    if B.ndim != 4 or B.shape[:2] != (1, length) or C.shape != B.shape:
        raise ValueError("Forest B and C must have matching shape [1, T, G, N]")
    if B.shape[2] < 1 or B.shape[3] < 1 or heads < 1 or headdim < 1 or heads % B.shape[2]:
        raise ValueError("Forest state dimensions must be positive and groups must divide heads")
    if dt.shape != (1, length, heads) or A.shape != (heads,):
        raise ValueError("Forest dt and A dimensions must match x heads and tokens")
    if D is not None and D.shape not in ((heads,), (heads, headdim)):
        raise ValueError("Forest D must have shape [H] or [H, P]")
    if dt_bias is not None and dt_bias.shape != (heads,):
        raise ValueError("Forest dt_bias must have shape [H]")
    tensors = (x, dt, A, B, C, segment_chunks, root_segments, D, dt_bias)
    if not x.is_cuda or any(t.device != x.device for t in tensors if t is not None):
        raise ValueError("Forest scan tensors must reside on one CUDA device")
    for boundaries in (segment_chunks, root_segments):
        if (
            boundaries.ndim != 1
            or boundaries.numel() < 2
            or boundaries.dtype != torch.int32
            or not boundaries.is_contiguous()
        ):
            raise ValueError("Forest boundaries must be contiguous int32 vectors of length >= 2")
    if state_dtype not in (None, C.dtype):
        raise NotImplementedError("Forest scan currently requires the pinned upstream state dtype")
    if not isinstance(save_intermediates, bool):
        raise TypeError("save_intermediates must be a bool")
    if save_intermediates:
        # Uniform checkpoint's first forward runs under no_grad; retain only
        # during its grad-enabled backward replay, not across checkpoint spans.
        save_intermediates = torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (x, dt, A, B, C, D, dt_bias) if tensor is not None
        )
    return _MambaChunkScanForest.apply(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        segment_chunks,
        root_segments,
        D,
        dt_bias,
        dt_softplus,
        save_intermediates,
    )
