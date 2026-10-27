# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from:
#   https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_state_passing.py
# Adapted from vLLM project (Apache-2.0).
# The varlen backward is adapted from (Apache-2.0):
# https://github.com/state-spaces/mamba/blob/v2.3.2.post1/mamba_ssm/ops/triton/ssd_state_passing.py

import torch
import triton
import triton.language as tl

from megatron.core.tuning import autotune_configs


@triton.autotune(
    configs=autotune_configs(
        [
            triton.Config({"BLOCK_SIZE": 64}),
            triton.Config({"BLOCK_SIZE": 128}),
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
            triton.Config({"BLOCK_SIZE": 2048}),
        ]
    ),
    key=["dim"],
)
@triton.jit
def _state_passing_fwd_kernel(
    # Pointers to matrices
    states_ptr,
    out_ptr,
    dA_cs_ptr,
    initstates_ptr,
    seq_idx_ptr,
    cu_chunk_seqlens_ptr,
    dst_states_ptr,
    dst_indices_ptr,
    dst_flags_ptr,
    # Matrix dimensions
    dim: tl.constexpr,
    nchunks,
    seqlen,
    chunk_size: tl.constexpr,
    # Strides
    stride_states_chunk: tl.int64,
    stride_states_head: tl.int64,
    stride_states_dim: tl.constexpr,
    stride_out_chunk: tl.int64,
    stride_out_head: tl.int64,
    stride_out_dim: tl.constexpr,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    stride_initstates_batch: tl.int64,
    stride_initstates_head: tl.int64,
    stride_initstates_dim: tl.constexpr,
    stride_seq_idx_chunk: tl.constexpr,
    stride_dst_batch: tl.int64,
    stride_dst_head: tl.int64,
    stride_dst_dim: tl.constexpr,
    # Meta-parameters
    HAS_INITSTATES: tl.constexpr,
    HAS_DST_STATES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_h = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=0)

    states_ptr += pid_h * stride_states_head
    dA_cs_ptr += pid_h * stride_dA_cs_head + (chunk_size - 1) * stride_dA_cs_csize
    out_ptr += pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim

    if HAS_INITSTATES and not HAS_DST_STATES:
        initstates_ptrs = (
            initstates_ptr + pid_h * stride_initstates_head + offs_m * stride_initstates_dim
        )

        states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    else:
        states = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    prev_seq_idx = tl.full((), 0, tl.int64)
    for c in range(nchunks):
        if HAS_DST_STATES:
            dst_flag = tl.load(dst_flags_ptr + c) != 0
        else:
            dst_flag = True
        if dst_flag:
            new_states = tl.load(states_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
            seq_idx = tl.load(seq_idx_ptr + c * stride_seq_idx_chunk).to(tl.int64)
            if HAS_DST_STATES:
                # Destination chunks start from their indexed initial state.
                is_new_seq = True
            else:
                is_new_seq = prev_seq_idx != seq_idx
            if is_new_seq:
                if HAS_INITSTATES:
                    initstates_ptrs = (
                        initstates_ptr
                        + seq_idx * stride_initstates_batch
                        + pid_h * stride_initstates_head
                        + offs_m * stride_initstates_dim
                    )
                    states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
                else:
                    states = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

            prev_seq_idx = seq_idx
            states = tl.exp(dA_cs) * states + new_states
            if HAS_DST_STATES:
                dst_idx = tl.load(dst_indices_ptr + c).to(tl.int64)
                dst_ptrs = (
                    dst_states_ptr
                    + dst_idx * stride_dst_batch
                    + pid_h * stride_dst_head
                    + offs_m * stride_dst_dim
                )
                tl.store(dst_ptrs, states, mask=offs_m < dim)
            else:
                tl.store(out_ptrs, states, mask=offs_m < dim)

        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk


def _state_passing_fwd(
    states,
    dA_cumsum,
    cu_chunk_seqlens,
    seq_idx,
    initial_states=None,
    out_dtype=None,
    dst_states=None,
    dst_indices=None,
    dst_flags=None,
    out=None,
):
    """
    dst_states/dst_indices/dst_flags write flagged boundary states directly to
    dst_states without allocating an output tensor.
    out: optional preallocated (nchunks, nheads, dim) output, e.g. a view into a larger buffer.
    """
    nchunks, nheads, dim = states.shape
    chunk_size = dA_cumsum.shape[-1]
    assert dA_cumsum.shape == (nheads, nchunks, chunk_size)
    seqlen = seq_idx.shape[-1]
    has_dst = dst_states is not None
    assert (dst_indices is not None) == has_dst and (
        dst_flags is not None
    ) == has_dst, "dst_states, dst_indices, and dst_flags must be provided together"
    if has_dst:
        assert out is None, "out cannot be combined with dst_states"
        out = states
    elif out is None:
        out_dtype = states.dtype if out_dtype is None else out_dtype
        out = torch.empty((nchunks, nheads, dim), device=states.device, dtype=out_dtype)
    else:
        assert out.shape == (nchunks, nheads, dim)

    initial_states_strides = (
        (initial_states.stride(0), initial_states.stride(1), initial_states.stride(2))
        if initial_states is not None
        else (0, 0, 0)
    )
    if has_dst:
        assert dst_states.shape[1] == nheads and dst_states.shape[2] == dim
    dst_strides = (
        (dst_states.stride(0), dst_states.stride(1), dst_states.stride(2)) if has_dst else (0, 0, 0)
    )

    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE"]), nheads)
    with torch.cuda.device(states.device.index):
        _state_passing_fwd_kernel[grid](
            states_ptr=states,
            out_ptr=out,
            dA_cs_ptr=dA_cumsum,
            initstates_ptr=initial_states,
            seq_idx_ptr=seq_idx,
            cu_chunk_seqlens_ptr=cu_chunk_seqlens,
            dst_states_ptr=dst_states,
            dst_indices_ptr=dst_indices,
            dst_flags_ptr=dst_flags,
            dim=dim,
            nchunks=nchunks,
            seqlen=seqlen if seq_idx is not None else 0,
            chunk_size=chunk_size if seq_idx is not None else 0,
            stride_states_chunk=states.stride(0),
            stride_states_head=states.stride(1),
            stride_states_dim=states.stride(2),
            stride_out_chunk=out.stride(0),
            stride_out_head=out.stride(1),
            stride_out_dim=out.stride(2),
            stride_dA_cs_head=dA_cumsum.stride(0),
            stride_dA_cs_chunk=dA_cumsum.stride(1),
            stride_dA_cs_csize=dA_cumsum.stride(2),
            stride_initstates_batch=initial_states_strides[0],
            stride_initstates_head=initial_states_strides[1],
            stride_initstates_dim=initial_states_strides[2],
            stride_seq_idx_chunk=seq_idx.stride(0),
            stride_dst_batch=dst_strides[0],
            stride_dst_head=dst_strides[1],
            stride_dst_dim=dst_strides[2],
            HAS_INITSTATES=initial_states is not None,
            HAS_DST_STATES=has_dst,
        )
    return None if has_dst else out


# ---------------------------------------------------------------------------------------------
# Backward. Adapted from mamba_ssm (state-spaces/mamba v2.3.2.post1) for varlen chunks: chunk c
# covers tokens [cu_chunk_seqlens[c], cu_chunk_seqlens[c + 1]) and never spans two sequences.
# Token addressing uses those offsets; seq_idx is per token of the (unpadded) packed layout, so
# the seq_idx tests of the original kernels stay valid unchanged.
# ---------------------------------------------------------------------------------------------


@triton.autotune(
    configs=autotune_configs(
        [
            triton.Config({'BLOCK_SIZE': 64}),
            triton.Config({'BLOCK_SIZE': 128}),
            triton.Config({'BLOCK_SIZE': 256}),
            triton.Config({'BLOCK_SIZE': 512}),
            triton.Config({'BLOCK_SIZE': 1024}),
            triton.Config({'BLOCK_SIZE': 2048}),
        ]
    ),
    key=['dim'],
)
@triton.jit
def _state_passing_bwd_kernel(
    chunk_offsets_ptr,  # varlen: (nchunks + 1,) token offsets
    # Pointers to matrices
    dout_ptr,
    out_ptr,
    dA_cs_ptr,
    dfinal_states_ptr,
    seq_idx_ptr,
    dstates_ptr,
    ddA_cs_ptr,
    dinitstates_ptr,
    states_converted_ptr,
    # Matrix dimensions
    dim,
    nchunks,
    seqlen,
    chunk_size,
    # Strides
    stride_dout_batch,
    stride_dout_chunk,
    stride_dout_head,
    stride_dout_dim,
    stride_out_batch,
    stride_out_chunk,
    stride_out_head,
    stride_out_dim,
    stride_dA_cs_batch,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dfinal_states_batch,
    stride_dfinal_states_head,
    stride_dfinal_states_dim,
    stride_seq_idx_batch,
    stride_seq_idx_seqlen,
    stride_dstates_batch,
    stride_dstates_chunk,
    stride_dstates_head,
    stride_dstates_dim,
    stride_ddA_cs_batch,
    stride_ddA_cs_chunk,
    stride_ddA_cs_head,
    stride_dinitstates_batch,
    stride_dinitstates_head,
    stride_dinitstates_dim,
    # Meta-parameters
    CONVERT_STATES: tl.constexpr,
    HAS_DFINAL_STATES: tl.constexpr,
    HAS_DINITSTATES: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2).to(tl.int64)
    pid_m = tl.program_id(axis=0)
    # tl.cast, not .to: Triton passes nchunks == 1 as a plain int.
    last_chunk = tl.cast(nchunks, tl.int64) - 1
    dstates_ptr += (
        pid_b * stride_dstates_batch
        + pid_h * stride_dstates_head
        + last_chunk * stride_dstates_chunk
    )
    dA_cs_ptr += (
        pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_head + last_chunk * stride_dA_cs_chunk
    )
    ddA_cs_ptr += (
        pid_b * stride_ddA_cs_batch
        + pid_h * stride_ddA_cs_head
        + last_chunk * stride_ddA_cs_chunk
        + pid_m
    )
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head + last_chunk * stride_out_chunk
    dout_ptr += (
        pid_b * stride_dout_batch + pid_h * stride_dout_head + last_chunk * stride_dout_chunk
    )
    if CONVERT_STATES:
        states_converted_ptr += (
            pid_b * stride_out_batch + pid_h * stride_out_head + last_chunk * stride_out_chunk
        )
    if HAS_DFINAL_STATES:
        dfinal_states_ptr += pid_b * stride_dfinal_states_batch + pid_h * stride_dfinal_states_head
    if HAS_DINITSTATES:
        dinitstates_ptr += pid_b * stride_dinitstates_batch + pid_h * stride_dinitstates_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dstates_ptrs = dstates_ptr + offs_m * stride_dstates_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    dout_ptrs = dout_ptr + offs_m * stride_dout_dim
    if CONVERT_STATES:
        states_converted_ptrs = states_converted_ptr + offs_m * stride_out_dim

    if HAS_DFINAL_STATES:
        dstates = tl.load(
            dfinal_states_ptr + offs_m * stride_dfinal_states_dim, mask=offs_m < dim, other=0.0
        ).to(tl.float32)
    else:
        dstates = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    tl.store(dstates_ptrs, dstates, mask=offs_m < dim)
    if HAS_SEQ_IDX:
        seq_idx = tl.load(seq_idx_ptr + (seqlen - 1) * stride_seq_idx_seqlen)
    dstates_ptrs -= stride_dstates_chunk
    for c in range(nchunks - 1):
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale = tl.exp(dA_cs)
        if HAS_SEQ_IDX:
            seq_idx_new = tl.load(
                seq_idx_ptr
                + ((tl.load(chunk_offsets_ptr + nchunks - c - 1) - 1) * stride_seq_idx_seqlen)
            )
            scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
            seq_idx = seq_idx_new
        out = tl.load(out_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if CONVERT_STATES:
            tl.store(states_converted_ptrs, out, mask=offs_m < dim)
        ddA = tl.sum(out * dstates) * scale
        tl.store(ddA_cs_ptr, ddA)
        dout = tl.load(dout_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dstates = scale * dstates + dout
        tl.store(dstates_ptrs, dstates, mask=offs_m < dim)
        dout_ptrs -= stride_dout_chunk
        dstates_ptrs -= stride_dstates_chunk
        dA_cs_ptr -= stride_dA_cs_chunk
        ddA_cs_ptr -= stride_ddA_cs_chunk
        out_ptrs -= stride_out_chunk
        if CONVERT_STATES:
            states_converted_ptrs -= stride_out_chunk
    if CONVERT_STATES:
        out = tl.load(out_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        tl.store(states_converted_ptrs, out, mask=offs_m < dim)
    if not HAS_DINITSTATES:
        tl.store(ddA_cs_ptr, 0.0)
    else:
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale = tl.exp(dA_cs)
        if HAS_SEQ_IDX:
            scale = tl.where(seq_idx == 0, scale, 0.0)
        out = tl.load(out_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        ddA = tl.sum(out * dstates) * scale
        tl.store(ddA_cs_ptr, ddA)
        dout = tl.load(dout_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dstates = scale * dstates + dout
        tl.store(dinitstates_ptr + offs_m * stride_dinitstates_dim, dstates, mask=offs_m < dim)


def _state_passing_bwd(
    cu_chunk_seqlens,
    states,
    dA_chunk_cumsum,
    dout,
    dfinal_states=None,
    seq_idx=None,
    has_initial_states=None,
    dstates_dtype=None,
    states_dtype=None,
    chunk_size=None,
):
    """
    states contains the initial_states at index 0. The final states are not included in states.
    """
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
    assert dout.shape == (batch, nchunks, nheads, dim)
    if seq_idx is not None:
        assert chunk_size is not None
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seqlen)
    dstates = torch.empty_like(
        dout, dtype=dstates_dtype if dstates_dtype is not None else dout.dtype
    )
    if states_dtype is not None and states_dtype != states.dtype:
        states_converted = torch.empty_like(
            states, dtype=dstates_dtype if dstates_dtype is not None else dout.dtype
        )
        assert states_converted.stride() == states.stride()
    else:
        states_converted = None
    if has_initial_states:
        dinitstates = torch.empty_like(dstates[:, 0])
    else:
        dinitstates = None
    if dfinal_states is not None:
        assert dfinal_states.shape == (batch, nheads, dim)
    BLOCK_SIZE_min = 64
    n_blocks = (dim + BLOCK_SIZE_min - 1) // BLOCK_SIZE_min
    ddA_chunk_cumsum = torch.empty(
        batch, nheads, nchunks, n_blocks, dtype=torch.float32, device=dA_chunk_cumsum.device
    )
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE']), batch, nheads)
    with torch.cuda.device(dout.device.index):
        _state_passing_bwd_kernel[grid](
            cu_chunk_seqlens,
            dout,
            states,
            dA_chunk_cumsum,
            dfinal_states,
            seq_idx,
            dstates,
            ddA_chunk_cumsum,
            dinitstates,
            states_converted,
            dim,
            nchunks,
            seqlen if seq_idx is not None else 0,
            chunk_size if seq_idx is not None else 0,
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            dout.stride(3),
            states.stride(0),
            states.stride(1),
            states.stride(2),
            states.stride(3),
            dA_chunk_cumsum.stride(0),
            dA_chunk_cumsum.stride(2),
            dA_chunk_cumsum.stride(1),
            *(
                (dfinal_states.stride(0), dfinal_states.stride(1), dfinal_states.stride(2))
                if dfinal_states is not None
                else (0, 0, 0)
            ),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            dstates.stride(0),
            dstates.stride(1),
            dstates.stride(2),
            dstates.stride(3),
            ddA_chunk_cumsum.stride(0),
            ddA_chunk_cumsum.stride(2),
            ddA_chunk_cumsum.stride(1),
            *(
                (dinitstates.stride(0), dinitstates.stride(1), dinitstates.stride(2))
                if dinitstates is not None
                else (0, 0, 0)
            ),
            CONVERT_STATES=states_converted is not None,
            HAS_DFINAL_STATES=dfinal_states is not None,
            HAS_DINITSTATES=dinitstates is not None,
            HAS_SEQ_IDX=seq_idx is not None,
        )
    BLOCK_SIZE_actual = _state_passing_bwd_kernel.best_config.kwargs["BLOCK_SIZE"]
    n_valid_blocks = (dim + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
    ddA_chunk_cumsum = (
        ddA_chunk_cumsum[..., :n_valid_blocks].sum(dim=-1).to(dtype=dA_chunk_cumsum.dtype)
    )
    if states_dtype is not None and states_dtype == states.dtype:
        states_converted = states
    return (
        (dstates, ddA_chunk_cumsum, dinitstates)
        if states_dtype is None
        else (dstates, ddA_chunk_cumsum, dinitstates, states_converted)
    )
