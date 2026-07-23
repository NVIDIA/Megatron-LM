# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from:
#   https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_state_passing.py
# Adapted from vLLM project (Apache-2.0).

import torch
import triton
import triton.language as tl

from megatron.core.ssm.ops.determinism import autotune_configs


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

    if HAS_INITSTATES:
        initstates_ptrs = (
            initstates_ptr + pid_h * stride_initstates_head + offs_m * stride_initstates_dim
        )

        states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    else:
        states = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    prev_seq_idx = 0
    for c in range(nchunks):
        if HAS_DST_STATES:
            dst_flag = tl.load(dst_flags_ptr + c) != 0
        else:
            dst_flag = True
        # Unflagged destination chunks have no chunk state.
        new_states = tl.load(states_ptrs, mask=(offs_m < dim) & dst_flag, other=0.0).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        seq_idx = tl.load(seq_idx_ptr + c * stride_seq_idx_chunk)
        if HAS_DST_STATES:
            # Destination chunks start from their indexed initial state.
            is_new_seq = True
        else:
            is_new_seq = prev_seq_idx != seq_idx
        # we have started a new sequence
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
        if not HAS_DST_STATES:
            tl.store(out_ptrs, states, mask=offs_m < dim)

        if HAS_DST_STATES:
            # Commit completed chunks directly to the state cache.
            if dst_flag:
                dst_idx = tl.load(dst_indices_ptr + c).to(tl.int64)
                dst_ptrs = (
                    dst_states_ptr
                    + dst_idx * stride_dst_batch
                    + pid_h * stride_dst_head
                    + offs_m * stride_dst_dim
                )
                tl.store(dst_ptrs, states, mask=offs_m < dim)

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
):
    """
    dst_states/dst_indices/dst_flags write flagged boundary states directly to
    dst_states without allocating an output tensor.
    """
    nchunks, nheads, dim = states.shape
    chunk_size = dA_cumsum.shape[-1]
    assert dA_cumsum.shape == (nheads, nchunks, chunk_size)
    seqlen = seq_idx.shape[-1]
    has_dst = dst_states is not None
    assert (dst_indices is not None) == has_dst and (
        dst_flags is not None
    ) == has_dst, "dst_states, dst_indices, and dst_flags must be provided together"
    if not has_dst:
        out_dtype = states.dtype if out_dtype is None else out_dtype
        out = torch.empty((nchunks, nheads, dim), device=states.device, dtype=out_dtype)
        out_strides = out.stride()
    else:
        out = states
        out_strides = (0, 0, 0)

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
            stride_out_chunk=out_strides[0],
            stride_out_head=out_strides[1],
            stride_out_dim=out_strides[2],
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
