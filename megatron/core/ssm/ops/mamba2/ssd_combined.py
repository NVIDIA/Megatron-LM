# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from:
#   https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_combined.py
# Adapted from vLLM project (Apache-2.0).
# The varlen backward is adapted from (Apache-2.0):
# https://github.com/state-spaces/mamba/blob/v2.3.2.post1/mamba_ssm/ops/triton/ssd_combined.py
# with the backward's state_dtype handling from state-spaces/mamba#972:
# https://github.com/state-spaces/mamba/commit/0048fbf2e7b2f214dcbe703ea3dec2b9647595e1

import math

import torch
import triton
import triton.language as tl
from packaging import version

from megatron.core.ssm.ops.common.determinism import alloc_tile_workspace, finalize_tile_workspace
from megatron.core.tuning import autotune_configs, use_deterministic_mode

from .ssd_bmm import _bmm_chunk_bwd, _bmm_chunk_fwd
from .ssd_chunk_scan import (
    _chunk_scan_bwd_dC,
    _chunk_scan_bwd_dcb,
    _chunk_scan_bwd_ddAcs_stable,
    _chunk_scan_bwd_dstates,
    _chunk_scan_bwd_dz,
    _chunk_scan_fwd,
)
from .ssd_chunk_state import (
    _chunk_cumsum_bwd,
    _chunk_cumsum_fwd,
    _chunk_state_bwd_db,
    _chunk_state_fwd,
    init_to_zero,
)
from .ssd_state_passing import _state_passing_bwd, _state_passing_fwd

TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")


def is_int_pow_2(n):
    """Return True if n is a positive integer power of 2."""
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


def _mamba_chunk_scan_combined_fwd(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    out,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    cu_chunk_seqlens=None,
    last_chunk_indices=None,
    return_raw_states=False,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    state_dtype=None,
    out_x=None,
):
    assert is_int_pow_2(chunk_size), "chunk_size must be integer power of 2"
    seqlen, nheads, headdim = x.shape
    _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (
        seqlen,
        ngroups,
        dstate,
    ), f"B.shape={B.shape} != ({seqlen}, {ngroups}, {dstate})"
    assert dt.shape == (seqlen, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (cu_chunk_seqlens.shape[0] - 1,)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if x.stride(-1) != 1 and x.stride(0) != 1:  # Either M or K dimension should be contiguous
        x = x.contiguous()
    if (
        z is not None and z.stride(-1) != 1 and z.stride(0) != 1
    ):  # Either M or K dimension should be contiguous
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    assert cu_chunk_seqlens is not None, "Assuming varlen input - must supply cu_chunk_seqlens"
    assert last_chunk_indices is not None, "last_chunk_indices must be provided"

    if initial_states is not None:
        num_seqs = last_chunk_indices.shape[0]
        assert initial_states.shape == (num_seqs, nheads, headdim, dstate)

    # This function executes 5 sub-functions for computing mamba
    # - a good resource is the blog https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/
    #   which has a minimal implementation to understand the below operations
    # - as explained by the blog, mamba is a special case of causal attention
    # - the idea is to chunk the attention matrix and compute each
    #   submatrix separately using different optimizations.
    # - see the blog and paper for a visualization of the submatrices
    #   which we refer to in the comments below

    # 1. Compute chunked cumsum of A * dt
    # - here dt may go through a softplus activation
    dA_cumsum, dt = _chunk_cumsum_fwd(
        dt,
        A,
        chunk_size,
        cu_chunk_seqlens,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
    )

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, cu_chunk_seqlens, states_in_fp32=True)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    # - for handling chunked prefill, this requires i) initial_states and
    #   ii) seq_idx to be all specified.
    # - When a new seq_idx is detected, we will stop passing the prev_state
    #   and switch accordingly to the init_state corresponding to the new seq_idx.
    states = _state_passing_fwd(
        states.flatten(-2),  # ... p n -> ... (p n)
        dA_cumsum,  # (nheads, nchunks, chunk_size)
        cu_chunk_seqlens,
        initial_states=(
            initial_states.flatten(-2) if initial_states is not None else None
        ),  # (batch, nheads, headdim*dstate)
        seq_idx=seq_idx,
        out_dtype=state_dtype if state_dtype is not None else C.dtype,
    )
    states = states.unflatten(-1, (-1, dstate))

    # 4. Compute batched matrix multiply for C_j^T B_i terms
    CB = _bmm_chunk_fwd(C, B, chunk_size, cu_chunk_seqlens, output_dtype=torch.float32)

    # 5. Scan and compute the diagonal blocks, taking into
    #    account past causal states.
    # - if initial states are provided, then states information will be
    #   augmented with initial_states.
    # - to do this properly, we need to account for example changes in
    #   the continuous batch, therefore we introduce pseudo chunks, which is
    #   a chunk that is split up each time an example changes.
    # - in each (pseudo) chunk, we detect if the previous (pseudo) chunk had
    #   a seq_idx change, in which case we take states information from
    #   init_states.
    _chunk_scan_fwd(
        CB,
        x,
        dt,
        dA_cumsum,
        C,
        states,
        cu_chunk_seqlens,
        out,  # in-place update
        seq_idx,
        D=D,
        z=z,
        initial_states=initial_states,
        out_x=out_x,
    )

    final_states = states[last_chunk_indices]
    if return_raw_states:
        # Caller extracts any chunk-boundary states itself from the raw states.
        return final_states, states
    return final_states


def mamba_chunk_scan_combined_varlen(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    cu_chunk_seqlens,
    last_chunk_indices,
    seq_idx,
    out,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    return_raw_states=False,
    state_dtype=None,
    out_x=None,
):
    """
    Argument:
        x: (seqlen, nheads, headdim)
        dt: (seqlen, nheads)
        A: (nheads)
        B: (seqlen, ngroups, dstate)
        C: (seqlen, ngroups, dstate)
        chunk_size: int
        cu_chunk_seqlens: (nchunks + 1,)
        last_chunk_indices: (batch,)
        seq_idx: (nchunks,)
        out: (seqlen, nheads, headdim) preallocated output tensor
        D: (nheads, headdim) or (nheads,)
        z: (seqlen, nheads, headdim)
        dt_bias: (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        dt_softplus: Whether to apply softplus to dt
        return_raw_states: If True, returns ``(varlen_states, raw_states)`` where
            ``raw_states`` is the full ``(nchunks, nheads, headdim, dstate)``
            chunk-boundary state tensor for the caller to extract from directly.
        state_dtype: The data type of the ssm state
        out_x: optional (seqlen, nheads, headdim) preallocated tensor, same strides as ``out``;
            requires ``z``. Receives the output before the ``z`` gate (with the ``D`` term), as
            ``mamba_ssm``'s training forward saves for its backward.
    Return:
        varlen_states: (batch, nheads, headdim, dstate), or
        (varlen_states, raw_states) if return_raw_states is True
    """

    assert seq_idx is not None

    varlen_states = _mamba_chunk_scan_combined_fwd(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        out,
        D=D,
        z=z,
        dt_bias=dt_bias,
        initial_states=initial_states,
        seq_idx=seq_idx,
        cu_chunk_seqlens=cu_chunk_seqlens,
        last_chunk_indices=last_chunk_indices,
        return_raw_states=return_raw_states,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        state_dtype=state_dtype,
        out_x=out_x,
    )

    return varlen_states


# ---------------------------------------------------------------------------------------------
# Backward. Adapted from mamba_ssm (state-spaces/mamba v2.3.2.post1) for varlen chunks: chunk c
# covers tokens [cu_chunk_seqlens[c], cu_chunk_seqlens[c + 1]) and never spans two sequences.
# Token addressing uses those offsets; seq_idx is per token of the (unpadded) packed layout, so
# the seq_idx tests of the original kernels stay valid unchanged.
# ---------------------------------------------------------------------------------------------


@triton.autotune(
    configs=autotune_configs(
        [
            triton.Config(
                {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64},
                num_stages=3,
                num_warps=8,
                pre_hook=init_to_zero(["ddt_ptr", "dD_ptr"]),
            ),
            triton.Config(
                {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32},
                num_stages=4,
                num_warps=4,
                pre_hook=init_to_zero(["ddt_ptr", "dD_ptr"]),
            ),
            triton.Config(
                {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32},
                num_stages=4,
                num_warps=4,
                pre_hook=init_to_zero(["ddt_ptr", "dD_ptr"]),
            ),
            triton.Config(
                {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32},
                num_stages=4,
                num_warps=4,
                pre_hook=init_to_zero(["ddt_ptr", "dD_ptr"]),
            ),
            triton.Config(
                {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32},
                num_stages=4,
                num_warps=4,
                pre_hook=init_to_zero(["ddt_ptr", "dD_ptr"]),
            ),
            triton.Config(
                {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32},
                num_stages=4,
                num_warps=4,
                pre_hook=init_to_zero(["ddt_ptr", "dD_ptr"]),
            ),
            triton.Config(
                {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32},
                num_stages=5,
                num_warps=4,
                pre_hook=init_to_zero(["ddt_ptr", "dD_ptr"]),
            ),
            triton.Config(
                {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32},
                num_stages=5,
                num_warps=4,
                pre_hook=init_to_zero(["ddt_ptr", "dD_ptr"]),
            ),
            triton.Config(
                {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32},
                num_stages=4,
                num_warps=4,
                pre_hook=init_to_zero(["ddt_ptr", "dD_ptr"]),
            ),
        ]
    ),
    key=['chunk_size', 'hdim', 'dstate'],
)
@triton.jit
def _chunk_scan_chunk_state_bwd_dx_kernel(
    chunk_offsets_ptr,  # varlen: (nchunks + 1,) token offsets
    # Pointers to matrices
    x_ptr,
    cb_ptr,
    dout_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    seq_idx_ptr,
    D_ptr,
    b_ptr,
    dstates_ptr,
    dx_ptr,
    ddt_ptr,
    dD_ptr,
    # Matrix dimensions
    chunk_size,
    hdim,
    dstate,
    batch,
    seqlen,
    nheads_ngroups_ratio,
    # Strides
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    stride_cb_batch,
    stride_cb_chunk,
    stride_cb_head,
    stride_cb_csize_m,
    stride_cb_csize_k,
    stride_dout_batch,
    stride_dout_seqlen,
    stride_dout_head,
    stride_dout_hdim,
    stride_dt_batch,
    stride_dt_chunk,
    stride_dt_head,
    stride_dt_csize,
    stride_dA_cs_batch,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    stride_seq_idx_batch,
    stride_seq_idx_seqlen,
    stride_D_head,
    stride_b_batch,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dstate,
    stride_dstates_batch,
    stride_dstates_chunk,
    stride_dstates_head,
    stride_dstates_hdim,
    stride_dstates_dstate,
    stride_dx_batch,
    stride_dx_seqlen,
    stride_dx_head,
    stride_dx_hdim,
    stride_ddt_batch,
    stride_ddt_chunk,
    stride_ddt_head,
    stride_ddt_csize,
    stride_ddt_tile,
    stride_dD_batch,
    stride_dD_chunk,
    stride_dD_head,
    stride_dD_csize,
    stride_dD_hdim,
    # Meta-parameters
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
    DETERMINISTIC_REDUCTION: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1).to(tl.int64)
    pid_c = pid_bc // batch
    chunk_start = tl.load(chunk_offsets_ptr + pid_c).to(tl.int64)
    chunk_end = tl.load(chunk_offsets_ptr + pid_c + 1).to(tl.int64)
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2).to(tl.int64)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += pid_b * stride_x_batch + chunk_start * stride_x_seqlen + pid_h * stride_x_head
    cb_ptr += (
        pid_b * stride_cb_batch
        + pid_c * stride_cb_chunk
        + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    )
    dout_ptr += (
        pid_b * stride_dout_batch + chunk_start * stride_dout_seqlen + pid_h * stride_dout_head
    )
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    ddt_ptr += (
        pid_b * stride_ddt_batch
        + pid_c * stride_ddt_chunk
        + pid_h * stride_ddt_head
        + pid_n * stride_ddt_tile
    )
    dA_cumsum_ptr += (
        pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    )
    b_ptr += (
        pid_b * stride_b_batch
        + chunk_start * stride_b_seqlen
        + (pid_h // nheads_ngroups_ratio) * stride_b_head
    )
    dstates_ptr += (
        pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk + pid_h * stride_dstates_head
    )
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + chunk_start * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    chunk_size_limit = chunk_end - chunk_start

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    dA_cs_m = tl.load(
        dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size_limit, other=0.0
    ).to(tl.float32)

    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    if not HAS_SEQ_IDX:
        # scale = tl.exp(dA_cs_last - dA_cs_m)
        scale = tl.exp(tl.minimum((dA_cs_last - dA_cs_m), 0.0))
    else:
        seq_idx_m = tl.load(
            seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1
        )
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)
        # scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(dA_cs_last - dA_cs_m), 0.0)
        scale = tl.where(
            seq_idx_m == seq_idx_last, tl.exp(tl.minimum((dA_cs_last - dA_cs_m), 0.0)), 0.0
        )
    # Might be faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
    # However, we're getting error with the Triton compiler 2.1.0 for that code path:
    # Unexpected mma -> mma layout conversion
    # Triton 2.2.0 fixes this
    offs_dstate = tl.arange(
        0, BLOCK_SIZE_DSTATE if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K
    )
    b_ptrs = b_ptr + (offs_m[:, None] * stride_b_seqlen + offs_dstate[None, :] * stride_b_dstate)
    dstates_ptrs = dstates_ptr + (
        offs_n[None, :] * stride_dstates_hdim + offs_dstate[:, None] * stride_dstates_dstate
    )
    if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128:
        b = tl.load(
            b_ptrs,
            mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate),
            other=0.0,
        )
        dstates = tl.load(
            dstates_ptrs, mask=(offs_dstate[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0
        )
        dstates = dstates.to(b_ptr.dtype.element_ty)
        acc = tl.dot(b, dstates) * scale[:, None]
    else:
        for k in range(0, dstate, BLOCK_SIZE_K):
            b = tl.load(
                b_ptrs,
                mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate - k),
                other=0.0,
            )
            dstates = tl.load(
                dstates_ptrs,
                mask=(offs_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim),
                other=0.0,
            )
            dstates = dstates.to(b_ptr.dtype.element_ty)
            acc += tl.dot(b, dstates)
            b_ptrs += BLOCK_SIZE_K * stride_b_dstate
            dstates_ptrs += BLOCK_SIZE_K * stride_dstates_dstate
        acc *= scale[:, None]

    # x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    # dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    # dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
    # ddt = tl.sum(acc * x, axis=1) * dt_m
    # ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
    # tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
    dout_ptrs = dout_ptr + (
        offs_k[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim
    )
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = chunk_size_limit
    K_MIN = pid_m * BLOCK_SIZE_M
    cb_ptrs += K_MIN * stride_cb_csize_k
    dout_ptrs += K_MIN * stride_dout_seqlen
    dA_cumsum_ptrs += K_MIN * stride_dA_cs_csize
    for k in range(K_MIN, K_MAX, BLOCK_SIZE_K):
        k = tl.multiple_of(k, BLOCK_SIZE_K)
        # For some reason setting mask to (offs_m[:, None] < chunk_size_limit) is much slower
        cb = tl.load(
            cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < K_MAX - k), other=0.0
        )
        dout = tl.load(
            dout_ptrs, mask=(offs_k[:, None] < K_MAX - k) & (offs_n[None, :] < hdim), other=0.0
        )
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < K_MAX - k, other=0.0).to(tl.float32)
        # cb *= tl.exp(dA_cs_k[None, :] - dA_cs_m[:, None])
        cb *= tl.exp(tl.minimum((dA_cs_k[None, :] - dA_cs_m[:, None]), 0.0))
        # If we don't have the (k + offs_k[None, :] < K_MAX) mask, for indices outside this range,
        # we might have dA_cs_m = 0.0 and dA_cs_k very negative, and tl.exp will return inf.
        # Multiplying with cb, which is 0.0 outside the range, will make the result NaN.
        # This will cause NaN in acc, and hence NaN in dx and ddt.
        mask = (k + offs_k[None, :] >= offs_m[:, None]) & (k + offs_k[None, :] < K_MAX)
        cb = tl.where(mask, cb, 0.0)
        cb = cb.to(dout_ptr.dtype.element_ty)
        acc += tl.dot(cb, dout)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        dout_ptrs += BLOCK_SIZE_K * stride_dout_seqlen
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
    dx = acc * dt_m[:, None]
    dx_ptr += pid_b * stride_dx_batch + chunk_start * stride_dx_seqlen + pid_h * stride_dx_head
    dx_ptrs = dx_ptr + (offs_m[:, None] * stride_dx_seqlen + offs_n[None, :] * stride_dx_hdim)
    if HAS_D:
        dout_res_ptrs = dout_ptr + (
            offs_m[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim
        )
        dout_res = tl.load(
            dout_res_ptrs,
            mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim),
            other=0.0,
        ).to(tl.float32)
        if D_HAS_HDIM:
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(
                tl.float32
            )
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        dx += dout_res * D
    tl.store(dx_ptrs, dx, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))

    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    x = tl.load(
        x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0
    ).to(tl.float32)
    if HAS_D:
        dD_ptr += (
            pid_b * stride_dD_batch
            + pid_c * stride_dD_chunk
            + pid_h * stride_dD_head
            + pid_m * stride_dD_csize
        )
        if D_HAS_HDIM:
            dD_ptrs = dD_ptr + offs_n * stride_dD_hdim
            dD = tl.sum(dout_res * x, axis=0)
            tl.store(dD_ptrs, dD, mask=offs_n < hdim)
        else:
            dD = tl.sum(dout_res * x)
            if DETERMINISTIC_REDUCTION:
                tl.store(dD_ptr + pid_n * stride_dD_hdim, dD)
            else:
                tl.atomic_add(dD_ptr, dD)
    ddt = tl.sum(acc * x, axis=1)
    ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
    if DETERMINISTIC_REDUCTION:
        tl.store(ddt_ptrs, ddt, mask=offs_m < chunk_size)
    else:
        tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)


_CHUNK_SCAN_CHUNK_STATE_BWD_DX_MIN_BLOCK_N = min(
    cfg.kwargs['BLOCK_SIZE_N'] for cfg in _chunk_scan_chunk_state_bwd_dx_kernel.configs
)


def _chunk_scan_chunk_state_bwd_dx(
    cu_chunk_seqlens, x, dt, dA_cumsum, B, CB, dout, dstates, D=None, seq_idx=None, dx=None
):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert CB.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dout.shape == x.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    deterministic = use_deterministic_mode()
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
        assert D.stride(-1) == 1
        BLOCK_SIZE_min = 32
        pid_m_tiles = triton.cdiv(chunk_size, BLOCK_SIZE_min)
        pid_n_tiles = math.ceil(headdim / _CHUNK_SCAN_CHUNK_STATE_BWD_DX_MIN_BLOCK_N)
        if D.dim() == 2:
            dD_hdim = headdim
        elif deterministic:
            dD_hdim = pid_n_tiles
        else:
            dD_hdim = 1
        dD = torch.zeros(
            pid_m_tiles, batch, nchunks, nheads, dD_hdim, device=D.device, dtype=torch.float32
        )
        dD_strides = (dD.stride(0), dD.stride(1), dD.stride(2), dD.stride(3), dD.stride(4))
    else:
        dD = None
        dD_strides = (0, 0, 0, 0, 0)
    if dx is None:
        dx = torch.empty_like(x)
    else:
        assert dx.shape == x.shape
    tile_count = math.ceil(headdim / _CHUNK_SCAN_CHUNK_STATE_BWD_DX_MIN_BLOCK_N)
    ddt, stride_ddt_tile = alloc_tile_workspace(
        (batch, nheads, nchunks, chunk_size),
        tile_count,
        torch.float32,
        dout.device,
        deterministic,
        zero_init=True,
    )
    grid_dx = lambda META: (
        triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
        batch * nchunks,
        nheads,
    )
    with torch.cuda.device(x.device.index):
        _chunk_scan_chunk_state_bwd_dx_kernel[grid_dx](
            cu_chunk_seqlens,
            x,
            CB,
            dout,
            dt,
            dA_cumsum,
            seq_idx,
            D,
            B,
            dstates,
            dx,
            ddt,
            dD,
            chunk_size,
            headdim,
            dstate,
            batch,
            seqlen,
            nheads // ngroups,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            CB.stride(0),
            CB.stride(1),
            CB.stride(2),
            CB.stride(-1),
            CB.stride(-2),
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            dout.stride(3),
            dt.stride(0),
            dt.stride(2),
            dt.stride(1),
            dt.stride(3),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),
            dA_cumsum.stride(1),
            dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            D.stride(0) if D is not None else 0,
            B.stride(0),
            B.stride(1),
            B.stride(2),
            B.stride(3),
            dstates.stride(0),
            dstates.stride(1),
            dstates.stride(2),
            dstates.stride(3),
            dstates.stride(4),
            dx.stride(0),
            dx.stride(1),
            dx.stride(2),
            dx.stride(3),
            ddt.stride(0),
            ddt.stride(2),
            ddt.stride(1),
            ddt.stride(3),
            stride_ddt_tile,
            dD_strides[1],
            dD_strides[2],
            dD_strides[3],
            dD_strides[0],
            dD_strides[4],
            D is not None,
            D.dim() == 2 if D is not None else True,
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
            IS_TRITON_22=TRITON_22,
            DETERMINISTIC_REDUCTION=deterministic,
        )
    if D is not None:
        BLOCK_SIZE_actual = _chunk_scan_chunk_state_bwd_dx_kernel.best_config.kwargs["BLOCK_SIZE_M"]
        n_valid_blocks = (chunk_size + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
        dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2))
        if D.dim() == 1:
            dD = dD.sum(dim=-1)
        dD = dD.to(dtype=D.dtype)
    ddt = finalize_tile_workspace(ddt, deterministic)
    return dx, ddt, dD


def _mamba_chunk_scan_combined_varlen_bwd(
    dout,
    x,
    dt,
    A,
    B,
    C,
    out,
    chunk_size,
    cu_chunk_seqlens,
    seq_idx,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    dx=None,
    ddt=None,
    dB=None,
    dC=None,
    dz=None,
    state_dtype=None,
):
    """Backward of the varlen SSD forward (no initial states).

    Argument (no batch dim, as in ``mamba_chunk_scan_combined_varlen``):
        dout, x: (seqlen, nheads, headdim)
        out: (seqlen, nheads, headdim) forward output before the ``z`` gate (``out_x``) if ``z`` is
            given, the output itself otherwise
        dt: (seqlen, nheads) raw dt, before ``dt_bias`` and softplus
        A: (nheads,)
        B, C: (seqlen, ngroups, dstate)
        cu_chunk_seqlens: (nchunks + 1,) chunk boundaries; a chunk never spans two sequences
        seq_idx: (nchunks,) sequence index of each chunk
        D: (nheads, headdim) or (nheads,)
        z: (seqlen, nheads, headdim)
        dt_bias: (nheads,)
        dx, ddt, dB, dC, dz: optional preallocated outputs, shaped as x, dt, B, C, z
        state_dtype: dtype of the materialized inter-chunk states and their gradients, as in the
            forward; ``x.dtype`` if None
    Return:
        dx, ddt, dA, dB, dC, dD, dz, ddt_bias
    """
    seqlen, nheads, headdim = x.shape
    _, ngroups, dstate = B.shape
    assert dout.shape == x.shape and out.shape == x.shape
    assert dt.shape == (seqlen, nheads)
    assert A.shape == (nheads,)
    assert nheads % ngroups == 0
    assert C.shape == B.shape
    assert seq_idx.shape == (cu_chunk_seqlens.shape[0] - 1,)
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    state_dtype = x.dtype if state_dtype is None else state_dtype

    # Recompute the forward intermediates with the forward kernels.
    dA_cumsum, dt_out = _chunk_cumsum_fwd(
        dt,
        A,
        chunk_size,
        cu_chunk_seqlens,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
    )
    CB = _bmm_chunk_fwd(C, B, chunk_size, cu_chunk_seqlens, output_dtype=torch.float32)
    states = _chunk_state_fwd(B, x, dt_out, dA_cumsum, cu_chunk_seqlens, states_in_fp32=True)
    # The forward state passing returns the state after each chunk; the backward kernels take the
    # state entering each chunk. Write the former one slot down a buffer whose slot 0 is the zero
    # initial state, and read the latter off its front. Across a sequence boundary the entering
    # state belongs to the previous sequence; the kernels mask it out with seq_idx.
    nchunks = states.shape[0]
    states_buf = torch.empty(
        (nchunks + 1, nheads, headdim * dstate), device=x.device, dtype=torch.float32
    )
    states_buf[0].zero_()
    _state_passing_fwd(states.flatten(-2), dA_cumsum, cu_chunk_seqlens, seq_idx, out=states_buf[1:])
    states = states_buf[:nchunks]

    # The backward kernels test seq_idx per token of the packed layout.
    token_seq_idx = torch.repeat_interleave(
        seq_idx, cu_chunk_seqlens.diff().long(), output_size=seqlen
    )[None]

    def b(t):
        return None if t is None else t.unsqueeze(0)

    x, B, C, dout, out, z = b(x), b(B), b(C), b(dout), b(out), b(z)
    dx, dB, dC, dz = b(dx), b(dB), b(dC), b(dz)
    dA_cumsum, dt_out, CB, states = b(dA_cumsum), b(dt_out), b(CB), b(states)
    dD = None
    if z is not None:
        dz, dout, dD = _chunk_scan_bwd_dz(
            cu_chunk_seqlens, x, z, out, dout, chunk_size=chunk_size, has_ddAcs=False, D=D, dz=dz
        )
    else:
        dz = None
    dstates = _chunk_scan_bwd_dstates(
        cu_chunk_seqlens, C, dA_cumsum, dout, seq_idx=token_seq_idx, dtype=states.dtype
    )
    dstates, ddA_chunk_cumsum, _, states = _state_passing_bwd(
        cu_chunk_seqlens,
        states,
        dA_cumsum[:, :, :, -1],
        dstates.flatten(-2),
        seq_idx=token_seq_idx,
        has_initial_states=False,
        dstates_dtype=state_dtype,
        states_dtype=state_dtype,
        chunk_size=chunk_size,
    )
    states = states.unflatten(-1, (headdim, dstate))
    dstates = dstates.unflatten(-1, (headdim, dstate))
    dx, ddt_chunk, dD_from_x = _chunk_scan_chunk_state_bwd_dx(
        cu_chunk_seqlens,
        x,
        dt_out,
        dA_cumsum,
        B,
        CB,
        dout,
        dstates,
        D=D,
        seq_idx=token_seq_idx,
        dx=dx,
    )
    dB_state, ddA_next = _chunk_state_bwd_db(
        cu_chunk_seqlens, x, dt_out, dA_cumsum, dstates, seq_idx=token_seq_idx, B=B, ngroups=ngroups
    )
    dC_state, ddA_cumsum_prev = _chunk_scan_bwd_dC(
        cu_chunk_seqlens,
        states.to(state_dtype),
        dA_cumsum,
        dout,
        seq_idx=token_seq_idx,
        C=C,
        ngroups=ngroups,
    )
    dCB = _chunk_scan_bwd_dcb(
        cu_chunk_seqlens, x, dt_out, dA_cumsum, dout, seq_idx=token_seq_idx, ngroups=ngroups
    )
    dCB = dCB.to(CB.dtype)
    dB_out = torch.empty_like(B) if dB is None else dB
    dC_out = torch.empty_like(C) if dC is None else dC
    _bmm_chunk_bwd(cu_chunk_seqlens, C, dCB, residual=dB_state, out=dB_out)
    _bmm_chunk_bwd(cu_chunk_seqlens, B, dCB.transpose(-1, -2), residual=dC_state, out=dC_out)
    if z is None:
        dD = dD_from_x

    ddA_cumsum_prev[..., -1] += ddA_chunk_cumsum
    ddA_prev = ddA_cumsum_prev.flip([-1]).cumsum(dim=-1).flip([-1])
    ddA = _chunk_scan_bwd_ddAcs_stable(cu_chunk_seqlens, x, dt_out, dA_cumsum, dout, CB)
    ddA += ddA_next + ddA_prev

    dt_in = dt.unsqueeze(0)
    ddt_in, dA, ddt_bias = _chunk_cumsum_bwd(
        cu_chunk_seqlens,
        ddA,
        ddt_chunk,
        dt_in,
        A,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        ddt=torch.empty_like(dt_in) if ddt is None else ddt.unsqueeze(0),
    )
    return (dx[0], ddt_in[0], dA, dB_out[0], dC_out[0], dD, None if dz is None else dz[0], ddt_bias)


class MambaChunkScanCombinedVarlenFn(torch.autograd.Function):
    """Varlen SSD with autograd: the varlen forward kernels and their varlen backward."""

    @staticmethod
    def forward(
        ctx,
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        cu_chunk_seqlens,
        last_chunk_indices,
        seq_idx,
        D=None,
        z=None,
        dt_bias=None,
        dt_softplus=False,
        dt_limit=(0.0, float("inf")),
        state_dtype=None,
    ):
        """Run the varlen forward, keeping the ungated output for the backward."""
        out = torch.empty_like(x)
        out_x = torch.empty_like(x) if z is not None else None
        _mamba_chunk_scan_combined_fwd(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            out,
            D=D,
            z=z,
            dt_bias=dt_bias,
            seq_idx=seq_idx,
            cu_chunk_seqlens=cu_chunk_seqlens,
            last_chunk_indices=last_chunk_indices,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
            state_dtype=state_dtype,
            out_x=out_x,
        )
        ctx.save_for_backward(
            x, dt, A, B, C, D, z, dt_bias, out if z is None else out_x, cu_chunk_seqlens, seq_idx
        )
        ctx.chunk_size = chunk_size
        ctx.dt_softplus = dt_softplus
        ctx.dt_limit = dt_limit
        ctx.state_dtype = state_dtype
        return out

    @staticmethod
    def backward(ctx, dout):
        """Gradients for x, dt, A, B, C, D, z and dt_bias."""
        x, dt, A, B, C, D, z, dt_bias, out_x, cu_chunk_seqlens, seq_idx = ctx.saved_tensors
        dx, ddt, dA, dB, dC, dD, dz, ddt_bias = _mamba_chunk_scan_combined_varlen_bwd(
            dout,
            x,
            dt,
            A,
            B,
            C,
            out_x,
            ctx.chunk_size,
            cu_chunk_seqlens,
            seq_idx,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=ctx.dt_softplus,
            dt_limit=ctx.dt_limit,
            state_dtype=ctx.state_dtype,
        )
        return dx, ddt, dA, dB, dC, None, None, None, None, dD, dz, ddt_bias, None, None, None


def mamba_chunk_scan_combined_varlen_fn(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    cu_chunk_seqlens,
    last_chunk_indices,
    seq_idx,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    state_dtype=None,
):
    """Differentiable varlen SSD for training on packed sequences.

    Same arguments as ``mamba_chunk_scan_combined_varlen`` minus ``out``,
    ``initial_states``, ``return_raw_states`` and ``out_x``.
    Chunks restart at every sequence start, so the output of a sequence does not depend on what
    else is packed with it (for a fixed kernel launch configuration).

    Argument:
        x: (seqlen, nheads, headdim)
        dt: (seqlen, nheads)
        A: (nheads)
        B, C: (seqlen, ngroups, dstate)
        chunk_size: int
        cu_chunk_seqlens: (nchunks + 1,)
        last_chunk_indices: (batch,)
        seq_idx: (nchunks,)
        D: (nheads, headdim) or (nheads,)
        z: (seqlen, nheads, headdim)
        dt_bias: (nheads,)
        state_dtype: dtype of the materialized inter-chunk states in the forward and the backward;
            the dtype of ``C`` (forward) and ``x`` (backward) if None
    Return:
        out: (seqlen, nheads, headdim)
    """
    return MambaChunkScanCombinedVarlenFn.apply(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        cu_chunk_seqlens,
        last_chunk_indices,
        seq_idx,
        D,
        z,
        dt_bias,
        dt_softplus,
        dt_limit,
        state_dtype,
    )
