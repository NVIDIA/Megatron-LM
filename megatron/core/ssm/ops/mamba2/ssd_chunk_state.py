# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from:
#   https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_chunk_state.py
# Adapted from vLLM project (Apache-2.0).
# The varlen backward is adapted from (Apache-2.0):
# https://github.com/state-spaces/mamba/blob/v2.3.2.post1/mamba_ssm/ops/triton/ssd_chunk_state.py

import math

import torch
import triton
import triton.language as tl
from packaging import version

from megatron.core.ssm.ops.common.determinism import alloc_tile_workspace, finalize_tile_workspace
from megatron.core.tuning import autotune_configs, use_deterministic_mode

try:
    TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")
except:
    raise ImportError("Triton version 3.0.0 or higher is required")

if TRITON3:

    @triton.jit
    def softplus(dt):  # pylint: disable=C0116
        dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)
        return dt

else:

    @triton.jit
    def softplus(dt):  # pylint: disable=C0116
        dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
        return dt


@triton.autotune(
    configs=autotune_configs(
        [
            triton.Config({"BLOCK_SIZE_H": 2}),
            triton.Config({"BLOCK_SIZE_H": 4}),
            triton.Config({"BLOCK_SIZE_H": 8}),
            triton.Config({"BLOCK_SIZE_H": 16}),
            triton.Config({"BLOCK_SIZE_H": 32}),
            triton.Config({"BLOCK_SIZE_H": 64}),
        ]
    ),
    key=["chunk_size", "nheads"],
)
@triton.jit
def _chunk_cumsum_fwd_kernel(
    # Pointers to matrices
    dt_ptr,
    A_ptr,
    dt_bias_ptr,
    dt_out_ptr,
    dA_cumsum_ptr,
    chunk_offsets_ptr,
    target_rows_ptr,
    # Matrix dimension
    seqlen,
    nheads: tl.constexpr,
    chunk_size: tl.constexpr,
    HAS_TARGET_ROWS: tl.constexpr,
    dt_min: tl.constexpr,
    dt_max: tl.constexpr,
    # Strides
    stride_dt_seqlen: tl.int64,
    stride_dt_head: tl.constexpr,
    stride_A_head: tl.constexpr,
    stride_dt_bias_head: tl.constexpr,
    stride_dt_out_head: tl.int64,
    stride_dt_out_chunk: tl.int64,
    stride_dt_out_csize: tl.constexpr,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_CHUNK: tl.constexpr,
):
    # if dt is long, may cause problems, so use 64 bit
    # https://github.com/triton-lang/triton/issues/1058
    pid_c = tl.program_id(axis=0).to(tl.int64)
    pid_h = tl.program_id(axis=1)
    if HAS_TARGET_ROWS:
        if tl.load(target_rows_ptr + pid_c) < 0:
            return

    chunk_seqlen_start = tl.load(chunk_offsets_ptr + pid_c)
    if HAS_TARGET_ROWS:
        # Fixed windows need only a start offset.
        chunk_seqlen_end = chunk_seqlen_start + chunk_size
    else:
        chunk_seqlen_end = tl.load(chunk_offsets_ptr + pid_c + 1)

    dt_ptr += chunk_seqlen_start * stride_dt_seqlen
    dt_out_ptr += pid_c * stride_dt_out_chunk
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    dt_ptrs = dt_ptr + (offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen)
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = dt_out_ptr + (
        offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_csize
    )
    dA_cs_ptrs = dA_cumsum_ptr + (
        offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize
    )
    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    dt = tl.load(
        dt_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0
    ).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(
            dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0
        ).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, softplus(dt), dt)

    dt = tl.clamp(dt, dt_min, dt_max)
    dt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0)
    tl.store(dt_out_ptrs, dt, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)
    tl.store(dA_cs_ptrs, dA_cs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))


@triton.autotune(
    configs=autotune_configs(
        [
            triton.Config(
                {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
                num_stages=5,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
                num_stages=5,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
                num_stages=4,
                num_warps=2,
            ),
        ]
    ),
    key=["hdim", "dstate", "chunk_size"],
)
@triton.jit
def _chunk_state_fwd_kernel(
    # Pointers to matrices
    x_ptr,
    b_ptr,
    states_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    chunk_offsets_ptr,
    chunk_flags_ptr,
    # Matrix dimensions
    hdim: tl.constexpr,
    dstate: tl.constexpr,
    chunk_size: tl.constexpr,
    seqlen,
    nheads_ngroups_ratio: tl.constexpr,
    # Strides
    stride_x_seqlen: tl.int64,
    stride_x_head: tl.int64,
    stride_x_hdim: tl.constexpr,
    stride_b_seqlen: tl.int64,
    stride_b_head: tl.int64,
    stride_b_dstate: tl.constexpr,
    stride_states_chunk: tl.int64,
    stride_states_head: tl.int64,
    stride_states_hdim: tl.int64,
    stride_states_dstate: tl.constexpr,
    stride_dt_head: tl.int64,
    stride_dt_chunk: tl.int64,
    stride_dt_csize: tl.constexpr,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    # Meta-parameters
    HAS_CHUNK_FLAGS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_c = tl.program_id(axis=1).to(tl.int64)
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    if HAS_CHUNK_FLAGS:
        # Only completed chunks need a boundary state.
        if tl.load(chunk_flags_ptr + pid_c) == 0:
            return
    chunk_seqlen_start = tl.load(chunk_offsets_ptr + pid_c)
    if HAS_CHUNK_FLAGS:
        # Chunk flags are paired with fixed-window starts.
        chunk_seqlen_end = chunk_seqlen_start + chunk_size
    else:
        chunk_seqlen_end = tl.load(chunk_offsets_ptr + pid_c + 1)
    b_ptr += chunk_seqlen_start * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    x_ptr += chunk_seqlen_start * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(
            tl.float32
        )
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        scale = tl.exp(tl.minimum(dA_cs_last - dA_cs_k, 0.0)) * dt_k
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)

        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    states = acc.to(states_ptr.dtype.element_ty)

    states_ptr += pid_c * stride_states_chunk + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (
        offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate
    )
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)


def _chunk_cumsum_fwd(
    dt,
    A,
    chunk_size,
    cu_chunk_seqlens,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    chunk_starts=None,
    target_rows=None,
):
    seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    has_target_rows = target_rows is not None
    assert (
        chunk_starts is not None
    ) == has_target_rows, "target_rows and chunk_starts must be provided together"
    if has_target_rows:
        chunk_offsets = chunk_starts
        nchunks = chunk_starts.shape[0]
    else:
        chunk_offsets = cu_chunk_seqlens
        nchunks = cu_chunk_seqlens.shape[0] - 1
    dt_out = torch.empty(nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    dA_cumsum = torch.empty(nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    grid_chunk_cs = lambda META: (nchunks, triton.cdiv(nheads, META["BLOCK_SIZE_H"]))
    with torch.cuda.device(dt.device.index):
        _chunk_cumsum_fwd_kernel[grid_chunk_cs](
            dt_ptr=dt,
            A_ptr=A,
            dt_bias_ptr=dt_bias,
            dt_out_ptr=dt_out,
            dA_cumsum_ptr=dA_cumsum,
            chunk_offsets_ptr=chunk_offsets,
            target_rows_ptr=target_rows,
            seqlen=seqlen,
            nheads=nheads,
            chunk_size=chunk_size,
            dt_min=dt_limit[0],
            dt_max=dt_limit[1],
            stride_dt_seqlen=dt.stride(0),
            stride_dt_head=dt.stride(1),
            stride_A_head=A.stride(0),
            stride_dt_bias_head=dt_bias.stride(0) if dt_bias is not None else 0,
            stride_dt_out_head=dt_out.stride(0),
            stride_dt_out_chunk=dt_out.stride(1),
            stride_dt_out_csize=dt_out.stride(2),
            stride_dA_cs_head=dA_cumsum.stride(0),
            stride_dA_cs_chunk=dA_cumsum.stride(1),
            stride_dA_cs_csize=dA_cumsum.stride(2),
            HAS_TARGET_ROWS=has_target_rows,
            DT_SOFTPLUS=dt_softplus,
            HAS_DT_BIAS=dt_bias is not None,
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        )
    return dA_cumsum, dt_out


def _chunk_state_fwd(
    B,
    x,
    dt,
    dA_cumsum,
    cu_chunk_seqlens,
    states=None,
    states_in_fp32=True,
    chunk_flags=None,
    chunk_starts=None,
):
    has_chunk_flags = chunk_flags is not None
    assert (
        chunk_starts is not None
    ) == has_chunk_flags, "chunk_flags and chunk_starts must be provided together"
    if has_chunk_flags:
        chunk_offsets = chunk_starts
    else:
        chunk_offsets = cu_chunk_seqlens
    seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (seqlen, ngroups, dstate)
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape

    if states is not None:
        assert states.shape == (nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty(
            (nchunks, nheads, headdim, dstate), device=x.device, dtype=states_dtype
        )

    grid = lambda META: (
        triton.cdiv(headdim, META["BLOCK_SIZE_M"]) * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        nchunks,
        nheads,
    )
    with torch.cuda.device(x.device.index):
        _chunk_state_fwd_kernel[grid](
            x_ptr=x,
            b_ptr=B,
            states_ptr=states,
            dt_ptr=dt,
            dA_cumsum_ptr=dA_cumsum,
            chunk_offsets_ptr=chunk_offsets,
            chunk_flags_ptr=chunk_flags,
            hdim=headdim,
            dstate=dstate,
            chunk_size=chunk_size,
            seqlen=seqlen,
            nheads_ngroups_ratio=nheads // ngroups,
            stride_x_seqlen=x.stride(0),
            stride_x_head=x.stride(1),
            stride_x_hdim=x.stride(2),
            stride_b_seqlen=B.stride(0),
            stride_b_head=B.stride(1),
            stride_b_dstate=B.stride(2),
            stride_states_chunk=states.stride(0),
            stride_states_head=states.stride(1),
            stride_states_hdim=states.stride(2),
            stride_states_dstate=states.stride(3),
            stride_dt_head=dt.stride(0),
            stride_dt_chunk=dt.stride(1),
            stride_dt_csize=dt.stride(2),
            stride_dA_cs_head=dA_cumsum.stride(0),
            stride_dA_cs_chunk=dA_cumsum.stride(1),
            stride_dA_cs_csize=dA_cumsum.stride(2),
            HAS_CHUNK_FLAGS=has_chunk_flags,
        )
    return states


@triton.autotune(
    configs=autotune_configs(
        [
            triton.Config(
                {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
                num_stages=5,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
                num_stages=5,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
                num_stages=4,
                num_warps=2,
            ),
        ]
    ),
    key=["hdim", "dstate", "chunk_size"],
)
@triton.jit
def _chunk_state_varlen_kernel(
    x_ptr,
    b_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    chunk_states_ptr,
    cu_seqlens_ptr,
    last_chunk_indices_ptr,
    cu_chunk_seqlens_ptr,
    states_ptr,
    initstates_ptr,
    hdim: tl.constexpr,
    dstate: tl.constexpr,
    chunk_size: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    stride_x_seqlen: tl.int64,
    stride_x_head: tl.int64,
    stride_x_hdim: tl.constexpr,
    stride_b_seqlen: tl.int64,
    stride_b_head: tl.int64,
    stride_b_dstate: tl.constexpr,
    stride_dt_head: tl.int64,
    stride_dt_chunk: tl.int64,
    stride_dt_csize: tl.constexpr,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    stride_chunk_states_chunk: tl.int64,
    stride_chunk_states_head: tl.int64,
    stride_chunk_states_hdim: tl.int64,
    stride_chunk_states_dstate: tl.constexpr,
    stride_states_batch: tl.int64,
    stride_states_head: tl.int64,
    stride_states_hdim: tl.int64,
    stride_states_dstate: tl.constexpr,
    stride_init_states_batch: tl.int64,
    stride_init_states_head: tl.int64,
    stride_init_states_hdim: tl.int64,
    stride_init_states_dstate: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_INITSTATES: tl.constexpr,
    USE_LAST_CHUNK_INDICES: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    end_idx = tl.load(cu_seqlens_ptr + pid_b + 1)
    start_idx = tl.load(cu_seqlens_ptr + pid_b)
    if USE_LAST_CHUNK_INDICES:
        pid_c = tl.load(last_chunk_indices_ptr + pid_b).to(tl.int64)
        chunk_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
        chunk_size_limit = tl.load(cu_chunk_seqlens_ptr + pid_c + 1) - chunk_start
    else:
        pid_c = (end_idx - 1) // chunk_size
        chunk_start = pid_c * chunk_size
        chunk_size_limit = end_idx - chunk_start
    b_ptr += chunk_start * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    x_ptr += chunk_start * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    chunk_states_ptr += pid_c * stride_chunk_states_chunk + pid_h * stride_chunk_states_head

    if HAS_INITSTATES:
        initstates_ptr += pid_h * stride_init_states_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (end_idx - 1 - chunk_start) * stride_dA_cs_csize).to(
        tl.float32
    )
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    start_idx_cur = tl.maximum(start_idx - chunk_start, 0)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < hdim)
            & (offs_k[None, :] < chunk_size_limit - k)
            & (offs_k[None, :] >= start_idx_cur - k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k)
            & (offs_n[None, :] < dstate)
            & (offs_k[:, None] >= start_idx_cur - k),
            other=0.0,
        ).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(
            tl.float32
        )
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        scale = tl.where(
            (offs_k >= start_idx_cur - k) & (offs_k < chunk_size_limit - k),
            tl.exp(tl.minimum(dA_cs_last - dA_cs_k, 0.0)) * dt_k,
            0.0,
        )
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    if (start_idx < chunk_start) or (HAS_INITSTATES):
        dA_cs_boundary = 0.0
        if not HAS_INITSTATES:
            past_states_ptrs = chunk_states_ptr + (
                offs_m[:, None] * stride_chunk_states_hdim
                + offs_n[None, :] * stride_chunk_states_dstate
            )
        else:
            if start_idx < chunk_start:
                past_states_ptrs = chunk_states_ptr + (
                    offs_m[:, None] * stride_chunk_states_hdim
                    + offs_n[None, :] * stride_chunk_states_dstate
                )
            else:
                past_states_ptrs = initstates_ptr + (
                    pid_b * stride_init_states_batch
                    + offs_m[:, None] * stride_init_states_hdim
                    + offs_n[None, :] * stride_init_states_dstate
                )
                if start_idx > chunk_start:
                    dA_cs_boundary = tl.load(
                        dA_cumsum_ptr + (start_idx - chunk_start - 1) * stride_dA_cs_csize
                    ).to(tl.float32)

        past_states = tl.load(
            past_states_ptrs, mask=(offs_m[:, None] < hdim) & (offs_n[None, :] < dstate), other=0.0
        ).to(tl.float32)
        scale = tl.exp(tl.minimum(dA_cs_last - dA_cs_boundary, 0.0))
        acc += past_states * scale

    states = acc.to(states_ptr.dtype.element_ty)
    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (
        offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate
    )
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)


def chunk_state_varlen(
    B,
    x,
    dt,
    dA_cumsum,
    cu_seqlens,
    chunk_states,
    initial_states=None,
    last_chunk_indices=None,
    cu_chunk_seqlens=None,
):
    """Compute per-sequence final SSM state from chunk states.

    Correct when sequences share chunks.
    """
    total_seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    batch = cu_seqlens.shape[0] - 1
    cu_seqlens = cu_seqlens.contiguous()
    assert nheads % ngroups == 0
    assert B.shape == (total_seqlen, ngroups, dstate)
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert chunk_states.shape == (nchunks, nheads, headdim, dstate)
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
    use_last_chunk = last_chunk_indices is not None and cu_chunk_seqlens is not None
    if use_last_chunk:
        last_chunk_indices = last_chunk_indices.contiguous().to(x.device)
        cu_chunk_seqlens = cu_chunk_seqlens.contiguous().to(x.device)
    else:
        last_chunk_indices = torch.zeros(1, dtype=torch.int64, device=x.device)
        cu_chunk_seqlens = cu_seqlens

    states = torch.empty(
        batch, nheads, headdim, dstate, dtype=chunk_states.dtype, device=chunk_states.device
    )
    initial_states_strides = (
        (
            initial_states.stride(0),
            initial_states.stride(1),
            initial_states.stride(2),
            initial_states.stride(3),
        )
        if initial_states is not None
        else (0, 0, 0, 0)
    )
    grid = lambda META: (
        triton.cdiv(headdim, META["BLOCK_SIZE_M"]) * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        batch,
        nheads,
    )
    with torch.cuda.device(x.device.index):
        _chunk_state_varlen_kernel[grid](
            x_ptr=x,
            b_ptr=B,
            dt_ptr=dt,
            dA_cumsum_ptr=dA_cumsum,
            chunk_states_ptr=chunk_states,
            cu_seqlens_ptr=cu_seqlens,
            last_chunk_indices_ptr=last_chunk_indices,
            cu_chunk_seqlens_ptr=cu_chunk_seqlens,
            states_ptr=states,
            initstates_ptr=initial_states,
            hdim=headdim,
            dstate=dstate,
            chunk_size=chunk_size,
            nheads_ngroups_ratio=nheads // ngroups,
            stride_x_seqlen=x.stride(0),
            stride_x_head=x.stride(1),
            stride_x_hdim=x.stride(2),
            stride_b_seqlen=B.stride(0),
            stride_b_head=B.stride(1),
            stride_b_dstate=B.stride(2),
            stride_dt_head=dt.stride(0),
            stride_dt_chunk=dt.stride(1),
            stride_dt_csize=dt.stride(2),
            stride_dA_cs_head=dA_cumsum.stride(0),
            stride_dA_cs_chunk=dA_cumsum.stride(1),
            stride_dA_cs_csize=dA_cumsum.stride(2),
            stride_chunk_states_chunk=chunk_states.stride(0),
            stride_chunk_states_head=chunk_states.stride(1),
            stride_chunk_states_hdim=chunk_states.stride(2),
            stride_chunk_states_dstate=chunk_states.stride(3),
            stride_states_batch=states.stride(0),
            stride_states_head=states.stride(1),
            stride_states_hdim=states.stride(2),
            stride_states_dstate=states.stride(3),
            stride_init_states_batch=initial_states_strides[0],
            stride_init_states_head=initial_states_strides[1],
            stride_init_states_hdim=initial_states_strides[2],
            stride_init_states_dstate=initial_states_strides[3],
            HAS_INITSTATES=initial_states is not None,
            USE_LAST_CHUNK_INDICES=use_last_chunk,
        )
    return states


# ---------------------------------------------------------------------------------------------
# Backward. Adapted from mamba_ssm (state-spaces/mamba v2.3.2.post1) for varlen chunks: chunk c
# covers tokens [cu_chunk_seqlens[c], cu_chunk_seqlens[c + 1]) and never spans two sequences.
# Token addressing uses those offsets; seq_idx is per token of the (unpadded) packed layout, so
# the seq_idx tests of the original kernels stay valid unchanged.
# pylint: disable=possibly-used-before-assignment  (variables guarded by the same
# constexpr flag, e.g. HAS_SEQ_IDX, at definition and use)
# ---------------------------------------------------------------------------------------------


def init_to_zero(names):
    """Autotune pre_hook: zero the named output buffers (they are accumulated with atomics)."""
    return lambda nargs: [nargs[name].zero_() for name in names if nargs[name] is not None]


@triton.autotune(
    configs=autotune_configs(
        [
            triton.Config(
                {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128},
                num_stages=3,
                num_warps=4,
                pre_hook=init_to_zero(["ddA_cumsum_ptr"]),
            ),
            triton.Config(
                {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32},
                num_stages=3,
                num_warps=4,
                pre_hook=init_to_zero(["ddA_cumsum_ptr"]),
            ),
            triton.Config(
                {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128},
                num_stages=3,
                num_warps=4,
                pre_hook=init_to_zero(["ddA_cumsum_ptr"]),
            ),
            triton.Config(
                {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64},
                num_stages=3,
                num_warps=4,
                pre_hook=init_to_zero(["ddA_cumsum_ptr"]),
            ),
            triton.Config(
                {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64},
                num_stages=3,
                num_warps=4,
                pre_hook=init_to_zero(["ddA_cumsum_ptr"]),
            ),
            triton.Config(
                {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32},
                num_stages=3,
                num_warps=4,
                pre_hook=init_to_zero(["ddA_cumsum_ptr"]),
            ),
            triton.Config(
                {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64},
                num_stages=3,
                num_warps=4,
                pre_hook=init_to_zero(["ddA_cumsum_ptr"]),
            ),
            triton.Config(
                {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32},
                num_stages=3,
                num_warps=4,
                pre_hook=init_to_zero(["ddA_cumsum_ptr"]),
            ),
        ]
    ),
    key=['chunk_size', 'dstate', 'hdim'],
)
@triton.jit
def _chunk_state_bwd_db_kernel(
    chunk_offsets_ptr,  # varlen: (nchunks + 1,) token offsets
    # Pointers to matrices
    x_ptr,
    dstates_ptr,
    b_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    seq_idx_ptr,
    db_ptr,
    ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size,
    dstate,
    hdim,
    batch,
    seqlen,
    nheads,
    nheads_per_program,
    ngroups,
    # Strides
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    stride_dstates_batch,
    stride_dstates_chunk,
    stride_states_head,
    stride_states_hdim,
    stride_states_dstate,
    stride_b_batch,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dstate,
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
    stride_db_batch,
    stride_db_seqlen,
    stride_db_split,
    stride_db_group,
    stride_db_dstate,
    stride_ddA_cs_batch,
    stride_ddA_cs_chunk,
    stride_ddA_cs_head,
    stride_ddA_cs_csize,
    stride_ddA_tile,
    # Meta-parameters
    HAS_DDA_CS: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    DETERMINISTIC_REDUCTION: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1).to(tl.int64)
    pid_c = pid_bc // batch
    chunk_start = tl.load(chunk_offsets_ptr + pid_c).to(tl.int64)
    chunk_end = tl.load(chunk_offsets_ptr + pid_c + 1).to(tl.int64)
    pid_b = pid_bc - pid_c * batch
    pid_sg = tl.program_id(axis=2)
    pid_s = pid_sg // ngroups
    pid_g = pid_sg - pid_s * ngroups
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += (
        pid_b * stride_x_batch
        + chunk_start * stride_x_seqlen
        + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_x_head
    )
    db_ptr += (
        pid_b * stride_db_batch
        + chunk_start * stride_db_seqlen
        + pid_g * stride_db_group
        + pid_s * stride_db_split
    )
    dstates_ptr += (
        pid_b * stride_dstates_batch
        + pid_c * stride_dstates_chunk
        + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_states_head
    )
    dt_ptr += (
        pid_b * stride_dt_batch
        + pid_c * stride_dt_chunk
        + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_dt_head
    )
    dA_cumsum_ptr += (
        pid_b * stride_dA_cs_batch
        + pid_c * stride_dA_cs_chunk
        + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_dA_cs_head
    )
    if HAS_DDA_CS:
        b_ptr += pid_b * stride_b_batch + chunk_start * stride_b_seqlen + pid_g * stride_b_head
        ddA_cumsum_ptr += (
            pid_b * stride_ddA_cs_batch
            + pid_c * stride_ddA_cs_chunk
            + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_ddA_cs_head
            + pid_n * stride_ddA_tile
        )
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + chunk_start * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_k[None, :] * stride_x_hdim)
    dstates_ptrs = dstates_ptr + (
        offs_n[None, :] * stride_states_dstate + offs_k[:, None] * stride_states_hdim
    )
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_m * stride_dA_cs_csize
    if HAS_DDA_CS:
        b_ptrs = b_ptr + (offs_m[:, None] * stride_b_seqlen + offs_n[None, :] * stride_b_dstate)
        ddA_cumsum_ptrs = ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize

    chunk_size_limit = chunk_end - chunk_start
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAS_DDA_CS:
        b = tl.load(
            b_ptrs,
            mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)
    if HAS_SEQ_IDX:
        seq_idx_m = tl.load(
            seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1
        )
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)
    nheads_iter = min(nheads_per_program, nheads // ngroups - pid_s * nheads_per_program)
    for h in range(nheads_iter):
        x = tl.load(
            x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim), other=0.0
        )
        dstates = tl.load(
            dstates_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < dstate), other=0.0
        )
        dstates = dstates.to(x_ptrs.dtype.element_ty)
        db = tl.dot(x, dstates)
        dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
        dA_cs_m = tl.load(dA_cumsum_ptrs, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
        dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
        if not HAS_SEQ_IDX:
            # scale = tl.exp(dA_cs_last - dA_cs_m)
            scale = tl.exp(tl.minimum((dA_cs_last - dA_cs_m), 0.0))
        else:
            # scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(dA_cs_last - dA_cs_m), 0.0)
            scale = tl.where(
                seq_idx_m == seq_idx_last, tl.exp(tl.minimum((dA_cs_last - dA_cs_m), 0.0)), 0.0
            )
        db *= (scale * dt_m)[:, None]
        if HAS_DDA_CS:
            # This is the gradient wrt (dA_cs_last - dA_cs_m), i.e. the exclusive reverse cumsum
            ddA_cs = tl.sum(db * b, axis=1)
            if DETERMINISTIC_REDUCTION:
                tl.store(
                    ddA_cumsum_ptrs + stride_ddA_cs_csize, ddA_cs, mask=offs_m < chunk_size - 1
                )
            else:
                tl.atomic_add(
                    ddA_cumsum_ptrs + stride_ddA_cs_csize, ddA_cs, mask=offs_m < chunk_size - 1
                )
        acc += db
        x_ptrs += stride_x_head
        dstates_ptrs += stride_states_head
        dt_ptrs += stride_dt_head
        dA_cumsum_ptr += stride_dA_cs_head
        dA_cumsum_ptrs += stride_dA_cs_head
        if HAS_DDA_CS:
            ddA_cumsum_ptrs += stride_ddA_cs_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # if HAS_SEQ_IDX:
    #     seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)
    #     acc = tl.where(seq_idx_m[:, None] == seq_idx_last, acc, 0.0)
    db_ptrs = db_ptr + (offs_m[:, None] * stride_db_seqlen + offs_n[None, :] * stride_db_dstate)
    tl.store(db_ptrs, acc, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate))


_CHUNK_STATE_BWD_DB_MIN_BLOCK_N = min(
    cfg.kwargs['BLOCK_SIZE_N'] for cfg in _chunk_state_bwd_db_kernel.configs
)


def _chunk_state_bwd_db(
    cu_chunk_seqlens, x, dt, dA_cumsum, dstates, seq_idx=None, B=None, ngroups=1
):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    dstate = dstates.shape[-1]
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    deterministic = use_deterministic_mode()
    if B is not None:
        assert B.shape == (batch, seqlen, ngroups, dstate)
        B_strides = (B.stride(0), B.stride(1), B.stride(2), B.stride(3))
        # Use torch.empty since the Triton kernel will call init_to_zero
        tile_count = math.ceil(dstate / _CHUNK_STATE_BWD_DB_MIN_BLOCK_N)
        ddA_cumsum, stride_ddA_tile = alloc_tile_workspace(
            (batch, nheads, nchunks, chunk_size),
            tile_count,
            torch.float32,
            x.device,
            deterministic,
            zero_init=True,
        )
        ddA_cumsum_strides = (
            ddA_cumsum.stride(0),
            ddA_cumsum.stride(2),
            ddA_cumsum.stride(1),
            ddA_cumsum.stride(3),
        )
    else:
        B_strides = (0, 0, 0, 0)
        ddA_cumsum = None
        ddA_cumsum_strides = (0, 0, 0, 0)
        stride_ddA_tile = 0
    nheads_ngroups_ratio = nheads // ngroups
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    nheads_per_program = max(
        min(math.ceil(batch * nchunks * nheads / sm_count), nheads_ngroups_ratio), 1
    )
    nsplits = triton.cdiv(nheads_ngroups_ratio, nheads_per_program)
    dB = torch.empty(batch, seqlen, nsplits, ngroups, dstate, device=x.device, dtype=torch.float32)
    grid_db = lambda META: (
        triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
        batch * nchunks,
        nsplits * ngroups,
    )
    with torch.cuda.device(x.device.index):
        _chunk_state_bwd_db_kernel[grid_db](
            cu_chunk_seqlens,
            x,
            dstates,
            B,
            dt,
            dA_cumsum,
            seq_idx,
            dB,
            ddA_cumsum,
            chunk_size,
            dstate,
            headdim,
            batch,
            seqlen,
            nheads,
            nheads_per_program,
            ngroups,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            dstates.stride(0),
            dstates.stride(1),
            dstates.stride(2),
            dstates.stride(3),
            dstates.stride(4),
            *B_strides,
            dt.stride(0),
            dt.stride(2),
            dt.stride(1),
            dt.stride(3),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),
            dA_cumsum.stride(1),
            dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            dB.stride(0),
            dB.stride(1),
            dB.stride(2),
            dB.stride(3),
            dB.stride(4),
            *ddA_cumsum_strides,
            stride_ddA_tile,
            HAS_DDA_CS=ddA_cumsum is not None,
            HAS_SEQ_IDX=seq_idx is not None,
            DETERMINISTIC_REDUCTION=deterministic,
            BLOCK_SIZE_K=max(triton.next_power_of_2(headdim), 16),
        )
    dB = dB.sum(2)
    if ddA_cumsum is not None:
        ddA_cumsum = finalize_tile_workspace(ddA_cumsum, deterministic)
        # The first element of ddA_cumsum is always zero, since that dA_cumsum does not contribute
        # to the state of the chunk.
        # torch.cumsum(ddA_cumsum[..., 1:], dim=-1, out=ddA_cumsum[..., 1:])
        # But it's easier to just do the cumsum for all elements, the result will be the same.
        torch.cumsum(ddA_cumsum, dim=-1, out=ddA_cumsum)
    return dB if B is None else (dB, ddA_cumsum)


@triton.autotune(
    configs=autotune_configs(
        [
            triton.Config({'BLOCK_SIZE_H': 1}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
            triton.Config({'BLOCK_SIZE_H': 2}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
            triton.Config({'BLOCK_SIZE_H': 4}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
            triton.Config({'BLOCK_SIZE_H': 8}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
            triton.Config({'BLOCK_SIZE_H': 16}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
            triton.Config({'BLOCK_SIZE_H': 32}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
            triton.Config({'BLOCK_SIZE_H': 64}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
        ]
    ),
    key=['chunk_size', 'nheads'],
)
@triton.jit
def _chunk_cumsum_bwd_kernel(
    chunk_offsets_ptr,  # varlen: (nchunks + 1,) token offsets
    # Pointers to matrices
    ddA_ptr,
    ddt_out_ptr,
    dt_ptr,
    A_ptr,
    dt_bias_ptr,
    ddt_ptr,
    dA_ptr,
    ddt_bias_ptr,
    # Matrix dimensions
    batch,
    seqlen,
    nheads,
    chunk_size,
    dt_min,
    dt_max,
    # Strides
    stride_ddA_batch,
    stride_ddA_chunk,
    stride_ddA_head,
    stride_ddA_csize,
    stride_ddt_out_batch,
    stride_ddt_out_chunk,
    stride_ddt_out_head,
    stride_ddt_out_csize,
    stride_dt_batch,
    stride_dt_seqlen,
    stride_dt_head,
    stride_A_head,
    stride_dt_bias_head,
    stride_ddt_batch,
    stride_ddt_seqlen,
    stride_ddt_head,
    stride_dA_batch,
    stride_dA_chunk,
    stride_dA_head,
    stride_ddt_bias_batch,
    stride_ddt_bias_chunk,
    stride_ddt_bias_head,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_CHUNK: tl.constexpr,
    DETERMINISTIC_REDUCTION: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1).to(tl.int64)
    chunk_start = tl.load(chunk_offsets_ptr + pid_c).to(tl.int64)
    chunk_end = tl.load(chunk_offsets_ptr + pid_c + 1).to(tl.int64)
    pid_h = tl.program_id(axis=2).to(tl.int64)
    ddt_out_ptr += pid_b * stride_ddt_out_batch + pid_c * stride_ddt_out_chunk
    ddA_ptr += pid_b * stride_ddA_batch + pid_c * stride_ddA_chunk
    dt_ptr += pid_b * stride_dt_batch + chunk_start * stride_dt_seqlen
    ddt_ptr += pid_b * stride_ddt_batch + chunk_start * stride_ddt_seqlen

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    ddt_out_ptrs = ddt_out_ptr + (
        offs_h[:, None] * stride_ddt_out_head + offs_c[None, :] * stride_ddt_out_csize
    )
    ddA_ptrs = ddA_ptr + (offs_h[:, None] * stride_ddA_head + offs_c[None, :] * stride_ddA_csize)
    dt_ptrs = dt_ptr + (offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen)
    ddt_ptrs = ddt_ptr + (offs_h[:, None] * stride_ddt_head + offs_c[None, :] * stride_ddt_seqlen)
    A_ptrs = A_ptr + offs_h * stride_A_head
    chunk_size_limit = chunk_end - chunk_start

    ddA = tl.load(
        ddA_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0
    ).to(tl.float32)
    ddt_out = tl.load(
        ddt_out_ptrs,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit),
        other=0.0,
    ).to(tl.float32)
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    ddt = ddA * A[:, None] + ddt_out
    dt = tl.load(
        dt_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0
    ).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(
            dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0
        ).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt_presoftplus = dt
        dt = tl.where(dt <= 20.0, softplus(dt), dt)
    clamp_mask = (dt < dt_min) | (dt > dt_max)
    # As of Triton 2.2.0, tl.clamp is not available yet
    # dt = tl.clamp(dt, dt_min, dt_max)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0)
    ddt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), ddt, 0.0)
    ddt = tl.where(clamp_mask, 0.0, ddt)
    if DT_SOFTPLUS:
        ddt = tl.where(dt_presoftplus <= 20.0, ddt * tl.sigmoid(dt_presoftplus), ddt)
    tl.store(ddt_ptrs, ddt, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit))
    dA = tl.sum(ddA * dt, axis=1)
    dA_ptr += pid_b * stride_dA_batch + pid_c * stride_dA_chunk
    if DETERMINISTIC_REDUCTION:
        tl.store(dA_ptr + offs_h * stride_dA_head, dA, mask=offs_h < nheads)
    else:
        tl.atomic_add(dA_ptr + offs_h * stride_dA_head, dA, mask=offs_h < nheads)
    if HAS_DT_BIAS:
        ddt_bias = tl.sum(ddt, axis=1)
        ddt_bias_ptr += pid_b * stride_ddt_bias_batch + pid_c * stride_ddt_bias_chunk
        if DETERMINISTIC_REDUCTION:
            tl.store(ddt_bias_ptr + offs_h * stride_ddt_bias_head, ddt_bias, mask=offs_h < nheads)
        else:
            tl.atomic_add(
                ddt_bias_ptr + offs_h * stride_ddt_bias_head, ddt_bias, mask=offs_h < nheads
            )


def _chunk_cumsum_bwd(
    cu_chunk_seqlens,
    ddA,
    ddt_out,
    dt,
    A,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    ddt=None,
):
    batch, seqlen, nheads = dt.shape
    _, _, nchunks, chunk_size = ddA.shape
    assert ddA.shape == (batch, nheads, nchunks, chunk_size)
    assert ddt_out.shape == (batch, nheads, nchunks, chunk_size)
    assert A.shape == (nheads,)
    deterministic = use_deterministic_mode()
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
        if deterministic:
            ddt_bias_workspace = torch.zeros(
                batch, nchunks, nheads, device=dt.device, dtype=torch.float32
            )
            ddt_bias = torch.empty_like(dt_bias, dtype=torch.float32)
            stride_ddt_bias_batch = ddt_bias_workspace.stride(0)
            stride_ddt_bias_chunk = ddt_bias_workspace.stride(1)
        else:
            ddt_bias_workspace = ddt_bias = torch.empty_like(dt_bias, dtype=torch.float32)
            stride_ddt_bias_batch = 0
            stride_ddt_bias_chunk = 0
    else:
        ddt_bias = None
        ddt_bias_workspace = None
        stride_ddt_bias_batch = 0
        stride_ddt_bias_chunk = 0
    if ddt is not None:
        assert ddt.shape == dt.shape
    else:
        ddt = torch.empty_like(dt)
    dA = torch.empty_like(A, dtype=torch.float32)
    if deterministic:
        dA_workspace = torch.zeros(batch, nchunks, nheads, device=dt.device, dtype=torch.float32)
        stride_dA_batch = dA_workspace.stride(0)
        stride_dA_chunk = dA_workspace.stride(1)
    else:
        dA_workspace = dA
        stride_dA_batch = 0
        stride_dA_chunk = 0
    grid_chunk_cs = lambda META: (batch, nchunks, triton.cdiv(nheads, META['BLOCK_SIZE_H']))
    with torch.cuda.device(dt.device.index):
        _chunk_cumsum_bwd_kernel[grid_chunk_cs](
            cu_chunk_seqlens,
            ddA,
            ddt_out,
            dt,
            A,
            dt_bias,
            ddt,
            dA_workspace,
            ddt_bias_workspace if ddt_bias is not None else None,
            batch,
            seqlen,
            nheads,
            chunk_size,
            dt_limit[0],
            dt_limit[1],
            ddA.stride(0),
            ddA.stride(2),
            ddA.stride(1),
            ddA.stride(3),
            ddt_out.stride(0),
            ddt_out.stride(2),
            ddt_out.stride(1),
            ddt_out.stride(3),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            A.stride(0),
            dt_bias.stride(0) if dt_bias is not None else 0,
            ddt.stride(0),
            ddt.stride(1),
            ddt.stride(2),
            stride_dA_batch,
            stride_dA_chunk,
            dA_workspace.stride(-1),
            stride_ddt_bias_batch,
            stride_ddt_bias_chunk,
            (ddt_bias_workspace.stride(-1) if ddt_bias is not None else 0),
            dt_softplus,
            HAS_DT_BIAS=dt_bias is not None,
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
            DETERMINISTIC_REDUCTION=deterministic,
        )
    if deterministic:
        dA.copy_(dA_workspace.sum(dim=(0, 1)))
        if ddt_bias is not None:
            ddt_bias.copy_(ddt_bias_workspace.sum(dim=(0, 1)))
    return ddt, dA, ddt_bias
