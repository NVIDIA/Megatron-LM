# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Row kernels for batch-invariant Mamba decode.

The buffered decode replays the training chunk-scan kernels for the one row a step
consumes. Those kernels still compute a whole training tile per (entry, head): the
chunk scan a BLOCK_SIZE_M-row M tile, CB a full M tile, and chunk state plus state
passing launch programs for every entry although only entries that fill their chunk
(about one in chunk_size) commit a boundary state.

The kernels here compute only what the step consumes, with the training kernels'
statements and tile decomposition so each stored value is bitwise the training one:

* ``_chunk_scan_row_kernel``: the chunk scan on the 16-row MMA tile holding the target
  row, with the training tile's BLOCK_SIZE_N / BLOCK_SIZE_K and causal bound K_MAX. A
  tensor-core MMA computes each output row from that row of A alone, so with the same
  instruction (``mma.sync``), k decomposition and accumulation order, the target row's
  bits do not depend on how many other rows share the tile.
* ``_bmm_row_kernel``: CB with the training kernel's config, loading and storing only
  the target row.
* ``_chunk_state_passing_rows_kernel``: chunk state (training config) and the state
  passing update ``exp(dA) * S + new`` fused, run only for entries that cross a chunk
  boundary. The update is one fma, as the state-passing kernel compiles it.

Each kernel takes its tile from the config the training autotuner launches with
(``configs`` holds exactly one entry once batch-invariant mode pins it). When a kernel
is not pinned to one config, or the chunk-scan tile does not use ``mma.sync``, the
caller keeps the training kernels.
"""

import torch
import triton
import triton.language as tl

from megatron.core.ssm.ops.mamba2.ssd_bmm import _bmm_chunk_fwd_kernel
from megatron.core.ssm.ops.mamba2.ssd_chunk_scan import _chunk_scan_fwd_kernel
from megatron.core.ssm.ops.mamba2.ssd_chunk_state import _chunk_state_fwd_kernel

# One MMA M tile: the row scan's M extent.
ROW_BLOCK_SIZE_M = 16


def _pinned_config(kernel):
    """The one config ``kernel``'s autotuner launches with, or None while it still chooses."""
    configs = getattr(kernel, "configs", ())
    return configs[0] if len(configs) == 1 else None


def row_kernel_configs(C: torch.Tensor, chunk_size: int):
    """Training configs (chunk scan, CB, chunk state) the row kernels mirror, or None.

    None keeps the training kernels: an autotuner not pinned to one config, a dstate the
    chunk scan splits over several dots, or a chunk-scan tile large enough for
    warpgroup / tcgen05 MMAs (``num_warps >= 4`` and ``BLOCK_SIZE_M >= 64``), whose
    instruction the 16-row tile cannot reproduce (on GB200 every such candidate differs
    from the row kernel in fp32; every ``mma.sync`` candidate matches it bitwise).
    """
    scan, bmm, state = (
        _pinned_config(k)
        for k in (_chunk_scan_fwd_kernel, _bmm_chunk_fwd_kernel, _chunk_state_fwd_kernel)
    )
    if scan is None or bmm is None or state is None:
        return None
    scan_m = scan.kwargs["BLOCK_SIZE_M"]
    if scan.num_warps >= 4 and scan_m >= 64:
        return None
    if triton.next_power_of_2(C.shape[-1]) > 128 or chunk_size % scan_m != 0:
        return None
    return scan, bmm, state


@triton.jit
def _chunk_scan_row_kernel(
    cb_ptr,
    x_ptr,
    z_ptr,
    out_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    C_ptr,
    D_ptr,
    initstates_ptr,
    chunk_starts_ptr,
    slots_ptr,
    target_rows_ptr,
    chunk_size: tl.constexpr,
    hdim: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    stride_cb_chunk: tl.int64,
    stride_cb_head: tl.int64,
    stride_cb_csize_m: tl.int64,
    stride_cb_csize_k: tl.constexpr,
    stride_x_seqlen: tl.int64,
    stride_x_head: tl.int64,
    stride_x_hdim: tl.constexpr,
    stride_z_seqlen: tl.int64,
    stride_z_head: tl.int64,
    stride_z_hdim: tl.constexpr,
    stride_out_seqlen: tl.int64,
    stride_out_head: tl.int64,
    stride_out_hdim: tl.constexpr,
    stride_dt_chunk: tl.int64,
    stride_dt_head: tl.int64,
    stride_dt_csize: tl.constexpr,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    stride_C_seqlen: tl.int64,
    stride_C_head: tl.int64,
    stride_C_dstate: tl.constexpr,
    stride_init_states_batch: tl.int64,
    stride_init_states_head: tl.int64,
    stride_init_states_hdim: tl.int64,
    stride_init_states_dstate: tl.constexpr,
    stride_D_head: tl.constexpr,
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_Z: tl.constexpr,
    TILE_SIZE_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    """``_chunk_scan_fwd_kernel`` in target-rows mode for the target row only.

    The training kernel's statements on a BLOCK_SIZE_M-row tile placed over the target
    row, with K_MAX from the training tile (TILE_SIZE_M rows) so the same zero-weight
    products are summed. Only the target row of C and cb is loaded; the other rows are
    zero and never reach the stored row.
    """
    pid_c = tl.program_id(axis=0).to(tl.int64)
    pid_h = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    tr = tl.load(target_rows_ptr + pid_c)
    if tr < 0:
        return
    cb_ptr += pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    chunk_seqlen_start = tl.load(chunk_starts_ptr + pid_c)
    x_ptr += chunk_seqlen_start * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    C_ptr += chunk_seqlen_start * stride_C_seqlen + (pid_h // nheads_ngroups_ratio) * stride_C_head

    offs_m = (tr // BLOCK_SIZE_M) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    is_target = offs_m == tr
    seq_idx = tl.load(slots_ptr + pid_c)
    prev_states_ptr = (
        initstates_ptr + seq_idx * stride_init_states_batch + pid_h * stride_init_states_head
    )
    chunk_size_limit = chunk_size

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dA_cs_m = tl.load(
        dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0
    ).to(tl.float32)

    offs_k_dstate = tl.arange(0, BLOCK_SIZE_DSTATE)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate)
    scale_m = tl.exp(dA_cs_m)
    C = tl.load(
        C_ptrs,
        mask=is_target[:, None]
        & (offs_m[:, None] < chunk_size_limit)
        & (offs_k_dstate[None, :] < dstate),
        other=0.0,
    )
    prev_states_ptrs = (
        prev_states_ptr
        + offs_n[None, :] * stride_init_states_hdim
        + offs_k_dstate[:, None] * stride_init_states_dstate
    )
    prev_states = tl.load(
        prev_states_ptrs,
        mask=(offs_k_dstate[:, None] < dstate) & (offs_n[None, :] < hdim),
        other=0.0,
    )
    prev_states = prev_states.to(C_ptr.dtype.element_ty)
    acc = tl.dot(C, prev_states) * scale_m[:, None]

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
    x_ptrs = x_ptr + (offs_k[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = min((tr // TILE_SIZE_M + 1) * TILE_SIZE_M, chunk_size_limit)
    for k in range(0, K_MAX, BLOCK_SIZE_K):
        cb = tl.load(
            cb_ptrs,
            mask=is_target[:, None]
            & (offs_m[:, None] < chunk_size)
            & (offs_k[None, :] < chunk_size - k),
            other=0.0,
        ).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        cb *= tl.exp(tl.minimum(dA_cs_m[:, None] - dA_cs_k[None, :], 0.0))
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        cb *= dt_k
        mask = offs_m[:, None] >= k + offs_k[None, :]
        cb = tl.where(mask, cb, 0.0)
        cb = cb.to(x_ptr.dtype.element_ty)
        x = tl.load(
            x_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < hdim),
            other=0.0,
        )
        acc += tl.dot(cb, x)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    if HAS_D:
        if D_HAS_HDIM:
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(
                tl.float32
            )
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        x_residual = tl.load(
            x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim),
            mask=is_target[:, None]
            & (offs_m[:, None] < chunk_size_limit)
            & (offs_n[None, :] < hdim),
            other=0.0,
        ).to(tl.float32)
        if x_ptr.dtype.element_ty == tl.float32:
            acc += x_residual * D
        else:
            # The training kernel converts 16-bit x to the MMA layout and its
            # ``acc += x_residual * D`` compiles to fma.rn.f32. Whether this kernel's
            # mul lands in the same layout as the add (and so fuses) is up to Triton's
            # layout pass; with Triton 3.8 and per-hdim D it does not, so the fused
            # form is written out.
            acc = tl.fma(x_residual, D, acc)

    if HAS_Z:
        # As the training kernel's target-rows mode: z is the entry's one row.
        z_ptr += pid_c * stride_z_seqlen + pid_h * stride_z_head
        z = tl.load(
            z_ptr + (offs_m[:, None] * 0 + stride_z_hdim * offs_n[None, :]),
            mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim),
            other=0.0,
        ).to(tl.float32)
        acc *= z * tl.sigmoid(z)

    out_ptr += pid_c * stride_out_seqlen + pid_h * stride_out_head
    out_ptrs = out_ptr + (offs_m[:, None] * 0 + offs_n[None, :] * stride_out_hdim)
    tl.store(out_ptrs, acc, mask=is_target[:, None] & (offs_n[None, :] < hdim))


@triton.jit
def _bmm_row_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    chunk_starts_ptr,
    target_rows_ptr,
    chunk_size: tl.constexpr,
    K: tl.constexpr,
    ngroups: tl.constexpr,
    stride_a_seqlen: tl.int64,
    stride_a_head: tl.int64,
    stride_ak: tl.constexpr,
    stride_b_seqlen: tl.int64,
    stride_b_head: tl.int64,
    stride_bk: tl.constexpr,
    stride_out_chunk: tl.int64,
    stride_out_head: tl.int64,
    stride_outm: tl.int64,
    stride_outn: tl.constexpr,
    dot_dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """``_bmm_chunk_fwd_kernel`` in target-rows mode, loading and storing only the target row."""
    pid_ch = tl.program_id(axis=1).to(tl.int64)
    pid_c = pid_ch // ngroups
    pid_h = pid_ch - pid_c * ngroups
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    tr = tl.load(target_rows_ptr + pid_c)
    if tr < 0:
        return
    if pid_m != tr // BLOCK_SIZE_M:
        return
    if pid_n * BLOCK_SIZE_N > tr:
        return

    chunk_seqlen_start = tl.load(chunk_starts_ptr + pid_c)
    a_ptr += chunk_seqlen_start * stride_a_seqlen + pid_h * stride_a_head
    b_ptr += chunk_seqlen_start * stride_b_seqlen + pid_h * stride_b_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_a_seqlen + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_b_seqlen)
    chunk_size_limit = chunk_size
    is_target = offs_m == tr

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=is_target[:, None]
            & (offs_m[:, None] < chunk_size_limit)
            & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        ).to(dot_dtype)
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < chunk_size_limit),
            other=0.0,
        ).to(dot_dtype)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    out = acc.to(out_ptr.dtype.element_ty)
    out_ptr += pid_c * stride_out_chunk + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_outm * offs_m[:, None] + offs_n[None, :] * stride_outn)
    tl.store(out_ptrs, out, mask=is_target[:, None] & (offs_n[None, :] < chunk_size))


@triton.jit
def _chunk_state_passing_rows_kernel(
    x_ptr,
    b_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    chunk_starts_ptr,
    chunk_flags_ptr,
    slots_ptr,
    states_ptr,
    nchunks,
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
    stride_states_batch: tl.int64,
    stride_states_head: tl.int64,
    stride_states_hdim: tl.int64,
    stride_states_dstate: tl.constexpr,
    CHUNKS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """``_chunk_state_fwd_kernel`` then ``_state_passing_fwd_kernel`` for crossing entries.

    For each entry with a chunk flag, the chunk-state tile (training statements and tile),
    then the state-passing update of the slot's state on the same fp32 values. A program
    owns CHUNKS_PER_PROGRAM entries of one (head, tile) and skips entries without a flag.
    """
    pid_t = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_r = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = pid_t // num_pid_n
    pid_n = pid_t % num_pid_n
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    for i in range(CHUNKS_PER_PROGRAM):
        c = (pid_r * CHUNKS_PER_PROGRAM + i).to(tl.int64)
        if tl.load(chunk_flags_ptr + c, mask=c < nchunks, other=0) != 0:
            chunk_seqlen_start = tl.load(chunk_starts_ptr + c)
            b_c = (
                b_ptr
                + chunk_seqlen_start * stride_b_seqlen
                + (pid_h // nheads_ngroups_ratio) * stride_b_head
            )
            x_c = x_ptr + chunk_seqlen_start * stride_x_seqlen + pid_h * stride_x_head
            dt_c = dt_ptr + c * stride_dt_chunk + pid_h * stride_dt_head
            dA_c = dA_cumsum_ptr + c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
            x_ptrs = x_c + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen)
            b_ptrs = b_c + (offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen)
            dt_ptrs = dt_c + offs_k * stride_dt_csize
            dA_cs_last = tl.load(dA_c + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
            dA_cumsum_ptrs = dA_c + offs_k * stride_dA_cs_csize
            chunk_size_limit = chunk_size

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
                dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(
                    tl.float32
                )
                scale = tl.exp(tl.minimum(dA_cs_last - dA_cs_k, 0.0)) * dt_k
                b *= scale[:, None]
                b = b.to(x_ptr.dtype.element_ty)
                acc += tl.dot(x, b)
                x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
                b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
                dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
                dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
            new_states = acc.to(tl.float32)

            # State passing: a crossing entry starts from its slot's state.
            slot = tl.load(slots_ptr + c).to(tl.int64)
            states_ptrs = (
                states_ptr
                + slot * stride_states_batch
                + pid_h * stride_states_head
                + offs_m[:, None] * stride_states_hdim
                + offs_n[None, :] * stride_states_dstate
            )
            states_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
            states = tl.load(states_ptrs, mask=states_mask, other=0.0).to(tl.float32)
            # The state-passing kernel's ``exp(dA_cs) * states + new_states`` compiles to one
            # fma; written out so this kernel rounds once as well.
            states = tl.fma(tl.exp(dA_cs_last), states, new_states)
            tl.store(states_ptrs, states, mask=states_mask)


def _launch_options(config):
    return {"num_warps": config.num_warps, "num_stages": config.num_stages}


def _block_sizes(config):
    return {key: config.kwargs[key] for key in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K")}


def chunk_scan_rows(
    cb, x, z, dt, dA_cumsum, C, out, slots, target_rows, chunk_starts, initial_states, D, config
):
    """``_chunk_scan_fwd(..., target_rows=, chunk_starts=)`` for the target rows only."""
    _, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = C.shape
    if z is not None:
        assert z.shape == (nchunks, nheads, headdim)
    block_size_n = config.kwargs["BLOCK_SIZE_N"]
    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)
    _chunk_scan_row_kernel[(nchunks, nheads, triton.cdiv(headdim, block_size_n))](
        cb,
        x,
        z,
        out,
        dt,
        dA_cumsum,
        C,
        D,
        initial_states,
        chunk_starts,
        slots,
        target_rows,
        chunk_size=chunk_size,
        hdim=headdim,
        dstate=dstate,
        nheads_ngroups_ratio=nheads // ngroups,
        stride_cb_chunk=cb.stride(0),
        stride_cb_head=cb.stride(1),
        stride_cb_csize_m=cb.stride(2),
        stride_cb_csize_k=cb.stride(3),
        stride_x_seqlen=x.stride(0),
        stride_x_head=x.stride(1),
        stride_x_hdim=x.stride(2),
        stride_z_seqlen=z_strides[0],
        stride_z_head=z_strides[1],
        stride_z_hdim=z_strides[2],
        stride_out_seqlen=out.stride(0),
        stride_out_head=out.stride(1),
        stride_out_hdim=out.stride(2),
        stride_dt_chunk=dt.stride(1),
        stride_dt_head=dt.stride(0),
        stride_dt_csize=dt.stride(2),
        stride_dA_cs_chunk=dA_cumsum.stride(1),
        stride_dA_cs_head=dA_cumsum.stride(0),
        stride_dA_cs_csize=dA_cumsum.stride(2),
        stride_C_seqlen=C.stride(0),
        stride_C_head=C.stride(1),
        stride_C_dstate=C.stride(2),
        stride_init_states_batch=initial_states.stride(0),
        stride_init_states_head=initial_states.stride(1),
        stride_init_states_hdim=initial_states.stride(2),
        stride_init_states_dstate=initial_states.stride(3),
        stride_D_head=D.stride(0) if D is not None else 0,
        HAS_D=D is not None,
        D_HAS_HDIM=D.dim() == 2 if D is not None else True,
        HAS_Z=z is not None,
        TILE_SIZE_M=config.kwargs["BLOCK_SIZE_M"],
        BLOCK_SIZE_M=ROW_BLOCK_SIZE_M,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=config.kwargs["BLOCK_SIZE_K"],
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        # Launch options only; the per-row instruction chain is set by the block sizes.
        num_warps=4,
        num_stages=3,
    )


def bmm_rows(C, B, chunk_size, target_rows, chunk_starts, config):
    """``_bmm_chunk_fwd(C, B, ..., output_dtype=fp32, target_rows=, chunk_starts=)``.

    Only each entry's target row of the result is written.
    """
    _, ngroups, k = C.shape
    assert B.shape == C.shape and C.stride(-1) == 1 and B.stride(-1) == 1
    nchunks = chunk_starts.shape[0]
    out = torch.empty(
        (nchunks, ngroups, chunk_size, chunk_size), device=C.device, dtype=torch.float32
    )
    dot_dtype = (
        tl.bfloat16
        if C.dtype == torch.bfloat16 or B.dtype == torch.bfloat16
        else (tl.float16 if C.dtype == torch.float16 or B.dtype == torch.float16 else tl.float32)
    )
    blocks = _block_sizes(config)
    grid = (
        triton.cdiv(chunk_size, blocks["BLOCK_SIZE_M"])
        * triton.cdiv(chunk_size, blocks["BLOCK_SIZE_N"]),
        nchunks * ngroups,
    )
    _bmm_row_kernel[grid](
        C,
        B,
        out,
        chunk_starts,
        target_rows,
        chunk_size=chunk_size,
        K=k,
        ngroups=ngroups,
        stride_a_seqlen=C.stride(0),
        stride_a_head=C.stride(1),
        stride_ak=C.stride(2),
        stride_b_seqlen=B.stride(0),
        stride_b_head=B.stride(1),
        stride_bk=B.stride(2),
        stride_out_chunk=out.stride(0),
        stride_out_head=out.stride(1),
        stride_outm=out.stride(-2),
        stride_outn=out.stride(-1),
        dot_dtype=dot_dtype,
        **blocks,
        **_launch_options(config),
    )
    return out


def _chunks_per_program(nchunks: int) -> int:
    """About one crossing per program: 1 below 32 entries, 4 at 64, 16 from 256."""
    if nchunks < 32:
        return 1
    return min(16, triton.next_power_of_2(nchunks // 16 + 1) // 2)


def chunk_state_passing_rows(x, B, dt, dA_cumsum, chunk_starts, chunk_flags, slots, states, config):
    """Commit the boundary state of every flagged entry into ``states[slot]``.

    ``_chunk_state_fwd(..., chunk_flags=)`` followed by ``_state_passing_fwd(...,
    dst_states=states, dst_indices=slots, dst_flags=chunk_flags)``.
    """
    _, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    assert states.shape[1:] == (nheads, headdim, dstate) and states.dtype == torch.float32
    blocks = _block_sizes(config)
    chunks_per_program = _chunks_per_program(nchunks)
    grid = (
        triton.cdiv(headdim, blocks["BLOCK_SIZE_M"]) * triton.cdiv(dstate, blocks["BLOCK_SIZE_N"]),
        nheads,
        triton.cdiv(nchunks, chunks_per_program),
    )
    _chunk_state_passing_rows_kernel[grid](
        x,
        B,
        dt,
        dA_cumsum,
        chunk_starts,
        chunk_flags,
        slots,
        states,
        nchunks,
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
        stride_states_batch=states.stride(0),
        stride_states_head=states.stride(1),
        stride_states_hdim=states.stride(2),
        stride_states_dstate=states.stride(3),
        CHUNKS_PER_PROGRAM=chunks_per_program,
        **blocks,
        **_launch_options(config),
    )
