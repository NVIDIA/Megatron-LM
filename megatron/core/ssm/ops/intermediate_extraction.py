# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused gather + conditional-scatter kernels for Mamba intermediate-state
extraction used by prefix caching.

These replace the two-step ``states[indices]`` (dense gather) + ``.copy_()``
(scratch write) pattern with a single kernel that:

1. Reads a runtime ``real_count`` from a fixed-address GPU tensor.
2. For each slot ``i < real_count``, gathers the source row indexed by the
   per-slot index/position and writes it directly into the destination scratch.
3. For each slot ``i >= real_count``, returns immediately (no work, no write).

This is CUDA-graph safe: the launch grid is sized at capture time to the maximum
possible slot count, but per-program execution is data-conditional on the
runtime ``real_count``, so padded slots cost almost nothing.
"""

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    from unittest.mock import MagicMock

    from megatron.core.utils import null_decorator

    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False


@triton.jit
def _scatter_intermediate_ssm_kernel(
    states_ptr,  # [num_chunks, state_flat]: source SSM states from the chunk scan
    chunk_indices_ptr,  # [max_count] int64: gather index per scratch slot
    real_count_ptr,  # int32[1]: number of meaningful slots this step
    out_ptr,  # [max_count, state_flat]: scratch destination (per layer)
    state_flat,  # tl.int32: product of (nheads, headdim, dstate)
    states_row_stride,  # tl.int64: stride between chunks in states_ptr
    out_row_stride,  # tl.int64: stride between slots in out_ptr
    BLOCK_N: tl.constexpr,  # columns per program
):
    """Conditional gather+scatter for SSM intermediate states.

    Grid: ``(max_count, ceil(state_flat / BLOCK_N))``. Each program owns one
    (slot, column-block) pair; programs with ``pid_slot >= real_count`` exit
    immediately, so padded slots produce no HBM traffic.
    """
    pid_slot = tl.program_id(0)
    pid_col = tl.program_id(1)

    real_count = tl.load(real_count_ptr).to(tl.int32)
    if pid_slot >= real_count:
        return

    chunk_idx = tl.load(chunk_indices_ptr + pid_slot).to(tl.int64)

    col_offset = pid_col * BLOCK_N
    cols = col_offset + tl.arange(0, BLOCK_N)
    mask = cols < state_flat

    src = states_ptr + chunk_idx * states_row_stride + cols.to(tl.int64)
    dst = out_ptr + pid_slot.to(tl.int64) * out_row_stride + cols.to(tl.int64)

    val = tl.load(src, mask=mask)
    tl.store(dst, val, mask=mask)


def scatter_intermediate_ssm(
    states: Tensor, chunk_indices: Tensor, real_count_gpu: Tensor, out: Tensor
) -> None:
    """Gather rows of ``states`` at ``chunk_indices`` and scatter into ``out``,
    for the first ``real_count_gpu`` slots only.

    Args:
        states: ``(num_chunks, *ssm_state_shape)`` chunk-scan output for one layer.
        chunk_indices: ``(max_count,)`` int64 per-slot gather index.
        real_count_gpu: ``int32[1]`` GPU tensor with the runtime real count.
        out: ``(max_count, *ssm_state_shape)`` destination scratch slice (one layer).
    """
    assert states.is_cuda and chunk_indices.is_cuda and real_count_gpu.is_cuda and out.is_cuda
    assert states.dim() >= 2, f"expected states to be at least 2D, got {states.shape}"
    assert (
        out.shape[1:] == states.shape[1:]
    ), f"per-slot shape mismatch: out {tuple(out.shape[1:])} vs states {tuple(states.shape[1:])}"
    assert chunk_indices.dtype == torch.int64, chunk_indices.dtype
    assert real_count_gpu.dtype == torch.int32 and real_count_gpu.numel() == 1

    # The grid follows the per-step view length (chunk_indices.numel()), bounded
    # above by out.shape[0] (the full scratch pool); programs past real_count
    # exit immediately inside the kernel.
    n_slots = int(chunk_indices.numel())
    if n_slots == 0:
        return
    if n_slots > out.shape[0]:
        raise ValueError(f"chunk_indices length {n_slots} exceeds scratch capacity {out.shape[0]}")

    state_flat = 1
    for d in states.shape[1:]:
        state_flat *= int(d)

    # Contiguity is required for the row-stride math; the engine pre-allocates
    # these contiguous, so assert to fail loudly in tests rather than corrupt.
    assert states.is_contiguous(), "states must be contiguous"
    assert out.is_contiguous(), "out must be contiguous"

    BLOCK_N = 1024
    grid = (n_slots, triton.cdiv(state_flat, BLOCK_N))
    _scatter_intermediate_ssm_kernel[grid](
        states,
        chunk_indices,
        real_count_gpu,
        out,
        state_flat=state_flat,
        states_row_stride=state_flat,
        out_row_stride=state_flat,
        BLOCK_N=BLOCK_N,
    )


@triton.jit
def _scatter_intermediate_conv_kernel(
    src_ptr,  # xBC_pre_conv batch-0 slice: addressed via strides
    abs_positions_ptr,  # [max_count] int32: extraction-window end position per slot
    real_count_ptr,  # int32[1]: meaningful slot count this step
    out_ptr,  # [max_count, conv_dim, d_conv]: scratch destination
    seq_len,  # tl.int32: bound for clamping
    conv_dim,  # tl.int32: feature dim
    src_stride_s,  # tl.int64: stride along seq_len
    src_stride_c,  # tl.int64: stride along conv_dim
    out_slot_stride,  # tl.int64: stride between slots in out_ptr (conv_dim * D_CONV)
    D_CONV: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Conditional gather of a length-``D_CONV`` conv window per slot.

    Reads window ``[abs_pos - D_CONV, abs_pos)`` from ``src_ptr`` (clamped into
    ``[0, seq_len)``) and writes it transposed into ``out[slot, :, :]`` of shape
    ``(conv_dim, D_CONV)``. The transpose is folded into the write pattern to
    match the slot allocator's storage layout.
    """
    pid_slot = tl.program_id(0)
    pid_c = tl.program_id(1)

    real_count = tl.load(real_count_ptr).to(tl.int32)
    if pid_slot >= real_count:
        return

    abs_pos = tl.load(abs_positions_ptr + pid_slot).to(tl.int32)

    c_offset = pid_c * BLOCK_C
    c_idxs = c_offset + tl.arange(0, BLOCK_C)
    c_mask = c_idxs < conv_dim

    # out[slot, c, j]: c outer, j inner (contiguous), so for fixed c the D_CONV
    # values are stride-1.
    slot_base = pid_slot.to(tl.int64) * out_slot_stride

    for j in tl.static_range(D_CONV):
        p_raw = abs_pos - D_CONV + j
        # Clamp into [0, seq_len - 1]. A no-op for real slots; defensive for any
        # future caller that doesn't filter padded slots via real_count.
        p = tl.maximum(0, tl.minimum(p_raw, seq_len - 1))

        src = src_ptr + p.to(tl.int64) * src_stride_s + c_idxs.to(tl.int64) * src_stride_c
        dst = out_ptr + slot_base + c_idxs.to(tl.int64) * D_CONV + j

        val = tl.load(src, mask=c_mask)
        tl.store(dst, val, mask=c_mask)


def scatter_intermediate_conv(
    src: Tensor, abs_positions: Tensor, real_count_gpu: Tensor, out: Tensor, d_conv: int
) -> None:
    """Gather length-``d_conv`` conv windows from ``src`` at ``abs_positions`` and
    scatter (transposed) into ``out``, for the first ``real_count_gpu`` slots only.

    Args:
        src: ``(batch, seq_len, conv_dim)`` pre-conv xBC tensor. Batch is assumed
            to be 1 (inference); only batch index 0 is read.
        abs_positions: ``(max_count,)`` int32 extraction-window end position per
            slot (window is ``[pos - d_conv, pos)``).
        real_count_gpu: ``int32[1]`` GPU tensor with the runtime real count.
        out: ``(max_count, conv_dim, d_conv)`` destination scratch slice (one layer).
        d_conv: conv window length (constexpr in the kernel).
    """
    assert src.is_cuda and abs_positions.is_cuda and real_count_gpu.is_cuda and out.is_cuda
    assert src.dim() == 3, f"expected src (batch, seq_len, conv_dim), got {src.shape}"
    assert src.shape[0] == 1, f"batch must be 1 for inference, got {src.shape[0]}"
    assert abs_positions.dtype == torch.int32, abs_positions.dtype
    assert real_count_gpu.dtype == torch.int32 and real_count_gpu.numel() == 1
    assert (
        out.dim() == 3 and out.shape[2] == d_conv
    ), f"out shape {tuple(out.shape)} does not match (max_count, conv_dim, {d_conv})"

    _, conv_dim, _ = out.shape
    n_slots = int(abs_positions.numel())
    if n_slots == 0:
        return
    if n_slots > out.shape[0]:
        raise ValueError(f"abs_positions length {n_slots} exceeds scratch capacity {out.shape[0]}")

    seq_len = int(src.shape[1])
    src_stride_s = int(src.stride(1))
    src_stride_c = int(src.stride(2))

    BLOCK_C = 128
    grid = (n_slots, triton.cdiv(conv_dim, BLOCK_C))
    _scatter_intermediate_conv_kernel[grid](
        src,
        abs_positions,
        real_count_gpu,
        out,
        seq_len=seq_len,
        conv_dim=conv_dim,
        src_stride_s=src_stride_s,
        src_stride_c=src_stride_c,
        out_slot_stride=conv_dim * d_conv,
        D_CONV=d_conv,
        BLOCK_C=BLOCK_C,
    )
