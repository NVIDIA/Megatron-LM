# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Row-bounded elementwise helpers for THD buffers (Triton, device-side row bound)."""

from __future__ import annotations

import torch

from megatron.core.fusions.padding_aware_switches import padding_aware_feature_disabled

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

if HAVE_TRITON:

    @triton.jit
    def _pa_add3_kernel(A, B, C, OUT, m_valid_ptr, N, row_stride, BLOCK_N: tl.constexpr):
        """out[row] = a + b + c for live rows (fp32 accumulation), zeros for the tail."""
        row = tl.program_id(0)
        col_block = tl.program_id(1)
        cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < N
        off = row.to(tl.int64) * row_stride + cols
        if row >= tl.load(m_valid_ptr):
            tl.store(
                OUT + off, tl.zeros([BLOCK_N], dtype=tl.float32).to(OUT.dtype.element_ty), mask=mask
            )
            return
        a = tl.load(A + off, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + off, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(C + off, mask=mask, other=0.0).to(tl.float32)
        tl.store(OUT + off, (a + b + c).to(OUT.dtype.element_ty), mask=mask)


def padding_aware_add3_available(*tensors: torch.Tensor) -> bool:
    """Triton, CUDA, same shape/dtype, row-major, kill-switch ``MCORE_PADDING_AWARE_SKIP=add``."""
    if not HAVE_TRITON or padding_aware_feature_disabled("add"):
        return False
    t0 = tensors[0]
    return all(
        t.is_cuda and t.shape == t0.shape and t.dtype == t0.dtype and t.is_contiguous()
        for t in tensors
    )


def padding_aware_add3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, m_valid: torch.Tensor):
    """``a + b + c`` over the live rows of ``[rows, ...]`` tensors; the padding tail is zero."""
    rows = a.shape[0]
    n_cols = a.numel() // rows
    out = torch.empty_like(a)
    block = 4096
    _pa_add3_kernel[(rows, triton.cdiv(n_cols, block))](
        a, b, c, out, m_valid, n_cols, n_cols, BLOCK_N=block, num_warps=8
    )
    return out


if HAVE_TRITON:

    @triton.jit
    def _pa_row_argmax_kernel(X, OUT, m_valid_ptr, V, x_row_stride, BLOCK: tl.constexpr):
        """OUT[row] = argmax over the row's V columns for live rows; -1 for the padding tail."""
        row = tl.program_id(0)
        if row >= tl.load(m_valid_ptr):
            tl.store(OUT + row, -1)
            return
        x_ptr = X + row.to(tl.int64) * x_row_stride
        best = float("-inf")
        best_idx = 0
        for off in range(0, V, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            x = tl.load(x_ptr + cols, mask=cols < V, other=float("-inf")).to(tl.float32)
            block_max = tl.max(x, 0)
            # first index attaining the block max (ties -> lowest index, like torch.argmax)
            block_idx = tl.min(tl.where(x == block_max, cols, V), 0)
            take = block_max > best
            best_idx = tl.where(take, block_idx, best_idx)
            best = tl.maximum(best, block_max)
        tl.store(OUT + row, best_idx.to(tl.int64))


def padding_aware_row_argmax(logits: torch.Tensor, m_valid: torch.Tensor) -> torch.Tensor:
    """``torch.argmax(logits, dim=-1)`` over the live rows of ``[..., V]`` logits; -1 for the tail
    (a value no label equals, so acceptance counts over masked rows stay unchanged)."""
    shape = logits.shape[:-1]
    x2d = logits.reshape(-1, logits.shape[-1])
    if x2d.stride(1) != 1:
        x2d = x2d.contiguous()
    out = torch.empty(x2d.shape[0], dtype=torch.int64, device=logits.device)
    block = min(16384, triton.next_power_of_2(x2d.shape[1]))
    _pa_row_argmax_kernel[(x2d.shape[0],)](
        x2d,
        out,
        m_valid,
        x2d.shape[1],
        x2d.stride(0),
        BLOCK=block,
        num_warps=max(4, min(16, block // 1024)),
    )
    return out.view(shape)
