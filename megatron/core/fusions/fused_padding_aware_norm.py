# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""RMSNorm over the live rows of a THD buffer (Triton, device-side row bound).

Rows at or beyond ``m_valid`` (an int64 ``(1,)`` device tensor read in-kernel, so the kernels are
CUDA-graph capturable) are written as zeros in the forward and get a zero input gradient in the
backward; the weight gradient sums over the live rows only (padding rows carry no loss gradient).
fp32 statistics and arithmetic, like Transformer Engine's RMSNorm; bf16/fp16/fp32 I/O.
"""

from __future__ import annotations

from typing import Optional

import torch

from megatron.core.fusions.padding_aware_switches import padding_aware_feature_disabled

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

MAX_COLS = 16384

if HAVE_TRITON:

    @triton.jit
    def _pa_rmsnorm_fwd_kernel(
        X,
        W,
        Y,
        RSTD,
        m_valid_ptr,
        N,
        n_rows,
        x_row_stride,
        y_row_stride,
        eps,
        HAS_WEIGHT: tl.constexpr,
        ZERO_CENTERED: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """ROWS rows per program as one [ROWS, BLOCK_N] tile:
        y = x * rsqrt(mean(x^2) + eps) * (w [+ 1]); rows at or beyond *m_valid_ptr -> zeros."""
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, BLOCK_N)
        row_ok = rows < n_rows
        live = rows < tl.load(m_valid_ptr)
        mask = row_ok[:, None] & (cols < N)[None, :]
        x = tl.load(
            X + rows.to(tl.int64)[:, None] * x_row_stride + cols[None, :], mask=mask, other=0.0
        ).to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(x * x, 1) / N + eps)
        y = x * rstd[:, None]
        if HAS_WEIGHT:
            w = tl.load(W + cols, mask=cols < N, other=0.0).to(tl.float32)
            if ZERO_CENTERED:
                w = w + 1.0
            y = y * w[None, :]
        y = tl.where(live[:, None], y, 0.0)
        tl.store(
            Y + rows.to(tl.int64)[:, None] * y_row_stride + cols[None, :],
            y.to(Y.dtype.element_ty),
            mask=mask,
        )
        tl.store(RSTD + rows, tl.where(live, rstd, 0.0), mask=row_ok)

    @triton.jit
    def _pa_rmsnorm_bwd_kernel(
        DY,
        X,
        W,
        RSTD,
        DX,
        DW_PART,
        m_valid_ptr,
        N,
        n_rows,
        dy_row_stride,
        x_row_stride,
        dx_row_stride,
        HAS_WEIGHT: tl.constexpr,
        ZERO_CENTERED: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """ROWS rows per program as one tile: dx = rstd * (dy*w - xhat * mean(dy*w*xhat)) for live
        rows (zeros for the tail); the program's partial dw = sum over its live rows of dy * xhat
        goes to DW_PART[program]."""
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, BLOCK_N)
        row_ok = rows < n_rows
        live = (rows < tl.load(m_valid_ptr)) & row_ok
        mask = live[:, None] & (cols < N)[None, :]
        x = tl.load(
            X + rows.to(tl.int64)[:, None] * x_row_stride + cols[None, :], mask=mask, other=0.0
        ).to(tl.float32)
        dy = tl.load(
            DY + rows.to(tl.int64)[:, None] * dy_row_stride + cols[None, :], mask=mask, other=0.0
        ).to(tl.float32)
        rstd = tl.load(RSTD + rows, mask=row_ok, other=0.0)
        xhat = x * rstd[:, None]
        if HAS_WEIGHT:
            w = tl.load(W + cols, mask=cols < N, other=0.0).to(tl.float32)
            if ZERO_CENTERED:
                w = w + 1.0
            dxhat = dy * w[None, :]
            tl.store(DW_PART + pid.to(tl.int64) * N + cols, tl.sum(dy * xhat, 0), mask=cols < N)
        else:
            dxhat = dy
        dx = rstd[:, None] * (dxhat - xhat * (tl.sum(dxhat * xhat, 1) / N)[:, None])
        # masked-off (tail / out of range) rows loaded zeros -> dx is zero there; store all rows
        tl.store(
            DX + rows.to(tl.int64)[:, None] * dx_row_stride + cols[None, :],
            dx.to(DX.dtype.element_ty),
            mask=row_ok[:, None] & (cols < N)[None, :],
        )


def _tile(n_cols: int):
    """(BLOCK_N, ROWS per program, warps): ~8K elements per program, 1-64 rows."""
    block = triton.next_power_of_2(n_cols)
    rows = max(1, min(64, 8192 // block))
    return block, rows, max(1, min(16, (block * rows) // 512))


class PaddingAwareRMSNorm(torch.autograd.Function):
    """RMSNorm of ``x2d [rows, N]`` over the first ``m_valid`` rows (weight optional)."""

    @staticmethod
    def forward(ctx, x2d, weight, eps, zero_centered, m_valid):
        """Normalise the live rows; zero the tail; keep rstd for the backward."""
        rows, n_cols = x2d.shape
        block, tile_rows, warps = _tile(n_cols)
        y = torch.empty_like(x2d)
        rstd = torch.empty(rows, dtype=torch.float32, device=x2d.device)
        has_weight = weight is not None
        _pa_rmsnorm_fwd_kernel[(triton.cdiv(rows, tile_rows),)](
            x2d,
            weight if has_weight else x2d,
            y,
            rstd,
            m_valid,
            n_cols,
            rows,
            x2d.stride(0),
            y.stride(0),
            eps,
            HAS_WEIGHT=has_weight,
            ZERO_CENTERED=bool(zero_centered) and has_weight,
            ROWS=tile_rows,
            BLOCK_N=block,
            num_warps=warps,
        )
        ctx.save_for_backward(x2d, weight, rstd, m_valid)
        ctx.zero_centered = bool(zero_centered) and has_weight
        return y

    @staticmethod
    def backward(ctx, dy):
        """Input gradient for the live rows (zero tail) and the weight gradient over them."""
        x2d, weight, rstd, m_valid = ctx.saved_tensors
        rows, n_cols = x2d.shape
        block, tile_rows, warps = _tile(n_cols)
        dy = dy.contiguous()
        dx = torch.empty_like(x2d)
        has_weight = weight is not None
        n_prog = triton.cdiv(rows, tile_rows)
        dw_part = (
            torch.empty((n_prog, n_cols), dtype=torch.float32, device=x2d.device)
            if has_weight
            else dx
        )
        _pa_rmsnorm_bwd_kernel[(n_prog,)](
            dy,
            x2d,
            weight if has_weight else x2d,
            rstd,
            dx,
            dw_part,
            m_valid,
            n_cols,
            rows,
            dy.stride(0),
            x2d.stride(0),
            dx.stride(0),
            HAS_WEIGHT=has_weight,
            ZERO_CENTERED=ctx.zero_centered,
            ROWS=tile_rows,
            BLOCK_N=block,
            num_warps=warps,
        )
        dw = dw_part.sum(dim=0).to(weight.dtype) if has_weight else None
        return dx, dw, None, None, None


def padding_aware_rms_norm_available(x: torch.Tensor) -> bool:
    """Triton, CUDA, unit last stride, <= MAX_COLS columns; env SKIP=norm disables."""
    return (
        HAVE_TRITON
        and x.is_cuda
        and x.stride(-1) == 1
        and x.shape[-1] <= MAX_COLS
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and not padding_aware_feature_disabled("norm")
    )


def padding_aware_rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    m_valid: torch.Tensor,
    zero_centered_gamma: bool = False,
) -> torch.Tensor:
    """RMSNorm over the last dim of ``x``; leading dims flatten to the bound's rows."""
    shape = x.shape
    x2d = x.reshape(-1, shape[-1])
    if x2d.stride(1) != 1:
        x2d = x2d.contiguous()
    y = PaddingAwareRMSNorm.apply(x2d, weight, eps, zero_centered_gamma, m_valid)
    return y.view(shape)
