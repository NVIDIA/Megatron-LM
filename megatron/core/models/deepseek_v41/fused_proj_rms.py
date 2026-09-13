# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused mHC mixing projection plus the V4.1 RMS factor (Triton).

The eager path (``HyperConnectionModule._projection_and_get_norm`` with
``v41_projection_and_rms``) upcasts the flattened streams ``x`` [M, K] to fp32, runs a tf32
matmul against the [N, K] mapping weight, and computes ``rsqrt(mean(x^2) + eps)`` with four
more full-width fp32 passes; the backward mirrors that. At 128K tokens x 20480 that is about
1.3 GB of fp32 activations written and re-read several times per sub-layer. The upstream fused
projection kernel is not reusable here: it implements the V4 factor ``norm / sqrt(K)`` and
fuses ``compute_h`` with it. This module is an independent implementation for the V4.1 form
that reads the bf16 streams once per pass and leaves ``_compute_h`` (tiny [M, N] tensors)
unchanged:

* forward:  ``proj = x @ W^T`` (fp32 accumulation, fp32-accurate 3xTF32 dot: about 21 mantissa
  bits, not bitwise IEEE; the omitted lo*lo term is below 2^-22 relative) and ``sumsq = sum(x^2)``
  in one sweep over K; ``r = rsqrt(sumsq / K + eps)``.
* backward: ``grad_x = grad_proj @ W + c * x`` with ``c = -grad_r * r^3 / K`` (one read of
  ``x``, one write in the stream dtype) and ``grad_W = grad_proj^T @ x`` (one read of ``x``,
  fixed reduction order over M, so the result is deterministic).

``N = n^2 + 2n`` is 24 for four streams; ``tl.dot`` needs a multiple of 16, so the N axis is
padded to 32 with masked loads and stores.
"""

from typing import Tuple

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - Triton is present in the training containers
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def fused_v41_proj_rms_available() -> bool:
    """Triton and a CUDA device are present."""
    return _TRITON_AVAILABLE and torch.cuda.is_available()


if _TRITON_AVAILABLE:

    @triton.jit
    def _split_tf32(a):
        """fp32 -> (tf32-exact high part, residual). Truncates the 13 low mantissa bits."""
        hi = (a.to(tl.int32, bitcast=True) & -8192).to(tl.float32, bitcast=True)
        return hi, a - hi

    @triton.jit
    def _dot_fp32(a, b, acc, A_EXACT: tl.constexpr, B_EXACT: tl.constexpr):
        """fp32-accurate a @ b via tensor-core tf32 dots (3xTF32 scheme).

        ``input_precision="ieee"`` is not honoured by every Triton/GPU combination (GB300 runs
        showed tf32-sized errors), so the operands are split into a tf32-exact high part and a
        residual; ``hi*hi + hi*lo + lo*hi`` recovers about 21 mantissa bits. Operands that are
        already tf32-exact (bf16 or fp16 inputs upcast to fp32) skip their residual terms.
        """
        a_hi, a_lo = _split_tf32(a)
        b_hi, b_lo = _split_tf32(b)
        acc = tl.dot(a_hi, b_hi, acc, input_precision="tf32")
        if not B_EXACT:
            acc = tl.dot(a_hi, b_lo, acc, input_precision="tf32")
        if not A_EXACT:
            acc = tl.dot(a_lo, b_hi, acc, input_precision="tf32")
        return acc

    @triton.jit
    def _v41_proj_rms_fwd_kernel(
        x_ptr,
        w_ptr,
        proj_ptr,
        r_ptr,
        M,
        K,
        N,
        eps,
        stride_xm,
        stride_wn,
        stride_pm,
        BM: tl.constexpr,
        BK: tl.constexpr,
        BN: tl.constexpr,
        X_EXACT: tl.constexpr,
    ):
        """One program per BM rows: proj[BM, N] = x @ W^T and r[BM] = rsqrt(mean(x^2) + eps)."""
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        row_ok = rows < M
        cols = tl.arange(0, BN)
        col_ok = cols < N
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        sumsq = tl.zeros((BM,), dtype=tl.float32)
        for k0 in range(0, K, BK):
            ks = k0 + tl.arange(0, BK)
            k_ok = ks < K
            x = tl.load(
                x_ptr + rows[:, None] * stride_xm + ks[None, :],
                mask=row_ok[:, None] & k_ok[None, :],
                other=0.0,
            ).to(tl.float32)
            w = tl.load(
                w_ptr + cols[None, :] * stride_wn + ks[:, None],
                mask=col_ok[None, :] & k_ok[:, None],
                other=0.0,
            ).to(tl.float32)
            acc = _dot_fp32(x, w, acc, X_EXACT, False)
            sumsq += tl.sum(x * x, axis=1)
        r = tl.rsqrt(sumsq / K.to(tl.float32) + eps)
        tl.store(
            proj_ptr + rows[:, None] * stride_pm + cols[None, :],
            acc,
            mask=row_ok[:, None] & col_ok[None, :],
        )
        tl.store(r_ptr + rows, r, mask=row_ok)

    @triton.jit
    def _v41_proj_rms_bwd_dx_kernel(
        x_ptr,
        w_ptr,
        gp_ptr,
        c_ptr,
        gx_ptr,
        M,
        K,
        N,
        stride_xm,
        stride_wn,
        stride_gm,
        stride_gxm,
        BM: tl.constexpr,
        BK: tl.constexpr,
        BN: tl.constexpr,
    ):
        """grad_x[BM, BK] = grad_proj[BM, N] @ W[N, BK] + c[BM] * x[BM, BK] (both fp32 operands)."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        rows = pid_m * BM + tl.arange(0, BM)
        row_ok = rows < M
        ks = pid_k * BK + tl.arange(0, BK)
        k_ok = ks < K
        cols = tl.arange(0, BN)
        col_ok = cols < N
        gp = tl.load(
            gp_ptr + rows[:, None] * stride_gm + cols[None, :],
            mask=row_ok[:, None] & col_ok[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            w_ptr + cols[:, None] * stride_wn + ks[None, :],
            mask=col_ok[:, None] & k_ok[None, :],
            other=0.0,
        ).to(tl.float32)
        x = tl.load(
            x_ptr + rows[:, None] * stride_xm + ks[None, :],
            mask=row_ok[:, None] & k_ok[None, :],
            other=0.0,
        ).to(tl.float32)
        c = tl.load(c_ptr + rows, mask=row_ok, other=0.0)
        acc = tl.zeros((BM, BK), dtype=tl.float32)
        gx = _dot_fp32(gp, w, acc, False, False) + c[:, None] * x
        tl.store(
            gx_ptr + rows[:, None] * stride_gxm + ks[None, :],
            gx.to(gx_ptr.dtype.element_ty),
            mask=row_ok[:, None] & k_ok[None, :],
        )

    @triton.jit
    def _v41_proj_rms_bwd_dw_kernel(
        x_ptr,
        gp_ptr,
        gw_ptr,
        M,
        K,
        N,
        stride_xm,
        stride_gm,
        stride_gwn,
        BM: tl.constexpr,
        BK: tl.constexpr,
        BN: tl.constexpr,
        X_EXACT: tl.constexpr,
    ):
        """grad_W[N, BK] = grad_proj^T @ x over all M rows, one program per K slice."""
        pid = tl.program_id(0)
        ks = pid * BK + tl.arange(0, BK)
        k_ok = ks < K
        cols = tl.arange(0, BN)
        col_ok = cols < N
        acc = tl.zeros((BN, BK), dtype=tl.float32)
        for m0 in range(0, M, BM):
            rows = m0 + tl.arange(0, BM)
            row_ok = rows < M
            gp = tl.load(
                gp_ptr + rows[:, None] * stride_gm + cols[None, :],
                mask=row_ok[:, None] & col_ok[None, :],
                other=0.0,
            ).to(tl.float32)
            x = tl.load(
                x_ptr + rows[:, None] * stride_xm + ks[None, :],
                mask=row_ok[:, None] & k_ok[None, :],
                other=0.0,
            ).to(tl.float32)
            acc = _dot_fp32(tl.trans(gp), x, acc, False, X_EXACT)
        tl.store(
            gw_ptr + cols[:, None] * stride_gwn + ks[None, :],
            acc,
            mask=col_ok[:, None] & k_ok[None, :],
        )


_BM = 64
_BK = 64


def _block_n(n_out: int) -> int:
    return max(16, triton.next_power_of_2(n_out))


def _tf32_exact(t: Tensor) -> bool:
    """bf16 / fp16 values upcast to fp32 are exactly representable in tf32 (10-bit mantissa)."""
    return t.dtype in (torch.bfloat16, torch.float16)


class _V41ProjRms(torch.autograd.Function):
    """``(proj, r) = (x @ W^T, rsqrt(mean(x^2) + eps))`` with the fused kernels above."""

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, eps: float) -> Tuple[Tensor, Tensor]:
        x = x.contiguous()
        weight = weight.contiguous()
        M, K = x.shape
        N = weight.shape[0]
        proj = torch.empty(M, N, dtype=torch.float32, device=x.device)
        r = torch.empty(M, dtype=torch.float32, device=x.device)
        grid = (triton.cdiv(M, _BM),)
        _v41_proj_rms_fwd_kernel[grid](
            x,
            weight,
            proj,
            r,
            M,
            K,
            N,
            eps,
            x.stride(0),
            weight.stride(0),
            proj.stride(0),
            BM=_BM,
            BK=_BK,
            BN=_block_n(N),
            X_EXACT=_tf32_exact(x),
            num_warps=4,
        )
        ctx.save_for_backward(x, weight, r)
        return proj, r.unsqueeze(1)

    @staticmethod
    def backward(ctx, grad_proj: Tensor, grad_r: Tensor):
        x, weight, r = ctx.saved_tensors
        M, K = x.shape
        N = weight.shape[0]
        if grad_proj is None:
            grad_proj = torch.zeros(M, N, dtype=torch.float32, device=x.device)
        grad_proj = grad_proj.to(torch.float32).contiguous()
        if grad_r is None:
            c = torch.zeros(M, dtype=torch.float32, device=x.device)
        else:
            # d r / d x = -r^3 x / K  (r = (mean(x^2) + eps)^{-1/2})
            c = -(grad_r.reshape(M).to(torch.float32) * r * r * r) / K
        grad_x = torch.empty_like(x)
        grad_w = torch.empty_like(weight, dtype=torch.float32)
        bn = _block_n(N)
        _v41_proj_rms_bwd_dx_kernel[(triton.cdiv(M, _BM), triton.cdiv(K, _BK))](
            x,
            weight,
            grad_proj,
            c,
            grad_x,
            M,
            K,
            N,
            x.stride(0),
            weight.stride(0),
            grad_proj.stride(0),
            grad_x.stride(0),
            BM=_BM,
            BK=_BK,
            BN=bn,
            num_warps=4,
        )
        _v41_proj_rms_bwd_dw_kernel[(triton.cdiv(K, _BK),)](
            x,
            grad_proj,
            grad_w,
            M,
            K,
            N,
            x.stride(0),
            grad_proj.stride(0),
            grad_w.stride(0),
            BM=_BM,
            BK=_BK,
            BN=bn,
            X_EXACT=_tf32_exact(x),
            num_warps=4,
        )
        return grad_x, grad_w.to(weight.dtype), None


def fused_v41_projection_and_rms(x: Tensor, weight: Tensor, eps: float) -> Tuple[Tensor, Tensor]:
    """``x`` [M, K] in the stream dtype (bf16 or fp32), ``weight`` [N, K] (kept in fp32 by the
    module). Returns ``proj`` [M, N] fp32 and ``r`` [M, 1] fp32, matching
    ``v41_projection_and_rms(x.float(), weight.float(), eps)``."""
    if not fused_v41_proj_rms_available() or not x.is_cuda:
        from megatron.core.models.deepseek_v41.hyper_connection import v41_projection_and_rms

        return v41_projection_and_rms(x.to(torch.float32), weight.to(torch.float32), eps)
    return _V41ProjRms.apply(x, weight, eps)
