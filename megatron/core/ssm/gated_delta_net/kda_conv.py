# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# ``causal_conv1d_fwd`` and ``causal_conv1d_bwd`` keep the interfaces and the semantics of the
# functions of the same names in flash-linear-attention 0.5.1
# (https://github.com/fla-org/flash-linear-attention, fla/modules/conv/triton/ops.py) for kernel
# width 4; the kernels are new.
#
# Licensed under the MIT license; see the LICENSE file in the repository root.

"""Row-walking kernels for FLA's short causal convolution, as KDA uses it.

FLA's Triton ``causal_conv1d`` kernels tile 64 rows by up to 128 channels. For GLM-5.3-Flash's
KDA (16,384 rows of 24,576 BF16 channels per CP rank, kernel width 4) its backward reruns the
forward to get the pre-activation and then spills registers (255 per thread, 12% occupancy):
4.1 ms per call on GB300, where reading ``x`` and ``dy`` and writing ``dx`` once takes about
0.5 ms. Its forward takes 0.62 ms.

The kernels here give each program a block of channels of one 64-row chunk and walk its rows,
keeping the last three input rows (and, backwards, the last three gradient rows) in registers:
every row of ``x``, ``dy`` and the output is read or written once, coalesced. The backward
recomputes the pre-activation on the way, writes ``dx`` three rows behind, the per-chunk weight
and bias gradients, and, at each sequence's first chunk, the initial-state gradient that FLA's
context parallelism sends to the previous rank: 0.91 ms per call; the forward takes 0.45 ms.
:func:`install` points FLA's conv ops (and so its CP conv) at them; inputs they do not cover
(another kernel width, a final state or its gradient, a residual, a batch of several unpacked
sequences) keep FLA's functions.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    from fla.modules.conv.triton import ops as _fla_conv_ops
    from fla.ops.utils import prepare_chunk_indices

    HAVE_FLA_CONV = True
except ImportError:
    HAVE_FLA_CONV = False

_FLA_CAUSAL_CONV1D_BWD = _fla_conv_ops.causal_conv1d_bwd if HAVE_FLA_CONV else None
_WIDTH = 4

if HAVE_FLA_CONV:

    @triton.jit
    def _chunk_bounds(cu_seqlens, chunk_indices, i_tg, T, IS_VARLEN: tl.constexpr):
        if IS_VARLEN:
            i_n = tl.load(chunk_indices + i_tg * 2).to(tl.int32)
            i_t = tl.load(chunk_indices + i_tg * 2 + 1).to(tl.int32)
            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            T = (tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos).to(tl.int32)
        else:
            i_n = 0
            i_t = i_tg
            bos = tl.cast(0, tl.int64)
        return i_n, i_t, bos, T

    @triton.autotune(
        configs=[
            triton.Config({"BD": bd}, num_warps=nw, num_stages=ns)
            for bd in (128, 256, 512)
            for nw in (2, 4, 8)
            for ns in (2, 3)
        ],
        key=["D", "BT", "ACT", "HAS_BIAS", "USE_H0"],
    )
    @triton.jit
    def _conv_w4_fwd_rows_kernel(
        x,
        y,
        weight,
        h0,
        cu_seqlens,
        chunk_indices,
        T,
        stride_x_t,
        stride_y_t,
        D: tl.constexpr,
        BT: tl.constexpr,
        BD: tl.constexpr,
        ACT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        USE_H0: tl.constexpr,
        IS_VARLEN: tl.constexpr,
    ):
        # y[t] = act(sum_j w[j] * x[t - 3 + j] + bias), walking a BT-row chunk row by row.
        i_d, i_tg = tl.program_id(0), tl.program_id(1)
        i_n, i_t, bos, T = _chunk_bounds(cu_seqlens, chunk_indices, i_tg, T, IS_VARLEN)
        o_d = i_d * BD + tl.arange(0, BD)
        m_d = o_d < D
        x_base = x + bos * stride_x_t + o_d
        y_base = y + bos * stride_y_t + o_d
        w0 = tl.load(weight + o_d * 4 + 0, mask=m_d, other=0.0).to(tl.float32)
        w1 = tl.load(weight + o_d * 4 + 1, mask=m_d, other=0.0).to(tl.float32)
        w2 = tl.load(weight + o_d * 4 + 2, mask=m_d, other=0.0).to(tl.float32)
        w3 = tl.load(weight + o_d * 4 + 3, mask=m_d, other=0.0).to(tl.float32)
        if HAS_BIAS:
            b_bias = tl.load(weight + D * 4 + o_d, mask=m_d, other=0.0).to(tl.float32)
        t0 = i_t * BT
        xa = tl.load(
            x_base + (t0 - 3).to(tl.int64) * stride_x_t, mask=m_d & (t0 - 3 >= 0), other=0.0
        ).to(tl.float32)
        xb = tl.load(
            x_base + (t0 - 2).to(tl.int64) * stride_x_t, mask=m_d & (t0 - 2 >= 0), other=0.0
        ).to(tl.float32)
        xc = tl.load(
            x_base + (t0 - 1).to(tl.int64) * stride_x_t, mask=m_d & (t0 - 1 >= 0), other=0.0
        ).to(tl.float32)
        if USE_H0:
            if i_t == 0:
                xa = tl.load(h0 + i_n * D * 4 + o_d * 4 + 1, mask=m_d, other=0.0).to(tl.float32)
                xb = tl.load(h0 + i_n * D * 4 + o_d * 4 + 2, mask=m_d, other=0.0).to(tl.float32)
                xc = tl.load(h0 + i_n * D * 4 + o_d * 4 + 3, mask=m_d, other=0.0).to(tl.float32)
        for r in range(0, BT):
            t = t0 + r
            m_t = m_d & (t < T)
            xt = tl.load(x_base + t.to(tl.int64) * stride_x_t, mask=m_t, other=0.0).to(tl.float32)
            b_y = xa * w0
            b_y += xb * w1
            b_y += xc * w2
            b_y += xt * w3
            if HAS_BIAS:
                b_y += b_bias
            if ACT:
                b_y = b_y * tl.sigmoid(b_y)
            tl.store(
                y_base + t.to(tl.int64) * stride_y_t,
                tl.cast(b_y, dtype=y.dtype.element_ty, fp_downcast_rounding="rtne"),
                mask=m_t,
            )
            xa = xb
            xb = xc
            xc = xt

    @triton.autotune(
        configs=[
            triton.Config({"BD": bd}, num_warps=nw, num_stages=ns)
            for bd in (128, 256, 512)
            for nw in (2, 4, 8)
            for ns in (2, 3)
        ],
        key=["D", "BT", "ACT", "HAS_BIAS", "USE_H0"],
    )
    @triton.jit
    def _conv_w4_bwd_rows_kernel(
        x,
        dy,
        dx,
        weight,
        h0,
        dw,
        db,
        dh0,
        cu_seqlens,
        chunk_indices,
        T,
        stride_x_t,
        stride_dy_t,
        stride_dx_t,
        D: tl.constexpr,
        BT: tl.constexpr,
        BD: tl.constexpr,
        ACT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        USE_H0: tl.constexpr,
        IS_VARLEN: tl.constexpr,
    ):
        # y_pre[t] = sum_j w[j] * x[t - 3 + j] (+ bias); g = dy * act'(y_pre);
        # dx[t] = sum_k w[3 - k] * g[t + k]; per BT-row chunk dw[j] = sum_t g[t] * x[t - 3 + j],
        # db = sum_t g[t]; dh0 at each sequence's first chunk.
        # A program owns BD channels of one chunk and walks its rows, keeping the last three x and
        # g rows in registers; it computes g for three rows past the chunk to finish dx there.
        i_d, i_tg = tl.program_id(0), tl.program_id(1)
        i_n, i_t, bos, T = _chunk_bounds(cu_seqlens, chunk_indices, i_tg, T, IS_VARLEN)
        o_d = i_d * BD + tl.arange(0, BD)
        m_d = o_d < D
        x_base = x + bos * stride_x_t + o_d
        dy_base = dy + bos * stride_dy_t + o_d
        dx_base = dx + bos * stride_dx_t + o_d
        w0 = tl.load(weight + o_d * 4 + 0, mask=m_d, other=0.0).to(tl.float32)
        w1 = tl.load(weight + o_d * 4 + 1, mask=m_d, other=0.0).to(tl.float32)
        w2 = tl.load(weight + o_d * 4 + 2, mask=m_d, other=0.0).to(tl.float32)
        w3 = tl.load(weight + o_d * 4 + 3, mask=m_d, other=0.0).to(tl.float32)
        if HAS_BIAS:
            b_bias = tl.load(weight + D * 4 + o_d, mask=m_d, other=0.0).to(tl.float32)
        t0 = i_t * BT
        # x rows t0 - 3 .. t0 - 1: the previous chunk's rows, or the initial state (rows -3..-1)
        xa = tl.load(
            x_base + (t0 - 3).to(tl.int64) * stride_x_t, mask=m_d & (t0 - 3 >= 0), other=0.0
        ).to(tl.float32)
        xb = tl.load(
            x_base + (t0 - 2).to(tl.int64) * stride_x_t, mask=m_d & (t0 - 2 >= 0), other=0.0
        ).to(tl.float32)
        xc = tl.load(
            x_base + (t0 - 1).to(tl.int64) * stride_x_t, mask=m_d & (t0 - 1 >= 0), other=0.0
        ).to(tl.float32)
        if USE_H0:
            if i_t == 0:
                xa = tl.load(h0 + i_n * D * 4 + o_d * 4 + 1, mask=m_d, other=0.0).to(tl.float32)
                xb = tl.load(h0 + i_n * D * 4 + o_d * 4 + 2, mask=m_d, other=0.0).to(tl.float32)
                xc = tl.load(h0 + i_n * D * 4 + o_d * 4 + 3, mask=m_d, other=0.0).to(tl.float32)
        ga = tl.zeros([BD], dtype=tl.float32)
        gb = tl.zeros([BD], dtype=tl.float32)
        gc = tl.zeros([BD], dtype=tl.float32)
        dw0 = tl.zeros([BD], dtype=tl.float32)
        dw1 = tl.zeros([BD], dtype=tl.float32)
        dw2 = tl.zeros([BD], dtype=tl.float32)
        dw3 = tl.zeros([BD], dtype=tl.float32)
        b_db = tl.zeros([BD], dtype=tl.float32)
        g_r0 = tl.zeros([BD], dtype=tl.float32)
        g_r1 = tl.zeros([BD], dtype=tl.float32)
        g_r2 = tl.zeros([BD], dtype=tl.float32)
        for r in range(0, BT + 3):
            t = t0 + r
            m_t = m_d & (t < T)
            xt = tl.load(x_base + t.to(tl.int64) * stride_x_t, mask=m_t, other=0.0).to(tl.float32)
            b_y = xa * w0 + xb * w1 + xc * w2 + xt * w3
            if HAS_BIAS:
                b_y += b_bias
            gt = tl.load(dy_base + t.to(tl.int64) * stride_dy_t, mask=m_t, other=0.0).to(tl.float32)
            if ACT:
                b_s = tl.sigmoid(b_y)
                gt = gt * b_s * (1 + b_y * (1 - b_s))
            own = gt * (r < BT).to(tl.float32)
            dw0 += own * xa
            dw1 += own * xb
            dw2 += own * xc
            dw3 += own * xt
            b_db += own
            # dx of row t - 3 is complete once g[t] is known
            dxt = ga * w3 + gb * w2 + gc * w1 + gt * w0
            tl.store(
                dx_base + (t - 3).to(tl.int64) * stride_dx_t,
                dxt.to(dx.dtype.element_ty),
                mask=m_d & (r >= 3) & (t - 3 < T),
            )
            if USE_H0:
                g_r0 = tl.where(r == 0, gt, g_r0)
                g_r1 = tl.where(r == 1, gt, g_r1)
                g_r2 = tl.where(r == 2, gt, g_r2)
            xa = xb
            xb = xc
            xc = xt
            ga = gb
            gb = gc
            gc = gt
        o_w = tl.arange(0, 4)
        b_dw = tl.where(o_w[None, :] == 0, dw0[:, None], 0.0)
        b_dw += tl.where(o_w[None, :] == 1, dw1[:, None], 0.0)
        b_dw += tl.where(o_w[None, :] == 2, dw2[:, None], 0.0)
        b_dw += tl.where(o_w[None, :] == 3, dw3[:, None], 0.0)
        tl.store(
            dw + i_tg.to(tl.int64) * D * 4 + o_d[:, None] * 4 + o_w[None, :],
            b_dw,
            mask=m_d[:, None],
        )
        if HAS_BIAS:
            tl.store(db + i_tg.to(tl.int64) * D + o_d, b_db, mask=m_d)
        if USE_H0:
            if i_t == 0:
                # dh0[c] = sum_{t < c} g[t] * w[c - 1 - t], c = 1..3 (index 0 is padding)
                b_dh0 = tl.where(o_w[None, :] == 1, (g_r0 * w0)[:, None], 0.0)
                b_dh0 += tl.where(o_w[None, :] == 2, (g_r0 * w1 + g_r1 * w0)[:, None], 0.0)
                b_dh0 += tl.where(
                    o_w[None, :] == 3, (g_r0 * w2 + g_r1 * w1 + g_r2 * w0)[:, None], 0.0
                )
                tl.store(
                    dh0 + i_n * D * 4 + o_d[:, None] * 4 + o_w[None, :],
                    b_dh0.to(dh0.dtype.element_ty),
                    mask=m_d[:, None],
                )


def _supported(x, dy, dht, weight, residual, activation, cu_seqlens):
    return (
        HAVE_FLA_CONV
        and weight is not None
        and weight.dim() == 2
        and weight.shape[1] == _WIDTH
        and dht is None
        and residual is None
        and activation in (None, "silu", "swish")
        and x.dim() == 3
        and x.shape[0] == 1
        and x.shape[-1] == weight.shape[0]
        and x.stride(-1) == 1
        and dy.shape == x.shape
        and dy.stride(-1) == 1
        and x.is_cuda
    )


_FLA_CAUSAL_CONV1D_FWD = _fla_conv_ops.causal_conv1d_fwd if HAVE_FLA_CONV else None


def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    BT: int = 64,
    layout_fallback: bool = False,
):
    """FLA's ``causal_conv1d_fwd``: ``(y, final_state)``."""
    if output_final_state or not _supported(x, x, None, weight, residual, activation, cu_seqlens):
        return _FLA_CAUSAL_CONV1D_FWD(
            x=x,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            output_final_state=output_final_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            BT=BT,
            layout_fallback=layout_fallback,
        )
    _, T, D = x.shape
    weight_c = weight.contiguous()
    if bias is not None:
        weight_c = torch.cat((weight_c.float().reshape(-1), bias.float().reshape(-1)))
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    h0 = initial_state.contiguous() if initial_state is not None else None

    def grid(meta):
        return (triton.cdiv(D, meta["BD"]), NT)

    _conv_w4_fwd_rows_kernel[grid](
        x,
        y,
        weight_c,
        h0,
        cu_seqlens,
        chunk_indices,
        T,
        x.stride(1),
        y.stride(1),
        D=D,
        BT=BT,
        ACT=activation in ("silu", "swish"),
        HAS_BIAS=bias is not None,
        USE_H0=initial_state is not None,
        IS_VARLEN=cu_seqlens is not None,
    )
    return y, None


def causal_conv1d_bwd(
    x: torch.Tensor,
    dy: torch.Tensor,
    dht: torch.Tensor | None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    BT: int = 64,
    layout_fallback: bool = False,
):
    """FLA's ``causal_conv1d_bwd``: ``(dx, dw, db, dresidual, dinitial_state)``."""
    if not _supported(x, dy, dht, weight, residual, activation, cu_seqlens):
        return _FLA_CAUSAL_CONV1D_BWD(
            x=x,
            dy=dy,
            dht=dht,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            BT=BT,
            layout_fallback=layout_fallback,
        )
    _, T, D = x.shape
    weight_c = weight.contiguous()
    if bias is not None:
        # The kernel reads the bias right after the weight; pass both in one buffer.
        weight_c = torch.cat((weight_c.float().reshape(-1), bias.float().reshape(-1)))
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    dx = torch.empty_like(x)
    dw = torch.empty(NT, D, _WIDTH, device=x.device, dtype=torch.float32)
    db = torch.empty(NT, D, device=x.device, dtype=torch.float32) if bias is not None else None
    dh0 = torch.zeros_like(initial_state) if initial_state is not None else None
    if initial_state is not None and initial_state.dim() != 3:
        raise ValueError("initial_state must be [N, D, W]")
    h0 = initial_state.contiguous() if initial_state is not None else None

    def grid(meta):
        return (triton.cdiv(D, meta["BD"]), NT)

    _conv_w4_bwd_rows_kernel[grid](
        x,
        dy,
        dx,
        weight_c,
        h0,
        dw,
        db if db is not None else dw,
        dh0 if dh0 is not None else dw,
        cu_seqlens,
        chunk_indices,
        T,
        x.stride(1),
        dy.stride(1),
        dx.stride(1),
        D=D,
        BT=BT,
        ACT=activation in ("silu", "swish"),
        HAS_BIAS=bias is not None,
        USE_H0=initial_state is not None,
        IS_VARLEN=cu_seqlens is not None,
    )
    dw = dw.sum(0).to(weight)
    db = db.sum(0).to(bias) if bias is not None else None
    return dx, dw, db, None, dh0


def install() -> None:
    """Point FLA's Triton conv ops (and so its CP conv) at :func:`causal_conv1d_bwd`.

    FLA's ``CausalConv1dFunction`` and ``CausalConv1dFunctionCP`` look the name up in
    ``fla.modules.conv.triton.ops`` when they run. Idempotent; a no-op without FLA.
    """
    if HAVE_FLA_CONV:
        _fla_conv_ops.causal_conv1d_fwd = causal_conv1d_fwd
        _fla_conv_ops.causal_conv1d_bwd = causal_conv1d_bwd
