# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused Triton kernels for the dense half of the Engram module.

The reference path (``Engram._dense_reference``) is a chain of about forty small PyTorch ops
over ``[tokens, num_streams * hidden]`` tensors: three group RMSNorms evaluated in fp32, the
key/query score, the sigmoid gate, the gated value, a channels-first permute for the
depthwise causal convolution, SiLU, the permute back, and the residual add. Most of its time
is strided copies and fp32 intermediates, not arithmetic.

This module computes everything after the two projections in two forward kernels and six
backward kernels. All but the first are 2-D ``[token block, channel block]`` tiles whose
channel block lies inside one residual stream, so per-token scalars (gate, norm statistics)
load once per row.

forward
    ``_gate_fwd_kernel``   one program per token: key/query norms, score, gate, the
                           conv-norm statistic of the gated value, and two row dot products
                           the key/query norm backward needs. All saved as ``[T, NS]``
                           scalars; the gated value itself is never materialized.
    ``_conv_fwd_kernel``   recompute the normed value at the ``kernel_size`` causal taps,
                           depthwise convolution, SiLU, residual add.
backward
    ``_conv_bwd_kernel``   SiLU backward (the convolution's output gradient) and per-tile
                           partial convolution-weight gradients.
    ``_cnorm_dot_kernel``  transposed convolution (the normed value's gradient), per-tile
                           partial row dot products for the conv-norm backward, partial
                           conv-norm gamma gradients.
    ``_gval_kernel``       conv-norm backward plus the residual gradient (the gated value's
                           gradient) and per-tile partial gate gradients.
    ``_value_bwd_kernel``  gradient of the shared value (sum over streams).
    ``_kq_bwd_kernel``     key and query norm backward and partial gamma gradients. The
                           backward of the scalar score chain (sigmoid, signed sqrt,
                           scaling) runs before it on ``[T, NS]`` scalars with the
                           reference's own PyTorch ops (``_score_chain_backward``); an
                           in-kernel evaluation is kept behind ``_SCALAR_CHAIN = "triton"``.

Numerics follow the reference op by op: every intermediate the reference holds in the
activation dtype is rounded to it here as well (normed key/query/value, the key*query
product, the score chain, the gate, the convolution output, SiLU, the products feeding the
reductions), and every reduction accumulates in fp32. What remains different is summation
order, the last ulp of transcendental functions, and one factorization: the key/query norm
backward uses ``sum_i round(g * q_i) * gamma_i * k_i ~= g * sum_i q_i * gamma_i * k_i`` with
the sum taken in the forward kernel, which avoids a second pass over two full-width tensors.

The gate is ``sigmoid(sign(s) * sqrt(|s|))`` of a bf16 score ``s``; its derivative is
``1 / (2 * sqrt(|s|))`` and grows without bound as ``s`` approaches zero. Any rounding
difference in ``s`` is therefore amplified in the gradients upstream of the score (key and
query projections, their norms, the Engram input, the tables). The reference itself is not
stable there: an input perturbation of one ulp moves those gradients by the same order as
the fused/reference difference. The tests measure that floor explicitly.

Partial parameter gradients are written per program and reduced with ``torch.sum``, so the
result is bitwise repeatable. No kernel reads a value on the host, and every specialization
is fixed by the model configuration (hidden size, stream count, kernel size, dilation), so
the path is CUDA-graph capturable and does not recompile when the token count changes.

Only the unsharded layout is handled: no context parallelism and no sequence-parallel halo.
The caller falls back to the reference path otherwise.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only without Triton
    HAVE_TRITON = False


# Tile sizes and warp counts. They depend only on the model configuration, never on the data.
_BLOCK_T = 32
_TILE_WARPS = 4
_ROW_WARPS = 8
_MAX_BLOCK_C = 128
_NUM_STATS = 7  # gate, score, rstd_key, rstd_query, rstd_value, key dot, query dot

# Where the backward of the scalar score chain (sigmoid, signed sqrt, scaling) runs: "torch"
# replays the reference's own autograd ops on the [T, NS] scalars, "triton" evaluates it
# inside _kq_bwd_kernel.
_SCALAR_CHAIN = "torch"
# With the torch chain, where the gate gradient sum(grad_value * shared_value) comes from:
# "triton" sums the per-tile partials of _gval_kernel, "torch" recomputes it with the
# reference's broadcast-multiply-and-sum (one extra pass over the gated-value gradient).
_GATE_GRAD = "triton"
# Tests set this to a dict to capture backward intermediates (stats, gate grad, score grad).
_DEBUG = None


def channel_block(hidden_size: int) -> int:
    """Largest power-of-two channel block (<= 128) that tiles one stream; 0 if none >= 16."""
    block = _MAX_BLOCK_C
    while block >= 16:
        if hidden_size % block == 0:
            return block
        block //= 2
    return 0


if HAVE_TRITON:

    @triton.jit
    def _rnd(x, DT: tl.constexpr):
        """Round an fp32 value through the activation dtype, as the reference stores it."""
        return x.to(DT).to(tl.float32)

    @triton.jit
    def _sign(x):
        return tl.where(x > 0, 1.0, tl.where(x < 0, -1.0, 0.0))

    @triton.jit(do_not_specialize=["T"])
    def _gate_fwd_kernel(
        k_ptr,
        h_ptr,
        sv_ptr,
        gk_ptr,
        gq_ptr,
        stats_ptr,
        T,
        eps,
        inv_sqrt_h,
        NS: tl.constexpr,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
        ZC: tl.constexpr,
    ):
        t = tl.program_id(0).to(tl.int64)
        DT = k_ptr.dtype.element_ty
        C: tl.constexpr = NS * H
        INV_H: tl.constexpr = 1.0 / H
        STAT = T * NS
        offs = tl.arange(0, BLOCK_H)
        m = offs < H
        sv = tl.load(sv_ptr + t * H + offs, mask=m, other=0.0).to(tl.float32)
        for s in tl.static_range(NS):
            base = t * C + s * H
            k = tl.load(k_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
            q = tl.load(h_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
            gk = tl.load(gk_ptr + s * H + offs, mask=m, other=0.0).to(tl.float32)
            gq = tl.load(gq_ptr + s * H + offs, mask=m, other=0.0).to(tl.float32)
            if ZC:
                gk = gk + 1.0
                gq = gq + 1.0
            rk = tl.rsqrt(tl.sum(k * k, axis=0) * INV_H + eps)
            rq = tl.rsqrt(tl.sum(q * q, axis=0) * INV_H + eps)
            kn = _rnd((k * rk) * gk, DT)
            qn = _rnd((q * rq) * gq, DT)
            prod = _rnd(kn * qn, DT)
            score = _rnd(tl.sum(prod, axis=0), DT)
            score = _rnd(score * inv_sqrt_h, DT)
            mag = _rnd(tl.maximum(tl.abs(score), 1e-6), DT)
            root = _rnd(tl.sqrt_rn(mag), DT)
            signed = _rnd(root * _sign(score), DT)
            gate = _rnd(1.0 / (1.0 + tl.exp(-signed)), DT)
            value = _rnd(gate * sv, DT)
            rv = tl.rsqrt(tl.sum(value * value, axis=0) * INV_H + eps)
            ak = tl.sum(qn * gk * k, axis=0)
            aq = tl.sum(kn * gq * q, axis=0)
            idx = t * NS + s
            tl.store(stats_ptr + idx, gate)
            tl.store(stats_ptr + STAT + idx, score)
            tl.store(stats_ptr + 2 * STAT + idx, rk)
            tl.store(stats_ptr + 3 * STAT + idx, rq)
            tl.store(stats_ptr + 4 * STAT + idx, rv)
            tl.store(stats_ptr + 5 * STAT + idx, ak)
            tl.store(stats_ptr + 6 * STAT + idx, aq)

    @triton.jit
    def _tile(T, NS: tl.constexpr, H: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr):
        """Rows, row mask, stream, in-stream channels, and global channels of this tile."""
        CT: tl.constexpr = H // BLOCK_C
        pid_c = tl.program_id(1)
        s = pid_c // CT
        hc = (pid_c % CT) * BLOCK_C + tl.arange(0, BLOCK_C)
        rows = tl.program_id(0).to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
        return rows, rows < T, s, hc, s * H + hc

    @triton.jit
    def _value_tile(
        sv_ptr,
        gate_ptr,
        rv_ptr,
        rows,
        valid,
        s,
        hc,
        NS: tl.constexpr,
        H: tl.constexpr,
        DT: tl.constexpr,
    ):
        """Return (gated value, its conv-norm rstd) at ``rows``; zeros where not ``valid``."""
        sv = tl.load(sv_ptr + rows[:, None] * H + hc[None, :], mask=valid[:, None], other=0.0)
        gate = tl.load(gate_ptr + rows * NS + s, mask=valid, other=0.0)
        rv = tl.load(rv_ptr + rows * NS + s, mask=valid, other=0.0)
        return _rnd(gate[:, None] * sv.to(tl.float32), DT), rv

    @triton.jit
    def _conv_tile(
        sv_ptr,
        gate_ptr,
        rv_ptr,
        w_ptr,
        rows,
        rmask,
        s,
        hc,
        cols,
        gamma,
        B,
        NS: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        DIL: tl.constexpr,
        DT: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Depthwise causal convolution of the normed value; also returns the value at rows."""
        acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
        value_t = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
        for j in tl.static_range(K):
            src = rows - ((K - 1 - j) * DIL) * B
            valid = rmask & (src >= 0)
            value, rv = _value_tile(sv_ptr, gate_ptr, rv_ptr, src, valid, s, hc, NS, H, DT)
            normed = _rnd((value * rv[:, None]) * gamma[None, :], DT)
            w = tl.load(w_ptr + cols * K + j).to(tl.float32)
            acc += w[None, :] * normed
            if j == K - 1:
                value_t = value
        return acc, value_t

    @triton.jit
    def _load_gamma(ptr, cols, ZC: tl.constexpr):
        gamma = tl.load(ptr + cols).to(tl.float32)
        if ZC:
            gamma = gamma + 1.0
        return gamma

    @triton.jit(do_not_specialize=["T"])
    def _conv_fwd_kernel(
        sv_ptr,
        stats_ptr,
        gc_ptr,
        w_ptr,
        out_ptr,
        T,
        B,
        NS: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        DIL: tl.constexpr,
        ZC: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        DT = out_ptr.dtype.element_ty
        C: tl.constexpr = NS * H
        rows, rmask, s, hc, cols = _tile(T, NS, H, BLOCK_T, BLOCK_C)
        gate_ptr = stats_ptr
        rv_ptr = stats_ptr + 4 * T * NS
        gamma = _load_gamma(gc_ptr, cols, ZC)
        acc, value_t = _conv_tile(
            sv_ptr, gate_ptr, rv_ptr, w_ptr, rows, rmask, s, hc, cols, gamma, B,
            NS, H, K, DIL, DT, BLOCK_T, BLOCK_C,
        )  # fmt: skip
        conv = _rnd(acc, DT)
        silu = _rnd(conv / (1.0 + tl.exp(-conv)), DT)
        out = value_t + silu
        tl.store(out_ptr + rows[:, None] * C + cols[None, :], out.to(DT), mask=rmask[:, None])

    @triton.jit(do_not_specialize=["T"])
    def _conv_bwd_kernel(
        go_ptr,
        sv_ptr,
        stats_ptr,
        gc_ptr,
        w_ptr,
        gconv_ptr,
        dw_ptr,
        T,
        B,
        NS: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        DIL: tl.constexpr,
        ZC: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        DT = gconv_ptr.dtype.element_ty
        C: tl.constexpr = NS * H
        rows, rmask, s, hc, cols = _tile(T, NS, H, BLOCK_T, BLOCK_C)
        gate_ptr = stats_ptr
        rv_ptr = stats_ptr + 4 * T * NS
        gamma = _load_gamma(gc_ptr, cols, ZC)
        acc, _ = _conv_tile(
            sv_ptr, gate_ptr, rv_ptr, w_ptr, rows, rmask, s, hc, cols, gamma, B,
            NS, H, K, DIL, DT, BLOCK_T, BLOCK_C,
        )  # fmt: skip
        conv = _rnd(acc, DT)
        sig = 1.0 / (1.0 + tl.exp(-conv))
        offs = rows[:, None] * C + cols[None, :]
        go = tl.load(go_ptr + offs, mask=rmask[:, None], other=0.0).to(tl.float32)
        gconv = _rnd(go * sig * (1.0 + conv * (1.0 - sig)), DT)
        tl.store(gconv_ptr + offs, gconv.to(DT), mask=rmask[:, None])
        # Weight gradient partial for this token tile: sum_t gconv[t] * normed[t - shift_j].
        pid_t = tl.program_id(0).to(tl.int64)
        for j in tl.static_range(K):
            src = rows - ((K - 1 - j) * DIL) * B
            valid = rmask & (src >= 0)
            value, rv = _value_tile(sv_ptr, gate_ptr, rv_ptr, src, valid, s, hc, NS, H, DT)
            normed = _rnd((value * rv[:, None]) * gamma[None, :], DT)
            tl.store(dw_ptr + (pid_t * K + j) * C + cols, tl.sum(gconv * normed, axis=0))

    @triton.jit(do_not_specialize=["T"])
    def _cnorm_dot_kernel(
        gconv_ptr,
        sv_ptr,
        stats_ptr,
        gc_ptr,
        w_ptr,
        gnormed_ptr,
        dot_ptr,
        dgc_ptr,
        T,
        B,
        NS: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        DIL: tl.constexpr,
        ZC: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        DT = gnormed_ptr.dtype.element_ty
        C: tl.constexpr = NS * H
        TILES: tl.constexpr = NS * (H // BLOCK_C)
        rows, rmask, s, hc, cols = _tile(T, NS, H, BLOCK_T, BLOCK_C)
        gamma = _load_gamma(gc_ptr, cols, ZC)
        # Transposed depthwise convolution: the output at row + shift_j used this row as tap j.
        gnormed = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
        for j in tl.static_range(K):
            dst = rows + ((K - 1 - j) * DIL) * B
            valid = rmask & (dst < T)
            w = tl.load(w_ptr + cols * K + j).to(tl.float32)
            g = tl.load(
                gconv_ptr + dst[:, None] * C + cols[None, :], mask=valid[:, None], other=0.0
            )
            gnormed += w[None, :] * g.to(tl.float32)
        gnormed = _rnd(gnormed, DT)
        offs = rows[:, None] * C + cols[None, :]
        tl.store(gnormed_ptr + offs, gnormed.to(DT), mask=rmask[:, None])
        value, rv = _value_tile(
            sv_ptr, stats_ptr, stats_ptr + 4 * T * NS, rows, rmask, s, hc, NS, H, DT
        )
        tl.store(
            dot_ptr + rows * TILES + tl.program_id(1),
            tl.sum(gnormed * gamma[None, :] * value, axis=1),
            mask=rmask,
        )
        pid_t = tl.program_id(0).to(tl.int64)
        tl.store(dgc_ptr + pid_t * C + cols, tl.sum(gnormed * (value * rv[:, None]), axis=0))

    @triton.jit(do_not_specialize=["T"])
    def _gval_kernel(
        go_ptr,
        gnormed_ptr,
        sv_ptr,
        stats_ptr,
        gc_ptr,
        dot_ptr,
        gval_ptr,
        ggate_ptr,
        T,
        NS: tl.constexpr,
        H: tl.constexpr,
        ZC: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
        CT_P2: tl.constexpr,
    ):
        DT = gval_ptr.dtype.element_ty
        C: tl.constexpr = NS * H
        CT: tl.constexpr = H // BLOCK_C
        TILES: tl.constexpr = NS * CT
        INV_H: tl.constexpr = 1.0 / H
        rows, rmask, s, hc, cols = _tile(T, NS, H, BLOCK_T, BLOCK_C)
        gamma = _load_gamma(gc_ptr, cols, ZC)
        ct = tl.arange(0, CT_P2)
        partial = tl.load(
            dot_ptr + rows[:, None] * TILES + s * CT + ct[None, :],
            mask=rmask[:, None] & (ct < CT)[None, :],
            other=0.0,
        )
        dot = tl.sum(partial, axis=1)
        value, rv = _value_tile(
            sv_ptr, stats_ptr, stats_ptr + 4 * T * NS, rows, rmask, s, hc, NS, H, DT
        )
        offs = rows[:, None] * C + cols[None, :]
        gnormed = tl.load(gnormed_ptr + offs, mask=rmask[:, None], other=0.0).to(tl.float32)
        rv2 = rv[:, None]
        gfrom_norm = _rnd(
            rv2 * (gnormed * gamma[None, :]) - (rv2 * rv2 * rv2) * value * (dot * INV_H)[:, None],
            DT,
        )
        go = tl.load(go_ptr + offs, mask=rmask[:, None], other=0.0).to(tl.float32)
        gval = _rnd(go + gfrom_norm, DT)
        tl.store(gval_ptr + offs, gval.to(DT), mask=rmask[:, None])
        sv = tl.load(sv_ptr + rows[:, None] * H + hc[None, :], mask=rmask[:, None], other=0.0)
        tl.store(
            ggate_ptr + rows * TILES + tl.program_id(1),
            tl.sum(_rnd(gval * sv.to(tl.float32), DT), axis=1),
            mask=rmask,
        )

    @triton.jit(do_not_specialize=["T"])
    def _value_bwd_kernel(
        gval_ptr,
        stats_ptr,
        gsv_ptr,
        T,
        NS: tl.constexpr,
        H: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        DT = gsv_ptr.dtype.element_ty
        C: tl.constexpr = NS * H
        rows = tl.program_id(0).to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
        hc = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        rmask = rows < T
        acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
        for s in tl.static_range(NS):
            gval = tl.load(
                gval_ptr + rows[:, None] * C + s * H + hc[None, :], mask=rmask[:, None], other=0.0
            )
            gate = tl.load(stats_ptr + rows * NS + s, mask=rmask, other=0.0)
            acc += _rnd(gval.to(tl.float32) * gate[:, None], DT)
        tl.store(gsv_ptr + rows[:, None] * H + hc[None, :], acc.to(DT), mask=rmask[:, None])

    @triton.jit(do_not_specialize=["T"])
    def _kq_bwd_kernel(
        k_ptr,
        h_ptr,
        gk_ptr,
        gq_ptr,
        stats_ptr,
        ggate_ptr,
        dk_ptr,
        dh_ptr,
        dgk_ptr,
        dgq_ptr,
        gsum_ptr,
        T,
        inv_sqrt_h,
        NS: tl.constexpr,
        H: tl.constexpr,
        ZC: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
        CT_P2: tl.constexpr,
        USE_GSUM: tl.constexpr,
    ):
        DT = dk_ptr.dtype.element_ty
        C: tl.constexpr = NS * H
        CT: tl.constexpr = H // BLOCK_C
        TILES: tl.constexpr = NS * CT
        INV_H: tl.constexpr = 1.0 / H
        rows, rmask, s, hc, cols = _tile(T, NS, H, BLOCK_T, BLOCK_C)
        gk = _load_gamma(gk_ptr, cols, ZC)
        gq = _load_gamma(gq_ptr, cols, ZC)
        STAT = T * NS
        idx = rows * NS + s
        gate = tl.load(stats_ptr + idx, mask=rmask, other=0.0)
        score = tl.load(stats_ptr + STAT + idx, mask=rmask, other=0.0)
        rk = tl.load(stats_ptr + 2 * STAT + idx, mask=rmask, other=0.0)
        rq = tl.load(stats_ptr + 3 * STAT + idx, mask=rmask, other=0.0)
        ak = tl.load(stats_ptr + 5 * STAT + idx, mask=rmask, other=0.0)
        aq = tl.load(stats_ptr + 6 * STAT + idx, mask=rmask, other=0.0)
        if USE_GSUM:
            # Score gradient precomputed by the caller (``_score_chain_backward``).
            gsum = tl.load(gsum_ptr + idx, mask=rmask, other=0.0)
        else:
            ct = tl.arange(0, CT_P2)
            partial = tl.load(
                ggate_ptr + rows[:, None] * TILES + s * CT + ct[None, :],
                mask=rmask[:, None] & (ct < CT)[None, :],
                other=0.0,
            )
            ggate = _rnd(tl.sum(partial, axis=1), DT)
            # Scalar chain: gate = sigmoid(sign(score) * sqrt(max(|score|, 1e-6))).
            gsigned = _rnd(ggate * (1.0 - gate) * gate, DT)
            sign = _sign(score)
            mag = tl.abs(score)
            root = _rnd(tl.sqrt_rn(_rnd(tl.maximum(mag, 1e-6), DT)), DT)
            groot = _rnd(gsigned * sign, DT)
            gmag = _rnd(tl.div_rn(groot, 2.0 * root), DT)
            gmag = tl.where(mag >= 1e-6, gmag, 0.0)
            gscore = _rnd(gmag * sign, DT)
            gsum = _rnd(gscore * inv_sqrt_h, DT)

        offs = rows[:, None] * C + cols[None, :]
        k = tl.load(k_ptr + offs, mask=rmask[:, None], other=0.0).to(tl.float32)
        q = tl.load(h_ptr + offs, mask=rmask[:, None], other=0.0).to(tl.float32)
        rk2 = rk[:, None]
        rq2 = rq[:, None]
        g2 = gsum[:, None]
        khat = k * rk2
        qhat = q * rq2
        gkn = _rnd(g2 * _rnd(qhat * gq[None, :], DT), DT)
        gqn = _rnd(g2 * _rnd(khat * gk[None, :], DT), DT)
        dk = rk2 * (gkn * gk[None, :]) - (rk2 * rk2 * rk2) * k * (g2 * ak[:, None] * INV_H)
        dh = rq2 * (gqn * gq[None, :]) - (rq2 * rq2 * rq2) * q * (g2 * aq[:, None] * INV_H)
        tl.store(dk_ptr + offs, dk.to(DT), mask=rmask[:, None])
        tl.store(dh_ptr + offs, dh.to(DT), mask=rmask[:, None])
        pid_t = tl.program_id(0).to(tl.int64)
        tl.store(dgk_ptr + pid_t * C + cols, tl.sum(gkn * khat, axis=0))
        tl.store(dgq_ptr + pid_t * C + cols, tl.sum(gqn * qhat, axis=0))


def _num_warps(block: int) -> int:
    return _ROW_WARPS if block >= 2048 else 4


def _score_chain_backward(grad_gate: Tensor, gate: Tensor, score: Tensor, hidden_size: int):
    """Gradient of ``sum(key * query)`` from the gate gradient, with the reference's own ops.

    The reference computes ``gate = sigmoid(sign(s) * sqrt(clamp_min(|s|, 1e-6)))`` with
    ``s = sum(key * query) / sqrt(H)``; these are the autograd formulas of exactly those ops,
    in the activation dtype, on ``[T, NS]`` tensors.
    """
    magnitude = score.abs()
    root = magnitude.clamp_min(1e-6).sqrt()
    grad_signed = torch.ops.aten.sigmoid_backward(grad_gate, gate)
    grad_root = grad_signed * score.sign()
    grad_clamped = grad_root / (2 * root)
    grad_magnitude = torch.where(magnitude >= 1e-6, grad_clamped, torch.zeros_like(grad_clamped))
    grad_score = grad_magnitude * score.sgn()
    return grad_score / float(hidden_size) ** 0.5


def _inv_sqrt(hidden_size: int) -> float:
    """``1 / sqrt(H)`` exactly as PyTorch divides a float32 tensor by a Python scalar.

    The CUDA kernel casts the scalar to float32 and multiplies by its float32 reciprocal.
    """
    root = torch.tensor(float(hidden_size) ** 0.5, dtype=torch.float32)
    return float(torch.reciprocal(root))


class FusedEngramDense(torch.autograd.Function):
    """Engram gate, conv-norm, causal depthwise convolution, SiLU, and residual in Triton.

    Inputs are ``[S, B, ...]`` tensors in the activation dtype: ``key`` is the key
    projection output and ``hidden`` the Engram input (both ``[S, B, NS * H]``), ``value``
    the shared value projection (``[S, B, H]``). Returns ``value * gate + conv`` with the
    reference path's layout ``[S, B, NS * H]``. ``H`` must be a multiple of 16.
    """

    @staticmethod
    def forward(
        ctx,
        key: Tensor,
        hidden: Tensor,
        value: Tensor,
        key_gamma: Tensor,
        query_gamma: Tensor,
        conv_gamma: Tensor,
        conv_weight: Tensor,
        num_streams: int,
        eps: float,
        zero_centered: bool,
        dilation: int,
    ) -> Tensor:
        """Gate, conv-norm and short convolution of the Engram dense half."""
        seq, batch = hidden.shape[0], hidden.shape[1]
        hidden_size = value.shape[-1]
        block_c = channel_block(hidden_size)
        if block_c == 0:
            raise ValueError(f"fused Engram needs hidden_size % 16 == 0, got {hidden_size}.")
        tokens = seq * batch
        kernel_size = conv_weight.shape[-1]
        key = key.contiguous()
        hidden = hidden.contiguous()
        value = value.contiguous()
        conv_weight = conv_weight.contiguous()
        key_gamma, query_gamma, conv_gamma = (
            key_gamma.contiguous(),
            query_gamma.contiguous(),
            conv_gamma.contiguous(),
        )
        stats = torch.empty(
            (_NUM_STATS, tokens, num_streams), device=hidden.device, dtype=torch.float32
        )
        output = torch.empty_like(hidden)
        block_h = triton.next_power_of_2(hidden_size)
        if tokens > 0:
            _gate_fwd_kernel[(tokens,)](
                key,
                hidden,
                value,
                key_gamma,
                query_gamma,
                stats,
                tokens,
                eps,
                _inv_sqrt(hidden_size),
                NS=num_streams,
                H=hidden_size,
                BLOCK_H=block_h,
                ZC=zero_centered,
                num_warps=_num_warps(block_h),
            )
            grid = (triton.cdiv(tokens, _BLOCK_T), num_streams * (hidden_size // block_c))
            _conv_fwd_kernel[grid](
                value,
                stats,
                conv_gamma,
                conv_weight,
                output,
                tokens,
                batch,
                NS=num_streams,
                H=hidden_size,
                K=kernel_size,
                DIL=dilation,
                ZC=zero_centered,
                BLOCK_T=_BLOCK_T,
                BLOCK_C=block_c,
                num_warps=_TILE_WARPS,
            )
        ctx.save_for_backward(
            key, hidden, value, key_gamma, query_gamma, conv_gamma, conv_weight, stats
        )
        ctx.config = (num_streams, zero_centered, dilation, seq, batch)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Fused backward: key, hidden, value, gamma and convolution-weight gradients."""
        key, hidden, value, key_gamma, query_gamma, conv_gamma, conv_weight, stats = (
            ctx.saved_tensors
        )
        num_streams, zero_centered, dilation, seq, batch = ctx.config
        hidden_size = value.shape[-1]
        channels = num_streams * hidden_size
        tokens = seq * batch
        kernel_size = conv_weight.shape[-1]
        block_c = channel_block(hidden_size)
        col_tiles = hidden_size // block_c
        tiles = num_streams * col_tiles
        ct_p2 = triton.next_power_of_2(col_tiles)
        grad_output = grad_output.contiguous()
        device = hidden.device
        f32 = torch.float32

        row_tiles = triton.cdiv(tokens, _BLOCK_T)
        grid = (row_tiles, tiles)
        # Every element of the partial buffers is written by exactly one program.
        dw_partial = torch.empty((row_tiles, kernel_size, channels), device=device, dtype=f32)
        gamma_partial = torch.empty((3, row_tiles, channels), device=device, dtype=f32)
        dgc_partial, dgk_partial, dgq_partial = gamma_partial.unbind(0)
        row_partial = torch.empty((2, tokens, tiles), device=device, dtype=f32)
        dot_partial, ggate_partial = row_partial.unbind(0)
        # grad_conv is dead after _cnorm_dot_kernel, so it holds the gated value's gradient.
        grad_conv = torch.empty_like(grad_output)
        grad_normed = torch.empty_like(grad_output)
        grad_value = torch.empty_like(value)
        grad_key = torch.empty_like(key)
        grad_hidden = torch.empty_like(hidden)
        common = dict(NS=num_streams, H=hidden_size, BLOCK_T=_BLOCK_T, BLOCK_C=block_c)
        conv = dict(K=kernel_size, DIL=dilation, ZC=zero_centered)

        if tokens > 0:
            _conv_bwd_kernel[grid](
                grad_output,
                value,
                stats,
                conv_gamma,
                conv_weight,
                grad_conv,
                dw_partial,
                tokens,
                batch,
                **common,
                **conv,
                num_warps=_TILE_WARPS,
            )
            _cnorm_dot_kernel[grid](
                grad_conv,
                value,
                stats,
                conv_gamma,
                conv_weight,
                grad_normed,
                dot_partial,
                dgc_partial,
                tokens,
                batch,
                **common,
                **conv,
                num_warps=_TILE_WARPS,
            )
            grad_value_streams = grad_conv
            _gval_kernel[grid](
                grad_output,
                grad_normed,
                value,
                stats,
                conv_gamma,
                dot_partial,
                grad_value_streams,
                ggate_partial,
                tokens,
                ZC=zero_centered,
                CT_P2=ct_p2,
                **common,
                num_warps=_TILE_WARPS,
            )
            _value_bwd_kernel[(row_tiles, col_tiles)](
                grad_value_streams, stats, grad_value, tokens, **common, num_warps=_TILE_WARPS
            )
            use_gsum = _SCALAR_CHAIN == "torch" or _DEBUG is not None
            if use_gsum:
                dtype = hidden.dtype
                if _GATE_GRAD == "torch":
                    grad_gate = (
                        grad_value_streams.view(tokens, num_streams, hidden_size)
                        * value.view(tokens, 1, hidden_size)
                    ).sum(-1)
                else:
                    grad_gate = ggate_partial.view(tokens, num_streams, col_tiles).sum(-1).to(dtype)
                gate_act, score_act = stats[0].to(dtype), stats[1].to(dtype)
                grad_sum = _score_chain_backward(grad_gate, gate_act, score_act, hidden_size)
                if _DEBUG is not None:
                    _DEBUG.update(
                        stats=stats.clone(), grad_gate=grad_gate.clone(), grad_sum=grad_sum.clone()
                    )
                grad_sum = grad_sum.float()
            else:
                grad_sum = stats[0]  # unused placeholder pointer
            _kq_bwd_kernel[grid](
                key,
                hidden,
                key_gamma,
                query_gamma,
                stats,
                ggate_partial,
                grad_key,
                grad_hidden,
                dgk_partial,
                dgq_partial,
                grad_sum,
                tokens,
                _inv_sqrt(hidden_size),
                ZC=zero_centered,
                CT_P2=ct_p2,
                USE_GSUM=_SCALAR_CHAIN == "torch",
                **common,
                num_warps=_TILE_WARPS,
            )
        else:
            dw_partial.zero_()
            gamma_partial.zero_()

        grad_conv_weight = dw_partial.sum(0).t().reshape(conv_weight.shape).to(conv_weight.dtype)
        gamma_grads = gamma_partial.sum(1)
        return (
            grad_key,
            grad_hidden,
            grad_value,
            gamma_grads[1].to(key_gamma.dtype),
            gamma_grads[2].to(query_gamma.dtype),
            gamma_grads[0].to(conv_gamma.dtype),
            grad_conv_weight,
            None,
            None,
            None,
            None,
        )


def fused_engram_dense(
    key: Tensor,
    hidden: Tensor,
    value: Tensor,
    key_gamma: Tensor,
    query_gamma: Tensor,
    conv_gamma: Tensor,
    conv_weight: Tensor,
    num_streams: int,
    eps: float,
    zero_centered: bool,
    dilation: int,
) -> Tensor:
    """Functional entry point for :class:`FusedEngramDense`."""
    return FusedEngramDense.apply(
        key,
        hidden,
        value,
        key_gamma,
        query_gamma,
        conv_gamma,
        conv_weight,
        num_streams,
        eps,
        zero_centered,
        dilation,
    )
