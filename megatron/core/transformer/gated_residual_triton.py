# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused Triton kernels for the gated-residual hyper-connection.

The torch path in ``gated_residual.py`` materializes the fp32 normalized
streams ``x̃`` ``[M, n*C]`` and re-reads them in several GEMM / elementwise
kernels. The module is memory-bound, so this path keeps only a bf16 copy of
``x̃`` (the operand of the down projection, also needed for its weight
gradient) and recomputes the fp32 value from ``h`` and the per-stream ``rstd``
wherever the torch path would read the fp32 tensor.

Notation (per token row ``m``; ``j`` = stream, ``c`` = channel, ``k = j*C + c``):

* ``rstd[m, j] = rsqrt(mean_c h[m, j, c]^2 + eps)``, ``x̂ = h * rstd``,
  ``x̃ = x̂ * g`` with ``g = 1 + gamma`` (zero-centered) or ``gamma``.
* ``P[j, m, :] = x̃_bf16[m, j, :] @ W_down[:, j, :]^T`` (fp32 per-stream partial
  sums of the down projection, cuBLAS); ``down = bf16(sum_j P)``.
* ``u = bf16(silu(down / n))``, ``L = u @ W_up^T``, ``G = sigmoid(L)``,
  ``mixed[m, c] = mean_j G[m, j, c] * x̃[m, j, c]``.
* ``z[m, j, i] = x̃[m, j, :] . W_inject[i, j, :]`` (fp32 partials),
  ``g_write = 2 * sigmoid(sum_j z / n)``.

Backward of the RMS norm needs ``S[m, j] = sum_c dx̃ * x̃`` before ``dh`` can be
written. ``dx̃`` has three terms (read-gate mean, down projection, write gate);
``S`` is assembled from per-term partial sums that avoid a second pass over
``dx̃``: the mean term from the read-gate backward kernel, the down term as
``ddown . P[j, m]`` and the write-gate term as ``dz . z[m, j]``.

All low-rank GEMMs run in cuBLAS (``P``, ``L = u @ W_up^T`` and
``ddown @ W_down`` in the weight dtype exactly like the torch path's
``F.linear``, plus the weight gradients); the Triton kernels are the memory-bound
glue around them, fused so the fp32 ``x̃`` and its gradient never reach memory.
``L`` is saved for the backward. Products of fp32 operands follow
torch's TF32 policy (``torch.backends.cuda.matmul.allow_tf32``) by rounding the
operands to TF32, as cuBLAS does for the torch path. Every reduction is a
fixed-order partial-sum buffer; no kernel uses atomics, so the forward is
bitwise reproducible (the mHC recompute path relies on that).

Selected by ``GatedResidualModule`` when ``config.deterministic_mode`` is
False; deterministic mode keeps the torch path.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    try:
        from triton.language.extra.cuda import libdevice
    except ImportError:  # pragma: no cover - older triton layout
        from triton.language.extra import libdevice

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - triton ships with every supported container
    HAVE_TRITON = False

# Fixed tile shapes. No autotuning: the reduction order of every kernel is a
# function of these constants only, so results do not depend on a tuning race.
_NORM_BM, _NORM_BC, _NORM_WARPS = 8, 128, 8
_MIX_BM, _MIX_BC, _MIX_WARPS = 8, 128, 4
_BMIX_BM, _BMIX_BC, _BMIX_WARPS = 4, 256, 4
# bwd norm: one program per (channel tile, stream, row split of ~_BNORM_ROWS_PER_SPLIT rows).
_BNORM_BM, _BNORM_BC, _BNORM_ROWS_PER_SPLIT, _BNORM_WARPS, _BNORM_STAGES = 8, 64, 512, 4, 1
_RED_BLOCK = 256
# The write-gate / write-back kernels unroll up to four streams.
MAX_FUSED_STREAMS = 4
_SMALL_BM, _BSMALL_BM = 2, 4
_WBF_BM, _WBF_BC = 4, 512
_WBB_BM, _WBB_BC = 2, 256


def _next_pow2(x: int) -> int:
    return 1 << (max(int(x), 1) - 1).bit_length()


def _round_tf32() -> bool:
    """Whether fp32 x fp32 products should be TF32-rounded, mirroring cuBLAS in torch."""
    return bool(torch.backends.cuda.matmul.allow_tf32)


if HAVE_TRITON:

    @triton.jit
    def _tf32(x):
        """Round fp32 to TF32 (10-bit mantissa, round-to-nearest, ties away)."""
        b = x.to(tl.uint32, bitcast=True)
        b = (b + 0x1000) & 0xFFFFE000
        return b.to(tl.float32, bitcast=True)

    @triton.jit
    def _fast_sigmoid(x):
        """sigmoid with an approximate (2-ulp) reciprocal instead of an IEEE division."""
        return libdevice.fast_dividef(1.0, 1.0 + tl.exp(-x))

    @triton.jit
    def _gr_fwd_norm_kernel(
        h_ptr,
        gamma_ptr,
        wi_ptr,
        rstd_ptr,
        xnb_ptr,
        z_ptr,
        M,
        eps,
        N: tl.constexpr,
        NP: tl.constexpr,
        C: tl.constexpr,
        ZERO_CENTERED: tl.constexpr,
        HAS_GATE: tl.constexpr,
        ROUND_TF32: tl.constexpr,
        BM: tl.constexpr,
        BC: tl.constexpr,
    ):
        """rstd, bf16 x̃ and the write-gate partials z for one (row block, stream j).

        A sum-of-squares pass and a normalize pass over [BM, C] in BC chunks. Partial sums are
        accumulated elementwise and reduced once, so the loop bodies contain no cross-thread
        reductions; the stream is a grid axis so enough programs are in flight.
        """
        NC: tl.constexpr = N * C
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        j = tl.program_id(1)
        rmask = rows < M
        ii = tl.arange(0, NP)
        base = rows[:, None] * NC + j * C
        ssq = tl.zeros([BM, BC], dtype=tl.float32)
        for c0 in range(0, C, BC):
            cc = c0 + tl.arange(0, BC)
            m2 = rmask[:, None] & (cc[None, :] < C)
            x = tl.load(h_ptr + base + cc[None, :], mask=m2, other=0.0).to(tl.float32)
            ssq += x * x
        rstd = tl.rsqrt(tl.sum(ssq, axis=1) / C + eps)
        tl.store(rstd_ptr + rows * N + j, rstd, mask=rmask)
        z0 = tl.zeros([BM, BC], dtype=tl.float32)
        z1 = tl.zeros([BM, BC], dtype=tl.float32)
        z2 = tl.zeros([BM, BC], dtype=tl.float32)
        z3 = tl.zeros([BM, BC], dtype=tl.float32)
        for c0 in range(0, C, BC):
            cc = c0 + tl.arange(0, BC)
            cmask = cc < C
            m2 = rmask[:, None] & cmask[None, :]
            x = tl.load(h_ptr + base + cc[None, :], mask=m2, other=0.0)
            gam = tl.load(gamma_ptr + j * C + cc, mask=cmask, other=0.0).to(tl.float32)
            if ZERO_CENTERED:
                gam = gam + 1.0
            xn = x.to(tl.float32) * rstd[:, None] * gam[None, :]
            tl.store(xnb_ptr + base + cc[None, :], xn.to(xnb_ptr.dtype.element_ty), mask=m2)
            if HAS_GATE:
                if ROUND_TF32:
                    xn = _tf32(xn)
                col = j * C + cc
                z0 += xn * _gr_load_wi_col(wi_ptr, 0, col, cmask, NC, ROUND_TF32)[None, :]
                if N > 1:
                    z1 += xn * _gr_load_wi_col(wi_ptr, 1, col, cmask, NC, ROUND_TF32)[None, :]
                if N > 2:
                    z2 += xn * _gr_load_wi_col(wi_ptr, 2, col, cmask, NC, ROUND_TF32)[None, :]
                if N > 3:
                    z3 += xn * _gr_load_wi_col(wi_ptr, 3, col, cmask, NC, ROUND_TF32)[None, :]
        if HAS_GATE:
            zacc = tl.where(ii[None, :] == 0, tl.sum(z0, axis=1)[:, None], 0.0)
            zacc += tl.where(ii[None, :] == 1, tl.sum(z1, axis=1)[:, None], 0.0)
            zacc += tl.where(ii[None, :] == 2, tl.sum(z2, axis=1)[:, None], 0.0)
            zacc += tl.where(ii[None, :] == 3, tl.sum(z3, axis=1)[:, None], 0.0)
            tl.store(
                z_ptr + rows[:, None] * (N * NP) + j * NP + ii[None, :], zacc, mask=rmask[:, None]
            )

    @triton.jit
    def _gr_fwd_small_kernel(
        p_ptr,
        z_ptr,
        down_ptr,
        u_ptr,
        gate_ptr,
        M,
        N: tl.constexpr,
        NP: tl.constexpr,
        R: tl.constexpr,
        RP: tl.constexpr,
        HAS_GATE: tl.constexpr,
        BM: tl.constexpr,
    ):
        """down = bf16(sum_j P); u = bf16(silu(down / n)); g_write = 2*sigmoid(sum_j z / n)."""
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < M
        rr = tl.arange(0, RP)
        m2 = rmask[:, None] & (rr[None, :] < R)
        acc = tl.zeros([BM, RP], dtype=tl.float32)
        for j in tl.static_range(N):
            acc += tl.load(p_ptr + (j * M + rows[:, None]) * R + rr[None, :], mask=m2, other=0.0)
        down = acc.to(down_ptr.dtype.element_ty)
        tl.store(down_ptr + rows[:, None] * R + rr[None, :], down, mask=m2)
        a = down.to(tl.float32) / N
        u = a * tl.sigmoid(a)
        tl.store(u_ptr + rows[:, None] * R + rr[None, :], u.to(u_ptr.dtype.element_ty), mask=m2)
        if HAS_GATE:
            ii = tl.arange(0, NP)
            zm = rmask[:, None] & (ii[None, :] < N)
            zs = tl.zeros([BM, NP], dtype=tl.float32)
            for j in tl.static_range(N):
                zs += tl.load(
                    z_ptr + rows[:, None] * (N * NP) + j * NP + ii[None, :], mask=zm, other=0.0
                )
            gate = 2.0 * tl.sigmoid(zs / N)
            tl.store(
                gate_ptr + rows[:, None] * N + ii[None, :],
                gate.to(gate_ptr.dtype.element_ty),
                mask=zm,
            )

    @triton.jit
    def _gr_stream_tile(
        h_ptr,
        gamma_ptr,
        rstd_ptr,
        l_ptr,
        rows,
        rmask,
        cc,
        cmask,
        j: tl.constexpr,
        N: tl.constexpr,
        C: tl.constexpr,
        ZERO_CENTERED: tl.constexpr,
        FAST_SIGMOID: tl.constexpr = False,
    ):
        """Stream j, channels cc: (G, x̂, g, rstd) as [BM, BC], [BM, BC], [BC], [BM]."""
        NC: tl.constexpr = N * C
        m2 = rmask[:, None] & cmask[None, :]
        off = rows[:, None] * NC + j * C + cc[None, :]
        L = tl.load(l_ptr + off, mask=m2, other=0.0).to(tl.float32)
        if FAST_SIGMOID:
            G = _fast_sigmoid(L)
        else:
            G = tl.sigmoid(L)
        x = tl.load(h_ptr + off, mask=m2, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + rows * N + j, mask=rmask, other=0.0)
        gam = tl.load(gamma_ptr + j * C + cc, mask=cmask, other=0.0).to(tl.float32)
        if ZERO_CENTERED:
            gam = gam + 1.0
        return G, x * rstd[:, None], gam, rstd

    @triton.jit
    def _gr_fwd_mix_kernel(
        h_ptr,
        gamma_ptr,
        rstd_ptr,
        l_ptr,
        mixed_ptr,
        M,
        N: tl.constexpr,
        C: tl.constexpr,
        ZERO_CENTERED: tl.constexpr,
        BM: tl.constexpr,
        BC: tl.constexpr,
    ):
        """mixed[m, c] = mean_j sigmoid(L)[m, j, c] * x̃[m, j, c]."""
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        rmask = rows < M
        cc = tl.program_id(1) * BC + tl.arange(0, BC)
        cmask = cc < C
        mixed = tl.zeros([BM, BC], dtype=tl.float32)
        for j in tl.static_range(N):
            G, xhat, gam, rstd = _gr_stream_tile(
                h_ptr, gamma_ptr, rstd_ptr, l_ptr, rows, rmask, cc, cmask, j, N, C, ZERO_CENTERED
            )
            mixed += G * xhat * gam[None, :]
        tl.store(
            mixed_ptr + rows[:, None] * C + cc[None, :],
            (mixed / N).to(mixed_ptr.dtype.element_ty),
            mask=rmask[:, None] & cmask[None, :],
        )

    @triton.jit
    def _gr_bwd_mix_kernel(
        h_ptr,
        gamma_ptr,
        rstd_ptr,
        l_ptr,
        dmixed_ptr,
        dl_ptr,
        sa_ptr,
        M,
        N: tl.constexpr,
        NP: tl.constexpr,
        C: tl.constexpr,
        ZERO_CENTERED: tl.constexpr,
        BM: tl.constexpr,
        BC: tl.constexpr,
    ):
        """dL = dmixed x̃ / n * G (1 - G); per-tile partial S_A[m, j] = sum_c (dmixed G / n) x̃."""
        NC: tl.constexpr = N * C
        pid_c = tl.program_id(1)
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        rmask = rows < M
        cc = pid_c * BC + tl.arange(0, BC)
        cmask = cc < C
        m2 = rmask[:, None] & cmask[None, :]
        dmn = tl.load(dmixed_ptr + rows[:, None] * C + cc[None, :], mask=m2, other=0.0)
        dmn = dmn.to(tl.float32) / N
        jj = tl.arange(0, NP)
        sa = tl.zeros([BM, NP], dtype=tl.float32)
        for j in tl.static_range(N):
            G, xhat, gam, rstd = _gr_stream_tile(
                h_ptr,
                gamma_ptr,
                rstd_ptr,
                l_ptr,
                rows,
                rmask,
                cc,
                cmask,
                j,
                N,
                C,
                ZERO_CENTERED,
                True,
            )
            xn = xhat * gam[None, :]
            dl = dmn * xn * G * (1.0 - G)
            tl.store(
                dl_ptr + rows[:, None] * NC + j * C + cc[None, :],
                dl.to(dl_ptr.dtype.element_ty),
                mask=m2,
            )
            sj = tl.sum(dmn * G * xn, axis=1)
            sa += tl.where(jj[None, :] == j, sj[:, None], 0.0)
        tl.store(sa_ptr + (pid_c * M + rows[:, None]) * NP + jj[None, :], sa, mask=rmask[:, None])

    @triton.jit
    def _gr_bwd_small_kernel(
        du_ptr,
        down_ptr,
        dgate_ptr,
        z_ptr,
        p_ptr,
        sa_ptr,
        ddown_ptr,
        dz_ptr,
        s_ptr,
        M,
        NCT,
        N: tl.constexpr,
        NP: tl.constexpr,
        R: tl.constexpr,
        RP: tl.constexpr,
        HAS_GATE: tl.constexpr,
        BM: tl.constexpr,
    ):
        """ddown, dz (write-gate logit grads) and the per-stream norm reduction S."""
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        rmask = rows < M
        rr = tl.arange(0, RP)
        m2 = rmask[:, None] & (rr[None, :] < R)
        du = tl.load(du_ptr + rows[:, None] * R + rr[None, :], mask=m2, other=0.0).to(tl.float32)
        down = tl.load(down_ptr + rows[:, None] * R + rr[None, :], mask=m2, other=0.0)
        a = down.to(tl.float32) / N
        sg = tl.sigmoid(a)
        ddown = (du * sg * (1.0 + a * (1.0 - sg)) / N).to(ddown_ptr.dtype.element_ty)
        tl.store(ddown_ptr + rows[:, None] * R + rr[None, :], ddown, mask=m2)
        ddf = ddown.to(tl.float32)
        jj = tl.arange(0, NP)
        jm = rmask[:, None] & (jj[None, :] < N)
        s = tl.zeros([BM, NP], dtype=tl.float32)
        for ct in range(0, NCT):
            s += tl.load(sa_ptr + (ct * M + rows[:, None]) * NP + jj[None, :], mask=jm, other=0.0)
        if HAS_GATE:
            zs = tl.zeros([BM, NP], dtype=tl.float32)
            for j in tl.static_range(N):
                zs += tl.load(
                    z_ptr + rows[:, None] * (N * NP) + j * NP + jj[None, :], mask=jm, other=0.0
                )
            sgz = tl.sigmoid(zs / N)
            dg = tl.load(dgate_ptr + rows[:, None] * N + jj[None, :], mask=jm, other=0.0)
            dz = dg.to(tl.float32) * 2.0 * sgz * (1.0 - sgz) / N  # [BM, NP] over gate index i
            tl.store(dz_ptr + rows[:, None] * NP + jj[None, :], dz, mask=rmask[:, None])
        for j in tl.static_range(N):
            p = tl.load(p_ptr + (j * M + rows[:, None]) * R + rr[None, :], mask=m2, other=0.0)
            sj = tl.sum(ddf * p, axis=1)
            if HAS_GATE:
                zj = tl.load(
                    z_ptr + rows[:, None] * (N * NP) + j * NP + jj[None, :], mask=jm, other=0.0
                )
                sj += tl.sum(dz * zj, axis=1)
            s += tl.where(jj[None, :] == j, sj[:, None], 0.0)
        tl.store(s_ptr + rows[:, None] * NP + jj[None, :], s, mask=rmask[:, None])

    @triton.jit
    def _gr_bwd_norm_kernel(
        h_ptr,
        gamma_ptr,
        rstd_ptr,
        l_ptr,
        b_ptr,
        wi_ptr,
        dmixed_ptr,
        dz_ptr,
        s_ptr,
        dh_ptr,
        dgam_part_ptr,
        dwi_part_ptr,
        M,
        ROWS_PER_SPLIT,
        N: tl.constexpr,
        NP: tl.constexpr,
        C: tl.constexpr,
        ZERO_CENTERED: tl.constexpr,
        HAS_GATE: tl.constexpr,
        ROUND_TF32: tl.constexpr,
        BM: tl.constexpr,
        BC: tl.constexpr,
    ):
        """dx̃ = B + dmixed G / n + dz @ W_inject -> dh through the grouped RMS norm.

        One program owns (channel tile, stream j, row split) and walks its rows in BM blocks,
        accumulating the dgamma / dW_inject partials elementwise; they are reduced over rows
        once at the end into the per-split partial buffers. B = ddown @ W_down (cuBLAS).
        """
        NC: tl.constexpr = N * C
        INV_N: tl.constexpr = 1.0 / N
        INV_C: tl.constexpr = 1.0 / C
        cc = tl.program_id(0) * BC + tl.arange(0, BC)
        j = tl.program_id(1)
        sp = tl.program_id(2)
        jj = tl.arange(0, NP)
        cmask = cc < C
        col = j * C + cc
        gam = tl.load(gamma_ptr + col, mask=cmask, other=0.0).to(tl.float32)
        if ZERO_CENTERED:
            gam = gam + 1.0
        if HAS_GATE:
            w0 = _gr_load_wi_col(wi_ptr, 0, col, cmask, NC, ROUND_TF32)
            w1 = _gr_load_wi_col(wi_ptr, 1 % N, col, cmask, NC, ROUND_TF32)
            w2 = _gr_load_wi_col(wi_ptr, 2 % N, col, cmask, NC, ROUND_TF32)
            w3 = _gr_load_wi_col(wi_ptr, 3 % N, col, cmask, NC, ROUND_TF32)
        a_gam = tl.zeros([BM, BC], dtype=tl.float32)
        a0 = tl.zeros([BM, BC], dtype=tl.float32)
        a1 = tl.zeros([BM, BC], dtype=tl.float32)
        a2 = tl.zeros([BM, BC], dtype=tl.float32)
        a3 = tl.zeros([BM, BC], dtype=tl.float32)
        row0 = sp * ROWS_PER_SPLIT
        for r0 in range(row0, row0 + ROWS_PER_SPLIT, BM):
            rows = r0 + tl.arange(0, BM)
            rmask = (rows < M) & (rows < row0 + ROWS_PER_SPLIT)
            m2 = rmask[:, None] & cmask[None, :]
            off = rows[:, None] * NC + col[None, :]
            G = _fast_sigmoid(tl.load(l_ptr + off, mask=m2, other=0.0).to(tl.float32))
            x = tl.load(h_ptr + off, mask=m2, other=0.0).to(tl.float32)
            rstd = tl.load(rstd_ptr + rows * N + j, mask=rmask, other=0.0)
            xhat = x * rstd[:, None]
            dmn = tl.load(dmixed_ptr + rows[:, None] * C + cc[None, :], mask=m2, other=0.0)
            dxn = tl.load(b_ptr + off, mask=m2, other=0.0).to(tl.float32)
            dxn += dmn.to(tl.float32) * G * INV_N
            if HAS_GATE:
                xg = xhat * gam[None, :]
                if ROUND_TF32:
                    xg = _tf32(xg)
                dzt = tl.load(
                    dz_ptr + rows[:, None] * NP + jj[None, :], mask=rmask[:, None], other=0.0
                )
                if ROUND_TF32:
                    dzt = _tf32(dzt)
                dz0 = tl.sum(tl.where(jj[None, :] == 0, dzt, 0.0), axis=1)
                dxn += dz0[:, None] * w0[None, :]
                a0 += dz0[:, None] * xg
                if N > 1:
                    dz1 = tl.sum(tl.where(jj[None, :] == 1, dzt, 0.0), axis=1)
                    dxn += dz1[:, None] * w1[None, :]
                    a1 += dz1[:, None] * xg
                if N > 2:
                    dz2 = tl.sum(tl.where(jj[None, :] == 2, dzt, 0.0), axis=1)
                    dxn += dz2[:, None] * w2[None, :]
                    a2 += dz2[:, None] * xg
                if N > 3:
                    dz3 = tl.sum(tl.where(jj[None, :] == 3, dzt, 0.0), axis=1)
                    dxn += dz3[:, None] * w3[None, :]
                    a3 += dz3[:, None] * xg
            s = tl.load(s_ptr + rows * NP + j, mask=rmask, other=0.0) * INV_C
            dh = rstd[:, None] * (dxn * gam[None, :] - xhat * s[:, None])
            tl.store(dh_ptr + off, dh.to(dh_ptr.dtype.element_ty), mask=m2)
            a_gam += tl.where(m2, dxn * xhat, 0.0)
        tl.store(dgam_part_ptr + sp * NC + col, tl.sum(a_gam, axis=0), mask=cmask)
        if HAS_GATE:
            tl.store(dwi_part_ptr + (sp * N + 0) * NC + col, tl.sum(a0, axis=0), mask=cmask)
            if N > 1:
                tl.store(dwi_part_ptr + (sp * N + 1) * NC + col, tl.sum(a1, axis=0), mask=cmask)
            if N > 2:
                tl.store(dwi_part_ptr + (sp * N + 2) * NC + col, tl.sum(a2, axis=0), mask=cmask)
            if N > 3:
                tl.store(dwi_part_ptr + (sp * N + 3) * NC + col, tl.sum(a3, axis=0), mask=cmask)

    @triton.jit
    def _gr_load_wi_col(
        wi_ptr, i: tl.constexpr, col, cmask, NC: tl.constexpr, ROUND_TF32: tl.constexpr
    ):
        w = tl.load(wi_ptr + i * NC + col, mask=cmask, other=0.0).to(tl.float32)
        if ROUND_TF32:
            w = _tf32(w)
        return w

    @triton.jit
    def _gr_param_reduce_kernel(
        dgam_part_ptr,
        dwi_part_ptr,
        dgam_ptr,
        dwi_ptr,
        S,
        N: tl.constexpr,
        NC: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Fixed-order sum over the row splits of the dgamma (output 0) / dW_inject row i
        (output 1 + i) partials; grid = (column blocks, 1 + n or 1)."""
        k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        o = tl.program_id(1)
        kmask = k < NC
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        if o == 0:
            for s in range(0, S):
                acc += tl.load(dgam_part_ptr + s * NC + k, mask=kmask, other=0.0)
            tl.store(dgam_ptr + k, acc.to(dgam_ptr.dtype.element_ty), mask=kmask)
        else:
            i = o - 1
            for s in range(0, S):
                acc += tl.load(dwi_part_ptr + (s * N + i) * NC + k, mask=kmask, other=0.0)
            tl.store(dwi_ptr + i * NC + k, acc.to(dwi_ptr.dtype.element_ty), mask=kmask)

    @triton.jit
    def _gr_write_back_fwd_kernel(
        res_ptr,
        gate_ptr,
        x_ptr,
        bias_ptr,
        out_ptr,
        M,
        N: tl.constexpr,
        C: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BM: tl.constexpr,
        BC: tl.constexpr,
    ):
        """out[m, j, c] = residual[m, j, c] + g_write[m, j] * (x[m, c] + bias[c])."""
        NC: tl.constexpr = N * C
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cc = tl.program_id(1) * BC + tl.arange(0, BC)
        rmask = rows < M
        m2 = rmask[:, None] & (cc[None, :] < C)
        x = tl.load(x_ptr + rows[:, None] * C + cc[None, :], mask=m2, other=0.0).to(tl.float32)
        if HAS_BIAS:
            x += tl.load(bias_ptr + cc, mask=cc < C, other=0.0).to(tl.float32)[None, :]
        for j in tl.static_range(N):
            g = tl.load(gate_ptr + rows * N + j, mask=rmask, other=0.0).to(tl.float32)
            off = rows[:, None] * NC + j * C + cc[None, :]
            r = tl.load(res_ptr + off, mask=m2, other=0.0).to(tl.float32)
            tl.store(out_ptr + off, (r + g[:, None] * x).to(out_ptr.dtype.element_ty), mask=m2)

    @triton.jit
    def _gr_write_back_bwd_kernel(
        go_ptr,
        gate_ptr,
        x_ptr,
        bias_ptr,
        dx_ptr,
        dgate_ptr,
        M,
        N: tl.constexpr,
        NP: tl.constexpr,
        C: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BM: tl.constexpr,
        BC: tl.constexpr,
    ):
        """dx = sum_j g_write_j * go_j; dg_write_j = sum_c go_j * (x + bias)."""
        NC: tl.constexpr = N * C
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        rmask = rows < M
        jj = tl.arange(0, NP)
        g0 = tl.load(gate_ptr + rows * N + 0, mask=rmask, other=0.0).to(tl.float32)
        g1 = tl.load(gate_ptr + rows * N + 1 % N, mask=rmask, other=0.0).to(tl.float32)
        g2 = tl.load(gate_ptr + rows * N + 2 % N, mask=rmask, other=0.0).to(tl.float32)
        g3 = tl.load(gate_ptr + rows * N + 3 % N, mask=rmask, other=0.0).to(tl.float32)
        a0 = tl.zeros([BM, BC], dtype=tl.float32)
        a1 = tl.zeros([BM, BC], dtype=tl.float32)
        a2 = tl.zeros([BM, BC], dtype=tl.float32)
        a3 = tl.zeros([BM, BC], dtype=tl.float32)
        for c0 in range(0, C, BC):
            cc = c0 + tl.arange(0, BC)
            m2 = rmask[:, None] & (cc[None, :] < C)
            x = tl.load(x_ptr + rows[:, None] * C + cc[None, :], mask=m2, other=0.0).to(tl.float32)
            if HAS_BIAS:
                x += tl.load(bias_ptr + cc, mask=cc < C, other=0.0).to(tl.float32)[None, :]
            base = go_ptr + rows[:, None] * NC + cc[None, :]
            go = tl.load(base, mask=m2, other=0.0).to(tl.float32)
            dx = g0[:, None] * go
            a0 += go * x
            if N > 1:
                go = tl.load(base + C, mask=m2, other=0.0).to(tl.float32)
                dx += g1[:, None] * go
                a1 += go * x
            if N > 2:
                go = tl.load(base + 2 * C, mask=m2, other=0.0).to(tl.float32)
                dx += g2[:, None] * go
                a2 += go * x
            if N > 3:
                go = tl.load(base + 3 * C, mask=m2, other=0.0).to(tl.float32)
                dx += g3[:, None] * go
                a3 += go * x
            tl.store(
                dx_ptr + rows[:, None] * C + cc[None, :], dx.to(dx_ptr.dtype.element_ty), mask=m2
            )
        dg = tl.where(jj[None, :] == 0, tl.sum(a0, axis=1)[:, None], 0.0)
        dg += tl.where(jj[None, :] == 1, tl.sum(a1, axis=1)[:, None], 0.0)
        dg += tl.where(jj[None, :] == 2, tl.sum(a2, axis=1)[:, None], 0.0)
        dg += tl.where(jj[None, :] == 3, tl.sum(a3, axis=1)[:, None], 0.0)
        tl.store(
            dgate_ptr + rows[:, None] * N + jj[None, :],
            dg.to(dgate_ptr.dtype.element_ty),
            mask=rmask[:, None] & (jj[None, :] < N),
        )


def _launch_gates_fwd(h2, gamma, wd, wi, eps, n, C, zero_centered):
    """Forward kernels before the read-gate mix. Returns all state the backward needs."""
    M = h2.shape[0]
    R = wd.shape[0]
    NP = _next_pow2(n)
    dev = h2.device
    has_gate = wi is not None
    rstd = torch.empty(M, n, device=dev, dtype=torch.float32)
    xnb = torch.empty(M, n * C, device=dev, dtype=wd.dtype)
    z = torch.empty(M, n, NP, device=dev, dtype=torch.float32) if has_gate else None
    _gr_fwd_norm_kernel[(triton.cdiv(M, _NORM_BM), n)](
        h2,
        gamma,
        wi if has_gate else gamma,
        rstd,
        xnb,
        z if has_gate else rstd,
        M,
        eps,
        N=n,
        NP=NP,
        C=C,
        ZERO_CENTERED=zero_centered,
        HAS_GATE=has_gate,
        ROUND_TF32=_round_tf32(),
        BM=_NORM_BM,
        BC=_NORM_BC,
        num_warps=_NORM_WARPS,
    )
    # Per-stream partials of the down projection, fp32 out: [n, M, C] @ [n, C, R].
    bmm_kwargs = {} if wd.dtype == torch.float32 else {"out_dtype": torch.float32}
    p = torch.bmm(
        xnb.view(M, n, C).transpose(0, 1), wd.view(R, n, C).permute(1, 2, 0), **bmm_kwargs
    )
    down = torch.empty(M, R, device=dev, dtype=wd.dtype)
    u = torch.empty(M, R, device=dev, dtype=wd.dtype)
    gate = torch.empty(M, n, device=dev, dtype=wd.dtype) if has_gate else None
    _gr_fwd_small_kernel[(triton.cdiv(M, _SMALL_BM),)](
        p,
        z if has_gate else p,
        down,
        u,
        gate if has_gate else down,
        M,
        N=n,
        NP=NP,
        R=R,
        RP=_next_pow2(R),
        HAS_GATE=has_gate,
        BM=_SMALL_BM,
        num_warps=4,
    )
    return rstd, xnb, p, z, down, u, gate


class _GatedResidualGatesFn(torch.autograd.Function):
    """(h, gamma, W_down, W_up[, W_inject]) -> (mixed[, g_write]) with a fused backward."""

    @staticmethod
    def forward(ctx, h, gamma, wd, wu, wi, eps, n, zero_centered):
        """Run the fused read and write gates; save what the fused backward needs."""
        shape = h.shape
        nC = shape[-1]
        C = nC // n
        h2 = h.reshape(-1, nC)
        if not h2.is_contiguous():
            h2 = h2.contiguous()
        M = h2.shape[0]
        rstd, xnb, p, z, down, u, gate = _launch_gates_fwd(
            h2, gamma, wd, wi, eps, n, C, zero_centered
        )
        logits = torch.mm(u, wu.t())  # [M, nC] in the weight dtype, as F.linear in the torch path
        mixed = torch.empty(M, C, device=h.device, dtype=wd.dtype)
        grid = (triton.cdiv(M, _MIX_BM), triton.cdiv(C, _MIX_BC))
        _gr_fwd_mix_kernel[grid](
            h2,
            gamma,
            rstd,
            logits,
            mixed,
            M,
            N=n,
            C=C,
            ZERO_CENTERED=zero_centered,
            BM=_MIX_BM,
            BC=_MIX_BC,
            num_warps=_MIX_WARPS,
        )
        ctx.save_for_backward(h2, gamma, wd, wu, wi, rstd, xnb, p, z, down, u, logits)
        ctx.has_gate = wi is not None
        ctx.n, ctx.C, ctx.zero_centered, ctx.shape = n, C, zero_centered, shape
        mixed = mixed.view(*shape[:-1], C)
        if wi is None:
            return mixed
        return mixed, gate.view(*shape[:-1], n)

    @staticmethod
    def backward(ctx, dmixed, dgate=None):
        """Fused backward of the gates: input, gamma and projection-weight gradients."""
        h2, gamma, wd, wu, wi, rstd, xnb, p, z, down, u, logits = ctx.saved_tensors
        n, C, zc, has_gate = ctx.n, ctx.C, ctx.zero_centered, ctx.has_gate
        M, nC = h2.shape
        R = wd.shape[0]
        NP = _next_pow2(n)
        dev = h2.device
        if dmixed is None:
            dmixed = torch.zeros(M, C, device=dev, dtype=wd.dtype)
        dmixed = dmixed.reshape(M, C)
        if not dmixed.is_contiguous():
            dmixed = dmixed.contiguous()
        nct = triton.cdiv(C, _BMIX_BC)
        grid = (triton.cdiv(M, _BMIX_BM), nct)

        dl = torch.empty(M, nC, device=dev, dtype=wu.dtype)
        sa = torch.empty(nct, M, NP, device=dev, dtype=torch.float32)
        _gr_bwd_mix_kernel[grid](
            h2,
            gamma,
            rstd,
            logits,
            dmixed,
            dl,
            sa,
            M,
            N=n,
            NP=NP,
            C=C,
            ZERO_CENTERED=zc,
            BM=_BMIX_BM,
            BC=_BMIX_BC,
            num_warps=_BMIX_WARPS,
        )
        du = torch.mm(dl, wu)  # [M, R]
        dwu = torch.mm(dl.t(), u)  # [nC, R]
        del dl

        ddown = torch.empty(M, R, device=dev, dtype=wd.dtype)
        s = torch.empty(M, NP, device=dev, dtype=torch.float32)
        if has_gate:
            dz = torch.empty(M, NP, device=dev, dtype=torch.float32)
            if dgate is None:
                dgate = torch.zeros(M, n, device=dev, dtype=wd.dtype)
            dgate = dgate.reshape(M, n).contiguous()
        else:
            dz = dgate = s
        _gr_bwd_small_kernel[(triton.cdiv(M, _BSMALL_BM),)](
            du,
            down,
            dgate,
            z if has_gate else s,
            p,
            sa,
            ddown,
            dz,
            s,
            M,
            nct,
            N=n,
            NP=NP,
            R=R,
            RP=_next_pow2(R),
            HAS_GATE=has_gate,
            BM=_BSMALL_BM,
            num_warps=4,
        )
        dxn_down = torch.mm(ddown, wd)  # [M, nC], as the torch path's F.linear input grad

        dh = torch.empty_like(h2)
        rows_per_split = min(_BNORM_ROWS_PER_SPLIT, triton.cdiv(M, _BNORM_BM) * _BNORM_BM)
        splits = triton.cdiv(M, rows_per_split)
        dgam_part = torch.empty(splits, nC, device=dev, dtype=torch.float32)
        dwi_part = (
            torch.empty(splits, n, nC, device=dev, dtype=torch.float32) if has_gate else dgam_part
        )
        _gr_bwd_norm_kernel[(triton.cdiv(C, _BNORM_BC), n, splits)](
            h2,
            gamma,
            rstd,
            logits,
            dxn_down,
            wi if has_gate else gamma,
            dmixed,
            dz,
            s,
            dh,
            dgam_part,
            dwi_part,
            M,
            rows_per_split,
            N=n,
            NP=NP,
            C=C,
            ZERO_CENTERED=zc,
            HAS_GATE=has_gate,
            ROUND_TF32=_round_tf32(),
            BM=_BNORM_BM,
            BC=_BNORM_BC,
            num_warps=_BNORM_WARPS,
            num_stages=_BNORM_STAGES,
        )
        dwd = torch.mm(ddown.t(), xnb)  # [R, nC]
        dgamma = torch.empty(nC, device=dev, dtype=gamma.dtype)
        dwi = torch.empty(n, nC, device=dev, dtype=wi.dtype) if has_gate else None
        _gr_param_reduce_kernel[(triton.cdiv(nC, _RED_BLOCK), 1 + n if has_gate else 1)](
            dgam_part,
            dwi_part,
            dgamma,
            dwi if has_gate else dgamma,
            splits,
            N=n,
            NC=nC,
            BLOCK=_RED_BLOCK,
            num_warps=4,
        )
        return dh.view(ctx.shape), dgamma, dwd, dwu, dwi, None, None, None


class _GatedResidualWriteBackFn(torch.autograd.Function):
    """out = residual + g_write ⊙ (x [+ bias]) broadcast over the n streams."""

    @staticmethod
    def forward(ctx, residual, gate, x, bias):
        """Write the gated sublayer output back into the n residual streams."""
        s_shape = residual.shape
        n = gate.shape[-1]
        nC = s_shape[-1]
        C = nC // n
        res2 = residual.reshape(-1, nC).contiguous()
        gate2 = gate.reshape(-1, n).contiguous()
        x2 = x.reshape(-1, C).contiguous()
        M = res2.shape[0]
        inj_dtype = torch.promote_types(gate.dtype, x.dtype)
        if bias is not None:
            inj_dtype = torch.promote_types(inj_dtype, bias.dtype)
        out_dtype = torch.promote_types(residual.dtype, inj_dtype)
        out = torch.empty(M, nC, device=residual.device, dtype=out_dtype)
        has_bias = bias is not None
        grid = (triton.cdiv(M, _WBF_BM), triton.cdiv(C, _WBF_BC))
        _gr_write_back_fwd_kernel[grid](
            res2,
            gate2,
            x2,
            bias if has_bias else x2,
            out,
            M,
            N=n,
            C=C,
            HAS_BIAS=has_bias,
            BM=_WBF_BM,
            BC=_WBF_BC,
            num_warps=4,
        )
        ctx.save_for_backward(gate2, x2, bias)
        ctx.has_bias = has_bias
        ctx.meta = (s_shape, gate.shape, x.shape, residual.dtype, n, C)
        return out.view(s_shape)

    @staticmethod
    def backward(ctx, go):
        """Gradients of the write-back w.r.t. residual, gate, sublayer output and bias."""
        gate2, x2, bias = ctx.saved_tensors
        s_shape, g_shape, x_shape, res_dtype, n, C = ctx.meta
        M = x2.shape[0]
        go2 = go.reshape(M, n * C)
        if not go2.is_contiguous():
            go2 = go2.contiguous()
        dx = torch.empty_like(x2)
        dgate = torch.empty_like(gate2)
        _gr_write_back_bwd_kernel[(triton.cdiv(M, _WBB_BM),)](
            go2,
            gate2,
            x2,
            bias if ctx.has_bias else x2,
            dx,
            dgate,
            M,
            N=n,
            NP=_next_pow2(n),
            C=C,
            HAS_BIAS=ctx.has_bias,
            BM=_WBB_BM,
            BC=_WBB_BC,
            num_warps=4,
        )
        dres = go if go.dtype == res_dtype else go.to(res_dtype)
        dbias = None
        if ctx.has_bias and ctx.needs_input_grad[3]:
            dbias = dx.float().sum(0).to(bias.dtype)
        return dres, dgate.view(g_shape), dx.view(x_shape), dbias


def fused_gated_residual_gates(
    hidden_states: Tensor,
    norm_weight: Tensor,
    down_weight: Tensor,
    up_weight: Tensor,
    inject_weight: Optional[Tensor],
    eps: float,
    n: int,
    zero_centered_gamma: bool,
) -> Tuple[Tensor, ...]:
    """Fused read gate (and write gate when ``inject_weight`` is given).

    Returns ``(mixed,)`` or ``(mixed, g_write)`` with the same dtypes as the torch path:
    both in the gate-projection weight dtype.
    """
    out = _GatedResidualGatesFn.apply(
        hidden_states,
        norm_weight,
        down_weight,
        up_weight,
        inject_weight,
        eps,
        n,
        zero_centered_gamma,
    )
    return out if isinstance(out, tuple) else (out,)


def fused_gated_residual_write_back(
    residual: Tensor, h_post: Tensor, x: Tensor, bias: Optional[Tensor]
) -> Tensor:
    """``residual + h_post ⊙ (x [+ bias])``; same semantics as ``_native_gr_write_back``."""
    return _GatedResidualWriteBackFn.apply(residual, h_post, x, bias)
