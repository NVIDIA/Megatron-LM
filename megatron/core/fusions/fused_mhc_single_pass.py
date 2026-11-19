# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fast single-pass (V4.1) mHC: fewer passes over the residual and cheap launches.

The fused single-pass mHC owner routes supported HybridModel inputs (a contiguous BF16
residual and a compatible projection) to these Triton kernels.
Per sublayer the default fused path makes about 16 kernel launches through the Triton and cuTile
Python launchers (tens of microseconds of host time each on Grace), and its backward sums the
residual's three gradients in a separate pass after an aggregate backward and a mapping backward
that each read and write a full residual-sized tensor. This path (n-stream residual x [T, n C]):

forward
  ``_sp_agg_sq_kernel``   aggregate ``sum_j pre_mix_j * x_j`` (the FP32 multiply-add chain of the
                          Triton aggregate, so bit-identical) and sum(x^2) partials, in one read.
  ``_sp_proj_kernel``     split-K projection x @ W^T on tensor cores, W split into BF16 hi + lo
                          terms (BF16 x makes every product exact; W keeps ~16 bits, against
                          TF32's 10 in the cuTile and Inductor paths).
  ``_sp_pre_fin_kernel``  reduce the partials, RMS 1 / sqrt(mean(x^2) + eps), the coefficient
                          activations and the Sinkhorn projection, per block of tokens.
  ``_sp_bda_fwd_kernel``  H_res^T x + H_post * y as explicit fused multiply-adds; it also writes
                          the next sublayer's aggregate and sum(x^2) partials from its output
                          (``NextAggregate``), so that sublayer skips ``_sp_agg_sq_kernel``.
backward
  ``_sp_bda_bwd4_kernel``  for n = 4, the values of fused_mhc_kernels' two Triton BDA backward
                          kernels (bit-identical) reading grad_out and the residual once (other n
                          lean-launch those two kernels); the residual gradient goes to a stash for
                          the same sublayer's mapping backward instead of autograd. On VR200,
                          ``_sp_bda_bwd4t_kernel`` (one 128-column tile per program, all loads
                          first) + ``_sp_bda_bwd_rowsum_kernel`` (the tiles' row sums added in
                          tile order) give the same bits faster.
  ``_sp_pre_bwd_fin_kernel``  Sinkhorn and coefficient backward:
                          dL/dproj (as BF16 hi + lo), the RMS term and alpha / bias partials; extra
                          programs split W into BF16 hi + lo.
  ``_sp_pre_bwd_main_kernel`` x's whole gradient
                          ``round(round(g_res + round(g_agg * pre_mix_j)) + round(g_map))``, the
                          operands, order and roundings of the default exact three-way sum
                          (MCORE_MHC_SP_FUSED_GRAD_ADD=exact), and pre-mix gradient partials.
                          On VR200, ``_sp_pre_bwd_main_tma_kernel`` does the same in token-major
                          tiles with TMA loads in a software-pipelined loop (on GB300,
                          ``_sp_pre_bwd_main_pf_kernel``: the next block's loads issued ahead).
                          The projection and W-gradient kernels have TMA variants too.
  ``_sp_gw_kernel``       W gradient a^T x, split over tokens.
  ``_sp_pre_bwd_reduce_kernel`` reduce the partials.

Numerics: not bit-identical to either existing mapping path (Inductor/TF32 cuBLAS or cuTile TF32
MMAs), hence opt-in; the projection and its gradients are more accurate than both. Given the
same inputs, the aggregate, the BDA backward (residual, branch and H gradients) and the
gradient-sum chain are bit-identical to the default Triton kernels'; the BDA output can differ
in the last bit (explicit fused multiply-adds). Checkpoint replays reproduce the forward bits.

All kernels take runtime arguments first and constexprs last and launch through ``_LeanLaunch``.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from megatron.core.env_flags import env_flag

# Optionally, each BDA also writes the next sublayer's aggregate and sum(x^2)
# partials (``NextAggregate``), so that sublayer skips ``_sp_agg_sq_kernel``. On the 64K proxy
# this saved about 40 ms per step (0.2%) but raised peak allocated memory by about 1.6 GB, for
# a reason not yet identified, so it stays off by default.
SINGLE_PASS_FAST_NEXT_AGG = env_flag("MCORE_MHC_SP_FAST_NEXT_AGG", False)

try:
    import triton
    import triton.language as tl
    from triton import knobs as _triton_knobs

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - Triton ships with the training containers
    _TRITON_AVAILABLE = False
try:
    # Host-side TMA descriptors (Triton 3.4+), for the kernels that load through the TMA unit.
    from triton.tools.tensor_descriptor import TensorDescriptor as _TensorDescriptor
except ImportError:  # pragma: no cover
    _TensorDescriptor = None

#: Width the n^2 + 2n mapping outputs are padded to (n = 4 gives 24).
NPAD = 32


def _spec_key(args) -> tuple:
    """What Triton specializes a kernel on, for the runtime arguments of one launch.

    Pointers specialize on dtype and 16-byte alignment; integers on being 1, divisible by 16,
    and fitting in 32 bits; floats not at all.
    """
    key = []
    for arg in args:
        if isinstance(arg, Tensor):
            key.append((arg.dtype, arg.data_ptr() % 16 == 0))
        elif isinstance(arg, bool) or arg is None:
            key.append(arg)
        elif isinstance(arg, int):
            key.append((arg == 1, arg % 16 == 0, -(2**31) <= arg < 2**31))
        elif isinstance(arg, float):
            key.append(float)
        elif _TensorDescriptor is not None and isinstance(arg, _TensorDescriptor):
            # Conservative: everything the descriptor encodes except the base address.
            key.append(
                (
                    _TensorDescriptor,
                    arg.base.dtype,
                    arg.base.data_ptr() % 16 == 0,
                    tuple(arg.shape),
                    tuple(arg.strides),
                    tuple(arg.block_shape),
                    getattr(arg, "padding", None),
                )
            )
        else:
            key.append(arg)
    return tuple(key)


class _LeanLaunch:
    """Launch one ``@triton.jit`` kernel through its cached ``CompiledKernel``.

    ``JITFunction.run`` binds, specializes and hashes every argument on every call. Here the first
    launch per specialization goes through Triton (compile and cache); later ones key on the same
    properties Triton specializes on (``_spec_key``) and call the compiled launcher directly.
    ``args`` are the runtime arguments and ``constexprs`` the constexpr values, each in signature
    order (``fn`` may be an ``Autotuner``; its JIT function is used with explicit configs). While
    a Triton launch hook is installed (e.g. a profiler), every launch takes Triton's own path.
    """

    __slots__ = ("fn", "cache", "slots")

    def __init__(self, fn):
        # Autotuner.fn is the JITFunction. Its own .fn is the Python implementation.
        self.fn = fn.fn if hasattr(fn, "configs") else fn
        self.cache = {}
        flags = [p.is_constexpr for p in self.fn.params]
        n_runtime = flags.count(False)
        # None when the constexprs trail the runtime arguments (plain concatenation); else the
        # (is_constexpr, index) of each parameter for merging.
        self.slots = None
        if flags != [False] * n_runtime + [True] * (len(flags) - n_runtime):
            counters = [0, 0]
            self.slots = []
            for is_cx in flags:
                self.slots.append((is_cx, counters[is_cx]))
                counters[is_cx] += 1

    def _ordered(self, args, constexprs):
        if self.slots is None:
            return (*args, *constexprs)
        return tuple(constexprs[i] if is_cx else args[i] for is_cx, i in self.slots)

    def __call__(self, grid, args, constexprs, num_warps, num_stages=3):
        key = (_spec_key(args), constexprs, num_warps, num_stages)
        kernel = self.cache.get(key)
        # Triton 3.4+ keeps the launch hooks as (normally empty) HookChain objects, older ones None.
        hook = _triton_knobs.runtime.launch_enter_hook
        if kernel is None or (hook is not None and getattr(hook, "calls", True)):
            compiled = self.fn[grid](
                *self._ordered(args, constexprs), num_warps=num_warps, num_stages=num_stages
            )
            if kernel is None and not torch.cuda.is_current_stream_capturing():
                self.cache[key] = compiled
            return
        kernel.run(
            grid[0],
            grid[1],
            grid[2],
            torch._C._cuda_getCurrentRawStream(
                (args[0] if isinstance(args[0], Tensor) else args[0].base).device.index
            ),
            kernel.function,
            kernel.packed_metadata,
            None,
            None,
            None,
            *self._ordered(args, constexprs),
        )


if _TRITON_AVAILABLE:
    _LOG2E = tl.constexpr(1.4426950408889634)

    @triton.jit
    def _sp_agg_sq_kernel(
        x_ptr,
        pre_ptr,
        agg_ptr,
        sq_ptr,
        T,
        C: tl.constexpr,
        N: tl.constexpr,
        BS: tl.constexpr,
        BC: tl.constexpr,
        HAS_PRE: tl.constexpr,
        SQ: tl.constexpr,
    ):
        """Grid (row blocks, column blocks): aggregate + per-column-block sum(x^2) partials."""
        pid_s = tl.program_id(0)
        pid_c = tl.program_id(1)
        offs_s = pid_s * BS + tl.arange(0, BS)
        offs_c = pid_c * BC + tl.arange(0, BC)
        mask_s = offs_s < T
        rows = offs_s
        acc = tl.zeros((BS, BC), dtype=tl.float32)
        sq = tl.zeros((BS,), dtype=tl.float32)
        for j in tl.static_range(N):
            xj = tl.load(
                x_ptr + rows[:, None] * (N * C) + j * C + offs_c[None, :],
                mask=mask_s[:, None],
                other=0.0,
            ).to(tl.float32)
            if SQ:
                sq += tl.sum(xj * xj, axis=1)
            if HAS_PRE:
                # The Triton aggregate's chain acc += h_j * x_j (FP32, j = 0..N-1), which compiles
                # to fused multiply-adds; spelled out so the bits do not depend on the launch shape.
                hj = tl.load(pre_ptr + rows * N + j, mask=mask_s, other=0.0).to(tl.float32)
                if j == 0:
                    acc = hj[:, None] * xj
                else:
                    acc = tl.fma(hj[:, None], xj, acc)
            elif j == 0:
                acc = xj
        tl.store(
            agg_ptr + rows[:, None] * C + offs_c[None, :],
            acc.to(agg_ptr.dtype.element_ty),
            mask=mask_s[:, None],
        )
        if SQ:
            tl.store(sq_ptr + pid_c * T + rows, sq, mask=mask_s)

    @triton.jit
    def _sp_proj_kernel(
        x_ptr,
        w_ptr,
        part_ptr,
        T,
        K: tl.constexpr,
        NP: tl.constexpr,
        NOUT: tl.constexpr,
        BM: tl.constexpr,
        BK: tl.constexpr,
        SPLIT: tl.constexpr,
    ):
        """Grid (row blocks, K splits): split-K x @ (W_hi + W_lo)^T partials [SPLIT, T, NP]."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        mask_m = offs_m < T
        rows = offs_m
        offs_o = tl.arange(0, NP)
        mask_o = offs_o < NOUT
        acc = tl.zeros((BM, NP), dtype=tl.float32)
        KS: tl.constexpr = K // SPLIT
        for k0 in range(pid_k * KS, pid_k * KS + KS, BK):
            offs_k = k0 + tl.arange(0, BK)
            xt = tl.load(
                x_ptr + rows[:, None] * K + offs_k[None, :], mask=mask_m[:, None], other=0.0
            )
            wf = tl.load(
                w_ptr + offs_o[None, :] * K + offs_k[:, None], mask=mask_o[None, :], other=0.0
            )
            wh = wf.to(tl.bfloat16)
            wl = (wf - wh.to(tl.float32)).to(tl.bfloat16)
            acc = tl.dot(xt, wh, acc)
            acc = tl.dot(xt, wl, acc)
        tl.store(
            part_ptr + (pid_k * T + rows)[:, None] * NP + offs_o[None, :], acc, mask=mask_m[:, None]
        )

    @triton.jit
    def _sp_pre_fin_kernel(
        part_ptr,
        sq_ptr,
        alpha_pre_ptr,
        alpha_post_ptr,
        alpha_res_ptr,
        bias_ptr,
        pre_ptr,
        post_ptr,
        hres_ptr,
        proj_ptr,
        inv_ptr,
        T,
        eps,
        h_eps,
        sk_eps,
        K: tl.constexpr,
        N: tl.constexpr,
        NOUT: tl.constexpr,
        NP: tl.constexpr,
        S: tl.constexpr,
        SQ_PARTS: tl.constexpr,
        ITERS: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """Grid (token blocks,): reduce partials, RMS, activations and Sinkhorn."""
        pid = tl.program_id(0)
        offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        rows = offs_t
        offs_o = tl.arange(0, NP)
        proj = tl.zeros((BLOCK_T, NP), dtype=tl.float32)
        for s in tl.static_range(S):
            proj += tl.load(
                part_ptr + (s * T + rows)[:, None] * NP + offs_o[None, :],
                mask=mask_t[:, None],
                other=0.0,
            )
        ss = tl.zeros((BLOCK_T,), dtype=tl.float32)
        for s in range(SQ_PARTS):
            ss += tl.load(sq_ptr + s * T + rows, mask=mask_t, other=0.0)
        inv = 1.0 / tl.sqrt(ss / K + eps)
        a_pre = tl.load(alpha_pre_ptr).to(tl.float32)
        a_post = tl.load(alpha_post_ptr).to(tl.float32)
        a_res = tl.load(alpha_res_ptr).to(tl.float32)
        alpha = tl.where(offs_o < N, a_pre, tl.where(offs_o < 2 * N, a_post, a_res))
        bias = tl.load(bias_ptr + offs_o, mask=offs_o < NOUT, other=0.0).to(tl.float32)
        z = (proj * inv[:, None]) * alpha[None, :] + bias[None, :]
        sig = tl.sigmoid(z)
        is_pre = offs_o < N
        is_post = (offs_o >= N) & (offs_o < 2 * N)
        tl.store(
            pre_ptr + rows[:, None] * N + offs_o[None, :],
            sig + h_eps,
            mask=mask_t[:, None] & is_pre[None, :],
        )
        tl.store(
            post_ptr + rows[:, None] * N + (offs_o[None, :] - N),
            2.0 * sig,
            mask=mask_t[:, None] & is_post[None, :],
        )
        tl.store(proj_ptr + rows[:, None] * NP + offs_o[None, :], proj, mask=mask_t[:, None])
        tl.store(inv_ptr + rows, inv, mask=mask_t)

        # Residual logits as [BLOCK_T, N, N] tiles: the same partial sums, re-read in 3-D.
        offs_i = tl.arange(0, N)
        o3 = 2 * N + offs_i[:, None] * N + offs_i[None, :]
        pr = tl.zeros((BLOCK_T, N, N), dtype=tl.float32)
        for s in tl.static_range(S):
            pr += tl.load(
                part_ptr + (s * T + rows)[:, None, None] * NP + o3[None, :, :],
                mask=mask_t[:, None, None],
                other=0.0,
            )
        b3 = tl.load(bias_ptr + o3).to(tl.float32)
        logits = (pr * inv[:, None, None]) * a_res + b3[None, :, :]
        # Sinkhorn: softmax over the last axis + eps, then alternate column / row scaling.
        row_max = tl.max(logits, axis=2)
        m = tl.exp2((logits - row_max[:, :, None]) * _LOG2E)
        for it in range(ITERS):
            row_sum = tl.sum(m, axis=2)
            first = it == 0
            m = m / (row_sum[:, :, None] + tl.where(first, 0.0, sk_eps)) + tl.where(
                first, sk_eps, 0.0
            )
            col_sum = tl.sum(m, axis=1)
            m = m / (col_sum[:, None, :] + sk_eps)
        tl.store(
            hres_ptr + rows[:, None, None] * (N * N) + o3[None, :, :] - 2 * N,
            m,
            mask=mask_t[:, None, None],
        )

    @triton.jit
    def _sp_bda_fwd_kernel(
        hr_ptr,
        x_ptr,
        hp_ptr,
        y_ptr,
        out_ptr,
        pn_ptr,
        nagg_ptr,
        nsq_ptr,
        T,
        C: tl.constexpr,
        N: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_C: tl.constexpr,
        NEXT_AGG: tl.constexpr,
    ):
        """out_i = h_post_i * y + sum_j h_res[j, i] * x_j as an explicit fused multiply-add chain.

        Explicit, so the forward, the checkpoint replay and the NEXT_AGG variant round the same
        way whatever the compiler would contract. (The default Triton kernel mixes contracted and
        separate operations depending on its launch shape, so the last bit can differ from it.)
        """
        pid_s = tl.program_id(0)
        pid_c = tl.program_id(1)
        offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_s = offs_s < T
        rows = offs_s
        y = tl.load(y_ptr + rows[:, None] * C + offs_c[None, :], mask=mask_s[:, None], other=0.0)
        y = y.to(tl.float32)
        xb = x_ptr + rows[:, None] * (N * C) + offs_c[None, :]
        if N == 4:
            # Each stream's tile loaded once (the stores below keep the compiler from reusing
            # loads across output streams): 34 -> 31 us on VR200, same operations and bits.
            x0 = tl.load(xb, mask=mask_s[:, None], other=0.0).to(tl.float32)
            x1 = tl.load(xb + C, mask=mask_s[:, None], other=0.0).to(tl.float32)
            x2 = tl.load(xb + 2 * C, mask=mask_s[:, None], other=0.0).to(tl.float32)
            x3 = tl.load(xb + 3 * C, mask=mask_s[:, None], other=0.0).to(tl.float32)
        for i in tl.static_range(N):
            hp_i = tl.load(hp_ptr + rows * N + i, mask=mask_s, other=0.0).to(tl.float32)
            out_i = hp_i[:, None] * y
            for j in tl.static_range(N):
                hr_ji = tl.load(hr_ptr + rows * (N * N) + j * N + i, mask=mask_s, other=0.0)
                if N == 4:
                    x_j = x0 if j == 0 else (x1 if j == 1 else (x2 if j == 2 else x3))
                else:
                    x_j = tl.load(xb + j * C, mask=mask_s[:, None], other=0.0).to(tl.float32)
                out_i = tl.fma(hr_ji.to(tl.float32)[:, None], x_j, out_i)
            out_i = out_i.to(out_ptr.dtype.element_ty)
            tl.store(
                out_ptr + rows[:, None] * (N * C) + i * C + offs_c[None, :],
                out_i,
                mask=mask_s[:, None],
            )
            if NEXT_AGG:
                # The next sublayer's aggregate and sum(x^2) from the stored (rounded) output:
                # the _sp_agg_sq_kernel chain on the same values, so the same bits.
                o = out_i.to(tl.float32)
                pn_i = tl.load(pn_ptr + rows * N + i, mask=mask_s, other=0.0).to(tl.float32)
                if i == 0:
                    nagg = pn_i[:, None] * o
                    nsq = tl.sum(o * o, axis=1)
                else:
                    nagg = tl.fma(pn_i[:, None], o, nagg)
                    nsq += tl.sum(o * o, axis=1)
        if NEXT_AGG:
            tl.store(
                nagg_ptr + rows[:, None] * C + offs_c[None, :],
                nagg.to(nagg_ptr.dtype.element_ty),
                mask=mask_s[:, None],
            )
            tl.store(nsq_ptr + pid_c * T + rows, nsq, mask=mask_s)

    @triton.jit
    def _sp_rowsum_into(acc, val, slot: tl.constexpr, NN: tl.constexpr):
        return acc + tl.where(tl.arange(0, NN)[None, :] == slot, val[:, None], 0.0)

    @triton.jit
    def _sp_bda_bwd4_kernel(
        go_ptr,
        hr_ptr,
        x_ptr,
        hp_ptr,
        y_ptr,
        gres_ptr,
        gy_ptr,
        ghp_ptr,
        ghr_ptr,
        T,
        C: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """BDA backward for n = 4 reading grad_out and the residual once.

        The same values as fused_mhc_kernels' two Triton BDA backward kernels: g_y = sum_j
        h_post_j go_j and g_res_i = sum_j h_res[i, j] go_j as their fused multiply-add chains, and
        the row sums g_h_post_i = sum_c go_i y, g_h_res[j, i] = sum_c go_i x_j accumulated per
        BLOCK_C tile in their order.
        """
        N: tl.constexpr = 4
        pid = tl.program_id(0)
        offs_s = pid * BLOCK_S + tl.arange(0, BLOCK_S)
        mask_s = offs_s < T
        m2 = mask_s[:, None]
        hrow = offs_s * (N * N)
        h00 = tl.load(hr_ptr + hrow + 0, mask=mask_s, other=0.0)
        h01 = tl.load(hr_ptr + hrow + 1, mask=mask_s, other=0.0)
        h02 = tl.load(hr_ptr + hrow + 2, mask=mask_s, other=0.0)
        h03 = tl.load(hr_ptr + hrow + 3, mask=mask_s, other=0.0)
        h10 = tl.load(hr_ptr + hrow + 4, mask=mask_s, other=0.0)
        h11 = tl.load(hr_ptr + hrow + 5, mask=mask_s, other=0.0)
        h12 = tl.load(hr_ptr + hrow + 6, mask=mask_s, other=0.0)
        h13 = tl.load(hr_ptr + hrow + 7, mask=mask_s, other=0.0)
        h20 = tl.load(hr_ptr + hrow + 8, mask=mask_s, other=0.0)
        h21 = tl.load(hr_ptr + hrow + 9, mask=mask_s, other=0.0)
        h22 = tl.load(hr_ptr + hrow + 10, mask=mask_s, other=0.0)
        h23 = tl.load(hr_ptr + hrow + 11, mask=mask_s, other=0.0)
        h30 = tl.load(hr_ptr + hrow + 12, mask=mask_s, other=0.0)
        h31 = tl.load(hr_ptr + hrow + 13, mask=mask_s, other=0.0)
        h32 = tl.load(hr_ptr + hrow + 14, mask=mask_s, other=0.0)
        h33 = tl.load(hr_ptr + hrow + 15, mask=mask_s, other=0.0)
        p0 = tl.load(hp_ptr + offs_s * N + 0, mask=mask_s, other=0.0)
        p1 = tl.load(hp_ptr + offs_s * N + 1, mask=mask_s, other=0.0)
        p2 = tl.load(hp_ptr + offs_s * N + 2, mask=mask_s, other=0.0)
        p3 = tl.load(hp_ptr + offs_s * N + 3, mask=mask_s, other=0.0)
        ghp = tl.zeros((BLOCK_S, N), dtype=tl.float32)
        ghr = tl.zeros((BLOCK_S, N * N), dtype=tl.float32)
        for c0 in range(0, C, BLOCK_C):
            offs_c = c0 + tl.arange(0, BLOCK_C)
            base = offs_s[:, None] * (N * C) + offs_c[None, :]
            ybase = offs_s[:, None] * C + offs_c[None, :]
            g0 = tl.load(go_ptr + base, mask=m2, other=0.0).to(tl.float32)
            g1 = tl.load(go_ptr + base + C, mask=m2, other=0.0).to(tl.float32)
            g2 = tl.load(go_ptr + base + 2 * C, mask=m2, other=0.0).to(tl.float32)
            g3 = tl.load(go_ptr + base + 3 * C, mask=m2, other=0.0).to(tl.float32)
            yy = tl.load(y_ptr + ybase, mask=m2, other=0.0).to(tl.float32)
            gy = p0[:, None] * g0
            gy = tl.fma(p1[:, None], g1, gy)
            gy = tl.fma(p2[:, None], g2, gy)
            gy = tl.fma(p3[:, None], g3, gy)
            tl.store(gy_ptr + ybase, gy.to(gy_ptr.dtype.element_ty), mask=m2)
            r0 = h00[:, None] * g0
            r0 = tl.fma(h01[:, None], g1, r0)
            r0 = tl.fma(h02[:, None], g2, r0)
            r0 = tl.fma(h03[:, None], g3, r0)
            r1 = h10[:, None] * g0
            r1 = tl.fma(h11[:, None], g1, r1)
            r1 = tl.fma(h12[:, None], g2, r1)
            r1 = tl.fma(h13[:, None], g3, r1)
            r2 = h20[:, None] * g0
            r2 = tl.fma(h21[:, None], g1, r2)
            r2 = tl.fma(h22[:, None], g2, r2)
            r2 = tl.fma(h23[:, None], g3, r2)
            r3 = h30[:, None] * g0
            r3 = tl.fma(h31[:, None], g1, r3)
            r3 = tl.fma(h32[:, None], g2, r3)
            r3 = tl.fma(h33[:, None], g3, r3)
            tl.store(gres_ptr + base, r0.to(gres_ptr.dtype.element_ty), mask=m2)
            tl.store(gres_ptr + base + C, r1.to(gres_ptr.dtype.element_ty), mask=m2)
            tl.store(gres_ptr + base + 2 * C, r2.to(gres_ptr.dtype.element_ty), mask=m2)
            tl.store(gres_ptr + base + 3 * C, r3.to(gres_ptr.dtype.element_ty), mask=m2)
            ghp = _sp_rowsum_into(ghp, tl.sum(g0 * yy, axis=1), 0, N)
            ghp = _sp_rowsum_into(ghp, tl.sum(g1 * yy, axis=1), 1, N)
            ghp = _sp_rowsum_into(ghp, tl.sum(g2 * yy, axis=1), 2, N)
            ghp = _sp_rowsum_into(ghp, tl.sum(g3 * yy, axis=1), 3, N)
            x0 = tl.load(x_ptr + base, mask=m2, other=0.0).to(tl.float32)
            x1 = tl.load(x_ptr + base + C, mask=m2, other=0.0).to(tl.float32)
            x2 = tl.load(x_ptr + base + 2 * C, mask=m2, other=0.0).to(tl.float32)
            x3 = tl.load(x_ptr + base + 3 * C, mask=m2, other=0.0).to(tl.float32)
            ghr = _sp_rowsum_into(ghr, tl.sum(g0 * x0, axis=1), 0, N * N)
            ghr = _sp_rowsum_into(ghr, tl.sum(g1 * x0, axis=1), 1, N * N)
            ghr = _sp_rowsum_into(ghr, tl.sum(g2 * x0, axis=1), 2, N * N)
            ghr = _sp_rowsum_into(ghr, tl.sum(g3 * x0, axis=1), 3, N * N)
            ghr = _sp_rowsum_into(ghr, tl.sum(g0 * x1, axis=1), 4, N * N)
            ghr = _sp_rowsum_into(ghr, tl.sum(g1 * x1, axis=1), 5, N * N)
            ghr = _sp_rowsum_into(ghr, tl.sum(g2 * x1, axis=1), 6, N * N)
            ghr = _sp_rowsum_into(ghr, tl.sum(g3 * x1, axis=1), 7, N * N)
            ghr = _sp_rowsum_into(ghr, tl.sum(g0 * x2, axis=1), 8, N * N)
            ghr = _sp_rowsum_into(ghr, tl.sum(g1 * x2, axis=1), 9, N * N)
            ghr = _sp_rowsum_into(ghr, tl.sum(g2 * x2, axis=1), 10, N * N)
            ghr = _sp_rowsum_into(ghr, tl.sum(g3 * x2, axis=1), 11, N * N)
            ghr = _sp_rowsum_into(ghr, tl.sum(g0 * x3, axis=1), 12, N * N)
            ghr = _sp_rowsum_into(ghr, tl.sum(g1 * x3, axis=1), 13, N * N)
            ghr = _sp_rowsum_into(ghr, tl.sum(g2 * x3, axis=1), 14, N * N)
            ghr = _sp_rowsum_into(ghr, tl.sum(g3 * x3, axis=1), 15, N * N)
        tl.store(ghp_ptr + offs_s[:, None] * N + tl.arange(0, N)[None, :], ghp, mask=m2)
        tl.store(ghr_ptr + offs_s[:, None] * (N * N) + tl.arange(0, N * N)[None, :], ghr, mask=m2)

    @triton.jit
    def _sp_pre_bwd_fin_kernel(
        gpre_ptr,
        gpost_ptr,
        ghres_ptr,
        proj_ptr,
        inv_ptr,
        alpha_pre_ptr,
        alpha_post_ptr,
        alpha_res_ptr,
        bias_ptr,
        w_ptr,
        ahi_ptr,
        alo_ptr,
        b_ptr,
        gab_ptr,
        whi_ptr,
        wlo_ptr,
        ws_m_ptr,
        ws_rs_ptr,
        ws_cs_ptr,
        T,
        P,
        sk_eps,
        K: tl.constexpr,
        N: tl.constexpr,
        NOUT: tl.constexpr,
        NP: tl.constexpr,
        ITERS: tl.constexpr,
        BLOCK_T: tl.constexpr,
        WB: tl.constexpr,
        HAS_GPRE: tl.constexpr,
        HAS_GPOST: tl.constexpr,
        HAS_GHRES: tl.constexpr,
    ):
        """Grid (P token blocks + W-split blocks,): coefficient + Sinkhorn backward.

        Token blocks write a = dL/dproj as BF16 hi + lo [T, NP] (zero-padded), b = -dL/dinv *
        inv^3 / K [T] (dL/dx = a @ W + b * x), and per-block partials [2 * NP] of dL/dbias and of
        dL/dz * u (alpha gradients by segment). The remaining programs split W into BF16 hi + lo
        [NP, K] (zero rows past NOUT) for the x-gradient kernel.
        """
        pid = tl.program_id(0)
        if pid >= P:
            offs = (pid - P) * WB + tl.arange(0, WB)
            mask = offs < NP * K
            wv = tl.load(w_ptr + offs, mask=mask & (offs < NOUT * K), other=0.0)
            wh = wv.to(tl.bfloat16)
            tl.store(whi_ptr + offs, wh, mask=mask)
            tl.store(wlo_ptr + offs, (wv - wh.to(tl.float32)).to(tl.bfloat16), mask=mask)
            return
        offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        rows = offs_t
        offs_o = tl.arange(0, NP)
        is_pre = offs_o < N
        is_post = (offs_o >= N) & (offs_o < 2 * N)
        proj = tl.load(
            proj_ptr + rows[:, None] * NP + offs_o[None, :], mask=mask_t[:, None], other=0.0
        )
        inv = tl.load(inv_ptr + rows, mask=mask_t, other=0.0)
        a_pre = tl.load(alpha_pre_ptr).to(tl.float32)
        a_post = tl.load(alpha_post_ptr).to(tl.float32)
        a_res = tl.load(alpha_res_ptr).to(tl.float32)
        alpha = tl.where(offs_o < N, a_pre, tl.where(offs_o < 2 * N, a_post, a_res))
        bias = tl.load(bias_ptr + offs_o, mask=offs_o < NOUT, other=0.0).to(tl.float32)
        u = proj * inv[:, None]
        sig = tl.sigmoid(u * alpha[None, :] + bias[None, :])
        dsig = sig * (1.0 - sig)
        g_post = tl.zeros((BLOCK_T, NP), dtype=tl.float32)
        post_mask = mask_t[:, None] & is_post[None, :]
        if HAS_GPOST:
            g_post += tl.load(
                gpost_ptr + rows[:, None] * N + (offs_o[None, :] - N), mask=post_mask, other=0.0
            ).to(tl.float32)
        gz = g_post * (2.0 * dsig)
        if HAS_GPRE:
            gp = tl.load(
                gpre_ptr + rows[:, None] * N + offs_o[None, :],
                mask=mask_t[:, None] & is_pre[None, :],
                other=0.0,
            )
            gz += gp.to(tl.float32) * dsig
        gz = tl.where(mask_t[:, None] & (offs_o[None, :] < 2 * N), gz, 0.0)
        g_u = gz * alpha[None, :]
        g_inv = tl.sum(g_u * proj, axis=1)
        a_pp = g_u * inv[:, None]
        a_hi = a_pp.to(tl.bfloat16)
        lo_mask = mask_t[:, None] & (offs_o[None, :] < 2 * N)
        tl.store(ahi_ptr + rows[:, None] * NP + offs_o[None, :], a_hi, mask=lo_mask)
        tl.store(
            alo_ptr + rows[:, None] * NP + offs_o[None, :],
            (a_pp - a_hi.to(tl.float32)).to(tl.bfloat16),
            mask=lo_mask,
        )
        gab_row = gab_ptr + pid * (2 * NP)
        tl.store(gab_row + offs_o, tl.sum(gz, axis=0), mask=offs_o < 2 * N)
        tl.store(gab_row + NP + offs_o, tl.sum(gz * u, axis=0), mask=offs_o < 2 * N)

        offs_i = tl.arange(0, N)
        ij = offs_i[:, None] * N + offs_i[None, :]
        o3 = 2 * N + ij
        g3 = tl.zeros((BLOCK_T, N, N), dtype=tl.float32)
        if HAS_GHRES:
            g3 += tl.load(
                ghres_ptr + rows[:, None, None] * (N * N) + ij[None, :, :],
                mask=mask_t[:, None, None],
                other=0.0,
            ).to(tl.float32)
        pr = tl.load(
            proj_ptr + rows[:, None, None] * NP + o3[None, :, :],
            mask=mask_t[:, None, None],
            other=0.0,
        )
        b3 = tl.load(bias_ptr + o3).to(tl.float32)
        u3 = pr * inv[:, None, None]
        logits = u3 * a_res + b3[None, :, :]
        row_max = tl.max(logits, axis=2)
        m0 = tl.exp2((logits - row_max[:, :, None]) * _LOG2E)
        # Forward states to the workspace, as the Triton Sinkhorn backward does.
        ws_m = ws_m_ptr + rows[:, None, None] * (2 * ITERS * N * N) + ij[None, :, :]
        ws_r = ws_rs_ptr + rows[:, None] * (ITERS * N) + offs_i[None, :]
        ws_c = ws_cs_ptr + rows[:, None] * (ITERS * N) + offs_i[None, :]
        m = m0
        for it in range(ITERS):
            first = it == 0
            tl.store(ws_m + (2 * it) * (N * N), m, mask=mask_t[:, None, None])
            row_sum = tl.sum(m, axis=2)
            tl.store(ws_r + it * N, row_sum, mask=mask_t[:, None])
            m = m / (row_sum[:, :, None] + tl.where(first, 0.0, sk_eps)) + tl.where(
                first, sk_eps, 0.0
            )
            tl.store(ws_m + (2 * it + 1) * (N * N), m, mask=mask_t[:, None, None])
            col_sum = tl.sum(m, axis=1)
            tl.store(ws_c + it * N, col_sum, mask=mask_t[:, None])
            m = m / (col_sum[:, None, :] + sk_eps)
        grad = g3
        for it_rev in range(ITERS):
            it = ITERS - 1 - it_rev
            first = it == 0
            col_s = tl.load(ws_c + it * N, mask=mask_t[:, None], other=1.0)
            grad = grad / (col_s[:, None, :] + sk_eps)
            col_corr = tl.sum(grad * m, axis=1)
            grad = grad - col_corr[:, None, :]
            m = tl.load(ws_m + (2 * it + 1) * (N * N), mask=mask_t[:, None, None], other=0.0)
            row_s = tl.load(ws_r + it * N, mask=mask_t[:, None], other=1.0)
            grad = grad / (row_s[:, :, None] + tl.where(first, 0.0, sk_eps))
            row_corr = tl.sum(grad * (m - tl.where(first, sk_eps, 0.0)), axis=2)
            grad = grad - row_corr[:, :, None]
            m = tl.load(ws_m + (2 * it) * (N * N), mask=mask_t[:, None, None], other=0.0)
        gz3 = tl.where(mask_t[:, None, None], grad * m0, 0.0)
        g_u3 = gz3 * a_res
        g_inv += tl.sum(tl.sum(g_u3 * pr, axis=2), axis=1)
        a3 = g_u3 * inv[:, None, None]
        a3_hi = a3.to(tl.bfloat16)
        a3_ptr = rows[:, None, None] * NP + o3[None, :, :]
        tl.store(ahi_ptr + a3_ptr, a3_hi, mask=mask_t[:, None, None])
        tl.store(
            alo_ptr + a3_ptr,
            (a3 - a3_hi.to(tl.float32)).to(tl.bfloat16),
            mask=mask_t[:, None, None],
        )
        tl.store(gab_row + o3, tl.sum(gz3, axis=0))
        tl.store(gab_row + NP + o3, tl.sum(gz3 * u3, axis=0))
        # Padding columns of a stay zero so the dot products ignore them.
        pad = mask_t[:, None] & (offs_o[None, :] >= NOUT)
        zero = tl.zeros((BLOCK_T, NP), dtype=tl.bfloat16)
        tl.store(ahi_ptr + rows[:, None] * NP + offs_o[None, :], zero, mask=pad)
        tl.store(alo_ptr + rows[:, None] * NP + offs_o[None, :], zero, mask=pad)
        tl.store(b_ptr + rows, -g_inv * (inv * inv * inv) / K, mask=mask_t)

    @triton.jit
    def _sp_pre_bwd_main_kernel(
        x_ptr,
        ahi_ptr,
        alo_ptr,
        b_ptr,
        whi_ptr,
        wlo_ptr,
        gagg_ptr,
        pre_ptr,
        gres_ptr,
        gx_ptr,
        gpm_part_ptr,
        T,
        TCH,
        C: tl.constexpr,
        N: tl.constexpr,
        NP: tl.constexpr,
        BT: tl.constexpr,
        BC: tl.constexpr,
        HAS_PRE: tl.constexpr,
        HAS_GAGG: tl.constexpr,
        HAS_GRES: tl.constexpr,
        NEED_GPM: tl.constexpr,
        EVEN_T: tl.constexpr,
    ):
        """Grid (column blocks over n C, token chunks): x's gradient and pre-mix partials.

        gx = round(round(g_res + round(g_agg * pre_j)) + round(a @ W + b * x)): the default exact
        three-way sum of the (BDA, aggregate, mapping) gradients, in that order. W stays in
        registers for the whole token chunk.
        """
        pid_k = tl.program_id(0)
        pid_t = tl.program_id(1)
        K: tl.constexpr = N * C
        CBLK: tl.constexpr = C // BC
        j = pid_k // CBLK
        offs_k = pid_k * BC + tl.arange(0, BC)
        offs_c = (pid_k % CBLK) * BC + tl.arange(0, BC)
        offs_o = tl.arange(0, NP)
        # Tiles are [BC, BT] (columns x tokens): the projection term is W^T [BC, NP] @ a^T [NP, BT],
        # which puts the columns on the MMA M axis (faster than tokens there on GB300).
        wht = tl.load(whi_ptr + offs_o[None, :] * K + offs_k[:, None])
        wlt = tl.load(wlo_ptr + offs_o[None, :] * K + offs_k[:, None])
        t_begin = pid_t * TCH
        t_end = tl.minimum(t_begin + TCH, T)
        for t0 in range(t_begin, t_end, BT):
            offs_t = t0 + tl.arange(0, BT)
            if EVEN_T:
                # TCH divides T and BT divides TCH: no masks (they slow this layout down).
                mask_t = None
                m2 = None
                oth = None
            else:
                mask_t = offs_t < t_end
                m2 = mask_t[None, :]
                oth = 0.0
            xk = offs_t[None, :] * K + offs_k[:, None]
            xt = tl.load(x_ptr + xk, mask=m2, other=oth).to(tl.float32)
            bb = tl.load(b_ptr + offs_t, mask=mask_t, other=oth)
            aht = tl.load(ahi_ptr + offs_t[None, :] * NP + offs_o[:, None], mask=m2, other=oth)
            alt = tl.load(alo_ptr + offs_t[None, :] * NP + offs_o[:, None], mask=m2, other=oth)
            g = tl.dot(wht, aht)
            g = tl.dot(wlt, aht, g)
            g = tl.dot(wht, alt, g)
            out = (g + bb[None, :] * xt).to(gx_ptr.dtype.element_ty)
            if HAS_GAGG:
                ga = tl.load(
                    gagg_ptr + offs_t[None, :] * C + offs_c[:, None], mask=m2, other=oth
                ).to(tl.float32)
                if HAS_PRE:
                    pj = tl.load(pre_ptr + offs_t * N + j, mask=mask_t, other=oth).to(tl.float32)
                    term = (ga * pj[None, :]).to(gx_ptr.dtype.element_ty)
                else:
                    term = tl.where(j == 0, ga, 0.0).to(gx_ptr.dtype.element_ty)
                if NEED_GPM:
                    tl.store(
                        gpm_part_ptr + pid_k * T + offs_t, tl.sum(ga * xt, axis=0), mask=mask_t
                    )
                if HAS_GRES:
                    r = tl.load(gres_ptr + xk, mask=m2, other=oth).to(tl.float32)
                    term = (r + term.to(tl.float32)).to(gx_ptr.dtype.element_ty)
                out = (term.to(tl.float32) + out.to(tl.float32)).to(gx_ptr.dtype.element_ty)
            elif HAS_GRES:
                r = tl.load(gres_ptr + xk, mask=m2, other=oth).to(tl.float32)
                out = (r + out.to(tl.float32)).to(gx_ptr.dtype.element_ty)
            tl.store(gx_ptr + xk, out, mask=m2)

    @triton.jit
    def _sp_gw_kernel(
        x_ptr,
        ahi_ptr,
        alo_ptr,
        gw_ptr,
        T,
        TCH,
        K: tl.constexpr,
        NP: tl.constexpr,
        NOUT: tl.constexpr,
        BK: tl.constexpr,
        BT: tl.constexpr,
    ):
        """Write the split-K weight-gradient partials for each token chunk."""
        pid_k = tl.program_id(0)
        pid_t = tl.program_id(1)
        offs_k = pid_k * BK + tl.arange(0, BK)
        offs_o = tl.arange(0, NP)
        acc = tl.zeros((NP, BK), dtype=tl.float32)
        t_begin = pid_t * TCH
        t_end = tl.minimum(t_begin + TCH, T)
        for t0 in range(t_begin, t_end, BT):
            offs_t = t0 + tl.arange(0, BT)
            mask_t = offs_t < t_end
            rows = offs_t
            xt = tl.load(
                x_ptr + rows[:, None] * K + offs_k[None, :], mask=mask_t[:, None], other=0.0
            )
            aht = tl.load(
                ahi_ptr + rows[None, :] * NP + offs_o[:, None], mask=mask_t[None, :], other=0.0
            )
            alt = tl.load(
                alo_ptr + rows[None, :] * NP + offs_o[:, None], mask=mask_t[None, :], other=0.0
            )
            acc = tl.dot(aht, xt, acc)
            acc = tl.dot(alt, xt, acc)
        tl.store(
            gw_ptr + pid_t * (NOUT * K) + offs_o[:, None] * K + offs_k[None, :],
            acc,
            mask=(offs_o < NOUT)[:, None],
        )

    @triton.jit
    def _sp_pre_bwd_reduce_kernel(
        gpm_part_ptr,
        gpm_ptr,
        gab_ptr,
        gab_out_ptr,
        gw_part_ptr,
        gw_ptr,
        T,
        P,
        GPM_PROGRAMS,
        NK,
        N: tl.constexpr,
        NOUT: tl.constexpr,
        NP: tl.constexpr,
        CBLK: tl.constexpr,
        GW_SPLITS: tl.constexpr,
        NEED_GPM: tl.constexpr,
        BLOCK: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        """1-D grid: pre-mix gradient rows, then one alpha/bias program, then W-gradient blocks."""
        pid = tl.program_id(0)
        if pid < GPM_PROGRAMS:
            if NEED_GPM:
                # Program = (stream, token block): coalesced loads along tokens.
                blocks = tl.cdiv(T, BLOCK)
                jj = pid // blocks
                t = (pid % blocks) * BLOCK + tl.arange(0, BLOCK)
                mask = t < T
                acc = tl.zeros((BLOCK,), dtype=tl.float32)
                for cb in range(CBLK):
                    acc += tl.load(gpm_part_ptr + (jj * CBLK + cb) * T + t, mask=mask, other=0.0)
                tl.store(gpm_ptr + t * N + jj, acc, mask=mask)
        elif pid == GPM_PROGRAMS:
            # One [BLOCK_P, NP] tile per step instead of a serial walk over the P partial rows.
            offs_o = tl.arange(0, NP)
            gb = tl.zeros((NP,), dtype=tl.float32)
            ga = tl.zeros((NP,), dtype=tl.float32)
            for p0 in range(0, P, BLOCK_P):
                offs_p = p0 + tl.arange(0, BLOCK_P)
                rows_p = gab_ptr + offs_p[:, None] * (2 * NP) + offs_o[None, :]
                mask_p = (offs_p < P)[:, None]
                gb += tl.sum(tl.load(rows_p, mask=mask_p, other=0.0), axis=0)
                ga += tl.sum(tl.load(rows_p + NP, mask=mask_p, other=0.0), axis=0)
            tl.store(gab_out_ptr + 3 + offs_o, gb, mask=offs_o < NOUT)
            tl.store(gab_out_ptr + 0, tl.sum(tl.where(offs_o < N, ga, 0.0), axis=0))
            tl.store(
                gab_out_ptr + 1, tl.sum(tl.where((offs_o >= N) & (offs_o < 2 * N), ga, 0.0), axis=0)
            )
            tl.store(
                gab_out_ptr + 2,
                tl.sum(tl.where((offs_o >= 2 * N) & (offs_o < NOUT), ga, 0.0), axis=0),
            )
        else:
            offs = (pid - GPM_PROGRAMS - 1) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < NK
            acc = tl.zeros((BLOCK,), dtype=tl.float32)
            for s in tl.static_range(GW_SPLITS):
                acc += tl.load(gw_part_ptr + s * NK + offs, mask=mask, other=0.0)
            tl.store(gw_ptr + offs, acc, mask=mask)

    @triton.jit
    def _sp_bda_bwd4t_kernel(
        go_ptr,
        hr_ptr,
        x_ptr,
        hp_ptr,
        y_ptr,
        gres_ptr,
        gy_ptr,
        part_ptr,
        T,
        C: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_C: tl.constexpr,
        EVEN: tl.constexpr,
    ):
        """Grid (row blocks, column tiles): the n = 4 BDA backward, one tile per program.

        All nine loads are issued first. g_y and g_res as in ``_sp_bda_bwd4_kernel``; the 20
        row sums of the tile go to ``part`` [tiles, 20, T], and ``_sp_bda_bwd_rowsum_kernel``
        adds them in tile order, the default kernels' accumulation order.
        """
        N: tl.constexpr = 4
        pid = tl.program_id(0)
        tile = tl.program_id(1)
        offs_s = pid * BLOCK_S + tl.arange(0, BLOCK_S)
        if EVEN:
            # BLOCK_S divides T: no masks (they cost ~15% here on VR200).
            mask_s = None
            m2 = None
            oth = None
        else:
            mask_s = offs_s < T
            m2 = mask_s[:, None]
            oth = 0.0
        offs_c = tile * BLOCK_C + tl.arange(0, BLOCK_C)
        base = offs_s[:, None] * (N * C) + offs_c[None, :]
        ybase = offs_s[:, None] * C + offs_c[None, :]
        g0 = tl.load(go_ptr + base, mask=m2, other=oth).to(tl.float32)
        g1 = tl.load(go_ptr + base + C, mask=m2, other=oth).to(tl.float32)
        g2 = tl.load(go_ptr + base + 2 * C, mask=m2, other=oth).to(tl.float32)
        g3 = tl.load(go_ptr + base + 3 * C, mask=m2, other=oth).to(tl.float32)
        yy = tl.load(y_ptr + ybase, mask=m2, other=oth).to(tl.float32)
        x0 = tl.load(x_ptr + base, mask=m2, other=oth).to(tl.float32)
        x1 = tl.load(x_ptr + base + C, mask=m2, other=oth).to(tl.float32)
        x2 = tl.load(x_ptr + base + 2 * C, mask=m2, other=oth).to(tl.float32)
        x3 = tl.load(x_ptr + base + 3 * C, mask=m2, other=oth).to(tl.float32)
        hrow = hr_ptr + offs_s * (N * N)
        p0 = tl.load(hp_ptr + offs_s * N + 0, mask=mask_s, other=oth)
        p1 = tl.load(hp_ptr + offs_s * N + 1, mask=mask_s, other=oth)
        p2 = tl.load(hp_ptr + offs_s * N + 2, mask=mask_s, other=oth)
        p3 = tl.load(hp_ptr + offs_s * N + 3, mask=mask_s, other=oth)
        gy = p0[:, None] * g0
        gy = tl.fma(p1[:, None], g1, gy)
        gy = tl.fma(p2[:, None], g2, gy)
        gy = tl.fma(p3[:, None], g3, gy)
        tl.store(gy_ptr + ybase, gy.to(gy_ptr.dtype.element_ty), mask=m2)
        for i in tl.static_range(N):
            r = tl.load(hrow + i * N + 0, mask=mask_s, other=oth)[:, None] * g0
            r = tl.fma(tl.load(hrow + i * N + 1, mask=mask_s, other=oth)[:, None], g1, r)
            r = tl.fma(tl.load(hrow + i * N + 2, mask=mask_s, other=oth)[:, None], g2, r)
            r = tl.fma(tl.load(hrow + i * N + 3, mask=mask_s, other=oth)[:, None], g3, r)
            tl.store(gres_ptr + base + i * C, r.to(gres_ptr.dtype.element_ty), mask=m2)
        # Slots: g_h_post_i = sum go_i y (0..3), then g_h_res[j, i] = sum go_i x_j (4 + 4 j + i).
        pb = part_ptr + tile * (20 * T) + offs_s
        tl.store(pb + 0 * T, tl.sum(g0 * yy, axis=1), mask=mask_s)
        tl.store(pb + 1 * T, tl.sum(g1 * yy, axis=1), mask=mask_s)
        tl.store(pb + 2 * T, tl.sum(g2 * yy, axis=1), mask=mask_s)
        tl.store(pb + 3 * T, tl.sum(g3 * yy, axis=1), mask=mask_s)
        tl.store(pb + 4 * T, tl.sum(g0 * x0, axis=1), mask=mask_s)
        tl.store(pb + 5 * T, tl.sum(g1 * x0, axis=1), mask=mask_s)
        tl.store(pb + 6 * T, tl.sum(g2 * x0, axis=1), mask=mask_s)
        tl.store(pb + 7 * T, tl.sum(g3 * x0, axis=1), mask=mask_s)
        tl.store(pb + 8 * T, tl.sum(g0 * x1, axis=1), mask=mask_s)
        tl.store(pb + 9 * T, tl.sum(g1 * x1, axis=1), mask=mask_s)
        tl.store(pb + 10 * T, tl.sum(g2 * x1, axis=1), mask=mask_s)
        tl.store(pb + 11 * T, tl.sum(g3 * x1, axis=1), mask=mask_s)
        tl.store(pb + 12 * T, tl.sum(g0 * x2, axis=1), mask=mask_s)
        tl.store(pb + 13 * T, tl.sum(g1 * x2, axis=1), mask=mask_s)
        tl.store(pb + 14 * T, tl.sum(g2 * x2, axis=1), mask=mask_s)
        tl.store(pb + 15 * T, tl.sum(g3 * x2, axis=1), mask=mask_s)
        tl.store(pb + 16 * T, tl.sum(g0 * x3, axis=1), mask=mask_s)
        tl.store(pb + 17 * T, tl.sum(g1 * x3, axis=1), mask=mask_s)
        tl.store(pb + 18 * T, tl.sum(g2 * x3, axis=1), mask=mask_s)
        tl.store(pb + 19 * T, tl.sum(g3 * x3, axis=1), mask=mask_s)

    @triton.jit
    def _sp_bda_bwd_rowsum_kernel(
        part_ptr, ghp_ptr, ghr_ptr, T, NT: tl.constexpr, BLOCK: tl.constexpr
    ):
        """Grid (token blocks, 20 slots): add the per-tile row sums in tile order."""
        pid = tl.program_id(0)
        slot = tl.program_id(1)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < T
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for tile in range(NT):
            acc += tl.load(part_ptr + tile * (20 * T) + slot * T + offs, mask=mask, other=0.0)
        if slot < 4:
            tl.store(ghp_ptr + offs * 4 + slot, acc, mask=mask)
        else:
            tl.store(ghr_ptr + offs * 16 + (slot - 4), acc, mask=mask)

    @triton.jit
    def _sp_pre_bwd_main_pf_kernel(
        x_ptr,
        ahi_ptr,
        alo_ptr,
        b_ptr,
        whi_ptr,
        wlo_ptr,
        gagg_ptr,
        pre_ptr,
        gres_ptr,
        gx_ptr,
        gpm_part_ptr,
        T,
        C: tl.constexpr,
        N: tl.constexpr,
        NP: tl.constexpr,
        BT: tl.constexpr,
        BC: tl.constexpr,
        TCH: tl.constexpr,
        HAS_PRE: tl.constexpr,
        HAS_GAGG: tl.constexpr,
        HAS_GRES: tl.constexpr,
        NEED_GPM: tl.constexpr,
    ):
        """``_sp_pre_bwd_main_kernel`` (TCH divides T, BT divides TCH) with the next token
        block's loads issued before this block's dots, so their latency overlaps the MMAs and
        layout conversions. The same operations, so the same bits."""
        pid_k = tl.program_id(0)
        pid_t = tl.program_id(1)
        K: tl.constexpr = N * C
        CBLK: tl.constexpr = C // BC
        NB: tl.constexpr = TCH // BT
        j = pid_k // CBLK
        offs_k = pid_k * BC + tl.arange(0, BC)
        offs_c = (pid_k % CBLK) * BC + tl.arange(0, BC)
        offs_o = tl.arange(0, NP)
        wht = tl.load(whi_ptr + offs_o[None, :] * K + offs_k[:, None])
        wlt = tl.load(wlo_ptr + offs_o[None, :] * K + offs_k[:, None])
        t_begin = pid_t * TCH
        offs_t = t_begin + tl.arange(0, BT)
        xk = offs_t[None, :] * K + offs_k[:, None]
        xt = tl.load(x_ptr + xk)
        bb = tl.load(b_ptr + offs_t)
        aht = tl.load(ahi_ptr + offs_t[None, :] * NP + offs_o[:, None])
        alt = tl.load(alo_ptr + offs_t[None, :] * NP + offs_o[:, None])
        if HAS_GAGG:
            ga = tl.load(gagg_ptr + offs_t[None, :] * C + offs_c[:, None])
            if HAS_PRE:
                pj = tl.load(pre_ptr + offs_t * N + j)
        if HAS_GRES:
            rr = tl.load(gres_ptr + xk)
        for i in range(NB):
            # Block i + 1 (block 0 again on the last step: in bounds, unused).
            nt = t_begin + ((i + 1) % NB) * BT + tl.arange(0, BT)
            nxk = nt[None, :] * K + offs_k[:, None]
            xt_n = tl.load(x_ptr + nxk)
            bb_n = tl.load(b_ptr + nt)
            aht_n = tl.load(ahi_ptr + nt[None, :] * NP + offs_o[:, None])
            alt_n = tl.load(alo_ptr + nt[None, :] * NP + offs_o[:, None])
            if HAS_GAGG:
                ga_n = tl.load(gagg_ptr + nt[None, :] * C + offs_c[:, None])
                if HAS_PRE:
                    pj_n = tl.load(pre_ptr + nt * N + j)
            if HAS_GRES:
                rr_n = tl.load(gres_ptr + nxk)
            xf = xt.to(tl.float32)
            g = tl.dot(wht, aht)
            g = tl.dot(wlt, aht, g)
            g = tl.dot(wht, alt, g)
            out = (g + bb[None, :] * xf).to(gx_ptr.dtype.element_ty)
            if HAS_GAGG:
                gaf = ga.to(tl.float32)
                if HAS_PRE:
                    term = (gaf * pj.to(tl.float32)[None, :]).to(gx_ptr.dtype.element_ty)
                else:
                    term = tl.where(j == 0, gaf, 0.0).to(gx_ptr.dtype.element_ty)
                if NEED_GPM:
                    tl.store(gpm_part_ptr + pid_k * T + offs_t, tl.sum(gaf * xf, axis=0))
                if HAS_GRES:
                    term = (rr.to(tl.float32) + term.to(tl.float32)).to(gx_ptr.dtype.element_ty)
                out = (term.to(tl.float32) + out.to(tl.float32)).to(gx_ptr.dtype.element_ty)
            elif HAS_GRES:
                out = (rr.to(tl.float32) + out.to(tl.float32)).to(gx_ptr.dtype.element_ty)
            tl.store(gx_ptr + xk, out)
            offs_t = nt
            xk = nxk
            xt = xt_n
            bb = bb_n
            aht = aht_n
            alt = alt_n
            if HAS_GAGG:
                ga = ga_n
                if HAS_PRE:
                    pj = pj_n
            if HAS_GRES:
                rr = rr_n

    @triton.jit
    def _sp_pre_bwd_main_tma_kernel(
        x_desc,
        ahi_desc,
        alo_desc,
        gres_desc,
        gagg_desc,
        gx_desc,
        b_ptr,
        pre_ptr,
        whi_ptr,
        wlo_ptr,
        gpm_part_ptr,
        T,
        C: tl.constexpr,
        N: tl.constexpr,
        NP: tl.constexpr,
        BT: tl.constexpr,
        BC: tl.constexpr,
        TCH: tl.constexpr,
        STAGES: tl.constexpr,
        HAS_PRE: tl.constexpr,
        HAS_GAGG: tl.constexpr,
        HAS_GRES: tl.constexpr,
        NEED_GPM: tl.constexpr,
    ):
        """``_sp_pre_bwd_main_kernel`` in token-major tiles [BT, BC], with x, a, g_res, g_agg and
        g_x moving through the TMA unit and the token loop software-pipelined (STAGES), so loads
        run ahead of the dots and layout conversions. The same operations, so the same bits."""
        pid_k = tl.program_id(0)
        pid_t = tl.program_id(1)
        K: tl.constexpr = N * C
        CBLK: tl.constexpr = C // BC
        j = pid_k // CBLK
        k0 = pid_k * BC
        c0 = (pid_k % CBLK) * BC
        offs_o = tl.arange(0, NP)
        offs_k = k0 + tl.arange(0, BC)
        wh = tl.load(whi_ptr + offs_o[:, None] * K + offs_k[None, :])
        wl = tl.load(wlo_ptr + offs_o[:, None] * K + offs_k[None, :])
        for i in tl.range(0, TCH // BT, num_stages=STAGES):
            t0 = pid_t * TCH + i * BT
            offs_t = t0 + tl.arange(0, BT)
            xf = x_desc.load([t0, k0]).to(tl.float32)
            ah = ahi_desc.load([t0, 0])
            al = alo_desc.load([t0, 0])
            bb = tl.load(b_ptr + offs_t)
            g = tl.dot(ah, wh)
            g = tl.dot(ah, wl, g)
            g = tl.dot(al, wh, g)
            out = (g + bb[:, None] * xf).to(tl.bfloat16)
            if HAS_GAGG:
                gaf = gagg_desc.load([t0, c0]).to(tl.float32)
                if HAS_PRE:
                    pj = tl.load(pre_ptr + offs_t * N + j).to(tl.float32)
                    term = (gaf * pj[:, None]).to(tl.bfloat16)
                else:
                    term = tl.where(j == 0, gaf, 0.0).to(tl.bfloat16)
                if NEED_GPM:
                    tl.store(gpm_part_ptr + pid_k * T + offs_t, tl.sum(gaf * xf, axis=1))
                if HAS_GRES:
                    rr = gres_desc.load([t0, k0]).to(tl.float32)
                    term = (rr + term.to(tl.float32)).to(tl.bfloat16)
                out = (term.to(tl.float32) + out.to(tl.float32)).to(tl.bfloat16)
            elif HAS_GRES:
                rr = gres_desc.load([t0, k0]).to(tl.float32)
                out = (rr + out.to(tl.float32)).to(tl.bfloat16)
            gx_desc.store([t0, k0], out)

    @triton.jit
    def _sp_proj_tma_kernel(
        x_desc,
        w_desc,
        part_ptr,
        T,
        K: tl.constexpr,
        NP: tl.constexpr,
        BM: tl.constexpr,
        BK: tl.constexpr,
        SPLIT: tl.constexpr,
        STAGES: tl.constexpr,
    ):
        """``_sp_proj_kernel`` with x and W ([NP, BK] tiles; rows past NOUT zero-filled by TMA)
        loaded through the TMA unit in a software-pipelined loop."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        m0 = pid_m * BM
        acc = tl.zeros((BM, NP), dtype=tl.float32)
        KS: tl.constexpr = K // SPLIT
        for i in tl.range(0, KS // BK, num_stages=STAGES):
            k0 = pid_k * KS + i * BK
            xt = x_desc.load([m0, k0])
            wf = w_desc.load([0, k0])
            wh = wf.to(tl.bfloat16)
            wl = (wf - wh.to(tl.float32)).to(tl.bfloat16)
            acc = tl.dot(xt, tl.trans(wh), acc)
            acc = tl.dot(xt, tl.trans(wl), acc)
        rows = m0 + tl.arange(0, BM)
        offs_o = tl.arange(0, NP)
        tl.store(
            part_ptr + (pid_k * T + rows)[:, None] * NP + offs_o[None, :],
            acc,
            mask=(rows < T)[:, None],
        )

    @triton.jit
    def _sp_gw_tma_kernel(
        x_desc,
        ahi_desc,
        alo_desc,
        gw_ptr,
        T,
        K: tl.constexpr,
        NP: tl.constexpr,
        NOUT: tl.constexpr,
        BK: tl.constexpr,
        BT: tl.constexpr,
        TCH: tl.constexpr,
        STAGES: tl.constexpr,
    ):
        """``_sp_gw_kernel`` with x and a loaded through the TMA unit (rows past T zero-filled)
        in a software-pipelined loop."""
        pid_k = tl.program_id(0)
        pid_t = tl.program_id(1)
        k0 = pid_k * BK
        acc = tl.zeros((NP, BK), dtype=tl.float32)
        for i in tl.range(0, TCH // BT, num_stages=STAGES):
            t0 = pid_t * TCH + i * BT
            xt = x_desc.load([t0, k0])
            acc = tl.dot(tl.trans(ahi_desc.load([t0, 0])), xt, acc)
            acc = tl.dot(tl.trans(alo_desc.load([t0, 0])), xt, acc)
        offs_o = tl.arange(0, NP)
        offs_k = k0 + tl.arange(0, BK)
        tl.store(
            gw_ptr + pid_t * (NOUT * K) + offs_o[:, None] * K + offs_k[None, :],
            acc,
            mask=(offs_o < NOUT)[:, None],
        )

    _agg_sq = _LeanLaunch(_sp_agg_sq_kernel)
    _proj = _LeanLaunch(_sp_proj_kernel)
    _pre_fin = _LeanLaunch(_sp_pre_fin_kernel)
    _bda_fwd = _LeanLaunch(_sp_bda_fwd_kernel)
    _bda_bwd4 = _LeanLaunch(_sp_bda_bwd4_kernel)
    _pre_bwd_fin = _LeanLaunch(_sp_pre_bwd_fin_kernel)
    _pre_bwd_main = _LeanLaunch(_sp_pre_bwd_main_kernel)
    _gw = _LeanLaunch(_sp_gw_kernel)
    _pre_bwd_reduce = _LeanLaunch(_sp_pre_bwd_reduce_kernel)
    _bda_bwd4t = _LeanLaunch(_sp_bda_bwd4t_kernel)
    _bda_bwd_rowsum = _LeanLaunch(_sp_bda_bwd_rowsum_kernel)
    _pre_bwd_main_pf = _LeanLaunch(_sp_pre_bwd_main_pf_kernel)
    _pre_bwd_main_tma = _LeanLaunch(_sp_pre_bwd_main_tma_kernel)
    _proj_tma = _LeanLaunch(_sp_proj_tma_kernel)
    _gw_tma = _LeanLaunch(_sp_gw_tma_kernel)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _tma_ok(*tensors: Tensor) -> bool:
    """2-D tensors a TMA descriptor can address: 16-byte aligned base and rows, unit stride."""
    if _TensorDescriptor is None:
        return False
    return all(
        t.dim() == 2
        and t.stride(1) == 1
        and t.data_ptr() % 16 == 0
        and (t.stride(0) * t.element_size()) % 16 == 0
        for t in tensors
    )


def _desc(t: Tensor, block):
    return _TensorDescriptor.from_tensor(t, list(block))


# Launch shapes, tuned on GB300 at T 4096, C 5120, n 4 (graph-timed kernel sweeps).
AGG_CFG = dict(bs=2, bc=256, num_warps=2, num_stages=1)
PROJ_CFG = dict(bm=128, bk=64, split=8, num_warps=4, num_stages=4)
PRE_FIN_CFG = dict(block_t=32, num_warps=4, num_stages=1)
BDA_FWD_CFG = dict(block_s=2, block_c=256, num_warps=2, num_stages=3)
# The default BDA backward kernels, at their autotuned GB300 choices (so their bits are reproduced).
BDA_BWD_A_CFG = dict(block_s=1, block_c=512, num_warps=2, num_stages=3)
BDA_BWD_B_CFG = dict(block_s=8, block_c=128, num_warps=4, num_stages=3)
# n = 4: one kernel for all four BDA gradients (BLOCK_C 128 keeps the row-sum blocking of B).
BDA_BWD4_CFG = dict(block_s=4, block_c=128, num_warps=2, num_stages=1)
# n = 4, one 128-column tile per program plus an ordered row-sum pass (same bits as the above);
# None keeps the row-loop kernel (faster on GB300 at the time it was tuned).
BDA_BWD4T_CFG: Optional[dict] = None
_BDA_BWD_LAUNCHERS = None


def _bda_bwd_launchers():
    """Lean launchers of fused_mhc_kernels' two Triton BDA backward kernels (imported lazily)."""
    global _BDA_BWD_LAUNCHERS
    if _BDA_BWD_LAUNCHERS is None:
        from megatron.core.fusions import fused_mhc_kernels as fk

        _BDA_BWD_LAUNCHERS = (
            _LeanLaunch(fk._triton_hpb_bwd_g_x_orig_kernel),
            _LeanLaunch(fk._triton_hpb_bwd_g_hp_hr_kernel),
        )
    return _BDA_BWD_LAUNCHERS


PRE_BWD_FIN_CFG = dict(block_t=32, w_block=4096, num_warps=4, num_stages=1)
PRE_BWD_MAIN_CFG = dict(bt=16, bc=128, tch=128, num_warps=4, num_stages=1)
GW_CFG = dict(bk=128, bt=64, tch=512, num_warps=4, num_stages=3)

# Per-device sets, from graph-timed sweeps at the 64K proxy shapes, applied to the dicts above
# on the first single_pass_fast_supports call on such a device.
# VR200 (compute capability 10.7, 212 SMs): the tiled BDA backward, TMA (tensor-descriptor)
# loads in software-pipelined loops for the projection, x-gradient and W-gradient kernels, and
# other launch shapes: about 21% less mHC kernel time per sublayer than the default shapes on
# VR200 (357 -> 282 us). Given the same inputs, the aggregate and the BDA forward and backward
# give the same bits with any set (checked on VR200). The TMA x-gradient kernel's mapping term
# goes through tensor-core tiles of another shape, so a few elements in 10^7 can differ in the
# last bit from the other kernels (each kernel's three-way sum stays exact); the sum(x^2),
# projection, pre-mix and W-gradient partial sums are blocked differently per set, so those can
# differ in the last bits too.
_DEVICE_CFGS = {
    (10, 7): {
        "AGG_CFG": dict(bs=8, bc=128, num_warps=4, num_stages=1),
        "PROJ_CFG": dict(bm=128, bk=64, split=16, num_warps=4, num_stages=4, tma=True),
        "PRE_FIN_CFG": dict(block_t=8, num_warps=1, num_stages=1),
        # 1024-column blocks measured 0.8 us faster but would need hidden % 1024 == 0.
        "BDA_FWD_CFG": dict(block_s=1, block_c=512, num_warps=2, num_stages=2),
        # TMA when the tensors allow it, else the prefetching kernel.
        "PRE_BWD_MAIN_CFG": dict(
            bt=32, bc=64, tch=256, num_warps=2, num_stages=3, tma=True, prefetch=True
        ),
        "BDA_BWD4T_CFG": dict(block_s=8, num_warps=2),
        "GW_CFG": dict(bk=256, bt=32, tch=1024, num_warps=4, num_stages=4, tma=True),
    },
    # GB300: only the TMA x-gradient kernel (17 us less per mapping backward than the plain one,
    # the prefetching kernel 13); the tiled BDA backward, the TMA projection and W-gradient
    # kernels and the other VR200 shapes measured slower or equal there.
    (10, 3): {
        "PRE_BWD_MAIN_CFG": dict(
            bt=32, bc=64, tch=256, num_warps=2, num_stages=3, tma=True, prefetch=True
        )
    },
}
_DEVICE_CFGS_APPLIED = False


def _apply_device_configs(device: torch.device) -> None:
    """Switch the launch shapes to the device's tuned set, once per process."""
    global _DEVICE_CFGS_APPLIED
    if _DEVICE_CFGS_APPLIED:
        return
    _DEVICE_CFGS_APPLIED = True
    tuned = _DEVICE_CFGS.get(torch.cuda.get_device_capability(device))
    if tuned is not None:
        for name, cfg in tuned.items():
            current = globals()[name]
            if current is None:
                globals()[name] = dict(cfg)
            else:
                current.update(cfg)


def single_pass_fast_supports(x: Tensor, n: int, weight: Tensor) -> bool:
    """Residuals and parameters the fast kernels handle (others keep the default path)."""
    if not (_TRITON_AVAILABLE and x.is_cuda and x.dim() == 3 and x.numel() and x.is_contiguous()):
        return False
    _apply_device_configs(x.device)
    # BF16 only: the BF16 hi + lo split of W makes the products exact for BF16 x.
    if x.dtype != torch.bfloat16 or n & (n - 1) or n * n + 2 * n > NPAD:
        return False
    width = x.shape[-1]
    if width % n or x.numel() >= 2**31 // 2:
        # Kernels use 32-bit offsets (twice the residual's elements covers every buffer).
        return False
    C = width // n
    return (
        C % AGG_CFG["bc"] == 0
        and width % (PROJ_CFG["split"] * PROJ_CFG["bk"]) == 0
        and C % BDA_FWD_CFG["block_c"] == 0
        and C % BDA_BWD_A_CFG["block_c"] == 0
        and C % PRE_BWD_MAIN_CFG["bc"] == 0
        and width % GW_CFG["bk"] == 0
        and weight.dtype == torch.float32
        and weight.is_contiguous()
        and tuple(weight.shape) == (n * n + 2 * n, width)
    )


# The entry predicate guards the Triton launchers defined under _TRITON_AVAILABLE.
# pylint: disable=possibly-used-before-assignment
def _launch_agg(x2d, pre_mix, agg, sq, n, with_sq):
    T, K = x2d.shape
    C = K // n
    cfg = AGG_CFG
    _agg_sq(
        (_cdiv(T, cfg["bs"]), C // cfg["bc"], 1),
        (x2d, pre_mix if pre_mix is not None else x2d, agg, sq, T),
        (C, n, cfg["bs"], cfg["bc"], pre_mix is not None, with_sq),
        cfg["num_warps"],
        cfg["num_stages"],
    )


def sp_pre_forward(
    x2d: Tensor,
    pre_mix: Optional[Tensor],
    weight: Tensor,
    alpha_pre: Tensor,
    alpha_post: Tensor,
    alpha_res: Tensor,
    bias: Tensor,
    n: int,
    norm_eps: float,
    h_eps: float,
    sk_eps: float,
    iters: int,
    precomputed: Optional[Tuple[Tensor, Tensor]] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return (aggregate [T, C], pre [T, n], post [T, n], h_res [T, n, n], proj [T, NP], inv [T]).

    ``precomputed`` = (aggregate, sum(x^2) partials [parts, T]) from the previous sublayer's BDA
    (``NextAggregate``) skips the aggregate kernel.
    """
    T, K = x2d.shape
    C = K // n
    nout = n * n + 2 * n
    dev = x2d.device
    split = PROJ_CFG["split"]
    # Partials are freed on return; the outputs (saved for backward) get their own block.
    if precomputed is None:
        sq_parts = C // AGG_CFG["bc"]
        agg = torch.empty(T, C, dtype=x2d.dtype, device=dev)
        scratch = torch.empty(split * T * NPAD + sq_parts * T, device=dev)
        part = scratch[: split * T * NPAD]
        sq = scratch[split * T * NPAD :]
    else:
        agg, sq = precomputed
        sq_parts = sq.shape[0]
        part = torch.empty(split * T * NPAD, device=dev)
    rest = torch.empty(T * (NPAD + 2 * n + n * n + 1), device=dev)
    proj = rest[: T * NPAD].view(T, NPAD)
    pre = rest[T * NPAD : T * (NPAD + n)].view(T, n)
    post = rest[T * (NPAD + n) : T * (NPAD + 2 * n)].view(T, n)
    hres = rest[T * (NPAD + 2 * n) : T * (NPAD + 2 * n + n * n)].view(T, n, n)
    inv = rest[T * (NPAD + 2 * n + n * n) :]
    if precomputed is None:
        _launch_agg(x2d, pre_mix, agg, sq, n, True)
    cfg = PROJ_CFG
    if cfg.get("tma") and _tma_ok(x2d, weight):
        _proj_tma(
            (_cdiv(T, cfg["bm"]), split, 1),
            (_desc(x2d, (cfg["bm"], cfg["bk"])), _desc(weight, (NPAD, cfg["bk"])), part, T),
            (K, NPAD, cfg["bm"], cfg["bk"], split, cfg["num_stages"]),
            cfg["num_warps"],
            cfg["num_stages"],
        )
    else:
        _proj(
            (_cdiv(T, cfg["bm"]), split, 1),
            (x2d, weight, part, T),
            (K, NPAD, nout, cfg["bm"], cfg["bk"], split),
            cfg["num_warps"],
            cfg["num_stages"],
        )
    fin = PRE_FIN_CFG
    _pre_fin(
        (_cdiv(T, fin["block_t"]), 1, 1),
        (
            part,
            sq,
            alpha_pre,
            alpha_post,
            alpha_res,
            bias,
            pre,
            post,
            hres,
            proj,
            inv,
            T,
            float(norm_eps),
            float(h_eps),
            float(sk_eps),
        ),
        (K, n, nout, NPAD, split, sq_parts, iters, fin["block_t"]),
        fin["num_warps"],
        fin["num_stages"],
    )
    return agg, pre, post, hres, proj, inv


def sp_aggregate(x2d: Tensor, pre_mix: Optional[Tensor], n: int) -> Tensor:
    """The aggregate alone, bit-identical to ``sp_pre_forward``'s (checkpoint replay)."""
    T, K = x2d.shape
    agg = torch.empty(T, K // n, dtype=x2d.dtype, device=x2d.device)
    _launch_agg(x2d, pre_mix, agg, agg, n, False)
    return agg


class NextAggregate:
    """The next sublayer's aggregate and sum(x^2) partials, written by this sublayer's BDA.

    ``pre`` is this sublayer's H_pre, the next sublayer's aggregation weights. The BDA forward
    fills ``agg`` / ``sq`` once (a checkpoint replay of the BDA leaves them alone) and drops
    ``pre``. ``out`` is a weak reference to the tensor the BDA returned, so the next sublayer only
    uses them for exactly that input. A checkpoint keeps this holder in its context until
    backward, so it must not hold the BDA output strongly: that would close a reference cycle
    through the autograd graph (context -> holder -> output -> grad_fn -> context) and leak it.
    """

    __slots__ = ("pre", "agg", "sq", "filled", "out")

    def __init__(self, pre: Tensor):
        self.pre = pre
        self.agg = self.sq = self.out = None
        self.filled = False

    def take(self, hidden_states: Tensor) -> Optional[Tuple[Tensor, Tensor]]:
        """(aggregate, sum(x^2) partials) if ``hidden_states`` is the filled output, once."""
        out = self.out() if self.out is not None else None
        result = (self.agg, self.sq) if self.filled and out is hidden_states else None
        self.agg = self.sq = self.out = None
        return result


def sp_bda_forward(
    h_res: Tensor,
    x2d: Tensor,
    h_post: Tensor,
    y2d: Tensor,
    n: int,
    next_agg: Optional[NextAggregate] = None,
) -> Tensor:
    """H_res^T x + H_post * y, output in y's dtype; h_res [T, n, n], x [T, n C], y [T, C].

    With an unfilled ``next_agg``, also writes the output's aggregate under ``next_agg.pre`` and
    its sum(x^2) partials (the next sublayer's ``_sp_agg_sq_kernel`` work) into it.
    """
    T, K = x2d.shape
    C = K // n
    cfg = BDA_FWD_CFG
    out = torch.empty(T, K, dtype=y2d.dtype, device=x2d.device)
    fill = next_agg is not None and not next_agg.filled
    if fill:
        next_agg.agg = torch.empty(T, C, dtype=y2d.dtype, device=x2d.device)
        next_agg.sq = torch.empty(C // cfg["block_c"], T, device=x2d.device)
        extra = (next_agg.pre, next_agg.agg, next_agg.sq)
    else:
        extra = (out, out, out)
    _bda_fwd(
        (_cdiv(T, cfg["block_s"]), C // cfg["block_c"], 1),
        (h_res, x2d, h_post, y2d, out, *extra, T),
        (C, n, cfg["block_s"], cfg["block_c"], fill),
        cfg["num_warps"],
        cfg["num_stages"],
    )
    if fill:
        next_agg.filled = True
        next_agg.pre = None
    return out


def sp_bda_backward(
    grad_out: Tensor, h_res: Tensor, x2d: Tensor, h_post: Tensor, y2d: Tensor, n: int
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (g_res [T, nC], g_y [T, C], g_h_post [T, n], g_h_res [T, n, n])."""
    T, K = x2d.shape
    C = K // n
    dev = x2d.device
    g_res = torch.empty(T, K, dtype=x2d.dtype, device=dev)
    g_y = torch.empty(T, C, dtype=y2d.dtype, device=dev)
    small = torch.empty(T * (n + n * n), dtype=h_res.dtype, device=dev)
    g_hp = small[: T * n].view(T, n)
    g_hr = small[T * n :].view(T, n, n)
    if n == 4 and C % 128 == 0 and BDA_BWD4T_CFG is not None:
        cfg = BDA_BWD4T_CFG
        tiles = C // 128
        part = torch.empty(tiles * 20 * T, device=dev)
        _bda_bwd4t(
            (_cdiv(T, cfg["block_s"]), tiles, 1),
            (grad_out, h_res, x2d, h_post, y2d, g_res, g_y, part, T),
            (C, cfg["block_s"], 128, T % cfg["block_s"] == 0),
            cfg["num_warps"],
            1,
        )
        _bda_bwd_rowsum((_cdiv(T, 256), 20, 1), (part, g_hp, g_hr, T), (tiles, 256), 4, 1)
        return g_res, g_y, g_hp, g_hr
    if n == 4 and C % BDA_BWD4_CFG["block_c"] == 0:
        cfg = BDA_BWD4_CFG
        _bda_bwd4(
            (_cdiv(T, cfg["block_s"]), 1, 1),
            (grad_out, h_res, x2d, h_post, y2d, g_res, g_y, g_hp, g_hr, T),
            (C, cfg["block_s"], cfg["block_c"]),
            cfg["num_warps"],
            cfg["num_stages"],
        )
        return g_res, g_y, g_hp, g_hr
    launch_a, launch_b = _bda_bwd_launchers()
    a = BDA_BWD_A_CFG
    launch_a(
        (_cdiv(T, a["block_s"]), _cdiv(C, a["block_c"]), 1),
        (grad_out, h_res, h_post, g_res, g_y, T, K, C, 1, n * n, n, 1, K, C, 1),
        (C, n, a["block_c"], a["block_s"]),
        a["num_warps"],
        a["num_stages"],
    )
    bcfg = BDA_BWD_B_CFG
    launch_b(
        (_cdiv(T, bcfg["block_s"]), 1, 1),
        (grad_out, x2d, y2d, y2d, g_hr, g_hp, T, K, C, 1, K, C, 1, n * n, n, 1),
        (C, n, False, bcfg["block_c"], bcfg["block_s"]),
        bcfg["num_warps"],
        bcfg["num_stages"],
    )
    return g_res, g_y, g_hp, g_hr


def sp_pre_backward(
    x2d: Tensor,
    pre_mix: Optional[Tensor],
    weight: Tensor,
    alpha_pre: Tensor,
    alpha_post: Tensor,
    alpha_res: Tensor,
    bias: Tensor,
    proj: Tensor,
    inv: Tensor,
    g_agg: Optional[Tensor],
    g_pre: Optional[Tensor],
    g_post: Optional[Tensor],
    g_hres: Optional[Tensor],
    g_res: Optional[Tensor],
    n: int,
    sk_eps: float,
    iters: int,
    need_pre_mix_grad: bool,
    need_param_grads: bool,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Tensor]:
    """Return (g_x [T, nC], g_pre_mix [T, n], g_weight [nout, nC], g_alpha_bias [3 + nout])."""
    T, K = x2d.shape
    C = K // n
    nout = n * n + 2 * n
    dev = x2d.device
    fin = PRE_BWD_FIN_CFG
    P = _cdiv(T, fin["block_t"])
    main = PRE_BWD_MAIN_CFG
    CBLK = C // main["bc"]
    has_gagg = g_agg is not None
    need_gpm = need_pre_mix_grad and has_gagg and pre_mix is not None
    # Scratch (freed on return) and returned gradients (kept by autograd) in separate blocks.
    bf = torch.empty(2 * T * NPAD + 2 * NPAD * K, dtype=torch.bfloat16, device=dev)
    a_hi = bf[: T * NPAD]
    a_lo = bf[T * NPAD : 2 * T * NPAD]
    w_hi = bf[2 * T * NPAD : 2 * T * NPAD + NPAD * K]
    w_lo = bf[2 * T * NPAD + NPAD * K :]
    gw_splits = _cdiv(T, GW_CFG["tch"]) if need_param_grads else 0
    f32 = torch.empty(
        T + P * 2 * NPAD + (n * CBLK * T if need_gpm else 0) + gw_splits * nout * K, device=dev
    )
    b = f32[:T]
    off = T
    gab = f32[off : off + P * 2 * NPAD]
    off += P * 2 * NPAD
    gpm_part = f32[off : off + n * CBLK * T] if need_gpm else gab
    off += n * CBLK * T if need_gpm else 0
    gw_part = f32[off:] if need_param_grads else gab
    outs = torch.empty(
        (3 + nout) + (T * n if need_gpm else 0) + (nout * K if need_param_grads else 0), device=dev
    )
    gab_out = outs[: 3 + nout]
    gpm = outs[3 + nout : 3 + nout + T * n].view(T, n) if need_gpm else None
    gw = outs[3 + nout + (T * n if need_gpm else 0) :].view(nout, K) if need_param_grads else None
    ws = torch.empty(T * iters * n * (2 * n + 2), device=dev)
    ws_m = ws[: T * 2 * iters * n * n]
    ws_rs = ws[T * 2 * iters * n * n : T * iters * n * (2 * n + 1)]
    ws_cs = ws[T * iters * n * (2 * n + 1) :]
    w_programs = _cdiv(NPAD * K, fin["w_block"])
    _pre_bwd_fin(
        (P + w_programs, 1, 1),
        (
            g_pre if g_pre is not None else b,
            g_post if g_post is not None else b,
            g_hres if g_hres is not None else b,
            proj,
            inv,
            alpha_pre,
            alpha_post,
            alpha_res,
            bias,
            weight,
            a_hi,
            a_lo,
            b,
            gab,
            w_hi,
            w_lo,
            ws_m,
            ws_rs,
            ws_cs,
            T,
            P,
            float(sk_eps),
        ),
        (
            K,
            n,
            nout,
            NPAD,
            iters,
            fin["block_t"],
            fin["w_block"],
            g_pre is not None,
            g_post is not None,
            g_hres is not None,
        ),
        fin["num_warps"],
        fin["num_stages"],
    )
    g_x = torch.empty(T, K, dtype=x2d.dtype, device=dev)
    has_gres = g_res is not None
    tma_ins = [x2d] + [t for t in (g_agg, g_res) if t is not None]
    if (
        main.get("tma")
        and T % main["tch"] == 0
        and main["tch"] % main["bt"] == 0
        and _tma_ok(*tma_ins)
    ):
        bt, bc = main["bt"], main["bc"]
        xd = _desc(x2d, (bt, bc))
        _pre_bwd_main_tma(
            (n * CBLK, T // main["tch"], 1),
            (
                xd,
                _desc(a_hi.view(T, NPAD), (bt, NPAD)),
                _desc(a_lo.view(T, NPAD), (bt, NPAD)),
                _desc(g_res, (bt, bc)) if has_gres else xd,
                _desc(g_agg, (bt, bc)) if has_gagg else xd,
                _desc(g_x, (bt, bc)),
                b,
                pre_mix if pre_mix is not None else b,
                w_hi,
                w_lo,
                gpm_part,
                T,
            ),
            (
                C,
                n,
                NPAD,
                bt,
                bc,
                main["tch"],
                main["num_stages"],
                pre_mix is not None,
                has_gagg,
                has_gres,
                need_gpm,
            ),
            main["num_warps"],
            main["num_stages"],
        )
    elif main.get("prefetch") and T % main["tch"] == 0 and main["tch"] // main["bt"] >= 2:
        _pre_bwd_main_pf(
            (n * CBLK, T // main["tch"], 1),
            (
                x2d,
                a_hi,
                a_lo,
                b,
                w_hi,
                w_lo,
                g_agg if has_gagg else x2d,
                pre_mix if pre_mix is not None else b,
                g_res if has_gres else x2d,
                g_x,
                gpm_part,
                T,
            ),
            (
                C,
                n,
                NPAD,
                main["bt"],
                main["bc"],
                main["tch"],
                pre_mix is not None,
                has_gagg,
                has_gres,
                need_gpm,
            ),
            main["num_warps"],
            main["num_stages"],
        )
    else:
        _pre_bwd_main(
            (n * CBLK, _cdiv(T, main["tch"]), 1),
            (
                x2d,
                a_hi,
                a_lo,
                b,
                w_hi,
                w_lo,
                g_agg if has_gagg else x2d,
                pre_mix if pre_mix is not None else b,
                g_res if has_gres else x2d,
                g_x,
                gpm_part,
                T,
                main["tch"],
            ),
            (
                C,
                n,
                NPAD,
                main["bt"],
                main["bc"],
                pre_mix is not None,
                has_gagg,
                has_gres,
                need_gpm,
                T % main["tch"] == 0 and main["tch"] % main["bt"] == 0,
            ),
            main["num_warps"],
            main["num_stages"],
        )
    if need_param_grads:
        gcfg = GW_CFG
        if gcfg.get("tma") and _tma_ok(x2d):
            bt = gcfg["bt"]
            _gw_tma(
                (K // gcfg["bk"], gw_splits, 1),
                (
                    _desc(x2d, (bt, gcfg["bk"])),
                    _desc(a_hi.view(T, NPAD), (bt, NPAD)),
                    _desc(a_lo.view(T, NPAD), (bt, NPAD)),
                    gw_part,
                    T,
                ),
                (K, NPAD, nout, gcfg["bk"], bt, gcfg["tch"], gcfg["num_stages"]),
                gcfg["num_warps"],
                gcfg["num_stages"],
            )
        else:
            _gw(
                (K // gcfg["bk"], gw_splits, 1),
                (x2d, a_hi, a_lo, gw_part, T, gcfg["tch"]),
                (K, NPAD, nout, gcfg["bk"], gcfg["bt"]),
                gcfg["num_warps"],
                gcfg["num_stages"],
            )
    block = 256
    gpm_programs = n * _cdiv(T, block) if need_gpm else 0
    gw_programs = _cdiv(nout * K, block) if need_param_grads else 0
    _pre_bwd_reduce(
        (gpm_programs + 1 + gw_programs, 1, 1),
        (
            gpm_part,
            gpm if gpm is not None else gab,
            gab,
            gab_out,
            gw_part,
            gw if gw is not None else gab,
            T,
            P,
            gpm_programs,
            nout * K,
        ),
        (n, nout, NPAD, CBLK, max(gw_splits, 1), need_gpm, block, 64),
        4,
        1,
    )
    return g_x, gpm, gw, gab_out


# pylint: enable=possibly-used-before-assignment


class ResidualGradStash:
    """Hands the BDA's residual gradient to the same sublayer's mapping backward.

    The BDA backward runs first (it produces the mapping outputs' gradients), so the mapping
    backward can fold the residual gradient into its single pass instead of autograd summing it.
    """

    __slots__ = ("grad", "active")

    def __init__(self):
        self.grad = None
        self.active = False


class SinglePassPreFunction(torch.autograd.Function):
    """Aggregate + mappings of one sublayer: (x, pre_mix, params) -> (agg, pre, post, h_res)."""

    @staticmethod
    def forward(
        ctx,
        x,
        pre_mix,
        weight,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        n,
        norm_eps,
        h_eps,
        sk_eps,
        iters,
        stash,
        precomputed,
    ):
        """Compute the aggregate and mappings and save their inputs for backward."""
        s, b, width = x.shape
        x2d = x.view(s * b, width)
        pre2d = None if pre_mix is None else pre_mix.reshape(s * b, n)
        if pre2d is not None and (pre2d.dtype != torch.float32 or not pre2d.is_contiguous()):
            pre2d = pre2d.float().contiguous()
        agg, pre, post, hres, proj, inv = sp_pre_forward(
            x2d,
            pre2d,
            weight,
            alpha_pre,
            alpha_post,
            alpha_res,
            bias,
            n,
            norm_eps,
            h_eps,
            sk_eps,
            iters,
            precomputed,
        )
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, pre2d, weight, alpha_pre, alpha_post, alpha_res, bias, proj, inv)
        ctx.n, ctx.sk_eps, ctx.iters, ctx.stash = n, sk_eps, iters, stash
        ctx.pre_mix_dtype = None if pre_mix is None else pre_mix.dtype
        return (
            agg.view(s, b, width // n),
            pre.view(s, b, n),
            post.view(s, b, n),
            hres.view(s, b, n, n),
        )

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g_agg, g_pre, g_post, g_hres):
        """Backpropagate through the mappings and the previous residual mix."""
        x, pre2d, weight, alpha_pre, alpha_post, alpha_res, bias, proj, inv = ctx.saved_tensors
        n = ctx.n
        s, b, width = x.shape
        T = s * b
        g_res = None
        stash = ctx.stash
        if stash is not None:
            g_res, stash.grad = stash.grad, None
        if g_res is not None:
            g_res = g_res.reshape(T, width)

        def flat(g, *shape):
            return None if g is None else g.reshape(*shape).contiguous()

        need_x = ctx.needs_input_grad[0]
        need_params = any(ctx.needs_input_grad[2:7])
        g_x, g_pm, g_w, g_ab = sp_pre_backward(
            x.view(T, width),
            pre2d,
            weight,
            alpha_pre,
            alpha_post,
            alpha_res,
            bias,
            proj,
            inv,
            flat(g_agg, T, width // n),
            flat(g_pre, T, n),
            flat(g_post, T, n),
            flat(g_hres, T, n, n),
            g_res,
            n,
            ctx.sk_eps,
            ctx.iters,
            ctx.needs_input_grad[1],
            need_params,
        )
        g_pre_mix = None
        if g_pm is not None:
            g_pre_mix = g_pm.view(s, b, n)
            if ctx.pre_mix_dtype != torch.float32:
                g_pre_mix = g_pre_mix.to(ctx.pre_mix_dtype)
        if not need_params:
            return (g_x.view(s, b, width) if need_x else None, g_pre_mix) + (None,) * 12
        return (
            g_x.view(s, b, width) if need_x else None,
            g_pre_mix,
            g_w,
            g_ab[0:1],
            g_ab[1:2],
            g_ab[2:3],
            g_ab[3:],
        ) + (None,) * 7


class SinglePassBDAFunction(torch.autograd.Function):
    """H_res^T x + H_post * y; with an active stash, the residual gradient goes to the stash."""

    @staticmethod
    def forward(ctx, h_res, x, h_post, y, n, stash, next_agg):
        """Combine the branch output and residual and optionally prepare the next aggregate."""
        s, b, width = x.shape
        T = s * b
        hr = h_res.reshape(T, n, n)
        hp = h_post.reshape(T, n)
        y2d = y.reshape(T, width // n)
        if not y2d.is_contiguous():
            y2d = y2d.contiguous()
        out = sp_bda_forward(hr, x.view(T, width), hp, y2d, n, next_agg)
        ctx.save_for_backward(hr, x, hp, y2d)
        ctx.n, ctx.stash = n, stash
        ctx.shapes = (h_res.shape, h_post.shape, y.shape)
        return out.view(s, b, width)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        """Return the branch gradients and pass the residual gradient to the paired mapping."""
        hr, x, hp, y2d = ctx.saved_tensors
        n = ctx.n
        s, b, width = x.shape
        T = s * b
        go = grad_out.reshape(T, width)
        if not go.is_contiguous():
            go = go.contiguous()
        g_res, g_y, g_hp, g_hr = sp_bda_backward(go, hr, x.view(T, width), hp, y2d, n)
        hr_shape, hp_shape, y_shape = ctx.shapes
        g_res = g_res.view(s, b, width) if ctx.needs_input_grad[1] else None
        stash = ctx.stash
        if g_res is not None and stash is not None and stash.active:
            # The same sublayer's mapping backward runs next and folds it into its pass.
            stash.grad, g_res = g_res, None
        return g_hr.view(hr_shape), g_res, g_hp.view(hp_shape), g_y.view(y_shape), None, None, None
