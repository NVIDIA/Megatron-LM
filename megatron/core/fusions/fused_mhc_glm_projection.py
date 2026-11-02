# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused mHC operations for the per-op GLM path (mhc_fused_ops).

The fixed-launch Triton Sinkhorn: fused_mhc_kernels' Sinkhorn kernels launched with a fixed
num_warps instead of their timing autotune, so every process runs the same compiled kernel.
The kernels are looked up through fused_mhc_kernels at call time: they exist only when Triton
is available, so not under MHC_FORCE_BACKEND=native, MHC_FORCE_BACKEND=cutile or
MHC_DISABLE_TRITON=1.

The GLM projection pair: per-op ``proj_rms`` with ``mhc_norm_eps_inside_sqrt`` and
``mhc_normalize_before_projection``, the drop-in of hyper_connection._proj_rms_impl(...,
eps_inside_sqrt=True, normalize_before_projection=True) as fixed-config Triton kernels (the RMS
scale, the split-K projection and its ordered reduce in the forward; a per-token pre-kernel, the
dgrad with its epilogue and the wgrad, and the wgrad's ordered reduce in the backward). Its
kernels are defined only when fused_mhc_kernels defined its own Triton kernels (and Triton has
the PDL externs). Each call reads torch's fp32 matmul precision: "tf32" runs the TF32 forms with
the rounding of mhc_fused_tf32_rounding, "ieee" and "none" run the strict forms. The kernel
lookup fails closed: it raises instead of returning kernels this process never defined.
"""

import contextvars
import functools
import os
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
from torch import Tensor

from megatron.core.fusions import fused_mhc_kernels as fmk

SINKHORN_NUM_WARPS = 1
"""Warps per program of both Sinkhorn kernels; the per-rank fused-ops line prints this value."""


def _sinkhorn_kernels():
    """Return the forward and backward JIT functions under fused_mhc_kernels' autotuners."""
    if not fmk.is_triton_available():
        raise RuntimeError("The fixed-launch mHC Sinkhorn needs Triton, which is unavailable.")
    return fmk._triton_sinkhorn_fwd_kernel.fn, fmk._triton_sinkhorn_bwd_kernel.fn


def _require_cuda(tensor: Tensor) -> None:
    """Fail closed on a non-CUDA tensor instead of falling back to the native Sinkhorn."""
    if not tensor.is_cuda:
        raise RuntimeError(
            f"The fixed-launch mHC Sinkhorn needs CUDA tensors, got device {tensor.device}. "
            "Drop 'sinkhorn' from mhc_fused_ops or set mhc_fused_backend='native'."
        )


def _sinkhorn_fwd(
    input_logits: Tensor, num_iterations: int, eps: float = 1e-6, *, num_warps: Optional[int] = None
) -> Tuple[Tensor, Tensor]:
    """fused_mhc_kernels' _triton_sinkhorn_fwd with a fixed-warp launch; returns (out, M_init)."""
    _require_cuda(input_logits)
    fwd_kernel, _ = _sinkhorn_kernels()
    if num_warps is None:
        num_warps = SINKHORN_NUM_WARPS  # read at launch time, never bound at definition
    original_shape = input_logits.shape
    hc = original_shape[-1]
    N_batch = input_logits.numel() // (hc * hc)
    dev = input_logits.device
    out = torch.empty(N_batch, hc, hc, dtype=input_logits.dtype, device=dev)
    M_init = torch.empty(N_batch, hc, hc, dtype=input_logits.dtype, device=dev)
    inp = input_logits.contiguous().view(N_batch, hc, hc)
    fwd_kernel[(N_batch,)](inp, out, M_init, N_batch, eps, hc, num_iterations, num_warps=num_warps)
    return out.view(original_shape), M_init.view(original_shape)


def _sinkhorn_bwd(
    grad_output: Tensor,
    M_init: Tensor,
    num_iterations: int,
    eps: float = 1e-6,
    *,
    num_warps: Optional[int] = None,
) -> Tensor:
    """fused_mhc_kernels' _triton_sinkhorn_bwd with a fixed-warp launch; fp32 workspaces."""
    _require_cuda(grad_output)
    _, bwd_kernel = _sinkhorn_kernels()
    if num_warps is None:
        num_warps = SINKHORN_NUM_WARPS  # read at launch time, never bound at definition
    original_shape = grad_output.shape
    hc = original_shape[-1]
    N_batch = grad_output.numel() // (hc * hc)
    dev = grad_output.device
    grad_input = torch.empty(N_batch, hc, hc, dtype=grad_output.dtype, device=dev)
    go = grad_output.contiguous().view(N_batch, hc, hc)
    mi = M_init.contiguous().view(N_batch, hc, hc)
    ws_M = torch.empty(N_batch * 2 * num_iterations * hc * hc, dtype=torch.float32, device=dev)
    ws_rs = torch.empty(N_batch * num_iterations * hc, dtype=torch.float32, device=dev)
    ws_cs = torch.empty(N_batch * num_iterations * hc, dtype=torch.float32, device=dev)
    bwd_kernel[(N_batch,)](
        go,
        mi,
        grad_input,
        ws_M,
        ws_rs,
        ws_cs,
        N_batch,
        eps,
        hc,
        num_iterations,
        num_warps=num_warps,
    )
    return grad_input.view(original_shape)


class FixedLaunchSinkhorn(torch.autograd.Function):
    """Sinkhorn with fused_mhc_kernels' Triton kernels at SINKHORN_NUM_WARPS warps.

    Saves M_init for the backward, as fused_mhc_kernels' TritonFusedSinkhorn does. It has no
    deterministic-mode branch: one fixed launch serves both modes.
    """

    @staticmethod
    def forward(ctx, input_logits: Tensor, num_iterations: int, eps: float = 1e-6) -> Tensor:
        """Run the Sinkhorn forward and save the initial matrix for the backward."""
        out, M_init = _sinkhorn_fwd(input_logits, num_iterations, eps)
        ctx.save_for_backward(M_init)
        ctx.num_iterations = num_iterations
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Run the Sinkhorn backward from the saved initial matrix."""
        (M_init,) = ctx.saved_tensors
        # Through the module global, so that a test can count the backward launches.
        grad_input = _sinkhorn_bwd(grad_output, M_init, ctx.num_iterations, ctx.eps)
        return grad_input, None, None


def fixed_launch_sinkhorn(input_logits: Tensor, num_iterations: int, eps: float = 1e-6) -> Tensor:
    """Apply the fixed-launch Triton Sinkhorn with autograd support.

    The per-op ``sinkhorn`` operation of HyperConnectionModule; it takes the arguments of
    hyper_connection.native_sinkhorn.
    """
    return FixedLaunchSinkhorn.apply(input_logits, num_iterations, eps)


# ============================================================================
# The GLM projection pair: constants and frozen launch configs
# ============================================================================
#
# Forward:  GP-1 s = rsqrt(sum_k xf^2 * fl32(1/K) + eps) (and, in TF32 mode, W_r = T(W) as [32, K]);
#           GP-2 split-K partials of T(xn) T(W_r)^T, xn = fl32(xf * s), each chain from acc = 0;
#           GP-3 p = their sum in chain order from +0.0.
# Backward: GP-4p per token g_s = fl32(sum_j fl32(G_p,j * p_j)) / s (+ G_s), coef = s^3 / K * g_s,
#           and the padded T(G_p) copies; GP-4 g_xn = T(G_p) T(W), g_x = bf16_RN(s * g_xn - coef *
#           xf) and the wgrad partials of T(xn)^T T(G_p); GP-4r g_W = their sum in program order
#           from +0.0. T() is the TF32 operand rounding (RN or RZ); the strict forms use the raw
#           fp32 operands.

PROJ_N = 24
"""Projection outputs, n * n + 2n at n = 4: the only n that per-op mode accepts."""

BLOCK_N = 32
"""PROJ_N padded to the MMA width; columns 24..31 of every padded operand are zero."""

GLM_PAIR_DEVICE_CAPABILITIES = ((10, 3),)
"""Compute capabilities the pair's kernels were validated on (GB300, sm_103)."""

_ROUND_MODE = {"rn": 1, "rz": 2}
# _tf32_round MODE per mhc_fused_tf32_rounding; under "rn", xn uses MODE 3 (cvt.rn.tf32.f32)
_XN_CVT = 3

GP4_NUM_SMS, GP4_OCC = 152, 2
"""GP-4's persistent grid holds at most GP4_NUM_SMS * GP4_OCC programs. Fixed, never queried
from the device: g_W's wgrad pieces, and so its bits, depend on them."""

GP1_CONFIG = dict(ROWS=8, BLOCK_K=1024, WR_BLOCK=128, num_warps=8, num_stages=1)
GP2_CONFIG = dict(BLOCK_M=128, BLOCK_K=64, SPLIT=14, ORDER=0, num_warps=8, num_stages=3)
GP2_STRICT_CONFIG = dict(
    BLOCK_M=128, BLOCK_K=64, SPLIT=16, PRECISION="tf32x3", num_warps=8, num_stages=3
)
GP3_CONFIG = dict(BLOCK_M=16, num_warps=4, num_stages=1)
GP4P_CONFIG = dict(BLOCK_R=16, num_warps=4, num_stages=1)
GP4_CONFIG = dict(
    BLOCK_T=64,
    BLOCK_C=128,
    num_warps=8,
    num_stages=1,
    maxnreg=128,
    maxnreg_dbg=None,
    ANCHOR=1,
    GT=1,
)
GP4_STRICT_CONFIG = dict(BLOCK_T=64, BLOCK_C=128, SPLIT_M=2, num_warps=8, num_stages=2)
GP4R_CONFIG = dict(RBC=64, num_warps=4, num_stages=1)
GP4R_STRICT_CONFIG = dict(BLOCK=512, num_warps=4, num_stages=1)

# Workspace sections: the forward's start on 1 KiB, the backward's on 128 B boundaries (fp32).
_FWD_ALIGN, _BWD_ALIGN = 256, 32
_TMA_SCRATCH_PER_CTA = 1024  # bytes reserved per GP-2 program for its device TMA descriptors


# ============================================================================
# The GLM projection pair: Triton kernels
# ============================================================================

if fmk._TRITON_AVAILABLE:  # the state under which fused_mhc_kernels defines its Triton kernels
    import triton
    import triton.language as tl

    try:
        from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
    except ImportError:  # no PDL externs in this Triton: the pair's kernels stay undefined
        gdc_launch_dependents = gdc_wait = None
else:
    triton = tl = gdc_launch_dependents = gdc_wait = None

_GLM_KERNELS_DEFINED = False
if gdc_wait is not None:

    @triton.jit
    def _tf32_round(v, MODE: tl.constexpr):
        """TF32 operand rounding: MODE 1 RNE and MODE 2 RZ by integer bit operations, with inf and
        NaN passed through; MODE 3 RNE by one cvt.rn.tf32.f32; MODE 0 none."""
        if MODE == 1:
            b = v.to(tl.uint32, bitcast=True)
            r = (b + 0xFFF + ((b >> 13) & 1)) & 0xFFFFE000
            out = tl.where((b & 0x7F800000) == 0x7F800000, v, r.to(tl.float32, bitcast=True))
        elif MODE == 2:
            b = v.to(tl.uint32, bitcast=True)
            r = b & 0xFFFFE000
            out = tl.where((b & 0x7F800000) == 0x7F800000, v, r.to(tl.float32, bitcast=True))
        elif MODE == 3:
            out = tl.inline_asm_elementwise(
                "cvt.rn.tf32.f32 $0, $1;", "=r,r", [v], dtype=tl.float32, is_pure=True, pack=1
            )
        else:
            out = v
        return out

    @triton.jit
    def _glm_wr_program(
        w_ptr,
        wr_ptr,
        j,
        K,
        N,
        stride_wn,
        WR_BLOCK: tl.constexpr,
        BLOCK_N: tl.constexpr,
        WROUND: tl.constexpr,
    ):
        """W_r[n, k] = T(W[n, k]) for n < N and 0 for N <= n < BLOCK_N, over the 16 sub-blocks of
        WR_BLOCK columns of program j; W_r is [BLOCK_N, K], K contiguous."""
        n_idx = tl.arange(0, BLOCK_N)
        for i in range(0, 16):
            kb = (j * 16 + i) * WR_BLOCK + tl.arange(0, WR_BLOCK)
            k_ok = kb[None, :] < K
            w = tl.load(
                w_ptr + n_idx[:, None] * stride_wn + kb[None, :],
                mask=(n_idx[:, None] < N) & k_ok,
                other=0.0,
            )
            tl.store(wr_ptr + n_idx[:, None] * K + kb[None, :], _tf32_round(w, WROUND), mask=k_ok)

    @triton.jit
    def _glm_rms_rows(
        x_ptr,
        s_ptr,
        pid,
        T,
        K,
        stride_xt,
        inv_k,
        eps,
        ROWS: tl.constexpr,
        BLOCK_K: tl.constexpr,
        EVEN: tl.constexpr,
    ):
        """s of ROWS rows: per (row, lane) a running fp32 sum of xf^2 over the K chunks in chunk
        order, one tl.sum over the lanes, then s = rsqrt(ss * inv_k + eps)."""
        rows = pid * ROWS + tl.arange(0, ROWS)
        acc = tl.zeros([ROWS, BLOCK_K], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            cols = k0 + tl.arange(0, BLOCK_K)
            if EVEN:
                xf = tl.load(x_ptr + rows[:, None] * stride_xt + cols[None, :]).to(tl.float32)
            else:
                xf = tl.load(
                    x_ptr + rows[:, None] * stride_xt + cols[None, :],
                    mask=(rows[:, None] < T) & (cols[None, :] < K),
                    other=0.0,
                ).to(tl.float32)
            acc += xf * xf  # the square of a bf16 value is exact in fp32
        ss = tl.sum(acc, axis=1)
        s = tl.math.rsqrt(ss * inv_k + eps)
        if EVEN:
            tl.store(s_ptr + rows, s)
        else:
            tl.store(s_ptr + rows, s, mask=rows < T)

    @triton.jit
    def _glm_rms_fwd_kernel(
        x_ptr,
        s_ptr,
        w_ptr,
        wr_ptr,
        T,
        K,
        N,
        stride_xt,
        stride_wn,
        inv_k,
        eps,
        n_row_progs,
        ROWS: tl.constexpr,
        BLOCK_K: tl.constexpr,
        EVEN: tl.constexpr,
        WPRE: tl.constexpr,
        WR_BLOCK: tl.constexpr,
        BLOCK_N: tl.constexpr,
        WROUND: tl.constexpr,
        PDL: tl.constexpr,
    ):
        """GP-1: the rms scale s of every token (programs < n_row_progs); with WPRE, the programs
        after them write W_r. With PDL, signals the dependent launch after its stores."""
        pid = tl.program_id(0)
        if WPRE:
            if pid < n_row_progs:
                _glm_rms_rows(x_ptr, s_ptr, pid, T, K, stride_xt, inv_k, eps, ROWS, BLOCK_K, EVEN)
            else:
                _glm_wr_program(
                    w_ptr, wr_ptr, pid - n_row_progs, K, N, stride_wn, WR_BLOCK, BLOCK_N, WROUND
                )
        else:
            _glm_rms_rows(x_ptr, s_ptr, pid, T, K, stride_xt, inv_k, eps, ROWS, BLOCK_K, EVEN)
        if PDL:
            gdc_launch_dependents()

    @triton.jit
    def _glm_proj_fwd_kernel(
        x_ptr,
        s_ptr,
        wr_ptr,
        ws_ptr,
        T,
        K,
        k_per_split,
        stride_xt,
        stride_ws_split,
        stride_ws_t,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ROUND: tl.constexpr,
        EVEN_M: tl.constexpr,
        ORDER: tl.constexpr,
        NS: tl.constexpr,
        PDL: tl.constexpr,
        PRECISION: tl.constexpr,
    ):
        """GP-2 (TF32): ws[chain, t, 0:32] = one MMA chain from acc = 0 over the chain's K columns,
        x and W_r by device TMA descriptors (out-of-range rows and columns zero-filled); the
        descriptors' scratch comes from the caller's allocator."""
        pid_m = tl.program_id(0)
        if ORDER == 1:
            pid_k = tl.num_programs(1) - 1 - tl.program_id(1)
        else:
            pid_k = tl.program_id(1)
        x_desc = tl.make_tensor_descriptor(
            x_ptr, shape=[T, K], strides=[stride_xt, 1], block_shape=[BLOCK_M, BLOCK_K]
        )
        w_desc = tl.make_tensor_descriptor(
            wr_ptr, shape=[BLOCK_N, K], strides=[K, 1], block_shape=[BLOCK_N, BLOCK_K]
        )
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_idx = tl.arange(0, BLOCK_N)
        k_begin = pid_k * k_per_split
        k_len = tl.minimum(k_per_split, K - k_begin)
        if PDL:
            gdc_wait()
        if EVEN_M:
            s = tl.load(s_ptr + rows)
        else:
            s = tl.load(s_ptr + rows, mask=rows < T, other=0.0)
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for kk in tl.range(0, k_len, BLOCK_K, num_stages=NS):
            x = x_desc.load([pid_m * BLOCK_M, k_begin + kk])
            w = w_desc.load([0, k_begin + kk])  # [BLOCK_N, BLOCK_K], K contiguous
            acc = tl.dot(
                _tf32_round(x.to(tl.float32) * s[:, None], ROUND),
                tl.trans(w),
                acc,
                input_precision=PRECISION,
                out_dtype=tl.float32,
            )
        if PDL:
            gdc_launch_dependents()
        if EVEN_M:
            tl.store(
                ws_ptr + pid_k * stride_ws_split + rows[:, None] * stride_ws_t + n_idx[None, :], acc
            )
        else:
            tl.store(
                ws_ptr + pid_k * stride_ws_split + rows[:, None] * stride_ws_t + n_idx[None, :],
                acc,
                mask=rows[:, None] < T,
            )

    @triton.jit
    def _glm_proj_fwd_strict_kernel(
        x_ptr,
        s_ptr,
        w_ptr,
        ws_ptr,
        T,
        K,
        N,
        k_per_split,
        stride_xt,
        stride_wn,
        stride_ws_split,
        stride_ws_t,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ROUND: tl.constexpr,
        PRECISION: tl.constexpr,
    ):
        """GP-2 (strict): the P2 prototype's split-K kernel; W read directly, empty splits store
        zero partials."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_ok = rows < T
        n_idx = tl.arange(0, BLOCK_N)
        n_ok = n_idx < N
        s = tl.load(s_ptr + rows, mask=row_ok, other=0.0)
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        k_begin = pid_k * k_per_split
        for kk in range(0, k_per_split, BLOCK_K):
            cols = k_begin + kk + tl.arange(0, BLOCK_K)
            col_ok = cols < K
            x = tl.load(
                x_ptr + rows[:, None] * stride_xt + cols[None, :],
                mask=row_ok[:, None] & col_ok[None, :],
                other=0.0,
            )
            xn = x.to(tl.float32) * s[:, None]  # fl32(xf * s)
            w_t = tl.load(
                w_ptr + n_idx[None, :] * stride_wn + cols[:, None],
                mask=col_ok[:, None] & n_ok[None, :],
                other=0.0,
            )
            acc = tl.dot(
                _tf32_round(xn, ROUND),
                _tf32_round(w_t, ROUND),
                acc,
                input_precision=PRECISION,
                out_dtype=tl.float32,
            )
        out_ok = row_ok[:, None] & (n_idx[None, :] < BLOCK_N)
        tl.store(
            ws_ptr + pid_k * stride_ws_split + rows[:, None] * stride_ws_t + n_idx[None, :],
            acc,
            mask=out_ok,
        )

    @triton.jit
    def _glm_proj_reduce_kernel(
        ws_ptr,
        p_ptr,
        T,
        N,
        stride_ws_split,
        stride_ws_t,
        stride_pt,
        SPLIT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        EVEN: tl.constexpr,
        PDL: tl.constexpr,
    ):
        """GP-3: p = ((0 + w_0) + w_1) + ... + w_{SPLIT-1}, fp32 round-to-nearest adds in chain
        order from +0.0, over BLOCK_M-row programs."""
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        n_idx = tl.arange(0, BLOCK_N)
        if EVEN:
            ok = n_idx[None, :] < N
        else:
            ok = (rows[:, None] < T) & (n_idx[None, :] < N)
        offs = rows[:, None] * stride_ws_t + n_idx[None, :]
        if PDL:
            gdc_wait()
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for i in tl.static_range(0, SPLIT):
            acc += tl.load(ws_ptr + i * stride_ws_split + offs, mask=ok, other=0.0)
        tl.store(p_ptr + rows[:, None] * stride_pt + n_idx[None, :], acc, mask=ok)

    @triton.jit
    def _glm_proj_bwd_rowscale_kernel(
        s_ptr,
        p_ptr,
        gp_ptr,
        gs_ptr,
        coef_ptr,
        gpr_ptr,
        gprt_ptr,
        gs_out_ptr,
        T,
        T_pad,
        inv_k,
        stride_pt,
        stride_gpt,
        stride_gpn,
        stride_gs,
        BLOCK_R: tl.constexpr,
        BLOCK_N: tl.constexpr,
        N: tl.constexpr,
        ROUND: tl.constexpr,
        GT: tl.constexpr,
        HAS_GP: tl.constexpr,
        HAS_GS: tl.constexpr,
        DEBUG: tl.constexpr,
    ):
        """GP-4p, per token: g_s = fl32(sum_j fl32(G_p,j * p_j)) summed in index order from +0.0,
        one correctly rounded division by s, then + G_s; coef = s * s * s * inv_k * g_s, left to
        right. Writes coef for all T_pad rows and, with G_p, T(G_p) as [T_pad, BLOCK_N] and (GT)
        [BLOCK_N, T_pad], rows >= T and columns >= N zero; DEBUG also writes g_s."""
        pid = tl.program_id(0)
        rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
        row_ok = rows < T
        st_ok = rows < T_pad
        s = tl.load(s_ptr + rows, mask=row_ok, other=1.0)
        if HAS_GP:
            acc = tl.zeros([BLOCK_R], dtype=tl.float32)
            for j in tl.static_range(N):
                g_j = tl.load(gp_ptr + rows * stride_gpt + j * stride_gpn, mask=row_ok, other=0.0)
                p_j = tl.load(p_ptr + rows * stride_pt + j, mask=row_ok, other=0.0)
                acc = acc + g_j * p_j
            g_s = tl.math.div_rn(acc, s)
            if HAS_GS:
                g_s = g_s + tl.load(gs_ptr + rows * stride_gs, mask=row_ok, other=0.0)
        elif HAS_GS:
            g_s = tl.load(gs_ptr + rows * stride_gs, mask=row_ok, other=0.0)
        else:
            g_s = tl.zeros([BLOCK_R], dtype=tl.float32)
        coef = s * s * s * inv_k * g_s
        tl.store(coef_ptr + rows, coef, mask=st_ok)
        if DEBUG:
            tl.store(gs_out_ptr + rows, g_s, mask=row_ok)
        if HAS_GP:
            n_idx = tl.arange(0, BLOCK_N)
            tn_ok = row_ok[:, None] & (n_idx < N)[None, :]
            gp = tl.load(
                gp_ptr + rows[:, None] * stride_gpt + n_idx[None, :] * stride_gpn,
                mask=tn_ok,
                other=0.0,
            )
            gr = _tf32_round(gp, ROUND)
            tl.store(gpr_ptr + rows[:, None] * BLOCK_N + n_idx[None, :], gr, mask=st_ok[:, None])
            if GT:
                tl.store(gprt_ptr + n_idx[None, :] * T_pad + rows[:, None], gr, mask=st_ok[:, None])

    @triton.jit
    def _glm_anchor(v, a, ANCHOR: tl.constexpr):
        """The identity on v (a 32-bit register move); with ANCHOR, the unused operand a places v,
        and what is computed from it, in a's register layout."""
        if ANCHOR:
            out = tl.inline_asm_elementwise(
                "mov.b32 $0, $1;", "=r,r,r", [v, a], dtype=tl.float32, is_pure=True, pack=1
            )
        else:
            out = v
        return out

    @triton.jit
    def _glm_gp4_program_plan(pid, NB, NU, NPROG):
        """GP-4's static partition: program pid owns the units [u_lo, u_hi) of u = c * NB + b, which
        lie in the column tiles c_lo..c_hi. The host's gp4_wgrad_split_bounds is this plan."""
        u_lo = (pid * NU) // NPROG
        u_hi = ((pid + 1) * NU) // NPROG
        c_lo = u_lo // NB
        c_hi = (u_hi - 1) // NB
        return u_lo, u_hi, c_lo, c_hi

    @triton.jit
    def _glm_gp4_piece_blocks(u_lo, u_hi, c, NB):
        """The token blocks [b_lo, b_hi) of column tile c inside the units [u_lo, u_hi)."""
        b_lo = tl.maximum(u_lo, c * NB) - c * NB
        b_hi = tl.minimum(u_hi, (c + 1) * NB) - c * NB
        return b_lo, b_hi

    @triton.jit
    def _glm_gp4_tile_programs(c, NB, NU, NPROG):
        """The programs p_first..p_last whose pieces cover column tile c: those holding the units
        c * NB and (c + 1) * NB - 1."""
        p_first = ((c * NB + 1) * NPROG - 1) // NU
        p_last = ((c + 1) * NB * NPROG - 1) // NU
        return p_first, p_last

    @triton.jit
    def _glm_gp4_slot(q, c):
        """The wgrad partial slot that program q's piece of column tile c writes and GP-4r reads."""
        return q + c

    @triton.jit
    def _glm_proj_bwd_piece(
        x_ptr,
        s_ptr,
        coef_ptr,
        gpr_ptr,
        gprt_ptr,
        w_ptr,
        gx_ptr,
        gx32_ptr,
        gxn_ptr,
        part_ptr,
        c,
        b_lo,
        b_hi,
        part_off,
        part_stride_n,
        T,
        K,
        N,
        T_pad,
        stride_xt,
        stride_wn,
        stride_gxt,
        BLOCK_T: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
        ROUND: tl.constexpr,
        XROUND: tl.constexpr,
        PRECISION: tl.constexpr,
        HAS_GP: tl.constexpr,
        DEBUG: tl.constexpr,
        GT: tl.constexpr,
        ANCHOR: tl.constexpr,
        NS: tl.constexpr,
    ):
        """One piece: column tile c over token blocks [b_lo, b_hi). Stores g_x of every tile and,
        with G_p, the piece's g_W^T partial [BLOCK_C, N] at part_ptr + part_off."""
        c0 = c * BLOCK_C
        j_idx = tl.arange(0, BLOCK_C)
        cols = c0 + j_idx
        col_ok = cols < K
        n_idx = tl.arange(0, BLOCK_N)
        n_ok = n_idx < N
        if HAS_GP:
            w = tl.load(
                w_ptr + n_idx[:, None] * stride_wn + cols[None, :],
                mask=n_ok[:, None] & col_ok[None, :],
                other=0.0,
            )
            w_op = _tf32_round(w, ROUND)  # [BLOCK_N, BLOCK_C], once per piece
            gw_acc = tl.zeros([BLOCK_C, BLOCK_N], dtype=tl.float32)  # transposed wgrad, in TMEM
        for b in tl.range(b_lo, b_hi, num_stages=NS):
            t0 = b * BLOCK_T
            rows = t0 + tl.arange(0, BLOCK_T)
            row_ok = rows < T
            tc_ok = row_ok[:, None] & col_ok[None, :]
            s = tl.load(s_ptr + rows, mask=row_ok, other=1.0)
            coef = tl.load(coef_ptr + rows, mask=row_ok, other=0.0)
            if HAS_GP:  # GP-4p wrote all T_pad rows, so these loads need no mask
                gp_op = tl.load(gpr_ptr + rows[:, None] * BLOCK_N + n_idx[None, :])
                if GT:
                    gpt_op = tl.load(gprt_ptr + n_idx[:, None] * T_pad + rows[None, :])
                else:
                    gpt_op = tl.trans(gp_op)
            xb = tl.load(x_ptr + rows[:, None] * stride_xt + cols[None, :], mask=tc_ok, other=0.0)
            if HAS_GP:
                g_xn = tl.dot(gp_op, w_op, input_precision=PRECISION, out_dtype=tl.float32)
            xf = xb.to(tl.float32)
            if HAS_GP:
                gx = s[:, None] * g_xn - coef[:, None] * xf
            else:
                gx = 0.0 - coef[:, None] * xf  # +0 where coef * xf is zero, as 0 - coef * xf
            tl.store(
                gx_ptr + rows[:, None] * stride_gxt + cols[None, :],
                gx.to(tl.bfloat16, fp_downcast_rounding="rtne"),
                mask=tc_ok,
            )
            if DEBUG:  # the fp32 g_x that the single bf16 rounding is applied to, and g_xn
                tl.store(gx32_ptr + rows[:, None] * stride_gxt + cols[None, :], gx, mask=tc_ok)
                if HAS_GP:
                    tl.store(gxn_ptr + rows[:, None] * stride_gxt + cols[None, :], g_xn, mask=tc_ok)
            if HAS_GP:
                xn_op = _tf32_round(_glm_anchor(xf, g_xn, ANCHOR) * s[:, None], XROUND)
                gw_acc = tl.dot(
                    tl.trans(xn_op),
                    tl.trans(gpt_op),
                    gw_acc,
                    input_precision=PRECISION,
                    out_dtype=tl.float32,
                )
        if HAS_GP:
            tl.store(
                part_ptr + part_off + n_idx[None, :] * part_stride_n + j_idx[:, None],
                gw_acc,
                mask=n_ok[None, :] & col_ok[:, None],
            )

    @triton.jit
    def _glm_proj_bwd_kernel(
        x_ptr,
        s_ptr,
        coef_ptr,
        gpr_ptr,
        gprt_ptr,
        w_ptr,
        gx_ptr,
        gx32_ptr,
        gxn_ptr,
        part_ptr,
        T,
        K,
        N,
        T_pad,
        NB,
        NU,
        NPROG,
        stride_xt,
        stride_wn,
        stride_gxt,
        BLOCK_T: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
        ROUND: tl.constexpr,
        XROUND: tl.constexpr,
        PRECISION: tl.constexpr,
        HAS_GP: tl.constexpr,
        DEBUG: tl.constexpr,
        GT: tl.constexpr,
        ANCHOR: tl.constexpr,
        NS: tl.constexpr,
    ):
        """GP-4 (TF32): dgrad, epilogue and transposed wgrad on a persistent grid. The NU = NB * NC
        units u = c * NB + b (column-major) are split statically over NPROG programs; program pid
        runs its pieces (one per column tile) in token order, writing the partial of tile c to
        slot pid + c."""
        pid = tl.program_id(0)
        u_lo, u_hi, c_lo, c_hi = _glm_gp4_program_plan(pid, NB, NU, NPROG)
        for c in range(c_lo, c_hi + 1):
            b_lo, b_hi = _glm_gp4_piece_blocks(u_lo, u_hi, c, NB)
            _glm_proj_bwd_piece(
                x_ptr,
                s_ptr,
                coef_ptr,
                gpr_ptr,
                gprt_ptr,
                w_ptr,
                gx_ptr,
                gx32_ptr,
                gxn_ptr,
                part_ptr,
                c,
                b_lo,
                b_hi,
                _glm_gp4_slot(pid, c) * (N * BLOCK_C),
                BLOCK_C,
                T,
                K,
                N,
                T_pad,
                stride_xt,
                stride_wn,
                stride_gxt,
                BLOCK_T,
                BLOCK_N,
                BLOCK_C,
                ROUND,
                XROUND,
                PRECISION,
                HAS_GP,
                DEBUG,
                GT,
                ANCHOR,
                NS,
            )

    @triton.jit
    def _glm_proj_bwd_strict_kernel(
        x_ptr,
        s_ptr,
        coef_ptr,
        gpr_ptr,
        w_ptr,
        gx_ptr,
        gx32_ptr,
        gxn_ptr,
        gws_ptr,
        T,
        K,
        N,
        t_per_split,
        stride_xt,
        stride_wn,
        stride_gxt,
        stride_gws_split,
        stride_gws_n,
        BLOCK_T: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
        ROUND: tl.constexpr,
        PRECISION: tl.constexpr,
        HAS_GP: tl.constexpr,
        DEBUG: tl.constexpr,
    ):
        """GP-4 (strict): the P2 prototype's backward in its specified orientation (g_W[N, C] +=
        G_p^T xn) on a (column tile, token split) grid, reading coef and G_p from GP-4p."""
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)
        cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        col_ok = cols < K
        n_idx = tl.arange(0, BLOCK_N)
        n_ok = n_idx < N
        if HAS_GP:
            w = tl.load(
                w_ptr + n_idx[:, None] * stride_wn + cols[None, :],
                mask=n_ok[:, None] & col_ok[None, :],
                other=0.0,
            )
            w_op = _tf32_round(w, ROUND)  # [BLOCK_N, BLOCK_C], loaded once per column slice
            gw_acc = tl.zeros([BLOCK_N, BLOCK_C], dtype=tl.float32)
        t_begin = pid_m * t_per_split
        t_stop = t_begin + t_per_split
        for t0 in range(t_begin, t_stop, BLOCK_T):
            rows = t0 + tl.arange(0, BLOCK_T)
            row_ok = (rows < T) & (rows < t_stop)
            tc_ok = row_ok[:, None] & col_ok[None, :]
            s = tl.load(s_ptr + rows, mask=row_ok, other=1.0)
            coef = tl.load(coef_ptr + rows, mask=row_ok, other=0.0)
            xf = tl.load(
                x_ptr + rows[:, None] * stride_xt + cols[None, :], mask=tc_ok, other=0.0
            ).to(tl.float32)
            if HAS_GP:
                gp_op = tl.load(
                    gpr_ptr + rows[:, None] * BLOCK_N + n_idx[None, :],
                    mask=row_ok[:, None],
                    other=0.0,
                )
                g_xn = tl.dot(gp_op, w_op, input_precision=PRECISION, out_dtype=tl.float32)
                gx = s[:, None] * g_xn - coef[:, None] * xf
            else:
                gx = 0.0 - coef[:, None] * xf  # +0 where coef * xf is zero, as 0 - coef * xf
            tl.store(
                gx_ptr + rows[:, None] * stride_gxt + cols[None, :],
                gx.to(tl.bfloat16, fp_downcast_rounding="rtne"),
                mask=tc_ok,
            )
            if DEBUG:
                tl.store(gx32_ptr + rows[:, None] * stride_gxt + cols[None, :], gx, mask=tc_ok)
                if HAS_GP:
                    tl.store(gxn_ptr + rows[:, None] * stride_gxt + cols[None, :], g_xn, mask=tc_ok)
            if HAS_GP:
                xn_op = _tf32_round(xf * s[:, None], ROUND)  # xn recomputed: fl32(xf * s)
                gw_acc = tl.dot(
                    tl.trans(gp_op), xn_op, gw_acc, input_precision=PRECISION, out_dtype=tl.float32
                )
        if HAS_GP:
            base = gws_ptr + pid_m * stride_gws_split
            tl.store(
                base + n_idx[:, None] * stride_gws_n + cols[None, :],
                gw_acc,
                mask=n_ok[:, None] & col_ok[None, :],
            )

    @triton.jit
    def _glm_wgrad_reduce_kernel(
        part_ptr,
        gw_ptr,
        N,
        K,
        NB,
        NU,
        NPROG,
        stride_gwn,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
        RBC: tl.constexpr,
    ):
        """GP-4r (TF32): g_W of RBC columns of tile c = the partials of programs p_first(c) to
        p_last(c) added in program (token) order from +0.0."""
        pid = tl.program_id(0)
        SUB: tl.constexpr = BLOCK_C // RBC
        c = pid // SUB
        j = (pid % SUB) * RBC + tl.arange(0, RBC)
        cols = c * BLOCK_C + j
        n_idx = tl.arange(0, BLOCK_N)
        ok = (n_idx[:, None] < N) & (cols[None, :] < K)
        p_first, p_last = _glm_gp4_tile_programs(c, NB, NU, NPROG)
        offs = n_idx[:, None] * BLOCK_C + j[None, :]
        slab = N * BLOCK_C
        acc = tl.zeros([BLOCK_N, RBC], dtype=tl.float32)
        for q in range(p_first, p_last + 1):
            acc += tl.load(part_ptr + _glm_gp4_slot(q, c) * slab + offs, mask=ok, other=0.0)
        tl.store(gw_ptr + n_idx[:, None] * stride_gwn + cols[None, :], acc, mask=ok)

    @triton.jit
    def _glm_wgrad_reduce_split_kernel(
        ws_ptr, gw_ptr, NK, stride_split, SPLIT: tl.constexpr, BLOCK: tl.constexpr
    ):
        """GP-4r (strict): g_W = ((0 + w_0) + w_1) + ... over the token splits in index order,
        on the flat [N * K] layout."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        ok = offs < NK
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for i in tl.static_range(0, SPLIT):
            acc += tl.load(ws_ptr + i * stride_split + offs, mask=ok, other=0.0)
        tl.store(gw_ptr + offs, acc, mask=ok)

    _GLM_KERNELS_DEFINED = True


class _GLMKernels(NamedTuple):
    """The JIT functions of the GLM projection pair, by role."""

    rms_fwd: object
    proj_fwd: object
    proj_fwd_strict: object
    proj_reduce: object
    bwd_rowscale: object
    bwd: object
    bwd_strict: object
    wgrad_reduce: object
    wgrad_reduce_split: object


def _glm_kernels() -> _GLMKernels:
    """Fail closed: raise instead of returning kernels that this process never defined."""
    if not (fmk.is_triton_available() and _GLM_KERNELS_DEFINED):
        raise RuntimeError("The Triton GLM projection pair needs Triton, which is unavailable.")
    return _GLMKernels(
        rms_fwd=_glm_rms_fwd_kernel,
        proj_fwd=_glm_proj_fwd_kernel,
        proj_fwd_strict=_glm_proj_fwd_strict_kernel,
        proj_reduce=_glm_proj_reduce_kernel,
        bwd_rowscale=_glm_proj_bwd_rowscale_kernel,
        bwd=_glm_proj_bwd_kernel,
        bwd_strict=_glm_proj_bwd_strict_kernel,
        wgrad_reduce=_glm_wgrad_reduce_kernel,
        wgrad_reduce_split=_glm_wgrad_reduce_split_kernel,
    )


# ============================================================================
# The GLM projection pair: split plans (pure Python)
# ============================================================================


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _align(n: int, alignment: int) -> int:
    return _cdiv(n, alignment) * alignment


def _gp2_chain_length(K: int) -> int:
    """GP-2's K columns per chain: BLOCK_K * cdiv(cdiv(K, n), BLOCK_K), with
    n = min(SPLIT, cdiv(K, BLOCK_K))."""
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    block_k = GP2_CONFIG["BLOCK_K"]
    n_split = min(GP2_CONFIG["SPLIT"], _cdiv(K, block_k))
    return block_k * _cdiv(_cdiv(K, n_split), block_k)


def gp2_split_bounds(K: int) -> Tuple[Tuple[int, int], ...]:
    """GP-2's split-K chains [start, stop) over K, none empty: chain i covers [i * k, min((i + 1)
    * k, K)) for i < cdiv(K, k), k = _gp2_chain_length(K). GP-3's SPLIT is their number."""
    per_chain = _gp2_chain_length(K)
    return tuple((start, min(start + per_chain, K)) for start in range(0, K, per_chain))


def _gp2_strict_split_length(K: int) -> int:
    """The strict GP-2's K columns per split: BLOCK_K * cdiv(cdiv(K, SPLIT), BLOCK_K)."""
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    block_k = GP2_STRICT_CONFIG["BLOCK_K"]
    return block_k * _cdiv(_cdiv(K, GP2_STRICT_CONFIG["SPLIT"]), block_k)


def gp2_strict_split_bounds(K: int) -> Tuple[Tuple[int, int], ...]:
    """The strict GP-2's SPLIT (16) splits [start, stop) over K, the P2 prototype's: empty
    splits past K are kept (their zero partials are reduced too)."""
    per_split = _gp2_strict_split_length(K)
    return tuple(
        (min(i * per_split, K), min((i + 1) * per_split, K))
        for i in range(GP2_STRICT_CONFIG["SPLIT"])
    )


class _GP4Partition(NamedTuple):
    """GP-4's static partition at (T, K): token blocks, column tiles, units and programs."""

    T_pad: int
    NB: int
    NC: int
    NU: int
    NPROG: int
    NSLOT: int


def _gp4_partition(T: int, K: int) -> _GP4Partition:
    """NB = cdiv(T, BLOCK_T) token blocks, NC = cdiv(K, BLOCK_C) column tiles, NU = NB * NC units,
    NPROG = min(GP4_NUM_SMS * GP4_OCC, NU) programs and NSLOT = NPROG + NC - 1 partial slots."""
    if T < 1 or K < 1:
        raise ValueError(f"T and K must be >= 1, got {T} and {K}")
    block_t, block_c = GP4_CONFIG["BLOCK_T"], GP4_CONFIG["BLOCK_C"]
    n_blocks, n_tiles = _cdiv(T, block_t), _cdiv(K, block_c)
    n_units = n_blocks * n_tiles
    n_prog = min(GP4_NUM_SMS * GP4_OCC, n_units)
    return _GP4Partition(
        T_pad=n_blocks * block_t,
        NB=n_blocks,
        NC=n_tiles,
        NU=n_units,
        NPROG=n_prog,
        NSLOT=n_prog + n_tiles - 1,
    )


def gp4_wgrad_split_bounds(
    T: int, K: int
) -> Tuple[Tuple[Tuple[int, int], Tuple[Tuple[int, int], ...]], ...]:
    """GP-4's wgrad pieces: for each BLOCK_C column tile c, in order, ((first column, stop column),
    token ranges [start, stop) of its pieces in the order GP-4r adds them). Program q owns the
    units [q * NU // NPROG, (q + 1) * NU // NPROG) of u = c * NB + b; a piece is one program's run
    inside one tile, and tile c's pieces are those of programs p_first(c) to p_last(c). The same
    arithmetic as the kernels' plan helpers (_glm_gp4_program_plan, _glm_gp4_piece_blocks,
    _glm_gp4_tile_programs), which a CPU test runs interpreted against this function."""
    part = _gp4_partition(T, K)
    block_t, block_c = GP4_CONFIG["BLOCK_T"], GP4_CONFIG["BLOCK_C"]
    n_blocks, n_units, n_prog = part.NB, part.NU, part.NPROG
    tiles = []
    for c in range(part.NC):
        p_first = ((c * n_blocks + 1) * n_prog - 1) // n_units
        p_last = ((c + 1) * n_blocks * n_prog - 1) // n_units
        pieces = []
        for q in range(p_first, p_last + 1):
            u_lo, u_hi = (q * n_units) // n_prog, ((q + 1) * n_units) // n_prog
            b_lo = max(u_lo, c * n_blocks) - c * n_blocks
            b_hi = min(u_hi, (c + 1) * n_blocks) - c * n_blocks
            if b_hi > b_lo:
                pieces.append((b_lo * block_t, min(b_hi * block_t, T)))
        tiles.append(((c * block_c, min((c + 1) * block_c, K)), tuple(pieces)))
    return tuple(tiles)


def _gp4_strict_split_length(T: int) -> int:
    """The strict GP-4's tokens per split: BLOCK_T * cdiv(cdiv(T, SPLIT_M), BLOCK_T)."""
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    block_t = GP4_STRICT_CONFIG["BLOCK_T"]
    return block_t * _cdiv(_cdiv(T, GP4_STRICT_CONFIG["SPLIT_M"]), block_t)


def gp4_strict_wgrad_split_bounds(T: int) -> Tuple[Tuple[int, int], ...]:
    """The strict GP-4's SPLIT_M (2) token splits [start, stop), the P2 prototype's: an empty
    split past T is kept (its zero partial is reduced too)."""
    per_split = _gp4_strict_split_length(T)
    return tuple(
        (min(i * per_split, T), min((i + 1) * per_split, T))
        for i in range(GP4_STRICT_CONFIG["SPLIT_M"])
    )


# ============================================================================
# The GLM projection pair: workspace layouts (fp32 elements of one 1-D workspace per call)
# ============================================================================


class _FwdLayout(NamedTuple):
    """The forward workspace: (offset, length) of W_r, the chain partials and the TMA scratch."""

    wr: Tuple[int, int]
    part: Tuple[int, int]
    tma: Tuple[int, int]
    total: int
    n_chain: int
    k_per_chain: int


class _BwdLayout(NamedTuple):
    """The backward workspace: (offset, length) of coef, T(G_p), T(G_p)^T and the wgrad partials."""

    coef: Tuple[int, int]
    gpr: Tuple[int, int]
    gprt: Tuple[int, int]
    part: Tuple[int, int]
    total: int


def _fwd_layout(T: int, K: int, mode: str) -> _FwdLayout:
    """TF32: [W_r BLOCK_N * K][partials n_chain * T * BLOCK_N][TMA n_CTA x 1 KiB]; strict: [partials
    16 * T * BLOCK_N]; every section starts on a 1 KiB boundary."""
    if mode == "tf32":
        n_chain, k_per_chain = len(gp2_split_bounds(K)), _gp2_chain_length(K)
        wr = (0, BLOCK_N * K)
        part = (_align(wr[1], _FWD_ALIGN), n_chain * T * BLOCK_N)
        n_cta = _cdiv(T, GP2_CONFIG["BLOCK_M"]) * n_chain
        tma = (_align(part[0] + part[1], _FWD_ALIGN), _cdiv(n_cta * _TMA_SCRATCH_PER_CTA, 4))
    else:
        n_chain, k_per_chain = GP2_STRICT_CONFIG["SPLIT"], _gp2_strict_split_length(K)
        wr = (0, 0)
        part = (0, n_chain * T * BLOCK_N)
        tma = (_align(part[1], _FWD_ALIGN), 0)
    total = max(_align(tma[0] + tma[1], _FWD_ALIGN), _FWD_ALIGN)
    return _FwdLayout(wr, part, tma, total, n_chain, k_per_chain)


def _bwd_layout(T: int, K: int, mode: str) -> _BwdLayout:
    """[coef T_pad][T(G_p) T_pad * BLOCK_N][T(G_p)^T BLOCK_N * T_pad][partials]: TF32 NSLOT
    slots of N * BLOCK_C, strict SPLIT_M * N * K; every section starts on a 128 B boundary."""
    part_ = _gp4_partition(T, K)
    t_pad = part_.T_pad
    coef = (0, t_pad)
    gpr = (_align(t_pad, _BWD_ALIGN), t_pad * BLOCK_N)
    gprt = (_align(gpr[0] + gpr[1], _BWD_ALIGN), BLOCK_N * t_pad)
    if mode == "tf32":
        n_part = part_.NSLOT * PROJ_N * GP4_CONFIG["BLOCK_C"]
    else:
        n_part = GP4_STRICT_CONFIG["SPLIT_M"] * PROJ_N * K
    part = (_align(gprt[0] + gprt[1], _BWD_ALIGN), n_part)
    return _BwdLayout(coef, gpr, gprt, part, _align(part[0] + part[1], _BWD_ALIGN))


def _section(ws: Tensor, section: Tuple[int, int]) -> Tensor:
    return ws.narrow(0, section[0], max(section[1], 1))


# ============================================================================
# The GLM projection pair: precision predicate and support
# ============================================================================


def _read_fp32_matmul_precision() -> str:
    """torch's fp32 matmul precision for CUDA; a seam the CPU tests patch."""
    return torch.backends.cuda.matmul.fp32_precision


def _matmul_mode() -> Tuple[Optional[str], Optional[str]]:
    """("tf32" or "ieee", None), or (None, reason).

    "tf32" selects the TF32 forms, except under NVIDIA_TF32_OVERRIDE=0, which makes cuBLAS run
    IEEE behind that reading; "ieee" and "none" select the strict forms.
    """
    try:
        precision = _read_fp32_matmul_precision()
    except Exception as error:
        return None, f"reading the fp32 matmul precision raised {type(error).__name__}"
    if precision == "tf32":
        if os.environ.get("NVIDIA_TF32_OVERRIDE") == "0":
            return "ieee", None
        return "tf32", None
    if precision in ("ieee", "none"):
        return "ieee", None
    return None, f"fp32 matmul precision reads {precision!r}"


@functools.lru_cache(maxsize=None)
def _device_capability(index: int) -> Tuple[int, int]:
    """(major, minor) of CUDA device index, read once per device."""
    major, minor = torch.cuda.get_device_capability(index)
    return int(major), int(minor)


def _layout_refusal(x2d: Tensor, weight: Tensor) -> Optional[str]:
    """Rows 3-10 of supported(): dtypes, shapes, strides and alignment, from metadata only."""
    if x2d.dtype != torch.bfloat16:
        return f"x2d is {x2d.dtype}, expected bfloat16"
    if weight.dtype != torch.float32:
        return f"weight is {weight.dtype}, expected float32"
    if x2d.dim() != 2 or weight.dim() != 2 or weight.shape[1] != x2d.shape[1]:
        return (
            f"shapes {tuple(x2d.shape)} and {tuple(weight.shape)} are not [T, K] and [{PROJ_N}, K]"
        )
    (T, K), N = x2d.shape, weight.shape[0]
    if N != PROJ_N:
        return f"N = {N}; validated at {PROJ_N}"
    if K == 0:
        return "K = 0: no columns"
    if K % GP4_CONFIG["BLOCK_C"]:
        return f"K = {K} is not a multiple of {GP4_CONFIG['BLOCK_C']}"
    if T == 0:
        return "no tokens"
    if T * K >= 2**31:
        return "T x K exceeds int32 offsets"
    # The kernels' other int32 offsets grow with K or T alone: the strict wgrad partials span
    # 2 * 24 * K (and bound W_r's 32 * K and the TF32 partial slots), the strict forward partials
    # 16 * T * 32 (and bound GP-4p's 32 * T_pad).
    if GP4_STRICT_CONFIG["SPLIT_M"] * PROJ_N * K >= 2**31:
        return f"K = {K} exceeds the int32 offsets of the weight-gradient partials"
    if GP2_STRICT_CONFIG["SPLIT"] * T * BLOCK_N >= 2**31:
        return f"T = {T} exceeds the int32 offsets of the forward partials"
    # A one-row view is contiguous whatever its row stride; the forward's TMA descriptor needs it.
    if not x2d.is_contiguous() or x2d.stride(0) != K:
        return "x2d is not contiguous"
    if not weight.is_contiguous():
        return "weight is not contiguous"
    if x2d.data_ptr() % 16:
        return "x2d is not 16-byte aligned"
    if weight.data_ptr() % 16:
        return "weight is not 16-byte aligned"
    return None


def _refusal(x2d: Tensor, weight: Tensor) -> Optional[str]:
    """None when the pair supports (x2d, weight), else the first reason it does not.

    Reads host metadata and Python state only: no synchronization, allocation or launch, so it is
    legal during CUDA-graph capture, and it does not depend on grad mode or requires_grad.
    """
    if x2d.device.type != "cuda" or weight.device != x2d.device:
        return f"inputs on {x2d.device} and {weight.device}, not one CUDA device"
    capability = _device_capability(x2d.device.index)
    if capability not in GLM_PAIR_DEVICE_CAPABILITIES:
        return f"compute capability {capability} is not validated"
    reason = _layout_refusal(x2d, weight)
    if reason is not None:
        return reason
    mode, reason = _matmul_mode()
    if mode is None:
        return reason
    if not (fmk.is_triton_available() and _GLM_KERNELS_DEFINED):
        return "Triton is unavailable"
    return None


# ============================================================================
# The GLM projection pair: host launchers
# ============================================================================


class _Launch(NamedTuple):
    """One kernel launch: its shipped name, JIT function, grid, runtime arguments, constexprs and
    launch options; tma marks the launch that needs the TMA scratch allocator."""

    name: str
    kernel: object
    grid: Tuple[int, ...]
    args: tuple
    constexprs: Dict[str, object]
    options: Dict[str, object]
    tma: bool = False


def _options(config: Dict[str, object], **extra: object) -> Dict[str, object]:
    options = dict(
        num_warps=config["num_warps"], num_stages=config["num_stages"], enable_fp_fusion=False
    )
    options.update(extra)
    return options


def _check_mode(rounding: str, mode: str, xround: Optional[int]) -> None:
    if rounding not in _ROUND_MODE:
        raise ValueError(f"rounding must be one of {sorted(_ROUND_MODE)}, got {rounding!r}")
    if mode not in ("tf32", "ieee"):
        raise ValueError(f"mode must be 'tf32' or 'ieee', got {mode!r}")
    if xround is not None and (xround != 1 or mode != "tf32" or rounding != "rn"):
        raise ValueError("_xround=1 applies to the TF32 'rn' mode only")


def _fwd_launches(
    kernels: _GLMKernels,
    x2d: Tensor,
    weight: Tensor,
    eps: float,
    p: Tensor,
    s: Tensor,
    ws: Tensor,
    *,
    rounding: str,
    mode: str,
    xround: Optional[int] = None,
    s_in: Optional[Tensor] = None,
) -> List[_Launch]:
    """The forward's launches in order (GP-1, GP-2, GP-3). Creates views of ws only."""
    T, K = x2d.shape
    layout = _fwd_layout(T, K, mode)
    s_read = s if s_in is None else s_in
    inv_k = 1.0 / K
    gp1 = GP1_CONFIG
    n_row = _cdiv(T, gp1["ROWS"])
    part = _section(ws, layout.part)
    gp1_cx = dict(
        ROWS=gp1["ROWS"],
        BLOCK_K=gp1["BLOCK_K"],
        EVEN=int(T % gp1["ROWS"] == 0 and K % gp1["BLOCK_K"] == 0),
        WR_BLOCK=gp1["WR_BLOCK"],
        BLOCK_N=BLOCK_N,
        PDL=1,
    )
    launches = []
    if mode == "tf32":
        wr = _section(ws, layout.wr)
        launches.append(
            _Launch(
                "_glm_rms_fwd_kernel",
                kernels.rms_fwd,
                (n_row + _cdiv(K, 16 * gp1["WR_BLOCK"]),),
                (
                    x2d,
                    s,
                    weight,
                    wr,
                    T,
                    K,
                    PROJ_N,
                    x2d.stride(0),
                    weight.stride(0),
                    inv_k,
                    eps,
                    n_row,
                ),
                dict(gp1_cx, WPRE=1, WROUND=_ROUND_MODE[rounding]),
                _options(gp1),
            )
        )
        gp2 = GP2_CONFIG
        if rounding == "rn":
            xn_round = _XN_CVT if xround is None else xround
        else:
            xn_round = _ROUND_MODE[rounding]
        launches.append(
            _Launch(
                "_glm_proj_fwd_kernel",
                kernels.proj_fwd,
                (_cdiv(T, gp2["BLOCK_M"]), layout.n_chain),
                (
                    x2d,
                    s_read,
                    wr,
                    part,
                    T,
                    K,
                    layout.k_per_chain,
                    x2d.stride(0),
                    T * BLOCK_N,
                    BLOCK_N,
                ),
                dict(
                    BLOCK_M=gp2["BLOCK_M"],
                    BLOCK_N=BLOCK_N,
                    BLOCK_K=gp2["BLOCK_K"],
                    ROUND=xn_round,
                    EVEN_M=int(T % gp2["BLOCK_M"] == 0),
                    ORDER=gp2["ORDER"],
                    NS=gp2["num_stages"],
                    PDL=1,
                    PRECISION="tf32",
                ),
                _options(gp2, launch_pdl=True),
                tma=True,
            )
        )
    else:
        launches.append(
            _Launch(
                "_glm_rms_fwd_kernel",
                kernels.rms_fwd,
                (n_row,),
                (
                    x2d,
                    s,
                    weight,
                    part,
                    T,
                    K,
                    PROJ_N,
                    x2d.stride(0),
                    weight.stride(0),
                    inv_k,
                    eps,
                    n_row,
                ),
                dict(gp1_cx, WPRE=0, WROUND=0),
                _options(gp1),
            )
        )
        gp2 = GP2_STRICT_CONFIG
        launches.append(
            _Launch(
                "_glm_proj_fwd_strict_kernel",
                kernels.proj_fwd_strict,
                (_cdiv(T, gp2["BLOCK_M"]), layout.n_chain),
                (
                    x2d,
                    s_read,
                    weight,
                    part,
                    T,
                    K,
                    PROJ_N,
                    layout.k_per_chain,
                    x2d.stride(0),
                    weight.stride(0),
                    T * BLOCK_N,
                    BLOCK_N,
                ),
                dict(
                    BLOCK_M=gp2["BLOCK_M"],
                    BLOCK_N=BLOCK_N,
                    BLOCK_K=gp2["BLOCK_K"],
                    ROUND=0,
                    PRECISION=gp2["PRECISION"],
                ),
                _options(gp2),
            )
        )
    gp3 = GP3_CONFIG
    launches.append(
        _Launch(
            "_glm_proj_reduce_kernel",
            kernels.proj_reduce,
            (_cdiv(T, gp3["BLOCK_M"]),),
            (part, p, T, PROJ_N, T * BLOCK_N, BLOCK_N, p.stride(0)),
            dict(
                SPLIT=layout.n_chain,
                BLOCK_M=gp3["BLOCK_M"],
                BLOCK_N=BLOCK_N,
                EVEN=int(T % gp3["BLOCK_M"] == 0),
                PDL=1,
            ),
            _options(gp3, launch_pdl=True),
        )
    )
    return launches


def _bwd_launches(
    kernels: _GLMKernels,
    x2d: Tensor,
    weight: Tensor,
    s: Tensor,
    p: Tensor,
    grad_p: Optional[Tensor],
    grad_s: Optional[Tensor],
    g_x: Tensor,
    g_w: Optional[Tensor],
    ws: Tensor,
    extras: Dict[str, Tensor],
    *,
    rounding: str,
    mode: str,
    debug: bool,
    xround: Optional[int] = None,
) -> List[_Launch]:
    """The backward's launches in order (GP-4p, GP-4 and, with G_p, GP-4r). grad_s is [T] or
    None; extras holds the debug outputs. Creates views of ws only."""
    T, K = x2d.shape
    layout = _bwd_layout(T, K, mode)
    has_gp, has_gs = grad_p is not None, grad_s is not None
    coef, gpr, gprt, part = (
        _section(ws, layout.coef),
        _section(ws, layout.gpr),
        _section(ws, layout.gprt),
        _section(ws, layout.part),
    )
    t_pad = _gp4_partition(T, K).T_pad
    g_round = _ROUND_MODE[rounding] if mode == "tf32" else 0
    gp_arg = grad_p if has_gp else p
    gs_arg = grad_s if has_gs else s
    gp4p = GP4P_CONFIG
    launches = [
        _Launch(
            "_glm_proj_bwd_rowscale_kernel",
            kernels.bwd_rowscale,
            (_cdiv(t_pad, gp4p["BLOCK_R"]),),
            (
                s,
                p,
                gp_arg,
                gs_arg,
                coef,
                gpr,
                gprt,
                extras.get("g_s", coef),
                T,
                t_pad,
                1.0 / K,
                p.stride(0),
                gp_arg.stride(0),
                gp_arg.stride(1),
                gs_arg.stride(0),
            ),
            dict(
                BLOCK_R=gp4p["BLOCK_R"],
                BLOCK_N=BLOCK_N,
                N=PROJ_N,
                ROUND=g_round,
                GT=GP4_CONFIG["GT"],
                HAS_GP=int(has_gp),
                HAS_GS=int(has_gs),
                DEBUG=int(debug),
            ),
            _options(gp4p),
        )
    ]
    gx32 = extras.get("g_x32", part)
    gxn = extras.get("g_xn", part)
    if mode == "tf32":
        gp4 = GP4_CONFIG
        partition = _gp4_partition(T, K)
        if rounding == "rn":
            xn_round = _XN_CVT if xround is None else xround
        else:
            xn_round = _ROUND_MODE[rounding]
        maxnreg = gp4["maxnreg_dbg"] if debug else gp4["maxnreg"]
        launches.append(
            _Launch(
                "_glm_proj_bwd_kernel",
                kernels.bwd,
                (partition.NPROG,),
                (
                    x2d,
                    s,
                    coef,
                    gpr,
                    gprt,
                    weight,
                    g_x,
                    gx32,
                    gxn,
                    part,
                    T,
                    K,
                    PROJ_N,
                    t_pad,
                    partition.NB,
                    partition.NU,
                    partition.NPROG,
                    x2d.stride(0),
                    weight.stride(0),
                    g_x.stride(0),
                ),
                dict(
                    BLOCK_T=gp4["BLOCK_T"],
                    BLOCK_N=BLOCK_N,
                    BLOCK_C=gp4["BLOCK_C"],
                    ROUND=g_round,
                    XROUND=xn_round,
                    PRECISION="tf32",
                    HAS_GP=int(has_gp),
                    DEBUG=int(debug),
                    GT=gp4["GT"],
                    ANCHOR=gp4["ANCHOR"],
                    NS=gp4["num_stages"],
                ),
                _options(gp4, **({} if maxnreg is None else dict(maxnreg=maxnreg))),
            )
        )
        if has_gp:
            gp4r = GP4R_CONFIG
            launches.append(
                _Launch(
                    "_glm_wgrad_reduce_kernel",
                    kernels.wgrad_reduce,
                    (partition.NC * (gp4["BLOCK_C"] // gp4r["RBC"]),),
                    (
                        part,
                        g_w,
                        PROJ_N,
                        K,
                        partition.NB,
                        partition.NU,
                        partition.NPROG,
                        g_w.stride(0),
                    ),
                    dict(BLOCK_N=BLOCK_N, BLOCK_C=gp4["BLOCK_C"], RBC=gp4r["RBC"]),
                    _options(gp4r),
                )
            )
    else:
        gp4 = GP4_STRICT_CONFIG
        launches.append(
            _Launch(
                "_glm_proj_bwd_strict_kernel",
                kernels.bwd_strict,
                (_cdiv(K, gp4["BLOCK_C"]), gp4["SPLIT_M"]),
                (
                    x2d,
                    s,
                    coef,
                    gpr,
                    weight,
                    g_x,
                    gx32,
                    gxn,
                    part,
                    T,
                    K,
                    PROJ_N,
                    _gp4_strict_split_length(T),
                    x2d.stride(0),
                    weight.stride(0),
                    g_x.stride(0),
                    PROJ_N * K,
                    K,
                ),
                dict(
                    BLOCK_T=gp4["BLOCK_T"],
                    BLOCK_N=BLOCK_N,
                    BLOCK_C=gp4["BLOCK_C"],
                    ROUND=0,
                    PRECISION="ieee",
                    HAS_GP=int(has_gp),
                    DEBUG=int(debug),
                ),
                _options(gp4),
            )
        )
        if has_gp:
            gp4r = GP4R_STRICT_CONFIG
            launches.append(
                _Launch(
                    "_glm_wgrad_reduce_split_kernel",
                    kernels.wgrad_reduce_split,
                    (_cdiv(PROJ_N * K, gp4r["BLOCK"]),),
                    (part, g_w, PROJ_N * K, PROJ_N * K),
                    dict(SPLIT=gp4["SPLIT_M"], BLOCK=gp4r["BLOCK"]),
                    _options(gp4r),
                )
            )
    return launches


class _TmaScratch:
    """A Triton allocator backed by the TMA section of the forward's per-call workspace."""

    __slots__ = ("_buffer",)

    def __init__(self, buffer: Tensor):
        self._buffer = buffer

    def __call__(self, size: int, alignment: int, stream: Optional[int]) -> Tensor:
        del stream
        reserved = self._buffer.numel() * self._buffer.element_size()
        if size > reserved:
            raise RuntimeError(
                f"Triton asked for {size} B of TMA descriptor scratch; the workspace reserves "
                f"{reserved} B ({_TMA_SCRATCH_PER_CTA} B per program)."
            )
        if self._buffer.data_ptr() % max(int(alignment), 1):
            raise RuntimeError(f"The workspace's TMA section is not {alignment}-byte aligned.")
        return self._buffer


def _launch_with_scratch(launch: _Launch, scratch: Tensor):
    """Launch with the TMA scratch allocator set; called in a copied context, which scopes it."""
    triton.set_allocator(_TmaScratch(scratch))
    return launch.kernel[launch.grid](*launch.args, **launch.constexprs, **launch.options)


def _run(
    launches: List[_Launch], scratch: Optional[Tensor], compiled: Optional[Dict[str, object]]
) -> None:
    """Launch in order on torch's current stream of the current device, without synchronizing."""
    for launch in launches:
        if launch.tma:
            handle = contextvars.copy_context().run(_launch_with_scratch, launch, scratch)
        else:
            handle = launch.kernel[launch.grid](*launch.args, **launch.constexprs, **launch.options)
        if compiled is not None:
            compiled[launch.name] = handle


def _glm_proj_fwd(
    x2d: Tensor,
    weight: Tensor,
    eps: float,
    *,
    rounding: str,
    mode: str,
    debug: bool = False,
    _compiled: Optional[Dict[str, object]] = None,
    _xround: Optional[int] = None,
    _s: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """The pair's forward: (p fp32 [T, 24], s fp32 [T, 1]) from x2d bf16 [T, K] and weight fp32
    [24, K]. mode is "tf32" or "ieee" (strict); rounding "rn" or "rz" (TF32 mode only).

    debug: the forward kernels have no debug build, so it launches the same kernels. _compiled,
    when a dict, receives each launch's compiled handle by kernel name; _xround=1 rounds xn by
    MODE 1 instead of the cvt under "rn"; _s, a contiguous fp32 [T, 1] tensor, is read by GP-2 in
    place of GP-1's s and returned as s (GP-1 still runs). The Function passes none of them.
    """
    del debug
    _check_mode(rounding, mode, _xround)
    kernels = _glm_kernels()
    T, K = x2d.shape
    if _s is not None and (
        _s.dtype != torch.float32
        or tuple(_s.shape) != (T, 1)
        or not _s.is_contiguous()
        or _s.device != x2d.device
    ):
        raise ValueError(f"_s must be a contiguous fp32 [{T}, 1] tensor on {x2d.device}")
    device = x2d.device
    p = torch.empty(T, PROJ_N, dtype=torch.float32, device=device)
    s = torch.empty(T, 1, dtype=torch.float32, device=device)
    layout = _fwd_layout(T, K, mode)
    ws = torch.empty(layout.total, dtype=torch.float32, device=device)
    launches = _fwd_launches(
        kernels,
        x2d,
        weight,
        float(eps),
        p,
        s,
        ws,
        rounding=rounding,
        mode=mode,
        xround=_xround,
        s_in=_s,
    )
    with torch.cuda.device(device):  # launch on x2d's device even when it is not the current one
        _run(launches, ws.narrow(0, layout.tma[0], layout.tma[1]), _compiled)
    return p, (s if _s is None else _s)


def _glm_proj_bwd(
    x2d: Tensor,
    weight: Tensor,
    s: Tensor,
    p: Tensor,
    grad_p: Optional[Tensor],
    grad_s: Optional[Tensor],
    *,
    rounding: str,
    mode: str,
    debug: bool = False,
    _compiled: Optional[Dict[str, object]] = None,
    _xround: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor], Dict[str, Tensor]]:
    """The pair's backward: (g_x bf16 [T, K], g_W fp32 [24, K] or None without grad_p, extras).

    grad_p is fp32 [T, 24] or None; grad_s is [T, 1] (the shape of s) or None. With debug, the
    debug builds run and extras holds "g_x32" (fp32 [T, K]), "g_s" (fp32 [T]) and, with grad_p,
    "g_xn" (fp32 [T, K]); otherwise extras is empty. _compiled and _xround as in _glm_proj_fwd.
    """
    _check_mode(rounding, mode, _xround)
    kernels = _glm_kernels()
    T, K = x2d.shape
    device = x2d.device
    if grad_p is not None and (grad_p.dtype != torch.float32 or tuple(grad_p.shape) != (T, PROJ_N)):
        raise RuntimeError(
            f"The GLM projection pair's backward takes an fp32 [{T}, {PROJ_N}] gradient of p, got "
            f"{grad_p.dtype} {tuple(grad_p.shape)}."
        )
    grad_s_flat = None
    if grad_s is not None:
        if grad_s.dtype != torch.float32 or grad_s.numel() != T:
            raise RuntimeError(
                f"The GLM projection pair's backward takes an fp32 [{T}, 1] gradient of s, got "
                f"{grad_s.dtype} {tuple(grad_s.shape)}."
            )
        grad_s_flat = grad_s.reshape(-1)
    g_x = torch.empty_like(x2d)
    g_w = None
    if grad_p is not None:
        g_w = torch.empty(PROJ_N, K, dtype=torch.float32, device=device)
    extras: Dict[str, Tensor] = {}
    if debug:
        extras["g_x32"] = torch.empty(T, K, dtype=torch.float32, device=device)
        extras["g_s"] = torch.empty(T, dtype=torch.float32, device=device)
        if grad_p is not None:
            extras["g_xn"] = torch.empty(T, K, dtype=torch.float32, device=device)
    ws = torch.empty(_bwd_layout(T, K, mode).total, dtype=torch.float32, device=device)
    launches = _bwd_launches(
        kernels,
        x2d,
        weight,
        s,
        p,
        grad_p,
        grad_s_flat,
        g_x,
        g_w,
        ws,
        extras,
        rounding=rounding,
        mode=mode,
        debug=debug,
        xround=_xround,
    )
    with torch.cuda.device(device):  # launch on x2d's device even when it is not the current one
        _run(launches, None, _compiled)
    return g_x, g_w, extras


# ============================================================================
# The GLM projection pair: autograd Function, public function and op object
# ============================================================================


class FusedGLMProjRms(torch.autograd.Function):
    """The GLM projection pair with autograd: (x2d, weight) -> (p, s).

    The forward reads the fp32 matmul precision once and the backward reuses its mode, so a step
    never mixes the TF32 and strict forms. Saves x2d, weight, s and p through save_for_backward.
    """

    @staticmethod
    def forward(ctx, x2d: Tensor, weight: Tensor, eps: float, rounding: str):
        """Run GP-1..GP-3 and save the backward's inputs."""
        mode, reason = _matmul_mode()
        if mode is None:
            raise RuntimeError(f"The Triton GLM projection pair cannot run: {reason}.")
        ctx.set_materialize_grads(False)
        p, s = _glm_proj_fwd(x2d, weight, eps, rounding=rounding, mode=mode)
        ctx.save_for_backward(x2d, weight, s, p)
        ctx.rounding, ctx.mode = rounding, mode
        return p, s

    @staticmethod
    def backward(ctx, grad_p: Optional[Tensor], grad_s: Optional[Tensor]):
        """Run GP-4p, GP-4 and (with grad_p) GP-4r; None for weight without grad_p."""
        if grad_p is None and grad_s is None:
            return None, None, None, None
        x2d, weight, s, p = ctx.saved_tensors
        # Through the module global, so that a test can count the backward launches.
        grad_x, grad_w, _ = _glm_proj_bwd(
            x2d, weight, s, p, grad_p, grad_s, rounding=ctx.rounding, mode=ctx.mode
        )
        return (
            grad_x if ctx.needs_input_grad[0] else None,
            grad_w if ctx.needs_input_grad[1] else None,
            None,
            None,
        )


def fused_glm_proj_rms(
    x2d: Tensor, weight: Tensor, eps: float, *, rounding: str = "rn"
) -> Tuple[Tensor, Tensor]:
    """The GLM projection pair: (p [T, 24] fp32, s [T, 1] fp32), the drop-in of
    _proj_rms_impl(..., eps_inside_sqrt=True, normalize_before_projection=True). Raises on an
    input it does not support."""
    if rounding not in _ROUND_MODE:
        raise ValueError(f"rounding must be one of {sorted(_ROUND_MODE)}, got {rounding!r}")
    reason = _refusal(x2d, weight)
    if reason is not None:
        raise RuntimeError(f"The Triton GLM projection pair does not support this input: {reason}.")
    return FusedGLMProjRms.apply(x2d, weight, eps, rounding)


class GLMProjRmsOp:
    """Per-op proj_rms with mhc_norm_eps_inside_sqrt and mhc_normalize_before_projection.

    Stateless: holds the TF32 rounding only.

    One object per rounding (glm_proj_rms_op); copy, deepcopy and pickle return that object.
    """

    __slots__ = ("rounding",)

    def __init__(self, rounding: str):
        if rounding not in _ROUND_MODE:
            raise ValueError(f"rounding must be one of {sorted(_ROUND_MODE)}, got {rounding!r}")
        self.rounding = rounding

    @property
    def label(self) -> str:
        """The per-rank fused-ops line's name of this operation."""
        return f"glm-triton(tf32={self.rounding})"

    def refusal(self, x2d: Tensor, weight: Tensor) -> Optional[str]:
        """None when the pair supports (x2d, weight), else the first reason it does not."""
        return _refusal(x2d, weight)

    def supported(self, x2d: Tensor, weight: Tensor) -> bool:
        """True when the pair supports (x2d, weight); host metadata only."""
        return _refusal(x2d, weight) is None

    def __call__(self, x2d: Tensor, weight: Tensor, eps: float) -> Tuple[Tensor, Tensor]:
        return fused_glm_proj_rms(x2d, weight, eps, rounding=self.rounding)

    def __copy__(self) -> "GLMProjRmsOp":
        return self

    def __deepcopy__(self, memo) -> "GLMProjRmsOp":
        return self

    def __reduce__(self):
        return (glm_proj_rms_op, (self.rounding,))


_GLM_PROJ_RMS_OPS = {rounding: GLMProjRmsOp(rounding) for rounding in _ROUND_MODE}


def glm_proj_rms_op(rounding: str) -> GLMProjRmsOp:
    """The op object of a TF32 rounding ("rn" or "rz"): one per rounding, so identity compares."""
    if rounding not in _GLM_PROJ_RMS_OPS:
        raise ValueError(f"rounding must be one of {sorted(_ROUND_MODE)}, got {rounding!r}")
    return _GLM_PROJ_RMS_OPS[rounding]
