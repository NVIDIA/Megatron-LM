# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Triton kernels used by the unified mHC fused path.

Only the Triton kernels that are faster than the cuTile implementations are kept
here:
  - Sinkhorn forward/backward
  - H_aggregate forward
  - H_post_bda forward/backward

The public mHC API lives in ``fused_mhc_kernels.py`` and handles fallback.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor


# ============================================================================
# Sinkhorn-Knopp
# ============================================================================


@triton.autotune(
    configs=[triton.Config({}, num_warps=nw) for nw in (1, 2, 4, 8)],
    key=["HC", "NUM_ITERS"],
)
@triton.jit
def _triton_sinkhorn_fwd_kernel(
    inp_ptr, out_ptr, M_init_ptr,
    N_batch, eps,
    HC: tl.constexpr, NUM_ITERS: tl.constexpr,
):
    """Grid: (N_batch,). Each program handles one [HC, HC] matrix."""
    pid = tl.program_id(0)
    if pid >= N_batch:
        return

    base = pid * HC * HC
    offs_r = tl.arange(0, HC)
    offs_c = tl.arange(0, HC)
    mat_ptrs = base + offs_r[:, None] * HC + offs_c[None, :]

    logits = tl.load(inp_ptr + mat_ptrs).to(tl.float32)
    row_max = tl.max(logits, axis=1)
    M = tl.exp2((logits - row_max[:, None]) * 1.4426950408889634)
    tl.store(M_init_ptr + mat_ptrs, M.to(M_init_ptr.dtype.element_ty))

    for _ in range(NUM_ITERS):
        row_sum = tl.sum(M, axis=1)
        M = M / (row_sum[:, None] + eps)
        col_sum = tl.sum(M, axis=0)
        M = M / (col_sum[None, :] + eps)

    tl.store(out_ptr + mat_ptrs, M.to(out_ptr.dtype.element_ty))


@triton.autotune(
    configs=[triton.Config({}, num_warps=nw) for nw in (1, 2, 4, 8)],
    key=["HC", "NUM_ITERS"],
)
@triton.jit
def _triton_sinkhorn_bwd_kernel(
    grad_out_ptr, M_init_ptr, grad_inp_ptr,
    ws_M_ptr, ws_rs_ptr, ws_cs_ptr,
    N_batch, eps,
    HC: tl.constexpr, NUM_ITERS: tl.constexpr,
):
    """Grid: (N_batch,). Each program handles one [HC, HC] backward."""
    pid = tl.program_id(0)
    if pid >= N_batch:
        return

    base = pid * HC * HC
    M_ws_base = pid * 2 * NUM_ITERS * HC * HC
    v_ws_base = pid * NUM_ITERS
    offs_r = tl.arange(0, HC)
    offs_c = tl.arange(0, HC)
    mat_ptrs = base + offs_r[:, None] * HC + offs_c[None, :]

    M = tl.load(M_init_ptr + mat_ptrs).to(tl.float32)
    for t in range(NUM_ITERS):
        ws_off = M_ws_base + (2 * t) * HC * HC
        tl.store(ws_M_ptr + ws_off + offs_r[:, None] * HC + offs_c[None, :], M)

        row_sum = tl.sum(M, axis=1)
        tl.store(ws_rs_ptr + (v_ws_base + t) * HC + offs_r, row_sum)
        M = M / (row_sum[:, None] + eps)

        ws_off = M_ws_base + (2 * t + 1) * HC * HC
        tl.store(ws_M_ptr + ws_off + offs_r[:, None] * HC + offs_c[None, :], M)

        col_sum = tl.sum(M, axis=0)
        tl.store(ws_cs_ptr + (v_ws_base + t) * HC + offs_c, col_sum)
        M = M / (col_sum[None, :] + eps)

    # M is the final forward output. It is the right value for the first VJP
    # through the last column-normalization step.
    grad = tl.load(grad_out_ptr + mat_ptrs).to(tl.float32)
    for t_rev in range(NUM_ITERS):
        t = NUM_ITERS - 1 - t_rev

        col_s = tl.load(ws_cs_ptr + (v_ws_base + t) * HC + offs_c).to(tl.float32)
        grad = grad / (col_s[None, :] + eps)
        col_corr = tl.sum(grad * M, axis=0)
        grad = grad - col_corr[None, :]
        M = tl.load(
            ws_M_ptr + M_ws_base + (2 * t + 1) * HC * HC
            + offs_r[:, None] * HC + offs_c[None, :]
        ).to(tl.float32)

        row_s = tl.load(ws_rs_ptr + (v_ws_base + t) * HC + offs_r).to(tl.float32)
        grad = grad / (row_s[:, None] + eps)
        row_corr = tl.sum(grad * M, axis=1)
        grad = grad - row_corr[:, None]
        M = tl.load(
            ws_M_ptr + M_ws_base + (2 * t) * HC * HC
            + offs_r[:, None] * HC + offs_c[None, :]
        ).to(tl.float32)

    M_init = tl.load(M_init_ptr + mat_ptrs).to(tl.float32)
    grad = grad * M_init
    tl.store(grad_inp_ptr + mat_ptrs, grad.to(grad_inp_ptr.dtype.element_ty))


def _triton_sinkhorn_fwd(
    input_logits: Tensor, num_iterations: int, eps: float = 1e-8
) -> Tuple[Tensor, Tensor]:
    original_shape = input_logits.shape
    hc = original_shape[-1]
    N_batch = input_logits.numel() // (hc * hc)
    dev = input_logits.device
    out = torch.empty(N_batch, hc, hc, dtype=input_logits.dtype, device=dev)
    M_init = torch.empty(N_batch, hc, hc, dtype=input_logits.dtype, device=dev)
    inp = input_logits.contiguous().view(N_batch, hc, hc)
    _triton_sinkhorn_fwd_kernel[(N_batch,)](
        inp, out, M_init, N_batch, eps, hc, num_iterations,
    )
    return out.view(original_shape), M_init.view(original_shape)


def _triton_sinkhorn_bwd(
    grad_output: Tensor, M_init: Tensor, num_iterations: int, eps: float = 1e-8
) -> Tensor:
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
    _triton_sinkhorn_bwd_kernel[(N_batch,)](
        go, mi, grad_input, ws_M, ws_rs, ws_cs, N_batch, eps, hc, num_iterations,
    )
    return grad_input.view(original_shape)


class TritonFusedSinkhorn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_logits: Tensor, num_iterations: int, eps: float = 1e-6):
        out, M_init = _triton_sinkhorn_fwd(input_logits, num_iterations, eps)
        ctx.save_for_backward(M_init)
        ctx.num_iterations = num_iterations
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (M_init,) = ctx.saved_tensors
        grad_input = _triton_sinkhorn_bwd(grad_output, M_init, ctx.num_iterations, ctx.eps)
        return grad_input, None, None


def triton_fused_sinkhorn(
    input_logits: Tensor, num_iterations: int, eps: float = 1e-6
) -> Tensor:
    return TritonFusedSinkhorn.apply(input_logits, num_iterations, eps)


# ============================================================================
# H_aggregate forward
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": bc, "BLOCK_S": bs}, num_warps=nw)
        for bc in (64, 128, 256, 512)
        for bs in (1, 2, 4, 8)
        for nw in (2, 4, 8)
    ],
    key=["C", "N"],
)
@triton.jit
def _triton_h_agg_fwd_kernel(
    x_ptr, h_ptr, out_ptr,
    sb, C: tl.constexpr, N: tl.constexpr,
    stride_x_s, stride_x_n, stride_x_c,
    BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr,
):
    """out[s, c] = sum_i x[s, i, c] * h[s, i]."""
    pid_s = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_s = offs_s < sb
    mask_c = offs_c < C
    mask_2d = mask_s[:, None] & mask_c[None, :]

    acc = tl.zeros((BLOCK_S, BLOCK_C), dtype=tl.float32)
    for i in tl.static_range(N):
        x_i = tl.load(
            x_ptr + offs_s[:, None] * stride_x_s + i * stride_x_n + offs_c[None, :],
            mask=mask_2d,
            other=0.0,
        ).to(tl.float32)
        h_i = tl.load(h_ptr + offs_s * N + i, mask=mask_s, other=0.0).to(tl.float32)
        acc += h_i[:, None] * x_i
    tl.store(
        out_ptr + offs_s[:, None] * C + offs_c[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_2d,
    )


def _triton_h_aggregate_fwd(x: Tensor, h_pre: Tensor) -> Tensor:
    s, b, n, C = x.shape
    sb = s * b
    out = torch.empty(sb, C, dtype=x.dtype, device=x.device)
    x_flat = x.contiguous().view(sb, n, C)
    h_flat = h_pre.contiguous().view(sb, n)

    grid = lambda META: (triton.cdiv(sb, META["BLOCK_S"]), triton.cdiv(C, META["BLOCK_C"]))
    _triton_h_agg_fwd_kernel[grid](
        x_flat, h_flat, out, sb, C, n,
        x_flat.stride(0), x_flat.stride(1), x_flat.stride(2),
    )
    return out.view(s, b, C)


# ============================================================================
# H_post BDA
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": bc, "BLOCK_S": bs}, num_warps=nw)
        for bc in (64, 128, 256, 512)
        for bs in (1, 2, 4, 8)
        for nw in (2, 4, 8)
    ],
    key=["C", "N"],
)
@triton.jit
def _triton_hpb_fwd_kernel(
    hr_ptr, orig_ptr, hp_ptr, x_ptr, bias_ptr, out_ptr,
    sb, C: tl.constexpr, N: tl.constexpr,
    stride_hr_s, stride_hr_i, stride_hr_j,
    stride_orig_s, stride_orig_n, stride_orig_c,
    stride_out_s, stride_out_n, stride_out_c,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr,
):
    """out = hr @ orig + hp * (x + bias)."""
    pid_s = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_s = offs_s < sb
    mask_c = offs_c < C
    mask_2d = mask_s[:, None] & mask_c[None, :]

    x_tile = tl.load(
        x_ptr + offs_s[:, None] * C + offs_c[None, :],
        mask=mask_2d,
        other=0.0,
    ).to(tl.float32)
    if HAS_BIAS:
        bias_tile = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        x_tile += bias_tile[None, :]

    for i in tl.static_range(N):
        hp_i = tl.load(hp_ptr + offs_s * N + i, mask=mask_s, other=0.0).to(tl.float32)
        out_i = hp_i[:, None] * x_tile

        for j in tl.static_range(N):
            hr_ij = tl.load(
                hr_ptr + offs_s * stride_hr_s + i * stride_hr_i + j * stride_hr_j,
                mask=mask_s,
                other=0.0,
            ).to(tl.float32)
            orig_j = tl.load(
                orig_ptr
                + offs_s[:, None] * stride_orig_s
                + j * stride_orig_n
                + offs_c[None, :],
                mask=mask_2d,
                other=0.0,
            ).to(tl.float32)
            out_i += hr_ij[:, None] * orig_j

        tl.store(
            out_ptr + offs_s[:, None] * stride_out_s + i * stride_out_n + offs_c[None, :],
            out_i.to(out_ptr.dtype.element_ty),
            mask=mask_2d,
        )


def _triton_h_post_bda_fwd(
    h_res: Tensor,
    original_residual: Tensor,
    h_post: Tensor,
    x: Tensor,
    bias: Optional[Tensor],
) -> Tensor:
    s, b, n, C = original_residual.shape
    sb = s * b
    dev = h_res.device
    out = torch.empty(sb, n, C, dtype=h_res.dtype, device=dev)
    hr_flat = h_res.contiguous().view(sb, n, n)
    orig_flat = original_residual.contiguous().view(sb, n, C)
    hp_flat = h_post.contiguous().view(sb, n)
    x_flat = x.contiguous().view(sb, C)

    grid = lambda META: (triton.cdiv(sb, META["BLOCK_S"]), triton.cdiv(C, META["BLOCK_C"]))
    _triton_hpb_fwd_kernel[grid](
        hr_flat, orig_flat, hp_flat, x_flat, bias if bias is not None else x_flat, out,
        sb, C, n,
        hr_flat.stride(0), hr_flat.stride(1), hr_flat.stride(2),
        orig_flat.stride(0), orig_flat.stride(1), orig_flat.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        HAS_BIAS=(bias is not None),
    )
    return out.view(s, b, n, C)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": bc, "BLOCK_S": bs}, num_warps=nw)
        for bc in (64, 128, 256, 512)
        for bs in (1, 2, 4, 8)
        for nw in (2, 4, 8)
    ],
    key=["C", "N"],
)
@triton.jit
def _triton_hpb_bwd_g_x_orig_kernel(
    go_ptr, hr_ptr, hp_ptr, g_orig_ptr, g_x_ptr,
    sb, C: tl.constexpr, N: tl.constexpr,
    stride_go_s, stride_go_n, stride_go_c,
    stride_hr_s, stride_hr_i, stride_hr_j,
    stride_orig_s, stride_orig_n, stride_orig_c,
    BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr,
):
    """g_x = hp @ go, g_orig = hr.T @ go."""
    pid_s = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_s = offs_s < sb
    mask_c = offs_c < C
    mask_2d = mask_s[:, None] & mask_c[None, :]

    g_x_acc = tl.zeros((BLOCK_S, BLOCK_C), dtype=tl.float32)
    for j in tl.static_range(N):
        go_j = tl.load(
            go_ptr + offs_s[:, None] * stride_go_s + j * stride_go_n + offs_c[None, :],
            mask=mask_2d, other=0.0,
        ).to(tl.float32)
        hp_j = tl.load(hp_ptr + offs_s * N + j, mask=mask_s, other=0.0).to(tl.float32)
        g_x_acc += hp_j[:, None] * go_j
    tl.store(
        g_x_ptr + offs_s[:, None] * C + offs_c[None, :],
        g_x_acc.to(g_x_ptr.dtype.element_ty),
        mask=mask_2d,
    )

    for i in tl.static_range(N):
        g_orig_i = tl.zeros((BLOCK_S, BLOCK_C), dtype=tl.float32)
        for j in tl.static_range(N):
            go_j = tl.load(
                go_ptr + offs_s[:, None] * stride_go_s + j * stride_go_n + offs_c[None, :],
                mask=mask_2d, other=0.0,
            ).to(tl.float32)
            hr_ji = tl.load(
                hr_ptr + offs_s * stride_hr_s + j * stride_hr_i + i * stride_hr_j,
                mask=mask_s, other=0.0,
            ).to(tl.float32)
            g_orig_i += hr_ji[:, None] * go_j
        tl.store(
            g_orig_ptr + offs_s[:, None] * stride_orig_s + i * stride_orig_n + offs_c[None, :],
            g_orig_i.to(g_orig_ptr.dtype.element_ty),
            mask=mask_2d,
        )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": bc, "BLOCK_S": bs}, num_warps=nw)
        for bc in (64, 128, 256, 512)
        for bs in (1, 2, 4, 8)
        for nw in (2, 4, 8)
    ],
    key=["C", "N"],
)
@triton.jit
def _triton_hpb_bwd_g_hp_hr_kernel(
    go_ptr, orig_ptr, x_ptr, bias_ptr,
    g_hr_ptr, g_hp_ptr,
    sb, C: tl.constexpr, N: tl.constexpr,
    stride_go_s, stride_go_n, stride_go_c,
    stride_orig_s, stride_orig_n, stride_orig_c,
    stride_hr_s, stride_hr_i, stride_hr_j,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr,
):
    """g_hp = sum_c go*(x+bias), g_hr = go @ orig.T."""
    pid_s = tl.program_id(0)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < sb

    g_hp_acc = tl.zeros((BLOCK_S, N), dtype=tl.float32)
    g_hr_acc = tl.zeros((BLOCK_S, N * N), dtype=tl.float32)

    for c_start in range(0, C, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        mask_2d = mask_s[:, None] & mask_c[None, :]

        x_tile = tl.load(
            x_ptr + offs_s[:, None] * C + offs_c[None, :],
            mask=mask_2d,
            other=0.0,
        ).to(tl.float32)
        if HAS_BIAS:
            bias_tile = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
            x_tile += bias_tile[None, :]

        for i in tl.static_range(N):
            go_i = tl.load(
                go_ptr + offs_s[:, None] * stride_go_s + i * stride_go_n + offs_c[None, :],
                mask=mask_2d,
                other=0.0,
            ).to(tl.float32)
            dot_hp = tl.sum(go_i * x_tile, axis=1)
            g_hp_acc += tl.where(
                tl.arange(0, N)[None, :] == i,
                dot_hp[:, None],
                tl.zeros((BLOCK_S, N), dtype=tl.float32),
            )
            for j in tl.static_range(N):
                orig_j = tl.load(
                    orig_ptr
                    + offs_s[:, None] * stride_orig_s
                    + j * stride_orig_n
                    + offs_c[None, :],
                    mask=mask_2d,
                    other=0.0,
                ).to(tl.float32)
                dot_hr = tl.sum(go_i * orig_j, axis=1)
                g_hr_acc += tl.where(
                    tl.arange(0, N * N)[None, :] == i * N + j,
                    dot_hr[:, None],
                    tl.zeros((BLOCK_S, N * N), dtype=tl.float32),
                )

    offs_n = tl.arange(0, N)
    tl.store(
        g_hp_ptr + offs_s[:, None] * N + offs_n[None, :],
        g_hp_acc.to(g_hp_ptr.dtype.element_ty),
        mask=mask_s[:, None],
    )

    nn_offs = tl.arange(0, N * N)
    for i in tl.static_range(N):
        for j in tl.static_range(N):
            col_mask = (nn_offs == (i * N + j)).to(tl.float32)
            val = tl.sum(g_hr_acc * col_mask[None, :], axis=1)
            tl.store(
                g_hr_ptr + offs_s * stride_hr_s + i * stride_hr_i + j * stride_hr_j,
                val.to(g_hr_ptr.dtype.element_ty),
                mask=mask_s,
            )


def _triton_h_post_bda_bwd(
    grad_output: Tensor,
    h_res: Tensor,
    original_residual: Tensor,
    h_post: Tensor,
    x: Tensor,
    bias: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    s, b, n, C = original_residual.shape
    sb = s * b
    dev = h_res.device

    g_hr = torch.empty(sb, n, n, dtype=h_res.dtype, device=dev)
    g_res = torch.empty(sb, n, C, dtype=original_residual.dtype, device=dev)
    g_hp = torch.empty(sb, n, dtype=h_post.dtype, device=dev)
    g_x = torch.empty(sb, C, dtype=x.dtype, device=dev)

    go_flat = grad_output.contiguous().view(sb, n, C)
    hr_flat = h_res.contiguous().view(sb, n, n)
    orig_flat = original_residual.contiguous().view(sb, n, C)
    hp_flat = h_post.contiguous().view(sb, n)
    x_flat = x.contiguous().view(sb, C)

    grid_a = lambda META: (triton.cdiv(sb, META["BLOCK_S"]), triton.cdiv(C, META["BLOCK_C"]))
    _triton_hpb_bwd_g_x_orig_kernel[grid_a](
        go_flat, hr_flat, hp_flat, g_res, g_x, sb, C, n,
        go_flat.stride(0), go_flat.stride(1), go_flat.stride(2),
        hr_flat.stride(0), hr_flat.stride(1), hr_flat.stride(2),
        g_res.stride(0), g_res.stride(1), g_res.stride(2),
    )

    grid_b = lambda META: (triton.cdiv(sb, META["BLOCK_S"]),)
    _triton_hpb_bwd_g_hp_hr_kernel[grid_b](
        go_flat, orig_flat, x_flat, bias if bias is not None else x_flat,
        g_hr, g_hp, sb, C, n,
        go_flat.stride(0), go_flat.stride(1), go_flat.stride(2),
        orig_flat.stride(0), orig_flat.stride(1), orig_flat.stride(2),
        g_hr.stride(0), g_hr.stride(1), g_hr.stride(2),
        HAS_BIAS=(bias is not None),
    )

    g_bias = g_x.sum(dim=0).to(dtype=bias.dtype) if bias is not None else None
    return (
        g_hr.view(s, b, n, n),
        g_res.view(s, b, n, C),
        g_hp.view(s, b, n),
        g_x.view(s, b, C),
        g_bias,
    )


__all__ = [
    "_triton_h_aggregate_fwd",
    "_triton_h_post_bda_fwd",
    "_triton_h_post_bda_bwd",
    "triton_fused_sinkhorn",
]
