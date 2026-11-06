# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Triton SwiGLU kernels for fused_bias_swiglu's bias-free plain, clamped and router-weighted paths.

Matches megatron.core.fusions.fused_bias_swiglu (bias-free) for:
    plain:     out = silu(gate) * up                         [gate | up] = chunk(x, 2, -1)
    clamped:   gate -> min(gate, c), up -> clamp(up, -c, c), gradients masked outside the clamp
    weighted:  out = act(x) * w   (w: one router probability per row, MoE experts)
               d_x = act'(x) * (g * w),  d_w = sum_cols(act(x) * g)

Each program loads one tile of gate and the matching tile of up once, computes in fp32, and writes
both gradient halves directly into one [.., 2F] output: no concat. The weighted backward uses one
program per row so the d_w row sum stays in registers instead of a second pass over x and g.
Launch configs are picked from the row length only (no autotuning), so the first call is safe
ahead of CUDA graph capture and changing row counts (MoE) never trigger re-tuning.
"""

from functools import cache

import torch
import triton
import triton.language as tl


def _config(n_cols: int, cap: int = 2048):
    """Tile size: the row length rounded up to a power of two, capped; warps scale with the tile."""
    block = min(cap, max(64, triton.next_power_of_2(n_cols)))
    return block, (4 if block >= 1024 else 2)


@cache
def clamp_includes_boundaries() -> bool:
    """Same boundary rule as MCore's clamped SwiGLU (its probe of this torch build)."""
    from megatron.core.fusions.fused_bias_swiglu import _probe_clamp_boundaries

    return bool(_probe_clamp_boundaries())


@triton.jit
def _act(gate_raw, up_raw, clamp, CLAMP: tl.constexpr):
    if CLAMP:
        gate = tl.minimum(gate_raw, clamp)
        up = tl.minimum(tl.maximum(up_raw, -clamp), clamp)
    else:
        gate = gate_raw
        up = up_raw
    sig = tl.sigmoid(gate)
    return gate, up, sig


@triton.jit
def _masks(gate_raw, up_raw, clamp, INCL: tl.constexpr):
    if INCL:
        gm = (gate_raw <= clamp).to(tl.float32)
        lm = ((up_raw >= -clamp) & (up_raw <= clamp)).to(tl.float32)
    else:
        gm = (gate_raw < clamp).to(tl.float32)
        lm = ((up_raw > -clamp) & (up_raw < clamp)).to(tl.float32)
    return gm, lm


@triton.jit
def _fwd_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    n_cols,
    x_row_stride,
    out_row_stride,
    clamp,
    BLOCK: tl.constexpr,
    CLAMP: tl.constexpr,
    WEIGHTED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    x_row = x_ptr + row * x_row_stride
    gate_raw = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    up_raw = tl.load(x_row + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    gate, up, sig = _act(gate_raw, up_raw, clamp, CLAMP)
    out = gate * sig * up
    if WEIGHTED:
        out = out * tl.load(w_ptr + row).to(tl.float32)
    tl.store(out_ptr + row * out_row_stride + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _bwd_kernel(
    grad_ptr,
    x_ptr,
    dx_ptr,
    n_cols,
    grad_row_stride,
    x_row_stride,
    dx_row_stride,
    clamp,
    BLOCK: tl.constexpr,
    CLAMP: tl.constexpr,
    INCL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    x_row = x_ptr + row * x_row_stride
    g = tl.load(grad_ptr + row * grad_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    gate_raw = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    up_raw = tl.load(x_row + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    gate, up, sig = _act(gate_raw, up_raw, clamp, CLAMP)
    d_gate = g * sig * (1.0 + gate * (1.0 - sig)) * up
    d_up = g * (gate * sig)
    if CLAMP:
        gm, lm = _masks(gate_raw, up_raw, clamp, INCL)
        d_gate = d_gate * gm
        d_up = d_up * lm
    dx_row = dx_ptr + row * dx_row_stride
    out_ty = dx_ptr.dtype.element_ty
    tl.store(dx_row + cols, d_gate.to(out_ty), mask=mask)
    tl.store(dx_row + n_cols + cols, d_up.to(out_ty), mask=mask)


@triton.jit
def _weighted_bwd_kernel(
    grad_ptr,
    x_ptr,
    w_ptr,
    dx_ptr,
    dw_ptr,
    n_cols,
    grad_row_stride,
    x_row_stride,
    dx_row_stride,
    clamp,
    BLOCK: tl.constexpr,
    CLAMP: tl.constexpr,
    INCL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * x_row_stride
    g_row = grad_ptr + row * grad_row_stride
    dx_row = dx_ptr + row * dx_row_stride
    out_ty = dx_ptr.dtype.element_ty
    w = tl.load(w_ptr + row).to(tl.float32)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        g = tl.load(g_row + cols, mask=mask, other=0.0).to(tl.float32)
        gate_raw = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        up_raw = tl.load(x_row + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        gate, up, sig = _act(gate_raw, up_raw, clamp, CLAMP)
        acc += tl.where(mask, gate * sig * up * g, 0.0)
        gw = g * w
        d_gate = gw * sig * (1.0 + gate * (1.0 - sig)) * up
        d_up = gw * (gate * sig)
        if CLAMP:
            gm, lm = _masks(gate_raw, up_raw, clamp, INCL)
            d_gate = d_gate * gm
            d_up = d_up * lm
        tl.store(dx_row + cols, d_gate.to(out_ty), mask=mask)
        tl.store(dx_row + n_cols + cols, d_up.to(out_ty), mask=mask)
    tl.store(dw_ptr + row, tl.sum(acc, axis=0).to(dw_ptr.dtype.element_ty))


def _as_rows(t: torch.Tensor) -> torch.Tensor:
    """View t as [rows, last_dim] with a unit stride along the last dim (copies only if needed)."""
    t2 = t.reshape(-1, t.shape[-1])
    return t2 if t2.stride(-1) == 1 else t2.contiguous()


def _clamp_args(clamp):
    use = clamp is not None and clamp > 0
    return use, (float(clamp) if use else 0.0)


def swiglu_fwd(
    x: torch.Tensor, weights: torch.Tensor | None = None, clamp: float | None = None
) -> torch.Tensor:
    """SwiGLU forward for x of shape [..., 2F]; weights (one per row) and clamp are optional."""
    assert x.shape[-1] % 2 == 0, f"last dim must be even, got {x.shape[-1]}"
    n_cols = x.shape[-1] // 2
    x2 = _as_rows(x)
    out = torch.empty((x2.shape[0], n_cols), dtype=x.dtype, device=x.device)
    use_clamp, c = _clamp_args(clamp)
    w = (
        weights.reshape(-1).contiguous() if weights is not None else x2
    )  # dummy pointer when unweighted
    if weights is not None:
        assert w.numel() == x2.shape[0], (w.shape, x2.shape)
    if x2.shape[0] and n_cols:
        block, warps = _config(n_cols)
        grid = (x2.shape[0], triton.cdiv(n_cols, block))
        _fwd_kernel[grid](
            x2,
            w,
            out,
            n_cols,
            x2.stride(0),
            out.stride(0),
            c,
            BLOCK=block,
            CLAMP=use_clamp,
            WEIGHTED=weights is not None,
            num_warps=warps,
        )
    return out.view(*x.shape[:-1], n_cols)


def swiglu_bwd(
    grad: torch.Tensor,
    x: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    weights: torch.Tensor | None = None,
    clamp: float | None = None,
):
    """Gradient of swiglu_fwd. Returns d_x ([..., 2F]), or (d_x, d_w) when weights are given.

    x may be stored as float8_e4m3fn; d_w has the dtype and shape of weights.
    """
    n_cols = x.shape[-1] // 2
    assert grad.shape[-1] == n_cols and grad.shape[:-1] == x.shape[:-1], (grad.shape, x.shape)
    out_dtype = out_dtype or grad.dtype
    g2, x2 = _as_rows(grad), _as_rows(x)
    dx = torch.empty((x2.shape[0], 2 * n_cols), dtype=out_dtype, device=x.device)
    use_clamp, c = _clamp_args(clamp)
    incl = clamp_includes_boundaries() if use_clamp else False
    if weights is None:
        if x2.shape[0] and n_cols:
            block, warps = _config(n_cols)
            grid = (x2.shape[0], triton.cdiv(n_cols, block))
            _bwd_kernel[grid](
                g2,
                x2,
                dx,
                n_cols,
                g2.stride(0),
                x2.stride(0),
                dx.stride(0),
                c,
                BLOCK=block,
                CLAMP=use_clamp,
                INCL=incl,
                num_warps=warps,
            )
        return dx.view(*x.shape)
    w = weights.reshape(-1).contiguous()
    dw = torch.empty_like(w)
    if x2.shape[0] and n_cols:
        block, warps = _config(n_cols, cap=4096)
        _weighted_bwd_kernel[(x2.shape[0],)](
            g2,
            x2,
            w,
            dx,
            dw,
            n_cols,
            g2.stride(0),
            x2.stride(0),
            dx.stride(0),
            c,
            BLOCK=block,
            CLAMP=use_clamp,
            INCL=incl,
            num_warps=warps,
        )
    return dx.view(*x.shape), dw.view(weights.shape)


# ------------------------------------------------------------------------------ multi-row
# Short rows (MoE experts): one program handles ROWS whole rows as a [ROWS, BLOCK] tile, so
# each program moves enough bytes even when a single row is only a few KB. The autograd
# functions use these for F <= MR_DISPATCH_MAX_COLS, where they beat the one-row kernels.
MR_MAX_COLS = 2048
MR_DISPATCH_MAX_COLS = 1024


def _mr_config(n_cols: int):
    block = max(64, triton.next_power_of_2(n_cols))
    rows = max(1, min(16, 4096 // block))
    return rows, block, (8 if rows * block >= 8192 else 4)


@triton.jit
def _fwd_kernel_mr(
    x_ptr,
    w_ptr,
    out_ptr,
    n_rows,
    n_cols,
    x_row_stride,
    out_row_stride,
    clamp,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    CLAMP: tl.constexpr,
    WEIGHTED: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    rmask = rows < n_rows
    mask = rmask[:, None] & (cols < n_cols)[None, :]
    r64 = rows.to(tl.int64)
    xo = r64[:, None] * x_row_stride + cols[None, :]
    gate_raw = tl.load(x_ptr + xo, mask=mask, other=0.0).to(tl.float32)
    up_raw = tl.load(x_ptr + xo + n_cols, mask=mask, other=0.0).to(tl.float32)
    gate, up, sig = _act(gate_raw, up_raw, clamp, CLAMP)
    out = gate * sig * up
    if WEIGHTED:
        w = tl.load(w_ptr + r64, mask=rmask, other=0.0).to(tl.float32)
        out = out * w[:, None]
    oo = r64[:, None] * out_row_stride + cols[None, :]
    tl.store(out_ptr + oo, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _bwd_kernel_mr(
    grad_ptr,
    x_ptr,
    w_ptr,
    dx_ptr,
    dw_ptr,
    n_rows,
    n_cols,
    grad_row_stride,
    x_row_stride,
    dx_row_stride,
    clamp,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    CLAMP: tl.constexpr,
    INCL: tl.constexpr,
    WEIGHTED: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    rmask = rows < n_rows
    mask = rmask[:, None] & (cols < n_cols)[None, :]
    r64 = rows.to(tl.int64)
    xo = r64[:, None] * x_row_stride + cols[None, :]
    g = tl.load(grad_ptr + r64[:, None] * grad_row_stride + cols[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    gate_raw = tl.load(x_ptr + xo, mask=mask, other=0.0).to(tl.float32)
    up_raw = tl.load(x_ptr + xo + n_cols, mask=mask, other=0.0).to(tl.float32)
    gate, up, sig = _act(gate_raw, up_raw, clamp, CLAMP)
    if WEIGHTED:
        w = tl.load(w_ptr + r64, mask=rmask, other=0.0).to(tl.float32)
        dw = tl.sum(tl.where(mask, gate * sig * up * g, 0.0), axis=1)
        tl.store(dw_ptr + r64, dw.to(dw_ptr.dtype.element_ty), mask=rmask)
        g = g * w[:, None]
    d_gate = g * sig * (1.0 + gate * (1.0 - sig)) * up
    d_up = g * (gate * sig)
    if CLAMP:
        gm, lm = _masks(gate_raw, up_raw, clamp, INCL)
        d_gate = d_gate * gm
        d_up = d_up * lm
    do = r64[:, None] * dx_row_stride + cols[None, :]
    out_ty = dx_ptr.dtype.element_ty
    tl.store(dx_ptr + do, d_gate.to(out_ty), mask=mask)
    tl.store(dx_ptr + do + n_cols, d_up.to(out_ty), mask=mask)


def swiglu_fwd_mr(x, weights=None, clamp=None):
    """Multi-row forward for short rows (F <= 2048); same contract as swiglu_fwd."""
    n_cols = x.shape[-1] // 2
    assert n_cols <= MR_MAX_COLS, n_cols
    x2 = _as_rows(x)
    out = torch.empty((x2.shape[0], n_cols), dtype=x.dtype, device=x.device)
    use_clamp, c = _clamp_args(clamp)
    w = weights.reshape(-1).contiguous() if weights is not None else x2
    if x2.shape[0] and n_cols:
        rows, block, warps = _mr_config(n_cols)
        _fwd_kernel_mr[(triton.cdiv(x2.shape[0], rows),)](
            x2,
            w,
            out,
            x2.shape[0],
            n_cols,
            x2.stride(0),
            out.stride(0),
            c,
            ROWS=rows,
            BLOCK=block,
            CLAMP=use_clamp,
            WEIGHTED=weights is not None,
            num_warps=warps,
        )
    return out.view(*x.shape[:-1], n_cols)


def swiglu_bwd_mr(grad, x, out_dtype=None, weights=None, clamp=None):
    """Multi-row backward for short rows (F <= 2048); same contract as swiglu_bwd."""
    n_cols = x.shape[-1] // 2
    assert n_cols <= MR_MAX_COLS, n_cols
    out_dtype = out_dtype or grad.dtype
    g2, x2 = _as_rows(grad), _as_rows(x)
    dx = torch.empty((x2.shape[0], 2 * n_cols), dtype=out_dtype, device=x.device)
    use_clamp, c = _clamp_args(clamp)
    incl = clamp_includes_boundaries() if use_clamp else False
    if weights is not None:
        w = weights.reshape(-1).contiguous()
        dw = torch.empty_like(w)
    else:
        w = dw = x2  # unused pointers
    if x2.shape[0] and n_cols:
        rows, block, warps = _mr_config(n_cols)
        _bwd_kernel_mr[(triton.cdiv(x2.shape[0], rows),)](
            g2,
            x2,
            w,
            dx,
            dw,
            x2.shape[0],
            n_cols,
            g2.stride(0),
            x2.stride(0),
            dx.stride(0),
            c,
            ROWS=rows,
            BLOCK=block,
            CLAMP=use_clamp,
            INCL=incl,
            WEIGHTED=weights is not None,
            num_warps=warps,
        )
    if weights is None:
        return dx.view(*x.shape)
    return dx.view(*x.shape), dw.view(weights.shape)


def _kernels(n_cols: int):
    """(fwd, bwd) implementation for this row length."""
    if n_cols <= MR_DISPATCH_MAX_COLS:
        return swiglu_fwd_mr, swiglu_bwd_mr
    return swiglu_fwd, swiglu_bwd


class TritonSwiGLUFunction(torch.autograd.Function):
    """Bias-free SwiGLU (optionally clamped); same contract as SwiGLUFunction without SiTU."""

    @staticmethod
    def forward(ctx, x, fp8_input_store=False, cpu_offload_input=False, clamp=None):
        """x: [..., 2F] as [gate | up]; returns [..., F]."""
        x_saved = x.to(torch.float8_e4m3fn) if fp8_input_store else x
        if cpu_offload_input:
            x_saved.activation_offloading = True
        ctx.save_for_backward(x_saved)
        ctx.x_dtype, ctx.clamp = x.dtype, clamp
        return _kernels(x.shape[-1] // 2)[0](x, clamp=clamp)

    @staticmethod
    def backward(ctx, grad):
        """Gradient for x only."""
        (x,) = ctx.saved_tensors
        bwd = _kernels(x.shape[-1] // 2)[1]
        return bwd(grad, x, out_dtype=ctx.x_dtype, clamp=ctx.clamp), None, None, None


class TritonWeightedSwiGLUFunction(torch.autograd.Function):
    """Router-weighted SwiGLU for MoE experts (optionally clamped).

    Same contract as WeightedSwiGLUFunction without SiTU-GLU.
    """

    @staticmethod
    def forward(ctx, x, weights, fp8_input_store=False, clamp=None):
        """x: [..., 2F] as [gate | up], weights: one value per row; returns [..., F]."""
        # Lazy import avoids the TE extension -> MLP -> fusion import cycle.
        from megatron.core.extensions.transformer_engine import propagate_te_grouped_tensor_marker

        x_saved = x.to(torch.float8_e4m3fn) if fp8_input_store else x
        propagate_te_grouped_tensor_marker(x, x_saved)
        ctx.save_for_backward(x_saved, weights)
        ctx.x_dtype, ctx.clamp = x.dtype, clamp
        return _kernels(x.shape[-1] // 2)[0](x, weights=weights, clamp=clamp)

    @staticmethod
    def backward(ctx, grad):
        """Gradients for x and weights."""
        x, weights = ctx.saved_tensors
        bwd = _kernels(x.shape[-1] // 2)[1]
        dx, dw = bwd(grad, x, out_dtype=ctx.x_dtype, weights=weights, clamp=ctx.clamp)
        return dx, dw, None, None
