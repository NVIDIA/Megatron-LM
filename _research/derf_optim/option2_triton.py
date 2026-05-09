"""Option 2: Triton fused (Derf | DyT) + Linear kernel.

Forward is a single Triton kernel that does

    out = (gamma * f(alpha*x + s) + beta) @ W^T  [+ b_lin]

with f = erf for Derf and f = tanh for DyT. The kernel reads `x` once, applies
the elementwise norm in registers, and accumulates the matmul, eliminating
the duplicated activation read that costs us throughput in the unfused
reference path.

Backward uses PyTorch ops with norm-output recomputation:
    pre = gamma * f(alpha*x + s) + beta            (no autograd, cheap)
    grad_pre = grad_out @ w_lin                    (standard linear bwd)
    grad_w_lin = grad_out^T @ pre
    grad_b_lin = grad_out.sum(over M) if has bias
    f'(z) = 2/sqrt(pi) * exp(-z^2)                 (z = alpha*x + s) for Derf
    f'(z) = 1 - tanh(z)^2                          (z = alpha*x)     for DyT
    grad_x       = grad_pre * gamma * f'(z) * alpha
    grad_gamma   = sum_M (grad_pre * f(z))
    grad_beta    = sum_M  grad_pre
    grad_alpha   = sum_all (grad_pre * gamma * f'(z) * x)
    grad_s       = sum_all (grad_pre * gamma * f'(z))             # Derf only

Saves only `x` for backward (same memory profile as TE's RMSNorm).
TP=1 only. bf16 + fp32 supported.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

from megatron.core.transformer.dynamic_norms import Derf, DyT


# -----------------------------------------------------------------------------
# Triton forward kernels (Derf and DyT). The bodies are nearly identical; the
# norm function is a constexpr-selected branch so we get both shapes from one
# template without repeating the matmul plumbing.
# -----------------------------------------------------------------------------


@triton.jit
def _norm_linear_fwd_kernel(
    x_ptr, w_lin_ptr, w_norm_ptr, b_norm_ptr, alpha_ptr, s_ptr, b_lin_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_LINEAR_BIAS: tl.constexpr,
    HAS_S: tl.constexpr,  # Derf has s, DyT does not
    NORM_KIND: tl.constexpr,  # 0=Derf (erf), 1=DyT (tanh)
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    alpha = tl.load(alpha_ptr).to(tl.float32)
    if HAS_S:
        s = tl.load(s_ptr).to(tl.float32)
    else:
        s = 0.0

    accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_kk = k0 + offs_k
        k_mask = offs_kk < K

        x_mask = (offs_m[:, None] < M) & (k_mask[None, :])
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_kk[None, :] * stride_xk,
            mask=x_mask,
            other=0.0,
        )
        w_norm = tl.load(w_norm_ptr + offs_kk, mask=k_mask, other=0.0)
        b_norm = tl.load(b_norm_ptr + offs_kk, mask=k_mask, other=0.0)

        x_f = x_tile.to(tl.float32)
        z = alpha * x_f + s
        if NORM_KIND == 0:
            fz = tl.erf(z)
        else:
            fz = (tl.exp(z) - tl.exp(-z)) / (tl.exp(z) + tl.exp(-z))
        pre = w_norm[None, :].to(tl.float32) * fz + b_norm[None, :].to(tl.float32)

        w_mask = (offs_n[:, None] < N) & k_mask[None, :]
        w_tile = tl.load(
            w_lin_ptr + offs_n[:, None] * stride_wn + offs_kk[None, :] * stride_wk,
            mask=w_mask,
            other=0.0,
        )

        # `pre @ W^T`: pre is [BLOCK_M, BLOCK_K], W is [BLOCK_N, BLOCK_K].
        # tl.dot wants [M,K] @ [K,N], so transpose W tile.
        accum += tl.dot(pre.to(x_tile.dtype), tl.trans(w_tile.to(x_tile.dtype)))

    if HAS_LINEAR_BIAS:
        b_lin = tl.load(b_lin_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        accum += b_lin[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        accum.to(x_tile.dtype),
        mask=out_mask,
    )


def _triton_fused_fwd(x, w_lin, w_norm, b_norm, alpha, s, b_lin, kind: str):
    """Public-ish entry: y = linear(norm(x)) via Triton.

    Shapes:
        x:      [..., K]
        w_lin:  [N, K]
        w_norm: [K]
        b_norm: [K]
        alpha:  scalar
        s:      scalar (Derf only; pass None for DyT)
        b_lin:  [N] or None
    """
    orig_shape = x.shape
    K = orig_shape[-1]
    x_2d = x.reshape(-1, K).contiguous()
    M = x_2d.shape[0]
    N = w_lin.shape[0]
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    s_dummy = torch.zeros((), dtype=alpha.dtype, device=alpha.device)
    s_used = s if s is not None else s_dummy
    b_lin_used = b_lin if b_lin is not None else x  # any tensor; mask=False protects load

    _norm_linear_fwd_kernel[grid](
        x_2d, w_lin.contiguous(), w_norm.contiguous(), b_norm.contiguous(),
        alpha, s_used, b_lin_used, out,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        w_lin.stride(0), w_lin.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        HAS_LINEAR_BIAS=b_lin is not None,
        HAS_S=(s is not None),
        NORM_KIND=0 if kind == "Derf" else 1,
    )
    return out.reshape(*orig_shape[:-1], N)


# -----------------------------------------------------------------------------
# Autograd: Triton forward, PyTorch backward with recomputed pre.
# -----------------------------------------------------------------------------


_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)


class _DerfLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_norm, b_norm, alpha, s, w_lin, b_lin):
        out = _triton_fused_fwd(x, w_lin, w_norm, b_norm, alpha, s, b_lin, "Derf")
        ctx.save_for_backward(x, w_norm, b_norm, alpha, s, w_lin)
        ctx.has_linear_bias = b_lin is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w_norm, b_norm, alpha, s, w_lin = ctx.saved_tensors
        # Recompute the post-norm tensor (same as TE's RMSNorm path).
        dtype = x.dtype
        z = alpha.to(dtype) * x + s.to(dtype)
        ez = torch.erf(z)
        pre = w_norm.to(dtype) * ez + b_norm.to(dtype)

        # Standard linear backward.
        K = x.shape[-1]
        N = grad_out.shape[-1]
        grad_pre = grad_out @ w_lin
        grad_w_lin = grad_out.reshape(-1, N).transpose(0, 1) @ pre.reshape(-1, K)
        grad_b_lin = grad_out.reshape(-1, N).sum(0) if ctx.has_linear_bias else None

        # Chain through Derf.
        # d_pre/d_z = w_norm * erf'(z) = w_norm * 2/sqrt(pi) * exp(-z^2)
        # d_pre/d_x = (d_pre/d_z) * alpha
        derf_dz = _TWO_OVER_SQRT_PI * torch.exp(-(z.float() ** 2)).to(dtype)
        chain = grad_pre * w_norm.to(dtype) * derf_dz  # [..., K]
        grad_x = chain * alpha.to(dtype)

        # Reduce param grads. w_norm and b_norm are 1D over K; reduce M dims.
        reduce_dims = tuple(range(grad_pre.dim() - 1))
        grad_w_norm = (grad_pre * ez).sum(dim=reduce_dims).to(w_norm.dtype)
        grad_b_norm = grad_pre.sum(dim=reduce_dims).to(b_norm.dtype)
        grad_alpha = (chain * x).sum().to(alpha.dtype)
        grad_s = chain.sum().to(s.dtype)

        return grad_x, grad_w_norm, grad_b_norm, grad_alpha, grad_s, grad_w_lin, grad_b_lin


class _DyTLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_norm, b_norm, alpha, w_lin, b_lin):
        out = _triton_fused_fwd(x, w_lin, w_norm, b_norm, alpha, None, b_lin, "DyT")
        ctx.save_for_backward(x, w_norm, b_norm, alpha, w_lin)
        ctx.has_linear_bias = b_lin is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w_norm, b_norm, alpha, w_lin = ctx.saved_tensors
        dtype = x.dtype
        z = alpha.to(dtype) * x
        tz = torch.tanh(z)
        pre = w_norm.to(dtype) * tz + b_norm.to(dtype)

        K = x.shape[-1]
        N = grad_out.shape[-1]
        grad_pre = grad_out @ w_lin
        grad_w_lin = grad_out.reshape(-1, N).transpose(0, 1) @ pre.reshape(-1, K)
        grad_b_lin = grad_out.reshape(-1, N).sum(0) if ctx.has_linear_bias else None

        # d_pre/d_z = w_norm * (1 - tanh(z)^2)
        dtanh_dz = 1.0 - tz * tz
        chain = grad_pre * w_norm.to(dtype) * dtanh_dz
        grad_x = chain * alpha.to(dtype)

        reduce_dims = tuple(range(grad_pre.dim() - 1))
        grad_w_norm = (grad_pre * tz).sum(dim=reduce_dims).to(w_norm.dtype)
        grad_b_norm = grad_pre.sum(dim=reduce_dims).to(b_norm.dtype)
        grad_alpha = (chain * x).sum().to(alpha.dtype)

        return grad_x, grad_w_norm, grad_b_norm, grad_alpha, grad_w_lin, grad_b_lin


def _triton_derf_linear(x, w_lin, w_norm, b_norm, alpha, s, b_lin):
    return _DerfLinearFn.apply(x, w_norm, b_norm, alpha, s, w_lin, b_lin)


def _triton_dyt_linear(x, w_lin, w_norm, b_norm, alpha, b_lin):
    return _DyTLinearFn.apply(x, w_norm, b_norm, alpha, w_lin, b_lin)


# -----------------------------------------------------------------------------
# Module wrappers (same shape as Option 1 so the spec wiring can swap easily)
# -----------------------------------------------------------------------------


class _TritonDynamicNormLinearTP1(nn.Module):
    norm_cls: type = None
    _is_derf: bool = False

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config,
        init_method,
        bias: bool = False,
        skip_bias_add: bool = False,
        gather_output: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer=None,
        grad_output_buffer=None,
        disable_grad_reduce: bool = False,
        **_unused,
    ):
        super().__init__()
        if config.tensor_model_parallel_size != 1:
            raise RuntimeError(
                "_TritonDynamicNormLinearTP1 only supports TP=1 "
                f"(got {config.tensor_model_parallel_size})."
            )
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add

        self.norm = self.norm_cls(config=config, hidden_size=input_size)

        weight = torch.empty(output_size, input_size, dtype=config.params_dtype)
        if config.perform_initialization:
            init_method(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size, dtype=config.params_dtype))
        else:
            self.register_parameter("bias", None)

    def forward(
        self, x: torch.Tensor, weight: Optional[torch.Tensor] = None, runtime_gather_output=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        w = weight if weight is not None else self.weight
        linear_bias = None if self.skip_bias_add else self.bias
        if self._is_derf:
            out = _triton_derf_linear(
                x, w, self.norm.weight, self.norm.bias, self.norm.alpha, self.norm.s, linear_bias
            )
        else:
            out = _triton_dyt_linear(
                x, w, self.norm.weight, self.norm.bias, self.norm.alpha, linear_bias
            )
        out_bias = self.bias if (self.skip_bias_add and self.bias is not None) else None
        return out, out_bias


class TritonDerfColumnLinearTP1(_TritonDynamicNormLinearTP1):
    norm_cls = Derf
    _is_derf = True


class TritonDyTColumnLinearTP1(_TritonDynamicNormLinearTP1):
    norm_cls = DyT
    _is_derf = False


def make_qkv_class(normalization: str) -> type:
    if normalization == "Derf":
        return TritonDerfColumnLinearTP1
    if normalization == "DyT":
        return TritonDyTColumnLinearTP1
    raise ValueError(f"Option 2 supports DyT/Derf, got {normalization}")


make_fc1_class = make_qkv_class
