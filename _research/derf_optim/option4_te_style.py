"""Option 4: TE-style pipeline — explicit Triton norm kernel + cuBLAS linear.

Rationale: TE's `LayerNormLinear` baseline (310 TFLOP/s) is two well-tuned,
well-pipelined kernel calls — `tex.rmsnorm_fwd` then a cuBLAS GEMM. It's NOT
a single fused CUTLASS kernel.

Option 1 (271 TFLOP/s, torch.compile) gets close by letting Inductor fuse
the elementwise Derf into a kernel, then calling cuBLAS via aten dispatch.
Option 4 isolates the kernel choice: we replace Inductor's auto-generated
Derf with a hand-written Triton elementwise kernel (single launch, vectorised),
and keep cuBLAS for the matmul via `F.linear`. If hand-tuned Triton beats
Inductor for this op, we get a free throughput bump.

Backward: standard cuBLAS linear backward + Derf chain rule with post-norm
recompute (saves only x, matches TE's RMSNorm memory profile).

bf16 + fp32, TP=1.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

from megatron.core.transformer.dynamic_norms import Derf, DyT


# -----------------------------------------------------------------------------
# Triton elementwise (Derf|DyT) norm kernel — one launch over flat n_total.
# -----------------------------------------------------------------------------


@triton.jit
def _norm_elementwise_kernel(
    x_ptr, w_ptr, b_ptr, alpha_ptr, s_ptr, out_ptr,
    n_total, K,
    BLOCK: tl.constexpr,
    HAS_S: tl.constexpr,
    NORM_KIND: tl.constexpr,  # 0=Derf (erf), 1=DyT (tanh)
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_total
    # Index into per-channel gamma / beta (one element per K-position).
    col = offs % K

    alpha = tl.load(alpha_ptr).to(tl.float32)
    if HAS_S:
        s = tl.load(s_ptr).to(tl.float32)
    else:
        s = 0.0

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + col, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + col, mask=mask, other=0.0).to(tl.float32)

    z = alpha * x + s
    if NORM_KIND == 0:
        fz = tl.erf(z)
    else:
        fz = (tl.exp(z) - tl.exp(-z)) / (tl.exp(z) + tl.exp(-z))
    out = w * fz + b

    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


def _triton_derf_norm(x, w_norm, b_norm, alpha, s, kind: str = "Derf"):
    out = torch.empty_like(x)
    K = x.shape[-1]
    n_total = x.numel()
    BLOCK = 4096
    grid = (triton.cdiv(n_total, BLOCK),)
    s_used = s if s is not None else torch.zeros((), dtype=alpha.dtype, device=alpha.device)
    _norm_elementwise_kernel[grid](
        x.contiguous(), w_norm.contiguous(), b_norm.contiguous(),
        alpha, s_used, out,
        n_total, K,
        BLOCK=BLOCK,
        HAS_S=(s is not None),
        NORM_KIND=0 if kind == "Derf" else 1,
        num_warps=8,
    )
    return out


# -----------------------------------------------------------------------------
# Autograd: Triton norm + F.linear (cuBLAS) forward; PyTorch backward with
# post-norm recompute.
# -----------------------------------------------------------------------------


_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)


class _DerfLinearTEStyleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_norm, b_norm, alpha, s, w_lin, b_lin):
        pre = _triton_derf_norm(x, w_norm, b_norm, alpha, s, kind="Derf")
        out = F.linear(pre, w_lin, b_lin)
        ctx.save_for_backward(x, w_norm, b_norm, alpha, s, w_lin)
        ctx.has_lin_bias = b_lin is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w_norm, b_norm, alpha, s, w_lin = ctx.saved_tensors
        dtype = x.dtype
        # Recompute pre (no autograd, single elementwise kernel each).
        with torch.no_grad():
            z = alpha.to(dtype) * x + s.to(dtype)
            ez = torch.erf(z)
            pre = w_norm.to(dtype) * ez + b_norm.to(dtype)

        # Standard linear backward.
        K = x.shape[-1]
        N = grad_out.shape[-1]
        grad_pre = grad_out @ w_lin
        grad_w_lin = grad_out.reshape(-1, N).transpose(0, 1) @ pre.reshape(-1, K)
        grad_b_lin = grad_out.reshape(-1, N).sum(0) if ctx.has_lin_bias else None

        # Chain through Derf.
        derf_dz = _TWO_OVER_SQRT_PI * torch.exp(-(z.float() ** 2)).to(dtype)
        chain = grad_pre * w_norm.to(dtype) * derf_dz
        grad_x = chain * alpha.to(dtype)

        reduce_dims = tuple(range(grad_pre.dim() - 1))
        grad_w_norm = (grad_pre * ez).sum(dim=reduce_dims).to(w_norm.dtype)
        grad_b_norm = grad_pre.sum(dim=reduce_dims).to(b_norm.dtype)
        grad_alpha = (chain * x).sum().to(alpha.dtype)
        grad_s = chain.sum().to(s.dtype)

        return grad_x, grad_w_norm, grad_b_norm, grad_alpha, grad_s, grad_w_lin, grad_b_lin


class _DyTLinearTEStyleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_norm, b_norm, alpha, w_lin, b_lin):
        pre = _triton_derf_norm(x, w_norm, b_norm, alpha, None, kind="DyT")
        out = F.linear(pre, w_lin, b_lin)
        ctx.save_for_backward(x, w_norm, b_norm, alpha, w_lin)
        ctx.has_lin_bias = b_lin is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w_norm, b_norm, alpha, w_lin = ctx.saved_tensors
        dtype = x.dtype
        with torch.no_grad():
            z = alpha.to(dtype) * x
            tz = torch.tanh(z)
            pre = w_norm.to(dtype) * tz + b_norm.to(dtype)

        K = x.shape[-1]
        N = grad_out.shape[-1]
        grad_pre = grad_out @ w_lin
        grad_w_lin = grad_out.reshape(-1, N).transpose(0, 1) @ pre.reshape(-1, K)
        grad_b_lin = grad_out.reshape(-1, N).sum(0) if ctx.has_lin_bias else None

        dtanh_dz = 1.0 - tz * tz
        chain = grad_pre * w_norm.to(dtype) * dtanh_dz
        grad_x = chain * alpha.to(dtype)

        reduce_dims = tuple(range(grad_pre.dim() - 1))
        grad_w_norm = (grad_pre * tz).sum(dim=reduce_dims).to(w_norm.dtype)
        grad_b_norm = grad_pre.sum(dim=reduce_dims).to(b_norm.dtype)
        grad_alpha = (chain * x).sum().to(alpha.dtype)

        return grad_x, grad_w_norm, grad_b_norm, grad_alpha, grad_w_lin, grad_b_lin


def _te_style_derf_linear(x, w_lin, w_norm, b_norm, alpha, s, b_lin):
    return _DerfLinearTEStyleFn.apply(x, w_norm, b_norm, alpha, s, w_lin, b_lin)


def _te_style_dyt_linear(x, w_lin, w_norm, b_norm, alpha, b_lin):
    return _DyTLinearTEStyleFn.apply(x, w_norm, b_norm, alpha, w_lin, b_lin)


# -----------------------------------------------------------------------------
# Module wrappers — match the signature used by Options 1/2/3 so the spec
# wiring can swap easily.
# -----------------------------------------------------------------------------


class _TEStyleDynamicNormLinearTP1(nn.Module):
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
                "TE-style Derf path only supports TP=1 "
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
            out = _te_style_derf_linear(
                x, w, self.norm.weight, self.norm.bias, self.norm.alpha, self.norm.s, linear_bias
            )
        else:
            out = _te_style_dyt_linear(
                x, w, self.norm.weight, self.norm.bias, self.norm.alpha, linear_bias
            )
        out_bias = self.bias if (self.skip_bias_add and self.bias is not None) else None
        return out, out_bias


class TEStyleDerfColumnLinearTP1(_TEStyleDynamicNormLinearTP1):
    norm_cls = Derf
    _is_derf = True


class TEStyleDyTColumnLinearTP1(_TEStyleDynamicNormLinearTP1):
    norm_cls = DyT
    _is_derf = False


def make_qkv_class(normalization: str) -> type:
    if normalization == "Derf":
        return TEStyleDerfColumnLinearTP1
    if normalization == "DyT":
        return TEStyleDyTColumnLinearTP1
    raise ValueError(f"Option 4 supports DyT/Derf, got {normalization}")


make_fc1_class = make_qkv_class
