"""Option 5: hypothesis test — Triton norm + TE's general_gemm (cuBLAS wrapper).

Identical to Option 4 except we route the matmul through TE's
`general_gemm` instead of `F.linear`. If this lands close to the RMSNorm
baseline (310 TFLOP/s), the missing 80 TFLOP/s in Option 4 was just the
PyTorch -> aten -> cuBLAS dispatch overhead vs TE's tighter wrapper. If
not, the gap is in the norm kernel itself (TE's hand-tuned RMSNorm vs our
Triton elementwise) and the next path forward is a hand-coded CUDA Derf
kernel that matches TE's RMSNorm performance.

bf16 + fp32, TP=1.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn

# TE's gemm wrapper is the same one TELayerNormColumnParallelLinear uses.
try:
    from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm
    _HAVE_TE_GEMM = True
except Exception:
    general_gemm = None
    _HAVE_TE_GEMM = False

from megatron.core.transformer.dynamic_norms import Derf, DyT
from _research.derf_optim.option4_te_style import _triton_derf_norm

_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)


class _DerfLinearTEGemmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_norm, b_norm, alpha, s, w_lin, b_lin):
        pre = _triton_derf_norm(x, w_norm, b_norm, alpha, s, kind="Derf")
        # TE's general_gemm expects 2D inputs in TN layout: out = pre @ w_lin.T
        # `A` is the right-hand operand in cuBLAS notation; pass weight as A
        # transposed (so layout='TN' means A^T @ B in column-major =
        # B @ A in row-major, matching torch.nn.functional.linear).
        K = pre.shape[-1]
        out_features = w_lin.shape[0]
        pre_2d = pre.reshape(-1, K).contiguous()
        out_2d, *_ = general_gemm(
            A=w_lin,             # [out_features, K]
            B=pre_2d,            # [batch_flat, K]
            out_dtype=pre.dtype,
            layout="TN",         # A is transposed: cuBLAS computes (A^T) @ B
            bias=b_lin,
        )
        # out_2d shape: [batch_flat, out_features]
        out = out_2d.reshape(*pre.shape[:-1], out_features)
        ctx.save_for_backward(x, w_norm, b_norm, alpha, s, w_lin)
        ctx.has_lin_bias = b_lin is not None
        ctx.batch_flat = pre_2d.shape[0]
        ctx.out_features = out_features
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w_norm, b_norm, alpha, s, w_lin = ctx.saved_tensors
        dtype = x.dtype
        with torch.no_grad():
            z = alpha.to(dtype) * x + s.to(dtype)
            ez = torch.erf(z)
            pre = w_norm.to(dtype) * ez + b_norm.to(dtype)

        K = x.shape[-1]
        N = grad_out.shape[-1]
        grad_pre = grad_out @ w_lin
        grad_w_lin = grad_out.reshape(-1, N).transpose(0, 1) @ pre.reshape(-1, K)
        grad_b_lin = grad_out.reshape(-1, N).sum(0) if ctx.has_lin_bias else None

        derf_dz = _TWO_OVER_SQRT_PI * torch.exp(-(z.float() ** 2)).to(dtype)
        chain = grad_pre * w_norm.to(dtype) * derf_dz
        grad_x = chain * alpha.to(dtype)

        reduce_dims = tuple(range(grad_pre.dim() - 1))
        grad_w_norm = (grad_pre * ez).sum(dim=reduce_dims).to(w_norm.dtype)
        grad_b_norm = grad_pre.sum(dim=reduce_dims).to(b_norm.dtype)
        grad_alpha = (chain * x).sum().to(alpha.dtype)
        grad_s = chain.sum().to(s.dtype)

        return grad_x, grad_w_norm, grad_b_norm, grad_alpha, grad_s, grad_w_lin, grad_b_lin


class _DyTLinearTEGemmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_norm, b_norm, alpha, w_lin, b_lin):
        pre = _triton_derf_norm(x, w_norm, b_norm, alpha, None, kind="DyT")
        K = pre.shape[-1]
        out_features = w_lin.shape[0]
        pre_2d = pre.reshape(-1, K).contiguous()
        out_2d, *_ = general_gemm(
            A=w_lin, B=pre_2d, out_dtype=pre.dtype, layout="TN", bias=b_lin,
        )
        out = out_2d.reshape(*pre.shape[:-1], out_features)
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


class _TEGemmDynamicNormLinearTP1(nn.Module):
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
        if not _HAVE_TE_GEMM:
            raise RuntimeError("Option 5 requires transformer_engine; not importable")
        if config.tensor_model_parallel_size != 1:
            raise RuntimeError(
                "Option 5 only supports TP=1 "
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
            out = _DerfLinearTEGemmFn.apply(
                x, self.norm.weight, self.norm.bias, self.norm.alpha, self.norm.s, w, linear_bias
            )
        else:
            out = _DyTLinearTEGemmFn.apply(
                x, self.norm.weight, self.norm.bias, self.norm.alpha, w, linear_bias
            )
        out_bias = self.bias if (self.skip_bias_add and self.bias is not None) else None
        return out, out_bias


class TEGemmDerfColumnLinearTP1(_TEGemmDynamicNormLinearTP1):
    norm_cls = Derf
    _is_derf = True


class TEGemmDyTColumnLinearTP1(_TEGemmDynamicNormLinearTP1):
    norm_cls = DyT
    _is_derf = False


def make_qkv_class(normalization: str) -> type:
    if normalization == "Derf":
        return TEGemmDerfColumnLinearTP1
    if normalization == "DyT":
        return TEGemmDyTColumnLinearTP1
    raise ValueError(f"Option 5 supports DyT/Derf, got {normalization}")


make_fc1_class = make_qkv_class
