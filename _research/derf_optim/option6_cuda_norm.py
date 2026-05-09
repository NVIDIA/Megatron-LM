"""Option 6: hand-tuned CUDA Derf|DyT norm + cuBLAS via F.linear.

Same shape as Option 4 (Triton norm + F.linear) but the norm runs through a
hand-tuned vectorised CUDA kernel — LDG.128 loads (8 bf16 elements per
load), one block per row, fp32 compute for erf/tanh, store back as input
dtype. Tests whether a hand-tuned C++/CUDA kernel beats Triton for the
elementwise norm at our shape.

Backward: PyTorch with post-norm recompute, same as Option 4. Saves x for
backward.

bf16 + fp32, TP=1.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.cpp_extension import load

from megatron.core.transformer.dynamic_norms import Derf, DyT


_THIS_DIR = Path(__file__).resolve().parent
_BUILD_DIR = Path(
    os.environ.get(
        "APERTUS_CPP_EXT_BUILD_DIR",
        os.environ.get("TORCHINDUCTOR_CACHE_DIR", "/tmp/torch_extensions"),
    )
) / "derf_norm_v2"
_BUILD_DIR.mkdir(parents=True, exist_ok=True)


_ext = None


def _ensure_loaded():
    global _ext
    if _ext is None:
        _ext = load(
            name="derf_norm_v2",
            sources=[str(_THIS_DIR / "cuda" / "derf_norm_v2.cu")],
            build_directory=str(_BUILD_DIR),
            extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
            verbose=False,
        )
    return _ext


_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)


class _DerfLinearCudaNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_norm, b_norm, alpha, s, w_lin, b_lin):
        ext = _ensure_loaded()
        pre = ext.derf_norm_fwd(x.contiguous(), w_norm, b_norm, alpha, s)
        out = F.linear(pre, w_lin, b_lin)
        ctx.save_for_backward(x, w_norm, b_norm, alpha, s, w_lin)
        ctx.has_lin_bias = b_lin is not None
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


class _DyTLinearCudaNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_norm, b_norm, alpha, w_lin, b_lin):
        ext = _ensure_loaded()
        pre = ext.dyt_norm_fwd(x.contiguous(), w_norm, b_norm, alpha)
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


class _CudaNormDynamicNormLinearTP1(nn.Module):
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
                f"Option 6 only supports TP=1 (got {config.tensor_model_parallel_size})."
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
            out = _DerfLinearCudaNormFn.apply(
                x, self.norm.weight, self.norm.bias, self.norm.alpha, self.norm.s, w, linear_bias
            )
        else:
            out = _DyTLinearCudaNormFn.apply(
                x, self.norm.weight, self.norm.bias, self.norm.alpha, w, linear_bias
            )
        out_bias = self.bias if (self.skip_bias_add and self.bias is not None) else None
        return out, out_bias


class CudaNormDerfColumnLinearTP1(_CudaNormDynamicNormLinearTP1):
    norm_cls = Derf
    _is_derf = True


class CudaNormDyTColumnLinearTP1(_CudaNormDynamicNormLinearTP1):
    norm_cls = DyT
    _is_derf = False


def make_qkv_class(normalization: str) -> type:
    if normalization == "Derf":
        return CudaNormDerfColumnLinearTP1
    if normalization == "DyT":
        return CudaNormDyTColumnLinearTP1
    raise ValueError(f"Option 6 supports DyT/Derf, got {normalization}")


make_fc1_class = make_qkv_class
