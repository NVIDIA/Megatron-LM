"""Option 3: hand-written CUDA fused (Derf | DyT) + Linear.

Wraps the CUDA kernel in cuda/derf_linear_kernel.cu via torch's JIT
``cpp_extension.load`` machinery. Same signature as Options 1/2 so the spec
wiring can swap in via APERTUS_DERF_OPTIM=cuda.

Forward: hand-rolled SMEM-tiled matmul with Derf/DyT prologue applied in
registers. No tensor cores yet (correctness-first); gives a fair "what does
plain CUDA fusion buy?" data point against Triton.

Backward: same recompute pattern as Option 2.

Compilation happens once at first import; cached under
``$TORCHINDUCTOR_CACHE_DIR/cpp_extensions/derf_linear_cuda``.
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

# Compile once. PyTorch caches the .so under the build_directory.
_BUILD_DIR = Path(
    os.environ.get(
        "APERTUS_CPP_EXT_BUILD_DIR",
        os.environ.get("TORCHINDUCTOR_CACHE_DIR", "/tmp/torch_extensions"),
    )
) / "derf_linear_cuda"
_BUILD_DIR.mkdir(parents=True, exist_ok=True)


_ext = None


def _ensure_loaded():
    global _ext
    if _ext is None:
        _ext = load(
            name="derf_linear_cuda",
            sources=[str(_THIS_DIR / "cuda" / "derf_linear_kernel.cu")],
            build_directory=str(_BUILD_DIR),
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _ext


_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)


class _DerfLinearCudaFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_norm, b_norm, alpha, s, w_lin, b_lin):
        ext = _ensure_loaded()
        b_lin_arg = b_lin if b_lin is not None else torch.empty(0, dtype=x.dtype, device=x.device)
        out = ext.derf_linear_fwd(x.contiguous(), w_lin.contiguous(), w_norm, b_norm, alpha, s, b_lin_arg)
        ctx.save_for_backward(x, w_norm, b_norm, alpha, s, w_lin)
        ctx.has_linear_bias = b_lin is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w_norm, b_norm, alpha, s, w_lin = ctx.saved_tensors
        dtype = x.dtype
        z = alpha.to(dtype) * x + s.to(dtype)
        ez = torch.erf(z)
        pre = w_norm.to(dtype) * ez + b_norm.to(dtype)

        K = x.shape[-1]
        N = grad_out.shape[-1]
        grad_pre = grad_out @ w_lin
        grad_w_lin = grad_out.reshape(-1, N).transpose(0, 1) @ pre.reshape(-1, K)
        grad_b_lin = grad_out.reshape(-1, N).sum(0) if ctx.has_linear_bias else None

        derf_dz = _TWO_OVER_SQRT_PI * torch.exp(-(z.float() ** 2)).to(dtype)
        chain = grad_pre * w_norm.to(dtype) * derf_dz
        grad_x = chain * alpha.to(dtype)

        reduce_dims = tuple(range(grad_pre.dim() - 1))
        grad_w_norm = (grad_pre * ez).sum(dim=reduce_dims).to(w_norm.dtype)
        grad_b_norm = grad_pre.sum(dim=reduce_dims).to(b_norm.dtype)
        grad_alpha = (chain * x).sum().to(alpha.dtype)
        grad_s = chain.sum().to(s.dtype)

        return grad_x, grad_w_norm, grad_b_norm, grad_alpha, grad_s, grad_w_lin, grad_b_lin


class _DyTLinearCudaFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_norm, b_norm, alpha, w_lin, b_lin):
        ext = _ensure_loaded()
        b_lin_arg = b_lin if b_lin is not None else torch.empty(0, dtype=x.dtype, device=x.device)
        out = ext.dyt_linear_fwd(x.contiguous(), w_lin.contiguous(), w_norm, b_norm, alpha, b_lin_arg)
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

        dtanh_dz = 1.0 - tz * tz
        chain = grad_pre * w_norm.to(dtype) * dtanh_dz
        grad_x = chain * alpha.to(dtype)

        reduce_dims = tuple(range(grad_pre.dim() - 1))
        grad_w_norm = (grad_pre * tz).sum(dim=reduce_dims).to(w_norm.dtype)
        grad_b_norm = grad_pre.sum(dim=reduce_dims).to(b_norm.dtype)
        grad_alpha = (chain * x).sum().to(alpha.dtype)

        return grad_x, grad_w_norm, grad_b_norm, grad_alpha, grad_w_lin, grad_b_lin


def _cuda_derf_linear(x, w_lin, w_norm, b_norm, alpha, s, b_lin):
    return _DerfLinearCudaFn.apply(x, w_norm, b_norm, alpha, s, w_lin, b_lin)


def _cuda_dyt_linear(x, w_lin, w_norm, b_norm, alpha, b_lin):
    return _DyTLinearCudaFn.apply(x, w_norm, b_norm, alpha, w_lin, b_lin)


class _CudaDynamicNormLinearTP1(nn.Module):
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
                "_CudaDynamicNormLinearTP1 only supports TP=1 "
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
            out = _cuda_derf_linear(
                x, w, self.norm.weight, self.norm.bias, self.norm.alpha, self.norm.s, linear_bias
            )
        else:
            out = _cuda_dyt_linear(
                x, w, self.norm.weight, self.norm.bias, self.norm.alpha, linear_bias
            )
        out_bias = self.bias if (self.skip_bias_add and self.bias is not None) else None
        return out, out_bias


class CudaDerfColumnLinearTP1(_CudaDynamicNormLinearTP1):
    norm_cls = Derf
    _is_derf = True


class CudaDyTColumnLinearTP1(_CudaDynamicNormLinearTP1):
    norm_cls = DyT
    _is_derf = False


def make_qkv_class(normalization: str) -> type:
    if normalization == "Derf":
        return CudaDerfColumnLinearTP1
    if normalization == "DyT":
        return CudaDyTColumnLinearTP1
    raise ValueError(f"Option 3 supports DyT/Derf, got {normalization}")


make_fc1_class = make_qkv_class
