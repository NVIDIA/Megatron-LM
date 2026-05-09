"""Option 1: torch.compile fused (Derf + Linear).

Wraps Derf and a plain `torch.nn.functional.linear` in a `torch.compile` region.
Bypasses TE and Megatron's TP plumbing in exchange for being visible to Inductor,
which can then fuse the elementwise Derf into the matmul prologue (eliminating
the duplicated read of the activation tensor that costs us throughput in the
unfused reference path).

Constraints:
    * TP=1 only. Megatron's `ColumnParallelLinear` uses a custom autograd
      Function with TP scatter/gather and async grad comms; torch.compile
      cannot see through that, so we use plain F.linear here. With TP=1 the
      semantics are identical.
    * Initialisation matches Megatron's `init_method` (the same scaled normal
      that `ColumnParallelLinear.__init__` uses for `self.weight`).
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from megatron.core.transformer.dynamic_norms import Derf, DyT


def _derf_linear_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    alpha: torch.Tensor,
    s: torch.Tensor,
    linear_bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Pure-PyTorch fused forward: y = linear(Derf(x))."""
    dtype = x.dtype
    pre = norm_weight.to(dtype) * torch.erf(alpha.to(dtype) * x + s.to(dtype)) + norm_bias.to(dtype)
    return F.linear(pre, weight, linear_bias)


def _dyt_linear_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    alpha: torch.Tensor,
    linear_bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Pure-PyTorch fused forward: y = linear(DyT(x))."""
    dtype = x.dtype
    pre = norm_weight.to(dtype) * torch.tanh(alpha.to(dtype) * x) + norm_bias.to(dtype)
    return F.linear(pre, weight, linear_bias)


_USE_COMPILE = os.environ.get("APERTUS_DERF_COMPILE", "1") != "0"
_COMPILE_MODE = os.environ.get("APERTUS_DERF_COMPILE_MODE", "default")


def _maybe_compile(fn):
    """Return torch.compile(fn) when enabled, else fn unchanged.

    `mode='default'` is the safest baseline; users can override via
    APERTUS_DERF_COMPILE_MODE=reduce-overhead / max-autotune to compare.
    """
    if not _USE_COMPILE:
        return fn
    return torch.compile(fn, mode=_COMPILE_MODE, dynamic=False, fullgraph=True)


_compiled_derf_linear = _maybe_compile(_derf_linear_fwd)
_compiled_dyt_linear = _maybe_compile(_dyt_linear_fwd)


class _CompiledDynamicNormLinearTP1(nn.Module):
    """Common TP=1 base for the DyT/Derf fused composites. Subclasses just
    pick which inner activation to use."""

    norm_cls: type = None  # set by subclass
    _compiled_fwd = None  # set by subclass

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
        # We rely on TP=1 semantics; otherwise we'd need a sharded matmul.
        if config.tensor_model_parallel_size != 1:
            raise RuntimeError(
                "_CompiledDynamicNormLinearTP1 only supports TP=1 "
                f"(got {config.tensor_model_parallel_size}). "
                "Use a sharded TE/Triton path for TP > 1."
            )

        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add

        # Norm parameters (Derf or DyT).
        self.norm = self.norm_cls(config=config, hidden_size=input_size)

        # Linear weight in params_dtype, init via init_method to match Megatron's
        # column-parallel-linear convention.
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
        # `bias` is always None in this fused path when skip_bias_add is on
        # (the dense MLP uses skip_bias_add=False but bias=False, so this is moot).
        linear_bias = None if self.skip_bias_add else self.bias
        out = self._call_compiled(x, w, linear_bias)
        out_bias = self.bias if (self.skip_bias_add and self.bias is not None) else None
        return out, out_bias

    def _call_compiled(self, x, weight, linear_bias):
        raise NotImplementedError


class CompiledDerfColumnLinearTP1(_CompiledDynamicNormLinearTP1):
    norm_cls = Derf

    def _call_compiled(self, x, weight, linear_bias):
        return _compiled_derf_linear(
            x,
            weight,
            self.norm.weight,
            self.norm.bias,
            self.norm.alpha,
            self.norm.s,
            linear_bias,
        )


class CompiledDyTColumnLinearTP1(_CompiledDynamicNormLinearTP1):
    norm_cls = DyT

    def _call_compiled(self, x, weight, linear_bias):
        return _compiled_dyt_linear(
            x,
            weight,
            self.norm.weight,
            self.norm.bias,
            self.norm.alpha,
            linear_bias,
        )


def make_qkv_class(normalization: str) -> type:
    """Return the fused norm-linear class to wire as `linear_qkv`."""
    if normalization == "Derf":
        return CompiledDerfColumnLinearTP1
    if normalization == "DyT":
        return CompiledDyTColumnLinearTP1
    raise ValueError(f"Option 1 supports DyT/Derf, got {normalization}")


make_fc1_class = make_qkv_class
