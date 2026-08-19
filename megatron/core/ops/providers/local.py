# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The backend built only from modules in Megatron Core."""

from __future__ import annotations

import warnings
from functools import partial
from typing import Optional

from megatron.core.ops.norm import apex_layer_norm, have_apex
from megatron.core.ops.norm.reference import WrappedTorchNorm
from megatron.core.ops.spec_provider import BackendSpecProvider
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.mlp import MLPSubmodules, TEActivationFunctionBuilder
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.moe.moe_layer import ExpertsBuilder
from megatron.core.transformer.torch_norm import LayerNormBuilder

__all__ = ["LocalSpecProvider"]


class LocalSpecProvider(BackendSpecProvider):
    """Provides Megatron Core submodules, using Apex's fused LayerNorm when it is installed."""

    def __init__(self) -> None:
        self._have_apex = have_apex()
        if not self._have_apex:
            warnings.warn("Apex is not installed. Falling back to Torch Norm")

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module the backend uses."""
        return ColumnParallelLinear

    def row_parallel_linear(self) -> type:
        """Which row parallel linear module the backend uses."""
        return RowParallelLinear

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Megatron Core has no fused layernorm plus linear module."""
        return None

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        """Apex's fused LayerNorm when available; Torch otherwise, and always for RMSNorm."""
        del for_qk, has_residual
        if rms_norm or not self._have_apex:
            return WrappedTorchNorm
        return apex_layer_norm()

    def core_attention(self) -> type:
        """Which module to use for attention."""
        return DotProductAttention

    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> ExpertsBuilder:
        """Which module and submodules to use for grouped mlp."""
        del moe_use_grouped_gemm
        return partial(
            SequentialMLP,
            submodules=MLPSubmodules(
                linear_fc1=ColumnParallelLinear,
                linear_fc2=RowParallelLinear,
                activation_func=self.activation_func(),
            ),
        )

    def activation_func(self) -> Optional[TEActivationFunctionBuilder]:
        """Megatron Core reads config.activation_func rather than building a module."""
        return None
