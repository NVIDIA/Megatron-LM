# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The Transformer Engine backend."""

from __future__ import annotations

import warnings
from functools import partial
from typing import Optional, cast

from megatron.core.extensions.transformer_engine import (
    TEActivationOp,
    TEColumnParallelGroupedLinear,
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TELinear,
    TERowParallelGroupedLinear,
    TERowParallelLinear,
)
from megatron.core.ops import _availability
from megatron.core.ops.norm.transformer_engine import TENormBackend
from megatron.core.ops.spec_provider import BackendSpecProvider
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.mlp import MLPSubmodules, TEActivationFunctionBuilder
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import ExpertsBuilder
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.utils import get_te_version, is_te_min_version

__all__ = ["TESpecProvider"]

_BACKEND_NAME = "transformer_engine"


class TESpecProvider(BackendSpecProvider):
    """Provides Transformer Engine submodules.

    TE stays a single backend: it keeps its own attention selection, FP8 and FP4 behavior, user
    buffers, and fused regions. Megatron does not re-implement any of that here.
    """

    def __init__(self) -> None:
        _availability.require("transformer_engine", backend=_BACKEND_NAME)
        self._norm = TENormBackend()

    def linear(self) -> type:
        """Which linear module TE backend uses."""
        return TELinear

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module TE backend uses."""
        return TEColumnParallelLinear

    def row_parallel_linear(self) -> type:
        """Which row parallel linear module TE backend uses."""
        return TERowParallelLinear

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module fuses layernorm and column parallel linear."""
        return TELayerNormColumnParallelLinear

    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        """Which module to use for layer norm."""
        return self._norm.layer_norm(rms_norm=rms_norm, for_qk=for_qk, has_residual=has_residual)

    def core_attention(self) -> type:
        """Which module to use for attention."""
        return TEDotProductAttention

    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> ExpertsBuilder:
        """Which module and submodules to use for grouped mlp."""
        if moe_use_grouped_gemm and TEColumnParallelGroupedLinear is not None:
            return partial(
                TEGroupedMLP,
                submodules=GroupedMLPSubmodules(
                    linear_fc1=TEColumnParallelGroupedLinear,
                    linear_fc2=TERowParallelGroupedLinear,
                    activation_func=self.activation_func(),
                ),
            )
        if not is_te_min_version("1.7.0.dev0"):
            warnings.warn(
                "Only transformer-engine>=1.7.0 supports MoE experts, "
                f"but your version is {get_te_version()}. "
                "Use local linear implementation instead."
            )
            linear_fc1, linear_fc2 = ColumnParallelLinear, RowParallelLinear
        else:
            linear_fc1, linear_fc2 = TEColumnParallelLinear, TERowParallelLinear
        return partial(
            SequentialMLP,
            submodules=MLPSubmodules(
                linear_fc1=linear_fc1, linear_fc2=linear_fc2, activation_func=self.activation_func()
            ),
        )

    def activation_func(self) -> Optional[TEActivationFunctionBuilder]:
        """Which module to use for activation function."""
        # transformer_engine.BasicOperation.forward has an overly permissive return type, but by
        # design these classes always meet the interface.
        return cast(TEActivationFunctionBuilder, TEActivationOp)
