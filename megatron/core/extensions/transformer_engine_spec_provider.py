# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import warnings
from functools import partial
from typing import cast

from typing_extensions import final, override

from megatron.core.extensions.transformer_engine import (
    TEActivationOp,
    TEColumnParallelGroupedLinear,
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TELinear,
    TENorm,
    TERowParallelGroupedLinear,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.models.backends import BackendSpecProvider
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.mlp import MLPSubmodules, TEActivationFunctionBuilder
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import ExpertsBuilder
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.utils import get_te_version, is_te_min_version


class _TENormWithResidual:
    """Class adapter for TENorm with residual fusion enabled."""

    def __new__(cls, *args, **kwargs):
        return TENorm(*args, has_residual=True, **kwargs)


@final
class TESpecProvider(BackendSpecProvider):
    """A protocol for providing the submodules used in Spec building."""

    @override
    def linear(self) -> type[TELinear]:
        """Which linear module TE backend uses"""
        return TELinear

    @override
    def column_parallel_linear(self) -> type[TEColumnParallelLinear]:
        """Which column parallel linear module TE backend uses"""
        return TEColumnParallelLinear

    @override
    def row_parallel_linear(self) -> type[TERowParallelLinear]:
        """Which row parallel linear module TE backend uses"""
        return TERowParallelLinear

    @override
    def column_parallel_layer_norm_linear(self) -> type[TELayerNormColumnParallelLinear]:
        """Which module for sequential layernorm and linear"""
        return TELayerNormColumnParallelLinear

    @override
    def layer_norm(
        self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False
    ) -> LayerNormBuilder:
        """Which module to use for layer norm"""
        if for_qk and not is_te_min_version("1.9.0"):
            # TENorm significantly harms convergence when used
            # for QKLayerNorm if TE Version < 1.9;
            # we instead use the Apex implementation.
            return FusedLayerNorm
        # Keep returning a class so this path stays aligned with build_module's class handling.
        return _TENormWithResidual if has_residual else TENorm

    @override
    def core_attention(self) -> type[TEDotProductAttention]:
        """Which module to use for attention"""
        return TEDotProductAttention

    @override
    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> ExpertsBuilder:
        """Which module and submodules to use for grouped mlp"""
        if moe_use_grouped_gemm and TEColumnParallelGroupedLinear is not None:
            return partial(
                TEGroupedMLP,
                submodules=GroupedMLPSubmodules(
                    linear_fc1=TEColumnParallelGroupedLinear,
                    linear_fc2=TERowParallelGroupedLinear,
                    activation_func=self.activation_func(),
                ),
            )
        else:
            if not is_te_min_version("1.7.0.dev0"):
                warnings.warn(
                    "Only transformer-engine>=1.7.0 supports MoE experts, "
                    f"but your version is {get_te_version()}. "
                    "Use local linear implementation instead."
                )
                return partial(
                    SequentialMLP,
                    submodules=MLPSubmodules(
                        linear_fc1=ColumnParallelLinear,
                        linear_fc2=RowParallelLinear,
                        activation_func=self.activation_func(),
                    ),
                )
            return partial(
                SequentialMLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                    activation_func=self.activation_func(),
                ),
            )

    @override
    def activation_func(self) -> TEActivationFunctionBuilder | None:
        """Which module to use for activation function"""
        # transformer_engine.BasicOperation.forward has an overly permissive return type, but by
        # design these classes always meet the interface.
        return cast(TEActivationFunctionBuilder, TEActivationOp)
