# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mixture-of-experts built on Transformer Engine."""

from __future__ import annotations

import warnings
from functools import partial
from typing import TYPE_CHECKING, Optional, cast

if TYPE_CHECKING:
    from megatron.core.transformer.mlp import TEActivationFunctionBuilder
    from megatron.core.transformer.moe.moe_layer import ExpertsBuilder

__all__ = ["Moe"]


class Moe:
    """Owns the MoE slots using Transformer Engine grouped linears."""

    REQUIRES = "transformer_engine"

    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> "ExpertsBuilder":
        """Grouped experts when TE offers them, sequential otherwise."""
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelGroupedLinear,
            TEColumnParallelLinear,
            TERowParallelGroupedLinear,
            TERowParallelLinear,
        )
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
        from megatron.core.transformer.mlp import MLPSubmodules
        from megatron.core.transformer.moe.experts import (
            GroupedMLPSubmodules,
            SequentialMLP,
            TEGroupedMLP,
        )
        from megatron.core.utils import get_te_version, is_te_min_version

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

    def activation_func(self) -> Optional["TEActivationFunctionBuilder"]:
        """Which module to use for activation function."""
        from megatron.core.extensions.transformer_engine import TEActivationOp

        # transformer_engine.BasicOperation.forward has an overly permissive return type, but by
        # design these classes always meet the interface.
        return cast("TEActivationFunctionBuilder", TEActivationOp)

    def moe_router(self) -> Optional[type]:
        """Keep the MoESubmodules default (the training TopKRouter)."""
        return None
