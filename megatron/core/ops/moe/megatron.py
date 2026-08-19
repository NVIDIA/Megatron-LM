# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mixture-of-experts built only from Megatron Core modules."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from megatron.core.transformer.mlp import TEActivationFunctionBuilder
    from megatron.core.transformer.moe.moe_layer import ExpertsBuilder

__all__ = ["Moe"]


class Moe:
    """Owns the MoE slots using sequential experts made of Megatron Core linears."""

    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> "ExpertsBuilder":
        """Megatron Core has no grouped GEMM, so experts are always sequential."""
        del moe_use_grouped_gemm
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
        from megatron.core.transformer.mlp import MLPSubmodules
        from megatron.core.transformer.moe.experts import SequentialMLP

        return partial(
            SequentialMLP,
            submodules=MLPSubmodules(
                linear_fc1=ColumnParallelLinear,
                linear_fc2=RowParallelLinear,
                activation_func=self.activation_func(),
            ),
        )

    def activation_func(self) -> Optional["TEActivationFunctionBuilder"]:
        """Megatron Core reads config.activation_func rather than building a module."""
        return None

    def moe_router(self) -> Optional[type]:
        """Keep the MoESubmodules default (the training TopKRouter)."""
        return None
