# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Inference-optimized mixture-of-experts."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Optional, cast

if TYPE_CHECKING:
    from megatron.core.transformer.mlp import TEActivationFunctionBuilder
    from megatron.core.transformer.moe.moe_layer import ExpertsBuilder

__all__ = ["Moe"]


class Moe:
    """Owns the MoE slots using the inference-optimized experts and router."""

    REQUIRES = "transformer_engine"

    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool) -> "ExpertsBuilder":
        """Inference always uses grouped experts."""
        del moe_use_grouped_gemm
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelGroupedLinear,
            TERowParallelGroupedLinear,
        )
        from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, InferenceGroupedMLP

        return partial(
            InferenceGroupedMLP,
            submodules=GroupedMLPSubmodules(
                linear_fc1=TEColumnParallelGroupedLinear,
                linear_fc2=TERowParallelGroupedLinear,
                activation_func=self.activation_func(),
            ),
        )

    def activation_func(self) -> Optional["TEActivationFunctionBuilder"]:
        """Which module to use for activation function."""
        from megatron.core.extensions.transformer_engine import TEActivationOp

        return cast("TEActivationFunctionBuilder", TEActivationOp)

    def moe_router(self) -> Optional[type]:
        """Inference needs compact [tokens, topk] index routing rather than a dense map."""
        from megatron.core.transformer.moe.router import InferenceTopKRouter

        return InferenceTopKRouter
