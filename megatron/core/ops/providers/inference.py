# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The inference-optimized backend."""

from __future__ import annotations

from functools import partial
from typing import Optional, cast

from megatron.core.extensions.transformer_engine import (
    TEActivationOp,
    TEColumnParallelGroupedLinear,
    TEDotProductAttention,
    TELinear,
    TERowParallelGroupedLinear,
)
from megatron.core.ops import _availability
from megatron.core.ops.norm.transformer_engine import TENormBackend
from megatron.core.ops.spec_provider import BackendSpecProvider
from megatron.core.tensor_parallel.inference_layers import (
    InferenceColumnParallelLinear,
    InferenceLayerNormColumnParallelLinear,
    InferenceRowParallelLinear,
)
from megatron.core.transformer.mlp import TEActivationFunctionBuilder
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, InferenceGroupedMLP
from megatron.core.transformer.moe.moe_layer import ExpertsBuilder
from megatron.core.transformer.torch_norm import LayerNormBuilder

__all__ = ["InferenceSpecProvider"]

_BACKEND_NAME = "inference_optimized"


class InferenceSpecProvider(BackendSpecProvider):
    """Provides inference-optimized submodules built on Transformer Engine."""

    def __init__(self) -> None:
        _availability.require("transformer_engine", backend=_BACKEND_NAME)
        # Inference layers manage the residual themselves, so norm never fuses one here.
        self._norm = TENormBackend(fuse_residual=False)

    def linear(self) -> type:
        """Which linear module the backend uses."""
        return TELinear

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module the backend uses."""
        return InferenceColumnParallelLinear

    def row_parallel_linear(self) -> type:
        """Which row parallel linear module the backend uses."""
        return InferenceRowParallelLinear

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module fuses layernorm and column parallel linear."""
        return InferenceLayerNormColumnParallelLinear

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
        del moe_use_grouped_gemm
        return partial(
            InferenceGroupedMLP,
            submodules=GroupedMLPSubmodules(
                linear_fc1=TEColumnParallelGroupedLinear,
                linear_fc2=TERowParallelGroupedLinear,
                activation_func=self.activation_func(),
            ),
        )

    def activation_func(self) -> Optional[TEActivationFunctionBuilder]:
        """Which module to use for activation function."""
        return cast(TEActivationFunctionBuilder, TEActivationOp)

    def moe_router(self) -> Optional[type]:
        """Inference needs compact [tokens, topk] index routing rather than a dense map."""
        from megatron.core.transformer.moe.router import InferenceTopKRouter

        return InferenceTopKRouter
