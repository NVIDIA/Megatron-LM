# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The inference-optimized backend for each operation family."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Optional, cast

if TYPE_CHECKING:
    from megatron.core.transformer.mlp import TEActivationFunctionBuilder
    from megatron.core.transformer.moe.moe_layer import ExpertsBuilder

__all__ = ["LinearInference", "MoeInference", "NormInference"]


class NormInference:
    """Transformer Engine's norm, never fusing the residual.

    Inference layers manage the residual themselves, so folding it into the norm would apply
    it twice. Composed rather than subclassed, and imported inside the method, so this module
    has no import-time dependency on ``megatron.core.ops`` -- which would be a cycle, since
    ``ops.norm`` names this class in its backend table.
    """

    #: Defers to the Transformer Engine norm, which is not audited here.
    DETERMINISM = "unknown"

    REQUIRES = "transformer_engine"

    def layer_norm(self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False):
        """Transformer Engine's choice, with residual fusion always declined."""
        from megatron.core.ops.norm.backends import NormTE

        del has_residual
        return NormTE().layer_norm(rms_norm=rms_norm, for_qk=for_qk, has_residual=False)


class LinearInference:
    """The inference-optimized linear layers."""

    #: Not audited.
    DETERMINISM = "unknown"

    REQUIRES = "transformer_engine"

    def linear(self) -> type:
        """Inference reuses the Transformer Engine non-parallel linear."""
        from megatron.core.extensions.transformer_engine import TELinear

        return TELinear

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module the backend uses."""
        from megatron.core.tensor_parallel.inference_layers import InferenceColumnParallelLinear

        return InferenceColumnParallelLinear

    def row_parallel_linear(self) -> type:
        """Which row parallel linear module the backend uses."""
        from megatron.core.tensor_parallel.inference_layers import InferenceRowParallelLinear

        return InferenceRowParallelLinear

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module fuses layernorm and column parallel linear."""
        from megatron.core.tensor_parallel.inference_layers import (
            InferenceLayerNormColumnParallelLinear,
        )

        return InferenceLayerNormColumnParallelLinear


class MoeInference:
    """The inference-optimized experts and router."""

    #: Not audited; routing and permutation are unexamined.
    DETERMINISM = "unknown"

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
