# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Inference-optimized linear layers, built on Transformer Engine."""

from __future__ import annotations

from typing import Optional

__all__ = ["Linear"]


class Linear:
    """Owns the linear slots using the inference-optimized layers."""

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
