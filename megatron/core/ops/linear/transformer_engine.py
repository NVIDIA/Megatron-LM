# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Transformer Engine linear layers."""

from __future__ import annotations

from typing import Optional

__all__ = ["Linear"]


class Linear:
    """Owns the linear slots using Transformer Engine."""

    REQUIRES = "transformer_engine"

    def linear(self) -> type:
        """Which non-parallel linear module the backend uses."""
        from megatron.core.extensions.transformer_engine import TELinear

        return TELinear

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module the backend uses."""
        from megatron.core.extensions.transformer_engine import TEColumnParallelLinear

        return TEColumnParallelLinear

    def row_parallel_linear(self) -> type:
        """Which row parallel linear module the backend uses."""
        from megatron.core.extensions.transformer_engine import TERowParallelLinear

        return TERowParallelLinear

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module fuses layernorm and column parallel linear."""
        from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear

        return TELayerNormColumnParallelLinear
