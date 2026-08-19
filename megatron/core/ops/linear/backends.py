# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Every linear backend, side by side. See :mod:`.contract` for what they must meet."""

from __future__ import annotations

from typing import Optional

__all__ = ["LinearLocal", "LinearTE"]


class LinearLocal:
    """The tensor-parallel linear layers in ``megatron.core.tensor_parallel``."""

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module the backend uses."""
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear

        return ColumnParallelLinear

    def row_parallel_linear(self) -> type:
        """Which row parallel linear module the backend uses."""
        from megatron.core.tensor_parallel.layers import RowParallelLinear

        return RowParallelLinear

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Megatron Core has no fused layernorm plus linear module."""
        return None


class LinearTE:
    """Transformer Engine's linear layers."""

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
