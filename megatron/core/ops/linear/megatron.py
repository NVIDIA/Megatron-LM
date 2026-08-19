# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron Core's own tensor-parallel linear layers."""

from __future__ import annotations

from typing import Optional

__all__ = ["Linear"]


class Linear:
    """Owns the linear slots using the layers in ``megatron.core.tensor_parallel``."""

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
