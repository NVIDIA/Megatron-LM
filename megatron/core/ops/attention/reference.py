# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reference dot-product attention."""

from __future__ import annotations

__all__ = ["Attention"]


class Attention:
    """Owns ``core_attention`` using Megatron Core's own dot-product attention."""

    def core_attention(self) -> type:
        """Which module to use for attention."""
        from megatron.core.transformer.dot_product_attention import DotProductAttention

        return DotProductAttention
