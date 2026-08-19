# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Every core-attention backend, side by side. See :mod:`.contract` for the requirements."""

from __future__ import annotations

__all__ = ["AttentionLocal", "AttentionTE"]


class AttentionLocal:
    """Megatron Core's own dot-product attention."""

    def core_attention(self) -> type:
        """Which module to use for attention."""
        from megatron.core.transformer.dot_product_attention import DotProductAttention

        return DotProductAttention


class AttentionTE:
    """Transformer Engine's fused attention.

    TE chooses flash, fused, or unfused internally from the shapes and environment it is
    given. Megatron does not copy that decision here.
    """

    REQUIRES = "transformer_engine"

    def core_attention(self) -> type:
        """Which module to use for attention."""
        from megatron.core.extensions.transformer_engine import TEDotProductAttention

        return TEDotProductAttention
