# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Transformer Engine attention."""

from __future__ import annotations

__all__ = ["Attention"]


class Attention:
    """Owns ``core_attention`` using Transformer Engine's fused attention.

    TE chooses flash, fused, or unfused internally from the shapes and environment it is
    given. Megatron does not copy that decision here.
    """

    REQUIRES = "transformer_engine"

    def core_attention(self) -> type:
        """Which module to use for attention."""
        from megatron.core.extensions.transformer_engine import TEDotProductAttention

        return TEDotProductAttention
