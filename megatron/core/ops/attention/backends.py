# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Every core-attention backend, side by side.

The contract they meet is in this package's ``__init__``.
"""

from __future__ import annotations

__all__ = ["AttentionLocal", "AttentionTE"]


class AttentionLocal:
    """Megatron Core's own dot-product attention."""

    #: Not audited.
    DETERMINISM = "unknown"

    def core_attention(self) -> type:
        """Which module to use for attention."""
        from megatron.core.transformer.dot_product_attention import DotProductAttention

        return DotProductAttention


class AttentionTE:
    """Transformer Engine's fused attention.

    TE chooses flash, fused, or unfused internally from the shapes and environment it is
    given. Megatron does not copy that decision here.
    """

    #: TE refuses to run under deterministic_mode unless NVTE_ALLOW_NONDETERMINISTIC_ALGO=0,
    #: which is one of the env defaults that mode sets. See extensions/transformer_engine.py.
    DETERMINISM = "deterministic"

    REQUIRES = "transformer_engine"

    def core_attention(self) -> type:
        """Which module to use for attention."""
        from megatron.core.extensions.transformer_engine import TEDotProductAttention

        return TEDotProductAttention
