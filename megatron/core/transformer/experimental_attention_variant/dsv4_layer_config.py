# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass

from megatron.core.transformer.transformer_config import MLATransformerConfig


@dataclass(kw_only=True)
class CSALayerConfig(MLATransformerConfig):
    """Configuration for a Compressed Sparse Attention layer."""

    compress_ratio: int = 0
    """Token compression ratio for this attention layer."""


@dataclass(kw_only=True)
class CSA2LayerConfig(MLATransformerConfig):
    """A CSA2 attention layer, with sharing sources indexed in the enclosing stack."""
