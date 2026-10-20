# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.transformer.experimental_attention_variant.dsv4_layer_config import (
    get_compact_indexer_workspace_registry,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig


class DSALayerConfig(MLATransformerConfig):
    """Configuration for a DeepSeek Sparse Attention layer in a hybrid stack.

    Due to backwards-compatibility, this config's arguments are defined in MLATransformerConfig.
    """

    @classmethod
    def from_config(cls, config):
        """Retain workspace sharing for dev's array-driven DSv4 D-layer factory."""
        if config.dsa_compact_indexer_workspace_sharing:
            get_compact_indexer_workspace_registry(config)
        return super().from_config(config)
