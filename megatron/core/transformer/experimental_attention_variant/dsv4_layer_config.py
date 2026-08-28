# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.transformer.transformer_config import MLATransformerConfig


class CSALayerConfig(MLATransformerConfig):
    """Configuration for a DeepSeek-V4 compressed sparse attention layer."""


class HCALayerConfig(MLATransformerConfig):
    """Configuration for a DeepSeek-V4 heavily compressed attention layer."""


class WindowAttentionLayerConfig(MLATransformerConfig):
    """Configuration for a DeepSeek-V4 sliding-window attention layer."""
