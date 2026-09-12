# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.transformer.transformer_config import MLATransformerConfig


class MLALayerConfig(MLATransformerConfig):
    """Configuration for a Multi-Latent Attention layer in a hybrid stack.

    Due to backwards-compatibility, this config's arguments are defined in MLATransformerConfig.
    """
