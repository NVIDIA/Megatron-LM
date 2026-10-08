# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.transformer.transformer_config import TransformerConfig


class AttentionLayerConfig(TransformerConfig):
    """Configuration for an attention layer in a hybrid stack.

    Due to backwards-compatibility, this config's arguments are defined in TransformerConfig.
    """
