# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.transformer.transformer_config import TransformerConfig


class KDALayerConfig(TransformerConfig):
    """Configuration for a KDA (Kimi Delta Attention) layer in a hybrid stack.

    Due to backwards-compatibility, this config's arguments are defined in TransformerConfig.
    """
