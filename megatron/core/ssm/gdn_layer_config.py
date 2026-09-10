# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.transformer.transformer_config import TransformerConfig


class GDNLayerConfig(TransformerConfig):
    """Configuration for a Gated DeltaNet layer in a hybrid stack.

    Due to backwards-compatibility, this config's arguments are defined in TransformerConfig.
    """
