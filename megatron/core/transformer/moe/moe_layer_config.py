# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.transformer.transformer_config import TransformerConfig


class MoELayerConfig(TransformerConfig):
    """Configuration for a Mixture-of-Experts layer in a hybrid stack.

    Due to backwards-compatibility, this config's arguments are defined in TransformerConfig.
    """
