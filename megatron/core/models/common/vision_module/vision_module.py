# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Megatron Vision Module."""

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


# Note: This is only a stub at the moment. This will be expanded in follow-up changes.
class VisionModule(MegatronModule):
    """Base vision module that has common helper functions used across CLIP, ViT, etc.

    Args:
        config (TransformerConfig): Input transformer config for the model
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config=config)
