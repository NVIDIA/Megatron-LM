# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Kimi K2.5 VL model provider for multimodal_dev training.

This package provides KimiK25VLModel whose checkpoint layout
(language_model.*, vision_tower.*, mm_projector.*) is identical to
Megatron-Bridge, enabling direct checkpoint loading.
"""

from examples.multimodal_dev.models.kimi_k25.configuration import (
    KIMI_K25_IMAGE_TOKEN_ID,
    KIMI_K25_VOCAB_SIZE,
    get_kimi_k25_language_config,
)
from examples.multimodal_dev.models.kimi_k25.model import KimiK25VLModel

__all__ = [
    "KimiK25VLModel",
    "get_kimi_k25_language_config",
    "KIMI_K25_IMAGE_TOKEN_ID",
    "KIMI_K25_VOCAB_SIZE",
]
