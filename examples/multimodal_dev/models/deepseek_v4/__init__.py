# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""DeepSeek-V4-Flash-Vision model package."""

from examples.multimodal_dev.models.deepseek_v4.configuration import (
    DEEPSEEK_V4_FLASH_VISION_COMPRESS_RATIOS,
    DEEPSEEK_V4_FLASH_VISION_HYBRID_PATTERN,
    DEEPSEEK_V4_VOCAB_SIZE,
    build_image_block,
    build_image_token_visibility,
    get_deepseek_v4_vision_config,
)
from examples.multimodal_dev.models.deepseek_v4.factory import (
    build_model,
    post_language_config,
    set_vision_flops_metadata,
)
from examples.multimodal_dev.models.deepseek_v4.model import DeepSeekV4VisionModel
from examples.multimodal_dev.models.deepseek_v4.vision_encoder import DeepSeekV4VisionEncoder

__all__ = [
    "DEEPSEEK_V4_FLASH_VISION_COMPRESS_RATIOS",
    "DEEPSEEK_V4_FLASH_VISION_HYBRID_PATTERN",
    "DEEPSEEK_V4_VOCAB_SIZE",
    "DeepSeekV4VisionEncoder",
    "DeepSeekV4VisionModel",
    "build_image_block",
    "build_image_token_visibility",
    "build_model",
    "get_deepseek_v4_vision_config",
    "post_language_config",
    "set_vision_flops_metadata",
]
