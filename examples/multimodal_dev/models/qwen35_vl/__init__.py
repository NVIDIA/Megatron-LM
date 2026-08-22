# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Qwen3.5-VL model components — the single source of truth.

Both the standalone ``multimodal_dev`` training path and the MIMO path
import from here.
"""

from examples.multimodal_dev.models.qwen35_vl.configuration import (
    MROPE_SECTION,
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VIDEO_TOKEN_ID,
    QWEN35_VL_VISION_END_TOKEN_ID,
    QWEN35_VL_VISION_START_TOKEN_ID,
    QWEN35_VL_VOCAB_SIZE,
    ROTARY_BASE,
    ROTARY_PERCENT,
    VISION_KWARGS,
    get_qwen35_vl_language_config,
    get_qwen35_vl_vision_config,
)
from examples.multimodal_dev.models.qwen35_vl.factory import (
    build_model,
    post_language_config,
    set_vision_flops_metadata,
)
from examples.multimodal_dev.models.qwen35_vl.model import Qwen35VLModel
from examples.multimodal_dev.models.qwen35_vl.mrope import get_rope_index
from examples.multimodal_dev.models.qwen35_vl.specs import (
    get_qwen35_vl_language_spec,
    get_qwen35_vl_vision_spec,
)
from examples.multimodal_dev.models.qwen35_vl.vision_encoder import (
    Qwen35VLPatchEmbed,
    Qwen35VLPatchMerger,
    Qwen35VLVisionEncoder,
    Qwen35VLVisionRotaryEmbedding,
)

__all__ = [
    # Model class
    "Qwen35VLModel",
    # Factory functions
    "build_model",
    "post_language_config",
    "set_vision_flops_metadata",
    # Vision encoder
    "Qwen35VLVisionEncoder",
    "Qwen35VLPatchEmbed",
    "Qwen35VLPatchMerger",
    "Qwen35VLVisionRotaryEmbedding",
    # Config helpers
    "get_qwen35_vl_vision_config",
    "get_qwen35_vl_language_config",
    # Spec helpers
    "get_qwen35_vl_language_spec",
    "get_qwen35_vl_vision_spec",
    # MRoPE
    "get_rope_index",
    # Constants
    "QWEN35_VL_IMAGE_TOKEN_ID",
    "QWEN35_VL_VIDEO_TOKEN_ID",
    "QWEN35_VL_VISION_START_TOKEN_ID",
    "QWEN35_VL_VISION_END_TOKEN_ID",
    "QWEN35_VL_VOCAB_SIZE",
    "ROTARY_BASE",
    "ROTARY_PERCENT",
    "MROPE_SECTION",
    "VISION_KWARGS",
]
