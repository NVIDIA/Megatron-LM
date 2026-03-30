# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Qwen3.5-VL model components for the MIMO training path.

This package is a self-contained duplicate of the qwen35_vl model code.
All imports reference local siblings rather than ``multimodal_v2``.
"""

from .configuration import (
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
from .mrope import compute_mrope_position_ids
from .specs import (
    get_qwen35_vl_language_spec,
    get_qwen35_vl_vision_spec,
)
from .vision_encoder import (
    Qwen35VLPatchEmbed,
    Qwen35VLPatchMerger,
    Qwen35VLVisionEncoder,
    Qwen35VLVisionRotaryEmbedding,
)

__all__ = [
    # Vision encoder  (Qwen35VLModel excluded — not used in MIMO path)
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
    "compute_mrope_position_ids",
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
