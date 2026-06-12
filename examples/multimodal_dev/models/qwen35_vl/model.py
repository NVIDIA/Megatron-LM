# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Qwen3.5-VL multimodal model for standalone FSDP + EP training.

Composes a Megatron-native Qwen3.5 vision encoder with a ``GPTModel``
language decoder using MRoPE and hybrid GatedDeltaNet / full-attention
layers.
"""

from typing import Optional

from torch import Tensor

from examples.multimodal_dev.models.base import MultimodalModel
from examples.multimodal_dev.models.qwen35_vl.configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VIDEO_TOKEN_ID,
    QWEN35_VL_VISION_START_TOKEN_ID,
    QWEN35_VL_VOCAB_SIZE,
    ROTARY_BASE,
    ROTARY_PERCENT,
    VISION_KWARGS,
)
from examples.multimodal_dev.models.qwen35_vl.mrope import get_rope_index
from examples.multimodal_dev.models.qwen35_vl.specs import get_qwen35_vl_vision_spec
from examples.multimodal_dev.models.qwen35_vl.vision_encoder import Qwen35VLVisionEncoder
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


class Qwen35VLModel(MultimodalModel):
    """Qwen3.5-VL multimodal model.

    Args:
        language_config: ``TransformerConfig`` for the language decoder.
        language_spec: ``ModuleSpec`` for language decoder layers.
        vision_config: ``TransformerConfig`` for the vision encoder.
        vision_spec: ``ModuleSpec`` for vision encoder layers.
        vocab_size: Vocabulary size.
        max_sequence_length: Maximum sequence length.
        image_token_id: Token ID for image placeholders.
        spatial_merge_size: Vision encoder spatial merge factor.
        mtp_block_spec: Optional MTP block spec.
        parallel_output: Keep outputs split across TP.
        share_embeddings_and_output_weights: Tie embeddings.
    """

    def __init__(
        self,
        language_config: TransformerConfig,
        language_spec: ModuleSpec,
        vision_config: TransformerConfig,
        vision_spec: ModuleSpec = None,
        vocab_size: int = QWEN35_VL_VOCAB_SIZE,
        max_sequence_length: int = 262144,
        image_token_id: int = QWEN35_VL_IMAGE_TOKEN_ID,
        video_token_id: int = QWEN35_VL_VIDEO_TOKEN_ID,
        vision_start_token_id: int = QWEN35_VL_VISION_START_TOKEN_ID,
        spatial_merge_size: int = 2,
        mtp_block_spec: ModuleSpec = None,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
    ):
        if vision_spec is None:
            vision_spec = get_qwen35_vl_vision_spec()

        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.spatial_merge_size = spatial_merge_size

        vkw = dict(VISION_KWARGS)
        vkw["spatial_merge_size"] = spatial_merge_size
        vkw["out_hidden_size"] = language_config.hidden_size

        vision_encoder = Qwen35VLVisionEncoder(
            config=vision_config,
            transformer_layer_spec=vision_spec,
            in_channels=vkw["in_channels"],
            patch_size=vkw["patch_size"],
            temporal_patch_size=vkw["temporal_patch_size"],
            spatial_merge_size=vkw["spatial_merge_size"],
            out_hidden_size=vkw["out_hidden_size"],
            max_num_positions=vkw["max_num_positions"],
        )

        super().__init__(
            language_config=language_config,
            language_spec=language_spec,
            vision_encoder=vision_encoder,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            image_token_id=image_token_id,
            position_embedding_type="mrope",
            rotary_percent=ROTARY_PERCENT,
            rotary_base=ROTARY_BASE,
            mrope_section=language_config.mrope_section,
            mtp_block_spec=mtp_block_spec,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=(
                share_embeddings_and_output_weights
            ),
        )

    def compute_position_ids(
        self,
        input_ids: Tensor,
        image_grid_thw: Optional[Tensor] = None,
        packed_seq_params=None,
    ) -> Tensor:
        """Compute 3D MRoPE position IDs for Qwen3.5-VL.

        In THD mode ``input_ids`` is ``[1, T]`` and ``packed_seq_params``
        supplies per-segment boundaries; positions restart at 0 per
        segment. In BSHD mode ``input_ids`` is ``[B, S]`` and
        ``packed_seq_params`` should be ``None``.

        Returns:
            ``[3, B, S]`` position IDs for MRoPE (``[3, 1, T]`` in THD).
        """
        position_ids, _ = get_rope_index(
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            packed_seq_params=packed_seq_params,
        )
        return position_ids
