# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Qwen3.5-VL multimodal model for standalone FSDP + EP training.

Composes a Megatron-native Qwen3.5 vision encoder with a ``GPTModel``
language decoder using MRoPE and hybrid GatedDeltaNet / full-attention
layers.
"""

from typing import Optional

import torch
from torch import Tensor

from examples.multimodal_dev.models.base import (
    _NO_CP_GROUP,
    MultimodalModel,
    _cp_debug_dump,
    _cp_split_tensor,
    _thd_cp_partition_index,
)
from examples.multimodal_dev.models.qwen35_vl.configuration import (
    MROPE_SECTION,
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


    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor = None,
        labels: Tensor = None,
        loss_mask: Tensor = None,
        pixel_values: Tensor = None,
        image_grid_thw: Tensor = None,
        decoder_input: Tensor = None,
        packed_seq_params=None,
        **kwargs,
    ):
        """Forward pass.

        Args:
            input_ids: ``[B, S]`` token IDs (or ``[1, T]`` in THD mode).
            position_ids: ``[3, B, S]`` for MRoPE or ``[B, S]``
                (``[3, 1, T]`` / ``[1, T]`` in THD mode).
            attention_mask: ``[B, S]`` attention mask (None in THD).
            labels: ``[B, S]`` target token IDs (``[1, T]`` in THD).
            loss_mask: ``[B, S]`` mask for loss (``[1, T]`` in THD).
            pixel_values: Preprocessed image pixels.
            image_grid_thw: ``[num_images, 3]`` grid dimensions.
            decoder_input: Pre-computed decoder input (skip embed).
            packed_seq_params: ``PackedSeqParams`` for THD attention.

        Returns:
            Loss tensor (post_process=True) or hidden states.
        """
        # Compute position_ids before packing (MRoPE needs [B, S] input_ids).
        if position_ids is None:
            position_ids = self.compute_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                packed_seq_params=packed_seq_params,
            )
        vision_embeddings = None
        if (
            self.vision_model is not None
            and pixel_values is not None
        ):
            vision_embeddings = self.vision_model(
                pixel_values, image_grid_thw,
            )

        if decoder_input is None and self.language_model is not None:
            text_embeddings = self.language_model.embedding(
                input_ids=input_ids, position_ids=None,
            )

            if vision_embeddings is not None:
                decoder_input = self._scatter_vision_embeddings(
                    input_ids, text_embeddings, vision_embeddings,
                )
            else:
                decoder_input = text_embeddings

        # --- Context Parallelism: split along sequence dimension ---
        # BSHD path: zigzag-split tensors along seq dim. position_ids is
        #   left alone — the MRoPE/RoPE embedding handles its own CP
        #   slicing via ``get_pos_emb_on_this_cp_rank()``.
        # THD path: use TE's ``thd_get_partitioned_indices`` so each
        #   packed sub-sample is split per-sample. MRoPE position_ids
        #   were pre-computed on the full [1, T] layout above and MUST
        #   be sliced with the same per-rank index here.
        from megatron.core import parallel_state
        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size > 1:
            cp_rank = parallel_state.get_context_parallel_rank()
            if packed_seq_params is not None:
                total_tokens = (
                    decoder_input.shape[0]
                    if decoder_input is not None
                    else input_ids.shape[1]
                )
                idx = _thd_cp_partition_index(
                    packed_seq_params.cu_seqlens_q_padded,
                    total_tokens, cp_size, cp_rank,
                )
                if decoder_input is not None:
                    decoder_input = decoder_input.index_select(0, idx)
                if input_ids is not None:
                    input_ids = input_ids.index_select(1, idx)
                if labels is not None:
                    labels = labels.index_select(1, idx)
                if loss_mask is not None:
                    loss_mask = loss_mask.index_select(1, idx)
                # position_ids is left at full length [3, 1, T]. MRoPE
                # returns full freqs; attention's _apply_rotary_pos_emb_thd
                # does per-sample CP zigzag via _get_thd_freqs_on_this_cp_rank
                # using cu_seqlens. We do NOT set packed_seq_params.cp_group
                # here — MTP reads it at runtime and would lose CP-aware
                # rolling of labels/loss_mask if forced to size-1.
                # MRoPE's BSHD-style zigzag is bypassed by mutating
                # mrope.cp_group directly below.
                # attention_mask is None in THD; skip.
            else:
                if decoder_input is not None:
                    decoder_input = _cp_split_tensor(
                        decoder_input, seq_dim=0,
                        cp_size=cp_size, cp_rank=cp_rank,
                    )
                if input_ids is not None:
                    input_ids = _cp_split_tensor(
                        input_ids, seq_dim=1,
                        cp_size=cp_size, cp_rank=cp_rank,
                    )
                if labels is not None:
                    labels = _cp_split_tensor(
                        labels, seq_dim=1,
                        cp_size=cp_size, cp_rank=cp_rank,
                    )
                if loss_mask is not None:
                    loss_mask = _cp_split_tensor(
                        loss_mask, seq_dim=1,
                        cp_size=cp_size, cp_rank=cp_rank,
                    )
                if attention_mask is not None:
                    attention_mask = _cp_split_tensor(
                        attention_mask, seq_dim=1,
                        cp_size=cp_size, cp_rank=cp_rank,
                    )

            _cp_debug_dump(
                "Qwen35VLModel.forward post-CP",
                decoder_input=decoder_input,
                input_ids=input_ids,
                labels=labels,
                loss_mask=loss_mask,
                position_ids=position_ids,
                cu_seqlens_q_padded=(
                    packed_seq_params.cu_seqlens_q_padded
                    if packed_seq_params is not None
                    else None
                ),
            )

        # In THD mode, temporarily override MRoPE's cp_group on the
        # language_model so it skips its BSHD-style zigzag of pre-computed
        # MRoPE embeddings.  Only affects ``MultimodalRotaryEmbedding``;
        # attention's RoPE / CP path and MTP both keep their real cp_group.
        _mrope = (
            getattr(self.language_model, "rotary_pos_emb", None)
            if packed_seq_params is not None
            and parallel_state.get_context_parallel_world_size() > 1
            else None
        )
        _saved_mrope_cp_group = (
            getattr(_mrope, "cp_group", None) if _mrope is not None else None
        )
        if _mrope is not None:
            _mrope.cp_group = _NO_CP_GROUP
        try:
            output = self.language_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=decoder_input,
                labels=labels,
                loss_mask=loss_mask,
                packed_seq_params=packed_seq_params,
            )
        finally:
            if _mrope is not None:
                _mrope.cp_group = _saved_mrope_cp_group

        return output
