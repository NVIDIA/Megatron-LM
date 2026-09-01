# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""DeepSeek-V4-Flash-Vision model with a native HybridModel decoder."""

from typing import Optional

import torch
from torch import Tensor, nn

from examples.multimodal_dev.models.base import MultimodalModel
from examples.multimodal_dev.models.deepseek_v4.configuration import (
    DEEPSEEK_V4_VOCAB_SIZE,
    IMAGE,
    NUM_IMAGE_TOKEN_TYPES,
    VISION_MAX_IMAGE_TOKENS,
    build_image_token_visibility,
    image_token_id,
)
from examples.multimodal_dev.models.deepseek_v4.vision_encoder import DeepSeekV4VisionEncoder
from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.hybrid.hybrid_layer_allocation import parse_hybrid_pattern
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


class DeepSeekV4VisionModel(MultimodalModel):
    """Compose the DSv4 vision tower, decoder-side image embeddings, and HybridModel.

    ``vision_embeddings`` plus ``vision_token_indices`` in :meth:`forward` form the stable
    boundary for a future MDP vision stage. The native and external paths share all decoder-side
    image-token semantics.
    """

    def __init__(
        self,
        language_config: TransformerConfig,
        hybrid_stack_spec: ModuleSpec,
        hybrid_layer_pattern: str,
        vision_config: TransformerConfig,
        vocab_size: int = DEEPSEEK_V4_VOCAB_SIZE,
        actual_vocab_size: int = DEEPSEEK_V4_VOCAB_SIZE,
        max_sequence_length: int = 4096,
        build_vision_encoder: bool = True,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
    ) -> None:
        parsed_pattern = parse_hybrid_pattern(hybrid_layer_pattern)
        if parsed_pattern.mtp_num_depths:
            raise ValueError("DeepSeek-V4-Vision phase one does not support MTP.")
        if actual_vocab_size > vocab_size:
            raise ValueError(
                f"actual_vocab_size={actual_vocab_size} exceeds vocab_size={vocab_size}."
            )

        self.actual_vocab_size = actual_vocab_size
        self.max_image_tokens = vision_config.vision_max_image_tokens
        vision_config.params_dtype = language_config.params_dtype
        vision_config.vision_out_hidden_size = language_config.hidden_size
        vision_encoder = DeepSeekV4VisionEncoder(vision_config) if build_vision_encoder else None

        super().__init__(
            language_config=language_config,
            language_spec=None,
            vision_encoder=vision_encoder,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            image_token_id=image_token_id(IMAGE, actual_vocab_size),
            position_embedding_type="rope",
            rotary_percent=language_config.rotary_percent,
            rotary_base=int(language_config.rotary_base),
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            hybrid_stack_spec=hybrid_stack_spec,
            hybrid_layer_pattern=hybrid_layer_pattern,
        )

        dtype = language_config.params_dtype
        self.image_start = nn.Parameter(torch.empty(language_config.hidden_size, dtype=dtype))
        self.image_end = nn.Parameter(torch.empty(language_config.hidden_size, dtype=dtype))
        self.image_newline = nn.Parameter(torch.empty(language_config.hidden_size, dtype=dtype))
        self.image_pad = nn.Parameter(torch.empty(language_config.hidden_size, dtype=dtype))
        if language_config.perform_initialization:
            for parameter in (self.image_start, self.image_end, self.image_newline, self.image_pad):
                language_config.init_method(parameter)

    def _embed_input_ids(self, input_ids: Tensor) -> Tensor:
        """Embed text IDs while replacing synthetic image IDs with a safe lookup row."""
        safe_input_ids = torch.where(
            (input_ids >= 0) & (input_ids < self.actual_vocab_size),
            input_ids,
            torch.zeros_like(input_ids),
        )
        return self.language_model.embedding(input_ids=safe_input_ids, position_ids=None)

    def _special_embedding_table(self) -> Tensor:
        # IMAGE uses image_pad as a temporary value and is overwritten by encoded rows below.
        return torch.stack(
            (self.image_start, self.image_pad, self.image_pad, self.image_newline, self.image_end)
        )

    @staticmethod
    def _flatten_vision_token_indices(indices: Tensor, batch_size: int, seqlen: int) -> Tensor:
        if indices.ndim == 1:
            return indices.long()
        if indices.ndim == 2 and indices.shape[1] == 2:
            return (indices[:, 0].long() * seqlen + indices[:, 1].long()).long()
        raise ValueError(
            "vision_token_indices must be flattened positions [N] or (batch, sequence) pairs [N, 2]."
        )

    def _scatter_vision_embeddings(
        self,
        input_ids: Tensor,
        text_embeddings: Tensor,
        vision_embeddings: Tensor,
        vision_token_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """Install special image embeddings and encoded IMAGE rows into decoder input."""
        sequence_parallel = (
            self.config.sequence_parallel
            and parallel_state.get_tensor_model_parallel_world_size() > 1
        )
        if sequence_parallel:
            text_embeddings = tensor_parallel.gather_from_sequence_parallel_region(
                text_embeddings, tensor_parallel_output_grad=False
            )

        batch_size, seqlen = input_ids.shape
        combined = text_embeddings.transpose(0, 1).contiguous()
        flat_embeddings = combined.view(-1, combined.shape[-1])
        flat_ids = input_ids.reshape(-1)
        image_types = flat_ids - self.actual_vocab_size
        visual_mask = (image_types >= 0) & (image_types < NUM_IMAGE_TOKEN_TYPES)
        invalid_visual = (flat_ids >= self.actual_vocab_size) & (~visual_mask)
        if invalid_visual.any():
            invalid_id = int(flat_ids[invalid_visual][0].item())
            raise ValueError(f"Unknown synthetic DeepSeek image token ID {invalid_id}.")

        visual_positions = torch.nonzero(visual_mask, as_tuple=False).flatten()
        visual_types = image_types.index_select(0, visual_positions)
        special_rows = self._special_embedding_table().index_select(0, visual_types)
        flat_embeddings = flat_embeddings.index_copy(
            0,
            visual_positions,
            special_rows.to(device=flat_embeddings.device, dtype=flat_embeddings.dtype),
        )

        if vision_embeddings.ndim == 3 and vision_embeddings.shape[1] == 1:
            vision_embeddings = vision_embeddings.squeeze(1)
        if vision_embeddings.ndim != 2:
            raise ValueError(
                f"vision_embeddings must be [num_rows, hidden], got {tuple(vision_embeddings.shape)}."
            )
        if vision_token_indices is None:
            decoder_positions = torch.nonzero(image_types == IMAGE, as_tuple=False).flatten()
        else:
            decoder_positions = self._flatten_vision_token_indices(
                vision_token_indices, batch_size, seqlen
            ).to(device=input_ids.device)
            if (decoder_positions < 0).any() or (decoder_positions >= flat_ids.numel()).any():
                raise ValueError("vision_token_indices contains an out-of-range decoder position.")
            if not torch.all(image_types.index_select(0, decoder_positions) == IMAGE):
                raise ValueError("Every vision_token_indices entry must point to an IMAGE token.")
        if decoder_positions.numel() != vision_embeddings.shape[0]:
            raise ValueError(
                f"Found {decoder_positions.numel()} IMAGE positions but received "
                f"{vision_embeddings.shape[0]} vision embedding rows."
            )
        flat_embeddings = flat_embeddings.index_copy(
            0,
            decoder_positions,
            vision_embeddings.to(device=flat_embeddings.device, dtype=flat_embeddings.dtype),
        )
        combined = flat_embeddings.view(batch_size, seqlen, -1).transpose(0, 1).contiguous()

        if sequence_parallel:
            combined = tensor_parallel.scatter_to_sequence_parallel_region(combined)
        return combined

    def prepare_attention_mask(self, input_ids: Tensor, attention_mask, packed_seq_params=None):
        """Expand each image span bidirectionally inside DSv4's sparse window."""
        has_image_tokens = torch.any(input_ids >= self.actual_vocab_size)
        if not has_image_tokens:
            return attention_mask
        if attention_mask is not None:
            raise ValueError(
                "DeepSeek-V4 image-span visibility cannot be combined with another attention mask."
            )
        return build_image_token_visibility(
            input_ids, vocab_size=self.actual_vocab_size, max_image_tokens=self.max_image_tokens
        )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor],
        *,
        pixel_values: Optional[Tensor] = None,
        vision_embeddings: Optional[Tensor] = None,
        decoder_input: Optional[Tensor] = None,
        **kwargs,
    ):
        """Run native vision encoding or consume externally supplied visual rows."""
        has_image_tokens = torch.any(input_ids >= self.actual_vocab_size)
        if has_image_tokens and all(
            value is None for value in (pixel_values, vision_embeddings, decoder_input)
        ):
            raise ValueError(
                "Image tokens require pixel_values, vision_embeddings, or a complete decoder_input."
            )
        if self.vision_model is None and pixel_values is not None and vision_embeddings is None:
            raise ValueError(
                "This model was built without a local vision encoder; pass vision_embeddings "
                "from the external/MDP vision stage instead of pixel_values."
            )
        return super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            pixel_values=pixel_values,
            vision_embeddings=vision_embeddings,
            decoder_input=decoder_input,
            **kwargs,
        )
