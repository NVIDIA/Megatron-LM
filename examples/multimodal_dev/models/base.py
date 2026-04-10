# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Base multimodal model for FSDP + EP training.

Composes a vision encoder and a ``GPTModel`` language decoder.  Designed
for FSDP + EP: always builds the **full** model on every rank (no PP
flags).  PP support is only available through the MIMO ``MimoModel``
assembly path.

Subclasses override ``compute_position_ids()`` for model-specific
position encoding (e.g. MRoPE for Qwen3.5-VL).
"""

from typing import Optional

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


class MultimodalModel(MegatronModule):
    """Base class for multimodal vision-language models.

    Composes a pre-constructed vision encoder and a ``GPTModel`` language
    decoder.  Designed for FSDP + EP; always builds the full model on
    every rank.

    Args:
        language_config: ``TransformerConfig`` for the language decoder.
        language_spec: ``ModuleSpec`` for decoder transformer layers.
        vision_encoder: Pre-constructed vision encoder module.
        vocab_size: Language model vocabulary size.
        max_sequence_length: Maximum sequence length.
        image_token_id: Token ID for image placeholder tokens.
        position_embedding_type: Position embedding type for the decoder.
        rotary_percent: Fraction of hidden dim for RoPE.
        rotary_base: Base frequency for RoPE.
        mrope_section: MRoPE channel sections.
        mtp_block_spec: Optional MTP block spec.
        parallel_output: Keep outputs split across TP ranks.
        share_embeddings_and_output_weights: Tie input/output embeddings.
    """

    def __init__(
        self,
        language_config: TransformerConfig,
        language_spec: ModuleSpec,
        vision_encoder: MegatronModule,
        vocab_size: int,
        max_sequence_length: int,
        image_token_id: int,
        position_embedding_type: str = "rope",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        mrope_section: list = None,
        mtp_block_spec: ModuleSpec = None,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
    ):
        super().__init__(config=language_config)

        self.image_token_id = image_token_id

        self.vision_model = vision_encoder
        self.language_model = GPTModel(
            config=language_config,
            transformer_layer_spec=language_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=True,
            post_process=True,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=(
                share_embeddings_and_output_weights
            ),
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            mtp_block_spec=mtp_block_spec,
        )

    def set_input_tensor(self, input_tensor):
        """Route input tensors (simplified, no PP routing)."""
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1
        self.language_model.set_input_tensor(input_tensor[0])

    def _scatter_vision_embeddings(
        self,
        input_ids: Tensor,
        text_embeddings: Tensor,
        vision_embeddings: Tensor,
    ) -> Tensor:
        """Replace image-token positions with vision embeddings.

        Handles sequence parallelism (gather → scatter → re-scatter).

        Args:
            input_ids: ``[B, S]`` token IDs.
            text_embeddings: ``[S, B, D]`` (or ``[S/TP, B, D]`` with SP).
            vision_embeddings: ``[num_visual_tokens, D]``.

        Returns:
            Combined embeddings, same shape as *text_embeddings*.
        """
        sp = (
            self.config.sequence_parallel
            and parallel_state.get_tensor_model_parallel_world_size()
            > 1
        )

        if sp:
            text_embeddings = (
                tensor_parallel.gather_from_sequence_parallel_region(
                    text_embeddings, tensor_parallel_output_grad=False,
                )
            )

        combined = text_embeddings.transpose(0, 1).contiguous()
        image_mask = input_ids == self.image_token_id
        mask_expanded = image_mask.unsqueeze(-1).expand_as(combined)
        combined = combined.masked_scatter(
            mask_expanded, vision_embeddings,
        )
        combined = combined.transpose(0, 1).contiguous()

        if sp:
            combined = (
                tensor_parallel.scatter_to_sequence_parallel_region(
                    combined,
                )
            )

        return combined

    def compute_position_ids(
        self,
        input_ids: Tensor,
        image_grid_thw: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute position IDs.  Override for MRoPE etc.

        Default: simple sequential positions.

        Args:
            input_ids: ``[B, S]`` token IDs.
            image_grid_thw: ``[num_images, 3]`` grid dimensions.

        Returns:
            Position IDs tensor.
        """
        B, S = input_ids.shape
        return (
            torch.arange(S, device=input_ids.device)
            .unsqueeze(0)
            .expand(B, -1)
        )

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
        **kwargs,
    ):
        """Forward pass.

        Args:
            input_ids: ``[B, S]`` token IDs.
            position_ids: ``[3, B, S]`` for MRoPE or ``[B, S]``.
            attention_mask: ``[B, S]`` attention mask.
            labels: ``[B, S]`` target token IDs.
            loss_mask: ``[B, S]`` mask for loss computation.
            pixel_values: Preprocessed image pixels.
            image_grid_thw: ``[num_images, 3]`` grid dimensions.
            decoder_input: Pre-computed decoder input (skip embed).

        Returns:
            Loss tensor (post_process=True) or hidden states.
        """
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

        output = self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            loss_mask=loss_mask,
        )
        return output
