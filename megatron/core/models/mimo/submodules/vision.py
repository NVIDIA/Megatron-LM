# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from megatron.core.models.mimo.submodules.base import ModalitySubmodules

# Initialize logger
logger = logging.getLogger(__name__)


class VisionModalitySubmodules(ModalitySubmodules):
    """Vision modality submodules for encoding, decoding, and projecting image data.

    Handles image processing through vision encoders and projections in a multi-modal model.
    """

    def __init__(
        self,
        encoders: Optional[Dict[str, nn.Module]] = None,
        decoders: Optional[Dict[str, nn.Module]] = None,
        input_projections: Optional[List[nn.Module]] = None,
        output_projections: Optional[List[nn.Module]] = None,
        **kwargs,
    ):
        """Initialize vision modality submodules.

        Args:
            encoders: Dictionary of encoder modules
            decoders: Dictionary of decoder modules
            input_projections: List of input projection modules
            output_projections: List of output projection modules
            **kwargs: Additional keyword arguments
        """
        super().__init__(
            encoders=encoders,
            decoders=decoders,
            input_projections=input_projections,
            output_projections=output_projections,
        )

        if self.input_projections:
            assert (
                len(self.input_projections) <= 1
            ), "VisionModalitySubmodules currently supports only one input projection"

        if self.output_projections:
            assert (
                len(self.output_projections) <= 1
            ), "VisionModalitySubmodules currently supports only one output projection"

    def encode(self, encoders_data_batch: Dict) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Encode image data batch into a list of tensors.

        Args:
            encoders_data_batch: Dictionary containing encoder-specific inputs.
                Keys should match encoder names in self.encoders.
                Each encoder receives its own specific inputs.

        Returns:
            Tuple of (embeddings list, position_embeddings list)
            Each embedding is a flattened tensor of shape [total_tokens, hidden_dim]

        Raises:
            ValueError: If no data is provided for any encoder or if there's a parameter mismatch.
        """
        if not encoders_data_batch:
            return [], []

        embeddings = []
        token_pos_embeddings = []

        for name, encoder in self.encoders.items():
            if name not in encoders_data_batch:
                raise ValueError(f"No inputs found for encoder '{name}'")

            encoder_inputs = encoders_data_batch[name]

            # Process inputs through the encoder
            for k, v in encoder_inputs.items():
                if hasattr(v, 'shape'):
                    logger.debug(f"Encoder '{name}' input '{k}' shape: {v.shape}")
            encoder_outputs = encoder(**encoder_inputs)
            # Handle case where encoder returns [embeddings, position_embeddings]
            if isinstance(encoder_outputs, (list, tuple)) and len(encoder_outputs) == 2:
                encoder_embed = encoder_outputs[0]
                encoder_pos_emb = encoder_outputs[1]
                token_pos_embeddings.append(encoder_pos_emb)
            else:
                encoder_embed = encoder_outputs
            
            logger.debug(f"Encoder '{name}' output shape: {encoder_embed.shape}")
            if encoder_embed.ndim == 3:
                # its b,s,h -> we need to flatten it to b*s,h
                encoder_embed = encoder_embed.reshape(-1, encoder_embed.size(-1))
                embeddings.append(encoder_embed)
            elif encoder_embed.ndim == 2:
                # its b*s,h -> encoder already returned the flattened output
                embeddings.append(encoder_embed)
            else:
                raise ValueError(
                    f"Encoder '{name}' output shape {encoder_embed.shape} is not supported. "
                    f"Expected 3D (b,s,h) or 2D (b*s,h) tensor, got {encoder_embed.ndim}D"
                )

        return embeddings, token_pos_embeddings

    def decode(self, embeddings: torch.Tensor, data_batch: Dict) -> torch.Tensor:
        """Decode embeddings into image tensors.

        Args:
            embeddings: Tensor of embeddings to decode.
            data_batch: Dictionary containing additional data for decoding.

        Returns:
            Tensor containing generated images.
        """

        raise NotImplementedError("No decoders support yet")

    def combine_embeddings(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Combine multiple embeddings from different encoders by concatenation.

        This method is used for combining encoder outputs before input projection.

        Args:
            embeddings: List of embeddings to combine

        Returns:
            Combined embedding tensor
        """
        if not embeddings:
            raise ValueError("Cannot combine empty list of embeddings")

        if len(embeddings) == 1:
            return embeddings[0]

        # each embedding is [total_tokens, hidden_dim]
        #  Make this configurable in the future
        combined = torch.cat(embeddings, dim=0)
        logger.debug(f"Combined embeddings shape after concatenation: {combined.shape}")
        return combined

    def project_embeddings(
        self, embeddings: List[torch.Tensor], is_input: bool = True
    ) -> torch.Tensor:
        """Project image embeddings using input or output projections.

        Args:
            embeddings: List of image embeddings to project
            is_input: If True, use input projections, otherwise use output projections

        Returns:
            Projected image embeddings or None if no embeddings
        """
        if is_input:
            embeddings = self.combine_embeddings(embeddings)

        # Get the appropriate projection (input or output)
        projections = self.input_projections if is_input else self.output_projections

        # Apply projection if available
        if projections:
            # We've asserted in __init__ that there's only one projection
            projection = projections[0]
            projected = projection(embeddings)
            logger.debug(f"Post-projection embeddings shape: {projected.shape}")
            return projected

        return embeddings

    def forward(self, encoder_inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Process image data through encoding and projection.
        Following Bagel's pipeline:
        1. VIT encoder -> embeddings (vit_hidden_size)
        2. MLP connector (input_projection) -> projects to llm_hidden_size
        3. Position embeddings -> added AFTER projection (in llm_hidden_size)

        Args:
            encoder_inputs: Dictionary where keys match encoder names in self.encoders
                and values are dictionaries of encoder-specific parameters.
                Example: {"clip": {"pixel_values": images}, "vit": {"images": vit_images}}

        Returns:
            Flattened image embeddings with shape [total_embeddings, hidden_dim],
            or None if no valid inputs were provided.
        """
        # Encode the images (returns embeddings list and token_pos_embeddings list)
        embeddings, token_pos_embeddings = self.encode(encoder_inputs)

        # If no embeddings were produced, return None
        if not embeddings:
            return None

        # Project embeddings through MLP connector (input_projection)
        projected = self.project_embeddings(embeddings, is_input=True)
        logger.debug(f"Projected vision embeddings shape: {projected.shape}")
        
        # Add position embeddings AFTER projection (Bagel style)
        # Position embeddings should already be in llm_hidden_size from the encoder
        if token_pos_embeddings and len(token_pos_embeddings) > 0:
            # Combine position embeddings if there are multiple
            if len(token_pos_embeddings) == 1:
                combined_pos_emb = token_pos_embeddings[0]
            else:
                combined_pos_emb = torch.cat(token_pos_embeddings, dim=0)
            
            # Verify shapes match
            if combined_pos_emb.shape == projected.shape:
                projected = projected + combined_pos_emb
                logger.debug(f"Added position embeddings to projected vision embeddings")
            else:
                logger.warning(
                    f"Position embedding shape {combined_pos_emb.shape} doesn't match "
                    f"projected shape {projected.shape}, skipping position embedding addition"
                )
        return projected  # [total_embeddings, hidden_dim]
