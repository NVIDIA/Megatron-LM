# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from megatron.core.models.mimo.submodules.base import ModalitySubmodules

# Initialize logger
logger = logging.getLogger(__name__)


class AudioModalitySubmodules(ModalitySubmodules):
    """Audio modality submodules for encoding, decoding, and projecting audio data."""

    def __init__(
        self,
        encoders: Optional[Dict[str, nn.Module]] = None,
        decoders: Optional[Dict[str, nn.Module]] = None,
        input_projections: Optional[List[nn.Module]] = None,
        output_projections: Optional[List[nn.Module]] = None,
        **kwargs,
    ):
        """Initialize audio modality submodules.

        Args:
            encoders: Dictionary of encoder modules
            decoders: Dictionary of decoder modules
            input_projections: List of input projection modules
            output_projections: List of output projection modules
            **kwargs: Additional keyword arguments
        """
        super().__init__(encoders, decoders, input_projections, output_projections, **kwargs)

        if self.input_projections:
            assert (
                len(self.input_projections) <= 1
            ), "AudioModalitySubmodules currently supports only one input projection"

        if self.output_projections:
            assert (
                len(self.output_projections) <= 1
            ), "AudioModalitySubmodules currently supports only one output projection"

    def encode(self, encoders_data_batch: Dict) -> List[torch.Tensor]:
        """Encode audio data into a sequence of embeddings.

        Args:
            encoders_data_batch: Dictionary containing encoder-specific inputs.
                Keys should match encoder names in self.encoders.
                Each encoder receives its own specific inputs.

        Returns:
            List of encoded audio embeddings, one from each encoder.
            Each embedding is a flattened tensor of shape [total_tokens, hidden_dim]

        Raises:
            ValueError: If no data is provided for any encoder or if there's a parameter mismatch.
        """
        if not encoders_data_batch:
            return []

        embeddings = []

        for name, encoder in self.encoders.items():
            if name not in encoders_data_batch:
                raise ValueError(f"No inputs found for encoder '{name}'")

            encoder_inputs = encoders_data_batch[name]

            # Process inputs through the encoder
            encoder_outputs = encoder(**encoder_inputs)
            logger.debug(f"Encoder '{name}' output shape: {encoder_outputs.shape}")
            if encoder_outputs.ndim == 3:
                # its b,s,h -> we need to flatten it to b*s,h
                encoder_outputs = encoder_outputs.reshape(-1, encoder_outputs.size(-1))
            elif encoder_outputs.ndim == 2:
                # its b*s,h -> encoder already returned the flattened output
                embeddings.append(encoder_outputs)
            else:
                raise ValueError(
                    f"Encoder '{name}' output shape {encoder_outputs.shape} is not supported"
                    "Expected 3D (b,s,h) or 2D (b*s,h) tensor, got {encoder_outputs.ndim}D"
                )
        return embeddings

    def decode(self, embeddings: torch.Tensor, data_batch: Dict) -> torch.Tensor:
        """Decode embeddings into audio data."""
        raise NotImplementedError("Audio decoding not implemented yet")

    def combine_embeddings(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Combine embeddings from different encoders."""
        if not embeddings:
            raise ValueError("Cannot combine empty list of embeddings")

        if len(embeddings) == 1:
            return embeddings[0]

        # Concatenate along sequence dimension
        # each embedding is [total_tokens, hidden_dim]
        combined = torch.cat(embeddings, dim=0)
        logger.debug(f"Combined audio embeddings shape: {combined.shape}")
        return combined

    def project_embeddings(
        self, embeddings: List[torch.Tensor], is_input: bool = True
    ) -> torch.Tensor:
        """Project embeddings to the language model dimension space."""

        if is_input:
            embeddings = self.combine_embeddings(embeddings)

        # Get the appropriate projections
        projections = self.input_projections if is_input else self.output_projections

        # Apply projection if available
        if projections:
            # We've asserted in __init__ that there's only one projection
            projection = projections[0]
            projected = projection(embeddings)
            logger.debug(f"Post-projection audio embeddings shape: {projected.shape}")
            return projected

        return embeddings

    def forward(self, encoder_inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Forward pass for audio modality submodules.

        Args:
            encoder_inputs: Dictionary where keys match encoder names in self.encoders
                and values are dictionaries of encoder-specific parameters.
                Example: {
                    "whisper": {"input_features": features},
                    "wav2vec": {"input_values": waveform}
                }

        Returns:
            Flattened audio embeddings with shape [total_embeddings, hidden_dim],
            or None if no valid inputs were provided.
        """

        embeddings = self.encode(encoder_inputs)
        # embeddings is a list of tensors, each tensor is a flattened audio embedding

        # If no embeddings were produced, return None
        if not embeddings:
            return None

        # Project embeddings
        projected = self.project_embeddings(embeddings, is_input=True)
        logger.debug(f"Projected audio embeddings shape: {projected.shape}")
        return projected  # [total_embeddings, hidden_dim]
