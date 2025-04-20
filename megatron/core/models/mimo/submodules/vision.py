# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import Dict, List, Optional

import torch

from megatron.core.models.mimo.submodules.base import ModalitySubmodules

# Initialize logger
logger = logging.getLogger(__name__)


class VisionModalitySubmodules(ModalitySubmodules):
    """Vision modality submodules for encoding, decoding, and projecting image data.

    Handles image processing through vision encoders and projections in a multi-modal model.
    """

    def __init__(
        self,
        encoders=None,
        decoders=None,
        input_projections=None,
        output_projections=None,
        **kwargs,
    ):
        """Initialize vision modality submodules.

        Args:
            encoders: List of encoder modules
            decoders: List of decoder modules
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

    def encode(self, data_batch: Dict) -> List[torch.Tensor]:
        """Encode image data batch into a list of tensors.

        Args:
            data_batch: Dictionary containing input data.
                Expected to have an 'images' key with tensor data of shape [num_images, 3, h, w].
                Each image in the list is treated as an independent input, not grouped by batch.

        Returns:
            List of encoded image embeddings, one from each encoder.
        """
        if "images" not in data_batch:
            return []

        images = data_batch["images"]  # [num_images, 3, h, w]
        logger.debug(f"Input images shape: {images.shape}")

        embeddings = []

        for i, encoder in enumerate(self.encoders):
            # Process all images as independent inputs (not grouped by batch)
            encoder_outputs = encoder(images)  # [num_images, seq_len, hidden_dim]
            logger.debug(
                f"Encoder {i+1}/{len(self.encoders)} "
                f"({type(encoder).__name__}) output shape: {encoder_outputs.shape}"
            )
            embeddings.append(encoder_outputs)

        return embeddings

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

        # concatenate along dimension 1 (seq) for now
        #  Make this configurable in the future
        combined = torch.cat(embeddings, dim=1)
        logger.debug(f"Combined embeddings shape after concatenation: {combined.shape}")
        return combined

    def project_embeddings(
        self, embeddings: List[torch.Tensor], is_input: bool = True
    ) -> Optional[torch.Tensor]:
        """Project image embeddings using input or output projections.

        Args:
            embeddings: List of image embeddings to project
            is_input: If True, use input projections, otherwise use output projections

        Returns:
            Projected image embeddings or None if no embeddings
        """
        if not embeddings:
            return None

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

    def forward(self, data_batch: Dict) -> Optional[torch.Tensor]:
        """Process image data through encoding and projection.

        Args:
            data_batch: Dictionary containing input data with 'images' key.
                Shape [num_tiles, 3, h, w]

        Returns:
            Projected image embeddings or None if no images in batch.
            Output shape is [total_embeddings, hidden_dim], where total_embeddings is the total
            number of embeddings across all images (num_tiles * tile_seq_len).
        """
        # Encode the images
        embeddings = self.encode(data_batch)

        # If no embeddings were produced, return None
        if not embeddings:
            return None

        projected = self.project_embeddings(embeddings, is_input=True)

        num_images, seq_len, hidden_dim = projected.shape
        flattened = projected.reshape(-1, hidden_dim)
        logger.debug(
            f"flattened embeddings shape: {flattened.shape}, "
            f"from {num_images} images with {seq_len} tokens each"
        )

        return flattened  # [total_embeddings, hidden_dim]
