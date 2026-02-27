# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from megatron.core.models.mimo.submodules.base import ModalitySubmodules

# Initialize logger
logger = logging.getLogger(__name__)


class DiffusionModalitySubmodules(ModalitySubmodules):
    """Diffusion modality submodules for encoding, decoding, and projecting image data.

    Handles image processing, timestep and position embeddings through vae encoders and projections in a multi-modal model.
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
            ), "DiffusionModalitySubmodules currently supports only one input projection"

        if self.output_projections:
            assert (
                len(self.output_projections) <= 1
            ), "DiffusionModalitySubmodules currently supports only one output projection"

        self.pre_process = False
        self.post_process = False
        self.share_embeddings_and_output_weights=False
        self.timestep_shift = kwargs['timestep_shift']
        self.dtype = kwargs['dtype'] if 'dtype' in kwargs else torch.float32

    def encode(self, encoders_data_batch: Dict) -> List[torch.Tensor]:
        """Encode image data batch into a list of tensors.

        Args:
            encoders_data_batch: Dictionary containing encoder-specific inputs.
                Keys should match encoder names in self.encoders.
                Each encoder receives its own specific inputs.

        Returns:
            List of encoded image embeddings, one from each encoder.
            Each embedding is a flattened tensor of shape [total_tokens, hidden_dim]

        Raises:
            ValueError: If no data is provided for any encoder or if there's a parameter mismatch.
        """
        if not encoders_data_batch:
            return []

        embeddings = {}
        
        shifted_timesteps = encoders_data_batch['shifted_timesteps']
        packed_timestep_embeds = self.encoders['timestep'](shifted_timesteps)
        embeddings['timestep_emb'] = packed_timestep_embeds

        latent_pos_ids = encoders_data_batch['latent_position_ids']
        latent_pos_emb = self.encoders['latent_position_ids'](latent_pos_ids)
        embeddings['pos_emb'] = latent_pos_emb

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

    def project_embeddings(
        self, embeddings: List[torch.Tensor], is_input: bool = True
    ) -> Optional[torch.Tensor]:

        raise NotImplementedError("No projections support yet")

    def combine_embeddings(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Combine multiple embeddings from different encoders by concatenation.

        This method is used for combining encoder outputs before input projection.

        Args:
            embeddings: List of embeddings to combine

        Returns:
            Combined embedding tensor
        """

        combined = self.input_projections[0](embeddings['latents'].to(self.dtype)) + embeddings['timestep_emb'] + embeddings['pos_emb']

        logger.debug(f"Combined embeddings shape after concatenation: {combined.shape}")
        return combined

    def forward(self, encoder_inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Process image data through encoding and projection.

        Args:
            encoder_inputs: Dictionary where keys match encoder names in self.encoders
                and values are dictionaries of encoder-specific parameters.
                Example: {"clip": {"pixel_values": images}, "vit": {"images": vit_images}}
                For intermediate PP stages: {"_intermediate_tensor": tensor}

        Returns:
            Flattened image embeddings with shape [total_embeddings, hidden_dim],
            or None if no valid inputs were provided.
        """
        embeddings = self.encode(encoder_inputs)
        embeddings['latents'] = encoder_inputs['latents']

        projected = self.combine_embeddings(embeddings)
        logger.debug(f"Projected vision embeddings shape: {projected.shape}")
        return projected  # [total_embeddings, hidden_dim]

    def llm2vae(self, llm_embeddings: torch.Tensor) -> torch.Tensor:
        """Project LLM embeddings to VAE space.

        Args:
            llm_embeddings: Tensor of LLM embeddings to project.

        Returns:
            Tensor of projected VAE embeddings.
        """
        return self.output_projections[0](llm_embeddings)
