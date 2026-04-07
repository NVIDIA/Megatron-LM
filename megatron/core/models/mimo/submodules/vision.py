# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Dict, List, Optional

import torch.nn as nn

from megatron.core.models.mimo.submodules.base import ModalitySubmodules


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
            **kwargs,
        )

        if self.input_projections:
            assert (
                len(self.input_projections) <= 1
            ), "VisionModalitySubmodules currently supports only one input projection"

        if self.output_projections:
            assert (
                len(self.output_projections) <= 1
            ), "VisionModalitySubmodules currently supports only one output projection"

    def decode(self, embeddings, data_batch: Dict):
        """Decode embeddings into image tensors."""
        raise NotImplementedError("No decoders support yet")
