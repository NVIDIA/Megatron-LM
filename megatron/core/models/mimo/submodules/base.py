# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from megatron.core.transformer.spec_utils import ModuleSpec, build_module

# Initialize logger
logger = logging.getLogger(__name__)


class ModalitySubmodules(ABC, nn.Module):
    """Base abstract class for modality-specific submodules.

    Manages encoders, decoders, and projection layers for a specific modality
    in a multi-modal model architecture. Subclasses must implement methods for
    encoding, decoding, combining embeddings, and projecting embeddings.

    .. warning::
        **EXPERIMENTAL**: This class is experimental, still under active development,
        and the API is subject to change without notice. Use at your own risk.

    Args:
        encoders (Dict[str, nn.Module]):
            Dictionary of encoder modules for processing modality inputs
        decoders (Dict[str, nn.Module]):
            Dictionary of decoder modules for generating modality outputs
        input_projections (List[nn.Module]):
            List of projection modules for transforming encoder outputs
        output_projections (List[nn.Module]):
            List of projection modules for transforming decoder inputs
    """

    def __init__(
        self,
        encoders: Optional[Dict[str, nn.Module]] = None,
        decoders: Optional[Dict[str, nn.Module]] = None,
        input_projections: Optional[List[nn.Module]] = None,
        output_projections: Optional[List[nn.Module]] = None,
        **kwargs,
    ) -> None:
        """Initialize the modality submodules."""
        super().__init__()
        self.encoders = nn.ModuleDict(encoders or {})
        self.decoders = nn.ModuleDict(decoders or {})
        self.input_projections = nn.ModuleList(input_projections or [])
        self.output_projections = nn.ModuleList(output_projections or [])

        warnings.warn(
            "ModalitySubmodules is experimental and still under active development. "
            "The API may change without notice in future releases.",
            category=UserWarning,
            stacklevel=2,
        )

    @classmethod
    def from_spec(cls, module_spec: ModuleSpec) -> 'ModalitySubmodules':
        """Create a modality submodule from ModuleSpec configuration.

        Args:
            module_spec (ModuleSpec): The module specification for this modality submodule

        Returns:
            ModalitySubmodules: An instance of the modality submodule
        """
        logger.debug(f"Creating {cls.__name__} from spec")
        params = module_spec.params or {}
        submodules = module_spec.submodules or {}

        # Build component lists from submodules dictionary
        encoders = {}
        if 'encoders' in submodules:
            for encoder_name, encoder_spec in submodules['encoders'].items():
                logger.debug(f"Building {cls.__name__} encoder: {encoder_spec.module.__name__}")
                encoder = build_module(encoder_spec)
                encoders[encoder_name] = encoder

        decoders = {}
        if 'decoders' in submodules:
            for decoder_name, decoder_spec in submodules['decoders'].items():
                logger.debug(f"Building {cls.__name__} decoder: {decoder_spec.module.__name__}")
                decoder = build_module(decoder_spec)
                decoders[decoder_name] = decoder

        input_projections = []
        if 'input_projections' in submodules:
            for proj_spec in submodules['input_projections']:
                logger.debug(
                    f"Building {cls.__name__} input projection: {proj_spec.module.__name__}"
                )
                projection = build_module(proj_spec)
                input_projections.append(projection)

        output_projections = []
        if 'output_projections' in submodules:
            for proj_spec in submodules['output_projections']:
                logger.debug(
                    f"Building {cls.__name__} output projection: {proj_spec.module.__name__}"
                )
                projection = build_module(proj_spec)
                output_projections.append(projection)

        # Pass any additional parameters from the params dictionary
        additional_params = params.copy()
        if additional_params:
            logger.debug(
                f"Using additional parameters for {cls.__name__}: {list(additional_params.keys())}"
            )

        return cls(
            encoders=encoders,
            decoders=decoders,
            input_projections=input_projections,
            output_projections=output_projections,
            **additional_params,
        )

    @abstractmethod
    def combine_embeddings(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Combine multiple embeddings from different encoders.

        Args:
            embeddings (List[torch.Tensor]):
                List of embeddings to combine

        Returns:
            torch.Tensor: Combined embedding tensor
        """
        pass

    @abstractmethod
    def encode(self, data_batch: Dict) -> List[torch.Tensor]:
        """Encode data batch into a list of tensors.

        Args:
            data_batch (Dict):
                Dictionary containing input data

        Returns:
            List[torch.Tensor]: List of encoded embeddings
        """
        pass

    @abstractmethod
    def decode(self, embeddings: torch.Tensor, data_batch: Dict) -> torch.Tensor:
        """Decode embeddings into a tensor.

        Args:
            embeddings (torch.Tensor):
                Embeddings to decode
            data_batch (Dict):
                Dictionary containing additional data for decoding

        Returns:
            torch.Tensor: Decoded output
        """
        pass

    @abstractmethod
    def project_embeddings(
        self, embeddings: List[torch.Tensor], is_input: bool = True
    ) -> Optional[torch.Tensor]:
        """Project embeddings into a tensor.

        Args:
            embeddings (List[torch.Tensor]):
                List of embeddings to project
            is_input (bool):
                If True, use input projections, otherwise use output projections

        Returns:
            Optional[torch.Tensor]: Projected embeddings or None
        """
        pass

    @abstractmethod
    def forward(self, encoder_inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Process data for this modality through encoding and projection.

        Args:
            encoder_inputs (Dict[str, Any]):
                Dictionary containing encoder-specific inputs. Keys should match encoder names.

        Returns:
            Optional[torch.Tensor]:
                Processed and projected embeddings tensor, or None if no embeddings were produced.
        """
        pass
