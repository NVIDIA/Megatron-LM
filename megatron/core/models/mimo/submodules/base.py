# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

try:
    from nemo.lens.helpers import managed_span as _otel_managed_span
    from nemo.lens.helpers import safe_set_span_attributes as _otel_safe_set_attrs
    from nemo.lens.state import is_span_group_enabled as _otel_sg_enabled
except ImportError:
    from megatron.core.telemetry.fallbacks import managed_span as _otel_managed_span
    from megatron.core.telemetry.fallbacks import safe_set_span_attributes as _otel_safe_set_attrs
    from megatron.core.telemetry.fallbacks import is_span_group_enabled as _otel_sg_enabled

from megatron.core._rank_utils import warn_single_rank
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import sharded_state_dict_default

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
        is_first_stage: bool = True,
        is_last_stage: bool = True,
        pg_collection=None,
        **kwargs,
    ) -> None:
        """Initialize the modality submodules.

        Args:
            encoders: Dict of encoder modules
            decoders: Dict of decoder modules
            input_projections: List of input projection modules
            output_projections: List of output projection modules
            is_first_stage: Whether this is the first PP stage for this module
            is_last_stage: Whether this is the last PP stage for this module
            pg_collection: Process group collection for this module
        """
        super().__init__()
        self.encoders = nn.ModuleDict(encoders or {})
        self.decoders = nn.ModuleDict(decoders or {})
        self.input_projections = nn.ModuleList(input_projections or [])
        self.output_projections = nn.ModuleList(output_projections or [])
        self.pg_collection = pg_collection
        self._is_first_stage: bool = is_first_stage
        self._is_last_stage: bool = is_last_stage

        warn_single_rank(
            "ModalitySubmodules is experimental and still under active development. "
            "The API may change without notice in future releases."
        )

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Iterate into ModuleDict/ModuleList children for TP-aware checkpointing.

        Injects dp_cp_group from pg_collection into metadata to avoid
        parallel_state global fallback in ensure_metadata_has_dp_cp_group.
        """
        if self.pg_collection is not None:
            assert (
                hasattr(self.pg_collection, 'dp_cp') and self.pg_collection.dp_cp is not None
            ), "pg_collection is missing dp_cp group"
            metadata = dict(metadata) if metadata else {}
            metadata['dp_cp_group'] = (
                getattr(self.pg_collection, 'dp_cp_gtp_remat', None) or self.pg_collection.dp_cp
            )

        sharded_sd = {}
        for name, container in self.named_children():
            for sub_name, module in container.named_children():
                sharded_sd.update(
                    sharded_state_dict_default(
                        module, f'{prefix}{name}.{sub_name}.', sharded_offsets, metadata
                    )
                )
        return sharded_sd

    @property
    def is_first_stage(self) -> bool:
        """Whether this is the first pipeline stage for this module."""
        return self._is_first_stage

    @property
    def is_last_stage(self) -> bool:
        """Whether this is the last pipeline stage for this module."""
        return self._is_last_stage

    @classmethod
    def from_spec(
        cls, module_spec: ModuleSpec, is_first_stage: bool = True, is_last_stage: bool = True
    ) -> 'ModalitySubmodules':
        """Create a modality submodule from ModuleSpec configuration.

        Args:
            module_spec (ModuleSpec): The module specification for this modality submodule
            is_first_stage (bool): Whether this is the first pipeline stage for this module.
                Controls encoder initialization and output projection initialization
                (output projections only built on first stage). Defaults to True.
            is_last_stage (bool): Whether this is the last pipeline stage for this module.
                Controls input projection initialization (only built on last stage).
                Defaults to True.

        Returns:
            ModalitySubmodules: An instance of the modality submodule
        """
        logger.debug(
            f"Creating {cls.__name__} from spec (is_first_stage={is_first_stage}, "
            f"is_last_stage={is_last_stage})"
        )
        params = module_spec.params or {}
        submodules = module_spec.submodules or {}

        # Build encoders (needed on all stages for pipeline processing)
        encoders = {}
        if 'encoders' in submodules:
            for encoder_name, encoder_spec in submodules['encoders'].items():
                logger.debug(f"Building {cls.__name__} encoder: {encoder_spec.module.__name__}")
                encoder = build_module(encoder_spec)
                encoders[encoder_name] = encoder

        # Build decoders (needed on all stages for pipeline processing)
        decoders = {}
        if 'decoders' in submodules:
            for decoder_name, decoder_spec in submodules['decoders'].items():
                logger.debug(f"Building {cls.__name__} decoder: {decoder_spec.module.__name__}")
                decoder = build_module(decoder_spec)
                decoders[decoder_name] = decoder

        # Build input projections only on last stage
        # (projection happens after encoding, before sending to language model)
        input_projections = []
        if is_last_stage and 'input_projections' in submodules:
            for proj_spec in submodules['input_projections']:
                logger.debug(
                    f"Building {cls.__name__} input projection: {proj_spec.module.__name__}"
                )
                projection = build_module(proj_spec)
                input_projections.append(projection)
        elif 'input_projections' in submodules:
            logger.debug(f"Skipping {cls.__name__} input projections (not last stage)")

        # Build output projections only on first stage
        # (projection happens before decoding, after receiving from language model)
        output_projections = []
        if is_first_stage and 'output_projections' in submodules:
            for proj_spec in submodules['output_projections']:
                logger.debug(
                    f"Building {cls.__name__} output projection: {proj_spec.module.__name__}"
                )
                projection = build_module(proj_spec)
                output_projections.append(projection)
        elif 'output_projections' in submodules:
            logger.debug(f"Skipping {cls.__name__} output projections (not first stage)")

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
            is_first_stage=is_first_stage,
            is_last_stage=is_last_stage,
            **additional_params,
        )

    def combine_embeddings(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Combine multiple embeddings from different encoders by concatenation.

        Args:
            embeddings (List[torch.Tensor]):
                List of embeddings to combine. Each is [total_tokens, hidden_dim].

        Returns:
            torch.Tensor: Combined embedding tensor
        """
        if not embeddings:
            raise ValueError("Cannot combine empty list of embeddings")

        if len(embeddings) == 1:
            return embeddings[0]

        combined = torch.cat(embeddings, dim=0)
        logger.debug(f"Combined embeddings shape after concatenation: {combined.shape}")
        return combined

    @staticmethod
    def _primary_encoder_output(encoder_outputs):
        """Return the primary tensor from an encoder result."""
        if (
            isinstance(encoder_outputs, tuple)
            and encoder_outputs
            and torch.is_tensor(encoder_outputs[0])
        ):
            return encoder_outputs[0]
        return encoder_outputs

    def encode(self, encoders_data_batch: Dict) -> List[torch.Tensor]:
        """Encode data batch into a list of tensors.

        Args:
            encoders_data_batch (Dict):
                Dictionary containing encoder-specific inputs.
                Keys should match encoder names in self.encoders.

        Returns:
            List[torch.Tensor]: List of encoded embeddings, each [total_tokens, hidden_dim]
        """
        if not encoders_data_batch:
            return []

        embeddings = []

        for name, encoder in self.encoders.items():
            if name not in encoders_data_batch:
                raise ValueError(f"No inputs found for encoder '{name}'")

            encoder_inputs = encoders_data_batch[name]
            if not _otel_sg_enabled('microbatch'):
                encoder_outputs = self._primary_encoder_output(encoder(**encoder_inputs))
            else:
                frozen_cache = self.__dict__.setdefault('_telemetry_encoder_frozen', {})
                if name not in frozen_cache:
                    frozen_cache[name] = not any(
                        parameter.requires_grad for parameter in encoder.parameters()
                    )
                primary_input = next(
                    (value for value in encoder_inputs.values() if torch.is_tensor(value)), None
                )
                attributes = {
                    'megatron.mimo.encoder.name': name,
                    'megatron.mimo.encoder.frozen': frozen_cache[name],
                }
                if primary_input is not None:
                    attributes.update(
                        {
                            'megatron.mimo.input.shape': list(primary_input.shape),
                            'megatron.mimo.input.dtype': str(primary_input.dtype),
                        }
                    )
                with _otel_managed_span(
                    'microbatch', 'megatron.mimo.encoder.forward', **attributes
                ) as span:
                    encoder_outputs = self._primary_encoder_output(encoder(**encoder_inputs))
                    _otel_safe_set_attrs(
                        span,
                        {
                            'megatron.mimo.output.shape': list(encoder_outputs.shape),
                            'megatron.mimo.output.dtype': str(encoder_outputs.dtype),
                        },
                    )
            logger.debug(f"Encoder '{name}' output shape: {encoder_outputs.shape}")

            if encoder_outputs.ndim == 3:
                encoder_outputs = encoder_outputs.reshape(-1, encoder_outputs.size(-1))
            elif encoder_outputs.ndim != 2:
                raise ValueError(
                    f"Encoder '{name}' output shape {encoder_outputs.shape} is not supported. "
                    f"Expected 3D (b,s,h) or 2D (b*s,h) tensor, got {encoder_outputs.ndim}D"
                )

            embeddings.append(encoder_outputs)

        return embeddings

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

    def project_embeddings(
        self, embeddings: List[torch.Tensor], is_input: bool = True
    ) -> Optional[torch.Tensor]:
        """Project embeddings using input or output projections.

        Args:
            embeddings (List[torch.Tensor]):
                List of embeddings to project
            is_input (bool):
                If True, use input projections, otherwise use output projections

        Returns:
            Optional[torch.Tensor]: Projected embeddings or None
        """
        combined = self.combine_embeddings(embeddings)

        projections = self.input_projections if is_input else self.output_projections

        if projections:
            projection = projections[0]
            if not _otel_sg_enabled('microbatch'):
                projected = projection(combined)
            else:
                with _otel_managed_span(
                    'microbatch',
                    'megatron.mimo.projection.forward',
                    **{
                        'megatron.mimo.projection.placement': 'encoder',
                        'megatron.mimo.projection.direction': 'input' if is_input else 'output',
                        'megatron.mimo.input.shape': list(combined.shape),
                        'megatron.mimo.input.dtype': str(combined.dtype),
                    },
                ) as span:
                    projected = projection(combined)
                    _otel_safe_set_attrs(
                        span,
                        {
                            'megatron.mimo.output.shape': list(projected.shape),
                            'megatron.mimo.output.dtype': str(projected.dtype),
                        },
                    )
            logger.debug(f"Post-projection embeddings shape: {projected.shape}")
            return projected

        return combined

    def forward(
        self,
        encoder_inputs: Optional[Dict[str, Any]] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Process data for this modality through encoding and projection.

        Args:
            encoder_inputs (Dict[str, Any]):
                Dictionary containing encoder-specific inputs. Keys should match encoder names.
                Used when is_first_stage=True.
            hidden_states (Optional[torch.Tensor]):
                Already-combined encoder features. When supplied, bypasses encoding on any stage.

        Returns:
            Optional[torch.Tensor]:
                Processed and projected embeddings tensor, or None if no embeddings were produced.
        """
        if encoder_inputs is not None and hidden_states is not None:
            raise ValueError("encoder_inputs and hidden_states are mutually exclusive")

        if hidden_states is not None:
            combined = hidden_states
        elif self.is_first_stage:
            if encoder_inputs is None:
                return None
            embeddings = self.encode(encoder_inputs)
            if not embeddings:
                return None
            combined = self.combine_embeddings(embeddings)
        else:
            return None

        if self.is_last_stage:
            return self.project_embeddings([combined], is_input=True)

        return combined
