# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
import warnings
from typing import Any, Dict, Optional
import dataclasses

import torch
import torch.distributed as dist

from megatron.core.models.mimo.config import MimoModelConfig
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import build_module

logger = logging.getLogger(__name__)


def _find_pg_collection_in_submodules(submodules) -> Optional[object]:
    """Recursively search for pg_collection in nested submodules."""
    if isinstance(submodules, dict):
        for nested_spec in submodules.values():
            if isinstance(nested_spec, dict):
                # Handle {"clip_encoder": spec}
                for spec in nested_spec.values():
                    if hasattr(spec, 'params') and spec.params and 'pg_collection' in spec.params:
                        return spec.params['pg_collection']
            elif isinstance(nested_spec, list):
                # Handle [spec1, spec2, ...]
                for spec in nested_spec:
                    if hasattr(spec, 'params') and spec.params and 'pg_collection' in spec.params:
                        return spec.params['pg_collection']
            elif hasattr(nested_spec, 'params') and nested_spec.params and 'pg_collection' in nested_spec.params:
                # Handle direct ModuleSpec
                return nested_spec.params['pg_collection']
    return None


class MimoModel(MegatronModule):
    """Multimodal In/Out Model supporting arbitrary combinations of modalities.

    .. warning::
        **EXPERIMENTAL**: This class is experimental, still under active development,
        and the API is subject to change without notice. Use at your own risk.

    .. note::
        This implementation is in development and may undergo API changes.


    This model processes multiple modalities (e.g., vision, audio) alongside text,
    combining their embeddings before passing them through a language model.

    Args:
        mimo_config (MimoModelConfig):
            Configuration for the model, including language model and modality submodules
    """

    def __init__(self, mimo_config: MimoModelConfig) -> None:
        """Initialize the multimodal model.

        Example:
            ```python
            # Create a model with default configuration
            model = MimoModel(mimo_config)
            ```
        """
        # Initialize with language model's transformer config for MegatronModule compatibility
        super().__init__(mimo_config.language_model_spec.params['config'])

        warnings.warn(
            "MimoModel is experimental and still under active development. "
            "The API may change without notice in future releases.",
            category=UserWarning,
            stacklevel=2,
        )

        self.mimo_config = mimo_config

        # Use special token IDs from the config
        self.special_token_ids = (
            mimo_config.special_token_ids.copy() if mimo_config.special_token_ids else {}
        )

        # Initialize modality submodules from specifications
        self.modality_submodules = torch.nn.ModuleDict()
        self._initialize_submodules()
        self._initialize_language_model()

    def _should_initialize_module(self, module_spec) -> bool:
        """Determine if the current rank should initialize a module based on its process groups."""
        params = module_spec.params or {}
        
        # Find pg_collection in params or nested submodules
        pg_collection = params.get('pg_collection') or (
            _find_pg_collection_in_submodules(module_spec.submodules) 
            if hasattr(module_spec, 'submodules') and module_spec.submodules else None
        )
        
        # No pg_collection means initialize on all ranks
        if not pg_collection:
            return True
            
        # Check if current rank is in any process group
        current_rank = dist.get_rank()
        for field in dataclasses.fields(pg_collection):
            pg = getattr(pg_collection, field.name, None)
            if pg and current_rank in dist.get_process_group_ranks(pg):
                return True
        
        return False

    def _initialize_submodules(self) -> None:
        """Initialize modality submodules from the ModuleSpec configurations.

        Only modalities present in the config will be instantiated.
        For each modality in the config, builds the corresponding submodule using from_spec.
        """

        for modality_name, submodule_spec in self.mimo_config.modality_submodules_spec.items():
            # Check if current rank should initialize this submodule
            if not self._should_initialize_module(submodule_spec):
                logger.debug(f"Rank {dist.get_rank()} skipping {modality_name} submodule initialization")
                self.modality_submodules[modality_name] = None
                continue
                
            # Get the submodule class
            submodule_class = submodule_spec.module
            logger.debug(f"[Rank - {dist.get_rank()}] Building {modality_name} submodule using {submodule_class.__name__}")

            # Use from_spec to instantiate the submodule
            submodule = submodule_class.from_spec(submodule_spec)
            self.modality_submodules[modality_name] = submodule

    def _initialize_language_model(self) -> None:
        """Initialize the language model."""
        # Check if current rank should initialize the language model
        if not self._should_initialize_module(self.mimo_config.language_model_spec):
            logger.debug(f"Rank {dist.get_rank()} skipping language model initialization")
            self.language_model = None
            return
            
        logger.debug(
            f"[Rank - {dist.get_rank()} Building language model using {self.mimo_config.language_model_spec.module.__name__}"
        )
        self.language_model = build_module(self.mimo_config.language_model_spec)

    def set_input_tensor(self, input_tensor):
        """Set input tensor for pipeline parallelism.

        This method is required by Megatron's pipeline parallel mechanism.
        It passes the output tensor from the previous stage as input to this stage.

        Args:
            input_tensor: Tensor or list of tensors passed between pipeline stages

        Returns:
            None
        """
        # Handle case where input_tensor might be a list or a single tensor
        if isinstance(input_tensor, list):
            # For simplicity, just use the first tensor
            input_tensor = input_tensor[0]

        # Pass the input tensor to the language model if it has a set_input_tensor method
        if hasattr(self.language_model, 'set_input_tensor'):
            self.language_model.set_input_tensor(input_tensor)

    def get_text_embeddings(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, special_token_ids: Dict[str, int]
    ) -> torch.Tensor:
        """Get embeddings for text tokens in the input.
        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len] containing text tokens
                and potentially special tokens for other modalities.
            position_ids: Position IDs corresponding to input tokens, used for positional encoding.
                Shape [batch_size, seq_len].
            special_token_ids: Dictionary mapping modality names to their special token IDs.
                Used to identify non-text tokens in the input_ids.

        Returns:
            torch.Tensor: Embeddings for text tokens, shape [num_text_tokens, hidden_dim].
        """
        if self.language_model is None:
            raise RuntimeError(f"Language model not initialized on rank {dist.get_rank()}")
            
        text_mask = torch.ones_like(input_ids, dtype=torch.bool)  # [b, s]
        for special_token_id in special_token_ids.values():
            text_mask &= input_ids != special_token_id

        batch_idx, seq_idx = text_mask.nonzero(as_tuple=True)
        input_ids_text = input_ids[batch_idx, seq_idx].unsqueeze(0)

        position_ids_text = (
            position_ids[batch_idx, seq_idx].unsqueeze(0) if position_ids is not None else None
        )

        text_embeddings = self.language_model.embedding(
            input_ids=input_ids_text, position_ids=position_ids_text
        ).squeeze(
            1
        )  # Shape: [num_text_tokens, hidden_dim]
        return text_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        modality_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Forward pass through the multimodal model.

        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            position_ids: Position IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            loss_mask: Loss mask [batch_size, seq_length]
            labels: Labels for training
            modality_inputs: Dictionary mapping modality names to encoder inputs. For example:
                {
                    "images": {
                        "clip_encoder": {"pixel_values": clip_images},
                        "vit_encoder": {"images": vit_images}
                    },
                    "audio": {
                        "whisper_encoder": {"input_features": whisper_features}
                    }
                }

        Returns:
            tuple: Tuple containing model outputs and loss mask
        """
        # 1. Process each modality to get embeddings
        modality_embeddings = {}

        for modality_name, submodule in self.modality_submodules.items():
            # Skip if submodule is None (not initialized on this rank)
            if submodule is None:
                continue
                
            # Process the modality through its submodule
            if (
                modality_inputs
                and modality_name in modality_inputs
                and modality_inputs[modality_name] is not None
            ):
                logger.debug(f"Processing {modality_name} modality")
                # Get embeddings for this modality
                embeddings = submodule.forward(encoder_inputs=modality_inputs[modality_name])
                if embeddings is not None:
                    # All embeddings are now in the format [num_tokens, hidden_dim]
                    modality_embeddings[modality_name] = embeddings
                    logger.debug(
                        f"Generated embeddings for {modality_name} with shape {embeddings.shape}"
                    )

        # Only process if language model is available on this rank
        if self.language_model is None:
            logger.debug(f"Rank {dist.get_rank()} has no language model, returning modality embeddings")
            return modality_embeddings, loss_mask

        # Get text embeddings
        text_embeddings = self.get_text_embeddings(input_ids, position_ids, self.special_token_ids)
        logger.debug(f"Generated text embeddings with shape {text_embeddings.shape}")

        modality_embeddings["text"] = text_embeddings

        # 2. Merge embeddings from different modalities
        logger.debug(f"Merging embeddings from {len(modality_embeddings)} modalities")
        combined_embeddings = self.align_embeddings_by_token_positions(
            modality_embeddings=modality_embeddings,  # [num_tokens, hidden_dim] for each modality
            input_ids=input_ids,  # Pass in batch-first format [b, s]
            special_token_ids=self.special_token_ids,
        )  # [s, b, h]
        logger.debug(f"Combined embeddings shape: {combined_embeddings.shape}")

        # 3. Forward pass through language model
        lm_output = self.language_model(
            input_ids=None,
            position_ids=None,
            decoder_input=combined_embeddings,
            labels=labels,
            attention_mask=attention_mask,
        )
        logger.debug(f"Language model output shape: {lm_output.shape}")

        return lm_output, loss_mask
