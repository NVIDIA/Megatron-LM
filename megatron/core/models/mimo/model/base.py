# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
import warnings
from typing import Any, Dict, Optional, List
import dataclasses

import torch
import torch.distributed as dist

from megatron.core.models.mimo.config import MimoModelConfig
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import build_module
from megatron.core.utils import unwrap_model

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
        
        # Store input tensors for pipeline parallelism
        # These will be set by set_input_tensor() and used in forward()
        self.modality_input_tensors = {}
        self.language_model_input_tensor = None
    
    def align_embeddings_by_token_positions(
        self,
        modality_embeddings: Dict[str, torch.Tensor],  # [num_embeddings, hidden_dim]
        input_ids: torch.Tensor,  # [bs, seq_len]
        special_token_ids: Dict[str, int],
    ) -> torch.Tensor:
        """Align embeddings from different modalities based on special token positions in input_ids.

        Args:
            modality_embeddings: Dictionary mapping modality names to their embeddings.
                For all modalities: tensor of shape [num_tokens_for_modality, hidden_dim]
            input_ids: Input token IDs of shape [batch_size, seq_len] containing special tokens
                that mark where each modality's embeddings should go. The number of special tokens
                for each modality should exactly match the number of embeddings for that modality.
            special_token_ids: Dictionary mapping modality names to their special token IDs

        Returns:
            Combined embeddings tensor of shape [seq_len, batch_size, hidden_dim]
        """
        # Ensure we have at least one modality
        if not modality_embeddings:
            raise ValueError("No modality embeddings provided. At least one modality is required.")

        logger.debug(f"Merging embeddings for modalities: {list(modality_embeddings.keys())}")

        # Use text embeddings if available, otherwise use any modality
        reference_embeddings = modality_embeddings.get(
            "text", next(iter(modality_embeddings.values()))
        )
        hidden_dim = reference_embeddings.size(-1)
        device = reference_embeddings.device
        dtype = reference_embeddings.dtype

        batch_size, seq_length = input_ids.size()  # input_ids is [b, s]

        logger.debug(
            f"Combined output tensor will have shape: [{seq_length}, {batch_size}, {hidden_dim}]"
        )

        combined_embeddings = torch.zeros(
            (batch_size, seq_length, hidden_dim), dtype=dtype, device=device
        )

        # Process each modality in modality_embeddings
        for modality_name, modality_emb in modality_embeddings.items():
            if modality_name == "text":
                # Text tokens: positions that are not any special token.
                mask = torch.ones_like(input_ids, dtype=torch.bool)
                for token_id in special_token_ids.values():
                    mask &= input_ids != token_id
            elif modality_name in special_token_ids:
                token_id = special_token_ids[modality_name]
                mask = input_ids == token_id
            else:
                raise ValueError(f"No special token ID defined for modality {modality_name}")

            num_tokens = mask.sum().item()
            if num_tokens != modality_emb.size(0):
                raise ValueError(
                    f"Number of {modality_name} tokens ({num_tokens}) does not match "
                    f"number of {modality_name} embeddings ({modality_emb.size(0)})"
                )

            expanded_mask = (
                mask.unsqueeze(-1).expand_as(combined_embeddings).to(combined_embeddings.device)
            )
            combined_embeddings.masked_scatter_(expanded_mask, modality_emb.flatten())
        return combined_embeddings.transpose(
            0, 1
        ).contiguous()  # Shape: [seq_length, batch_size, hidden_dim]


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

    def _get_submodule_pp_group(self, module_spec):
        """Get the pipeline parallel process group for a submodule."""
        params = module_spec.params or {}
        
        # Find pg_collection in params or nested submodules
        pg_collection = params.get('pg_collection') or (
            _find_pg_collection_in_submodules(module_spec.submodules) 
            if hasattr(module_spec, 'submodules') and module_spec.submodules else None
        )
        
        if pg_collection and hasattr(pg_collection, 'pp'):
            return pg_collection.pp
        
        return None
    
    def _is_submodule_pp_first_stage(self, modality_name: str) -> bool:
        """Check if the current rank is at the first PP stage for a given submodule.
        
        Args:
            modality_name: Name of the modality submodule
            
        Returns:
            True if at first PP stage or if no PP is configured, False otherwise
        """
        if modality_name not in self.mimo_config.modality_submodules_spec:
            return True  # Default to True if submodule not found
            
        submodule_spec = self.mimo_config.modality_submodules_spec[modality_name]
        pp_group = self._get_submodule_pp_group(submodule_spec)
        
        if pp_group is None:
            return True  # No PP configured, treat as first stage
            
        return is_pp_first_stage(pp_group)

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

    def set_input_tensor(self, input_tensor: List[Dict[str, torch.Tensor]]):
        """Set input tensor for pipeline parallelism.
        """
        current_rank = dist.get_rank()

        assert isinstance(input_tensor, list), "Input tensor must be a list"
        assert len(input_tensor) == 1, "Input tensor must be a list of length 1"
        assert isinstance(input_tensor[0], dict), "Input tensor[0] must be a dictionary"
        
        input_dict = input_tensor[0]
            
        logger.debug(
            f"[Rank {current_rank}][MimoModel][set_input_tensor] Received dict with keys: {list(input_dict.keys())}"
        )
        
        # Process each modality submodule
        for modality_name, submodule in self.modality_submodules.items():
            if modality_name in input_dict:
                tensor = input_dict[modality_name]
                if isinstance(tensor, list):
                    tensor = tensor[0]
                
                self.modality_input_tensors[modality_name] = tensor
                logger.debug(
                    f"[Rank {current_rank}][MimoModel][set_input_tensor][{modality_name}] "
                    f"Stored input tensor with shape: {tensor.shape}"
                )
                
                # If the submodule has its own set_input_tensor method, call it
                if hasattr(submodule, 'set_input_tensor'):
                    submodule.set_input_tensor(tensor)
        
        if self.language_model is not None and 'language_module' in input_dict:
            lm_tensor = input_dict['language_module']
            if isinstance(lm_tensor, list):
                lm_tensor = lm_tensor[0]
            
            self.language_model_input_tensor = lm_tensor
            logger.debug(
                f"[Rank {current_rank}][MimoModel][set_input_tensor][language_module] "
                f"Stored LM intermediate tensor with shape: {lm_tensor.shape}"
            )
            
            # Pass to language model's set_input_tensor if it exists
            if hasattr(unwrap_model(self.language_model), 'set_input_tensor'):
                unwrap_model(self.language_model).set_input_tensor(lm_tensor)

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


        text_embeddings = unwrap_model(self.language_model).embedding(
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
                Note: This is only required at the first PP stage of each modality.
                At intermediate PP stages, the input comes from set_input_tensor.

        Returns:
            tuple: Tuple containing model outputs and loss mask
        """
        current_rank = dist.get_rank()
        
        # 1. Process each modality to get embeddings
        modality_embeddings = {}

        for modality_name, submodule in self.modality_submodules.items():
            # Skip if submodule is None (not initialized on this rank)
            if submodule is None:
                continue
            
            # Determine input source based on PP stage
            is_first_stage = self._is_submodule_pp_first_stage(modality_name)
            
            if is_first_stage:
                # First PP stage: use modality_inputs provided to forward
                if (
                    modality_inputs
                    and modality_name in modality_inputs
                    and modality_inputs[modality_name] is not None
                ):
                    logger.debug(
                        f"[Rank {current_rank}][MimoModel][forward][{modality_name}] "
                        f"First PP stage, using modality_inputs"
                    )
                    # Get embeddings for this modality
                    embeddings = submodule.forward(encoder_inputs=modality_inputs[modality_name])
                    if embeddings is not None:
                        # All embeddings are now in the format [num_tokens, hidden_dim]
                        modality_embeddings[modality_name] = embeddings
                        logger.debug(
                            f"[Rank {current_rank}][MimoModel][forward][{modality_name}] "
                            f"Generated embeddings with shape {embeddings.shape}"
                        )
                else:
                    logger.debug(
                        f"[Rank {current_rank}][MimoModel][forward][{modality_name}] "
                        f"First PP stage but no modality inputs provided"
                    )
            else:
                # Intermediate PP stage: use stored input tensor from set_input_tensor
                if modality_name in self.modality_input_tensors:
                    input_tensor = self.modality_input_tensors[modality_name]
                    logger.debug(
                        f"[Rank {current_rank}][MimoModel][forward][{modality_name}] "
                        f"Intermediate PP stage, using stored input tensor with shape {input_tensor.shape}"
                    )
                    
                    # Pass through the submodule
                    # Note: The submodule's forward method should handle tensor inputs appropriately
                    embeddings = submodule({'_intermediate_tensor': input_tensor})
                    if embeddings is not None:
                        modality_embeddings[modality_name] = embeddings
                        logger.debug(
                            f"[Rank {current_rank}][MimoModel][forward][{modality_name}] "
                            f"Generated embeddings with shape {embeddings.shape}"
                        )
                else:
                    logger.warning(
                        f"[Rank {current_rank}][MimoModel][forward][{modality_name}] "
                        f"Intermediate PP stage but no input tensor stored from set_input_tensor"
                    )

        # Only process if language model is available on this rank
        if self.language_model is None:
            logger.debug(
                f"[Rank {current_rank}][MimoModel][forward] "
                f"No language model on this rank, returning modality embeddings with keys: {modality_embeddings.keys()}"
            )
            # Return as dictionary for multimodule pipeline compatibility
            # The keys should be the modality names (e.g., 'vision', 'audio')
            return modality_embeddings, loss_mask

        # Check if we're at the first PP stage of the language model
        lm_pp_group = self._get_submodule_pp_group(self.mimo_config.language_model_spec)
        is_lm_first_stage = lm_pp_group is None or is_pp_first_stage(lm_pp_group)
        
        if is_lm_first_stage:
            # First LM PP stage: process modality embeddings and text embeddings
            logger.debug(
                f"[Rank {current_rank}][MimoModel][forward] "
                f"Language model first PP stage, processing text and modality embeddings"
            )
            
            # At LM first stage, modality submodules are None (not initialized on this rank)
            # We need to use modality embeddings that were received from encoders
            # via set_input_tensor (stored in self.modality_input_tensors)
            # These are the final outputs from encoder PP stages
            logger.debug(
                f"[Rank {current_rank}][MimoModel][forward] "
                f"Available stored modality embeddings: {list(self.modality_input_tensors.keys())}"
            )
            
            for modality_name, stored_embedding in self.modality_input_tensors.items():
                if stored_embedding is not None:
                    logger.debug(
                        f"[Rank {current_rank}][MimoModel][forward] "
                        f"Using stored {modality_name} embedding from set_input_tensor with shape {stored_embedding.shape}"
                    )
                    modality_embeddings[modality_name] = stored_embedding
            
            # Check if we have any modality embeddings (excluding locally computed ones from first loop)
            # If special tokens for modalities are present but we don't have embeddings, warn
            if not self.modality_input_tensors and not modality_embeddings:
                logger.warning(
                    f"[Rank {current_rank}][MimoModel][forward] "
                    f"LM first stage but no modality embeddings received via set_input_tensor. "
                    f"This may be expected if encoders are on the same rank, or an error if using multimodule pipeline."
                )
            
            # Get text embeddings
            text_embeddings = self.get_text_embeddings(input_ids, position_ids, self.special_token_ids)
            logger.debug(
                f"[Rank {current_rank}][MimoModel][forward] "
                f"Generated text embeddings with shape {text_embeddings.shape}"
            )

            modality_embeddings["text"] = text_embeddings

            # Merge embeddings from different modalities
            logger.debug(
                f"[Rank {current_rank}][MimoModel][forward] "
                f"Merging embeddings from {len(modality_embeddings)} modalities"
            )
            combined_embeddings = self.align_embeddings_by_token_positions(
                modality_embeddings=modality_embeddings,  # [num_tokens, hidden_dim] for each modality
                input_ids=input_ids,  # Pass in batch-first format [b, s]
                special_token_ids=self.special_token_ids,
            )  # [s, b, h]
            logger.debug(
                f"[Rank {current_rank}][MimoModel][forward] "
                f"Combined embeddings shape: {combined_embeddings.shape}"
            )

            # Forward pass through language model
            lm_output = self.language_model(
                input_ids=None,
                position_ids=None,
                decoder_input=combined_embeddings,
                labels=labels,
                attention_mask=attention_mask,
            )
        else:
            # Intermediate LM PP stage: use stored input tensor
            if self.language_model_input_tensor is not None:
                logger.debug(
                    f"[Rank {current_rank}][MimoModel][forward] "
                    f"Language model intermediate PP stage, using stored input tensor with shape {self.language_model_input_tensor.shape}"
                )
                # The language model's set_input_tensor should have already been called
                # Just do the forward pass
                lm_output = self.language_model(
                    input_ids=None,
                    position_ids=None,
                    decoder_input=None,  # The LM will use its stored input tensor
                    labels=labels,
                    attention_mask=attention_mask,
                )
            else:
                raise RuntimeError(
                    f"[Rank {current_rank}][MimoModel][forward] "
                    f"Language model at intermediate PP stage but no input tensor stored from set_input_tensor"
                )
        
        logger.debug(
            f"[Rank {current_rank}][MimoModel][forward] "
            f"Language model output shape: {lm_output.shape}"
        )
        
        if not is_pp_last_stage(lm_pp_group):
            return {'language_module': lm_output}, loss_mask


        return lm_output, loss_mask
