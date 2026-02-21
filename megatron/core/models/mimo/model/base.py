# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
import warnings
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from megatron.core.models.mimo.config.role import ModuleStageInfo, RankRole

from megatron.core.models.mimo.config import MimoModelConfig
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import build_module
from megatron.core.utils import unwrap_model

logger = logging.getLogger(__name__)


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
        self._validate_grid_map()
        self.role = self._determine_role()

        # Use special token IDs from the config
        self.special_token_ids = (
            mimo_config.special_token_ids.copy() if mimo_config.special_token_ids else {}
        )

        # Initialize modality submodules from specifications
        self.modality_submodules = torch.nn.ModuleDict()
        self._initialize_submodules()
        self._initialize_language_model()

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

    def _initialize_submodules(self) -> None:
        """Initialize modality submodules from the ModuleSpec configurations.

        Only modalities present in the config will be instantiated.
        When role is set, only initializes submodules this rank participates in.
        Stage info is passed to from_spec() to conditionally skip projection
        initialization on non-last stages (saves memory in pipeline parallelism).
        """
        for modality_name, submodule_spec in self.mimo_config.modality_submodules_spec.items():
            # Skip if we have a role and this module isn't in it
            if self.role is not None and modality_name not in self.role.modules:
                logger.debug(f"Skipping {modality_name} submodule (not in role)")
                continue

            # Determine stage info for this module
            is_first_stage = True
            is_last_stage = True
            if self.role is not None and modality_name in self.role.modules:
                stage_info = self.role.modules[modality_name]
                is_first_stage = stage_info.is_first_stage
                is_last_stage = stage_info.is_last_stage

            submodule_class = submodule_spec.module
            logger.debug(
                f"Building {modality_name} submodule using {submodule_class.__name__} "
                f"(is_first_stage={is_first_stage}, is_last_stage={is_last_stage})"
            )

            # Pass stage info to from_spec so projections are only built when needed
            submodule = submodule_class.from_spec(
                submodule_spec,
                is_first_stage=is_first_stage,
                is_last_stage=is_last_stage,
            )

            self.modality_submodules[modality_name] = submodule

    def _initialize_language_model(self) -> None:
        """Initialize the language model.

        When role is set, only initializes if this rank participates in language module.
        """
        # Skip if we have a role and don't participate in language module
        if self.role is not None and not self.role.has_language_module:
            logger.debug("Skipping language model initialization (not in role)")
            self.language_model = None
            return

        logger.debug(
            f"Building language model using {self.mimo_config.language_model_spec.module.__name__}"
        )
        self.language_model = build_module(self.mimo_config.language_model_spec)

    def _validate_grid_map(self) -> None:
        """Validate module_to_grid_map consistency with submodule config.

        Validates that:
        - language_module_key is set when module_to_grid_map is provided
        - module_to_grid_map keys exactly match modality_submodules_spec keys + language_module_key

        Raises:
            ValueError: If validation fails.
        """
        if not self.mimo_config.module_to_grid_map:
            return

        # Require language_module_key when using multi-module PP
        if self.mimo_config.language_module_key is None:
            raise ValueError(
                "language_module_key must be set when module_to_grid_map is provided. "
                "Specify which module key identifies the language model."
            )

        grid_map_keys = set(self.mimo_config.module_to_grid_map.keys())
        submodule_keys = set(self.mimo_config.modality_submodules_spec.keys())
        submodule_keys.add(self.mimo_config.language_module_key)

        if grid_map_keys != submodule_keys:
            missing_in_grid = submodule_keys - grid_map_keys
            extra_in_grid = grid_map_keys - submodule_keys
            raise ValueError(
                f"module_to_grid_map keys must match modality_submodules_spec keys + "
                f"language_module_key. Missing in grid_map: {missing_in_grid}, "
                f"Extra in grid_map: {extra_in_grid}"
            )

    def _determine_role(self) -> Optional[RankRole]:
        """Determine this rank's role based on grid map.

        Returns:
            RankRole describing which modules this rank participates in,
            or None if module_to_grid_map is not set (all modules on all ranks).
        """
        if not self.mimo_config.module_to_grid_map:
            return None

        current_rank = dist.get_rank()
        modules = {}

        for module_name, grid in self.mimo_config.module_to_grid_map.items():
            # Check if current rank is in this grid
            if not (grid.rank_offset <= current_rank < grid.rank_offset + grid.size):
                continue

            # Check if PP dimension exists
            if "pp" not in grid.dim_names:
                # No PP dimension means single stage (both first and last)
                modules[module_name] = ModuleStageInfo(
                    is_first_stage=True,
                    is_last_stage=True,
                )
                continue

            # Get PP process group and determine stage
            pp_group = grid.get_pg("pp")
            pp_rank = pp_group.rank()
            pp_size = pp_group.size()
            is_first = (pp_rank == 0)
            is_last = (pp_rank == pp_size - 1)
            logger.info(
                f"[_determine_role] Rank {current_rank}: module={module_name}, "
                f"pp_rank={pp_rank}/{pp_size}, is_first_stage={is_first}, is_last_stage={is_last}"
            )
            modules[module_name] = ModuleStageInfo(
                is_first_stage=is_first,
                is_last_stage=is_last,
            )

        return RankRole(
            modules=modules,
            language_module_name=self.mimo_config.language_module_key,
        )

    def set_input_tensor(self, input_tensor):
        """Set input tensor for pipeline parallelism.

        This method is required by Megatron's pipeline parallel mechanism.
        It passes the output tensor from the previous stage as input to this stage.

        Args:
            input_tensor: Either:
                - Dict[str, Tensor]: Maps module names to their input tensors (for multi-module PP)
                - Tensor or List[Tensor]: Single tensor for language model (backward compat)

        Returns:
            None
        """
        # Store dict input for multi-module PP
        if isinstance(input_tensor, dict):
            self.input_tensors = input_tensor
            return

        # Backward compatibility: single tensor or list
        if isinstance(input_tensor, list):
            input_tensor = input_tensor[0]

        # Store as input_tensors for consistency
        self.input_tensors = input_tensor

        # Also delegate to language model for backward compatibility
        if self.language_model is not None and hasattr(self.language_model, 'set_input_tensor'):
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
            modality_inputs: Dictionary mapping modality names to encoder inputs.

        Returns:
            tuple: (output, loss_mask) where output semantics depend on role:
                - Encoder-only ranks: Dict[str, Tensor] of encoder outputs
                - Language module ranks: language model output (logits or loss)
                - No role (all modules colocated): language model output
        """
        # Get any tensors passed via set_input_tensor
        input_tensors = getattr(self, 'input_tensors', None)

        if self.role is None:
            # Original behavior: all modules on all ranks
            return self._forward_all_modules(
                input_ids, position_ids, attention_mask,
                loss_mask, labels, modality_inputs
            )

        if self.role.has_modality_modules and not self.role.has_language_module:
            # Encoder-only rank
            return self._forward_encoders(modality_inputs, input_tensors), loss_mask

        if self.role.has_language_module and not self.role.has_modality_modules:
            # Language-module-only rank
            return self._forward_language_module(
                input_ids, position_ids, attention_mask,
                labels, input_tensors
            ), loss_mask

        if self.role.has_modality_modules and self.role.has_language_module:
            # Colocated encoders and language module is a configuration error
            raise ValueError(
                "Invalid configuration: Colocated encoders and language module on the same "
                "rank is not supported in multi-module pipeline parallelism. Use separate "
                "grids for encoders and language module, or disable multi-module PP by not "
                "setting module_to_grid_map."
            )

        raise RuntimeError(f"Rank has no modules assigned in role: {self.role}")

    def _forward_encoders(
        self,
        modality_inputs: Optional[Dict[str, Dict[str, Any]]],
        input_tensors: Optional[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for encoder modules on this rank.

        Args:
            modality_inputs: Raw inputs for each modality (images, audio, etc.)
            input_tensors: Hidden states from previous pipeline stages

        Returns:
            Dict mapping encoder names to their output tensors
        """
        outputs = {}

        for encoder_name in self.role.modality_module_names:
            if encoder_name not in self.modality_submodules:
                continue

            submodule = self.modality_submodules[encoder_name]

            # Determine input based on stage position
            if self.role.is_first_stage(encoder_name):
                # First stage: use raw modality inputs
                encoder_input = modality_inputs.get(encoder_name) if modality_inputs else None
                if encoder_input is not None:
                    output = submodule.forward(encoder_inputs=encoder_input)
                else:
                    output = None
            else:
                # Non-first stage: use hidden states from previous stage
                hidden_states = input_tensors.get(encoder_name) if input_tensors else None
                if hidden_states is not None:
                    output = submodule.forward(hidden_states=hidden_states)
                else:
                    output = None

            if output is not None:
                outputs[encoder_name] = output

        return outputs

    def _forward_language_module(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        input_tensors: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Forward pass for language module on this rank.

        Args:
            input_ids: Token IDs
            position_ids: Position IDs
            attention_mask: Attention mask
            labels: Labels for loss computation
            input_tensors: Hidden states or embeddings from previous stage

        Returns:
            Language model output (hidden states, logits, or loss depending on stage)
        """
        lang_name = self.role.language_module_name

        if self.role.is_first_stage(lang_name):
            # First stage: receive encoder embeddings, combine with text, pass to LM
            # Build modality embeddings dict from encoder outputs
            modality_embeddings = {}
            if input_tensors:
                for name, tensor in input_tensors.items():
                    if name != lang_name:
                        modality_embeddings[name] = tensor

            # Get text embeddings
            text_embeddings = self.get_text_embeddings(
                input_ids, position_ids, self.special_token_ids
            )
            modality_embeddings["text"] = text_embeddings

            # Combine all embeddings
            combined_embeddings = self.align_embeddings_by_token_positions(
                modality_embeddings=modality_embeddings,
                input_ids=input_ids,
                special_token_ids=self.special_token_ids,
            )

            lm_output = self.language_model(
                input_ids=None,
                position_ids=None,
                decoder_input=combined_embeddings,
                labels=labels,
                attention_mask=attention_mask,
            )
        else:
            # Non-first stage: receive hidden states from previous LM stage
            hidden_states = input_tensors.get(lang_name) if input_tensors else None

            # Set input tensor on language model for PP
            if hidden_states is not None and hasattr(self.language_model, 'set_input_tensor'):
                self.language_model.set_input_tensor(hidden_states)

            lm_output = self.language_model(
                input_ids=None,
                position_ids=None,
                decoder_input=None,
                labels=labels,
                attention_mask=attention_mask,
            )

        # Key output for non-last stages so schedule can route to next LM stage
        if not self.role.is_last_stage(lang_name):
            return {lang_name: lm_output}

        return lm_output

    def _forward_all_modules(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        loss_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        modality_inputs: Optional[Dict[str, Dict[str, Any]]],
    ):
        """Forward pass when all modules are on all ranks (no multi-module PP).

        This is the original behavior, preserved for backward compatibility.
        """
        # 1. Process each modality to get embeddings
        modality_embeddings = {}

        for modality_name, submodule in self.modality_submodules.items():
            if (
                modality_inputs
                and modality_name in modality_inputs
                and modality_inputs[modality_name] is not None
            ):
                logger.debug(f"Processing {modality_name} modality")
                embeddings = submodule.forward(encoder_inputs=modality_inputs[modality_name])
                if embeddings is not None:
                    modality_embeddings[modality_name] = embeddings
                    logger.debug(
                        f"Generated embeddings for {modality_name} with shape {embeddings.shape}"
                    )

        # Get text embeddings
        text_embeddings = self.get_text_embeddings(input_ids, position_ids, self.special_token_ids)
        logger.debug(f"Generated text embeddings with shape {text_embeddings.shape}")

        modality_embeddings["text"] = text_embeddings

        # 2. Merge embeddings from different modalities
        logger.debug(f"Merging embeddings from {len(modality_embeddings)} modalities")
        combined_embeddings = self.align_embeddings_by_token_positions(
            modality_embeddings=modality_embeddings,
            input_ids=input_ids,
            special_token_ids=self.special_token_ids,
        )
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
