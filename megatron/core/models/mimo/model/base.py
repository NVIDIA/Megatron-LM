# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
import warnings
from typing import Any, Dict, List, Optional

import torch

from megatron.core.models.mimo.config import MimoModelConfig
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import build_module

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
        For each modality in the config, builds the corresponding submodule using from_spec.
        """

        for modality_name, submodule_spec in self.mimo_config.modality_submodules_spec.items():
            # Get the submodule class
            submodule_class = submodule_spec.module
            logger.debug(f"Building {modality_name} submodule using {submodule_class.__name__}")

            # Use from_spec to instantiate the submodule
            submodule = submodule_class.from_spec(submodule_spec)
            self.modality_submodules[modality_name] = submodule

    def _initialize_language_model(self) -> None:
        """Initialize the language model."""
        logger.debug(
            f"Building language model using {self.mimo_config.language_model_spec.module.__name__}"
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

    def _get_language_model_embeddings(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings from the language model.

        Handles both HuggingFace (has embedding method) and Megatron Core (has get_word_embeddings method).

        Args:
            input_ids: [batch_size, seq_len]
            position_ids: [batch_size, seq_len]

        Returns:
            embeddings: [seq_len, batch_size, hidden_dim]
        """
        # Check for HuggingFace style (callable embedding method)
        if hasattr(self.language_model, 'embedding') and callable(self.language_model.embedding):
            return self.language_model.embedding(input_ids=input_ids, position_ids=position_ids)

        # Check for Megatron Core BagelMCoreModel style
        if hasattr(self.language_model, 'get_word_embeddings'):
            return self.language_model.get_word_embeddings(input_ids=input_ids, position_ids=position_ids)

        # Fallback: Megatron Core GPTModel style - use embedding module directly
        if hasattr(self.language_model, 'embedding') and hasattr(self.language_model.embedding, 'word_embeddings'):
            embeddings = self.language_model.embedding.word_embeddings(input_ids)
            return embeddings.transpose(0, 1).contiguous()

        raise RuntimeError("Language model does not have a recognized embedding interface")

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

        text_embeddings = self._get_language_model_embeddings(
            input_ids=input_ids_text, position_ids=position_ids_text
        ).squeeze(
            1
        )  # Shape: [num_text_tokens, hidden_dim]
        return text_embeddings

    def get_all_text_embeddings(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for ALL tokens in input_ids (for Bagel-style where input_ids contains only text).

        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]
            position_ids: Position IDs corresponding to input tokens. Shape [batch_size, seq_len].

        Returns:
            torch.Tensor: Embeddings for all tokens, shape [num_tokens, hidden_dim].
        """
        text_embeddings = self._get_language_model_embeddings(
            input_ids=input_ids, position_ids=position_ids
        ).squeeze(1)  # Shape: [num_tokens, hidden_dim]
        return text_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        modality_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
        # Parameters for Bagel-style training
        sample_lens: Optional[List[int]] = None,
        packed_position_ids: Optional[torch.Tensor] = None,
        ce_loss_indexes: Optional[torch.Tensor] = None,
        packed_label_ids: Optional[torch.Tensor] = None,
        # Additional Bagel-specific parameters for sparse sequence construction
        sequence_length: Optional[int] = None,
        packed_text_indexes: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.Tensor] = None,
        packed_vae_token_indexes: Optional[torch.Tensor] = None,
        mse_loss_indexes: Optional[torch.Tensor] = None,
        vis_gen_target: Optional[torch.Tensor] = None,
        # Parameters for attention mask creation (BlockMask)
        split_lens: Optional[List[int]] = None,
        attn_modes: Optional[List[str]] = None,
        nested_attention_masks: Optional[List[torch.Tensor]] = None,
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
            sample_lens: List of sample lengths for packed sequences (Bagel-style)
            packed_position_ids: Packed position IDs for packed sequences
            ce_loss_indexes: Boolean tensor indicating where to compute CE loss
            packed_label_ids: Packed label IDs for loss computation
            sequence_length: Total length of the full packed sequence (Bagel-style)
            packed_text_indexes: Indices where text tokens go in the full sequence
            packed_vit_token_indexes: Indices where vision tokens go in the full sequence
            split_lens: List of split lengths for attention mask creation
            attn_modes: List of attention modes for each split

        Returns:
            tuple: Tuple containing model outputs and loss mask
        """        # Check if we're using Bagel-style sparse sequence construction
        use_bagel_style = (sequence_length is not None and packed_text_indexes is not None)

        # 1. Process each modality to get embeddings
        modality_embeddings = {}
        vision_embeddings = None

        for modality_name, submodule in self.modality_submodules.items():
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
                    if modality_name == "images":
                        vision_embeddings = embeddings
                    if modality_name == "diffusion":
                        visual_latents = embeddings
                    logger.debug(
                        f"Generated embeddings for {modality_name} with shape {embeddings.shape}"
                    )

        if use_bagel_style:
            # Bagel-style: Pass input_ids directly to language model
            # Let the language model handle embedding computation and sparse sequence construction
            # This avoids computing embeddings twice
            logger.debug(f"Using Bagel-style with input_ids shape {input_ids.shape}")

            # Forward pass through language model with Bagel-style interface
            # Language model (BagelMCoreModel or BagelLLMHuggingFaceModel) will:
            # 1. Compute text embeddings from input_ids
            # 2. Construct full packed_sequence with text + vision embeddings
            # 3. Run transformer and compute loss
            lm_output = self.language_model(
                input_ids=input_ids,
                position_ids=position_ids,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_ids=packed_position_ids,
                ce_loss_indexes=ce_loss_indexes,
                packed_label_ids=packed_label_ids,
                # Bagel-specific for sparse sequence construction
                sequence_length=sequence_length,
                packed_text_indexes=packed_text_indexes,
                packed_vit_token_indexes=packed_vit_token_indexes,
                packed_vae_token_indexes=packed_vae_token_indexes,
                vision_embeddings=vision_embeddings,
                visual_latents=visual_latents,
                # For BlockMask creation
                split_lens=split_lens,
                attn_modes=attn_modes,
                    nested_attention_masks=nested_attention_masks,
            )

            # Step 11: Compute MSE loss at specific indexes
            mse = None
            if mse_loss_indexes is not None and vis_gen_target is not None:
                hidden_state = lm_output['last_hidden_state']
                mse_loss_indexes = mse_loss_indexes.to(hidden_state.device)
                vis_gen_target = vis_gen_target.to(hidden_state.device)
                gen_hidden_state = hidden_state[mse_loss_indexes]
                noise_pred = self.modality_submodules['diffusion'].llm2vae(gen_hidden_state)
                # vis_gen_target = (noise - clean_latent)[shifted_timesteps>0]
                # mse = torch.nn.functional.mse_loss(noise_pred, vis_gen_target, reduction="none")
                mse = (noise_pred - vis_gen_target) ** 2
            lm_output['mse'] = mse
        else:

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
            # Check if language model supports Bagel-style interface
            if hasattr(self.language_model, 'model') and sample_lens is not None:
                # Bagel-style interface (HuggingFace models like BagelLLMHuggingFaceModel)
                lm_output = self.language_model(
                    decoder_input=combined_embeddings,
                    sample_lens=sample_lens,
                    attention_mask=attention_mask,
                    packed_position_ids=packed_position_ids,
                    ce_loss_indexes=ce_loss_indexes,
                    packed_label_ids=packed_label_ids,
                    split_lens=split_lens,
                    attn_modes=attn_modes,
                    nested_attention_masks=nested_attention_masks,
                )
            else:
                # Standard Megatron interface
                lm_output = self.language_model(
                    input_ids=None,
                    position_ids=position_ids,
                    decoder_input=combined_embeddings,
                    labels=labels,
                    attention_mask=attention_mask,
                )
        logger.debug(f"Language model output: {type(lm_output)}")

        return lm_output, loss_mask
