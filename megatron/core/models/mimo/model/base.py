# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
import warnings
from typing import Any, Dict, Optional

import torch  # type: ignore[import-not-found]

from megatron.core.models.mimo.config import MimoModelConfig
from megatron.core.models.mimo.partition.utils import PartitionAdapter, PartitionConfig
from megatron.core.packed_seq_params import PackedSeqParams
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

    def __init__(self, mimo_config: MimoModelConfig, cp_group=None, tp_group=None) -> None:
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

        # Extract language model config for partition adapter
        language_config = mimo_config.language_model_spec.params['config']
        assert (
            language_config.pipeline_model_parallel_size == 1
        ), "Pipeline parallelism is not supported in MimoModel"
        max_seq_len = mimo_config.language_model_spec.params.get('max_sequence_length', 4096)

        self.partition_adapter: Optional[PartitionAdapter] = None
        # Create partition adapter only if parallelism is enabled
        if language_config.context_parallel_size > 1 or language_config.sequence_parallel:
            partition_config = PartitionConfig.from_mp_config(
                mp=language_config,
                max_seq_len=max_seq_len,
                kv_format=mimo_config.kv_format,
                cp_group=cp_group,
                tp_group=tp_group,
            )
            self.partition_adapter = PartitionAdapter(partition_config)

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
                For all modalities: tensor of shape (N, H).
                Shape: (num_tokens_for_modality, hidden_dim)
            input_ids: Input token IDs. Shape: (B, S) or (S,)
                Contains special tokens that mark where each modality's embeddings should go.
                The number of special tokens for each modality should exactly match the number
                of embeddings for that modality.
            special_token_ids: Dictionary mapping modality names to their special token IDs

        Returns:
            Combined embeddings tensor. Shape: (S, B, H) or (S, H)
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

        batch_size, seq_length = input_ids.size()  # input_ids is [B, S]
        logger.debug(
            f"Combined output tensor will have shape: [{seq_length}, {batch_size}, {hidden_dim}]"
        )

        combined_embeddings = torch.zeros(
            (batch_size, seq_length, hidden_dim), dtype=dtype, device=device
        )

        # Process each modality in modality_embeddings
        for modality_name, modality_emb in modality_embeddings.items():
            if modality_name == "text":
                mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)
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

            expanded_mask = mask.unsqueeze(-1).expand_as(combined_embeddings)
            combined_embeddings.masked_scatter_(expanded_mask, modality_emb.flatten())

        return combined_embeddings.transpose(0, 1).contiguous()  # [S, B, H]

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

    def get_text_embeddings(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, special_token_ids: Dict[str, int]
    ) -> torch.Tensor:
        """Get embeddings for text tokens in the input.
        Args:
            input_ids: Input token IDs. Shape: (B, S)
                Contains text tokens and potentially special tokens for other modalities.
            position_ids: Position IDs corresponding to input tokens, used for positional encoding.
                Shape: (B, S)
            special_token_ids: Dictionary mapping modality names to their special token IDs.
                Used to identify non-text tokens in the input_ids.

        Returns:
            torch.Tensor: Embeddings for text tokens.
            Shape: (N, H), where N is the number of text tokens.
        """
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
        packing_kwargs: Optional[dict] = None,
    ):
        """Forward pass through the multimodal model.

        Args:
            input_ids: Input token IDs. Shape: (B, S)
            position_ids: Position IDs. Shape: (B, S)
            attention_mask: Attention mask. Shape: (B, S)
            loss_mask: Loss mask. Shape: (B, S)
            labels: Labels for training. Shape: (B, S)
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
            packing_kwargs: Optional dictionary of kwargs to construct PackedSeqParams
                            if packed_seq_params is not provided. For example:
                                {
                                    "cu_seqlens_q": cu_seqlens,
                                    "cu_seqlens_kv": cu_seqlens,
                                    "cu_seqlens_q_padded": cu_seqlens_padded,
                                    "cu_seqlens_kv_padded": cu_seqlens_padded,
                                    "max_seqlen_q": torch.tensor(
                                        max(seqlens_padded), dtype=torch.int32
                                    ),
                                    "max_seqlen_kv": torch.tensor(
                                        max(seqlens_padded), dtype=torch.int32
                                    ),
                                }

        Returns:
            tuple: Tuple containing model outputs and loss mask
                - lm_output: Model output. Shape: (B, S, ...) or (B, S, V)
                - loss_mask: Loss mask. Shape: (B, S)
        """
        # If packing_kwargs is provided, construct PackedSeqParams
        packed_seq_params = None
        if packing_kwargs is not None:
            # Ensure correct dtype for seqlens tensors
            for key in packing_kwargs:
                if 'cu_seqlens' in key and packing_kwargs[key] is not None:
                    packing_kwargs[key] = packing_kwargs[key].to(dtype=torch.int32)
            packed_seq_params = PackedSeqParams(**packing_kwargs)
            packed_seq_params.qkv_format = 'thd'
            logger.debug(f"Packed sequence parameters: {packed_seq_params}")

        # 1. Process each modality to get embeddings
        modality_embeddings = {}

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

        # 3. If sharding is needed, apply PartitionAdapter.
        # combined_embeddings is [S, B, H]; transpose to [B, S, H] for shard() which expects
        # batch-first layout (required by get_batch_on_this_cp_rank). After CP sharding each
        # rank holds [B, S/cp, H]; transpose back to [S/cp, B, H] for the language model.
        if self.partition_adapter is not None:
            combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()  # [B, S, H]
            combined_embeddings, labels, loss_mask, _, packed_seq_params = (
                self.partition_adapter.shard(
                    embeddings=combined_embeddings,
                    labels=labels,
                    loss_mask=loss_mask,
                    attention_mask=attention_mask,
                    packed_seq_params=packed_seq_params,
                )
            )
            # shard() returns embeddings in [B, S/cp, H]; transpose to [S/cp, B, H]
            # which is what the language model expects.
            if combined_embeddings is not None:
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

        # 5. Forward pass through language model
        lm_output = self.language_model(
            input_ids=None,
            position_ids=None,
            decoder_input=combined_embeddings,
            labels=labels,
            attention_mask=None,
            packed_seq_params=packed_seq_params,
        )

        logger.debug(f"Language model output shape: {lm_output.shape}")

        return lm_output, loss_mask
