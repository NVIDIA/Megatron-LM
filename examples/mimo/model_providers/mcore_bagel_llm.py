# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""
Megatron Core based Bagel Language Model.

This module provides a wrapper around GPTModel that handles Bagel-specific
input/output formats while leveraging Megatron Core's efficient implementation.
"""

from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_mot_block import (
    MoTTransformerLayerSubmodules,
    TransformerMoTBlock,
    TransformerMoTBlockSubmodules,
    get_mot_layer_spec,
)
from megatron.core.transformer.spec_utils import ModuleSpec


class BagelMCoreModel(GPTModel):
    """
    Bagel Language Model based on Megatron Core GPTModel.

    This model wraps GPTModel to handle Bagel-specific training scenarios:
    - Packed sequences with variable sample lengths
    - Sparse sequence construction (text + vision at specific indexes)
    - BlockMask attention for flex attention
    - Cross-entropy loss computation at specific indexes
    """

    def __init__(self, *args, llm_config=None, use_flex_attention=False, **kwargs):
        """Initialize BagelMCoreModel.

        Args:
            llm_config: Bagel LLM config (Qwen2Config) for additional settings
            *args, **kwargs: Arguments passed to GPTModel
        """
        super().__init__(*args, **kwargs)
        self.share_embeddings_and_output_weights = True

        self.llm_config = llm_config
        self.num_heads = kwargs.get('config').num_attention_heads if 'config' in kwargs else 28

        # Check if model uses MoE (Mixture of Experts)
        self.use_mo = False
        if llm_config is not None and hasattr(llm_config, 'layer_module'):
            self.use_mo = "Mo" in llm_config.layer_module

        # If using MoT, replace decoder with TransformerMoTBlock
        if self.use_mo:
            self._setup_mot_decoder(use_flex_attention=use_flex_attention, **kwargs)

    def _setup_mot_decoder(self, use_flex_attention=False, **kwargs):
        """Setup TransformerMoTBlock as decoder for MoT mode.

        This replaces the standard TransformerBlock with TransformerMoTBlock
        which supports separate processing paths for understanding (und) and
        generation (gen) tokens.
        """
        config = kwargs.get('config')
        transformer_layer_spec = kwargs.get('transformer_layer_spec')

        if config is None:
            return

        # Get MoT-specific settings from llm_config
        freeze_und = getattr(self.llm_config, 'freeze_und', False) if self.llm_config else False

        # Build MoT layer spec if not already MoT spec
        if transformer_layer_spec is not None:
            # Check if already a MoT spec
            if isinstance(transformer_layer_spec, TransformerMoTBlockSubmodules):
                mot_spec = transformer_layer_spec
            elif hasattr(transformer_layer_spec, 'submodules') and isinstance(
                transformer_layer_spec.submodules, (TransformerMoTBlockSubmodules, MoTTransformerLayerSubmodules)
            ):
                mot_spec = transformer_layer_spec
            else:
                # Convert standard spec to MoT spec using helper function
                mot_spec = get_mot_layer_spec(use_flex_attention=use_flex_attention, qk_layernorm=True)
        else:
            mot_spec = get_mot_layer_spec(use_flex_attention=use_flex_attention, qk_layernorm=True)

        # Replace decoder with TransformerMoTBlock
        self.decoder = TransformerMoTBlock(
            config=config,
            spec=mot_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            pg_collection=self.pg_collection,
            vp_stage=self.vp_stage,
            freeze_und=freeze_und,
        )

        print(f"[BagelMCoreModel] Initialized with TransformerMoTBlock (freeze_und={freeze_und})")

    def get_word_embeddings(self, input_ids: Tensor, position_ids: Optional[Tensor] = None) -> Tensor:
        """Get embeddings for input tokens.

        This method provides a consistent interface for getting embeddings,
        used by MimoModel to get text embeddings.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len] (optional)

        Returns:
            embeddings: [seq_len, batch_size, hidden_dim]
        """
        # Use GPTModel's embedding module (LanguageModelEmbedding)
        # LanguageModelEmbedding returns [seq_len, batch, hidden]
        return self.embedding(input_ids=input_ids, position_ids=position_ids)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        decoder_input: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        loss_mask: Optional[Tensor] = None,
        # Bagel-specific parameters (matching MimoModel forward signature)
        sample_lens: Optional[List[int]] = None,
        packed_position_ids: Optional[Tensor] = None,
        ce_loss_indexes: Optional[Tensor] = None,
        packed_label_ids: Optional[Tensor] = None,
        sequence_length: Optional[int] = None,
        packed_text_indexes: Optional[Tensor] = None,
        packed_vit_token_indexes: Optional[Tensor] = None,
        packed_vae_token_indexes: Optional[Tensor] = None,
        vision_embeddings: Optional[Tensor] = None,
        visual_latents: Optional[Tensor] = None,
        split_lens: Optional[List[int]] = None,
        attn_modes: Optional[List[str]] = None,
        nested_attention_masks: Optional[List[Tensor]] = None,
        # # MoT-specific parameters (can be auto-derived if not provided)
        # packed_und_token_indexes: Optional[Tensor] = None,
        # packed_gen_token_indexes: Optional[Tensor] = None,
        **kwargs,
    ) -> dict:
        """Forward pass with Bagel-specific handling.

        For Bagel-style, we receive input_ids (text tokens only) and construct
        the full packed sequence internally.

        Args:
            input_ids: Text token IDs [batch, num_text_tokens] (for Bagel-style)
            position_ids: Position IDs for text tokens [batch, num_text_tokens]
            attention_mask: Attention mask
            decoder_input: Pre-computed embeddings (not used in Bagel-style)
            labels: Labels for loss computation
            loss_mask: Mask for loss computation
            sample_lens: List of sample lengths in packed sequence
            packed_position_ids: Position IDs for the full packed sequence
            ce_loss_indexes: Indexes where to compute CE loss
            packed_label_ids: Label IDs at ce_loss_indexes
            sequence_length: Total length of packed sequence
            packed_text_indexes: Where text tokens go in full sequence
            packed_vit_token_indexes: Where vision tokens (ViT) go in full sequence
            packed_vae_token_indexes: Where VAE tokens go in full sequence (for generation)
            vision_embeddings: Vision embeddings from encoder
            split_lens: Split lengths for attention mask
            attn_modes: Attention modes for each split

        Returns:
            dict: Dictionary with 'last_hidden_state' and 'ce' (cross-entropy loss)
        """
        # Check if we're using Bagel-style (packed sequence with sparse construction)
        use_bagel_style = (sequence_length is not None and packed_text_indexes is not None)

        if use_bagel_style:
            # Bagel-style: construct full packed sequence from text input_ids + vision embeddings
            device = input_ids.device if input_ids is not None else vision_embeddings.device

            # Step 1: Get text embeddings using GPTModel's embedding layer
            # self.embedding is inherited from GPTModel (LanguageModelEmbedding)
            print("language_model.model.embed_tokens forward")
            print("packed_text_ids", input_ids.shape, input_ids.sum())
            print("sequence_length", sequence_length)
            text_embeddings = self.embedding(
                input_ids=input_ids,
                position_ids=position_ids
            )  # [seq_len, batch, hidden] or [batch, seq_len, hidden]
            print("after language model.model.embed_tokens forward", text_embeddings.shape, text_embeddings.to(torch.float32).sum())

            # Ensure we have [num_text_tokens, hidden] format
            if text_embeddings.dim() == 3:
                if text_embeddings.shape[1] == 1:  # [seq, 1, hidden]
                    text_embeddings = text_embeddings.squeeze(1)
                elif text_embeddings.shape[0] == 1:  # [1, seq, hidden]
                    text_embeddings = text_embeddings.squeeze(0)
                else:
                    text_embeddings = text_embeddings[0]  # Take first batch

            hidden_size = text_embeddings.shape[-1]
            dtype = text_embeddings.dtype

            # Step 2: Create full packed_sequence and place embeddings at their indexes
            packed_sequence = torch.zeros(
                (sequence_length, hidden_size),
                dtype=dtype,
                device=device
            )

            # Place text embeddings at their indexes
            packed_text_indexes = packed_text_indexes.to(device)
            packed_sequence[packed_text_indexes] = text_embeddings

            # Place vision embeddings at their indexes if present
            if vision_embeddings is not None and packed_vit_token_indexes is not None:
                packed_vit_token_indexes = packed_vit_token_indexes.to(device)
                packed_sequence[packed_vit_token_indexes] = vision_embeddings

            # Place visual latents at their indexes if present
            if visual_latents is not None and packed_vae_token_indexes is not None:
                packed_vae_token_indexes = packed_vae_token_indexes.to(device)
                packed_sequence[packed_vae_token_indexes] = visual_latents

            # Step 3: Prepare decoder_input for GPTModel
            # GPTModel expects [seq_len, batch, hidden]
            decoder_input_for_gpt = packed_sequence.unsqueeze(1)

            # Step 4: Create attention mask (BlockMask) if needed
            if nested_attention_masks is not None:
                # Use nested_attention_masks directly if provided
                attention_mask = nested_attention_masks
            elif split_lens is not None and attn_modes is not None:
                from bagel.data.data_utils import create_sparse_mask
                from torch.nn.attention.flex_attention import create_block_mask
                sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, device)
                seqlen = sum(sample_lens)
                attention_mask = create_block_mask(
                    sparse_mask,
                    B=1, H=self.num_heads,
                    Q_LEN=seqlen,
                    KV_LEN=seqlen,
                    device=device,
                    BLOCK_SIZE=128,
                    _compile=True
                )

            # Step 5: Prepare position_ids for GPTModel
            # Use packed_position_ids for the full sequence
            if packed_position_ids is not None:
                gpt_position_ids = packed_position_ids.unsqueeze(0)  # [1, seq_len]
            else:
                gpt_position_ids = torch.arange(sequence_length, device=device).unsqueeze(0)

            # Step 6: Prepare token indexes
            mo_kwargs = {}
            if self.use_mo:
                # packed_und_token_indexes: understanding tokens (text + vit)
                packed_und_token_indexes = packed_text_indexes
                if packed_vit_token_indexes is not None:
                    packed_und_token_indexes = torch.cat([packed_text_indexes, packed_vit_token_indexes.to(device)], dim=0)
                mo_kwargs['packed_und_token_indexes'] = packed_und_token_indexes

                # Build packed_gen_token_indexes: generation tokens (vae)
                if packed_vae_token_indexes is not None:
                    mo_kwargs['packed_gen_token_indexes'] = packed_vae_token_indexes.to(device)
                else:
                    mo_kwargs['packed_gen_token_indexes'] = torch.tensor([], dtype=torch.long, device=device)

                # print("BagelMCoreModel packed_und_token_indexes:", mo_kwargs['packed_und_token_indexes'].shape)
                # if len(mo_kwargs['packed_gen_token_indexes']) > 0:
                #     print("BagelMCoreModel packed_gen_token_indexes:", mo_kwargs['packed_gen_token_indexes'].shape)
                # else:
                #     print("BagelMCoreModel packed_gen_token_indexes is empty")

            # print("language_model forward")
            # print("packed_sequence", packed_sequence.shape, packed_sequence.to(torch.float32).sum())
            # print("sample_lens", sample_lens)
            # print("attention_mask", attention_mask.shape)
            # print("packed_position_ids", packed_position_ids.shape, packed_position_ids.sum())
            # print("packed_text_indexes", packed_text_indexes.shape, packed_text_indexes.sum())

            # Step 8: Call decoder directly (bypass parent forward to have control)
            hidden_states = self._forward_decoder(
                decoder_input=decoder_input_for_gpt,
                position_ids=gpt_position_ids,
                attention_mask=attention_mask,
                packed_position_ids=packed_position_ids,
                sample_lens=sample_lens,
                **mo_kwargs,
            )


            # Step 9: Process output hidden states
            if isinstance(hidden_states, tuple):
                # MoT decoder returns (hidden_states, context)
                hidden_states = hidden_states[0]

            if hidden_states.dim() == 3:
                if hidden_states.shape[0] == 1:
                    last_hidden_state = hidden_states.squeeze(0)  # [seq_len, hidden]
                elif hidden_states.shape[1] == 1:
                    last_hidden_state = hidden_states.squeeze(1)  # [seq_len, hidden]
                else:
                    last_hidden_state = hidden_states[0]
            else:
                last_hidden_state = hidden_states
            # print("after language_model forward, last_hidden_state", last_hidden_state.shape, last_hidden_state.to(torch.float32).sum())
            # print("================================================")

            # Step 10: Compute CE loss at specific indexes
            ce = None
            if ce_loss_indexes is not None and packed_label_ids is not None:
                ce_loss_indexes = ce_loss_indexes.to(device)
                packed_label_ids = packed_label_ids.to(device)

                # Apply output_layer to get logits
                # Shape: [seq_len, hidden_size] -> [num_loss_tokens, vocab_size]
                output_weight = None
                if self.share_embeddings_and_output_weights:
                    output_weight = self.shared_embedding_or_output_weight()
                # Use shared embedding weight as output projection
                # print("before language_model.lm_head last_hidden_state", last_hidden_state[ce_loss_indexes].shape, last_hidden_state[ce_loss_indexes].to(torch.float32).sum(), last_hidden_state[ce_loss_indexes].flatten()[:10])
                # print("before language_model.lm_head weight", output_weight.shape, output_weight.dtype, output_weight.sum())
                logits_at_loss = F.linear(
                    last_hidden_state[ce_loss_indexes],
                    output_weight
                )
                # print("after language_model.lm_head forward, logits_at_loss", logits_at_loss.shape, logits_at_loss.sum(), logits_at_loss.flatten()[:10])
                # print("before language_model.lm_head cross_entropy, packed_label_ids", packed_label_ids.shape, packed_label_ids.sum(), packed_label_ids.flatten()[:10])
                ce = F.cross_entropy(logits_at_loss, packed_label_ids, reduction="none")
                # print("after language_model.lm_head cross_entropy", ce.shape, ce.sum(), ce)

            return dict(last_hidden_state=last_hidden_state, ce=ce)

        else:
            # Standard GPTModel forward for non-Bagel cases
            output = super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=decoder_input,
                labels=labels,
                loss_mask=loss_mask,
            )

            # Wrap output in dict format for consistency
            if isinstance(output, dict):
                return output
            else:
                return dict(last_hidden_state=output, ce=None)

    def _forward_decoder(
        self,
        decoder_input: Tensor,
        position_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        packed_position_ids: Optional[Tensor] = None,
        sample_lens: Optional[List[int]] = None,
        packed_und_token_indexes: Optional[Tensor] = None,
        packed_gen_token_indexes: Optional[Tensor] = None,
        packed_seq_params=None,
        inference_context=None,
        **kwargs,
    ) -> Union[Tensor, tuple]:
        """Forward pass through decoder with proper input alignment.

        This method handles the differences between standard TransformerBlock
        and TransformerMoTBlock interfaces.

        Args:
            decoder_input: Input embeddings [seq_len, batch, hidden]
            position_ids: Position IDs [batch, seq_len]
            attention_mask: Attention mask
            packed_position_ids: Packed position IDs for rotary embeddings
            sample_lens: Sample lengths for packed sequence
            packed_und_token_indexes: MoT - indexes of understanding tokens
            packed_gen_token_indexes: MoT - indexes of generation tokens
            packed_seq_params: Packed sequence parameters
            inference_context: Inference context for KV cache

        Returns:
            Hidden states from decoder (Tensor or tuple)
        """
        # Compute rotary embeddings if needed (following GPTModel._preprocess pattern)
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None

        if self.position_embedding_type == 'rope' and hasattr(self, 'rotary_pos_emb') and self.rotary_pos_emb is not None:
            # Use forward_with_position_ids to use position_ids instead of rotary_seq_len to properly compute the sequence length for rotary embeddings
            rotary_pos_emb = self.rotary_pos_emb.forward_with_position_ids(packed_position_ids)
        elif self.position_embedding_type == 'yarn' and hasattr(self, 'rotary_pos_emb') and self.rotary_pos_emb is not None:
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_context, self.decoder, decoder_input, self.config, packed_seq_params
            )
            rotary_pos_emb, _ = self.rotary_pos_emb(rotary_seq_len)

        # Call decoder based on type
        if self.use_mo:
            # TransformerMoTBlock interface
            hidden_states = self.decoder(
                hidden_states=decoder_input,
                attention_mask=attention_mask,
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
            )
        else:
            # Standard TransformerBlock interface - do NOT pass MoT-specific or unexpected kwargs
            hidden_states = self.decoder(
                hidden_states=decoder_input,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
            )

        return hidden_states
