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
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from megatron.core.models.bagel.transformer_mot_block import (
    MoTTransformerLayerSubmodules,
    TransformerMoTBlock,
    TransformerMoTBlockSubmodules,
    get_mot_layer_spec,
)

from megatron.core.models.bagel.bagel_rope import BagelRotaryEmbedding
from megatron.core.models.bagel.mot_packed_seq_params import MoTPackedSeqParams


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
        # kwargs['share_embeddings_and_output_weights'] = True
        super().__init__(*args, **kwargs)
        # self.share_embeddings_and_output_weights = True

        self.llm_config = llm_config
        self.num_heads = kwargs.get('config').num_attention_heads if 'config' in kwargs else 28

        

        #use bagel rope
        self.rotary_pos_emb = BagelRotaryEmbedding(
            kv_channels=self.config.kv_channels,
            rotary_percent=self.rotary_percent,
            rotary_interleaved=self.config.rotary_interleaved,
            seq_len_interpolation_factor=kwargs.get('seq_len_interpolation_factor', None),
            rotary_base=self.rotary_base,
            rope_scaling=self.rotary_scaling,
            rope_scaling_factor=kwargs.get('rope_scaling_factor', 8.0),
            use_cpu_initialization=self.config.use_cpu_initialization,
            cp_group=self.pg_collection.cp,
        )


        # Check if model uses MoT (Mixture of Transformers)
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
        decoder_input: Tensor,
        attention_mask=None,
        labels: Optional[Tensor] = None,
        loss_mask: Optional[Tensor] = None,
        packed_position_ids: Optional[Tensor] = None,
        sample_lens: Optional[List[int]] = None,
        split_lens: Optional[List[int]] = None,
        attn_modes: Optional[List[str]] = None,
        packed_seq_params: Optional[MoTPackedSeqParams] = None,
        **kwargs,
    ) -> dict:
        """Forward pass for BagelMCoreModel.

        Receives pre-assembled, pre-sharded inputs from the caller
        (BagelMimoModel, which calls align_bagel_embeddings before this).
        No embedding assembly, no CP sharding logic here.

        Args:
            decoder_input: Compact embeddings [Lund+Lgen, 1, H], pre-assembled
                by align_bagel_embeddings().
            attention_mask: Pre-built BlockMask or None.
            labels: Per-rank labels [actual_lund] pre-sliced by align_bagel_embeddings
                (0 at non-CE positions), or None.
            loss_mask: Per-rank loss mask [actual_lund] pre-sliced by align_bagel_embeddings
                (0.0 at non-CE), or None.
            packed_position_ids: Compact position IDs [Lund+Lgen] pre-built by
                align_bagel_embeddings(), or None.
            sample_lens: Sample lengths for BlockMask creation.
            split_lens: Split lengths for BlockMask creation.
            attn_modes: Attention modes for BlockMask creation.
            packed_seq_params: MoTPackedSeqParams (from BagelMimoModel.get_packed_seq_params).

        Returns:
            dict: {last_hidden_state, ce, packed_seq_params}
        """
        # Non-first PP stage: decoder_input is None — the hidden activation
        # arrives via set_input_tensor inside the inherited GPTModel layers.
        # Use the language model's parameter device as a fallback.
        if decoder_input is not None:
            device = decoder_input.device
        else:
            device = next(self.parameters()).device
        decoder_input_for_gpt = decoder_input

        # Step 4: Build FlexAttention BlockMask from split_lens/attn_modes if provided.
        # Callers may also pass a pre-built BlockMask via attention_mask directly.
        if split_lens is not None and attn_modes is not None:
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
        # When compact packed_position_ids [Lund+Lgen] is provided, use it directly.
        # Fallback: ascending range over decoder_input length.
        if packed_position_ids is not None:
            gpt_position_ids = packed_position_ids.unsqueeze(0)   # [1, Lund+Lgen] or [1, S]
        else:
            seq_len = decoder_input_for_gpt.shape[0]
            gpt_position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # Step 6: Use pre-built MoTPackedSeqParams from BagelMimoModel (required).
        packed_seq_params_for_decoder = None
        if self.use_mo:
            assert packed_seq_params is not None, \
                "packed_seq_params must be provided by the caller (BagelMimoModel.get_packed_seq_params)"
            packed_seq_params_for_decoder = packed_seq_params

        # Step 7: Call decoder directly (bypass parent forward to have control)
        hidden_states = self._forward_decoder(
            decoder_input=decoder_input_for_gpt,
            position_ids=gpt_position_ids,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            sample_lens=sample_lens,
            packed_seq_params=packed_seq_params_for_decoder,
        )


        # Step 8: Process output hidden states
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

        # Step 9: Compute CE loss — last PP stage only.
        # On non-last stages (post_process=False) the LM head and CE don't exist
        # on this rank; return the raw hidden state so Megatron's pipeline schedule
        # can send it to the next stage as the activation tensor.
        # PP requires a 3-D ``[seq, batch=1, hidden]`` tensor — Megatron's
        # _communicate_shapes hardcodes a 3-element shape buffer
        # (p2p_communication.py:194). Returning the 2-D ``last_hidden_state``
        # would cause the irecv buffer (3,) to mismatch the isend tensor (2,)
        # and deadlock forever. Use the unsqueezed form for the activation hop.
        if not self.post_process:
            if last_hidden_state.dim() == 2:
                return last_hidden_state.unsqueeze(1)  # [seq, 1, hidden]
            return last_hidden_state

        # For MoT (compact hidden states): filter ce_loss_indexes to this rank's und tokens
        # and index directly into compact last_hidden_state — no scatter-back to [S, H] needed.
        # For non-MoT: ce_loss_indexes index into global last_hidden_state as before.
        # Always compute CE so output_layer participates in backward (FSDP correctness).
        # labels/loss_mask are always tensors from the dataloader; zero loss_mask means
        # CE contributes 0 to loss but output_layer.weight is still in the autograd graph.
        actual_Lund = len(packed_seq_params.local_und_token_indexes)
        mask = loss_mask > 0
        logits_hidden_states = last_hidden_state[:actual_Lund][mask]
        output_weight = self.output_layer.weight
        logits = F.linear(logits_hidden_states, output_weight)
        # can not use full logits for ce calc would cause a OOM when Lund is large.
        # output_layer is column-parallel: at TP>1 each rank holds vocab/TP
        # columns, so use vocab_parallel_cross_entropy which all-reduces
        # across the TP group instead of plain F.cross_entropy.
        tp_size = torch.distributed.get_world_size(self.pg_collection.tp)
        if tp_size > 1:
            ce = vocab_parallel_cross_entropy(logits, labels[mask]) * loss_mask[mask]
        else:
            ce = F.cross_entropy(logits, labels[mask], reduction="none") * loss_mask[mask]

        return dict(last_hidden_state=last_hidden_state, ce=ce,
                    packed_seq_params=packed_seq_params_for_decoder)

    def _forward_decoder(
        self,
        decoder_input: Tensor,
        position_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        packed_position_ids: Optional[Tensor] = None,
        sample_lens: Optional[List[int]] = None,
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
            packed_seq_params: MoTPackedSeqParams carrying all MoT metadata and CP shard info.
                Must be pre-built by the caller before invoking this method.
            inference_context: Inference context for KV cache

        Returns:
            Hidden states from decoder (Tensor or tuple)
        """
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None

        if self.use_mo and packed_seq_params is not None:
            # ── MoT compact rearrangement ────────────────────────────────────
            # packed_seq_params carries all index arrays and CP shard info.
            Lund = packed_seq_params.padded_und_seqlen
            Lgen = packed_seq_params.padded_gen_seqlen
            local_und_idx = packed_seq_params.local_und_token_indexes
            local_gen_idx = packed_seq_params.local_gen_token_indexes


            # Compact decoder_input if it arrives as full [S, 1, H] (e.g. direct test calls).
            # When called via forward() or align_embeddings_by_token_positions, it is
            # already compact [Lund+Lgen, 1, H] and this branch is skipped.
            # On non-first PP stages decoder_input is None — the activation
            # arrives via set_input_tensor inside TransformerMoTBlock; skip the
            # shape check there.
            if decoder_input is not None:
                assert decoder_input.shape[0] == Lund + Lgen, f"decoder_input shape mismatch, expected {Lund + Lgen}, got {decoder_input.shape[0]}"

            # Compute RoPE on compact position IDs.
            if (
                self.position_embedding_type == 'rope'
                and hasattr(self, 'rotary_pos_emb')
                and self.rotary_pos_emb is not None
                and packed_position_ids is not None
            ):
                assert len(packed_position_ids) == Lund + Lgen, f"packed_position_ids shape mismatch, expected {Lund + Lgen}, got {packed_position_ids.shape}"
                rotary_pos_emb = self.rotary_pos_emb.forward_with_position_ids(packed_position_ids)

        # Call decoder based on type
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
