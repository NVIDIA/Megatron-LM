# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
from typing import Optional
import torch

from bagel.modeling.bagel import SiglipVisionModel
from bagel.modeling.bagel.modeling_utils import PositionEmbedding
import numpy as np


class HFBagelVisionEncoderWrapper(torch.nn.Module):
    """
    ============================================================================
    Bagel Vision Architecture
    ============================================================================
    Bagel's vision pipeline includes:
    1. SigLIP VIT encoder (1152-dim output)
    2. MLP connector (1152 → 3584)
    3. 2D position embeddings (added after connector)
    """

    def __init__(self, feature_layer_index=-2, bagel_config=None, vit_path=None,
                 dtype=torch.float32, recompute_vit=False):
        """Initialize the HFBagelVisionEncoderWrapper.

        Args:
            feature_layer_index (int): Index of the feature layer to extract from the encoder's hidden states.
                                       Default is -2 (second to last layer).
            model_path (str): Path to the Bagel model.
            vit_path (str): Path to the SigLIP VIT model.
            recompute_vit (bool): Wrap each SigLIP encoder layer's forward with
                ``torch.utils.checkpoint``. Only layer-boundary activations are
                retained; intermediate activations are recomputed during backward.
        """
        super().__init__()
        llm_config = bagel_config.llm_config
        vit_config = bagel_config.vit_config
        self.dtype = dtype
        if vit_path is None:
            self.encoder = SiglipVisionModel(vit_config)
        else:
            self.encoder = SiglipVisionModel.from_pretrained(vit_path, config=vit_config)
        print("=== vit_model ===")
        for name, param in self.encoder.named_parameters():
            print(f"{name}: {param.shape}, {param.sum()}")
        print("================================================")
        self.encoder.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
        self.encoder.to(dtype)

        add_postion_embedding = True
        self.add_postion_embedding = add_postion_embedding
        if self.add_postion_embedding:
            self.vit_pos_embed = PositionEmbedding(bagel_config.vit_max_num_patch_per_side, llm_config.hidden_size)
            self.vit_pos_embed.to(dtype)

        self.recompute_vit = recompute_vit
        if self.recompute_vit:
            self._enable_vit_activation_checkpointing()

    def _enable_vit_activation_checkpointing(self):
        """Wrap the entire ``SiglipEncoder.forward`` (all transformer layers)
        as a single ``torch.utils.checkpoint`` block — "full-ViT" recompute.

        During forward: only the encoder's input tensor is retained; all
        per-layer boundary activations and internal activations are freed.
        During backward: the whole SigLIP encoder is re-executed once to
        regenerate the full activation graph, then gradients flow back.

        Trade-off vs. per-layer checkpointing:
          • Forward peak:  full-encoder is ~N× smaller (only input retained,
            instead of N layer-boundary tensors).
          • Backward peak: full-encoder is larger — briefly holds ALL N
            layers' intermediates simultaneously during recompute. This is
            acceptable here because the ViT backward happens AFTER the LLM
            backward has already freed most of its activations.

        Since rank 24's OOM is driven by iter-2 FORWARD peak (not backward),
        full-encoder recompute is the more effective lever.
        """
        from torch.utils.checkpoint import checkpoint as _checkpoint

        try:
            siglip_encoder = self.encoder.vision_model.encoder
        except AttributeError as e:
            raise RuntimeError(
                "Cannot locate SigLIP encoder at "
                "self.encoder.vision_model.encoder — has the SigLIP model "
                "structure changed?"
            ) from e

        if getattr(siglip_encoder, '_ckpt_wrapped', False):
            return

        orig_forward = siglip_encoder.forward

        def _ckpt_encoder_forward(*args, **kwargs):
            # Skip checkpointing during eval / inference (no grad needed).
            if torch.is_grad_enabled():
                return _checkpoint(
                    orig_forward, *args, use_reentrant=False, **kwargs
                )
            return orig_forward(*args, **kwargs)

        siglip_encoder._ckpt_original_forward = orig_forward
        siglip_encoder.forward = _ckpt_encoder_forward
        siglip_encoder._ckpt_wrapped = True

        n_layers = len(siglip_encoder.layers)
        print(
            f"[HFBagelVisionEncoder] Full-ViT activation checkpointing enabled "
            f"— wrapping all {n_layers} SigLIP encoder layers as a single "
            f"checkpoint block"
        )


    def forward(
        self,
        packed_vit_tokens,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
    ):
        """Input: packed_vit_tokens of shape (num_patches, patch_dim)."""

        # Get the dtype of the encoder weights for consistent computation
        encoder_dtype = next(self.encoder.parameters()).dtype

        # Convert input to the same dtype as the encoder
        packed_vit_tokens = packed_vit_tokens.to(dtype=encoder_dtype)

        # Process through encoder and extract last_hidden_state
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()
        # if cu_seqlens is not None:
        #     print("cu_seqlens", cu_seqlens.shape, cu_seqlens.sum())
        # if max_seqlen is not None:
        #     print("max_seqlen", max_seqlen)

        packed_vit_token_embed = self.encoder(
            packed_pixel_values=packed_vit_tokens,
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        # print("after vit model forward, packed_vit_token_embed", packed_vit_token_embed.shape, packed_vit_token_embed.sum())
        if self.add_postion_embedding:
            vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            # print("after vit pos embed forward, vit_token_pos_emb", vit_token_pos_emb.shape, vit_token_pos_emb.sum())
        # print("================================================")
        return [packed_vit_token_embed, vit_token_pos_emb] if self.add_postion_embedding else packed_vit_token_embed
