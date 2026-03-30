# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
from typing import Optional
import torch

from bagel.modeling.bagel import SiglipVisionModel
from bagel.modeling.bagel.modeling_utils import PositionEmbedding
import numpy as np

_DEBUG = os.environ.get("BAGEL_DEBUG", "0") == "1"


class HFBagelVisionEncoderWrapper(torch.nn.Module):
    """
    ============================================================================
    Bagel Vision Architecture
    ============================================================================
    Bagel's vision pipeline includes:
    1. SigLIP VIT encoder (1152-dim output)
    2. MLP connector (1152 → 3584)
    3. 2D position embeddings (added after connector)

    Note: The SigLIP encoder is stored as a plain attribute (not an nn.Module
    submodule) so that its parameters are invisible to FSDP. This avoids FSDP
    trying to shard HF ViT parameters whose hidden_size (1152) is incompatible
    with the flat-buffer sharding layout.
    """

    def __init__(self, feature_layer_index=-2, bagel_config=None, vit_path=None, dtype=torch.float32):
        """Initialize the HFBagelVisionEncoderWrapper.

        Args:
            feature_layer_index (int): Index of the feature layer to extract from the encoder's hidden states.
                                       Default is -2 (second to last layer).
            model_path (str): Path to the Bagel model.
            vit_path (str): Path to the SigLIP VIT model.
        """
        super().__init__()
        llm_config = bagel_config.llm_config
        vit_config = bagel_config.vit_config
        self.dtype = dtype
        if vit_path is None:
            encoder = SiglipVisionModel(vit_config)
        else:
            encoder = SiglipVisionModel.from_pretrained(vit_path, config=vit_config)
        if _DEBUG:
            print("=== vit_model ===")
            for name, param in encoder.named_parameters():
                print(f"{name}: {param.shape}, {param.sum()}")
            print("================================================")
        encoder.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
        encoder.to(dtype)

        # Store encoder as a plain attribute so its parameters are NOT registered
        # as submodule parameters, keeping them invisible to FSDP.
        object.__setattr__(self, '_encoder', encoder)
        self._encoder_on_device = False

        add_postion_embedding = True
        self.add_postion_embedding = add_postion_embedding
        if self.add_postion_embedding:
            self.vit_pos_embed = PositionEmbedding(bagel_config.vit_max_num_patch_per_side, llm_config.hidden_size)
            self.vit_pos_embed.to(dtype)

    @property
    def encoder(self):
        return self._encoder

    def forward(
        self,
        packed_vit_tokens,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
    ):
        """Input: packed_vit_tokens of shape (num_patches, patch_dim)."""

        # Ensure encoder is on the same device as input (only once).
        if not self._encoder_on_device:
            target_device = packed_vit_tokens.device
            self._encoder.to(target_device)
            self._encoder_on_device = True

        # Convert input to the same dtype as the encoder
        packed_vit_tokens = packed_vit_tokens.to(dtype=self.dtype)

        # Process through encoder (no_grad since ViT is not managed by FSDP optimizer)
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()

        with torch.no_grad():
            packed_vit_token_embed = self._encoder(
                packed_pixel_values=packed_vit_tokens,
                packed_flattened_position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        if self.add_postion_embedding:
            vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
        return [packed_vit_token_embed, vit_token_pos_emb] if self.add_postion_embedding else packed_vit_token_embed
