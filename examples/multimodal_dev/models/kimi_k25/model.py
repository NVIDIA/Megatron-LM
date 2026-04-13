# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Kimi K2.5 Vision-Language model for multimodal_dev training.

This module mirrors the Megatron-Bridge ``KimiK25VLModel`` so that
checkpoint keys are identical:
    language_model.*   — MoE + MLA GPTModel
    vision_tower.*     — MoonViT3d (HF dynamic module)
    mm_projector.*     — PatchMergerMLP (HF dynamic module)

The forward pass follows the Bridge exactly:
  1. Embed text tokens via language_model.embedding()
  2. Extract & project image features via vision_tower + mm_projector
  3. Replace image-placeholder embeddings with projected features
  4. Run the merged embeddings through language_model as decoder_input

Adapted for multimodal_dev's forward_step convention: forward() returns
the output tensor directly (not a tuple).
"""

import importlib
import logging
from typing import List, Optional

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import MLATransformerConfig
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from examples.multimodal_dev.models.kimi_k25.configuration import (
    KIMI_K25_IMAGE_TOKEN_ID,
    KIMI_K25_PAD_TOKEN_ID,
)

logger = logging.getLogger(__name__)


class KimiK25VLModel(MegatronModule):
    """Kimi K2.5 Vision-Language model aligned with Megatron-Bridge.

    Submodule names (``language_model``, ``vision_tower``, ``mm_projector``)
    match the Bridge checkpoint layout so ``dist_checkpointing.load`` works
    without key remapping.

    Args:
        language_model: Pre-built MCore GPTModel (MoE + MLA).
        hf_model_path: HF hub id or local path for MoonViT3d dynamic loading.
        media_placeholder_token_id: Token id for image placeholders.
        pad_token_id: Padding token id.
        ignore_index: Label ignore index for loss masking.
        freeze_vision_model: Freeze vision tower parameters.
        freeze_vision_projection: Freeze mm_projector parameters.
    """

    def __init__(
        self,
        language_model: GPTModel,
        hf_model_path: str,
        media_placeholder_token_id: int = KIMI_K25_IMAGE_TOKEN_ID,
        pad_token_id: int = KIMI_K25_PAD_TOKEN_ID,
        ignore_index: int = -100,
        freeze_vision_model: bool = True,
        freeze_vision_projection: bool = True,
    ):
        super().__init__(config=language_model.config)

        self.media_placeholder_token_id = media_placeholder_token_id
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

        self.language_model = language_model
        self._init_vision_modules(hf_model_path)

        if freeze_vision_model and hasattr(self, "vision_tower"):
            for p in self.vision_tower.parameters():
                p.requires_grad = False
        if freeze_vision_projection and hasattr(self, "mm_projector"):
            for p in self.mm_projector.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # Vision init
    # ------------------------------------------------------------------
    def _init_vision_modules(self, hf_model_path: str):
        """Load MoonViT3d and PatchMergerMLP from HF dynamic modules."""
        MoonViT3dPretrainedModel = get_class_from_dynamic_module(
            "modeling_kimi_k25.MoonViT3dPretrainedModel", hf_model_path,
        )
        PatchMergerMLP = get_class_from_dynamic_module(
            "modeling_kimi_k25.PatchMergerMLP", hf_model_path,
        )
        VisionTowerConfig = get_class_from_dynamic_module(
            "modeling_kimi_k25.VisionTowerConfig", hf_model_path,
        )
        ProjectorConfig = get_class_from_dynamic_module(
            "modeling_kimi_k25.ProjectorConfig", hf_model_path,
        )

        # Patch MoonViT3dEncoder missing attribute
        _vit_module = importlib.import_module(MoonViT3dPretrainedModel.__module__)
        if not getattr(_vit_module.MoonViT3dEncoder, "_patched", False):
            _orig_init = _vit_module.MoonViT3dEncoder.__init__

            def _patched_init(self, *a, **kw):
                self.use_deterministic_attn = False
                _orig_init(self, *a, **kw)

            _vit_module.MoonViT3dEncoder.__init__ = _patched_init
            _vit_module.MoonViT3dEncoder._patched = True

        MoonViT3dEncoder = get_class_from_dynamic_module(
            "modeling_kimi_k25.MoonViT3dEncoder", hf_model_path,
        )
        if not hasattr(MoonViT3dEncoder, "use_deterministic_attn"):
            MoonViT3dEncoder.use_deterministic_attn = False

        hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
        vision_config = hf_config.vision_config

        self.vision_tower = MoonViT3dPretrainedModel(VisionTowerConfig(vision_config))
        self.mm_projector = PatchMergerMLP(ProjectorConfig(vision_config))

        logger.info("Vision modules loaded from %s", hf_model_path)

    # ------------------------------------------------------------------
    # Pipeline parallel
    # ------------------------------------------------------------------
    def set_input_tensor(self, input_tensor):
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1
        self.language_model.set_input_tensor(input_tensor[0])

    # ------------------------------------------------------------------
    # Vision feature extraction
    # ------------------------------------------------------------------
    def _extract_image_features(self, pixel_values, grid_thw):
        """Run vision tower + projector, return list of feature tensors."""
        # Reshape flat pixels to 4D if needed (from TP broadcast)
        if pixel_values.ndim <= 3:
            patch_size = self.vision_tower.patch_embed.proj.kernel_size[0]
            in_channels = self.vision_tower.patch_embed.proj.in_channels
            expected = in_channels * patch_size * patch_size
            if pixel_values.shape[-1] == expected:
                pixel_values = pixel_values.reshape(-1, in_channels, patch_size, patch_size)

        if grid_thw.ndim == 3:
            grid_thw = grid_thw.reshape(-1, grid_thw.shape[-1])

        image_features = self.vision_tower(pixel_values, grid_thw)
        projected = self.mm_projector(image_features)
        if isinstance(projected, torch.Tensor):
            projected = [projected]
        return projected

    # ------------------------------------------------------------------
    # Embedding merge (copied from Bridge)
    # ------------------------------------------------------------------
    def _merge_input_ids_with_image_features(
        self,
        image_features: List[torch.Tensor],
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """Replace image placeholder embeddings with projected image features."""
        _, embed_dim = image_features[0].shape
        feature_lengths = [x.shape[0] for x in image_features]
        total_image_features = sum(feature_lengths)
        image_features_cat = torch.cat(image_features, dim=0)
        num_images = len(image_features)

        image_token_index = self.media_placeholder_token_id
        batch_size, sequence_length = input_ids.shape

        num_placeholders = (input_ids == image_token_index).sum().item()
        is_pre_expanded = num_placeholders == total_image_features
        is_truncated = (
            not is_pre_expanded
            and num_placeholders > num_images
            and num_placeholders < total_image_features
        )

        if is_pre_expanded or is_truncated:
            if is_truncated:
                per_sample = (input_ids == image_token_index).sum(dim=1)
                parts, fi = [], 0
                for si in range(batch_size):
                    rem = per_sample[si].item()
                    while rem > 0 and fi < num_images:
                        n = min(image_features[fi].shape[0], rem)
                        parts.append(image_features[fi][:n])
                        rem -= n
                        fi += 1
                image_features_cat = torch.cat(parts, dim=0)

            final_embedding = inputs_embeds.clone()
            image_mask = input_ids == image_token_index
            final_embedding[image_mask] = image_features_cat.to(inputs_embeds.dtype)

            if attention_mask is None:
                attention_mask = (input_ids != self.pad_token_id).long()
            position_ids = (attention_mask.cumsum(-1) - 1).masked_fill_(
                attention_mask == 0, 1,
            )

            if labels is not None:
                final_labels = labels.clone()
                final_labels[image_mask] = self.ignore_index
            else:
                final_labels = None

            return final_embedding, None, final_labels, position_ids

        raise NotImplementedError(
            "Dynamic expansion mode not implemented. "
            "Ensure mock data pre-expands image placeholders."
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # 1. Text embeddings
        inputs_embeds = self.language_model.embedding(
            input_ids=input_ids, position_ids=None,
        )  # [S, B, H]
        inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()  # [B, S, H]

        # 2. Vision features
        has_pixels = pixel_values is not None and pixel_values.numel() > 0
        not_generation = input_ids is not None and input_ids.shape[1] != 1

        if has_pixels and not_generation:
            pixel_values = pixel_values.to(self.vision_tower.dtype)
            image_features = self._extract_image_features(pixel_values, image_grid_thw)
            inputs_embeds = inputs_embeds.to(image_features[0].dtype)

            # 3. Merge
            inputs_embeds, attention_mask, labels, position_ids = (
                self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels,
                )
            )
            attention_mask = None  # causal mask handled internally

        # 4. Transpose to Megatron format (T, B, D)
        inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()

        if self.config.sequence_parallel:
            inputs_embeds = tensor_parallel.scatter_to_sequence_parallel_region(
                inputs_embeds,
            )

        # 5. Language model forward
        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=inputs_embeds,
            labels=labels,
            loss_mask=loss_mask,
        )
        return output
