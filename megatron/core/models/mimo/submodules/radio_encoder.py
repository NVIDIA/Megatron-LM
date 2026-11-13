# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reusable RADIO vision encoder wrapper for MIMO models."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch

from megatron.core.models.multimodal.llava_model import pixel_shuffle, pixel_shuffle_dynamic_res
from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import sharded_state_dict_default

RADIO_ENCODER_MODULE_NAME = "radio_encoder"

# Preserve the previous helper name for example imports.
_pixel_shuffle_dynamic_res = pixel_shuffle_dynamic_res


class RADIOEncoderWrapper(MegatronModule):
    """RADIO encoder wrapper matching the Nemotron6-MoE VLM provider."""

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        pg_collection: Optional[ProcessGroupCollection],
        img_h: int,
        img_w: int,
        patch_dim: int,
        class_token_len: int,
        drop_class_token: bool = True,
        apply_pixel_shuffle: bool = True,
        force_eval_mode: bool = False,
        dynamic_resolution: bool = False,
    ) -> None:
        super().__init__(config=transformer_config)
        self.class_token_len = class_token_len
        self.drop_class_token = drop_class_token
        self.apply_pixel_shuffle = apply_pixel_shuffle
        self.force_eval_mode = force_eval_mode
        self.dynamic_resolution = dynamic_resolution
        self.radio_model = RADIOViTModel(
            transformer_config=transformer_config,
            transformer_layer_spec=transformer_layer_spec,
            patch_dim=patch_dim,
            img_h=img_h,
            img_w=img_w,
            class_token_len=class_token_len,
            add_class_token=True,
            max_img_h=2048,
            max_img_w=2048,
            has_cpe=True,
            embedder_bias=False,
            dynamic_resolution=dynamic_resolution,
            force_eval_mode=force_eval_mode,
            pg_collection=pg_collection,
        )

    def forward(
        self, x: torch.Tensor, imgs_sizes: Optional[torch.Tensor] = None, packed_seq_params=None
    ) -> torch.Tensor:
        """Run RADIO, drop class tokens, and apply pixel shuffle."""
        context = torch.no_grad() if self.force_eval_mode else nullcontext()
        with context:
            x = x.to(dtype=self.radio_model.embedder.weight.dtype)
            embeddings = self.radio_model(
                x, imgs_sizes=imgs_sizes, packed_seq_params=packed_seq_params
            )
        if self.drop_class_token:
            if self.dynamic_resolution and imgs_sizes is not None and self.class_token_len > 0:
                # Class tokens are interleaved between tiles; build mask to remove them.
                remove_mask = torch.full(
                    (embeddings.shape[-2],), True, dtype=torch.bool, device=embeddings.device
                )
                patch_dim = self.radio_model.patch_dim
                if torch.is_tensor(imgs_sizes):
                    seq_lens = torch.prod(imgs_sizes // patch_dim, dim=-1)
                else:
                    seq_lens = torch.tensor(
                        [(h // patch_dim) * (w // patch_dim) for h, w in imgs_sizes]
                    )
                current_length = 0
                for sl in seq_lens:
                    remove_mask[current_length : current_length + self.class_token_len] = False
                    current_length += int(sl) + self.class_token_len
                embeddings = embeddings[:, remove_mask, :]
            else:
                embeddings = embeddings[:, self.class_token_len :, :]
        if self.apply_pixel_shuffle:
            if self.dynamic_resolution and imgs_sizes is not None:
                embeddings = _pixel_shuffle_dynamic_res(
                    embeddings, imgs_sizes, self.radio_model.patch_dim
                )
            else:
                embeddings = pixel_shuffle(embeddings, scale_factor=0.5)
        return embeddings

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Delegate checkpoint serialization while preserving child module prefixes."""
        # Param-less wrapper: delegate straight to the child so checkpoint keys keep
        # the ``radio_model.`` prefix without the base-class tp/dp_cp_group machinery.
        sharded_sd = {}
        for name, child in self.named_children():
            sharded_sd.update(
                sharded_state_dict_default(child, f"{prefix}{name}.", sharded_offsets, metadata)
            )
        return sharded_sd
