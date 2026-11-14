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


def _postprocess_output_counts(
    imgs_sizes: torch.Tensor,
    *,
    patch_dim: int,
    class_token_len: int,
    drop_class_token: bool,
    apply_pixel_shuffle: bool,
) -> torch.Tensor:
    """Derive the projected embedding count for each grouped image or tubelet."""
    sizes = imgs_sizes
    if sizes.numel() == 0:
        return torch.empty(0, dtype=torch.int64)
    if sizes.ndim != 2 or sizes.shape[1] != 2:
        raise ValueError(
            f"grouped imgs_sizes must have shape [num_media, 2], got {tuple(sizes.shape)}"
        )
    if torch.any(sizes <= 0) or torch.any(sizes % patch_dim != 0):
        raise ValueError(
            f"grouped image sizes must be positive multiples of patch_dim={patch_dim}: "
            f"{sizes.tolist()}"
        )

    patch_grid = sizes // patch_dim
    counts = torch.prod(patch_grid, dim=-1)
    if not drop_class_token:
        counts = counts + class_token_len
    if apply_pixel_shuffle:
        if not drop_class_token and class_token_len:
            raise ValueError("dynamic-resolution pixel shuffle requires class tokens to be dropped")
        if torch.any(patch_grid % 2 != 0):
            raise ValueError(
                f"pixel shuffle requires even patch-grid dimensions, got {patch_grid.tolist()}"
            )
        counts = torch.prod(patch_grid // 2, dim=-1)
    return counts


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
        *,
        force_cpe_eval_mode: bool = False,
        interpolate_only_cpe: bool = False,
        cpe_aspect_ratio_select: bool = False,
        disable_cpe: bool = False,
        temporal_patch_dim: int = 1,
        temporal_ckpt_compat: bool = False,
        separate_video_embedder: bool = False,
    ) -> None:
        super().__init__(config=transformer_config)
        if patch_dim <= 0 or temporal_patch_dim <= 0:
            raise ValueError("patch_dim and temporal_patch_dim must be positive")
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
            has_cpe=not disable_cpe,
            embedder_bias=False,
            dynamic_resolution=dynamic_resolution,
            force_eval_mode=force_eval_mode,
            force_cpe_eval_mode=force_cpe_eval_mode,
            interpolate_only_cpe=interpolate_only_cpe,
            cpe_aspect_ratio_select=cpe_aspect_ratio_select,
            temporal_patch_dim=temporal_patch_dim,
            temporal_ckpt_compat=temporal_ckpt_compat,
            separate_video_embedder=separate_video_embedder,
            pg_collection=pg_collection,
        )

    def forward(
        self,
        x: torch.Tensor,
        imgs_sizes: Optional[torch.Tensor | list[tuple[int, int]]] = None,
        packed_seq_params=None,
        *,
        num_frames: Optional[torch.Tensor | list[int]] = None,
        expected_output_counts: Optional[torch.Tensor | list[int]] = None,
    ) -> torch.Tensor:
        """Run RADIO and postprocess embeddings using the grouped frame geometry.

        ``imgs_sizes`` contains one (height, width) pair per input frame. When
        temporal grouping is enabled, ``num_frames`` gives the frame count per
        media item (one for an image). ``expected_output_counts``, if supplied,
        contains one output token count per image or resulting video tubelet,
        in packed order, rather than one count per original video.
        """
        if expected_output_counts is not None and (
            not self.dynamic_resolution or imgs_sizes is None
        ):
            raise ValueError("expected_output_counts requires dynamic-resolution imgs_sizes")
        if self.dynamic_resolution and imgs_sizes is not None:
            # Retain the backbone's tensor geometry contract, including for callers
            # that keep their input metadata as host lists.
            imgs_sizes = torch.as_tensor(imgs_sizes)
            if imgs_sizes.numel() == 0:
                imgs_sizes = imgs_sizes.reshape(0, 2)
            if imgs_sizes.ndim != 2 or imgs_sizes.shape[1] != 2:
                raise ValueError("imgs_sizes must have shape [num_frames, 2]")
            if (
                imgs_sizes.is_floating_point()
                or imgs_sizes.is_complex()
                or imgs_sizes.dtype == torch.bool
            ):
                raise ValueError("imgs_sizes must contain integer dimensions")
            if torch.any(imgs_sizes <= 0) or torch.any(
                imgs_sizes % self.radio_model.patch_dim != 0
            ):
                raise ValueError("imgs_sizes must contain positive multiples of patch_dim")
            if self.apply_pixel_shuffle:
                if not self.drop_class_token and self.class_token_len:
                    raise ValueError(
                        "dynamic-resolution pixel shuffle requires class tokens to be dropped"
                    )
                if torch.any((imgs_sizes // self.radio_model.patch_dim) % 2 != 0):
                    raise ValueError("pixel shuffle requires even patch-grid dimensions")

        context = torch.no_grad() if self.force_eval_mode else nullcontext()
        with context:
            x = x.to(dtype=self.radio_model.embedder.weight.dtype)
            radio_output = self.radio_model(
                x, imgs_sizes=imgs_sizes, packed_seq_params=packed_seq_params, num_frames=num_frames
            )
            if self.radio_model.temporal_patch_dim > 1:
                embeddings, grouped_imgs_sizes, _ = radio_output
            else:
                embeddings = radio_output
                grouped_imgs_sizes = imgs_sizes

        if self.drop_class_token:
            if (
                self.dynamic_resolution
                and grouped_imgs_sizes is not None
                and self.class_token_len > 0
            ):
                keep_mask = torch.ones(
                    embeddings.shape[-2], dtype=torch.bool, device=embeddings.device
                )
                sequence_lengths = torch.prod(
                    grouped_imgs_sizes // self.radio_model.patch_dim, dim=-1
                ).tolist()
                offset = 0
                for sequence_length in sequence_lengths:
                    keep_mask[offset : offset + self.class_token_len] = False
                    offset += sequence_length + self.class_token_len
                embeddings = embeddings[:, keep_mask, :]
            else:
                embeddings = embeddings[:, self.class_token_len :, :]

        if self.apply_pixel_shuffle:
            if self.dynamic_resolution and grouped_imgs_sizes is not None:
                embeddings = _pixel_shuffle_dynamic_res(
                    embeddings, grouped_imgs_sizes, self.radio_model.patch_dim
                )
            else:
                embeddings = pixel_shuffle(embeddings, scale_factor=0.5)

        if expected_output_counts is not None:
            actual_counts = _postprocess_output_counts(
                grouped_imgs_sizes,
                patch_dim=self.radio_model.patch_dim,
                class_token_len=self.class_token_len,
                drop_class_token=self.drop_class_token,
                apply_pixel_shuffle=self.apply_pixel_shuffle,
            )
            expected_counts = torch.as_tensor(expected_output_counts, device=actual_counts.device)
            if (
                expected_counts.ndim != 1
                or expected_counts.is_floating_point()
                or expected_counts.is_complex()
                or expected_counts.dtype == torch.bool
                or torch.any(expected_counts < 0)
            ):
                raise ValueError("expected_output_counts must be a vector of nonnegative integers")
            if not torch.equal(actual_counts, expected_counts):
                raise ValueError(
                    "RADIO media counts diverged after temporal grouping: "
                    f"expected={expected_counts.tolist()}, actual={actual_counts.tolist()}"
                )
            expected_total = int(expected_counts.sum().item())
            if expected_total != embeddings.shape[-2]:
                raise ValueError(
                    "RADIO output length does not match expected media counts: "
                    f"output={embeddings.shape[-2]}, expected_sum={expected_total}"
                )
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
