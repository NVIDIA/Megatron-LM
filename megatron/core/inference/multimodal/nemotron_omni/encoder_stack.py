# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Device-side encoder stack for Nemotron Omni.

Runs at request admission, deliberately outside the graphed per-step region: encoder output
shape varies per request, so capturing a tower inside the step would freeze one request's
shapes into the graph, and chunked prefill would re-run the tower once per chunk.

Images and videos are encoded in a *single* batched RADIO call. The reference vLLM
implementation sets `requires_sequential_video_encoding = True` and encodes videos one at a
time, but only because batched dynamic-resolution video is unimplemented there; mcore's
`RADIOViTModel._apply_temporal_grouping` already handles mixed image/video items in one packed
batch. The tubelet grouping and last-frame padding are per-item, so batching is numerically
identical and materially faster.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from megatron.core.inference.multimodal.nemotron_omni.config import NemotronOmniConfig
from megatron.core.models.vision.evs import compute_retention_mask
from megatron.core.models.vision.pixel_shuffle import pixel_shuffle_dynamic_res
from megatron.core.packed_seq_params import PackedSeqParams


@dataclass
class EncodedMedia:
    """Encoder rows for one media item, projected to the language model's hidden size."""

    embeddings: torch.Tensor
    """`[num_tokens, hidden_size]`."""

    tokens_per_tubelet: List[int]
    """Row count per tubelet. Single-element for images."""


def _build_packed_seq_params(seq_lens: Sequence[int], device: torch.device) -> PackedSeqParams:
    """Build `thd` varlen attention params for a packed, per-tile sequence.

    A fresh object per call is mandatory: `RADIOViTModel.forward` *mutates*
    `packed_seq_params.cu_seqlens_q/kv` and `max_seqlen_q/kv` in place when it prepends class
    tokens, so a reused instance would accumulate the class-token offset on every call.
    """
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(list(seq_lens)), dim=0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    max_seqlen = torch.tensor(max(seq_lens) if seq_lens else 0, dtype=torch.int32, device=device)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )


class NemotronOmniEncoderStack(torch.nn.Module):
    """Owns the vision tower, the two projectors, and the audio tower.

    Args:
        config (NemotronOmniConfig): Resolved model configuration.
        vision_model: `RADIOViTModel` instance.
        vision_projection: Vision projector, `4 * vit_hidden -> hidden_size`.
        audio_model: Optional projected Parakeet tower.
    """

    def __init__(
        self,
        config: NemotronOmniConfig,
        vision_model: torch.nn.Module,
        vision_projection: torch.nn.Module,
        audio_model: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vision_model = vision_model
        self.vision_projection = vision_projection
        self.audio_model = audio_model

    @property
    def patch_dim(self) -> int:
        """Vision patch size in pixels."""
        return self.config.vision.patch_size

    @property
    def class_token_len(self) -> int:
        """Non-patch prefix tokens RADIO prepends per tile."""
        return self.config.vision.class_token_len

    def _strip_class_tokens(
        self, features: torch.Tensor, tile_seq_lens: Sequence[int]
    ) -> torch.Tensor:
        """Drop the CLS + register prefix that RADIO prepends to every tile.

        In the dynamic-resolution path the prefixes are *interleaved* between tiles rather than
        sitting at the front of the sequence, so this builds a keep-mask instead of slicing.
        """
        if self.class_token_len == 0:
            return features

        keep = torch.ones(features.shape[-2], dtype=torch.bool, device=features.device)
        cursor = 0
        for seq_len in tile_seq_lens:
            keep[cursor : cursor + self.class_token_len] = False
            cursor += int(seq_len) + self.class_token_len
        assert cursor == features.shape[-2], (
            f"class-token stripping covered {cursor} of {features.shape[-2]} tokens; "
            f"class_token_len={self.class_token_len} is probably wrong"
        )
        return features[:, keep, :]

    def _tile_seq_lens(self, imgs_sizes: torch.Tensor) -> List[int]:
        """Patch count per tile, from pixel `(height, width)` pairs."""
        return torch.prod(imgs_sizes // self.patch_dim, dim=-1).tolist()

    @torch.inference_mode()
    def encode_visual(
        self, pixel_values: torch.Tensor, imgs_sizes: torch.Tensor, num_frames: Sequence[int]
    ) -> List[EncodedMedia]:
        """Encode images and videos in one batched RADIO call.

        Args:
            pixel_values (torch.Tensor): `[1, total_patches, 3 * patch**2]`, patchified frames
                for every item concatenated in prompt order.
            imgs_sizes (torch.Tensor): `[total_frames, 2]` pixel `(height, width)`, one row per
                *frame* (before tubelet grouping).
            num_frames (Sequence[int]): Frames per media item; 1 marks a still image.

        Return:
            (List[EncodedMedia]) One entry per media item, in the input order.
        """
        device = next(self.vision_model.parameters()).device
        dtype = self.vision_model.embedder.weight.dtype
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        imgs_sizes = imgs_sizes.to(device=device)

        packed_seq_params = _build_packed_seq_params(self._tile_seq_lens(imgs_sizes), device)

        # temporal_patch_dim > 1 makes forward() return the regrouped sizes and per-item
        # tubelet counts alongside the features.
        features, grouped_sizes, tubelets_per_item = self.vision_model(
            pixel_values,
            imgs_sizes=imgs_sizes,
            packed_seq_params=packed_seq_params,
            num_frames=list(num_frames),
        )

        tubelet_seq_lens = self._tile_seq_lens(grouped_sizes)
        features = self._strip_class_tokens(features, tubelet_seq_lens)
        features = pixel_shuffle_dynamic_res(features, grouped_sizes, self.patch_dim)
        features = self.vision_projection(features)

        # Post-shuffle token count per tubelet: patches // 4.
        tokens_per_tubelet = [seq_len // 4 for seq_len in tubelet_seq_lens]
        rows = features[0]
        assert rows.shape[0] == sum(tokens_per_tubelet), (
            f"projector produced {rows.shape[0]} rows but the tubelet grid implies "
            f"{sum(tokens_per_tubelet)}"
        )

        encoded: List[EncodedMedia] = []
        row_cursor = 0
        tubelet_cursor = 0
        for item_frames, item_tubelets in zip(num_frames, tubelets_per_item):
            item_counts = tokens_per_tubelet[tubelet_cursor : tubelet_cursor + item_tubelets]
            num_rows = sum(item_counts)
            item_rows = rows[row_cursor : row_cursor + num_rows]

            if item_frames > 1 and self.config.video_pruning_rate:
                item_rows, item_counts = self._prune_video(
                    item_rows, tokens_per_tubelet=item_counts, grid=grouped_sizes[tubelet_cursor]
                )

            encoded.append(EncodedMedia(embeddings=item_rows, tokens_per_tubelet=item_counts))
            row_cursor += num_rows
            tubelet_cursor += item_tubelets

        assert (
            row_cursor == rows.shape[0]
        ), f"consumed {row_cursor} of {rows.shape[0]} projected rows"
        return encoded

    def _prune_video(
        self, rows: torch.Tensor, tokens_per_tubelet: Sequence[int], grid: torch.Tensor
    ) -> Tuple[torch.Tensor, List[int]]:
        """Apply EVS to one video's projected rows.

        Every tubelet of a video shares one spatial grid (all frames were resized to a common
        target), which is what lets the rows be viewed as `[T, H, W, hidden]`.

        Return:
            (Tuple[torch.Tensor, List[int]]) Retained rows and the per-tubelet counts after
            pruning. The counts are re-derived from the mask rather than predicted, because the
            prompt's separator tokens must be re-rendered against the actual retention.
        """
        num_tubelets = len(tokens_per_tubelet)
        token_h = int(grid[0]) // self.patch_dim // 2
        token_w = int(grid[1]) // self.patch_dim // 2

        mask = compute_retention_mask(
            rows,
            video_size_thw=(num_tubelets, token_h, token_w),
            q=float(self.config.video_pruning_rate),
        )
        retained_per_tubelet = mask.view(num_tubelets, -1).sum(dim=-1).tolist()
        return rows[mask], [int(count) for count in retained_per_tubelet]

    @torch.inference_mode()
    def encode_audio(
        self,
        mel_features: torch.Tensor,
        attention_mask: torch.Tensor,
        clips_per_item: Sequence[int],
    ) -> List[EncodedMedia]:
        """Encode audio clips and regroup them into per-item row blocks.

        Args:
            mel_features (torch.Tensor): `[total_clips, max_frames, num_mel_bins]`.
            attention_mask (torch.Tensor): `[total_clips, max_frames]` bool, True on valid
                frames. The tower trims each clip's output to the matching row count.
            clips_per_item (Sequence[int]): Clips belonging to each audio item, in order.

        Return:
            (List[EncodedMedia]) One entry per audio item.
        """
        assert self.audio_model is not None, "encode_audio requires an audio tower"
        device = next(self.audio_model.parameters()).device
        dtype = self.audio_model.projection.fc1.weight.dtype
        rows_per_clip = self.audio_model(
            mel_features.to(device=device, dtype=dtype), attention_mask.to(device=device)
        )

        encoded: List[EncodedMedia] = []
        cursor = 0
        for num_clips in clips_per_item:
            item_rows = torch.cat(rows_per_clip[cursor : cursor + num_clips], dim=0)
            encoded.append(
                EncodedMedia(embeddings=item_rows, tokens_per_tubelet=[item_rows.shape[0]])
            )
            cursor += num_clips
        return encoded

    def num_tubelets(self, num_frames: int) -> int:
        """Tubelet count for a video with `num_frames` frames.

        The tail tubelet is padded by repeating the last frame, so a frame count that does not
        divide the temporal patch size still yields a whole tubelet.
        """
        temporal = self.config.vision.video_temporal_patch_size
        if num_frames == 1 or temporal <= 1:
            return 1
        return math.ceil(num_frames / temporal)
