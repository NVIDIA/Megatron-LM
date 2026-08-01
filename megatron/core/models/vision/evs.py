# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Efficient Video Sampling: prune video tokens that repeat the previous frame.

Scores every token by how much it differs from the token at the same spatial position in the
previous frame, then keeps the most-changed ones. Static regions of a video collapse to a few
tokens while moving regions survive.

Three details are load-bearing for reproducibility against the reference implementation:

- The first frame's dissimilarity is the literal sentinel `255`, not `+inf`. With `+inf` the
  stable sort's tie-breaking among first-frame tokens changes, so the retained set differs.
- The sort is `stable=True` descending. Ties are extremely common (identical static regions),
  so an unstable sort makes the retained set non-deterministic.
- Pruning runs *after* the projector, on language-model-dimension embeddings, not on tower
  features.
"""

from typing import Tuple, Union

import torch


def compute_retained_tokens_count(tokens_per_frame: int, num_frames: int, q: float) -> int:
    """Number of video tokens to keep.

    Floored at one full frame's worth, so the first frame always survives intact regardless of
    how aggressive the pruning rate is.

    Args:
        tokens_per_frame (int): Tokens per frame (or per tubelet).
        num_frames (int): Number of frames (or tubelets).
        q (float): Pruning rate in [0, 1); 0.7 discards ~70% of tokens.

    Return:
        (int) Retained token count.
    """
    total_tokens = tokens_per_frame * num_frames
    return max(tokens_per_frame, int(total_tokens * (1 - q)))


def compute_retention_mask(
    video_embeds: torch.Tensor,
    video_size_thw: Union[torch.Tensor, Tuple[int, int, int]],
    q: float,
    spatial_merge_size: int = 1,
) -> torch.Tensor:
    """Select which video tokens to keep.

    Args:
        video_embeds (torch.Tensor): `[T * H * W // spatial_merge_size**2, hidden]`, already
            projected to the language model's hidden size.
        video_size_thw: `(T, H, W)` in post-pixel-shuffle token units.
        q (float): Pruning rate in [0, 1).
        spatial_merge_size (int): Further spatial reduction. 1 for Nemotron Omni, whose
            reduction already happened in pixel shuffle.

    Return:
        (torch.Tensor) Bool mask over the flattened token axis, True where retained.
    """
    num_frames, height, width = (int(v) for v in video_size_thw)

    # reshape rather than einops: this runs under torch.compile in some callers, and einops
    # forces a graph break.
    video_embeds = video_embeds.reshape(
        num_frames, height // spatial_merge_size, width // spatial_merge_size, video_embeds.size(-1)
    )
    tokens_per_frame = (height // spatial_merge_size) * (width // spatial_merge_size)

    similarity = torch.nn.functional.cosine_similarity(
        video_embeds[1:, ...], video_embeds[:-1, ...], dim=-1
    )
    dissimilarity = 1 - similarity

    # Sentinel, not infinity -- see the module docstring.
    dissimilarity = torch.cat(
        [255 * torch.ones_like(video_embeds[:1, :, :, 0]), dissimilarity], dim=0
    )

    dissimilarity_flat = dissimilarity.view(-1)
    order = torch.argsort(dissimilarity_flat, dim=-1, descending=True, stable=True)
    retain_num_tokens = compute_retained_tokens_count(
        tokens_per_frame=tokens_per_frame, num_frames=num_frames, q=q
    )

    retention_mask = torch.zeros_like(dissimilarity_flat, dtype=torch.bool)
    retention_mask.index_fill_(0, order[:retain_num_tokens], True)
    return retention_mask
