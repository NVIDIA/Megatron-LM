# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""MRoPE (Multimodal Rotary Position Embedding) position ID computation.

Computes 3D position IDs for Qwen3.5-VL: for text tokens all three
dimensions share sequential positions; for image tokens the three
dimensions encode (temporal, height, width) in the merged spatial grid.

Reference: HF ``Qwen3VLForConditionalGeneration.get_rope_index``.
"""

from typing import Optional

import torch
from torch import Tensor


def compute_mrope_position_ids(
    input_ids: Tensor,
    image_grid_thw: Optional[Tensor],
    image_token_id: int,
    spatial_merge_size: int = 2,
) -> Tensor:
    """Compute 3D MRoPE position IDs for Qwen3.5-VL.

    For text tokens: sequential positions on all 3 dimensions.
    For image tokens: 2D spatial grid positions (temporal, height, width).

    Args:
        input_ids: ``[B, S]`` token IDs.
        image_grid_thw: ``[num_images, 3]`` per-image
            ``(temporal, height, width)`` in patch-grid units.
        image_token_id: Token ID used for image placeholders.
        spatial_merge_size: Merge factor (positions use merged grid
            coordinates).

    Returns:
        Position IDs ``[3, B, S]`` for MRoPE (temporal, height, width).
    """
    B, S = input_ids.shape
    device = input_ids.device

    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    mrope_ids = position_ids.unsqueeze(0).expand(3, -1, -1).clone()

    if image_grid_thw is None or image_grid_thw.numel() == 0:
        return mrope_ids

    img_idx = 0

    for b in range(B):
        image_mask = input_ids[b] == image_token_id
        if not image_mask.any():
            continue

        pos_offset = 0
        text_pos = 0
        t = h = w = n_tokens = 0

        for s in range(S):
            if input_ids[b, s] == image_token_id:
                if pos_offset == 0:
                    if img_idx >= image_grid_thw.shape[0]:
                        break
                    t = int(image_grid_thw[img_idx, 0].item())
                    h = (
                        int(image_grid_thw[img_idx, 1].item())
                        // spatial_merge_size
                    )
                    w = (
                        int(image_grid_thw[img_idx, 2].item())
                        // spatial_merge_size
                    )
                    n_tokens = t * h * w

                temporal_idx = pos_offset // (h * w)
                spatial_idx = pos_offset % (h * w)
                h_idx = spatial_idx // w
                w_idx = spatial_idx % w

                mrope_ids[0, b, s] = temporal_idx + text_pos
                mrope_ids[1, b, s] = h_idx + text_pos
                mrope_ids[2, b, s] = w_idx + text_pos

                pos_offset += 1
                if pos_offset >= n_tokens:
                    text_pos += max(t, h, w)
                    pos_offset = 0
                    img_idx += 1
            else:
                mrope_ids[0, b, s] = text_pos
                mrope_ids[1, b, s] = text_pos
                mrope_ids[2, b, s] = text_pos
                text_pos += 1

    return mrope_ids
