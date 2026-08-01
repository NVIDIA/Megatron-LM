# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Pixel shuffle for dynamic-resolution vision features.

Folds a 2x2 spatial neighbourhood of patches into the channel dimension, quartering the token
count. Element ordering intentionally differs from `llava_model.pixel_shuffle`'s packed
dynamic-resolution branch, which reshapes 4 horizontally adjacent patches into one token
instead of a 2x2 block -- a different operation, not a different spelling of the same one.
This is the version that matches the reference Nemotron Omni implementation and is
e2e-validated; do not "fix" it to agree with the other one.
"""

from typing import List, Tuple, Union

import torch


def pixel_shuffle_dynamic_res(
    x: torch.Tensor,
    imgs_sizes: Union[torch.Tensor, List[Tuple[int, int]]],
    patch_dim: int,
    scale_factor: float = 0.5,
    version: int = 2,
) -> torch.Tensor:
    """Apply pixel shuffle per tile across a packed, variable-length sequence.

    Args:
        x (torch.Tensor): Packed features, `[b, total_patches, hidden]`, class tokens already
            removed.
        imgs_sizes: Per-tile `(height, width)` in pixels. Tensor `[num_tiles, 2]` or a list of
            pairs.
        patch_dim (int): Patch size in pixels.
        scale_factor (float): Spatial reduction per axis. 0.5 quarters the token count.
        version (int): 2 restores row-major ordering after the fold; 1 leaves it transposed.

    Return:
        (torch.Tensor) `[b, total_patches * scale_factor**2, hidden / scale_factor**2]`.
    """
    if not torch.is_tensor(imgs_sizes):
        imgs_sizes = torch.tensor(imgs_sizes, dtype=torch.long, device=x.device)

    seq_lens = torch.prod(imgs_sizes // patch_dim, dim=-1)
    splits = torch.split(x, seq_lens.tolist(), dim=-2)

    out = []
    for i, sv in enumerate(splits):
        h = int(imgs_sizes[i][0]) // patch_dim
        w = int(imgs_sizes[i][1]) // patch_dim
        sv = sv.reshape(sv.shape[0], h, w, -1)

        n, h, w, c = sv.size()
        sv = sv.view(n, h, int(w * scale_factor), int(c / scale_factor))
        sv = sv.permute(0, 2, 1, 3).contiguous()
        sv = sv.view(
            n, int(w * scale_factor), int(h * scale_factor), int(c / (scale_factor * scale_factor))
        )

        if version == 2:
            sv = sv.permute(0, 2, 1, 3).contiguous()

        sv = sv.reshape(sv.shape[0], -1, sv.shape[-1])
        out.append(sv)

    return torch.cat(out, dim=-2)
