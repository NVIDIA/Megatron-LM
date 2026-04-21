# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved. Except portions as noted which are Copyright (c) 2023 OpenGVLab and licensed under the MIT license found in LICENSE.
import torch
from einops import rearrange
from typing import Optional, Union, List, Tuple


def patchify_image(x: torch.Tensor, patch_dim: int) -> torch.Tensor:
    """Patchify an image:
        Option 1: (C,H,W) == (3,H,W) -> (L,3*patch_dim*patch_dim)
        Option 2: (B,C,H,W) == (B,3,H,W) -> (B,L,3*patch_dim*patch_dim)
    """
    assert x.shape[-2] % patch_dim == 0 and x.shape[-1] % patch_dim == 0, \
        f"H and W must be divisible by patch_dim={patch_dim}, found {x.shape[-2:]}"

    if x.ndim == 3:
        assert x.shape[0] == 3, f"Expected (C,H,W) tensor, found {x.shape}"
        py = x.shape[-2] // patch_dim  # H
        px = x.shape[-1] // patch_dim  # W
        x = rearrange(x, 'c (py yy) (px xx) -> (py px) (c yy xx)',
                            py=py, yy=patch_dim,
                            px=px, xx=patch_dim,
        )
    elif x.ndim == 4:
        assert x.shape[1] == 3, f"Expected (B,C,H,W) tensor, found {x.shape}"
        py = x.shape[-2] // patch_dim  # H
        px = x.shape[-1] // patch_dim  # W
        x = rearrange(x, 'b c (py yy) (px xx) -> b (py px) (c yy xx)',
                            py=py, yy=patch_dim,
                            px=px, xx=patch_dim,
        )
    else:
        raise NotImplementedError(f"Expected (B,C,H,W) or (C,H,W) input tensor, found shape {x.shape}")

    return x


def unpatchify_image(x: torch.Tensor, img_H: int, img_W: int, patch_dim: int) -> torch.Tensor:
    """Unpatchify an image (reverse of patchify_image()):
        Option 1: (L,C) == (L,3*patch_dim*patch_dim) -> (3,H,W)
        Option 2: (B,L,C) == (B,L,3*patch_dim*patch_dim) -> (B,3,H,W)
    """
    assert img_H % patch_dim == 0 and img_W % patch_dim == 0 and patch_dim > 0, \
        f"Expected img_H/img_W to be divisible by patch_dim={patch_dim}, found img_H={img_H}, img_W={img_W}"

    py = img_H // patch_dim
    px = img_W // patch_dim

    if x.ndim == 2:
        expected_shape = (py * px, 3 * patch_dim * patch_dim)
        assert x.shape == expected_shape, f"Expected x.shape={expected_shape}, found x.shape={x.shape}"

        x = rearrange(x, '(py px) (c yy xx) -> c (py yy) (px xx)',
                            py=py, yy=patch_dim,
                            px=px, xx=patch_dim,
        )

    elif x.ndim == 3:
        expected_shape = (x.shape[0], py * px, 3 * patch_dim * patch_dim)
        assert x.shape == expected_shape, f"Expected x.shape={expected_shape}, found x.shape={x.shape}"

        x = rearrange(x, 'b (py px) (c yy xx) -> b c (py yy) (px xx)',
                            py=py, yy=patch_dim,
                            px=px, xx=patch_dim,
        )

    else:
        raise NotImplementedError(f"Expected (B,L,C) or (L,C) input tensor, found shape {x.shape}")

    return x
