# Copyright (c) 2026 DeepSeek-AI.
# Adapted from deepseek-ai/DeepSeek-V4.1-Flash (MIT; see LICENSE.deepseek).
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Image patch preparation and image-span metadata."""

import math
from dataclasses import dataclass

import numpy as np
import torch

TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)


@dataclass
class ImageInput:
    """A patch grid and its typed span in one language-model input sequence."""

    start: int
    patches: torch.Tensor
    n_vit_h: int
    n_vit_w: int
    types: torch.Tensor


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    """Count patch slots, newlines and image delimiters for a projected grid."""
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height: int, best_width: int, patch_size: int, downsample_ratio: int):
    """Token grid the aligner produces from a patch grid of this pixel size."""
    return math.ceil((best_height // patch_size) / downsample_ratio), math.ceil(
        (best_width // patch_size) / downsample_ratio
    )


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    """Largest aspect-preserving pixel size whose token grid still fits in max_n_token."""
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:  # very tall: collapse to a single column
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:  # very wide: collapse to a single row
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return (
        math.floor(height * beta / patch_size) * patch_size,
        math.floor(width * beta / patch_size) * patch_size,
    )


def safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token):
    """Shrink the pixel size until the image costs at most max_n_token LLM tokens."""
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, max_n_token
        )
        n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def plan_image_grid(width: int, height: int, args):
    """Resize plan for an image of the given original size; a pure function of its arguments."""
    p = args.vision_patch_size
    if args.vision_max_wh_ratio is not None and width > height * args.vision_max_wh_ratio:
        width = height * args.vision_max_wh_ratio
    if 0 < width * height < args.vision_min_pixels:
        ratio = (args.vision_min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    return safe_resize(
        height,
        width,
        best_height,
        best_width,
        p,
        args.vision_downsample_ratio,
        args.vision_max_n_token,
    )


def image_token_types(n_llm_h: int, n_llm_w: int) -> torch.Tensor:
    """Default layout: the aligner grid in reading order, one IMAGE_NEW_LINE per row."""
    types = [IMAGE_START]
    types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
    types.append(IMAGE_END)
    return torch.tensor(types, dtype=torch.int64)


def prepare_image(image, args, start: int = 0) -> ImageInput:
    """Convert a decoded image to the released model's normalized, padded patch grid."""
    from PIL import ImageOps

    image = image.convert("RGB")
    p = args.vision_patch_size
    nh, nw, height, width = plan_image_grid(image.width, image.height, args)
    if (
        args.vision_max_wh_ratio is not None
        and image.width >= args.vision_max_wh_ratio * image.height
    ):
        image = image.resize((width, height))
    else:
        image = ImageOps.pad(image, (width, height), color=(127, 127, 127))
    x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255
    x = (x - 0.5) / 0.5
    h, w = height // p, width // p
    patches = x.reshape(3, h, p, w, p).permute(1, 3, 0, 2, 4).reshape(h * w, 3, p, p)
    return ImageInput(start, patches, h, w, image_token_types(nh, nw))
