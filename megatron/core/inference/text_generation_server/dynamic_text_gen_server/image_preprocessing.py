# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Image preprocessing for multimodal inference servers.

Shared between vlm_server.py and the coordinator/engine image dispatch in
run_dynamic_text_generation_server.py. Lives in core/inference so the engine
can import it without circular dependencies.
"""

import io
import math
from typing import Optional

import torch

from megatron.core.inference.config import ImageProcessingConfig
from megatron.core.models.vision.encoder_registry import REGISTRY as _ENCODER_REGISTRY


def _resolve_pixel_stats(vision_model_type: str):
    """Return (pixel_mean, pixel_std) for a vision encoder.

    Reads from the canonical encoder registry so training and inference share
    one source of truth. Falls back to CLIP-style stats for unknown encoders
    (matching the registry's own EncoderSpec defaults).
    """
    spec = _ENCODER_REGISTRY.get(vision_model_type)
    if spec is not None:
        return list(spec.pixel_mean), list(spec.pixel_std)
    # Fall back to CLIP defaults pulled from the registry's dataclass field
    # defaults rather than instantiating (which would need the four required
    # geometry fields) or copying locally.
    from megatron.core.models.vision.encoder_registry import EncoderSpec

    fields = EncoderSpec.__dataclass_fields__
    return list(fields["pixel_mean"].default), list(fields["pixel_std"].default)


def dynamic_res_preprocess(
    image,
    min_patches=1,
    max_patches=128,
    res_step=16,
    factor_max=1.0,
    pixel_shuffle=False,
    spatial_merge_size=1,
):
    """Resize image to fit within [min_patches, max_patches] preserving aspect ratio.

    For pixel_shuffle, patch grid dimensions are rounded to even numbers for
    compatibility.

    NOTE: Training uses ``DynamicResolutionImageTilingStrategy._process_single``
    (in megatron.energon.task_encoder.multimodal.image_tiling) as the canonical
    resize. The math here is intentionally a subset of that strategy and could
    drift if energon's implementation changes (e.g. ``min_side`` floor, tiling
    augmentation). For full parity, inference should call into the energon
    strategy directly — TODO once we have a clean way to import it that doesn't
    require energon at engine-drain time.
    """
    orig_width, orig_height = image.size

    # Use math.ceil, not round(x + 0.5) — the latter is banker's rounding and
    # produces off-by-one, non-monotonic patch counts on exactly-aligned sides.
    closest_patch_height = math.ceil(orig_height / res_step)
    closest_patch_width = math.ceil(orig_width / res_step)
    patches = closest_patch_height * closest_patch_width

    factor = min(math.sqrt(max_patches / patches), factor_max)
    target_patch_height = math.floor(factor * closest_patch_height)
    target_patch_width = math.floor(factor * closest_patch_width)

    if target_patch_height * target_patch_width < min_patches:
        up_factor = math.sqrt(min_patches / max(target_patch_height * target_patch_width, 1))
        target_patch_height = math.ceil(up_factor * target_patch_height)
        target_patch_width = math.ceil(up_factor * target_patch_width)

    grid_multiple = max(2 if pixel_shuffle else 1, spatial_merge_size)
    if grid_multiple > 1:
        if target_patch_height % grid_multiple:
            increase = grid_multiple - target_patch_height % grid_multiple
            if (target_patch_height + increase) * target_patch_width <= max_patches:
                target_patch_height += increase
            else:
                target_patch_height -= target_patch_height % grid_multiple
        if target_patch_width % grid_multiple:
            increase = grid_multiple - target_patch_width % grid_multiple
            if target_patch_height * (target_patch_width + increase) <= max_patches:
                target_patch_width += increase
            else:
                target_patch_width -= target_patch_width % grid_multiple

        target_patch_height = max(grid_multiple, target_patch_height)
        target_patch_width = max(grid_multiple, target_patch_width)

    assert target_patch_height * target_patch_width <= max_patches

    resized_img = image.resize((target_patch_width * res_step, target_patch_height * res_step))
    return resized_img


def preprocess_image_bytes(
    image_bytes: bytes,
    config: ImageProcessingConfig,
    target_hw=None,
    device: Optional[torch.device] = None,
) -> tuple:
    """Preprocess raw image bytes into tensors for dynamic-resolution inference.

    Args:
        image_bytes: Raw image file bytes (e.g. JPEG/PNG).
        config: Image preprocessing configuration.
        target_hw: Optional (H, W) tuple in pixels. If given, resize to exactly
            this size instead of running dynamic_res_preprocess. Used to keep
            all images in a multi-image request at the same patch dimensions.

    Returns:
        (imgs, imgs_sizes) tensors on CUDA.
        imgs shape: [1, num_patches, C*patch_dim*patch_dim]
        imgs_sizes shape: [1, 2] with [H, W] in pixels.
    """
    from PIL import Image
    from torchvision import transforms as T

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    patch_dim = config.patch_dim

    if target_hw is not None:
        target_h, target_w = target_hw
        img = img.resize((target_w, target_h))
    else:
        img = dynamic_res_preprocess(
            img,
            min_patches=config.dynamic_resolution_min_patches,
            max_patches=config.dynamic_resolution_max_patches,
            res_step=patch_dim,
            pixel_shuffle=config.pixel_shuffle,
            spatial_merge_size=config.spatial_merge_size,
        )

    vision_type = config.vision_model_type
    pixel_mean = config.pixel_mean
    pixel_std = config.pixel_std
    if pixel_mean is None or pixel_std is None:
        pixel_mean, pixel_std = _resolve_pixel_stats(vision_type)

    transform = T.Compose([T.ToTensor(), T.Normalize(mean=pixel_mean, std=pixel_std)])

    img_tensor = transform(img)  # [C, H, W]
    C, H, W = img_tensor.shape

    py, px = H // patch_dim, W // patch_dim
    patches = img_tensor.reshape(C, py, patch_dim, px, patch_dim)
    patches = patches.permute(1, 3, 0, 2, 4).contiguous()
    patches = patches.reshape(py * px, C * patch_dim * patch_dim)

    images = patches.unsqueeze(0)
    imgs_sizes = torch.tensor([[H, W]], dtype=torch.int32)

    if device is not None:
        return images.to(device), imgs_sizes.to(device)
    return images, imgs_sizes


def preprocess_image_bytes_list(
    image_bytes_list,
    config: ImageProcessingConfig,
    device: Optional[torch.device] = None,
) -> dict:
    """Preprocess a list of raw image bytes into engine.add_request image kwargs.

    Selects the dynamic-resolution or tiling path from the inference config.
    Each image is preprocessed independently so its aspect ratio is preserved.

    Args:
        image_bytes_list: List of raw image bytes (one entry per image).
        config: Image preprocessing configuration.
        device: Optional target device for the returned tensors. If None,
            tensors are returned on CPU and the caller is responsible for
            transfer.

    Returns:
        dict suitable for ``**kwargs`` to ``DynamicInferenceEngine.add_request``.
    """
    if not image_bytes_list:
        return {}

    dynamic_res = config.dynamic_resolution and not config.use_tiling

    if dynamic_res:
        # Preprocess each image independently so its aspect ratio is preserved.
        # Downstream (llava_model._preprocess_data / vision encoder pack) handles
        # per-image cu_seqlens, so ragged patch counts are fine.
        all_imgs, all_sizes = [], []
        for image_bytes in image_bytes_list:
            imgs, imgs_sizes = preprocess_image_bytes(image_bytes, config, device=device)
            all_imgs.append(imgs)
            all_sizes.append(imgs_sizes)
        imgs = torch.cat(all_imgs, dim=1) if len(all_imgs) > 1 else all_imgs[0]
        imgs_sizes = torch.cat(all_sizes, dim=0) if len(all_sizes) > 1 else all_sizes[0]
        return {"imgs": imgs, "imgs_sizes": imgs_sizes}

    all_imgs, all_num_tiles = [], []
    for image_bytes in image_bytes_list:
        imgs, num_tiles = preprocess_image_bytes_tiled(image_bytes, config, device=device)
        all_imgs.append(imgs)
        all_num_tiles.append(num_tiles)
    imgs = torch.cat(all_imgs, dim=0) if len(all_imgs) > 1 else all_imgs[0]
    num_tiles = torch.cat(all_num_tiles, dim=0) if len(all_num_tiles) > 1 else all_num_tiles[0]
    return {
        "imgs": imgs,
        "num_tiles": num_tiles,
        "num_img_embeddings_per_tile": config.num_img_embeddings_per_tile,
    }


def preprocess_image_bytes_tiled(
    image_bytes: bytes,
    config: ImageProcessingConfig,
    device: Optional[torch.device] = None,
) -> tuple:
    """Preprocess raw image bytes into tiled tensors for static-resolution inference.

    Returns:
        (imgs, num_tiles) where imgs is [num_tiles, C, H, W] and num_tiles is a [1] int tensor.

    Note: depends on examples/multimodal/image_processing.py being importable.
    Callers that use the tiling path must ensure that path is on sys.path.
    """
    from PIL import Image

    from examples.multimodal.image_processing import ImageTransform

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if config.img_h is None or config.img_w is None:
        raise ValueError("Tiled image preprocessing requires img_h and img_w.")
    transform = ImageTransform(input_size=config.img_h, vision_model_type=config.vision_model_type)
    imgs_list = transform(
        img,
        config.img_h,
        config.img_w,
        use_tiling=config.use_tiling,
        max_num_tiles=config.max_num_tiles,
        use_thumbnail=config.use_thumbnail,
    )

    imgs = torch.stack(imgs_list)
    num_tiles = torch.tensor([len(imgs_list)], dtype=torch.int)
    if device is not None:
        return imgs.to(device), num_tiles.to(device)
    return imgs, num_tiles
