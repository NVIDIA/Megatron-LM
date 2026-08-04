# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Image preprocessing for VLM inference servers.

Shared between vlm_server.py and the coordinator/engine VLM dispatch in
run_dynamic_text_generation_server.py. Lives in core/inference so the engine
can import it without circular dependencies.
"""

import io
import math

import torch

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
    # Fall back to CLIP defaults pulled from the registry rather than a local
    # copy, so changes to the canonical constants flow through.
    from megatron.core.models.vision.encoder_registry import EncoderSpec
    default_spec = EncoderSpec()
    return list(default_spec.pixel_mean), list(default_spec.pixel_std)


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

    closest_patch_height = round(orig_height / res_step + 0.5)
    closest_patch_width = round(orig_width / res_step + 0.5)
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


def preprocess_image_bytes(image_bytes: bytes, args, target_hw=None) -> tuple:
    """Preprocess raw image bytes into tensors for dynamic-resolution VLM inference.

    Args:
        image_bytes: Raw image file bytes (e.g. JPEG/PNG).
        args: Megatron args (must have patch_dim and dynamic_resolution_* attrs).
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

    patch_dim = args.patch_dim
    pixel_shuffle = getattr(args, 'pixel_shuffle', False)
    spatial_merge_size = getattr(args, 'spatial_merge_size', 1)
    min_patches = getattr(args, 'dynamic_resolution_min_patches', 1)
    max_patches = getattr(args, 'dynamic_resolution_max_patches', 128)

    if target_hw is not None:
        target_h, target_w = target_hw
        img = img.resize((target_w, target_h))
    else:
        img = dynamic_res_preprocess(
            img,
            min_patches=min_patches,
            max_patches=max_patches,
            res_step=patch_dim,
            pixel_shuffle=pixel_shuffle,
            spatial_merge_size=spatial_merge_size,
        )

    vision_type = getattr(args, 'vision_model_type', 'radio')
    pixel_mean = getattr(args, 'pixel_mean', None)
    pixel_std = getattr(args, 'pixel_std', None)
    if pixel_mean is None or pixel_std is None:
        pixel_mean, pixel_std = _resolve_pixel_stats(vision_type)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=pixel_mean, std=pixel_std),
    ])

    img_tensor = transform(img)  # [C, H, W]
    C, H, W = img_tensor.shape

    py, px = H // patch_dim, W // patch_dim
    patches = img_tensor.reshape(C, py, patch_dim, px, patch_dim)
    patches = patches.permute(1, 3, 0, 2, 4).contiguous()
    patches = patches.reshape(py * px, C * patch_dim * patch_dim)

    images = patches.unsqueeze(0)
    imgs_sizes = torch.tensor([[H, W]], dtype=torch.int32)

    return images.cuda(), imgs_sizes.cuda()


def preprocess_image_bytes_list(image_bytes_list, args) -> dict:
    """Preprocess a list of raw image bytes into engine.add_request VLM kwargs.

    Selects the dynamic-resolution or tiling path based on args.dynamic_resolution
    and args.use_tiling. Within a single request, dynamic-resolution images are
    resized to the first image's H/W to keep all images at matching patch counts.

    Args:
        image_bytes_list: List of raw image bytes (one entry per image).
        args: Megatron args.

    Returns:
        dict suitable for ``**kwargs`` to ``DynamicInferenceEngine.add_request``.
    """
    if not image_bytes_list:
        return {}

    dynamic_res = (
        getattr(args, 'dynamic_resolution', False)
        and not getattr(args, 'use_tiling', False)
    )

    if dynamic_res:
        all_imgs, all_sizes = [], []
        ref_hw = None
        for image_bytes in image_bytes_list:
            imgs, imgs_sizes = preprocess_image_bytes(image_bytes, args, target_hw=ref_hw)
            if ref_hw is None:
                ref_hw = (imgs_sizes[0][0].item(), imgs_sizes[0][1].item())
            all_imgs.append(imgs)
            all_sizes.append(imgs_sizes)
        imgs = torch.cat(all_imgs, dim=1) if len(all_imgs) > 1 else all_imgs[0]
        imgs_sizes = torch.cat(all_sizes, dim=0) if len(all_sizes) > 1 else all_sizes[0]
        return {"imgs": imgs, "imgs_sizes": imgs_sizes}

    all_imgs, all_num_tiles = [], []
    for image_bytes in image_bytes_list:
        imgs, num_tiles = preprocess_image_bytes_tiled(image_bytes, args)
        all_imgs.append(imgs)
        all_num_tiles.append(num_tiles)
    imgs = torch.cat(all_imgs, dim=0) if len(all_imgs) > 1 else all_imgs[0]
    num_tiles = torch.cat(all_num_tiles, dim=0) if len(all_num_tiles) > 1 else all_num_tiles[0]
    return {
        "imgs": imgs,
        "num_tiles": num_tiles,
        "num_img_embeddings_per_tile": getattr(args, 'num_img_embeddings_per_tile', 0),
    }


def preprocess_image_bytes_tiled(image_bytes: bytes, args) -> tuple:
    """Preprocess raw image bytes into tiled tensors for static-resolution VLM inference.

    Returns:
        (imgs, num_tiles) where imgs is [num_tiles, C, H, W] and num_tiles is a [1] int tensor.

    Note: depends on examples/multimodal/image_processing.py being importable.
    Callers that use the tiling path must ensure that path is on sys.path.
    """
    from PIL import Image

    from examples.multimodal.image_processing import ImageTransform

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    transform = ImageTransform(input_size=args.img_h, vision_model_type=args.vision_model_type)
    imgs_list = transform(
        img, args.img_h, args.img_w,
        use_tiling=args.use_tiling,
        max_num_tiles=args.max_num_tiles,
        use_thumbnail=args.use_thumbnail,
    )

    imgs = torch.stack(imgs_list)
    num_tiles = torch.tensor([len(imgs_list)], dtype=torch.int)
    return imgs.cuda(), num_tiles.cuda()
