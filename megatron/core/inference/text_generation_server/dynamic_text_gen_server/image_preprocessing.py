# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Image preprocessing for multimodal inference servers.

Shared between vlm_server.py and the coordinator/engine image dispatch in
run_dynamic_text_generation_server.py. Lives in core/inference so the engine
can import it without circular dependencies.
"""

import io
import json
import math
from pathlib import Path
from typing import Optional

import torch

from megatron.core.inference.config import ImageProcessingConfig, VideoProcessingConfig
from megatron.core.models.vision.encoder_registry import REGISTRY as _ENCODER_REGISTRY


def _video_target_resolution(image, config: ImageProcessingConfig) -> tuple[int, int]:
    """Return a video frame target as ``(height, width)``."""
    target_patches = config.dynamic_resolution_max_patches
    aspect_ratio = image.width / max(image.height, 1)
    patch_height = max(1, round(math.sqrt(target_patches / aspect_ratio)))
    patch_width = max(1, round(math.sqrt(target_patches * aspect_ratio)))
    required_divisor = 2 if config.pixel_shuffle else 1
    if required_divisor > 1:
        height_remainder = patch_height % required_divisor
        width_remainder = patch_width % required_divisor
        height_up = patch_height + (required_divisor - height_remainder if height_remainder else 0)
        width_up = patch_width + (required_divisor - width_remainder if width_remainder else 0)
        if height_up * width_up <= target_patches:
            patch_height, patch_width = height_up, width_up
        else:
            patch_height = max(required_divisor, patch_height - height_remainder)
            patch_width = max(required_divisor, patch_width - width_remainder)
    return patch_height * config.patch_dim, patch_width * config.patch_dim


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


def _load_frame_sequence_manifest(payload: bytes, frame_manifest_magic: Optional[bytes]):
    """Load PIL images from a configured frame-sequence manifest."""
    if not frame_manifest_magic or not payload.startswith(frame_manifest_magic):
        return None

    from PIL import Image

    try:
        manifest = json.loads(payload[len(frame_manifest_magic) :])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid frame-sequence manifest JSON.") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Frame-sequence manifest must be a JSON object.")

    frame_paths = manifest.get("frame_paths")
    if (
        not isinstance(frame_paths, list)
        or not frame_paths
        or not all(isinstance(path, str) and path for path in frame_paths)
    ):
        raise ValueError("Frame-sequence manifest requires non-empty string frame_paths.")

    frames = []
    for frame_path in frame_paths:
        resolved = Path(frame_path).expanduser().resolve()
        with Image.open(resolved) as image:
            frames.append(image.convert("RGB").copy())
    return frames


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


def preprocess_image(
    image, config: ImageProcessingConfig, target_hw=None, device: Optional[torch.device] = None
) -> tuple:
    """Convert one PIL image into packed vision patches and its resized shape."""
    try:
        from torchvision import transforms as T
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for VLM image preprocessing. Install a "
            "torchvision build matching your torch version, or use the NGC "
            "PyTorch container that ships one."
        ) from exc

    img = image.convert("RGB")

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


def preprocess_image_bytes(
    image_bytes: bytes,
    config: ImageProcessingConfig,
    target_hw=None,
    device: Optional[torch.device] = None,
) -> tuple:
    """Decode image bytes and return packed vision patches and its resized shape."""
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as image:
        return preprocess_image(image, config, target_hw=target_hw, device=device)


def preprocess_image_bytes_list(
    image_bytes_list, config: ImageProcessingConfig, device: Optional[torch.device] = None
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

    if not dynamic_res:
        # Static tiling used to live here as ``preprocess_image_bytes_tiled``,
        # but it delegated to ``examples.multimodal.image_processing.ImageTransform``,
        # a bad dependency direction (``megatron/core`` importing from
        # ``examples/``). No in-tree caller currently hits this branch
        # (all supported encoders are on the dynamic-resolution path), so we
        # drop the tiling helper rather than move its 168-line dep into core.
        # Wire clients that need static tiling should preprocess bytes
        # themselves and submit a tensor payload
        # (``multi_modal_data['image'] = {'imgs': Tensor, 'num_tiles': Tensor,
        # 'num_img_embeddings_per_tile': int}``); the engine already accepts
        # that shape without touching examples.
        raise NotImplementedError(
            "Wire-side static-tiling preprocessing has moved out of "
            "``megatron/core``. Submit a preprocessed tensor payload as "
            "``multi_modal_data['image']`` "
            "({'imgs': Tensor, 'num_tiles': Tensor, "
            "'num_img_embeddings_per_tile': int}), or set "
            "``ImageProcessingConfig.dynamic_resolution=True`` to use the "
            "dynamic-resolution path that stays in-core."
        )

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


def preprocess_video_bytes_list(
    video_bytes_list, config: VideoProcessingConfig, device: Optional[torch.device] = None
) -> dict:
    """Decode videos and return packed dynamic-resolution engine inputs.

    Frames are sampled uniformly, resized using the same image preprocessing
    configuration as still images, and kept grouped through ``num_frames``.
    """
    if not video_bytes_list:
        return {}
    if config.num_frames <= 0:
        raise ValueError("VideoProcessingConfig.num_frames must be positive.")
    if config.temporal_patch_size <= 0:
        raise ValueError("VideoProcessingConfig.temporal_patch_size must be positive.")
    if not config.image_config.dynamic_resolution or config.image_config.use_tiling:
        raise NotImplementedError(
            "Raw video preprocessing currently requires dynamic-resolution, "
            "non-tiled vision inputs."
        )

    import numpy as np

    def decode_frames(encoded_video):
        frames = _load_frame_sequence_manifest(encoded_video, config.frame_manifest_magic)
        if frames is not None:
            return frames, True

        import av

        with av.open(io.BytesIO(encoded_video)) as container:
            return ([frame.to_image().convert("RGB") for frame in container.decode(video=0)], False)

    packed_videos = []
    packed_sizes = []
    frame_counts = []
    reference_hw = None

    for encoded_video in video_bytes_list:
        if not isinstance(encoded_video, (bytes, bytearray)):
            raise TypeError("video payloads must contain only bytes.")
        frames, is_frame_sequence = decode_frames(bytes(encoded_video))
        if not frames:
            raise ValueError("Decoded video contains no frames.")

        if is_frame_sequence:
            if len(frames) != config.num_frames:
                raise ValueError(
                    "Frame-sequence count must match the configured count: "
                    f"{len(frames)} != {config.num_frames}."
                )
            sampled_frames = frames
        else:
            sample_count = min(config.num_frames, len(frames))
            if config.temporal_patch_size > 1 and sample_count % config.temporal_patch_size:
                rounded_down = (
                    sample_count // config.temporal_patch_size
                ) * config.temporal_patch_size
                sample_count = (
                    rounded_down
                    if rounded_down > 0
                    else min(config.temporal_patch_size, len(frames))
                )
            sample_indices = (
                np.rint(np.linspace(0, len(frames) - 1, num=sample_count)).astype(np.int64).tolist()
            )
            sampled_frames = [frames[index] for index in sample_indices]
        sample_count = len(sampled_frames)

        frame_tensors = []
        frame_sizes = []
        if reference_hw is None:
            reference_hw = _video_target_resolution(sampled_frames[0], config.image_config)
        for frame in sampled_frames:
            imgs, imgs_sizes = preprocess_image(
                frame, config.image_config, target_hw=reference_hw, device=device
            )
            frame_tensors.append(imgs)
            frame_sizes.append(imgs_sizes)

        packed_videos.append(torch.cat(frame_tensors, dim=1))
        packed_sizes.append(torch.cat(frame_sizes, dim=0))
        frame_counts.append(sample_count)

    return {
        "imgs": torch.cat(packed_videos, dim=1),
        "imgs_sizes": torch.cat(packed_sizes, dim=0),
        "num_frames": torch.tensor(frame_counts, dtype=torch.int32, device=packed_videos[0].device),
    }
