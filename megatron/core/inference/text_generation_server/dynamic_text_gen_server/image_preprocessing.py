# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Image preprocessing for multimodal inference servers.

Shared between vlm_server.py and the coordinator/engine image dispatch in
run_dynamic_text_generation_server.py. Lives in core/inference so the engine
can import it without circular dependencies.
"""

import io
import json
import logging
import math
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from megatron.core.inference.config import ImageProcessingConfig, VideoProcessingConfig
from megatron.core.models.vision.encoder_registry import REGISTRY as _ENCODER_REGISTRY

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _warn_multi_image_patch_budget() -> None:
    """Warn once per process about Nemotron's counter-intuitive image budget."""
    logger.warning(
        "Nemotron VL 3.5 may use more patches per image when given more images. "
        "This is counter-intuitive and likely a processor bug, but users can set "
        "dynamic_resolution_max_patches to cap the patches used per image."
    )


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
    """Load frames and timing metadata from a configured frame-sequence manifest."""
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
    metadata = manifest.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("Frame-sequence manifest metadata must be an object.")
    frame_indices = metadata.get("frames_indices", list(range(len(frame_paths))))
    fps = metadata.get("fps", 1.0)
    if (
        not isinstance(frame_indices, list)
        or len(frame_indices) != len(frame_paths)
        or any(type(index) is not int or index < 0 for index in frame_indices)
    ):
        raise ValueError(
            "Frame-sequence manifest metadata.frames_indices must contain one "
            "non-negative integer per frame."
        )
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps):
        raise ValueError("Frame-sequence manifest metadata.fps must be a finite number.")
    fps = float(fps)
    if fps <= 0:
        raise ValueError("Frame-sequence manifest metadata.fps must be positive.")

    frames = []
    for frame_path in frame_paths:
        resolved = Path(frame_path).expanduser().resolve()
        with Image.open(resolved) as image:
            frames.append(image.convert("RGB").copy())
    return frames, frame_indices, fps


def dynamic_res_preprocess(
    image,
    min_patches=1,
    max_patches=128,
    res_step=16,
    factor_max=1.0,
    pixel_shuffle=False,
    spatial_merge_size=1,
    video_maintain_aspect_ratio=None,
    rounding_mode="ceil",
):
    """Resize image to fit within [min_patches, max_patches] preserving aspect ratio.

    When ``video_maintain_aspect_ratio`` is not None, use a fixed per-frame
    ``max_patches`` budget. True preserves the source aspect ratio; False uses
    a square grid. Images leave this as None and use adaptive resizing.
    For pixel_shuffle, patch grid dimensions are rounded to even numbers for
    compatibility.

    ``rounding_mode`` makes the source processor's grid contract explicit.
    The default preserves MCore/Energon ceil behavior; ``round_plus_half``
    reproduces HF processors that intentionally use Python's half-to-even
    ``round(x + 0.5)`` rule.
    """
    orig_width, orig_height = image.size

    if video_maintain_aspect_ratio is not None:
        if video_maintain_aspect_ratio:
            aspect_ratio = orig_width / max(orig_height, 1)
            target_patch_height = max(1, round(math.sqrt(max_patches / aspect_ratio)))
            target_patch_width = max(1, round(math.sqrt(max_patches * aspect_ratio)))
            # Preserve the former _video_target_resolution behavior exactly.
            grid_multiple = 2 if pixel_shuffle else 1
        else:
            target_patch_height = target_patch_width = max(1, math.isqrt(max_patches))
            grid_multiple = max(2 if pixel_shuffle else 1, spatial_merge_size)
        if grid_multiple > 1:
            height_remainder = target_patch_height % grid_multiple
            width_remainder = target_patch_width % grid_multiple
            height_up = target_patch_height + (
                grid_multiple - height_remainder if height_remainder else 0
            )
            width_up = target_patch_width + (
                grid_multiple - width_remainder if width_remainder else 0
            )
            if height_up * width_up <= max_patches:
                target_patch_height, target_patch_width = height_up, width_up
            else:
                target_patch_height = max(grid_multiple, target_patch_height - height_remainder)
                target_patch_width = max(grid_multiple, target_patch_width - width_remainder)
    else:
        grid_multiple = max(2 if pixel_shuffle else 1, spatial_merge_size)
        if rounding_mode == "ceil":
            closest_patch_height = math.ceil(orig_height / res_step)
            closest_patch_width = math.ceil(orig_width / res_step)
        elif rounding_mode == "round_plus_half":
            # Some HF processors intentionally use Python's half-to-even
            # ``round(x + 0.5)`` contract. Preserve it exactly: replacing this
            # with ceil changes the projected-token count for aligned odd grids.
            closest_patch_height = round(orig_height / res_step + 0.5)
            closest_patch_width = round(orig_width / res_step + 0.5)
        else:
            raise ValueError(
                "rounding_mode must be 'ceil' or 'round_plus_half', got " f"{rounding_mode!r}."
            )
        patches = closest_patch_height * closest_patch_width

        factor = min(math.sqrt(max_patches / patches), factor_max)
        target_patch_height = math.floor(factor * closest_patch_height)
        target_patch_width = math.floor(factor * closest_patch_width)

        should_enforce_minimum = rounding_mode != "round_plus_half" or max_patches > min_patches
        if should_enforce_minimum and target_patch_height * target_patch_width < min_patches:
            up_factor = math.sqrt(min_patches / max(target_patch_height * target_patch_width, 1))
            target_patch_height = math.ceil(up_factor * target_patch_height)
            target_patch_width = math.ceil(up_factor * target_patch_width)

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
    source_img = img

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
            rounding_mode=config.dynamic_resolution_rounding_mode,
        )

    vision_type = config.vision_model_type
    pixel_mean = config.pixel_mean
    pixel_std = config.pixel_std
    if pixel_mean is None or pixel_std is None:
        pixel_mean, pixel_std = _resolve_pixel_stats(vision_type)

    if config.dynamic_resolution_resize_mode == "pil":
        transform = T.Compose([T.ToTensor(), T.Normalize(mean=pixel_mean, std=pixel_std)])
        img_tensor = transform(img)  # [C, H, W]
    elif config.dynamic_resolution_resize_mode == "torch_bicubic_antialias":
        import torch.nn.functional as F

        target_hw = (img.height, img.width)
        source_array = np.asarray(source_img, dtype=np.uint8)
        img_tensor = (
            torch.from_numpy(source_array).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
        )
        if img_tensor.shape[-2:] != target_hw:
            img_tensor = F.interpolate(
                img_tensor, size=target_hw, mode="bicubic", align_corners=False, antialias=True
            )
        mean = torch.tensor(pixel_mean, dtype=img_tensor.dtype).view(1, -1, 1, 1)
        std = torch.tensor(pixel_std, dtype=img_tensor.dtype).view(1, -1, 1, 1)
        img_tensor = ((img_tensor / 255.0 - mean) / std).squeeze(0)
    else:
        raise ValueError(
            "dynamic_resolution_resize_mode must be 'pil' or "
            f"'torch_bicubic_antialias', got {config.dynamic_resolution_resize_mode!r}."
        )
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

    if config.dynamic_resolution_model_length is not None:
        model_length = int(config.dynamic_resolution_model_length)
        if model_length <= 4:
            raise ValueError("dynamic_resolution_model_length must be greater than 4.")
        merge_size = max(int(config.spatial_merge_size), 1)
        model_patch_budget = (model_length - 4) * (merge_size * merge_size)
        # NOTE(@cspades): Oddly, Nemotron VL 3.5 uses more patches per image when given more images.
        # This is counter-intuitive and likely a bug:
        # https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/blob/main/image_processing.py#L98
        # Until fixed, manually control patches per image using dynamic_resolution_max_patches.
        if len(image_bytes_list) > 1:
            _warn_multi_image_patch_budget()
        request_patch_budget = max(
            model_patch_budget, int(config.dynamic_resolution_min_patches) * len(image_bytes_list)
        )
        configured_max = int(config.dynamic_resolution_max_patches)
        if configured_max > 0:
            request_patch_budget = min(configured_max, request_patch_budget)
        config = replace(
            config,
            dynamic_resolution_max_patches=max(
                int(config.dynamic_resolution_min_patches), request_patch_budget
            ),
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


def _video_sample_indices(total_frames: int, config: VideoProcessingConfig) -> list[int]:
    """Return the existing uniformly spaced sample indices for a video."""
    import numpy as np

    if total_frames <= 0:
        return []
    if total_frames < config.temporal_patch_size:
        sample_count = min(config.num_frames, config.temporal_patch_size)
    else:
        sample_count = min(config.num_frames, total_frames)
    if config.temporal_patch_size > 1 and sample_count % config.temporal_patch_size:
        rounded_down = (sample_count // config.temporal_patch_size) * config.temporal_patch_size
        sample_count = (
            rounded_down if rounded_down > 0 else min(config.temporal_patch_size, total_frames)
        )
    return np.rint(np.linspace(0, total_frames - 1, num=sample_count)).astype(np.int64).tolist()


def _decode_sampled_video_frames(encoded_video: bytes, config: VideoProcessingConfig):
    """Decode sampled frames once and retain their source indices and FPS."""
    import av

    def decode_selected(total_frames: int):
        sample_indices = _video_sample_indices(total_frames, config)
        wanted = set(sample_indices)
        sampled_frames_by_index = {}
        decoded_count = 0
        with av.open(io.BytesIO(encoded_video)) as container:
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                decoded_count = index + 1
                if index in wanted:
                    sampled_frames_by_index[index] = frame.to_image().convert("RGB")
        sampled_frames = [
            sampled_frames_by_index[index]
            for index in sample_indices
            if index in sampled_frames_by_index
        ]
        return sampled_frames, decoded_count, sample_indices

    # Prefer container metadata so indexed streams need only one decode pass.
    with av.open(io.BytesIO(encoded_video)) as container:
        stream = container.streams.video[0]
        declared_frames = int(stream.frames or 0)
        average_rate = stream.average_rate
        fps = float(average_rate) if average_rate is not None else 0.0

    if declared_frames > 0:
        sampled_frames, decoded_count, sample_indices = decode_selected(declared_frames)
        if decoded_count == declared_frames:
            return sampled_frames, sample_indices, fps
        # Some containers report an inaccurate frame count. Redecode using the
        # observed count to preserve the original exact uniform indices.
        if decoded_count > 0:
            sampled_frames, _, sample_indices = decode_selected(decoded_count)
        return sampled_frames, sample_indices, fps

    # Unindexed streams need a count-only pass before exact uniform indices can
    # be selected. No AVFrame or RGB image is retained during this pass.
    total_frames = 0
    with av.open(io.BytesIO(encoded_video)) as container:
        stream = container.streams.video[0]
        for total_frames, _ in enumerate(container.decode(stream), start=1):
            pass
    if total_frames == 0:
        return [], [], fps
    sampled_frames, _, sample_indices = decode_selected(total_frames)
    return sampled_frames, sample_indices, fps


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

    def decode_frames(encoded_video):
        manifest_result = _load_frame_sequence_manifest(encoded_video, config.frame_manifest_magic)
        if manifest_result is not None:
            frames, frame_indices, fps = manifest_result
            return frames, frame_indices, fps, True

        frames, frame_indices, fps = _decode_sampled_video_frames(encoded_video, config)
        return frames, frame_indices, fps, False

    packed_videos = []
    packed_sizes = []
    frame_counts = []
    video_frame_indices = []
    video_fps = []

    for encoded_video in video_bytes_list:
        if not isinstance(encoded_video, (bytes, bytearray)):
            raise TypeError("video payloads must contain only bytes.")
        frames, frame_indices, fps, is_frame_sequence = decode_frames(bytes(encoded_video))
        if not frames:
            raise ValueError("Decoded video contains no frames.")

        if is_frame_sequence and len(frames) != config.num_frames:
            raise ValueError(
                "Frame-sequence count must match the configured count: "
                f"{len(frames)} != {config.num_frames}."
            )
        sampled_frames = frames
        sample_count = len(sampled_frames)
        if len(frame_indices) != sample_count:
            raise ValueError(
                "Video timing metadata must contain one frame index per sampled frame."
            )

        frame_tensors = []
        frame_sizes = []
        reference_image = dynamic_res_preprocess(
            sampled_frames[0],
            min_patches=config.image_config.dynamic_resolution_min_patches,
            max_patches=config.image_config.dynamic_resolution_max_patches,
            res_step=config.image_config.patch_dim,
            pixel_shuffle=config.image_config.pixel_shuffle,
            spatial_merge_size=config.image_config.spatial_merge_size,
            video_maintain_aspect_ratio=config.video_maintain_aspect_ratio,
        )
        reference_hw = (reference_image.height, reference_image.width)
        for frame in sampled_frames:
            imgs, imgs_sizes = preprocess_image(
                frame, config.image_config, target_hw=reference_hw, device=device
            )
            frame_tensors.append(imgs)
            frame_sizes.append(imgs_sizes)

        packed_videos.append(torch.cat(frame_tensors, dim=1))
        packed_sizes.append(torch.cat(frame_sizes, dim=0))
        frame_counts.append(sample_count)
        video_frame_indices.append(frame_indices)
        video_fps.append(fps)

    return {
        "imgs": torch.cat(packed_videos, dim=1),
        "imgs_sizes": torch.cat(packed_sizes, dim=0),
        "num_frames": torch.tensor(frame_counts, dtype=torch.int32, device=packed_videos[0].device),
        "video_frame_indices": video_frame_indices,
        "video_fps": video_fps,
    }
