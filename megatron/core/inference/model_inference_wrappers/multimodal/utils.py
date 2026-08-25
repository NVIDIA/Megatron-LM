# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch

def dynamic_media_embedding_counts(
    imgs_sizes: torch.Tensor,
    patch_dim: int,
    *,
    pixel_shuffle: bool,
    pixel_shuffle_size: int = 2,
    spatial_merge_size: int = 1,
) -> list[int]:
    """Return projected token counts for dynamic-resolution media frames."""
    if patch_dim <= 0:
        raise ValueError("patch_dim must be greater than 0.")
    if imgs_sizes.ndim != 2 or imgs_sizes.shape[1] != 2:
        raise ValueError(f"imgs_sizes must have shape [N, 2], got {tuple(imgs_sizes.shape)}.")
    if pixel_shuffle and pixel_shuffle_size <= 0:
        raise ValueError("pixel_shuffle_size must be greater than 0.")
    if spatial_merge_size <= 0:
        raise ValueError("spatial_merge_size must be greater than 0.")

    counts = []
    for height, width in imgs_sizes.tolist():
        if height % patch_dim or width % patch_dim:
            raise ValueError("Media dimensions must be divisible by patch_dim.")
        patch_height = height // patch_dim
        patch_width = width // patch_dim
        if pixel_shuffle and (
            patch_height % pixel_shuffle_size or patch_width % pixel_shuffle_size
        ):
            raise ValueError("Media patch grids must be divisible by pixel_shuffle_size.")
        count = patch_height * patch_width
        if pixel_shuffle:
            count //= pixel_shuffle_size**2
        merge_factor = spatial_merge_size**2
        if count % merge_factor:
            raise ValueError("Media token count must be divisible by the spatial merge factor.")
        counts.append(count // merge_factor)
    return counts


def dynamic_media_replacement_counts(
    frame_embedding_counts: list[int],
    *,
    num_frames,
    temporal_patch_size: int,
) -> list[int]:
    """Map per-frame counts to one compact placeholder per image or video."""
    if num_frames is None:
        return frame_embedding_counts

    if isinstance(num_frames, int):
        frame_groups = [num_frames]
    elif hasattr(num_frames, "tolist"):
        values = num_frames.tolist()
        frame_groups = (
            [int(values)] if not isinstance(values, list) else [int(value) for value in values]
        )
    else:
        frame_groups = [int(value) for value in num_frames]
    if any(value <= 0 for value in frame_groups):
        raise ValueError("num_frames entries must be positive.")
    if sum(frame_groups) != len(frame_embedding_counts):
        raise ValueError(
            "num_frames must partition imgs_sizes exactly: "
            f"sum(num_frames)={sum(frame_groups)}, "
            f"imgs_sizes={len(frame_embedding_counts)}."
        )
    if temporal_patch_size <= 0:
        raise ValueError("temporal_patch_size must be positive.")

    per_video_counts = []
    frame_offset = 0
    for frame_count in frame_groups:
        video_frame_counts = frame_embedding_counts[frame_offset : frame_offset + frame_count]
        tubelet_counts = video_frame_counts[::temporal_patch_size]
        per_video_counts.append(sum(tubelet_counts))
        frame_offset += frame_count

    return per_video_counts
