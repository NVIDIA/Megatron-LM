# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Host-side video preprocessing for Nemotron Omni.

Unlike images, every frame of a video is resized to one *common* target, because the tubelet
grouping concatenates frames along the channel axis and pixel shuffle later views the whole
video as a `[T, H, W, hidden]` block. Per-frame resolution would break both.

Note on provenance: the reference implementation's aspect-preserving resize claims in a comment
to mirror `examples/multimodal/image_processing.py`, but that file's
`find_closest_area_weighted_aspect_ratio` selects a *tile count* for fixed-tile processing and
is a different computation entirely. The logic below follows the reference implementation's
actual code -- distribute the target patch area by aspect ratio, then snap both axes to a
multiple of 2 -- since that is what produced the checkpoint's behaviour.
"""

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch

from megatron.core.inference.multimodal.nemotron_omni.image_processor import (
    bicubic_resize_and_normalize,
)


@dataclass
class ProcessedVideo:
    """One video resized to a common frame geometry."""

    pixel_values: torch.Tensor
    """`[num_frames, 3, height, width]`, normalized."""

    frame_indices: List[int]
    """Original indices of the sampled frames, used to render timestamps."""

    frame_duration_ms: Optional[int]
    """Integer ms per frame. Integer by contract -- see `prompt.py`."""

    tokens_per_frame: int
    """Post-pixel-shuffle token count for one frame's grid."""

    @property
    def num_frames(self) -> int:
        """Sampled frame count."""
        return self.pixel_values.shape[0]

    @property
    def size_hw(self) -> Tuple[int, int]:
        """Pixel `(height, width)` shared by every frame."""
        return (self.pixel_values.shape[-2], self.pixel_values.shape[-1])


def compute_aspect_preserving_size(
    orig_w: int, orig_h: int, target_num_patches: int, patch_size: int
) -> Tuple[int, int]:
    """Distribute a patch-area budget across the two axes at the source aspect ratio.

    Snaps both axes to a multiple of 2 (required by the 2x2 pixel-shuffle fold), preferring to
    round *up* when the enlarged grid still fits the budget.

    Args:
        orig_w (int): Source width in pixels.
        orig_h (int): Source height in pixels.
        target_num_patches (int): Patch-area budget.
        patch_size (int): Patch side in pixels.

    Return:
        (Tuple[int, int]) Target `(width, height)` in pixels.
    """
    aspect_wh = orig_w / max(orig_h, 1)
    patch_h = max(round(math.sqrt(target_num_patches / aspect_wh)), 1)
    patch_w = max(round(math.sqrt(target_num_patches * aspect_wh)), 1)

    divisor = 2
    rem_h = patch_h % divisor
    rem_w = patch_w % divisor
    up_h = patch_h + (divisor - rem_h if rem_h else 0)
    up_w = patch_w + (divisor - rem_w if rem_w else 0)
    if up_h * up_w <= target_num_patches:
        patch_h, patch_w = up_h, up_w
    else:
        patch_h = max(divisor, patch_h - rem_h)
        patch_w = max(divisor, patch_w - rem_w)

    return patch_w * patch_size, patch_h * patch_size


def compute_video_geometry(
    orig_w: int, orig_h: int, target_num_patches: int, maintain_aspect_ratio: bool, patch_size: int
) -> Tuple[int, int, int]:
    """Target frame size and the token count it produces.

    Args:
        orig_w (int): Source width in pixels.
        orig_h (int): Source height in pixels.
        target_num_patches (int): Patch-area budget per frame.
        maintain_aspect_ratio (bool): Whether to preserve the source aspect ratio.
        patch_size (int): Patch side in pixels.

    Return:
        (Tuple[int, int, int]) `(width, height, tokens_per_frame)`.
    """
    if maintain_aspect_ratio:
        target_w, target_h = compute_aspect_preserving_size(
            orig_w, orig_h, target_num_patches, patch_size
        )
    else:
        side = int(math.sqrt(target_num_patches))
        side = max(2, (side // 2) * 2)
        target_w = target_h = side * patch_size

    # Two independent halvings, floored separately -- matching the reference's
    # `int(h * ratio) * int(w * ratio)` rather than halving the product.
    tokens_per_frame = (target_h // patch_size // 2) * (target_w // patch_size // 2)
    return target_w, target_h, tokens_per_frame


def _frames_to_nfhwc(video: Any) -> Tuple[torch.Tensor, int, int]:
    """Coerce a video into a `[T, H, W, 3]` uint8 tensor."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("Pillow and numpy are required for video preprocessing") from exc

    if isinstance(video, (list, tuple)):
        frames = []
        for frame in video:
            if isinstance(frame, Image.Image):
                frame = np.asarray(
                    frame if frame.mode == "RGB" else frame.convert("RGB"), dtype=np.uint8
                )
            elif torch.is_tensor(frame):
                frame = frame.to(torch.uint8).cpu().numpy()
            frames.append(np.asarray(frame, dtype=np.uint8))
        array = np.stack(frames, axis=0)
    elif torch.is_tensor(video):
        array = video.to(torch.uint8).cpu().numpy()
    else:
        array = np.asarray(video, dtype=np.uint8)

    assert array.ndim == 4 and array.shape[-1] == 3, f"expected THWC RGB, got {array.shape}"
    return torch.from_numpy(array), array.shape[2], array.shape[1]


class NemotronOmniVideoProcessor:
    """Resizes and normalizes sampled video frames to a common geometry."""

    def __init__(
        self,
        *,
        patch_size: int,
        target_num_patches: int,
        maintain_aspect_ratio: bool,
        norm_mean: Sequence[float],
        norm_std: Sequence[float],
    ) -> None:
        self.patch_size = patch_size
        self.target_num_patches = target_num_patches
        self.maintain_aspect_ratio = maintain_aspect_ratio
        self.norm_mean = torch.tensor(norm_mean).reshape(3, 1, 1)
        self.norm_std = torch.tensor(norm_std).reshape(3, 1, 1)

    def process(
        self,
        video: Any,
        fps: Optional[float] = None,
        frame_indices: Optional[Sequence[int]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> ProcessedVideo:
        """Resize every frame to the shared target geometry.

        Args:
            video (Any): Frames as a `[T, H, W, 3]` array/tensor or a list of PIL images.
            fps (Optional[float]): Sampling rate, used only to render timestamps.
            frame_indices (Optional[Sequence[int]]): Original frame indices; defaults to
                `range(T)`.
            dtype (torch.dtype): Output dtype for the pixel values.

        Return:
            (ProcessedVideo) Resized frames plus the metadata the prompt expander needs.
        """
        frames, orig_w, orig_h = _frames_to_nfhwc(video)
        target_w, target_h, tokens_per_frame = compute_video_geometry(
            orig_w=orig_w,
            orig_h=orig_h,
            target_num_patches=self.target_num_patches,
            maintain_aspect_ratio=self.maintain_aspect_ratio,
            patch_size=self.patch_size,
        )
        pixel_values = bicubic_resize_and_normalize(
            frames, (target_h, target_w), self.norm_mean, self.norm_std, dtype
        )

        # Integer division, not float: a float frame duration makes the host-rendered timestamp
        # string differ from the device-rendered one, which changes the separator token count.
        frame_duration_ms = int(1000.0 / fps) if fps else None

        return ProcessedVideo(
            pixel_values=pixel_values,
            frame_indices=list(frame_indices) if frame_indices else list(range(frames.shape[0])),
            frame_duration_ms=frame_duration_ms,
            tokens_per_frame=tokens_per_frame,
        )
