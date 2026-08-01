# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Host-side dynamic-resolution image preprocessing for Nemotron Omni.

Every arithmetic detail here is load-bearing for token-count parity with the reference
implementation. The ones that look like typos but are not:

- `round(dim / patch + 0.5)`, *not* `math.ceil`. They differ for exact multiples: a 512px side
  with patch 16 gives 33 here and 32 from ceil.
- `factor = min(sqrt(budget / patches), 1.0)`, so an image is never upscaled.
- The budget is in *patches*, four per post-shuffle token, hence the x4 conversion.
- Patch grids are snapped to a multiple of 2 preferring *up*, but only when rounding up still
  fits the budget; otherwise down, floored at 2.
- The rebalancing loop runs at most 10 times and then gives up on the min-patches floor.
"""

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch

try:
    import einops

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False

try:
    import numpy as np
    from PIL import Image

    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False


@dataclass
class TiledImage:
    """One image resized to a patch grid, with the token count it will produce."""

    pixel_values: torch.Tensor
    """`[3, height, width]`, normalized, fp32 by default."""

    patch_grid: Tuple[int, int]
    """`(patch_width, patch_height)`."""

    num_tokens: int
    """Post-pixel-shuffle token count, i.e. `patches // 4`."""

    @property
    def size_hw(self) -> Tuple[int, int]:
        """Pixel `(height, width)`, the order `RADIOViTModel` expects in `imgs_sizes`."""
        return (self.pixel_values.shape[-2], self.pixel_values.shape[-1])


def _to_nhwc_uint8(image: Any) -> torch.Tensor:
    """Coerce a PIL image, numpy array, or path into a `[1, H, W, 3]` uint8 tensor."""
    if not HAVE_PIL:
        raise ImportError("Pillow and numpy are required for Nemotron Omni image preprocessing")
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        array = np.asarray(image if image.mode == "RGB" else image.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        array = image.astype(np.uint8, copy=False)
    elif torch.is_tensor(image):
        array = image.to(torch.uint8).cpu().numpy()
    else:
        raise TypeError(f"unsupported image type {type(image).__name__}")
    assert array.ndim == 3 and array.shape[-1] == 3, f"expected HWC RGB, got {array.shape}"
    return torch.from_numpy(np.expand_dims(array, axis=0))


def bicubic_resize_and_normalize(
    tensor: torch.Tensor,
    size: Optional[Tuple[int, int]],
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Permute NHWC to NCHW, bicubic resize, then rescale and normalize.

    Args:
        tensor (torch.Tensor): `[N, H, W, C]` uint8.
        size (Optional[Tuple[int, int]]): Target `(height, width)`; skips resize when None.
        norm_mean (torch.Tensor): `[3, 1, 1]` mean, in 0-1 units.
        norm_std (torch.Tensor): `[3, 1, 1]` std, in 0-1 units.
        dtype (torch.dtype): Output dtype.

    Return:
        (torch.Tensor) `[N, C, H, W]` normalized.
    """
    tensor = tensor.permute(0, 3, 1, 2).to(dtype=torch.float32)
    if size is not None:
        tensor = torch.nn.functional.interpolate(
            tensor, size=size, mode="bicubic", align_corners=False, antialias=True
        )
    return ((tensor / 255.0 - norm_mean) / norm_std).to(dtype=dtype).contiguous()


def patchify(images: Sequence[torch.Tensor], patch_size: int) -> torch.Tensor:
    """Flatten resized images into the packed patch sequence `RADIOViTModel` consumes.

    Args:
        images (Sequence[torch.Tensor]): Each `[3, H, W]`, with H and W multiples of
            `patch_size`.
        patch_size (int): Patch side in pixels.

    Return:
        (torch.Tensor) `[1, total_patches, 3 * patch_size**2]`.
    """
    if not HAVE_EINOPS:
        raise ImportError("einops is required for Nemotron Omni image preprocessing")

    flattened = []
    for img in images:
        py = img.shape[-2] // patch_size
        px = img.shape[-1] // patch_size
        flattened.append(
            einops.rearrange(
                img,
                "c (py yy) (px xx) -> (py px) (c yy xx)",
                py=py,
                yy=patch_size,
                px=px,
                xx=patch_size,
            )
        )
    return torch.cat(flattened, dim=0).unsqueeze(0)


class DynamicResolutionImageTiler:
    """Chooses a patch grid per image under a shared token budget, then resizes.

    Unlike fixed-tile preprocessors, every image keeps its native aspect ratio and the budget
    is shared across all images in the request, so adding a second image shrinks the first.
    """

    # Pixel shuffle folds 2x2 patch neighbourhoods, so 4 patches produce 1 token.
    PATCHES_PER_TOKEN = 4
    # Patch grids must be even on both axes for the 2x2 fold to tile exactly.
    REQUIRED_DIVISOR = 2

    def __init__(
        self,
        *,
        patch_size: int,
        min_num_patches: int,
        max_num_patches: int,
        norm_mean: Sequence[float],
        norm_std: Sequence[float],
        factor_max: float = 1.0,
    ) -> None:
        self.patch_size = patch_size
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches if max_num_patches > 0 else math.inf
        self.factor_max = factor_max
        self.norm_mean = torch.tensor(norm_mean).reshape(3, 1, 1)
        self.norm_std = torch.tensor(norm_std).reshape(3, 1, 1)

    def num_tokens_for_grid(self, patch_width: int, patch_height: int) -> int:
        """Post-shuffle token count for a patch grid."""
        return (patch_width * patch_height) // self.PATCHES_PER_TOKEN

    def _choose_grid(self, width: int, height: int, patch_budget: int) -> Tuple[int, int]:
        """Pick `(patch_width, patch_height)` for one image under a patch budget.

        Return:
            (Tuple[int, int]) Patch grid, both axes even.
        """
        # +0.5 then round, which is not ceil: an exact multiple of patch_size rounds *up* to
        # one extra patch here. Reproduced deliberately.
        closest_patch_height = round(height / self.patch_size + 0.5)
        closest_patch_width = round(width / self.patch_size + 0.5)
        patches = closest_patch_height * closest_patch_width

        # min(..., 1.0) means images are downscaled to fit but never upscaled to fill.
        factor = min(math.sqrt(patch_budget / patches), self.factor_max)
        target_h = math.floor(factor * closest_patch_height)
        target_w = math.floor(factor * closest_patch_width)

        # Pull tiny images back up to the floor, but only if the floor itself fits.
        if patch_budget > self.min_num_patches and target_h * target_w < self.min_num_patches:
            up_factor = math.sqrt(self.min_num_patches / max(target_h * target_w, 1))
            target_h = math.ceil(up_factor * target_h)
            target_w = math.ceil(up_factor * target_w)

        divisor = self.REQUIRED_DIVISOR
        rem_h = target_h % divisor
        if rem_h != 0:
            inc_h = divisor - rem_h
            if (target_h + inc_h) * target_w <= patch_budget:
                target_h += inc_h
            else:
                target_h = max(divisor, target_h - rem_h)

        rem_w = target_w % divisor
        if rem_w != 0:
            inc_w = divisor - rem_w
            if target_h * (target_w + inc_w) <= patch_budget:
                target_w += inc_w
            else:
                target_w = max(divisor, target_w - rem_w)

        return target_w, target_h

    def compute_grids(
        self, sizes: Sequence[Tuple[int, int]], token_budget: int
    ) -> List[Tuple[int, int]]:
        """Allocate the shared budget across images and return one patch grid each.

        Args:
            sizes (Sequence[Tuple[int, int]]): Original `(width, height)` per image.
            token_budget (int): Post-shuffle tokens available for all images combined.

        Return:
            (List[Tuple[int, int]]) `(patch_width, patch_height)` per image.
        """
        assert sizes, "no images to tile"

        patch_budget = token_budget * self.PATCHES_PER_TOKEN
        # A budget too small for even the floor is allowed; the prompt gets truncated instead
        # of the request being rejected.
        patch_budget = max(patch_budget, self.min_num_patches * len(sizes))

        per_image_budget = [
            int(max(min(patch_budget, self.max_num_patches), self.min_num_patches)) for _ in sizes
        ]

        # Each round re-slices the budget proportionally to what the previous round asked for.
        # Bounded at 10 rounds; the reference implementation raises past that, and so do we,
        # because silently over-budget grids would desync the host token count from the
        # encoder output length.
        for _ in range(10):
            grids = [
                self._choose_grid(w, h, budget) for (w, h), budget in zip(sizes, per_image_budget)
            ]
            patch_counts = [pw * ph for pw, ph in grids]
            total = sum(patch_counts)
            if total <= patch_budget:
                return grids

            scaling = patch_budget / total
            scaled = [max(self.min_num_patches, int(count * scaling)) for count in patch_counts]
            if any(scaled[i] < per_image_budget[i] for i in range(len(scaled))):
                per_image_budget = scaled
            else:
                # No further reduction possible; pin everything to the floor and retry once.
                per_image_budget = [self.min_num_patches] * len(sizes)

        raise ValueError(
            f"failed to fit {len(sizes)} images into {patch_budget} patches after 10 rounds"
        )

    def process(
        self, images: Sequence[Any], token_budget: int, dtype: torch.dtype = torch.float32
    ) -> List[TiledImage]:
        """Resize and normalize every image under a shared token budget.

        Args:
            images (Sequence[Any]): PIL images, numpy arrays, or paths.
            token_budget (int): Post-shuffle tokens available for all images combined.
            dtype (torch.dtype): Output dtype for the pixel values.

        Return:
            (List[TiledImage]) One entry per input image, in order.
        """
        raw = [_to_nhwc_uint8(image) for image in images]
        sizes = [(t.shape[2], t.shape[1]) for t in raw]  # (width, height)
        grids = self.compute_grids(sizes, token_budget)

        tiled = []
        for tensor, (patch_w, patch_h) in zip(raw, grids):
            target = (patch_h * self.patch_size, patch_w * self.patch_size)
            pixel_values = bicubic_resize_and_normalize(
                tensor, target, self.norm_mean, self.norm_std, dtype
            )
            tiled.append(
                TiledImage(
                    pixel_values=pixel_values[0],
                    patch_grid=(patch_w, patch_h),
                    num_tokens=self.num_tokens_for_grid(patch_w, patch_h),
                )
            )
        return tiled
