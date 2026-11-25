# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/conversion/fp8_export.py
# Source license: Apache-2.0.

"""Topology-neutral helpers for direct Transformer Engine FP8 parameter export."""

import logging
import math
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FP8ExportLayout:
    """Topology-neutral FP8 parameter layout used during checkpoint export."""

    format_name: str
    fp8_dtype: str | None
    block_shape: tuple[int | None, int | None]
    data_dtype: torch.dtype | None
    scale_dtype: torch.dtype | None
    scale_shape: tuple[int, ...]
    compact_scale_shape: tuple[int, int] | None
    with_gemm_swizzled_scales: bool

    def validate(self) -> None:
        """Validate format-specific export invariants."""
        if self.fp8_dtype != "kFloat8E4M3":
            raise ValueError(
                f"FP8 parameter export requires fp8_dtype=kFloat8E4M3, got {self.fp8_dtype!r}"
            )
        if self.data_dtype != torch.uint8:
            raise ValueError(f"FP8 parameter export requires uint8 data, got {self.data_dtype}")
        expected_scale_dtype = torch.uint8 if self.format_name == "mxfp8" else torch.float32
        if self.scale_dtype != expected_scale_dtype:
            raise ValueError(
                f"{self.format_name} parameter export requires {expected_scale_dtype} scales, got {self.scale_dtype}"
            )
        if self.format_name != "mxfp8":
            return
        if self.with_gemm_swizzled_scales:
            raise ValueError("MXFP8 parameter export requires compact, non-swizzled scales")

        compact_shape = self.compact_scale_shape
        if (
            compact_shape is None
            or len(self.scale_shape) != 2
            or self.scale_shape[0] < compact_shape[0]
            or self.scale_shape[1] < compact_shape[1]
        ):
            raise ValueError(
                "MXFP8 scale tensor is smaller than the compact scale shape: "
                f"scale_shape={self.scale_shape}, expected_scale_shape={compact_shape}"
            )


def detect_fp8_export_layout(
    local_weights: torch.Tensor, *, fp8_recipe: str | None, fp8_scale_inv_attr: str
) -> FP8ExportLayout | None:
    """Inspect TE metadata without validating before the bridge's PP collective.

    Args:
        local_weights: Local parameter to inspect.
        fp8_recipe: Configured FP8 recipe.
        fp8_scale_inv_attr: Scale metadata key, optionally prefixed with an underscore.

    Returns:
        Layout for a supported FP8 parameter, or None when no usable scale metadata exists.
    """
    metadata = {}
    get_metadata = getattr(local_weights, "get_metadata", None)
    if callable(get_metadata):
        try:
            candidate_metadata = get_metadata()
        except (AttributeError, RuntimeError, TypeError):
            pass
        else:
            if isinstance(candidate_metadata, dict):
                metadata = candidate_metadata

    rowwise_data = metadata.get("rowwise_data")
    scale_tensor = metadata.get(fp8_scale_inv_attr.removeprefix("_"))
    # Uninitialized scales must not introduce a sidecar task into the export stream.
    if scale_tensor is None:
        return None

    # Python and pybind TE dtype enums expose the same names; no TE import is needed.
    fp8_dtype = getattr(metadata.get("fp8_dtype"), "name", None)
    is_mxfp8 = fp8_recipe == "mxfp8" and "with_gemm_swizzled_scales" in metadata
    if is_mxfp8:
        return FP8ExportLayout(
            format_name="mxfp8",
            fp8_dtype=fp8_dtype,
            block_shape=(1, 32),
            data_dtype=rowwise_data.dtype if isinstance(rowwise_data, torch.Tensor) else None,
            scale_dtype=scale_tensor.dtype if isinstance(scale_tensor, torch.Tensor) else None,
            scale_shape=tuple(scale_tensor.shape) if isinstance(scale_tensor, torch.Tensor) else (),
            compact_scale_shape=(
                math.prod(local_weights.shape[:-1]),
                math.ceil(local_weights.shape[-1] / 32),
            ),
            with_gemm_swizzled_scales=bool(metadata["with_gemm_swizzled_scales"]),
        )
    if "is_2D_scaled" in metadata:
        has_valid_row_ratio = (
            metadata.get("is_2D_scaled")
            and isinstance(scale_tensor, torch.Tensor)
            and local_weights.ndim > 0
            and scale_tensor.ndim > 0
            and scale_tensor.shape[0] > 0
            and local_weights.shape[0] % scale_tensor.shape[0] == 0
        )
        row_block_size = (
            local_weights.shape[0] // scale_tensor.shape[0] if has_valid_row_ratio else None
        )
        block_len = getattr(metadata.get("quantizer"), "block_len", None)
        return FP8ExportLayout(
            format_name="blockwise",
            fp8_dtype=fp8_dtype,
            block_shape=(row_block_size, block_len if isinstance(block_len, int) else None),
            data_dtype=rowwise_data.dtype if isinstance(rowwise_data, torch.Tensor) else None,
            scale_dtype=scale_tensor.dtype if isinstance(scale_tensor, torch.Tensor) else None,
            scale_shape=tuple(scale_tensor.shape) if isinstance(scale_tensor, torch.Tensor) else (),
            compact_scale_shape=None,
            with_gemm_swizzled_scales=False,
        )
    return None


def get_fp8_export_tensors(
    local_weights: torch.Tensor | None, *, fp8_scale_inv_attr: str
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Extract FP8 weight and inverse-scale tensors after collective layout validation.

    Args:
        local_weights: Local FP8 parameter, or None on a non-owning rank.
        fp8_scale_inv_attr: Scale metadata key, optionally prefixed with an underscore.

    Returns:
        E4M3 weight data and compact inverse scales in their source dtype.
    """
    metadata = getattr(local_weights, "get_metadata")() if local_weights is not None else {}
    export_weight_tensor = local_weights
    rowwise_data = metadata.get("rowwise_data")
    if rowwise_data is not None:
        # FP8 parameter weights are E4M3 in both E4M3 and hybrid recipes;
        # E5M2 is used only for backward gradients.
        export_weight_tensor = rowwise_data.contiguous().view(torch.float8_e4m3fn)
    scale_tensor = metadata.get(fp8_scale_inv_attr.removeprefix("_"))
    if local_weights is None or scale_tensor is None:
        return export_weight_tensor, scale_tensor

    # TE pads MXFP8 scales to multiples of (128, 4), while 2D blockwise
    # scales may pad their K dimension.
    if "with_gemm_swizzled_scales" in metadata:
        expected_m = math.prod(local_weights.shape[:-1])
        expected_k_tiles = math.ceil(local_weights.shape[-1] / 32)
        if scale_tensor.shape != (expected_m, expected_k_tiles):
            scale_tensor = scale_tensor[:expected_m, :expected_k_tiles].contiguous()
        return export_weight_tensor, scale_tensor

    quantizer = metadata.get("quantizer")
    block_len = getattr(quantizer, "block_len", None)
    is_2d_scaled = metadata.get("is_2D_scaled")
    if block_len is None or not is_2d_scaled:
        logger.warning("WARNING: block_len or not is_2d_scaled")
        return export_weight_tensor, scale_tensor

    q_k = local_weights.shape[-1]
    expected_k_tiles = math.ceil(q_k / block_len)
    if scale_tensor.shape[1] != expected_k_tiles:
        scale_tensor = scale_tensor[:, :expected_k_tiles].contiguous()
    return export_weight_tensor, scale_tensor
