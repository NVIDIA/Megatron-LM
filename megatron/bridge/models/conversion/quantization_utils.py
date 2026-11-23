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


# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/b57951ad292e43209f1cc7d4b5500768365a20ac/src/megatron/bridge/models/conversion/quantization_utils.py
# Source license: Apache-2.0.

import math
from collections.abc import Mapping

import torch

FP8_BLOCK_SIZE = 128
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def is_fp8_tensor(tensor: torch.Tensor) -> bool:
    """Return whether *tensor* uses one of PyTorch's FP8 dtypes."""
    return tensor.dtype in FP8_DTYPES


def fp8_weight_block_size(hf_config: object) -> tuple[int, int] | None:
    """Return the ``quantization_config.weight_block_size`` an HF config declares, if any."""
    quantization_config = getattr(hf_config, "quantization_config", None)
    if hasattr(quantization_config, "to_dict"):
        quantization_config = quantization_config.to_dict()
    if (
        not isinstance(quantization_config, Mapping)
        or quantization_config.get("weight_block_size") is None
    ):
        return None
    rows, cols = quantization_config["weight_block_size"]
    return int(rows), int(cols)


def dequantize_fp8_scale_inv(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    *,
    block_size: tuple[int, int] | None = None,
    name: str = "",
) -> torch.Tensor:
    """Dequantize FP8 codes with their ``*_scale_inv`` companion into float32.

    Fine-grained FP8 checkpoints (``quant_method: "fp8"``, as in DeepSeek-V3 and Qwen3 FP8) store
    one scale per ``block_size`` block of the last two weight dimensions, with partial blocks at
    the edges (``ceil``-shaped grids), or a single scale for the whole tensor. The weight is
    ``weight.float() * scale_inv`` broadcast over each block. Leading dimensions, such as stacked
    experts, must match between the weight and its scale grid.

    Args:
        weight: FP8 tensor with at least two dimensions.
        scale_inv: Scale grid, or a tensor holding one scale.
        block_size: ``(rows, cols)`` of one block; defaults to ``FP8_BLOCK_SIZE`` squared.
        name: Tensor name for error messages.

    Raises:
        ValueError: If the scale grid does not match the weight shape and block size.
    """
    if scale_inv.numel() == 1:
        return weight.float() * scale_inv.float().reshape(())
    label = name or "<unnamed>"
    if weight.ndim < 2:
        raise ValueError(
            f"Cannot dequantize FP8 tensor {label} of shape {tuple(weight.shape)} with a scale_inv of shape "
            f"{tuple(scale_inv.shape)}: block scales need at least two weight dimensions."
        )
    rows, cols = block_size or (FP8_BLOCK_SIZE, FP8_BLOCK_SIZE)
    *leading, m, n = weight.shape
    expected = (*leading, math.ceil(m / rows), math.ceil(n / cols))
    if tuple(scale_inv.shape) != expected:
        raise ValueError(
            f"Cannot dequantize FP8 tensor {label} of shape {tuple(weight.shape)}: its scale_inv "
            f"has shape {tuple(scale_inv.shape)}, but {rows}x{cols} blocks need {expected}."
        )
    # Scale one row of blocks at a time so the only full-size temporary is the float32 result.
    out = weight.float()
    row_scales = scale_inv.float().repeat_interleave(cols, dim=-1)[..., :n]
    for i in range(expected[-2]):
        out[..., i * rows : (i + 1) * rows, :] *= row_scales[..., i : i + 1, :]
    return out


def dequantize_fp8_scale_inv_weights(
    hf_param: str | Mapping[str, str],
    hf_weights: torch.Tensor | Mapping[str, torch.Tensor],
    hf_state_dict: Mapping[str, torch.Tensor],
    *,
    block_size: tuple[int, int] | None = None,
) -> torch.Tensor | Mapping[str, torch.Tensor]:
    """Dequantize loaded FP8 tensors that have a ``<name>_scale_inv`` companion in the checkpoint.

    ``hf_weights`` is what a bridge's ``maybe_modify_loaded_hf_weight`` returned for ``hf_param``.
    Tensors that are not FP8 or have no companion are returned unchanged, and so is ``hf_weights``
    when nothing needs dequantizing.
    """

    def dequantize(name: str, tensor: torch.Tensor) -> torch.Tensor:
        scale_inv = hf_state_dict.get(f"{name}_scale_inv") if is_fp8_tensor(tensor) else None
        if scale_inv is None:
            return tensor
        return dequantize_fp8_scale_inv(tensor, scale_inv, block_size=block_size, name=name)

    if isinstance(hf_weights, torch.Tensor):
        return dequantize(hf_param, hf_weights) if isinstance(hf_param, str) else hf_weights
    if not isinstance(hf_param, Mapping) or not any(
        isinstance(tensor, torch.Tensor) and is_fp8_tensor(tensor) for tensor in hf_weights.values()
    ):
        return hf_weights
    return {
        key: (
            dequantize(hf_param[key], tensor)
            if key in hf_param and isinstance(tensor, torch.Tensor)
            else tensor
        )
        for key, tensor in hf_weights.items()
    }
