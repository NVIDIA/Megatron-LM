# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pure-tensor block-FP8 checkpoint serialization."""

from __future__ import annotations

from typing import Literal

import torch

ScaleFormat = Literal["float32", "e8m0"]


def _validate(tensor: torch.Tensor, block_shape: tuple[int, int]) -> None:
    if tensor.ndim < 2:
        raise ValueError(
            f"block-FP8 tensor must have at least two dimensions, got {tensor.shape}"
        )
    if not tensor.dtype.is_floating_point:
        raise TypeError(f"block-FP8 source must be floating point, got {tensor.dtype}")
    block_m, block_k = block_shape
    if block_m <= 0 or block_k <= 0:
        raise ValueError(f"block shape must be positive, got {block_shape}")
    if tensor.shape[-2] % block_m or tensor.shape[-1] % block_k:
        raise ValueError(
            f"tensor trailing shape {tuple(tensor.shape[-2:])} must be divisible by "
            f"block shape {block_shape}"
        )


def _encode_e8m0(scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    exponent = torch.ceil(torch.log2(scale)).clamp(-127, 127)
    encoded = (exponent + 127).to(torch.uint8)
    rounded = torch.exp2(exponent)
    return rounded, encoded.view(torch.float8_e8m0fnu)


def quantize_block_fp8(
    tensor: torch.Tensor,
    block_shape: tuple[int, int] = (128, 128),
    *,
    scale_format: ScaleFormat = "float32",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize trailing 2-D blocks to E4M3 and checkpoint-format descales."""
    _validate(tensor, block_shape)
    if scale_format not in {"float32", "e8m0"}:
        raise ValueError(f"unsupported block-FP8 scale format: {scale_format!r}")

    source = tensor.float()
    block_m, block_k = block_shape
    *leading, rows, columns = source.shape
    row_blocks = rows // block_m
    column_blocks = columns // block_k
    blocked = source.reshape(*leading, row_blocks, block_m, column_blocks, block_k)
    amax = blocked.abs().amax(dim=(-3, -1))
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    # Match vLLM/DeepGEMM checkpoint quantization: clamp the block amax before
    # dividing by E4M3's finite maximum, then optionally ceil to UE8M0.
    scale_f = amax.clamp_min(1e-4) / fp8_max
    if scale_format == "e8m0":
        scale_f, serialized_scale = _encode_e8m0(scale_f)
    else:
        serialized_scale = scale_f

    expanded = scale_f.repeat_interleave(block_m, dim=-2).repeat_interleave(
        block_k, dim=-1
    )
    quantized = (source / expanded).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return quantized, serialized_scale.contiguous()


def dequantize_block_fp8(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    block_shape: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """Dequantize a tensor emitted by :func:`quantize_block_fp8`."""
    _validate(tensor, block_shape)
    block_m, block_k = block_shape
    expected = (
        *tensor.shape[:-2],
        tensor.shape[-2] // block_m,
        tensor.shape[-1] // block_k,
    )
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"scale shape {tuple(scale.shape)} does not match expected {expected}"
        )
    scale_f = scale.float()
    expanded = scale_f.repeat_interleave(block_m, dim=-2).repeat_interleave(
        block_k, dim=-1
    )
    return tensor.float() * expanded


__all__ = ["ScaleFormat", "dequantize_block_fp8", "quantize_block_fp8"]
