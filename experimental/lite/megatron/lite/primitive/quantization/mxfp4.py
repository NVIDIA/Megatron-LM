# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MXFP4 E2M1 + UE8M0 checkpoint serialization."""

from __future__ import annotations

import torch

MXFP4_BLOCK_SIZE = 32
_E2M1_POSITIVE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _validate(tensor: torch.Tensor) -> None:
    if tensor.ndim < 1:
        raise ValueError("MXFP4 tensor must have at least one dimension")
    if not tensor.dtype.is_floating_point:
        raise TypeError(f"MXFP4 source must be floating point, got {tensor.dtype}")
    if tensor.shape[-1] % MXFP4_BLOCK_SIZE:
        raise ValueError(
            f"tensor last dimension {tensor.shape[-1]} must be divisible by "
            f"{MXFP4_BLOCK_SIZE}"
        )


def _select_scale(blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = blocks.abs().amax(dim=-1)
    floor = 6.0 * (2.0**-126)
    exponent = torch.ceil(torch.log2(amax.clamp_min(floor) / 6.0)).clamp(-127, 127)
    encoded = (exponent + 127).to(torch.uint8)
    return torch.exp2(exponent), encoded.view(torch.float8_e8m0fnu)


def _quantize_nibbles(values: torch.Tensor) -> torch.Tensor:
    magnitude = values.abs()
    # Midpoints between [0, .5, 1, 1.5, 2, 3, 4, 6]. Start with strict
    # greater-than so ties stay on the lower encoding, then promote only the
    # three midpoints whose upper encoding is even. This reproduces PTX
    # ``cvt.rn.satfinite.e2m1x2.f32`` without an 8x codebook-distance tensor.
    boundaries = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    index = torch.zeros_like(magnitude, dtype=torch.uint8)
    for boundary in boundaries:
        index += (magnitude > boundary).to(torch.uint8)
    for upper_even_boundary in (0.75, 1.75, 3.5):
        index += (magnitude == upper_even_boundary).to(torch.uint8)
    sign = torch.where(values.signbit(), torch.tensor(8, device=values.device), 0).to(
        torch.uint8
    )
    return index | sign


def quantize_mxfp4(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Serialize the last dimension in 32-value MXFP4 blocks."""
    _validate(tensor)
    source = tensor.float()
    blocks = source.reshape(*source.shape[:-1], -1, MXFP4_BLOCK_SIZE)
    scale_f, scale = _select_scale(blocks)
    normalized = blocks / scale_f.unsqueeze(-1)
    nibbles = _quantize_nibbles(normalized).reshape(
        *source.shape[:-1], source.shape[-1]
    )
    packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
    return packed.contiguous().view(torch.int8), scale.contiguous()


def dequantize_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize checkpoint-format MXFP4 for CPU validation."""
    if packed.dtype != torch.int8:
        raise TypeError(f"MXFP4 packed tensor must be int8, got {packed.dtype}")
    expected = (*packed.shape[:-1], packed.shape[-1] * 2 // MXFP4_BLOCK_SIZE)
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"scale shape {tuple(scale.shape)} does not match expected {expected}"
        )
    table = torch.tensor(
        (*_E2M1_POSITIVE, *(value * -1.0 for value in _E2M1_POSITIVE)),
        dtype=torch.float32,
        device=packed.device,
    )
    raw = packed.view(torch.uint8)
    values = torch.stack((table[(raw & 0x0F).long()], table[(raw >> 4).long()]), dim=-1)
    values = values.flatten(-2)
    expanded_scale = scale.float().repeat_interleave(MXFP4_BLOCK_SIZE, dim=-1)
    return values * expanded_scale


__all__ = ["MXFP4_BLOCK_SIZE", "dequantize_mxfp4", "quantize_mxfp4"]
