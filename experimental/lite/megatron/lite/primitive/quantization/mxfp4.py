# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MXFP4 E2M1 + UE8M0 encoding — the single source of truth for MXFP4 numerics.

This module owns the two rules that fully determine an MXFP4 encoding:

1. :func:`mx_shared_scale_exponent` — the E8M0 per-block power-of-two scale;
2. :func:`e2m1_round_index` — the element rounding onto the E2M1 grid.

Both are defined to be **bit-identical to the quantizer the rollout actually
runs**: ModelOpt's ``MXFP4QTensor.quantize`` (``modelopt/torch/quantization/
qtensor/mxfp4_tensor.py``), which verl's ``utils/modelopt/qat_weight_exporter.py``
calls (via ``to_quantized_weight``) to produce the weights vLLM serves.

Everything else in Lite that touches MXFP4 — in particular the training-time
fake-quant in :mod:`megatron.lite.primitive.quantization.qat` — imports these
two functions rather than reimplementing them. QAT is only meaningful if the
error the training graph compensates is the error the rollout actually makes,
so a second, "equivalent" implementation is not acceptable here.

``tests/unit/primitive/test_mxfp4_modelopt_parity_unit.py`` locks both rules
against a verbatim transcription of the ModelOpt kernel (and against the real
package when it is importable). If you change either rule, that test goes red;
that is the point.
"""

from __future__ import annotations

import torch

MXFP4_BLOCK_SIZE = 32
_E2M1_POSITIVE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# OCP E2M1 representable magnitudes, the midpoints between them, and the max.
E2M1_LEVELS: tuple[float, ...] = _E2M1_POSITIVE
E2M1_MIDPOINTS: tuple[float, ...] = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
E2M1_MAX: float = 6.0

# E8M0 stores an unsigned biased exponent; bias 127, so 2^e with e >= -127.
E8M0_BIAS: int = 127
E8M0_EXP_MIN: int = -127


def mx_shared_scale_exponent(block_amax: torch.Tensor) -> torch.Tensor:
    """E8M0 block-scale exponent ``e`` (float32) such that the scale is ``2^e``.

    ``e = ceil(max(log2(amax / 6), -127))``.

    This is ModelOpt's rule verbatim::

        descale = input_amax / cls.E2M1_max
        e8m0_scale = torch.ceil(torch.maximum(torch.log2(descale), -127.0))

    It guarantees ``amax / 2^e <= 6``, i.e. the block maximum is *never*
    saturated. Do not "simplify" it to the OCP Alg. 1 form
    ``floor(log2(amax)) - emax_elem``: that picks a scale 2x too small whenever
    the amax mantissa exceeds 1.5 (~29-42% of blocks on real weights) and then
    clips the block maximum — the single most important value in the block — to
    6.0. An all-zero block yields ``log2(0) = -inf -> e = -127``; its codes are
    all zero regardless, matching ModelOpt byte for byte.

    There is deliberately no upper clamp: bf16/fp32 amax cannot reach
    ``6 * 2^127``, and adding one would diverge from ModelOpt.
    """
    descale = block_amax.float() / E2M1_MAX
    floor_exp = torch.tensor(
        float(E8M0_EXP_MIN), dtype=torch.float32, device=descale.device
    )
    return torch.ceil(torch.maximum(torch.log2(descale), floor_exp))


def mx_shared_scale(block_amax: torch.Tensor) -> torch.Tensor:
    """The E8M0 block scale ``X = 2^e`` as float32. See :func:`mx_shared_scale_exponent`."""
    return torch.exp2(mx_shared_scale_exponent(block_amax))


def e2m1_round_index(magnitude: torch.Tensor) -> torch.Tensor:
    """Index (0..7) into :data:`E2M1_LEVELS` for a non-negative tensor.

    **Ties round down**, matching ModelOpt's ``cast_fp4``::

        ord_ = torch.sum((x.abs().unsqueeze(-1) - E2M1_bounds) > 0, dim=-1)

    i.e. the number of midpoints *strictly* below the magnitude — which is
    exactly ``torch.bucketize(..., right=False)``. This is **not** round-half-to-
    even: after scaling by an exact power of two the E2M1 midpoints (0.75, 1.75,
    3.5, ...) are exactly representable in bf16, so ~1.1% of elements of a real
    weight tensor land on one and the tie rule is observable, not theoretical.
    Values above 5.0 saturate to the top level (6.0).
    """
    mids = torch.tensor(E2M1_MIDPOINTS, dtype=torch.float32, device=magnitude.device)
    return torch.bucketize(magnitude, mids, right=False)


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
    exponent = mx_shared_scale_exponent(blocks.abs().amax(dim=-1))
    encoded = (exponent + E8M0_BIAS).to(torch.uint8)
    return torch.exp2(exponent), encoded.view(torch.float8_e8m0fnu)


def _quantize_nibbles(values: torch.Tensor) -> torch.Tensor:
    index = e2m1_round_index(values.abs()).to(torch.uint8)
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


__all__ = [
    "E2M1_LEVELS",
    "E2M1_MAX",
    "E2M1_MIDPOINTS",
    "E8M0_BIAS",
    "E8M0_EXP_MIN",
    "MXFP4_BLOCK_SIZE",
    "dequantize_mxfp4",
    "e2m1_round_index",
    "mx_shared_scale",
    "mx_shared_scale_exponent",
    "quantize_mxfp4",
]
