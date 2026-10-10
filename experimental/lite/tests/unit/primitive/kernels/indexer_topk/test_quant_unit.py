# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts of the indexer top-k operand quantizers and weight folding."""

from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.kernels.indexer_topk import (
    fold_indexer_weights,
    quantize_indexer_fp8_rows,
    quantize_indexer_mxfp4_rows,
    quantize_indexer_mxfp4_rows_reference,
)
from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4

pytestmark = pytest.mark.mlite

_TINY = torch.finfo(torch.float32).tiny
# Next bfloat16 value above each E2M1 midpoint (the midpoint plus one bfloat16 ulp).
_ABOVE = {0.25: 0.251953125, 0.75: 0.75390625, 1.25: 1.2578125, 1.75: 1.7578125}
_ABOVE.update({2.5: 2.515625, 3.5: 3.515625, 5.0: 5.03125})


def _bf16(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).to(torch.bfloat16)


def _pack_codes(codes: list[int]) -> list[int]:
    """Two E2M1 codes per byte, even element in the low nibble, as signed int8 values."""
    packed = [codes[i] | (codes[i + 1] << 4) for i in range(0, len(codes), 2)]
    return [value - 256 if value > 127 else value for value in packed]


def _pack_exponents(exponents: tuple[int, int, int, int]) -> int:
    """Four UE8M0 exponents, group g in bits [8g, 8g + 8), as a signed int32 value."""
    return int.from_bytes(bytes(exponents), "little", signed=True)


def test_fp8_rows_amax_over_448_with_tiny_floor():
    torch.manual_seed(20260930)
    value = torch.randn(4, 3, 128, dtype=torch.bfloat16) * 7
    value[1, 2] = 0.0
    value[2, 0, 5] = 3.0e38

    data, scale = quantize_indexer_fp8_rows(value)

    assert data.dtype == torch.float8_e4m3fn and tuple(data.shape) == (4, 3, 128)
    assert scale.dtype == torch.float32 and tuple(scale.shape) == (4, 3)
    assert data.is_contiguous() and scale.is_contiguous()
    amax = value.float().abs().amax(dim=-1)
    assert torch.equal(scale, (amax / 448.0).clamp_min(_TINY))
    assert scale[1, 2].item() == _TINY and torch.isfinite(scale).all()
    assert not data[1, 2].float().any()
    # The largest magnitude of every nonzero row lands exactly on the E4M3 maximum.
    row_max = data.float().abs().amax(dim=-1)
    assert torch.equal(row_max[amax > 0], torch.full_like(row_max[amax > 0], 448.0))
    expected = (value.float() / scale.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    assert torch.equal(data.view(torch.uint8), expected.view(torch.uint8))


def test_mxfp4_reference_golden_codes():
    above = _ABOVE
    # Row 0, group 0: amax 6.0 -> 6 / 6 = 1.0 is a power of two -> exponent 127, scale 1.0.
    group0 = [0.0, -0.0, 0.25, -0.25, above[0.25], 0.75, above[0.75], 1.25, above[1.25], 1.75]
    group0 += [above[1.75], 2.5, above[2.5], 3.5, above[3.5], 5.0, above[5.0], 6.0, -6.0, -0.75]
    group0 += [-above[0.75], -above[5.0], -0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, -2.0, -3.0, -1.0]
    codes0 = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0xF, 0x9]
    codes0 += [0xA, 0xF, 0, 1, 2, 3, 4, 5, 6, 0xC, 0xD, 0xA]
    # Group 1: amax / 6 is below the 1e-4 floor -> float32(1e-4) has biased exponent 113 and a
    # nonzero mantissa -> exponent 114 (scale 2**-13); every code is +0.
    group1 = [1.0e-5, -1.0e-5, 0.0, -0.0] * 8
    codes1 = [0] * 32
    # Group 2: amax 3.0 -> 3 / 6 = 0.5 is exact -> exponent 126 (scale 0.5).
    group2 = [3.0, 1.0, 0.375, -0.625, 0.125] + [0.0] * 27
    codes2 = [7, 4, 1, 0xA, 0] + [0] * 27
    # Group 3: amax one bfloat16 ulp above 3.0 -> 0.5026 rounds up to exponent 127 (scale 1).
    group3 = [3.015625, 1.0, 0.75, -0.5] + [0.0] * 28
    codes3 = [5, 2, 1, 0x9] + [0] * 28
    # Row 1: the UE8M0 clamp. inf / 6 has biased exponent 255 -> clamped to 254 (scale 2**127),
    # so +-inf saturate to +-6 and finite values vanish; the bfloat16 maximum 3.39e38 gives
    # 5.65e37 = 1.33 * 2**125 -> exponent 253 and 3.39e38 / 2**126 = 3.98 -> code 6; zero and
    # negative-zero groups take the 1e-4 floor.
    bf16_max = torch.finfo(torch.bfloat16).max
    row1 = [float("inf"), float("-inf"), 1.0, -1.0] + [0.0] * 28
    row1 += [bf16_max, -bf16_max] + [0.0] * 30 + [0.0] * 32 + [-0.0] * 32
    codes_row1 = [7, 0xF, 0, 0] + [0] * 28 + [6, 0xE] + [0] * 30 + [0] * 64

    value = torch.stack([_bf16(group0 + group1 + group2 + group3), _bf16(row1)])
    packed, scales = quantize_indexer_mxfp4_rows_reference(value)

    assert packed.dtype == torch.int8 and tuple(packed.shape) == (2, 64)
    assert scales.dtype == torch.int32 and tuple(scales.shape) == (2,)
    expected_packed = [_pack_codes(codes0 + codes1 + codes2 + codes3), _pack_codes(codes_row1)]
    assert packed.tolist() == expected_packed
    assert scales.tolist() == [
        _pack_exponents((127, 114, 126, 127)),
        _pack_exponents((254, 253, 114, 114)),
    ]
    # On CPU the public entry point is the reference; leading dimensions are preserved.
    packed_3d, scales_3d = quantize_indexer_mxfp4_rows(value.view(2, 1, 128))
    assert torch.equal(packed_3d, packed.view(2, 1, 64))
    assert torch.equal(scales_3d, scales.view(2, 1))


def test_mxfp4_indexer_differs_from_checkpoint_quantizer():
    torch.manual_seed(7)
    generic = torch.randn(8, 128, dtype=torch.bfloat16).abs() + 0.01
    packed, scales = quantize_indexer_mxfp4_rows_reference(generic)
    checkpoint_packed, checkpoint_scale = quantize_mxfp4(generic)
    # Away from the three differences both encodings agree byte for byte.
    assert torch.equal(packed, checkpoint_packed)
    assert torch.equal(scales.view(torch.uint8).view(8, 4), checkpoint_scale.view(torch.uint8))

    # 1. Scale floor: tiny and all-zero blocks share the indexer scale 2**-13 (exponent 114);
    #    the checkpoint exponents are ceil(log2(1e-5 / 6)) + 127 = 108 and 0.
    small = torch.zeros(1, 128, dtype=torch.bfloat16)
    small[0, :32] = 1.0e-5
    _, small_scales = quantize_indexer_mxfp4_rows_reference(small)
    _, small_checkpoint = quantize_mxfp4(small)
    assert small_scales.view(torch.uint8).tolist() == [114, 114, 114, 114]
    assert small_checkpoint.view(torch.uint8).tolist() == [[108, 0, 0, 0]]

    # 2. Sign of zero codes: -0.0 and -0.1 (at scale 1) encode as +0 here and as -0 there.
    signed = torch.zeros(1, 128, dtype=torch.bfloat16)
    signed[0, 0], signed[0, 2], signed[0, 3] = 6.0, -0.0, -0.1
    signed_packed, _ = quantize_indexer_mxfp4_rows_reference(signed)
    signed_checkpoint, _ = quantize_mxfp4(signed)
    assert signed_packed.view(torch.uint8)[0, 1].item() == 0x00
    assert signed_checkpoint.view(torch.uint8)[0, 1].item() == 0x88

    # 3. Layout: one packed int32 per 128-value row versus one float8_e8m0fnu per 32 values.
    assert scales.dtype == torch.int32 and tuple(scales.shape) == (8,)
    assert checkpoint_scale.dtype == torch.float8_e8m0fnu
    assert tuple(checkpoint_scale.shape) == (8, 4)
    short_rows = torch.ones(2, 64, dtype=torch.bfloat16)
    assert tuple(quantize_mxfp4(short_rows)[1].shape) == (2, 2)
    with pytest.raises(ValueError, match="rows of 128 values"):
        quantize_indexer_mxfp4_rows_reference(short_rows)
    with pytest.raises(TypeError, match="bfloat16, float16 or float32"):
        quantize_indexer_mxfp4_rows(torch.ones(2, 128, dtype=torch.float64))


def test_fold_indexer_weights_order():
    torch.manual_seed(11)
    weights = torch.randn(256, 32, dtype=torch.bfloat16)
    q_scale = torch.rand(256, 32) * 3 + 0.01
    softmax_scale = 128**-0.5

    folded = fold_indexer_weights(weights, softmax_scale=softmax_scale, q_scale=q_scale)

    assert folded.dtype == torch.float32 and folded.shape == weights.shape
    assert folded.is_contiguous()
    # Softmax scale first, then the query scale: the other association rounds differently.
    assert torch.equal(folded, weights.float().mul(softmax_scale).mul(q_scale))
    assert not torch.equal(folded, weights.float().mul(q_scale).mul(softmax_scale))
    # softmax_scale == 1.0 is skipped (FP8 callers pass pre-scaled weights) and q_scale=None
    # leaves the query scale to the score kernel (MXFP4).
    unit = fold_indexer_weights(weights, softmax_scale=1.0, q_scale=q_scale)
    assert torch.equal(unit, weights.float().mul(q_scale))
    scaled = fold_indexer_weights(weights, softmax_scale=softmax_scale, q_scale=None)
    assert torch.equal(scaled, weights.float().mul(softmax_scale))
    # Same semantics as the upstream indexer, which rounds the scaled weights back to bfloat16,
    # but not the same bits.
    upstream = (weights.float() * softmax_scale).to(torch.bfloat16).float()
    assert not torch.equal(scaled, upstream)
    torch.testing.assert_close(scaled, upstream, rtol=2**-7, atol=0)

    with pytest.raises(ValueError, match="softmax_scale must be positive"):
        fold_indexer_weights(weights, softmax_scale=0.0, q_scale=None)
    with pytest.raises(ValueError, match="q_scale shape"):
        fold_indexer_weights(weights, softmax_scale=1.0, q_scale=q_scale[:, :1])
