# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Block-FP8 quantize/dequantize round-trip fidelity.

Validates numerical fidelity only (the quantize->dequantize round-trip error stays
within the E4M3 blockwise magnitude); it does not validate layout, axis order, or
fusion. Zero GPU, pure tensors. DS4 routed experts use expert_dtype=fp8 ->
quantize_block_fp8 with scale_format="float32", so every case here exercises the
float32 scale branch.
"""

from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.quantization.block_fp8 import (
    dequantize_block_fp8,
    quantize_block_fp8,
)

# E4M3 has 3 mantissa bits -> per-value resolution ~2^-3 = 12.5%. With blockwise
# per-128x128 absmax scaling, the overall (Frobenius) energy relative error should
# be ~2^-3/sqrt(3), i.e. a few percent. We gate on 6%; element-wise max-rel is not
# a criterion (near-zero elements inside a block have naturally large rel error).
FROBENIUS_TOL = 0.06


def _frobenius_rel_err(restored: torch.Tensor, source: torch.Tensor) -> float:
    return (
        torch.linalg.vector_norm(restored.float() - source.float())
        / torch.linalg.vector_norm(source.float())
    ).item()


def _roundtrip(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight, scale = quantize_block_fp8(source, (128, 128), scale_format="float32")
    restored = dequantize_block_fp8(weight, scale, (128, 128))
    return weight, scale, restored


def test_roundtrip_extreme_negative_and_near_zero_values() -> None:
    """Tensor with E4M3 extremes / negatives / near-zero / exact-zero round-trips within fp8 magnitude."""
    torch.manual_seed(0)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    source = torch.empty(256, 256)
    # Mix in block-level large magnitudes (+/- near fp8_max), tiny values, exact 0,
    # and a regular small-weight scale.
    source[:128, :128] = torch.linspace(-fp8_max, fp8_max, 128 * 128).reshape(128, 128)
    source[:128, 128:] = torch.full((128, 128), 1e-4)
    source[128:, :128] = torch.randn(128, 128) * 0.02
    block = torch.randn(128, 128) * 0.02
    block[0, 0] = fp8_max          # one large outlier in the block sets the scale
    block[1, 1] = -fp8_max
    block[2, 2] = 0.0              # exact zero must round-trip back to zero
    source[128:, 128:] = block

    weight, scale, restored = _roundtrip(source)

    assert weight.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32
    assert scale.shape == (2, 2)  # 256/128 x 256/128
    assert torch.isfinite(restored).all(), "dequant produced NaN/Inf"
    # Exact-zero elements must restore to 0 (finite scale, quantization preserves 0).
    assert restored[128 + 2, 128 + 2].abs().item() == 0.0
    rel = _frobenius_rel_err(restored, source)
    assert rel < FROBENIUS_TOL, f"Frobenius rel err {rel:.4f} >= {FROBENIUS_TOL}"


@pytest.mark.parametrize(
    "shape,label",
    [
        ((512, 1024), "w1_gate_up_I_by_H"),   # expert gate/up: [I, H]
        ((1024, 512), "w2_down_H_by_I"),      # expert down:    [H, I]
    ],
)
def test_roundtrip_ds4_expert_shapes(shape: tuple[int, int], label: str) -> None:
    """DS4 routed-expert weight shapes (w1[I,H]/w2[H,I]) round-trip under a realistic weight distribution."""
    torch.manual_seed(1234)
    # Realistic weight-like distribution: small-scale normal + a few outliers (which set each block's scale).
    source = torch.randn(*shape) * 0.02
    outlier_mask = torch.rand(*shape) < 0.001
    source = torch.where(outlier_mask, torch.sign(source) * 5.0, source)

    weight, scale, restored = _roundtrip(source)

    assert weight.shape == source.shape, f"{label}: quantization changed the shape (suspect axis/layout error)"
    assert scale.shape == (shape[0] // 128, shape[1] // 128)
    assert torch.isfinite(restored).all()
    rel = _frobenius_rel_err(restored, source)
    assert rel < FROBENIUS_TOL, f"{label}: Frobenius rel err {rel:.4f} >= {FROBENIUS_TOL}"


def test_roundtrip_is_deterministic() -> None:
    """The same input quantized/dequantized twice is bit-identical (no randomness; usable as a CI baseline)."""
    source = torch.randn(128, 128) * 0.05
    _, _, a = _roundtrip(source)
    _, _, b = _roundtrip(source)
    assert torch.equal(a, b)
