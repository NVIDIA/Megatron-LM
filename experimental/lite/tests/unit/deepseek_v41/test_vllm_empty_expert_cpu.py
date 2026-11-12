# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Empty native expert slots retain their shape and contribute zero gradients."""
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm.experts import _Grouped, dequantize_activation
from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4


@pytest.mark.parametrize('width', [512, 2304, 5120])
@pytest.mark.parametrize('rows', [0, 2])
def test_decode_empty_and_nonempty(rows, width):
    codes = torch.ones(rows, width).to(torch.float8_e4m3fn)
    scale = torch.full((rows, width // 32), 2.0)
    if not rows:
        with pytest.raises(RuntimeError, match='ambiguous'):
            codes.float().reshape(*scale.shape, -1)
    actual = dequantize_activation(codes, scale)
    assert actual.shape == codes.shape and actual.dtype == torch.float32
    assert torch.equal(actual, torch.full((rows, width), 2.0))


@pytest.mark.parametrize('counts', [(0, 0), (2, 0), (0, 2)])
def test_grouped_backward_empty_slots(counts):
    rows = sum(counts)
    packed, scales = quantize_mxfp4(torch.full((64, 64), 0.5))
    ctx = SimpleNamespace(
        saved_tensors=(
            torch.ones(rows, 64).to(torch.float8_e4m3fn),
            torch.ones(rows, 2),
            torch.stack([packed, packed]),
            torch.stack([scales, scales]),
        ),
        counts=counts,
        dtypes=[torch.float32, torch.float32],
        input_dtype=torch.bfloat16,
    )
    dx, ignored_counts, ignored_codes, *dw = _Grouped.backward(
        ctx, torch.ones(rows, 64, dtype=torch.bfloat16)
    )
    assert ignored_counts is ignored_codes is None
    assert dx.shape == (rows, 64) and dx.dtype == torch.bfloat16
    assert torch.equal(dx, torch.full_like(dx, 32))
    for count, grad in zip(counts, dw):
        assert torch.isfinite(grad).all()
        assert torch.equal(grad, torch.full((64, 64), float(count)))
