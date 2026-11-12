# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts of the indexer top-k operand quantizers and weight folding."""

from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.kernels.indexer_topk import (
    fold_indexer_weights,
    quantize_indexer_fp8_rows,
)

pytestmark = pytest.mark.mlite

_TINY = torch.finfo(torch.float32).tiny


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
    # folds only the softmax scale.
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
