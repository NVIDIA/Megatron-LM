# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.moe import flashinfer_cutedsl_mxfp8


def _run_with_fake_kernels(monkeypatch, out, num_tokens=3, hidden=64):
    """Call flashinfer_cutedsl_mxfp8_moe on CPU with the FlashInfer kernels replaced."""

    def fake_quantize(x, is_sf_swizzled_layout, alignment):
        return torch.zeros_like(x, dtype=torch.uint8), torch.zeros(
            x.shape[0] * x.shape[1] // 32, dtype=torch.uint8
        )

    def fake_moe(**kwargs):
        return torch.full((num_tokens, hidden), 7, dtype=torch.bfloat16)

    monkeypatch.setattr(flashinfer_cutedsl_mxfp8, "mxfp8_quantize", fake_quantize, raising=False)
    monkeypatch.setattr(flashinfer_cutedsl_mxfp8, "cute_dsl_fused_moe", fake_moe, raising=False)
    weights = SimpleNamespace(
        hidden_size=hidden,
        localized_shards=None,
        fc1_data=torch.zeros(2, 1),
        fc1_scale=None,
        fc2_data=None,
        fc2_scale=None,
        alpha=None,
    )
    return flashinfer_cutedsl_mxfp8.flashinfer_cutedsl_mxfp8_moe(
        torch.zeros(num_tokens, hidden, dtype=torch.bfloat16),
        torch.zeros(num_tokens, 2, dtype=torch.int64),
        torch.zeros(num_tokens, 2, dtype=torch.float32),
        weights,
        num_experts=2,
        local_expert_offset=0,
        activation_type=0,
        out=out,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_output_buffer_receives_copy(monkeypatch, dtype):
    out = torch.zeros(8, 64, dtype=dtype)
    result = _run_with_fake_kernels(monkeypatch, out)
    assert result is out
    assert torch.all(out[:3] == 7) and torch.all(out[3:] == 0)


def test_output_buffer_too_small_rejected(monkeypatch):
    with pytest.raises(ValueError, match="cannot hold"):
        _run_with_fake_kernels(monkeypatch, torch.zeros(2, 64, dtype=torch.bfloat16))
