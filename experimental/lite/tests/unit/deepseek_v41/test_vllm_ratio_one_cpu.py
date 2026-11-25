# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Literal ratio-one metadata and the independently evaluated RMS derivative."""
import torch

from megatron.lite.model.deepseek_v41.vllm.compressed import metadata, pool_vjp


def test_ratio_one_rows_and_norm_vjp():
    m = metadata(7, 1, 'cpu')
    assert torch.equal(m.slots, torch.arange(7)) and m.lengths.tolist() == [7]
    torch.manual_seed(35)
    raw = torch.randn(7, 512, requires_grad=True)
    weight = torch.randn(512, requires_grad=True)
    grad = torch.randn_like(raw)
    output = torch.nn.functional.rms_norm(raw, (512,), weight, 1e-6)
    expected = torch.autograd.grad(output, (raw, weight), grad)
    actual = pool_vjp(raw, weight, 1, grad, 1e-6)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
