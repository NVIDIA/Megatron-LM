# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The active projection VJP agrees with autograd through rounded operands."""
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm.linear import _BF16Linear


def test_head_projection_vjp_uses_compute_weight_and_master_dtype():
    torch.manual_seed(71)
    x = torch.randn(5, 7, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(11, 7, dtype=torch.float32, requires_grad=True)
    grad = torch.randn(5, 11)
    compute = weight + (weight.bfloat16().float() - weight).detach()
    output = torch.nn.functional.linear(x.float(), compute)
    expected = torch.autograd.grad(output, (x, weight), grad)
    ctx = SimpleNamespace(
        saved_tensors=(x.detach(), weight.detach().bfloat16()), weight_dtype=weight.dtype
    )
    actual = _BF16Linear.backward(ctx, grad)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert actual[0].dtype == x.dtype and actual[1].dtype == weight.dtype


def test_deployment_projection_rejects_cpu_forward():
    with pytest.raises(RuntimeError, match='requires CUDA'):
        _BF16Linear.apply(torch.ones(2, 8), torch.ones(16, 8))
