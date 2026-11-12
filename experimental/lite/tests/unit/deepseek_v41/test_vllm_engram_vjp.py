# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Masked Engram rows are exact identity, including their owned gradients."""
import torch

from megatron.lite.model.deepseek_v41.vllm.engram import reference


def test_masked_engram_identity_and_gradient_ownership():
    torch.manual_seed(19)
    h = torch.randn(3, 1, 4, 16, requires_grad=True)
    kv = torch.randn(3, 1, 80, requires_grad=True)
    q = torch.randn(4, 16, requires_grad=True)
    k = torch.randn(4, 16, requires_grad=True)
    mask = torch.tensor([[True], [False], [True]])
    output = reference(h, kv, q, k, 1e-6, mask)
    assert torch.equal(output[1], h[1])
    output.sum().backward()
    assert torch.equal(h.grad[1], torch.ones_like(h[1]))
    assert torch.count_nonzero(kv.grad[1]) == 0
    assert all(torch.isfinite(x.grad).all() and x.grad.count_nonzero() for x in (h, kv, q, k))
