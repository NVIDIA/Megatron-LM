# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Owned mHC post/collapse derivatives; native CUDA forward is a separate gate."""
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm.mhc import _Collapse, _Post, collapse, post_mix


@pytest.mark.parametrize("rows", [0, 3])
def test_post_owned_vjp_matches_composed_residual_reference(rows):
    torch.manual_seed(71)
    shapes = [(rows, 1, 16), (rows, 1, 4, 16), (rows, 1, 4), (rows, 1, 4, 4)]
    operands = [torch.randn(s, requires_grad=True) for s in shapes]
    output, residual, post, comb = operands
    mixed = torch.einsum("...ij,...id->...jd", comb, residual)
    target = mixed + post.unsqueeze(-1) * output.unsqueeze(-2)
    grad = torch.randn_like(target)
    expected = torch.autograd.grad(target, operands, grad)
    actual = _Post.backward(SimpleNamespace(saved_tensors=operands), grad)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)


def test_collapse_owned_vjp_keeps_all_stream_dependencies():
    values = torch.randn(3, 1, 4, 16, requires_grad=True)
    pre = torch.randn(3, 1, 4, requires_grad=True)
    grad = torch.randn(3, 1, 16)
    expected = torch.autograd.grad((values * pre.unsqueeze(-1)).sum(-2), (values, pre), grad)
    actual = _Collapse.backward(SimpleNamespace(saved_tensors=(values, pre)), grad)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)


def test_materialized_mhc_boundaries_reject_cpu_before_vllm_import():
    x = torch.zeros(1, 1, 4, 16)
    pre = torch.zeros(1, 1, 4)
    with pytest.raises(RuntimeError, match="CUDA"):
        collapse(x, pre)
    with pytest.raises(RuntimeError, match="CUDA"):
        post_mix(x[..., 0, :], x, pre, torch.zeros(1, 1, 4, 4))
