# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU adjoint check for the owned rotary VJP; no deployment forward claim."""
import torch

from megatron.lite.model.deepseek_v41.vllm.attention import _rotate


def test_rotary_vjp_is_transpose_for_query_and_shared_kv():
    torch.manual_seed(413)
    positions = torch.tensor([2, 0, 3])
    angles = torch.randn(4, 4)
    cache = torch.cat((angles.cos(), angles.sin()), -1)
    for shape in ((3, 2, 12), (3, 12)):
        x = torch.randn(shape, requires_grad=True)
        grad = torch.randn(shape)
        y = _rotate(x, positions, cache)
        (actual,) = torch.autograd.grad(y, x, grad)
        expected = _rotate(grad, positions, cache, inverse=True)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(y[..., :4], x[..., :4], rtol=0, atol=0)
