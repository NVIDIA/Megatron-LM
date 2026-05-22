# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch
import torch.nn.functional as F

from megatron.core.activations import fast_gelu, quick_gelu, squared_relu


def test_squared_relu_matches_relu_squared():
    x = torch.tensor([-2.0, -0.5, 0.0, 1.5])

    assert torch.equal(squared_relu(x), torch.square(F.relu(x)))


def test_quick_gelu_matches_formula():
    x = torch.tensor([-2.0, 0.0, 2.0])

    assert torch.allclose(quick_gelu(x), x * torch.sigmoid(1.702 * x))


def test_fast_gelu_matches_formula():
    x = torch.tensor([-2.0, 0.0, 2.0])
    expected = 0.5 * x * (1.0 + torch.tanh(x * 0.79788456 * (1.0 + 0.044715 * x * x)))

    assert torch.allclose(fast_gelu(x), expected)
