# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from megatron.core.optimizer.clip_grads import zero_grads_manual_impl
from megatron.core.optimizer.optimizer_config import OptimizerConfig


def test_zero_grads_manual_impl():
    """Test that zero_grads_manual_impl zeros out parameter gradients."""
    params = [torch.nn.Parameter(torch.randn(4, 4).cuda()) for _ in range(3)]
    for p in params:
        p.grad = torch.randn_like(p)
        assert p.grad.abs().sum() > 0

    zero_grads_manual_impl(params)

    for p in params:
        assert p.grad.abs().sum() == 0


def test_zero_grads_manual_impl_decoupled():
    """Test zero_grads_manual_impl with decoupled gradients."""
    params = [torch.nn.Parameter(torch.randn(4, 4).cuda()) for _ in range(3)]
    for p in params:
        p.decoupled_grad = torch.randn(4, 4, device='cuda', dtype=torch.float32)
        assert p.decoupled_grad.abs().sum() > 0

    zero_grads_manual_impl(params, use_decoupled_grad=True)

    for p in params:
        assert p.decoupled_grad.abs().sum() == 0


def test_zero_grads_manual_impl_skips_none_grads():
    """Test that zero_grads_manual_impl skips parameters without gradients."""
    p_with_grad = torch.nn.Parameter(torch.randn(4, 4).cuda())
    p_with_grad.grad = torch.randn_like(p_with_grad)
    p_without_grad = torch.nn.Parameter(torch.randn(4, 4).cuda())

    zero_grads_manual_impl([p_with_grad, p_without_grad])

    assert p_with_grad.grad.abs().sum() == 0
    assert p_without_grad.grad is None


def test_grad_norm_skip_threshold_config():
    """Test that grad_norm_skip_threshold config has correct default."""
    config = OptimizerConfig()
    assert config.grad_norm_skip_threshold == 1000
