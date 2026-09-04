# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from megatron.core.optimizer import ChainedOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from tests.unit_tests.test_utilities import Utils


def test_grad_norm_skip_threshold_config():
    """Test that grad_norm_skip_threshold config has correct default."""
    config = OptimizerConfig()
    assert config.grad_norm_skip_threshold == float('inf')


def test_default_grad_norm_skip_threshold_does_not_compare_grad_norm():
    """The disabled skip threshold must not inspect a device-backed gradient norm."""

    class UncomparableGradNorm:
        def __gt__(self, _other):
            raise AssertionError(
                "The default infinite threshold should short-circuit the comparison"
            )

    class MockOptimizer:
        def __init__(self):
            self.config = OptimizerConfig(clip_grad=0.0)
            self.param = torch.nn.Parameter(torch.ones(1))
            self.is_stub_optimizer = False
            self.step_called = False

        def prepare_grads(self):
            return False

        def get_grad_norm(self):
            return UncomparableGradNorm()

        def get_parameters(self):
            return [self.param]

        def step_with_ready_grads(self):
            self.step_called = True
            return True

    optimizer = MockOptimizer()

    update_successful, _, _ = ChainedOptimizer([optimizer]).step()

    assert update_successful
    assert optimizer.step_called
