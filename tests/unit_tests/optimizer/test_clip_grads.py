# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from megatron.core.optimizer.optimizer_config import OptimizerConfig


def test_grad_norm_skip_threshold_config():
    """Test that grad_norm_skip_threshold config has correct default."""
    config = OptimizerConfig()
    assert config.grad_norm_skip_threshold == float('inf')
