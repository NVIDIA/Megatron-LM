# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
from torch import inf

from megatron.core.optimizer.clip_grads import get_grad_norm_fp32
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from tests.unit_tests.test_utilities import Utils


def test_grad_norm_skip_threshold_config():
    """Test that grad_norm_skip_threshold config has correct default."""
    config = OptimizerConfig()
    assert config.grad_norm_skip_threshold == float('inf')


def _as_float(total_norm):
    """total_norm may be returned as a 0-d/1-element tensor or a python float
    depending on whether transformer_engine's multi_tensor_scale_tensor is
    available; normalize before comparing in tests."""
    return total_norm.item() if torch.is_tensor(total_norm) else total_norm


class TestGetGradNormFp32:
    """Regression tests for get_grad_norm_fp32, including the empty
    grads_for_norm crash fixed in #5530 (issue #5529)."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel(1, 1)
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("norm_type", [2.0, inf, 1.0])
    def test_empty_grads_returns_zero(self, norm_type):
        """An empty grads_for_norm must return 0.0, not raise.

        Previously: norm_type == inf raised ValueError (max() over an
        empty generator), and norm_type not in {2.0, inf} raised TypeError
        (a python float was passed to torch.distributed.all_reduce).
        """
        total_norm = get_grad_norm_fp32([], norm_type=norm_type)
        assert _as_float(total_norm) == 0.0

    @pytest.mark.parametrize("norm_type", [2.0, inf, 1.0])
    def test_nonempty_grads_matches_reference(self, norm_type):
        """Non-empty path should match a plain torch.norm-based computation."""
        torch.manual_seed(0)
        grads = [torch.randn(4, 4, device='cuda') for _ in range(3)]

        total_norm = _as_float(get_grad_norm_fp32(grads, norm_type=norm_type))

        if norm_type == inf:
            expected = max(g.abs().max() for g in grads).item()
        else:
            expected = sum(g.norm(norm_type) ** norm_type for g in grads).item() ** (
                1.0 / norm_type
            )
        assert total_norm == pytest.approx(expected, rel=1e-5)
