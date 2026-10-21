# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.optimizer.clip_grads import (
    clip_grad_by_total_norm_fp32,
    count_zeros_fp32,
    multi_tensor_scale_tensor_impl,
)
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


class TestCountZerosFp32GtpPadding:
    """count_zeros_fp32 must exclude GTP alignment-padding rows: they are structural zeros
    (never written by the wgrad GEMM, see generalized_tensor_parallelism), not real zero
    gradients, so counting them inflates GTP's num_zeros relative to 3D. The distributed
    optimizer stamps the per-shard pad-element count onto `.gtp_pad_zeros`
    (tensor_parallel.gtp_local_pad_zero_count) for count_zeros_fp32 to subtract."""

    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def _make_param(self, grad, gtp_pad_zeros=0):
        param = torch.nn.Parameter(torch.zeros_like(grad))
        param.grad = grad
        if gtp_pad_zeros:
            param.gtp_pad_zeros = gtp_pad_zeros
        return param

    def test_padding_elements_excluded_from_zero_count(self):
        grad = torch.zeros(8, 4, device='cuda')
        grad[0, 0] = 1.0
        # Last 3 rows (12 elements) are structural GTP padding, not real zero gradient.
        param = self._make_param(grad, gtp_pad_zeros=12)

        num_zeros = count_zeros_fp32(
            [param], grad_stats_parallel_group=parallel_state.get_model_parallel_group()
        )

        assert num_zeros == grad.numel() - 1 - 12

    def test_no_gtp_pad_zeros_attribute_counts_all_zeros(self):
        """Non-GTP params (no .gtp_pad_zeros stamped) are unaffected by the correction."""
        grad = torch.zeros(8, 4, device='cuda')
        grad[0, 0] = 1.0
        param = self._make_param(grad)

        num_zeros = count_zeros_fp32(
            [param], grad_stats_parallel_group=parallel_state.get_model_parallel_group()
        )

        assert num_zeros == grad.numel() - 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="clipping kernels need a GPU")
@pytest.mark.skipif(
    multi_tensor_scale_tensor_impl is None, reason="tensor-scale kernel (Transformer Engine) needed"
)
# 0.3 is not a power of two: a reciprocal-and-multiply coefficient would differ from the division.
@pytest.mark.parametrize("max_norm", [1.0, 0.5, 0.3])
def test_float64_device_norm_clips_like_the_python_float_path(max_norm):
    """ChainedOptimizer combines its optimizers' norms into a float64 device tensor (so that an
    optimizer-step CUDA graph capture has no host synchronization). Clipping with that tensor must
    scale the gradients bit for bit like the Python-float path the golden values were made with."""
    torch.manual_seed(0)
    for trial in range(32):
        # Norms straddle max_norm so both the clipped and the unclipped branch are exercised.
        grads = [
            torch.randn(size, device="cuda") * (0.02 + 0.06 * (trial % 4))
            for size in (7, 128, 1000)
        ]
        # Per-optimizer norms as get_grad_norm_fp32 returns them: float32 tensors of shape (1,).
        norms = [torch.linalg.vector_norm(g).reshape(1) for g in grads]
        squares = sum(x**2 for x in norms)
        python_norm = math.sqrt(squares)
        device_norm = torch.sqrt(squares.to(torch.float64))
        assert device_norm.item() == python_norm

        def clip(total_norm):
            params = []
            for g in grads:
                p = torch.nn.Parameter(torch.zeros_like(g))
                p.grad = g.clone()
                params.append(p)
            clip_grad_by_total_norm_fp32(params, max_norm, total_norm)
            return [p.grad for p in params]

        for got, want in zip(clip(device_norm), clip(python_norm)):
            assert torch.equal(got, want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device norms need a GPU")
def test_chained_optimizer_combines_device_norms_in_float64():
    """Optimizers with separate grad-stats groups (e.g. Muon + Adam): the combined norm is the
    float64 root of the fp32 sum of squares, taken on the device (no host synchronization) with
    exactly the value of the Python-float form ``math.sqrt(float(sum))``."""

    class MockOptimizer:
        def __init__(self, norm, group):
            self.config = OptimizerConfig(clip_grad=1.0)
            self.norm = torch.tensor([norm], device="cuda", dtype=torch.float32)
            self.group = group
            self.is_stub_optimizer = False
            self.model_chunks = []

        def get_grad_stats_parallel_group(self):
            return self.group

        def get_grad_norm(self):
            return self.norm

    chained = ChainedOptimizer([MockOptimizer(1.7, "dp"), MockOptimizer(0.3, "dp-ep")])
    norm = chained.get_grad_norm()
    assert isinstance(norm, torch.Tensor) and norm.dtype == torch.float64
    squares = (
        torch.tensor([1.7], dtype=torch.float32) ** 2
        + torch.tensor([0.3], dtype=torch.float32) ** 2
    )
    assert norm.item() == math.sqrt(squares)


def test_chained_optimizer_combines_python_float_norms_and_skips_missing_ones():
    """Optimizers that report Python floats keep the ``math.sqrt`` form; optimizers that report no
    norm (``None``) are left out, and no norms at all give 0.0."""

    class MockOptimizer:
        def __init__(self, norm, group):
            self.config = OptimizerConfig(clip_grad=1.0)
            self.norm = norm
            self.group = group
            self.is_stub_optimizer = False
            self.model_chunks = []

        def get_grad_stats_parallel_group(self):
            return self.group

        def get_grad_norm(self):
            return self.norm

    norm = ChainedOptimizer([MockOptimizer(3.0, "dp"), MockOptimizer(4.0, "dp-ep")]).get_grad_norm()
    assert isinstance(norm, float) and norm == 5.0
    assert (
        ChainedOptimizer([MockOptimizer(None, "dp"), MockOptimizer(4.0, "dp-ep")]).get_grad_norm()
        == 4.0
    )
    assert (
        ChainedOptimizer([MockOptimizer(None, "dp"), MockOptimizer(None, "dp-ep")]).get_grad_norm()
        == 0.0
    )
