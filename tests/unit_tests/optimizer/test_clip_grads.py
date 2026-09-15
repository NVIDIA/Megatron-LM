# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.optimizer import clip_grads
from megatron.core.optimizer.clip_grads import count_zeros_fp32
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from tests.unit_tests.test_utilities import Utils


def test_grad_norm_skip_threshold_config():
    """Test that grad_norm_skip_threshold config has correct default."""
    config = OptimizerConfig()
    assert config.grad_norm_skip_threshold == float('inf')


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


_DTYPES = [
    (torch.bfloat16, torch.float32),
    (torch.float32, torch.bfloat16),
    (torch.bfloat16, torch.bfloat16),
    (torch.float32, torch.float32),
]


def _padded_grads(dtypes, numel, value=1.0 / 64):
    # Keep a regressed fused kernel's wider dtype interpretation inside its allocation.
    # Padding also exposes writes beyond the logical gradient view.
    bases = [torch.ones(4 * numel, device='cuda', dtype=dtype) for dtype in dtypes]
    grads = [base[:numel].fill_(value) for base in bases]
    return bases, grads


@pytest.mark.skipif(
    clip_grads.l2_norm_impl.__name__ == 'local_multi_tensor_l2_norm',
    reason='Requires fused multi-tensor kernels; the Python fallback already accepts mixed lists',
)
class TestMixedDtypeGradNormAndClip:
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('dtypes', _DTYPES)
    @pytest.mark.parametrize('numel', [128, 130])
    @pytest.mark.parametrize('layout', ['both', 'rank0_empty', 'disjoint', 'all_empty'])
    def test_grad_norm(self, dtypes, numel, layout):
        bases, grads = _padded_grads(dtypes, numel)
        rank = torch.distributed.get_rank()
        if layout == 'all_empty' or (layout == 'rank0_empty' and rank == 0):
            grads = []
        elif layout == 'disjoint':
            grads = [grads[rank % 2]]
        squared_norm = torch.zeros(1, device='cuda', dtype=torch.float64)
        for grad in grads:
            squared_norm += grad.double().square().sum()
        torch.distributed.all_reduce(squared_norm)

        actual = clip_grads.get_grad_norm_fp32(
            grads, grad_stats_parallel_group=torch.distributed.group.WORLD
        )

        assert float(actual) == pytest.approx(float(squared_norm.sqrt()), rel=2e-6, abs=1e-8)
        for base in bases:
            assert torch.equal(base[numel:], torch.ones_like(base[numel:]))

    @pytest.mark.parametrize('dtypes', _DTYPES[:2])
    def test_zero_grad_norm_ignores_padding(self, dtypes):
        bases, grads = _padded_grads(dtypes, 128, value=0.0)
        actual = clip_grads.get_grad_norm_fp32(
            grads, grad_stats_parallel_group=torch.distributed.group.WORLD
        )
        assert float(actual) == 0.0
        assert all(torch.all(base[128:] == 1) for base in bases)

    @pytest.mark.parametrize('dtypes', _DTYPES)
    @pytest.mark.parametrize('numel', [127, 130])
    @pytest.mark.parametrize('use_decoupled_grad', [False, True])
    @pytest.mark.parametrize('tensor_coefficient', [False, True])
    @pytest.mark.parametrize('clip', [False, True])
    def test_clip_gradients(self, dtypes, numel, use_decoupled_grad, tensor_coefficient, clip):
        if tensor_coefficient and clip_grads.multi_tensor_scale_tensor_impl is None:
            pytest.skip('Backend has no tensor-coefficient scaling API')
        bases, grads = _padded_grads(dtypes, numel)
        params = []
        for grad in grads:
            # The dtype of a decoupled gradient need not match its parameter's dtype.
            dtype = torch.bfloat16 if use_decoupled_grad else grad.dtype
            param = torch.nn.Parameter(torch.zeros_like(grad, dtype=dtype))
            if use_decoupled_grad:
                param.decoupled_grad = grad
            else:
                param.grad = grad
            params.append(param)
        params.append(torch.nn.Parameter(torch.zeros(1, device='cuda')))  # No gradient.
        total_norm = torch.tensor([2.0], device='cuda') if tensor_coefficient else 2.0
        max_norm = 1.0000005 if clip else 4.0
        coefficient = min(float(max_norm / (total_norm + 1e-6)), 1.0)
        expected = [base.clone() for base in bases]
        for base in expected:
            base[:numel].mul_(coefficient)

        clip_grads.clip_grad_by_total_norm_fp32(
            params, max_norm, total_norm, use_decoupled_grad=use_decoupled_grad
        )

        for actual, reference in zip(bases, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    @pytest.mark.parametrize('use_decoupled_grad', [False, True])
    @pytest.mark.parametrize('tensor_coefficient', [False, True])
    def test_empty_gradients_do_not_launch_kernel(
        self, monkeypatch, use_decoupled_grad, tensor_coefficient
    ):
        if tensor_coefficient and clip_grads.multi_tensor_scale_tensor_impl is None:
            pytest.skip('Backend has no tensor-coefficient scaling API')

        def unexpected_launch(*args, **kwargs):
            pytest.fail('An empty gradient list must not reach a fused kernel')

        # Fail before dispatch: some native backends dereference the first list entry.
        monkeypatch.setattr(clip_grads, 'multi_tensor_applier', unexpected_launch)
        total_norm = torch.tensor([2.0], device='cuda') if tensor_coefficient else 2.0
        clip_grads.clip_grad_by_total_norm_fp32(
            [], 1.0, total_norm, use_decoupled_grad=use_decoupled_grad
        )
