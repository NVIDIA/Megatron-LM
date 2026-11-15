# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from megatron.core import parallel_state
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


class TestCountZerosFp32UnderCudaGraphCapture:
    """Inside the optimizer CUDA graph the count cannot be read on the host: the function
    returns the device tensor, which every replay refreshes."""

    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_returns_host_number_outside_capture(self):
        grad = torch.zeros(8, 4, device='cuda')
        grad[0, 0] = 1.0
        param = torch.nn.Parameter(torch.zeros_like(grad))
        param.grad = grad

        num_zeros = count_zeros_fp32(
            [param], grad_stats_parallel_group=parallel_state.get_model_parallel_group()
        )

        assert not isinstance(num_zeros, torch.Tensor)
        assert num_zeros == grad.numel() - 1

    def test_returns_device_tensor_refreshed_by_replay(self, monkeypatch):
        grad = torch.zeros(8, 4, device='cuda')
        grad[0, 0] = 1.0
        param = torch.nn.Parameter(torch.zeros_like(grad))
        param.grad = grad
        group = parallel_state.get_model_parallel_group()
        # On one rank the reduction is the identity. Capturing a collective needs a
        # communicator set up for it, which this test is not about.
        monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, **_: None)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            num_zeros = count_zeros_fp32([param], grad_stats_parallel_group=group)
        assert isinstance(num_zeros, torch.Tensor) and num_zeros.is_cuda

        graph.replay()
        assert num_zeros.item() == grad.numel() - 1

        # The same tensor reports the new gradient after the next replay.
        grad[1, :] = 1.0
        graph.replay()
        assert num_zeros.item() == grad.numel() - 5
