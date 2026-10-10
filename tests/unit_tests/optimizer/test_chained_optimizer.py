# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math

import pytest
import torch

from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.clip_grads import get_grad_norm_fp32
from megatron.core.optimizer.optimizer import ChainedOptimizer, FP32Optimizer
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("grad_norm_group", [None, "mtp"])
def test_chained_optimizer_eager_grad_norm(grad_norm_group):
    """Eager aggregation must preserve the scalar norm and clipping precision."""
    Utils.initialize_model_parallel()
    extra_group = torch.distributed.new_group()
    try:
        config = OptimizerConfig(optimizer="sgd", lr=0.1, optimizer_cuda_graph=False)
        params = [torch.nn.Parameter(torch.zeros(1, device="cuda")) for _ in range(2)]
        optimizers = []
        for param, group in zip(params, [torch.distributed.group.WORLD, extra_group]):
            param.grad = torch.zeros_like(param)
            param.grad_norm_group = grad_norm_group
            optimizer = FP32Optimizer(
                torch.optim.SGD([param], lr=config.lr), config, lambda _: None
            )
            optimizer.grad_stats_parallel_group = group
            optimizers.append(optimizer)
        optimizer = ChainedOptimizer(optimizers)

        for values in [(1.0, 2.0), (0.0, 2.0), (0.0, 0.0)]:
            for param, value in zip(params, values):
                param.grad.fill_(value)
            norms = [
                get_grad_norm_fp32(
                    opt.get_grads_for_grad_norm(grad_norm_group),
                    grad_stats_parallel_group=opt.get_grad_stats_parallel_group(),
                )
                for opt in optimizers
            ]
            expected_norm = math.sqrt(sum((norm if norm else 0.0) ** 2 for norm in norms))
            if grad_norm_group is None:
                actual_norm = optimizer.get_grad_norm()
            else:
                actual_norm = optimizer._get_grad_norm_for_group(grad_norm_group)
            assert isinstance(actual_norm, float)
            assert actual_norm == expected_norm
    finally:
        torch.distributed.destroy_process_group(extra_group)
        Utils.destroy_model_parallel()


def test_chained_optimizer_cuda_graph():
    """Graph replay must match an eager step across distinct gradient-statistics groups."""
    Utils.initialize_model_parallel()
    # A distinct group forces ChainedOptimizer to combine per-optimizer norms.
    # Sharing WORLD would bypass the aggregation that failed during graph capture.
    extra_group = torch.distributed.new_group()
    graph = torch.cuda.CUDAGraph()
    try:
        config = OptimizerConfig(optimizer="sgd", lr=0.1, clip_grad=1.0, optimizer_cuda_graph=True)
        params = [torch.nn.Parameter(torch.zeros(1, device="cuda")) for _ in range(2)]
        optimizers = []
        for param, group in zip(params, [torch.distributed.group.WORLD, extra_group]):
            param.grad = torch.zeros_like(param)
            optimizer = FP32Optimizer(
                torch.optim.SGD([param], lr=config.lr), config, lambda _: None
            )
            optimizer.grad_stats_parallel_group = group
            optimizers.append(optimizer)
        optimizer = ChainedOptimizer(optimizers)

        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())

        # Warm up on the same stream used for capture.
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                for param, value in zip(params, [3.0, 4.0]):
                    param.grad.fill_(value)
                optimizer.step()

        with torch.cuda.graph(graph, stream=capture_stream):
            _, grad_norm, _ = optimizer.step()
        torch.cuda.current_stream().wait_stream(capture_stream)

        for values in [(3.0, 4.0), (0.0, 0.0), (3.0, 4.0)]:
            for param, value in zip(params, values):
                param.data.zero_()
                param.grad.fill_(value)
            _, expected_norm, _ = optimizer.step()
            expected_params = [param.detach().clone() for param in params]

            for param, value in zip(params, values):
                param.data.zero_()
                param.grad.fill_(value)
            graph.replay()
            torch.testing.assert_close(grad_norm, expected_norm)
            torch.testing.assert_close(params, expected_params)
    finally:
        torch.cuda.synchronize()
        graph.reset()
        torch.distributed.destroy_process_group(extra_group)
        Utils.destroy_model_parallel()
