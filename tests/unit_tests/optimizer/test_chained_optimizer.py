# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.optimizer import ChainedOptimizer, FP32Optimizer
from tests.unit_tests.test_utilities import Utils


def test_chained_optimizer_cuda_graph():
    """Capture and replay clipping across distinct gradient-statistics process groups."""
    Utils.initialize_model_parallel()
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

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                for param, value in zip(params, [3.0, 4.0]):
                    param.grad.fill_(value)
                optimizer.step()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph, stream=stream):
            _, grad_norm, _ = optimizer.step()

        for values, local_norm in [((3.0, 4.0), 5.0), ((0.0, 0.0), 0.0), ((3.0, 4.0), 5.0)]:
            for param, value in zip(params, values):
                param.data.zero_()
                param.grad.fill_(value)
            graph.replay()
            expected_norm = local_norm * torch.distributed.get_world_size() ** 0.5
            torch.testing.assert_close(grad_norm, torch.full_like(grad_norm, expected_norm))
            coefficient = min(1.0, config.clip_grad / (expected_norm + 1e-6))
            for param, value in zip(params, values):
                torch.testing.assert_close(
                    param, torch.full_like(param, -config.lr * value * coefficient)
                )
    finally:
        torch.cuda.synchronize()
        graph.reset()
        torch.distributed.destroy_process_group(extra_group)
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("norms, expected", [([3.0, 4.0], 5.0), ([0.0, 0.0], 0.0)])
def test_chained_optimizer_float_norms(norms, expected):
    """Backends returning Python floats retain the same combined norm."""
    optimizers = [
        SimpleNamespace(
            get_grad_norm=lambda norm=norm: norm,
            get_grad_stats_parallel_group=lambda group=object(): group,
        )
        for norm in norms
    ]
    assert ChainedOptimizer(optimizers).get_grad_norm() == expected
