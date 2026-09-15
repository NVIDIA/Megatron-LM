# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.ssm.gated_delta_net import torch_chunk_gated_delta_rule
from tests.unit_tests.test_utilities import Utils


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('with_initial_state', [False, True])
def test_native_gdn_cuda_graph_replay(dtype, with_initial_state):
    """Capture native GDN and compare changed-input outputs, state and all gradients."""
    torch.cuda.set_device(Utils.local_rank)
    torch.manual_seed(1234)
    inputs = [torch.randn(1, 8, 2, 4, device='cuda', dtype=dtype) * 0.1 for _ in range(3)]
    inputs += [
        -torch.rand(1, 8, 2, device='cuda', dtype=dtype) * 0.1,
        torch.rand(1, 8, 2, device='cuda', dtype=dtype),
    ]
    if with_initial_state:
        inputs.append(torch.randn(1, 2, 4, 4, device='cuda') * 0.1)
    inputs = [value.requires_grad_() for value in inputs]

    def evaluate(values):
        output, state = torch_chunk_gated_delta_rule(
            *values[:5],
            chunk_size=4,
            initial_state=values[5] if with_initial_state else None,
            output_final_state=True,
        )
        gradients = torch.autograd.grad(
            output.float().square().sum() + state.square().sum(), values
        )
        return output, state, gradients

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            evaluate(inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = evaluate(inputs)
    for scale in (0.9, 1.1):
        reference = [(value.detach() * scale).requires_grad_() for value in inputs]
        with torch.no_grad():
            for target, source in zip(inputs, reference):
                target.copy_(source)
        graph.replay()
        torch.cuda.synchronize()
        expected = evaluate(reference)
        torch.testing.assert_close(actual, expected)
        assert actual[1].device == inputs[2].device and actual[1].dtype == torch.float32
