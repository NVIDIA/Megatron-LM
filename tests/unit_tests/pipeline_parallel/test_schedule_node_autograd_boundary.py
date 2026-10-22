# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import contextmanager

import pytest
import torch

from megatron.core.pipeline_parallel.utils import ScheduleNode


class _TestScheduleNode(ScheduleNode):
    """ScheduleNode test double that omits event synchronization."""

    @contextmanager
    def stream_acquire_context(self, name=None):
        yield


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ScheduleNode requires CUDA")
def test_input_wrapper_does_not_retain_node_inputs_or_outputs():
    """Input wrappers preserve segmented backward without retaining real tensors."""
    weight_1 = torch.tensor(3.0, device="cuda", requires_grad=True)
    weight_2 = torch.tensor(5.0, device="cuda", requires_grad=True)
    input_tensor = torch.tensor([2.0, 4.0], device="cuda", requires_grad=True)
    node_2_backward_called = False

    def backward(output_edges, output_grads):
        torch.autograd.backward(output_edges, grad_tensors=output_grads)
        return output_grads

    def node_2_backward(output_edges, output_grads):
        nonlocal node_2_backward_called
        node_2_backward_called = True
        return backward(output_edges, output_grads)

    stream = torch.cuda.current_stream()
    node_1 = _TestScheduleNode(
        lambda tensor: tensor * weight_1,
        stream=stream,
        event=None,
        backward_func=backward,
        name="node_1",
    )
    node_2 = _TestScheduleNode(
        lambda tensor: tensor * weight_2,
        stream=stream,
        event=None,
        backward_func=node_2_backward,
        name="node_2",
    )

    intermediate = node_1.forward(input_tensor)
    output = node_2.forward(intermediate)

    assert type(intermediate) is torch.Tensor
    assert type(output) is torch.Tensor

    # Each node retains only its storage-free input-gradient leaf.
    node_1_wrapper = node_1._input_wrappers[0]
    node_2_wrapper = node_2._input_wrappers[0]
    assert node_1_wrapper._real_tensor is None
    assert node_2_wrapper._real_tensor is None
    assert node_1_wrapper.shape == input_tensor.shape
    assert node_2_wrapper.shape == input_tensor.shape
    assert node_1_wrapper.untyped_storage().nbytes() == 0
    assert node_2_wrapper.untyped_storage().nbytes() == 0
    assert len(node_1._output_edges) == 1
    assert len(node_2._output_edges) == 1

    node_2_input_grad = node_2.backward(torch.ones_like(output))
    assert node_2_backward_called
    assert torch.equal(node_2_input_grad, torch.full_like(input_tensor, 5.0))
    assert torch.equal(weight_2.grad, torch.full_like(weight_2, 18.0))
    assert weight_1.grad is None

    node_1_input_grad = node_1.backward(node_2_input_grad)
    assert torch.equal(node_1_input_grad, torch.full_like(input_tensor, 15.0))
    assert torch.equal(weight_1.grad, torch.full_like(weight_1, 30.0))
    # The first boundary deliberately stops traversal into the caller's graph.
    assert input_tensor.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ScheduleNode requires CUDA")
def test_backward_ignores_non_differentiable_outputs():
    """Output gradients remain positionally aligned when an output has no autograd edge."""
    weight = torch.tensor(3.0, device="cuda", requires_grad=True)
    input_tensor = torch.tensor([2.0, 4.0], device="cuda", requires_grad=True)
    node = _TestScheduleNode(
        lambda tensor: (tensor * weight, tensor.detach()),
        stream=torch.cuda.current_stream(),
        event=None,
    )

    differentiable_output, detached_output = node.forward(input_tensor)
    assert node._output_edges[0] is not None
    assert node._output_edges[1] is None

    input_grad = node.backward(
        (torch.ones_like(differentiable_output), torch.ones_like(detached_output))
    )
    assert torch.equal(input_grad, torch.full_like(input_tensor, 3.0))
    assert torch.equal(weight.grad, torch.full_like(weight, 6.0))
