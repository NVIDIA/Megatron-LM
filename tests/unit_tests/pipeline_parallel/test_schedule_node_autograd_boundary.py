# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import contextmanager

import pytest
import torch

from megatron.core.models.common.utils import TransformerLayerNode
from megatron.core.pipeline_parallel.utils import ScheduleNode, ScheduleNodeInputWrapper


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
        free_input=True,
        name="node_1",
    )
    node_2 = _TestScheduleNode(
        lambda tensor: tensor * weight_2,
        stream=stream,
        event=None,
        backward_func=node_2_backward,
        free_input=True,
        name="node_2",
    )

    intermediate = node_1.forward(input_tensor)
    output = node_2.forward(intermediate)

    assert type(intermediate) is torch.Tensor
    assert type(output) is torch.Tensor

    # Each node retains only its storage-free input-gradient leaf.
    node_1_wrapper = node_1.inputs[0]
    node_2_wrapper = node_2.inputs[0]
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
    assert node_2.inputs is None
    assert torch.equal(node_2_input_grad, torch.full_like(input_tensor, 5.0))
    assert torch.equal(weight_2.grad, torch.full_like(weight_2, 18.0))
    assert weight_1.grad is None

    node_1_input_grad = node_1.backward(node_2_input_grad)
    assert node_1.inputs is None
    assert torch.equal(node_1_input_grad, torch.full_like(input_tensor, 15.0))
    assert torch.equal(weight_1.grad, torch.full_like(weight_1, 30.0))
    # The first boundary deliberately stops traversal into the caller's graph.
    assert input_tensor.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ScheduleNode requires CUDA")
def test_retained_input_uses_an_ordinary_detached_leaf():
    """Non-releasable boundaries retain input storage without wrapper dispatch."""
    weight = torch.tensor(3.0, device="cuda", requires_grad=True)
    input_tensor = torch.tensor([2.0, 4.0], device="cuda", requires_grad=True)
    node = _TestScheduleNode(
        lambda tensor: tensor * weight,
        stream=torch.cuda.current_stream(),
        event=None,
        free_input=False,
    )

    output = node.forward(input_tensor)
    grad_source = node.inputs[0]
    assert type(grad_source) is torch.Tensor
    assert not isinstance(grad_source, ScheduleNodeInputWrapper)
    assert grad_source.is_leaf
    assert grad_source.untyped_storage().nbytes() == input_tensor.untyped_storage().nbytes()

    input_grad = node.backward(torch.ones_like(output))
    assert torch.equal(input_grad, torch.full_like(input_tensor, 3.0))
    assert torch.equal(weight.grad, torch.full_like(weight, 6.0))
    assert input_tensor.grad is None


def test_side_output_uses_an_ordinary_detached_leaf():
    """Side-output gradients cross the node boundary without another wrapper node."""
    input_tensor = torch.tensor([2.0, 4.0], requires_grad=True)
    produced = input_tensor * 2.0
    node = object.__new__(TransformerLayerNode)
    node.detached = tuple()
    node.before_detached = tuple()

    detached = node.detach(produced)
    assert type(detached) is torch.Tensor
    assert not isinstance(detached, ScheduleNodeInputWrapper)
    assert detached.is_leaf

    (detached * 3.0).sum().backward()
    detached_grads = node.backward_impl(tuple(), tuple())

    assert torch.equal(detached_grads[0], torch.full_like(input_tensor, 3.0))
    assert torch.equal(input_tensor.grad, torch.full_like(input_tensor, 6.0))
    assert node.detached == tuple()
    assert node.before_detached == tuple()


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ScheduleNode requires CUDA")
@pytest.mark.parametrize("free_input", [False, True])
def test_input_storage_survives_cross_stream_backward(monkeypatch, free_input):
    """Input storage stays live until its consumer's backward kernels finish."""
    torch.cuda.synchronize()
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    event = torch.cuda.Event()
    size = 1024
    with torch.cuda.stream(consumer):
        weight = torch.ones((size, size), device="cuda", requires_grad=True)
        # Initialize cuBLAS before delaying the asynchronous backward below.
        warmup = torch.ones_like(weight, requires_grad=True)
        (warmup @ weight).backward(torch.ones_like(weight))
        weight.grad = None
        del warmup
    torch.cuda.synchronize()
    with torch.cuda.stream(producer):
        input_tensor = torch.ones_like(weight, requires_grad=True)
        input_pointer = input_tensor.data_ptr()
        event.record()

    recorded_consumer = False
    original_record_stream = torch.Tensor.record_stream

    def record_stream(tensor, stream):
        nonlocal recorded_consumer
        # Retain metadata only: saving the tensor here would conceal premature frees.
        if tensor.data_ptr() == input_pointer and stream == consumer:
            recorded_consumer = True
        return original_record_stream(tensor, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", record_stream)
    node = ScheduleNode(
        lambda tensor: tensor @ weight, stream=consumer, event=event, free_input=free_input
    )
    output = node.forward(input_tensor)
    # Check the lifetime contract independently of allocator reuse and GPU timing.
    assert recorded_consumer
    with torch.cuda.stream(consumer):
        output_grad = torch.ones_like(output)
    del output, input_tensor
    with torch.cuda.stream(consumer):
        torch.cuda._sleep(200_000_000)
    input_grad = node.backward(output_grad)
    with torch.cuda.stream(producer):
        # Reuse pressure on the allocating stream must not overwrite backward inputs.
        replacements = [torch.empty_like(weight) for _ in range(16)]
        for replacement in replacements:
            replacement.fill_(float("nan"))
    torch.cuda.synchronize()
    assert torch.equal(weight.grad, torch.full_like(weight, size))
    assert torch.equal(input_grad, torch.full_like(input_grad, size))
