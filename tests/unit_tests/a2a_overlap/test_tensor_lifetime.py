# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for schedule-local cross-stream tensor lifetime management."""

import gc
import weakref

import pytest
import torch

from megatron.core.pipeline_parallel.tensor_lifetime import ScheduleTensorLifetimeManager
from megatron.core.pipeline_parallel.utils import NoopScheduleNode, ScheduleNode

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.launch_on_gb200,
]


def _owner_bindings(manager):
    return list(manager._owners.values())


def _pending_releases(manager):
    return [entry for entries in manager._pending.values() for entry in entries]


def _publish(manager, tensor, owner, node="producer"):
    manager.consume_inputs_and_publish_outputs(
        (), tensor, stream=owner, node=node, retire_consumed=False
    )


def _consume_inputs_and_publish_outputs(
    manager, consumed, produced, consumer, retire=True, node="consumer"
):
    manager.consume_inputs_and_publish_outputs(
        consumed, produced, stream=consumer, node=node, retire_consumed=retire
    )


def _consume_output_grads_and_publish_input_grads(
    manager, consumed, produced, consumer, node="consumer", forward_outputs=()
):
    manager.consume_forward_outputs(forward_outputs)
    manager.consume_output_grads_and_publish_input_grads(
        consumed, produced, stream=consumer, node=node
    )


def _wait_and_drain(manager, event, stream):
    event.wait(stream)
    with torch.cuda.stream(stream):
        manager.drain(stream)


def test_same_stream_empty_storage_is_immediate():
    owner = torch.cuda.Stream()
    manager = ScheduleTensorLifetimeManager()
    with torch.cuda.stream(owner):
        tensor = torch.empty(1024, device="cuda")
    _publish(manager, tensor, owner)
    _consume_inputs_and_publish_outputs(manager, tensor, (), owner)

    assert tensor.untyped_storage().nbytes() == 0
    assert not _owner_bindings(manager)
    assert not _pending_releases(manager)


def test_cross_stream_release_waits_for_owner_acquire():
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    event = torch.cuda.Event()
    manager = ScheduleTensorLifetimeManager()
    with torch.cuda.stream(owner):
        tensor = torch.empty(1024, device="cuda")
        event.record(owner)
    _publish(manager, tensor, owner)
    event.wait(consumer)
    with torch.cuda.stream(consumer):
        tensor.add_(1)
        event.record(consumer)
    _consume_inputs_and_publish_outputs(manager, tensor, (), consumer)

    assert tensor.untyped_storage().nbytes() > 0
    assert not _owner_bindings(manager)
    assert len(_pending_releases(manager)) == 1

    _wait_and_drain(manager, event, owner)
    assert tensor.untyped_storage().nbytes() == 0
    assert not _pending_releases(manager)


def test_non_retiring_consumer_replaces_input_binding_with_output_binding():
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    manager = ScheduleTensorLifetimeManager()
    with torch.cuda.stream(owner):
        tensor = torch.empty(16, device="cuda")
    with torch.cuda.stream(consumer):
        output = tensor + 1
    _publish(manager, tensor, owner)

    _consume_inputs_and_publish_outputs(manager, tensor, output, consumer, retire=False)

    owners = _owner_bindings(manager)
    assert tensor.untyped_storage().nbytes() > 0
    assert not _pending_releases(manager)
    assert len(owners) == 1
    assert owners[0].tensor is output
    assert owners[0].stream == consumer
    manager.export(output)


@pytest.mark.parametrize("retire_consumed", [False, True])
def test_forward_rejects_partial_slice_alias_output(retire_consumed):
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    manager = ScheduleTensorLifetimeManager()
    with torch.cuda.stream(owner):
        tensor = torch.empty(16, device="cuda")
        output = tensor[2:10]
    _publish(manager, tensor, owner)

    with pytest.raises(RuntimeError, match="does not support storage aliases"):
        _consume_inputs_and_publish_outputs(
            manager, tensor, output, consumer, retire=retire_consumed
        )

    # Alias validation must happen before ownership or storage is mutated.
    owners = _owner_bindings(manager)
    assert len(owners) == 1
    assert owners[0].tensor is tensor
    assert owners[0].stream == owner
    assert tensor.untyped_storage().nbytes() > 0
    manager.export(tensor)


def test_structured_output_storage_aliases_are_rejected():
    stream = torch.cuda.Stream()
    manager = ScheduleTensorLifetimeManager()
    tensor = torch.empty(16, device="cuda")
    output_a = tensor[:8]
    output_b = tensor[4:]

    with pytest.raises(RuntimeError, match="does not support storage aliases"):
        manager.consume_inputs_and_publish_outputs(
            (), (output_a, output_b), stream=stream, node="aliased outputs", retire_consumed=False
        )


def test_structured_tensors_are_deduplicated():
    stream = torch.cuda.Stream()
    manager = ScheduleTensorLifetimeManager()
    tensor = torch.empty(16, device="cuda")

    _publish(manager, (tensor, None, tensor), stream, "structured producer")

    assert len(_owner_bindings(manager)) == 1
    manager.export((tensor, tensor))
    assert not _owner_bindings(manager)


def test_two_managers_do_not_drain_another_microbatch():
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    backward_event = torch.cuda.Event()
    forward_event = torch.cuda.Event()
    backward = ScheduleTensorLifetimeManager()
    forward = ScheduleTensorLifetimeManager()
    with torch.cuda.stream(owner):
        tensor = torch.empty(1024, device="cuda")
        backward_event.record(owner)
    _publish(backward, tensor, owner, node="B producer")
    backward_event.wait(consumer)
    with torch.cuda.stream(consumer):
        tensor.add_(1)
        backward_event.record(consumer)
    _consume_inputs_and_publish_outputs(backward, tensor, (), consumer, node="B MLP")

    # An intervening F dispatch uses the same physical stream but a different
    # microbatch plan.  It must not release B's tensor or add a B -> F edge.
    with torch.cuda.stream(owner):
        forward_event.record(owner)
    _wait_and_drain(forward, forward_event, owner)
    assert tensor.untyped_storage().nbytes() > 0
    assert len(_pending_releases(backward)) == 1

    _wait_and_drain(backward, backward_event, owner)
    assert tensor.untyped_storage().nbytes() == 0
    assert not _pending_releases(backward)


def test_terminal_hand_back_drains_all_owner_streams():
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    event = torch.cuda.Event()
    manager = ScheduleTensorLifetimeManager()
    with torch.cuda.stream(owner):
        tensor = torch.empty(1024, device="cuda")
        event.record(owner)
    _publish(manager, tensor, owner)
    event.wait(consumer)
    with torch.cuda.stream(consumer):
        tensor.add_(1)
        event.record(consumer)
    _consume_inputs_and_publish_outputs(manager, tensor, (), consumer)

    manager.finalize_phase(event, "test")

    assert tensor.untyped_storage().nbytes() == 0
    assert not _pending_releases(manager)


def test_finalize_exports_phase_output():
    owner = torch.cuda.Stream()
    event = torch.cuda.Event()
    manager = ScheduleTensorLifetimeManager()
    with torch.cuda.stream(owner):
        output = torch.empty(16, device="cuda")
        event.record(owner)
    _publish(manager, output, owner)

    manager.finalize_phase(event, "test", outputs=output)

    assert output.untyped_storage().nbytes() > 0
    assert not _owner_bindings(manager)


def test_finalize_rejects_unconsumed_owner_binding():
    owner = torch.cuda.Stream()
    event = torch.cuda.Event()
    manager = ScheduleTensorLifetimeManager()
    tensor = torch.empty(16, device="cuda")
    _publish(manager, tensor, owner, node="forgotten producer")

    with pytest.raises(RuntimeError, match="unconsumed tensor-owner bindings"):
        manager.finalize_phase(event, "test")

    manager.export(tensor)
    manager.finalize_phase(event, "test")


def test_cross_stream_drop_reference_holds_gradient_until_owner_acquire():
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    event = torch.cuda.Event()
    manager = ScheduleTensorLifetimeManager()
    with torch.cuda.stream(owner):
        grad = torch.empty(1024, device="cuda")
        event.record(owner)
    _publish(manager, grad, owner)
    grad_ref = weakref.ref(grad)
    event.wait(consumer)
    with torch.cuda.stream(consumer):
        grad.add_(1)
        event.record(consumer)
    _consume_output_grads_and_publish_input_grads(manager, grad, (), consumer)
    del grad
    gc.collect()

    assert grad_ref() is not None
    _wait_and_drain(manager, event, owner)
    gc.collect()
    assert grad_ref() is None


def test_unknown_external_gradient_uses_record_stream_fallback(monkeypatch):
    consumer = torch.cuda.Stream()
    manager = ScheduleTensorLifetimeManager()
    grad = torch.empty(16, device="cuda")
    recorded = []
    original_record_stream = torch.Tensor.record_stream

    def track_record_stream(tensor, stream):
        recorded.append((tensor, stream))
        return original_record_stream(tensor, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", track_record_stream)

    _consume_output_grads_and_publish_input_grads(manager, grad, (), consumer)

    assert len(recorded) == 1
    assert recorded[0][0] is grad
    assert recorded[0][1] == consumer
    assert not _owner_bindings(manager)
    assert not _pending_releases(manager)


def test_unknown_external_forward_input_preserves_record_stream_fallback(monkeypatch):
    consumer = torch.cuda.Stream()
    manager = ScheduleTensorLifetimeManager()
    tensor = torch.empty(16, device="cuda")
    recorded = []
    original_record_stream = torch.Tensor.record_stream

    def track_record_stream(value, stream):
        recorded.append((value, stream))
        return original_record_stream(value, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", track_record_stream)

    _consume_inputs_and_publish_outputs(manager, tensor, (), consumer)

    assert tensor.untyped_storage().nbytes() == 0
    assert len(recorded) == 1
    assert recorded[0][0] is tensor
    assert recorded[0][1] == consumer
    assert not _owner_bindings(manager)
    assert not _pending_releases(manager)


def test_backward_consumes_recompute_output_binding_and_publishes_gradient(monkeypatch):
    stream = torch.cuda.Stream()
    manager = ScheduleTensorLifetimeManager()
    forward_output = torch.empty(16, device="cuda")
    incoming_grad = torch.empty_like(forward_output)
    produced_grad = torch.empty_like(forward_output)
    _publish(manager, forward_output, stream, node="recompute forward")
    recorded = []
    original_record_stream = torch.Tensor.record_stream

    def track_record_stream(tensor, consumer_stream):
        recorded.append((tensor, consumer_stream))
        return original_record_stream(tensor, consumer_stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", track_record_stream)

    _consume_output_grads_and_publish_input_grads(
        manager, incoming_grad, produced_grad, stream, forward_outputs=forward_output
    )

    owners = _owner_bindings(manager)
    assert len(owners) == 1
    assert owners[0].tensor is produced_grad
    assert len(recorded) == 1
    assert recorded[0][0] is incoming_grad
    assert recorded[0][1] == stream
    manager.export(produced_grad)


def test_backward_rejects_output_and_input_grad_storage_alias():
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    manager = ScheduleTensorLifetimeManager()
    with torch.cuda.stream(owner):
        output_grad = torch.empty(16, device="cuda")
        input_grad = output_grad[2:10]
    _publish(manager, output_grad, owner, node="upstream backward")

    with pytest.raises(RuntimeError, match="does not support storage aliases"):
        _consume_output_grads_and_publish_input_grads(manager, output_grad, input_grad, consumer)

    owners = _owner_bindings(manager)
    assert len(owners) == 1
    assert owners[0].tensor is output_grad
    assert owners[0].stream == owner
    manager.export(output_grad)


def test_noop_preserves_tensor_binding_until_real_consumer():
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    manager = ScheduleTensorLifetimeManager()
    tensor = torch.empty(16, device="cuda")
    _publish(manager, tensor, owner)

    assert NoopScheduleNode().backward(tensor) is tensor
    _consume_output_grads_and_publish_input_grads(manager, tensor, (), consumer)

    assert not _owner_bindings(manager)
    assert len(_pending_releases(manager)) == 1
    assert tuple(manager._pending) == (owner,)


def test_external_consumed_grad_records_stream_in_schedule_node(monkeypatch):
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    event = torch.cuda.Event()
    manager = ScheduleTensorLifetimeManager()
    with torch.cuda.stream(producer):
        leaf = torch.randn(16, device="cuda", requires_grad=True)
        (leaf.square().sum()).backward()
    external_grad = leaf.grad
    node_input = torch.empty(16, device="cuda", requires_grad=True)
    node_output = torch.empty(16, device="cuda", requires_grad=True)
    output_grad = torch.empty_like(node_output)
    node = ScheduleNode(
        lambda value: value,
        consumer,
        event,
        backward_func=lambda _outputs, output_grads: (*output_grads, external_grad),
        lifetime_manager=manager,
    )
    node.inputs = [node_input]
    node.output = node_output
    _publish(manager, node_output, consumer, "forward")
    _publish(manager, output_grad, consumer, "upstream backward")

    recorded = []
    original_record_stream = torch.Tensor.record_stream

    def track_record_stream(tensor, stream):
        recorded.append((tensor, stream))
        return original_record_stream(tensor, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", track_record_stream)

    node.backward(output_grad)

    assert len(recorded) == 1
    assert recorded[0][0] is external_grad
    assert recorded[0][1].cuda_stream == consumer.cuda_stream
    assert not _owner_bindings(manager)
    assert not _pending_releases(manager)


def test_no_external_consumed_grad_skips_record_stream(monkeypatch):
    stream = torch.cuda.Stream()
    event = torch.cuda.Event()
    manager = ScheduleTensorLifetimeManager()
    node_input = torch.empty(16, device="cuda", requires_grad=True)
    node_output = torch.empty(16, device="cuda", requires_grad=True)
    output_grad = torch.empty_like(node_output)
    input_grad = torch.empty_like(node_input)
    node = ScheduleNode(
        lambda value: value,
        stream,
        event,
        backward_func=lambda _outputs, output_grads: output_grads,
        lifetime_manager=manager,
    )
    node.inputs = [node_input]
    node.inputs[0].grad = input_grad
    node.output = node_output
    _publish(manager, node_output, stream, "forward")
    _publish(manager, output_grad, stream, "upstream backward")

    def unexpected_record_stream(_tensor, _stream):
        raise AssertionError("empty external gradient path must not call record_stream")

    monkeypatch.setattr(torch.Tensor, "record_stream", unexpected_record_stream)

    assert node.backward(output_grad) is input_grad
    owners = _owner_bindings(manager)
    assert len(owners) == 1
    assert owners[0].tensor is input_grad
    manager.export(input_grad)


def test_allocator_reuse_poison_does_not_overwrite_delayed_consumer():
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("torch.cuda._sleep is unavailable")

    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    event = torch.cuda.Event()
    manager = ScheduleTensorLifetimeManager()
    observed = []

    for pattern in range(1, 17):
        with torch.cuda.stream(owner):
            tensor = torch.full((256 * 1024,), pattern, dtype=torch.int32, device="cuda")
            event.record(owner)
        _publish(manager, tensor, owner, node=f"producer {pattern}")
        event.wait(consumer)
        with torch.cuda.stream(consumer):
            torch.cuda._sleep(1_000_000)
            observed.append(tensor.clone())
            event.record(consumer)
        _consume_inputs_and_publish_outputs(
            manager, tensor, (), consumer, node=f"consumer {pattern}"
        )

        # The wait is enqueued before drain makes storage allocator-visible.
        _wait_and_drain(manager, event, owner)
        with torch.cuda.stream(owner):
            torch.full((256 * 1024,), -pattern, dtype=torch.int32, device="cuda")

    manager.finalize_phase(event, "test")
    torch.cuda.synchronize()
    for pattern, value in enumerate(observed, start=1):
        assert torch.all(value == pattern)


def test_full_graph_capture_drains_during_capture_not_replay():
    capture_stream = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph, stream=capture_stream):
        event = torch.cuda.Event()
        manager = ScheduleTensorLifetimeManager()
        tensor = torch.full((1024,), 7, dtype=torch.int32, device="cuda")
        event.record(capture_stream)
        _publish(manager, tensor, capture_stream)

        event.wait(consumer)
        with torch.cuda.stream(consumer):
            output = tensor + 1
            event.record(consumer)
        _consume_inputs_and_publish_outputs(manager, tensor, output, consumer)
        _wait_and_drain(manager, event, capture_stream)
        manager.finalize_phase(event, "capture", outputs=output)

    assert not _pending_releases(manager)
    assert not _owner_bindings(manager)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    assert torch.all(output == 8)
    # Python owner bookkeeping only ran during capture, not during replay.
    assert not _pending_releases(manager)
    assert not _owner_bindings(manager)
