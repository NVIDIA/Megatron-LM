# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for combined 1F1B scheduled cross-stream tensor release."""

import gc
import weakref

import pytest
import torch

from megatron.core.pipeline_parallel.combined_1f1b_tensor_release import Combined1F1BTensorRelease
from megatron.core.pipeline_parallel.utils import NoopScheduleNode, ScheduleNode

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.launch_on_gb200,
]


def _owner_bindings(tensor_release):
    return list(tensor_release._owners.values())


def _pending_releases(tensor_release):
    return [entry for entries in tensor_release._pending.values() for entry in entries]


def _publish(tensor_release, tensor, owner, node="producer"):
    tensor_release.consume_inputs_and_publish_outputs(
        (), tensor, stream=owner, node=node, release_consumed=False
    )


def _consume_inputs_and_publish_outputs(
    tensor_release, consumed, produced, consumer, release=True, node="consumer"
):
    tensor_release.consume_inputs_and_publish_outputs(
        consumed, produced, stream=consumer, node=node, release_consumed=release
    )


def _consume_output_grads_and_publish_input_grads(
    tensor_release, consumed, produced, consumer, node="consumer", forward_outputs=()
):
    tensor_release.consume_forward_outputs(forward_outputs)
    tensor_release.consume_output_grads_and_publish_input_grads(
        consumed, produced, stream=consumer, node=node
    )


def _wait_and_drain(tensor_release, event, stream):
    event.wait(stream)
    with torch.cuda.stream(stream):
        tensor_release.drain(stream)


def test_same_stream_empty_storage_is_immediate():
    owner = torch.cuda.Stream()
    tensor_release = Combined1F1BTensorRelease()
    with torch.cuda.stream(owner):
        tensor = torch.empty(1024, device="cuda")
    _publish(tensor_release, tensor, owner)
    _consume_inputs_and_publish_outputs(tensor_release, tensor, (), owner)

    assert tensor.untyped_storage().nbytes() == 0
    assert not _owner_bindings(tensor_release)
    assert not _pending_releases(tensor_release)


def test_cross_stream_release_waits_for_owner_acquire():
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    event = torch.cuda.Event()
    tensor_release = Combined1F1BTensorRelease()
    with torch.cuda.stream(owner):
        tensor = torch.empty(1024, device="cuda")
        event.record(owner)
    _publish(tensor_release, tensor, owner)
    event.wait(consumer)
    with torch.cuda.stream(consumer):
        tensor.add_(1)
        event.record(consumer)
    _consume_inputs_and_publish_outputs(tensor_release, tensor, (), consumer)

    assert tensor.untyped_storage().nbytes() > 0
    assert not _owner_bindings(tensor_release)
    assert len(_pending_releases(tensor_release)) == 1

    _wait_and_drain(tensor_release, event, owner)
    assert tensor.untyped_storage().nbytes() == 0
    assert not _pending_releases(tensor_release)


def test_non_releasing_consumer_replaces_input_binding_with_output_binding():
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    tensor_release = Combined1F1BTensorRelease()
    with torch.cuda.stream(owner):
        tensor = torch.empty(16, device="cuda")
    with torch.cuda.stream(consumer):
        output = tensor + 1
    _publish(tensor_release, tensor, owner)

    _consume_inputs_and_publish_outputs(tensor_release, tensor, output, consumer, release=False)

    owners = _owner_bindings(tensor_release)
    assert tensor.untyped_storage().nbytes() > 0
    assert not _pending_releases(tensor_release)
    assert len(owners) == 1
    assert owners[0].tensor is output
    assert owners[0].stream == consumer
    tensor_release.export(output)


@pytest.mark.parametrize("release_consumed", [False, True])
def test_forward_rejects_partial_slice_alias_output(release_consumed):
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    tensor_release = Combined1F1BTensorRelease()
    with torch.cuda.stream(owner):
        tensor = torch.empty(16, device="cuda")
        output = tensor[2:10]
    _publish(tensor_release, tensor, owner)

    with pytest.raises(RuntimeError, match="does not support storage aliases"):
        _consume_inputs_and_publish_outputs(
            tensor_release, tensor, output, consumer, release=release_consumed
        )

    # Alias validation must happen before ownership or storage is mutated.
    owners = _owner_bindings(tensor_release)
    assert len(owners) == 1
    assert owners[0].tensor is tensor
    assert owners[0].stream == owner
    assert tensor.untyped_storage().nbytes() > 0
    tensor_release.export(tensor)


def test_structured_output_storage_aliases_are_rejected():
    stream = torch.cuda.Stream()
    tensor_release = Combined1F1BTensorRelease()
    tensor = torch.empty(16, device="cuda")
    output_a = tensor[:8]
    output_b = tensor[4:]

    with pytest.raises(RuntimeError, match="does not support storage aliases"):
        tensor_release.consume_inputs_and_publish_outputs(
            (), (output_a, output_b), stream=stream, node="aliased outputs", release_consumed=False
        )


def test_structured_tensors_are_deduplicated():
    stream = torch.cuda.Stream()
    tensor_release = Combined1F1BTensorRelease()
    tensor = torch.empty(16, device="cuda")

    _publish(tensor_release, (tensor, None, tensor), stream, "structured producer")

    assert len(_owner_bindings(tensor_release)) == 1
    tensor_release.export((tensor, tensor))
    assert not _owner_bindings(tensor_release)


def test_two_managers_do_not_drain_another_microbatch():
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    backward_event = torch.cuda.Event()
    forward_event = torch.cuda.Event()
    backward = Combined1F1BTensorRelease()
    forward = Combined1F1BTensorRelease()
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
    tensor_release = Combined1F1BTensorRelease()
    with torch.cuda.stream(owner):
        tensor = torch.empty(1024, device="cuda")
        event.record(owner)
    _publish(tensor_release, tensor, owner)
    event.wait(consumer)
    with torch.cuda.stream(consumer):
        tensor.add_(1)
        event.record(consumer)
    _consume_inputs_and_publish_outputs(tensor_release, tensor, (), consumer)

    tensor_release.finalize_phase(event, "test")

    assert tensor.untyped_storage().nbytes() == 0
    assert not _pending_releases(tensor_release)


def test_finalize_exports_phase_output():
    owner = torch.cuda.Stream()
    event = torch.cuda.Event()
    tensor_release = Combined1F1BTensorRelease()
    with torch.cuda.stream(owner):
        output = torch.empty(16, device="cuda")
        event.record(owner)
    _publish(tensor_release, output, owner)

    tensor_release.finalize_phase(event, "test", outputs=output)

    assert output.untyped_storage().nbytes() > 0
    assert not _owner_bindings(tensor_release)


def test_finalize_rejects_unconsumed_owner_binding():
    owner = torch.cuda.Stream()
    event = torch.cuda.Event()
    tensor_release = Combined1F1BTensorRelease()
    tensor = torch.empty(16, device="cuda")
    _publish(tensor_release, tensor, owner, node="forgotten producer")

    with pytest.raises(RuntimeError, match="unconsumed tensor-owner bindings"):
        tensor_release.finalize_phase(event, "test")

    tensor_release.export(tensor)
    tensor_release.finalize_phase(event, "test")


def test_cross_stream_drop_reference_holds_gradient_until_owner_acquire():
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    event = torch.cuda.Event()
    tensor_release = Combined1F1BTensorRelease()
    with torch.cuda.stream(owner):
        grad = torch.empty(1024, device="cuda")
        event.record(owner)
    _publish(tensor_release, grad, owner)
    grad_ref = weakref.ref(grad)
    event.wait(consumer)
    with torch.cuda.stream(consumer):
        grad.add_(1)
        event.record(consumer)
    _consume_output_grads_and_publish_input_grads(tensor_release, grad, (), consumer)
    del grad
    gc.collect()

    assert grad_ref() is not None
    _wait_and_drain(tensor_release, event, owner)
    gc.collect()
    assert grad_ref() is None


def test_unknown_external_gradient_uses_record_stream_fallback(monkeypatch):
    consumer = torch.cuda.Stream()
    tensor_release = Combined1F1BTensorRelease()
    grad = torch.empty(16, device="cuda")
    recorded = []
    original_record_stream = torch.Tensor.record_stream

    def track_record_stream(tensor, stream):
        recorded.append((tensor, stream))
        return original_record_stream(tensor, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", track_record_stream)

    _consume_output_grads_and_publish_input_grads(tensor_release, grad, (), consumer)

    assert len(recorded) == 1
    assert recorded[0][0] is grad
    assert recorded[0][1] == consumer
    assert not _owner_bindings(tensor_release)
    assert not _pending_releases(tensor_release)


def test_unknown_external_forward_input_preserves_record_stream_fallback(monkeypatch):
    consumer = torch.cuda.Stream()
    tensor_release = Combined1F1BTensorRelease()
    tensor = torch.empty(16, device="cuda")
    recorded = []
    original_record_stream = torch.Tensor.record_stream

    def track_record_stream(value, stream):
        recorded.append((value, stream))
        return original_record_stream(value, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", track_record_stream)

    _consume_inputs_and_publish_outputs(tensor_release, tensor, (), consumer)

    assert tensor.untyped_storage().nbytes() == 0
    assert len(recorded) == 1
    assert recorded[0][0] is tensor
    assert recorded[0][1] == consumer
    assert not _owner_bindings(tensor_release)
    assert not _pending_releases(tensor_release)


def test_backward_consumes_recompute_output_binding_and_publishes_gradient(monkeypatch):
    stream = torch.cuda.Stream()
    tensor_release = Combined1F1BTensorRelease()
    forward_output = torch.empty(16, device="cuda")
    incoming_grad = torch.empty_like(forward_output)
    produced_grad = torch.empty_like(forward_output)
    _publish(tensor_release, forward_output, stream, node="recompute forward")
    recorded = []
    original_record_stream = torch.Tensor.record_stream

    def track_record_stream(tensor, consumer_stream):
        recorded.append((tensor, consumer_stream))
        return original_record_stream(tensor, consumer_stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", track_record_stream)

    _consume_output_grads_and_publish_input_grads(
        tensor_release, incoming_grad, produced_grad, stream, forward_outputs=forward_output
    )

    owners = _owner_bindings(tensor_release)
    assert len(owners) == 1
    assert owners[0].tensor is produced_grad
    assert len(recorded) == 1
    assert recorded[0][0] is incoming_grad
    assert recorded[0][1] == stream
    tensor_release.export(produced_grad)


def test_backward_rejects_output_and_input_grad_storage_alias():
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    tensor_release = Combined1F1BTensorRelease()
    with torch.cuda.stream(owner):
        output_grad = torch.empty(16, device="cuda")
        input_grad = output_grad[2:10]
    _publish(tensor_release, output_grad, owner, node="upstream backward")

    with pytest.raises(RuntimeError, match="does not support storage aliases"):
        _consume_output_grads_and_publish_input_grads(
            tensor_release, output_grad, input_grad, consumer
        )

    owners = _owner_bindings(tensor_release)
    assert len(owners) == 1
    assert owners[0].tensor is output_grad
    assert owners[0].stream == owner
    tensor_release.export(output_grad)


def test_noop_preserves_tensor_binding_until_real_consumer():
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    tensor_release = Combined1F1BTensorRelease()
    tensor = torch.empty(16, device="cuda")
    _publish(tensor_release, tensor, owner)

    assert NoopScheduleNode().backward(tensor) is tensor
    _consume_output_grads_and_publish_input_grads(tensor_release, tensor, (), consumer)

    assert not _owner_bindings(tensor_release)
    assert len(_pending_releases(tensor_release)) == 1
    assert tuple(tensor_release._pending) == (owner,)


def test_external_consumed_grad_records_stream_in_schedule_node(monkeypatch):
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    event = torch.cuda.Event()
    tensor_release = Combined1F1BTensorRelease()
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
        tensor_release=tensor_release,
    )
    node.inputs = [node_input]
    node.output = node_output
    _publish(tensor_release, node_output, consumer, "forward")
    _publish(tensor_release, output_grad, consumer, "upstream backward")

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
    assert not _owner_bindings(tensor_release)
    assert not _pending_releases(tensor_release)


def test_no_external_consumed_grad_skips_record_stream(monkeypatch):
    stream = torch.cuda.Stream()
    event = torch.cuda.Event()
    tensor_release = Combined1F1BTensorRelease()
    node_input = torch.empty(16, device="cuda", requires_grad=True)
    node_output = torch.empty(16, device="cuda", requires_grad=True)
    output_grad = torch.empty_like(node_output)
    input_grad = torch.empty_like(node_input)
    node = ScheduleNode(
        lambda value: value,
        stream,
        event,
        backward_func=lambda _outputs, output_grads: output_grads,
        tensor_release=tensor_release,
    )
    node.inputs = [node_input]
    node.inputs[0].grad = input_grad
    node.output = node_output
    _publish(tensor_release, node_output, stream, "forward")
    _publish(tensor_release, output_grad, stream, "upstream backward")

    def unexpected_record_stream(_tensor, _stream):
        raise AssertionError("empty external gradient path must not call record_stream")

    monkeypatch.setattr(torch.Tensor, "record_stream", unexpected_record_stream)

    assert node.backward(output_grad) is input_grad
    owners = _owner_bindings(tensor_release)
    assert len(owners) == 1
    assert owners[0].tensor is input_grad
    tensor_release.export(input_grad)


def test_allocator_reuse_poison_does_not_overwrite_delayed_consumer():
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("torch.cuda._sleep is unavailable")

    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    event = torch.cuda.Event()
    tensor_release = Combined1F1BTensorRelease()
    observed = []

    for pattern in range(1, 17):
        with torch.cuda.stream(owner):
            tensor = torch.full((256 * 1024,), pattern, dtype=torch.int32, device="cuda")
            event.record(owner)
        _publish(tensor_release, tensor, owner, node=f"producer {pattern}")
        event.wait(consumer)
        with torch.cuda.stream(consumer):
            torch.cuda._sleep(1_000_000)
            observed.append(tensor.clone())
            event.record(consumer)
        _consume_inputs_and_publish_outputs(
            tensor_release, tensor, (), consumer, node=f"consumer {pattern}"
        )

        # The wait is enqueued before drain makes storage allocator-visible.
        _wait_and_drain(tensor_release, event, owner)
        with torch.cuda.stream(owner):
            torch.full((256 * 1024,), -pattern, dtype=torch.int32, device="cuda")

    tensor_release.finalize_phase(event, "test")
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
        tensor_release = Combined1F1BTensorRelease()
        tensor = torch.full((1024,), 7, dtype=torch.int32, device="cuda")
        event.record(capture_stream)
        _publish(tensor_release, tensor, capture_stream)

        event.wait(consumer)
        with torch.cuda.stream(consumer):
            output = tensor + 1
            event.record(consumer)
        _consume_inputs_and_publish_outputs(tensor_release, tensor, output, consumer)
        _wait_and_drain(tensor_release, event, capture_stream)
        tensor_release.finalize_phase(event, "capture", outputs=output)

    assert not _pending_releases(tensor_release)
    assert not _owner_bindings(tensor_release)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    assert torch.all(output == 8)
    # Python owner bookkeeping only ran during capture, not during replay.
    assert not _pending_releases(tensor_release)
    assert not _owner_bindings(tensor_release)
