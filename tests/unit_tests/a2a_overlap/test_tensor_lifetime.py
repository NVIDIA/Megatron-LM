# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for schedule-local cross-stream tensor lifetime management."""

import gc
import weakref

import pytest
import torch

from megatron.core.pipeline_parallel import tensor_lifetime
from megatron.core.pipeline_parallel.tensor_lifetime import ScheduleTensorLifetimeManager
from megatron.core.pipeline_parallel.utils import NoopScheduleNode

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.launch_on_gb200,
]


def _publish(manager, tensor, owner, node="producer"):
    manager.publish(tensor, owner, node)


def _consume_inputs_and_publish_outputs(
    manager, consumed, produced, consumer, retire=True, node="consumer"
):
    manager.consume_inputs_and_publish_outputs(
        consumed, produced, stream=consumer, node=node, retire_consumed=retire
    )


def _consume_output_grads_and_publish_input_grads(
    manager,
    consumed,
    produced,
    consumer,
    node="consumer",
    forward_outputs=(),
    additional_consumed_grads=(),
):
    manager.consume_forward_outputs(forward_outputs)
    manager.consume_output_grads_and_publish_input_grads(
        consumed,
        produced,
        stream=consumer,
        node=node,
        additional_consumed_grads=additional_consumed_grads,
    )


def _wait_and_drain(manager, event, stream, node="owner acquire"):
    event.wait(stream)
    with torch.cuda.stream(stream):
        manager.drain(stream, node)


def test_same_stream_empty_storage_is_immediate():
    owner = torch.cuda.Stream()
    manager = ScheduleTensorLifetimeManager()
    with torch.cuda.stream(owner):
        tensor = torch.empty(1024, device="cuda")
    _publish(manager, tensor, owner)
    _consume_inputs_and_publish_outputs(manager, tensor, (), owner)

    assert tensor.untyped_storage().nbytes() == 0
    assert not manager.owners
    assert not manager.pending
    assert manager.stats["same_stream"] == 1


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
    assert not manager.owners
    assert len(manager.pending) == 1

    _wait_and_drain(manager, event, owner)
    assert tensor.untyped_storage().nbytes() == 0
    assert not manager.pending
    assert manager.last_release_node == "owner acquire"


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

    assert tensor.untyped_storage().nbytes() > 0
    assert not manager.pending
    assert len(manager.owners) == 1
    assert manager.owners[0].tensor is output
    assert manager.owners[0].stream.cuda_stream == consumer.cuda_stream
    assert manager.owners[0].stream_key == (consumer.device, int(consumer.cuda_stream))
    manager.export(output)


def test_single_tensor_hot_path_skips_structured_iterator(monkeypatch):
    stream = torch.cuda.Stream()
    manager = ScheduleTensorLifetimeManager()
    external_input = torch.empty(16, device="cuda")
    output = torch.empty_like(external_input)
    additional_grad = torch.empty_like(external_input)

    def unexpected_structured_iteration(_value):
        raise AssertionError("single tensors must bypass structured iteration")
        yield

    monkeypatch.setattr(
        tensor_lifetime, "_iter_unique_cuda_tensors", unexpected_structured_iteration
    )

    # The external input exercises the direct record_stream fallback, while the
    # output exercises the direct publish path.
    manager.consume_inputs_and_publish_outputs(
        external_input, output, stream=stream, node="single tensor", retire_consumed=True
    )
    manager.consume_forward_outputs(output)
    manager.consume_output_grads_and_publish_input_grads(
        (),
        (),
        stream=stream,
        node="single additional grad",
        additional_consumed_grads=(additional_grad,),
    )

    assert not manager.owners
    assert manager.stats["record_stream_fallback"] == 2


def test_empty_additional_consumed_grads_skip_record_stream(monkeypatch):
    stream = torch.cuda.Stream()
    manager = ScheduleTensorLifetimeManager()

    def unexpected_record_stream(_tensors, _stream):
        raise AssertionError("empty additional gradients must skip record_stream")

    monkeypatch.setattr(manager, "_record_stream", unexpected_record_stream)
    manager.consume_output_grads_and_publish_input_grads(
        (), (), stream=stream, node="backward", additional_consumed_grads=()
    )

    assert manager.stats["record_stream_fallback"] == 0


def test_structured_tensors_are_deduplicated():
    stream = torch.cuda.Stream()
    manager = ScheduleTensorLifetimeManager()
    tensor = torch.empty(16, device="cuda")

    manager.publish((tensor, None, tensor), stream, "structured producer")

    assert len(manager.owners) == 1
    assert manager.stats["published"] == 1
    manager.export((tensor, tensor))
    assert not manager.owners


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
    _wait_and_drain(forward, forward_event, owner, "F dispatch")
    assert tensor.untyped_storage().nbytes() > 0
    assert len(backward.pending) == 1

    _wait_and_drain(backward, backward_event, owner, "B dispatch")
    assert tensor.untyped_storage().nbytes() == 0
    assert not backward.pending


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
    assert not manager.pending
    assert manager.stats["released_terminal"] == 1
    assert manager.last_release_node == "test:phase_finalize"


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
    assert not manager.owners
    assert manager.stats["owners_before_finalize"] == 1
    assert manager.stats["owners_at_finalize"] == 0
    assert manager.stats["exported"] == 1


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


def test_unknown_external_gradient_uses_record_stream_fallback():
    consumer = torch.cuda.Stream()
    manager = ScheduleTensorLifetimeManager()
    grad = torch.empty(16, device="cuda")

    _consume_output_grads_and_publish_input_grads(manager, grad, (), consumer)

    assert not manager.owners
    assert not manager.pending
    assert manager.stats["record_stream_fallback"] == 1


def test_unknown_external_forward_input_preserves_record_stream_fallback():
    consumer = torch.cuda.Stream()
    manager = ScheduleTensorLifetimeManager()
    tensor = torch.empty(16, device="cuda")

    _consume_inputs_and_publish_outputs(manager, tensor, (), consumer)

    assert tensor.untyped_storage().nbytes() == 0
    assert not manager.owners
    assert not manager.pending
    assert manager.stats["record_stream_fallback"] == 1


def test_backward_consumes_recompute_output_binding_and_publishes_gradient():
    stream = torch.cuda.Stream()
    manager = ScheduleTensorLifetimeManager()
    forward_output = torch.empty(16, device="cuda")
    incoming_grad = torch.empty_like(forward_output)
    produced_grad = torch.empty_like(forward_output)
    _publish(manager, forward_output, stream, node="recompute forward")

    _consume_output_grads_and_publish_input_grads(
        manager, incoming_grad, produced_grad, stream, forward_outputs=forward_output
    )

    assert len(manager.owners) == 1
    assert manager.owners[0].tensor is produced_grad
    assert manager.stats["consumed"] == 1
    assert manager.stats["record_stream_fallback"] == 1
    manager.export(produced_grad)


def test_noop_preserves_tensor_binding_until_real_consumer():
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    manager = ScheduleTensorLifetimeManager()
    tensor = torch.empty(16, device="cuda")
    _publish(manager, tensor, owner)

    assert NoopScheduleNode().backward(tensor) is tensor
    _consume_output_grads_and_publish_input_grads(manager, tensor, (), consumer)

    assert not manager.owners
    assert len(manager.pending) == 1
    assert manager.pending[0].owner_stream.cuda_stream == owner.cuda_stream


def test_detached_gradient_without_registered_producer_falls_back():
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    manager = ScheduleTensorLifetimeManager()
    with torch.cuda.stream(producer):
        leaf = torch.randn(16, device="cuda", requires_grad=True)
        (leaf.square().sum()).backward()
    detached_grad = leaf.grad

    _consume_output_grads_and_publish_input_grads(
        manager, (), (), consumer, additional_consumed_grads=(detached_grad,)
    )

    assert manager.stats["record_stream_fallback"] == 1
    assert not manager.pending


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
        _wait_and_drain(manager, event, capture_stream, "captured owner hand-back")
        manager.finalize_phase(event, "capture", outputs=output)

    assert not manager.pending
    assert not manager.owners
    capture_stats = manager.stats.copy()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    assert torch.all(output == 8)
    # Python owner bookkeeping only ran during capture, not during replay.
    assert manager.stats == capture_stats
