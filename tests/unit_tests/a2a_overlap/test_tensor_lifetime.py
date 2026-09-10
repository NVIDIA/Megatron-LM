# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for schedule-local cross-stream tensor lifetime management."""

import gc
import weakref

import pytest
import torch

from megatron.core.pipeline_parallel.tensor_lifetime import ScheduleTensorLifetimeManager
from megatron.core.pipeline_parallel.utils import NoopScheduleNode

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _retire_forward(manager, tensor, owner, consumer, node="consumer"):
    manager.retire_forward_inputs(tensor, owner_stream=owner, consumer_stream=consumer, node=node)


def _retire_gradient(manager, tensor, owner, consumer, node="consumer", fallback_consumed=()):
    manager.retire_backward_inputs(
        tensor,
        owner_stream=owner,
        consumer_stream=consumer,
        node=node,
        fallback_consumed=fallback_consumed,
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
    _retire_forward(manager, tensor, owner, owner)

    assert tensor.untyped_storage().nbytes() == 0
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
    event.wait(consumer)
    with torch.cuda.stream(consumer):
        tensor.add_(1)
        event.record(consumer)
    _retire_forward(manager, tensor, owner, consumer)

    assert tensor.untyped_storage().nbytes() > 0
    assert len(manager.pending) == 1

    _wait_and_drain(manager, event, owner)
    assert tensor.untyped_storage().nbytes() == 0
    assert not manager.pending
    assert manager.last_release_node == "owner acquire"


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
    backward_event.wait(consumer)
    with torch.cuda.stream(consumer):
        tensor.add_(1)
        backward_event.record(consumer)
    _retire_forward(backward, tensor, owner, consumer, node="B MLP")

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
    event.wait(consumer)
    with torch.cuda.stream(consumer):
        tensor.add_(1)
        event.record(consumer)
    _retire_forward(manager, tensor, owner, consumer)

    manager.finalize_phase(event, "test")

    assert tensor.untyped_storage().nbytes() == 0
    assert not manager.pending
    assert manager.stats["released_terminal"] == 1
    assert manager.last_release_node == "test:phase_finalize"


def test_cross_stream_drop_reference_holds_gradient_until_owner_acquire():
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    event = torch.cuda.Event()
    manager = ScheduleTensorLifetimeManager()
    with torch.cuda.stream(owner):
        grad = torch.empty(1024, device="cuda")
        event.record(owner)
    grad_ref = weakref.ref(grad)
    event.wait(consumer)
    with torch.cuda.stream(consumer):
        grad.add_(1)
        event.record(consumer)
    _retire_gradient(manager, grad, owner, consumer)
    del grad
    gc.collect()

    assert grad_ref() is not None
    _wait_and_drain(manager, event, owner)
    gc.collect()
    assert grad_ref() is None


def test_unknown_external_gradient_uses_record_stream_fallback():
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    manager = ScheduleTensorLifetimeManager()
    grad = torch.empty(16, device="cuda")
    manager.mark_external_gradients(grad)

    _retire_gradient(manager, grad, owner, consumer)

    assert not manager.pending
    assert manager.stats["record_stream_fallback"] == 1


def test_noop_preserves_tensor_until_statically_mapped_real_consumer():
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    manager = ScheduleTensorLifetimeManager()
    tensor = torch.empty(16, device="cuda")

    assert NoopScheduleNode().forward(tensor) is tensor
    _retire_gradient(manager, tensor, owner, consumer)

    assert len(manager.pending) == 1
    assert manager.pending[0].owner_stream.cuda_stream == owner.cuda_stream


def test_detached_gradient_without_explicit_producer_tag_falls_back():
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    manager = ScheduleTensorLifetimeManager()
    with torch.cuda.stream(producer):
        leaf = torch.randn(16, device="cuda", requires_grad=True)
        (leaf.square().sum()).backward()
    detached_grad = leaf.grad

    _retire_gradient(manager, detached_grad, None, consumer)

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
        event.wait(consumer)
        with torch.cuda.stream(consumer):
            torch.cuda._sleep(1_000_000)
            observed.append(tensor.clone())
            event.record(consumer)
        _retire_forward(manager, tensor, owner, consumer)

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

        event.wait(consumer)
        with torch.cuda.stream(consumer):
            output = tensor + 1
            event.record(consumer)
        manager.retire_forward_inputs(
            tensor, owner_stream=capture_stream, consumer_stream=consumer, node="captured consumer"
        )
        _wait_and_drain(manager, event, capture_stream, "captured owner hand-back")
        manager.finalize_phase(event, "capture")

    assert not manager.pending
    capture_stats = manager.stats.copy()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    assert torch.all(output == 8)
    assert manager.stats == capture_stats
