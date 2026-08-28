# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for schedule-aware cross-stream tensor lifetime management."""

import gc
import weakref

import pytest
import torch

from megatron.core.pipeline_parallel.tensor_lifetime import (
    _PROVENANCE_REGISTRY,
    ReleaseAction,
    ScheduleTensorLifetimeManager,
    _stream_key,
    register_external_tensor,
)
from megatron.core.pipeline_parallel.utils import NoopScheduleNode

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _new_manager(phase="test"):
    manager = ScheduleTensorLifetimeManager(torch.cuda.Event())
    manager.begin_phase(phase)
    manager.record_root(torch.cuda.current_stream(), f"{phase}:root")
    return manager


def _complete_node(manager, stream, name):
    token = manager.acquire(stream, name)
    generation = manager.record(stream, token)
    assert token.completion_generation == generation
    return generation


def test_same_stream_empty_storage_is_immediate():
    owner = torch.cuda.Stream()
    with torch.cuda.stream(owner):
        tensor = torch.empty(1024, device="cuda")
    register_external_tensor(tensor, owner, "same-stream producer")

    manager = _new_manager()
    generation = _complete_node(manager, owner, "same-stream consumer")
    manager.retire(
        tensor,
        action=ReleaseAction.EMPTY_STORAGE,
        consumer_stream=owner,
        consumer_generation=generation,
        node="same-stream consumer",
    )

    assert tensor.untyped_storage().nbytes() == 0
    assert not manager.pending
    assert manager.stats["same_stream"] == 1
    manager.finalize_phase("test")


def test_cross_stream_release_waits_for_owner_acquire():
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(owner):
        tensor = torch.empty(1024, device="cuda")
    register_external_tensor(tensor, owner, "cross-stream producer")

    manager = _new_manager()
    generation = _complete_node(manager, consumer, "cross-stream consumer")
    manager.retire(
        tensor,
        action=ReleaseAction.EMPTY_STORAGE,
        consumer_stream=consumer,
        consumer_generation=generation,
        node="cross-stream consumer",
    )

    assert tensor.untyped_storage().nbytes() > 0
    assert len(manager.pending) == 1

    # Waiting on an older event generation is not sufficient, even on the owner stream.
    manager._drain(owner, generation - 1, release_node="too-early owner acquire", terminal=False)
    assert tensor.untyped_storage().nbytes() > 0

    _complete_node(manager, owner, "safe owner acquire")
    assert tensor.untyped_storage().nbytes() == 0
    assert not manager.pending
    assert manager.last_release_node == "safe owner acquire"
    manager.finalize_phase("test")


def test_terminal_hand_back_drains_all_owner_streams():
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(owner):
        tensor = torch.empty(1024, device="cuda")
    register_external_tensor(tensor, owner, "terminal producer")

    manager = _new_manager()
    generation = _complete_node(manager, consumer, "terminal consumer")
    manager.retire(
        tensor,
        action=ReleaseAction.EMPTY_STORAGE,
        consumer_stream=consumer,
        consumer_generation=generation,
        node="terminal consumer",
    )
    manager.finalize_phase("test")

    assert tensor.untyped_storage().nbytes() == 0
    assert not manager.pending
    assert manager.stats["released_terminal"] == 1
    assert manager.last_release_node == "test:phase_finalize"


def test_cross_stream_drop_reference_holds_gradient_until_owner_acquire():
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    grad = torch.empty(1024, device="cuda")
    grad_ref = weakref.ref(grad)
    register_external_tensor(grad, owner, "backward gradient producer")

    manager = _new_manager()
    generation = _complete_node(manager, consumer, "backward gradient consumer")
    manager.retire(
        grad,
        action=ReleaseAction.DROP_REFERENCE,
        consumer_stream=consumer,
        consumer_generation=generation,
        node="backward gradient consumer",
    )
    del grad
    gc.collect()

    assert grad_ref() is not None
    _complete_node(manager, owner, "backward gradient release")
    gc.collect()
    assert grad_ref() is None
    manager.finalize_phase("test")


def test_wrong_manager_cannot_consume_a_live_lease():
    owner = torch.cuda.Stream()
    tensor = torch.empty(16, device="cuda")
    first = _new_manager("first")
    first.publish(tensor, owner, "first producer")
    second = _new_manager("second")
    generation = _complete_node(second, owner, "wrong consumer")

    with pytest.raises(RuntimeError, match="belongs to lifetime manager"):
        second.retire(
            tensor,
            action=ReleaseAction.DROP_REFERENCE,
            consumer_stream=owner,
            consumer_generation=generation,
            node="wrong consumer",
        )

    first.finalize_phase("first", outputs=tensor)
    second.finalize_phase("second")


def test_noop_preserves_external_lease_until_real_consumer():
    owner = torch.cuda.Stream()
    tensor = torch.empty(16, device="cuda")
    register_external_tensor(tensor, owner, "noop producer")
    assert NoopScheduleNode().forward(tensor) is tensor

    manager = _new_manager()
    generation = _complete_node(manager, owner, "real consumer")
    manager.retire(
        tensor,
        action=ReleaseAction.DROP_REFERENCE,
        consumer_stream=owner,
        consumer_generation=generation,
        node="real consumer",
    )
    manager.finalize_phase("test")


def test_free_input_output_alias_is_rejected():
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    tensor = torch.empty(16, device="cuda")
    alias = tensor.view_as(tensor)
    register_external_tensor(tensor, owner, "alias producer")
    manager = _new_manager()
    generation = _complete_node(manager, consumer, "alias consumer")

    with pytest.raises(RuntimeError, match="aliases output storage"):
        manager.retire_and_publish(
            tensor,
            alias,
            action=ReleaseAction.EMPTY_STORAGE,
            producer_stream=consumer,
            consumer_stream=consumer,
            consumer_generation=generation,
            node="alias consumer",
        )

    manager.export(tensor)
    manager.finalize_phase("test")


def test_drop_reference_alias_transfers_creation_stream():
    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    tensor = torch.empty(16, device="cuda")
    alias = tensor.view_as(tensor)
    register_external_tensor(tensor, owner, "alias producer")
    manager = _new_manager()
    generation = _complete_node(manager, consumer, "alias consumer")

    manager.retire_and_publish(
        tensor,
        alias,
        action=ReleaseAction.DROP_REFERENCE,
        producer_stream=consumer,
        consumer_stream=consumer,
        consumer_generation=generation,
        node="alias consumer",
    )

    provenance = _PROVENANCE_REGISTRY.peek(alias)
    assert provenance is not None
    assert provenance.creation_stream_key == _stream_key(owner)
    assert not manager.pending
    manager.finalize_phase("test", outputs=alias)


def test_unknown_provenance_and_stale_lease_are_strict_errors():
    owner = torch.cuda.Stream()
    unknown = torch.empty(16, device="cuda")
    manager = _new_manager()
    generation = _complete_node(manager, owner, "unknown consumer")
    with pytest.raises(RuntimeError, match="Missing creation-stream provenance"):
        manager.retire(
            unknown,
            action=ReleaseAction.DROP_REFERENCE,
            consumer_stream=owner,
            consumer_generation=generation,
            node="unknown consumer",
        )
    assert manager.stats["unknown_provenance"] == 1
    manager.finalize_phase("test")

    tensor = torch.empty(16, device="cuda")
    register_external_tensor(tensor, owner, "first lease")
    with pytest.raises(RuntimeError, match="unconsumed scheduled-lifetime lease"):
        register_external_tensor(tensor, owner, "stale lease")
    cleanup = _new_manager("cleanup")
    cleanup.export(tensor)
    cleanup.finalize_phase("cleanup")


def test_detached_grad_hook_publishes_materialized_grad_on_consumer_stream():
    consumer = torch.cuda.Stream()
    manager = _new_manager()
    with torch.cuda.stream(consumer):
        leaf = torch.randn(16, device="cuda", requires_grad=True)
    handle = manager.track_detached_leaf(leaf)

    token = manager.acquire(consumer, "detached backward")
    with torch.cuda.stream(consumer):
        (leaf.square().sum()).backward()
        manager.publish_dirty_detached_grads(consumer, "detached backward")
    generation = manager.record(consumer, token)

    provenance = _PROVENANCE_REGISTRY.peek(leaf.grad)
    assert provenance is not None
    assert provenance.creation_stream_key == _stream_key(consumer)
    manager.retire(
        leaf.grad,
        action=ReleaseAction.DROP_REFERENCE,
        consumer_stream=consumer,
        consumer_generation=generation,
        node="detached backward",
    )
    handle.remove()
    manager.finalize_phase("test")


def test_finalize_rejects_unexported_live_lease():
    manager = _new_manager()
    tensor = torch.empty(16, device="cuda")
    manager.publish(tensor, torch.cuda.current_stream(), "forgotten output")

    with pytest.raises(RuntimeError, match="live tensor leases"):
        manager.finalize_phase("test")

    manager.export(tensor)
    manager.finalize_phase("test")


def test_allocator_reuse_poison_does_not_overwrite_delayed_consumer():
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("torch.cuda._sleep is unavailable")

    owner = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    manager = _new_manager()
    observed = []

    for pattern in range(1, 17):
        with torch.cuda.stream(owner):
            tensor = torch.full((256 * 1024,), pattern, dtype=torch.int32, device="cuda")
        _complete_node(manager, owner, f"producer {pattern}")
        manager.publish(tensor, owner, f"producer {pattern}")

        token = manager.acquire(consumer, f"consumer {pattern}")
        with torch.cuda.stream(consumer):
            torch.cuda._sleep(1_000_000)
            observed.append(tensor.clone())
        generation = manager.record(consumer, token)
        manager.retire(
            tensor,
            action=ReleaseAction.EMPTY_STORAGE,
            consumer_stream=consumer,
            consumer_generation=generation,
            node=f"consumer {pattern}",
        )

        # The owner-side wait is the hand-back edge.  Allocator pressure starts only
        # after that edge has been enqueued, so reusing the address cannot race the read.
        _complete_node(manager, owner, f"owner reuse {pattern}")
        with torch.cuda.stream(owner):
            torch.full((256 * 1024,), -pattern, dtype=torch.int32, device="cuda")

    manager.finalize_phase("test")
    torch.cuda.synchronize()
    for pattern, value in enumerate(observed, start=1):
        assert torch.all(value == pattern)


def test_full_graph_capture_finalizes_leases_before_python_free_replay():
    capture_stream = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph, stream=capture_stream):
        manager = ScheduleTensorLifetimeManager(torch.cuda.Event())
        manager.begin_phase("capture")
        manager.record_root(capture_stream, "capture root")

        tensor = torch.full((1024,), 7, dtype=torch.int32, device="cuda")
        manager.publish(tensor, capture_stream, "captured producer")
        token = manager.acquire(consumer, "captured consumer")
        with torch.cuda.stream(consumer):
            output = tensor + 1
        generation = manager.record(consumer, token)
        manager.retire_and_publish(
            tensor,
            output,
            action=ReleaseAction.EMPTY_STORAGE,
            producer_stream=consumer,
            consumer_stream=consumer,
            consumer_generation=generation,
            node="captured consumer",
        )
        _complete_node(manager, capture_stream, "captured owner hand-back")
        manager.finalize_phase("capture", outputs=output)

    assert not manager.pending
    assert not manager._live_lease_ids
    capture_stats = manager.stats.copy()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    assert torch.all(output == 8)
    # Python lifetime bookkeeping only ran during capture, not during replay.
    assert manager.stats == capture_stats
