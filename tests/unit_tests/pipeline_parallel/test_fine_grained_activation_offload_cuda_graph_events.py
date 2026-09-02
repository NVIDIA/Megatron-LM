# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import nullcontext
from unittest.mock import Mock, call

import pytest
import torch

from megatron.core.pipeline_parallel import fine_grained_activation_offload as offload
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    ChunkOffloadHandler,
    PipelineOffloadManager,
)


@pytest.fixture
def mocked_offload_manager(monkeypatch):
    """Build the singleton without requiring a CUDA device."""
    events = []
    streams = []

    def make_event(**kwargs):
        event = Mock(name=f"event_{len(events)}")
        event.external = kwargs.get("external", False)
        events.append(event)
        return event

    def make_stream():
        stream = Mock(name=f"stream_{len(streams)}")
        streams.append(stream)
        return stream

    monkeypatch.setattr(torch.cuda, "Event", make_event)
    monkeypatch.setattr(torch.cuda, "Stream", make_stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(offload, "nvtx_range_push", Mock())
    monkeypatch.setattr(offload, "nvtx_range_pop", Mock())

    old_manager = PipelineOffloadManager.OFFLOAD_MGR
    manager = PipelineOffloadManager()
    PipelineOffloadManager.OFFLOAD_MGR = manager
    yield manager, events, streams
    PipelineOffloadManager.OFFLOAD_MGR = old_manager


def _new_handler(manager, max_inflight_offloads):
    return ChunkOffloadHandler(
        min_offloaded_tensor_size=1,
        cpu_tensor_pool=manager.cpu_tensor_pool,
        max_inflight_offloads=max_inflight_offloads,
    )


def _empty_group(handler, name):
    handler.on_group_start_forward(name)
    return handler.offload_groups[-1]


def _bulk_empty_group(handler, name):
    group = _empty_group(handler, name)
    handler.bulk_offload_group(group)
    return group


@pytest.mark.parametrize("max_inflight_offloads", [None, 0, 2])
def test_handler_allocates_external_event_only_for_positive_throttling_cap(
    mocked_offload_manager, max_inflight_offloads
):
    manager, _, _ = mocked_offload_manager
    handler = _new_handler(manager, max_inflight_offloads=max_inflight_offloads)

    group = _empty_group(handler, "core_attn")

    assert (group._offload_throttle_event is not None) == (
        max_inflight_offloads is not None and max_inflight_offloads > 0
    )


def test_capture_scope_nests_but_sibling_scopes_get_distinct_owners(mocked_offload_manager):
    manager, _, _ = mocked_offload_manager

    assert manager.cuda_graph_capture_owner is None
    with manager.cuda_graph_capture_scope(may_cross_graphs=False):
        first_owner = manager.cuda_graph_capture_owner
        assert first_owner is not None
        with manager.cuda_graph_capture_scope(may_cross_graphs=True):
            assert manager.cuda_graph_capture_owner is first_owner
            assert not manager.cuda_graph_capture_owner.may_cross_graphs
        assert manager.cuda_graph_capture_owner is first_owner

    assert manager.cuda_graph_capture_owner is None
    with manager.cuda_graph_capture_scope(may_cross_graphs=False):
        sibling_owner = manager.cuda_graph_capture_owner

    assert sibling_owner is not first_owner
    assert manager.cuda_graph_capture_owner is None


def test_capture_scope_restores_owner_after_exception(mocked_offload_manager):
    manager, _, _ = mocked_offload_manager

    with pytest.raises(RuntimeError, match="capture failed"):
        with manager.cuda_graph_capture_scope(may_cross_graphs=True):
            assert manager.cuda_graph_capture_owner is not None
            raise RuntimeError("capture failed")

    assert manager.cuda_graph_capture_owner is None


@pytest.mark.parametrize("use_capture_scope", [False, True], ids=["eager", "same-owner"])
def test_same_owner_and_eager_throttle_wait_on_internal_event(
    mocked_offload_manager, monkeypatch, use_capture_scope
):
    manager, _, _ = mocked_offload_manager
    main_stream = Mock()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main_stream)
    handler = _new_handler(manager, max_inflight_offloads=0)

    scope = (
        manager.cuda_graph_capture_scope(may_cross_graphs=True)
        if use_capture_scope
        else nullcontext()
    )
    with scope:
        group = _bulk_empty_group(handler, "core_attn")

    main_stream.wait_event.assert_called_once_with(group._offload_event)
    assert not handler._offload_pending_by_name["core_attn"]


def test_different_capture_owner_waits_on_external_event(mocked_offload_manager, monkeypatch):
    manager, _, _ = mocked_offload_manager
    main_stream = Mock()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main_stream)
    handler = _new_handler(manager, max_inflight_offloads=1)

    with manager.cuda_graph_capture_scope(may_cross_graphs=True):
        group = _bulk_empty_group(handler, "core_attn")

    with manager.cuda_graph_capture_scope(may_cross_graphs=True):
        second_group = _bulk_empty_group(handler, "core_attn")

    assert group._offload_throttle_event is not None
    main_stream.wait_event.assert_called_once_with(group._offload_throttle_event)
    assert [entry.group for entry in handler._offload_pending_by_name["core_attn"]] == [
        second_group
    ]


@pytest.mark.parametrize(
    "producer_is_captured", [False, True], ids=["eager-to-graph", "graph-to-eager"]
)
def test_mixed_eager_and_capture_owner_fails_fast(
    mocked_offload_manager, monkeypatch, producer_is_captured
):
    manager, _, _ = mocked_offload_manager
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: Mock())
    handler = _new_handler(manager, max_inflight_offloads=1)

    producer_scope = (
        manager.cuda_graph_capture_scope(may_cross_graphs=True)
        if producer_is_captured
        else nullcontext()
    )
    with producer_scope:
        _bulk_empty_group(handler, "core_attn")

    consumer_scope = (
        nullcontext()
        if producer_is_captured
        else manager.cuda_graph_capture_scope(may_cross_graphs=True)
    )
    with consumer_scope, pytest.raises(RuntimeError, match="eager execution.*CUDA graph capture"):
        _bulk_empty_group(handler, "core_attn")


@pytest.mark.parametrize("max_inflight_offloads", [0, 2])
def test_max_inflight_fifo_is_per_group_name(
    mocked_offload_manager, monkeypatch, max_inflight_offloads
):
    manager, _, _ = mocked_offload_manager
    main_stream = Mock()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main_stream)
    handler = _new_handler(manager, max_inflight_offloads=max_inflight_offloads)

    core_groups = [_bulk_empty_group(handler, "core_attn") for _ in range(3)]
    mlp_group = _bulk_empty_group(handler, "mlp")

    if max_inflight_offloads == 0:
        expected_waits = [
            *[call(group._offload_event) for group in core_groups],
            call(mlp_group._offload_event),
        ]
        assert main_stream.wait_event.call_args_list == expected_waits
        assert not handler._offload_pending_by_name["core_attn"]
        assert not handler._offload_pending_by_name["mlp"]
    else:
        main_stream.wait_event.assert_called_once_with(core_groups[0]._offload_event)
        assert [
            entry.group for entry in handler._offload_pending_by_name["core_attn"]
        ] == core_groups[1:]
        assert [entry.group for entry in handler._offload_pending_by_name["mlp"]] == [mlp_group]
