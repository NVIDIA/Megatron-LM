# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from contextlib import nullcontext
from unittest.mock import Mock, call

import pytest
import torch

from megatron.core.pipeline_parallel import fine_grained_activation_offload as offload
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    ChunkOffloadHandler,
    FineGrainedActivationOffloadingInterface,
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


def _set_local_cuda_device():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank >= torch.cuda.device_count():
        pytest.fail(
            f"LOCAL_RANK={local_rank} is outside visible CUDA device count "
            f"{torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)


@pytest.fixture
def cuda_offload_manager():
    _set_local_cuda_device()
    torch.cuda.synchronize()
    FineGrainedActivationOffloadingInterface.reset_instance()
    manager = PipelineOffloadManager.get_instance()
    try:
        yield manager
    finally:
        torch.cuda.synchronize()
        FineGrainedActivationOffloadingInterface.reset_instance()
        torch.cuda.empty_cache()


def _prewarm_cpu_pool(handler, count, shape, dtype):
    backups = [handler.cpu_tensor_pool.allocate(shape, dtype=dtype) for _ in range(count)]
    for backup in backups:
        handler.cpu_tensor_pool.free(backup)


def _capture_real_d2h_group(handler, group, source, tag):
    handler.d2h_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(handler.d2h_stream):
        torch.cuda._sleep(50_000_000)
    group.push_tensor(tag, source)
    handler.bulk_offload_group(group)


def _assert_cpu_backup(group, tag, expected):
    _, backup, _ = group._tensors[tag]
    torch.testing.assert_close(backup, torch.full_like(backup, expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for graph capture.")
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize(
    ("max_inflight_offloads", "may_cross_graphs"),
    [
        pytest.param(0, False, id="full-iteration-cap0"),
        pytest.param(2, False, id="full-iteration-cap2"),
        pytest.param(0, True, id="te-like-same-owner-cap0"),
        pytest.param(2, True, id="te-like-same-owner-cap2"),
    ],
)
def test_same_graph_throttle_orders_real_d2h_before_source_reuse(
    cuda_offload_manager, max_inflight_offloads, may_cross_graphs
):
    manager = cuda_offload_manager
    handler = _new_handler(manager, max_inflight_offloads)
    group_count = max_inflight_offloads + 1
    shape = (4096,)
    dtype = torch.float32
    _prewarm_cpu_pool(handler, group_count, shape, dtype)
    sources = [torch.empty(shape, dtype=dtype, device="cuda") for _ in range(group_count)]
    tags = [(index + 1, 0) for index in range(group_count)]
    groups = []
    graph = torch.cuda.CUDAGraph()

    capture_scope = (
        FineGrainedActivationOffloadingInterface.cuda_graph_capture_scope
        if may_cross_graphs
        else manager.cuda_graph_capture_scope
    )
    with capture_scope(may_cross_graphs=may_cross_graphs):
        for _ in range(group_count):
            groups.append(_empty_group(handler, "core_attn"))
        with torch.cuda.graph(graph):
            for group, source, tag in zip(groups, sources, tags):
                _capture_real_d2h_group(handler, group, source, tag)
            # The oldest copy is the one drained by cap=0/cap=2. Reusing its
            # source immediately after the drain exposes a missing graph edge.
            sources[0].add_(1000)
            torch.cuda.current_stream().wait_stream(handler.d2h_stream)

    try:
        for replay, value in enumerate((11.0, 22.0)):
            for source in sources:
                source.fill_(value)
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            _assert_cpu_backup(groups[0], tags[0], value)
            assert sources[0][0].item() == value + 1000
    finally:
        torch.cuda.synchronize()
        graph = None
        groups.clear()
        sources.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for graph capture.")
@pytest.mark.launch_on_gb200
def test_cross_graph_throttle_captures_real_d2h_with_external_event(cuda_offload_manager):
    manager = cuda_offload_manager
    handler = _new_handler(manager, max_inflight_offloads=2)
    shape = (4096,)
    dtype = torch.float32
    _prewarm_cpu_pool(handler, 3, shape, dtype)
    sources = [torch.empty(shape, dtype=dtype, device="cuda") for _ in range(3)]
    tags = [(index + 1, 0) for index in range(3)]
    groups = []
    graphs = []

    for index in range(3):
        graph = torch.cuda.CUDAGraph()
        with manager.cuda_graph_capture_scope(may_cross_graphs=True):
            group = _empty_group(handler, "core_attn")
            groups.append(group)
            with torch.cuda.graph(graph):
                _capture_real_d2h_group(handler, group, sources[index], tags[index])
                if index == 2:
                    # Graph three drains graph one's pending entry. Capture must
                    # encode an external wait; an internal event is invalid here.
                    sources[0].add_(1000)
                # CUDA capture requires every forked stream to rejoin its own
                # graph. Because this local join may also order payload completion,
                # capture/replay success is the cross-graph event regression signal;
                # payload checks below only prove that each graph performed real D2H.
                torch.cuda.current_stream().wait_stream(handler.d2h_stream)
        graphs.append(graph)

    try:
        for value in (31.0, 42.0):
            for source in sources:
                source.fill_(value)
            torch.cuda.synchronize()
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            for group, tag in zip(groups, tags):
                _assert_cpu_backup(group, tag, value)
            assert sources[0][0].item() == value + 1000
    finally:
        torch.cuda.synchronize()
        graphs.clear()
        groups.clear()
        sources.clear()
