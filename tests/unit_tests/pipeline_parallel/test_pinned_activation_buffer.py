# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

import pytest
import torch

from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    ChunkOffloadHandler,
    OffloadTensorGroup,
    OffloadTensorPool,
)
from megatron.core.pipeline_parallel.pinned_activation_buffer import PinnedActivationBuffer
from megatron.core.transformer.transformer_config import TransformerConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pinned copies required")
MIB = 2**20


@pytest.fixture(autouse=True)
def select_device():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    yield
    torch.cuda.synchronize()


def test_fixed_capacity_preserves_live_data_and_reuses_storage():
    arena = PinnedActivationBuffer(4 * MIB)
    assert arena.capacity_bytes == 4 * MIB
    first = arena.allocate((MIB,), torch.uint8)
    first.fill_(17)
    ptr = first.data_ptr()
    second = arena.allocate((3 * MIB,), torch.uint8)
    second.fill_(29)
    assert second.data_ptr() == ptr + MIB
    assert first.data_ptr() == ptr and torch.all(first == 17)
    with pytest.raises(RuntimeError, match="budget exceeded"):
        arena.allocate((1,), torch.uint8)
    assert torch.all(second == 29)
    arena.release(first)
    # Individual releases do not permit overwriting outstanding backups.
    with pytest.raises(RuntimeError, match="budget exceeded"):
        arena.allocate((1,), torch.uint8)
    arena.release(second)
    again = arena.allocate((4 * MIB,), torch.uint8)
    assert arena.capacity_bytes == 4 * MIB and again.data_ptr() == ptr
    arena.release(again)
    torch.cuda.synchronize()


@pytest.mark.parametrize("explicit_stream", [False, True])
def test_reuse_waits_for_all_reader_streams(explicit_stream):
    arena = OffloadTensorPool(device="cpu", pin_memory=True, capacity_bytes=2 * MIB)
    a = arena.allocate((MIB,), torch.uint8)
    b = arena.allocate((MIB,), torch.uint8)
    a.fill_(17)
    b.fill_(29)
    streams = [torch.cuda.Stream() for _ in range(4)]
    outputs = [torch.empty(MIB, dtype=torch.uint8, device="cuda") for _ in range(2)]
    for host, output, stream in zip((a, b), outputs, streams[:2]):
        with torch.cuda.stream(stream):
            torch.cuda._sleep(10_000_000)
            output.copy_(host, non_blocking=True)
            arena.free(host, stream=stream if explicit_stream else None)
    arena.reset()
    arena.clear()
    # Reuse before either old read completes, on two distinct writer streams.
    source = torch.full((MIB,), 43, dtype=torch.uint8, device="cuda")
    for writer in streams[2:]:
        writer.wait_stream(torch.cuda.current_stream())
    new = []
    for stream in streams[2:]:
        with torch.cuda.stream(stream):
            host = arena.allocate((MIB,), torch.uint8, stream=stream if explicit_stream else None)
            host.copy_(source, non_blocking=True)
            new.append(host)
    torch.cuda.synchronize()
    assert torch.all(outputs[0] == 17) and torch.all(outputs[1] == 29)
    assert all(torch.all(host == 43) for host in new)
    for host in new:
        arena.free(host)
    torch.cuda.synchronize()


def test_offload_views_and_non_lifo_release():
    handler = ChunkOffloadHandler.__new__(ChunkOffloadHandler)
    handler.cpu_tensor_pool = OffloadTensorPool(device="cpu", pin_memory=True, capacity_bytes=MIB)
    base = torch.randn(64, 96, device="cuda", dtype=torch.bfloat16)
    inputs = [base[:, :80].detach(), base[:, :8], torch.arange(3072, device="cuda")]
    states = [handler.offload(x, use_cpu_pool=False) for x in inputs]
    for index in (0, 2, 1):
        recovered = handler.reload(states[index])
        torch.cuda.synchronize()
        assert torch.equal(recovered, inputs[index])
        if index == 0:
            assert recovered.stride() == inputs[index].stride()
    assert handler.cpu_tensor_pool.get_pool_status()["fixed_buffer"]["live_bytes"] == 0


def test_graph_replay_keeps_backing_addresses():
    handler = ChunkOffloadHandler.__new__(ChunkOffloadHandler)
    handler.cpu_tensor_pool = OffloadTensorPool(device="cpu", pin_memory=True, capacity_bytes=MIB)
    source = torch.ones(8192, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        handler.reload(handler.offload(source, use_cpu_pool=False))
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = handler.reload(handler.offload(source, use_cpu_pool=False))
        # A second epoch within capture exercises the reuse event dependency.
        output2 = handler.reload(handler.offload(source, use_cpu_pool=False))
    for value in (3, 5, 7):
        handler.cpu_tensor_pool.reset()
        handler.cpu_tensor_pool.clear()
        source.fill_(value)
        graph.replay()
        assert torch.all(output == value) and torch.all(output2 == value)
    torch.cuda.synchronize()


def test_capture_exhaustion_preserves_fixed_capacity():
    arena = PinnedActivationBuffer(MIB)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        host = arena.allocate((MIB,), torch.uint8)
        with pytest.raises(RuntimeError, match="budget exceeded"):
            arena.allocate((1,), torch.uint8)
        arena.release(host)
    torch.cuda.synchronize()
    assert arena.capacity_bytes == MIB


def test_split_graph_backups_are_not_overwritten_by_eager_offloads():
    handler = ChunkOffloadHandler.__new__(ChunkOffloadHandler)
    pool = OffloadTensorPool(device="cpu", pin_memory=True, capacity_bytes=2 * MIB)
    handler.cpu_tensor_pool = pool
    source = torch.full((MIB,), 17, dtype=torch.uint8, device="cuda")
    eager_source = torch.full_like(source, 29)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        handler.reload(handler.offload(source))
    stream.synchronize()

    forward, backward = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(forward, stream=stream):
        state = handler.offload(source)
    with torch.cuda.graph(backward, stream=stream):
        output = handler.reload(state)

    for _ in range(3):
        pool.reset()
        pool.clear()
        forward.replay()
        eager_state = handler.offload(eager_source)
        backward.replay()
        eager_output = handler.reload(eager_state)
        assert torch.all(output == 17)
        assert torch.all(eager_output == 29)
        assert pool.get_pool_status()["fixed_buffer"]["capacity_bytes"] == 2 * MIB

    # Backward capture/replay cannot return graph-owned space to eager allocation.
    with pytest.raises(RuntimeError, match="budget exceeded"):
        pool.allocate((2 * MIB,), torch.uint8)


def test_each_capture_reserves_its_own_reusable_range():
    handler = ChunkOffloadHandler.__new__(ChunkOffloadHandler)
    pool = OffloadTensorPool(device="cpu", pin_memory=True, capacity_bytes=2 * MIB)
    handler.cpu_tensor_pool = pool
    source = torch.full((MIB,), 17, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        handler.reload(handler.offload(source))
    stream.synchronize()

    forward, backward, round_trips = (torch.cuda.CUDAGraph() for _ in range(3))
    with torch.cuda.graph(forward, stream=stream):
        state = handler.offload(source)
    with torch.cuda.graph(backward, stream=stream):
        output = handler.reload(state)
    with torch.cuda.graph(round_trips, stream=stream):
        # Four MiB of traffic fits in one additional MiB by reusing within capture.
        outputs = [handler.reload(handler.offload(source)) for _ in range(4)]
    assert pool.get_pool_status()["fixed_buffer"]["graph_reserved_bytes"] == 2 * MIB
    forward.replay()
    source.fill_(29)
    round_trips.replay()
    backward.replay()
    assert torch.all(output == 17)
    assert all(torch.all(result == 29) for result in outputs)
    with pytest.raises(RuntimeError, match="budget exceeded"):
        pool.allocate((1,), torch.uint8)


def test_captured_read_reserves_an_eager_backup():
    handler = ChunkOffloadHandler.__new__(ChunkOffloadHandler)
    pool = OffloadTensorPool(device="cpu", pin_memory=True, capacity_bytes=2 * MIB)
    handler.cpu_tensor_pool = pool
    source = torch.full((MIB,), 17, dtype=torch.uint8, device="cuda")
    state = handler.offload(source)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = handler.reload(state)
    eager_state = handler.offload(torch.full_like(source, 29))
    graph.replay()
    eager_output = handler.reload(eager_state)
    assert torch.all(output == 17)
    assert torch.all(eager_output == 29)


@pytest.mark.parametrize("explicit_stream", [False, True])
def test_graph_replay_with_multiple_copy_streams_and_epochs(explicit_stream):
    handler = ChunkOffloadHandler.__new__(ChunkOffloadHandler)
    handler.cpu_tensor_pool = OffloadTensorPool(device="cpu", pin_memory=True, capacity_bytes=MIB)
    source = torch.ones(8192, dtype=torch.bfloat16, device="cuda")
    capture, writer, reader = (torch.cuda.Stream() for _ in range(3))
    capture.wait_stream(torch.cuda.current_stream())

    def round_trip():
        writer.wait_stream(capture)
        with torch.cuda.stream(writer):
            state = handler.offload(
                source, use_cpu_pool=False, stream=writer if explicit_stream else None
            )
        reader.wait_stream(writer)
        with torch.cuda.stream(reader):
            output = handler.reload(state, stream=reader if explicit_stream else None)
        capture.wait_stream(reader)
        return output

    with torch.cuda.stream(capture):
        round_trip()
    capture.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture):
        outputs = [round_trip() for _ in range(8)]
    for value in range(3, 103):
        handler.cpu_tensor_pool.reset()
        source.fill_(value)
        graph.replay()
        assert all(torch.all(output == value) for output in outputs)
    torch.cuda.synchronize()


def test_pool_reset_preserves_outstanding_backups():
    pool = OffloadTensorPool(device="cpu", pin_memory=True, capacity_bytes=MIB)
    host = pool.allocate((MIB // 4,), torch.float32)
    host.fill_(7)
    pointer = host.data_ptr()
    pool.reset()
    pool.clear()
    with pytest.raises(RuntimeError, match="budget exceeded"):
        pool.allocate((1,), torch.uint8)
    assert torch.all(host == 7)
    assert pool.get_pool_status()["global_stats"]["current_in_use"] == 1
    pool.free(host)
    # A different shape and dtype can reuse the exact same backing storage.
    reused = pool.allocate((MIB // 2, 2), torch.uint8)
    assert reused.data_ptr() == pointer
    pool.free(reused)
    assert pool.get_pool_status()["global_stats"]["current_in_use"] == 0


@pytest.mark.parametrize("capacity", [0, MIB])
@pytest.mark.parametrize("group_name", ["fused_group_mlp", "moe_act", "expert_fc1", "core_attn"])
def test_pool_routes_variable_shape_groups(capacity, group_name):
    pool = OffloadTensorPool(device="cpu", pin_memory=True, capacity_bytes=capacity)
    handler = ChunkOffloadHandler.__new__(ChunkOffloadHandler)
    handler.cpu_tensor_pool = pool
    group = OffloadTensorGroup(group_name)
    source = torch.randn(32, 64, device="cuda")
    state = handler.offload(source, use_cpu_pool=group.use_cpu_pool)
    assert state[2] is bool(capacity or group.use_cpu_pool)
    result = handler.reload(state)
    assert torch.equal(source, result)
    assert pool.get_pool_status()["global_stats"]["current_in_use"] == 0
    if capacity:
        assert pool.get_pool_status()["pools"] == {}
        assert pool.get_pool_status()["fixed_buffer"]["capacity_bytes"] == capacity


def test_pool_cannot_change_backing_storage_after_use():
    pool = OffloadTensorPool(device="cpu", pin_memory=True)
    host = pool.allocate((16,), torch.uint8)
    pool.free(host)
    with pytest.raises(RuntimeError, match="before allocations"):
        pool.configure_fixed_buffer(MIB)

    fixed = OffloadTensorPool(device="cpu", pin_memory=True, capacity_bytes=MIB)
    with pytest.raises(RuntimeError, match="only once"):
        fixed.configure_fixed_buffer(2 * MIB)


@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (3, 5), (2, 3, 4)])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.bfloat16, torch.float32, torch.int64])
def test_typed_views_preserve_shape_stride_and_alignment(shape, dtype):
    pool = OffloadTensorPool(device="cpu", pin_memory=True, capacity_bytes=MIB)
    prefix = pool.allocate((7,), torch.uint8)
    tensor = pool.allocate(shape, dtype)
    following = pool.allocate((8,), torch.int64)
    assert tensor.shape == shape and tensor.dtype == dtype
    assert tensor.stride() == torch.empty(shape, dtype=dtype).stride()
    assert tensor.is_contiguous() and tensor.is_pinned()
    if tensor.numel():
        assert tensor.data_ptr() == prefix.data_ptr() + 256
    assert following.data_ptr() % 256 == 0
    prefix.fill_(11)
    tensor.fill_(7)
    following.fill_(13)
    assert torch.all(prefix == 11) and torch.all(following == 13)
    assert torch.all(tensor == 7)
    for host in (tensor, following, prefix):
        pool.free(host)


@pytest.mark.parametrize("capacity", [0, -MIB, 128, 3 * MIB])
def test_invalid_fixed_capacity(capacity):
    with pytest.raises(ValueError, match="power of two"):
        PinnedActivationBuffer(capacity)


@pytest.mark.parametrize("size_gib", [0, 0.5, 8])
def test_pinned_buffer_config(size_gib):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=8,
        fine_grained_activation_offloading=bool(size_gib),
        offload_modules=["moe_act"],
        fine_grained_offloading_buffer_size_gib=size_gib,
    )
    assert config.fine_grained_offloading_buffer_size_gib == size_gib


@pytest.mark.parametrize("size_gib", [-1, float("nan"), float("inf"), 3, 2**-24])
def test_invalid_pinned_buffer_config(size_gib):
    with pytest.raises(ValueError, match="Pinned offload buffer"):
        TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            fine_grained_activation_offloading=True,
            offload_modules=["moe_act"],
            fine_grained_offloading_buffer_size_gib=size_gib,
        )
