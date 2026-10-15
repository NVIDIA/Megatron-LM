# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
import os

import pytest
import torch

from megatron.core.pipeline_parallel import pinned_activation_buffer
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    ChunkOffloadHandler,
    OffloadTensorPool,
)
from megatron.core.pipeline_parallel.pinned_activation_buffer import PinnedActivationBuffer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pinned copies required")
MIB = 2**20


@pytest.fixture(autouse=True)
def select_device():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    yield
    torch.cuda.synchronize()


def test_fixed_capacity_preserves_live_data_and_reuses_storage():
    arena = OffloadTensorPool(device="cpu", pin_memory=True, capacity_bytes=4 * MIB)
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
    arena.reset()
    arena.clear()
    assert arena.get_pool_status()["global_stats"]["current_in_use"] == 2
    arena.free(first)
    # Individual releases do not permit overwriting outstanding backups.
    with pytest.raises(RuntimeError, match="budget exceeded"):
        arena.allocate((1,), torch.uint8)
    assert torch.all(second == 29)
    arena.free(second)
    # Reuse the same storage for a different shape and dtype after every backup retires.
    again = arena.allocate((MIB, 1), torch.float32)
    assert again.data_ptr() == ptr
    assert arena.get_pool_status()["fixed_buffer"]["capacity_bytes"] == 4 * MIB
    arena.free(again)
    assert arena.get_pool_status()["global_stats"]["current_in_use"] == 0
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


@pytest.mark.parametrize("other_capture", [False, True], ids=["eager", "another-graph"])
def test_split_graph_backups_are_not_overwritten(other_capture):
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

    if other_capture:
        round_trips = torch.cuda.CUDAGraph()
        with torch.cuda.graph(round_trips, stream=stream):
            # Four MiB of traffic reuses a separate one-MiB range within this capture.
            outputs = [handler.reload(handler.offload(eager_source)) for _ in range(4)]
        assert pool.get_pool_status()["fixed_buffer"]["graph_reserved_bytes"] == 2 * MIB

    for _ in range(3):
        pool.reset()
        pool.clear()
        forward.replay()
        if other_capture:
            round_trips.replay()
        else:
            eager_state = handler.offload(eager_source)
        backward.replay()
        if not other_capture:
            outputs = [handler.reload(eager_state)]
        assert torch.all(output == 17)
        assert all(torch.all(result == 29) for result in outputs)
        assert pool.get_pool_status()["fixed_buffer"]["capacity_bytes"] == 2 * MIB

    # Backward capture/replay cannot return graph-owned space to eager allocation.
    with pytest.raises(RuntimeError, match="budget exceeded"):
        pool.allocate((2 * MIB,), torch.uint8)


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
        host = handler.cpu_tensor_pool.allocate((MIB,), torch.uint8)
        with pytest.raises(RuntimeError, match="budget exceeded"):
            handler.cpu_tensor_pool.allocate((1,), torch.uint8)
        handler.cpu_tensor_pool.free(host)
    for value in range(3, 103):
        handler.cpu_tensor_pool.reset()
        handler.cpu_tensor_pool.clear()
        source.fill_(value)
        graph.replay()
        assert all(torch.all(output == value) for output in outputs)
    assert handler.cpu_tensor_pool.get_pool_status()["fixed_buffer"]["capacity_bytes"] == MIB
    torch.cuda.synchronize()


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


@pytest.mark.parametrize("capacity", [0, -MIB, 128, 3 * MIB + 1])
def test_invalid_fixed_capacity(capacity):
    with pytest.raises(ValueError, match="multiple of 256 bytes"):
        PinnedActivationBuffer(capacity)


def _mapping_bytes(address):
    """Return the size of the /proc/self/maps mapping that contains ``address``."""
    with open("/proc/self/maps") as maps:
        for line in maps:
            start, end = (int(x, 16) for x in line.split(maxsplit=1)[0].split("-"))
            if start <= address < end:
                return end - start
    raise AssertionError(f"no mapping contains {address:#x}")


# Each capacity leaves several MiB below the next power of two for page rounding.
@pytest.mark.parametrize("capacity", [16 * MIB + 256, 5 * MIB])
def test_non_power_of_two_capacity_is_pinned_exactly(capacity):
    before = torch.cuda.host_memory_stats()
    arena = PinnedActivationBuffer(capacity)
    after = torch.cuda.host_memory_stats()
    # The storage does not come from the pinned caching allocator, which rounds up.
    assert after["active_requests.allocated"] == before["active_requests.allocated"]
    assert arena._storage.numel() == capacity and arena._storage.is_pinned()
    if os.path.exists("/proc/self/maps"):
        # The driver rounds to whole pages only, never to the next power of two.
        assert capacity <= _mapping_bytes(arena._storage.data_ptr()) < 1 << capacity.bit_length()
    host = arena.allocate((capacity // 4,), torch.float32)
    with pytest.raises(RuntimeError, match="budget exceeded"):
        arena.allocate((1,), torch.uint8)
    source = torch.randn(capacity // 4, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        host.copy_(source, non_blocking=True)
        result = torch.empty_like(source)
        result.copy_(host, non_blocking=True)
        arena.release(host, stream=stream)
    torch.cuda.synchronize()
    assert torch.equal(result, source)


def test_storage_is_freed_only_after_the_last_view(caplog):
    arena = PinnedActivationBuffer(MIB)
    free = arena._free
    view = arena.allocate((MIB,), torch.uint8)
    del arena
    gc.collect()
    # A surviving view keeps the pinned pages allocated and usable.
    assert free.alive and view.is_pinned()
    view.fill_(5)
    assert torch.all(view == 5)
    del view
    gc.collect()
    assert not free.alive
    assert "cudaFreeHost" not in caplog.text


def test_failed_host_allocation_raises(monkeypatch):
    runtime = pinned_activation_buffer._cuda_runtime()
    monkeypatch.setattr(
        runtime, "cudaHostAlloc", lambda *_: (runtime.cudaError_t.cudaErrorMemoryAllocation, 0)
    )
    with pytest.raises(RuntimeError, match=f"cudaHostAlloc of {3 * MIB} bytes"):
        PinnedActivationBuffer(3 * MIB)
