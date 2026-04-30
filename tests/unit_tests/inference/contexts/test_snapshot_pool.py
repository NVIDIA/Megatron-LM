# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.inference.contexts.snapshot_pool import (
    SNAPSHOT_FORWARD_IN_FLIGHT,
    SNAPSHOT_FREE,
    SNAPSHOT_POPULATING,
    SNAPSHOT_RETIRING,
    GpuSnapshotBufferPool,
)


class FakeEvent:
    def __init__(self, complete: bool):
        self.complete = complete
        self.synchronize_calls = 0

    def query(self):
        return self.complete

    def synchronize(self):
        self.complete = True
        self.synchronize_calls += 1


def _pool(slot_count: int = 2) -> GpuSnapshotBufferPool:
    return GpuSnapshotBufferPool(
        slot_count=slot_count,
        max_requests=4,
        max_tokens=16,
        max_kv_blocks=8,
        device=torch.cuda.current_device(),
    )


def test_snapshot_pool_acquires_and_releases_in_order_with_fixed_addresses():
    pool = _pool(slot_count=1)
    first = pool.acquire(step_id=10)
    gpu_ptr = first.gpu_view._buf.data_ptr()
    cpu_ptr = first.cpu_mirror.data_ptr()

    assert first.slot_id == 0
    assert first.state == SNAPSHOT_POPULATING
    assert first.gpu_view.current_dynamic_step_id == 10

    pool.release(first)
    assert first.state == SNAPSHOT_FREE

    second = pool.acquire(step_id=11)
    assert second.slot_id == 0
    assert second.gpu_view._buf.data_ptr() == gpu_ptr
    assert second.cpu_mirror.data_ptr() == cpu_ptr
    assert second.gpu_view.current_dynamic_step_id == 11


def test_snapshot_pool_releases_slots_out_of_order():
    pool = _pool(slot_count=2)
    first = pool.acquire(step_id=1)
    second = pool.acquire(step_id=2)

    pool.release(second)
    reacquired = pool.acquire(step_id=3)

    assert reacquired.slot_id == second.slot_id
    assert first.state == SNAPSHOT_POPULATING
    assert sorted(pool.active_slot_ids) == [0, 1]


def test_snapshot_pool_acquires_specific_slot_for_graph_capture():
    pool = _pool(slot_count=2)
    first = pool.acquire_specific(1, step_id=10)

    assert first.slot_id == 1
    assert first.state == SNAPSHOT_POPULATING
    assert first.gpu_view.current_dynamic_step_id == 10

    with pytest.raises(RuntimeError, match="not reusable"):
        pool.acquire_specific(1, step_id=11)

    pool.release(first)
    second = pool.acquire_specific(1, step_id=12)
    assert second.slot_id == 1
    assert second.gpu_view.current_dynamic_step_id == 12


def test_snapshot_pool_holds_forward_slot_while_another_slot_populates():
    pool = _pool(slot_count=2)
    held = pool.acquire(step_id=1)
    pool.mark_ready(held)
    read_event = FakeEvent(complete=False)
    pool.mark_forward_in_flight(held, gpu_read_event=read_event)

    assert held.state == SNAPSHOT_FORWARD_IN_FLIGHT
    pool.release(held)
    assert held.state == SNAPSHOT_RETIRING

    next_slot = pool.acquire(step_id=2)
    assert next_slot.slot_id != held.slot_id

    read_event.complete = True
    pool.poll_retired()
    assert held.state == SNAPSHOT_FREE


def test_snapshot_pool_requires_gpu_event_and_journal_retirement_before_reuse():
    pool = _pool(slot_count=1)
    held = pool.acquire(step_id=1)
    read_event = FakeEvent(complete=False)
    pool.mark_forward_in_flight(held, gpu_read_event=read_event)

    with pytest.raises(RuntimeError, match="No reusable"):
        pool.acquire(step_id=2)

    pool.release(held)
    assert held.state == SNAPSHOT_RETIRING
    with pytest.raises(RuntimeError, match="No reusable"):
        pool.acquire(step_id=2)

    read_event.complete = True
    reacquired = pool.acquire(step_id=2)
    assert reacquired.slot_id == held.slot_id


def test_snapshot_pool_memory_accounting_includes_graph_captures():
    pool = _pool(slot_count=2)
    accounting = pool.memory_accounting()

    assert accounting["snapshot_slot_count"] == 2
    assert accounting["metadata_buffer_bytes"] > 0
    assert accounting["pinned_mirror_bytes"] == accounting["metadata_buffer_bytes"]
    assert accounting["graph_capture_count"] == 0

    pool.record_graph_cache_lookup(1, hit=False)
    pool.register_graph_capture(1, byte_count=4096, cache_key=("shape-a", 1), max_captures=1)
    pool.record_graph_cache_lookup(1, hit=True)
    pool.register_graph_capture(1, byte_count=8192, cache_key=("shape-a", 1), max_captures=1)
    accounting = pool.memory_accounting()
    assert accounting["graph_capture_count"] == 1
    assert accounting["graph_capture_bytes"] == 4096
    assert accounting["graph_capture_key_count"] == 1
    assert accounting["graph_cache_hits"] == 1
    assert accounting["graph_cache_misses"] == 1

    with pytest.raises(RuntimeError, match="exceeded CUDA graph capture budget"):
        pool.register_graph_capture(1, byte_count=1024, cache_key=("shape-b", 1), max_captures=1)


def test_snapshot_pool_reset_for_suspend_releases_active_slots_and_graphs():
    pool = _pool(slot_count=1)
    held = pool.acquire(step_id=1)
    read_event = FakeEvent(complete=False)
    pool.mark_forward_in_flight(held, gpu_read_event=read_event)
    pool.release(held)
    pool.register_graph_capture(0, byte_count=1024, cache_key=("shape", 0), max_captures=1)

    pool.reset_for_suspend()

    assert held.state == SNAPSHOT_FREE
    assert held.owning_step_id is None
    assert held.last_gpu_read_event is None
    assert read_event.synchronize_calls == 1
    assert pool.active_slot_ids == ()
    accounting = pool.memory_accounting()
    assert accounting["graph_capture_count"] == 0
    assert accounting["graph_capture_key_count"] == 0
