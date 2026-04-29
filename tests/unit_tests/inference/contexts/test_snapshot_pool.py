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

    def query(self):
        return self.complete


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

    pool.register_graph_capture(1, byte_count=4096)
    accounting = pool.memory_accounting()
    assert accounting["graph_capture_count"] == 1
    assert accounting["graph_capture_bytes"] == 4096
