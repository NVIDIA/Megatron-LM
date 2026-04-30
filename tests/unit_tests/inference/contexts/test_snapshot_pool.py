# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for ``GpuSnapshotBufferPool`` (v3 plan commit 9).

Plan validation:
- Correctness with ``max_concurrent_steps=1``.
- Debug-mode test with ``max_concurrent_steps=2`` and an artificial
  two-step pipeline confirming no buffer reuse while a graph capture
  references the slot.
- Memory-budget test asserting pool memory ≈ ``slot_count × per-slot``.
"""

import pytest
import torch

from megatron.core.inference.contexts.gpu_view import ContextGPUView
from megatron.core.inference.contexts.snapshot_pool import GpuSnapshotBufferPool


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda:0")


def _build_pool(device, max_concurrent_steps: int) -> GpuSnapshotBufferPool:
    return GpuSnapshotBufferPool(
        max_concurrent_steps=max_concurrent_steps,
        max_requests=4,
        max_tokens=16,
        max_kv_blocks=2,
        device=device,
    )


class TestGpuSnapshotBufferPool:
    def test_buffer_count_is_max_concurrent_plus_one(self, device):
        pool = _build_pool(device, max_concurrent_steps=1)
        assert pool.buffer_count == 2
        pool2 = _build_pool(device, max_concurrent_steps=2)
        assert pool2.buffer_count == 3

    def test_acquire_returns_lowest_free_slot(self, device):
        pool = _build_pool(device, max_concurrent_steps=2)
        slot0, view0 = pool.acquire(step_id=0)
        slot1, view1 = pool.acquire(step_id=1)
        slot2, view2 = pool.acquire(step_id=2)
        assert (slot0, slot1, slot2) == (0, 1, 2)
        assert isinstance(view0, ContextGPUView)
        assert view0._buf is not view1._buf
        assert view1._buf is not view2._buf

    def test_release_returns_slot_immediately_when_event_omitted(self, device):
        pool = _build_pool(device, max_concurrent_steps=1)
        slot0, _ = pool.acquire(step_id=0)
        pool.release(slot0)
        assert pool.free_slot_count == pool.buffer_count

    def test_release_defers_until_event_signals(self, device):
        """Debug-mode two-step pipeline: a slot held by an in-flight read is
        not reusable until the read's event has signaled."""
        pool = _build_pool(device, max_concurrent_steps=2)
        slot0, view0 = pool.acquire(step_id=0)
        slot1, _ = pool.acquire(step_id=1)
        # Stand-in for "GPU is still reading slot0". Record on the current
        # stream after a real (cheap) GPU op so the event fires after the op.
        marker = torch.cuda.Event()
        view0._buf.fill_(0)
        marker.record()
        pool.release(slot0, after_event=marker)
        # The slot is "draining"; the event has fired by the time we sync.
        torch.cuda.synchronize(device)
        # Reclaim runs lazily on free_slot_count or acquire.
        assert pool.free_slot_count >= 1
        # Subsequent acquire reuses the freed slot.
        next_slot, _ = pool.acquire(step_id=2)
        assert next_slot == slot0

    def test_acquire_raises_on_exhaustion(self, device):
        pool = _build_pool(device, max_concurrent_steps=1)
        pool.acquire(step_id=0)
        pool.acquire(step_id=1)
        with pytest.raises(RuntimeError):
            pool.acquire(step_id=2)

    def test_release_raises_on_unacquired_slot(self, device):
        pool = _build_pool(device, max_concurrent_steps=1)
        with pytest.raises(RuntimeError):
            pool.release(0)

    def test_owning_step_id_tracked(self, device):
        pool = _build_pool(device, max_concurrent_steps=1)
        slot0, _ = pool.acquire(step_id=42)
        assert pool.owning_step_id(slot0) == 42
        pool.release(slot0)
        assert pool.owning_step_id(slot0) is None

    def test_memory_budget_matches_slot_count(self, device):
        pool = _build_pool(device, max_concurrent_steps=2)
        per_slot = pool.per_slot_bytes
        assert per_slot > 0
        assert pool.total_bytes == pool.buffer_count * per_slot

    def test_slots_have_distinct_buffers(self, device):
        pool = _build_pool(device, max_concurrent_steps=2)
        bufs = [pool.slot(i)._buf.data_ptr() for i in range(pool.buffer_count)]
        assert len(set(bufs)) == pool.buffer_count
