# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the experimental Megatron-FSDP trace-pool allocator."""

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.allocator import (
    TracePoolAllocator,
)


def test_trace_key_keeps_storage_identity_across_free() -> None:
    """Trace reallocation must resurrect the Storage seen by saved tensor views."""
    allocator = TracePoolAllocator()
    first = allocator.allocate("weight", 8, torch.float32, "cpu", arena="allgather")
    storage_id = first.untyped_storage()._cdata

    allocator.free("weight")
    assert first.untyped_storage().nbytes() == 0

    second = allocator.allocate("weight", 8, torch.float32, "cpu", arena="allgather")
    assert second.untyped_storage()._cdata == storage_id
    second.fill_(7)
    torch.testing.assert_close(first, torch.full_like(first, 7))


def test_plan_shares_slot_for_non_overlapping_keys() -> None:
    """Non-overlapping keys in one stream arena should share a steady slot."""
    allocator = TracePoolAllocator()
    allocator.allocate("left", 8, torch.float32, "cpu", arena="allgather")
    allocator.free("left")
    allocator.allocate("right", 4, torch.float32, "cpu", arena="allgather")
    allocator.free("right")

    assert allocator.plan() == 8 * torch.empty((), dtype=torch.float32).element_size()
    left = allocator.allocate("left", 8, torch.float32, "cpu", arena="allgather")
    allocator.free("left")
    right = allocator.allocate("right", 4, torch.float32, "cpu", arena="allgather")
    assert left.data_ptr() == right.data_ptr()


def test_plan_does_not_share_slots_across_stream_arenas() -> None:
    """Host lifetimes from independently executing CUDA streams must not alias."""
    allocator = TracePoolAllocator()
    allocator.allocate("ag", 8, torch.float32, "cpu", arena="allgather")
    allocator.free("ag")
    allocator.allocate("rs", 8, torch.float32, "cpu", arena="reduce_scatter")
    allocator.free("rs")
    allocator.plan()

    ag = allocator.allocate("ag", 8, torch.float32, "cpu", arena="allgather")
    rs = allocator.allocate("rs", 8, torch.float32, "cpu", arena="reduce_scatter")
    assert ag.data_ptr() != rs.data_ptr()


def test_optimized_slot_collision_reports_schedule_divergence() -> None:
    """A later lifetime overlap that differs from the trace should fail loudly."""
    allocator = TracePoolAllocator()
    allocator.allocate("first", 8, torch.float32, "cpu", arena="allgather")
    allocator.free("first")
    allocator.allocate("second", 8, torch.float32, "cpu", arena="allgather")
    allocator.free("second")
    allocator.plan()

    allocator.allocate("first", 8, torch.float32, "cpu", arena="allgather")
    with pytest.raises(RuntimeError, match="still used"):
        allocator.allocate("second", 8, torch.float32, "cpu", arena="allgather")


def test_plan_rejects_live_trace_allocations() -> None:
    """Planning is legal only at a quiescent global-batch boundary."""
    allocator = TracePoolAllocator()
    allocator.allocate("live", 1, torch.float32, "cpu", arena="allgather")

    with pytest.raises(RuntimeError, match="live allocations"):
        allocator.plan()
