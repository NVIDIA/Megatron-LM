# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.inference.contexts.kv_block_allocator import KVBlockAllocator


class FakeContext:
    def __init__(self):
        self.async_overlap_debug_counters = {
            "reservation_commits": 0,
            "reservation_rollbacks": 0,
        }


def test_kv_reservation_commit_keeps_blocks_owned():
    context = FakeContext()
    allocator = KVBlockAllocator(context, total_count=5, paused_count=0)

    reservation = allocator.reserve_blocks(request_slot=0, count=2, step_id=3)
    allocator.commit_reservation(reservation)

    assert reservation.kv_block_ids == (2, 3)
    assert allocator.total_avail == 2
    assert allocator.get_total_used() == 2
    assert allocator._open_reservations == {}
    assert allocator._committed_reservations[reservation.reservation_id] == reservation
    assert context.async_overlap_debug_counters["reservation_commits"] == 1


def test_kv_reservation_rollback_returns_blocks():
    context = FakeContext()
    allocator = KVBlockAllocator(context, total_count=5, paused_count=0)
    reservation = allocator.reserve_blocks(request_slot=0, count=2, step_id=4)

    allocator.rollback_reservation(reservation)

    assert allocator.total_avail == 4
    assert allocator.get_total_used() == 0
    assert allocator._rolled_back_reservations[reservation.reservation_id] == reservation
    assert context.async_overlap_debug_counters["reservation_rollbacks"] == 1

    allocator.rollback_reservation(reservation)
    assert allocator.total_avail == 4


def test_kv_reservation_reports_unavailable_blocks():
    allocator = KVBlockAllocator(FakeContext(), total_count=3, paused_count=0)

    reservation = allocator.reserve_blocks(request_slot=0, count=3, step_id=5)

    assert reservation is None
    assert allocator.total_avail == 2
    assert allocator._open_reservations == {}


def test_deferred_release_waits_for_snapshot_retirement():
    allocator = KVBlockAllocator(FakeContext(), total_count=5, paused_count=0)
    blocks = allocator.allocate_memory_blocks(2)

    allocator.defer_release_until_snapshot_retired(blocks, snapshot_slot_id=7)

    assert allocator.total_avail == 2
    allocator.release_deferred_blocks_for_snapshot(6)
    assert allocator.total_avail == 2
    allocator.release_deferred_blocks_for_snapshot(7)
    assert allocator.total_avail == 4


def test_reset_clears_reservation_state():
    allocator = KVBlockAllocator(FakeContext(), total_count=5, paused_count=0)
    reservation = allocator.reserve_blocks(request_slot=0, count=1, step_id=6)
    allocator.defer_release_until_snapshot_retired(
        torch.tensor([reservation.kv_block_ids[0]], dtype=torch.int32), snapshot_slot_id=1
    )

    allocator.reset()

    assert allocator.total_avail == 4
    assert allocator._open_reservations == {}
    assert allocator._committed_reservations == {}
    assert allocator._rolled_back_reservations == {}
    assert allocator._deferred_snapshot_block_releases == {}
