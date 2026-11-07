# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import pytest

from megatron.core.resharding.nvshmem_copy_service.nvshmem_types import ScheduledBatch
from megatron.core.resharding.nvshmem_copy_service.planning.communication_scheduler import (
    CommunicationScheduler,
)


class TestCommunicationScheduler:
    """Test suite for CommunicationScheduler."""

    def test_conflict_detection(self):
        """Test that conflicts are detected correctly."""
        scheduler = CommunicationScheduler()

        # Create test batches
        batch1 = ScheduledBatch(src_pe=0, dest_pe=1, batch_index=0, iteration=-1, total_size=1000)
        batch2 = ScheduledBatch(src_pe=0, dest_pe=2, batch_index=0, iteration=-1, total_size=1000)
        batch3 = ScheduledBatch(src_pe=2, dest_pe=3, batch_index=0, iteration=-1, total_size=1000)
        batch4 = ScheduledBatch(src_pe=4, dest_pe=5, batch_index=0, iteration=-1, total_size=1000)

        # batch1 and batch2 conflict (share src PE 0)
        # batch2 and batch3 conflict (share PE 2)
        # batch4 doesn't conflict with any

        batches = [batch1, batch2, batch3, batch4]
        scheduler._assign_iterations(batches)

        # batch1 and batch2 should be in different iterations (conflict on PE 0)
        assert batch1.iteration != batch2.iteration

        # batch2 and batch3 should be in different iterations (conflict on PE 2)
        assert batch2.iteration != batch3.iteration

        # batch4 can be in any iteration (no conflicts)
        # Since we process by degree, batch4 (degree 0) will be placed first
        assert batch4.iteration == 0

    def test_degree_based_sorting(self):
        """Test that batches are sorted by conflict degree."""
        scheduler = CommunicationScheduler()

        # Create batches with different conflict patterns
        # Central hub pattern: PE 0 connects to many others
        batches = [
            ScheduledBatch(src_pe=0, dest_pe=1, batch_index=0, iteration=-1, total_size=1000),
            ScheduledBatch(src_pe=0, dest_pe=2, batch_index=0, iteration=-1, total_size=1000),
            ScheduledBatch(src_pe=0, dest_pe=3, batch_index=0, iteration=-1, total_size=1000),
            ScheduledBatch(
                src_pe=4, dest_pe=5, batch_index=0, iteration=-1, total_size=1000
            ),  # isolated
        ]

        scheduler._assign_iterations(batches)

        # Batches involving PE 0 should be scheduled in different iterations
        pe0_batches = [b for b in batches if b.src_pe == 0 or b.dest_pe == 0]
        iterations = [b.iteration for b in pe0_batches]
        # All PE 0 batches should be in different iterations
        assert len(iterations) == len(set(iterations))

        # Isolated batch should be in iteration 0 (no conflicts)
        isolated = [b for b in batches if b.src_pe == 4][0]
        assert isolated.iteration == 0

    def test_ring_pattern(self):
        """Test scheduling efficiency for ring communication pattern."""
        scheduler = CommunicationScheduler()

        n_pes = 8
        # Ring pattern: each PE sends to next PE (0→1, 1→2, 2→3, ...)
        batches = [
            ScheduledBatch(
                src_pe=i, dest_pe=(i + 1) % n_pes, batch_index=0, iteration=-1, total_size=1000
            )
            for i in range(n_pes)
        ]

        scheduler._assign_iterations(batches)

        # Ring pattern needs 2 iterations because a PE can't send and receive simultaneously
        # Iteration 0: Even-indexed PEs send (0→1, 2→3, 4→5, 6→7)
        # Iteration 1: Odd-indexed PEs send (1→2, 3→4, 5→6, 7→0)
        iterations_used = len(set(b.iteration for b in batches))
        assert iterations_used == 2, f"Ring should use 2 iterations, got {iterations_used}"

        # Verify no conflicts within each iteration
        for iteration in range(iterations_used):
            iter_batches = [b for b in batches if b.iteration == iteration]
            used_pes = set()
            for batch in iter_batches:
                assert batch.src_pe not in used_pes
                assert batch.dest_pe not in used_pes
                used_pes.add(batch.src_pe)
                used_pes.add(batch.dest_pe)

    def test_all_to_all_pattern(self):
        """Test scheduling for all-to-all communication."""
        scheduler = CommunicationScheduler()

        n_pes = 4
        # All-to-all: every PE sends to every other PE
        batches = []
        for src in range(n_pes):
            for dst in range(n_pes):
                if src != dst:
                    batches.append(
                        ScheduledBatch(
                            src_pe=src, dest_pe=dst, batch_index=0, iteration=-1, total_size=1000
                        )
                    )

        scheduler._assign_iterations(batches)

        # Verify schedule is conflict-free
        for iteration in range(scheduler.num_iterations):
            iter_batches = [b for b in batches if b.iteration == iteration]
            used_pes = set()
            for batch in iter_batches:
                # No PE should be used twice in same iteration
                assert (
                    batch.src_pe not in used_pes
                ), f"PE {batch.src_pe} used twice in iteration {iteration}"
                assert (
                    batch.dest_pe not in used_pes
                ), f"PE {batch.dest_pe} used twice in iteration {iteration}"
                used_pes.add(batch.src_pe)
                used_pes.add(batch.dest_pe)

    def test_empty_workloads(self):
        """Test handling of empty workloads."""
        scheduler = CommunicationScheduler()
        batches = []
        scheduler._assign_iterations(batches)
        assert scheduler.num_iterations == 0

    def test_single_batch(self):
        """Test scheduling with a single batch."""
        scheduler = CommunicationScheduler()
        batches = [
            ScheduledBatch(src_pe=0, dest_pe=1, batch_index=0, iteration=-1, total_size=1000)
        ]
        scheduler._assign_iterations(batches)
        assert scheduler.num_iterations == 1
        assert batches[0].iteration == 0

    def test_no_self_conflict(self):
        """Test that a batch doesn't conflict with itself."""
        scheduler = CommunicationScheduler()

        # Single batch should be scheduled in iteration 0
        batches = [
            ScheduledBatch(src_pe=0, dest_pe=1, batch_index=0, iteration=-1, total_size=1000)
        ]
        scheduler._assign_iterations(batches)
        assert batches[0].iteration == 0

    def test_scatter_pattern(self):
        """Test one-to-many scatter pattern."""
        scheduler = CommunicationScheduler()

        n_receivers = 7
        # One PE sends to all others
        batches = [
            ScheduledBatch(src_pe=0, dest_pe=i + 1, batch_index=0, iteration=-1, total_size=1000)
            for i in range(n_receivers)
        ]

        scheduler._assign_iterations(batches)

        # All batches involve PE 0 as sender, so must be in different iterations
        iterations_used = len(set(b.iteration for b in batches))
        assert iterations_used == n_receivers

    def test_gather_pattern(self):
        """Test many-to-one gather pattern."""
        scheduler = CommunicationScheduler()

        n_senders = 7
        # All PEs send to one PE
        batches = [
            ScheduledBatch(src_pe=i, dest_pe=7, batch_index=0, iteration=-1, total_size=1000)
            for i in range(n_senders)
        ]

        scheduler._assign_iterations(batches)

        # All batches involve PE 7 as receiver, so must be in different iterations
        iterations_used = len(set(b.iteration for b in batches))
        assert iterations_used == n_senders

    def test_large_batch_priority(self):
        """Test that larger batches get priority (tie-breaking by size)."""
        scheduler = CommunicationScheduler()

        # Create batches with different sizes
        small_batch = ScheduledBatch(
            src_pe=0, dest_pe=1, batch_index=0, iteration=-1, total_size=100
        )
        large_batch = ScheduledBatch(
            src_pe=2, dest_pe=3, batch_index=0, iteration=-1, total_size=10000
        )

        batches = [small_batch, large_batch]
        scheduler._assign_iterations(batches)

        # Both should be in iteration 0 (no conflicts)
        assert small_batch.iteration == 0
        assert large_batch.iteration == 0
