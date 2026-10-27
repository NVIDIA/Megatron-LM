# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import pytest

from megatron.core.resharding.nvshmem_copy_service.nvshmem_types import SendRequest
from megatron.core.resharding.nvshmem_copy_service.planning.workload_packer import WorkloadPacker


class TestWorkloadPacker:
    """Test suite for WorkloadPacker."""

    def test_pack_single_request(self):
        """Test packing a single send request."""
        packer = WorkloadPacker()

        requests = [SendRequest(task_id=1, src_tensor=None, src_pos=0, size=1000, dest_pe=1)]
        workloads = packer.pack_workloads(requests, n_pes=2)

        # Should create one batch for destination PE 1
        assert 1 in workloads
        assert len(workloads[1]) == 1
        assert workloads[1][0].total_size == 1000
        assert len(workloads[1][0].tasks) == 1

    def test_pack_multiple_requests_same_dest(self):
        """Test packing multiple requests to the same destination."""
        packer = WorkloadPacker()

        requests = [
            SendRequest(task_id=1, src_tensor=None, src_pos=0, size=1000, dest_pe=1),
            SendRequest(task_id=2, src_tensor=None, src_pos=0, size=2000, dest_pe=1),
            SendRequest(task_id=3, src_tensor=None, src_pos=0, size=3000, dest_pe=1),
        ]
        workloads = packer.pack_workloads(requests, n_pes=2)

        # All requests fit in one batch (under 256MB default limit)
        assert 1 in workloads
        assert len(workloads[1]) == 1
        assert workloads[1][0].total_size == 6000
        assert len(workloads[1][0].tasks) == 3

    def test_pack_exceeds_batch_size(self):
        """Test that requests are split when size limit exceeded."""
        packer = WorkloadPacker()

        # Create requests that exceed 256MB limit
        mb_256 = 256 * 1024 * 1024
        requests = [
            SendRequest(task_id=1, src_tensor=None, src_pos=0, size=mb_256 - 1000, dest_pe=1),
            SendRequest(task_id=2, src_tensor=None, src_pos=0, size=5000, dest_pe=1),
            SendRequest(task_id=3, src_tensor=None, src_pos=0, size=2000, dest_pe=1),
        ]
        workloads = packer.pack_workloads(requests, n_pes=2)

        # Should create 2 batches (first request alone, others together)
        assert 1 in workloads
        assert len(workloads[1]) == 2

    def test_pack_multiple_destinations(self):
        """Test packing requests to multiple destinations."""
        packer = WorkloadPacker()

        requests = [
            SendRequest(task_id=1, src_tensor=None, src_pos=0, size=1000, dest_pe=1),
            SendRequest(task_id=2, src_tensor=None, src_pos=0, size=2000, dest_pe=2),
            SendRequest(task_id=3, src_tensor=None, src_pos=0, size=3000, dest_pe=1),
            SendRequest(task_id=4, src_tensor=None, src_pos=0, size=4000, dest_pe=3),
        ]
        workloads = packer.pack_workloads(requests, n_pes=4)

        # PE 1: requests 1 and 3 (4000 total)
        assert len(workloads[1]) == 1
        assert workloads[1][0].total_size == 4000

        # PE 2: request 2 (2000 total)
        assert len(workloads[2]) == 1
        assert workloads[2][0].total_size == 2000

        # PE 3: request 4 (4000 total)
        assert len(workloads[3]) == 1
        assert workloads[3][0].total_size == 4000

    def test_pack_empty_requests(self):
        """Test packing with no requests."""
        packer = WorkloadPacker()
        workloads = packer.pack_workloads([], n_pes=4)
        # All PEs should have empty lists
        for pe in range(4):
            assert pe in workloads
            assert len(workloads[pe]) == 0

    def test_pack_descending_size_order(self):
        """Test that packing sorts by size descending (largest first)."""
        packer = WorkloadPacker()

        requests = [
            SendRequest(task_id=1, src_tensor=None, src_pos=0, size=1000, dest_pe=1),
            SendRequest(task_id=2, src_tensor=None, src_pos=0, size=5000, dest_pe=1),
            SendRequest(task_id=3, src_tensor=None, src_pos=0, size=3000, dest_pe=1),
            SendRequest(task_id=4, src_tensor=None, src_pos=0, size=2000, dest_pe=1),
        ]
        workloads = packer.pack_workloads(requests, n_pes=2)

        # All should be in one batch
        assert 1 in workloads
        assert len(workloads[1]) == 1

        # Check that tasks are sorted by size (descending)
        sizes = [req.size for req in workloads[1][0].tasks]
        assert sizes == sorted(sizes, reverse=True)

    def test_pack_preserves_task_ids(self):
        """Test that packing preserves task IDs."""
        packer = WorkloadPacker()

        requests = [
            SendRequest(task_id=100, src_tensor=None, src_pos=0, size=1000, dest_pe=1),
            SendRequest(task_id=200, src_tensor=None, src_pos=0, size=2000, dest_pe=1),
            SendRequest(task_id=300, src_tensor=None, src_pos=0, size=3000, dest_pe=1),
        ]
        workloads = packer.pack_workloads(requests, n_pes=2)

        # All requests should be in one batch
        assert 1 in workloads
        assert len(workloads[1]) == 1

        # Check task IDs are preserved (sorted by size descending: 300, 200, 100)
        task_ids = [req.task_id for req in workloads[1][0].tasks]
        assert task_ids == [300, 200, 100]
