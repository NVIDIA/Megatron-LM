# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.inference.contexts.kernels.request_completion_kernel import (
    triton_classify_and_release,
)


class TestRequestCompletionKernel:

    def _make_state(self, active_count, paused_count, max_kv_blocks, blocks_per_request=2):
        """Create test state mimicking update_requests context."""
        device = "cuda"
        max_requests = active_count + paused_count + 16
        total_blocks = max_requests * max_kv_blocks + 32

        kv_block_ids = torch.full(
            (max_requests, max_kv_blocks), -1, dtype=torch.int32, device=device
        )
        block_bag = torch.arange(total_blocks, dtype=torch.int32, device=device)
        total_avail = total_blocks - 1 - active_count * blocks_per_request
        total_avail_tensor = torch.tensor([total_avail], dtype=torch.int32, device=device)

        # Assign blocks to active requests
        block_id = total_blocks - 2  # skip dummy block
        for i in range(active_count):
            abs_idx = paused_count + i
            for j in range(blocks_per_request):
                kv_block_ids[abs_idx, j] = block_id
                block_id -= 1

        return kv_block_ids, block_bag, total_avail_tensor, total_avail

    def test_basic_one_finished(self):
        active_count = 4
        paused_count = 0
        max_kv_blocks = 8
        blocks_per_req = 2

        kv_block_ids, block_bag, avail_t, initial_avail = self._make_state(
            active_count, paused_count, max_kv_blocks, blocks_per_req
        )

        # Request 1 (index 1) finishes
        mask = torch.tensor([1, 0, 1, 1], dtype=torch.int32, device="cuda")
        actual_active = (mask == 1).sum().item()  # 3

        fl, ar, nc, fi, nf = triton_classify_and_release(
            mask, kv_block_ids, block_bag, avail_t,
            None, None, actual_active, paused_count, max_kv_blocks, False,
        )

        # 1 finished request released 2 blocks
        assert avail_t.item() == initial_avail + blocks_per_req
        assert nf == 1
        # kv_block_ids for finished request should be -1
        assert (kv_block_ids[1, :] == -1).all()
        # Active requests' blocks should be untouched
        assert (kv_block_ids[0, :blocks_per_req] != -1).all()

    def test_all_finished(self):
        active_count = 3
        paused_count = 0
        max_kv_blocks = 4
        blocks_per_req = 2

        kv_block_ids, block_bag, avail_t, initial_avail = self._make_state(
            active_count, paused_count, max_kv_blocks, blocks_per_req
        )

        mask = torch.tensor([0, 0, 0], dtype=torch.int32, device="cuda")
        actual_active = 0  # all finished

        fl, ar, nc, fi, nf = triton_classify_and_release(
            mask, kv_block_ids, block_bag, avail_t,
            None, None, actual_active, paused_count, max_kv_blocks, False,
        )

        assert nf == 3
        assert nc == 0  # no compaction needed when all finished
        assert avail_t.item() == initial_avail + active_count * blocks_per_req

    def test_none_finished(self):
        active_count = 3
        paused_count = 0
        max_kv_blocks = 4

        kv_block_ids, block_bag, avail_t, initial_avail = self._make_state(
            active_count, paused_count, max_kv_blocks
        )

        mask = torch.tensor([1, 1, 1], dtype=torch.int32, device="cuda")
        actual_active = 3  # none finished

        fl, ar, nc, fi, nf = triton_classify_and_release(
            mask, kv_block_ids, block_bag, avail_t,
            None, None, actual_active, paused_count, max_kv_blocks, False,
        )

        assert nf == 0
        assert nc == 0
        assert avail_t.item() == initial_avail

    def test_compaction_indices_correct(self):
        """Verify compaction indices match the Python reference computation."""
        active_count = 6
        paused_count = 2
        max_kv_blocks = 4

        kv_block_ids, block_bag, avail_t, _ = self._make_state(
            active_count, paused_count, max_kv_blocks
        )

        # mask: [active, finished, active, finished, active, active]
        # actual_active = 4, finished_count = 2
        # partition boundary at index 4
        # finished in left (i < 4): index 1 (abs 3), index 3 (abs 5)
        # active in right (i >= 4): index 4 (abs 6), index 5 (abs 7)
        mask = torch.tensor([1, 0, 1, 0, 1, 1], dtype=torch.int32, device="cuda")
        actual_active = 4  # count of mask==1

        fl, ar, nc, fi, nf = triton_classify_and_release(
            mask, kv_block_ids, block_bag, avail_t,
            None, None, actual_active, paused_count, max_kv_blocks, False,
        )

        assert nf == 2
        assert nc == 2
        # finished_left should be [3, 5] (abs indices of finished in left partition)
        assert fl[0].item() == paused_count + 1  # 3
        assert fl[1].item() == paused_count + 3  # 5
        # active_right should be [6, 7] (abs indices of active in right partition)
        assert ar[0].item() == paused_count + 4  # 6
        assert ar[1].item() == paused_count + 5  # 7

    def test_with_paused_requests(self):
        active_count = 3
        paused_count = 2
        max_kv_blocks = 4
        blocks_per_req = 1

        kv_block_ids, block_bag, avail_t, initial_avail = self._make_state(
            active_count, paused_count, max_kv_blocks, blocks_per_req
        )

        # 1 finished out of 3 active
        mask = torch.tensor([1, 0, 1], dtype=torch.int32, device="cuda")
        actual_active = 2  # count of mask==1

        fl, ar, nc, fi, nf = triton_classify_and_release(
            mask, kv_block_ids, block_bag, avail_t,
            None, None, actual_active, paused_count, max_kv_blocks, False,
        )

        assert nf == 1
        assert avail_t.item() == initial_avail + blocks_per_req
        # Finished request at abs index paused_count + 1 = 3
        assert fi[0].item() == paused_count + 1

    def test_prefix_cache_ref_count_decrement(self):
        """With prefix caching, blocks should have ref_count decremented."""
        active_count = 2
        paused_count = 0
        max_kv_blocks = 4
        total_blocks = 64

        kv_block_ids = torch.full((16, max_kv_blocks), -1, dtype=torch.int32, device="cuda")
        block_bag = torch.arange(total_blocks, dtype=torch.int32, device="cuda")
        avail_t = torch.tensor([total_blocks - 5], dtype=torch.int32, device="cuda")
        ref_counts = torch.zeros(total_blocks, dtype=torch.int32, device="cuda")
        block_hashes = torch.full((total_blocks,), -1, dtype=torch.int64, device="cuda")

        # Request 0 has blocks 60, 61 with ref_count=1
        kv_block_ids[0, 0] = 60
        kv_block_ids[0, 1] = 61
        ref_counts[60] = 1
        ref_counts[61] = 1
        # Request 1 has block 62 with ref_count=1
        kv_block_ids[1, 0] = 62
        ref_counts[62] = 1

        # Request 0 finishes
        mask = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        actual_active = 1  # count of mask==1

        fl, ar, nc, fi, nf = triton_classify_and_release(
            mask, kv_block_ids, block_bag, avail_t,
            ref_counts, block_hashes, actual_active, paused_count, max_kv_blocks, True,
        )

        # Ref counts should be decremented
        assert ref_counts[60].item() == 0
        assert ref_counts[61].item() == 0
        # Unregistered blocks (hash==-1) with ref_count==0 should be returned to pool
        assert avail_t.item() == total_blocks - 5 + 2
        # Request 1's block should be untouched
        assert ref_counts[62].item() == 1
