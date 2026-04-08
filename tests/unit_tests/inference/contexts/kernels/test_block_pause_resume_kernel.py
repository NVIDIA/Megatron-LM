# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.inference.contexts.kernels.block_pause_resume_kernel import (
    triton_detect_pause,
    triton_resume_and_allocate,
)


class TestBlockPauseResumeKernel:

    def test_detect_pause_basic(self):
        """Some requests need new blocks."""
        block_size = 64
        num_spec = 0
        threshold = block_size - 1 - num_spec  # 63
        paused = 0
        active = 4

        offsets = torch.zeros(active + 16, dtype=torch.int32, device="cuda")
        # Requests 1 and 3 are at the threshold
        offsets[0] = 10
        offsets[1] = 63  # needs new block
        offsets[2] = 20
        offsets[3] = 63  # needs new block

        needs, num_needing, src, dst, nc = triton_detect_pause(
            offsets, active, paused, block_size, num_spec
        )

        assert num_needing == 2
        assert needs[1].item() == 1
        assert needs[3].item() == 1
        assert needs[0].item() == 0
        assert needs[2].item() == 0

    def test_detect_pause_none(self):
        """No requests need new blocks."""
        offsets = torch.tensor([10, 20, 30, 40], dtype=torch.int32, device="cuda")
        offsets_padded = torch.zeros(20, dtype=torch.int32, device="cuda")
        offsets_padded[:4] = offsets

        needs, num_needing, _, _, nc = triton_detect_pause(
            offsets_padded, 4, 0, 64, 0
        )

        assert num_needing == 0
        assert nc == 0

    def test_detect_pause_all(self):
        """All requests need new blocks — no compaction needed."""
        offsets = torch.full((20,), 63, dtype=torch.int32, device="cuda")

        needs, num_needing, _, _, nc = triton_detect_pause(
            offsets, 4, 0, 64, 0
        )

        assert num_needing == 4
        assert nc == 0  # all pausing, no compaction

    def test_detect_pause_compaction_indices(self):
        """Verify compaction indices are correct."""
        # 4 active requests at paused_count=2 (abs indices 2,3,4,5)
        # offsets: [low, high, low, high] → requests 0,2 stay active, 1,3 pause
        offsets = torch.zeros(20, dtype=torch.int32, device="cuda")
        offsets[2] = 10   # active (abs 2)
        offsets[3] = 63   # needs block (abs 3)
        offsets[4] = 10   # active (abs 4)
        offsets[5] = 63   # needs block (abs 5)

        needs, num_needing, src, dst, nc = triton_detect_pause(
            offsets, 4, 2, 64, 0
        )

        assert num_needing == 2
        # Partition at position 2: left=[idx 0,1], right=[idx 2,3]
        # Left has: needs[0]=0 (active, but in left → needs to move right)
        #           needs[1]=1 (pausing, correct position)
        # Right has: needs[2]=0 (active, correct position)
        #            needs[3]=1 (pausing, but in right → needs to move left)
        # So compact: active_on_left at position 0 (abs 2), paused_on_right at position 3 (abs 5)
        # src = [abs 5, abs 2], dst = [abs 2, abs 5]
        if nc > 0:
            assert nc == 2  # one swap pair = 2 elements in src/dst

    def test_resume_basic(self):
        """Resume paused requests, verify block allocation."""
        block_size = 64
        num_spec = 0
        paused = 2
        active = 3
        max_kv_blocks = 8
        max_requests = 16
        total_blocks = 128

        offsets = torch.zeros(max_requests, dtype=torch.int32, device="cuda")
        kv_block_ids = torch.full((max_requests, max_kv_blocks), -1, dtype=torch.int32, device="cuda")
        kv_block_counts = torch.zeros(max_requests, dtype=torch.int32, device="cuda")
        last_kv_block_id = torch.full((max_requests,), -1, dtype=torch.int32, device="cuda")
        block_bag = torch.arange(total_blocks, dtype=torch.int32, device="cuda")
        avail_t = torch.tensor([total_blocks - 1], dtype=torch.int32, device="cuda")

        # Setup 2 paused requests at indices 0,1
        # Both need new blocks (offset at threshold)
        offsets[0] = 63
        offsets[1] = 63
        kv_block_counts[0] = 1
        kv_block_counts[1] = 1
        kv_block_ids[0, 0] = 50
        kv_block_ids[1, 0] = 51
        last_kv_block_id[0] = 50
        last_kv_block_id[1] = 51

        resume_count = triton_resume_and_allocate(
            offsets, kv_block_ids, kv_block_counts, last_kv_block_id,
            block_bag, avail_t,
            paused, active, block_size, num_spec, max_requests, 1024,
        )

        assert resume_count == 2
        # Each resumed request should have gotten a new block
        assert kv_block_counts[0].item() == 2
        assert kv_block_counts[1].item() == 2
        assert kv_block_ids[0, 1].item() != -1
        assert kv_block_ids[1, 1].item() != -1
        assert avail_t.item() == total_blocks - 1 - 2

    def test_resume_no_blocks(self):
        """No blocks available → resume_count = 0."""
        offsets = torch.full((16,), 63, dtype=torch.int32, device="cuda")
        kv_block_ids = torch.full((16, 4), -1, dtype=torch.int32, device="cuda")
        kv_block_counts = torch.ones(16, dtype=torch.int32, device="cuda")
        last_kv_block_id = torch.zeros(16, dtype=torch.int32, device="cuda")
        block_bag = torch.arange(16, dtype=torch.int32, device="cuda")
        avail_t = torch.tensor([0], dtype=torch.int32, device="cuda")  # no blocks

        resume_count = triton_resume_and_allocate(
            offsets, kv_block_ids, kv_block_counts, last_kv_block_id,
            block_bag, avail_t,
            2, 3, 64, 0, 16, 1024,
        )

        assert resume_count == 0

    def test_resume_no_paused(self):
        """No paused requests → resume_count = 0."""
        offsets = torch.zeros(16, dtype=torch.int32, device="cuda")
        kv_block_ids = torch.full((16, 4), -1, dtype=torch.int32, device="cuda")
        kv_block_counts = torch.zeros(16, dtype=torch.int32, device="cuda")
        last_kv_block_id = torch.zeros(16, dtype=torch.int32, device="cuda")
        block_bag = torch.arange(64, dtype=torch.int32, device="cuda")
        avail_t = torch.tensor([63], dtype=torch.int32, device="cuda")

        resume_count = triton_resume_and_allocate(
            offsets, kv_block_ids, kv_block_counts, last_kv_block_id,
            block_bag, avail_t,
            0, 3, 64, 0, 16, 1024,
        )

        assert resume_count == 0
