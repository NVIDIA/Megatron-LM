# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.inference.contexts.kernels.update_requests_kernel import triton_fused_counts


class TestFusedCountsKernel:

    def test_basic_counts(self):
        mask = torch.tensor([1, 0, 1, 1, 0], dtype=torch.int32, device="cuda")
        offsets = torch.zeros(20, dtype=torch.int32, device="cuda")
        # Set some offsets at threshold (63 for block_size=64, num_spec=0)
        offsets[0] = 63  # active request 0 (mask pos 0) needs new block
        offsets[2] = 10  # active request 2 (mask pos 2) doesn't
        offsets[3] = 63  # active request 3 (mask pos 3) needs new block

        active, finished, needs = triton_fused_counts(mask, offsets, 0, 64, 0)

        assert active == 3
        assert finished == 2
        assert needs == 2

    def test_all_active(self):
        mask = torch.tensor([1, 1, 1], dtype=torch.int32, device="cuda")
        offsets = torch.full((16,), 10, dtype=torch.int32, device="cuda")

        active, finished, needs = triton_fused_counts(mask, offsets, 0, 64, 0)

        assert active == 3
        assert finished == 0
        assert needs == 0

    def test_all_finished(self):
        mask = torch.tensor([0, 0, 0], dtype=torch.int32, device="cuda")
        offsets = torch.zeros(16, dtype=torch.int32, device="cuda")

        active, finished, needs = triton_fused_counts(mask, offsets, 0, 64, 0)

        assert active == 0
        assert finished == 3
        assert needs == 0

    def test_with_paused_offset(self):
        """Paused requests offset the index into last_kv_block_offset."""
        mask = torch.tensor([1, 0, 1], dtype=torch.int32, device="cuda")
        offsets = torch.zeros(16, dtype=torch.int32, device="cuda")
        # Paused at indices 0,1. Active at 2,3,4.
        offsets[2] = 63  # mask pos 0 → abs idx 2 → needs block
        offsets[3] = 10  # mask pos 1 → abs idx 3 → finished (mask=0), skip
        offsets[4] = 63  # mask pos 2 → abs idx 4 → needs block

        active, finished, needs = triton_fused_counts(mask, offsets, 2, 64, 0)

        assert active == 2
        assert finished == 1
        assert needs == 2

    def test_speculative_threshold(self):
        """With speculative tokens, threshold is lower."""
        mask = torch.tensor([1, 1], dtype=torch.int32, device="cuda")
        offsets = torch.zeros(16, dtype=torch.int32, device="cuda")
        # block_size=64, num_spec=2 → threshold=61
        offsets[0] = 61  # at threshold
        offsets[1] = 60  # below threshold

        active, finished, needs = triton_fused_counts(mask, offsets, 0, 64, 2)

        assert active == 2
        assert finished == 0
        assert needs == 1

    def test_empty_mask(self):
        mask = torch.empty(0, dtype=torch.int32, device="cuda")
        offsets = torch.zeros(16, dtype=torch.int32, device="cuda")

        active, finished, needs = triton_fused_counts(mask, offsets, 0, 64, 0)

        assert active == 0
        assert finished == 0
        assert needs == 0
