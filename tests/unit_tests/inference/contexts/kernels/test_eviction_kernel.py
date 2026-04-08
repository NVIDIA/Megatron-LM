# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.inference.contexts.kernels.eviction_kernel import triton_evict_overflow


class TestEvictionKernel:

    def _make_state(self, paused, active, max_kv_blocks, blocks_per_paused=2):
        device = "cuda"
        max_requests = paused + active + 16
        total_blocks = max_requests * max_kv_blocks + 64
        total_request_count = paused + active

        kv_block_counts = torch.zeros(max_requests, dtype=torch.int32, device=device)
        kv_block_ids = torch.full(
            (max_requests, max_kv_blocks), -1, dtype=torch.int32, device=device
        )
        request_ids = torch.arange(max_requests, dtype=torch.int32, device=device)
        block_bag = torch.arange(total_blocks, dtype=torch.int32, device=device)
        avail_t = torch.tensor([total_blocks - 1], dtype=torch.int32, device=device)

        # Assign blocks to paused requests
        bid = total_blocks - 2
        for i in range(paused):
            kv_block_counts[i] = blocks_per_paused
            for j in range(blocks_per_paused):
                kv_block_ids[i, j] = bid
                bid -= 1

        return (kv_block_counts, kv_block_ids, request_ids, block_bag, avail_t,
                total_request_count, total_blocks)

    def test_no_overflow(self):
        paused, active = 3, 5
        kv_block_counts, kv_block_ids, req_ids, bag, avail, trc, _ = self._make_state(
            paused, active, 8, blocks_per_paused=2
        )
        # paused_used = 3*2 = 6, limit = 10 → no overflow
        ec, _, _, _, _, sc = triton_evict_overflow(
            kv_block_counts, kv_block_ids, req_ids, bag, avail,
            None, None, paused, active, trc, 10, 8, False,
        )
        assert ec == 0
        assert sc == 0

    def test_basic_eviction(self):
        paused, active = 4, 3
        kv_block_counts, kv_block_ids, req_ids, bag, avail, trc, tb = self._make_state(
            paused, active, 8, blocks_per_paused=3
        )
        initial_avail = avail.item()
        # paused_used = 4*3 = 12, limit = 6 → overflow
        # cumsum: [3, 6, 9, 12]. valid_count where cumsum <= 6: indices 0,1 → valid=2
        # overflow_count = 4-2 = 2
        ec, evict_ids, evict_idxs, src, dst, sc = triton_evict_overflow(
            kv_block_counts, kv_block_ids, req_ids, bag, avail,
            None, None, paused, active, trc, 6, 8, False,
        )

        assert ec > 0
        # Blocks should be released
        assert avail.item() > initial_avail
        # Evicted request block IDs should be -1
        for i in range(ec):
            idx = evict_idxs[i].item()
            assert (kv_block_ids[idx, :] == -1).all()

    def test_no_paused(self):
        """No paused requests → nothing to evict."""
        kv_block_counts = torch.zeros(16, dtype=torch.int32, device="cuda")
        kv_block_ids = torch.full((16, 4), -1, dtype=torch.int32, device="cuda")
        req_ids = torch.arange(16, dtype=torch.int32, device="cuda")
        bag = torch.arange(64, dtype=torch.int32, device="cuda")
        avail = torch.tensor([63], dtype=torch.int32, device="cuda")

        ec, _, _, _, _, sc = triton_evict_overflow(
            kv_block_counts, kv_block_ids, req_ids, bag, avail,
            None, None, 0, 5, 5, 10, 4, False,
        )
        assert ec == 0

    def test_swap_pattern_evict_lt_active(self):
        """When evict < active, swap evicted with rightmost active."""
        paused, active = 4, 6
        kv_block_counts, kv_block_ids, req_ids, bag, avail, trc, _ = self._make_state(
            paused, active, 4, blocks_per_paused=3
        )
        # paused_used=12, limit=3 → lots of overflow
        ec, _, _, src, dst, sc = triton_evict_overflow(
            kv_block_counts, kv_block_ids, req_ids, bag, avail,
            None, None, paused, active, trc, 3, 4, False,
        )

        if ec > 0 and ec < active:
            # Pattern 1: src starts at paused-evict, dst starts at total-evict
            assert sc == ec
            assert src[0].item() == paused - ec
            assert dst[0].item() == trc - ec

    def test_swap_pattern_evict_ge_active(self):
        """When evict >= active, swap active with leftmost evicted."""
        paused, active = 6, 2
        kv_block_counts, kv_block_ids, req_ids, bag, avail, trc, _ = self._make_state(
            paused, active, 4, blocks_per_paused=3
        )
        # paused_used=18, limit=3 → massive overflow, evict >= active
        ec, _, _, src, dst, sc = triton_evict_overflow(
            kv_block_counts, kv_block_ids, req_ids, bag, avail,
            None, None, paused, active, trc, 3, 4, False,
        )

        if ec > 0 and ec >= active:
            # Pattern 2: src starts at paused-evict, dst starts at paused
            assert sc == active
            assert src[0].item() == paused - ec
            assert dst[0].item() == paused
