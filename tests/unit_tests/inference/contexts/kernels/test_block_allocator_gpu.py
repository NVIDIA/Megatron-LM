# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.inference.contexts.kernels.block_allocator_gpu import (
    triton_allocate_blocks,
    triton_release_blocks,
)


class TestBlockAllocatorGPU:

    def _make_allocator_state(self, total_count, initial_avail=None):
        """Create allocator state tensors mimicking KVBlockAllocator."""
        device = "cuda"
        if initial_avail is None:
            initial_avail = total_count - 1  # -1 for dummy block
        block_bag = torch.arange(total_count, dtype=torch.int32, device=device)
        total_avail_tensor = torch.tensor([initial_avail], dtype=torch.int32, device=device)
        ref_counts = torch.zeros(total_count, dtype=torch.int32, device=device)
        return block_bag, total_avail_tensor, ref_counts

    def test_allocate_basic(self):
        block_bag, avail, ref_counts = self._make_allocator_state(64)
        initial_avail = avail.item()
        num_blocks = 5

        output = torch.empty(num_blocks, dtype=torch.int32, device="cuda")
        triton_allocate_blocks(block_bag, avail, output, num_blocks)

        assert avail.item() == initial_avail - num_blocks
        expected = block_bag[avail.item() : avail.item() + num_blocks]
        assert torch.equal(output, expected)

    def test_allocate_matches_python_reference(self):
        total = 128
        num_alloc = 10
        block_bag, avail, ref_counts = self._make_allocator_state(total)

        # Python reference
        ref_avail = avail.item()
        ref_avail -= num_alloc
        ref_ids = block_bag[ref_avail : ref_avail + num_alloc].clone()

        # Triton
        output = torch.empty(num_alloc, dtype=torch.int32, device="cuda")
        triton_allocate_blocks(block_bag, avail, output, num_alloc)

        assert avail.item() == ref_avail
        assert torch.equal(output, ref_ids)

    def test_release_matches_python_reference(self):
        total = 128
        block_bag, avail, ref_counts = self._make_allocator_state(total)

        # Allocate some blocks first (via Python path)
        num_alloc = 10
        old_avail = avail.item()
        avail.fill_(old_avail - num_alloc)
        allocated = block_bag[avail.item() : avail.item() + num_alloc].clone()

        # Now release them
        # Python reference
        ref_bag = block_bag.clone()
        ref_avail = avail.item()
        ref_bag[ref_avail : ref_avail + num_alloc] = allocated
        ref_avail += num_alloc

        # Triton
        triton_release_blocks(block_bag, avail, allocated, num_alloc)

        assert avail.item() == ref_avail
        assert torch.equal(block_bag[ref_avail - num_alloc : ref_avail], allocated)

    def test_allocate_then_release_roundtrip(self):
        total = 64
        block_bag, avail, ref_counts = self._make_allocator_state(total)
        initial_avail = avail.item()

        # Allocate all available
        output = torch.empty(initial_avail, dtype=torch.int32, device="cuda")
        triton_allocate_blocks(block_bag, avail, output, initial_avail)
        assert avail.item() == 0

        # Release all back
        triton_release_blocks(block_bag, avail, output, initial_avail)
        assert avail.item() == initial_avail

    def test_allocate_with_prefix_cache_sets_ref_counts(self):
        total = 64
        block_bag, avail, ref_counts = self._make_allocator_state(total)
        num_alloc = 8

        output = torch.empty(num_alloc, dtype=torch.int32, device="cuda")
        triton_allocate_blocks(
            block_bag, avail, output, num_alloc,
            ref_counts=ref_counts, has_prefix_cache=True,
        )

        # All allocated blocks should have ref_count == 1
        assert (ref_counts[output.long()] == 1).all()
        # Unallocated blocks should still be 0
        assert ref_counts.sum().item() == num_alloc

    def test_empty_allocate(self):
        block_bag, avail, ref_counts = self._make_allocator_state(32)
        initial_avail = avail.item()

        output = torch.empty(0, dtype=torch.int32, device="cuda")
        triton_allocate_blocks(block_bag, avail, output, 0)
        assert avail.item() == initial_avail

    def test_empty_release(self):
        block_bag, avail, ref_counts = self._make_allocator_state(32)
        initial_avail = avail.item()

        blocks = torch.empty(0, dtype=torch.int32, device="cuda")
        triton_release_blocks(block_bag, avail, blocks, 0)
        assert avail.item() == initial_avail

    def test_allocate_exact_capacity(self):
        total = 16
        block_bag, avail, ref_counts = self._make_allocator_state(total)
        capacity = avail.item()  # 15 (total - 1 for dummy)

        output = torch.empty(capacity, dtype=torch.int32, device="cuda")
        triton_allocate_blocks(block_bag, avail, output, capacity)
        assert avail.item() == 0
        assert output.numel() == capacity

    def test_multiple_allocate_release_cycles(self):
        total = 64
        block_bag, avail, ref_counts = self._make_allocator_state(total)
        initial_avail = avail.item()

        for _ in range(5):
            output = torch.empty(10, dtype=torch.int32, device="cuda")
            triton_allocate_blocks(block_bag, avail, output, 10)
            triton_release_blocks(block_bag, avail, output, 10)

        assert avail.item() == initial_avail

    def test_total_avail_gpu_scalar_consistency(self):
        total = 32
        block_bag, avail, ref_counts = self._make_allocator_state(total)

        output = torch.empty(7, dtype=torch.int32, device="cuda")
        triton_allocate_blocks(block_bag, avail, output, 7)
        assert avail.item() == total - 1 - 7

        triton_release_blocks(block_bag, avail, output[:3], 3)
        assert avail.item() == total - 1 - 7 + 3
