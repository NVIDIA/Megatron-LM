# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.inference.contexts.kernels.prefix_cache_gpu import GPUHashTable


class TestGPUHashTable:

    def test_insert_and_lookup_basic(self):
        ht = GPUHashTable(total_count=64, device=torch.device("cuda"))
        hashes = [101, 202, 303, 404, 505]
        block_ids = [0, 1, 2, 3, 4]
        ht.insert(hashes, block_ids)

        query = torch.tensor(hashes, dtype=torch.int64, device="cuda")
        result = ht.lookup(query)
        expected = torch.tensor(block_ids, dtype=torch.int32, device="cuda")
        assert torch.equal(result, expected)

    def test_lookup_miss(self):
        ht = GPUHashTable(total_count=64, device=torch.device("cuda"))
        ht.insert([101, 202], [0, 1])

        query = torch.tensor([999, 888, 777], dtype=torch.int64, device="cuda")
        result = ht.lookup(query)
        assert (result == -1).all()

    def test_insert_overwrite(self):
        ht = GPUHashTable(total_count=64, device=torch.device("cuda"))
        ht.insert([101], [0])
        ht.insert([101], [5])  # overwrite

        assert ht.get(101) == 5

    def test_delete_basic(self):
        ht = GPUHashTable(total_count=64, device=torch.device("cuda"))
        ht.insert([101, 202, 303], [0, 1, 2])

        # Simulate block_hashes tensor
        block_hashes = torch.full((64,), -1, dtype=torch.int64, device="cuda")
        block_hashes[0] = 101
        block_hashes[1] = 202

        # Delete blocks 0 and 1
        delete_ids = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        ht.delete_by_block_ids(delete_ids, block_hashes)

        assert 101 not in ht
        assert 202 not in ht
        assert 303 in ht
        assert ht.get(303) == 2

    def test_tombstone_probe_chain(self):
        """Delete middle of a probe chain, verify later entries still findable."""
        ht = GPUHashTable(total_count=64, device=torch.device("cuda"))
        table_size = ht.table_size
        mask = ht.table_size_mask

        # Create two hashes that map to the same slot (collision)
        base_slot = 42
        h1 = base_slot  # maps to slot 42
        # Find h2 that also maps to slot 42
        h2 = base_slot + table_size  # also maps to slot 42 (mod table_size)

        # Insert sequentially to avoid race on same slot with parallel grid
        ht.insert([h1], [10])
        ht.insert([h2], [20])

        # Verify both are findable
        assert ht.get(h1) == 10
        assert ht.get(h2) == 20

        # Delete h1 (creates tombstone at slot 42)
        block_hashes = torch.full((64,), -1, dtype=torch.int64, device="cuda")
        block_hashes[10] = h1
        ht.delete_by_block_ids(torch.tensor([10], dtype=torch.int32, device="cuda"), block_hashes)

        # h1 should be gone, h2 should still be findable (probe past tombstone)
        assert h1 not in ht
        assert ht.get(h2) == 20

    def test_clear(self):
        ht = GPUHashTable(total_count=64, device=torch.device("cuda"))
        ht.insert([101, 202, 303], [0, 1, 2])
        assert len(ht) == 3

        ht.clear()
        assert len(ht) == 0
        assert 101 not in ht

    def test_contains_and_get(self):
        ht = GPUHashTable(total_count=64, device=torch.device("cuda"))
        ht.insert([101, 202], [0, 1])

        assert 101 in ht
        assert 999 not in ht
        assert ht.get(101) == 0
        assert ht.get(999) is None
        assert ht.get(999, -1) == -1

    def test_len(self):
        ht = GPUHashTable(total_count=64, device=torch.device("cuda"))
        assert len(ht) == 0

        ht.insert([101, 202, 303], [0, 1, 2])
        assert len(ht) == 3

    def test_keys(self):
        ht = GPUHashTable(total_count=64, device=torch.device("cuda"))
        hashes = [101, 202, 303]
        ht.insert(hashes, [0, 1, 2])

        assert ht.keys() == set(hashes)

    def test_high_load(self):
        """Insert total_count entries (50% load), verify all found."""
        total_count = 256
        ht = GPUHashTable(total_count=total_count, device=torch.device("cuda"))
        hashes = list(range(1, total_count + 1))  # hashes must be > 0
        block_ids = list(range(total_count))
        ht.insert(hashes, block_ids)

        query = torch.tensor(hashes, dtype=torch.int64, device="cuda")
        result = ht.lookup(query)
        expected = torch.tensor(block_ids, dtype=torch.int32, device="cuda")
        assert torch.equal(result, expected)
        assert len(ht) == total_count

    def test_empty_operations(self):
        ht = GPUHashTable(total_count=64, device=torch.device("cuda"))

        # Empty insert
        ht.insert([], [])

        # Empty lookup
        result = ht.lookup(torch.empty(0, dtype=torch.int64, device="cuda"))
        assert result.numel() == 0

        # Empty delete
        block_hashes = torch.full((64,), -1, dtype=torch.int64, device="cuda")
        ht.delete_by_block_ids(torch.empty(0, dtype=torch.int32, device="cuda"), block_hashes)

    def test_insert_after_delete_reuses_tombstone(self):
        ht = GPUHashTable(total_count=64, device=torch.device("cuda"))
        ht.insert([101], [0])

        block_hashes = torch.full((64,), -1, dtype=torch.int64, device="cuda")
        block_hashes[0] = 101
        ht.delete_by_block_ids(torch.tensor([0], dtype=torch.int32, device="cuda"), block_hashes)

        assert 101 not in ht

        # Re-insert at same hash
        ht.insert([101], [5])
        assert ht.get(101) == 5
