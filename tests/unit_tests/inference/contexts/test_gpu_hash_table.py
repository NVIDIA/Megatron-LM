# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.inference.contexts.gpu_hash_table import GPUHashTable, next_power_of_2


# =========================================================================
# Helper
# =========================================================================

DEVICE = "cuda"


def _keys(vals):
    """Create int64 key tensor on GPU."""
    return torch.tensor(vals, dtype=torch.int64, device=DEVICE)


def _vals(vals):
    """Create int32 value tensor on GPU."""
    return torch.tensor(vals, dtype=torch.int32, device=DEVICE)


# =========================================================================
# Tests
# =========================================================================


class TestNextPowerOf2:
    def test_basic(self):
        assert next_power_of_2(1) == 1
        assert next_power_of_2(2) == 2
        assert next_power_of_2(3) == 4
        assert next_power_of_2(5) == 8
        assert next_power_of_2(16) == 16
        assert next_power_of_2(17) == 32
        assert next_power_of_2(1000) == 1024


class TestGPUHashTableInsertLookup:
    """Insert/lookup correctness tests."""

    @pytest.mark.internal
    def test_single_insert_lookup(self):
        ht = GPUHashTable(max_entries=64, device=DEVICE)
        ht.insert_batch(_keys([42]), _vals([7]))
        result = ht.lookup_batch_alloc(_keys([42]))
        assert result[0].item() == 7

    @pytest.mark.internal
    def test_batch_insert_lookup(self):
        ht = GPUHashTable(max_entries=128, device=DEVICE)
        keys = _keys([100, 200, 300, 400, 500])
        vals = _vals([10, 20, 30, 40, 50])
        ht.insert_batch(keys, vals)

        results = ht.lookup_batch_alloc(keys)
        assert results.tolist() == [10, 20, 30, 40, 50]

    @pytest.mark.internal
    def test_missing_key_returns_neg1(self):
        ht = GPUHashTable(max_entries=64, device=DEVICE)
        ht.insert_batch(_keys([42]), _vals([7]))
        result = ht.lookup_batch_alloc(_keys([99]))
        assert result[0].item() == -1

    @pytest.mark.internal
    def test_empty_table_lookup(self):
        ht = GPUHashTable(max_entries=64, device=DEVICE)
        result = ht.lookup_batch_alloc(_keys([1, 2, 3]))
        assert result.tolist() == [-1, -1, -1]

    @pytest.mark.internal
    def test_empty_batch_operations(self):
        ht = GPUHashTable(max_entries=64, device=DEVICE)
        ht.insert_batch(_keys([]), _vals([]))
        result = ht.lookup_batch_alloc(_keys([]))
        assert result.numel() == 0

    @pytest.mark.internal
    def test_duplicate_key_updates_value(self):
        ht = GPUHashTable(max_entries=64, device=DEVICE)
        ht.insert_batch(_keys([42]), _vals([7]))
        ht.insert_batch(_keys([42]), _vals([99]))
        result = ht.lookup_batch_alloc(_keys([42]))
        assert result[0].item() == 99

    @pytest.mark.internal
    def test_sentinel_key_skipped(self):
        """Keys equal to -1 (empty sentinel) should be skipped."""
        ht = GPUHashTable(max_entries=64, device=DEVICE)
        ht.insert_batch(_keys([-1, 42]), _vals([1, 7]))
        result = ht.lookup_batch_alloc(_keys([-1, 42]))
        # -1 should not be inserted, so lookup returns -1
        assert result[0].item() == -1
        assert result[1].item() == 7


class TestGPUHashTableCollisions:
    """Collision handling tests — keys that hash to the same slot."""

    @pytest.mark.internal
    def test_collision_resolution(self):
        """Insert keys that may collide due to masking and verify all are retrievable."""
        ht = GPUHashTable(max_entries=64, device=DEVICE)
        # Keys that differ only in high bits will collide when masked
        cap_mask = ht.capacity_mask
        base_slot = 42 & cap_mask
        # Create keys that all map to the same initial slot
        colliding_keys = [base_slot + i * ht.capacity for i in range(5)]
        keys = _keys(colliding_keys)
        vals = _vals(list(range(5)))
        ht.insert_batch(keys, vals)

        results = ht.lookup_batch_alloc(keys)
        assert results.tolist() == list(range(5))

    @pytest.mark.internal
    def test_high_load_factor(self):
        """Fill table to ~40% load factor and verify all entries."""
        max_entries = 256
        ht = GPUHashTable(max_entries=max_entries, device=DEVICE)
        n = max_entries  # ~50% load factor due to 2x capacity
        keys = _keys(list(range(1, n + 1)))  # Avoid 0 and -1
        vals = _vals(list(range(n)))
        ht.insert_batch(keys, vals)

        results = ht.lookup_batch_alloc(keys)
        assert results.tolist() == list(range(n))


class TestGPUHashTableRebuild:
    """Rebuild (delete-via-rebuild) tests."""

    @pytest.mark.internal
    def test_rebuild_removes_deleted_keys(self):
        ht = GPUHashTable(max_entries=64, device=DEVICE)
        keys = _keys([100, 200, 300, 400])
        vals = _vals([1, 2, 3, 4])
        ht.insert_batch(keys, vals)

        # Rebuild with only a subset
        ht.rebuild(_keys([100, 300]), _vals([1, 3]))

        results = ht.lookup_batch_alloc(keys)
        assert results[0].item() == 1   # kept
        assert results[1].item() == -1  # removed
        assert results[2].item() == 3   # kept
        assert results[3].item() == -1  # removed

    @pytest.mark.internal
    def test_rebuild_empty(self):
        ht = GPUHashTable(max_entries=64, device=DEVICE)
        ht.insert_batch(_keys([1, 2, 3]), _vals([10, 20, 30]))
        ht.rebuild(_keys([]), _vals([]))
        results = ht.lookup_batch_alloc(_keys([1, 2, 3]))
        assert results.tolist() == [-1, -1, -1]

    @pytest.mark.internal
    def test_clear(self):
        ht = GPUHashTable(max_entries=64, device=DEVICE)
        ht.insert_batch(_keys([1, 2, 3]), _vals([10, 20, 30]))
        ht.clear()
        results = ht.lookup_batch_alloc(_keys([1, 2, 3]))
        assert results.tolist() == [-1, -1, -1]
        assert ht.size == 0


class TestGPUHashTableCapacity:
    """Capacity and sizing tests."""

    @pytest.mark.internal
    def test_capacity_is_power_of_2(self):
        for max_entries in [1, 10, 33, 100, 500]:
            ht = GPUHashTable(max_entries=max_entries, device=DEVICE)
            assert (ht.capacity & (ht.capacity - 1)) == 0  # power of 2
            assert ht.capacity >= 2 * max_entries

    @pytest.mark.internal
    def test_min_capacity(self):
        """Even with max_entries=1, should have reasonable capacity."""
        ht = GPUHashTable(max_entries=1, device=DEVICE)
        assert ht.capacity >= 32  # min 16 entries * 2


class TestGPUHashTablePrefixMatch:
    """Tests for prefix_match_batch (Triton kernel)."""

    @pytest.mark.internal
    def test_full_prefix_match(self):
        """All hashes found in order."""
        ht = GPUHashTable(max_entries=64, device=DEVICE)
        ht.insert_batch(_keys([10, 20, 30, 40]), _vals([0, 1, 2, 3]))

        hashes = _keys([10, 20, 30, 40])
        offsets = torch.tensor([0, 4], dtype=torch.int32, device=DEVICE)
        pending = torch.zeros(64, dtype=torch.bool, device=DEVICE)

        num_matched, has_pending, matched_ids = ht.prefix_match_batch(
            hashes, offsets, pending, max_blocks_per_req=4
        )
        assert num_matched.tolist() == [4]
        assert has_pending.tolist() == [0]
        assert matched_ids[0].tolist() == [0, 1, 2, 3]

    @pytest.mark.internal
    def test_partial_prefix_match(self):
        """Chain breaks at first miss."""
        ht = GPUHashTable(max_entries=64, device=DEVICE)
        ht.insert_batch(_keys([10, 20]), _vals([0, 1]))

        hashes = _keys([10, 20, 30, 40])
        offsets = torch.tensor([0, 4], dtype=torch.int32, device=DEVICE)
        pending = torch.zeros(64, dtype=torch.bool, device=DEVICE)

        num_matched, has_pending, matched_ids = ht.prefix_match_batch(
            hashes, offsets, pending, max_blocks_per_req=4
        )
        assert num_matched.tolist() == [2]
        assert matched_ids[0, :2].tolist() == [0, 1]
        assert matched_ids[0, 2:].tolist() == [-1, -1]

    @pytest.mark.internal
    def test_gap_breaks_chain(self):
        """Middle hash missing breaks the chain."""
        ht = GPUHashTable(max_entries=64, device=DEVICE)
        ht.insert_batch(_keys([10, 30, 40]), _vals([0, 2, 3]))  # hash 20 missing

        hashes = _keys([10, 20, 30, 40])
        offsets = torch.tensor([0, 4], dtype=torch.int32, device=DEVICE)
        pending = torch.zeros(64, dtype=torch.bool, device=DEVICE)

        num_matched, has_pending, matched_ids = ht.prefix_match_batch(
            hashes, offsets, pending, max_blocks_per_req=4
        )
        assert num_matched.tolist() == [1]
        assert matched_ids[0, 0].item() == 0

    @pytest.mark.internal
    def test_pending_detection(self):
        """Matched block with pending bitmap set."""
        ht = GPUHashTable(max_entries=64, device=DEVICE)
        ht.insert_batch(_keys([10, 20, 30]), _vals([0, 1, 2]))

        pending = torch.zeros(64, dtype=torch.bool, device=DEVICE)
        pending[1] = True  # block 1 is pending

        hashes = _keys([10, 20, 30])
        offsets = torch.tensor([0, 3], dtype=torch.int32, device=DEVICE)

        num_matched, has_pending, matched_ids = ht.prefix_match_batch(
            hashes, offsets, pending, max_blocks_per_req=3
        )
        assert num_matched.tolist() == [3]
        assert has_pending.tolist() == [1]
        assert matched_ids[0].tolist() == [0, 1, 2]

    @pytest.mark.internal
    def test_multiple_requests_batch(self):
        """Multiple requests in one kernel launch."""
        ht = GPUHashTable(max_entries=64, device=DEVICE)
        ht.insert_batch(_keys([10, 20, 30, 40, 50]), _vals([0, 1, 2, 3, 4]))

        # Req 0: [10, 20, 30] -> 3 match
        # Req 1: [10, 20, 99] -> 2 match (99 missing)
        # Req 2: [50]         -> 1 match
        hashes = _keys([10, 20, 30, 10, 20, 99, 50])
        offsets = torch.tensor([0, 3, 6, 7], dtype=torch.int32, device=DEVICE)
        pending = torch.zeros(64, dtype=torch.bool, device=DEVICE)

        num_matched, has_pending, matched_ids = ht.prefix_match_batch(
            hashes, offsets, pending, max_blocks_per_req=3
        )
        assert num_matched.tolist() == [3, 2, 1]
        assert matched_ids[0, :3].tolist() == [0, 1, 2]
        assert matched_ids[1, :2].tolist() == [0, 1]
        assert matched_ids[2, 0].item() == 4

    @pytest.mark.internal
    def test_no_match(self):
        """No hashes in table."""
        ht = GPUHashTable(max_entries=64, device=DEVICE)
        # Empty table — no inserts.

        hashes = _keys([10, 20])
        offsets = torch.tensor([0, 2], dtype=torch.int32, device=DEVICE)
        pending = torch.zeros(64, dtype=torch.bool, device=DEVICE)

        num_matched, has_pending, matched_ids = ht.prefix_match_batch(
            hashes, offsets, pending, max_blocks_per_req=2
        )
        assert num_matched.tolist() == [0]
        assert has_pending.tolist() == [0]
        assert (matched_ids == -1).all()

    @pytest.mark.internal
    def test_empty_request(self):
        """Zero-length request."""
        ht = GPUHashTable(max_entries=64, device=DEVICE)
        ht.insert_batch(_keys([10, 20]), _vals([0, 1]))

        hashes = _keys([])
        offsets = torch.tensor([0, 0], dtype=torch.int32, device=DEVICE)
        pending = torch.zeros(64, dtype=torch.bool, device=DEVICE)

        num_matched, has_pending, matched_ids = ht.prefix_match_batch(
            hashes, offsets, pending, max_blocks_per_req=4
        )
        assert num_matched.tolist() == [0]
        assert has_pending.tolist() == [0]


class TestGPUHashTableLargeScale:
    """Larger-scale correctness tests."""

    @pytest.mark.internal
    def test_1000_entries(self):
        n = 1000
        ht = GPUHashTable(max_entries=n, device=DEVICE)
        keys = torch.arange(1, n + 1, dtype=torch.int64, device=DEVICE)
        vals = torch.arange(0, n, dtype=torch.int32, device=DEVICE)
        ht.insert_batch(keys, vals)

        results = ht.lookup_batch_alloc(keys)
        assert torch.equal(results, vals)

        # Check some missing keys
        missing = torch.arange(n + 1, n + 101, dtype=torch.int64, device=DEVICE)
        missing_results = ht.lookup_batch_alloc(missing)
        assert (missing_results == -1).all()

    @pytest.mark.internal
    def test_rebuild_preserves_remaining(self):
        """Insert 500, rebuild with 250, verify correctness."""
        n = 500
        ht = GPUHashTable(max_entries=n, device=DEVICE)
        keys = torch.arange(1, n + 1, dtype=torch.int64, device=DEVICE)
        vals = torch.arange(0, n, dtype=torch.int32, device=DEVICE)
        ht.insert_batch(keys, vals)

        # Keep only even-indexed entries
        keep_mask = torch.arange(n) % 2 == 0
        keep_keys = keys[keep_mask]
        keep_vals = vals[keep_mask]
        ht.rebuild(keep_keys, keep_vals)

        # Verify kept entries
        results = ht.lookup_batch_alloc(keep_keys)
        assert torch.equal(results, keep_vals)

        # Verify removed entries
        removed_keys = keys[~keep_mask]
        removed_results = ht.lookup_batch_alloc(removed_keys)
        assert (removed_results == -1).all()
