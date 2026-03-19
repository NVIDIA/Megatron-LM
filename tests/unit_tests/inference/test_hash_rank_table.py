# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import numpy as np
import pytest

from megatron.core.inference.config import PrefixCachingCoordinatorPolicy
from megatron.core.inference.hash_rank_table import HashRankTable


class TestRecordAndQuery:
    """Basic write/read operations."""

    def test_record_and_has(self):
        t = HashRankTable(n_ranks=3)
        t.record(0, [10, 20])
        assert t.has(10, 0)
        assert t.has(20, 0)
        assert not t.has(10, 1)
        assert not t.has(30, 0)

    def test_record_increments_assignment_counter(self):
        t = HashRankTable(n_ranks=2)
        assert t.assignment_counter == 0
        t.record(0, [1, 2])
        assert t.assignment_counter == 1
        t.record(1, [3])
        assert t.assignment_counter == 2

    def test_set_explicit_timestamp(self):
        t = HashRankTable(n_ranks=2)
        t.set(42, 1, 99)
        assert t.get_timestamp(42, 1) == 99
        assert t.get_timestamp(42, 0) == 0

    def test_get_row_returns_none_for_unseen_hash(self):
        t = HashRankTable(n_ranks=2)
        assert t.get_row(999) is None

    def test_get_row_returns_timestamps(self):
        t = HashRankTable(n_ranks=3)
        t.set(10, 0, 5)
        t.set(10, 2, 7)
        row = t.get_row(10)
        assert row is not None
        np.testing.assert_array_equal(row, [5, 0, 7])

    def test_get_timestamp_unseen_hash(self):
        t = HashRankTable(n_ranks=2)
        assert t.get_timestamp(999, 0) == 0

    def test_multiple_ranks_same_hash(self):
        t = HashRankTable(n_ranks=3)
        t.record(0, [10])
        t.record(2, [10])
        assert t.has(10, 0)
        assert not t.has(10, 1)
        assert t.has(10, 2)


class TestMatchVectorFirstPrefixBlock:
    """match_vector with FIRST_PREFIX_BLOCK policy (default)."""

    def test_empty_hashes(self):
        t = HashRankTable(n_ranks=2)
        result = t.match_vector([])
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_no_match(self):
        t = HashRankTable(n_ranks=2)
        t.record(0, [10])
        result = t.match_vector([99])
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_only_checks_first_hash(self):
        """FIRST_PREFIX_BLOCK only looks at hashes[0], ignoring the rest."""
        t = HashRankTable(n_ranks=2)
        t.record(1, [20])  # rank 1 has hash 20 but not hash 10
        # hashes[0]=10 is not cached, even though hashes[1]=20 is.
        result = t.match_vector([10, 20], policy=PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_binary_match(self):
        t = HashRankTable(n_ranks=3)
        t.record(0, [10])
        t.record(2, [10])
        result = t.match_vector([10, 20])
        np.testing.assert_array_equal(result, [1.0, 0.0, 1.0])

    def test_default_policy_is_first_prefix_block(self):
        t = HashRankTable(n_ranks=2)
        t.record(0, [10])
        default = t.match_vector([10])
        explicit = t.match_vector([10], policy=PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK)
        np.testing.assert_array_equal(default, explicit)


class TestMatchVectorLongestPrefix:
    """match_vector with LONGEST_PREFIX policy."""

    POLICY = PrefixCachingCoordinatorPolicy.LONGEST_PREFIX

    def test_empty_hashes(self):
        t = HashRankTable(n_ranks=2)
        result = t.match_vector([], policy=self.POLICY)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_no_match(self):
        t = HashRankTable(n_ranks=2)
        t.record(0, [10])
        result = t.match_vector([99, 88], policy=self.POLICY)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_full_match_scores_one(self):
        """A rank matching the deepest hash in a length-1 sequence scores 1.0."""
        t = HashRankTable(n_ranks=2)
        t.record(0, [10])
        result = t.match_vector([10], policy=self.POLICY)
        np.testing.assert_array_equal(result, [1.0, 0.0])

    def test_normalized_depth_score(self):
        """Score is (deepest_index + 1) / len(hashes)."""
        t = HashRankTable(n_ranks=2)
        # hashes = [A, B, C, D] — rank 0 has C (index 2).
        t.record(0, [30])
        result = t.match_vector([10, 20, 30, 40], policy=self.POLICY)
        expected_score = 3.0 / 4.0  # (2 + 1) / 4
        np.testing.assert_allclose(result, [expected_score, 0.0])

    def test_deepest_match_wins(self):
        """Reverse scan finds the deepest hash, not the shallowest."""
        t = HashRankTable(n_ranks=2)
        # Rank 0 has hash at index 1 and index 3.
        t.record(0, [20, 40])
        result = t.match_vector([10, 20, 30, 40], policy=self.POLICY)
        # Deepest match is index 3 → score = 4/4 = 1.0
        np.testing.assert_allclose(result, [1.0, 0.0])

    def test_multiple_ranks_different_depths(self):
        """When two ranks share the deepest hash, both get the same score."""
        t = HashRankTable(n_ranks=3)
        t.record(0, [30])  # index 2
        t.record(1, [30])  # index 2 (same deepest hash)
        # rank 2 has nothing
        result = t.match_vector([10, 20, 30, 40], policy=self.POLICY)
        expected = 3.0 / 4.0
        np.testing.assert_allclose(result, [expected, expected, 0.0])

    def test_only_deepest_hash_matters(self):
        """A rank with only a shallow hash gets 0 if a deeper hash exists for another rank."""
        t = HashRankTable(n_ranks=2)
        t.record(0, [10])  # index 0 (shallow)
        t.record(1, [30])  # index 2 (deeper)
        # The reverse scan stops at index 2 (hash 30) — rank 0 doesn't have it.
        result = t.match_vector([10, 20, 30], policy=self.POLICY)
        np.testing.assert_allclose(result, [0.0, 1.0])


class TestCapacityGrowth:
    """Table auto-grows when capacity is exceeded."""

    def test_grows_beyond_initial_capacity(self):
        t = HashRankTable(n_ranks=2, initial_capacity=4)
        assert t._timestamps.shape[0] == 4
        # Insert 5 unique hashes to exceed initial capacity.
        for i in range(5):
            t.record(0, [i])
        assert t._timestamps.shape[0] >= 5
        # All hashes are still accessible.
        for i in range(5):
            assert t.has(i, 0)

    def test_capacity_doubles(self):
        t = HashRankTable(n_ranks=1, initial_capacity=4)
        for i in range(5):
            t.record(0, [i])
        # Should have doubled from 4 to 8.
        assert t._timestamps.shape[0] == 8


class TestCompact:
    """compact() removes dead rows and shrinks the backing array."""

    def test_removes_dead_rows(self):
        t = HashRankTable(n_ranks=2, initial_capacity=4)
        t.set(10, 0, 1)
        t.set(20, 0, 2)
        t.set(30, 0, 3)
        # Kill hash 20 by zeroing its timestamps.
        row = t._hash_to_row[20]
        t._timestamps[row, :] = 0

        t.compact()

        assert t.has(10, 0)
        assert not t.has(20, 0)
        assert t.has(30, 0)
        assert t._next_row == 2

    def test_compact_preserves_data(self):
        t = HashRankTable(n_ranks=2, initial_capacity=4)
        t.set(10, 0, 5)
        t.set(10, 1, 7)
        t.set(20, 0, 3)
        # Kill hash 20.
        row = t._hash_to_row[20]
        t._timestamps[row, :] = 0

        t.compact()

        assert t.get_timestamp(10, 0) == 5
        assert t.get_timestamp(10, 1) == 7

    def test_compact_noop_when_no_dead_rows(self):
        t = HashRankTable(n_ranks=2, initial_capacity=4)
        t.set(10, 0, 1)
        t.set(20, 1, 2)
        old_next_row = t._next_row

        t.compact()

        assert t._next_row == old_next_row
        assert t.has(10, 0)
        assert t.has(20, 1)

    def test_shrinks_when_half_capacity(self):
        """After compacting, the array shrinks if usage is well below capacity."""
        t = HashRankTable(n_ranks=2, initial_capacity=4)
        # Insert enough hashes to grow the table.
        for i in range(300):
            t.set(i, 0, i + 1)
        big_capacity = t._timestamps.shape[0]
        assert big_capacity >= 300

        # Kill most rows, leaving only a few live.
        for i in range(295):
            row = t._hash_to_row[i]
            t._timestamps[row, :] = 0

        t.compact()

        assert t._next_row == 5
        assert t._timestamps.shape[0] < big_capacity
        # Remaining hashes are intact.
        for i in range(295, 300):
            assert t.has(i, 0)

    def test_compact_all_dead(self):
        """Compacting when every row is dead leaves an empty table."""
        t = HashRankTable(n_ranks=2, initial_capacity=4)
        t.set(10, 0, 1)
        t.set(20, 1, 2)
        # Remove everything.
        t._timestamps[:] = 0

        t.compact()

        assert t._next_row == 0
        assert len(t._hash_to_row) == 0
        assert not t.has(10, 0)

    def test_new_inserts_work_after_compact(self):
        t = HashRankTable(n_ranks=2, initial_capacity=4)
        t.set(10, 0, 1)
        t.set(20, 0, 2)
        # Remove hash 10.
        t._timestamps[t._hash_to_row[10], :] = 0
        t.compact()

        # Insert new hashes after compacting.
        t.record(1, [30, 40])
        assert t.has(30, 1)
        assert t.has(40, 1)
        assert t.has(20, 0)


class TestMaxTimestamps:
    """max_timestamps returns per-rank element-wise max across hashes."""

    def test_single_hash(self):
        t = HashRankTable(n_ranks=2)
        t.set(10, 0, 5)
        t.set(10, 1, 3)
        result = t.max_timestamps([10])
        np.testing.assert_array_equal(result, [5, 3])

    def test_max_across_multiple_hashes(self):
        t = HashRankTable(n_ranks=2)
        t.set(10, 0, 5)
        t.set(10, 1, 3)
        t.set(20, 0, 2)
        t.set(20, 1, 9)
        result = t.max_timestamps([10, 20])
        np.testing.assert_array_equal(result, [5, 9])

    def test_unseen_hashes_ignored(self):
        t = HashRankTable(n_ranks=2)
        t.set(10, 0, 5)
        result = t.max_timestamps([10, 999])
        np.testing.assert_array_equal(result, [5, 0])

    def test_all_unseen_returns_zeros(self):
        t = HashRankTable(n_ranks=3)
        result = t.max_timestamps([99, 88])
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_empty_hashes(self):
        t = HashRankTable(n_ranks=2)
        result = t.max_timestamps([])
        np.testing.assert_array_equal(result, [0, 0])


class TestAddRank:
    """add_rank() appends a new rank column."""

    def test_add_rank_returns_new_index(self):
        t = HashRankTable(n_ranks=2)
        new_idx = t.add_rank()
        assert new_idx == 2
        assert t.n_ranks == 3

    def test_add_rank_preserves_existing_data(self):
        t = HashRankTable(n_ranks=2)
        t.set(10, 0, 5)
        t.set(10, 1, 3)
        t.add_rank()
        assert t.get_timestamp(10, 0) == 5
        assert t.get_timestamp(10, 1) == 3
        assert t.get_timestamp(10, 2) == 0

    def test_add_rank_new_rank_is_usable(self):
        t = HashRankTable(n_ranks=2)
        new_idx = t.add_rank()
        t.record(new_idx, [42])
        assert t.has(42, new_idx)
        assert not t.has(42, 0)

    def test_add_multiple_ranks(self):
        t = HashRankTable(n_ranks=1)
        idx1 = t.add_rank()
        idx2 = t.add_rank()
        assert idx1 == 1
        assert idx2 == 2
        assert t.n_ranks == 3
        t.set(10, idx2, 7)
        assert t.get_timestamp(10, idx2) == 7

    def test_match_vector_after_add_rank(self):
        t = HashRankTable(n_ranks=2)
        t.record(0, [10])
        new_idx = t.add_rank()
        t.record(new_idx, [10])
        result = t.match_vector([10])
        np.testing.assert_array_equal(result, [1.0, 0.0, 1.0])
