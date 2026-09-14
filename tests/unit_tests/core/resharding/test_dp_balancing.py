# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import pytest

from megatron.core.resharding.utils import ParameterMetadata, select_src_metadata_balanced


class TestDPBalancing:
    """Test suite for DP load balancing."""

    def _create_metadata(self, rank, tp_group, dp_group, ep_group=None):
        """Helper to create ParameterMetadata for testing."""
        return ParameterMetadata(
            name="test.weight",
            shape=(128, 256),
            dtype="float32",
            element_size=4,  # 4 bytes for float32
            owner_rank=rank,
            tensor_parallel_group_ranks=tp_group,
            data_parallel_group_ranks=dp_group,
            expert_parallel_group_ranks=ep_group,
            is_tp=True,
            partition_dim=0,
        )

    def test_dp_balancing_basic(self):
        """Test basic DP balancing with 2 DP groups."""
        # Setup: TP=2, DP=2, World=4
        # TP groups: [[0,1], [2,3]]
        # DP groups: [[0,2], [1,3]]  (TP-local-0 and TP-local-1)

        # Source metadata from all ranks
        src_meta_list = [
            self._create_metadata(rank=0, tp_group=[0, 1], dp_group=[0, 2]),
            self._create_metadata(rank=1, tp_group=[0, 1], dp_group=[1, 3]),
            self._create_metadata(rank=2, tp_group=[2, 3], dp_group=[0, 2]),
            self._create_metadata(rank=3, tp_group=[2, 3], dp_group=[1, 3]),
        ]

        # Destination metadata (TP=1, DP=4)
        dst_meta = self._create_metadata(rank=0, tp_group=[0], dp_group=[0, 1, 2, 3])

        # Test each destination rank's selection
        selections = {}
        for dst_rank in range(4):
            selected = select_src_metadata_balanced(src_meta_list, dst_meta, dst_rank)
            selections[dst_rank] = (selected.owner_rank, tuple(selected.data_parallel_group_ranks))

        # Verify round-robin across DP groups
        # dst_rank 0: 0 % 2 = 0 -> should select DP group [0,2]
        # dst_rank 1: 1 % 2 = 1 -> should select DP group [1,3]
        # dst_rank 2: 2 % 2 = 0 -> should select DP group [0,2]
        # dst_rank 3: 3 % 2 = 1 -> should select DP group [1,3]
        assert selections[0][1] in [(0, 2), (1, 3)]  # DP group 0 or 1
        assert selections[1][1] in [(0, 2), (1, 3)]
        assert selections[2][1] == selections[0][1]  # Same as rank 0
        assert selections[3][1] == selections[1][1]  # Same as rank 1

        # Verify different DP groups selected
        assert selections[0][1] != selections[1][1]

    def test_dp_balancing_non_collocated(self):
        """Test DP balancing in non-collocated mode (dst ranks not in source ranks)."""
        # Setup: TP=2, DP=2, World=4 (non-collocated, same config on both sides)

        src_meta_list = [
            self._create_metadata(rank=0, tp_group=[0, 1], dp_group=[0, 2]),
            self._create_metadata(rank=1, tp_group=[0, 1], dp_group=[1, 3]),
            self._create_metadata(rank=2, tp_group=[2, 3], dp_group=[0, 2]),
            self._create_metadata(rank=3, tp_group=[2, 3], dp_group=[1, 3]),
        ]

        # Destination with TP=2 (same as source), dst rank not in source ranks
        dst_meta = self._create_metadata(rank=4, tp_group=[4, 5], dp_group=[4, 5])

        # Should select via DP balancing (no local copy available)
        selected = select_src_metadata_balanced(src_meta_list, dst_meta, dst_rank=4)

        # dst_rank=4, 4 % 2 = 0 -> selects from first sorted DP group
        assert selected.owner_rank in [0, 1, 2, 3]

    def test_dp_balancing_distribution(self):
        """Test that many destination ranks are evenly distributed across source DP groups."""
        # Setup: TP=2, DP=4, World=8
        # TP groups: [[0,1], [2,3], [4,5], [6,7]]
        # DP groups: [[0,2,4,6], [1,3,5,7]]

        src_meta_list = [
            self._create_metadata(rank=0, tp_group=[0, 1], dp_group=[0, 2, 4, 6]),
            self._create_metadata(rank=1, tp_group=[0, 1], dp_group=[1, 3, 5, 7]),
            self._create_metadata(rank=2, tp_group=[2, 3], dp_group=[0, 2, 4, 6]),
            self._create_metadata(rank=3, tp_group=[2, 3], dp_group=[1, 3, 5, 7]),
            self._create_metadata(rank=4, tp_group=[4, 5], dp_group=[0, 2, 4, 6]),
            self._create_metadata(rank=5, tp_group=[4, 5], dp_group=[1, 3, 5, 7]),
            self._create_metadata(rank=6, tp_group=[6, 7], dp_group=[0, 2, 4, 6]),
            self._create_metadata(rank=7, tp_group=[6, 7], dp_group=[1, 3, 5, 7]),
        ]

        dst_meta = self._create_metadata(rank=0, tp_group=[0], dp_group=list(range(8)))

        # Count selections per DP group
        dp_group_counts = {}
        for dst_rank in range(16):  # Test with more dst ranks than src
            selected = select_src_metadata_balanced(src_meta_list, dst_meta, dst_rank)
            dp_group = tuple(selected.data_parallel_group_ranks)
            dp_group_counts[dp_group] = dp_group_counts.get(dp_group, 0) + 1

        # Should have exactly 2 DP groups
        assert len(dp_group_counts) == 2

        # Each should be selected 8 times (16 ranks / 2 groups = 8)
        assert all(count == 8 for count in dp_group_counts.values())

    def test_dp_balancing_with_ep(self):
        """Test DP balancing with expert parallelism."""
        # Setup: TP=2, EP=2, DP=2, World=8
        # When EP sizes match, should filter by EP local rank
        #
        # EP local rank is computed from ep_group.index(owner_rank):
        #   rank=0 in ep_group=[0, 2] -> EP local 0
        #   rank=2 in ep_group=[0, 2] -> EP local 1
        #   rank=4 in ep_group=[4, 6] -> EP local 0
        #   rank=6 in ep_group=[4, 6] -> EP local 1

        src_meta_list = [
            # EP local 0 (rank 0 is at index 0 in ep_group [0, 2])
            self._create_metadata(rank=0, tp_group=[0, 1], dp_group=[0, 4], ep_group=[0, 2]),
            # EP local 1 (rank 2 is at index 1 in ep_group [0, 2])
            self._create_metadata(rank=2, tp_group=[2, 3], dp_group=[2, 6], ep_group=[0, 2]),
            # EP local 0 (rank 4 is at index 0 in ep_group [4, 6])
            self._create_metadata(rank=4, tp_group=[4, 5], dp_group=[0, 4], ep_group=[4, 6]),
            # EP local 1 (rank 6 is at index 1 in ep_group [4, 6])
            self._create_metadata(rank=6, tp_group=[6, 7], dp_group=[2, 6], ep_group=[4, 6]),
        ]

        # Destination with same EP size=2, EP local rank = 0
        # (rank 8 is at index 0 in ep_group [8, 9])
        dst_meta = self._create_metadata(rank=8, tp_group=[8, 9], dp_group=[8, 9], ep_group=[8, 9])

        # When EP sizes match (2->2), should filter by EP local rank
        selected = select_src_metadata_balanced(src_meta_list, dst_meta, dst_rank=8)

        # Should select from EP-local-0 ranks only (0 or 4)
        ep_local = selected.expert_parallel_group_ranks.index(selected.owner_rank)
        assert ep_local == 0
        assert selected.owner_rank in [0, 4]

    def test_dp_balancing_single_dp_group(self):
        """Test fast path when only one DP group exists."""
        # Setup: TP=2, DP=1, World=2 (single DP group)

        src_meta_list = [
            self._create_metadata(rank=0, tp_group=[0, 1], dp_group=[0, 1]),
            self._create_metadata(rank=1, tp_group=[0, 1], dp_group=[0, 1]),
        ]

        dst_meta = self._create_metadata(rank=0, tp_group=[0], dp_group=[0])

        # Should hit fast path and return first metadata
        selected = select_src_metadata_balanced(src_meta_list, dst_meta, dst_rank=0)

        # Fast path returns first entry (after any TP/EP filtering)
        assert selected == src_meta_list[0]

    def test_tp_size_mismatch_no_filter(self):
        """Test DP balancing when TP sizes differ (resharding mode)."""
        # Setup: TP=4 -> TP=2 (resharding)

        src_meta_list = [
            self._create_metadata(rank=0, tp_group=[0, 1, 2, 3], dp_group=[0, 4]),
            self._create_metadata(rank=1, tp_group=[0, 1, 2, 3], dp_group=[1, 5]),
            self._create_metadata(rank=2, tp_group=[0, 1, 2, 3], dp_group=[2, 6]),
            self._create_metadata(rank=3, tp_group=[0, 1, 2, 3], dp_group=[3, 7]),
            self._create_metadata(rank=4, tp_group=[4, 5, 6, 7], dp_group=[0, 4]),
            self._create_metadata(rank=5, tp_group=[4, 5, 6, 7], dp_group=[1, 5]),
            self._create_metadata(rank=6, tp_group=[4, 5, 6, 7], dp_group=[2, 6]),
            self._create_metadata(rank=7, tp_group=[4, 5, 6, 7], dp_group=[3, 7]),
        ]

        # Destination with TP=2 (different from source TP=4)
        dst_meta = self._create_metadata(rank=8, tp_group=[8, 9], dp_group=[8, 9])

        # Should only do DP balancing (no TP filtering)
        selected = select_src_metadata_balanced(src_meta_list, dst_meta, dst_rank=8)

        # Since dst_rank % 4 = 0, should select DP group 0
        # Could be any TP local rank (not filtered)
        assert selected.owner_rank in [0, 4]  # Both have DP group [0,4]

    def test_ep_size_mismatch_no_filter(self):
        """Test that EP filtering is skipped when EP sizes differ."""
        # Setup: EP=4 -> EP=8 (expert parallel resharding)

        src_meta_list = [
            self._create_metadata(rank=0, tp_group=[0], dp_group=[0, 4], ep_group=[0, 1, 2, 3]),
            self._create_metadata(rank=1, tp_group=[1], dp_group=[1, 5], ep_group=[0, 1, 2, 3]),
            self._create_metadata(rank=2, tp_group=[2], dp_group=[2, 6], ep_group=[0, 1, 2, 3]),
            self._create_metadata(rank=3, tp_group=[3], dp_group=[3, 7], ep_group=[0, 1, 2, 3]),
        ]

        # Destination with EP=8 (different from source EP=4)
        # EP sizes differ (4 vs 8), so EP filtering should be skipped
        dst_meta = self._create_metadata(
            rank=8, tp_group=[8], dp_group=[8, 9], ep_group=list(range(8, 16))
        )

        # Should NOT filter by EP local rank (sizes differ)
        selected = select_src_metadata_balanced(src_meta_list, dst_meta, dst_rank=8)

        # Should work without error (no EP filtering when sizes differ)
        assert selected.owner_rank in [0, 1, 2, 3]

    def test_load_distribution_across_parameters(self):
        """Test that different dst ranks select different DP groups for load balancing."""
        # Setup: TP=1, DP=4, World=4

        src_meta_list = [
            self._create_metadata(rank=0, tp_group=[0], dp_group=[0, 1, 2, 3]),
            self._create_metadata(rank=1, tp_group=[1], dp_group=[0, 1, 2, 3]),
            self._create_metadata(rank=2, tp_group=[2], dp_group=[0, 1, 2, 3]),
            self._create_metadata(rank=3, tp_group=[3], dp_group=[0, 1, 2, 3]),
        ]

        dst_meta = self._create_metadata(rank=0, tp_group=[0], dp_group=[0, 1, 2, 3, 4, 5, 6, 7])

        # Simulate 8 destination ranks selecting sources
        # Since there's only 1 DP group with 4 members, all should select the same group
        # But round-robin based on dst_rank should still distribute across src ranks
        selections = []
        for dst_rank in range(8):
            selected = select_src_metadata_balanced(src_meta_list, dst_meta, dst_rank)
            selections.append(selected.owner_rank)

        # All should select the same DP group (only one exists)
        # But within that group, should cycle through available ranks
        # Since there's only 1 DP group, they all select from it
        assert all(rank in [0, 1, 2, 3] for rank in selections)

    def test_within_dp_group_distribution(self):
        """Test that dst ranks distribute across source ranks within a DP group."""
        # This tests the optimization: when multiple dst ranks map to the same DP group,
        # they should use different source ranks within that group for load balancing.

        # Setup: TP=2, World=8 -> TP=1, World=8
        # Source TP groups: [[0,1], [2,3], [4,5], [6,7]]
        # Source DP groups: [[0,2,4,6], [1,3,5,7]]  (2 DP replicas)

        src_meta_list = [
            self._create_metadata(rank=0, tp_group=[0, 1], dp_group=[0, 2, 4, 6]),
            self._create_metadata(rank=1, tp_group=[0, 1], dp_group=[1, 3, 5, 7]),
            self._create_metadata(rank=2, tp_group=[2, 3], dp_group=[0, 2, 4, 6]),
            self._create_metadata(rank=3, tp_group=[2, 3], dp_group=[1, 3, 5, 7]),
            self._create_metadata(rank=4, tp_group=[4, 5], dp_group=[0, 2, 4, 6]),
            self._create_metadata(rank=5, tp_group=[4, 5], dp_group=[1, 3, 5, 7]),
            self._create_metadata(rank=6, tp_group=[6, 7], dp_group=[0, 2, 4, 6]),
            self._create_metadata(rank=7, tp_group=[6, 7], dp_group=[1, 3, 5, 7]),
        ]

        dst_meta = self._create_metadata(rank=0, tp_group=[0], dp_group=list(range(8)))

        # Test 8 destination ranks
        selections = {}
        for dst_rank in range(8):
            selected = select_src_metadata_balanced(src_meta_list, dst_meta, dst_rank)
            selections[dst_rank] = (
                selected.owner_rank,
                tuple(selected.tensor_parallel_group_ranks),
            )

        # Verify distribution:
        # dst_rank 0: DP group 0 ([0,2,4,6]), within-group idx 0 -> rank 0, TP [0,1]
        # dst_rank 1: DP group 1 ([1,3,5,7]), within-group idx 0 -> rank 1, TP [0,1]
        # dst_rank 2: DP group 0 ([0,2,4,6]), within-group idx 1 -> rank 2, TP [2,3]
        # dst_rank 3: DP group 1 ([1,3,5,7]), within-group idx 1 -> rank 3, TP [2,3]
        # dst_rank 4: DP group 0 ([0,2,4,6]), within-group idx 2 -> rank 4, TP [4,5]
        # dst_rank 5: DP group 1 ([1,3,5,7]), within-group idx 2 -> rank 5, TP [4,5]
        # dst_rank 6: DP group 0 ([0,2,4,6]), within-group idx 3 -> rank 6, TP [6,7]
        # dst_rank 7: DP group 1 ([1,3,5,7]), within-group idx 3 -> rank 7, TP [6,7]

        assert selections[0] == (0, (0, 1))
        assert selections[1] == (1, (0, 1))
        assert selections[2] == (2, (2, 3))
        assert selections[3] == (3, (2, 3))
        assert selections[4] == (4, (4, 5))
        assert selections[5] == (5, (4, 5))
        assert selections[6] == (6, (6, 7))
        assert selections[7] == (7, (6, 7))

        # Verify ALL source ranks are used (good load distribution!)
        source_ranks_used = {sel[0] for sel in selections.values()}
        assert source_ranks_used == {0, 1, 2, 3, 4, 5, 6, 7}, "All source ranks should be used"

        # Verify each TP group used by 2 dst ranks (evenly distributed)
        tp_group_usage = {}
        for sel in selections.values():
            tp_group = sel[1]
            tp_group_usage[tp_group] = tp_group_usage.get(tp_group, 0) + 1

        # Each of 4 TP groups should be used by exactly 2 destination ranks
        assert all(count == 2 for count in tp_group_usage.values())
        assert len(tp_group_usage) == 4  # 4 different TP groups

    def test_local_copy_preference_collocated(self):
        """Test that collocated mode prefers local copies when available."""
        # Setup: Collocated TP=2, World=8, DP=4
        # Each rank has both src and dst models with same configuration
        # Should always prefer local copy (dst_rank == src_rank)

        # Source metadata from all ranks
        src_meta_list = [
            self._create_metadata(rank=0, tp_group=[0, 1], dp_group=[0, 2, 4, 6]),
            self._create_metadata(rank=2, tp_group=[2, 3], dp_group=[0, 2, 4, 6]),
            self._create_metadata(rank=4, tp_group=[4, 5], dp_group=[0, 2, 4, 6]),
            self._create_metadata(rank=6, tp_group=[6, 7], dp_group=[0, 2, 4, 6]),
        ]

        # Destination rank 0 (collocated - has both src and dst)
        dst_meta = self._create_metadata(rank=0, tp_group=[0, 1], dp_group=[0, 2, 4, 6])

        # Should select rank 0 for local copy (not rank 2, 4, or 6)
        selected = select_src_metadata_balanced(src_meta_list, dst_meta, dst_rank=0)
        assert selected.owner_rank == 0, "Should prefer local copy in collocated mode"

        # Try rank 4
        dst_meta = self._create_metadata(rank=4, tp_group=[4, 5], dp_group=[0, 2, 4, 6])
        selected = select_src_metadata_balanced(src_meta_list, dst_meta, dst_rank=4)
        assert selected.owner_rank == 4, "Should prefer local copy for rank 4"

    def test_no_local_copy_non_collocated(self):
        """Test that non-collocated mode still uses DP balancing."""
        # Setup: Non-collocated - dst rank 8 not in source ranks [0,2,4,6]

        src_meta_list = [
            self._create_metadata(rank=0, tp_group=[0, 1], dp_group=[0, 2, 4, 6]),
            self._create_metadata(rank=2, tp_group=[2, 3], dp_group=[0, 2, 4, 6]),
            self._create_metadata(rank=4, tp_group=[4, 5], dp_group=[0, 2, 4, 6]),
            self._create_metadata(rank=6, tp_group=[6, 7], dp_group=[0, 2, 4, 6]),
        ]

        # Destination rank 8 (not in source ranks - non-collocated)
        dst_meta = self._create_metadata(rank=8, tp_group=[8, 9], dp_group=[8, 9])

        # Should fall back to DP balancing (not trying to find rank 8 in sources)
        selected = select_src_metadata_balanced(src_meta_list, dst_meta, dst_rank=8)
        assert selected.owner_rank in [0, 2, 4, 6], "Should select from available source ranks"
