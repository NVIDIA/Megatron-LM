# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the resharding planner functions.

These test the TP planners (_plan_multi_dim_lcm, _plan_block_interleaved),
DP fallback (_finalize_dp_transfers), and descriptor building in isolation
without requiring distributed init or GPU.
"""

import math

import pytest

from megatron.core.resharding.planner import (
    _build_descriptors_for_param,
    _finalize_dp_transfers,
    _plan_block_interleaved,
    _plan_multi_dim_lcm,
)
from megatron.core.resharding.utils import ParameterMetadata, ShardingDescriptor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meta(
    name="weight",
    shape=(64, 128),
    is_tp=False,
    partition_dim=0,
    partition_stride=1,
    partition_sizes=None,
    owner_rank=0,
    tp_ranks=None,
    dp_ranks=None,
    ep_ranks=None,
    resolved_name=None,
):
    """Create a ParameterMetadata for testing."""
    import torch

    return ParameterMetadata(
        name=name,
        shape=shape,
        dtype=torch.float32,
        element_size=4,
        is_tp=is_tp,
        partition_dim=partition_dim,
        partition_stride=partition_stride,
        partition_sizes=partition_sizes,
        owner_rank=owner_rank,
        tensor_parallel_group_ranks=tp_ranks,
        data_parallel_group_ranks=dp_ranks,
        expert_parallel_group_ranks=ep_ranks,
        resolved_name=resolved_name or name,
    )


def _tp_descriptor(dim, src_ranks, dst_ranks, src_stride=1, dst_stride=1):
    """Create a TP ShardingDescriptor for testing."""
    return ShardingDescriptor(
        name="tp",
        dim=dim,
        src_stride=src_stride,
        dst_stride=dst_stride,
        src_dim_ranks=src_ranks,
        dst_dim_ranks=dst_ranks,
    )


def _verify_full_coverage(ops, dim, expected_full_len):
    """Verify that transfer ops cover every element of the destination tensor exactly once."""
    covered = set()
    for _, _, dst_slice in ops:
        s = dst_slice[dim]
        for i in range(s.start, s.stop):
            assert i not in covered, f"Duplicate coverage at dst offset {i}"
            covered.add(i)
    assert covered == set(range(expected_full_len)), (
        f"Expected coverage [0, {expected_full_len}), got gaps: "
        f"{set(range(expected_full_len)) - covered}"
    )


# ===========================================================================
# _plan_multi_dim_lcm
# ===========================================================================


class TestPlanMultiDimLcm:
    """Tests for the LCM-based TP planner."""

    def test_tp2_to_tp1(self):
        """TP2 → TP1: destination rank 0 should receive from both source ranks."""
        # Source: TP2, each rank has shape (64, 64) on dim 1
        # Destination: TP1, rank 0 has shape (64, 128) on dim 1
        src = _meta(shape=(64, 64), is_tp=True, partition_dim=1, tp_ranks=[0, 1])
        dst = _meta(shape=(64, 128), is_tp=False, tp_ranks=[0])
        desc = _tp_descriptor(dim=1, src_ranks=[0, 1], dst_ranks=[0])

        ops = _plan_multi_dim_lcm("weight", src, dst, [desc], my_global_rank=0)
        assert len(ops) == 2
        # Should receive from rank 0 and rank 1
        src_ranks = {op[0] for op in ops}
        assert src_ranks == {0, 1}
        _verify_full_coverage(ops, dim=1, expected_full_len=128)

    def test_tp1_to_tp2(self):
        """TP1 → TP2: each destination rank receives half."""
        src = _meta(shape=(64, 128), is_tp=False, tp_ranks=[0])
        dst = _meta(shape=(64, 64), is_tp=True, partition_dim=1, tp_ranks=[0, 1])
        desc = _tp_descriptor(dim=1, src_ranks=[0], dst_ranks=[0, 1])

        # Rank 0 receives first half
        ops_r0 = _plan_multi_dim_lcm("weight", src, dst, [desc], my_global_rank=0)
        assert len(ops_r0) == 1
        assert ops_r0[0][0] == 0  # from rank 0
        _verify_full_coverage(ops_r0, dim=1, expected_full_len=64)

        # Rank 1 receives second half
        ops_r1 = _plan_multi_dim_lcm("weight", src, dst, [desc], my_global_rank=1)
        assert len(ops_r1) == 1
        assert ops_r1[0][0] == 0  # from rank 0

    def test_tp2_to_tp4(self):
        """TP2 → TP4: each dst rank receives from 1 or more src ranks."""
        src = _meta(shape=(64, 64), is_tp=True, partition_dim=1, tp_ranks=[0, 1])
        dst = _meta(shape=(64, 32), is_tp=True, partition_dim=1, tp_ranks=[0, 1, 2, 3])
        desc = _tp_descriptor(dim=1, src_ranks=[0, 1], dst_ranks=[0, 1, 2, 3])

        for rank in range(4):
            ops = _plan_multi_dim_lcm("weight", src, dst, [desc], my_global_rank=rank)
            assert len(ops) >= 1
            _verify_full_coverage(ops, dim=1, expected_full_len=32)

    def test_same_tp_size(self):
        """TP2 → TP2: each dst rank receives from its corresponding src rank."""
        src = _meta(shape=(64, 64), is_tp=True, partition_dim=1, tp_ranks=[0, 1])
        dst = _meta(shape=(64, 64), is_tp=True, partition_dim=1, tp_ranks=[0, 1])
        desc = _tp_descriptor(dim=1, src_ranks=[0, 1], dst_ranks=[0, 1])

        ops = _plan_multi_dim_lcm("weight", src, dst, [desc], my_global_rank=0)
        assert len(ops) == 1
        assert ops[0][0] == 0  # from self

    def test_rank_not_in_dst(self):
        """Rank not in destination group returns empty ops."""
        src = _meta(shape=(64, 64), is_tp=True, partition_dim=1, tp_ranks=[0, 1])
        dst = _meta(shape=(64, 128), is_tp=False, tp_ranks=[2])
        desc = _tp_descriptor(dim=1, src_ranks=[0, 1], dst_ranks=[2])

        ops = _plan_multi_dim_lcm("weight", src, dst, [desc], my_global_rank=0)
        assert ops == []

    def test_empty_descriptors(self):
        """No descriptors returns empty ops."""
        src = _meta()
        dst = _meta()
        ops = _plan_multi_dim_lcm("weight", src, dst, [], my_global_rank=0)
        assert ops == []

    def test_size_mismatch_raises(self):
        """Mismatched TP dimensions should raise."""
        src = _meta(shape=(64, 64), is_tp=True, partition_dim=1, tp_ranks=[0, 1])
        dst = _meta(shape=(64, 100), is_tp=False, tp_ranks=[0])
        desc = _tp_descriptor(dim=1, src_ranks=[0, 1], dst_ranks=[0])

        with pytest.raises(RuntimeError, match="size mismatch"):
            _plan_multi_dim_lcm("weight", src, dst, [desc], my_global_rank=0)

    def test_dim0_partition(self):
        """TP on dimension 0 (row-parallel)."""
        src = _meta(shape=(32, 128), is_tp=True, partition_dim=0, tp_ranks=[0, 1])
        dst = _meta(shape=(64, 128), is_tp=False, tp_ranks=[0])
        desc = _tp_descriptor(dim=0, src_ranks=[0, 1], dst_ranks=[0])

        ops = _plan_multi_dim_lcm("weight", src, dst, [desc], my_global_rank=0)
        assert len(ops) == 2
        _verify_full_coverage(ops, dim=0, expected_full_len=64)

    def test_conservation_all_ranks(self):
        """TP4 → TP2: verify all source elements are accounted for across all dst ranks."""
        src = _meta(shape=(64, 32), is_tp=True, partition_dim=1, tp_ranks=[0, 1, 2, 3])
        dst = _meta(shape=(64, 64), is_tp=True, partition_dim=1, tp_ranks=[0, 1])
        desc = _tp_descriptor(dim=1, src_ranks=[0, 1, 2, 3], dst_ranks=[0, 1])

        all_ops = []
        for rank in range(2):
            ops = _plan_multi_dim_lcm("weight", src, dst, [desc], my_global_rank=rank)
            all_ops.extend(ops)

        # Total transferred elements should equal full tensor size on dim 1
        total = sum(op[2][1].stop - op[2][1].start for op in all_ops)
        assert total == 128  # 64 per rank * 2 ranks


# ===========================================================================
# _plan_block_interleaved
# ===========================================================================


class TestPlanBlockInterleaved:
    """Tests for the block-interleaved TP planner (Mamba in_proj style)."""

    def test_tp2_to_tp1_two_blocks(self):
        """TP2 → TP1 with two blocks of different sizes."""
        # Block sizes per rank: [32, 16] → full sizes: [64, 32]
        # Source local dim = 32+16 = 48, Dest local dim = 64+32 = 96
        src = _meta(
            shape=(64, 48), is_tp=True, partition_dim=1, partition_sizes=[32, 16], tp_ranks=[0, 1]
        )
        dst = _meta(shape=(64, 96), is_tp=False, partition_sizes=None, tp_ranks=[0])
        desc = _tp_descriptor(dim=1, src_ranks=[0, 1], dst_ranks=[0])

        ops = _plan_block_interleaved("weight", src, dst, [desc], my_global_rank=0)
        assert len(ops) > 0
        _verify_full_coverage(ops, dim=1, expected_full_len=96)

    def test_tp1_to_tp2_two_blocks(self):
        """TP1 → TP2 with two blocks."""
        src = _meta(shape=(64, 96), is_tp=False, partition_sizes=None, tp_ranks=[0])
        dst = _meta(
            shape=(64, 48), is_tp=True, partition_dim=1, partition_sizes=[32, 16], tp_ranks=[0, 1]
        )
        desc = _tp_descriptor(dim=1, src_ranks=[0], dst_ranks=[0, 1])

        ops_r0 = _plan_block_interleaved("weight", src, dst, [desc], my_global_rank=0)
        ops_r1 = _plan_block_interleaved("weight", src, dst, [desc], my_global_rank=1)
        assert len(ops_r0) > 0
        assert len(ops_r1) > 0
        _verify_full_coverage(ops_r0, dim=1, expected_full_len=48)
        _verify_full_coverage(ops_r1, dim=1, expected_full_len=48)

    def test_no_partition_sizes_raises(self):
        """Both src and dst missing partition_sizes should raise."""
        src = _meta(shape=(64, 48), is_tp=True, partition_dim=1, tp_ranks=[0, 1])
        dst = _meta(shape=(64, 96), is_tp=False, tp_ranks=[0])
        desc = _tp_descriptor(dim=1, src_ranks=[0, 1], dst_ranks=[0])

        with pytest.raises(RuntimeError, match="partition_sizes"):
            _plan_block_interleaved("weight", src, dst, [desc], my_global_rank=0)

    def test_rank_not_in_dst(self):
        """Rank not in destination returns empty."""
        src = _meta(
            shape=(64, 48), is_tp=True, partition_dim=1, partition_sizes=[32, 16], tp_ranks=[0, 1]
        )
        dst = _meta(shape=(64, 96), tp_ranks=[2])
        desc = _tp_descriptor(dim=1, src_ranks=[0, 1], dst_ranks=[2])

        ops = _plan_block_interleaved("weight", src, dst, [desc], my_global_rank=0)
        assert ops == []

    def test_three_blocks_tp2_to_tp4(self):
        """TP2 → TP4 with three blocks (simulates Mamba z,x,B,C,dt packing)."""
        # Per-rank sizes: [16, 8, 4] → full: [32, 16, 8]
        src = _meta(
            shape=(64, 28), is_tp=True, partition_dim=1, partition_sizes=[16, 8, 4], tp_ranks=[0, 1]
        )
        dst = _meta(
            shape=(64, 14),
            is_tp=True,
            partition_dim=1,
            partition_sizes=[8, 4, 2],
            tp_ranks=[0, 1, 2, 3],
        )
        desc = _tp_descriptor(dim=1, src_ranks=[0, 1], dst_ranks=[0, 1, 2, 3])

        for rank in range(4):
            ops = _plan_block_interleaved("weight", src, dst, [desc], my_global_rank=rank)
            assert len(ops) > 0
            _verify_full_coverage(ops, dim=1, expected_full_len=14)


# ===========================================================================
# _finalize_dp_transfers
# ===========================================================================


class TestFinalizeDpTransfers:
    """Tests for the DP/replicated parameter fallback planner."""

    def test_same_dp_local_copy(self):
        """Same DP group → local copy from self."""
        src = _meta(owner_rank=0, dp_ranks=[0, 1])
        dst = _meta(owner_rank=0, dp_ranks=[0, 1])

        ops = _finalize_dp_transfers("weight", src, dst, my_global_rank=0)
        assert len(ops) == 1
        assert ops[0][0] == 0  # source is self
        # Full tensor copy
        assert ops[0][1] == (slice(None), slice(None))
        assert ops[0][2] == (slice(None), slice(None))

    def test_different_dp_uses_owner_rank(self):
        """Different DP groups → uses src_metadata.owner_rank."""
        src = _meta(owner_rank=2, dp_ranks=[2, 3])
        dst = _meta(owner_rank=0, dp_ranks=[0, 1])

        ops = _finalize_dp_transfers("weight", src, dst, my_global_rank=0)
        assert len(ops) == 1
        assert ops[0][0] == 2  # src owner rank

    def test_rank_not_in_dst_dp(self):
        """Rank not in destination DP group returns empty."""
        src = _meta(owner_rank=0, dp_ranks=[0, 1])
        dst = _meta(owner_rank=2, dp_ranks=[2, 3])

        ops = _finalize_dp_transfers("weight", src, dst, my_global_rank=0)
        assert ops == []

    def test_non_collocated_dp(self):
        """Non-collocated: src and dst have completely disjoint ranks."""
        src = _meta(owner_rank=0, dp_ranks=[0, 1, 2, 3])
        dst = _meta(owner_rank=4, dp_ranks=[4, 5, 6, 7])

        ops = _finalize_dp_transfers("weight", src, dst, my_global_rank=4)
        assert len(ops) == 1
        assert ops[0][0] == 0  # from src owner


# ===========================================================================
# _build_descriptors_for_param
# ===========================================================================


class TestBuildDescriptors:
    """Tests for TP descriptor construction."""

    def test_tp_both_sides(self):
        """Both src and dst are TP → produces TP descriptor."""
        src = _meta(shape=(64, 64), is_tp=True, partition_dim=1, tp_ranks=[0, 1])
        dst = _meta(shape=(64, 128), is_tp=True, partition_dim=1, tp_ranks=[0])

        descs = _build_descriptors_for_param(src, dst)
        assert len(descs) == 1
        assert descs[0].name == "tp"
        assert descs[0].dim == 1

    def test_tp_one_side_only(self):
        """Only src is TP → still produces TP descriptor."""
        src = _meta(shape=(64, 64), is_tp=True, partition_dim=1, tp_ranks=[0, 1])
        dst = _meta(shape=(64, 128), is_tp=False, tp_ranks=[0])

        descs = _build_descriptors_for_param(src, dst)
        assert len(descs) == 1

    def test_neither_tp(self):
        """Neither side is TP → no descriptors."""
        src = _meta(shape=(64, 128), is_tp=False, tp_ranks=None)
        dst = _meta(shape=(64, 128), is_tp=False, tp_ranks=None)

        descs = _build_descriptors_for_param(src, dst)
        assert descs == []

    def test_size_conservation_failure(self):
        """Mismatched global sizes should raise."""
        src = _meta(shape=(64, 64), is_tp=True, partition_dim=1, tp_ranks=[0, 1])
        # Global = 128, but dst claims global = 100 (1 rank * 100)
        dst = _meta(shape=(64, 100), is_tp=True, partition_dim=1, tp_ranks=[0])

        with pytest.raises(RuntimeError, match="Cannot build TP descriptor"):
            _build_descriptors_for_param(src, dst)

    def test_missing_tp_ranks(self):
        """Missing TP group ranks → no descriptors (not enough context)."""
        src = _meta(shape=(64, 64), is_tp=True, partition_dim=1, tp_ranks=None)
        dst = _meta(shape=(64, 128), is_tp=False, tp_ranks=[0])

        descs = _build_descriptors_for_param(src, dst)
        assert descs == []
