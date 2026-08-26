# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the resharding planner functions.

These test the TP planner (_plan_tp, covering both plain and block-interleaved
layouts), DP fallback (_finalize_dp_transfers), and descriptor building in
isolation without requiring distributed init or GPU.
"""

import math

import pytest

import megatron.core.resharding.planner as planner
from megatron.core.resharding.planner import (
    _build_descriptors_for_param,
    _build_execution_batch_ids,
    _build_tensor_reshard_specs,
    _finalize_dp_transfers,
    _plan_tp,
    build_plan_from_rosters,
    index_metadata_rosters,
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
# _plan_tp
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

        ops = _plan_tp("weight", src, dst, [desc], my_global_rank=0)
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
        ops_r0 = _plan_tp("weight", src, dst, [desc], my_global_rank=0)
        assert len(ops_r0) == 1
        assert ops_r0[0][0] == 0  # from rank 0
        _verify_full_coverage(ops_r0, dim=1, expected_full_len=64)

        # Rank 1 receives second half
        ops_r1 = _plan_tp("weight", src, dst, [desc], my_global_rank=1)
        assert len(ops_r1) == 1
        assert ops_r1[0][0] == 0  # from rank 0

    def test_tp2_to_tp4(self):
        """TP2 → TP4: each dst rank receives from 1 or more src ranks."""
        src = _meta(shape=(64, 64), is_tp=True, partition_dim=1, tp_ranks=[0, 1])
        dst = _meta(shape=(64, 32), is_tp=True, partition_dim=1, tp_ranks=[0, 1, 2, 3])
        desc = _tp_descriptor(dim=1, src_ranks=[0, 1], dst_ranks=[0, 1, 2, 3])

        for rank in range(4):
            ops = _plan_tp("weight", src, dst, [desc], my_global_rank=rank)
            assert len(ops) >= 1
            _verify_full_coverage(ops, dim=1, expected_full_len=32)

    def test_same_tp_size(self):
        """TP2 → TP2: each dst rank receives from its corresponding src rank."""
        src = _meta(shape=(64, 64), is_tp=True, partition_dim=1, tp_ranks=[0, 1])
        dst = _meta(shape=(64, 64), is_tp=True, partition_dim=1, tp_ranks=[0, 1])
        desc = _tp_descriptor(dim=1, src_ranks=[0, 1], dst_ranks=[0, 1])

        ops = _plan_tp("weight", src, dst, [desc], my_global_rank=0)
        assert len(ops) == 1
        assert ops[0][0] == 0  # from self

    def test_rank_not_in_dst(self):
        """Rank not in destination group returns empty ops."""
        src = _meta(shape=(64, 64), is_tp=True, partition_dim=1, tp_ranks=[0, 1])
        dst = _meta(shape=(64, 128), is_tp=False, tp_ranks=[2])
        desc = _tp_descriptor(dim=1, src_ranks=[0, 1], dst_ranks=[2])

        ops = _plan_tp("weight", src, dst, [desc], my_global_rank=0)
        assert ops == []

    def test_empty_descriptors(self):
        """No descriptors returns empty ops."""
        src = _meta()
        dst = _meta()
        ops = _plan_tp("weight", src, dst, [], my_global_rank=0)
        assert ops == []

    def test_size_mismatch_raises(self):
        """Mismatched TP dimensions should raise."""
        src = _meta(shape=(64, 64), is_tp=True, partition_dim=1, tp_ranks=[0, 1])
        dst = _meta(shape=(64, 100), is_tp=False, tp_ranks=[0])
        desc = _tp_descriptor(dim=1, src_ranks=[0, 1], dst_ranks=[0])

        with pytest.raises(RuntimeError, match="size mismatch"):
            _plan_tp("weight", src, dst, [desc], my_global_rank=0)

    def test_dim0_partition(self):
        """TP on dimension 0 (row-parallel)."""
        src = _meta(shape=(32, 128), is_tp=True, partition_dim=0, tp_ranks=[0, 1])
        dst = _meta(shape=(64, 128), is_tp=False, tp_ranks=[0])
        desc = _tp_descriptor(dim=0, src_ranks=[0, 1], dst_ranks=[0])

        ops = _plan_tp("weight", src, dst, [desc], my_global_rank=0)
        assert len(ops) == 2
        _verify_full_coverage(ops, dim=0, expected_full_len=64)

    def test_conservation_all_ranks(self):
        """TP4 → TP2: verify all source elements are accounted for across all dst ranks."""
        src = _meta(shape=(64, 32), is_tp=True, partition_dim=1, tp_ranks=[0, 1, 2, 3])
        dst = _meta(shape=(64, 64), is_tp=True, partition_dim=1, tp_ranks=[0, 1])
        desc = _tp_descriptor(dim=1, src_ranks=[0, 1, 2, 3], dst_ranks=[0, 1])

        all_ops = []
        for rank in range(2):
            ops = _plan_tp("weight", src, dst, [desc], my_global_rank=rank)
            all_ops.extend(ops)

        # Total transferred elements should equal full tensor size on dim 1
        total = sum(op[2][1].stop - op[2][1].start for op in all_ops)
        assert total == 128  # 64 per rank * 2 ranks


# ===========================================================================
# _plan_tp
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

        ops = _plan_tp("weight", src, dst, [desc], my_global_rank=0)
        assert len(ops) > 0
        _verify_full_coverage(ops, dim=1, expected_full_len=96)

    def test_tp1_to_tp2_two_blocks(self):
        """TP1 → TP2 with two blocks."""
        src = _meta(shape=(64, 96), is_tp=False, partition_sizes=None, tp_ranks=[0])
        dst = _meta(
            shape=(64, 48), is_tp=True, partition_dim=1, partition_sizes=[32, 16], tp_ranks=[0, 1]
        )
        desc = _tp_descriptor(dim=1, src_ranks=[0], dst_ranks=[0, 1])

        ops_r0 = _plan_tp("weight", src, dst, [desc], my_global_rank=0)
        ops_r1 = _plan_tp("weight", src, dst, [desc], my_global_rank=1)
        assert len(ops_r0) > 0
        assert len(ops_r1) > 0
        _verify_full_coverage(ops_r0, dim=1, expected_full_len=48)
        _verify_full_coverage(ops_r1, dim=1, expected_full_len=48)

    def test_rank_not_in_dst(self):
        """Rank not in destination returns empty."""
        src = _meta(
            shape=(64, 48), is_tp=True, partition_dim=1, partition_sizes=[32, 16], tp_ranks=[0, 1]
        )
        dst = _meta(shape=(64, 96), tp_ranks=[2])
        desc = _tp_descriptor(dim=1, src_ranks=[0, 1], dst_ranks=[2])

        ops = _plan_tp("weight", src, dst, [desc], my_global_rank=0)
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
            ops = _plan_tp("weight", src, dst, [desc], my_global_rank=rank)
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


# ===========================================================================
# build_plan_from_rosters (local, deterministic planning + node-add stability)
# ===========================================================================


def _plan_edges(plans):
    """Collect (task_id, src_rank, dst_rank) transfers from a {rank: ReshardPlan}.

    Reads them once from every send op and once from every recv op; the two sets
    must be equal for the plan to be consistent (a matching, same-task_id recv for
    every send).
    """
    sends = {(op.task_id, r, op.peer_rank) for r, p in plans.items() for op in p.send_ops}
    recvs = {(op.task_id, op.peer_rank, r) for r, p in plans.items() for op in p.recv_ops}
    return sends, recvs


def _build_all(gathered_pairs, execution_batch_bytes=None):
    """Build every rank's plan from a rank-ordered list of (src_meta, dst_meta)."""
    dst_by_rank, src_by_name = index_metadata_rosters(gathered_pairs)
    return {
        rank: build_plan_from_rosters(
            dst_by_rank,
            src_by_name,
            rank,
            execution_batch_bytes=execution_batch_bytes,
        )
        for rank in dst_by_rank
    }


def _recv_sig(plan):
    """Identity of a plan's recv ops (task_id + slices), for stability comparisons."""
    return [(op.task_id, op.peer_rank, op.my_slice, op.peer_slice) for op in plan.recv_ops]


class TestBuildPlanFromRosters:
    """Local plan building replayed independently per rank."""

    def test_task_ids_match_across_ranks(self):
        """Sender and receiver, planned independently, agree on task_id per transfer.

        rank 0 sources a replicated weight; ranks 1 and 2 each receive a full copy.
        """
        gathered = [
            ([_meta(owner_rank=0, tp_ranks=[0], dp_ranks=[0])], []),  # rank 0: source
            ([], [_meta(owner_rank=1, tp_ranks=[1], dp_ranks=[1])]),  # rank 1: dest
            ([], [_meta(owner_rank=2, tp_ranks=[2], dp_ranks=[2])]),  # rank 2: dest
        ]
        plans = _build_all(gathered)
        sends, recvs = _plan_edges(plans)

        # Every send has a matching recv with the same task_id, and vice versa.
        assert sends == recvs
        # Two transfers: 0->1 and 0->2, with distinct task_ids.
        assert len(sends) == 2
        assert {(s, d) for _, s, d in sends} == {(0, 1), (0, 2)}
        assert len({tid for tid, _, _ in sends}) == 2

    def test_execution_batches_match_across_ranks(self):
        """Large complete parameters are split without separating their slices."""
        num_elements = 40_000_000  # 160 MB at float32
        gathered = [
            (
                [
                    _meta(
                        name=name, shape=(num_elements,), owner_rank=0, tp_ranks=[0], dp_ranks=[0]
                    )
                    for name in ("first", "second")
                ],
                [],
            ),
            (
                [],
                [
                    _meta(
                        name=name, shape=(num_elements,), owner_rank=1, tp_ranks=[1], dp_ranks=[1]
                    )
                    for name in ("first", "second")
                ],
            ),
        ]

        default_plans = _build_all(gathered)
        assert {plan.num_batches for plan in default_plans.values()} == {1}

        plans = _build_all(gathered, execution_batch_bytes=256 * 1024 * 1024)

        assert {plan.num_batches for plan in plans.values()} == {2}
        send_batches = {op.task_id: op.batch_id for op in plans[0].send_ops}
        recv_batches = {op.task_id: op.batch_id for op in plans[1].recv_ops}
        assert send_batches == recv_batches == {0: 0, 1: 1}

        dst_by_rank, src_by_name = index_metadata_rosters(gathered)
        larger_limit_plans = {
            rank: build_plan_from_rosters(
                dst_by_rank, src_by_name, rank, execution_batch_bytes=400_000_000
            )
            for rank in dst_by_rank
        }
        assert {plan.num_batches for plan in larger_limit_plans.values()} == {1}

    def test_node_add_keeps_existing_task_ids_stable(self):
        """Appending a rank rebuilds locally without renumbering existing transfers."""
        base = [
            ([_meta(owner_rank=0, tp_ranks=[0], dp_ranks=[0])], []),
            ([], [_meta(owner_rank=1, tp_ranks=[1], dp_ranks=[1])]),
            ([], [_meta(owner_rank=2, tp_ranks=[2], dp_ranks=[2])]),
        ]
        before = _build_all(base)

        # A new destination rank 3 joins; everyone replays over the grown roster.
        grown = base + [([], [_meta(owner_rank=3, tp_ranks=[3], dp_ranks=[3])])]
        after = _build_all(grown)

        sends_after, recvs_after = _plan_edges(after)
        assert sends_after == recvs_after
        # Existing receivers keep the exact same recv ops (task_id + slices).
        for rank in (1, 2):
            assert _recv_sig(before[rank]) == _recv_sig(after[rank])
        # The new rank added exactly one transfer with a fresh task_id.
        assert len(sends_after) == 3
        assert {(s, d) for _, s, d in sends_after} == {(0, 1), (0, 2), (0, 3)}


def test_execution_batch_ids_keep_replicas_together():
    """All replicas of a logical parameter receive one global batch ID."""
    dst_by_rank = {
        2: {
            "first": _meta(name="first", shape=(2,), owner_rank=2),
            "second": _meta(name="second", shape=(2,), owner_rank=2),
        },
        3: {"first": _meta(name="first", shape=(2,), owner_rank=3)},
    }
    src_by_name = {
        "first": [
            _meta(name="first", shape=(2,), owner_rank=0),
            _meta(name="first", shape=(2,), owner_rank=1),
        ],
        "second": [_meta(name="second", shape=(2,), owner_rank=0)],
    }

    batch_ids, num_batches = _build_execution_batch_ids(
        dst_by_rank, src_by_name, max_batch_bytes=12
    )

    assert batch_ids == {"first": 0, "second": 1}
    assert num_batches == 2


@pytest.mark.parametrize(
    ("gathered_limits", "local_limit", "expected_limit"),
    [
        ([200, 100], 200, 100),
        ([None, None], None, None),
        ([None, 100], None, 100),
    ],
)
def test_local_plan_resolves_rank_execution_batch_bytes(
    monkeypatch, gathered_limits, local_limit, expected_limit
):
    """Ranks agree on the smallest configured limit or preserve the unset default."""
    sentinel = object()
    forwarded = {}

    class FakeGroup:
        def rank(self):
            return 0

        def size(self):
            return 2

    monkeypatch.setattr(planner, "_extract_module_metadata", lambda *_args: [])

    def fake_all_gather_object(output, _local_entry, group):
        assert isinstance(group, FakeGroup)
        output[:] = [([], [], limit) for limit in gathered_limits]

    def fake_build(dst_by_rank, src_by_name, rank, execution_batch_bytes):
        forwarded["args"] = (dst_by_rank, src_by_name, rank)
        forwarded["execution_batch_bytes"] = execution_batch_bytes
        return sentinel

    monkeypatch.setattr(planner.dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(planner, "build_plan_from_rosters", fake_build)

    result = planner.build_local_reshard_plan(
        None, None, group=FakeGroup(), execution_batch_bytes=local_limit
    )

    assert result is sentinel
    assert forwarded == {
        "args": ({0: {}, 1: {}}, {}, 0),
        "execution_batch_bytes": expected_limit,
    }


def test_centralized_planner_compatibility_wrapper(monkeypatch):
    """The previous public planner name warns and forwards every argument."""
    sentinel = object()
    forwarded = {}

    def fake_local(src_module, dst_module, **kwargs):
        forwarded["args"] = (src_module, dst_module)
        forwarded["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(planner, "build_local_reshard_plan", fake_local)
    with pytest.warns(DeprecationWarning, match="build_local_reshard_plan"):
        result = planner.build_centralized_reshard_plan(
            "src",
            "dst",
            num_experts=8,
            group="group",
            src_rank_offset=3,
            dst_rank_offset=7,
            execution_batch_bytes=123,
        )

    assert result is sentinel
    assert forwarded == {
        "args": ("src", "dst"),
        "kwargs": {
            "num_experts": 8,
            "group": "group",
            "src_rank_offset": 3,
            "dst_rank_offset": 7,
            "execution_batch_bytes": 123,
        },
    }


class TestTensorReshardSpecs:
    """Whole-parameter metadata retained for native reshard backends."""

    def test_tp_transition_uses_local_names_and_global_shape(self):
        resolved_name = "decoder.layers.7.linear.weight"
        src_mesh = [0, 1, 2, 3]
        dst_mesh = [4, 5]
        src = {
            resolved_name: [
                _meta(
                    name="decoder.layers.3.linear.weight",
                    resolved_name=resolved_name,
                    shape=(64, 32),
                    is_tp=True,
                    partition_dim=1,
                    owner_rank=rank,
                    tp_ranks=src_mesh,
                    dp_ranks=[rank],
                )
                for rank in src_mesh
            ]
        }
        dst = {
            rank: {
                resolved_name: _meta(
                    name="decoder.layers.1.linear.weight",
                    resolved_name=resolved_name,
                    shape=(64, 64),
                    is_tp=True,
                    partition_dim=1,
                    owner_rank=rank,
                    tp_ranks=dst_mesh,
                    dp_ranks=[rank],
                )
            }
            for rank in dst_mesh
        }

        src_specs, error = _build_tensor_reshard_specs(dst, src, my_global_rank=2)
        dst_specs, dst_error = _build_tensor_reshard_specs(dst, src, my_global_rank=5)

        assert error is None
        assert dst_error is None
        assert len(src_specs) == len(dst_specs) == 1
        src_spec = src_specs[0]
        dst_spec = dst_specs[0]
        assert src_spec.global_shape == dst_spec.global_shape == (64, 128)
        assert src_spec.src_ranks == dst_spec.src_ranks == tuple(src_mesh)
        assert src_spec.dst_ranks == dst_spec.dst_ranks == tuple(dst_mesh)
        assert src_spec.src_shard_dim == dst_spec.src_shard_dim == 1
        assert src_spec.dst_shard_dim == dst_spec.dst_shard_dim == 1
        assert src_spec.src_param_name == "decoder.layers.3.linear.weight"
        assert src_spec.dst_param_name is None
        assert dst_spec.src_param_name is None
        assert dst_spec.dst_param_name == "decoder.layers.1.linear.weight"

    def test_pp_stage_pairs_are_preserved_per_parameter(self):
        src = {}
        dst = {rank: {} for rank in range(4, 6)}
        for name, src_mesh, dst_mesh in (
            ("decoder.layers.0.weight", [0, 1], [4]),
            ("decoder.layers.1.weight", [2, 3], [5]),
        ):
            src[name] = [
                _meta(
                    name=name,
                    resolved_name=name,
                    shape=(8, 4),
                    is_tp=True,
                    partition_dim=1,
                    owner_rank=rank,
                    tp_ranks=src_mesh,
                    dp_ranks=[rank],
                )
                for rank in src_mesh
            ]
            for rank in dst_mesh:
                dst[rank][name] = _meta(
                    name=name,
                    resolved_name=name,
                    shape=(8, 8),
                    owner_rank=rank,
                    tp_ranks=dst_mesh,
                    dp_ranks=[rank],
                )

        specs, error = _build_tensor_reshard_specs(dst, src, my_global_rank=0)

        assert error is None
        assert [(spec.src_ranks, spec.dst_ranks) for spec in specs] == [
            ((0, 1), (4,)),
            ((2, 3), (5,)),
        ]

    def test_strided_layout_records_reason(self):
        src_metadata = _meta(
            shape=(8, 4),
            is_tp=True,
            partition_dim=1,
            partition_stride=2,
            owner_rank=0,
            tp_ranks=[0, 1],
            dp_ranks=[0],
        )
        src_peer = _meta(
            shape=(8, 4),
            is_tp=True,
            partition_dim=1,
            partition_stride=2,
            owner_rank=1,
            tp_ranks=[0, 1],
            dp_ranks=[1],
        )
        dst_metadata = _meta(shape=(8, 8), owner_rank=2, tp_ranks=[2], dp_ranks=[2])

        specs, error = _build_tensor_reshard_specs(
            {2: {"weight": dst_metadata}}, {"weight": [src_metadata, src_peer]}, my_global_rank=0
        )

        assert specs is None
        assert "partition_stride=2" in error

    def test_block_interleaved_parameter_becomes_regular_component_specs(self):
        src_mesh = [0, 1]
        src = {
            "weight": [
                _meta(
                    shape=(8, 4),
                    is_tp=True,
                    partition_dim=1,
                    partition_sizes=[3, 1],
                    owner_rank=rank,
                    tp_ranks=src_mesh,
                    dp_ranks=[rank],
                )
                for rank in src_mesh
            ]
        }
        dst_metadata = _meta(shape=(8, 8), owner_rank=2, tp_ranks=[2], dp_ranks=[2])

        specs, error = _build_tensor_reshard_specs(
            {2: {"weight": dst_metadata}}, src, my_global_rank=2
        )

        assert error is None
        assert [spec.global_shape for spec in specs] == [(8, 6), (8, 2)]
        assert [spec.src_local_shape for spec in specs] == [(8, 3), (8, 1)]
        assert [spec.dst_local_shape for spec in specs] == [(8, 6), (8, 2)]
        assert [spec.part_index for spec in specs] == [0, 1]
        assert all(spec.part_count == 2 for spec in specs)
        assert specs[0].src_slice[1] == slice(0, 3)
        assert specs[1].src_slice[1] == slice(3, 4)
        assert specs[0].dst_slice[1] == slice(0, 6)
        assert specs[1].dst_slice[1] == slice(6, 8)

    def test_data_parallel_replicas_get_independent_mesh_specs(self):
        src = {
            "weight": [
                _meta(owner_rank=0, tp_ranks=[0], dp_ranks=[0]),
                _meta(owner_rank=1, tp_ranks=[1], dp_ranks=[1]),
            ]
        }
        dst = {
            2: {"weight": _meta(owner_rank=2, tp_ranks=[2], dp_ranks=[2])},
            3: {"weight": _meta(owner_rank=3, tp_ranks=[3], dp_ranks=[3])},
        }

        specs, error = _build_tensor_reshard_specs(dst, src, my_global_rank=0)

        assert error is None
        assert [(spec.src_ranks, spec.dst_ranks) for spec in specs] == [((0,), (2,)), ((1,), (3,))]
