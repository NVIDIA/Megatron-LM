# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for logical-coordinate reshard planning."""

import math
from itertools import product

import pytest
import torch

import megatron.core.resharding.planner as planner
from megatron.core.resharding.gtp_planner import plan_gtp
from megatron.core.resharding.planner import (
    _build_descriptors_for_param,
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
    is_gtp=False,
    gtp_ranks=None,
    gtp_pad_length=0,
    resolved_name=None,
):
    """Create parameter metadata for testing."""
    return ParameterMetadata(
        name=name,
        shape=shape,
        dtype=torch.float32,
        element_size=4,
        is_tp=is_tp,
        partition_dim=partition_dim,
        partition_stride=partition_stride,
        partition_sizes=partition_sizes,
        is_gtp=is_gtp,
        gtp_remat_group_ranks=gtp_ranks,
        gtp_pad_length=gtp_pad_length,
        owner_rank=owner_rank,
        tensor_parallel_group_ranks=tp_ranks,
        data_parallel_group_ranks=dp_ranks,
        expert_parallel_group_ranks=ep_ranks,
        resolved_name=resolved_name or name,
    )


def _tp_metadata(shape, partition_dim, tp_ranks, **kwargs):
    """Create metadata for every rank in one TP group."""
    return [
        _meta(
            shape=shape,
            is_tp=len(tp_ranks) > 1,
            partition_dim=partition_dim,
            owner_rank=rank,
            tp_ranks=tp_ranks,
            dp_ranks=[rank],
            **kwargs,
        )
        for rank in tp_ranks
    ]


def _verify_nd_coverage(ops, expected_shape):
    """Verify that destination slices cover each logical local element once."""
    covered = set()
    for _, _, dst_slice in ops:
        ranges = [range(part.start, part.stop) for part in dst_slice]
        for index in product(*ranges):
            assert index not in covered, f"Duplicate coverage at destination index {index}"
            covered.add(index)
    assert len(covered) == math.prod(expected_shape)


def _tp_global_index(local_index, tp_rank, tp_size, local_size, stride):
    """Map one plain/strided TP-local index to its logical global index."""
    segment_size = local_size // stride
    segment, offset = divmod(local_index, segment_size)
    return segment * segment_size * tp_size + tp_rank * segment_size + offset


class TestLogicalShardPlanner:
    """One algorithm covers replicated, TP, packed, strided, and GTP layouts."""

    @staticmethod
    def _tp2_gtp2_metadata(shape, partition_dim, **kwargs):
        topology = {
            0: ([0, 1], [0, 2]),
            1: ([0, 1], [1, 3]),
            2: ([2, 3], [0, 2]),
            3: ([2, 3], [1, 3]),
        }
        return [
            _meta(
                shape=shape,
                is_tp=True,
                partition_dim=partition_dim,
                owner_rank=rank,
                tp_ranks=tp_ranks,
                dp_ranks=[rank],
                is_gtp=True,
                gtp_ranks=gtp_ranks,
                **kwargs,
            )
            for rank, (tp_ranks, gtp_ranks) in topology.items()
        ]

    def test_replicated_transfer(self):
        src = [_meta(owner_rank=2, dp_ranks=[2, 3])]
        dst = _meta(owner_rank=4, dp_ranks=[4, 5])

        ops = plan_sharded_transfer("weight", src, src[0], dst)

        assert ops == [(2, (slice(0, 64), slice(0, 128)), (slice(0, 64), slice(0, 128)))]

    def test_tp2_to_unsharded(self):
        src = _tp_metadata(shape=(64, 64), partition_dim=1, tp_ranks=[0, 1])
        dst = _meta(shape=(64, 128), owner_rank=2, tp_ranks=[2], dp_ranks=[2])

        ops = plan_sharded_transfer("weight", src, src[0], dst)

        assert ops == [
            (0, (slice(0, 64), slice(0, 64)), (slice(0, 64), slice(0, 64))),
            (1, (slice(0, 64), slice(0, 64)), (slice(0, 64), slice(64, 128))),
        ]

    def test_unsharded_to_tp2(self):
        src = [_meta(shape=(64, 128), owner_rank=0, tp_ranks=[0], dp_ranks=[0])]
        dst_group = [1, 2]

        for tp_rank, rank in enumerate(dst_group):
            dst = _meta(
                shape=(64, 64),
                is_tp=True,
                partition_dim=1,
                owner_rank=rank,
                tp_ranks=dst_group,
                dp_ranks=[rank],
            )
            ops = plan_sharded_transfer("weight", src, src[0], dst)
            assert ops == [
                (
                    0,
                    (slice(0, 64), slice(tp_rank * 64, (tp_rank + 1) * 64)),
                    (slice(0, 64), slice(0, 64)),
                )
            ]

    def test_tp2_to_tp4(self):
        src = _tp_metadata(shape=(64, 64), partition_dim=1, tp_ranks=[0, 1])
        dst_group = [2, 3, 4, 5]

        for rank in dst_group:
            dst = _meta(
                shape=(64, 32),
                is_tp=True,
                partition_dim=1,
                owner_rank=rank,
                tp_ranks=dst_group,
                dp_ranks=[rank],
            )
            ops = plan_sharded_transfer("weight", src, src[0], dst)
            _verify_nd_coverage(ops, (64, 32))

    @pytest.mark.parametrize(
        ("partition_dim", "src_size", "dst_size", "src_stride", "dst_stride"),
        [(0, 2, 1, 1, 1), (1, 4, 2, 1, 1), (0, 3, 4, 1, 1), (1, 2, 3, 2, 3), (0, 4, 2, 3, 2)],
    )
    def test_tp_layouts_preserve_logical_indices(
        self, partition_dim, src_size, dst_size, src_stride, dst_stride
    ):
        global_size = 144
        src_group = list(range(src_size))
        dst_group = list(range(100, 100 + dst_size))
        src_shape = [6, 6]
        dst_shape = [6, 6]
        src_shape[partition_dim] = global_size // src_size
        dst_shape[partition_dim] = global_size // dst_size
        src = _tp_metadata(
            shape=tuple(src_shape),
            partition_dim=partition_dim,
            partition_stride=src_stride,
            tp_ranks=src_group,
        )

        for dst_tp_rank, dst_rank in enumerate(dst_group):
            dst = _meta(
                shape=tuple(dst_shape),
                is_tp=dst_size > 1,
                partition_dim=partition_dim,
                partition_stride=dst_stride,
                owner_rank=dst_rank,
                tp_ranks=dst_group,
                dp_ranks=[dst_rank],
            )
            ops = plan_sharded_transfer("weight", src, src[0], dst)
            _verify_nd_coverage(ops, tuple(dst_shape))
            for src_rank, src_slice, dst_slice in ops:
                for src_index, dst_index in zip(
                    range(src_slice[partition_dim].start, src_slice[partition_dim].stop),
                    range(dst_slice[partition_dim].start, dst_slice[partition_dim].stop),
                ):
                    src_global = _tp_global_index(
                        src_index, src_rank, src_size, src_shape[partition_dim], src_stride
                    )
                    dst_global = _tp_global_index(
                        dst_index, dst_tp_rank, dst_size, dst_shape[partition_dim], dst_stride
                    )
                    assert src_global == dst_global

    def test_strided_tp(self):
        src = _tp_metadata(shape=(64, 32), partition_dim=1, partition_stride=2, tp_ranks=[0, 1])
        dst = _meta(shape=(64, 64), owner_rank=2, tp_ranks=[2], dp_ranks=[2])

        ops = plan_sharded_transfer("weight", src, src[0], dst)

        assert [rank for rank, _, _ in ops] == [0, 1, 0, 1]
        _verify_nd_coverage(ops, (64, 64))

    def test_packed_tp2_to_tp4(self):
        src = _tp_metadata(
            shape=(64, 28), partition_dim=1, partition_sizes=[16, 8, 4], tp_ranks=[0, 1]
        )
        dst_group = [2, 3, 4, 5]

        for rank in dst_group:
            dst = _meta(
                shape=(64, 14),
                is_tp=True,
                partition_dim=1,
                partition_sizes=[8, 4, 2],
                owner_rank=rank,
                tp_ranks=dst_group,
                dp_ranks=[rank],
            )
            ops = plan_sharded_transfer("weight", src, src[0], dst)
            _verify_nd_coverage(ops, (64, 14))

    def test_logical_shape_mismatch_raises(self):
        src = _tp_metadata(shape=(64, 64), partition_dim=1, tp_ranks=[0, 1])
        dst = _meta(shape=(64, 100), owner_rank=2, tp_ranks=[2], dp_ranks=[2])

        with pytest.raises(RuntimeError, match="logical shape mismatch"):
            plan_sharded_transfer("weight", src, src[0], dst)

    def test_overlapping_source_metadata_raises(self):
        src = [
            _meta(
                shape=(4, 3),
                owner_rank=rank,
                tp_ranks=[rank],
                dp_ranks=[rank],
                is_gtp=True,
                gtp_ranks=group,
            )
            for rank, group in ((0, [0, 1]), (1, [1, 0]))
        ]
        dst = _meta(shape=(8, 3), owner_rank=2, tp_ranks=[2], dp_ranks=[2])

        with pytest.raises(RuntimeError, match="overlapping destination coverage"):
            plan_sharded_transfer("weight", src, src[0], dst)

    def test_missing_tp_group_raises(self):
        src = [_meta(shape=(64, 64), is_tp=True, partition_dim=1, owner_rank=0)]
        dst = _meta(shape=(64, 128), owner_rank=1)

        with pytest.raises(RuntimeError, match="missing tensor-parallel group"):
            plan_sharded_transfer("weight", src, src[0], dst)

    def test_gtp2_to_unsharded(self):
        src = [
            _meta(
                shape=(4, 3),
                owner_rank=rank,
                tp_ranks=[rank],
                dp_ranks=[rank],
                is_gtp=True,
                gtp_ranks=[0, 1],
            )
            for rank in (0, 1)
        ]
        dst = _meta(shape=(8, 3), owner_rank=2, tp_ranks=[2], dp_ranks=[2])

        ops = plan_sharded_transfer("weight", src, src[0], dst)

        assert [rank for rank, _, _ in ops] == [0, 1]
        _verify_nd_coverage(ops, (8, 3))

    def test_uses_only_selected_source_replica(self):
        src = []
        for group in ([0, 1], [2, 3]):
            src.extend(
                _meta(
                    shape=(4, 3),
                    owner_rank=rank,
                    tp_ranks=[rank],
                    dp_ranks=[rank % 2, rank % 2 + 2],
                    is_gtp=True,
                    gtp_ranks=group,
                )
                for rank in group
            )
        dst = _meta(shape=(8, 3), owner_rank=4, tp_ranks=[4], dp_ranks=[4])

        ops = plan_sharded_transfer("weight", src, src[2], dst)

        assert [rank for rank, _, _ in ops] == [2, 3]
        _verify_nd_coverage(ops, (8, 3))

    def test_unsharded_to_padded_gtp2(self):
        src = [_meta(shape=(6, 2), owner_rank=0, tp_ranks=[0], dp_ranks=[0])]
        dst = _meta(
            shape=(4, 2),
            owner_rank=2,
            tp_ranks=[2],
            dp_ranks=[2],
            is_gtp=True,
            gtp_ranks=[1, 2],
            gtp_pad_length=2,
        )

        ops = plan_sharded_transfer("weight", src, src[0], dst)

        assert ops == [(0, (slice(4, 6), slice(0, 2)), (slice(0, 2), slice(0, 2)))]

    def test_column_tp2_gtp2_to_unsharded(self):
        src = self._tp2_gtp2_metadata(shape=(2, 3), partition_dim=0)
        dst = _meta(shape=(8, 3), owner_rank=4, tp_ranks=[4], dp_ranks=[4])

        ops = plan_sharded_transfer("weight", src, src[0], dst)

        assert [rank for rank, _, _ in ops] == [0, 2, 1, 3]
        _verify_nd_coverage(ops, (8, 3))

    def test_row_tp2_gtp2_to_unsharded(self):
        src = self._tp2_gtp2_metadata(shape=(4, 4), partition_dim=1)
        dst = _meta(shape=(8, 8), owner_rank=4, tp_ranks=[4], dp_ranks=[4])

        ops = plan_sharded_transfer("weight", src, src[0], dst)

        assert len(ops) == 4
        assert {rank for rank, _, _ in ops} == {0, 1, 2, 3}
        _verify_nd_coverage(ops, (8, 8))

    def test_strided_tp_gtp_to_unsharded(self):
        src = self._tp2_gtp2_metadata(shape=(2, 3), partition_dim=0, partition_stride=2)
        dst = _meta(shape=(8, 3), owner_rank=4, tp_ranks=[4], dp_ranks=[4])

        ops = plan_sharded_transfer("weight", src, src[0], dst)

        assert [rank for rank, _, _ in ops] == [0, 1, 2, 3]
        assert [dst_slice[0] for _, _, dst_slice in ops] == [
            slice(0, 2),
            slice(2, 4),
            slice(4, 6),
            slice(6, 8),
        ]
        _verify_nd_coverage(ops, (8, 3))

    def test_packed_tp_gtp_layout_with_padding(self):
        src = self._tp2_gtp2_metadata(
            shape=(2, 2), partition_dim=0, partition_sizes=[2, 1], gtp_pad_length=1
        )
        dst = _meta(
            shape=(6, 2),
            is_tp=True,
            partition_dim=0,
            partition_sizes=[4, 2],
            owner_rank=4,
            tp_ranks=[4],
            dp_ranks=[4],
        )

        ops = plan_sharded_transfer("weight", src, src[0], dst)

        assert [rank for rank, _, _ in ops] == [0, 1, 2, 3]
        _verify_nd_coverage(ops, (6, 2))


def _plan_edges(plans):
    """Collect matching send and receive edges from per-rank plans."""
    sends = {
        (op.task_id, rank, op.peer_rank) for rank, plan in plans.items() for op in plan.send_ops
    }
    recvs = {
        (op.task_id, op.peer_rank, rank) for rank, plan in plans.items() for op in plan.recv_ops
    }
    return sends, recvs


def _build_all(gathered_pairs):
    """Build every rank's plan from a rank-ordered metadata roster."""
    dst_by_rank, src_by_name = index_metadata_rosters(gathered_pairs)
    return {rank: build_plan_from_rosters(dst_by_rank, src_by_name, rank) for rank in dst_by_rank}


def _recv_sig(plan):
    """Return the stable identity of a plan's receive operations."""
    return [(op.task_id, op.peer_rank, op.my_slice, op.peer_slice) for op in plan.recv_ops]


class TestBuildPlanFromRosters:
    """The global deterministic schedule uses the same shard algorithm."""

    def test_gtp_schedule_matches_across_ranks(self):
        src = [
            _meta(
                shape=(4, 3),
                owner_rank=rank,
                tp_ranks=[rank],
                dp_ranks=[rank],
                is_gtp=True,
                gtp_ranks=[0, 1],
            )
            for rank in (0, 1)
        ]
        dst = _meta(shape=(8, 3), owner_rank=2, tp_ranks=[2], dp_ranks=[2])
        plans = _build_all([([src[0]], []), ([src[1]], []), ([], [dst])])

        sends, recvs = _plan_edges(plans)
        assert sends == recvs
        assert {(src_rank, dst_rank) for _, src_rank, dst_rank in sends} == {(0, 2), (1, 2)}
        assert [op.my_slice[0] for op in plans[2].recv_ops] == [slice(0, 4), slice(4, 8)]

    def test_task_ids_match_across_ranks(self):
        gathered = [
            ([_meta(owner_rank=0, tp_ranks=[0], dp_ranks=[0])], []),
            ([], [_meta(owner_rank=1, tp_ranks=[1], dp_ranks=[1])]),
            ([], [_meta(owner_rank=2, tp_ranks=[2], dp_ranks=[2])]),
        ]
        plans = _build_all(gathered)
        sends, recvs = _plan_edges(plans)

        assert sends == recvs
        assert len(sends) == 2
        assert {(src, dst) for _, src, dst in sends} == {(0, 1), (0, 2)}
        assert len({task_id for task_id, _, _ in sends}) == 2

    def test_node_add_keeps_existing_task_ids_stable(self):
        base = [
            ([_meta(owner_rank=0, tp_ranks=[0], dp_ranks=[0])], []),
            ([], [_meta(owner_rank=1, tp_ranks=[1], dp_ranks=[1])]),
            ([], [_meta(owner_rank=2, tp_ranks=[2], dp_ranks=[2])]),
        ]
        before = _build_all(base)
        after = _build_all(base + [([], [_meta(owner_rank=3, tp_ranks=[3], dp_ranks=[3])])])

        sends, recvs = _plan_edges(after)
        assert sends == recvs
        for rank in (1, 2):
            assert _recv_sig(before[rank]) == _recv_sig(after[rank])
        assert len(sends) == 3
        assert {(src, dst) for _, src, dst in sends} == {(0, 1), (0, 2), (0, 3)}


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
            "src", "dst", num_experts=8, group="group", src_rank_offset=3, dst_rank_offset=7
        )

    assert result is sentinel
    assert forwarded == {
        "args": ("src", "dst"),
        "kwargs": {"num_experts": 8, "group": "group", "src_rank_offset": 3, "dst_rank_offset": 7},
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
