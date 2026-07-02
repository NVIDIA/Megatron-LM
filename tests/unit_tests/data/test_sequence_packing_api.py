# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.datasets.sequence_packing import (
    ExecutionGroup,
    ExecutionPlan,
    PackAssignment,
    PackDescriptor,
    PackingConstraints,
    PlacementKind,
    PlacementResources,
    PlacementScheduler,
    RankGroup,
    SequenceDescriptor,
    SequencePackingScheduler,
)


def test_pack_descriptor_builds_thd_metadata():
    pack = PackDescriptor(
        pack_id=7,
        sample_ids=[10, 11, 12],
        sequence_lengths=[5, 2, 4],
        padded_sequence_lengths=[8, 4, 4],
    )

    assert pack.sample_ids == (10, 11, 12)
    assert pack.sequence_lengths == (5, 2, 4)
    assert pack.total_tokens == 11
    assert pack.materialized_tokens == 16
    assert pack.cu_seqlens == (0, 5, 7, 11)
    assert pack.cu_seqlens_padded == (0, 8, 12, 16)


def test_execution_plan_preserves_execution_order():
    group0 = ExecutionGroup(
        group_id=0,
        assignments=[
            PackAssignment(
                pack_id=3,
                placement_kind=PlacementKind.DYNAMIC_CP,
                executor_ranks=[0, 1],
                rank_group_id="cp2_0",
                local_cp_size=2,
            )
        ],
    )
    group1 = ExecutionGroup(
        group_id=1,
        assignments=[
            PackAssignment(
                pack_id=4,
                placement_kind=PlacementKind.DATA_PARALLEL,
                executor_ranks=[2],
            )
        ],
        requires_barrier_after=False,
    )

    plan = ExecutionPlan(groups=[group0, group1], num_microbatches=2)

    assert plan.assignment_count == 2
    assert plan.pack_ids == (3, 4)
    assert plan.groups[0].requires_barrier_after is True
    assert plan.groups[1].requires_barrier_after is False


def test_api_rejects_invalid_shapes_and_lengths():
    with pytest.raises(ValueError, match="num_tokens"):
        SequenceDescriptor(sample_id=0, num_tokens=0)

    with pytest.raises(ValueError, match="padded_num_tokens"):
        SequenceDescriptor(sample_id=0, num_tokens=8, padded_num_tokens=4)

    with pytest.raises(ValueError, match="same length"):
        PackDescriptor(pack_id=0, sample_ids=[0, 1], sequence_lengths=[8])

    with pytest.raises(ValueError, match="unique"):
        RankGroup(group_id="bad", ranks=[0, 0])

    with pytest.raises(ValueError, match="local_cp_size"):
        PackAssignment(
            pack_id=0,
            placement_kind=PlacementKind.DYNAMIC_CP,
            executor_ranks=[0],
            local_cp_size=0,
        )


def test_scheduler_protocols_are_structural():
    class SimplePackingScheduler:
        def build_packs(self, sequences, constraints):
            del constraints
            return [
                PackDescriptor(
                    pack_id=0,
                    sample_ids=[sequence.sample_id for sequence in sequences],
                    sequence_lengths=[sequence.num_tokens for sequence in sequences],
                )
            ]

    class SimplePlacementScheduler:
        def build_plan(self, packs, resources):
            assignment = PackAssignment(
                pack_id=packs[0].pack_id,
                placement_kind=PlacementKind.STATIC_CP,
                executor_ranks=resources.rank_groups["cp"].ranks,
                rank_group_id="cp",
            )
            return ExecutionPlan(groups=[ExecutionGroup(group_id=0, assignments=[assignment])])

    packing_scheduler = SimplePackingScheduler()
    placement_scheduler = SimplePlacementScheduler()

    assert isinstance(packing_scheduler, SequencePackingScheduler)
    assert isinstance(placement_scheduler, PlacementScheduler)

    sequences = [
        SequenceDescriptor(sample_id=0, num_tokens=5),
        SequenceDescriptor(sample_id=1, num_tokens=7),
    ]
    packs = packing_scheduler.build_packs(
        sequences, PackingConstraints(max_tokens_per_pack=16, pad_to_multiple=8)
    )
    resources = PlacementResources(cp_size=2, rank_groups={"cp": RankGroup("cp", [0, 1])})
    plan = placement_scheduler.build_plan(packs, resources)

    assert packs[0].cu_seqlens == (0, 5, 12)
    assert plan.groups[0].assignments[0].executor_ranks == (0, 1)
