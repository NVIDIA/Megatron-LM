# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from megatron.rl.shared_prefix_metadata import (
    plan_fixed_execution_slots,
    plan_group_coherent_shards,
)


def test_single_completion_groups_keep_dense_fallback_rows_and_order() -> None:
    """PPO may sample one answer per prompt; no row may disappear."""
    groups = ["ppo:0", "ppo:1", "ppo:2", "ppo:3"]
    lengths = [9, 8, 2, 1]
    slots = plan_fixed_execution_slots(
        group_ids=groups, sequence_lengths=lengths, bin_capacity=12, sequence_length_pad_multiple=4
    )
    shards = plan_group_coherent_shards(group_ids=groups, sequence_lengths=lengths, num_shards=2)
    assert len(set(groups)) == 4
    assert slots.row_slot_ids == (0, 0, 0, 0)
    assert slots.units_per_group_by_chunk == (1,)
    assert shards.shard_indices == ((0, 3), (1, 2))
    assert shards.rank_order_permutation == (0, 3, 1, 2)


def test_single_completion_slots_still_enforce_physical_capacity() -> None:
    with pytest.raises(ValueError, match="exceeds the execution bin capacity"):
        plan_fixed_execution_slots(
            group_ids=["ppo:0"],
            sequence_lengths=[9],
            bin_capacity=10,
            sequence_length_pad_multiple=4,
        )


@pytest.mark.parametrize("pad_multiple", [None, 0, 2.0, True])
def test_slot_planning_rejects_a_non_integer_pad_multiple(pad_multiple) -> None:
    with pytest.raises(ValueError, match="sequence_length_pad_multiple must be a positive integer"):
        plan_fixed_execution_slots(
            group_ids=["a", "a"],
            sequence_lengths=[3, 4],
            bin_capacity=16,
            sequence_length_pad_multiple=pad_multiple,
        )


def test_group_coherent_shards_balance_work_without_splitting_groups() -> None:
    plan = plan_group_coherent_shards(
        group_ids=["a", "a", "b", "b", "c", "c", "d", "d"],
        sequence_lengths=[9, 9, 8, 8, 2, 2, 1, 1],
        num_shards=2,
    )

    assert plan.shard_indices == ((0, 1, 6, 7), (2, 3, 4, 5))
    assert plan.rank_order_permutation == (0, 1, 6, 7, 2, 3, 4, 5)
    for shard in plan.shard_indices:
        shard_groups = {["a", "a", "b", "b", "c", "c", "d", "d"][index] for index in shard}
        assert len(shard_groups) == 2


def test_fixed_execution_slots_equalize_real_units_without_dummies() -> None:
    group_ids = ["large"] * 4 + ["small"] * 4
    lengths = [6, 6, 6, 6, 4, 4, 4, 4]

    plan = plan_fixed_execution_slots(
        group_ids=group_ids, sequence_lengths=lengths, bin_capacity=10
    )

    assert plan.units_per_group_by_chunk == (4,)
    for group_id in ("large", "small"):
        indices = [index for index, value in enumerate(group_ids) if value == group_id]
        assert {plan.row_slot_ids[index] for index in indices} == {0, 1, 2, 3}
    for group_id in ("large", "small"):
        for slot_id in range(4):
            slot_work = sum(
                lengths[index]
                for index, value in enumerate(group_ids)
                if value == group_id and plan.row_slot_ids[index] == slot_id
            )
            assert 0 < slot_work <= 10


def test_fixed_execution_slots_uses_full_length_capacity_and_padding() -> None:
    with pytest.raises(ValueError, match="exceeds the execution bin capacity"):
        plan_fixed_execution_slots(
            group_ids=["a", "a"],
            sequence_lengths=[9, 8],
            bin_capacity=10,
            sequence_length_pad_multiple=4,
        )


def test_fixed_execution_slots_plan_each_global_batch_independently() -> None:
    plan = plan_fixed_execution_slots(
        group_ids=["a", "a", "b", "b", "c", "c", "d", "d"],
        sequence_lengths=[6, 6, 6, 6, 4, 4, 4, 4],
        bin_capacity=10,
        batch_size=4,
    )

    assert plan.units_per_group_by_chunk == (2, 1)
    assert plan.row_slot_ids == (0, 1, 0, 1, 0, 0, 0, 0)


def test_group_coherent_shards_preserve_global_batch_boundaries() -> None:
    plan = plan_group_coherent_shards(
        group_ids=["a", "a", "b", "b", "c", "c", "d", "d"],
        sequence_lengths=[9, 9, 8, 8, 2, 2, 1, 1],
        num_shards=2,
        batch_size=4,
    )

    assert plan.shard_indices == ((0, 1, 4, 5), (2, 3, 6, 7))
    assert plan.rank_order_permutation == (0, 1, 4, 5, 2, 3, 6, 7)


def test_group_coherent_shards_reject_groups_crossing_batch_boundaries() -> None:
    with pytest.raises(ValueError, match="must not cross"):
        plan_group_coherent_shards(
            group_ids=["a", "b", "a", "b"],
            sequence_lengths=[1, 1, 1, 1],
            num_shards=1,
            batch_size=2,
        )


@pytest.mark.parametrize(
    "group_ids,num_shards,error",
    [(["a", "a", "b"], 1, "equal-size"), (["a", "a", "b", "b", "c", "c"], 2, "integral number")],
)
def test_group_coherent_shards_reject_unsafe_dp_shapes(
    group_ids: list[str], num_shards: int, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        plan_group_coherent_shards(
            group_ids=group_ids, sequence_lengths=[1] * len(group_ids), num_shards=num_shards
        )
