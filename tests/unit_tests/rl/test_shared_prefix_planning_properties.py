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

"""Seeded end-to-end properties of shard -> slot -> plan -> align."""

import random
from collections import defaultdict

import pytest

from megatron.rl.shared_prefix_alignment import (
    align_physical_units,
    align_training_units,
    materialize_alignment,
)
from megatron.rl.shared_prefix_dense_bins import (
    plan_dense_training_bins,
    share_prefixes_in_dense_training_bins,
)
from megatron.rl.shared_prefix_execution import plan_shared_prefix_execution_units
from megatron.rl.shared_prefix_metadata import (
    get_prescribed_shared_prefix_slots,
    plan_fixed_execution_slots,
    plan_group_coherent_shards,
)
from megatron.rl.shared_prefix_packing import (
    MAX_SHARED_PREFIX_BRANCHES,
    SharedPrefixForestLayout,
    SharedPrefixRow,
    _round_up,
)

WORKER_MODES = {
    "slots": dict(),
    "forest": dict(pack_groups=True, repack_groups=True),
    "forest_dense": dict(pack_groups=True, repack_groups=True, pack_dense_fallbacks=True),
}


def _first_fit_decreasing(capacity):
    def pack(costs):
        bins, loads = [], []
        for index in sorted(range(len(costs)), key=lambda index: (-costs[index], index)):
            for bin_index, load in enumerate(loads):
                if load + costs[index] <= capacity:
                    bins[bin_index].append(index)
                    loads[bin_index] += costs[index]
                    break
            else:
                bins.append([index])
                loads.append(costs[index])
        return bins

    return pack


def _reorder_data(values, permutation):
    """NeMo-RL ``BatchedDataDict.reorder_data``: output position -> source row."""
    return [values[source] for _, source in sorted(zip(permutation, range(len(values))))]


def _random_batch(rng):
    multiple, dp = rng.choice([1, 2, 4, 8]), rng.choice([1, 2, 4])
    group_size = rng.choice([1, 2, 3, 4, 8, 17])
    groups_per_chunk, num_chunks = rng.randint(1, 3) * dp, rng.randint(1, 2)
    rows = []
    for chunk in range(num_chunks):
        chunk_rows = []
        for group in range(groups_per_chunk):
            prompt = tuple(rng.randrange(50) for _ in range(rng.choice([0, 1, 4, 16])))
            for _ in range(group_size):
                row_prompt = prompt
                if prompt and rng.random() < 0.05:
                    row_prompt = prompt[:-1] + (prompt[-1] + 1,)
                completion = rng.randint(0 if row_prompt else 1, rng.choice([4, 32]))
                chunk_rows.append((f"c{chunk}g{group}", row_prompt, completion))
        rng.shuffle(chunk_rows)
        rows.extend(chunk_rows)
    max_padded = max(_round_up(len(prompt) + length, multiple) for _, prompt, length in rows)
    capacity = _round_up(max_padded * rng.choice([1, 2, 3]), multiple)
    return rows, multiple, dp, groups_per_chunk * group_size, capacity


def _expanded(unit, costs):
    if unit.shared_layout is None:
        return sum(costs[row] for row in unit.row_indices)
    return sum(
        len(root.row_indices) * root.prompt_length + sum(root.physical_completion_lengths)
        for _, root in unit.shared_layout.iter_roots()
    )


def _check_units(units, costs, *, capacity, multiple, num_rows, expanded_budget):
    assert sorted(row for unit in units for row in unit.row_indices) == list(range(num_rows))
    for unit in units:
        assert _round_up(unit.physical_length, multiple) <= capacity
        if unit.shared_layout is not None:
            assert unit.shared_layout.row_indices == unit.row_indices
            assert unit.shared_layout.physical_total_length == unit.physical_length
        if expanded_budget:
            assert _expanded(unit, costs) <= capacity


@pytest.mark.parametrize("seed", range(4))
def test_shard_slot_plan_align_invariants(seed):
    rng = random.Random(seed)
    for _ in range(25):
        rows, multiple, dp, chunk_size, capacity = _random_batch(rng)
        group_ids = [group for group, _, _ in rows]
        lengths = [len(prompt) + completion for _, prompt, completion in rows]
        slot_plan = plan_fixed_execution_slots(
            group_ids=group_ids,
            sequence_lengths=lengths,
            bin_capacity=capacity,
            batch_size=chunk_size,
            sequence_length_pad_multiple=multiple,
        )
        shards = plan_group_coherent_shards(
            group_ids=group_ids, sequence_lengths=lengths, num_shards=dp, batch_size=chunk_size
        )
        flat = [row for shard in shards.shard_indices for row in shard]
        restored = (
            flat
            if shards.rank_order_permutation is None
            else _reorder_data(flat, shards.rank_order_permutation)
        )
        assert restored == list(range(len(rows)))

        # Every group has K nonempty, in-budget slots of at most the branch limit.
        for chunk, slots_per_group in enumerate(slot_plan.units_per_group_by_chunk):
            members = defaultdict(lambda: defaultdict(list))
            for row in range(chunk * chunk_size, (chunk + 1) * chunk_size):
                members[group_ids[row]][slot_plan.row_slot_ids[row]].append(row)
            for slots in members.values():
                assert sorted(slots) == list(range(slots_per_group))
                for slot_rows in slots.values():
                    assert len(slot_rows) <= MAX_SHARED_PREFIX_BRANCHES
                    assert sum(_round_up(lengths[row], multiple) for row in slot_rows) <= capacity

        local_size = chunk_size // dp
        for chunk in range(len(rows) // chunk_size):
            locals_ = [
                shard[chunk * local_size : (chunk + 1) * local_size]
                for shard in shards.shard_indices
            ]
            # Ranks keep global-batch boundaries and never split a group.
            assert all(row // chunk_size == chunk for local in locals_ for row in local)
            owners = {group_ids[row]: rank for rank, local in enumerate(locals_) for row in local}
            assert all(
                owners[group_ids[row]] == rank for rank, l in enumerate(locals_) for row in l
            )
            for mode, options in WORKER_MODES.items():
                for forward_only, evaluation in ((False, False), (True, False), (True, True)):
                    if evaluation and not options:
                        continue
                    _check_rank_plans(
                        rows,
                        locals_,
                        slot_plan.row_slot_ids,
                        capacity=capacity,
                        multiple=multiple,
                        options=dict(options, forward_only=forward_only),
                        evaluation=evaluation,
                        aligned=bool(options),
                    )
            _check_dense_bins(rows, locals_, capacity=capacity, multiple=multiple)


def _local_rows(rows, local):
    return [
        SharedPrefixRow(index, rows[source][0], rows[source][1], rows[source][2])
        for index, source in enumerate(local)
    ]


def _check_rank_plans(
    rows, locals_, row_slot_ids, *, capacity, multiple, options, evaluation, aligned
):
    physical_only = options["forward_only"] and evaluation
    plans = []
    for local in locals_:
        local_rows = _local_rows(rows, local)
        row_slots = get_prescribed_shared_prefix_slots(
            group_ids=[row.group_id for row in local_rows],
            slot_ids=[row_slot_ids[source] for source in local],
        )
        units = plan_shared_prefix_execution_units(
            local_rows,
            row_slots=row_slots,
            bin_capacity=capacity,
            padding_multiple=multiple,
            evaluation_packing=evaluation,
            dense_packer=_first_fit_decreasing(capacity),
            **options,
        )
        costs = [_round_up(row.total_length, multiple) for row in local_rows]
        _check_units(
            units,
            costs,
            capacity=capacity,
            multiple=multiple,
            num_rows=len(local_rows),
            expanded_budget=not physical_only,
        )
        plans.append((units, costs))
    counts = [len(units) for units, _ in plans]
    if not aligned:
        # Driver slots alone give every DP rank the same forward count.
        assert len(set(counts)) == 1
        return
    target = max(counts)
    if target > min(len(costs) for _, costs in plans):
        return
    for units, costs in plans:
        if physical_only:
            aligned_units, _ = align_physical_units(
                units,
                costs=costs,
                capacity=capacity,
                target_count=target,
                padding_multiple=multiple,
            )
        elif options["forward_only"]:
            aligned_units, _ = materialize_alignment(
                units, costs=costs, capacity=capacity, target_count=target
            )
        else:
            aligned_units, _ = align_training_units(
                units,
                costs=costs,
                capacity=capacity,
                target_count=target,
                padding_multiple=multiple,
            )
        assert len(aligned_units) == target
        _check_units(
            aligned_units,
            costs,
            capacity=capacity,
            multiple=multiple,
            num_rows=len(costs),
            expanded_budget=not physical_only,
        )


def _check_dense_bins(rows, locals_, *, capacity, multiple):
    plans = []
    for local in locals_:
        local_rows = _local_rows(rows, local)
        costs = [_round_up(row.total_length, multiple) for row in local_rows]
        units = plan_dense_training_bins(
            costs=costs, bin_capacity=capacity, dense_packer=_first_fit_decreasing(capacity)
        )
        plans.append((local_rows, costs, units))
    target = max(len(units) for _, _, units in plans)
    if target > min(len(costs) for _, costs, _ in plans):
        return
    for local_rows, costs, units in plans:
        aligned_units, _ = materialize_alignment(
            units, costs=costs, capacity=capacity, target_count=target
        )
        shared = share_prefixes_in_dense_training_bins(
            local_rows, aligned_units, costs=costs, padding_multiple=multiple, bin_capacity=capacity
        )
        assert len(shared) == target
        _check_units(
            shared,
            costs,
            capacity=capacity,
            multiple=multiple,
            num_rows=len(costs),
            expanded_budget=True,
        )
        for dense, unit in zip(aligned_units, shared):
            assert set(dense.row_indices) == set(unit.row_indices)
            assert _expanded(unit, costs) == dense.physical_length
            if isinstance(unit.shared_layout, SharedPrefixForestLayout):
                roots = unit.shared_layout.roots
                assert unit.shared_layout.mtp_loss_group_root_counts == (len(roots),)
                assert all(len(root.row_indices) <= MAX_SHARED_PREFIX_BRANCHES for root in roots)
