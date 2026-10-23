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

"""Planning examples, invalid inputs, branch-limit splits and seeded planning properties."""

import itertools
import math
import pickle
import random
from collections import defaultdict
from dataclasses import FrozenInstanceError, replace
from functools import partial

import pytest
import torch

from megatron.rl.shared_prefix_alignment import (
    align_physical_units,
    align_rows_to_count,
    align_training_units,
    materialize_alignment,
)
from megatron.rl.shared_prefix_cost import estimate_shared_prefix_row_work
from megatron.rl.shared_prefix_dense_bins import (
    plan_dense_training_bins,
    share_prefixes_in_dense_training_bins,
)
from megatron.rl.shared_prefix_execution import (
    SharedPrefixExecutionPlan,
    SharedPrefixExecutionUnit,
    plan_shared_prefix_execution_units,
    validate_shared_prefix_execution_units,
)
from megatron.rl.shared_prefix_metadata import (
    get_prescribed_shared_prefix_slots,
    plan_fixed_execution_slots,
    plan_group_coherent_shards,
)
from megatron.rl.shared_prefix_packing import (
    MAX_SHARED_PREFIX_BRANCHES,
    SharedPrefixForestLayout,
    SharedPrefixLayout,
    SharedPrefixRow,
    _round_up,
    build_shared_prefix_layout,
    plan_shared_prefix_bins,
    split_rows_evenly,
)
from megatron.rl.shared_prefix_tensors import (
    get_shared_prefix_context_parallel_indices,
    materialize_shared_prefix_layout,
    resolve_shared_prefix_parallel_topology,
    resolve_shared_prefix_physical_padding_multiple,
)
from megatron.rl.tree_layout import PackedTreeLayout
from tests.unit_tests.rl.shared_prefix.shared_prefix_oracles import (
    assert_observed,
    batch_rows,
    covered_rows,
    nested_tree,
    note,
    pack,
    row,
)

# Two stars stored interleaved: root a, root b, then a's leaf and b's leaf.
_INTERLEAVED_STARS = ((0, 2, 4, 5), (2, 2, 1, 1), (-1, -1, 0, 1))
WORKER_MODES = {
    "slots": dict(),
    "forest": dict(pack_groups=True, repack_groups=True),
    "forest_dense": dict(pack_groups=True, repack_groups=True, pack_dense_fallbacks=True),
}


def rows(count, *, group="g", prompt=(1, 2, 3), completion=1, start=0) -> list[SharedPrefixRow]:
    """``count`` rows of one exact-prompt group with consecutive row indices."""
    return [SharedPrefixRow(start + index, group, prompt, completion) for index in range(count)]


def two_stars():
    """Stars ``a`` (rows 0-1, completions 2) and ``b`` (rows 2-3, completions 1), prompt 3."""
    a = build_shared_prefix_layout(rows(2, group="a", completion=2))
    return a, build_shared_prefix_layout(rows(2, group="b", completion=1, start=2))


def planned_star_unit() -> SharedPrefixExecutionUnit:
    """A three-row star planned through the public execution planner."""
    kwargs = dict(row_slots=((0, 1, 2),), bin_capacity=16, padding_multiple=1)
    (unit,) = plan_shared_prefix_execution_units(rows(3, group="a", completion=2), **kwargs)
    return unit


def fixed_slots(group_ids, lengths, capacity, multiple=1, **kwargs):
    kwargs |= dict(bin_capacity=capacity, sequence_length_pad_multiple=multiple)
    return plan_fixed_execution_slots(group_ids=list(group_ids), sequence_lengths=lengths, **kwargs)


def shards(group_ids, lengths, num_shards, **kwargs):
    kwargs |= dict(sequence_lengths=lengths, num_shards=num_shards)
    return plan_group_coherent_shards(group_ids=list(group_ids), **kwargs)


def prescribed_slots(group_ids, slot_ids):
    return get_prescribed_shared_prefix_slots(group_ids=tuple(group_ids), slot_ids=slot_ids)


def estimate(group_ids, lengths, prompts, **kwargs):
    kwargs |= dict(sequence_lengths=lengths, prompt_lengths=prompts)
    return estimate_shared_prefix_row_work(group_ids=list(group_ids), **kwargs)


def first_fit_decreasing(capacity):
    def pack_bins(costs):
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

    return pack_bins


def expanded_length(unit, costs):
    if unit.shared_layout is None:
        return sum(costs[row] for row in unit.row_indices)
    return sum(
        len(root.row_indices) * root.prompt_length + sum(root.physical_completion_lengths)
        for _, root in unit.shared_layout.iter_roots()
    )


def check_units(units, costs, *, capacity, multiple, expanded_budget=True):
    """Units cover every row once, fit the budget and execute exactly their layouts."""
    assert covered_rows(units) == list(range(len(costs)))
    for unit in units:
        assert _round_up(unit.physical_length, multiple) <= capacity
        if unit.shared_layout is not None:
            assert unit.shared_layout.row_indices == unit.row_indices
            assert unit.shared_layout.physical_total_length == unit.physical_length
        assert not expanded_budget or expanded_length(unit, costs) <= capacity
    validate_shared_prefix_execution_units(tuple(units), batch_size=len(costs))


def check_slot_plan(slot_plan, group_ids, lengths, *, capacity, multiple, chunk_size):
    """Every group has K nonempty, in-budget slots of at most the branch limit."""
    for chunk, slots_per_group in enumerate(slot_plan.units_per_group_by_chunk):
        members = defaultdict(lambda: defaultdict(list))
        for row in range(chunk * chunk_size, (chunk + 1) * chunk_size):
            members[group_ids[row]][slot_plan.row_slot_ids[row]].append(row)
        for slots in members.values():
            assert sorted(slots) == list(range(slots_per_group))
            for slot_rows in slots.values():
                assert len(slot_rows) <= MAX_SHARED_PREFIX_BRANCHES
                assert sum(_round_up(lengths[row], multiple) for row in slot_rows) <= capacity


def slot_work(lengths, prefixes, *, physical_weight=4, expanded_weight=1):
    """Executed work of one slot: one prompt copy when its rows can share."""
    expanded = sum(lengths)
    shares = len(lengths) > 1 and len(set(prefixes)) == 1
    backbone = expanded - (len(lengths) - 1) * prefixes[0] if shares else expanded
    return physical_weight * backbone + expanded_weight * expanded


def _kind(layout):
    return "dense" if layout is None else getattr(layout, "mtp_loss_group_root_counts", "star")


def _summary(units):
    """``(rows, kind, physical length)`` per unit; kind is dense, star or a forest's MTP groups."""
    return [(unit.row_indices, _kind(unit.shared_layout), unit.physical_length) for unit in units]


def _plan(plan_rows, capacity, multiple=1, **kwargs):
    kwargs |= dict(bin_capacity=capacity, sequence_length_pad_multiple=multiple)
    plan = plan_shared_prefix_bins(plan_rows, **kwargs)
    bins = [layout.row_indices for layout in plan.shared_bins]
    lengths = [layout.physical_total_length for layout in plan.shared_bins]
    return dict(plan=plan, bins=bins, lengths=lengths, fallback=plan.fallback_row_indices)


def _order_invariant_plan():
    plan_rows = [row(0, completion=3), row(1, completion=8), row(2, completion=3)]
    forward = _plan(plan_rows, 10)
    return forward | {"reversed_equal": forward["plan"] == _plan(plan_rows[::-1], 10)["plan"]}


def _snapshot_row_plan():
    prompt = [10, 11, 12]
    snapshot = SharedPrefixRow(0, "group", prompt, 2)
    prompt.append(99)
    return {"prompt": snapshot.prompt_token_ids, **_plan([snapshot, row(1)], 20)}


def _forest(order, *args):
    """Pack stars ``a`` and ``b`` (``two_stars``) in ``order`` with ``pack(*args)``."""
    a, b = two_stars()
    packed = pack([a, b] if order == "ab" else [b, a], *args)
    merged_roots = getattr(packed[0], "roots", None) == (a, b)
    return {"packed": packed, "unmerged": packed == (a, b), "merged_roots": merged_roots}


def _evaluation_vs_training_budget():
    plan_rows = rows(4, group="group", prompt=tuple(range(8)), completion=2)
    kwargs = dict(row_slots=((0, 1), (2, 3)), bin_capacity=20, padding_multiple=1)
    kwargs |= dict(pack_groups=True, repack_groups=True, evaluation_packing=True)
    train = plan_shared_prefix_execution_units(plan_rows, **kwargs)
    evaluation = plan_shared_prefix_execution_units(plan_rows, forward_only=True, **kwargs)
    return {"train": _summary(train), "evaluation": _summary(evaluation)}


def _mismatch_and_singleton_units():
    plan_rows = [row(0, group="a", prompt=(1, 2)), row(1, group="a", prompt=(1, 3))]
    plan_rows.append(row(2, group="b", prompt=(5, 6)))
    kwargs = dict(row_slots=((0, 1), (2,)), bin_capacity=8, padding_multiple=4)
    plan = SharedPrefixExecutionPlan(plan_shared_prefix_execution_units(plan_rows, **kwargs))
    pickled = pickle.loads(pickle.dumps(plan))
    return {"units": _summary(plan.units), "pickled": pickled, "round_trip": pickled == plan}


def _merged_dense_fallbacks(capacity):
    """Star ``a`` (rows 0-1), ungrouped row 2 and rows 3-4 of a prompt-mismatched group ``b``."""
    specs = [(0, "a", (1, 2), 2), (1, "a", (1, 2), 2), (2, None, (5, 6), 2)]
    specs += [(3, "b", (7, 8), 1), (4, "b", (7, 9), 1)]
    merge_rows = [SharedPrefixRow(*spec) for spec in specs]
    kwargs = dict(row_slots=((0, 1), (2,), (3, 4)), bin_capacity=capacity, padding_multiple=1)
    kwargs |= dict(pack_groups=True, pack_dense_fallbacks=True, merge_dense_fallbacks=True)
    kwargs["dense_packer"] = lambda costs: [[0], [1, 2]]
    units = plan_shared_prefix_execution_units(merge_rows, **kwargs)
    costs = [merge_row.total_length for merge_row in merge_rows]
    return {"units": _summary(units), "expanded": [expanded_length(u, costs) for u in units]}


def _packer_called_twice():
    seen, kwargs = [], dict(costs=(4, 4), bin_capacity=8)
    units = plan_dense_training_bins(dense_packer=lambda c: seen.append(c) or [(1, 0)], **kwargs)
    return {"seen": seen, "units": _summary(units)}


def _aligned(units, lengths, capacity, target):
    kwargs = dict(padded_row_lengths=lengths, bin_capacity=capacity, target_count=target)
    return [(unit.row_indices, unit.source_unit) for unit in align_rows_to_count(units, **kwargs)]


def _star_unit(prompt):
    """A four-row star with completions of 2 as one execution unit."""
    layout = build_shared_prefix_layout(rows(4, group="a", prompt=prompt, completion=2))
    return SharedPrefixExecutionUnit(layout.row_indices, layout, layout.physical_total_length)


def _training_alignment():
    args = dict(costs=(4,) * 4, capacity=16, target_count=3)
    dense, count = materialize_alignment((_star_unit((1, 2)),), **args)
    shared, shared_count = align_training_units((_star_unit((1, 2)),), padding_multiple=1, **args)
    return {"counts": (count, shared_count), "dense": _summary(dense), "shared": _summary(shared)}


def _physical_alignment():
    args = dict(costs=(10,) * 4, capacity=20, target_count=2, padding_multiple=1)
    units, stats = align_physical_units((_star_unit(tuple(range(8))),), **args)
    return {"units": _summary(units), "rebuilt_units": stats["rebuilt_units"]}


def _ppo_singleton_groups():
    """PPO may sample one answer per prompt; no row may disappear."""
    group_ids, lengths = [f"ppo:{index}" for index in range(4)], [9, 8, 2, 1]
    return [fixed_slots(group_ids, lengths, 12, 4), shards(group_ids, lengths, 2)]


def _equalized_slots():
    group_ids, lengths = ["large"] * 4 + ["small"] * 4, [6] * 4 + [4] * 4
    plan = fixed_slots(group_ids, lengths, 10)
    check_slot_plan(plan, group_ids, lengths, capacity=10, multiple=1, chunk_size=8)
    return plan


def _clean_group_work():
    work = estimate("aabb", [5, 7, 4, 9], [2, 2, 3, 3])
    return {"work": work, "group_sums": [sum(work[:2]), sum(work[2:])]}


def _tree_summary(tree):
    """Every descriptor method's result, keyed by method name, plus per-node queries."""
    methods = "depth roots leaf_nodes node_pos_offset position_ids first_predecessors"
    summary = {name: getattr(tree, name)() for name in methods.split() + ["prev_token_index"]}
    nodes, paths = range(tree.num_nodes), map(tree.path_token_indices, tree.leaf_nodes())
    pairs = [(a, n) for a in nodes for n in nodes if tree.is_ancestor(a, n)]
    summary |= dict(ancestors=tuple(map(tree.ancestors, nodes)), ancestor_pairs=pairs)
    return summary | dict(tree=tree, padding=tree.padding_positions(), paths=tuple(paths))


def _tree_snapshot():
    starts, lengths, parents = [0, 2], [2, 1], [-1, 0]
    tree = PackedTreeLayout(starts, lengths, parents)
    starts[0], lengths[0], parents[1] = 99, 99, -1
    return {"tree": tree, "pickle_round_trip": pickle.loads(pickle.dumps(tree)) == tree}


def _concatenated_trees():
    nested, star_tree = nested_tree(), PackedTreeLayout.from_shared_prefix(2, (1, 2))
    tree = PackedTreeLayout.concat((nested, star_tree, nested))
    rebased = tuple(16 + index for index in nested.path_token_indices(2))
    positions = (*nested.position_ids(), *star_tree.position_ids(), *nested.position_ids())
    summary = _tree_summary(tree)
    checks = {"rebased_path": tree.path_token_indices(9) == rebased}
    return summary | checks | {"concatenated_positions": summary["position_ids"] == positions}


def _root_only_star_roots():
    trees = (PackedTreeLayout.from_shared_prefix(2, (1, 2)), PackedTreeLayout((0,), (3,), (-1,)))
    return tuple(PackedTreeLayout.concat(trees).iter_star_roots())


PLANNING_EXAMPLES = {
    "layout-gather-fanout-scatter": (
        lambda: build_shared_prefix_layout([row(4), row(7, completion=1), row(9, completion=3)]),
        {
            "row_indices, completion_lengths, total_length": ((4, 7, 9), (2, 1, 3), 9),
            "branch_starts": (3, 5, 6),
            "position_ids": (0, 1, 2, 3, 4, 3, 3, 4, 5),
            "token_gather_rows": (4, 4, 4, 4, 4, 7, 9, 9, 9),
            "token_gather_columns": (0, 1, 2, 3, 4, 3, 3, 4, 5),
            "completion_positions": (3, 4, 5, 6, 7, 8),
            "predecessor_positions": (2, 3, 2, 2, 6, 7),
            "completion_scatter_rows": (4, 4, 7, 9, 9, 9),
            "completion_scatter_columns": (2, 3, 2, 2, 3, 4),
        },
    ),
    "layout-physical-branch-padding": (
        lambda: build_shared_prefix_layout(
            [row(0), row(1, completion=4)], sequence_length_pad_multiple=4
        ),
        {
            "completion_lengths, physical_completion_lengths": ((2, 4), (5, 5)),
            "total_length, physical_total_length, branch_starts": (9, 13, (3, 8)),
            "position_ids": (0, 1, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7),
            "physical_padding_positions": (5, 6, 7, 12),
            "completion_positions": (3, 4, 8, 9, 10, 11),
            "predecessor_positions": (2, 3, 2, 8, 9, 10),
        },
    ),
    # A hand-built layout derives every index map from its rows.
    "layout-ctor-derives-index-maps": (
        lambda: SharedPrefixLayout("g", (1, 2), (0, 1), (1, 1), physical_completion_lengths=[2, 1]),
        {
            "physical_completion_lengths, total_length, physical_total_length": ((2, 1), 4, 5),
            "position_ids, completion_positions": ((0, 1, 2, 3, 2), (2, 4)),
            "predecessor_positions, physical_padding_positions": ((1, 1), (3,)),
            "token_gather_rows, token_gather_columns": ((0, 0, 0, 0, 1), (0, 1, 2, 3, 2)),
        },
    ),
    # The padded star needs 13 tokens: it falls back at capacity 12 and shares at 16.
    "plan-capacity-counts-branch-padding": (
        lambda: [_plan([row(0), row(1, completion=4)], c, 4) for c in (12, 16)],
        {"0.bins, 0.fallback": ([], (0, 1)), "1.bins, 1.lengths": ([(1, 0)], [13])},
    ),
    "plan-order-invariant-ffd": (
        _order_invariant_plan,
        {"reversed_equal, bins, fallback": (True, [(0, 2)], (1,))},
    ),
    "plan-split-at-token-and-branch-limits": (
        lambda: _plan([row(i) for i in range(5)], 9, max_completions_per_bin=3),
        {"bins, fallback": ([(0, 1, 2), (3, 4)], ())},
    ),
    "plan-ineligible-rows-fall-back": (
        lambda: _plan(
            [row(0, group=None), row(1, prompt=()), row(2, completion=0), row(3, group="singleton")]
            + [row(4, group="capacity", completion=5), row(5, group="capacity", completion=5)],
            8,
        ),
        {"bins, fallback": ([], (0, 1, 2, 3, 4, 5))},
    ),
    "plan-covers-noncontiguous-rows": (
        lambda: _plan(
            [row(7, group="a"), row(2, group="a"), row(9, group="b"), row(4, group=None)], 20
        ),
        {"bins, fallback": ([(2, 7)], (4, 9))},
    ),
    "row-snapshots-list-prompt": (_snapshot_row_plan, {"prompt, bins": ((10, 11, 12), [(0, 1)])}),
    # Backbones 7 + 5 fit 12 and expanded 10 + 8 fit 20; stable FFD by expanded cost puts a first.
    "forest-merges-stars-ffd": (
        lambda: _forest("ba", 12, 20),
        {
            "packed.len, merged_roots": (1, True),
            "packed.0.row_indices, packed.0.physical_total_length": ((0, 1, 2, 3), 12),
            "packed.0.mtp_loss_group_root_counts": (),
        },
    ),
    # 7 + 5 = 12 tokens pad to 16 under a multiple of 8.
    "forest-budgets-padded-backbone": (
        lambda: [_forest("ab", 16, None, 8), _forest("ab", 8, None, 8)],
        {"0.packed.len, 0.packed.0.physical_total_length, 1.unmerged": (1, 12, True)},
    ),
    # Merging needs 18 expanded tokens > 17; evaluation omits the expanded budget and merges.
    "forest-respects-expanded-budget": (
        lambda: [_forest("ab", 12, 17), _forest("ab", 12)],
        {"0.unmerged, 1.packed.len, 1.packed.0.row_indices": (True, 1, (0, 1, 2, 3))},
    ),
    "execution-eval-physical-vs-train-expanded": (
        _evaluation_vs_training_budget,
        {
            "train": [((0, 1), "star", 12), ((2, 3), "star", 12)],
            "evaluation": [((0, 1, 2, 3), "star", 16)],
        },
    ),
    "execution-mismatch-and-singleton-stay-dense": (
        _mismatch_and_singleton_units,
        {
            "units": [((0, 1), "dense", 8), ((2,), "dense", 4)],
            "round_trip, pickled.max_physical_length": (True, 8),
        },
    ),
    # The bin holding the ungrouped row would fit, but stays dense instead of raising.
    # Star (0, 1) absorbs the dense bin (3, 4) as one MTP normalization group.
    "execution-merge-dense-fallbacks-cap20": (
        lambda: _merged_dense_fallbacks(20),
        {"units": [((2,), "dense", 4), ((0, 1, 3, 4), (1, 2), 12)], "expanded": [4, 14]},
    ),
    # Absorbing (3, 4) would need 14 expanded tokens > 12, so nothing merges.
    "execution-merge-dense-fallbacks-cap12": (
        lambda: _merged_dense_fallbacks(12),
        {"units": [((2,), "dense", 4), ((3, 4), "dense", 6), ((0, 1), "star", 6)]},
    ),
    # Called twice to enforce determinism.
    "dense-bins-packer-called-twice": (
        _packer_called_twice,
        {"seen": [[4, 4]] * 2, "units": [((1, 0), "dense", 8)]},
    ),
    # A packer may sort its own copy of the costs without failing the determinism check.
    "dense-bins-packer-may-sort-its-copy": (
        lambda: plan_dense_training_bins(
            costs=(4, 3), bin_capacity=8, dense_packer=lambda c: c.sort() or [[0, 1]]
        ),
        {"len": 1},
    ),
    "align-rows-equal-count-keeps-units": (
        lambda: _aligned(((2, 0), (1, 3)), (8, 16, 24, 32), 64, 2),
        [((2, 0), 0), ((1, 3), 1)],
    ),
    "align-rows-split-units-become-dense": (
        lambda: _aligned(((0, 1, 2, 3), (4,)), (40, 8, 8, 8, 8), 64, 3),
        [((0,), None), ((1, 2, 3), None), ((4,), 1)],
    ),
    "align-rows-to-singletons-in-order": (
        lambda: _aligned(((3, 1, 0), (2, 4)), (1,) * 5, 8, 5),
        [((3,), None), ((1,), None), ((0,), None), ((2,), None), ((4,), None)],
    ),
    # Shared alignment keeps the conservative dense cuts, singletons stay dense and (2, 3) shares.
    "align-training-keeps-conservative-cuts": (
        _training_alignment,
        {
            "counts": (3, 3),
            "dense": [((0,), "dense", 4), ((1,), "dense", 4), ((2, 3), "dense", 8)],
            "shared": [((0,), "dense", 4), ((1,), "dense", 4), ((2, 3), "star", 6)],
        },
    ),
    "align-physical-rebuilds-real-subsets": (
        _physical_alignment,
        {"units": [((0, 1), "star", 12), ((2, 3), "star", 12)], "rebuilt_units": 2},
    ),
    "metadata-prescribed-slots-order": (
        lambda: prescribed_slots("abab", (0, 0, 1, 1)),
        ((0,), (1,), (2,), (3,)),
    ),
    "metadata-ppo-singleton-groups": (
        _ppo_singleton_groups,
        {
            "0.row_slot_ids, 0.units_per_group_by_chunk": ((0, 0, 0, 0), (1,)),
            "1.shard_indices, 1.rank_order_permutation": (((0, 3), (1, 2)), (0, 3, 1, 2)),
        },
    ),
    "metadata-shards-balance-whole-groups": (
        lambda: shards("aabbccdd", [9, 9, 8, 8, 2, 2, 1, 1], 2),
        {
            "shard_indices": ((0, 1, 6, 7), (2, 3, 4, 5)),
            "rank_order_permutation": (0, 1, 6, 7, 2, 3, 4, 5),
        },
    ),
    "metadata-slots-equalize-real-units": (_equalized_slots, {"units_per_group_by_chunk": (4,)}),
    "metadata-slots-per-global-batch": (
        lambda: fixed_slots("aabbccdd", [6, 6, 6, 6, 4, 4, 4, 4], 10, batch_size=4),
        {"units_per_group_by_chunk, row_slot_ids": ((2, 1), (0, 1, 0, 1, 0, 0, 0, 0))},
    ),
    "metadata-shards-keep-batch-boundaries": (
        lambda: shards("aabbccdd", [9, 9, 8, 8, 2, 2, 1, 1], 2, batch_size=4),
        {
            "shard_indices": ((0, 1, 4, 5), (2, 3, 6, 7)),
            "rank_order_permutation": (0, 1, 4, 5, 2, 3, 6, 7),
        },
    ),
    # Group size 2 scales every weight: 2 * (4 * star backbone + 1 * expanded).
    "cost-clean-groups": (
        _clean_group_work,
        {
            "work": [42, 62, 28, 78],
            "group_sums": [2 * slot_work([5, 7], [2, 2]), 2 * slot_work([4, 9], [3, 3])],
        },
    ),
    "cost-prompt-mismatch-is-dense": (
        lambda: estimate("aa", [5, 7], [2, 3]),
        [2 * 5 * 5, 2 * 5 * 7],
    ),
    "cost-singleton-groups-no-credit": (lambda: estimate("ab", [5, 9], [2, 4]), [5 * 5, 5 * 9]),
    "tree-nested-descriptors": (
        lambda: _tree_summary(nested_tree()),
        {
            "depth, roots, leaf_nodes, ancestors.3": (2, (0,), (2, 3), (0, 1)),
            "ancestor_pairs": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)],
            "node_pos_offset, first_predecessors, padding": ((0, 2, 4, 4), (-1, 1, 3, 3), (4, 5)),
            "position_ids": (0, 1, 2, 3, 4, 5, 4, 5, 4, 5, 6),
            "prev_token_index": (-1, 0, 1, 2, 3, 4, 3, 6, 3, 8, 9),
            "paths": ((0, 1, 2, 3, 6, 7), (0, 1, 2, 3, 8, 9, 10)),
            "tree.total_len, tree.logical_total_len": (11, 9),
        },
    ),
    "tree-snapshots-inputs-and-pickles": (
        _tree_snapshot,
        {
            "tree.node_start, tree.node_len, tree.node_parent": ((0, 2), (2, 1), (-1, 0)),
            "pickle_round_trip": True,
        },
    ),
    "tree-concat-rebases-links": (
        _concatenated_trees,
        {
            "roots, ancestors.10": ((0, 4, 7), (7, 8)),
            "tree.node_parent": (-1, 0, 1, 1, -1, 4, 4, -1, 7, 8, 8),
            "tree.node_start": (0, 2, 6, 8, 11, 13, 14, 16, 18, 22, 24),
            # Token 22 continues from the real end (19) of the intermediate, not its padding.
            "prev_token_index.16, prev_token_index.22": (-1, 19),
            "rebased_path, concatenated_positions": (True, True),
        },
    ),
    "tree-star-lowering-keeps-root-only": (_root_only_star_roots, ((0, (1, 2)), (3, ()))),
    # Interleaved stars remain a valid generic tree descriptor.
    "tree-legacy-layouts-stay-valid": (
        lambda: PackedTreeLayout(*_INTERLEAVED_STARS).path_token_indices(2),
        (0, 1, 4),
    ),
}


@pytest.mark.parametrize("build, expected", PLANNING_EXAMPLES.values(), ids=list(PLANNING_EXAMPLES))
def test_planning_examples(build, expected):
    assert_observed(build(), expected)


# Invalid inputs. Each area lists ``(name, call, match[, error])``; ids are ``<area>-<name>``,
# so ``-k tree-`` selects one area. Calls are lambdas or partials: nothing runs at collection.

_BATCH = dict(input_ids=[[2, 3, 4, 5], [2, 3, 6, 7]], lengths=[4, 4], prompts=[2, 2])
_pad_slots = partial(fixed_slots, "aa", (3, 4), 16)


def _pair(**second):
    return build_shared_prefix_layout([row(0), row(1, **second)])


def _pair_plan(capacity=20, multiple=1, branches=MAX_SHARED_PREFIX_BRANCHES):
    return _plan(
        [row(0), row(1, completion=4)], capacity, multiple, max_completions_per_bin=branches
    )


def _layout(rows=(0, 1), completions=(1, 1), physical=(), prompt=(1, 2), group="g"):
    return SharedPrefixLayout(group, prompt, rows, completions, physical)


def _align(units, target, costs=(8, 8), capacity=16):
    kwargs = dict(padded_row_lengths=costs, bin_capacity=capacity, target_count=target)
    return align_rows_to_count(units, **kwargs)


def _estimate(**fields):
    return estimate(**dict(group_ids="aaa", lengths=[3] * 3, prompts=[1] * 3) | fields)


def _validate_star(**changes):
    unit = planned_star_unit()  # three rows, prompt 3, completions 2: physical length 9
    assert unit.shared_layout is not None
    validate_shared_prefix_execution_units((replace(unit, **changes),), batch_size=3)


def _slots(row_slots, plan_rows=None, **kwargs):
    """Plan ``plan_rows`` (default: one two-row group) into prescribed slots of 8 tokens."""
    plan_rows = plan_rows or rows(2, group="a", prompt=(1, 2), completion=2)
    kwargs |= dict(row_slots=row_slots, bin_capacity=8, padding_multiple=1)
    return plan_shared_prefix_execution_units(plan_rows, **kwargs)


def _dense_singletons(count, completion, bins):
    """``count`` single-row groups through group packing and a dense packer returning ``bins``."""
    singletons = [row(i, group=str(i), prompt=(1, 2), completion=completion) for i in range(count)]
    options = dict(pack_groups=True, pack_dense_fallbacks=True, dense_packer=lambda costs: bins)
    return _slots(tuple((i,) for i in range(count)), singletons, **options)


def _unstable_dense_bins():
    orders, kwargs = itertools.cycle(([[0], [1]], [[1], [0]])), dict(costs=(4, 4), bin_capacity=8)
    return plan_dense_training_bins(dense_packer=lambda c: next(orders), **kwargs)


def _materialize(completions, row0=(1, 2, 3), row1=(1, 2, 4)):
    """Materialize a prompt-(1, 2) star with ``completions`` from a batch that may disagree."""
    star_rows = [row(i, group="g", prompt=(1, 2), completion=c) for i, c in enumerate(completions)]
    input_ids = torch.tensor([list(tokens) + [0] * (5 - len(tokens)) for tokens in (row0, row1)])
    lengths, layout = torch.tensor([len(row0), len(row1)]), build_shared_prefix_layout(star_rows)
    return materialize_shared_prefix_layout(input_ids, input_lengths=lengths, layout=layout)


def _batch_rows(group_ids="gg", **fields):
    """Rows of the valid two-row ``_BATCH`` with ``fields`` replaced."""
    batch = {name: torch.as_tensor(value) for name, value in (_BATCH | fields).items()}
    return batch_rows(group_ids=group_ids, **batch)


def _with_dtype(field, dtype):
    """``_BATCH`` rows with ``field`` cast to ``dtype``; float values are made non-integral."""
    field = {"input_lengths": "lengths", "prompt_lengths": "prompts"}.get(field, field)
    value = torch.tensor(_BATCH[field]).to(getattr(torch, dtype))
    return _batch_rows(**{field: value + 0.25 if dtype == "float32" else value})


def _topology(tp, cp, sp):
    return resolve_shared_prefix_parallel_topology(tp_size=tp, cp_size=cp, sequence_parallel=sp)


def _padding_multiple(raw):
    kwargs = dict(tp_size=2, cp_size=2, padding_multiple=raw)
    return resolve_shared_prefix_physical_padding_multiple(**kwargs)


def _cp_indices(length, rank, size):
    return get_shared_prefix_context_parallel_indices(length, cp_rank=rank, cp_size=size)


PACKING = [
    ("layout-singleton", lambda: build_shared_prefix_layout([row(0)]), "at least two"),
    ("layout-mixed-group", lambda: _pair(group="other"), "same group_id"),
    ("layout-prompt-mismatch", lambda: _pair(prompt=(10, 99, 12)), "identical prompt"),
    ("layout-empty-completion", lambda: _pair(completion=0), "non-empty completion"),
    ("plan-duplicate-row", lambda: _plan([row(0), row(0)], 20), "row_index values must be unique"),
    ("plan-capacity-zero", lambda: _pair_plan(0), "bin_capacity must be a positive integer"),
    ("plan-capacity-float", lambda: _pair_plan(20.0), "bin_capacity must be a positive integer"),
    ("plan-capacity-bool", lambda: _pair_plan(True), "bin_capacity must be a positive integer"),
    ("plan-branches-one", lambda: _pair_plan(branches=1), "must be an integer of at least 2"),
    ("plan-branches-float", lambda: _pair_plan(branches=2.0), "must be an integer of at least 2"),
    # The 13-token star padded to 4 would occupy 16 tokens, past a 13-token budget.
    ("plan-capacity-unaligned", lambda: _pair_plan(13, 4), "multiple of sequence_length_pad"),
    ("row-index-bool", lambda: row(True), "row_index must be a nonnegative integer"),
    ("row-index-negative", lambda: row(-1), "row_index must be a nonnegative integer"),
    ("row-completion-float", lambda: row(0, completion=2.0), "completion_length must be"),
    ("row-group-empty", lambda: row(0, group=""), "non-empty string"),
    ("row-group-int", lambda: row(0, group=7), "non-empty string"),
    ("row-prompt-bool", lambda: row(0, prompt=(1, True)), "prompt_token_ids must contain integers"),
    ("row-prompt-float", lambda: row(0, prompt=(1, 2.0)), "prompt_token_ids must contain integers"),
    # An unaligned budget could admit a forest that topology padding overflows.
    ("forest-unaligned-budget", lambda: pack(two_stars(), 12, None, 8), "aligned to the padding"),
    ("forest-star-over-budget", lambda: pack(two_stars()[:1], 6), "aligned backbone budget"),
    (
        "layout-ctor-logical-over-physical",
        lambda: _layout(completions=(2, 1), physical=(1, 1)),
        "no longer than storage",
    ),
    ("layout-ctor-physical-zero", lambda: _layout(physical=(0, 1)), "positive and contiguous"),
    ("layout-ctor-physical-count", lambda: _layout(physical=(1,)), "must have equal length"),
    ("layout-ctor-row-count", lambda: _layout(rows=(0,)), "node must own one source row"),
    ("layout-ctor-row-duplicate", lambda: _layout(rows=(1, 1)), "unique nonnegative integers"),
    ("layout-ctor-row-bool", lambda: _layout(rows=(0, True)), "unique nonnegative integers"),
    ("layout-ctor-completion-zero", lambda: _layout(completions=(1, 0)), "positive"),
    ("layout-ctor-empty-prompt", lambda: _layout(prompt=()), "non-empty prompt"),
    ("layout-ctor-empty-group", lambda: _layout(group=""), "non-empty group_id"),
]

ALIGNMENT = [
    ("duplicate-row", lambda: _align(((0, 0), (1,)), 2), None),
    ("missing-row", lambda: _align(((0,),), 1), None),
    ("unknown-row", lambda: _align(((0,), (1,), (2,)), 3), None),
    ("bool-row", lambda: _align(((True,), (1,)), 2), None),
    ("empty-unit", lambda: _align(((), (0, 1)), 2), None),
    ("target-zero", lambda: _align(((0, 1),), 0), None),
    ("target-over-rows", lambda: _align(((0, 1),), 3), None),
    ("target-bool", lambda: _align(((0, 1),), True), None),
    ("unit-over-capacity", lambda: _align(((0, 1),), 2, (16, 16)), None),
    ("cost-zero", lambda: _align(((0, 1),), 2, (0, 8)), None),
    ("cost-float", lambda: _align(((0, 1),), 2, (1.5, 8)), None),
    ("capacity-zero", lambda: _align(((0, 1),), 2, (8, 8), 0), None),
]

COST = [
    ("unequal-groups", lambda: _estimate(group_ids="aab"), "equal-size"),
    ("slot-count", lambda: _estimate(row_slot_ids=[0]), "one entry per row"),
    ("slot-negative", lambda: _estimate(row_slot_ids=[0, -1, 0]), "nonnegative integers"),
    ("empty-group-id", lambda: _estimate(group_ids=["a", "", "a"]), "prompt-group identity"),
]

EXECUTION = [
    ("validate-reordered-rows", lambda: _validate_star(row_indices=(2, 0, 1)), "in layout order"),
    ("validate-padded-length", lambda: _validate_star(physical_length=13), "length must equal"),
    ("slots-lost", lambda: _slots(((0,),)), "every source row exactly once"),
    ("slots-duplicated", lambda: _slots(((0, 0),)), "every source row exactly once"),
    ("slots-empty", lambda: _slots(((0, 1), ())), "every source row exactly once"),
    ("slots-unknown", lambda: _slots(((0, 2),)), "every source row exactly once"),
    ("dense-packer-drops-row", lambda: _dense_singletons(2, 2, [[0]]), "every source row exactly"),
    ("dense-packer-over-budget", lambda: _dense_singletons(4, 6, [[0, 1, 2, 3]]), "token budget"),
]

DENSE_BINS = [("unstable-packer", _unstable_dense_bins, "deterministic")]

METADATA = [
    ("prescribed-slots-uneven", lambda: prescribed_slots("aab", (0, 1, 0)), "same number"),
    ("slots-singleton-over-capacity", lambda: fixed_slots(["ppo:0"], [9], 10, 4), "execution bin"),
    ("slots-pad-none", lambda: _pad_slots(None), "pad_multiple must be a positive integer"),
    ("slots-pad-zero", lambda: _pad_slots(0), "pad_multiple must be a positive integer"),
    ("slots-pad-float", lambda: _pad_slots(2.0), "pad_multiple must be a positive integer"),
    ("slots-pad-bool", lambda: _pad_slots(True), "pad_multiple must be a positive integer"),
    ("slots-padded-rows-over-capacity", lambda: fixed_slots("aa", (9, 8), 10, 4), "execution bin"),
    ("shards-group-crosses-batch", lambda: shards("abab", [1] * 4, 1, batch_size=2), "not cross"),
    ("shards-unequal-groups", lambda: shards("aab", [1] * 3, 1), "equal-size"),
    ("shards-non-integral", lambda: shards("aabbcc", [1] * 6, 2), "integral number"),
]

TENSORS = [
    ("topology-tp-bool", lambda: _topology(True, 1, False), None),
    ("topology-tp-float", lambda: _topology(4.0, 1, True), None),
    ("topology-tp-str", lambda: _topology("4", 1, True), None),
    ("topology-cp-bool", lambda: _topology(1, False, False), None),
    ("topology-cp-float", lambda: _topology(1, 2.0, False), None),
    ("topology-cp-str", lambda: _topology(1, "2", False), None),
    ("topology-sp-int", lambda: _topology(1, 1, 0), None),
    ("topology-sp-str", lambda: _topology(1, 1, "false"), None),
    ("padding-multiple-true", lambda: _padding_multiple(True), "padding_multiple"),
    ("padding-multiple-false", lambda: _padding_multiple(False), "padding_multiple"),
    ("padding-multiple-zero", lambda: _padding_multiple(0), "padding_multiple"),
    ("padding-multiple-negative", lambda: _padding_multiple(-8), "padding_multiple"),
    ("padding-multiple-float", lambda: _padding_multiple(8.0), "padding_multiple"),
    ("padding-multiple-str", lambda: _padding_multiple("8"), "padding_multiple"),
    ("padding-multiple-not-topology-aligned", lambda: _padding_multiple(12), "alignment 8"),
    ("cp-indices-size-zero", lambda: _cp_indices(8, 0, 0), "cp_size must be positive"),
    ("cp-indices-rank-out-of-range", lambda: _cp_indices(8, 2, 2), "cp_rank must be in"),
    ("cp-indices-length-not-divisible", lambda: _cp_indices(6, 0, 2), r"divisible by 2 \* cp_size"),
    ("materialize-stale-prompt", lambda: _materialize((1, 1), row1=(1, 9, 4)), "exact prompt"),
    ("materialize-row-shorter-than-layout", lambda: _materialize((2, 1)), "length differs"),
    # A stale layout must not silently drop the source row's extra completion tokens.
    (
        "materialize-row-longer-than-layout",
        lambda: _materialize((2, 1), (1, 2, 7, 8, 9)),
        "completion length differs from source row",
    ),
    ("rows-1d-input", lambda: _batch_rows(input_ids=[1, 2, 3]), "input_ids must have shape"),
    # Integral float tokens must still be rejected by dtype, not by value.
    (
        "rows-input_ids-float32-integral",
        lambda: _batch_rows(input_ids=torch.zeros((2, 4), dtype=torch.float32)),
        "integer dtype",
    ),
    ("rows-length-over-width", lambda: _batch_rows(lengths=[4, 5]), "input_lengths must be within"),
    ("rows-prompt-over-length", lambda: _batch_rows(lengths=[4, 1]), "prompt_lengths must satisfy"),
    ("rows-group-count", lambda: _batch_rows("g"), "group_ids must have 2 entries"),
    *[
        (f"rows-dtype-{field}-{dtype}", partial(_with_dtype, field, dtype), "integer dtype")
        for field in ("input_ids", "input_lengths", "prompt_lengths")
        for dtype in ("float32", "complex64", "bool")
    ],
]

TREE = [
    ("frozen", lambda: setattr(nested_tree(), "node_len", (99, 1)), None, FrozenInstanceError),
    *[
        (f"malformed-{name}", partial(PackedTreeLayout, *spec), None)
        for name, spec in (
            ("empty", ((), (), (), ())),
            ("parent-count", ((0,), (2,), (), ())),
            ("overlap", ((0, 1), (2, 1), (-1, 0), ())),
            ("gap", ((0, 3), (2, 1), (-1, 0), ())),
            ("zero-length", ((0,), (0,), (-1,), ())),
            ("self-parent", ((0, 2), (2, 1), (-1, 1), ())),
            ("forward-edge", ((0, 2), (2, 1), (1, 0), ())),
            ("parent-below-root", ((0,), (2,), (-2,), ())),
            ("length-bool", ((0,), (True,), (-1,), ())),
            ("parent-float", ((0,), (2,), (-1.0,), ())),
            ("logical-over-physical", ((0,), (2,), (-1,), (3,))),
            ("logical-zero", ((0,), (2,), (-1,), (0,))),
            ("logical-count", ((0, 2), (2, 1), (-1, 0), (1,))),
        )
    ],
    ("node-query-minus-one", lambda: nested_tree().path_token_indices(-1), "node index"),
    ("node-query-past-end", lambda: nested_tree().path_token_indices(4), "node index"),
    ("node-query-bool", lambda: nested_tree().path_token_indices(True), "node index"),
    ("node-query-float", lambda: nested_tree().path_token_indices(1.0), "node index"),
    (
        "star-lowering-nested",
        lambda: tuple(nested_tree().iter_star_roots()),
        "forest of stars",
        NotImplementedError,
    ),
    (
        "star-lowering-interleaved",
        lambda: tuple(PackedTreeLayout(*_INTERLEAVED_STARS).iter_star_roots()),
        "contiguous star storage",
        NotImplementedError,
    ),
    (
        "star-lowering-padded-root",
        lambda: tuple(PackedTreeLayout((0, 4), (4, 1), (-1, 0), (2, 1)).iter_star_roots()),
        "unpadded prompt roots",
        NotImplementedError,
    ),
]

AREAS = {
    "packing": PACKING,
    "alignment": ALIGNMENT,
    "cost": COST,
    "execution": EXECUTION,
    "dense-bins": DENSE_BINS,
    "metadata": METADATA,
    "tensors": TENSORS,
    "tree": TREE,
}


@pytest.mark.parametrize(
    "call, match, error",
    [
        pytest.param(call, match, *error or [ValueError], id=f"{area}-{name}")
        for area, cases in AREAS.items()
        for name, call, match, *error in cases
    ],
)
def test_rejects_invalid_inputs(call, match, error):
    with pytest.raises(error, match=match):
        call()


def _split_rows_evenly(count):
    source = rows(count, completion=2)
    chunks = split_rows_evenly(source, max_completions_per_bin=MAX_SHARED_PREFIX_BRANCHES)
    flat = [chunk_row for chunk in chunks for chunk_row in chunk]
    sizes, indices = tuple(map(len, chunks)), [chunk_row.row_index for chunk_row in flat]
    return dict(sizes=sizes, rows=indices, same=flat == source)


def _plan_shared_prefix_bins(count):
    plan = plan_shared_prefix_bins(rows(count), bin_capacity=1000)
    sizes = tuple(len(layout.row_indices) for layout in plan.shared_bins)
    return {"sizes": sizes, "rows": covered_rows(plan), "fallback": plan.fallback_row_indices}


def _execution_units(count, **kwargs):
    kwargs |= dict(row_slots=(tuple(range(count)),), bin_capacity=4 * count, padding_multiple=1)
    units = plan_shared_prefix_execution_units(rows(count), **kwargs)
    layouts = [unit.shared_layout for unit in units if unit.shared_layout is not None]
    sizes = tuple(len(root.row_indices) for layout in layouts for _, root in layout.iter_roots())
    return {"sizes": sizes, "rows": [i for unit in units for i in unit.row_indices], "units": units}


def _fixed_execution_slots(count):
    plan = fixed_slots("g" * count, [4] * count, 4 * count)
    (slots,) = plan.units_per_group_by_chunk
    sizes = tuple(plan.row_slot_ids.count(slot) for slot in range(slots))
    rows_in_slots = [index for index, slot in enumerate(plan.row_slot_ids) if slot < slots]
    return {"sizes": sizes, "rows": rows_in_slots, "groups": plan.units_per_group_by_chunk}


def _dense_training_bins(count):
    dense = (SharedPrefixExecutionUnit(tuple(range(count)), None, 4 * count),)
    kwargs = dict(costs=(4,) * count, padding_multiple=1, bin_capacity=4 * count)
    (unit,) = units = share_prefixes_in_dense_training_bins(rows(count), dense, **kwargs)
    layout = unit.shared_layout
    sizes = tuple(len(root.row_indices) for root in layout.roots)
    observed = dict(groups=layout.mtp_loss_group_root_counts, length=unit.physical_length)
    return {"sizes": sizes, "rows": list(unit.row_indices), "units": units, **observed}


BRANCH_LIMIT_SIZES = {0: (), 3: (3,), 16: (16,), 17: (9, 8), 32: (16, 16), 33: (11, 11, 11)}
_COUNTS = (3, 16, 17, 32, 33)
BRANCH_LIMIT_PLANNERS = {
    "split_rows_evenly": (_split_rows_evenly, (0, *_COUNTS)),
    "plan_shared_prefix_bins": (_plan_shared_prefix_bins, (0, *_COUNTS)),
    "execution-slots": (partial(_execution_units, pack_groups=True), _COUNTS),
    "execution-repack": (partial(_execution_units, pack_groups=True, repack_groups=True), _COUNTS),
    "fixed-slots": (_fixed_execution_slots, _COUNTS),
    "dense-bins": (_dense_training_bins, _COUNTS),
    # Without group packing, a slot past the branch limit warns and stays one dense forward.
    "execution-unpacked": (_execution_units, _COUNTS),
    # Raising the limit keeps the slot as one shared star.
    "execution-unpacked-limit17": (partial(_execution_units, max_completions_per_bin=17), (17,)),
}


def _branch_limit_case(planner, count):
    sizes = BRANCH_LIMIT_SIZES[count]
    if planner == "execution-unpacked-limit17":
        sizes = (count,)
    elif planner == "execution-unpacked" and count > MAX_SHARED_PREFIX_BRANCHES:
        sizes = ()
    return pytest.param(planner, count, sizes, id=f"{planner}-{count}")


@pytest.mark.parametrize(
    "planner, count, sizes",
    [
        _branch_limit_case(planner, count)
        for planner, (_, counts) in BRANCH_LIMIT_PLANNERS.items()
        for count in counts
    ],
)
def test_branch_limit_splits_evenly(planner, count, sizes, recwarn):
    """Every planner splits a group past the branch limit into near-equal shared chunks.

    Each planner reports what it can observe; every report is checked against the same
    expectation. A dense bin stores one 3-token prompt per root plus one token per completion.
    """
    result = BRANCH_LIMIT_PLANNERS[planner][0](count)
    dense = planner == "execution-unpacked" and count > MAX_SHARED_PREFIX_BRANCHES
    warned = [w.category for w in recwarn if "exceeds max_completions_per_bin=16" in str(w.message)]
    assert warned == [UserWarning] * dense
    expected = dict(sizes=sizes, rows=list(range(count)), same=True, fallback=())
    expected |= dict(groups=(len(sizes),), length=3 * len(sizes) + count)
    assert_observed(result, {key: value for key, value in expected.items() if key in result})
    if "units" in result:
        check_units(result["units"], [4] * count, capacity=4 * count, multiple=1)
        if not sizes:
            assert _summary(result["units"]) == [(tuple(range(count)), "dense", 4 * count)]


def _random_batch(rng):
    multiple, dp = rng.choice([1, 2, 4, 8]), rng.choice([1, 2, 4])
    group_size = rng.choice([1, 2, 3, 4, 8, 17])
    groups_per_chunk, num_chunks = rng.randint(1, 3) * dp, rng.randint(1, 2)
    batch = []
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
        batch.extend(chunk_rows)
    max_padded = max(_round_up(len(prompt) + length, multiple) for _, prompt, length in batch)
    capacity = _round_up(max_padded * rng.choice([1, 2, 3]), multiple)
    return batch, multiple, dp, groups_per_chunk * group_size, capacity


def _reorder_data(values, permutation):
    """NeMo-RL ``BatchedDataDict.reorder_data``: output position -> source row."""
    return [values[source] for _, source in sorted(zip(permutation, range(len(values))))]


def _pipeline(rng):
    """Shard, slot, plan and align random multi-chunk batches like the NeMo-RL driver."""
    for iteration in range(25):
        batch, multiple, dp, chunk_size, capacity = _random_batch(rng)
        limits = dict(capacity=capacity, multiple=multiple)
        group_ids = [group for group, _, _ in batch]
        lengths = [len(prompt) + completion for _, prompt, completion in batch]
        with note(iteration=iteration, stage="shard-and-slot"):
            slot_plan = fixed_slots(group_ids, lengths, capacity, multiple, batch_size=chunk_size)
            shard_plan = shards(group_ids, lengths, dp, batch_size=chunk_size)
            flat = [row for shard in shard_plan.shard_indices for row in shard]
            if shard_plan.rank_order_permutation is not None:
                flat = _reorder_data(flat, shard_plan.rank_order_permutation)
            assert flat == list(range(len(batch)))
            check_slot_plan(slot_plan, group_ids, lengths, chunk_size=chunk_size, **limits)

        size = chunk_size // dp
        for chunk in range(len(batch) // chunk_size):
            locals_ = [s[chunk * size : (chunk + 1) * size] for s in shard_plan.shard_indices]
            with note(iteration=iteration, chunk=chunk, stage="ranks"):
                # Ranks keep global-batch boundaries and never split a group.
                assert all(row // chunk_size == chunk for local in locals_ for row in local)
                owner = {group_ids[r]: rank for rank, local in enumerate(locals_) for r in local}
                assert all(owner[group_ids[r]] == k for k, rs in enumerate(locals_) for r in rs)
            ranks = []
            for local in locals_:
                local_rows = [SharedPrefixRow(i, *batch[row]) for i, row in enumerate(local)]
                slot_ids = [slot_plan.row_slot_ids[row] for row in local]
                row_slots = prescribed_slots([r.group_id for r in local_rows], slot_ids)
                costs = [_round_up(local_row.total_length, multiple) for local_row in local_rows]
                ranks.append((local_rows, row_slots, costs))
            for mode, options in WORKER_MODES.items():
                for forward_only, evaluation in ((False, False), (True, False), (True, True)):
                    if evaluation and not options:
                        continue
                    context = dict(mode=mode, forward_only=forward_only, evaluation=evaluation)
                    with note(iteration=iteration, chunk=chunk, **context):
                        _check_rank_plans(ranks, options, forward_only, evaluation, **limits)
            with note(iteration=iteration, chunk=chunk, mode="dense-bins"):
                _check_dense_bins(ranks, **limits)


def _check_rank_plans(ranks, options, forward_only, evaluation, *, capacity, multiple):
    physical_only = forward_only and evaluation
    limits = dict(capacity=capacity, multiple=multiple, expanded_budget=not physical_only)
    kwargs = dict(bin_capacity=capacity, padding_multiple=multiple, forward_only=forward_only)
    kwargs |= dict(evaluation_packing=evaluation, dense_packer=first_fit_decreasing(capacity))
    plans = []
    for local_rows, slots, costs in ranks:
        units = plan_shared_prefix_execution_units(local_rows, row_slots=slots, **kwargs, **options)
        check_units(units, costs, **limits)
        plans.append((units, costs))
    counts = [len(units) for units, _ in plans]
    if not options:
        # Driver slots alone give every DP rank the same forward count.
        assert len(set(counts)) == 1
        return
    target = max(counts)
    if target > min(len(costs) for _, costs in plans):
        return
    for units, costs in plans:
        budget = dict(costs=costs, capacity=capacity, target_count=target)
        if physical_only:
            aligned, _ = align_physical_units(units, padding_multiple=multiple, **budget)
        elif forward_only:
            aligned, _ = materialize_alignment(units, **budget)
        else:
            aligned, _ = align_training_units(units, padding_multiple=multiple, **budget)
        assert len(aligned) == target
        check_units(aligned, costs, **limits)


def _check_dense_bins(ranks, *, capacity, multiple):
    packer = dict(bin_capacity=capacity, dense_packer=first_fit_decreasing(capacity))
    plans = [(rs, costs, plan_dense_training_bins(costs=costs, **packer)) for rs, _, costs in ranks]
    target = max(len(units) for _, _, units in plans)
    if target > min(len(costs) for _, costs, _ in plans):
        return
    for local_rows, costs, units in plans:
        budget = dict(costs=costs, capacity=capacity, target_count=target)
        aligned, _ = materialize_alignment(units, **budget)
        kwargs = dict(costs=costs, padding_multiple=multiple, bin_capacity=capacity)
        shared = share_prefixes_in_dense_training_bins(local_rows, aligned, **kwargs)
        assert len(shared) == target
        check_units(shared, costs, capacity=capacity, multiple=multiple)
        for dense, unit in zip(aligned, shared):
            assert set(dense.row_indices) == set(unit.row_indices)
            assert expanded_length(unit, costs) == dense.physical_length
            if isinstance(unit.shared_layout, SharedPrefixForestLayout):
                roots = unit.shared_layout.roots
                assert unit.shared_layout.mtp_loss_group_root_counts == (len(roots),)
                assert all(len(root.row_indices) <= MAX_SHARED_PREFIX_BRANCHES for root in roots)


def _align_rows(rng):
    """Eight ranks with random first-fit cuts align to the largest count without copying rows."""
    for iteration in range(100):
        ranks = []
        for _rank in range(8):
            costs = tuple(8 * rng.randint(1, 16) for _ in range(32))
            order = list(range(32))
            rng.shuffle(order)
            units, used = [[]], 0
            for index in order:
                if used + costs[index] > 512:
                    units.append([])
                    used = 0
                units[-1].append(index)
                used += costs[index]
            ranks.append((costs, units))
        target = max(len(units) for _, units in ranks)
        for rank, (costs, units) in enumerate(ranks):
            with note(iteration=iteration, rank=rank, target=target):
                kwargs = dict(padded_row_lengths=costs, bin_capacity=512, target_count=target)
                result = align_rows_to_count(units, **kwargs)
                assert len(result) == target
                flat = [index for unit in result for index in unit.row_indices]
                assert flat == [index for unit in units for index in unit]
                assert sorted(flat) == list(range(32))
                for unit in result:
                    assert sum(costs[index] for index in unit.row_indices) <= 512
                    if unit.source_unit is not None:
                        assert unit.row_indices == tuple(units[unit.source_unit])


def _row_work(rng):
    """Row weights sum, per (group, slot), to that slot's executed work times a common scale."""
    for iteration in range(200):
        group_size = rng.randint(1, 6)
        groups = [f"g{index}" for index in range(rng.randint(1, 4)) for _ in range(group_size)]
        prompts, lengths, slots = [], [], []
        for group in dict.fromkeys(groups):
            prompt = rng.randint(0, 5)
            num_slots = rng.randint(1, group_size)
            for index in range(group_size):
                prompts.append(prompt if rng.random() > 0.1 else rng.randint(0, 5))
                lengths.append(prompts[-1] + rng.randint(0, 9))
                slots.append(index % num_slots)
        with note(iteration=iteration, groups=groups, slots=slots):
            work = estimate(groups, lengths, prompts, row_slot_ids=slots)
            assert all(value >= 0 for value in work)
            members = defaultdict(list)
            for index, key in enumerate(zip(groups, slots)):
                members[key].append(index)
            scale = math.lcm(*(len(indices) for indices in members.values()))
            for indices in members.values():
                expected = slot_work([lengths[i] for i in indices], [prompts[i] for i in indices])
                assert sum(work[i] for i in indices) == scale * expected


PLANNING_PROPERTIES = {f"pipeline-seed{seed}": (_pipeline, seed) for seed in range(4)}
PLANNING_PROPERTIES["align_rows_to_count-seed185"] = (_align_rows, 185)
PLANNING_PROPERTIES["row_work-seed0"] = (_row_work, 0)


@pytest.mark.parametrize("check, seed", PLANNING_PROPERTIES.values(), ids=list(PLANNING_PROPERTIES))
def test_planning_properties(check, seed):
    check(random.Random(seed))
