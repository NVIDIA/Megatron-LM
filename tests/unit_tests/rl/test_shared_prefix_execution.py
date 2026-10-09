# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Packing contracts independent of NeMo's batch/configuration classes."""

import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from megatron.rl.shared_prefix_alignment import (
    align_physical_units,
    align_training_units,
    materialize_alignment,
)
from megatron.rl.shared_prefix_dense_bins import (
    plan_dense_training_bins,
    share_prefixes_in_dense_training_bins,
)
from megatron.rl.shared_prefix_execution import (
    SharedPrefixExecutionPlan,
    SharedPrefixExecutionUnit,
    plan_shared_prefix_execution_units,
)
from megatron.rl.shared_prefix_metadata import get_prescribed_shared_prefix_slots
from megatron.rl.shared_prefix_packing import SharedPrefixRow, build_shared_prefix_layout
from megatron.rl.shared_prefix_tensors import (
    build_shared_prefix_rows,
    materialize_shared_prefix_layout,
)
from tests.unit_tests.rl.shared_prefix_oracles import build_star_attention_allow_mask


def test_evaluation_uses_physical_budget_training_keeps_expanded_mtp_budget():
    rows = [SharedPrefixRow(index, "group", tuple(range(8)), 2) for index in range(4)]
    kwargs = dict(
        row_slots=((0, 1), (2, 3)),
        bin_capacity=20,
        padding_multiple=1,
        pack_groups=True,
        repack_groups=True,
        evaluation_packing=True,
    )
    train = plan_shared_prefix_execution_units(rows, **kwargs)
    evaluation = plan_shared_prefix_execution_units(rows, forward_only=True, **kwargs)
    assert len(train) == 2 and len(evaluation) == 1
    assert [u.physical_length for u in train] == [12, 12]
    assert evaluation[0].physical_length == 16
    assert sorted(index for unit in train for index in unit.row_indices) == list(range(4))
    assert evaluation[0].row_indices == (0, 1, 2, 3)


def test_mismatch_and_single_completion_groups_retain_real_dense_rows():
    rows = [
        SharedPrefixRow(0, "a", (1, 2), 2),
        SharedPrefixRow(1, "a", (1, 3), 2),
        SharedPrefixRow(2, "b", (5, 6), 2),
    ]
    units = plan_shared_prefix_execution_units(
        rows, row_slots=((0, 1), (2,)), bin_capacity=8, padding_multiple=4
    )
    assert units == (
        SharedPrefixExecutionUnit((0, 1), None, 8),
        SharedPrefixExecutionUnit((2,), None, 4),
    )
    restored = pickle.loads(pickle.dumps(SharedPrefixExecutionPlan(units)))
    assert restored.units == units and restored.max_physical_length == 8


@pytest.mark.parametrize("slots", [((0,),), ((0, 0),), ((0, 1), ()), ((0, 2),)])
def test_public_planner_rejects_lost_duplicated_empty_or_unknown_rows(slots):
    rows = [SharedPrefixRow(index, "a", (1, 2), 2) for index in range(2)]
    with pytest.raises(ValueError, match="every source row exactly once"):
        plan_shared_prefix_execution_units(
            rows, row_slots=slots, bin_capacity=8, padding_multiple=1
        )


def test_length_only_callback_stays_independent_and_must_preserve_coverage():
    rows = [SharedPrefixRow(index, str(index), (1, 2), 2) for index in range(2)]
    with pytest.raises(ValueError, match="every source row exactly once"):
        plan_shared_prefix_execution_units(
            rows,
            row_slots=((0,), (1,)),
            bin_capacity=8,
            padding_multiple=1,
            pack_groups=True,
            pack_dense_fallbacks=True,
            dense_packer=lambda costs: ((0,),),
        )
    seen = []

    def pack(costs):
        seen.append(tuple(costs))
        return ((1, 0),)

    units = plan_dense_training_bins(costs=(4, 4), bin_capacity=8, dense_packer=pack)
    assert seen == [(4, 4), (4, 4)]  # called twice to enforce determinism
    assert units == (SharedPrefixExecutionUnit((1, 0), None, 8),)


def test_dense_bin_reconstruction_retains_one_mtp_group_and_causal_rows():
    inputs = torch.tensor([[1, 2, 3, 8], [4, 5, 6, 9], [1, 2, 3, 10]])
    rows = build_shared_prefix_rows(
        input_ids=inputs,
        input_lengths=torch.tensor([4, 4, 4]),
        prompt_lengths=torch.tensor([3, 3, 3]),
        group_ids=("a", "b", "a"),
    )
    (unit,) = share_prefixes_in_dense_training_bins(
        rows,
        (SharedPrefixExecutionUnit((0, 1, 2), None, 12),),
        costs=(4, 4, 4),
        padding_multiple=1,
        bin_capacity=12,
    )
    assert unit.row_indices == (0, 2, 1)
    assert unit.physical_length == 9
    assert unit.shared_layout.mtp_loss_group_root_counts == (2,)
    tensor_bin = materialize_shared_prefix_layout(
        inputs, input_lengths=torch.tensor([4, 4, 4]), layout=unit.shared_layout
    )
    mask = build_star_attention_allow_mask(tensor_bin.layout)
    assert not bool(mask[5:, :5].any()) and not bool(mask[:5, 5:].any())
    assert not bool(mask[3, 4]) and not bool(mask[4, 3])


def test_training_alignment_keeps_same_cuts_as_conservative_alignment():
    layout = build_shared_prefix_layout(
        [SharedPrefixRow(index, "a", (1, 2), 2) for index in range(4)]
    )
    unit = SharedPrefixExecutionUnit(layout.row_indices, layout, layout.physical_total_length)
    args = dict(costs=(4, 4, 4, 4), capacity=16, target_count=3)
    dense, count = materialize_alignment((unit,), **args)
    shared, shared_count = align_training_units((unit,), padding_multiple=1, **args)
    assert count == shared_count == 3
    assert [u.row_indices for u in dense] == [u.row_indices for u in shared]
    assert sum(u.physical_length for u in shared) < sum(u.physical_length for u in dense)
    assert all(u.shared_layout is None for u in shared if len(u.row_indices) == 1)


def test_physical_alignment_rebuilds_real_subsets_without_dummy_units():
    layout = build_shared_prefix_layout(
        [SharedPrefixRow(index, "a", tuple(range(8)), 2) for index in range(4)]
    )
    unit = SharedPrefixExecutionUnit(layout.row_indices, layout, layout.physical_total_length)
    units, stats = align_physical_units(
        (unit,), costs=(10, 10, 10, 10), capacity=20, target_count=2, padding_multiple=1
    )
    assert [u.row_indices for u in units] == [(0, 1), (2, 3)]
    assert [u.physical_length for u in units] == [12, 12]
    assert stats["rebuilt_units"] == 2


def test_prescribed_slot_validation_keeps_group_and_slot_order():
    assert get_prescribed_shared_prefix_slots(
        group_ids=("a", "b", "a", "b"), slot_ids=(0, 0, 1, 1)
    ) == ((0,), (1,), (2,), (3,))
    with pytest.raises(ValueError, match="same number"):
        get_prescribed_shared_prefix_slots(group_ids=("a", "a", "b"), slot_ids=(0, 1, 0))


def test_pure_packing_imports_do_not_load_model_or_optional_generation_dependencies():
    repo = Path(__file__).resolve().parents[3]
    code = '''
import sys
sys.path.insert(0, sys.argv[1])
from megatron.rl.shared_prefix_packing import SharedPrefixRow, plan_shared_prefix_bins
from megatron.rl.shared_prefix_execution import plan_shared_prefix_execution_units
from megatron.rl.shared_prefix_alignment import align_rows_to_count
from megatron.rl.shared_prefix_cost import estimate_shared_prefix_row_work
rows = [SharedPrefixRow(i, "a", (1, 2), 1) for i in range(2)]
assert len(plan_shared_prefix_bins(rows, bin_capacity=4).shared_bins) == 1
assert not any(k == "torch" or k == "pydantic" or k.startswith("nemo_rl")
               or k.startswith("megatron.core") for k in sys.modules)
'''
    subprocess.run([sys.executable, "-I", "-S", "-c", code, str(repo)], check=True)


def test_historical_generation_request_api_and_pickle_paths_are_preserved():
    pytest.importorskip("pydantic")
    from megatron.rl import GenericGenerationArgs, Request, TypeLookupable

    args = GenericGenerationArgs(temperature=0.8, top_k=16)
    updated = args.add(GenericGenerationArgs(max_tokens=32))
    assert updated.temperature == 0.8 and updated.top_k == 16 and updated.max_tokens == 32
    request = Request(generation_args=updated)
    assert pickle.loads(pickle.dumps(request)) == request
    assert GenericGenerationArgs.__module__ == Request.__module__ == "megatron.rl"
    assert TypeLookupable.__module__ == "megatron.rl"


def _merge_rows():
    # Two stars of group "a" plus three dense rows: one ungrouped, two of group "b".
    return [
        SharedPrefixRow(0, "a", (1, 2), 2),
        SharedPrefixRow(1, "a", (1, 2), 2),
        SharedPrefixRow(2, None, (5, 6), 2),
        SharedPrefixRow(3, "b", (7, 8), 1),
        SharedPrefixRow(4, "b", (7, 9), 1),
    ]


def _first_fit(capacity):
    def pack(costs):
        bins, loads = [], []
        for index, cost in sorted(enumerate(costs), key=lambda item: (-item[1], item[0])):
            for bin_index, load in enumerate(loads):
                if load + cost <= capacity:
                    bins[bin_index].append(index)
                    loads[bin_index] += cost
                    break
            else:
                bins.append([index])
                loads.append(cost)
        return bins

    return pack


def _merge_units(rows, *, bin_capacity, dense_packer):
    return plan_shared_prefix_execution_units(
        rows,
        row_slots=((0, 1), (2,), (3, 4)),
        bin_capacity=bin_capacity,
        padding_multiple=1,
        pack_groups=True,
        pack_dense_fallbacks=True,
        merge_dense_fallbacks=True,
        dense_packer=dense_packer,
    )


def test_merge_dense_fallbacks_keeps_ungrouped_rows_dense_and_mtp_groups():
    rows = _merge_rows()
    units = _merge_units(rows, bin_capacity=20, dense_packer=lambda costs: [[0], [1, 2]])
    assert sorted(index for unit in units for index in unit.row_indices) == [0, 1, 2, 3, 4]
    # The bin holding the ungrouped row would fit, but stays dense instead of raising.
    assert SharedPrefixExecutionUnit((2,), None, 4) in units
    (forest,) = [unit for unit in units if unit.shared_layout is not None]
    # Star (0, 1) absorbs the dense bin (3, 4) as one MTP normalization group.
    assert forest.row_indices == (0, 1, 3, 4)
    assert forest.shared_layout.mtp_loss_group_root_counts == (1, 2)
    expanded = sum(
        len(root.row_indices) * root.prompt_length + sum(root.physical_completion_lengths)
        for _, root in forest.shared_layout.iter_roots()
    )
    assert forest.physical_length == 12 and expanded == 14


def test_merge_dense_fallbacks_respects_the_expanded_budget():
    rows = _merge_rows()
    units = _merge_units(rows, bin_capacity=12, dense_packer=lambda costs: [[0], [1, 2]])
    # Absorbing (3, 4) would need 14 expanded tokens > 12, so nothing merges.
    assert all(unit.shared_layout is None or unit.row_indices == (0, 1) for unit in units)
    assert sorted(index for unit in units for index in unit.row_indices) == [0, 1, 2, 3, 4]


def test_dense_packer_bins_must_fit_the_token_budget():
    rows = [SharedPrefixRow(index, str(index), (1, 2), 6) for index in range(4)]
    with pytest.raises(ValueError, match="within the token budget"):
        plan_shared_prefix_execution_units(
            rows,
            row_slots=((0,), (1,), (2,), (3,)),
            bin_capacity=8,
            padding_multiple=1,
            pack_groups=True,
            pack_dense_fallbacks=True,
            dense_packer=lambda costs: [[0, 1, 2, 3]],
        )


def test_dense_packer_must_be_deterministic():
    calls = []

    def unstable(costs):
        calls.append(None)
        return [[0], [1]] if len(calls) % 2 else [[1], [0]]

    with pytest.raises(ValueError, match="deterministic"):
        plan_dense_training_bins(costs=(4, 4), bin_capacity=8, dense_packer=unstable)

    def mutating(costs):
        costs.sort()  # a packer may reorder its own copy without breaking the check
        return [[0, 1]]

    assert len(plan_dense_training_bins(costs=(4, 3), bin_capacity=8, dense_packer=mutating)) == 1


def _large_group(count, completion_length=1):
    return [SharedPrefixRow(index, "g", (1, 2, 3), completion_length) for index in range(count)]


@pytest.mark.parametrize(
    "count, sizes", [(16, (16,)), (17, (9, 8)), (32, (16, 16)), (33, (11,) * 3)]
)
def test_slot_larger_than_branch_limit_splits_evenly_under_group_packing(count, sizes):
    rows = _large_group(count)
    units = plan_shared_prefix_execution_units(
        rows,
        row_slots=(tuple(range(count)),),
        bin_capacity=4 * count,
        padding_multiple=1,
        pack_groups=True,
    )
    roots = [root for unit in units for _, root in unit.shared_layout.iter_roots()]
    assert tuple(len(root.row_indices) for root in roots) == sizes
    assert sorted(index for unit in units for index in unit.row_indices) == list(range(count))


def test_slot_larger_than_branch_limit_warns_and_stays_one_dense_forward():
    rows = _large_group(17)
    with pytest.warns(UserWarning, match="exceeds max_completions_per_bin=16"):
        units = plan_shared_prefix_execution_units(
            rows, row_slots=(tuple(range(17)),), bin_capacity=68, padding_multiple=1
        )
    assert units == (SharedPrefixExecutionUnit(tuple(range(17)), None, 68),)
    # Raising the limit keeps the slot as one shared star.
    (unit,) = plan_shared_prefix_execution_units(
        rows,
        row_slots=(tuple(range(17)),),
        bin_capacity=68,
        padding_multiple=1,
        max_completions_per_bin=17,
    )
    assert len(unit.shared_layout.row_indices) == 17


def test_dense_bin_reconstruction_splits_large_groups_evenly():
    rows = _large_group(17)
    (unit,) = share_prefixes_in_dense_training_bins(
        rows,
        (SharedPrefixExecutionUnit(tuple(range(17)), None, 68),),
        costs=(4,) * 17,
        padding_multiple=1,
        bin_capacity=68,
    )
    roots = unit.shared_layout.roots
    assert [len(root.row_indices) for root in roots] == [9, 8]
    assert unit.shared_layout.mtp_loss_group_root_counts == (2,)
    assert unit.physical_length == 2 * 3 + 17
