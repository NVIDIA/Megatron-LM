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

from megatron.rl.shared_prefix_packing import (
    SharedPrefixForestLayout,
    SharedPrefixLayout,
    SharedPrefixRow,
    build_shared_prefix_layout,
    pack_shared_prefix_groups,
    plan_shared_prefix_bins,
    split_rows_evenly,
)


def _row(
    row_index: int,
    *,
    group_id: str | None = "group",
    prompt: tuple[int, ...] = (10, 11, 12),
    completion_length: int = 2,
) -> SharedPrefixRow:
    return SharedPrefixRow(
        row_index=row_index,
        group_id=group_id,
        prompt_token_ids=prompt,
        completion_length=completion_length,
    )


def test_build_layout_emits_gather_fanout_and_scatter_indices() -> None:
    layout = build_shared_prefix_layout(
        [_row(4, completion_length=2), _row(7, completion_length=1), _row(9, completion_length=3)]
    )

    assert layout.row_indices == (4, 7, 9)
    assert layout.completion_lengths == (2, 1, 3)
    assert layout.total_length == 9
    assert layout.branch_starts == (3, 5, 6)
    assert layout.position_ids == (0, 1, 2, 3, 4, 3, 3, 4, 5)

    assert layout.token_gather_rows == (4, 4, 4, 4, 4, 7, 9, 9, 9)
    assert layout.token_gather_columns == (0, 1, 2, 3, 4, 3, 3, 4, 5)
    assert layout.completion_positions == (3, 4, 5, 6, 7, 8)
    assert layout.predecessor_positions == (2, 3, 2, 2, 6, 7)
    assert layout.completion_scatter_rows == (4, 4, 7, 9, 9, 9)
    assert layout.completion_scatter_columns == (2, 3, 2, 2, 3, 4)


def test_layout_materializes_each_branch_to_the_dense_sequence_alignment() -> None:
    rows = [_row(0, completion_length=2), _row(1, completion_length=4)]

    layout = build_shared_prefix_layout(rows, sequence_length_pad_multiple=4)

    assert layout.completion_lengths == (2, 4)
    assert layout.physical_completion_lengths == (5, 5)
    assert layout.total_length == 9
    assert layout.physical_total_length == 13
    assert layout.branch_starts == (3, 8)
    assert layout.position_ids == (0, 1, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7)
    assert layout.physical_padding_positions == (5, 6, 7, 12)
    assert layout.completion_positions == (3, 4, 8, 9, 10, 11)
    assert layout.predecessor_positions == (2, 3, 2, 8, 9, 10)


def test_planner_accounts_for_physical_branch_padding_in_capacity() -> None:
    rows = [_row(0, completion_length=2), _row(1, completion_length=4)]

    too_small = plan_shared_prefix_bins(rows, bin_capacity=12, sequence_length_pad_multiple=4)
    exact = plan_shared_prefix_bins(rows, bin_capacity=16, sequence_length_pad_multiple=4)

    assert not too_small.shared_bins
    assert too_small.fallback_row_indices == (0, 1)
    assert len(exact.shared_bins) == 1
    assert exact.shared_bins[0].physical_total_length == 13


def test_layout_requires_a_valid_exact_prompt_group() -> None:
    with pytest.raises(ValueError, match="at least two"):
        build_shared_prefix_layout([_row(0)])

    with pytest.raises(ValueError, match="same group_id"):
        build_shared_prefix_layout([_row(0), _row(1, group_id="other")])

    with pytest.raises(ValueError, match="identical prompt"):
        build_shared_prefix_layout([_row(0), _row(1, prompt=(10, 99, 12))])

    with pytest.raises(ValueError, match="non-empty completion"):
        build_shared_prefix_layout([_row(0), _row(1, completion_length=0)])


def test_planner_is_input_order_invariant_and_first_fit_decreasing() -> None:
    rows = [
        _row(0, completion_length=3),
        _row(1, completion_length=8),
        _row(2, completion_length=3),
    ]

    forward = plan_shared_prefix_bins(rows, bin_capacity=10)
    reverse = plan_shared_prefix_bins(list(reversed(rows)), bin_capacity=10)

    assert forward == reverse
    assert len(forward.shared_bins) == 1
    assert forward.shared_bins[0].row_indices == (0, 2)
    assert forward.fallback_row_indices == (1,)


def test_planner_splits_a_group_at_token_and_branch_limits() -> None:
    rows = [_row(i, completion_length=2) for i in range(5)]

    plan = plan_shared_prefix_bins(rows, bin_capacity=9, max_completions_per_bin=3)

    assert [layout.row_indices for layout in plan.shared_bins] == [(0, 1, 2), (3, 4)]
    assert not plan.fallback_row_indices


def test_exact_prompt_mismatch_does_not_share_within_group_id() -> None:
    plan = plan_shared_prefix_bins(
        [_row(0, prompt=(1, 2)), _row(1, prompt=(1, 9)), _row(2, prompt=(1, 2))], bin_capacity=20
    )

    assert len(plan.shared_bins) == 1
    assert plan.shared_bins[0].row_indices == (0, 2)
    assert plan.fallback_row_indices == (1,)


def test_planner_keeps_every_ineligible_row_as_a_fallback() -> None:
    plan = plan_shared_prefix_bins(
        [
            _row(0, group_id=None),
            _row(1, prompt=()),
            _row(2, completion_length=0),
            _row(3, group_id="singleton"),
            _row(4, group_id="capacity", completion_length=5),
            _row(5, group_id="capacity", completion_length=5),
        ],
        bin_capacity=8,
    )

    assert plan.shared_bins == ()
    assert plan.fallback_row_indices == (0, 1, 2, 3, 4, 5)


def test_planner_covers_every_row_exactly_once() -> None:
    rows = [
        _row(7, group_id="a"),
        _row(2, group_id="a"),
        _row(9, group_id="b"),
        _row(4, group_id=None),
    ]

    plan = plan_shared_prefix_bins(rows, bin_capacity=20)
    covered = (
        tuple(row for layout in plan.shared_bins for row in layout.row_indices)
        + plan.fallback_row_indices
    )

    assert sorted(covered) == [2, 4, 7, 9]
    assert len(covered) == len(set(covered))


@pytest.mark.parametrize(
    "rows, bin_capacity, max_completions_per_bin, message",
    [
        ([_row(0), _row(0)], 20, 16, "row_index values must be unique"),
        ([_row(0), _row(1)], 0, 16, "bin_capacity must be a positive integer"),
        ([_row(0), _row(1)], 20.0, 16, "bin_capacity must be a positive integer"),
        ([_row(0), _row(1)], True, 16, "bin_capacity must be a positive integer"),
        ([_row(0), _row(1)], 20, 1, "must be an integer of at least 2"),
        ([_row(0), _row(1)], 20, 2.0, "must be an integer of at least 2"),
    ],
)
def test_planner_rejects_invalid_inputs(
    rows: list[SharedPrefixRow], bin_capacity: int, max_completions_per_bin: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        plan_shared_prefix_bins(
            rows, bin_capacity=bin_capacity, max_completions_per_bin=max_completions_per_bin
        )


def test_planner_requires_capacity_aligned_to_the_padding_multiple() -> None:
    rows = [_row(0, completion_length=2), _row(1, completion_length=4)]
    # A 13-token star padded to 4 would occupy 16 tokens, past a 13-token budget.
    with pytest.raises(ValueError, match="multiple of sequence_length_pad_multiple"):
        plan_shared_prefix_bins(rows, bin_capacity=13, sequence_length_pad_multiple=4)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(row_index=True), "row_index must be a nonnegative integer"),
        (dict(row_index=-1), "row_index must be a nonnegative integer"),
        (dict(completion_length=2.0), "completion_length must be a nonnegative integer"),
        (dict(group_id=""), "non-empty string"),
        (dict(group_id=7), "non-empty string"),
        (dict(prompt_token_ids=(1, True)), "prompt_token_ids must contain integers"),
        (dict(prompt_token_ids=(1, 2.0)), "prompt_token_ids must contain integers"),
    ],
)
def test_row_rejects_invalid_fields(kwargs: dict, message: str) -> None:
    fields = dict(row_index=0, group_id="g", prompt_token_ids=(1, 2), completion_length=1)
    fields.update(kwargs)
    with pytest.raises(ValueError, match=message):
        SharedPrefixRow(**fields)


def test_row_snapshots_list_prompts_as_hashable_tuples() -> None:
    prompt = [10, 11, 12]
    row = SharedPrefixRow(0, "group", prompt, 2)
    prompt.append(99)
    assert row.prompt_token_ids == (10, 11, 12)
    plan = plan_shared_prefix_bins([row, _row(1)], bin_capacity=20)
    assert plan.shared_bins[0].row_indices == (0, 1)


@pytest.mark.parametrize(
    "count, limit, sizes",
    [(0, 16, ()), (3, 16, (3,)), (16, 16, (16,)), (17, 16, (9, 8)), (33, 16, (11, 11, 11))],
)
def test_split_rows_evenly_avoids_singleton_remainders(count, limit, sizes) -> None:
    rows = [_row(index) for index in range(count)]
    chunks = split_rows_evenly(rows, max_completions_per_bin=limit)
    assert tuple(len(chunk) for chunk in chunks) == sizes
    assert [row for chunk in chunks for row in chunk] == rows


@pytest.mark.parametrize("count, sizes", [(16, [16]), (17, [9, 8]), (33, [11, 11, 11])])
def test_planner_splits_groups_past_the_branch_limit_evenly(count, sizes) -> None:
    rows = [_row(index) for index in range(count)]

    plan = plan_shared_prefix_bins(rows, bin_capacity=1000)

    assert [len(layout.row_indices) for layout in plan.shared_bins] == sizes
    assert not plan.fallback_row_indices


def _star(group: str, rows: int, completion_length: int, start: int = 0) -> SharedPrefixLayout:
    return build_shared_prefix_layout(
        [
            SharedPrefixRow(start + index, group, (1, 2, 3), completion_length)
            for index in range(rows)
        ]
    )


def test_group_packing_merges_stars_into_a_forest() -> None:
    a, b = _star("a", 2, 2), _star("b", 2, 1, start=2)
    # Backbones 7 + 5 fit 12; expanded 10 + 8 fit 20.
    (forest,) = pack_shared_prefix_groups(
        [b, a], bin_capacity=12, dense_capacity=20, padding_multiple=1
    )
    assert isinstance(forest, SharedPrefixForestLayout)
    # Stable first-fit decreasing by expanded cost: the larger star leads.
    assert forest.roots == (a, b)
    assert forest.row_indices == (0, 1, 2, 3)
    assert forest.physical_total_length == 12
    assert forest.mtp_loss_group_root_counts == ()


def test_group_packing_budgets_the_padded_backbone() -> None:
    a, b = _star("a", 2, 2), _star("b", 2, 1, start=2)
    # 7 + 5 = 12 tokens pad to 16 under a multiple of 8.
    (forest,) = pack_shared_prefix_groups(
        [a, b], bin_capacity=16, dense_capacity=None, padding_multiple=8
    )
    assert forest.physical_total_length == 12
    assert pack_shared_prefix_groups(
        [a, b], bin_capacity=8, dense_capacity=None, padding_multiple=8
    ) == (a, b)
    # An unaligned budget could admit a forest that topology padding overflows.
    with pytest.raises(ValueError, match="aligned to the padding multiple"):
        pack_shared_prefix_groups([a, b], bin_capacity=12, dense_capacity=None, padding_multiple=8)


def test_group_packing_respects_the_expanded_mtp_budget() -> None:
    a, b = _star("a", 2, 2), _star("b", 2, 1, start=2)
    assert pack_shared_prefix_groups(
        [a, b], bin_capacity=12, dense_capacity=17, padding_multiple=1
    ) == (a, b)
    # Evaluation omits the expanded budget and merges.
    (forest,) = pack_shared_prefix_groups(
        [a, b], bin_capacity=12, dense_capacity=None, padding_multiple=1
    )
    assert forest.row_indices == (0, 1, 2, 3)


def test_group_packing_rejects_a_star_over_the_backbone_budget() -> None:
    with pytest.raises(ValueError, match="aligned backbone budget"):
        pack_shared_prefix_groups(
            [_star("a", 2, 2)], bin_capacity=6, dense_capacity=None, padding_multiple=1
        )
