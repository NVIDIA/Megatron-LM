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
    SharedPrefixFallbackReason,
    SharedPrefixRow,
    build_shared_prefix_layout,
    plan_shared_prefix_bins,
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
    assert layout.baseline_length == 15
    assert layout.tokens_saved == 6
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
    exact = plan_shared_prefix_bins(rows, bin_capacity=13, sequence_length_pad_multiple=4)

    assert not too_small.shared_bins
    assert [fallback.reason for fallback in too_small.fallbacks] == [
        SharedPrefixFallbackReason.NO_CAPACITY_COMPATIBLE_PEER,
        SharedPrefixFallbackReason.NO_CAPACITY_COMPATIBLE_PEER,
    ]
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
    assert forward.fallbacks[0].reason is SharedPrefixFallbackReason.SEQUENCE_EXCEEDS_BIN
    assert not forward.fallbacks[0].fits_block_diagonal_bin


def test_planner_splits_a_group_at_token_and_branch_limits() -> None:
    rows = [_row(i, completion_length=2) for i in range(5)]

    plan = plan_shared_prefix_bins(rows, bin_capacity=9, max_completions_per_bin=3)

    assert [layout.row_indices for layout in plan.shared_bins] == [(0, 1, 2), (3, 4)]
    assert not plan.fallbacks


def test_exact_prompt_mismatch_does_not_share_within_group_id() -> None:
    plan = plan_shared_prefix_bins(
        [_row(0, prompt=(1, 2)), _row(1, prompt=(1, 9)), _row(2, prompt=(1, 2))], bin_capacity=20
    )

    assert len(plan.shared_bins) == 1
    assert plan.shared_bins[0].row_indices == (0, 2)
    assert plan.fallback_row_indices == (1,)
    assert plan.fallbacks[0].reason is SharedPrefixFallbackReason.PROMPT_MISMATCH


def test_planner_keeps_explicit_mixed_fallback_reasons() -> None:
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
    assert [fallback.reason for fallback in plan.fallbacks] == [
        SharedPrefixFallbackReason.MISSING_GROUP_ID,
        SharedPrefixFallbackReason.EMPTY_PROMPT,
        SharedPrefixFallbackReason.EMPTY_COMPLETION,
        SharedPrefixFallbackReason.NO_EXACT_PROMPT_PEER,
        SharedPrefixFallbackReason.NO_CAPACITY_COMPATIBLE_PEER,
        SharedPrefixFallbackReason.NO_CAPACITY_COMPATIBLE_PEER,
    ]
    assert all(fallback.fits_block_diagonal_bin for fallback in plan.fallbacks)


def test_planner_covers_every_row_exactly_once() -> None:
    rows = [
        _row(7, group_id="a"),
        _row(2, group_id="a"),
        _row(9, group_id="b"),
        _row(4, group_id=None),
    ]

    plan = plan_shared_prefix_bins(rows, bin_capacity=20)
    covered = plan.shared_row_indices + plan.fallback_row_indices

    assert sorted(covered) == [2, 4, 7, 9]
    assert len(covered) == len(set(covered))


@pytest.mark.parametrize(
    "rows, bin_capacity, max_completions_per_bin, message",
    [
        ([_row(0), _row(0)], 20, 16, "row_index values must be unique"),
        ([_row(0), _row(1)], 0, 16, "bin_capacity must be positive"),
        ([_row(0), _row(1)], 20, 1, "must be at least 2"),
    ],
)
def test_planner_rejects_invalid_inputs(
    rows: list[SharedPrefixRow], bin_capacity: int, max_completions_per_bin: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        plan_shared_prefix_bins(
            rows, bin_capacity=bin_capacity, max_completions_per_bin=max_completions_per_bin
        )
