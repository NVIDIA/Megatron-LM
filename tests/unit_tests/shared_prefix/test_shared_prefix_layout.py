# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SharedPrefixLayout metadata against the dense rows it represents (CPU)."""

import pytest
import torch

from megatron.core.shared_prefix_layout import SharedPrefixForestLayout, SharedPrefixLayout


def test_layout_prompt_multiplicities_equal_dense_gather_counts():
    layout = SharedPrefixLayout(3, (2, 5, 1))
    indices = torch.cat(layout.dense_branch_indices("cpu"))
    expected = torch.bincount(indices, minlength=layout.total_len).float()
    torch.testing.assert_close(
        layout.padded_token_multiplicities(layout.total_len, "cpu"), expected, rtol=0, atol=0
    )
    assert layout.position_ids("cpu").tolist() == [0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 3]


def test_cp_local_indices_follow_the_zigzag_split():
    """Rank r of C owns chunks r and 2C-r-1 of 2C; together the ranks own every index once."""
    assert SharedPrefixLayout.cp_local_indices(8, 2, 0, "cpu").tolist() == [0, 1, 6, 7]
    assert SharedPrefixLayout.cp_local_indices(8, 2, 1, "cpu").tolist() == [2, 3, 4, 5]
    for cp_size in (1, 2, 4):
        owned = [
            SharedPrefixForestLayout.cp_local_indices(16, cp_size, rank, "cpu")
            for rank in range(cp_size)
        ]
        assert all(indices.dtype == torch.long for indices in owned)
        assert torch.equal(torch.cat(owned).sort().values, torch.arange(16))


@pytest.mark.parametrize(
    "length, cp_size, cp_rank, match",
    [
        (8, 0, 0, "invalid CP geometry"),
        (8, 2, -1, "invalid CP geometry"),
        (8, 2, 2, "invalid CP geometry"),
        (10, 4, 0, r"divisible by 2 \* cp_size"),
    ],
)
def test_cp_local_indices_reject_invalid_geometry(length, cp_size, cp_rank, match):
    with pytest.raises(ValueError, match=match):
        SharedPrefixLayout.cp_local_indices(length, cp_size, cp_rank, "cpu")


def _two_root_forest():
    """Roots [3 | 4, 2] and [2 | 2] with per-branch padding (logical 3, 1 and 2)."""
    return SharedPrefixForestLayout(
        (
            SharedPrefixLayout(3, (4, 2), logical_completion_lens=(3, 1), padding_multiple=2),
            SharedPrefixLayout(2, [2], logical_completion_lens=[2], padding_multiple=2),
        )
    )


def test_forest_layout_concatenates_independent_roots():
    forest = _two_root_forest()
    assert forest.total_len == 13
    assert forest.padding_multiple == 2
    assert [offset for offset, _ in forest.iter_roots()] == [0, 9]
    assert forest.forest == [(0, 3, [4, 2]), (9, 2, [2])]
    assert forest.dense_branch_lengths == (7, 5, 4)
    # Each dense row repeats its own root's prompt and never reaches into another root.
    assert [branch.tolist() for branch in forest.dense_branch_indices("cpu")] == [
        [0, 1, 2, 3, 4, 5, 6],
        [0, 1, 2, 7, 8],
        [9, 10, 11, 12],
    ]
    assert forest.padded_position_ids(16, "cpu").tolist() == (
        [0, 1, 2, 3, 4, 5, 6, 3, 4] + [0, 1, 2, 3] + [0, 0, 0]
    )


@pytest.mark.parametrize(
    "exclude_sequence_padding, completion_rows",
    [(False, [1, 1, 1, 1, 1, 1]), (True, [1, 1, 1, 0, 1, 0])],
)
def test_forest_multiplicities_count_dense_copies(exclude_sequence_padding, completion_rows):
    """Prompts count once per completion; per-branch padding counts only in the unmasked
    baseline; trailing topology padding never counts."""
    forest = _two_root_forest()
    multiplicities = forest.padded_token_multiplicities(
        16, "cpu", exclude_sequence_padding=exclude_sequence_padding
    )
    assert multiplicities.tolist() == [2, 2, 2] + completion_rows + [1, 1, 1, 1] + [0, 0, 0]
    if not exclude_sequence_padding:
        dense_rows = torch.cat(forest.dense_branch_indices("cpu"))
        assert multiplicities[:13].tolist() == torch.bincount(dense_rows).tolist()


def test_forest_mtp_loss_groups_partition_dense_lengths():
    roots = _two_root_forest().roots
    assert SharedPrefixForestLayout(roots).mtp_loss_group_lengths == (12, 4)
    grouped = SharedPrefixForestLayout(roots, mtp_loss_group_root_counts=[2])
    assert grouped.mtp_loss_group_lengths == (16,)


def test_layouts_reject_physical_lengths_shorter_than_the_layout():
    forest = _two_root_forest()
    for padded in (forest.padded_position_ids, forest.padded_token_multiplicities):
        with pytest.raises(ValueError, match="shorter than shared-prefix forest"):
            padded(12, "cpu")
    star = forest.roots[0]
    for padded in (star.padded_position_ids, star.padded_token_multiplicities):
        with pytest.raises(ValueError, match="physical length 8 is shorter than layout 9"):
            padded(8, "cpu")


def test_layouts_normalize_fields():
    """Sequences become tuples and star lengths become ints. ``padding_multiple`` and the MTP
    group counts are not coerced: they must already be Python ints."""
    import numpy as np

    star = SharedPrefixLayout(
        np.int64(3), [np.int64(4), 2], logical_completion_lens=[np.int64(3), 2], padding_multiple=2
    )
    assert star.prefix_len == 3 and star.completion_lens == (4, 2)
    assert star.logical_completion_lens == (3, 2)
    lengths = (star.prefix_len, *star.completion_lens, *star.logical_completion_lens)
    assert all(type(length) is int for length in lengths)
    forest = SharedPrefixForestLayout([star, star], mtp_loss_group_root_counts=[1, 1])
    assert forest.roots == (star, star)
    assert forest.mtp_loss_group_root_counts == (1, 1)
    with pytest.raises(ValueError, match="padding_multiple must be an integer"):
        SharedPrefixLayout(3, (4,), logical_completion_lens=(4,), padding_multiple=np.int64(2))
    with pytest.raises(ValueError, match="positive integers"):
        SharedPrefixForestLayout((star,), mtp_loss_group_root_counts=(np.int64(1),))


@pytest.mark.parametrize(
    "prefix_len, completion_lens, logical_completion_lens, padding_multiple, match",
    [
        (0, (2,), None, None, "non-empty prefix"),
        (3, (), None, None, "one or more non-empty completions"),
        (3, (2, 0), None, None, "one or more non-empty completions"),
        (3, (2,), (3,), 4, "not exceed physical completion lengths"),
        (3, (2,), (0,), 4, "logical completion lengths must be positive"),
        (3, (2, 2), (2,), 4, "match the physical branch count"),
        (3, (2,), (2,), None, "must be provided together"),
        (3, (2,), None, 4, "must be provided together"),
        (3, (2,), (2,), True, "padding_multiple must be an integer"),
        (3, (2,), (2,), 1.5, "padding_multiple must be an integer"),
        (3, (2,), (2,), 0, "padding_multiple must be positive"),
    ],
)
def test_star_layout_rejects_invalid_lengths(
    prefix_len, completion_lens, logical_completion_lens, padding_multiple, match
):
    with pytest.raises(ValueError, match=match):
        SharedPrefixLayout(
            prefix_len,
            completion_lens,
            logical_completion_lens=logical_completion_lens,
            padding_multiple=padding_multiple,
        )


_STAR = SharedPrefixLayout(3, (2,))
_PADDED_STAR = SharedPrefixLayout(3, (2,), logical_completion_lens=(2,), padding_multiple=8)


@pytest.mark.parametrize(
    "roots, counts, match",
    [
        ((), (), "one or more star layouts"),
        ((object(),), (), "one or more star layouts"),
        ((_PADDED_STAR, _STAR), (), "same padding multiple"),
        ((_STAR, _STAR), "12", "sequence of integers"),
        ((_STAR, _STAR), 3, "sequence of integers"),
        ((_STAR, _STAR), (True, 1), "positive integers"),
        ((_STAR, _STAR), (0, 2), "positive integers"),
        ((_STAR, _STAR), (1,), "partition every forest root"),
        ((_STAR, _STAR), (2, 1), "partition every forest root"),
    ],
)
def test_forest_layout_rejects_invalid_roots_and_groups(roots, counts, match):
    with pytest.raises(ValueError, match=match):
        SharedPrefixForestLayout(roots, mtp_loss_group_root_counts=counts)
