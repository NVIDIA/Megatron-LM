# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SharedPrefixTree and SharedPrefixPack metadata against the dense rows they represent (CPU)."""

import pytest
import torch

from megatron.core.datasets.shared_prefix.layout import (
    SharedPrefixPack,
    SharedPrefixTree,
    shared_prefix_topology_multiple,
)


def test_layout_prompt_multiplicities_equal_dense_gather_counts():
    layout = SharedPrefixTree(3, (2, 5, 1))
    indices = torch.cat(layout.dense_branch_indices("cpu"))
    expected = torch.bincount(indices, minlength=layout.total_len).float()
    torch.testing.assert_close(
        layout.padded_token_multiplicities(layout.total_len, "cpu"), expected, rtol=0, atol=0
    )
    assert layout.position_ids("cpu").tolist() == [0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 3]


def test_cp_local_indices_follow_the_zigzag_split():
    """Rank r of C owns chunks r and 2C-r-1 of 2C; together the ranks own every index once."""
    assert SharedPrefixTree.cp_local_indices(8, 2, 0, "cpu").tolist() == [0, 1, 6, 7]
    assert SharedPrefixTree.cp_local_indices(8, 2, 1, "cpu").tolist() == [2, 3, 4, 5]
    for cp_size in (1, 2, 4):
        owned = [
            SharedPrefixPack.cp_local_indices(16, cp_size, rank, "cpu") for rank in range(cp_size)
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
        SharedPrefixTree.cp_local_indices(length, cp_size, cp_rank, "cpu")


def _two_tree_pack():
    """Trees [3 | 4, 2] and [2 | 2] with per-branch padding (logical 3, 1 and 2)."""
    return SharedPrefixPack(
        (
            SharedPrefixTree(3, (4, 2), logical_completion_lens=(3, 1), padding_multiple=2),
            SharedPrefixTree(2, [2], logical_completion_lens=[2], padding_multiple=2),
        )
    )


def test_pack_concatenates_independent_trees():
    pack = _two_tree_pack()
    assert pack.total_len == 13
    assert pack.padding_multiple == 2
    assert [offset for offset, _ in pack.iter_roots()] == [0, 9]
    assert pack.forest == [(0, 3, [4, 2]), (9, 2, [2])]
    assert pack.dense_branch_lengths == (7, 5, 4)
    # Each dense row repeats its own tree's prompt and never reaches into another tree.
    assert [branch.tolist() for branch in pack.dense_branch_indices("cpu")] == [
        [0, 1, 2, 3, 4, 5, 6],
        [0, 1, 2, 7, 8],
        [9, 10, 11, 12],
    ]
    assert pack.padded_position_ids(16, "cpu").tolist() == (
        [0, 1, 2, 3, 4, 5, 6, 3, 4] + [0, 1, 2, 3] + [0, 0, 0]
    )


@pytest.mark.parametrize(
    "exclude_sequence_padding, completion_rows",
    [(False, [1, 1, 1, 1, 1, 1]), (True, [1, 1, 1, 0, 1, 0])],
)
def test_pack_multiplicities_count_dense_copies(exclude_sequence_padding, completion_rows):
    """Prompts count once per completion; per-branch padding counts only in the unmasked
    baseline; trailing topology padding never counts."""
    pack = _two_tree_pack()
    multiplicities = pack.padded_token_multiplicities(
        16, "cpu", exclude_sequence_padding=exclude_sequence_padding
    )
    assert multiplicities.tolist() == [2, 2, 2] + completion_rows + [1, 1, 1, 1] + [0, 0, 0]
    if not exclude_sequence_padding:
        dense_rows = torch.cat(pack.dense_branch_indices("cpu"))
        assert multiplicities[:13].tolist() == torch.bincount(dense_rows).tolist()


def test_pack_mtp_loss_groups_partition_dense_lengths():
    roots = _two_tree_pack().roots
    assert SharedPrefixPack(roots).mtp_loss_group_lengths == (12, 4)
    grouped = SharedPrefixPack(roots, mtp_loss_group_root_counts=[2])
    assert grouped.mtp_loss_group_lengths == (16,)


def test_layouts_reject_physical_lengths_shorter_than_the_layout():
    pack = _two_tree_pack()
    for padded in (pack.padded_position_ids, pack.padded_token_multiplicities):
        with pytest.raises(ValueError, match="shorter than the shared-prefix pack"):
            padded(12, "cpu")
    tree = pack.roots[0]
    for padded in (tree.padded_position_ids, tree.padded_token_multiplicities):
        with pytest.raises(ValueError, match="physical length 8 is shorter than layout 9"):
            padded(8, "cpu")


def test_layouts_normalize_fields():
    """Sequences become tuples and tree lengths become ints. ``padding_multiple`` and the MTP
    group counts are not coerced: they must already be Python ints."""
    import numpy as np

    tree = SharedPrefixTree(
        np.int64(3), [np.int64(4), 2], logical_completion_lens=[np.int64(3), 2], padding_multiple=2
    )
    assert tree.prefix_len == 3 and tree.completion_lens == (4, 2)
    assert tree.logical_completion_lens == (3, 2)
    lengths = (tree.prefix_len, *tree.completion_lens, *tree.logical_completion_lens)
    assert all(type(length) is int for length in lengths)
    pack = SharedPrefixPack([tree, tree], mtp_loss_group_root_counts=[1, 1])
    assert pack.roots == (tree, tree)
    assert pack.mtp_loss_group_root_counts == (1, 1)
    with pytest.raises(ValueError, match="padding_multiple must be an integer"):
        SharedPrefixTree(3, (4,), logical_completion_lens=(4,), padding_multiple=np.int64(2))
    with pytest.raises(ValueError, match="positive integers"):
        SharedPrefixPack((tree,), mtp_loss_group_root_counts=(np.int64(1),))


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
def test_tree_rejects_invalid_lengths(
    prefix_len, completion_lens, logical_completion_lens, padding_multiple, match
):
    with pytest.raises(ValueError, match=match):
        SharedPrefixTree(
            prefix_len,
            completion_lens,
            logical_completion_lens=logical_completion_lens,
            padding_multiple=padding_multiple,
        )


_TREE = SharedPrefixTree(3, (2,))
_PADDED_TREE = SharedPrefixTree(3, (2,), logical_completion_lens=(2,), padding_multiple=8)


@pytest.mark.parametrize(
    "roots, counts, match",
    [
        ((), (), "one or more SharedPrefixTree roots"),
        ((object(),), (), "one or more SharedPrefixTree roots"),
        ((_PADDED_TREE, _TREE), (), "same padding multiple"),
        ((_TREE, _TREE), "12", "sequence of integers"),
        ((_TREE, _TREE), 3, "sequence of integers"),
        ((_TREE, _TREE), (True, 1), "positive integers"),
        ((_TREE, _TREE), (0, 2), "positive integers"),
        ((_TREE, _TREE), (1,), "partition every root of the pack"),
        ((_TREE, _TREE), (2, 1), "partition every root of the pack"),
    ],
)
def test_pack_rejects_invalid_roots_and_groups(roots, counts, match):
    with pytest.raises(ValueError, match=match):
        SharedPrefixPack(roots, mtp_loss_group_root_counts=counts)


@pytest.mark.parametrize(
    "tp_size, cp_size, multiple", [(1, 1, 1), (1, 2, 4), (2, 1, 4), (2, 2, 8), (4, 4, 32)]
)
def test_topology_multiple(tp_size, cp_size, multiple):
    assert shared_prefix_topology_multiple(tp_size=tp_size, cp_size=cp_size) == multiple


@pytest.mark.parametrize("tp_size, cp_size", [(0, 1), (1, 0), (True, 1), (1, 2.0)])
def test_topology_multiple_rejects_invalid_sizes(tp_size, cp_size):
    with pytest.raises(ValueError, match="must be a positive integer"):
        shared_prefix_topology_multiple(tp_size=tp_size, cp_size=cp_size)
