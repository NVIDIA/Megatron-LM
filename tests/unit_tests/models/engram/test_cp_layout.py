# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The zigzag layout arithmetic the n-gram memory uses under context parallelism.

These are pure index helpers, so they are checked against an explicit statement of the
convention rather than against a second implementation: rank ``r`` of ``cp_size`` owns
``chunk[r]`` and ``chunk[2 * cp_size - 1 - r]`` of a sequence cut into ``2 * cp_size`` pieces.
Getting this wrong is the kind of bug an end-to-end loss comparison hides, because a
consistently permuted sequence still trains.
"""

import pytest
import torch

from megatron.core.models.engram.cp_layout import restore_zigzag, select_zigzag


def _global(length):
    return torch.arange(length, dtype=torch.int64).view(1, length)


def test_select_matches_the_documented_chunk_pairing():
    sequence = _global(8)
    # cp_size=2 -> 4 chunks of 2: rank 0 owns chunks 0 and 3, rank 1 owns chunks 1 and 2.
    torch.testing.assert_close(
        select_zigzag(sequence, 2, 0, sequence_dim=1), torch.tensor([[0, 1, 6, 7]])
    )
    torch.testing.assert_close(
        select_zigzag(sequence, 2, 1, sequence_dim=1), torch.tensor([[2, 3, 4, 5]])
    )


def test_restore_is_the_inverse_of_select():
    for cp_size in (1, 2, 4):
        sequence = _global(8 * cp_size)
        shards = [select_zigzag(sequence, cp_size, rank, 1) for rank in range(cp_size)]
        gathered = torch.cat(shards, dim=1)  # what an all-gather produces: rank-major
        torch.testing.assert_close(restore_zigzag(gathered, cp_size, sequence_dim=1), sequence)


def test_round_trip_holds_on_the_sequence_leading_dimension():
    """Hidden states are [S, B, ...]; tokens are [B, S]. Both layouts must work."""
    hidden = torch.arange(8 * 3).view(8, 3)
    shards = [select_zigzag(hidden, 2, rank, sequence_dim=0) for rank in range(2)]
    gathered = torch.cat(shards, dim=0)
    torch.testing.assert_close(restore_zigzag(gathered, 2, sequence_dim=0), hidden)


def test_cp_size_one_is_the_identity():
    sequence = _global(5)
    torch.testing.assert_close(select_zigzag(sequence, 1, 0, 1), sequence)
    torch.testing.assert_close(restore_zigzag(sequence, 1, 1), sequence)


def test_thd_partition_matches_per_document_zigzag():
    """Packed rows zigzag inside each document, not across the row."""
    from megatron.core.models.engram.cp_layout import thd_partition_index

    # Two documents of length 8 each, cp_size=2 -> 4 chunks of 2 per document.
    cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32)
    # Rank 0 owns chunks 0 and 3 of every document; rank 1 owns chunks 1 and 2.
    torch.testing.assert_close(
        thd_partition_index(cu_seqlens, 2, 0), torch.tensor([0, 1, 6, 7, 8, 9, 14, 15])
    )
    torch.testing.assert_close(
        thd_partition_index(cu_seqlens, 2, 1), torch.tensor([2, 3, 4, 5, 10, 11, 12, 13])
    )
    # Together the ranks cover every position exactly once.
    covered = torch.cat([thd_partition_index(cu_seqlens, 2, r) for r in range(2)]).sort().values
    torch.testing.assert_close(covered, torch.arange(16))


def test_thd_partition_rejects_indivisible_documents():
    from megatron.core.models.engram.cp_layout import thd_partition_index

    cu_seqlens = torch.tensor([0, 6], dtype=torch.int32)  # 6 is not divisible by 2*cp_size=4
    with pytest.raises(ValueError, match="divisible by 2 \\* cp_size"):
        thd_partition_index(cu_seqlens, 2, 0)
