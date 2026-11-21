# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts of the indexer top-k query layouts."""

from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.kernels.indexer_topk import IndexerGeometry, QueryLayout, QuerySegment

pytestmark = pytest.mark.mlite


def _visible_key_rows(layout: QueryLayout) -> dict[int, list[int]]:
    """Map every covered local row to the key tensor rows it sees."""
    rows = {}
    for segment in layout.segments:
        for row in range(segment.row_start, segment.row_end):
            visible = layout.visible_keys(segment, row)
            rows[row] = list(range(segment.key_start, segment.key_start + visible))
    return rows


@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
def test_contiguous_cp_covers_rows_once(cp_size):
    sequence = 96 * cp_size
    local = sequence // cp_size
    positions = []
    for rank in range(cp_size):
        layout = QueryLayout.contiguous(local, position=rank * local, keys=sequence)
        (segment,) = layout.segments
        assert (segment.row_start, segment.row_end, segment.position) == (0, local, rank * local)
        for row in range(local):
            position = rank * local + row
            positions.append(position)
            assert layout.visible_keys(segment, row) == position + 1
    assert sorted(positions) == list(range(sequence))

    # Packed sequences split into contiguous shards: every token of every sequence is a local row
    # of exactly one rank, and the rows past the last sequence are uncovered padding.
    cu_seqlens = [0, 5, 5, 61, 64, 90]
    local = -(-cu_seqlens[-1] // cp_size) + 3
    seen = []
    for rank in range(cp_size):
        layout = QueryLayout.packed(
            cu_seqlens, row_start=rank * local, rows=local, absolute_ids=True
        )
        for segment in layout.segments:
            seen.extend(rank * local + row for row in range(segment.row_start, segment.row_end))
    assert seen == list(range(cu_seqlens[-1]))


def test_packed_segments_intersect_sequences():
    cu_seqlens = [0, 10, 10, 30, 47]  # the second sequence is empty
    layout = QueryLayout.packed(cu_seqlens, row_start=8, rows=30, absolute_ids=False)
    assert layout.segments == (
        QuerySegment(row_start=0, row_end=2, position=8, key_start=0, key_count=10, index_base=0),
        QuerySegment(row_start=2, row_end=22, position=0, key_start=10, key_count=20, index_base=0),
        QuerySegment(
            row_start=22, row_end=30, position=0, key_start=30, key_count=17, index_base=0
        ),
    )
    assert [layout.visible_keys(layout.segments[0], row) for row in (0, 1)] == [9, 10]
    assert [layout.visible_keys(layout.segments[1], row) for row in (2, 5, 21)] == [1, 4, 20]
    assert layout.visible_keys(layout.segments[2], 29) == 8

    # Absolute ids: the output ids are key tensor rows; padding rows past the last sequence.
    layout = QueryLayout.packed(cu_seqlens, row_start=40, rows=16, absolute_ids=True)
    assert layout.segments == (
        QuerySegment(
            row_start=0, row_end=7, position=10, key_start=30, key_count=17, index_base=30
        ),
    )
    assert layout.rows == 16


def test_visible_keys_match_dsa_cp_mask(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.attention.dsa import (
        _build_cp_causal_mask,
        _dense_cp_layout,
        _packed_cp_layout,
    )

    cp_size, local = 4, 24
    for rank in range(cp_size):
        query_positions, key_order = _dense_cp_layout(
            local_seq=local, cp_size=cp_size, cp_rank=rank, device=torch.device("cpu")
        )
        mask = _build_cp_causal_mask(query_positions, key_order)
        layout = QueryLayout.contiguous(local, position=rank * local, keys=cp_size * local)
        expected = {
            row: torch.nonzero(torch.isfinite(mask[row])).flatten().tolist() for row in range(local)
        }
        assert _visible_key_rows(layout) == expected

    cu_seqlens = [0, 7, 7, 40, 61, 96]
    for rank in range(cp_size):
        query_positions, key_order = _packed_cp_layout(
            torch.tensor(cu_seqlens), cp_size=cp_size, cp_rank=rank, device=torch.device("cpu")
        )
        mask = _build_cp_causal_mask(
            query_positions, key_order, cu_seqlens=torch.tensor(cu_seqlens)
        )
        layout = QueryLayout.packed(
            cu_seqlens, row_start=rank * local, rows=local, absolute_ids=True
        )
        expected = {
            row: torch.nonzero(torch.isfinite(mask[row])).flatten().tolist() for row in range(local)
        }
        assert _visible_key_rows(layout) == expected


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_zigzag_two_segments(cp_size):
    chunk = 6
    total = 2 * cp_size * chunk
    covered = []
    for rank in range(cp_size):
        layout = QueryLayout(
            rows=2 * chunk,
            segments=(
                QuerySegment(0, chunk, rank * chunk, 0, total, 0),
                QuerySegment(chunk, 2 * chunk, (2 * cp_size - 1 - rank) * chunk, 0, total, 0),
            ),
        )
        for segment in layout.segments:
            for row in range(segment.row_start, segment.row_end):
                position = segment.position + row - segment.row_start
                covered.append(position)
                assert layout.visible_keys(segment, row) == position + 1
    assert sorted(covered) == list(range(total))


def test_full_layout_and_empty_layouts():
    layout = QueryLayout.full(10, keys=3)
    assert layout.segments == (QuerySegment(0, 10, 0, 0, 3, 0),)
    assert [layout.visible_keys(layout.segments[0], row) for row in range(10)] == [
        1,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
    ]
    # Rows past the last key see every key of the sequence, never more.
    layout = QueryLayout.full(10, keys=2)
    assert [layout.visible_keys(layout.segments[0], row) for row in (0, 1, 9)] == [1, 2, 2]
    layout = QueryLayout.contiguous(4, position=6, keys=5)
    assert [layout.visible_keys(layout.segments[0], row) for row in range(4)] == [5, 5, 5, 5]
    assert QueryLayout.full(0, keys=0).segments == ()
    assert QueryLayout.packed([0], row_start=0, rows=4, absolute_ids=False).segments == ()


def test_layout_validation():
    with pytest.raises(ValueError, match="ordered, disjoint"):
        QueryLayout(rows=10, segments=(QuerySegment(0, 6, 0, 0, 6, 0),) * 2)
    with pytest.raises(ValueError, match="ordered, disjoint"):
        QueryLayout(rows=5, segments=(QuerySegment(0, 6, 0, 0, 6, 0),))
    with pytest.raises(ValueError, match="row_end"):
        QuerySegment(3, 3, 0, 0, 1, 0)
    with pytest.raises(ValueError, match="int32"):
        QuerySegment(0, 1, 0, 0, 2, 2**31 - 1)
    with pytest.raises(ValueError, match="outside the segment"):
        layout = QueryLayout.full(4, keys=4)
        layout.visible_keys(layout.segments[0], 4)
    with pytest.raises(TypeError, match="host integers"):
        QueryLayout.packed(torch.tensor([0, 4]), row_start=0, rows=4, absolute_ids=False)
    with pytest.raises(ValueError, match="non-decreasing"):
        QueryLayout.packed([0, 5, 3], row_start=0, rows=4, absolute_ids=False)
    with pytest.raises(ValueError, match="num_heads"):
        IndexerGeometry(num_heads=0, head_dim=128, topk=2048)
    assert IndexerGeometry(32, 128, 2048) == IndexerGeometry(32, 128, 2048)
