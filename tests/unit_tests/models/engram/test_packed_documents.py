# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed (THD) document boundaries for variants whose windows do not reset on a token value.

The `qwen` variant splits its n-gram windows on an EOS token, so packed rows work for it
without any extra information. The `deepseek` variant has no such token, so its boundaries
come from `cu_seqlens`. Both end up in the same place: a window never reaches back past the
first token of its own document, while the causal convolution deliberately still mixes across
the boundary — that is what both published references do.
"""

import torch

from megatron.core.models.engram.hashing import (
    build_ngram_hashes,
    segment_starts_from_cu_seqlens,
    shift_right_within_segments,
)

_PAD = 0
_MULTIPLIERS = torch.tensor([13, 17, 19], dtype=torch.int64)
_TABLE_SIZES = torch.tensor([11, 13, 17, 19], dtype=torch.int64)
_ORDER = 3
_HEADS = 2


def _hash(tokens, cu_seqlens=None):
    return build_ngram_hashes(
        input_ids=tokens,
        tokenizer_remap=None,
        multipliers=_MULTIPLIERS,
        table_sizes=_TABLE_SIZES,
        max_ngram_order=_ORDER,
        num_hash_heads=_HEADS,
        boundary_token_id=_PAD,
        reset_at_boundary=False,
        cu_seqlens=cu_seqlens,
    )


def test_segment_starts_from_cu_seqlens_marks_each_document():
    # Two documents of length 4 and 3 inside a row of capacity 9; the tail is padding.
    cu_seqlens = torch.tensor([0, 4, 7], dtype=torch.int32)
    starts = segment_starts_from_cu_seqlens(cu_seqlens, batch_size=1, sequence_length=9)
    torch.testing.assert_close(
        starts, torch.tensor([[0, 0, 0, 0, 4, 4, 4, 7, 7]], dtype=torch.int64)
    )


def test_shift_never_crosses_a_packed_boundary():
    tokens = torch.arange(1, 10, dtype=torch.int64).view(1, 9)
    cu_seqlens = torch.tensor([0, 4, 7], dtype=torch.int32)
    starts = segment_starts_from_cu_seqlens(cu_seqlens, 1, 9)

    shifted = shift_right_within_segments(tokens, 1, _PAD, starts)
    # Positions 0, 4 and 7 start a document, so their shifted source is the fill value.
    torch.testing.assert_close(
        shifted, torch.tensor([[_PAD, 1, 2, 3, _PAD, 5, 6, _PAD, 8]], dtype=torch.int64)
    )

    shifted_two = shift_right_within_segments(tokens, 2, _PAD, starts)
    torch.testing.assert_close(
        shifted_two,
        torch.tensor([[_PAD, _PAD, 1, 2, _PAD, _PAD, 5, _PAD, _PAD]], dtype=torch.int64),
    )


def test_packed_row_hashes_match_the_documents_standing_alone():
    """The deepseek convention (pad at document start) must be per-document under packing."""
    torch.manual_seed(0)
    doc_a = torch.randint(1, 16, (1, 11), dtype=torch.int64)
    doc_b = torch.randint(1, 16, (1, 9), dtype=torch.int64)
    packed = torch.cat([doc_a, doc_b], dim=1)
    cu_seqlens = torch.tensor([0, doc_a.shape[1], packed.shape[1]], dtype=torch.int32)

    packed_hashes = _hash(packed, cu_seqlens=cu_seqlens)

    # Each document inside the packed row hashes exactly as it does on its own.
    torch.testing.assert_close(packed_hashes[:, : doc_a.shape[1]], _hash(doc_a))
    torch.testing.assert_close(packed_hashes[:, doc_a.shape[1] :], _hash(doc_b))


def test_without_cu_seqlens_the_window_does_reach_across():
    """Guards the test above: without boundaries the second document is contaminated."""
    torch.manual_seed(0)
    doc_a = torch.randint(1, 16, (1, 11), dtype=torch.int64)
    doc_b = torch.randint(1, 16, (1, 9), dtype=torch.int64)
    packed = torch.cat([doc_a, doc_b], dim=1)

    unpacked_hashes = _hash(packed)
    assert not torch.equal(unpacked_hashes[:, doc_a.shape[1] :], _hash(doc_b))


def test_eos_and_cu_seqlens_boundaries_compose():
    """A row can be packed and carry EOS; either boundary alone must still stop a window."""
    eos = 5
    tokens = torch.tensor([[1, 2, eos, 3, 4, 6, 7, 8]], dtype=torch.int64)
    # cu_seqlens splits at 5, EOS splits at 3 -> position 3 and position 5 both start a segment.
    cu_seqlens = torch.tensor([0, 5, 8], dtype=torch.int32)

    both = build_ngram_hashes(
        input_ids=tokens,
        tokenizer_remap=None,
        multipliers=_MULTIPLIERS,
        table_sizes=_TABLE_SIZES,
        max_ngram_order=_ORDER,
        num_hash_heads=_HEADS,
        boundary_token_id=eos,
        reset_at_boundary=True,
        cu_seqlens=cu_seqlens,
    )
    # Position 5 begins a packed document: its 2-gram window is filled, not taken from pos 4.
    only_eos = build_ngram_hashes(
        input_ids=tokens,
        tokenizer_remap=None,
        multipliers=_MULTIPLIERS,
        table_sizes=_TABLE_SIZES,
        max_ngram_order=_ORDER,
        num_hash_heads=_HEADS,
        boundary_token_id=eos,
        reset_at_boundary=True,
        cu_seqlens=None,
    )
    assert not torch.equal(both[:, 5], only_eos[:, 5])
    # Position 3 begins an EOS-delimited document and is unaffected by the packed boundary.
    torch.testing.assert_close(both[:, 3], only_eos[:, 3])
