# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Focused CPU tests for QSA's tiled, document-relative block selection."""

import math
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import qsa


def _reference_bits(query, pooled_keys, doc_ids, positions, ratio, topk):
    """Score each query against only its own visible complete blocks."""
    tokens, _, head_dim = query.shape
    max_blocks = pooled_keys.shape[1]
    nbytes = (max_blocks + 8) // 8
    expected = torch.zeros(tokens, nbytes, dtype=torch.uint8, device=query.device)
    for token in range(tokens):
        visible = min((int(positions[token]) + 1) // ratio, max_blocks)
        if visible <= topk:
            chosen = range(visible)
        else:
            scores = torch.matmul(query[token].float(), pooled_keys[int(doc_ids[token])].float().T)
            scores = scores.relu().sum(0) / math.sqrt(head_dim)
            scores[visible:] = float("-inf")
            chosen = torch.topk(scores, k=min(topk, max_blocks)).indices.tolist()
        for block in chosen:
            expected[token, block // 8] |= 1 << (block % 8)
    return expected


@pytest.mark.parametrize("lengths", [[1, 15, 37], [16, 0, 29]])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_select_blocks_matches_uneven_packed_documents_across_tiles(monkeypatch, lengths, device):
    """Document boundaries, an empty document and incomplete tails stay independent."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    monkeypatch.setattr(qsa, "_QSA_SELECT_TILE_BYTES", 256)
    generator = torch.Generator().manual_seed(21)
    ratio, topk, heads, head_dim = 4, 2, 2, 3
    doc_ids = torch.repeat_interleave(
        torch.arange(len(lengths), dtype=torch.int32), torch.tensor(lengths)
    )
    positions = torch.cat([torch.arange(length, dtype=torch.int32) for length in lengths])
    max_blocks = max(lengths) // ratio
    query = torch.randint(-3, 4, (sum(lengths), heads, head_dim), generator=generator).float()
    pooled = torch.randint(-3, 4, (len(lengths), max_blocks, head_dim), generator=generator).float()
    valid = torch.arange(max_blocks)[None, :] < (torch.tensor(lengths) // ratio)[:, None]
    doc_ids, positions, query, pooled, valid = (
        tensor.to(device) for tensor in (doc_ids, positions, query, pooled, valid)
    )
    indexer = SimpleNamespace(compress_ratio=ratio, block_topk=topk)

    actual, all_selected = qsa.QSAIndexer._select_blocks(
        indexer, query, pooled, valid, doc_ids, positions
    )
    expected = _reference_bits(query, pooled, doc_ids, positions, ratio, topk)
    assert not all_selected
    assert actual.dtype == torch.uint8
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_select_blocks_ties_do_not_depend_on_tile_size(monkeypatch):
    """Tied top-k block IDs remain the same when a document is split into tiles."""
    ratio, topk, nblocks = 2, 3, 11
    lengths = [24, 25]
    doc_ids = torch.repeat_interleave(torch.arange(2, dtype=torch.int32), torch.tensor(lengths))
    positions = torch.cat([torch.arange(length, dtype=torch.int32) for length in lengths])
    query = torch.zeros(sum(lengths), 2, 4)
    pooled = torch.randn(2, nblocks, 4, generator=torch.Generator().manual_seed(22))
    valid = torch.ones(2, nblocks, dtype=torch.bool)
    indexer = SimpleNamespace(compress_ratio=ratio, block_topk=topk)

    monkeypatch.setattr(qsa, "_QSA_SELECT_TILE_BYTES", 128)
    tiled, tiled_all = qsa.QSAIndexer._select_blocks(
        indexer, query, pooled, valid, doc_ids, positions
    )
    monkeypatch.setattr(qsa, "_QSA_SELECT_TILE_BYTES", 64 << 20)
    untiled, untiled_all = qsa.QSAIndexer._select_blocks(
        indexer, query, pooled, valid, doc_ids, positions
    )
    assert not tiled_all and not untiled_all
    torch.testing.assert_close(tiled, untiled, atol=0, rtol=0)
    torch.testing.assert_close(
        tiled, _reference_bits(query, pooled, doc_ids, positions, ratio, topk), atol=0, rtol=0
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_select_blocks_many_short_packed_documents_use_bounded_tiles(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    monkeypatch.setattr(qsa, "_QSA_SELECT_TILE_BYTES", 32 << 10)
    lengths = [16 + (doc % 3) - 1 for doc in range(32)]
    lengths[9] = 0
    ratio, topk, heads, head_dim = 4, 2, 4, 8
    doc_ids = torch.repeat_interleave(
        torch.arange(len(lengths), dtype=torch.int32), torch.tensor(lengths)
    )
    positions = torch.cat([torch.arange(length, dtype=torch.int32) for length in lengths])
    generator = torch.Generator().manual_seed(34)
    query = torch.randint(-3, 4, (sum(lengths), heads, head_dim), generator=generator).float()
    pooled = torch.randint(-3, 4, (len(lengths), 4, head_dim), generator=generator).float()
    valid = torch.arange(4)[None, :] < (torch.tensor(lengths) // ratio)[:, None]
    doc_ids, positions, query, pooled, valid = (
        tensor.to(device) for tensor in (doc_ids, positions, query, pooled, valid)
    )
    expected = _reference_bits(query, pooled, doc_ids, positions, ratio, topk)

    def fail_if_document_loop(*_args, **_kwargs):
        raise AssertionError("short packed documents should use the tiled batched path")

    monkeypatch.setattr(torch, "searchsorted", fail_if_document_loop)
    actual, all_selected = qsa.QSAIndexer._select_blocks(
        SimpleNamespace(compress_ratio=ratio, block_topk=topk),
        query,
        pooled,
        valid,
        doc_ids,
        positions,
    )
    assert not all_selected
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "doc_count,length,workspace,uniform",
    [
        (1, 37, 64, False),
        (2, 16, 32 << 10, False),
        (7, 16, 32 << 10, False),
        (2, 64, 4 << 10, True),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_select_blocks_avoids_offset_sync_for_known_document_layouts(
    monkeypatch, doc_count, length, workspace, uniform, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    monkeypatch.setattr(qsa, "_QSA_SELECT_TILE_BYTES", workspace)
    ratio, topk, heads, head_dim = 4, 2, 2, 4
    nblocks = length // ratio
    doc_ids = torch.arange(doc_count, dtype=torch.int32).repeat_interleave(length).to(device)
    positions = torch.arange(length, dtype=torch.int32).repeat(doc_count).to(device)
    generator = torch.Generator().manual_seed(35)
    query = (
        torch.randint(-3, 4, (doc_count * length, heads, head_dim), generator=generator)
        .float()
        .to(device)
    )
    pooled = (
        torch.randint(-3, 4, (doc_count, nblocks, head_dim), generator=generator).float().to(device)
    )
    valid = torch.ones(doc_count, nblocks, dtype=torch.bool, device=device)
    expected = _reference_bits(query, pooled, doc_ids, positions, ratio, topk)

    def fail_if_offset_sync(*_args, **_kwargs):
        raise AssertionError("offset search would require an additional GPU-to-CPU sync")

    monkeypatch.setattr(torch, "searchsorted", fail_if_offset_sync)
    actual, all_selected = qsa.QSAIndexer._select_blocks(
        SimpleNamespace(compress_ratio=ratio, block_topk=topk),
        query,
        pooled,
        valid,
        doc_ids,
        positions,
        length if uniform else None,
    )
    assert not all_selected
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("nblocks", [0, 3])
def test_select_blocks_all_visible_path_is_tiled(monkeypatch, nblocks):
    monkeypatch.setattr(qsa, "_QSA_SELECT_TILE_BYTES", 64)
    ratio, topk = 4, 3
    positions = torch.arange(14, dtype=torch.int32)
    doc_ids = torch.zeros(14, dtype=torch.int32)
    query = torch.randn(14, 2, 3)
    pooled = torch.randn(1, nblocks, 3)
    valid = torch.ones(1, nblocks, dtype=torch.bool)
    indexer = SimpleNamespace(compress_ratio=ratio, block_topk=topk)

    actual, all_selected = qsa.QSAIndexer._select_blocks(
        indexer, query, pooled, valid, doc_ids, positions
    )
    assert all_selected
    torch.testing.assert_close(
        actual, _reference_bits(query, pooled, doc_ids, positions, ratio, topk), atol=0, rtol=0
    )
