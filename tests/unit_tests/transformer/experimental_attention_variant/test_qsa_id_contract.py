# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU contract probe for retaining QSA TopK IDs before bitset scatter."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.qsa import (
    QSAIndexer,
    QSASelection,
    build_qsa_dense_mask,
)
from megatron.core.transformer.experimental_attention_variant.qsa_id_sparse import (
    validate_qsa_block_ids,
)


def _indexer_inputs(rows, ratio):
    seq_len = sum(rows[0])
    assert all(sum(row) == seq_len for row in rows)
    doc_lengths = [length for row in rows for length in row]
    doc_ids = torch.cat(
        [torch.full((length,), doc, dtype=torch.int32) for doc, length in enumerate(doc_lengths)]
    )
    positions = torch.cat([torch.arange(length, dtype=torch.int32) for length in doc_lengths])
    nblocks = max(doc_lengths) // ratio
    generator = torch.Generator().manual_seed(208)
    q = torch.rand(doc_ids.numel(), 2, 8, generator=generator) + 0.1
    pooled = torch.rand(len(doc_lengths), nblocks, 8, generator=generator) + 0.1
    block_valid = torch.arange(nblocks)[None] < (torch.tensor(doc_lengths) // ratio)[:, None]
    return seq_len, doc_ids, positions, q, pooled, block_valid


@pytest.mark.parametrize(
    "rows,ratio,topk,uniform",
    [
        ([[57]], 4, 2, None),
        ([[31, 26]], 4, 3, None),
        ([[1, 128]], 4, 3, None),
        ([[3, 5, 7]], 4, 4, None),
        ([[57], [57]], 4, 2, 57),
        ([[33, 24]], 2, 4, None),
        ([[33, 24]], 8, 2, None),
    ],
)
def test_qsa_selected_ids_match_original_bits_and_exact_mask(rows, ratio, topk, uniform):
    seq_len, doc_ids, positions, q, pooled, block_valid = _indexer_inputs(rows, ratio)
    indexer = SimpleNamespace(compress_ratio=ratio, block_topk=topk)
    args = (q, pooled, block_valid, doc_ids, positions)
    bits, bits_all_selected = QSAIndexer._select_blocks(indexer, *args, uniform_doc_len=uniform)
    ids, ids_all_selected = QSAIndexer._select_blocks(
        indexer, *args, uniform_doc_len=uniform, output_format="ids"
    )
    assert ids_all_selected == bits_all_selected
    assert ids.dtype == torch.int32 and ids.shape == (len(rows) * seq_len, topk)
    validate_qsa_block_ids(
        ids.reshape(len(rows), seq_len, topk), positions.reshape(len(rows), seq_len), ratio
    )

    reconstructed_bits = torch.zeros_like(bits)
    for row in range(ids.shape[0]):
        for block in ids[row].tolist():
            if block >= 0:
                reconstructed_bits[row, block // 8] |= 1 << (block & 7)
    torch.testing.assert_close(reconstructed_bits, bits, atol=0, rtol=0)

    # The dense mask is deliberately used only at these tiny CPU lengths.
    selection = QSASelection(
        doc_ids=doc_ids.view(1, -1),
        positions=positions.view(1, -1),
        selected_bits=bits.flatten(),
        bits_per_row=bits.shape[1],
        bits_per_row_t=torch.tensor(bits.shape[1]),
        compress_ratio=ratio,
        all_selected=bits_all_selected,
    )
    exact_bits_mask = build_qsa_dense_mask(selection, doc_ids.numel())[0]
    exact_ids_mask = torch.zeros_like(exact_bits_mask)
    for row in range(ids.shape[0]):
        pos = int(positions[row])
        start = row - pos
        for block in ids[row].tolist():
            if block >= 0:
                exact_ids_mask[row, start + block * ratio : start + (block + 1) * ratio] = True
        tail_start = ((pos + 1) // ratio) * ratio
        exact_ids_mask[row, start + tail_start : row + 1] = True
    torch.testing.assert_close(exact_ids_mask, exact_bits_mask, atol=0, rtol=0)
