# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts of the MSA kernel helpers (``primitive.kernels.msa_kernels``).

The selection table, the block-level BlockMask metadata and the backend dispatch are
pure torch and must hold without a GPU; the flex kernel itself is covered by
``test_msa_unit.py``.
"""

from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.kernels import msa_kernels as mk


def test_block_indices_to_table_marks_selected_blocks_and_ignores_padding():
    idx = torch.tensor([[[[0, 2, -1], [1, -1, -1]]]])  # [B=1, H=1, S=2, K=3]
    table = mk.block_indices_to_table(idx, n_kv_blocks=4)
    assert table.shape == (1, 1, 2, 4)
    assert table[0, 0, 0].tolist() == [True, False, True, False]
    assert table[0, 0, 1].tolist() == [False, True, False, False]


def test_flex_block_mask_lists_the_union_of_selected_kv_blocks():
    pytest.importorskip("torch.nn.attention.flex_attention")
    block = 4
    S = 8  # two query blocks
    pos = torch.arange(S).unsqueeze(0)
    # query block 0 (tokens 0-3) selects kv block 0; query block 1 selects kv blocks 0 and 1 (token 5 also 1)
    idx = torch.full((1, 1, S, 2), -1, dtype=torch.int64)
    idx[0, 0, :4, 0] = 0
    idx[0, 0, 4:, 0] = 1
    idx[0, 0, 5, 1] = 0
    table = mk.block_indices_to_table(idx, n_kv_blocks=2)
    mask = mk.build_flex_block_mask(table, pos, num_q_heads=2, key_length=S, block_size=block)
    kv_num = mask.kv_num_blocks[0]  # [H_q, n_q_blocks]
    assert kv_num.tolist() == [[1, 2], [1, 2]]
    first_listed = mask.kv_indices[0, :, :, 0]
    assert first_listed.tolist() == [[0, 0], [0, 0]]


@pytest.mark.parametrize("backend", ["dense", "sdpa", ""])
def test_core_attention_rejects_unknown_backend(backend):
    q = torch.zeros(1, 2, 4, 8)
    k = torch.zeros(1, 1, 4, 8)
    idx = torch.zeros(1, 1, 4, 1, dtype=torch.int64)
    pos = torch.arange(4).unsqueeze(0)
    with pytest.raises(ValueError, match="unknown MSA backend"):
        mk.msa_core_attention(q, k, k, idx, pos, block_size=4, backend=backend)


def test_core_attention_refuses_magi_backend():
    q = torch.zeros(1, 2, 4, 8)
    k = torch.zeros(1, 1, 4, 8)
    idx = torch.zeros(1, 1, 4, 1, dtype=torch.int64)
    pos = torch.arange(4).unsqueeze(0)
    with pytest.raises(ValueError, match="magi"):
        mk.msa_core_attention(q, k, k, idx, pos, block_size=4, backend="magi")


@pytest.mark.parametrize(
    "seq_len,block,topk,degenerate",
    [(512, 128, 4, True), (500, 128, 4, True), (513, 128, 4, False), (4096, 128, 16, False)],
)
def test_degenerate_predicate(seq_len, block, topk, degenerate):
    assert mk.msa_is_degenerate(seq_len, block, topk) is degenerate
