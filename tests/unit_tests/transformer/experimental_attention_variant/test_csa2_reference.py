# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU tests for the framework-free CSA2 reference operators."""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa2.reference import (
    candidate_blocks_to_mask,
    compressed_causal_mask,
    compressed_visible_counts,
    concat_window_and_compressed_indices,
    dense_reference_attention,
    indexer_scores,
    indexer_topk_indices,
    pool_groups_with_softmax_gate,
    select_candidate_block_ids,
    select_candidate_blocks,
    sliding_window_indices,
    sparse_attention_with_sink,
)

torch.manual_seed(0)


class TestIndices:
    def test_sliding_window(self):
        idx = sliding_window_indices(6, 4)
        assert idx.shape == (6, 4) and idx.dtype == torch.int32
        for i in range(6):
            row = set(idx[i].tolist()) - {-1}
            assert row == set(range(max(0, i - 3), i + 1))
        assert (idx[0] == torch.tensor([0, -1, -1, -1], dtype=torch.int32)).all()

    def test_visible_counts_and_mask(self):
        counts = compressed_visible_counts(9, 2)
        assert counts.tolist() == [(i + 1) // 2 for i in range(9)]
        mask = compressed_causal_mask(9, 4, 2)
        assert mask.shape == (9, 4)
        # position 1 has completed group 0 (tokens 0-1); position 0 sees nothing
        assert not mask[0].any()
        assert mask[1].tolist() == [True, False, False, False]
        assert mask[8].tolist() == [True, True, True, True]
        # ratio 1: every entry up to and including self is visible
        counts1 = compressed_visible_counts(5, 1)
        assert counts1.tolist() == [1, 2, 3, 4, 5]

    def test_concat_indices(self):
        window = sliding_window_indices(4, 2)
        comp = torch.tensor([[[-1, -1]], [[0, -1]], [[0, 1]], [[1, -1]]], dtype=torch.int32)
        merged = concat_window_and_compressed_indices(window, comp, n_window_keys=4, batch_size=1)
        assert merged.shape == (4, 1, 4)
        assert merged[2, 0].tolist() == [2, 1, 4, 5]
        assert merged[1, 0].tolist() == [1, 0, 4, -1]
        no_comp = concat_window_and_compressed_indices(window, None, 4, 3)
        assert no_comp.shape == (4, 3, 2)


class TestPooling:
    def test_softmax_gate_matches_loop(self):
        s, b, d, r = 7, 2, 5, 3
        values = torch.randn(s, b, d)
        gates = torch.randn(s, b, d)
        pooled = pool_groups_with_softmax_gate(values, gates, r)
        assert pooled.shape == (2, b, d)
        for g in range(2):
            v = values[g * r : (g + 1) * r]
            w = torch.softmax(gates[g * r : (g + 1) * r], dim=0)
            torch.testing.assert_close(pooled[g], (v * w).sum(0))


class TestCandidateBlocks:
    def test_keeps_newest_block_and_topk(self):
        n, block, keep = 16, 4, 2
        scores = torch.full((1, 1, n), -float("inf"))
        visible = 10  # positions 0..9 reachable
        scores[..., :visible] = torch.arange(visible, dtype=torch.float32)
        scores[..., 4:8] = 100.0  # block 1 dominates
        mask = select_candidate_blocks(scores, visible, keep, block)
        assert mask.shape == scores.shape
        blocks = mask.view(1, 1, n // block, block).all(-1)[0, 0].tolist()
        # block 2 holds the newest position (9) and is pinned, block 1 has the top score
        assert blocks == [False, True, True, False]

    def test_unreachable_blocks_dropped(self):
        n, block = 8, 4
        scores = torch.full((1, 1, n), -float("inf"))
        scores[..., :2] = 1.0
        mask = select_candidate_blocks(scores, 2, topk_blocks=2, block_size=block)
        assert mask[..., :4].all() and not mask[..., 4:].any()

    def test_broadcast_visible_counts(self):
        s, n, block = 5, 8, 2
        scores = torch.randn(s, 1, n)
        visible = compressed_visible_counts(s, 1)
        scores = scores.masked_fill(
            torch.arange(n).view(1, 1, n) >= visible.view(s, 1, 1), -float("inf")
        )
        mask = select_candidate_blocks(scores, visible.view(s, 1), topk_blocks=1, block_size=block)
        for i in range(s):
            newest = (int(visible[i]) - 1) // block
            assert mask[i, 0, newest * block : (newest + 1) * block].all()


class TestTopk:
    def test_topk_valid_and_sorted(self):
        s, b, n = 6, 2, 5
        scores = torch.randn(s, b, n)
        visible = compressed_visible_counts(s, 1)  # 1..6, but only n=5 entries exist
        scores = scores.masked_fill(
            torch.arange(n).view(1, 1, n) >= visible.view(s, 1, 1), -float("inf")
        )
        topk = indexer_topk_indices(scores, visible, 3)
        assert topk.shape == (s, b, 3) and topk.dtype == torch.int32
        for i in range(s):
            for j in range(b):
                row = topk[i, j].tolist()
                valid = [x for x in row if x >= 0]
                assert valid == sorted(valid)
                assert all(x < visible[i] for x in valid)
                assert len(valid) == min(3, int(visible[i]), n)

    def test_topk_zero_entries(self):
        scores = torch.empty(3, 1, 0)
        topk = indexer_topk_indices(scores, torch.zeros(3, dtype=torch.long), 4)
        assert topk.shape == (3, 1, 0)

    def test_indexer_scores_shape(self):
        q = torch.randn(4, 2, 3, 8)
        k = torch.randn(5, 2, 8)
        w = torch.rand(4, 2, 3)
        out = indexer_scores(q, k, w)
        assert out.shape == (4, 2, 5)
        expected = (torch.relu(torch.einsum("sbhd,nbd->sbhn", q, k)) * w.unsqueeze(-1)).sum(2)
        torch.testing.assert_close(out, expected)

    def test_indexer_scores_rows_matches_batched(self):
        from megatron.core.transformer.experimental_attention_variant.csa2.indexer import (
            _rows_within_budget,
            indexer_scores_rows,
        )

        q = torch.randn(7, 3, 8)
        k = torch.randn(5, 8)
        w = torch.rand(7, 3)
        rows = indexer_scores_rows(q, k.t(), w)
        batched = indexer_scores(q.unsqueeze(1), k.unsqueeze(1), w.unsqueeze(1)).squeeze(1)
        torch.testing.assert_close(rows, batched)
        # budget: 1 GiB over 128K fp32 keys (three buffers) leaves a few hundred rows
        assert _rows_within_budget(131072) == (1 << 30) // (3 * 4 * 131072)
        assert _rows_within_budget(1 << 40) == 1


class TestSparseAttention:
    @pytest.mark.parametrize("ratio", [1, 2])
    def test_matches_dense_masked_attention(self, ratio):
        s, b, h, d, window = 12, 2, 3, 8, 4
        n_comp = s // ratio
        query = torch.randn(s, b, h, d)
        window_keys = torch.randn(s, b, d)
        comp_keys = torch.randn(n_comp, b, d)
        sink = torch.randn(h)
        scale = d**-0.5

        window_idx = sliding_window_indices(s, window)
        visible = compressed_visible_counts(s, ratio)
        # take every visible compressed entry (top-k == all)
        scores = torch.zeros(s, b, n_comp)
        scores = scores.masked_fill(
            torch.arange(n_comp).view(1, 1, -1) >= visible.view(s, 1, 1), -float("inf")
        )
        topk = indexer_topk_indices(scores, visible, n_comp)
        indices = concat_window_and_compressed_indices(window_idx, topk, s, b)
        keys = torch.cat([window_keys, comp_keys], dim=0)
        sparse = sparse_attention_with_sink(query, keys, sink, indices, scale)

        allowed = torch.zeros(s, s + n_comp, dtype=torch.bool)
        for i in range(s):
            allowed[i, max(0, i - window + 1) : i + 1] = True
            allowed[i, s : s + int(visible[i])] = True
        dense = dense_reference_attention(query, keys, sink, allowed, scale)
        torch.testing.assert_close(sparse, dense, rtol=1e-5, atol=1e-5)

    def test_all_invalid_row_is_zero(self):
        query = torch.randn(2, 1, 2, 4)
        keys = torch.randn(3, 1, 4)
        idx = torch.full((2, 1, 3), -1, dtype=torch.int32)
        out = sparse_attention_with_sink(query, keys, torch.zeros(2), idx, 1.0)
        assert torch.isfinite(out).all() and (out == 0).all()

    def test_sink_absorbs_mass(self):
        query = torch.randn(1, 1, 1, 4)
        keys = torch.randn(1, 1, 4)
        idx = torch.zeros(1, 1, 1, dtype=torch.int32)
        small = sparse_attention_with_sink(query, keys, torch.tensor([-50.0]), idx, 1.0)
        large = sparse_attention_with_sink(query, keys, torch.tensor([50.0]), idx, 1.0)
        torch.testing.assert_close(small[0, 0, 0], keys[0, 0])
        assert large.abs().max() < 1e-6


class TestTopkCandidateInteraction:
    def test_masked_positions_are_not_resurrected(self):
        # 6 reachable positions, only two finite (candidate-kept) scores, top-k 4
        neg = -float("inf")
        scores = torch.tensor([[[10.0, 9.0, neg, neg, neg, neg]]])
        visible = torch.tensor([6])
        topk = indexer_topk_indices(scores, visible, 4)
        kept = sorted(x for x in topk[0, 0].tolist() if x >= 0)
        assert kept == [0, 1]
        assert (topk[0, 0] == -1).sum() == 2


class TestCandidateBlockIds:
    def test_ids_match_mask(self):
        s, n, block, keep = 5, 16, 4, 2
        scores = torch.randn(s, 1, n)
        visible = torch.tensor([3, 7, 9, 12, 16])
        unreachable = torch.arange(n).view(1, 1, n) >= visible.view(s, 1, 1)
        scores = scores.masked_fill(unreachable, -float("inf"))
        ids = select_candidate_block_ids(scores, visible.view(s, 1), keep, block)
        assert ids.shape == (s, 1, keep) and ids.dtype == torch.int32
        mask = candidate_blocks_to_mask(ids, n, block)
        expected = select_candidate_blocks(scores, visible.view(s, 1), keep, block)
        torch.testing.assert_close(mask, expected)
        # row 0 sees one block only: the second slot is padding
        assert ids[0, 0].tolist()[1] == -1

    def test_padding_when_topk_exceeds_blocks(self):
        scores = torch.zeros(1, 1, 6)
        ids = select_candidate_block_ids(scores, 6, topk_blocks=5, block_size=3)
        assert ids.shape == (1, 1, 5)
        assert sorted(x for x in ids[0, 0].tolist() if x >= 0) == [0, 1]
