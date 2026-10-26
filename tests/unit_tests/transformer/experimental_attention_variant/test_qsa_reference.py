# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QSA reference-oracle test matrix (QSA-1).

Pure torch, no Megatron imports, CPU-capable. Covers: dense degeneration
(seq <= budget), exact-multiple-of-4 vs ragged-tail lengths, zero-score tie
avalanches (deterministic top-k), THD multi-document boundaries, real sparsity
(seq > 2.5k with the production budget), repeat-run/recompute bit-identity,
the own-block-not-force-kept caveat, and main-attention fwd/bwd sanity.
"""

import math

import pytest
import torch

from tests.unit_tests.transformer.experimental_attention_variant.qsa_reference import (
    QSAAttentionParams,
    QSAIndexerParams,
    batched_indexer_selected_mask,
    build_causal_visible,
    build_thd_positions,
    build_thd_visible,
    deterministic_topk_indices,
    reference_attention_forward,
    reference_indexer_selected_sets,
)

HIDDEN = 256  # small proxy hidden size for speed; projection is 256 -> 640


def make_params(**over):
    p = QSAIndexerParams.init_random(hidden_size=HIDDEN, seed=1)
    for k, v in over.items():
        setattr(p, k, v)
    return p


def make_hidden(B, S, seed=2):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, S, HIDDEN, generator=g)


def sets_from_mask(mask):
    """[B, S, S] bool -> per-(b, t) python sets."""
    return [
        [set(torch.nonzero(mask[b, t]).flatten().tolist()) for t in range(mask.shape[1])]
        for b in range(mask.shape[0])
    ]


def assert_set_equal(literal, batched_mask):
    got = sets_from_mask(batched_mask)
    for b, (ref_rows, got_rows) in enumerate(zip(literal, got)):
        for t, (r, g) in enumerate(zip(ref_rows, got_rows)):
            assert r == g, (
                f"selected-set mismatch at (b={b}, t={t}): "
                f"missing={sorted(r - g)[:8]} extra={sorted(g - r)[:8]}"
            )


# ---------------------------------------------------------------------------
# deterministic top-k
# ---------------------------------------------------------------------------


class TestDeterministicTopK:
    def test_ties_break_by_ascending_index(self):
        scores = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
        idx = deterministic_topk_indices(scores, 4).tolist()
        assert idx == [1, 3, 0, 2]

    def test_all_zero_scores_select_prefix(self):
        scores = torch.zeros(1000)
        idx = deterministic_topk_indices(scores, 512).tolist()
        assert idx == list(range(512))

    def test_k_larger_than_n(self):
        scores = torch.tensor([2.0, 1.0])
        assert deterministic_topk_indices(scores, 512).tolist() == [0, 1]

    def test_repeat_runs_bitwise_identical(self):
        g = torch.Generator().manual_seed(3)
        scores = torch.randn(4096, generator=g)
        scores[::3] = 0.0  # inject a tie avalanche
        a = deterministic_topk_indices(scores, 512)
        b = deterministic_topk_indices(scores.clone(), 512)
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# literal (visible-prefix) vs batched (causal-prefix) equivalence
# ---------------------------------------------------------------------------


class TestLiteralVsBatchedBSHD:
    @pytest.mark.parametrize("S", [64, 66, 67, 65])  # multiple-of-4 and every ragged phase
    def test_dense_regime_set_equality(self, S):
        # S << budget: every block selected; exercises pooling/tail plumbing.
        p = make_params()
        h = make_hidden(2, S)
        pos = torch.arange(S).expand(2, S)
        vis = build_causal_visible(S).expand(2, S, S)
        literal = reference_indexer_selected_sets(h, p, pos, vis)
        batched = batched_indexer_selected_mask(h, p)
        assert_set_equal(literal, batched)

    def test_dense_regime_mask_equals_causal(self):
        # seq <= budget => QSA degenerates to dense causal attention.
        p = make_params()
        S = 60
        batched = batched_indexer_selected_mask(make_hidden(1, S), p)
        assert torch.equal(batched[0], build_causal_visible(S))

    @pytest.mark.parametrize("S", [300, 301])
    def test_sparse_regime_small_budget_set_equality(self, S):
        # Reduced budget so selection is real at small scale (fast CPU test).
        p = make_params(token_budget=64)  # block_topk = 16, sparse for t >= 67
        h = make_hidden(2, S, seed=5)
        pos = torch.arange(S).expand(2, S)
        vis = build_causal_visible(S).expand(2, S, S)
        literal = reference_indexer_selected_sets(h, p, pos, vis)
        batched = batched_indexer_selected_mask(h, p)
        assert_set_equal(literal, batched)
        # sanity: sparsity actually happened
        assert batched[0, -1].sum() < S - 1

    def test_tie_avalanche_set_equality(self):
        # Force every relu score to zero: q·k <= 0 for all pairs. With w := -A
        # (A = |randn|), q = -A h, k = -A h ... instead simpler: zero the q rows
        # of the projection so q == 0 -> all scores exactly 0 -> pure tie-break.
        p = make_params(token_budget=64)
        p.index_qk_proj_weight = p.index_qk_proj_weight.clone()
        p.index_qk_proj_weight[: p.n_heads * p.head_dim] = 0.0
        S = 300
        h = make_hidden(1, S, seed=7)
        pos = torch.arange(S).expand(1, S)
        vis = build_causal_visible(S).expand(1, S, S)
        literal = reference_indexer_selected_sets(h, p, pos, vis)
        batched = batched_indexer_selected_mask(h, p)
        assert_set_equal(literal, batched)
        # all-zero scores => first block_topk blocks by ascending id + tail
        t = S - 1  # last query, ragged phase (S-1+1) % 4 == 0 ? S=300 -> m=75
        m = (t + 1) // 4
        k = min(64 // 4, m)
        expect = set(range(k * 4)) | set(range(m * 4, t + 1))
        assert sets_from_mask(batched)[0][t] == expect

    def test_own_block_not_force_kept(self):
        # (t+1) % 4 == 0 -> empty tail; with all-zero scores the own block
        # (id m-1) loses to blocks 0..k-1, so token t does NOT attend to itself.
        p = make_params(token_budget=64)
        p.index_qk_proj_weight = p.index_qk_proj_weight.clone()
        p.index_qk_proj_weight[: p.n_heads * p.head_dim] = 0.0
        S = 300
        batched = batched_indexer_selected_mask(make_hidden(1, S, seed=8), p)
        t = 299  # (299 + 1) % 4 == 0, m = 75 > k = 16
        assert not bool(batched[0, t, t]), "own block must NOT be force-kept (HF semantics)"

    def test_repeat_run_bitwise_identical(self):
        p = make_params(token_budget=64)
        h = make_hidden(1, 301, seed=9)
        a = batched_indexer_selected_mask(h, p)
        b = batched_indexer_selected_mask(h.clone(), p)
        assert torch.equal(a, b)


class TestLiteralVsBatchedTHD:
    @pytest.mark.parametrize(
        "doc_lens", [[5, 4, 20], [8, 3, 1, 12], [40, 2]]
    )  # ragged docs, doc==1, multiple-of-4 docs
    def test_multi_document_set_equality(self, doc_lens):
        p = make_params(token_budget=16)  # block_topk 4: sparsity inside longer docs
        cu = torch.tensor([0] + list(torch.tensor(doc_lens).cumsum(0)))
        T = int(cu[-1])
        h = make_hidden(1, T, seed=11)[0]  # [T, H]
        vis = build_thd_visible(cu).unsqueeze(0)
        pos = build_thd_positions(cu).unsqueeze(0)
        literal = reference_indexer_selected_sets(h.unsqueeze(0), p, pos, vis)
        batched = batched_indexer_selected_mask(h, p, cu_seqlens=cu)
        assert_set_equal(literal, batched)

    def test_blocks_never_cross_documents(self):
        p = make_params(token_budget=16)
        cu = torch.tensor([0, 6, 30])  # doc0 len 6 (ragged), doc1 len 24
        h = make_hidden(1, 30, seed=12)[0]
        batched = batched_indexer_selected_mask(h, p, cu_seqlens=cu)[0]
        # nothing in doc1 may see doc0 and vice versa
        assert not batched[6:, :6].any()
        assert not batched[:6, 6:].any()

    def test_thd_matches_bshd_per_document(self):
        # each packed document must select exactly as it would standalone
        p = make_params(token_budget=16)
        lens = [10, 23]
        cu = torch.tensor([0, 10, 33])
        h = make_hidden(1, 33, seed=13)[0]
        packed = batched_indexer_selected_mask(h, p, cu_seqlens=cu)[0]
        for d in range(2):
            s, e = int(cu[d]), int(cu[d + 1])
            solo = batched_indexer_selected_mask(h[s:e].unsqueeze(0), p)[0]
            assert torch.equal(packed[s:e, s:e], solo)


class TestFullScaleSparsity:
    def test_production_budget_seq_3072(self):
        # Real sparsity with the production budget (2048 tokens / 512 blocks):
        # m_t up to 768 > 512. Set-equality on sampled queries to bound runtime.
        p = make_params()  # token_budget 2048
        S = 3072
        h = make_hidden(1, S, seed=21)
        batched = batched_indexer_selected_mask(h, p)
        # sparsity really engaged: t = 3071 has m = 768 complete blocks, empty
        # tail ((t+1) % 4 == 0), so exactly 512 * 4 = 2048 tokens are selected.
        assert int(batched[0, S - 1].sum()) == 2048

        # full literal-port sweep (slow: O(S) python loop with per-query pooling)
        pos = torch.arange(S).expand(1, S)
        vis = build_causal_visible(S).expand(1, S, S)
        literal = reference_indexer_selected_sets(h, p, pos, vis)
        assert_set_equal(literal, batched)


# ---------------------------------------------------------------------------
# main attention fwd/bwd
# ---------------------------------------------------------------------------


class TestMainAttention:
    def _setup(self, dtype=torch.float32):
        S, B = 48, 2
        ap = QSAAttentionParams.init_random(
            hidden_size=HIDDEN, n_heads=8, n_kv_heads=2, head_dim=32, seed=31
        )
        ap.rotary_dim = 8
        g = torch.Generator().manual_seed(32)
        h = torch.randn(B, S, HIDDEN, generator=g, dtype=dtype, requires_grad=True)
        mask = build_causal_visible(S).expand(B, S, S)
        return h, ap, mask

    def test_matches_sdpa_with_gate_removed(self):
        # Independent check: with gate forced open (sigmoid -> 1 impossible;
        # instead compare against a manual sdpa implementation of the same math).
        h, ap, mask = self._setup()
        out = reference_attention_forward(h, ap, mask)

        # independent re-implementation via F.scaled_dot_product_attention
        B, S, _ = h.shape
        p = ap
        from tests.unit_tests.transformer.experimental_attention_variant.qsa_reference import (
            apply_partial_rope,
            build_rope_cos_sin,
            rms_norm,
        )

        qf = (h @ p.q_proj_weight.t()).view(B, S, p.n_heads, p.head_dim * 2)
        q, gate = torch.chunk(qf, 2, dim=-1)
        q = rms_norm(q, p.q_norm_weight, p.rms_norm_eps)
        k = rms_norm(
            (h @ p.k_proj_weight.t()).view(B, S, p.n_kv_heads, p.head_dim),
            p.k_norm_weight,
            p.rms_norm_eps,
        )
        v = (h @ p.v_proj_weight.t()).view(B, S, p.n_kv_heads, p.head_dim)
        pos = torch.arange(S).expand(B, S)
        cos, sin = build_rope_cos_sin(pos, p.rotary_dim, p.rope_theta, h.dtype)
        q = apply_partial_rope(q, cos.unsqueeze(-2), sin.unsqueeze(-2)).transpose(1, 2)
        k = apply_partial_rope(k, cos.unsqueeze(-2), sin.unsqueeze(-2)).transpose(1, 2)
        v = v.transpose(1, 2)
        o = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask.unsqueeze(1), enable_gqa=True
        )
        o = o.transpose(1, 2).reshape(B, S, -1) * torch.sigmoid(gate.reshape(B, S, -1))
        expect = o @ p.o_proj_weight.t()
        assert torch.allclose(out, expect, atol=2e-5, rtol=2e-5)

    def test_backward_runs_and_grads_finite(self):
        h, ap, mask = self._setup()
        out = reference_attention_forward(h, ap, mask)
        out.square().mean().backward()
        assert h.grad is not None and torch.isfinite(h.grad).all()

    def test_selection_mask_actually_gates(self):
        h, ap, mask = self._setup()
        sparse = mask.clone()
        sparse[:, 32:, 1:16] = False  # drop some visible keys for late queries
        dense_out = reference_attention_forward(h, ap, mask)
        sparse_out = reference_attention_forward(h, ap, sparse)
        assert not torch.allclose(dense_out[:, 32:], sparse_out[:, 32:])
        # early queries (unchanged rows) are bitwise identical
        assert torch.equal(dense_out[:, :16], sparse_out[:, :16])

    def test_fwd_repeat_bitwise(self):
        h, ap, mask = self._setup()
        a = reference_attention_forward(h, ap, mask)
        b = reference_attention_forward(h.detach().clone().requires_grad_(), ap, mask)
        assert torch.equal(a, b)
