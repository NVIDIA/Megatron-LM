# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU tests for the bounded-memory (row-chunked) CSA indexer top-K.

The cuDNN indexer kernels are replaced by a PyTorch reference with the same
contract, so these tests check the chunking logic itself: row slicing,
per-chunk ``cu_seqlens``, causal offsets, padding and score gathering.
Kernel numerics are covered by the GPU tests in
``test_csa_fused_sparse_attention.py``.
"""

import pytest
import torch

import megatron.core.transformer.experimental_attention_variant.csa_utils.fused_sparse_attention as fsa


class ReferenceDSA:
    """PyTorch stand-in for the cudnn-frontend ``DSA`` indexer wrappers.

    ``indexer_forward_wrapper``: ``sum_h relu(q_h . k) * w_h`` with keys at or
    beyond ``(pos + 1) // ratio`` set to ``-inf``. ``indexer_top_k_wrapper``:
    top-K over the first ``seq_lens[row]`` columns, ``-1`` padded.
    Records the largest score block it materialized.
    """

    def __init__(self):
        self.max_score_elems = 0
        self.forward_calls = 0

    def _record(self, scores):
        self.forward_calls += 1
        self.max_score_elems = max(self.max_score_elems, scores.numel())

    def indexer_forward_wrapper(
        self,
        q,
        k,
        w,
        ratio,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        q_causal_offsets=None,
    ):
        if cu_seqlens_q is None:
            # BSHD: q (b, sq, h, d), k (b, sk, 1, d), w (b, sq, h).
            logits = torch.einsum("bshd,btd->bsht", q.float(), k[:, :, 0].float())
            scores = (logits.relu() * w.float().unsqueeze(-1)).sum(dim=2)
            sq, sk = scores.shape[1:]
            valid = (torch.arange(sq).view(sq, 1) + 1) // ratio
            scores = scores.masked_fill(torch.arange(sk).view(1, sk) >= valid, float("-inf"))
            self._record(scores)
            return {"scores": scores}

        # THD: q (total_q, h, d), k (total_k, 1, d), w (total_q, h).
        total_q = q.shape[0]
        scores = torch.full((total_q, int(max_seqlen_k)), float("-inf"))
        num_segments = cu_seqlens_q.numel() - 1
        for seg in range(num_segments):
            q0, q1 = int(cu_seqlens_q[seg]), int(cu_seqlens_q[seg + 1])
            k0, k1 = int(cu_seqlens_k[seg]), int(cu_seqlens_k[seg + 1])
            if q1 <= q0 or k1 <= k0:
                continue
            logits = torch.einsum("shd,td->sht", q[q0:q1].float(), k[k0:k1, 0].float())
            seg_scores = (logits.relu() * w[q0:q1].float().unsqueeze(-1)).sum(dim=1)
            offset = int(q_causal_offsets[seg]) if q_causal_offsets is not None else 0
            positions = torch.arange(q1 - q0).view(-1, 1) + offset
            keys = torch.arange(k1 - k0).view(1, -1)
            seg_scores = seg_scores.masked_fill(keys >= (positions + 1) // ratio, float("-inf"))
            scores[q0:q1, : k1 - k0] = seg_scores
        self._record(scores)
        return {"scores": scores}

    @staticmethod
    def indexer_top_k_wrapper(scores, seq_lens, top_k, next_n=1, return_val=False):
        rows, cols = scores.shape
        keep = torch.arange(cols).view(1, cols) < seq_lens.view(rows, 1).long()
        masked = scores.masked_fill(~keep, float("-inf"))
        values, indices = masked.topk(top_k, dim=-1)
        indices = indices.masked_fill(values == float("-inf"), -1).int()
        return {"indices": indices}


@pytest.fixture
def ref_dsa(monkeypatch):
    stub = ReferenceDSA()
    monkeypatch.setattr(fsa, "_DSA", stub)
    return stub


def _core(*args, **kwargs):
    """Call the dense-fallback core; the compact softmax slot must stay empty."""
    topk_indices, topk_length, scores, compact_softmax = fsa._indexer_topk_core(*args, **kwargs)
    assert compact_softmax is None
    return topk_indices, topk_length, scores


def _force_chunk_rows(monkeypatch, rows, sk):
    """Make the score budget hold exactly ``rows`` query rows."""
    monkeypatch.setattr(fsa, "_CSA_INDEXER_SCORE_CHUNK_MAX_BYTES", rows * sk * 4)
    monkeypatch.setattr(fsa, "_CSA_INDEXER_SCORE_CHUNK_ROW_ALIGNMENT", 1)


def _bshd_inputs(b=2, sq=48, sk=12, nh=4, hd=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, sq, nh, hd, generator=g)
    k = torch.randn(b, sk, hd, generator=g)
    w = torch.randn(b, sq, nh, generator=g)
    return q, k, w


def _thd_inputs(q_lens, k_lens, padding_rows=0, nh=4, hd=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    total_q = sum(q_lens) + padding_rows
    q = torch.randn(total_q, nh, hd, generator=g)
    k = torch.randn(sum(k_lens), hd, generator=g)
    w = torch.randn(total_q, nh, generator=g)
    cu_q = torch.tensor([0] + q_lens, dtype=torch.int32).cumsum(0).int()
    cu_k = torch.tensor([0] + k_lens, dtype=torch.int32).cumsum(0).int()
    return q, k, w, cu_q, cu_k


def _assert_matches_full(full, bounded, *, check_scores):
    full_idx, full_len, full_scores = full
    idx, length, topk_scores = bounded
    assert torch.equal(idx, full_idx)
    assert torch.equal(length, full_len)
    if check_scores:
        rows = full_scores.reshape(-1, full_scores.shape[-1])
        expected = fsa._gather_topk_scores(rows, full_idx.reshape(rows.shape[0], -1))
        # The PyTorch stand-in reduces BSHD and THD blocks in different orders,
        # so compare values to fp32 tolerance (``-inf`` slots must match exactly).
        torch.testing.assert_close(topk_scores.reshape(expected.shape), expected)
    else:
        assert topk_scores is None


@pytest.mark.parametrize("chunk_rows", [1, 5, 16, 47, 96])
@pytest.mark.parametrize("scores_output", ["topk", "none"])
def test_bshd_chunked_matches_single_pass(ref_dsa, monkeypatch, chunk_rows, scores_output):
    q, k, w = _bshd_inputs()
    topk = 6
    full = _core(q, k, w, topk, ratio=4)

    _force_chunk_rows(monkeypatch, chunk_rows, k.shape[1])
    bounded = _core(q, k, w, topk, ratio=4, scores_output=scores_output)

    _assert_matches_full(full, bounded, check_scores=scores_output == "topk")
    assert bounded[0].shape == (2, 48, topk)
    assert bounded[0].dtype == torch.int32


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("chunk_rows", [1, 3, 8, 17, 40, 200])
def test_thd_chunked_matches_single_pass(ref_dsa, monkeypatch, chunk_rows, deterministic):
    # Segments: short (< ratio), long, empty, medium; plus CUDA-graph padding rows.
    q_lens, k_lens = [3, 37, 0, 22], [1, 9, 1, 6]
    q, k, w, cu_q, cu_k = _thd_inputs(q_lens, k_lens, padding_rows=6)
    offsets = torch.tensor([0, 8, 0, 100], dtype=torch.int32)  # CP-style causal offsets
    kwargs = dict(
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(k_lens),
        q_causal_offsets=offsets,
        deterministic=deterministic,
    )
    topk = 8
    full = _core(q, k, w, topk, ratio=4, **kwargs)

    _force_chunk_rows(monkeypatch, chunk_rows, max(k_lens))
    bounded = _core(q, k, w, topk, ratio=4, scores_output="topk", **kwargs)

    _assert_matches_full(full, bounded, check_scores=True)
    # Padding rows (past cu_seqlens_q[-1]) select nothing.
    assert torch.all(bounded[0][sum(q_lens) :] == -1)
    assert torch.all(bounded[1][sum(q_lens) :] == 0)


def test_thd_chunked_matches_independent_reference(ref_dsa, monkeypatch):
    """Check against a from-scratch per-row computation, not only the single-pass path."""
    q_lens, k_lens = [13, 30], [3, 7]
    q, k, w, cu_q, cu_k = _thd_inputs(q_lens, k_lens)
    ratio, topk = 4, 4
    _force_chunk_rows(monkeypatch, 7, max(k_lens))
    idx, length, topk_scores = _core(
        q,
        k,
        w,
        topk,
        ratio,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(k_lens),
        scores_output="topk",
    )
    for seg in range(len(q_lens)):
        q0, k0 = int(cu_q[seg]), int(cu_k[seg])
        for pos in range(q_lens[seg]):
            row = q0 + pos
            n_valid = min((pos + 1) // ratio, k_lens[seg])
            keys = k[k0 : k0 + n_valid]
            scores = ((q[row] @ keys.T).relu() * w[row].unsqueeze(-1)).sum(0)
            n_sel = min(topk, n_valid)
            expected_vals, expected_idx = scores.topk(n_sel)
            assert int(length[row]) == n_sel
            assert idx[row, :n_sel].tolist() == expected_idx.tolist()
            assert torch.all(idx[row, n_sel:] == -1)
            assert torch.allclose(topk_scores[row, :n_sel], expected_vals)
            assert torch.all(topk_scores[row, n_sel:] == float("-inf"))


def test_peak_score_block_is_bounded(ref_dsa, monkeypatch):
    q, k, w = _bshd_inputs(b=2, sq=256, sk=64)
    sk = k.shape[1]
    _core(q, k, w, 8, ratio=4)
    full_elems = ref_dsa.max_score_elems
    assert full_elems == 2 * 256 * sk

    ref_dsa.max_score_elems = 0
    ref_dsa.forward_calls = 0
    _force_chunk_rows(monkeypatch, 32, sk)
    _core(q, k, w, 8, ratio=4, scores_output="topk")
    assert ref_dsa.max_score_elems <= 32 * sk
    assert ref_dsa.max_score_elems * 16 == full_elems
    assert ref_dsa.forward_calls == 2 * 256 // 32


def test_small_problem_uses_single_pass(ref_dsa):
    q, k, w = _bshd_inputs()
    _core(q, k, w, 6, ratio=4, scores_output="topk")
    assert ref_dsa.forward_calls == 1


def test_chunk_rows_respects_budget_and_alignment(monkeypatch):
    monkeypatch.setattr(fsa, "_CSA_INDEXER_SCORE_CHUNK_MAX_BYTES", 1024 * 1024 * 1024)
    monkeypatch.setattr(fsa, "_CSA_INDEXER_SCORE_CHUNK_ROW_ALIGNMENT", 512)
    # 1M-token context, ratio 4 -> 262144 compressed keys: 1 GiB holds 1024 rows.
    assert fsa._indexer_score_chunk_rows(1 << 20, 1 << 18) == 1024
    assert fsa._indexer_score_chunk_rows(100, 1 << 18) == 100
    # Budget below one alignment unit falls back to the raw row count.
    assert fsa._indexer_score_chunk_rows(1 << 20, 3 << 20) == 85
    assert fsa._indexer_score_chunk_rows(10, 1 << 40) == 1


def test_sparse_predict_matches_full_score_loss_preparation():
    """Chunked path's predict equals ``prepare_sparse_loss`` on the full score matrix."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import (
        csa_indexer_loss_kernels,
    )

    g = torch.Generator().manual_seed(3)
    rows, sk, topk = 9, 20, 5
    full_scores = torch.randn(rows, sk, generator=g)
    topk_indices = full_scores.topk(topk, dim=-1).indices.int()
    topk_indices[2, 3:] = -1  # short row
    padding = torch.zeros(rows, dtype=torch.bool)
    padding[6] = True
    physical = topk_indices + 100

    expected_predict, expected_idx, expected_phys = (
        csa_indexer_loss_kernels._prepare_sparse_loss_fallback(
            full_scores, topk_indices, padding, physical
        )
    )

    # What FusedCSAIndexerSparseAttnFunc does with the chunked scores.
    selected = fsa._gather_topk_scores(full_scores, topk_indices)
    row_mask = padding.unsqueeze(-1)
    masked_idx = topk_indices.masked_fill(row_mask, -1)
    masked_phys = physical.masked_fill(row_mask, -1)
    predict = fsa._sparse_indexer_predict(selected, masked_idx)

    assert torch.equal(masked_idx, expected_idx)
    assert torch.equal(masked_phys, expected_phys)
    torch.testing.assert_close(predict, expected_predict, rtol=0, atol=0)


def test_rejects_unknown_scores_output():
    q, k, w = _bshd_inputs()
    with pytest.raises(ValueError, match="scores_output"):
        _core(q, k, w, 4, scores_output="dense")
