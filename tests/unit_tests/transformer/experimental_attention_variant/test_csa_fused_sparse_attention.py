# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Focused tests for the SBHD fused CSA integration surface."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    fused_sparse_attention as fused_csa,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa_teacher_lse import (
    can_use_fused_csa_teacher_lse,
    fused_csa_teacher_lse,
)


def test_local_to_global_flat_preserves_sbhd_row_order_and_invalid_entries():
    local = torch.tensor([[[0, -1], [2, 3], [4, -1]], [[1, 2], [0, -1], [3, 4]]], dtype=torch.int64)

    actual = fused_csa.local_to_global_flat(local, batch_size=2)
    expected = torch.tensor([[0, -1], [3, 5], [4, 6], [1, -1], [8, -1], [7, 9]], dtype=torch.int32)

    assert torch.equal(actual, expected)


def test_build_flat_topk_idxs_compacts_valid_entries_on_cpu():
    window = torch.tensor([[[0, -1, 2], [1, -1, -1]]], dtype=torch.int32)
    compressed = torch.tensor([[[4, -1], [5, 6]]], dtype=torch.int32)

    indices, lengths = fused_csa.build_flat_topk_idxs(
        window, compressed, batch_size=1, compact=True
    )

    assert torch.equal(
        indices, torch.tensor([[0, 2, 4, -1, -1], [1, 5, 6, -1, -1]], dtype=torch.int32)
    )
    assert torch.equal(lengths, torch.tensor([3, 3], dtype=torch.int32))


def test_non_compressed_teacher_lse_includes_window_and_sink():
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.5, 1.0], [1.0, -0.5]]], dtype=torch.float32)
    kv_full = torch.tensor(
        [[1.0, 2.0], [-1.0, 0.5], [0.25, -0.75], [0.5, 0.5]], dtype=torch.float32
    )
    window_indices = torch.tensor([[0, -1], [1, 2]], dtype=torch.int32)
    sink = torch.tensor([0.25, -0.5])

    actual = fused_csa._compute_csa_non_compressed_lse(
        query, kv_full, sink, window_indices, softmax_scale=0.5
    )

    gathered = kv_full.index_select(0, window_indices.clamp_min(0).reshape(-1)).reshape(2, 2, 2)
    logits = torch.einsum("rhd,rkd->rhk", query, gathered) * 0.5
    logits = logits.masked_fill((window_indices < 0).unsqueeze(1), float("-inf"))
    expected = torch.logaddexp(torch.logsumexp(logits, dim=-1), sink.view(1, -1))
    torch.testing.assert_close(actual, expected)

    sink_only = fused_csa._compute_csa_non_compressed_lse(
        query, kv_full, sink, torch.full_like(window_indices, -1), softmax_scale=0.5
    )
    torch.testing.assert_close(sink_only, sink.view(1, -1).expand_as(sink_only))


def test_dense_teacher_lse_uses_all_causal_compressed_keys_in_bounded_chunks(monkeypatch):
    torch.manual_seed(17)
    batch, seqlen_q, heads, dim, seqlen_k, ratio = 1, 8, 2, 3, 4, 2
    query = torch.randn(batch, seqlen_q, heads, dim)
    compressed_kv = torch.randn(batch, seqlen_k, dim)
    non_compressed_lse = torch.randn(batch, seqlen_q, heads)

    # The production limit is 1 GiB. Shrink it here so the test proves that
    # score recomputation is row-chunked instead of allocating B*S*H*K.
    monkeypatch.setattr(fused_csa, "_CSA_TEACHER_LSE_CHUNK_MAX_BYTES", 64)
    original_einsum = torch.einsum
    query_chunk_sizes = []

    def recording_einsum(equation, *operands):
        if equation == "bqhd,bkd->bqhk":
            query_chunk_sizes.append(operands[0].shape[1])
        return original_einsum(equation, *operands)

    monkeypatch.setattr(torch, "einsum", recording_einsum)
    actual = fused_csa._compute_dense_csa_teacher_lse(
        query, compressed_kv, non_compressed_lse, softmax_scale=0.25, ratio=ratio
    )

    scores = original_einsum("bqhd,bkd->bqhk", query, compressed_kv) * 0.25
    key_positions = torch.arange(seqlen_k).view(1, 1, 1, seqlen_k)
    visible_keys = (torch.arange(1, seqlen_q + 1) // ratio).view(1, seqlen_q, 1, 1)
    scores = scores.masked_fill(key_positions >= visible_keys, float("-inf"))
    expected = torch.logaddexp(non_compressed_lse, torch.logsumexp(scores, dim=-1))

    torch.testing.assert_close(actual, expected)
    assert len(query_chunk_sizes) > 1
    assert max(query_chunk_sizes) <= 2


def test_eager_teacher_oracle_matches_window_sink_and_all_compressed_mass():
    torch.manual_seed(29)
    seqlen_q, batch, heads, dim, seqlen_k, ratio = 5, 2, 3, 4, 3, 2
    query = torch.randn(seqlen_q, batch, heads, dim)
    original_kv = torch.randn(seqlen_q, batch, dim)
    compressed_kv = torch.randn(seqlen_k, batch, dim)
    kv_full = torch.cat([original_kv, compressed_kv], dim=0)
    sink = torch.randn(heads)
    window_local = torch.tensor(
        [[[0, -1], [0, 1], [1, 2], [2, 3], [3, 4]], [[0, -1], [0, 1], [0, 2], [1, 3], [2, 4]]],
        dtype=torch.int32,
    )
    window_global = fused_csa.local_to_global_flat(window_local, batch)
    query_flat = query.reshape(seqlen_q * batch, heads, dim)
    kv_full_flat = kv_full.reshape((seqlen_q + seqlen_k) * batch, dim)
    query_bshd = query.permute(1, 0, 2, 3).contiguous()
    compressed_bsd = compressed_kv.permute(1, 0, 2).contiguous()

    non_compressed = fused_csa._compute_csa_non_compressed_lse(
        query_flat, kv_full_flat, sink, window_global, softmax_scale=0.5
    )
    non_compressed = non_compressed.reshape(seqlen_q, batch, heads).permute(1, 0, 2).contiguous()
    actual = fused_csa._compute_dense_csa_teacher_lse(
        query_bshd, compressed_bsd, non_compressed, softmax_scale=0.5, ratio=ratio
    )

    expected = torch.empty(batch, seqlen_q, heads)
    for batch_idx in range(batch):
        for query_idx in range(seqlen_q):
            for head_idx in range(heads):
                logits = [sink[head_idx]]
                for key_idx in window_local[batch_idx, query_idx].tolist():
                    if key_idx >= 0:
                        logits.append(
                            torch.dot(
                                query[query_idx, batch_idx, head_idx],
                                original_kv[key_idx, batch_idx],
                            )
                            * 0.5
                        )
                for key_idx in range((query_idx + 1) // ratio):
                    logits.append(
                        torch.dot(
                            query[query_idx, batch_idx, head_idx], compressed_kv[key_idx, batch_idx]
                        )
                        * 0.5
                    )
                expected[batch_idx, query_idx, head_idx] = torch.logsumexp(
                    torch.stack(logits), dim=0
                )

    torch.testing.assert_close(actual, expected)


def test_production_teacher_fails_fast_instead_of_allocating_eager_scores(monkeypatch):
    monkeypatch.setattr(
        fused_csa, "csa_teacher_lse_unsupported_reason", lambda *_args: "Triton is not available"
    )
    fused_teacher = MagicMock(side_effect=AssertionError("fused teacher should not be invoked"))
    eager_window = MagicMock(side_effect=AssertionError("eager window teacher was invoked"))
    eager_dense = MagicMock(side_effect=AssertionError("eager dense teacher was invoked"))
    monkeypatch.setattr(fused_csa, "fused_csa_teacher_lse", fused_teacher)
    monkeypatch.setattr(fused_csa, "_compute_csa_non_compressed_lse", eager_window)
    monkeypatch.setattr(fused_csa, "_compute_dense_csa_teacher_lse", eager_dense)

    with pytest.raises(RuntimeError, match="Refusing the eager score-matrix fallback"):
        fused_csa._compute_full_csa_teacher_lse(
            torch.empty(1, 2, 3, 4),
            torch.empty(2, 3, 4),
            torch.empty(4, 4),
            torch.empty(1, 2, 4),
            torch.empty(3),
            torch.empty(2, 1, dtype=torch.int32),
            softmax_scale=0.5,
            ratio=4,
        )

    fused_teacher.assert_not_called()
    eager_window.assert_not_called()
    eager_dense.assert_not_called()


def test_indexer_topk_pads_to_requested_width_when_compressed_kv_is_short(monkeypatch):
    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q, k, w, ratio):
            del k, w, ratio
            return {"scores": torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)}

        @staticmethod
        def indexer_top_k_wrapper(scores, seq_lens, top_k, next_n, return_val):
            del scores, seq_lens, next_n, return_val
            return {"indices": torch.arange(top_k, dtype=torch.int32).expand(2, -1).clone()}

    monkeypatch.setattr(fused_csa, "_DSA", FakeDSA)
    q = torch.zeros(1, 2, 2, 4)
    k = torch.zeros(1, 3, 4)
    w = torch.ones(1, 2, 2)

    indices, lengths, _ = fused_csa._indexer_topk_bshd(q, k, w, topk=5, ratio=1)

    assert indices.shape == (1, 2, 5)
    assert torch.equal(indices[..., :3], torch.tensor([[[0, 1, 2], [0, 1, 2]]]))
    assert torch.all(indices[..., 3:] == -1)
    assert torch.equal(lengths, torch.tensor([[3, 3]], dtype=torch.int32))


def test_csa_sparse_attn_flattens_and_restores_sbhd_layout():
    sq, batch, heads, dim, value_dim = 3, 2, 4, 8, 5
    query = torch.randn(sq, batch, heads, dim)
    kv = torch.randn(7, batch, dim)
    sink = torch.zeros(heads)
    indices = torch.zeros(sq * batch, 2, dtype=torch.int32)
    flat_output = torch.randn(sq * batch, heads, value_dim)
    lse = torch.randn(sq * batch, heads)

    with patch.object(
        fused_csa.CSASparseAttnFunc, "apply", return_value=(flat_output, lse, None)
    ) as apply:
        actual = fused_csa.csa_sparse_attn(query, kv, sink, indices, 0.125)

    args = apply.call_args.args
    assert args[0].shape == (sq * batch, heads, dim)
    assert args[1].shape == (7 * batch, dim)
    assert args[3] is indices
    assert actual.shape == (sq, batch, heads * value_dim)
    assert torch.equal(actual, flat_output.reshape(sq, batch, heads * value_dim))


def test_flash_mla_adapter_pads_and_restores_small_head_count(monkeypatch):
    seen = {}

    def fake_flash(q, _kv, _indices, _scale, **kwargs):
        seen["q"] = q.detach().clone()
        seen["sink"] = kwargs["attn_sink"].detach().clone()
        rows, heads, _ = q.shape
        out = torch.arange(rows * heads * 3, dtype=q.dtype).reshape(rows, heads, 3)
        lse = torch.arange(rows * heads, dtype=torch.float32).reshape(rows, heads)
        lse_indexer = lse + 1000
        return out, torch.empty(0), lse, lse_indexer

    monkeypatch.setattr(fused_csa, "_flash_mla_sparse_fwd", fake_flash)
    monkeypatch.setattr(fused_csa, "_get_head_padding", lambda _heads: 64)
    monkeypatch.setattr(fused_csa, "_get_topk_alignment", lambda: 1)

    q = torch.randn(3, 2, 8)
    kv = torch.randn(5, 8)
    indices = torch.zeros(3, 4, dtype=torch.int32)
    sink = torch.tensor([0.25, -0.5])

    out, lse, lse_indexer = fused_csa._csa_fwd_flash_mla(
        q, kv, indices, 0.125, d_v=3, attn_sink=sink, indexer_topk=1
    )

    assert seen["q"].shape == (3, 64, 8)
    assert torch.equal(seen["q"][:, :2], q)
    assert torch.count_nonzero(seen["q"][:, 2:]) == 0
    assert torch.equal(seen["sink"][:2], sink)
    assert torch.all(seen["sink"][2:] == float("-inf"))
    assert out.shape == (3, 2, 3)
    assert lse.shape == (3, 2)
    assert lse_indexer.shape == (3, 2)
    expected_out = torch.arange(3 * 64 * 3, dtype=q.dtype).reshape(3, 64, 3)[:, :2]
    expected_lse = torch.arange(3 * 64, dtype=torch.float32).reshape(3, 64)[:, :2]
    assert torch.equal(out, expected_out)
    assert torch.equal(lse, expected_lse)
    assert torch.equal(lse_indexer, expected_lse + 1000)


def test_attn_target_pads_small_head_count(monkeypatch):
    seen = {}

    class FakeDSA:
        @staticmethod
        def sparse_attn_score_recompute_wrapper(q, _k, lse, indices, _scale, **kwargs):
            seen["q"] = q.detach().clone()
            seen["lse"] = lse.detach().clone()
            seen["qhead_per_kv_head"] = kwargs["qhead_per_kv_head"]
            return {"target": torch.zeros_like(indices, dtype=torch.float32)}

    monkeypatch.setattr(fused_csa, "_DSA", FakeDSA)
    q = torch.randn(1, 3, 2, 8)
    k = torch.randn(1, 4, 8)
    lse = torch.randn(1, 3, 2)
    indices = torch.zeros(1, 3, 4, dtype=torch.int32)

    target = fused_csa._compute_attn_target(q, k, lse, indices, 0.125, 2)

    assert seen["q"].shape == (1, 3, 8, 8)
    assert torch.equal(seen["q"][:, :, :2], q)
    assert torch.count_nonzero(seen["q"][:, :, 2:]) == 0
    assert torch.equal(seen["lse"][:, :, :2], lse)
    assert torch.all(seen["lse"][:, :, 2:] == float("inf"))
    assert seen["qhead_per_kv_head"] == 8
    assert target.shape == indices.shape


def test_dense_kl_ignores_ratio_masked_negative_infinity_scores():
    attn_score = torch.tensor([[[1.0, 0.0, 0.0]]])
    attn_l1norm = torch.tensor([[1.0]])
    index_score = torch.tensor([[[0.0, float("-inf"), float("-inf")]]])
    index_lse = torch.tensor([[0.0]])

    loss = fused_csa._kl_loss_from_dense_scores(
        attn_score, attn_l1norm, index_score, index_lse, loss_coeff=0.5
    )

    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_fused_training_wrapper_uses_csa_namespaced_autograd_function():
    output = torch.randn(2, 1, 4)
    loss = torch.tensor(0.25)
    tensors = [torch.empty(0) for _ in range(7)]

    with patch.object(
        fused_csa.FusedCSAIndexerSparseAttnFunc, "apply", return_value=(output, loss)
    ) as apply:
        actual = fused_csa.fused_csa_indexer_sparse_attn(
            *tensors, indexer_topk=8, ratio=4, softmax_scale=0.125
        )

    apply.assert_called_once()
    assert actual[0] is output
    assert actual[1] is loss


def test_fused_dense_training_uses_full_teacher_and_compact_indices_for_backward(monkeypatch):
    seqlen, batch, heads, dim = 4, 1, 2, 3
    compressed_rows, indexer_heads, indexer_dim = 2, 2, 2
    query = torch.randn(seqlen, batch, heads, dim, requires_grad=True)
    kv_full = torch.randn(seqlen + compressed_rows, batch, dim, requires_grad=True)
    sink = torch.randn(heads, requires_grad=True)
    window = torch.tensor([[[0, -1], [0, 1], [1, 2], [2, 3]]], dtype=torch.int32)
    q_indexer = torch.randn(seqlen, batch, indexer_heads, indexer_dim, requires_grad=True)
    k_indexer = torch.randn(compressed_rows, batch, indexer_dim, requires_grad=True)
    weights = torch.randn(seqlen, batch, indexer_heads, requires_grad=True)
    selected = torch.tensor([[[-1, -1], [-1, -1], [0, -1], [1, 0]]], dtype=torch.int32)
    indexer_scores = torch.zeros(batch, seqlen, compressed_rows)

    monkeypatch.setattr(
        fused_csa,
        "_indexer_topk_bshd",
        lambda *_args, **_kwargs: (
            selected.clone(),
            (selected >= 0).sum(dim=-1).int(),
            indexer_scores.clone(),
        ),
    )

    flash_seen = {}

    def fake_flash(q, kv, indices, scale, **kwargs):
        del scale
        flash_seen["indices"] = indices.detach().clone()
        flash_seen["topk_length"] = kwargs["topk_length"].detach().clone()
        flash_seen["indexer_topk"] = kwargs["indexer_topk"]
        return q.new_zeros(q.shape), q.new_zeros(q.shape[:2]), None

    monkeypatch.setattr(fused_csa, "_csa_fwd_flash_mla", fake_flash)

    teacher_lse = torch.full((batch, seqlen, heads), 1.75)
    teacher_seen = {}

    def fake_fused_teacher(*args, **kwargs):
        teacher_seen["window_indices"] = args[4].detach().clone()
        teacher_seen["batch_size"] = kwargs["batch_size"]
        teacher_seen["seqlen_q"] = kwargs["seqlen_q"]
        return teacher_lse

    monkeypatch.setattr(fused_csa, "csa_teacher_lse_unsupported_reason", lambda *_args: None)
    monkeypatch.setattr(fused_csa, "fused_csa_teacher_lse", fake_fused_teacher)
    monkeypatch.setattr(
        fused_csa,
        "_compute_csa_non_compressed_lse",
        MagicMock(side_effect=AssertionError("production path used eager window teacher")),
    )
    monkeypatch.setattr(
        fused_csa,
        "_compute_dense_csa_teacher_lse",
        MagicMock(side_effect=AssertionError("production path allocated eager dense scores")),
    )

    dense_lse_seen = {}
    backward_seen = {}
    fake_dsa = SimpleNamespace()

    def fake_dense_attn(q, k, lse, scale, **kwargs):
        del q, scale, kwargs
        dense_lse_seen["lse"] = lse.detach().clone()
        shape = (batch, seqlen, k.shape[1])
        return {
            "out": torch.ones(shape),
            "denom": torch.full(shape[:2], k.shape[1], dtype=torch.float32),
        }

    def fake_dense_backward(q, w, k, *_args, **_kwargs):
        return {
            "d_index_q": torch.zeros_like(q),
            "d_index_k": torch.zeros_like(k),
            "d_weights": torch.zeros_like(w),
        }

    def fake_sparse_backward(q, kv, _out, _dout, _lse, attn_sink, indices, **kwargs):
        backward_seen["indices"] = indices.detach().clone()
        backward_seen["topk_length"] = kwargs["topk_length"].detach().clone()
        return {
            "dq": torch.zeros_like(q),
            "dkv": torch.zeros_like(kv),
            "d_sink": torch.zeros_like(attn_sink),
        }

    fake_dsa.dense_attn_score_recompute_wrapper = MagicMock(side_effect=fake_dense_attn)
    fake_dsa.dense_indexer_backward_wrapper = MagicMock(side_effect=fake_dense_backward)
    fake_dsa.indexer_backward_wrapper = MagicMock(
        side_effect=AssertionError("dense fused training must not use sparse indexer backward")
    )
    fake_dsa.sparse_attention_backward_wrapper = MagicMock(side_effect=fake_sparse_backward)
    monkeypatch.setattr(fused_csa, "_DSA", fake_dsa)

    output, indexer_loss = fused_csa.fused_csa_indexer_sparse_attn(
        query,
        kv_full,
        sink,
        window,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk=2,
        ratio=4,
        softmax_scale=0.5,
        loss_coeff=1.0,
        sparse_loss=False,
        kv_offset=seqlen,
    )
    (output.sum() + indexer_loss).backward()

    assert flash_seen["indexer_topk"] == 0
    assert torch.equal(
        teacher_seen["window_indices"], fused_csa.local_to_global_flat(window, batch)
    )
    assert teacher_seen["batch_size"] == batch
    assert teacher_seen["seqlen_q"] == seqlen
    torch.testing.assert_close(dense_lse_seen["lse"], teacher_lse)
    assert torch.equal(backward_seen["topk_length"], flash_seen["topk_length"])
    assert torch.all(backward_seen["indices"] >= 0)
    valid_prefix = torch.arange(flash_seen["indices"].shape[-1]).view(1, -1) < flash_seen[
        "topk_length"
    ].view(-1, 1)
    assert torch.all(flash_seen["indices"][valid_prefix] >= 0)
    fake_dsa.dense_indexer_backward_wrapper.assert_called_once()
    fake_dsa.indexer_backward_wrapper.assert_not_called()


def test_sparse_indexer_backward_preserves_invalid_minus_one(monkeypatch):
    """Invalid top-k slots must reach the guarded cuDNN backward as ``-1``."""
    seqlen, batch, heads, dim = 3, 1, 2, 4
    compressed_rows, indexer_heads, indexer_dim = 2, 2, 3
    query = torch.randn(seqlen, batch, heads, dim, requires_grad=True)
    kv_full = torch.randn(seqlen + compressed_rows, batch, dim, requires_grad=True)
    sink = torch.randn(heads, requires_grad=True)
    window = torch.tensor([[[0, -1], [0, 1], [1, 2]]], dtype=torch.int32)
    q_indexer = torch.randn(seqlen, batch, indexer_heads, indexer_dim, requires_grad=True)
    k_indexer = torch.randn(compressed_rows, batch, indexer_dim, requires_grad=True)
    weights = torch.randn(seqlen, batch, indexer_heads, requires_grad=True)
    selected = torch.tensor([[[-1, -1], [0, -1], [1, 0]]], dtype=torch.int32)

    monkeypatch.setattr(
        fused_csa,
        "_indexer_topk_bshd",
        lambda *_args, **_kwargs: (
            selected.clone(),
            (selected >= 0).sum(dim=-1).int(),
            torch.zeros(batch, seqlen, compressed_rows),
        ),
    )
    monkeypatch.setattr(
        fused_csa,
        "_csa_fwd_flash_mla",
        lambda q, *_args, **_kwargs: (q.new_zeros(q.shape), q.new_zeros(q.shape[:2]), None),
    )
    monkeypatch.setattr(
        fused_csa,
        "_compute_attn_target",
        lambda *_args, **_kwargs: torch.where(
            selected >= 0,
            torch.full(selected.shape, 0.5, dtype=torch.float32),
            torch.zeros(selected.shape, dtype=torch.float32),
        ),
    )

    backward_seen = {}
    fake_dsa = SimpleNamespace()

    def fake_indexer_backward(q, w, k, _attn_score, _index_score, indices, **_kwargs):
        backward_seen["indices"] = indices.detach().clone()
        return {
            "d_index_q": torch.zeros_like(q),
            "d_index_k": torch.zeros_like(k),
            "d_weights": torch.zeros_like(w),
        }

    def fake_sparse_backward(q, kv, _out, _dout, _lse, attn_sink, _indices, **_kwargs):
        return {
            "dq": torch.zeros_like(q),
            "dkv": torch.zeros_like(kv),
            "d_sink": torch.zeros_like(attn_sink),
        }

    fake_dsa.indexer_backward_wrapper = MagicMock(side_effect=fake_indexer_backward)
    fake_dsa.sparse_attention_backward_wrapper = MagicMock(side_effect=fake_sparse_backward)
    monkeypatch.setattr(fused_csa, "_DSA", fake_dsa)

    output, indexer_loss = fused_csa.fused_csa_indexer_sparse_attn(
        query,
        kv_full,
        sink,
        window,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk=2,
        ratio=4,
        softmax_scale=0.5,
        loss_coeff=1.0,
        sparse_loss=True,
        kv_offset=seqlen,
    )
    (output.sum() + indexer_loss).backward()

    assert torch.equal(backward_seen["indices"], selected)
    assert torch.all(backward_seen["indices"][selected < 0] == -1)
    fake_dsa.indexer_backward_wrapper.assert_called_once()


def test_ratio4_training_dispatch_never_touches_native_dense_fallback(monkeypatch):
    import torch.nn as nn

    from megatron.core.transformer.experimental_attention_variant import csa as csa_module

    module = csa_module.CompressedSparseAttention.__new__(csa_module.CompressedSparseAttention)
    nn.Module.__init__(module)
    module.use_fused_kernels = True
    module.compress_ratio = 4
    module.window_size = 2
    module.softmax_scale = 0.5
    module.attn_sink = nn.Parameter(torch.zeros(2))
    module.num_attention_heads = module.attn_sink.numel()
    module.layer_number = 1
    module.config = SimpleNamespace(
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_use_sparse_loss=False,
        calculate_per_token_loss=False,
        num_layers=1,
        mtp_num_layers=0,
    )

    class FakeCompressor:
        def __call__(self, x):
            return x.new_zeros(1, x.shape[1], 3)

    class FakeIndexer:
        index_topk = 2
        softmax_scale = 0.25

        def forward_before_topk(self, x, qr):
            return (x.new_zeros(4, 1, 2, 2), x.new_zeros(1, 1, 2), qr.new_zeros(4, 1, 2))

    module.__dict__["compressor"] = FakeCompressor()
    module.__dict__["indexer"] = FakeIndexer()
    module.train()

    expected = torch.randn(4, 1, 6)
    fused_call = MagicMock(return_value=(expected, torch.tensor(0.0)))
    monkeypatch.setattr(csa_module, "fused_csa_indexer_sparse_attn", fused_call)
    monkeypatch.setattr(csa_module, "nvtx_range_push", lambda *_args: None)
    monkeypatch.setattr(csa_module, "nvtx_range_pop", lambda *_args: None)
    monkeypatch.setattr(
        csa_module,
        "unfused_compressed_sparse_attn",
        MagicMock(side_effect=AssertionError("native final attention fallback was used")),
    )
    monkeypatch.setattr(
        csa_module,
        "_compute_unfused_csa_non_compressed_lse",
        MagicMock(side_effect=AssertionError("native dense teacher allocation was used")),
    )
    monkeypatch.setattr(
        csa_module.FusedDSAIndexerLoss,
        "apply",
        MagicMock(side_effect=AssertionError("native FusedDSAIndexerLoss was used")),
    )
    monkeypatch.setattr(csa_module.DSAIndexerLossAutoScaler, "apply", lambda output, _loss: output)

    query = torch.randn(4, 1, 2, 3)
    key = torch.randn(4, 1, 1, 3)
    x = torch.randn(4, 1, 5)
    qr = torch.randn(4, 1, 2)
    actual = module.forward(query, key, key, None, x=x, qr=qr)

    assert actual is expected
    fused_call.assert_called_once()


def test_public_surface_is_csa_namespaced_and_sbhd_only():
    assert "csa_sparse_attn" in fused_csa.__all__
    assert "fused_csa_indexer_sparse_attn" in fused_csa.__all__
    assert not hasattr(fused_csa, "dsa_sparse_attn")
    assert not hasattr(fused_csa, "FusedCSAIndexerSparseAttnFromTopkFunc")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_real_sbhd_teacher_lse_kernel_matches_bounded_reference():
    torch.manual_seed(41)
    device = torch.device("cuda")
    batch, seqlen_q, heads, dim = 2, 23, 16, 32
    seqlen_full, seqlen_compressed, window_width, ratio = 23, 9, 5, 3
    query = torch.randn(seqlen_q, batch, heads, dim, device=device, dtype=torch.bfloat16)
    full_kv = torch.randn(seqlen_full, batch, dim, device=device, dtype=torch.bfloat16)
    compressed_kv = torch.randn(batch, seqlen_compressed, dim, device=device, dtype=torch.bfloat16)
    sink = torch.randn(heads, device=device, dtype=torch.float32)
    window_local = torch.randint(
        0, seqlen_full, (batch, seqlen_q, window_width), device=device, dtype=torch.int32
    )
    window_local[:, ::4, -1] = -1
    window_global = fused_csa.local_to_global_flat(window_local, batch)
    query_flat = query.reshape(seqlen_q * batch, heads, dim)
    full_kv_flat = full_kv.reshape(seqlen_full * batch, dim)

    if not can_use_fused_csa_teacher_lse(
        query_flat, full_kv_flat, compressed_kv, sink, window_global
    ):
        pytest.skip("Triton CSA teacher-LSE kernel is unavailable")

    actual = fused_csa_teacher_lse(
        query_flat,
        full_kv_flat,
        compressed_kv,
        sink,
        window_global,
        softmax_scale=dim**-0.5,
        ratio=ratio,
        batch_size=batch,
        seqlen_q=seqlen_q,
    )
    non_compressed = fused_csa._compute_csa_non_compressed_lse(
        query_flat, full_kv_flat, sink, window_global, dim**-0.5
    )
    non_compressed = non_compressed.reshape(seqlen_q, batch, heads).permute(1, 0, 2).contiguous()
    expected = fused_csa._compute_dense_csa_teacher_lse(
        query.permute(1, 0, 2, 3).contiguous(), compressed_kv, non_compressed, dim**-0.5, ratio
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_real_fused_sbhd_forward_backward_matches_native_reference(num_heads):
    """Compare small-head FlashMLA/cuDNN forward and backward with native CSA."""
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("fused CSA requires SM90+")
    try:
        from cudnn import DSA  # noqa: F401
        from flash_mla import flash_mla_sparse_fwd  # noqa: F401
    except ImportError:
        pytest.skip("fused CSA dependencies are unavailable")

    from megatron.core.transformer.experimental_attention_variant.csa import (
        get_window_topk_idxs,
        unfused_compressed_sparse_attn,
    )

    torch.manual_seed(1234)
    seq = 128
    query = torch.randn(seq, 1, num_heads, 512, device="cuda", dtype=torch.bfloat16) * 0.05
    kv = torch.randn(seq, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.05
    sink = torch.zeros(num_heads, device="cuda", dtype=torch.float32)
    query_native = query.detach().clone().requires_grad_(True)
    query_fused = query.detach().clone().requires_grad_(True)
    kv_native = kv.detach().clone().requires_grad_(True)
    kv_fused = kv.detach().clone().requires_grad_(True)
    sink_native = sink.detach().clone().requires_grad_(True)
    sink_fused = sink.detach().clone().requires_grad_(True)
    local_indices = get_window_topk_idxs(seq, 1, seq, query.device).int()
    flat_indices, _ = fused_csa.build_flat_topk_idxs(local_indices, batch_size=1)

    expected = unfused_compressed_sparse_attn(
        query_native, kv_native, sink_native, local_indices, 512**-0.5
    )
    actual = fused_csa.csa_sparse_attn(query_fused, kv_fused, sink_fused, flat_indices, 512**-0.5)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=3e-2, atol=3e-2)
    expected.float().sum().backward()
    actual.float().sum().backward()
    torch.testing.assert_close(
        query_fused.grad.float(), query_native.grad.float(), rtol=5e-2, atol=5e-2
    )
    torch.testing.assert_close(kv_fused.grad.float(), kv_native.grad.float(), rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(
        sink_fused.grad.float(), sink_native.grad.float(), rtol=5e-2, atol=5e-2
    )
