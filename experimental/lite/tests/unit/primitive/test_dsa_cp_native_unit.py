# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytestmark = pytest.mark.mlite


@pytest.fixture(autouse=True)
def _te_import_stub(transformer_engine_import_stub):
    transformer_engine_import_stub()


def test_dense_cp_layout_keeps_local_queries_and_restores_global_kv_order():
    from megatron.lite.primitive.modules.attention.dsa import _dense_cp_layout

    query_positions, kv_reorder = _dense_cp_layout(
        local_seq=4,
        cp_size=2,
        cp_rank=0,
        device=torch.device("cpu"),
    )

    assert query_positions.tolist() == [0, 1, 2, 3]
    gathered_positions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    assert gathered_positions.index_select(0, kv_reorder).tolist() == list(range(8))


def test_packed_cp_layout_uses_contiguous_global_coordinates():
    from megatron.lite.primitive.modules.attention.dsa import _packed_cp_layout

    cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32)
    query_positions, kv_reorder = _packed_cp_layout(
        cu_seqlens,
        cp_size=2,
        cp_rank=0,
        device=torch.device("cpu"),
    )

    assert query_positions.tolist() == list(range(8))
    gathered_positions = torch.arange(16)
    assert gathered_positions.index_select(0, kv_reorder).tolist() == list(range(16))


def test_cp_projected_kv_is_physically_aligned_for_cudnn_topk():
    from megatron.lite.primitive.modules.attention.dsa import _pad_cp_projected_kv

    kv = torch.arange(520 * 3, dtype=torch.float32).view(520, 1, 3)
    index_k = torch.arange(520 * 2, dtype=torch.float32).view(520, 1, 2)

    padded_kv, padded_index_k = _pad_cp_projected_kv(kv, index_k)

    assert padded_kv.shape == (1024, 1, 3)
    assert padded_index_k is not None and padded_index_k.shape == (1024, 1, 2)
    assert torch.equal(padded_kv[:520], kv)
    assert torch.equal(padded_index_k[:520], index_k)
    assert torch.count_nonzero(padded_kv[520:]) == 0
    assert torch.count_nonzero(padded_index_k[520:]) == 0


def test_cp_causal_mask_blocks_future_and_cross_packed_sequence_keys():
    from megatron.lite.primitive.modules.attention.dsa import _build_cp_causal_mask

    query_positions = torch.tensor([1, 8, 15], dtype=torch.long)
    # Keys beyond the final logical boundary model physically aligned CP padding.
    key_positions = torch.arange(20, dtype=torch.long)
    cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32)

    mask = _build_cp_causal_mask(
        query_positions,
        key_positions,
        cu_seqlens=cu_seqlens,
    )
    valid = mask == 0

    assert valid[0].nonzero().flatten().tolist() == [0, 1]
    assert valid[1].nonzero().flatten().tolist() == [8]
    assert valid[2].nonzero().flatten().tolist() == list(range(8, 16))


def test_ragged_packed_kv_is_physically_aligned_without_changing_valid_rows():
    from megatron.lite.primitive.kernels.dsa_kernels import _pad_sparse_kv_rows

    kv = torch.arange(520 * 3, dtype=torch.float32).view(520, 3)
    padded = _pad_sparse_kv_rows(kv, 128)

    assert padded.shape == (640, 3)
    assert torch.equal(padded[:520], kv)
    assert torch.count_nonzero(padded[520:]) == 0
    assert _pad_sparse_kv_rows(padded, 128) is padded


def test_ragged_indexer_scores_keep_physical_width_and_report_cudnn_contract(capsys):
    from megatron.lite.primitive.kernels.dsa_kernels import _cudnn_topk_block

    scores = torch.randn(3, 520)
    fake_dsa = Mock()
    returned = torch.zeros((3, 512), dtype=torch.int32)
    returned[0, -1] = 700
    fake_dsa.indexer_top_k_wrapper.return_value = {"indices": returned}

    values, indices = _cudnn_topk_block(fake_dsa, scores, width=512)

    call = fake_dsa.indexer_top_k_wrapper.call_args
    input_values, seq_lens = call.args
    assert input_values.shape == (3, 520)
    assert torch.equal(input_values, scores)
    assert seq_lens.tolist() == [520, 520, 520]
    assert call.kwargs == {"top_k": 512, "next_n": 1, "return_val": False}
    assert "input_values.shape=(3, 520)" in capsys.readouterr().out
    assert values.shape == indices.shape == (3, 512)
    assert indices[0, -1] == -1
    assert torch.isneginf(values[0, -1])


def test_dense_cp_native_collective_only_receives_projected_kv_and_indexer_k():
    from megatron.lite.primitive.modules.attention.dsa import DynamicSparseAttention

    gathered_widths = []
    gather_modes = []
    sentinel = torch.randn(1, 4, 64)

    def project_inputs(x, cos, sin, position_ids):
        del x, cos, sin, position_ids
        return (
            torch.randn(4, 1, 2, 8),
            torch.randn(4, 1, 8),
            torch.randn(2, 2, 4),
            torch.randn(4, 1, 2, 4),
            torch.randn(4, 1, 4),
            torch.randn(4, 1, 2),
        )

    def gather_projected(tensor, reorder, *, contiguous=False):
        gathered_widths.append(tensor.shape[-1])
        gather_modes.append(contiguous)
        return torch.cat([tensor, tensor], dim=0).index_select(0, reorder)

    def run_sparse(query, kv, q_idx, k_idx, weights, mask, **kwargs):
        del q_idx, k_idx, weights, mask, kwargs
        assert query.shape[0] == 4
        assert kv.shape[0] == 8
        return torch.randn(4, 1, 8)

    fake = SimpleNamespace(
        cp_size=2,
        cp_rank=0,
        _project_cp_inputs=project_inputs,
        _gather_projected_cp=gather_projected,
        _run_cp_sparse_segment=run_sparse,
        _project_cp_output=lambda out, weight: sentinel,
    )
    x = torch.randn(1, 4, 64)
    result = DynamicSparseAttention._forward_dense_cp_native(
        fake,
        x,
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        index_share_state=None,
    )

    assert result is sentinel
    assert gathered_widths == [8, 4]
    assert gather_modes == [False, False]
    assert x.shape[-1] not in gathered_widths


def test_packed_cp_native_explicitly_selects_contiguous_projected_gather():
    from megatron.lite.primitive.modules.attention.dsa import DynamicSparseAttention

    gather_modes = []
    sentinel = torch.randn(1, 4, 64)

    def project_inputs(x, cos, sin, position_ids):
        del x, cos, sin, position_ids
        return (
            torch.randn(4, 1, 2, 8),
            torch.randn(4, 1, 8),
            torch.randn(2, 2, 4),
            torch.randn(4, 1, 2, 4),
            torch.randn(4, 1, 4),
            torch.randn(4, 1, 2),
        )

    def gather_projected(tensor, reorder, *, contiguous=False):
        gather_modes.append(contiguous)
        return torch.cat([tensor, tensor], dim=0).index_select(0, reorder)

    def run_sparse(query, kv, q_idx, k_idx, weights, mask, **kwargs):
        del q_idx, k_idx, weights, mask, kwargs
        assert query.shape[0] == 4
        assert kv.shape[0] == 8
        return torch.randn(4, 1, 8)

    fake = SimpleNamespace(
        cp_size=2,
        cp_rank=0,
        _packed_cu_seqlens=lambda params, device: params.cu_seqlens_q.to(device),
        _project_cp_inputs=project_inputs,
        _gather_projected_cp=gather_projected,
        _run_cp_sparse_segment=run_sparse,
        _project_cp_output=lambda out, weight: sentinel,
    )
    params = SimpleNamespace(
        cp_layout="contiguous",
        cu_seqlens_q=torch.tensor([0, 8], dtype=torch.int32),
    )
    result = DynamicSparseAttention._forward_packed_cp_native(
        fake,
        torch.randn(1, 4, 64),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        params,
        index_share_state=None,
    )

    assert result is sentinel
    assert gather_modes == [True, True]


def test_cp_indexer_topk_respects_explicit_global_position_mask():
    from megatron.lite.primitive.kernels.dsa_kernels import indexer_topk_with_mask

    torch.manual_seed(7)
    q = torch.randn(2, 1, 2, 4)
    k = torch.randn(8, 1, 4)
    weights = torch.randn(2, 1, 2)
    mask = torch.full((2, 8), float("-inf"))
    mask[0, :2] = 0
    mask[1, :7] = 0

    _scores, actual = indexer_topk_with_mask(q, k, weights, 4, mask, 4**-0.5)
    reference = torch.einsum("qbhd,kbd->bqhk", q.float(), k.float())
    reference = torch.relu(reference).mul(weights.permute(1, 0, 2).unsqueeze(-1)).sum(dim=2)
    reference = reference * (4**-0.5) + mask.unsqueeze(0)
    values, expected = torch.topk(reference, k=4, dim=-1)
    expected = torch.where(torch.isfinite(values), expected, torch.full_like(expected, -1))

    assert torch.equal(actual, expected.int())
    assert (actual[0, 0][actual[0, 0] >= 0] < 2).all()
    assert (actual[0, 1][actual[0, 1] >= 0] < 7).all()


def test_cp_dense_indexer_loss_matches_full_score_reference_and_backpropagates():
    from megatron.lite.primitive.kernels.dsa_kernels import cp_indexer_loss, indexer_topk_with_mask

    torch.manual_seed(11)
    q = torch.randn(3, 1, 2, 4, requires_grad=True)
    k = torch.randn(6, 1, 4, requires_grad=True)
    weights = torch.randn(3, 1, 2, requires_grad=True)
    query = torch.randn(3, 1, 2, 4)
    kv = torch.randn(6, 1, 4)
    mask = torch.full((3, 6), float("-inf"))
    mask[0, :1] = 0
    mask[1, :4] = 0
    mask[2, :6] = 0
    _scores, topk = indexer_topk_with_mask(q, k, weights, 3, mask, 4**-0.5)

    actual = cp_indexer_loss(
        q,
        k,
        weights,
        topk,
        query,
        kv,
        mask=mask,
        softmax_scale=4**-0.5,
        loss_coeff=0.7,
        sparse_loss=False,
        calculate_per_token_loss=False,
    )

    index_logits = torch.einsum("qbhd,kbd->bqhk", q.float(), k.float())
    index_logits = torch.relu(index_logits).mul(weights.permute(1, 0, 2).unsqueeze(-1)).sum(dim=2)
    index_logits = index_logits * (4**-0.5) + mask.unsqueeze(0)
    predict = torch.softmax(index_logits, dim=-1)
    attn_logits = torch.einsum("qbhd,kbd->bqhk", query.float(), kv.float())
    attn_logits = attn_logits * (4**-0.5) + mask.unsqueeze(0).unsqueeze(2)
    target = torch.softmax(attn_logits, dim=-1).mean(dim=2)
    expected = (
        target * (torch.log(target + 1.0e-10) - torch.log(predict + 1.0e-10))
    ).sum(dim=-1).mean() * 0.7

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual.backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert weights.grad is not None and torch.isfinite(weights.grad).all()


def test_cp_sparse_indexer_loss_ignores_invalid_topk_sentinels():
    from megatron.lite.primitive.kernels.dsa_kernels import cp_indexer_loss

    torch.manual_seed(19)
    q = torch.randn(2, 1, 2, 4, requires_grad=True)
    k = torch.randn(5, 1, 4, requires_grad=True)
    weights = torch.randn(2, 1, 2, requires_grad=True)
    query = torch.randn(2, 1, 2, 4)
    kv = torch.randn(5, 1, 4)
    topk = torch.tensor([[[0, -1, -1], [2, 1, 0]]], dtype=torch.int32)
    mask = torch.zeros((2, 5), dtype=torch.float32)

    loss = cp_indexer_loss(
        q,
        k,
        weights,
        topk,
        query,
        kv,
        mask=mask,
        softmax_scale=4**-0.5,
        loss_coeff=0.5,
        sparse_loss=True,
        calculate_per_token_loss=False,
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert weights.grad is not None and torch.isfinite(weights.grad).all()


def test_zero_cp_indexer_loss_keeps_zero_grad_edges_for_indexer_parameters():
    from megatron.lite.primitive.kernels.dsa_kernels import cp_indexer_loss

    q = torch.randn(2, 1, 2, 4, requires_grad=True)
    k = torch.randn(3, 1, 4, requires_grad=True)
    weights = torch.randn(2, 1, 2, requires_grad=True)
    loss = cp_indexer_loss(
        q,
        k,
        weights,
        torch.zeros((1, 2, 1), dtype=torch.int32),
        torch.randn(2, 1, 2, 4),
        torch.randn(3, 1, 4),
        mask=torch.zeros((2, 3)),
        softmax_scale=0.5,
        loss_coeff=0.0,
        sparse_loss=True,
        calculate_per_token_loss=False,
    )

    loss.backward()
    assert q.grad is not None and torch.count_nonzero(q.grad) == 0
    assert k.grad is not None and torch.count_nonzero(k.grad) == 0
    assert weights.grad is not None and torch.count_nonzero(weights.grad) == 0
