# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Focused tests for the SBHD fused CSA integration surface."""

from unittest.mock import patch

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    fused_sparse_attention as fused_csa,
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


def test_public_surface_is_csa_namespaced_and_sbhd_only():
    assert "csa_sparse_attn" in fused_csa.__all__
    assert "fused_csa_indexer_sparse_attn" in fused_csa.__all__
    assert not hasattr(fused_csa, "dsa_sparse_attn")
    assert not hasattr(fused_csa, "FusedCSAIndexerSparseAttnFromTopkFunc")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_real_fused_sbhd_forward_matches_native_reference():
    """Compare the real FlashMLA forward with the native CSA implementation."""
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
    query = torch.randn(seq, 1, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.05
    kv = torch.randn(seq, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.05
    sink = torch.zeros(1, device="cuda", dtype=torch.float32)
    local_indices = get_window_topk_idxs(seq, 1, seq, query.device).int()
    flat_indices, _ = fused_csa.build_flat_topk_idxs(local_indices, batch_size=1)

    expected = unfused_compressed_sparse_attn(query, kv, sink, local_indices, 512**-0.5)
    actual = fused_csa.csa_sparse_attn(query, kv, sink, flat_indices, 512**-0.5)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=3e-2, atol=3e-2)
