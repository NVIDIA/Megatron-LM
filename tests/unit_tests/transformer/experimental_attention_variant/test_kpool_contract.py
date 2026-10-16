# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""PR 7522 selector contracts and independent dev adaptation regressions."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.experimental_attention_variant.dsa import (
    _kpool_compress_keys_per_seg,
    _unfused_absorbed_dsa_fn,
    _validate_dsa_kpool_config,
    fused_qk_topk_kpool,
)


@pytest.fixture(autouse=True)
def _bind_local_cuda_device():
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))


@pytest.mark.parametrize(
    ('pool_size', 'topk', 'loss_coeff', 'match'),
    [
        (0, 16, 0, 'positive'),
        (4, None, 0, 'must be set'),
        (4, 0, 0, 'positive'),
        (4, 15, 0, 'divisible'),
        (4, 16, 0.01, 'not supported'),
    ],
)
def test_kpool_rejects_invalid_selector_contract(pool_size, topk, loss_coeff, match):
    with pytest.raises(ValueError, match=match):
        _validate_dsa_kpool_config(
            SimpleNamespace(
                dsa_indexer_kpool=pool_size,
                dsa_indexer_topk=topk,
                dsa_indexer_loss_coeff=loss_coeff,
            )
        )


@pytest.mark.parametrize('loss_coeff', [None, 0.0])
def test_kpool_allows_frozen_selector_without_auxiliary_loss(loss_coeff):
    _validate_dsa_kpool_config(
        SimpleNamespace(
            dsa_indexer_kpool=4, dsa_indexer_topk=2048, dsa_indexer_loss_coeff=loss_coeff
        )
    )


def test_kpool_segment_compression_matches_independent_torch_reference():
    # GLM53 indexer head width is 128; only the number of tokens is reduced.
    torch.manual_seed(571)
    lengths = (3, 9, 5)
    cu = torch.tensor([0, 3, 12, 17], dtype=torch.int32, device='cuda')
    keys = torch.randn(sum(lengths), 1, 128, device='cuda', dtype=torch.bfloat16)
    gates = torch.randn_like(keys)
    ape = torch.randn(4, 128, device='cuda')
    pooled, bases = _kpool_compress_keys_per_seg(keys, gates, ape, 4, cu)
    reference, reference_bases = [], []
    start = 0
    for length in lengths:
        for offset in range(0, length - 3, 4):
            base = start + offset
            probabilities = torch.softmax(gates[base : base + 4].float() + ape[:, None], dim=0)
            reference.append((probabilities * keys[base : base + 4].float()).sum(0))
            reference_bases.append(base)
        start += length
    expected = torch.stack(reference).to(torch.bfloat16)
    actual = pooled[: len(reference)]
    assert bases[: len(reference)].tolist() == reference_bases
    assert torch.all(bases[len(reference) :] == -1)
    lhs, rhs = actual.flatten().double(), expected.flatten().double()
    assert F.cosine_similarity(lhs[None], rhs[None]).item() > 0.999
    assert (2 * (lhs * rhs).sum() / (lhs.square().sum() + rhs.square().sum())).item() > 0.999


def test_kpool_selection_without_scores_does_not_retain_autograd_graph():
    torch.manual_seed(572)
    q = torch.randn(7, 1, 2, 128, device='cuda', requires_grad=True)
    k = torch.randn(7, 1, 128, device='cuda', requires_grad=True)
    weights = torch.randn(7, 1, 2, device='cuda', requires_grad=True)
    positions = torch.arange(1, 8, device='cuda')
    with torch.no_grad():
        scores, indices = fused_qk_topk_kpool(
            q,
            k,
            weights,
            2048,
            4,
            gate_score=torch.zeros_like(k),
            ape=torch.zeros(4, 128, device='cuda'),
            varlen_starts=torch.zeros_like(positions),
            varlen_ends=positions,
            return_index_scores=False,
        )
    assert scores is None
    assert not indices.requires_grad
    assert indices.shape == (1, 7, 2051)
    assert q.grad is None and k.grad is None and weights.grad is None


def test_absorbed_sparse_attention_empty_rows_have_finite_zero_gradients():
    torch.manual_seed(573)
    query = torch.randn(2, 1, 2, 8, device='cuda', requires_grad=True)
    key = torch.randn(3, 1, 1, 8, device='cuda', requires_grad=True)
    indices = torch.tensor([[[-1, -1, -1], [0, 1, -1]]], device='cuda', dtype=torch.int32)
    output = _unfused_absorbed_dsa_fn(query, key, indices, 1 / 8**0.5, v_channels=8)
    torch.testing.assert_close(output[0], torch.zeros_like(output[0]))
    output.square().sum().backward()
    assert torch.isfinite(query.grad).all() and torch.isfinite(key.grad).all()
    torch.testing.assert_close(query.grad[0], torch.zeros_like(query.grad[0]))
