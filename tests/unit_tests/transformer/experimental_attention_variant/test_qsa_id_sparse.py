# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""One-GPU correctness checks for the standalone selected-ID QSA prototype."""

import math

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.experimental_attention_variant.qsa_id_sparse import (
    qsa_sparse_attention_id,
)


def _routes(docs, topk, ratio, seed):
    rng = torch.Generator().manual_seed(seed)
    batch, seq_len = len(docs), sum(docs[0])
    positions = torch.empty(batch, seq_len, dtype=torch.int32)
    ids = torch.full((batch, seq_len, topk), -1, dtype=torch.int32)
    mask = torch.zeros(batch, seq_len, seq_len, dtype=torch.bool)
    for b, lengths in enumerate(docs):
        assert sum(lengths) == seq_len
        start = 0
        for length in lengths:
            for pos in range(length):
                q = start + pos
                positions[b, q] = pos
                visible = (pos + 1) // ratio
                chosen = torch.randperm(visible, generator=rng)[:topk].tolist()
                ids[b, q, : len(chosen)] = torch.tensor(chosen, dtype=torch.int32)
                for block in chosen:
                    mask[b, q, start + block * ratio : start + (block + 1) * ratio] = True
                tail = ((pos + 1) // ratio) * ratio
                mask[b, q, start + tail : q + 1] = True
            start += length
    return ids.cuda(), positions.cuda(), mask.cuda()


def _dense_reference(q, k, v, mask, scale):
    repeat = q.shape[1] // k.shape[1]
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        return F.scaled_dot_product_attention(
            q,
            k.repeat_interleave(repeat, dim=1),
            v.repeat_interleave(repeat, dim=1),
            attn_mask=mask[:, None],
            scale=scale,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("docs", [[[37]], [[21, 16]], [[1, 1, 35]], [[21, 16], [37]]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_selected_id_qsa_matches_dense_forward_backward(docs, dtype):
    ids, positions, mask = _routes(docs, topk=6, ratio=4, seed=19)
    batch, seq_len = len(docs), sum(docs[0])
    torch.manual_seed(31)
    q = torch.randn(batch, 4, seq_len, 32, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(batch, 2, seq_len, 32, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(batch, 2, seq_len, 32, device="cuda", dtype=dtype, requires_grad=True)
    scale = 1 / math.sqrt(32)
    actual = qsa_sparse_attention_id(q, k, v, ids, positions, ratio=4, scale=scale)
    ref_q, ref_k, ref_v = (x.detach().clone().requires_grad_() for x in (q, k, v))
    expected = _dense_reference(ref_q, ref_k, ref_v, mask, scale)
    atol = 0.04 if dtype == torch.bfloat16 else 2e-4
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=atol)
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (q, k, v), grad)
    expected_grads = torch.autograd.grad(expected, (ref_q, ref_k, ref_v), grad)
    for got, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got.float(), want.float(), atol=atol, rtol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("ratio,topk,dim", [(2, 33, 64), (4, 12, 128), (8, 9, 64)])
def test_selected_id_qsa_multiple_chunks_and_ratios(ratio, topk, dim):
    ids, positions, mask = _routes([[47, 82]], topk=topk, ratio=ratio, seed=59)
    torch.manual_seed(61)
    q = torch.randn(1, 4, 129, dim, device="cuda", requires_grad=True)
    k = torch.randn(1, 2, 129, dim, device="cuda", requires_grad=True)
    v = torch.randn(1, 2, 129, dim, device="cuda", requires_grad=True)
    actual = qsa_sparse_attention_id(q, k, v, ids, positions, ratio=ratio, validate=True)
    rq, rk, rv = (x.detach().clone().requires_grad_() for x in (q, k, v))
    expected = _dense_reference(rq, rk, rv, mask, dim**-0.5)
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)
    grad = torch.randn_like(actual)
    for got, want in zip(
        torch.autograd.grad(actual, (q, k, v), grad),
        torch.autograd.grad(expected, (rq, rk, rv), grad),
    ):
        torch.testing.assert_close(got, want, atol=5e-4, rtol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_selected_id_qsa_rejects_empty_and_duplicate_complete_block_routes():
    ids, positions, _ = _routes([[16]], topk=3, ratio=4, seed=71)
    q = torch.randn(1, 2, 16, 16, device="cuda")
    k = torch.randn(1, 1, 16, 16, device="cuda")
    empty = ids.clone()
    empty[0, 3, :] = -1  # no incomplete tail at the fourth token
    with pytest.raises(ValueError, match="fill every available top-k slot"):
        qsa_sparse_attention_id(q, k, k, empty, positions, validate=True)
    duplicated = ids.clone()
    duplicated[0, 7, 1] = duplicated[0, 7, 0]
    with pytest.raises(ValueError, match="distinct"):
        qsa_sparse_attention_id(q, k, k, duplicated, positions, validate=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_selected_id_qsa_accepts_noncontiguous_invalid_slots():
    """Padding holes and the p=R-1/R boundary preserve the same exact mask."""
    ids, positions, mask = _routes([[17, 28]], topk=3, ratio=4, seed=73)
    generator = torch.Generator().manual_seed(79)
    shuffled = ids.clone()
    for row in range(ids.shape[1]):
        order = torch.randperm(ids.shape[-1], generator=generator).to(ids.device)
        shuffled[0, row] = ids[0, row, order]
    assert (shuffled[0, 3] >= 0).sum() == 1
    assert (shuffled[0, 4] >= 0).sum() == 1
    torch.manual_seed(83)
    q = torch.randn(1, 4, 45, 32, device="cuda", requires_grad=True)
    k = torch.randn(1, 2, 45, 32, device="cuda", requires_grad=True)
    v = torch.randn(1, 2, 45, 32, device="cuda", requires_grad=True)
    out = qsa_sparse_attention_id(q, k, v, shuffled, positions, validate=True)
    rq, rk, rv = (x.detach().clone().requires_grad_() for x in (q, k, v))
    expected = _dense_reference(rq, rk, rv, mask, 32**-0.5)
    torch.testing.assert_close(out, expected, atol=3e-4, rtol=3e-4)
    grad = torch.randn_like(out)
    for actual, reference in zip(
        torch.autograd.grad(out, (q, k, v), grad), torch.autograd.grad(expected, (rq, rk, rv), grad)
    ):
        torch.testing.assert_close(actual, reference, atol=5e-4, rtol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_selected_id_qsa_released_geometry_sparse_row_matches_dense_gradients():
    """BF16 D256, 24Q/2KV, K512 and S>2051 exercise real QSA sparse rows."""
    seq_len, ratio, topk, dim = 2064, 4, 512, 256
    positions = torch.arange(seq_len, device="cuda", dtype=torch.int32).unsqueeze(0)
    visible = (positions + 1) // ratio
    slot = torch.arange(topk, device="cuda", dtype=torch.int32)
    candidate = visible.unsqueeze(-1) - slot - 1
    ids = torch.where(candidate >= 0, candidate, -1).contiguous()
    assert int(ids[0, -1, topk - 1]) > 0
    keys = torch.arange(seq_len, device="cuda", dtype=torch.int32)
    complete = (keys[None, None, :] // ratio >= visible.unsqueeze(-1) - topk) & (
        keys[None, None, :] // ratio < visible.unsqueeze(-1)
    )
    tail = keys[None, None, :] >= visible.unsqueeze(-1) * ratio
    causal = keys[None, None, :] <= positions.unsqueeze(-1)
    mask = (complete | tail) & causal
    torch.manual_seed(89)
    q = torch.randn(1, 24, seq_len, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 2, seq_len, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 2, seq_len, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = qsa_sparse_attention_id(q, k, v, ids, positions, ratio=ratio, validate=True)
    rq, rk, rv = (x.detach().clone().requires_grad_() for x in (q, k, v))
    expected = _dense_reference(rq, rk, rv, mask, dim**-0.5)
    torch.testing.assert_close(out.float(), expected.float(), atol=7e-2, rtol=7e-2)
    grad = torch.randn_like(out)
    for actual, reference in zip(
        torch.autograd.grad(out, (q, k, v), grad), torch.autograd.grad(expected, (rq, rk, rv), grad)
    ):
        torch.testing.assert_close(actual.float(), reference.float(), atol=8e-2, rtol=8e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_selected_id_qsa_finite_difference():
    ids, positions, _ = _routes([[8]], topk=2, ratio=4, seed=5)
    torch.manual_seed(47)
    q = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn(1, 1, 8, 16, device="cuda", dtype=torch.float32, requires_grad=True)
    v = torch.randn(1, 1, 8, 16, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn_like(q)

    def loss(q_arg, k_arg, v_arg):
        return (qsa_sparse_attention_id(q_arg, k_arg, v_arg, ids, positions) * weight).sum()

    analytic = torch.autograd.grad(loss(q, k, v), (q, k, v))
    epsilon = 1e-3
    for tensor_index, element in [(0, (0, 0, 5, 3)), (1, (0, 0, 1, 2)), (2, (0, 0, 5, 4))]:
        plus = [x.detach().clone() for x in (q, k, v)]
        minus = [x.detach().clone() for x in (q, k, v)]
        plus[tensor_index][element] += epsilon
        minus[tensor_index][element] -= epsilon
        numerical = (loss(*plus) - loss(*minus)) / (2 * epsilon)
        torch.testing.assert_close(analytic[tensor_index][element], numerical, atol=2e-3, rtol=2e-3)
