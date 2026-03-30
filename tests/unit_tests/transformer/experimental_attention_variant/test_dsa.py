# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.experimental_attention_variant.dsa_fused_kernels import (
    DSAFunction,
    indexer_bwd_interface,
    indexer_topk_reducesum_interface,
    sparse_mla_bwd_interface,
    sparse_mla_fwd_interface,
    sparse_mla_topk_reducesum_interface,
)
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(autouse=True)
def distributed_setup_teardown():
    """Initialize / destroy megatron parallel state around every test."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    yield
    Utils.destroy_model_parallel()


def _cosine_sim(a, b):
    return F.cosine_similarity(
        a.flatten().double().unsqueeze(0), b.flatten().double().unsqueeze(0)
    ).item()


def _tensor_sim(a, b):
    a, b = a.double(), b.double()
    denom = (a * a + b * b).sum()
    return (2.0 * (a * b).sum() / denom).item() if denom else 1.0


def _assert_similarity(a, b, eps=1e-3):
    c = _cosine_sim(a, b)
    t = _tensor_sim(a, b)
    assert c > 1 - eps, f"cosine_sim={c:.6f}"
    assert t > 1 - eps, f"tensor_sim={t:.6f}"


def _generate_causal_indices(S, HKV, topk, device="cuda"):
    indices = torch.full((S, HKV, topk), S, dtype=torch.int32, device=device)
    for t in range(S):
        for h in range(HKV):
            valid = torch.randperm(max(1, t), device=device)[:topk]
            indices[t, h, : len(valid)] = valid
    return indices


# ---------------------------------------------------------------------------
# 1. Sparse MLA Forward
# ---------------------------------------------------------------------------
def test_sparse_mla_fwd(S=4096, H=128, HKV=1, DQK=576, DV=512, topk=2048, dtype=torch.bfloat16):
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    q = torch.randn((S, H, DQK), dtype=dtype, device="cuda")
    kv = torch.randn((S, HKV, DQK), dtype=dtype, device="cuda")
    offsets = torch.tensor([0, S], dtype=torch.int32, device="cuda")
    sm_scale = (1.0 / DQK) ** 0.5

    indices = _generate_causal_indices(S, HKV, topk)

    fused_out, fused_lse = sparse_mla_fwd_interface(
        q, kv, indices, offsets, sm_scale, DV, use_unfused=False
    )
    ref_out, ref_lse = sparse_mla_fwd_interface(
        q, kv, indices, offsets, sm_scale, DV, use_unfused=True
    )

    _assert_similarity(fused_out, ref_out)
    _assert_similarity(fused_lse, ref_lse)


# ---------------------------------------------------------------------------
# 2. Sparse MLA Backward
# ---------------------------------------------------------------------------
def test_sparse_mla_bwd(S=4096, H=128, HKV=1, DQK=576, DV=512, topk=2048, dtype=torch.bfloat16):
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    q = torch.randn((S, H, DQK), dtype=dtype, device="cuda")
    kv = torch.randn((S, HKV, DQK), dtype=dtype, device="cuda")
    do = torch.randn((S, H, DV), dtype=dtype, device="cuda")
    offsets = torch.tensor([0, S], dtype=torch.int32, device="cuda")
    sm_scale = (1.0 / DQK) ** 0.5

    indices = _generate_causal_indices(S, HKV, topk)

    o, lse = sparse_mla_fwd_interface(q, kv, indices, offsets, sm_scale, DV, use_unfused=False)

    fused_dq, fused_dkv = sparse_mla_bwd_interface(
        q, kv, o, do, indices, lse, offsets, sm_scale, DV, use_unfused=False
    )
    ref_dq, ref_dkv = sparse_mla_bwd_interface(
        q, kv, o, do, indices, lse, offsets, sm_scale, DV, use_unfused=True
    )

    _assert_similarity(fused_dq, ref_dq)
    _assert_similarity(fused_dkv, ref_dkv)


# ---------------------------------------------------------------------------
# 3. Indexer Top-K + ReduceSum
# ---------------------------------------------------------------------------
def test_indexer_topk_reducesum(S=4096, H=64, D=128, topk=2048, dtype=torch.bfloat16):
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    q = torch.randn((S, H, D), dtype=dtype, device="cuda")
    weights = torch.randn((S, H), dtype=dtype, device="cuda")
    k = torch.randn((S, D), dtype=dtype, device="cuda")
    offsets = torch.tensor([0, S], dtype=torch.int32, device="cuda")

    ref_indices, ref_score = indexer_topk_reducesum_interface(
        q, weights, k, topk, offsets, use_unfused=True
    )
    fused_indices, fused_score = indexer_topk_reducesum_interface(
        q, weights, k, topk, offsets, use_unfused=False
    )

    _assert_similarity(fused_score, ref_score)


# ---------------------------------------------------------------------------
# 4. Indexer Backward
# ---------------------------------------------------------------------------
def test_indexer_bwd(S=4096, H=64, D=128, topk=2048, dtype=torch.bfloat16):
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    q = torch.randn((S, H, D), dtype=dtype, device="cuda")
    weights = torch.randn((S, H), dtype=dtype, device="cuda")
    k = torch.randn((S, D), dtype=dtype, device="cuda")
    offsets = torch.tensor([0, S], dtype=torch.int32, device="cuda")

    topk_indices, index_score = indexer_topk_reducesum_interface(
        q, weights, k, topk, offsets, use_unfused=False
    )

    causal_mask = (
        torch.arange(S, device="cuda")[:, None] >= torch.arange(topk, device="cuda")[None, :]
    )
    uniform_logits = torch.where(causal_mask, torch.ones(S, topk, device="cuda"), float("-inf"))
    attn_score = F.softmax(uniform_logits, dim=-1, dtype=torch.float32)

    fused_dq, fused_dweights, fused_dk = indexer_bwd_interface(
        q, weights, k, attn_score, index_score, topk_indices, offsets, use_unfused=False
    )
    ref_dq, ref_dweights, ref_dk = indexer_bwd_interface(
        q, weights, k, attn_score, index_score, topk_indices, offsets, use_unfused=True
    )

    _assert_similarity(fused_dq, ref_dq)
    _assert_similarity(fused_dweights, ref_dweights)
    _assert_similarity(fused_dk, ref_dk)


# ---------------------------------------------------------------------------
# 5. Sparse MLA Top-K ReduceSum
# ---------------------------------------------------------------------------
def test_sparse_mla_topk_reducesum(
    S=4096, H=128, DV=512, D_tail=64, topk=2048, dtype=torch.bfloat16
):
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    DQK = DV + D_tail
    q = torch.randn((S, H, DQK), dtype=dtype, device="cuda")
    kv = torch.randn((S, 1, DQK), dtype=dtype, device="cuda")
    offsets = torch.tensor([0, S], dtype=torch.int32, device="cuda")
    sm_scale = (1.0 / DQK) ** 0.5

    topk_indices = (
        torch.arange(topk, dtype=torch.int32, device="cuda")
        .view(1, 1, topk)
        .expand(S, 1, -1)
        .contiguous()
    )

    _, lse = sparse_mla_fwd_interface(q, kv, topk_indices, offsets, sm_scale, DV, use_unfused=True)

    fused_score = sparse_mla_topk_reducesum_interface(
        q, kv, topk_indices, lse, offsets, DV, sm_scale, use_unfused=False
    )
    ref_score = sparse_mla_topk_reducesum_interface(
        q, kv, topk_indices, lse, offsets, DV, sm_scale, use_unfused=True
    )

    _assert_similarity(fused_score, ref_score)


# ---------------------------------------------------------------------------
# 6. DSAFunction
# ---------------------------------------------------------------------------
def test_dsa_function(
    S=4096,
    B=1,
    H=128,
    HKV=1,
    DQK=576,
    DV=512,
    index_h=64,
    index_d=128,
    topk=2048,
    loss_coeff=1.0,
    dtype=torch.bfloat16,
):
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    device = "cuda"

    query = torch.randn((S, B, H, DQK), dtype=dtype, device=device).requires_grad_(True)
    key = torch.randn((S, B, HKV, DQK), dtype=dtype, device=device).requires_grad_(True)
    index_q = torch.randn((S, B, index_h, index_d), dtype=dtype, device=device).requires_grad_(True)
    index_k = torch.randn((S, B, index_d), dtype=dtype, device=device).requires_grad_(True)
    weights = torch.randn((S, B, index_h), dtype=dtype, device=device).requires_grad_(True)

    sm_scale = (1.0 / DQK) ** 0.5
    do = torch.randn((S, B, H, DV), dtype=dtype, device=device)

    logged_losses = []

    # --- Fused path ---
    fused_out = DSAFunction.apply(
        query,
        key,
        index_q,
        index_k,
        weights,
        None,
        topk,
        DV,
        sm_scale,
        loss_coeff,
        lambda l: logged_losses.append(l.item()),
        None,
        False,
    )
    assert fused_out.shape == (S, B, H, DV)

    fused_out.backward(do)
    assert len(logged_losses) == 1
    assert not torch.isnan(torch.tensor(logged_losses[0]))

    for name, param in [
        ("query", query),
        ("key", key),
        ("index_q", index_q),
        ("index_k", index_k),
        ("weights", weights),
    ]:
        assert param.grad is not None, f"{name}.grad is None"
        assert not param.grad.isnan().any(), f"{name}.grad has NaN"

    fused_dq = query.grad.clone()
    fused_dk = key.grad.clone()
    fused_dindex_q = index_q.grad.clone()
    fused_dindex_k = index_k.grad.clone()
    fused_dweights = weights.grad.clone()

    for p in [query, key, index_q, index_k, weights]:
        p.grad = None

    # --- Unfused path ---
    ref_out = DSAFunction.apply(
        query,
        key,
        index_q,
        index_k,
        weights,
        None,
        topk,
        DV,
        sm_scale,
        loss_coeff,
        None,
        None,
        True,
    )
    assert ref_out.shape == (S, B, H, DV)

    _assert_similarity(fused_out, ref_out)

    ref_out.backward(do)

    _assert_similarity(fused_dq, query.grad)
    _assert_similarity(fused_dk, key.grad)
    _assert_similarity(fused_dindex_q, index_q.grad)
    _assert_similarity(fused_dindex_k, index_k.grad)
    _assert_similarity(fused_dweights, weights.grad)
