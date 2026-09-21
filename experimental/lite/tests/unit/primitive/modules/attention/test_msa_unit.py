# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax Sparse Attention primitive (flex backend), bf16 on one GPU.

References are pure-torch masked dense attention on the same block selection (softmax in
fp32) and, for the full ``MSAttention`` module, HF ``MiniMaxM3VLAttention`` with identical
weights. bf16 flips a few percent of the indexer's top-k rows whenever the GEMM order
changes, so selections are compared as sets with a flip budget and outputs are compared
on the rows whose selection agrees; per-case numbers are printed as evidence.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.gpus(1)
DEV = "cuda"
KERNEL_REL = 2e-2  # bf16 kernel vs fp32 masked reference, rel-to-max
GRAD_REL = 5e-2
GRAD_COS = 0.999
TOPK_FLIP_BUDGET = 0.2  # fraction of (batch, head, token) rows whose top-k set may differ from HF in bf16
MATCHED_ROWS_REL = 5e-2


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _block_scores(iq, ik, pos, block):
    B, H, S_q, _ = iq.shape
    S_k = ik.shape[2]
    n_kv = -(-S_k // block)
    scores = torch.matmul(iq, ik.transpose(-1, -2))
    scores = scores.masked_fill(torch.arange(S_k, device=iq.device)[None, None, None, :] > pos[:, None, :, None], float("-inf"))
    scores = F.pad(scores, (0, n_kv * block - S_k), value=float("-inf"))
    return scores.view(B, H, S_q, n_kv, block).amax(-1)


def _select_blocks(scores, pos, block, topk):
    scores = scores.clone()
    local = (pos // block)[:, None, :, None].expand(-1, scores.shape[1], -1, 1)
    scores.scatter_(-1, local, float("inf"))
    top_scores, top_idx = scores.topk(min(topk, scores.shape[-1]), dim=-1)
    return top_idx.masked_fill(top_scores == float("-inf"), -1)


def _random_selection(B, H_idx, S, K, block):
    iq = torch.randn(B, H_idx, S, 64, device=DEV)
    ik = torch.randn(B, 1, S, 64, device=DEV)
    pos = torch.arange(S, device=DEV).unsqueeze(0).expand(B, -1)
    return _select_blocks(_block_scores(iq, ik, pos, block), pos, block, K), pos


def _masked_reference(q, k, v, idx, pos, block):
    """fp32 dense GQA attention restricted to the selected blocks (and causal)."""
    B, H_idx, S_q, _ = idx.shape
    S_k = k.shape[2]
    n_kv = -(-S_k // block)
    keep = torch.zeros(B, H_idx, S_q, n_kv + 1, dtype=torch.bool, device=DEV)
    keep.scatter_(-1, idx.masked_fill(idx < 0, n_kv).long(), True)
    keep = keep[..., :n_kv].repeat_interleave(block, dim=-1)[..., :S_k]
    keep = keep.repeat_interleave(q.shape[1] // H_idx, dim=1)
    keep &= ~(torch.arange(S_k, device=DEV)[None, None, None, :] > pos[:, None, :, None])
    n_rep = q.shape[1] // k.shape[1]
    q, k, v = q.float(), k.float().repeat_interleave(n_rep, 1), v.float().repeat_interleave(n_rep, 1)
    scores = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    return torch.softmax(scores.masked_fill(~keep, float("-inf")), dim=-1) @ v


@pytest.mark.parametrize("S", [1000, 1024, 2048])
def test_flex_matches_masked_reference_forward_backward_bf16(S):
    from megatron.lite.primitive.kernels import msa_kernels as mk

    torch.manual_seed(0)
    B, Hq, Hkv, D, K, blk = 2, 8, 2, 128, 4, 128
    idx, pos = _random_selection(B, Hkv, S, K, blk)
    q = torch.randn(B, Hq, S, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, Hkv, S, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, Hkv, S, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(B, Hq, S, D, device=DEV, dtype=torch.bfloat16)

    out = mk.msa_core_attention(q, k, v, idx, pos, block_size=blk, backend="flex")
    q32, k32, v32 = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    ref = _masked_reference(q32, k32, v32, idx, pos, blk)
    fwd_rel = _rel(out, ref)
    grads = torch.autograd.grad(out, (q, k, v), g)
    ref_grads = torch.autograd.grad(ref, (q32, k32, v32), g.float())
    grad_rel = [_rel(a, b) for a, b in zip(grads, ref_grads)]
    grad_cos = [_cos(a, b) for a, b in zip(grads, ref_grads)]
    print(f"msa_flex_vs_masked_ref S={S} fwd_rel={fwd_rel:.3e} grad_rel={[f'{r:.2e}' for r in grad_rel]} grad_cos={[f'{c:.5f}' for c in grad_cos]}")
    assert fwd_rel < KERNEL_REL
    assert max(grad_rel) < GRAD_REL and min(grad_cos) > GRAD_COS


def test_degenerate_selection_equals_causal_sdpa_bf16():
    from megatron.lite.primitive.kernels import msa_kernels as mk

    torch.manual_seed(0)
    B, Hq, Hkv, S, D, K, blk = 1, 8, 2, 512, 128, 4, 128  # 4 blocks <= top-4: every query sees every visible key
    assert mk.msa_is_degenerate(S, blk, K)
    idx, pos = _random_selection(B, Hkv, S, K, blk)
    q = torch.randn(B, Hq, S, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, S, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out = mk.msa_core_attention(q, k, v, idx, pos, block_size=blk, backend="flex")
    ref = F.scaled_dot_product_attention(q, k.repeat_interleave(Hq // Hkv, 1), v.repeat_interleave(Hq // Hkv, 1), is_causal=True)
    rel = _rel(out, ref)
    print(f"msa_flex_degenerate_vs_sdpa rel={rel:.3e}")
    assert rel < KERNEL_REL


def _hf_available():
    try:
        import transformers.models.minimax_m3_vl.modeling_minimax_m3_vl  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _hf_available(), reason="transformers without minimax_m3_vl")
@pytest.mark.parametrize("S", [1000, 2048])
def test_msattention_matches_hf_bf16(S):
    """Full module (fused-norm qkv + QK-norm + partial RoPE + indexer + flex core) vs HF eager, bf16, TP=1."""
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3VLAttention,
        MiniMaxM3VLRotaryEmbedding,
    )

    from megatron.lite.primitive.modules.attention.msa import MSAttention
    from megatron.lite.primitive.parallel import ParallelState

    torch.manual_seed(0)
    hidden, Hq, Hkv, D, Hidx, Didx, K, blk = 256, 8, 2, 128, 2, 128, 4, 128
    eps, theta = 1e-6, 5e6
    hf_cfg = MiniMaxM3VLTextConfig(
        vocab_size=1024, hidden_size=hidden, num_hidden_layers=1, num_attention_heads=Hq,
        num_key_value_heads=Hkv, head_dim=D, rms_norm_eps=eps, rope_theta=theta, rotary_dim=D // 2,
        partial_rotary_factor=0.5, index_n_heads=Hidx, index_head_dim=Didx, index_block_size=blk,
        index_topk_blocks=K, index_local_blocks=1, layer_types=["minimax_m3_sparse"], mlp_layer_types=["dense"],
        attn_implementation="eager",
    )
    hf = MiniMaxM3VLAttention(hf_cfg, layer_idx=0).to(DEV)
    for p in hf.parameters():
        torch.nn.init.normal_(p, std=0.05)
    hf = hf.to(torch.bfloat16)
    hf_rope = MiniMaxM3VLRotaryEmbedding(hf_cfg).to(DEV)

    lite = MSAttention(
        hidden, Hq, Hkv, D, ParallelState(), index_n_heads=Hidx, index_head_dim=Didx, block_size=blk,
        topk_blocks=K, rms_norm_eps=eps, rope_theta=theta, rotary_percent=0.5, backend="flex",
    ).to(DEV).to(torch.bfloat16)
    with torch.no_grad():
        # fused pre-attention RMSNorm: gamma = 1 (zero-centred weight 0) so lite(x) == hf(rmsnorm(x))
        lite.qkv.linear.layer_norm_weight.zero_()
        lite.qkv.linear.weight.copy_(torch.cat([hf.q_proj.weight, hf.k_proj.weight, hf.v_proj.weight], 0))
        lite.proj.linear.weight.copy_(hf.o_proj.weight)
        lite.q_norm.weight.copy_(hf.q_norm.weight)
        lite.k_norm.weight.copy_(hf.k_norm.weight)
        lite.indexer.q_proj.linear.weight.copy_(hf.indexer.q_proj.weight)
        lite.indexer.k_proj.weight.copy_(hf.indexer.k_proj.weight)
        lite.indexer.q_norm.weight.copy_(hf.indexer.q_norm.weight)
        lite.indexer.k_norm.weight.copy_(hf.indexer.k_norm.weight)

    B = 2
    x = torch.randn(B, S, hidden, device=DEV)
    x_normed = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16)
    pos = torch.arange(S, device=DEV).unsqueeze(0).expand(B, -1)
    with torch.no_grad():
        cos, sin = hf_rope(x_normed, pos)
        hf_out, _ = hf(x_normed, (cos, sin), attention_mask=None, position_ids=pos)
        hf_idx = hf.indexer(x_normed, (cos, sin), None, pos)
        lite_out = lite(x.to(torch.bfloat16).transpose(0, 1).contiguous(), position_ids=pos).transpose(0, 1)

    lite_sets = lite.last_block_indices.sort(-1).values
    hf_sets = hf_idx.sort(-1).values.to(torch.int32)
    mismatch = (lite_sets != hf_sets).any(-1)  # [B, H_idx, S]
    flip_frac = mismatch.float().mean().item()
    matched = ~mismatch.any(1)  # [B, S]: every index head agrees -> outputs must agree at kernel precision
    matched_rel = _rel(lite_out[matched], hf_out[matched])
    cos_all = _cos(lite_out, hf_out)
    print(f"msattention_vs_hf_bf16 S={S} topk_flip_rows={flip_frac:.3%} matched_rows_rel={matched_rel:.3e} cos_all={cos_all:.6f}")
    assert flip_frac < TOPK_FLIP_BUDGET
    assert matched_rel < MATCHED_ROWS_REL
    assert cos_all > 0.999
