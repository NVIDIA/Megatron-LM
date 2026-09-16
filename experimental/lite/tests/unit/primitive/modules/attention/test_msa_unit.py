# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for the MiniMax Sparse Attention primitive (P1).

fp32 is the acceptance gate (TF32 disabled); references are the fp64-capable
``ref/minimax_m3/msa_ref.py`` and Hugging Face ``MiniMaxM3VLAttention``.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

_REF = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "ref", "minimax_m3")
sys.path.insert(0, os.path.abspath(_REF))

pytestmark = pytest.mark.gpus(1)
DEV = "cuda"


def _fp32_env():
    # Transformer Engine fp32 GEMMs use TF32 unless cuBLAS is told otherwise at process start.
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        pytest.skip("fp32 gate needs NVIDIA_TF32_OVERRIDE=0 in the environment (TE GEMMs default to TF32)")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _random_selection(B, H_idx, S, K, block, dtype=torch.float64):
    """HF-format block indices via the fp64 reference indexer on random projections."""
    import msa_ref

    iq = torch.randn(B, H_idx, S, 64, dtype=dtype, device=DEV)
    ik = torch.randn(B, 1, S, 64, dtype=dtype, device=DEV)
    pos = torch.arange(S, device=DEV).unsqueeze(0).expand(B, -1)
    scores = msa_ref.msa_block_scores(iq, ik, pos, block)
    return msa_ref.msa_select_blocks(scores, pos, block, K, 1), pos


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


# --------------------------------------------------------------------------- #
def test_dense_backend_matches_msa_ref_fp64():
    import msa_ref

    from megatron.lite.primitive.kernels import msa_kernels as mk

    torch.manual_seed(0)
    B, Hq, Hkv, S, D, K, blk = 2, 8, 2, 1000, 64, 4, 128
    idx, pos = _random_selection(B, Hkv, S, K, blk)
    q = torch.randn(B, Hq, S, D, dtype=torch.float64, device=DEV)
    k = torch.randn(B, Hkv, S, D, dtype=torch.float64, device=DEV)
    v = torch.randn_like(k)
    out = mk.msa_core_attention(q, k, v, idx, pos, block_size=blk, backend="dense")
    keep = msa_ref.block_indices_to_mask(idx, pos, S, blk, Hq)
    ref = msa_ref.masked_attention(q, k, v, keep, D**-0.5)
    assert (out - ref).abs().max().item() < 1e-12


@pytest.mark.parametrize("S", [1000, 1024, 2048])
def test_flex_backend_matches_dense_fp32_fwd_bwd(S):
    from megatron.lite.primitive.kernels import msa_kernels as mk

    _fp32_env()
    torch.manual_seed(0)
    B, Hq, Hkv, D, K, blk = 2, 8, 2, 128, 4, 128
    idx, pos = _random_selection(B, Hkv, S, K, blk, dtype=torch.float32)
    mk_q = lambda h: torch.randn(B, h, S, D, device=DEV, requires_grad=True)
    q, k, v = mk_q(Hq), mk_q(Hkv), mk_q(Hkv)
    g = torch.randn(B, Hq, S, D, device=DEV)
    out_f = mk.msa_core_attention(q, k, v, idx, pos, block_size=blk, backend="flex")
    out_d = mk.msa_core_attention(q, k, v, idx, pos, block_size=blk, backend="dense")
    assert _rel(out_f, out_d) < 1e-5
    gf = torch.autograd.grad(out_f, (q, k, v), g, retain_graph=True)
    gd = torch.autograd.grad(out_d, (q, k, v), g)
    for a, b in zip(gf, gd):
        assert _rel(a, b) < 1e-5


def test_degenerate_equals_dense_causal_sdpa_fp32():
    from megatron.lite.primitive.kernels import msa_kernels as mk

    _fp32_env()
    torch.manual_seed(0)
    B, Hq, Hkv, S, D, K, blk = 1, 8, 2, 512, 128, 4, 128  # 4 blocks <= top-4 -> dense
    assert mk.msa_is_degenerate(S, blk, K)
    idx, pos = _random_selection(B, Hkv, S, K, blk, dtype=torch.float32)
    q = torch.randn(B, Hq, S, D, device=DEV)
    k = torch.randn(B, Hkv, S, D, device=DEV)
    v = torch.randn_like(k)
    for backend in ("dense", "flex"):
        out = mk.msa_core_attention(q, k, v, idx, pos, block_size=blk, backend=backend)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k.repeat_interleave(Hq // Hkv, 1), v.repeat_interleave(Hq // Hkv, 1), is_causal=True
        )
        assert _rel(out, ref) < 1e-5, backend


# --------------------------------------------------------------------------- #
def _hf_available():
    try:
        import transformers.models.minimax_m3_vl.modeling_minimax_m3_vl  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _hf_available(), reason="transformers without minimax_m3_vl")
@pytest.mark.parametrize("S", [1000, 2048])
def test_msattention_matches_hf_fp32(S):
    """Full module (fused-norm qkv + QK-norm + partial RoPE + indexer + sparse core) vs HF, fp32, TP=1."""
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3VLAttention,
        MiniMaxM3VLRotaryEmbedding,
    )

    from megatron.lite.primitive.modules.attention.msa import MSAttention
    from megatron.lite.primitive.parallel import ParallelState

    _fp32_env()
    torch.manual_seed(0)
    # head_dim 128 as in M3: compiled flex_attention is only true-fp32 for head_dim 128 (D=64 runs at TF32 precision)
    hidden, Hq, Hkv, D, Hidx, Didx, K, blk = 256, 8, 2, 128, 2, 128, 4, 128
    eps, theta = 1e-6, 5e6
    hf_cfg = MiniMaxM3VLTextConfig(
        vocab_size=1024, hidden_size=hidden, num_hidden_layers=1, num_attention_heads=Hq,
        num_key_value_heads=Hkv, head_dim=D, rms_norm_eps=eps, rope_theta=theta, rotary_dim=D // 2,
        partial_rotary_factor=0.5, index_n_heads=Hidx, index_head_dim=Didx, index_block_size=blk,
        index_topk_blocks=K, index_local_blocks=1, layer_types=["minimax_m3_sparse"], mlp_layer_types=["dense"],
        attn_implementation="eager",
    )
    hf = MiniMaxM3VLAttention(hf_cfg, layer_idx=0).to(DEV).float()
    hf_rope = MiniMaxM3VLRotaryEmbedding(hf_cfg).to(DEV)
    for p in hf.parameters():
        torch.nn.init.normal_(p, std=0.05)

    lite = MSAttention(
        hidden, Hq, Hkv, D, ParallelState(), index_n_heads=Hidx, index_head_dim=Didx, block_size=blk,
        topk_blocks=K, rms_norm_eps=eps, rope_theta=theta, rotary_percent=0.5, backend="flex",
    ).to(DEV).float()
    with torch.no_grad():
        # fused pre-attention RMSNorm: gamma = 1 (zero-centered weight 0) so lite(x) == hf(rmsnorm(x))
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
    x_normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    pos = torch.arange(S, device=DEV).unsqueeze(0).expand(B, -1)
    with torch.no_grad():
        cos, sin = hf_rope(x_normed, pos)
        hf_out, _ = hf(x_normed, (cos, sin), attention_mask=None, position_ids=pos)
        hf_idx = hf.indexer(x_normed, (cos, sin), None, pos)
        lite_out = lite(x.transpose(0, 1).contiguous(), position_ids=pos).transpose(0, 1)
        # margin-aware discrete check (thresholds.FP32): a flipped row is acceptable only if its top-k
        # margin is below 10x the observed indexer-score perturbation between the two implementations.
        import msa_ref
        from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import apply_rotary_pos_emb

        from megatron.lite.primitive.utils.rope import _apply_rotary_pos_emb_bshd

        x_sb = x.transpose(0, 1).contiguous()
        xn_l = lite._qkv_lora_input(x_sb)
        freqs = lite.rotary(S)
        iq_l = _apply_rotary_pos_emb_bshd(lite.indexer.q_norm(lite.indexer.q_proj(xn_l).view(S, B, Hidx, Didx)), freqs)
        ik_l = _apply_rotary_pos_emb_bshd(lite.indexer.k_norm(lite.indexer.k_proj(xn_l).view(S, B, 1, Didx)), freqs)
        iq_h = hf.indexer.q_norm(hf.indexer.q_proj(x_normed).view(B, S, Hidx, Didx)).transpose(1, 2)
        ik_h = hf.indexer.k_norm(hf.indexer.k_proj(x_normed).view(B, S, 1, Didx)).transpose(1, 2)
        iq_h, ik_h = apply_rotary_pos_emb(iq_h, ik_h, cos[..., :Didx], sin[..., :Didx])
        sc_l = msa_ref.msa_block_scores(iq_l.permute(1, 2, 0, 3).float(), ik_l.permute(1, 2, 0, 3).float(), pos, blk)
        sc_h = msa_ref.msa_block_scores(iq_h.float(), ik_h.float(), pos, blk)
        finite = torch.isfinite(sc_h)
        perturbation = (sc_l - sc_h)[finite].abs().max().item()
        # margin of the HF selection: min selected score (excluding the forced local block) - max unselected score
        sel = torch.zeros_like(sc_h, dtype=torch.bool).scatter_(-1, hf_idx.clamp(min=0).long(), hf_idx >= 0)
        own = (pos // blk)[:, None, :, None]
        sel_nonlocal = sel & (torch.arange(sc_h.shape[-1], device=DEV)[None, None, None, :] != own)
        min_sel = sc_h.masked_fill(~sel_nonlocal, float("inf")).amin(-1)
        max_unsel = sc_h.masked_fill(sel | ~finite, float("-inf")).amax(-1)
        margin = (min_sel - max_unsel).nan_to_num(posinf=1e9)
    lite_sets = lite.last_block_indices.sort(-1).values
    hf_sets = hf_idx.sort(-1).values.to(torch.int32)
    mismatch = (lite_sets != hf_sets).any(-1)
    unexplained = mismatch & ~(margin < 10.0 * perturbation)
    assert int(unexplained.sum()) == 0, (int(mismatch.sum()), perturbation)
    assert mismatch.float().mean().item() < 1e-2
    ok = ~mismatch.any(1)  # [B, S]: tokens whose selection matches in every group -> outputs must meet the fp32 gate
    assert _rel(lite_out[ok], hf_out[ok]) < 1e-5, _rel(lite_out[ok], hf_out[ok])
    print(f"S={S}: indexer score perturbation {perturbation:.2e}, flipped rows {int(mismatch.sum())}/{mismatch.numel()}")
