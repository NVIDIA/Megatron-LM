# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for ``MSAttention`` (MagiAttention MSA extension + msa_v1 kernels), CP=1.

The reference reuses the module's own projections/norms/RoPE/indexer projections and runs the
indexer scoring + top-k and the block-sparse attention as fp32 masked dense torch, one document
at a time. The msa_v1 kernels are bf16-only so thresholds are bf16-level.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.gpus(1, min_architecture="blackwell"), pytest.mark.optional]
DEV = "cuda"
HIDDEN, HQ, HKV, D, HIDX, DIDX, BLK, TOPK = 512, 64, 4, 128, 4, 128, 128, 16
CHUNK = 512

magi_msa = pytest.importorskip("megatron.lite.primitive.kernels.magi_msa")
pytest.importorskip("magi_attn_extensions.MSA")
pytest.importorskip("msa_v1")


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


def _build(ps, seed: int = 0):
    from megatron.lite.primitive.modules.attention.msa import MSAttention

    torch.manual_seed(seed)
    m = MSAttention(
        HIDDEN, HQ, HKV, D, ps, index_n_heads=HIDX, index_head_dim=DIDX, block_size=BLK, topk_blocks=TOPK,
        local_blocks=1, rotary_percent=0.5, rope_theta=5_000_000.0,
    )
    return m.to(DEV).to(torch.bfloat16)


@pytest.fixture(scope="module")
def pair():
    from megatron.lite.primitive.parallel import ParallelState

    magi_msa.ensure_single_process_group()
    ps = ParallelState()
    ref = _build(ps)
    magi = _build(ps)
    magi.load_state_dict(ref.state_dict(), strict=True)
    return ps, ref, magi


def _ctx(ps, seq_lens):
    settings = magi_msa.MagiMsaSettings(chunk_size=CHUNK)
    cfg = magi_msa.build_msa_config(settings)
    return magi_msa.plan_magi_batch(seq_lens, ps=ps, settings=settings, msa_config=cfg, need_dense_key=False)


def _select_blocks(iq, ik, pos):
    """fp32 indexer scores ``[1, H_idx, S, D] x [1, 1, S, D]`` -> causal block max -> own block forced -> top-k."""
    S = ik.shape[2]
    n_kv = -(-S // BLK)
    scores = (iq @ ik.transpose(-1, -2)).masked_fill(torch.arange(S, device=DEV) > pos[:, None, :, None], float("-inf"))
    scores = F.pad(scores, (0, n_kv * BLK - S), value=float("-inf")).view(*scores.shape[:3], n_kv, BLK).amax(-1)
    scores.scatter_(-1, (pos // BLK)[:, None, :, None].expand(-1, scores.shape[1], -1, 1), float("inf"))
    top_scores, top_idx = scores.topk(min(TOPK, n_kv), dim=-1)
    return top_idx.masked_fill(top_scores == float("-inf"), -1)


def _masked_attention(q, k, v, idx, pos):
    """fp32 dense GQA attention restricted to the selected blocks (and causal)."""
    S, n_kv = k.shape[2], -(-k.shape[2] // BLK)
    keep = torch.zeros(*idx.shape[:3], n_kv + 1, dtype=torch.bool, device=DEV)
    keep.scatter_(-1, idx.masked_fill(idx < 0, n_kv).long(), True)
    keep = keep[..., :n_kv].repeat_interleave(BLK, dim=-1)[..., :S].repeat_interleave(HQ // HIDX, dim=1)
    keep &= torch.arange(S, device=DEV) <= pos[:, None, :, None]
    k, v = k.float().repeat_interleave(HQ // HKV, 1), v.float().repeat_interleave(HQ // HKV, 1)
    scores = (q.float() @ k.transpose(-1, -2)) * D**-0.5
    return torch.softmax(scores.masked_fill(~keep, float("-inf")), dim=-1) @ v


def _torch_ref(m, x_docs):
    from megatron.lite.primitive.utils.rope import _apply_rotary_pos_emb_bshd

    outs = []
    for x in x_docs:
        S = x.shape[0]
        pos = torch.arange(S, device=DEV)[None]
        q, _gate, k, v = m._split_qkv(m.qkv(x))
        freqs = m.rotary(S)
        q = _apply_rotary_pos_emb_bshd(m.q_norm(q), freqs)
        k = _apply_rotary_pos_emb_bshd(m.k_norm(k), freqs)
        iq, ik = m.indexer.index_qk(m._qkv_lora_input(x), freqs)
        idx = _select_blocks(iq.permute(1, 2, 0, 3).float(), ik.permute(1, 2, 0, 3).float(), pos)
        out = _masked_attention(*(t.permute(1, 2, 0, 3) for t in (q, k, v)), idx, pos)
        outs.append(m.proj(out.to(x.dtype).permute(2, 0, 1, 3).reshape(S, 1, HQ * D)))
    return torch.cat(outs, dim=0)


def _grads(m):
    names = ["qkv.linear.weight", "qkv.linear.layer_norm_weight", "proj.linear.weight", "q_norm.weight", "k_norm.weight"]
    params = dict(m.named_parameters())
    return {n: params[n].grad for n in names}


def test_magi_matches_torch_single_doc(pair):
    ps, ref_m, magi = pair
    S = 4096  # 32 KV blocks > topk 16 -> genuinely sparse
    torch.manual_seed(1)
    x = torch.randn(S, 1, HIDDEN, device=DEV, dtype=torch.bfloat16)
    g = torch.randn(S, 1, HIDDEN, device=DEV, dtype=torch.bfloat16) * 0.1
    ctx = _ctx(ps, [S])
    assert ctx.pad == 0
    ref = _torch_ref(ref_m, [x])
    out = magi(x, magi_ctx=ctx)
    assert out.shape == ref.shape
    c, r = _cos(out, ref), _rel(out, ref)
    print(f"[magi vs torch] single-doc cos={c:.6f} rel={r:.3e}")
    assert c >= 0.999 and r <= 5e-3
    ref_m.zero_grad(set_to_none=True)
    magi.zero_grad(set_to_none=True)
    (ref * g).sum().backward()
    (out * g).sum().backward()
    gf, gm = _grads(ref_m), _grads(magi)
    for n in gf:
        assert gf[n] is not None and gm[n] is not None, n
        cg = _cos(gm[n], gf[n])
        print(f"[grad] {n}: cos={cg:.6f}")
        assert cg >= 0.99, n
    for n, p in magi.indexer.named_parameters():
        assert p.grad is None and not p.requires_grad, n
    for n, p in ref_m.indexer.named_parameters():
        assert p.grad is None, n


def test_magi_packed_two_docs_matches_per_doc_torch(pair):
    ps, ref_m, magi = pair
    lens = [1536, 2560]
    torch.manual_seed(2)
    xs = [torch.randn(L, 1, HIDDEN, device=DEV, dtype=torch.bfloat16) for L in lens]
    ctx = _ctx(ps, lens)
    assert ctx.pad == 0 and ctx.num_real_docs == 2
    ref = _torch_ref(ref_m, xs)
    out = magi(torch.cat(xs, 0), magi_ctx=ctx)
    c, r = _cos(out, ref), _rel(out, ref)
    print(f"[magi vs torch] packed 2 docs cos={c:.6f} rel={r:.3e}")
    assert c >= 0.999 and r <= 5e-3


def test_magi_cp1_odd_length_is_not_padded(pair):
    ps, ref_m, magi = pair
    S = 3000
    torch.manual_seed(3)
    x = torch.randn(S, 1, HIDDEN, device=DEV, dtype=torch.bfloat16)
    ctx = _ctx(ps, [S])
    assert ctx.pad == 0 and ctx.cu_seqlens_host[-1] == S
    out = magi(x, magi_ctx=ctx)
    ref = _torch_ref(ref_m, [x])
    c, r = _cos(out, ref), _rel(out, ref)
    print(f"[magi vs torch] cp1 unpadded cos={c:.6f} rel={r:.3e}")
    assert c >= 0.999 and r <= 5e-3


def test_magi_rejects_missing_ctx_and_tp(pair):
    from megatron.lite.primitive.parallel import ParallelState

    ps, _ref, magi = pair
    x = torch.randn(CHUNK, 1, HIDDEN, device=DEV, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        magi(x)
    ps_tp2 = ParallelState(tp_size=2)
    with pytest.raises(NotImplementedError):
        _build(ps_tp2)
