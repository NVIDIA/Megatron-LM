# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for the ``magi`` MSA backend (MagiAttention MSA extension + msa_v1 kernels), CP=1.

The ``flex`` backend (exact, fp32-validated in ``test_msa_unit.py``) is the reference; the msa_v1
kernels are bf16-only so thresholds are bf16-level (P1 measured official-kernel-vs-fp32 rel ~2e-3).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.gpus(1, min_architecture="blackwell")
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


def _build(backend: str, ps, seed: int = 0):
    from megatron.lite.primitive.modules.attention.msa import MSAttention

    torch.manual_seed(seed)
    m = MSAttention(
        HIDDEN, HQ, HKV, D, ps, index_n_heads=HIDX, index_head_dim=DIDX, block_size=BLK, topk_blocks=TOPK,
        local_blocks=1, backend=backend, rotary_percent=0.5, rope_theta=5_000_000.0,
    )
    return m.to(DEV).to(torch.bfloat16)


@pytest.fixture(scope="module")
def pair():
    from megatron.lite.primitive.parallel import ParallelState

    magi_msa.ensure_single_process_group()
    ps = ParallelState()
    flex = _build("flex", ps)
    magi = _build("magi", ps)
    magi.load_state_dict(flex.state_dict(), strict=True)
    return ps, flex, magi


def _ctx(ps, seq_lens):
    settings = magi_msa.MagiMsaSettings(chunk_size=CHUNK)
    cfg = magi_msa.build_msa_config(settings)
    return magi_msa.plan_magi_batch(seq_lens, ps=ps, settings=settings, msa_config=cfg, need_dense_key=False)


def _flex_ref(flex, x_docs):
    outs = []
    for x in x_docs:
        pos = torch.arange(x.shape[0], device=DEV)[None]
        outs.append(flex(x, position_ids=pos))
    return torch.cat(outs, dim=0)


def _grads(m):
    names = ["qkv.linear.weight", "qkv.linear.layer_norm_weight", "proj.linear.weight", "q_norm.weight", "k_norm.weight"]
    params = dict(m.named_parameters())
    return {n: params[n].grad for n in names}


def test_magi_matches_flex_single_doc(pair):
    ps, flex, magi = pair
    S = 4096  # 32 KV blocks > topk 16 -> genuinely sparse
    torch.manual_seed(1)
    x = torch.randn(S, 1, HIDDEN, device=DEV, dtype=torch.bfloat16)
    g = torch.randn(S, 1, HIDDEN, device=DEV, dtype=torch.bfloat16) * 0.1
    ctx = _ctx(ps, [S])
    assert ctx.pad == 0
    ref = _flex_ref(flex, [x])
    out = magi(x, magi_ctx=ctx)
    assert out.shape == ref.shape
    c, r = _cos(out, ref), _rel(out, ref)
    print(f"[magi vs flex] single-doc cos={c:.6f} rel={r:.3e}")
    assert c >= 0.999 and r <= 5e-3
    flex.zero_grad(set_to_none=True)
    magi.zero_grad(set_to_none=True)
    (ref * g).sum().backward()
    (out * g).sum().backward()
    gf, gm = _grads(flex), _grads(magi)
    for n in gf:
        assert gf[n] is not None and gm[n] is not None, n
        cg = _cos(gm[n], gf[n])
        print(f"[grad] {n}: cos={cg:.6f}")
        assert cg >= 0.99, n
    for n, p in magi.indexer.named_parameters():
        assert p.grad is None and not p.requires_grad, n
    for n, p in flex.indexer.named_parameters():
        assert p.grad is None, n


def test_magi_packed_two_docs_matches_per_doc_flex(pair):
    ps, flex, magi = pair
    lens = [1536, 2560]
    torch.manual_seed(2)
    xs = [torch.randn(L, 1, HIDDEN, device=DEV, dtype=torch.bfloat16) for L in lens]
    ctx = _ctx(ps, lens)
    assert ctx.pad == 0 and ctx.num_real_docs == 2
    ref = _flex_ref(flex, xs)
    out = magi(torch.cat(xs, 0), magi_ctx=ctx)
    c, r = _cos(out, ref), _rel(out, ref)
    print(f"[magi vs flex] packed 2 docs cos={c:.6f} rel={r:.3e}")
    assert c >= 0.999 and r <= 5e-3


def test_magi_padding_doc_does_not_change_real_tokens(pair):
    ps, flex, magi = pair
    S = 3000
    torch.manual_seed(3)
    x = torch.randn(S, 1, HIDDEN, device=DEV, dtype=torch.bfloat16)
    ctx = _ctx(ps, [S])
    assert ctx.pad == (-S) % CHUNK and ctx.cu_seqlens_host[-1] == S + ctx.pad
    x_pad = torch.cat([x, torch.zeros(ctx.pad, 1, HIDDEN, device=DEV, dtype=torch.bfloat16)], 0)
    out = magi(x_pad, magi_ctx=ctx)[:S]
    ref = _flex_ref(flex, [x])
    c, r = _cos(out, ref), _rel(out, ref)
    print(f"[magi vs flex] padded cos={c:.6f} rel={r:.3e}")
    assert c >= 0.999 and r <= 5e-3


def test_magi_rejects_missing_ctx_and_tp(pair):
    from megatron.lite.primitive.parallel import ParallelState

    ps, _flex, magi = pair
    x = torch.randn(CHUNK, 1, HIDDEN, device=DEV, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        magi(x)
    ps_tp2 = ParallelState(tp_size=2)
    with pytest.raises(NotImplementedError):
        _build("magi", ps_tp2)
