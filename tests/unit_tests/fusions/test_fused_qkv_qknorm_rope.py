# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

try:
    import transformer_engine as te
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb as te_rope

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from megatron.core.fusions import fused_qkv_qknorm_rope as fused_module
from megatron.core.fusions.fused_qkv_qknorm_rope import HAVE_TRITON, fused_qkv_qknorm_rope

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TE and HAVE_TRITON), reason="needs CUDA, TE and Triton"
)


def _freqs(sq, hn, base=1e6):
    inv = 1.0 / (base ** (torch.arange(0, hn, 2, device="cuda", dtype=torch.float32) / hn))
    f = torch.outer(torch.arange(sq, device="cuda", dtype=torch.float32), inv)
    return torch.cat((f, f), dim=-1)[:, None, None, :]


def _unfused(mixed, freqs, qn, kn, nq, hn):
    """MCore's split path: strided split, reshape, TE RMSNorm, TE fused RoPE."""
    q, k, v = torch.split(mixed, [nq * hn, hn, hn], dim=3)
    q = q.reshape(q.size(0), q.size(1), -1, hn)
    q = te_rope(qn(q), freqs, tensor_format="sbhd", fused=True)
    k = te_rope(kn(k), freqs, tensor_format="sbhd", fused=True)
    return q, k, v


def _fp64_oracle(mixed, freqs, qw, kw, nq, hn, eps, zc):
    x = mixed.double()
    q, k, v = torch.split(x, [nq * hn, hn, hn], dim=3)
    q = q.reshape(q.size(0), q.size(1), -1, hn)
    cos, sin = freqs.double().cos(), freqs.double().sin()

    def nr(t, w):
        t = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps) * (w.double() + (1 if zc else 0))
        t1, t2 = t.chunk(2, -1)
        return t * cos + torch.cat((-t2, t1), -1) * sin

    return nr(q, qw), nr(k, kw), v


@pytest.mark.parametrize(
    "sq,b,ng,nq,hn", [(256, 2, 4, 8, 128), (128, 3, 2, 4, 64), (64, 1, 1, 1, 128)]
)
@pytest.mark.parametrize("zc", [False, True])
# None: automatic (1 token per program at these sizes); 7: the last program gets a partial block.
@pytest.mark.parametrize("bwd_tokens", [None, 7])
def test_matches_unfused_te_path(sq, b, ng, nq, hn, zc, bwd_tokens, monkeypatch):
    monkeypatch.setattr(fused_module, "BWD_TOKENS", bwd_tokens)
    torch.manual_seed(0)
    eps = 1e-6
    row = (nq + 2) * hn
    mixed = torch.randn(sq, b, ng * row, device="cuda", dtype=torch.bfloat16).view(sq, b, ng, row)
    freqs = _freqs(sq, hn)
    qn = te.pytorch.RMSNorm(hn, eps=eps, zero_centered_gamma=zc, params_dtype=torch.bfloat16).cuda()
    kn = te.pytorch.RMSNorm(hn, eps=eps, zero_centered_gamma=zc, params_dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        qn.weight.copy_(torch.randn(hn) * 0.1 + (0 if zc else 1))
        kn.weight.copy_(torch.randn(hn) * 0.1 + (0 if zc else 1))

    m_ref = mixed.clone().requires_grad_(True)
    ref = _unfused(m_ref, freqs, qn, kn, nq, hn)
    grads = [torch.randn_like(t) for t in ref]
    torch.autograd.backward(ref, grads)
    ref_dqw, ref_dkw = qn.weight.grad.clone(), kn.weight.grad.clone()
    qn.weight.grad = kn.weight.grad = None

    m_fus = mixed.clone().requires_grad_(True)
    out = fused_qkv_qknorm_rope(m_fus, freqs, freqs, qn.weight, kn.weight, nq, eps, zc)
    torch.autograd.backward(out, grads)

    for o, r in zip(out, ref):
        assert o.shape == r.shape and o.is_contiguous()
    # Same op order and rounding points as the TE path. A rare rounding flip of the bf16 norm output
    # (values of order 1) can survive RoPE even where cos/sin terms nearly cancel, so the bound is
    # absolute at bf16 resolution, plus a bound on how often outputs differ at all.
    for o, r in zip(out, ref):
        torch.testing.assert_close(o, r, atol=1.6e-2, rtol=1.6e-2)
        assert (o != r).float().mean().item() < 1e-4
    torch.testing.assert_close(m_fus.grad, m_ref.grad, atol=1.6e-2, rtol=1.6e-2)
    assert (m_fus.grad != m_ref.grad).float().mean().item() < 1e-3
    torch.testing.assert_close(qn.weight.grad.float(), ref_dqw.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(kn.weight.grad.float(), ref_dkw.float(), atol=1e-2, rtol=1e-2)

    # Not less accurate than the unfused path against a float64 oracle.
    ora = _fp64_oracle(mixed, freqs, qn.weight, kn.weight, nq, hn, eps, zc)
    for o, r, x in zip(out, ref, ora):
        assert (o.double() - x).abs().max() <= (r.double() - x).abs().max() * 1.01 + 1e-6


def test_bwd_tokens_follow_work_per_sm(monkeypatch):
    monkeypatch.setattr(fused_module, "BWD_TOKENS", None)
    # Largest power of 2 up to 8 leaving at least 32 programs per SM.
    picks = [fused_module._bwd_tokens(t, 4, 100) for t in (1024, 1600, 3200, 6400, 65536)]
    assert picks == [1, 2, 4, 8, 8]
    monkeypatch.setattr(fused_module, "BWD_TOKENS", 16)
    assert fused_module._bwd_tokens(1024, 4, 100) == 16


def test_value_and_layout_copy_exact():
    sq, b, ng, nq, hn = 32, 2, 2, 4, 128
    mixed = torch.randn(sq, b, ng, (nq + 2) * hn, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(hn, device="cuda", dtype=torch.bfloat16)
    q, k, v = fused_qkv_qknorm_rope(mixed, _freqs(sq, hn), None, w, w, nq, 1e-6)
    assert torch.equal(v, mixed[..., (nq + 1) * hn :])
    assert q.shape == (sq, b, ng * nq, hn) and k.shape == (sq, b, ng, hn)


def test_self_attention_qk_layernorm_fused_matches_unfused():
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.utils import PARAM_READY_CALLBACK_ATTR
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        cfg = TransformerConfig(
            num_layers=1,
            hidden_size=512,
            num_attention_heads=8,
            num_query_groups=2,
            kv_channels=64,
            qk_layernorm=True,
            normalization="RMSNorm",
            layernorm_epsilon=1e-6,
            apply_rope_fusion=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True)
        attn = build_module(spec.submodules.self_attention, config=cfg, layer_number=1).cuda()
        assert attn._can_fuse_qk_norm_rope()
        sq, b = 128, 2
        emb = RotaryEmbedding(kv_channels=64, rotary_percent=1.0, rotary_base=1e6).cuda()(sq)
        x = torch.randn(sq, b, 512, device="cuda", dtype=torch.bfloat16)
        gout = torch.randn_like(x)
        # Stand-ins for a sharding backend's publication callbacks (e.g. DDP overlap_param_gather).
        published = []
        norm_weights = {"q": attn.q_layernorm.weight, "k": attn.k_layernorm.weight}
        for name, w in norm_weights.items():
            setattr(w, PARAM_READY_CALLBACK_ATTR, lambda name=name: published.append(name))
        res = []
        for flag in (False, True):
            cfg.fused_single_qkv_rope = flag
            attn.zero_grad(set_to_none=True)
            xi = x.clone().requires_grad_(True)
            out, _ = attn(xi, attention_mask=None, rotary_pos_emb=emb)
            out.backward(gout)
            res.append((out, xi.grad, attn.q_layernorm.weight.grad, attn.k_layernorm.weight.grad))
        for a, r in zip(res[1], res[0]):
            torch.testing.assert_close(a.float(), r.float(), atol=2e-2, rtol=2e-2)
        # The fused pass skips the norm modules' pre-hooks, so it must publish both weights itself.
        assert published == ["q", "k"]
        for w in norm_weights.values():
            delattr(w, PARAM_READY_CALLBACK_ATTR)
    finally:
        Utils.destroy_model_parallel()
