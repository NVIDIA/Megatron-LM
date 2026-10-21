# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import math

import pytest
import torch

from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.exact_attention.autograd import thd_paging, zigzag_positions
from megatron.core.transformer.exact_attention.cp_plan import build_cp_plan
from megatron.core.transformer.transformer_config import TransformerConfig


def _config(**kw):
    base = dict(
        num_layers=2,
        hidden_size=512,
        num_attention_heads=8,
        num_query_groups=2,
        kv_channels=128,
        attention_backend=AttnBackend.exact,
        attention_dropout=0.0,
        bf16=True,
    )
    base.update(kw)
    return TransformerConfig(**base)


class TestExactAttentionConfig:
    def test_valid(self):
        assert _config().exact_attention_super_block_tiles == 8

    @pytest.mark.parametrize(
        "kw",
        [
            dict(kv_channels=64),
            dict(attention_dropout=0.1),
            dict(window_size=(128, 0)),
            dict(attn_logit_softcapping=30.0),
            dict(softmax_type='learnable'),
            dict(exact_attention_super_block_tiles=0),
            dict(context_parallel_size=2, cp_comm_type='p2p'),
            dict(context_parallel_size=2, cp_comm_type='a2a+p2p'),
        ],
    )
    def test_rejected(self, kw):
        with pytest.raises(ValueError):
            _config(**kw)

    @pytest.mark.parametrize("comm", ['all_gather', 'a2a'])
    def test_cp_comm_types(self, comm):
        _config(context_parallel_size=2, cp_comm_type=comm)


class TestExactAttentionLayouts:
    def test_thd_paging(self):
        cu = torch.tensor([0, 130, 130 + 256, 130 + 256 + 7], dtype=torch.int32)
        page_table, seqused, rows, num_pages, max_len = thd_paging(cu)
        assert seqused.tolist() == [130, 256, 7]
        assert num_pages == 2 + 2 + 1 and max_len == 256
        assert page_table[0, :2].tolist() == [0, 1] and page_table[2, 0].item() == 4
        assert rows.unique().numel() == rows.numel() == 393
        assert rows[129].item() == 128 + 1 and rows[130].item() == 2 * 128

    @pytest.mark.parametrize("cp", [2, 4])
    def test_zigzag_positions_match_plan(self, cp):
        cu = torch.tensor([0, 64, 64 + 128, 64 + 128 + 192], dtype=torch.int32)
        pos = zigzag_positions(cu, cp)
        plans = [build_cp_plan(cu, cp, r, 2, "cpu") for r in range(cp)]
        assert torch.equal(pos, torch.cat([p.local_positions for p in plans]))
        assert torch.equal(pos.sort().values, torch.arange(int(cu[-1])))


def _supported_gpu():
    if not torch.cuda.is_available():
        return False
    try:
        from megatron.core.transformer.exact_attention.api import HAVE_EXACT_ATTENTION
    except ImportError:
        return False
    return HAVE_EXACT_ATTENTION and torch.cuda.get_device_capability()[0] == 10


gpu = pytest.mark.skipif(not _supported_gpu(), reason="needs a compute capability 10.x GPU, cutlass-dsl and flash-attn-4")
HQ, HKV = 8, 2


def _bits(t):
    return t.contiguous().view(torch.int16)


def _rel(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return ((a - b).norm() / b.norm()).item()


def _ref_grads(q, k, v, w, scale):
    """FP64 causal GQA attention on one sequence [L, H, D]: out and grads of sum(out * w)."""
    qd, kd, vd = (t.double().detach().requires_grad_() for t in (q, k, v))
    rep = q.shape[1] // k.shape[1]
    kk = kd.repeat_interleave(rep, 1).transpose(0, 1)
    vv = vd.repeat_interleave(rep, 1).transpose(0, 1)
    s = torch.einsum("hqd,hkd->hqk", qd.transpose(0, 1), kk) * scale
    n = q.shape[0]
    s = s.masked_fill(~torch.ones(n, n, dtype=torch.bool, device=q.device).tril(), float("-inf"))
    o = torch.einsum("hqk,hkd->hqd", s.softmax(-1), vv).transpose(0, 1)
    (o * w.double()).sum().backward()
    return o.detach(), qd.grad, kd.grad, vd.grad


@gpu
class TestExactAttentionGpu:
    def _setup(self, d, g):
        from megatron.core.transformer.exact_attention.api import ExactAttention

        attn = ExactAttention(
            "cuda", head_dim=d, num_q_heads=HQ, num_kv_heads=HKV, super_block_tiles=g
        )
        gen = torch.Generator(device="cuda").manual_seed(1234)

        def rand(*shape):
            return torch.randn(*shape, device="cuda", generator=gen).to(torch.bfloat16)

        return attn, rand

    @pytest.mark.parametrize("d", [128, 256])
    @pytest.mark.parametrize("g", [8, 16])
    def test_dense_training(self, d, g):
        from megatron.core.transformer.exact_attention.autograd import dense_attention

        attn, rand = self._setup(d, g)
        b, n = 2, 3 * 128 * g + 172
        q, k, v, w = rand(b, n, HQ, d), rand(b, n, HKV, d), rand(b, n, HKV, d), rand(b, n, HQ, d)
        runs = []
        for _ in range(3):
            inputs = [t.clone().requires_grad_() for t in (q, k, v)]
            out = dense_attention(*inputs, attn)
            (out.float() * w.float()).sum().backward()
            runs.append([out.detach()] + [t.grad for t in inputs])
        assert torch.equal(_bits(runs[0][0]), _bits(attn.prefill(q, k, v)))
        for later in runs[1:]:
            for a, c in zip(runs[0], later):
                assert torch.equal(_bits(a), _bits(c))
        for i in range(b):
            ref = _ref_grads(q[i], k[i], v[i], w[i], attn.softmax_scale)
            assert _rel(runs[0][0][i], ref[0]) < 4e-3
            for got, want in zip(runs[0][1:], ref[1:]):
                assert _rel(got[i], want) < 4e-3

    @pytest.mark.parametrize("d", [128, 256])
    def test_varlen_and_cp(self, d):
        from megatron.core.transformer.exact_attention.autograd import (
            cp_backward,
            cp_forward,
            varlen_attention,
        )

        attn, rand = self._setup(d, 8)
        lens = [700, 2052, 1500]
        cu = torch.tensor([0, 700, 2752, 4252], dtype=torch.int32, device="cuda")
        t = int(cu[-1])
        q, k, v, w = rand(t, HQ, d), rand(t, HKV, d), rand(t, HKV, d), rand(t, HQ, d)
        inputs = [x.clone().requires_grad_() for x in (q, k, v)]
        out = varlen_attention(*inputs, attn, cu)
        (out.float() * w.float()).sum().backward()
        grads = [x.grad for x in inputs]
        start = 0
        for n in lens:
            sl = slice(start, start + n)
            dense = attn.prefill(
                q[sl][None].contiguous(), k[sl][None].contiguous(), v[sl][None].contiguous()
            )
            assert torch.equal(_bits(out[sl]), _bits(dense[0]))
            ref = _ref_grads(q[sl], k[sl], v[sl], w[sl], attn.softmax_scale)
            for got, want in zip(grads, ref[1:]):
                assert _rel(got[sl], want) < 4e-3
            start += n

        cp = 2
        dq = torch.zeros(t, HQ, d, device="cuda")
        dk = torch.zeros(t, HKV, d, device="cuda")
        dv = torch.zeros_like(dk)
        for r in range(cp):
            plan = build_cp_plan(cu, cp, r, HKV, q.device)
            lp = plan.local_positions
            q_l = q.index_select(0, lp).contiguous()
            o_l, lse_l = cp_forward(q_l, k, v, attn, plan)
            assert torch.equal(_bits(o_l), _bits(out.detach().index_select(0, lp)))
            gq, gk, gv = cp_backward(q_l, k, v, o_l, w.index_select(0, lp), lse_l, attn, plan)
            dq.index_copy_(0, lp, gq.float())
            dk += gk
            dv += gv
        assert torch.equal(_bits(dq.bfloat16()), _bits(grads[0]))
        assert _rel(dk.bfloat16(), grads[1]) < 6e-3 and _rel(dv.bfloat16(), grads[2]) < 6e-3

    def test_heads_are_independent(self):
        """The a2a CP path runs a subset of heads; its bytes must equal the full-head run."""
        from megatron.core.transformer.exact_attention.api import ExactAttention
        from megatron.core.transformer.exact_attention.autograd import varlen_attention

        attn, rand = self._setup(256, 8)
        half = ExactAttention("cuda", head_dim=256, num_q_heads=HQ // 2, num_kv_heads=HKV // 2)
        cu = torch.tensor([0, 900, 2000], dtype=torch.int32, device="cuda")
        q, k, v = rand(2000, HQ, 256), rand(2000, HKV, 256), rand(2000, HKV, 256)
        full = varlen_attention(q, k, v, attn, cu)
        for h in range(2):
            qs = q[:, h * HQ // 2 : (h + 1) * HQ // 2].contiguous()
            ks = k[:, h * HKV // 2 : (h + 1) * HKV // 2].contiguous()
            vs = v[:, h * HKV // 2 : (h + 1) * HKV // 2].contiguous()
            part = varlen_attention(qs, ks, vs, half, cu)
            assert torch.equal(_bits(part), _bits(full[:, h * HQ // 2 : (h + 1) * HQ // 2]))

    def test_module_sbhd_and_thd(self):
        from types import SimpleNamespace

        from megatron.core.packed_seq_params import PackedSeqParams
        from megatron.core.transformer.enums import AttnMaskType
        from megatron.core.transformer.exact_attention.dot_product_attention import (
            ExactDotProductAttention,
        )

        class _Group:
            def size(self):
                return 1

        config = _config(kv_channels=256, hidden_size=2048)
        module = ExactDotProductAttention(
            config,
            1,
            AttnMaskType.causal,
            "self",
            pg_collection=SimpleNamespace(tp=_Group(), cp=None),
        )
        gen = torch.Generator(device="cuda").manual_seed(7)
        rand = lambda *s: torch.randn(*s, device="cuda", generator=gen).to(torch.bfloat16)
        s, b = 640, 2
        q, k, v = rand(s, b, HQ, 256), rand(s, b, HKV, 256), rand(s, b, HKV, 256)
        out = module(q, k, v, None)
        assert out.shape == (s, b, HQ * 256)
        thd = lambda x: x.transpose(0, 1).reshape(b * s, x.shape[2], x.shape[3]).contiguous()
        cu = torch.tensor([0, s, 2 * s], dtype=torch.int32, device="cuda")
        packed = PackedSeqParams(qkv_format='thd', cu_seqlens_q=cu, cu_seqlens_kv=cu)
        out_thd = module(thd(q), thd(k), thd(v), None, packed_seq_params=packed)
        assert torch.equal(_bits(out_thd.view(b, s, -1).transpose(0, 1)), _bits(out))
