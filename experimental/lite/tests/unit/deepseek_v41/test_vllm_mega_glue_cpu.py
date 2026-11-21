# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""attention_core=mega glue: fused-layout VJPs, option plumbing and dispatch."""
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm import attention, linear


def test_row_permuted_linear_vjp_is_the_natural_vjp():
    torch.manual_seed(91)
    x, w = torch.randn(5, 64).bfloat16(), torch.randn(96, 64).bfloat16()
    perm = torch.randperm(96)
    grad = torch.randn(5, 96).bfloat16()  # w.r.t. the permuted output columns
    ctx = SimpleNamespace(
        saved_tensors=(x, w), input_shape=x.shape, weight_dtype=torch.float32, row_perm=perm
    )
    dx, dw, *_ = linear._MXFP8Linear.backward(ctx, grad)
    natural = torch.empty_like(grad).index_copy_(1, perm, grad)
    ctx.row_perm = None
    expected = linear._MXFP8Linear.backward(ctx, natural)
    torch.testing.assert_close(dx, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(dw, expected[1], rtol=0, atol=0)


def test_mega_woa_backward_is_the_einsum_vjp_with_unpermuted_columns(monkeypatch):
    from megatron.lite.model.deepseek_v41.vllm import mega_attention

    torch.manual_seed(92)
    tokens, groups, rank, width = 3, 2, 32, 64
    out = torch.randn(tokens, groups, width).bfloat16().float()  # dequantized fused output
    monkeypatch.setattr(mega_attention, 'dequantize_output', lambda data, scale: out)
    from megatron.lite.primitive.quantization.block_fp8 import (
        dequantize_block_fp8,
        quantize_block_fp8,
    )

    codes, blocks = quantize_block_fp8(
        torch.randn(groups * rank, width), (32, 32), scale_format='e8m0'
    )
    decoded = dequantize_block_fp8(codes, blocks, (32, 32)).bfloat16()  # fused columns
    perm = torch.randperm(width // 32).repeat_interleave(32) * 32 + torch.arange(width) % 32
    weight = torch.zeros(groups * rank, width)
    saved = (None, None, codes, blocks, perm)
    ctx = SimpleNamespace(saved_tensors=saved, weight=weight, groups=groups)
    grad = torch.randn(tokens, groups, rank).bfloat16()
    _, _, d_out, dw, *_ = attention._MegaWoA.backward(ctx, grad)
    w = decoded.float().view(groups, rank, width)
    expected_out = torch.einsum('tgr,grw->tgw', grad.float(), w)
    expected_dw = torch.einsum('tgr,tgw->grw', grad.float(), out).reshape(-1, width)
    natural = torch.empty_like(expected_dw)
    natural[:, perm] = expected_dw
    torch.testing.assert_close(d_out.float(), expected_out, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dw.float(), natural, rtol=1e-5, atol=1e-5)


def test_attention_core_option_selects_flashmla_or_mega(monkeypatch):
    from megatron.lite.model.deepseek_v41.vllm import protocol

    assert protocol.ImplConfig().attention_core == 'flashmla'
    protocol.validate_impl_cfg_keys({'attention_core': 'mega'})
    monkeypatch.setattr(protocol.native, 'build_model', lambda *a, **k: 'built')
    impl = protocol.ImplConfig(attention_core='mega', token_map=torch.zeros(1))
    assert protocol.build_model(SimpleNamespace(engram_layer_ids=()), impl_cfg=impl) == 'built'
    with pytest.raises(ValueError, match="'flashmla' or 'mega'"):
        protocol.build_model(
            SimpleNamespace(engram_layer_ids=()),
            impl_cfg=protocol.ImplConfig(attention_core='sparse', token_map=torch.zeros(1)),
        )


def test_swa_attention_dispatches_the_mega_core(monkeypatch):
    calls = []

    def mega(module, hidden, layout, cu, source):
        calls.append((hidden.shape[0], cu.tolist(), layout.positions.tolist()))
        assert cu is layout.host_cu  # host offsets: no device sync for the plan
        return hidden * 3

    monkeypatch.setattr(attention, 'mega_forward', mega)
    module = SimpleNamespace(
        pg_collection=SimpleNamespace(cp=SimpleNamespace(size=lambda: 1, rank=lambda: 0)),
        _deployment_rope=torch.empty(64, 8),
        _dsv4_compress_ratio=2,
        attention_core='mega',
    )
    params = SimpleNamespace(
        qkv_format='thd', cu_seqlens_q_padded=torch.tensor([0, 3, 7]), cu_seqlens_q=None
    )
    x = torch.randn(7, 1, 16)
    out, bias = attention.SWAAttention.forward(module, x, packed_seq_params=params)
    torch.testing.assert_close(out, x * 3, rtol=0, atol=0)
    assert bias is None and calls == [(7, [0, 3, 7], [0, 1, 2, 0, 1, 2, 3])]


@pytest.mark.parametrize('ratio', [1, 2])
def test_context_parallel_ranks_take_their_own_boundary_rows(ratio):
    documents = torch.tensor([0, 5, 6, 16])  # odd lengths straddle the rank split
    rows, world = 16, 2
    every = attention._documents(documents, 'cpu')
    flag = ((every.global_positions + 1) % ratio == 0).nonzero().flatten()
    spans = []
    for rank in range(world):
        layout = attention._documents(
            documents, 'cpu', local_rows=rows // world, rank=rank, world=world
        )
        lo, hi = attention._local_boundaries(layout, ratio)
        local = (flag >= layout.offset) & (flag < layout.offset + layout.local_rows)
        assert local.nonzero().flatten().tolist() == list(range(lo, hi))
        spans.append((lo, hi))
    assert spans[0][0] == 0 and spans[0][1] == spans[1][0] and spans[1][1] == flag.numel()


@pytest.mark.parametrize('deterministic', [False, True])
def test_deterministic_mode_selects_the_fixed_order_dkv(monkeypatch, deterministic):
    from megatron.lite.model.deepseek_v41.vllm import mega_attention

    seen = {}

    def core(*args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(data=None, scale=None, grad=None)

    monkeypatch.setattr(mega_attention, 'mega_attention', core)
    monkeypatch.setattr(attention, 'fused_permutations', lambda *a: (None, None))
    monkeypatch.setattr(attention, '_projections', lambda module, x, row_perm=None: (x, x, x, x, x))
    monkeypatch.setattr(attention._MegaWoA, 'apply', lambda *a: torch.zeros(2, 1, 4))
    monkeypatch.setattr(attention, 'linear', lambda z, w, **k: z)
    module = SimpleNamespace(
        _dsv4_compress_ratio=0,
        _deployment_rope=None,
        config=SimpleNamespace(csa_window_size=8, o_groups=1, deterministic_mode=deterministic),
        release=SimpleNamespace(index_topk=4),
        core_attention=SimpleNamespace(native=SimpleNamespace(attn_sink=None)),
        linear_o_group_proj=None,
        linear_proj=SimpleNamespace(weight=None),
    )
    layout = SimpleNamespace(cp_group=None, positions=torch.arange(2))
    attention.mega_forward(module, torch.zeros(2, 1, 4), layout, None, None)
    assert seen['deterministic'] is deterministic and seen['cp_group'] is None


def test_boundary_rows_backward_matches_indexing_without_sort():
    from megatron.lite.model.deepseek_v41.vllm import compressed

    torch.manual_seed(93)
    m = compressed.metadata([5, 0, 6, 9], 2, 'cpu')
    latent = torch.randn(20, 8, requires_grad=True)
    rows = compressed._BoundaryRows.apply(latent, m.closers, m.closing)
    torch.testing.assert_close(rows, latent[m.closers], rtol=0, atol=0)
    grad = torch.randn_like(rows)
    (actual,) = torch.autograd.grad(rows, latent, grad)
    (expected,) = torch.autograd.grad(latent[m.closers], latent, grad)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
