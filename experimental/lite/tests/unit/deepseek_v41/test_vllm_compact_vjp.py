# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Packed-context byte/VJP controls; GPU gates exercise native forwards."""
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm import attention, experts, linear
from megatron.lite.primitive.quantization.block_fp8 import dequantize_block_fp8, quantize_block_fp8


def raw_equal(a, b):
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('rows', [0, 3, 41])
@pytest.mark.parametrize('limit', [None, 1.0])
def test_swiglu_recompute_preserves_output_and_vjp(dtype, rows, limit):
    torch.manual_seed(42)
    first, score = torch.randn(rows, 128, dtype=dtype), torch.randn(rows)
    if rows:
        first[0, :6] = torch.tensor([0.0, -0.0, 1.0, -1.0, 2.0, -2.0], dtype=dtype)
    operands = [(first.clone().requires_grad_(), score.clone().requires_grad_()) for _ in range(2)]
    reference = experts._activation_surrogate(*operands[0], limit)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        actual = experts._Activation.apply(*operands[1], limit)
    assert saved == list(operands[1])
    raw_equal(reference, actual)
    gradient = torch.randn_like(reference)
    expected = torch.autograd.grad(reference, operands[0], gradient)
    observed = torch.autograd.grad(actual, operands[1], gradient)
    for a, b in zip(expected, observed):
        raw_equal(a, b)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_packed_linear_and_shared_context_reconstruct_exact_operands(dtype):
    torch.manual_seed(43)
    codes = torch.randn(7, 128).to(torch.float8_e4m3fn)
    raw = torch.randint(122, 130, (7, 4), dtype=torch.uint8)
    packed, sf = quantize_block_fp8(torch.randn(64, 128), (32, 32), scale_format='e8m0')
    x = (
        (codes.float().reshape(7, 4, 32) * torch.exp2(raw.float() - 127).unsqueeze(-1))
        .reshape(7, 128)
        .bfloat16()
        .to(dtype)
    )
    weight = dequantize_block_fp8(packed, sf, (32, 32)).to(dtype)
    g = torch.randn(7, 64).bfloat16()
    ctx = SimpleNamespace(input_shape=x.shape, weight_dtype=torch.float32)
    ctx.saved_tensors = (x, weight)
    reference = linear._MXFP8Linear.backward(ctx, g)
    ctx.compact, ctx.input_dtype, ctx.saved_tensors = True, dtype, (codes, raw, packed, sf)
    for a, b in zip(reference[:2], linear._MXFP8Linear.backward(ctx, g)[:2]):
        raw_equal(a, b)
    scale = torch.exp2(raw.float() - 127)
    ctx.saved_tensors = (codes, scale, packed, sf)
    for a, b in zip(reference[:2], experts._Shared.backward(ctx, g)[:2]):
        raw_equal(a, b)


def test_wo_a_packed_context_preserves_pre_rotation_vjp(monkeypatch):
    # Only the unchanged rotation is bypassed; compare old/new decoded operands.
    monkeypatch.setattr(attention, '_rotate', lambda x, *_: x)
    torch.manual_seed(44)
    codes = torch.randn(3, 2, 128).to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.randint(-3, 3, (3, 2, 4)).float())
    packed_scale = torch.stack([experts.packed_scales(scale[:, g]) for g in range(2)], 1)
    packed, sf = quantize_block_fp8(torch.randn(64, 128), (32, 32), scale_format='e8m0')
    x = torch.stack(
        [experts.dequantize_activation(codes[:, g], scale[:, g]) for g in range(2)], 1
    ).bfloat16()
    weight = dequantize_block_fp8(packed, sf, (32, 32)).reshape(2, 32, 128).bfloat16()
    ctx = SimpleNamespace(shape=(3, 4, 64), weight_shape=(64, 128), weight_dtype=torch.float32)
    position, cache = torch.arange(3), torch.empty(0)
    ctx.saved_tensors = (x, weight, position, cache)
    grad = torch.randn(3, 2, 32).bfloat16()
    expected = attention._FP8OutputProjection.backward(ctx, grad)
    ctx.compact, ctx.input_dtype = True, torch.bfloat16
    ctx.saved_tensors = (codes, packed_scale, packed, sf, position, cache)
    for a, b in zip(expected[:2], attention._FP8OutputProjection.backward(ctx, grad)[:2]):
        raw_equal(a, b)


@pytest.mark.parametrize('kind', ['bf16', 'mxfp8', 'wo_a'])
@pytest.mark.parametrize('gradient_dtype', [torch.bfloat16, torch.float32])
def test_tensor_core_vjp_matches_fp32_control(kind, gradient_dtype, monkeypatch):
    torch.manual_seed(45)

    def check(a, b):
        a, b = a.double().flatten(), b.double().flatten()
        cosine = torch.nn.functional.cosine_similarity(a, b, dim=0)
        assert cosine >= 0.9999
        assert (a - b).norm() / b.norm() <= 1e-2
        assert -cosine < 0.9999

    if kind == 'wo_a':
        monkeypatch.setattr(attention, '_rotate', lambda x, *_: x)
        x, w = torch.randn(7, 2, 128).bfloat16(), torch.randn(2, 32, 128).bfloat16()
        g = torch.randn(7, 2, 32).to(gradient_dtype)
        ctx = SimpleNamespace(
            saved_tensors=(x, w, torch.arange(7), torch.empty(0)),
            shape=(7, 4, 64),
            weight_shape=(64, 128),
            weight_dtype=torch.float32,
        )
        dx, dw, *_ = attention._FP8OutputProjection.backward(ctx, g)
        check(dx, torch.einsum('tgd,gdk->tgk', g.float(), w.float()).bfloat16().reshape_as(dx))
        check(dw, torch.einsum('tgd,tgk->gdk', g.float(), x.float()).reshape_as(dw))
    else:
        x, w = torch.randn(7, 128).bfloat16(), torch.randn(64, 128).bfloat16()
        g = torch.randn(7, 64).to(gradient_dtype)
        ctx = SimpleNamespace(saved_tensors=(x, w), input_shape=x.shape, weight_dtype=torch.float32)
        dx, dw, *_ = (linear._BF16Linear if kind == 'bf16' else linear._MXFP8Linear).backward(
            ctx, g
        )
        check(dx, (g.float() @ w.float()).bfloat16())
        check(dw, g.float().T @ x.float())


def test_thd_projections_are_batched_and_metadata_tracks_boundary_changes(monkeypatch):
    state = SimpleNamespace(
        pg_collection=SimpleNamespace(cp=SimpleNamespace(size=lambda: 1)),
        _deployment_rope=torch.empty(6, 64),
        _dsv4_compress_ratio=0,
    )
    params = SimpleNamespace(qkv_format='thd', cu_seqlens_q_padded=torch.tensor([0, 3, 3, 9]))
    calls, layouts = [], []

    def projections(module, x):
        calls.append(x.shape[0])
        return x

    def batched(module, hidden, positions, cache, *, projected, layout, source, **kwargs):
        # One call over every document; positions restart per document.
        assert source is None
        raw_equal(hidden.flatten(1), projected)
        raw_equal(positions, torch.cat([torch.arange(n) for n in layout.lengths]))
        raw_equal(layout.cu, torch.tensor([lo for lo, _ in layout.documents] + [9]).int())
        raw_equal(layout.seq, torch.tensor(layout.lengths).int())
        layouts.append([tuple(pair) for pair in layout.documents.tolist()])
        return hidden * 2

    monkeypatch.setattr(attention, '_projections', projections)
    monkeypatch.setattr(attention, 'swa_forward', batched)
    x = torch.randn(9, 1, 32, requires_grad=True)
    for grad_enabled in (False, True):
        with torch.set_grad_enabled(grad_enabled):
            output, _ = attention.SWAAttention.forward(state, x, packed_seq_params=params)
        raw_equal(output, x * 2)
    assert calls == [9, 9] and layouts == [[(0, 3), (3, 3), (3, 9)]] * 2
    cached = params._deployment_documents
    assert params._deployment_documents is cached
    params.cu_seqlens_q_padded[1] = 2
    state._deployment_rope = torch.empty(5, 64)
    with pytest.raises(ValueError, match="Document length 6 exceeds RoPE cache capacity 5"):
        attention.SWAAttention.forward(state, x, packed_seq_params=params)
    assert calls == [9, 9]  # Reject before projections or native kernels.
    state._deployment_rope = torch.empty(6, 64)
    attention.SWAAttention.forward(state, x, packed_seq_params=params)
    assert layouts[-1] == [(0, 2), (2, 3), (3, 9)] and params._deployment_documents is not cached
    output.sum().backward()
    raw_equal(x.grad, torch.full_like(x, 2))
    params.cu_seqlens_q_padded = torch.tensor([0, 4])
    with pytest.raises(ValueError, match='tiling'):
        attention.SWAAttention.forward(state, x, packed_seq_params=params)
    params.cu_seqlens_q_padded = torch.tensor([0, 0])
    output, _ = attention.SWAAttention.forward(state, x[:0], packed_seq_params=params)
    assert output.shape == x[:0].shape and calls == [9, 9, 9]


def _combine_reference(topk, cu, lengths, window, ratio, top_k, stride, width):
    """vLLM ``CombineTopkSwaIndicesKernel`` (op-build 8a7fa58d76) for full prefills."""
    rows = []
    for b, (lo, hi) in enumerate(zip(cu, cu[1:])):
        for pos in range(hi - lo):
            count = min((pos + 1) // ratio, top_k) if ratio else 0
            picked = [
                stride * b + t if 0 <= t < width else -1 for t in topk[lo + pos][:count].tolist()
            ]
            swa = [stride * b + width + j for j in range(max(pos - window + 1, 0), pos + 1)]
            rows.append(picked + swa)
    return rows


@pytest.mark.parametrize('ratio', [0, 1, 2])
def test_batched_index_remap_reads_each_documents_own_rows(ratio):
    torch.manual_seed(47)
    lengths, window, top_k = [31, 0, 17, 40], 8, 6
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    documents = list(zip(cu, cu[1:]))
    counts = [n // ratio if ratio else 0 for n in lengths]
    topk = torch.full((cu[-1], top_k), -1, dtype=torch.int32)
    for (lo, hi), count in zip(documents, counts):
        for pos in range(hi - lo):
            valid = min((pos + 1) // ratio, top_k) if ratio else 0
            topk[lo + pos, :valid] = torch.randperm(max((pos + 1) // max(ratio, 1), 1))[:valid]
    layout = attention._documents(documents, 'cpu')
    total, width = sum(counts), max(counts)
    stride, table = attention._remap(layout, ratio, width, total, 'cpu')
    batched = _combine_reference(topk, cu, lengths, window, ratio, top_k, stride, width)
    for (lo, hi), count, first in zip(documents, counts, [sum(counts[:d]) for d in range(4)]):
        alone = _combine_reference(
            topk[lo:hi], [0, hi - lo], [hi - lo], window, ratio, top_k, count + hi - lo, count
        )
        for pos in range(hi - lo):
            got = [int(table[v]) if v >= 0 else -1 for v in batched[lo + pos]]
            # Single-document KV is [own compressed | own rows]; batched KV is
            # [all compressed | all rows]: translate the single-document address.
            want = [
                -1 if v < 0 else first + v if v < count else total + lo + v - count
                for v in alone[pos]
            ]
            assert got == want


def test_projection_rows_preserve_fp8_codes_and_scale_bytes(monkeypatch):
    import sys
    from dataclasses import make_dataclass

    from megatron.lite.model.deepseek_v41.vllm.linear import _row_scales

    def swizzle(sf, rows, width):
        tiles, groups = (rows + 127) // 128, (width + 127) // 128
        padded = torch.zeros(tiles * 128, groups * 4, dtype=torch.uint8)
        padded[:rows, : width // 32] = sf
        return padded.view(tiles, 4, 32, groups, 4).transpose(1, 3).contiguous().flatten()

    qa_type = make_dataclass(
        'Activation', ['data', 'scale', 'orig_dtype', 'orig_shape', 'quant_key']
    )
    qr = torch.randn(271, 160, requires_grad=True)
    codes = torch.randn_like(qr).to(torch.float8_e4m3fn)
    scales = torch.randint(100, 150, (271, 5), dtype=torch.uint8)
    qa = qa_type(codes, swizzle(scales, 271, 160), qr.dtype, qr.shape, object())
    packed = (qr, qr, qr, qr, qa, _row_scales(qa.scale, *qr.shape))
    with monkeypatch.context() as patch:
        patch.setitem(
            sys.modules,
            'vllm.model_executor.layers.quantization.utils.mxfp8_utils',
            SimpleNamespace(swizzle_mxfp8_scale=swizzle),
        )
        for lo, hi in ((0, 3), (3, 3), (3, 130), (130, 271)):
            result = attention._projection_rows(packed, lo, hi)
            for tensor in result[:4]:
                raw_equal(tensor, qr[lo:hi])
            sliced = result[-1]
            raw_equal(sliced.data, codes[lo:hi])
            raw_equal(_row_scales(sliced.scale, hi - lo, 160), scales[lo:hi])
            assert sliced.quant_key is qa.quant_key and sliced.orig_shape == qr[lo:hi].shape
        attention._projection_rows(packed, 3, 130)[0].sum().backward()
        expected = torch.zeros_like(qr)
        expected[3:130] = 1
        raw_equal(qr.grad, expected)


def test_owner_weight_gradients_accumulate_into_fp32_main_grad(monkeypatch):
    import sys

    # The placeholder provider imports Transformer Engine; only its shape is observed.
    monkeypatch.setitem(
        sys.modules,
        'megatron.lite.primitive.parallel.linear',
        SimpleNamespace(_dummy_wgrad=lambda w: torch.empty_like(w)),
    )
    torch.manual_seed(46)
    a, b = torch.randn(4, 8), torch.randn(2, 8)
    dw = torch.randn(6, 8)
    for owner in (a, b):
        owner.main_grad = torch.ones(owner.shape)
    like = torch.cat((a, b))
    placeholder = linear.accumulate_wgrad((a, b), dw, like)
    assert placeholder.shape == like.shape
    raw_equal(a.main_grad, 1 + dw[:4])
    raw_equal(b.main_grad, 1 + dw[4:])
    assert a.grad_added_to_main_grad and b.grad_added_to_main_grad
    # Without an FP32 main_grad the gradient goes back through autograd.
    c = torch.randn(6, 8, dtype=torch.bfloat16)
    raw_equal(linear.accumulate_wgrad((c,), dw, c), dw.bfloat16())
    ctx = SimpleNamespace(
        saved_tensors=(torch.randn(3, 8).bfloat16(), b.bfloat16()),
        input_shape=(3, 8),
        weight_dtype=torch.bfloat16,
        owners=(b,),
        like=b,
    )
    before = b.main_grad.clone()
    g = torch.randn(3, 2).bfloat16()
    linear._BF16Linear.backward(ctx, g)
    raw_equal(b.main_grad, before + linear.tc_mm(g.T, ctx.saved_tensors[0], torch.float32))


@pytest.mark.parametrize('world', [2, 4])
def test_contiguous_cp_metadata_keeps_empty_documents_and_query_offsets(world):
    from megatron.lite.model.deepseek_v41.vllm.compressed import metadata

    cu = torch.tensor([0, 3, 3, 13, 32])
    full = attention._documents(cu, 'cpu')
    seen = []
    for rank in range(world):
        layout = attention._documents(cu, 'cpu', local_rows=32 // world, rank=rank, world=world)
        seen.append(layout.positions)
        assert int(layout.query_cu[-1]) == 32 // world
        for doc in range(4):
            begin = max(int(cu[doc]), layout.offset)
            end = min(int(cu[doc + 1]), layout.offset + layout.local_rows)
            assert int(layout.query_cu[doc + 1] - layout.query_cu[doc]) == max(end - begin, 0)
        for ratio in (1, 2):
            m = metadata(layout.lengths, ratio, 'cpu')
            assert m.counts == [n // ratio for n in layout.lengths]
            assert torch.equal(m.positions, full.positions)
    raw_equal(torch.cat(seen), full.positions)


@pytest.mark.parametrize('ratio', [1, 2])
def test_indexer_submits_all_documents_once_with_global_key_ranges(monkeypatch, ratio):
    import sys

    from megatron.lite.model.deepseek_v41.vllm import compressed

    m = compressed.metadata([3, 0, 10, 19], ratio, 'cpu')
    calls = []

    def quant(positions, q, rope, weights, *args, **kwargs):
        assert torch.equal(positions, m.query_positions)
        return (q.to(torch.int8), torch.ones(q.shape[0], 1)), weights

    def logits(query, keys, weights, starts, ends, width):
        calls.append((starts.clone(), ends.clone(), width))
        assert torch.equal(starts, m.key_starts[m.query_requests])
        assert torch.equal(ends - starts, (m.query_positions + 1) // ratio)
        return torch.arange(width).float().expand(32, -1).clone()

    def topk(logits, starts, ends, output, *args):
        assert torch.equal(starts, torch.zeros_like(starts))
        assert torch.equal(ends, (m.query_positions + 1) // ratio)
        output[:, : logits.shape[1]] = torch.arange(logits.shape[1])

    modules = {
        'vllm.model_executor.kernels.batch_invariant': SimpleNamespace(
            top_k_per_row_prefill_index_order=topk
        ),
        'vllm.model_executor.layers.sparse_attn_indexer': SimpleNamespace(
            as_fp32_logits=lambda x, ends: x
        ),
        'vllm.models.deepseek_v41.common.ops': SimpleNamespace(fused_indexer_q_rope_quant=quant),
        'vllm.utils.deep_gemm': SimpleNamespace(
            fp8_fp4_mqa_logits=logits, mqa_logits_mx_inputs=lambda q, sf, w: (sf, w)
        ),
    }
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    total = sum(m.counts)
    result = compressed.index_topk(
        torch.zeros(32, 32, 64),
        torch.ones(32, 32),
        (torch.zeros(total, 64, dtype=torch.int8), torch.zeros(total, 4, dtype=torch.uint8)),
        torch.empty(1),
        ratio,
        lambda *args: None,
        m=m,
    )
    assert len(calls) == 1 and calls[0][2] == max(m.counts)
    assert result.shape == (32, 512)
