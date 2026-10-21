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
