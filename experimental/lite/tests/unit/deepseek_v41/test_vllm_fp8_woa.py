# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Decoded-operand derivative and publication precision contracts; no GPU claim."""
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm import protocol
from megatron.lite.model.deepseek_v41.vllm.attention import _FP8OutputProjection, _rotate


@pytest.mark.parametrize('rows', [0, 3])
def test_woa_ste_uses_decoded_operands(rows):
    torch.manual_seed(17)
    activation = torch.randn(rows, 2, 64).bfloat16()
    weight = torch.randn(2, 32, 64).bfloat16()
    grad = torch.randn(rows, 2, 32).bfloat16()
    pos = torch.arange(rows)
    cache = torch.zeros(max(rows, 1), 8)
    cache[:, :4] = 1
    ctx = SimpleNamespace(
        saved_tensors=(activation, weight, pos, cache),
        shape=(rows, 4, 32),
        weight_shape=(64, 64),
        weight_dtype=torch.float32,
    )
    dx, dw, *metadata = _FP8OutputProjection.backward(ctx, grad)
    x = activation.float().requires_grad_()
    w = weight.float().requires_grad_()
    y = torch.einsum('tgk,gdk->tgd', x, w)
    expected_x, expected_w = torch.autograd.grad(y, (x, w), grad.float())
    assert torch.equal(dx, _rotate(expected_x.bfloat16().reshape(ctx.shape), pos, cache))
    torch.testing.assert_close(dw, expected_w.reshape(ctx.weight_shape), rtol=1e-5, atol=1e-5)
    assert metadata == [None, None, None]


def test_woa_forward_has_no_cpu_numerical_fallback():
    with pytest.raises(RuntimeError, match='requires CUDA'):
        _FP8OutputProjection.apply(
            torch.zeros(3, 4, 32), torch.zeros(64, 64), torch.arange(3), torch.zeros(3, 8), 2
        )


def test_online_publication_uses_same_release_encoding(monkeypatch):
    calls = []

    def export(*args, **kwargs):
        calls.append(kwargs['target'])
        yield ('native-code', torch.zeros(1, dtype=torch.uint8))

    monkeypatch.setattr(protocol.native, 'export_hf_weights', export)
    assert len(list(protocol.export_hf_weights(None, None, None, target='vllm'))) == 1
    assert calls == ['release']
    list(protocol.export_hf_weights(None, None, None, target='hf'))
    assert calls[-1] == 'hf'


def test_lora_prequant_interface_has_no_cpu_fallback():
    from megatron.lite.model.deepseek_v41.vllm.attention import inverse_rope_bf16

    with pytest.raises(RuntimeError, match='requires CUDA'):
        inverse_rope_bf16(torch.zeros(3, 4, 32), torch.arange(3), torch.zeros(3, 8), 2)
