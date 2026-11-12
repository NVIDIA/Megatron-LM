# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU transport/math controls for the routed backward; GPU parity is a separate gate."""
import sys
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm import experts, moe, weights


def test_encoding_cache_lifetime_includes_data_updates():
    weights.clear()
    owner = torch.nn.Parameter(torch.ones(32, 32))
    calls = []

    def encode():
        calls.append(1)
        return owner.detach().clone()

    try:
        first = weights.encoded((owner,), 'test', encode)
        assert weights.encoded((owner,), 'test', encode) is first
        version = owner._version
        owner.data.add_(1)
        assert owner._version == version  # optimizer writes need a forward-entry reset
        weights.clear()
        second = weights.encoded((owner,), 'test', encode)
        assert torch.equal(second, first + 1) and len(calls) == 2
        weights.encoded((owner,), 'other', encode)
        assert len(calls) == 3
        plain = owner.detach()
        weights.encoded((plain,), 'test', encode)
        weights.encoded((plain,), 'test', encode)
        assert len(calls) == 5  # transient stacked tensors must never hit by identity
    finally:
        weights.clear()


def _te_cpu(a, b, outputs, quantizers, dtype, **kw):
    # Oracle only: production dispatches the entire list through one TE API call.
    values = []
    for x, y in zip(a, b):
        if kw['layout'] == 'TN':
            z = y.float() @ x.float().T
        elif kw['layout'] == 'NN':
            z = y.float() @ x.float()
        else:
            z = y.float().T @ x.float()
        values.append(z.to(dtype))
    if kw.get('single_output', False):
        outputs[0].copy_(torch.cat(values))
    else:
        for target, value in zip(outputs, values):
            target.add_(value) if kw.get('accumulate') else target.copy_(value)


@pytest.mark.parametrize('rows,unrouted', [(5, False), (5, True), (0, True)])
def test_routed_backward_empty_experts_and_unrouted_slots(monkeypatch, rows, unrouted):
    torch.manual_seed(19)
    hidden, inter, local = 64, 32, 3
    fc1 = [torch.nn.Parameter(torch.randn(2 * inter, hidden)) for _ in range(local)]
    fc2 = [torch.nn.Parameter(torch.randn(hidden, inter)) for _ in range(local)]
    w1, w2 = moe._mx(torch.stack(fc1)), moe._mx(torch.stack(fc2))
    codes = torch.randn(rows, hidden).to(torch.float8_e4m3fn)
    scale = torch.ones(rows, hidden // 32)
    grad, scores = torch.randn(rows, hidden), torch.rand(rows, 3)
    # Expert 1 is always empty; repeated expert slots and -1 are intentional.
    ids = torch.tensor([[2, 0, 2], [0, -1, 2], [2, 2, 0], [-1, 0, 0], [0, 2, -1]])[:rows]
    if unrouted:
        ids = torch.full_like(ids, -1)
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    exchange_calls = []

    def exchange(output, input, receives, sends, **kw):
        assert receives == sends and kw['async_op']
        output.copy_(input)
        exchange_calls.append(tuple(input.shape))
        return SimpleNamespace(wait=lambda: None)

    monkeypatch.setattr(torch.distributed, 'all_to_all_single', exchange)
    monkeypatch.setattr(moe, 'all_to_all', lambda group, value: value.clone())
    monkeypatch.setattr(moe, '_decoded_experts', lambda *args: (w1, w2))
    monkeypatch.setattr(
        experts,
        '_gather_decode_bf16',
        lambda c, s, order: experts.dequantize_activation(
            c.view(torch.uint8)[order].view(torch.float8_e4m3fn), s[order]
        ).bfloat16(),
    )
    monkeypatch.setitem(
        sys.modules,
        'transformer_engine.pytorch.cpp_extensions',
        SimpleNamespace(general_grouped_gemm=_te_cpu),
    )
    marker = object()
    actual = moe._routed_grads(
        codes, scale, grad, scores, ids, fc1, fc2, 3.0, group, lambda: marker
    )
    assert actual[-1] is marker and len(exchange_calls) == 6
    dx = torch.zeros(rows, 3, hidden)
    ds = torch.zeros(rows, 3)
    d1, d2 = [], []
    decoded = experts.dequantize_activation(codes, scale).bfloat16()
    for expert in range(local):
        token, slot = (ids == expert).nonzero(as_tuple=True)
        a, g, score = decoded[token], grad.bfloat16()[token], scores[token, slot]
        h = (a.float() @ w1[expert].float().T).bfloat16().float().requires_grad_()
        s = score.clone().requires_grad_()
        gate, up = h.chunk(2, -1)
        act = torch.nn.functional.silu(gate.clamp(max=3.0)) * up.clamp(-3.0, 3.0) * s[:, None]
        dact = (g.float() @ w2[expert].float()).bfloat16().float()
        dh, dscore = torch.autograd.grad(act, (h, s), dact)
        dh = dh.bfloat16()
        dx[token, slot] = (dh.float() @ w1[expert].float()).bfloat16().float()
        ds[token, slot] = dscore
        d1.append(dh.float().T @ a.float())
        d2.append(g.float().T @ moe._requant_l2(act).bfloat16().float())
    for value, reference in zip(actual[:4], (dx.sum(1), ds, d1, d2)):
        if isinstance(value, list):
            for a, b in zip(value, reference):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        else:
            torch.testing.assert_close(value, reference, atol=0, rtol=0)
        assert all(torch.isfinite(t).all() for t in (value if isinstance(value, list) else [value]))


@pytest.mark.parametrize('splits', [(3, 0, 2), (0, 0, 0)])
def test_grouped_wgrad_accumulates_main_grad_without_overwriting(monkeypatch, splits):
    owners = [torch.nn.Parameter(torch.zeros(8, 4)) for _ in splits]
    for owner in owners:
        owner.main_grad = torch.ones_like(owner)
    monkeypatch.setitem(
        sys.modules,
        'transformer_engine.pytorch.cpp_extensions',
        SimpleNamespace(general_grouped_gemm=_te_cpu),
    )
    monkeypatch.setitem(
        sys.modules,
        'megatron.lite.primitive.parallel.linear',
        SimpleNamespace(_dummy_wgrad=lambda p: torch.empty_like(p)),
    )
    x, grad = torch.randn(sum(splits), 4).bfloat16(), torch.randn(sum(splits), 8).bfloat16()
    with torch.no_grad():
        for _ in range(2):
            placeholders = moe._grouped_wgrad(x, grad, splits, owners)
    for owner, a, g, placeholder in zip(owners, x.split(splits), grad.split(splits), placeholders):
        torch.testing.assert_close(
            owner.main_grad, torch.ones_like(owner) + 2 * (g.float().T @ a.float())
        )
        assert owner.grad_added_to_main_grad and placeholder.shape == owner.shape


def test_inference_does_not_retain_weight_encodings():
    owner = torch.nn.Parameter(torch.ones(32, 32))
    calls = []

    def produce():
        calls.append(1)
        return owner.detach().clone()

    try:
        weights.clear(enabled=False)
        first = weights.encoded((owner,), 'test', produce)
        second = weights.encoded((owner,), 'test', produce)
        assert first is not second and len(calls) == 2
        assert torch.equal(first, second) and not weights._CACHE
        weights.clear(enabled=True)
        third = weights.encoded((owner,), 'test', produce)
        assert weights.encoded((owner,), 'test', produce) is third
        assert len(calls) == 3
    finally:
        weights.clear()
