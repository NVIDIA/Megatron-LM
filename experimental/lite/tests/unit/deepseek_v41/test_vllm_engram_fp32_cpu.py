# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native modular Engram VJP; no inference kernel runs in this CPU gate."""
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm.engram import _Post, reference
from megatron.lite.primitive.ckpt.hf_weights import _FP4_E2M1_TABLE
from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4


@pytest.mark.parametrize('rows', [0, 257])
@pytest.mark.parametrize('master_dtype', [torch.bfloat16, torch.float32])
def test_post_backward_promotes_query_key_before_making_leaves(rows, master_dtype):
    torch.manual_seed(20261020)
    hidden = torch.randn(rows, 4, 32).bfloat16()
    kv = torch.randn(rows, 160).bfloat16()
    query, key = [torch.randn(4, 32).to(master_dtype) for _ in range(2)]
    gradient = torch.randn_like(hidden)
    leaves = [x.float().detach().requires_grad_() for x in (hidden, kv, query, key)]
    out = reference(*leaves, 1e-6, None)
    expected = torch.autograd.grad(out, leaves, gradient.float())
    ctx = SimpleNamespace(saved_tensors=(hidden, kv, query, key), eps=1e-6, mask=None)
    actual = _Post.backward(ctx, gradient)
    for got, want in zip(actual[:4], expected):
        assert torch.equal(got, want.to(got.dtype))
        assert torch.isfinite(got).all()
    for got in actual[2:4]:
        assert got.dtype == torch.float32
        if rows:
            assert not torch.equal(
                got, got.bfloat16().float()
            ), 'BF16-leaf negative control'


def test_fp4_negative_zero_survives_decode_and_requantization():
    values = torch.tensor(_FP4_E2M1_TABLE).repeat(2).reshape(1, 32)
    assert values[0, 8].signbit()
    codes, _ = quantize_mxfp4(values)
    unsigned = codes.to(torch.int16) & 255
    unpacked = torch.stack((unsigned & 15, unsigned >> 4), dim=-1).flatten(-2)
    assert unpacked.tolist() == [list(range(16)) * 2]


@pytest.mark.parametrize('publish', [False, True])
def test_modular_forward_consults_readiness_between_updates(monkeypatch, publish):
    from megatron.core.utils import PARAM_READY_CALLBACK_ATTR
    from megatron.lite.model.deepseek_v41.vllm import model as module

    model = module.DeepseekV41Model.__new__(module.DeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.child = torch.nn.Linear(1, 1, bias=False)
    model.child.weight.data.fill_(1)
    model.layout = SimpleNamespace(has_head=False)
    master = torch.nn.Parameter(torch.ones(1, 1))
    optimizer = torch.optim.SGD([master], lr=0.1)

    def ready():
        model.child.weight.data.copy_(master.detach())

    setattr(model.child.weight, PARAM_READY_CALLBACK_ATTR, ready)
    monkeypatch.setattr(
        module.NativeModel, 'forward', lambda self, x, **kw: x @ self.child.weight
    )
    if not publish:
        monkeypatch.setattr(module, 'ensure_params_ready', lambda params: None)
    for _ in range(2):
        optimizer.zero_grad()
        master.square().sum().backward()
        optimizer.step()
        output = model(torch.ones(1, 1))
        assert torch.equal(output, master.detach()) is publish
