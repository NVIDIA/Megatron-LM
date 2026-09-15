# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU tests of the production router backward, not GPU parity evidence."""

from types import SimpleNamespace

import pytest
import torch

import megatron.core  # noqa: F401


def test_production_router_wgrad_does_not_round_each_chunk(
    transformer_engine_import_stub, monkeypatch
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import moe_ep_chunk_overlap as ep
    from megatron.lite.primitive.utils import moe

    # Real backward with a CPU GEMM reference, including its output dtype.
    calls = []

    def gemm(a, b, out_dtype, *, layout, out=None, accumulate=False, **kwargs):
        value = {
            "TN": lambda: b.float() @ a.float().T,
            "NN": lambda: b.float() @ a.float(),
            "NT": lambda: b.float().T @ a.float(),
        }[layout]()
        value = value.to(out_dtype)
        if out is not None:
            calls.append((out_dtype, accumulate, out.data_ptr()))
            out.add_(value) if accumulate else out.copy_(value)
            value = out
        return (value,)

    monkeypatch.setattr(moe, "_te_general_gemm", gemm)
    torch.manual_seed(31)
    weight = torch.randn(3, 8, dtype=torch.bfloat16, requires_grad=True)
    x = torch.randn(12, 8, dtype=torch.bfloat16)
    grad = torch.randn(12, 3, dtype=torch.bfloat16)
    accum = [None]
    for xx, gg in zip(x.chunk(2), grad.chunk(2)):
        xx = xx.detach().requires_grad_(True)
        scores = moe.router_gating_linear(xx, weight, None, torch.bfloat16)
        chunk = SimpleNamespace(x=xx, scores=scores, scores_edge=None, scores_dtype=scores.dtype)
        dx = ep._backward_router(chunk, torch.zeros_like(xx), gg, (weight,), accum)
        assert torch.equal(dx, (gg.float() @ weight.detach().float()).bfloat16())
        assert not hasattr(weight, "_wgrad_accumulator")
    expected = (grad.float().T @ x.float()).bfloat16()
    assert torch.equal(ep._materialize((weight,), accum)[0], expected)
    assert len(calls) == 2 and all(c[0] == torch.float32 and c[1] for c in calls)
    assert calls[0][2] == calls[1][2]
    assert weight.grad is None
    # Without a bound workspace the ordinary router still returns its gradient.
    native = moe.router_gating_linear(x, weight, None, torch.bfloat16)
    assert torch.equal(torch.autograd.grad(native, weight, grad)[0], expected)
    assert len(calls) == 2


def test_double_router_is_not_downcast(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import moe_ep_chunk_overlap as ep
    from megatron.lite.primitive.utils.moe import router_gating_linear

    weight = torch.ones(1, 1, dtype=torch.float64, requires_grad=True)
    x = torch.tensor([[1.00000000001]], dtype=torch.float64, requires_grad=True)
    scores = router_gating_linear(x, weight, None, torch.float64)
    chunk = SimpleNamespace(x=x, scores=scores, scores_edge=None, scores_dtype=scores.dtype)
    accum = [None]
    ep._backward_router(chunk, torch.zeros_like(x), torch.ones_like(scores), (weight,), accum)
    assert torch.equal(ep._materialize((weight,), accum)[0], x.detach())


def test_router_accumulator_binding_cleans_up_on_error(transformer_engine_import_stub, monkeypatch):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import moe_ep_chunk_overlap as ep

    weight = torch.ones(2, 2, requires_grad=True)
    x = torch.ones(2, 2, requires_grad=True)
    chunk = SimpleNamespace(x=x, scores=x, scores_edge=None, scores_dtype=x.dtype)

    def fail(*args, **kwargs):
        assert weight._wgrad_accumulator.dtype == torch.float32
        raise RuntimeError("injected backward failure")

    monkeypatch.setattr(torch.autograd, "grad", fail)
    with pytest.raises(RuntimeError, match="injected"):
        ep._backward_router(chunk, x, x, (weight,), [None])
    assert not hasattr(weight, "_wgrad_accumulator")


def test_router_accumulator_rejects_overlapping_owner(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import moe_ep_chunk_overlap as ep

    weight = torch.ones(2, 2, requires_grad=True)
    owner = torch.zeros_like(weight)
    weight._wgrad_accumulator = owner
    x = torch.ones(2, 2, requires_grad=True)
    chunk = SimpleNamespace(x=x, scores=x, scores_edge=None, scores_dtype=x.dtype)
    with pytest.raises(RuntimeError, match="already leased"):
        ep._backward_router(chunk, x, x, (weight,), [None])
    assert weight._wgrad_accumulator is owner
    del weight._wgrad_accumulator
