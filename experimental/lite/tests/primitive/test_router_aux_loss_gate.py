# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Aux-loss gating: a zero coefficient must not attach aux-loss gradients.

Configurations with ``moe_router_load_balancing_type='none'`` require no
load-balancing gradient. MLite routers previously attached the switch loss through
``MoEAuxLossAutoScaler`` whenever ``compute_aux_loss=True`` and training,
even when the coefficient resolved to zero, injecting a coherent all-token
gradient into every parameter below the router.
"""

from types import SimpleNamespace

import pytest
import torch

import megatron.lite.primitive.modules.router as router_module
from megatron.lite.primitive.modules.router import SigmoidTopKRouter, TopKRouter

pytestmark = [
    pytest.mark.gpus(1),
    pytest.mark.env(CUDA_DEVICE_MAX_CONNECTIONS="1"),
]


def _ps():
    return SimpleNamespace(tp_size=1)


def _config(coef: float) -> SimpleNamespace:
    return SimpleNamespace(
        num_experts_per_tok=2,
        num_experts=4,
        n_routed_experts=4,
        router_aux_loss_coef=coef,
        aux_loss_alpha=coef,
        hidden_size=8,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        scoring_func="sigmoid",
        n_group=None,
        topk_group=None,
    )


class _CountingScaler(torch.autograd.Function):
    calls = 0

    @staticmethod
    def forward(ctx, output, aux_loss):
        _CountingScaler.calls += 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


@pytest.fixture
def counting_scaler(monkeypatch):
    _CountingScaler.calls = 0
    monkeypatch.setattr(router_module, "MoEAuxLossAutoScaler", _CountingScaler)
    return _CountingScaler


@pytest.mark.parametrize("coef,expected_calls", [(0.0, 0), (0.001, 1)])
def test_topk_router_skips_aux_loss_when_coef_zero(counting_scaler, coef, expected_calls):
    if not torch.cuda.is_available():
        pytest.skip("TopKRouter gating GEMM requires CUDA")
    router = TopKRouter(_config(coef), _ps()).cuda()
    router.train()
    x = torch.randn(6, 8, device="cuda", requires_grad=True)
    scores, indices = router(x)
    assert scores.shape == (6, 2)
    assert indices.shape == (6, 2)
    assert counting_scaler.calls == expected_calls


@pytest.mark.parametrize("coef,expected_calls", [(0.0, 0), (0.001, 1)])
def test_sigmoid_router_skips_aux_loss_when_coef_zero(counting_scaler, coef, expected_calls):
    router = SigmoidTopKRouter(_config(coef), _ps())
    router.train()
    x = torch.randn(6, 8, requires_grad=True)
    scores, indices = router(x)
    assert scores.shape == (6, 2)
    assert indices.shape == (6, 2)
    assert counting_scaler.calls == expected_calls
