# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SigmoidTopKRouter gate options (CPU)."""

from types import SimpleNamespace

import torch


def _router_config():
    return SimpleNamespace(
        hidden_size=8,
        n_routed_experts=8,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        scoring_func="sigmoid",
        aux_loss_alpha=0.0,
    )


def test_router_gemm_without_te_is_a_plain_mm(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.primitive.utils.moe import router_gating_linear

    gen = torch.Generator().manual_seed(0)
    x = torch.randn(6, 8, generator=gen).to(torch.bfloat16).requires_grad_(True)
    w = torch.randn(4, 8, generator=gen).requires_grad_(True)
    x_mm = x.detach().clone().requires_grad_(True)
    w_mm = w.detach().clone().requires_grad_(True)

    out = router_gating_linear(x, w, None, torch.float32, use_te_gemm=False)
    expected = torch.mm(x_mm.float(), w_mm.t())
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, expected)
    grad = torch.randn(6, 4, generator=gen)
    out.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(x.grad, x_mm.grad)
    torch.testing.assert_close(w.grad, w_mm.grad)


def test_sigmoid_router_keeps_an_fp32_gate_weight(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.router import SigmoidTopKRouter

    ps = SimpleNamespace(tp_size=1, tp_group=None)
    router = SigmoidTopKRouter(_router_config(), ps, compute_aux_loss=False, gate_weight_fp32=True)
    router = router.to(torch.bfloat16)
    assert router.gate.weight.dtype == torch.float32
    assert router.expert_bias.dtype == torch.float32
