from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.mlite


def _split_grouped_qkvg():
    from megatron.lite.primitive.modules import split_grouped_qkvg

    return split_grouped_qkvg


def _moe_aux_scaler():
    from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler

    return MoEAuxLossAutoScaler


def _router_and_parallel_state(monkeypatch):
    from megatron.core.transformer.moe import moe_utils

    if not hasattr(moe_utils, "te_general_gemm"):
        monkeypatch.setattr(moe_utils, "te_general_gemm", None, raising=False)

    from megatron.lite.primitive.modules.router import TopKRouter
    from megatron.lite.primitive.parallel import ParallelState

    return TopKRouter, ParallelState


def _router_config():
    return SimpleNamespace(
        hidden_size=4, num_experts=4, num_experts_per_tok=2, router_aux_loss_coef=0.1
    )


def _walk_grad_fn_names(tensor: torch.Tensor) -> set[str]:
    names: set[str] = set()
    stack = [tensor.grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None:
            continue
        names.add(type(fn).__name__)
        stack.extend(parent for parent, _idx in fn.next_functions)
    return names


def test_gqa_split_grouped_qkvg_preserves_q_gate_kv_order():
    split_grouped_qkvg = _split_grouped_qkvg()
    qkv = torch.arange(24).reshape(1, 24)

    query, gate, key, value = split_grouped_qkvg(qkv, num_heads=4, num_kv_heads=2, head_dim=2)

    assert query.shape == (1, 4, 2)
    assert gate.shape == (1, 4, 2)
    assert key.shape == (1, 2, 2)
    assert value.shape == (1, 2, 2)
    assert torch.equal(query, torch.tensor([[[0, 1], [2, 3], [12, 13], [14, 15]]]))
    assert torch.equal(gate, torch.tensor([[[4, 5], [6, 7], [16, 17], [18, 19]]]))
    assert torch.equal(key, torch.tensor([[[8, 9], [20, 21]]]))
    assert torch.equal(value, torch.tensor([[[10, 11], [22, 23]]]))


def test_moe_aux_loss_auto_scaler_threads_scaled_aux_gradient():
    MoEAuxLossAutoScaler = _moe_aux_scaler()
    MoEAuxLossAutoScaler.set_loss_scale(torch.tensor([0.25]))
    output = torch.randn(3, requires_grad=True)
    aux_loss = torch.tensor(2.0, requires_grad=True)

    scaled_output = MoEAuxLossAutoScaler.apply(output * 2.0, aux_loss)
    scaled_output.sum().backward()

    torch.testing.assert_close(output.grad, torch.full_like(output, 2.0))
    torch.testing.assert_close(aux_loss.grad, torch.tensor(0.25))
    MoEAuxLossAutoScaler.main_loss_backward_scale = None


def test_topk_router_returns_finite_scores_and_valid_expert_indices(monkeypatch):
    TopKRouter, ParallelState = _router_and_parallel_state(monkeypatch)
    config = _router_config()
    router = TopKRouter(config, ParallelState(), compute_aux_loss=False)
    hidden = torch.randn(5, 4)

    scores, indices = router(hidden)

    assert scores.shape == (5, 2)
    assert indices.shape == (5, 2)
    assert scores.dtype == hidden.dtype
    assert torch.isfinite(scores).all()
    assert indices.min().item() >= 0
    assert indices.max().item() < config.num_experts


def test_topk_router_scores_are_normalized_and_deterministic_in_eval(monkeypatch):
    TopKRouter, ParallelState = _router_and_parallel_state(monkeypatch)
    config = _router_config()
    router = TopKRouter(config, ParallelState(), compute_aux_loss=False)
    hidden = torch.randn(5, config.hidden_size)

    router.eval()
    scores_1, indices_1 = router(hidden)
    scores_2, indices_2 = router(hidden)

    torch.testing.assert_close(scores_1.sum(dim=-1), torch.ones(hidden.size(0)))
    torch.testing.assert_close(scores_1, scores_2, atol=0, rtol=0)
    assert torch.equal(indices_1, indices_2)


def test_topk_router_does_not_attach_aux_scaler_in_eval(monkeypatch):
    TopKRouter, ParallelState = _router_and_parallel_state(monkeypatch)
    config = _router_config()
    router = TopKRouter(config, ParallelState(), compute_aux_loss=True)
    hidden = torch.randn(5, config.hidden_size)

    router.eval()
    scores, _indices = router(hidden)

    assert not any("MoEAuxLoss" in name for name in _walk_grad_fn_names(scores))


def test_topk_router_aux_loss_contributes_gate_gradient(monkeypatch):
    TopKRouter, ParallelState = _router_and_parallel_state(monkeypatch)
    config = _router_config()
    router = TopKRouter(config, ParallelState(), compute_aux_loss=True)
    hidden = torch.randn(8, config.hidden_size)

    router.train()
    scores, _indices = router(hidden)
    scores.sum().backward()
    grad_with_aux = router.gate.weight.grad.detach().clone()

    router.zero_grad()
    saved_coeff = router.aux_loss_coeff
    router.aux_loss_coeff = 0.0
    scores_no_aux, _indices = router(hidden)
    scores_no_aux.sum().backward()
    grad_no_aux = router.gate.weight.grad.detach().clone()
    router.aux_loss_coeff = saved_coeff

    assert torch.isfinite(grad_with_aux).all()
    assert torch.isfinite(grad_no_aux).all()
    assert (grad_with_aux - grad_no_aux).abs().sum().item() > 0.0
