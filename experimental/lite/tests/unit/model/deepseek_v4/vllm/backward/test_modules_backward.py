"""Numerical VJP gates for the production vLLM-visible bridges."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from megatron.lite.model.deepseek_v4.vllm.primitive.dense import (
    _inverse_rope,
    _pre_graph,
    block_fp8_linear,
    fixed_route_vjp,
    fused_block_fp8_linear,
    fused_qkv_rms_norm,
    gate_linear,
    mhc_head,
    mhc_post,
    mhc_pre_broadcast,
    o_projection,
    rms_norm,
)
from megatron.lite.primitive.modules.attention.hca import HyperConnection
from megatron.lite.primitive.recompute import wrap_checkpoint

pytestmark = pytest.mark.gpus(1)


def _grad_like(value: torch.Tensor) -> torch.Tensor:
    return torch.arange(1, value.numel() + 1, dtype=torch.float32).reshape_as(value).div(value.numel()).to(value)


def _post_graph(x, residual, post, comb):
    return HyperConnection.post(x, residual, post.squeeze(-1), comb)


@pytest.mark.parametrize("fused", [False, True])
def test_visible_linear_value_and_master_vjp(fused: bool) -> None:
    torch.manual_seed(1)
    x = torch.randn(3, 4, requires_grad=True)
    weights = tuple(
        torch.randn(rows, 4, requires_grad=True)
        for rows in ((2, 3) if fused else (5,))
    )

    def visible(value, *ws):
        return F.linear(value, torch.cat(ws)) + 0.25
    output = (
        fused_block_fp8_linear(visible, x, *weights)
        if fused
        else block_fp8_linear(visible, x, weights[0])
    )
    assert torch.equal(output, visible(x, *weights).detach())
    output.backward(_grad_like(output))
    refs = tuple(value.detach().requires_grad_(True) for value in (x, *weights))
    F.linear(refs[0].float(), torch.cat(refs[1:]).float()).backward(_grad_like(output).float())
    for actual, reference in zip((x, *weights), refs, strict=True):
        torch.testing.assert_close(actual.grad.float(), reference.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("fused", [False, True])
def test_visible_linear_empty_rows_do_not_call_deployment_kernel(fused: bool) -> None:
    value = torch.empty(0, 4, requires_grad=True)
    weights = tuple(
        torch.randn(rows, 4, requires_grad=True)
        for rows in ((2, 3) if fused else (5,))
    )

    def unsupported_empty_m(*_args):
        raise AssertionError("deployment kernel must not receive M=0")

    output = (
        fused_block_fp8_linear(unsupported_empty_m, value, *weights)
        if fused
        else block_fp8_linear(unsupported_empty_m, value, weights[0])
    )
    assert output.shape == (0, 5)
    output.sum().backward()
    assert value.grad is not None and value.grad.numel() == 0
    for weight in weights:
        assert weight.grad is not None
        assert torch.count_nonzero(weight.grad) == 0


def test_forward_only_uses_visible_path_without_autograd_owner() -> None:
    x = torch.randn(3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)
    def visible(value, master):
        return F.linear(value, master) + 0.25

    with torch.inference_mode():
        output = block_fp8_linear(visible, x, weight)
        projected = o_projection(
            lambda value, *_weights: value.sum(dim=1),
            torch.randn(3, 2, 4),
            torch.randn(2, 8),
            torch.randn(4, 2),
            positions=torch.arange(3),
            cos_sin_cache=torch.randn(3, 4),
            n_groups=1,
            heads_per_group=2,
            nope_dim=2,
            rope_dim=2,
            o_lora_rank=2,
        )

    assert output.is_inference()
    assert projected.is_inference()
    torch.testing.assert_close(output, visible(x, weight), rtol=0, atol=0)


def test_checkpoint_recompute_owns_no_grad_visible_output() -> None:
    class Projection(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(5, 4))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            def visible(x, weight):
                # Deployment kernels run without an autograd tape, but must
                # return ordinary tensors for the functional VJP bridge.
                with torch.no_grad():
                    return F.linear(x, weight)

            return block_fp8_linear(visible, value, self.weight)

    module = Projection()
    wrap_checkpoint(module)
    value = torch.randn(3, 4, requires_grad=True)
    module(value).sum().backward()
    assert value.grad is not None
    assert module.weight.grad is not None


def test_linear_backward_rejects_master_mutation_after_forward() -> None:
    value = torch.randn(2, 4, requires_grad=True)
    weight = torch.randn(3, 4, requires_grad=True)
    output = block_fp8_linear(F.linear, value, weight)
    with torch.no_grad():
        weight.add_(1)
    with pytest.raises(RuntimeError, match="changed between forward and backward"):
        output.sum().backward()


def test_gate_linear_uses_bound_master_vjp() -> None:
    x = torch.randn(4, 6, requires_grad=True)
    weight = torch.randn(9, 6, requires_grad=True)
    output = gate_linear(lambda value: (F.linear(value, weight), None), x, weight)
    grad = _grad_like(output)
    output.backward(grad)
    torch.testing.assert_close(x.grad, grad @ weight.detach())
    torch.testing.assert_close(weight.grad, grad.T @ x.detach())


@pytest.mark.parametrize("fused", [False, True])
def test_rms_norm_vjp_matches_pytorch(fused: bool) -> None:
    torch.manual_seed(2)
    eps = 1e-6
    x = torch.randn(3, 8, requires_grad=True)
    weight = torch.randn(8, requires_grad=True)
    def visible(value, w, epsilon):
        return F.rms_norm(value, (8,), w, epsilon)

    if fused:
        y = torch.randn(3, 6, requires_grad=True)
        wy = torch.randn(6, requires_grad=True)
        def pair(a, b, wa, wb, epsilon):
            return (
                F.rms_norm(a, (8,), wa, epsilon),
                F.rms_norm(b, (6,), wb, epsilon),
            )
        outputs = fused_qkv_rms_norm(pair, x, y, weight, wy, eps)
        grads = tuple(_grad_like(value) for value in outputs)
        torch.autograd.backward(outputs, grads)
        candidates = (x, y, weight, wy)
        refs = tuple(value.detach().requires_grad_(True) for value in candidates)
        torch.autograd.backward(pair(*refs, eps), grads)
    else:
        output = rms_norm(visible, x, weight, eps)
        grad = _grad_like(output)
        output.backward(grad)
        candidates = (x, weight)
        refs = tuple(value.detach().requires_grad_(True) for value in candidates)
        visible(*refs, eps).backward(grad)
    for actual, reference in zip(candidates, refs, strict=True):
        torch.testing.assert_close(actual.grad, reference.grad, rtol=1e-5, atol=1e-6)


def test_mhc_visible_values_use_functional_vjps() -> None:
    torch.manual_seed(3)
    mult, hidden, eps = 2, 4, 1e-6
    mix = (2 + mult) * mult
    inputs = (
        torch.randn(3, hidden, requires_grad=True),
        torch.randn(mix, mult * hidden, requires_grad=True),
        torch.randn(3, requires_grad=True),
        torch.randn(mix, requires_grad=True),
        torch.randn(hidden, requires_grad=True),
    )

    def pre(*values):
        residual, post, comb, x = _pre_graph(*values[:4], mult=mult, iters=3, eps=eps)
        return residual, post, comb, F.rms_norm(x, (hidden,), values[4], eps)

    outputs = mhc_pre_broadcast(pre, *inputs, mult=mult, iters=3, eps=eps, norm_eps=eps)
    grads = tuple(_grad_like(value) for value in outputs)
    torch.autograd.backward(outputs, grads)
    refs = tuple(value.detach().requires_grad_(True) for value in inputs)
    torch.autograd.backward(pre(*refs), grads)
    for actual, reference in zip(inputs, refs, strict=True):
        torch.testing.assert_close(actual.grad, reference.grad, rtol=1e-5, atol=1e-6)

    post_inputs = tuple(
        value.detach().requires_grad_(True)
        for value in (outputs[3], outputs[0], outputs[1], outputs[2])
    )
    post = mhc_post(lambda *values: _post_graph(*values) + 0.125, *post_inputs)
    post.backward(_grad_like(post))
    assert all(value.grad is not None for value in post_inputs)


def test_mhc_head_and_o_projection_cover_parameters() -> None:
    torch.manual_seed(4)
    x = torch.randn(3, 2, 4, requires_grad=True)
    fn = torch.randn(2, 8, requires_grad=True)
    scale = torch.randn(2, requires_grad=True)
    base = torch.randn(2, requires_grad=True)

    def head(value, fn_, scale_, base_):
        flat = value.flatten(-2).float()
        mixes = F.linear(flat, fn_.float()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
        return ((torch.sigmoid(mixes * scale_ + base_) + 1e-6).unsqueeze(-1) * value.float()).sum(-2).to(value)

    output = mhc_head(head, x, fn, scale, base, eps=1e-6)
    output.backward(_grad_like(output))
    assert all(value.grad is not None for value in (x, fn, scale, base))

    o = torch.randn(3, 4, 8, requires_grad=True)
    wa = torch.randn(6, 16, requires_grad=True)
    wb = torch.randn(5, 6, requires_grad=True)
    positions = torch.tensor([2, 0, 3])
    cache = torch.randn(5, 4)

    def projection(o_, wa_, wb_):
        inverse = _inverse_rope(o_, positions, cache, 4, 4).reshape(3, 2, -1)
        z = torch.einsum("tgd,grd->tgr", inverse, wa_.reshape(2, 3, -1))
        return F.linear(z.flatten(1), wb_)

    projected = o_projection(
        projection, o, wa, wb, positions=positions, cos_sin_cache=cache,
        n_groups=2, heads_per_group=2, nope_dim=4, rope_dim=4, o_lora_rank=3,
    )
    projected.backward(_grad_like(projected))
    assert all(value.grad is not None for value in (o, wa, wb))


def test_router_backward_keeps_ids_snapshot_when_consumer_mutates_output() -> None:
    torch.manual_seed(31)
    logits = torch.randn(2, 5, requires_grad=True)
    original_ids = torch.tensor([[4, 1], [0, 3]], dtype=torch.int32)

    def visible(value):
        scores = torch.sqrt(F.softplus(value)).gather(-1, original_ids.long())
        return scores / scores.sum(-1, keepdim=True), original_ids.clone()

    weights, returned_ids = fixed_route_vjp(
        visible, logits, renormalize=True, route_scale=1.0
    )
    assert torch.equal(returned_ids, original_ids)
    returned_ids.fill_(99)
    grad = _grad_like(weights)
    weights.backward(grad)

    reference = logits.detach().requires_grad_(True)
    visible(reference)[0].backward(grad)
    torch.testing.assert_close(logits.grad, reference.grad)
