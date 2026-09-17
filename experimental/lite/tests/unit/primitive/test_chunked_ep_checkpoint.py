# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Numerical contracts for the model-independent EP checkpoint bridge."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import megatron.core  # noqa: F401


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_head_publishes_weight_before_cast(dtype):
    from megatron.core.utils import PARAM_READY_CALLBACK_ATTR
    from megatron.lite.model.qwen3_moe.lite.model import Qwen3MoEModel

    weight = torch.nn.Parameter(torch.zeros(3, 4))
    calls = []

    def publish():
        calls.append(True)
        with torch.no_grad():
            weight.fill_(2)

    setattr(weight, PARAM_READY_CALLBACK_ATTR, publish)
    model = SimpleNamespace(
        head=SimpleNamespace(col=SimpleNamespace(linear=SimpleNamespace(weight=weight)))
    )
    hidden = torch.zeros(1, 4, dtype=dtype)
    actual = Qwen3MoEModel._head_weight_for_fused_ce(model, hidden)
    assert calls == [True]
    assert actual.dtype == dtype
    assert torch.equal(actual, torch.full_like(actual, 2))
    actual.sum().backward()
    assert torch.equal(weight.grad, torch.ones_like(weight))
    delattr(weight, PARAM_READY_CALLBACK_ATTR)
    assert torch.equal(Qwen3MoEModel._head_weight_for_fused_ce(model, hidden), actual)


@pytest.mark.parametrize("shape", [(3, 2), (3, 1, 2), (3, 2, 2), (0, 1, 2)])
def test_saved_bridge_preserves_input_gradient_shape(transformer_engine_import_stub, shape):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import EPChunkForwardOp

    router, experts = torch.nn.Identity(), torch.nn.Identity()
    backward = SimpleNamespace(
        router=router, experts=experts, backward=lambda context, grad: (2 * grad, [], [])
    )
    workspace = SimpleNamespace(key=SimpleNamespace(shape_profile=SimpleNamespace(chunk_count=2)))
    forward = EPChunkForwardOp(
        router=router, experts=experts, workspace=workspace, backward_op=backward
    )
    forward._forward_saved_context_async = lambda x, ranges, shape, dtype: (
        (2 * x).view(shape),
        None,
    )
    x = torch.ones(shape, requires_grad=True)
    output = forward(x)
    assert output.shape == x.shape
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 2))


@pytest.mark.parametrize("full_recompute", [False, True])
@pytest.mark.parametrize("zero_out_wgrad", [False, True])
def test_checkpoint_publishes_main_grad_to_outer_ddp_hook(
    transformer_engine_import_stub, full_recompute, zero_out_wgrad
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _SavedContextEPChunkFunction,
        checkpoint_ep_chunk,
    )

    experts = torch.nn.Linear(2, 2, bias=False)
    weight = experts.weight
    weight.main_grad = torch.zeros_like(weight)
    weight.grad_added_to_main_grad = False
    weight.zero_out_wgrad = zero_out_wgrad
    hooks = []

    def ddp_hook(param):
        assert param.grad is not None, "DDP requires an outer autograd gradient sentinel"
        assert param.grad_added_to_main_grad
        if param.zero_out_wgrad:
            param.main_grad.add_(param.grad)
        param.grad = None
        hooks.append(param.main_grad.clone())

    weight.register_post_accumulate_grad_hook(ddp_hook)

    def forward(x):
        return experts(x)

    forward.router = torch.nn.Identity()
    forward.experts = experts

    def fused(x, grad):
        weight.main_grad.add_(grad.T @ x)
        weight.grad_added_to_main_grad = True
        return grad @ weight, [], [None]

    execution = SimpleNamespace(
        forward_op=forward, fused_op=SimpleNamespace(forward_backward=fused)
    )
    for rows in (3, 5, 2):
        x = torch.ones(rows, 2, requires_grad=True)
        if full_recompute:
            output = checkpoint_ep_chunk(lambda value: (value, value), x, execution, ())
        else:
            forward.backward_op = SimpleNamespace(backward=fused)
            forward._logical_chunk_count = 2
            forward._forward_saved_context_async = lambda value, *args: (experts(value), value)
            output = _SavedContextEPChunkFunction.apply(x, forward, x.shape, weight)
        output.sum().backward()
    assert len(hooks) == 3
    for actual, expected in zip(hooks, (3, 8, 10), strict=True):
        torch.testing.assert_close(actual, torch.full_like(weight, expected))


@pytest.mark.parametrize("layers", [1, 3])
@pytest.mark.parametrize("frozen_prefix", [False, True])
@pytest.mark.parametrize("park_each_layer", [False, True])
def test_checkpoint_matches_native_across_microbatches(
    transformer_engine_import_stub, layers, frozen_prefix, park_each_layer
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        EPChunkForwardOp,
        checkpoint_ep_chunk,
    )

    torch.manual_seed(314)
    native = torch.nn.ModuleList(
        [
            torch.nn.ModuleList([torch.nn.Linear(4, 4).double() for _ in range(3)])
            for _ in range(layers)
        ]
    )
    candidate = deepcopy(native)
    if frozen_prefix:
        for stack in (native, candidate):
            for block in stack:
                block[0].requires_grad_(False)
    calls = []

    def prefix(block, x):
        residual = x + torch.nn.functional.dropout(block[0](x), p=0.2, training=True)
        norm = torch.tanh(residual)
        index = next((i for i, item in enumerate(candidate) if item is block), None)
        if index is not None and (park_each_layer or index == 0) and torch.is_grad_enabled():
            finish = executions[index].finish_backward
            prior_calls = finish.call_count

            def check_parked(grad):
                assert finish.call_count == prior_calls + 1, "expert arena outlived fused backward"
                return grad

            norm.register_hook(check_parked)
        return norm, residual

    def execution(block):
        def forward(x):
            assert not torch.is_grad_enabled()
            calls.append("fwd")
            return block[2](torch.tanh(block[1](x)))

        def fused(x, grad):
            calls.append("fused")
            x = x.detach().requires_grad_(True)
            out = block[2](torch.tanh(block[1](x)))
            grads = torch.autograd.grad(
                out, (x, *block[1].parameters(), *block[2].parameters()), grad
            )
            return grads[0], grads[1:3], grads[3:]

        class Forward(EPChunkForwardOp):
            router, experts = block[1], block[2]
            backward_op = None
            _logical_chunk_count = 2

            def __init__(self):
                pass  # CPU boundary test: no GPU workspace or transport initialization.

            def _forward_output_async(self, x, ranges, input_shape, dtype):
                return forward(x)

        return SimpleNamespace(
            forward_op=Forward(),
            fused_op=SimpleNamespace(forward_backward=fused),
            finish_backward=Mock(),
        )

    executions = [execution(block) for block in candidate]
    with torch.enable_grad(), pytest.raises(RuntimeError, match="requires a paired backward op"):
        executions[0].forward_op(torch.ones(1, 4, dtype=torch.double))
    for rows in (4, 7, 3):
        x = torch.randn(rows, 4, dtype=torch.double, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        seed = torch.get_rng_state()
        out = x
        for block in native:
            norm, residual = prefix(block, out)
            out = residual + block[2](torch.tanh(block[1](norm)))
        native_rng = torch.get_rng_state()
        out.square().mean().backward()
        torch.set_rng_state(seed)
        actual = y
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
            for i, block in enumerate(candidate):
                actual = checkpoint_ep_chunk(
                    lambda t, block=block: prefix(block, t),
                    actual,
                    executions[i],
                    tuple(block[0].parameters()),
                    finish_backward=park_each_layer or i == 0,
                )
        # The bridge retains only each layer input, not attention/norm activations.
        assert len(saved) == layers
        torch.testing.assert_close(actual, out, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), native_rng, rtol=0, atol=0)
        actual.square().mean().backward()
        torch.testing.assert_close(torch.get_rng_state(), native_rng, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-12, atol=1e-12)
        for reference, changed in zip(native.parameters(), candidate.parameters(), strict=True):
            if not reference.requires_grad:
                assert changed.grad is None
            else:
                torch.testing.assert_close(reference.grad, changed.grad, rtol=1e-12, atol=1e-12)
    assert calls.count("fwd") == calls.count("fused") == 3 * layers
    assert executions[0].finish_backward.call_count == 3
    for execution in executions[1:]:
        assert execution.finish_backward.call_count == (3 if park_each_layer else 0)


@pytest.mark.parametrize("full_recompute", [False, True])
def test_qwen_layer_assembly_keeps_parameter_paths(
    transformer_engine_import_stub, monkeypatch, full_recompute
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite import chunked_ep as adapter
    from megatron.lite.model.qwen3_moe.lite import model
    from megatron.lite.primitive.modules import moe_ep_chunk_overlap as ep

    class LinearStub(torch.nn.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(4, 4, bias=False)

        def forward(self, x, *args, **kwargs):
            return super().forward(x)

    class Router(LinearStub):
        def forward(self, x):
            return super().forward(x), None

    def dispatcher(*args, **kwargs):
        return SimpleNamespace(
            dispatch=lambda x, scores, indices: (scores, None, None),
            wait_dispatch_event=lambda: None,
            combine=lambda x: x,
        )

    class Execution:
        def __init__(self, *, router, experts, retain_backward, **kwargs):
            def run(x):
                return experts(router(x)[0])

            self.forward_op = run
            self.forward_op.router, self.forward_op.experts = router, experts
            self.backward_op = object() if retain_backward else None

            def fused(x, grad):
                x = x.detach().requires_grad_(True)
                grads = torch.autograd.grad(
                    run(x), (x, *router.parameters(), *experts.parameters()), grad
                )
                return grads[0], grads[1:2], grads[2:]

            self.fused_op = None if retain_backward else SimpleNamespace(forward_backward=fused)
            self.finish_backward = Mock()
            self.finish_forward = Mock()

    monkeypatch.setattr(model, "GQAttention", LinearStub)
    monkeypatch.setattr(model.te, "RMSNorm", lambda *args, **kwargs: torch.nn.LayerNorm(4))
    monkeypatch.setattr(ep, "EPChunkExecution", Execution)
    for module in (model, adapter):
        monkeypatch.setattr(module, "TopKRouter", Router)
        monkeypatch.setattr(module, "Experts", LinearStub)
        monkeypatch.setattr(module, "TokenDispatcher", dispatcher)
    cfg = SimpleNamespace(
        hidden_size=4,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        rms_norm_eps=1e-5,
        rope_theta=10000,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=4,
    )
    ps = SimpleNamespace(ep_size=2, tp_ep_group=object())
    native = model.TransformerLayer(cfg, ps, 0).double()
    options = adapter.Qwen3ChunkedEP(16, full_recompute=full_recompute)
    changed = options.layer(cfg, ps, 0).double()
    assert tuple(native.state_dict()) == tuple(changed.state_dict())
    assert tuple(dict(native.named_parameters())) == tuple(dict(changed.named_parameters()))
    changed.load_state_dict(native.state_dict(), strict=True)
    sibling = options.layer(cfg, ps, 1).double()
    selected = options.bind([changed, sibling])
    assert changed.full_recompute == sibling.full_recompute == full_recompute
    assert selected is changed.moe.chunked_ep
    assert options.bind([]) is None
    x = torch.randn(5, 1, 4, dtype=torch.double, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    expected = native(x)
    actual = changed(y)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad)
    for a, b in zip(native.parameters(), changed.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad)
    assert selected.finish_backward.call_count == int(full_recompute)
