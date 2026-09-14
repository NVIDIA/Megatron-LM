# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Numerical contracts for the model-independent EP checkpoint bridge."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import megatron.core  # noqa: F401


@pytest.mark.parametrize("layers", [1, 3])
@pytest.mark.parametrize("frozen_prefix", [False, True])
def test_checkpoint_matches_native_across_microbatches(
    transformer_engine_import_stub, layers, frozen_prefix
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import checkpoint_ep_chunk

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
        return torch.tanh(residual), residual

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

        class Forward:
            router, experts = block[1], block[2]

            def __call__(self, x):
                return forward(x)

        return SimpleNamespace(
            forward_op=Forward(),
            fused_op=SimpleNamespace(forward_backward=fused),
            finish_backward=Mock(),
        )

    executions = [execution(block) for block in candidate]
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
                    finish_backward=i == 0,
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
        execution.finish_backward.assert_not_called()


@pytest.mark.parametrize("full_recompute", [False, True])
def test_qwen_layer_assembly_keeps_parameter_paths(
    transformer_engine_import_stub, monkeypatch, full_recompute
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite import chunked_ep as adapter
    from megatron.lite.model.qwen3_moe.lite import model
    from megatron.lite.primitive.modules import moe_ep_chunk_overlap as ep

    class Attention(torch.nn.Linear):
        def __init__(self, **kwargs):
            super().__init__(4, 4, bias=False)

        def forward(self, x, **kwargs):
            return super().forward(x)

    class Router(torch.nn.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(4, 4, bias=False)

        def forward(self, x):
            return super().forward(x), None

    class Experts(torch.nn.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(4, 4, bias=False)

        def forward(self, x, *args, **kwargs):
            return super().forward(x)

    class Dispatcher:
        def __init__(self, *args, **kwargs):
            pass

        def dispatch(self, x, scores, indices):
            return scores, None, None

        def wait_dispatch_event(self):
            pass

        def combine(self, x):
            return x

    class Execution:
        def __init__(self, *, router, experts, retain_backward, **kwargs):
            def run(x):
                return experts(router(x)[0])

            class Forward:
                def __call__(self, x):
                    return run(x)

            self.forward_op = Forward()
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

    monkeypatch.setattr(model, "GQAttention", Attention)
    monkeypatch.setattr(model.te, "RMSNorm", lambda *args, **kwargs: torch.nn.LayerNorm(4))
    monkeypatch.setattr(ep, "EPChunkExecution", Execution)
    for module in (model, adapter):
        monkeypatch.setattr(module, "TopKRouter", Router)
        monkeypatch.setattr(module, "Experts", Experts)
        monkeypatch.setattr(module, "TokenDispatcher", Dispatcher)
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
    selected = options.bind([changed])
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
