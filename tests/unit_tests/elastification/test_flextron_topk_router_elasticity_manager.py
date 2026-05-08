# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for FlextronTopKRouterElasticityManager.

Focuses on the hard-mask path: it replaces the router's ``routing`` method
with a wrapper that masks the upper expert indices before delegating, and
must save/restore ``router.expert_bias`` around the call (regression
guard for the prior permanent-mutation bug).

The soft-mask path goes through ``topk_softmax_with_capacity``, which
requires real MoE plumbing — covered by integration tests, not here.

Run with:
    torchrun --nproc_per_node=1 -m pytest tests/unit_tests/elastification/test_flextron_topk_router_elasticity_manager.py
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.elastification.flextron_elasticity_hooks import (
    FlextronTopKRouterElasticityManager,
    add_flextron_topk_router_elasticity,
)


def _config(num_moe_experts=8, soft_mask=False, flex_hetero_moe_expert=False):
    return SimpleNamespace(
        flextron=True,
        soft_mask=soft_mask,
        flex_hetero_moe_expert=flex_hetero_moe_expert,
        num_moe_experts=num_moe_experts,
        moe_expert_int_list=[num_moe_experts, num_moe_experts // 2],
        hybrid_layer_pattern="E",
    )


class _StubRouter:
    """Minimal router: holds an ``expert_bias`` tensor and an ``original_routing``
    method that records its inputs so we can assert what was passed."""

    def __init__(self, expert_bias):
        self.expert_bias = expert_bias
        # Record (logits_clone, expert_bias_clone) at call-time so we can verify
        # what the inner call observed.
        self.calls = []

        def routing(logits, **kwargs):
            self.calls.append(
                {
                    "logits": logits.detach().clone(),
                    "expert_bias": self.expert_bias.detach().clone(),
                    "kwargs": kwargs,
                }
            )
            return logits, kwargs

        self.routing = routing


@pytest.mark.internal
class TestFlextronTopKRouterElasticityManager:
    def test_attach_replaces_routing_method(self):
        cfg = _config()
        router = _StubRouter(expert_bias=torch.zeros(8))
        original = router.routing
        mgr = FlextronTopKRouterElasticityManager(cfg)
        mgr.attach_hooks(router)
        assert router.routing is not original
        # The handle list records the method-replacement entry for detach.
        assert len(mgr.hook_handles) == 1
        assert mgr.hook_handles[0][0] == "method_replacement"

    def test_no_elasticity_params_delegates_to_original(self):
        """When current_router_moe_expert is None, wrapped_routing must
        forward (logits, kwargs) unchanged to the original method."""
        cfg = _config()
        router = _StubRouter(expert_bias=torch.zeros(8))
        mgr = FlextronTopKRouterElasticityManager(cfg)
        mgr.attach_hooks(router)

        logits = torch.randn(4, 8)
        out_logits, out_kwargs = router.routing(logits, foo="bar")

        assert len(router.calls) == 1
        # Logits passed through untouched.
        torch.testing.assert_close(router.calls[0]["logits"], logits)
        assert router.calls[0]["kwargs"] == {"foo": "bar"}
        torch.testing.assert_close(out_logits, logits)

    def test_hard_mask_truncates_upper_logits(self):
        """With expert_int=4 (half), logits[:, 4:] should be -inf when the
        original routing sees them, and logits[:, :4] should equal the input
        scaled by the router_moe_expert logit (max of one-hot)."""
        cfg = _config(num_moe_experts=8, soft_mask=False)
        router = _StubRouter(expert_bias=torch.zeros(8))
        mgr = FlextronTopKRouterElasticityManager(cfg)
        mgr.attach_hooks(router)

        # One-hot on the half-experts choice (index 1 of moe_expert_int_list = 4 experts).
        per_logits = torch.tensor([0.0, 1.0])
        mgr.set_elasticity_params(router_moe_expert=(per_logits, 4))

        logits = torch.ones(2, 8)
        router.routing(logits)

        seen = router.calls[0]["logits"]
        # Lower 4 columns: scaled by router_moe_expert_logits = max(per_logits) = 1.0.
        torch.testing.assert_close(seen[:, :4], torch.ones(2, 4))
        # Upper 4 columns: -inf.
        assert torch.isinf(seen[:, 4:]).all() and (seen[:, 4:] < 0).all()

    def test_hard_mask_preserves_expert_bias_after_call(self):
        """Regression: the wrapper must save and restore router.expert_bias.
        Previously it left a truncated clone bound to router.expert_bias,
        leaking into subsequent forwards."""
        cfg = _config(num_moe_experts=8, soft_mask=False)
        original_bias = torch.arange(8, dtype=torch.float32) + 1.0
        router = _StubRouter(expert_bias=original_bias.clone())
        mgr = FlextronTopKRouterElasticityManager(cfg)
        mgr.attach_hooks(router)
        mgr.set_elasticity_params(router_moe_expert=(torch.tensor([0.0, 1.0]), 4))

        bias_before = router.expert_bias.clone()
        router.routing(torch.zeros(2, 8))
        bias_after = router.expert_bias

        # The bias seen *during* the call should have indices 4: zeroed.
        seen_bias = router.calls[0]["expert_bias"]
        assert (seen_bias[:4] == bias_before[:4]).all()
        assert (seen_bias[4:] == 0).all()
        # But the bias on the router after the call must be the original.
        torch.testing.assert_close(bias_after, original_bias)
        # Same Python object, not just equal values.
        assert bias_after is not seen_bias

    def test_hard_mask_bias_restored_even_if_inner_raises(self):
        """``try/finally`` must restore expert_bias when original_routing raises."""
        cfg = _config(num_moe_experts=8, soft_mask=False)
        original_bias = torch.arange(8, dtype=torch.float32) + 1.0
        router = _StubRouter(expert_bias=original_bias.clone())

        def boom(logits, **kwargs):
            raise RuntimeError("simulated downstream failure")

        router.routing = boom

        mgr = FlextronTopKRouterElasticityManager(cfg)
        mgr.attach_hooks(router)
        mgr.set_elasticity_params(router_moe_expert=(torch.tensor([0.0, 1.0]), 4))

        with pytest.raises(RuntimeError, match="simulated"):
            router.routing(torch.zeros(2, 8))

        torch.testing.assert_close(router.expert_bias, original_bias)

    def test_hard_mask_with_no_expert_bias(self):
        """When router has no expert_bias, the save/restore branch must skip
        cleanly and the inner call must still see the masked logits."""
        cfg = _config(num_moe_experts=8, soft_mask=False)

        # Minimal router: no expert_bias attribute, original_routing records
        # only the logits it saw (avoids the StubRouter's bias.detach()).
        class _RouterNoBias:
            def __init__(self):
                self.expert_bias = None
                self.seen = None

                def routing(logits, **kwargs):
                    self.seen = logits.detach().clone()
                    return logits, kwargs

                self.routing = routing

        router = _RouterNoBias()
        mgr = FlextronTopKRouterElasticityManager(cfg)
        mgr.attach_hooks(router)
        mgr.set_elasticity_params(router_moe_expert=(torch.tensor([0.0, 1.0]), 4))

        router.routing(torch.ones(2, 8))
        assert torch.isinf(router.seen[:, 4:]).all()
        # The bias attribute must remain None — no accidental clone-binding.
        assert router.expert_bias is None

    def test_detach_restores_original_routing(self):
        cfg = _config()
        router = _StubRouter(expert_bias=torch.zeros(8))
        original_callable = router.routing  # capture pre-attach reference
        mgr = FlextronTopKRouterElasticityManager(cfg)
        mgr.attach_hooks(router)
        wrapped = router.routing
        assert wrapped is not original_callable

        mgr.detach_hooks()

        # After detach: routing is back, the helper attribute is gone, the
        # handle list is empty.
        assert router.routing is original_callable
        assert not hasattr(router, "_original_routing")
        assert mgr.hook_handles == []


@pytest.mark.internal
class TestAddFlextronTopKRouterElasticity:
    def test_factory_returns_manager(self):
        cfg = _config()
        router = _StubRouter(expert_bias=torch.zeros(8))
        mgr = add_flextron_topk_router_elasticity(router, cfg, layer_idx=0)
        assert isinstance(mgr, FlextronTopKRouterElasticityManager)
        assert len(mgr.hook_handles) == 1
        mgr.detach_hooks()
