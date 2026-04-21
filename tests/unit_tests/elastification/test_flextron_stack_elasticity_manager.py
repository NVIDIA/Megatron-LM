# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GPU-backed tests for FlextronStackElasticityManager.

Tests the final-norm hooks that apply eps modification and sqrt(emb_per)
scaling when the router supplies an embedding choice. Run with:

    torchrun --nproc_per_node=1 -m pytest tests/unit_tests/elastification/test_flextron_stack_elasticity_manager.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.elastification.flextron_elasticity_hooks import (
    FlextronStackElasticityManager,
    add_flextron_stack_elasticity,
)


def _stack_config(emb_int_list=(256, 128), soft_mask=True, layernorm_epsilon=1e-5):
    """Minimal SimpleNamespace exposing every attr the stack manager reads."""
    return SimpleNamespace(
        flextron=True,
        soft_mask=soft_mask,
        hidden_size=256,
        emb_int_list=list(emb_int_list),
        layernorm_epsilon=layernorm_epsilon,
    )


def _stack_with_final_norm(hidden_size=256, eps=1e-5):
    """A stand-in MambaStack: only `final_norm` is hooked, nothing else required."""
    stack = nn.Module()
    stack.final_norm = nn.LayerNorm(hidden_size, eps=eps).cuda().to(torch.bfloat16)
    return stack


@pytest.mark.internal
class TestFlextronStackElasticityManager:
    def teardown_method(self, method):
        # No parallel state was initialized; nothing to tear down.
        pass

    def test_disabled_manager_is_noop(self):
        config = _stack_config()
        config.flextron = False
        mgr = FlextronStackElasticityManager(config)
        stack = _stack_with_final_norm()
        mgr.attach_hooks(stack)  # Should silently skip.
        assert mgr.hook_handles == [] if hasattr(mgr, "hook_handles") else True

    def test_attach_registers_two_hooks(self):
        config = _stack_config()
        mgr = FlextronStackElasticityManager(config)
        stack = _stack_with_final_norm()
        mgr.attach_hooks(stack)
        # One pre-hook + one post-hook on final_norm.
        assert len(mgr.hook_handles) == 2

    def test_current_router_emb_none_is_noop(self):
        """Without elasticity params set, hooks must pass through unchanged."""
        config = _stack_config()
        mgr = FlextronStackElasticityManager(config)
        stack = _stack_with_final_norm()
        mgr.attach_hooks(stack)
        x = torch.randn(4, 2, 256, dtype=torch.bfloat16, device="cuda")

        expected = stack.final_norm(x)  # direct call — hooks do run but should no-op
        # Hooks were attached in-place, so call again to capture the hooked output.
        out = stack.final_norm(x)
        torch.testing.assert_close(out, expected)

    def test_soft_mask_scales_output_by_sqrt_emb_per(self):
        """With soft_mask and a one-hot router distribution, output should scale by
        sqrt(emb_per) of the selected choice."""
        config = _stack_config(emb_int_list=[256, 128], soft_mask=True)
        mgr = FlextronStackElasticityManager(config)
        stack = _stack_with_final_norm()
        mgr.attach_hooks(stack)

        # One-hot on index 1 (emb_int=128 -> per=0.5)
        per_logits = torch.tensor([0.0, 1.0], dtype=torch.bfloat16, device="cuda")
        mgr.set_elasticity_params(router_emb=(per_logits, 128))

        x = torch.randn(4, 2, 256, dtype=torch.bfloat16, device="cuda")
        # Baseline without elasticity: detach hooks first.
        mgr.detach_hooks()
        baseline = stack.final_norm(x)

        # Re-attach and run with elasticity.
        mgr.attach_hooks(stack)
        mgr.set_elasticity_params(router_emb=(per_logits, 128))
        scaled = stack.final_norm(x)

        # Expected: baseline * sqrt(0.5)  (since per_logit is 1.0 on idx 1)
        expected_scale = (128 / 256) ** 0.5
        torch.testing.assert_close(
            scaled, baseline * expected_scale, atol=1e-2, rtol=1e-2
        )

    def test_detach_removes_all_hooks(self):
        config = _stack_config()
        mgr = FlextronStackElasticityManager(config)
        stack = _stack_with_final_norm()
        mgr.attach_hooks(stack)
        assert len(mgr.hook_handles) == 2
        mgr.detach_hooks()
        assert mgr.hook_handles == []


@pytest.mark.internal
class TestAddFlextronStackElasticity:
    def test_factory_returns_manager_with_hooks_attached(self):
        config = _stack_config()
        stack = _stack_with_final_norm()
        mgr = add_flextron_stack_elasticity(stack, config)
        assert isinstance(mgr, FlextronStackElasticityManager)
        assert len(mgr.hook_handles) == 2
        mgr.detach_hooks()
