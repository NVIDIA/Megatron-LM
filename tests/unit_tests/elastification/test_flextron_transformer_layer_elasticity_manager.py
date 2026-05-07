# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GPU-backed tests for FlextronTransformerLayerElasticityManager.

Tests the pre_mlp_layernorm pre/post hooks for E-layers. Run with:

    torchrun --nproc_per_node=1 -m pytest tests/unit_tests/elastification/test_flextron_transformer_layer_elasticity_manager.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.elastification.flextron_elasticity_hooks import (
    FlextronTransformerLayerElasticityManager,
    add_flextron_transformer_layer_elasticity,
)


def _tl_config(emb_int_list=(256, 128), soft_mask=True, layernorm_epsilon=1e-5):
    return SimpleNamespace(
        flextron=True,
        soft_mask=soft_mask,
        hidden_size=256,
        emb_int_list=list(emb_int_list),
        layernorm_epsilon=layernorm_epsilon,
    )


def _fake_transformer_layer(hidden_size=256, eps=1e-5):
    """Minimal module exposing .pre_mlp_layernorm (the only submodule hooked)."""
    layer = nn.Module()
    layer.pre_mlp_layernorm = nn.LayerNorm(hidden_size, eps=eps).cuda().to(torch.bfloat16)
    return layer


@pytest.mark.internal
class TestFlextronTransformerLayerElasticityManager:
    def teardown_method(self, method):
        pass

    def test_attach_registers_two_hooks(self):
        config = _tl_config()
        mgr = FlextronTransformerLayerElasticityManager(config)
        layer = _fake_transformer_layer()
        mgr.attach_hooks(layer)
        assert len(mgr.hook_handles) == 2

    def test_current_router_emb_none_is_noop(self):
        """With current_router_emb unset, hook behavior must match no-hook forward."""
        config = _tl_config()
        layer = _fake_transformer_layer()
        x = torch.randn(4, 2, 256, dtype=torch.bfloat16, device="cuda")
        expected = layer.pre_mlp_layernorm(x)

        mgr = FlextronTransformerLayerElasticityManager(config)
        mgr.attach_hooks(layer)
        # current_router_emb is None — no masking, no scaling.
        out = layer.pre_mlp_layernorm(x)
        torch.testing.assert_close(out, expected)

    def test_soft_mask_scales_output(self):
        """With soft_mask one-hot on a smaller choice, the hook should:
        (1) zero input channels beyond the chosen emb_int, then
        (2) scale the LN output by sqrt(emb_per).
        Reproduce the expected output by applying that mask+scale manually
        against plain LN and assert equality (within bf16 tolerance)."""
        config = _tl_config(emb_int_list=[256, 128], soft_mask=True)
        layer = _fake_transformer_layer()

        x = torch.randn(4, 2, 256, dtype=torch.bfloat16, device="cuda")

        # Build expected output: mask upper half, LN, scale by sqrt(emb_per).
        mask = torch.zeros(256, dtype=torch.bfloat16, device="cuda")
        mask[:128] = 1.0
        expected = layer.pre_mlp_layernorm(x * mask[None, None, :]) * (128 / 256) ** 0.5

        mgr = FlextronTransformerLayerElasticityManager(config)
        mgr.attach_hooks(layer)
        # One-hot on index 1 (emb_int=128 -> per=0.5)
        per_logits = torch.tensor([0.0, 1.0], dtype=torch.bfloat16, device="cuda")
        mgr.set_elasticity_params(router_emb=(per_logits, 128))
        out = layer.pre_mlp_layernorm(x)

        # Tolerance accommodates the tiny eps drift (5e-6 vs 1e-5) inside LN.
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_full_budget_one_hot_preserves_magnitude_order(self):
        """When router is one-hot on full budget (index 0 = 100% emb), the
        pre-hook masks nothing and post-hook scales by sqrt(1.0)=1.0."""
        config = _tl_config(emb_int_list=[256, 128], soft_mask=True)
        layer = _fake_transformer_layer()
        x = torch.randn(4, 2, 256, dtype=torch.bfloat16, device="cuda")
        baseline = layer.pre_mlp_layernorm(x)

        mgr = FlextronTransformerLayerElasticityManager(config)
        mgr.attach_hooks(layer)
        per_logits = torch.tensor([1.0, 0.0], dtype=torch.bfloat16, device="cuda")
        mgr.set_elasticity_params(router_emb=(per_logits, 256))

        out = layer.pre_mlp_layernorm(x)
        # Full-budget path: input mask is all-ones, scale is sqrt(1.0). Output
        # should equal baseline within bf16 tolerance (eps adjustment may add
        # tiny drift).
        torch.testing.assert_close(out, baseline, atol=5e-2, rtol=5e-2)

    def test_detach_restores_forward(self):
        config = _tl_config()
        layer = _fake_transformer_layer()
        x = torch.randn(4, 2, 256, dtype=torch.bfloat16, device="cuda")

        mgr = FlextronTransformerLayerElasticityManager(config)
        mgr.attach_hooks(layer)
        per_logits = torch.tensor([0.0, 1.0], dtype=torch.bfloat16, device="cuda")
        mgr.set_elasticity_params(router_emb=(per_logits, 128))

        masked_out = layer.pre_mlp_layernorm(x)
        mgr.detach_hooks()
        detached_out = layer.pre_mlp_layernorm(x)

        # After detach, the output should match the un-hooked LN output.
        expected = nn.LayerNorm(256, eps=layer.pre_mlp_layernorm.eps).cuda().to(torch.bfloat16)
        expected.weight.data.copy_(layer.pre_mlp_layernorm.weight.data)
        expected.bias.data.copy_(layer.pre_mlp_layernorm.bias.data)
        torch.testing.assert_close(detached_out, expected(x), atol=1e-2, rtol=1e-2)
        # The masked output from before detach should differ from the detached one.
        assert not torch.allclose(masked_out, detached_out, atol=1e-2)


@pytest.mark.internal
class TestAddFlextronTransformerLayerElasticity:
    def test_factory_returns_manager(self):
        config = _tl_config()
        layer = _fake_transformer_layer()
        mgr = add_flextron_transformer_layer_elasticity(layer, config, layer_idx=3)
        assert isinstance(mgr, FlextronTransformerLayerElasticityManager)
        assert mgr.layer_idx == 3
        assert len(mgr.hook_handles) == 2
        mgr.detach_hooks()
