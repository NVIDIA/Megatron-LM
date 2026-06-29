# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for ``apply_flextron_elasticity_to_model``.

These tests focus on the layer-class-name-based routing logic (which manager
gets attached to which layer type). They use stub nn.Modules so the tests are
pure-Python and run without a GPU or distributed setup. The individual manager
classes are exercised via GPU-backed tests elsewhere.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.elastification import flextron_elasticity_hooks as hooks_module
from megatron.elastification.flextron_elasticity_hooks import apply_flextron_elasticity_to_model


def _make_submod(class_name):
    """Build a bare nn.Module with a given __class__.__name__."""
    mod = nn.Module()
    mod.__class__ = type(class_name, (nn.Module,), {})
    return mod


def _mamba_layer():
    layer = nn.Module()
    layer.__class__ = type("MambaLayer", (nn.Module,), {})
    layer.add_module("mixer", _make_submod("MambaMixer"))
    return layer


def _moe_layer(*, cls="MoETransformerLayer"):
    layer = nn.Module()
    layer.__class__ = type(cls, (nn.Module,), {})
    layer.add_module("pre_mlp_layernorm", _make_submod("RMSNorm"))
    mlp = _make_submod("MoELayer")
    mlp.add_module("router", _make_submod("TopKRouter"))
    mlp.add_module("experts", _make_submod("TEGroupedMLP"))
    layer.add_module("mlp", mlp)
    return layer


def _attention_layer():
    layer = nn.Module()
    layer.__class__ = type("TransformerLayer", (nn.Module,), {})
    attn = _make_submod("SelfAttention")
    layer.add_module("self_attention", attn)
    return layer


class _StubModel(nn.Module):
    """Minimal model exposing .decoder.layers and (optionally) .decoder.final_norm."""

    def __init__(self, layers, with_final_norm=False):
        super().__init__()
        decoder = nn.Module()
        decoder.layers = nn.ModuleList(layers)
        if with_final_norm:
            decoder.add_module("final_norm", _make_submod("RMSNorm"))
        self.decoder = decoder


def _make_config(pattern="MEM*", flextron=True):
    return SimpleNamespace(hybrid_layer_pattern=pattern, flextron=flextron)


@pytest.fixture(autouse=True)
def stub_managers(monkeypatch):
    """Stub every add_flextron_* entry point to a call-recorder.

    The real managers attach PyTorch hooks to real submodules; testing the
    routing logic does not need that machinery. Each stub returns a Sentinel
    whose ``.target`` field points at the module it would have hooked.
    """
    calls = {
        "transformer_layer": [],
        "moe": [],
        "topk_router": [],
        "grouped_mlp": [],
        "mamba": [],
        "attention": [],
        "stack": [],
    }

    def _record(bucket):
        def _stub(module, config, layer_idx=None):
            entry = SimpleNamespace(target=module, layer_idx=layer_idx, config=config)
            calls[bucket].append(entry)
            return entry

        return _stub

    def _record_stack(module, config):
        entry = SimpleNamespace(target=module, config=config)
        calls["stack"].append(entry)
        return entry

    monkeypatch.setattr(
        hooks_module, "add_flextron_transformer_layer_elasticity", _record("transformer_layer")
    )
    monkeypatch.setattr(hooks_module, "add_flextron_moe_elasticity", _record("moe"))
    monkeypatch.setattr(hooks_module, "add_flextron_topk_router_elasticity", _record("topk_router"))
    monkeypatch.setattr(hooks_module, "add_flextron_grouped_mlp_elasticity", _record("grouped_mlp"))
    monkeypatch.setattr(hooks_module, "add_flextron_mamba_elasticity", _record("mamba"))
    monkeypatch.setattr(hooks_module, "add_flextron_attention_elasticity", _record("attention"))
    monkeypatch.setattr(hooks_module, "add_flextron_stack_elasticity", _record_stack)

    return calls


class TestEarlyReturns:
    def test_missing_hybrid_pattern_returns_empty(self):
        model = _StubModel([_mamba_layer()])
        config = SimpleNamespace()  # no hybrid_layer_pattern
        assert apply_flextron_elasticity_to_model(model, config) == []

    def test_empty_hybrid_pattern_returns_empty(self):
        model = _StubModel([_mamba_layer()])
        config = _make_config(pattern="")
        assert apply_flextron_elasticity_to_model(model, config) == []

    def test_missing_decoder_returns_empty(self):
        model = nn.Module()  # no .decoder
        config = _make_config()
        assert apply_flextron_elasticity_to_model(model, config) == []


class TestLayerRouting:
    def test_m_layer_registers_mamba_only(self, stub_managers):
        model = _StubModel([_mamba_layer()])
        config = _make_config(pattern="M")
        apply_flextron_elasticity_to_model(model, config)
        assert len(stub_managers["mamba"]) == 1
        assert stub_managers["mamba"][0].layer_idx == 0
        assert stub_managers["mamba"][0].target.__class__.__name__ == "MambaMixer"
        for key in ("transformer_layer", "moe", "topk_router", "grouped_mlp", "attention"):
            assert stub_managers[key] == []

    def test_star_layer_registers_attention_only(self, stub_managers):
        model = _StubModel([_attention_layer()])
        config = _make_config(pattern="*")
        apply_flextron_elasticity_to_model(model, config)
        assert len(stub_managers["attention"]) == 1
        assert stub_managers["attention"][0].target.__class__.__name__ == "SelfAttention"
        for key in ("transformer_layer", "moe", "topk_router", "grouped_mlp", "mamba"):
            assert stub_managers[key] == []

    def test_e_layer_registers_all_four_moe_managers(self, stub_managers):
        model = _StubModel([_moe_layer()])
        config = _make_config(pattern="E")
        apply_flextron_elasticity_to_model(model, config)
        assert len(stub_managers["transformer_layer"]) == 1
        assert len(stub_managers["moe"]) == 1
        assert len(stub_managers["topk_router"]) == 1
        assert len(stub_managers["grouped_mlp"]) == 1

    def test_e_layer_accepts_both_class_names(self, stub_managers):
        """Regression: the E-layer hook should fire whether the layer class is
        TransformerLayer (modelopt spec) or MoETransformerLayer (default spec)."""
        model = _StubModel(
            [_moe_layer(cls="TransformerLayer"), _moe_layer(cls="MoETransformerLayer")]
        )
        config = _make_config(pattern="EE")
        apply_flextron_elasticity_to_model(model, config)
        # Both E-layers should have TransformerLayer elasticity attached.
        assert len(stub_managers["transformer_layer"]) == 2

    def test_hybrid_pattern_routes_each_layer(self, stub_managers):
        layers = [_mamba_layer(), _moe_layer(), _mamba_layer(), _attention_layer()]
        model = _StubModel(layers, with_final_norm=True)
        config = _make_config(pattern="MEM*")
        apply_flextron_elasticity_to_model(model, config)

        # One mamba manager per M, one attention per *, and all four moe managers per E.
        assert len(stub_managers["mamba"]) == 2
        assert len(stub_managers["attention"]) == 1
        assert len(stub_managers["transformer_layer"]) == 1
        assert len(stub_managers["moe"]) == 1
        assert len(stub_managers["topk_router"]) == 1
        assert len(stub_managers["grouped_mlp"]) == 1
        # And a single stack-level manager for the final norm.
        assert len(stub_managers["stack"]) == 1


class TestStackManager:
    def test_stack_manager_registered_when_final_norm_present(self, stub_managers):
        model = _StubModel([_mamba_layer()], with_final_norm=True)
        config = _make_config(pattern="M")
        apply_flextron_elasticity_to_model(model, config)
        assert len(stub_managers["stack"]) == 1

    def test_stack_manager_skipped_when_no_final_norm(self, stub_managers):
        model = _StubModel([_mamba_layer()], with_final_norm=False)
        config = _make_config(pattern="M")
        apply_flextron_elasticity_to_model(model, config)
        assert stub_managers["stack"] == []


class TestMissingSubmodules:
    def test_mamba_layer_without_mixer_is_skipped(self, stub_managers):
        """M-layer without a MambaMixer submodule should not crash."""
        layer = nn.Module()
        layer.__class__ = type("MambaLayer", (nn.Module,), {})
        # intentionally no 'mixer' submodule
        model = _StubModel([layer])
        config = _make_config(pattern="M")
        apply_flextron_elasticity_to_model(model, config)
        assert stub_managers["mamba"] == []

    def test_attention_layer_without_self_attention_is_skipped(self, stub_managers):
        layer = nn.Module()
        layer.__class__ = type("TransformerLayer", (nn.Module,), {})
        # no SelfAttention submodule
        model = _StubModel([layer])
        config = _make_config(pattern="*")
        apply_flextron_elasticity_to_model(model, config)
        assert stub_managers["attention"] == []


class TestManagersStoredOnModel:
    def test_model_gets_flextron_managers_attribute(self, stub_managers):
        model = _StubModel([_mamba_layer()])
        config = _make_config(pattern="M")
        returned = apply_flextron_elasticity_to_model(model, config)
        assert model._flextron_managers is returned
        assert len(returned) == len(stub_managers["mamba"])

    def test_pattern_shorter_than_layers_only_uses_pattern_length(self, stub_managers):
        layers = [_mamba_layer(), _mamba_layer(), _mamba_layer()]
        model = _StubModel(layers)
        config = _make_config(pattern="M")  # only first layer is covered
        apply_flextron_elasticity_to_model(model, config)
        assert len(stub_managers["mamba"]) == 1
