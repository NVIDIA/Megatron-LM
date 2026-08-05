# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Allocation-free tests for direct layer descriptors at the HybridStack boundary."""

from copy import deepcopy
from types import SimpleNamespace

from torch import nn

import megatron.core.models.hybrid.hybrid_block as hybrid_block_module
from megatron.core.models.hybrid.hybrid_architecture import HybridLayerSpec
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.transformer import TransformerConfig


class _BuiltLayer(nn.Module):
    """Parameter-free layer returned by the patched module builder."""

    def __init__(self, layer_number: int):
        super().__init__()
        self.layer_number = layer_number

    def mamba_state_shapes_per_request(self):
        return ("conv", self.layer_number), ("ssm", self.layer_number)


def _config(num_layers: int) -> TransformerConfig:
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=64,
        num_attention_heads=8,
        num_query_groups=4,
        kv_channels=8,
        ffn_hidden_size=128,
        num_moe_experts=8,
        moe_ffn_hidden_size=96,
        moe_router_topk=2,
        mamba_state_dim=16,
        mamba_head_dim=8,
        mamba_num_heads=8,
        mamba_num_groups=2,
        add_bias_linear=False,
    )


def _patch_builder(monkeypatch):
    calls = []

    def fake_build_module(module_spec, **kwargs):
        calls.append((module_spec, kwargs))
        return _BuiltLayer(kwargs["layer_number"])

    monkeypatch.setattr(hybrid_block_module, "build_module", fake_build_module)
    return calls


def _build_stack(config, *, layer_specs=None, layer_type_list=None, pp_layer_offset=0):
    return HybridStack(
        config=config,
        submodules=hybrid_stack_spec.submodules,
        pre_process=False,
        layer_specs=layer_specs,
        layer_type_list=layer_type_list,
        pp_layer_offset=pp_layer_offset,
        post_layer_norm=False,
        post_process=False,
        pg_collection=SimpleNamespace(pp=None, tp=None),
        name="decoder",
    )


def test_direct_stack_builds_existing_specs_with_occurrence_configs_and_global_numbers(monkeypatch):
    calls = _patch_builder(monkeypatch)
    base_config = _config(3)
    mamba_config = deepcopy(base_config)
    attention_config = deepcopy(base_config)
    moe_config = deepcopy(base_config)
    mamba_config.mamba_state_dim = 24
    attention_config.kv_channels = 16
    moe_config.moe_ffn_hidden_size = 112
    moe_config.moe_router_topk = 4

    stock_specs = [
        hybrid_stack_spec.submodules.mamba_layer,
        hybrid_stack_spec.submodules.attention_layer,
        hybrid_stack_spec.submodules.moe_layer,
    ]
    occurrence_configs = [mamba_config, attention_config, moe_config]
    layer_specs = [
        HybridLayerSpec(module_spec=module_spec, config=layer_config)
        for module_spec, layer_config in zip(stock_specs, occurrence_configs)
    ]

    stack = _build_stack(base_config, layer_specs=layer_specs, pp_layer_offset=11)

    assert len(calls) == 3
    assert all(call[0] is expected for call, expected in zip(calls, stock_specs))
    assert all(call[1]["config"] is expected for call, expected in zip(calls, occurrence_configs))
    assert [call[1]["layer_number"] for call in calls] == [12, 13, 14]
    assert [layer.layer_number for layer in stack.layers] == [12, 13, 14]
    assert [call[1]["name"] for call in calls] == [
        "decoder.layers.0",
        "decoder.layers.1",
        "decoder.layers.2",
    ]

    assert calls[0][1]["pp_layer_offset"] == 11
    assert calls[1][1]["pp_layer_offset"] == 11
    assert "pp_layer_offset" not in calls[2][1]
    assert calls[1][1]["add_layer_offset"] is False
    assert calls[2][1]["add_layer_offset"] is False


def test_legacy_and_equivalent_direct_stack_build_same_semantic_and_module_order(monkeypatch):
    calls = _patch_builder(monkeypatch)
    config = _config(4)
    submodules = hybrid_stack_spec.submodules
    direct_specs = [
        HybridLayerSpec(submodules.mamba_layer, config),
        HybridLayerSpec(submodules.attention_layer, config),
        HybridLayerSpec(submodules.moe_layer, config),
        HybridLayerSpec(submodules.mlp_layer, config),
    ]

    direct_stack = _build_stack(config, layer_specs=direct_specs, pp_layer_offset=3)
    direct_calls = list(calls)
    calls.clear()
    legacy_stack = _build_stack(config, layer_type_list=["M", "*", "E", "-"], pp_layer_offset=3)

    expected_types = ["mamba", "attention", "moe", "mlp"]
    assert direct_stack.layer_type_list == expected_types
    assert legacy_stack.layer_type_list == ["M", "*", "E", "-"]
    assert not hasattr(legacy_stack, "layer_specs")
    assert [call[0].module for call in direct_calls] == [call[0].module for call in calls]
    assert [call[1]["layer_number"] for call in direct_calls] == [4, 5, 6, 7]
    assert [call[1]["layer_number"] for call in calls] == [4, 5, 6, 7]
    assert all(call[1]["config"] is config for call in calls)
    assert legacy_stack.mamba_state_shapes_per_request() == (("conv", 4), ("ssm", 4))


def test_legacy_stack_preserves_exact_family_specific_builder_kwargs(monkeypatch):
    calls = _patch_builder(monkeypatch)
    config = _config(7)
    symbols = ["M", "*", "D", "+", "-", "E", "G"]

    stack = _build_stack(config, layer_type_list=symbols, pp_layer_offset=4)

    common = {"config", "layer_number", "pg_collection"}
    expected_keys = [
        common | {"pp_layer_offset", "name"},
        common | {"is_mtp_layer", "add_layer_offset", "pp_layer_offset", "name"},
        common | {"is_mtp_layer", "add_layer_offset", "pp_layer_offset", "name"},
        common | {"is_mtp_layer", "add_layer_offset", "pp_layer_offset"},
        common | {"add_layer_offset", "name"},
        common | {"add_layer_offset", "name"},
        common | {"add_layer_offset", "name"},
    ]

    assert stack.layer_type_list is symbols
    assert [set(kwargs) for _, kwargs in calls] == expected_keys
    assert all(kwargs["config"] is config for _, kwargs in calls)
    assert "name" not in calls[3][1]


def test_legacy_stack_still_accepts_a_layer_class(monkeypatch):
    calls = _patch_builder(monkeypatch)
    config = _config(1)
    submodules = deepcopy(hybrid_stack_spec.submodules)
    submodules.mamba_layer = _BuiltLayer

    stack = HybridStack(
        config=config,
        submodules=submodules,
        layer_type_list=["M"],
        post_layer_norm=False,
        post_process=False,
        pg_collection=SimpleNamespace(pp=None, tp=None),
    )

    assert stack.layer_type_list == ["M"]
    assert calls[0][0] is _BuiltLayer
    assert not hasattr(stack, "layer_specs")


def test_new_layer_specs_parameter_does_not_shift_legacy_positional_arguments(monkeypatch):
    calls = _patch_builder(monkeypatch)
    config = _config(1)
    pg_collection = SimpleNamespace(pp=None, tp=None)

    stack = HybridStack(
        config,
        hybrid_stack_spec.submodules,
        False,
        ["M"],
        9,
        False,
        False,
        None,
        None,
        pg_collection,
        False,
        "decoder",
    )

    assert stack.layer_type_list == ["M"]
    assert calls[0][1]["layer_number"] == 10
    assert calls[0][1]["name"] == "decoder.layers.0"
