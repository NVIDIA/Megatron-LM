# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""FSDP unit selection without running model kernels."""

from types import SimpleNamespace

import pytest
from torch import nn

from megatron.core.distributed.fsdp.fsdp_unit_collector import FsdpUnitCollector
from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer
from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_layer import TransformerLayer


def _empty_module(module_type):
    """Build only the module hierarchy, without initializing distributed kernels."""
    module = module_type.__new__(module_type)
    nn.Module.__init__(module)
    if isinstance(module, TransformerLayer):
        module.is_moe_layer = False
    return module


@pytest.mark.parametrize(
    "unit_types, expected_names",
    [
        ((), [""]),
        ((nn.Linear,), ["block.0", "projection", ""]),
        ((nn.Sequential,), ["block", ""]),
        ((nn.Module,), ["block.0", "block.1", "block", "projection", ""]),
    ],
)
def test_custom_units_and_root_are_collected_once(unit_types, expected_names):
    """Configured units precede their owner, including when the root matches the policy."""
    root = nn.Module()
    root.block = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    root.projection = nn.Linear(4, 4)
    names = {module: name for name, module in root.named_modules()}
    collector = FsdpUnitCollector(unit_types)

    units = collector.collect(root)
    assert [names[unit] for unit in units] == expected_names
    assert collector.collect(root) == units


@pytest.mark.parametrize("expert_type", [nn.Sequential, TEGroupedMLP, SequentialMLP])
@pytest.mark.parametrize(
    "unit_types, expected_names",
    [
        ((), ["layer.mlp.experts", ""]),
        ((TransformerLayer,), ["layer.mlp.experts", "layer", ""]),
        ((MoELayer,), ["layer.mlp.experts", ""]),
        ((nn.Sequential,), ["layer.mlp.experts", ""]),
        ((nn.Linear,), ["layer.mlp.experts", ""]),
    ],
)
def test_routed_experts_are_explicit_units(expert_type, unit_types, expected_names):
    """Transformer layers select routed experts directly and own their remaining internals."""
    layer = _empty_module(TransformerLayer)
    layer.mlp = _empty_module(MoELayer)
    layer.is_moe_layer = True
    layer.mlp.experts = _empty_module(expert_type)
    layer.mlp.experts.projection = nn.Linear(4, 4)
    layer.mlp.shared = nn.Linear(4, 4)
    layer.mlp.auxiliary = nn.Sequential(nn.Linear(4, 4))
    root = nn.Module()
    root.layer = layer
    names = {module: name for name, module in root.named_modules()}
    collector = FsdpUnitCollector(unit_types)

    assert [names[unit] for unit in collector.collect(root)] == expected_names


@pytest.mark.parametrize("layer_type", [TransformerLayer, MambaLayer])
def test_shortcut_owns_helper_called_layers(layer_type):
    """Wrapper units keep helper-called layers and leave expert units independent."""
    experts = nn.Sequential(nn.Linear(4, 4))
    layer = _empty_module(layer_type)
    layer.projection = nn.Linear(4, 4)
    if isinstance(layer, TransformerLayer):
        layer.mlp = nn.Identity()
    moe_layer = _empty_module(TransformerLayer)
    moe_layer.mlp = _empty_module(MoELayer)
    moe_layer.is_moe_layer = True
    moe_layer.mlp.shared = nn.Linear(4, 4)
    moe_layer.mlp.experts = experts
    wrapper = _empty_module(ShortcutMoEBlock)
    wrapper.compute_layer = layer
    wrapper.moe_layer = moe_layer
    # A sibling layer outside the wrapper remains an independent unit.
    sibling = _empty_module(layer_type)
    sibling.projection = nn.Linear(4, 4)
    if isinstance(sibling, TransformerLayer):
        sibling.mlp = nn.Identity()
    root = nn.ModuleList([wrapper, sibling])
    collector = FsdpUnitCollector((ShortcutMoEBlock, TransformerLayer, MambaLayer))

    assert collector.collect(root) == [experts, wrapper, sibling, root]


def test_mhc_keeps_replayed_norms_and_routed_experts_independent():
    """mHC owns the inner layer while norms and routed experts retain their hooks."""
    layer = _empty_module(TransformerLayer)
    layer.input_layernorm = nn.LayerNorm(4)
    layer.pre_mlp_layernorm = nn.Identity()
    layer.mlp = _empty_module(MoELayer)
    layer.is_moe_layer = True
    layer.mlp.shared = nn.Linear(4, 4)
    layer.mlp.experts = nn.Sequential(nn.Linear(4, 4))
    wrapper = _empty_module(HyperConnectionHybridLayer)
    wrapper.inner_layer = layer
    wrapper.hyper_connection = nn.Linear(4, 4)
    wrapper.config = SimpleNamespace(recompute_granularity="selective", recompute_modules=["mhc"])
    collector = FsdpUnitCollector((HyperConnectionHybridLayer, TransformerLayer))

    assert collector.collect(wrapper) == [layer.input_layernorm, layer.mlp.experts, wrapper]


def test_mhc_owns_mamba_norm_and_mixer():
    """A wrapped Mamba has no routed experts or independently replayed Transformer norms."""
    layer = _empty_module(MambaLayer)
    layer.norm = nn.LayerNorm(4)
    layer.mixer = nn.Linear(4, 4)
    wrapper = _empty_module(HyperConnectionHybridLayer)
    wrapper.inner_layer = layer
    wrapper.hyper_connection = nn.Linear(4, 4)
    wrapper.config = SimpleNamespace(recompute_granularity="selective", recompute_modules=["mhc"])
    collector = FsdpUnitCollector((HyperConnectionHybridLayer, MambaLayer))

    assert collector.collect(wrapper) == [wrapper]
