# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for list-native Flextron model setup validation."""

from types import SimpleNamespace

import pytest

from megatron.core.models.hybrid import MTPSplit, PipelineSplit
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.ssm.mlp_layer_config import MLPLayerConfig
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
from megatron.elastification import flextron_utils
from megatron.elastification.flextron_utils import FlextronModelManager
from megatron.elastification.memory_config import MemoryConfig


def _root_config(num_layers=3):
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=16,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=4,
        ffn_hidden_size=32,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_shared_expert_intermediate_size=8,
        mamba_num_heads=4,
        mamba_head_dim=4,
        mamba_state_dim=8,
    )


def _supported_architecture(config):
    return (
        MambaLayerConfig.from_config(config),
        AttentionLayerConfig.from_config(config),
        MoELayerConfig.from_config(config),
    )


def _model(layer_config_list):
    return SimpleNamespace(
        layer_config_list=layer_config_list,
        vocab_size=32,
        share_embeddings_and_output_weights=False,
    )


@pytest.fixture(autouse=True)
def stub_runtime_config(monkeypatch):
    args = SimpleNamespace()
    monkeypatch.setattr(flextron_utils, "get_args", lambda: args)
    monkeypatch.setattr(flextron_utils, "load_memory_config", lambda _: MemoryConfig())


def test_manager_snapshots_model_layer_config_list_on_runtime_config():
    config = _root_config()
    source = list(_supported_architecture(config))

    manager = FlextronModelManager(_model(source), config)

    assert manager.layer_config_list == tuple(source)
    assert all(actual is expected for actual, expected in zip(manager.layer_config_list, source))
    assert config.flextron_layer_config_list is manager.layer_config_list
    assert not hasattr(config, "hybrid_layer_pattern")


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [("pipeline_model_parallel_size", 2), ("virtual_pipeline_model_parallel_size", 2)],
)
def test_manager_rejects_pipeline_parallelism(field_name, field_value):
    config = _root_config()
    setattr(config, field_name, field_value)

    with pytest.raises(NotImplementedError, match="pipeline model parallelism"):
        FlextronModelManager(_model(_supported_architecture(config)), config)


@pytest.mark.parametrize("marker", [PipelineSplit, MTPSplit])
def test_manager_rejects_architecture_markers(marker):
    config = _root_config(num_layers=2)
    architecture = (
        MambaLayerConfig.from_config(config),
        marker,
        MoELayerConfig.from_config(config),
    )

    with pytest.raises(NotImplementedError, match="does not support"):
        FlextronModelManager(_model(architecture), config)


def test_manager_rejects_unsupported_layer_config_type():
    config = _root_config(num_layers=1)

    with pytest.raises(NotImplementedError, match="supports only"):
        FlextronModelManager(_model((MLPLayerConfig.from_config(config),)), config)


@pytest.mark.parametrize(
    ("layer_config_type", "field_name", "different_value"),
    [
        (MambaLayerConfig, "hidden_size", 32),
        (MambaLayerConfig, "mamba_num_heads", 8),
        (MambaLayerConfig, "mamba_head_dim", 8),
        (MambaLayerConfig, "mamba_state_dim", 16),
        (MambaLayerConfig, "mamba_num_groups", 4),
        (AttentionLayerConfig, "num_attention_heads", 8),
        (AttentionLayerConfig, "num_query_groups", 1),
        (AttentionLayerConfig, "kv_channels", 8),
        (MoELayerConfig, "ffn_hidden_size", 64),
        (MoELayerConfig, "moe_ffn_hidden_size", 64),
        (MoELayerConfig, "num_moe_experts", 8),
        (MoELayerConfig, "moe_shared_expert_intermediate_size", 16),
        (MoELayerConfig, "moe_router_topk", 1),
    ],
)
def test_manager_rejects_heterogeneous_structural_sizing(
    layer_config_type, field_name, different_value
):
    config = _root_config(num_layers=1)
    layer_config = layer_config_type.from_config(config)
    setattr(layer_config, field_name, different_value)

    with pytest.raises(NotImplementedError, match=field_name):
        FlextronModelManager(_model((layer_config,)), config)


def test_manager_rejects_distinct_root_moe_ffn_hidden_size():
    config = _root_config(num_layers=1)
    config.moe_ffn_hidden_size = config.ffn_hidden_size // 2
    layer_config = MoELayerConfig.from_config(config)

    with pytest.raises(NotImplementedError, match="requires moe_ffn_hidden_size to equal"):
        FlextronModelManager(_model((layer_config,)), config)


def test_manager_rejects_layer_count_mismatch():
    config = _root_config(num_layers=2)
    architecture = (MambaLayerConfig.from_config(config),)

    with pytest.raises(ValueError, match="1 layer configs.*num_layers=2"):
        FlextronModelManager(_model(architecture), config)
