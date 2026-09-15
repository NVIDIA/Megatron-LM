# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Small reproducer for Puzzle's Python-defined heterogeneous config-list path."""

import sys
from argparse import ArgumentParser, Namespace
from unittest.mock import Mock, patch

import pytest

from megatron.core.models.hybrid import HybridLayerConfigListEntry, MTPSplit
from megatron.core.models.hybrid.hybrid_layer_allocation import parse_hybrid_layer_config_list
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.argument_utils import (
    hybrid_config_from_args,
    pretrain_cfg_container_from_args,
)
from megatron.training.arguments import parse_args, validate_args
from megatron.training.models.hybrid import HybridModelBuilder, HybridModelConfig


@pytest.fixture
def small_puzzle_args(monkeypatch):
    def add_model_args(parser: ArgumentParser) -> ArgumentParser:
        parser.set_defaults(
            is_hybrid_model=True,
            num_layers=4,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=4,
            group_query_attention=True,
            num_query_groups=2,
            mamba_num_heads=8,
            mamba_head_dim=32,
            mamba_state_dim=16,
            mamba_num_groups=1,
            num_experts=4,
            moe_ffn_hidden_size=64,
            moe_router_topk=1,
            moe_router_score_function="sigmoid",
            moe_latent_size=32,
            moe_shared_expert_intermediate_size=64,
            moe_shared_expert_overlap=False,
            normalization="RMSNorm",
            add_bias_linear=False,
            squared_relu=True,
            bias_gelu_fusion=False,
            position_embedding_type="none",
            mtp_num_layers=None,
            padded_vocab_size=128,
            tokenizer_type="NullTokenizer",
            seq_length=8,
            max_position_embeddings=8,
            micro_batch_size=1,
            train_iters=1,
            lr=1e-4,
        )
        return parser

    monkeypatch.setattr(sys, "argv", ["test_puzzle_config.py"])
    return validate_args(parse_args(extra_args_provider=add_model_args))


def _build_layer_configs(config: TransformerConfig) -> list[HybridLayerConfigListEntry]:
    small_moe = MoELayerConfig.from_config(config)
    large_moe = MoELayerConfig.from_config(config)
    large_moe.moe_ffn_hidden_size = 128
    large_moe.moe_router_topk = 2
    return [
        MambaLayerConfig.from_config(config),
        small_moe,
        AttentionLayerConfig.from_config(config),
        large_moe,
        MTPSplit,
        AttentionLayerConfig.from_config(config),
        MoELayerConfig.from_config(large_moe),
    ]


def _build_model_config(args: Namespace) -> HybridModelConfig:
    config = hybrid_config_from_args(args)
    config.hybrid_layer_config_list = _build_layer_configs(config.transformer)
    return config


def test_small_puzzle_layer_order_and_heterogeneous_dimensions(small_puzzle_args):
    config = _build_model_config(small_puzzle_args)
    parsed = parse_hybrid_layer_config_list(config.hybrid_layer_config_list, expected_num_layers=4)
    assert [type(layer) for layer in parsed.main_layer_config_list] == [
        MambaLayerConfig,
        MoELayerConfig,
        AttentionLayerConfig,
        MoELayerConfig,
    ]
    assert parsed.mtp_num_depths == 1
    assert [type(layer) for layer in parsed.mtp_layer_config_list] == [
        AttentionLayerConfig,
        MoELayerConfig,
    ]
    layers = [layer for layer in config.hybrid_layer_config_list if layer is not MTPSplit]
    assert all(layer.hidden_size == 128 and layer.is_hybrid_model for layer in layers)
    assert config.vocab_size == 128
    assert layers[0].mamba_state_dim == 16
    moe_layers = [layer for layer in layers if type(layer) is MoELayerConfig]
    assert [(layer.moe_ffn_hidden_size, layer.moe_router_topk) for layer in moe_layers] == [
        (64, 1),
        (128, 2),
        (128, 2),
    ]
    assert all(layer.num_moe_experts == 4 and layer.moe_latent_size == 32 for layer in moe_layers)
    assert all(layer.moe_shared_expert_intermediate_size == 64 for layer in moe_layers)


def test_small_puzzle_factory_builds_independent_configs(small_puzzle_args):
    source = hybrid_config_from_args(small_puzzle_args).transformer
    source_before = vars(source).copy()
    first = _build_layer_configs(source)
    second = _build_layer_configs(source)
    assert vars(source) == source_before
    assert first is not second
    for first_layer, second_layer in zip(first, second, strict=True):
        if first_layer is MTPSplit:
            assert second_layer is MTPSplit
        else:
            assert first_layer is not second_layer
            assert first_layer is not source
    assert len({id(layer) for layer in first if layer is not MTPSplit}) == 6

    first[3].moe_ffn_hidden_size = 32
    assert first[6].moe_ffn_hidden_size == 128
    assert second[3].moe_ffn_hidden_size == 128
    assert vars(source) == source_before


def test_small_puzzle_builder_infers_mtp_without_mutating_sources(small_puzzle_args):
    initialized_as_hybrid = []
    original_post_init = TransformerConfig.__post_init__

    def check_post_init(config):
        initialized_as_hybrid.append(config.is_hybrid_model)
        original_post_init(config)

    with patch.object(TransformerConfig, "__post_init__", check_post_init):
        config = _build_model_config(small_puzzle_args)
    assert initialized_as_hybrid and all(initialized_as_hybrid)
    assert config.mtp_num_layers is None
    assert small_puzzle_args.mtp_num_layers is None
    source_layers = config.hybrid_layer_config_list
    source_before = [vars(layer).copy() for layer in source_layers if layer is not MTPSplit]

    # Exercise the real builder and parser; module allocation is covered separately.
    with (
        patch("megatron.training.models.hybrid.get_args", return_value=small_puzzle_args),
        patch("megatron.training.models.hybrid.HybridModel") as model_class,
    ):
        HybridModelBuilder(config).build_model(Mock(), pre_process=True, post_process=True)

    assert config.mtp_num_layers == small_puzzle_args.mtp_num_layers == 1
    assert model_class.call_args.kwargs["hybrid_layer_config_list"] is source_layers
    assert model_class.call_args.kwargs["hybrid_layer_pattern"] is None
    assert [vars(layer) for layer in source_layers if layer is not MTPSplit] == source_before
    assert all(layer.mtp_num_layers is None for layer in source_layers if layer is not MTPSplit)
    assert small_puzzle_args.hybrid_layer_pattern is None
    assert not hasattr(small_puzzle_args, "hybrid_layer_config_list")

    training_config = pretrain_cfg_container_from_args(small_puzzle_args, config)
    assert training_config.model is config
    assert training_config.model.hybrid_layer_config_list is source_layers
    assert training_config.checkpoint.load_optim is True
    assert training_config.checkpoint.load_rng is True
