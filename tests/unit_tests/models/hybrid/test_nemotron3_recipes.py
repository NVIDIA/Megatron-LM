# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Structural tests for the architecture-only Nemotron 3 examples."""

from itertools import chain

from examples.nemotron3.nemotron_3_5_nano_30b_a3b import make_model_config as make_nano_model_config
from examples.nemotron3.nemotron_labs_3_puzzle_75b_a9b import (
    make_model_config as make_puzzle_model_config,
)
from megatron.core.models.hybrid.hybrid_architecture import (
    flatten_hybrid_layer_pattern,
    resolve_hybrid_architecture,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec

NANO_PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
PUZZLE_PATTERN = (
    "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*" "EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"
)
PUZZLE_MOE_CONFIGS = [
    (1280, 4),
    (1280, 8),
    (1280, 10),
    (1280, 8),
    (1280, 8),
    (1280, 8),
    (1280, 12),
    (1280, 8),
    (1280, 10),
    (1280, 8),
    (2688, 12),
    (1536, 14),
    (2688, 12),
    (1536, 12),
    (1536, 12),
    (2688, 12),
    (2688, 12),
    (2688, 12),
    (1536, 10),
    (2688, 12),
    (2688, 12),
    (1792, 12),
    (1792, 14),
    (1280, 10),
    (1280, 10),
    (1280, 12),
    (1280, 8),
    (1280, 12),
    (1280, 10),
    (1280, 8),
    (1280, 8),
    (1280, 10),
    (1280, 10),
    (1280, 10),
    (1280, 12),
    (1280, 12),
    (1280, 14),
    (1280, 16),
    (1792, 18),
    (2048, 18),
]


def _segments_and_layers(pattern):
    segments = flatten_hybrid_layer_pattern(pattern)
    return segments, tuple(chain.from_iterable(segments))


def _layer_symbols(layers) -> str:
    symbols = {"mamba": "M", "attention": "*", "moe": "E"}
    return "".join(symbols[layer.module_spec.metainfo["hybrid_layer_type"]] for layer in layers)


def _layers_of_type(layers, layer_type: str):
    return [
        layer for layer in layers if layer.module_spec.metainfo["hybrid_layer_type"] == layer_type
    ]


def _assert_transformer_values(config, expected):
    assert {name: getattr(config.transformer, name) for name in expected} == expected


def _resolve(config):
    return resolve_hybrid_architecture(
        config=config.transformer,
        hybrid_stack_spec=hybrid_stack_spec,
        layer_specs=config.layer_specs,
        mtp_layer_specs=config.mtp_layer_specs,
    )


def test_nano_direct_architecture_matches_published_order_and_pp_vpp_split():
    config = make_nano_model_config()
    segments, layers = _segments_and_layers(config.layer_specs)

    assert [len(segment) for segment in segments] == [13, 13, 13, 13]
    assert [sum(map(len, segments[:index])) for index in range(4)] == [0, 13, 26, 39]
    assert _layer_symbols(layers) == NANO_PATTERN
    assert len(_layers_of_type(layers, "mamba")) == 23
    assert len(_layers_of_type(layers, "moe")) == 23
    assert [index for index, symbol in enumerate(_layer_symbols(layers)) if symbol == "*"] == [
        5,
        12,
        19,
        26,
        33,
        42,
    ]

    assert config.hybrid_layer_pattern is None
    assert config.vocab_size == 131072
    assert config.seq_length == 262144
    assert config.transformer.virtual_pipeline_model_parallel_size is None
    _assert_transformer_values(
        config,
        {
            "pipeline_model_parallel_size": 2,
            "num_layers": 52,
            "hidden_size": 2688,
            "num_attention_heads": 32,
            "num_query_groups": 2,
            "kv_channels": 128,
            "ffn_hidden_size": 1856,
            "mamba_state_dim": 128,
            "mamba_head_dim": 64,
            "mamba_num_heads": 64,
            "mamba_num_groups": 8,
            "num_moe_experts": 128,
            "moe_ffn_hidden_size": 1856,
            "moe_router_topk": 6,
            "moe_router_num_groups": 1,
            "moe_router_group_topk": 1,
            "moe_shared_expert_intermediate_size": 3712,
        },
    )

    architecture = _resolve(config)
    assert [len(segment) for segment in architecture.segments] == [13, 13, 13, 13]
    assert config.transformer.virtual_pipeline_model_parallel_size == 2


def test_nano_reuses_uniform_family_configs_and_defines_repeated_mtp():
    config = make_nano_model_config()
    _, layers = _segments_and_layers(config.layer_specs)

    for layer_type in ("mamba", "attention", "moe"):
        family = _layers_of_type(layers, layer_type)
        assert len({id(layer.config) for layer in family}) == 1

    moe_configs = _layers_of_type(layers, "moe")
    assert {
        (layer.config.moe_ffn_hidden_size, layer.config.moe_router_topk) for layer in moe_configs
    } == {(1856, 6)}

    mtp_segments, mtp_layers = _segments_and_layers(config.mtp_layer_specs)
    assert [len(segment) for segment in mtp_segments] == [2]
    assert _layer_symbols(mtp_layers) == "*E"
    assert config.transformer.mtp_num_layers == 2
    assert config.transformer.mtp_use_repeated_layer is True


def test_puzzle_direct_architecture_matches_published_order_and_pp_vpp_split():
    config = make_puzzle_model_config()
    segments, layers = _segments_and_layers(config.layer_specs)

    assert [len(segment) for segment in segments] == [22, 22, 22, 22]
    assert [sum(map(len, segments[:index])) for index in range(4)] == [0, 22, 44, 66]
    assert _layer_symbols(layers) == PUZZLE_PATTERN
    assert len(_layers_of_type(layers, "mamba")) == 40
    assert len(_layers_of_type(layers, "moe")) == 40
    assert len(_layers_of_type(layers, "attention")) == 8

    assert config.hybrid_layer_pattern is None
    assert config.vocab_size == 131072
    assert config.seq_length == 262144
    assert config.transformer.virtual_pipeline_model_parallel_size is None
    _assert_transformer_values(
        config,
        {
            "pipeline_model_parallel_size": 2,
            "num_layers": 88,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_query_groups": 2,
            "kv_channels": 128,
            "ffn_hidden_size": 21504,
            "mamba_state_dim": 96,
            "mamba_head_dim": 64,
            "mamba_num_heads": 128,
            "mamba_num_groups": 8,
            "num_moe_experts": 512,
            "moe_router_num_groups": 1,
            "moe_router_group_topk": 1,
            "moe_shared_expert_intermediate_size": 5376,
            "moe_latent_size": 1024,
        },
    )

    architecture = _resolve(config)
    assert [len(segment) for segment in architecture.segments] == [22, 22, 22, 22]
    assert config.transformer.virtual_pipeline_model_parallel_size == 2


def test_puzzle_preserves_every_occurrence_specific_moe_config_and_mtp():
    config = make_puzzle_model_config()
    _, layers = _segments_and_layers(config.layer_specs)
    moe_layers = _layers_of_type(layers, "moe")

    assert [
        (layer.config.moe_ffn_hidden_size, layer.config.moe_router_topk) for layer in moe_layers
    ] == PUZZLE_MOE_CONFIGS
    assert len({id(layer.config) for layer in moe_layers}) == 40
    assert {layer.config.num_moe_experts for layer in moe_layers} == {512}

    mtp_segments, mtp_layers = _segments_and_layers(config.mtp_layer_specs)
    assert [len(segment) for segment in mtp_segments] == [2]
    assert _layer_symbols(mtp_layers) == "*E"
    assert mtp_layers[1].config.moe_ffn_hidden_size == 2688
    assert mtp_layers[1].config.moe_router_topk == 22
    assert config.transformer.mtp_num_layers == 1
    assert config.transformer.mtp_use_repeated_layer is False
