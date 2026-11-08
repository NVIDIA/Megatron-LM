# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pattern and config-list models share the established layer FLOP estimates."""

import pytest
import torch

from megatron.core.flops import kda_layer_flops
from megatron.core.models.hybrid import MTPSplit, PipelineSplit
from megatron.core.models.hybrid.hybrid_layer_specs import gdp_stack_spec, hybrid_stack_spec
from megatron.core.models.hybrid.layers.flops import estimate_hybrid_config_list_flops
from megatron.core.models.hybrid.layers.utils import create_layer_config
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.training.models import GPTModelConfig, HybridModelConfig
from megatron.training.training import num_floating_point_operations


def _config(**overrides):
    options = dict(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        ffn_hidden_size=32,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        multi_latent_attention=False,
        q_lora_rank=4,
        kv_lora_rank=6,
        qk_head_dim=4,
        qk_pos_emb_head_dim=2,
        v_head_dim=4,
        mamba_state_dim=4,
        mamba_head_dim=8,
        mamba_num_groups=1,
        mamba_num_heads=6,
        gdp_num_householder=2,
        linear_key_head_dim=4,
        linear_value_head_dim=6,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=3,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=4,
        dsa_indexer_topk=4,
        csa_window_size=2,
        output_projection_groups=2,
        output_projection_lora_rank=4,
    )
    options.update(overrides)
    return MLATransformerConfig(**options)


def _pattern_flops(config, pattern, tokens, squared_lengths, spec=hybrid_stack_spec):
    model_config = HybridModelConfig(
        transformer=config, vocab_size=32, hybrid_layer_pattern=pattern, hybrid_stack_spec=spec
    )
    return num_floating_point_operations(
        model_config,
        2,
        sequence_length=16,
        total_real_tokens_in_batch=tokens,
        seqlen_squared_sum_in_batch=squared_lengths,
    )


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    "symbol,overrides,use_gdp",
    [
        ("-", {}, False),
        ("*", {"num_query_groups": 1}, False),
        ("M", {}, False),
        ("M", {}, True),
        ("G", {}, False),
        ("G", {"experimental_attention_variant": "gdn2"}, False),
        ("D", {"dsa_indexer_loss_coeff": 0.0}, False),
        ("D", {"dsa_indexer_loss_coeff": 0.1, "dsa_indexer_use_sparse_loss": True}, False),
        ("D", {"dsa_indexer_loss_coeff": 0.0, "dsa_indexer_kpool": 2}, False),
        ("C", {}, False),
        ("C", {"csa_dense_mode": True}, False),
        ("H", {}, False),
        ("W", {}, False),
        (
            "E",
            {
                "num_moe_experts": 4,
                "moe_router_topk": 2,
                "moe_ffn_hidden_size": 24,
                "moe_latent_size": 8,
                "moe_shared_expert_intermediate_size": 12,
            },
            False,
        ),
    ],
)
def test_homogeneous_lists_preserve_pattern_flops(symbol, overrides, use_gdp, packed):
    config = _config(**overrides)
    spec = gdp_stack_spec if use_gdp else hybrid_stack_spec
    layers = [create_layer_config(config, symbol) for _ in range(2)]
    tokens, squared_lengths = (10, 3**2 + 7**2) if packed else (32, 2 * 16**2)
    expected = _pattern_flops(config, symbol * 2, tokens, squared_lengths, spec)
    actual = estimate_hybrid_config_list_flops(layers, spec, config, 32, tokens, squared_lengths)
    # The public legacy counter rounds its final total down to an integer.
    assert actual == pytest.approx(expected, rel=0, abs=1)


@pytest.mark.parametrize("q_lora_rank", [None, 4])
def test_mla_and_mlp_list_matches_standard_transformer_formula(q_lora_rank):
    config = _config(num_layers=1, multi_latent_attention=True, q_lora_rank=q_lora_rank)
    layers = [create_layer_config(config, "+"), create_layer_config(config, "-")]
    model_config = GPTModelConfig(transformer=config, vocab_size=32)
    expected = num_floating_point_operations(
        model_config,
        2,
        sequence_length=16,
        total_real_tokens_in_batch=10,
        seqlen_squared_sum_in_batch=58,
    )
    actual = estimate_hybrid_config_list_flops(layers, hybrid_stack_spec, config, 32, 10, 58)
    assert actual == pytest.approx(expected, rel=0, abs=1)


@pytest.mark.parametrize("two_stage,forward_per_token", [(False, 2048), (True, 1920)])
def test_kda_gate_projection_shapes(two_stage, forward_per_token):
    # Direct Q/K/V/F/G plus beta versus QKV and two low-rank gates.
    assert (
        kda_layer_flops(10, 16, 4, 4, 2, 2, conv_kernel_dim=4, two_stage_gates=two_stage)
        == 10 * forward_per_token
    )


@pytest.mark.parametrize("shared_weights", [False, True])
def test_mixed_widths_and_mtp_sum_each_layers_shared_formula(shared_weights):
    root = _config(num_layers=3, mtp_num_layers=2, mtp_use_repeated_layer=shared_weights)
    narrow = _config(ffn_hidden_size=24)
    wide = _config(ffn_hidden_size=48)
    attention = _config(num_query_groups=1)
    narrow_layer = create_layer_config(narrow, "-")
    wide_layer = create_layer_config(wide, "-")
    attention_layer = create_layer_config(attention, "*")
    layers = [
        narrow_layer,
        PipelineSplit,
        attention_layer,
        wide_layer,
        MTPSplit,
        wide_layer,
        MTPSplit,
        wide_layer,
    ]
    tokens, squared_lengths = 10, 58
    # Each single-layer reference includes logits. Five layer executions need
    # only three logits heads; every MTP depth still executes its norms/projection.
    expected = _pattern_flops(narrow, "-", tokens, squared_lengths)
    expected += _pattern_flops(attention, "*", tokens, squared_lengths)
    expected += 3 * _pattern_flops(wide, "-", tokens, squared_lengths)
    expected -= 2 * (6 * tokens * 16 * 32)
    expected += 6 * tokens * 2 * (3 * 16 + 2 * 16**2)
    assert (
        estimate_hybrid_config_list_flops(
            layers, hybrid_stack_spec, root, 32, tokens, squared_lengths
        )
        == expected
    )
