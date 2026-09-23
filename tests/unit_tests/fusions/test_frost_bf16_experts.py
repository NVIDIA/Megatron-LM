# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import argparse

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import _add_network_size_args


def _config(**changes):
    values = dict(
        num_layers=1,
        hidden_size=256,
        ffn_hidden_size=256,
        moe_ffn_hidden_size=256,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_grouped_gemm=True,
        moe_bf16_expert_backend="frost",
        moe_mlp_glu_interleave_size=32,
        activation_func=F.silu,
        activation_func_clamp_value=10.0,
        glu_linear_offset=0.0,
        gated_linear_unit=True,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        gradient_accumulation_fusion=True,
        recompute_granularity="selective",
        recompute_modules=["moe_act"],
    )
    values.update(changes)
    return TransformerConfig(**values)


def test_frost_backend_uses_real_generated_cli():
    parser = _add_network_size_args(argparse.ArgumentParser())
    args = parser.parse_args(["--moe-bf16-expert-backend", "frost"])
    assert args.moe_bf16_expert_backend == "frost"
    assert parser.parse_args([]).moe_bf16_expert_backend == "transformer_engine"


def test_frost_config_opt_in_preserves_native_default():
    assert _config().moe_bf16_expert_backend == "frost"
    config = _config(moe_bf16_expert_backend="transformer_engine", moe_mlp_glu_interleave_size=None)
    assert config.moe_bf16_expert_backend == "transformer_engine"


def test_frost_allows_source_recipe_attention_offload_configuration():
    config = _config(
        fine_grained_activation_offloading=True, offload_modules=["core_attn", "attn_proj"]
    )
    assert config.fine_grained_activation_offloading
    assert config.offload_modules == ["core_attn", "attn_proj"]


@pytest.mark.parametrize(
    "changes",
    [
        dict(moe_bf16_expert_backend="cutedsl"),
        dict(moe_mlp_glu_interleave_size=None),
        dict(activation_func_clamp_value=None),
        dict(activation_func_clamp_value=0.0),
        dict(activation_func=F.relu),
        dict(glu_linear_offset=1.0),
        dict(add_bias_linear=True),
        dict(moe_use_grouped_tensor=True),
        dict(use_transformer_engine_op_fuser=True),
        dict(fp8="e4m3"),
        dict(gradient_accumulation_fusion=False),
        dict(delay_wgrad_compute=True),
        dict(moe_apply_probs_on_input=True, moe_router_topk=1),
        dict(fine_grained_activation_offloading=True, offload_modules=["expert_fc1"]),
        dict(fine_grained_activation_offloading=True, offload_modules=["moe_act"]),
        dict(fine_grained_activation_offloading=True, offload_modules=["fused_group_mlp"]),
    ],
)
def test_frost_rejects_unsupported_training_contracts(changes):
    with pytest.raises(ValueError, match="[Ff]rost|moe_bf16_expert_backend"):
        _config(**changes)
