# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Load-plan coverage for the Qwen3.5 HF WeightSpec."""

from __future__ import annotations

import torch
from megatron.lite.model.qwen3_5.config import Qwen35Config
from megatron.lite.model.qwen3_5.lite.checkpoint import Qwen35WeightSpec


def _config() -> Qwen35Config:
    return Qwen35Config(
        num_hidden_layers=2,
        layer_types=["full_attention", "linear_attention"],
        num_experts=2,
        num_experts_per_tok=1,
    )


def test_qwen35_load_plan_covers_both_attention_types_and_experts() -> None:
    weight_map = Qwen35WeightSpec(_config()).weight_map()

    assert weight_map["layers.0.full_attn.qkv.linear.weight"] == [
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.v_proj.weight",
    ]
    assert weight_map["layers.1.linear_attn.in_proj.linear.weight"] == [
        "model.language_model.layers.1.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.1.linear_attn.in_proj_z.weight",
        "model.language_model.layers.1.linear_attn.in_proj_b.weight",
        "model.language_model.layers.1.linear_attn.in_proj_a.weight",
    ]
    assert weight_map["layers.1.moe.experts.fc1.weight1"] == [
        "model.language_model.layers.1.mlp.experts.1.gate_proj.weight",
        "model.language_model.layers.1.mlp.experts.1.up_proj.weight",
    ]


def test_qwen35_packed_expert_source_is_selected_without_model_load_logic() -> None:
    spec = Qwen35WeightSpec(_config())
    native_name = "layers.0.moe.experts.fc1.weight1"
    canonical = "model.language_model.layers.0.mlp.experts.1.gate_proj.weight"
    packed = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    tensor = torch.arange(24).reshape(2, 4, 3)

    assert spec.hf_name_candidates(native_name, canonical) == [canonical, packed]
    assert torch.equal(
        spec.transform_hf_source(native_name, 0, packed, tensor), tensor[1, :2]
    )
    assert torch.equal(
        spec.transform_hf_source(native_name, 1, packed, tensor), tensor[1, 2:]
    )
