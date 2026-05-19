# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.transformer.transformer_config import TransformerConfig


def _group_mlp_offload_config(**overrides):
    kwargs = {
        "num_layers": 1,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_moe_experts": 2,
        "moe_grouped_gemm": True,
        "use_transformer_engine_op_fuser": True,
        "fine_grained_activation_offloading": True,
        "offload_modules": ["group_mlp"],
    }
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_group_mlp_offload_config_is_valid_with_te_op_fuser():
    config = _group_mlp_offload_config()

    assert config.offload_modules == ["group_mlp"]


def test_group_mlp_offload_requires_moe():
    with pytest.raises(ValueError, match="group_mlp.*num_moe_experts"):
        _group_mlp_offload_config(num_moe_experts=None)


def test_group_mlp_offload_requires_grouped_gemm():
    with pytest.raises(ValueError, match="group_mlp.*moe_grouped_gemm"):
        _group_mlp_offload_config(moe_grouped_gemm=False)


def test_group_mlp_offload_requires_te_op_fuser():
    with pytest.raises(ValueError, match="group_mlp.*use_transformer_engine_op_fuser"):
        _group_mlp_offload_config(use_transformer_engine_op_fuser=False)


@pytest.mark.parametrize("module", ["expert_fc1", "moe_act"])
def test_group_mlp_offload_excludes_non_fused_moe_offload_groups(module):
    with pytest.raises(ValueError, match="group_mlp.*cannot.*combined"):
        _group_mlp_offload_config(offload_modules=["group_mlp", module])


def test_group_mlp_offload_conflicts_with_moe_recompute():
    with pytest.raises(AssertionError, match="group_mlp"):
        _group_mlp_offload_config(recompute_granularity="selective", recompute_modules=["moe"])
