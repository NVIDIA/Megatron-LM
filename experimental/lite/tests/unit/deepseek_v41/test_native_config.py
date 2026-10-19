# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Configuration and unsupported candidate/CP combinations fail without allocation."""
import pytest

from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
from megatron.lite.model.deepseek_v41.lite.components import attention
from megatron.lite.model.deepseek_v41.lite.components.attention import DS41SelfAttention
from megatron.lite.model.deepseek_v41.lite.components.config import core_config, validate_runtime_cp
from megatron.lite.primitive.modules.attention import cross_layer as compressor
from megatron.lite.primitive.modules.attention import ratio_one
from megatron.lite.runtime.contracts import ParallelConfig


@pytest.mark.parametrize("deterministic", [False, True])
def test_native_core_configuration_is_bf16_and_keeps_literal_ratios(deterministic):

    config = core_config(
        DeepseekV41Config(num_hidden_layers=4),
        ParallelConfig(cp=2, ep=4),
        deterministic_mode=deterministic,
    )
    assert config.csa_compress_ratios == [0, 0, 2, 2]
    assert config.num_moe_experts == 384 and config.context_parallel_size == 1
    assert config.fp8 is None and config.fp4 is None
    assert config.deterministic_mode == deterministic and not config.moe_router_fusion
    assert config.moe_router_score_function == "sqrtsoftplus"
    assert config.moe_token_dispatcher_type == "alltoall"


@pytest.mark.parametrize(
    "overrides",
    [
        dict([pair])
        for pair in [
            ("recompute_granularity", "full"),
            ("recompute_granularity", "selective"),
            ("recompute_method", "uniform"),
            ("recompute_method", "block"),
        ]
    ],
)
def test_native_recompute_rejected_before_core_build(overrides):

    with pytest.raises(ValueError, match="must use KVSourceState.checkpoint"):
        core_config(DeepseekV41Config(num_hidden_layers=4), ParallelConfig(), **overrides)


def test_runtime_cp_is_explicit_and_checked_against_model_and_packed_params():
    from types import SimpleNamespace

    release = DeepseekV41Config(cp_size=2)
    group = SimpleNamespace(size=lambda: 2)
    params = SimpleNamespace(qkv_format="thd", cp_partition_mode="contiguous", cp_group=group)
    validate_runtime_cp(release, group, params)
    with pytest.raises(ValueError, match="configured CP size"):
        validate_runtime_cp(release, SimpleNamespace(size=lambda: 1))
    params.cp_group = object()
    with pytest.raises(ValueError, match="differs from model group"):
        validate_runtime_cp(release, group, params)


def test_native_cp_config_readers_are_unreachable_from_ds41_attention():
    import ast
    import inspect

    from megatron.core.models.common.embeddings import rotary_pos_embedding
    from megatron.core.transformer import multi_latent_attention
    from megatron.core.transformer.experimental_attention_variant import (
        csa,
        deepseek_v4_hybrid_attention,
    )

    assert multi_latent_attention.MultiLatentAttention not in DS41SelfAttention.__mro__
    expected = {("rotary_pos_embedding", 302), ("multi_latent_attention", 1014)}
    found = set()
    for module in (rotary_pos_embedding, multi_latent_attention, csa, deepseek_v4_hybrid_attention):
        for node in ast.walk(ast.parse(inspect.getsource(module))):
            if isinstance(node, ast.Attribute) and node.attr in {
                "context_parallel_size",
                "cp_partition_mode",
            }:
                value = ast.unparse(node.value)
                if value.endswith("config"):
                    found.add((module.__name__.rsplit(".", 1)[-1], node.lineno))
    assert found == expected
    for module in (attention, ratio_one, compressor):
        tree = ast.parse(inspect.getsource(module))
        assert not any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "get_rotary_seq_len"
            for n in ast.walk(tree)
        )
