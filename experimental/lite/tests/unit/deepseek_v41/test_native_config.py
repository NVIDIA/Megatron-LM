# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Configuration and unsupported candidate/CP combinations fail without allocation."""
import pytest

from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
from megatron.lite.model.deepseek_v41.topology import validate_build


@pytest.mark.parametrize("cp", [1, 2])
def test_candidate_free_prefix(cp):
    config = DeepseekV41Config(num_hidden_layers=4)
    policies = validate_build(config, context_parallel_size=cp)
    assert [(p.kv, p.index) for p in policies] == [(None, None), (None, None), (2, 2), (2, 2)]
    assert not any(p.candidate for p in policies)


def test_candidate_cp_rejected_before_allocation():
    with pytest.raises(NotImplementedError, match="upstream candidate-column injection"):
        validate_build(DeepseekV41Config(), context_parallel_size=2)
    policies = validate_build(DeepseekV41Config(), context_parallel_size=1)
    assert policies[20].candidate and policies[-1].candidate
    assert policies[39].kv == 20 and policies[39].index == 36


def test_release_text_config_and_overrides():
    config = DeepseekV41Config._from_hf_dict(
        {
            "text_config": {
                "hidden_size": 5120,
                "kv_source_layer_ids": [2, 8],
                "rope_scaling": {"factor": 16, "original_max_position_embeddings": 65536},
            }
        },
        num_hidden_layers=4,
    )
    assert config.kv_source_layer_ids == (2, 8)
    assert config.num_hidden_layers == 4 and config.head_dim == 512
    assert config.rotary_scaling_factor == 16 and config.original_max_position_embeddings == 65536
    assert len(config.compress_ratios) == 43


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kv_source_layer_ids": (8,)},
        {"index_source_layer_ids": (2, 2)},
        {"compress_ratios": [0, 0, 3, 2]},
        {"compress_ratios": [0, 0]},
    ],
)
def test_invalid_ownership(kwargs):
    with pytest.raises(ValueError):
        validate_build(DeepseekV41Config(num_hidden_layers=4, **kwargs), context_parallel_size=1)


def test_native_core_configuration_is_bf16_and_keeps_literal_ratios():
    from megatron.lite.model.deepseek_v41.lite.native import core_config
    from megatron.lite.runtime.contracts import ParallelConfig

    config = core_config(DeepseekV41Config(num_hidden_layers=4), ParallelConfig(cp=2, ep=4))
    assert config.csa_compress_ratios == [0, 0, 2, 2]
    assert config.qk_head_dim + config.qk_pos_emb_head_dim == 512
    assert config.dsa_indexer_head_dim == 128 and config.dsa_indexer_n_heads == 32
    assert config.num_moe_experts == 384 and config.context_parallel_size == 2
    assert config.fp8 is None and config.fp4 is None
    assert not config.deterministic_mode and not config.moe_router_fusion
    assert config.moe_router_score_function == "sqrtsoftplus"
    assert config.moe_token_dispatcher_type == "alltoall"


@pytest.mark.parametrize(
    "overrides",
    [
        {"recompute_granularity": "full"},
        {"recompute_granularity": "selective"},
        {"recompute_method": "uniform"},
        {"recompute_method": "block"},
    ],
)
def test_native_recompute_rejected_before_core_build(overrides):
    from megatron.lite.model.deepseek_v41.lite.native import core_config
    from megatron.lite.runtime.contracts import ParallelConfig

    with pytest.raises(ValueError, match="must use KVSourceState.checkpoint"):
        core_config(DeepseekV41Config(num_hidden_layers=4), ParallelConfig(), **overrides)
