# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU tests: DeepSeek-V4.1 configuration validation and hybrid pattern helpers."""

import pytest
import torch

from megatron.core.models.deepseek_v41.layer_specs import (
    build_dsv41_hybrid_layer_pattern,
    dsv41_config_kwargs_from_model_layers,
)
from megatron.core.models.hybrid.hybrid_layer_allocation import (
    get_hybrid_total_layer_count,
    parse_hybrid_pattern,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig

V41_RATIOS = [0, 0] + [2] * 18 + [1] * 20

# Tiny CSA2 layout: 6 model layers, two window layers, one ratio-2 group, one ratio-1 group.
TINY_RATIOS = [0, 0, 2, 2, 1, 1]
TINY_KV = [2, 4]
TINY_INDEX = [2, 4]


def make_tiny_config(**overrides):
    kwargs = dict(
        hidden_size=32,
        num_attention_heads=4,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        add_bias_linear=False,
        q_lora_rank=16,
        qk_pos_emb_head_dim=8,
        v_head_dim=32,
        o_groups=2,
        o_lora_rank=16,
        rope_type="rope",
        multi_latent_attention=True,
        qk_layernorm=True,
        normalization="RMSNorm",
        experimental_attention_variant="dsv4_hybrid",
        dsv4_version="v4.1",
        enable_hyper_connections=True,
        num_residual_streams=4,
        csa_window_size=4,
        csa_compress_rotary_base=160000.0,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=16,
        dsa_indexer_topk=4,
        dsa_indexer_loss_coeff=0.0,
        csa2_kv_source_layers=TINY_KV,
        csa2_index_source_layers=TINY_INDEX,
        csa2_candidate_source_layer=4,
        csa2_candidate_topk_blocks=2,
        csa2_candidate_block_size=2,
        num_moe_experts=4,
        moe_ffn_hidden_size=32,
        moe_router_topk=2,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        activation_func_clamp_value=10.0,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    kwargs.update(dsv41_config_kwargs_from_model_layers(TINY_RATIOS))
    kwargs.update(overrides)
    return MLATransformerConfig(**kwargs)


class TestPatternBuilder:
    def test_released_pattern(self):
        pattern = build_dsv41_hybrid_layer_pattern(V41_RATIOS, pipeline_split_after_layers=[19])
        stages = pattern.split("|")
        assert len(stages) == 2
        assert stages[0] == "WEWE" + "DE" * 18
        assert stages[1] == "DE" * 20
        assert get_hybrid_total_layer_count(pattern) == 80
        parsed = parse_hybrid_pattern(pattern)
        assert parsed.mtp_pattern is None and parsed.main_pattern == pattern

    def test_no_split(self):
        pattern = build_dsv41_hybrid_layer_pattern(TINY_RATIOS)
        assert pattern == "WEWEDEDEDEDE"

    def test_bad_split(self):
        with pytest.raises(ValueError):
            build_dsv41_hybrid_layer_pattern(TINY_RATIOS, pipeline_split_after_layers=[5])

    def test_config_kwargs(self):
        kwargs = dsv41_config_kwargs_from_model_layers(TINY_RATIOS)
        assert kwargs["num_layers"] == 12
        assert kwargs["csa_compress_ratios"] == [0, 0, 0, 0, 2, 0, 2, 0, 1, 0, 1, 0]


class TestConfigValidation:
    def test_valid(self):
        config = make_tiny_config()
        assert config.dsv4_version == "v4.1"
        assert config.num_layers == 12
        # dsv4_hybrid derivation still applies
        assert config.qk_head_dim == config.v_head_dim - config.qk_pos_emb_head_dim

    def test_released_layout_valid(self):
        config = make_tiny_config(
            csa2_kv_source_layers=[2, 8, 14, 20],
            csa2_index_source_layers=[2, 8, 14, 20, 24, 28, 32, 36],
            csa2_candidate_source_layer=20,
            csa2_candidate_topk_blocks=2048,
            csa2_candidate_block_size=8,
            **dsv41_config_kwargs_from_model_layers(V41_RATIOS),
        )
        assert config.num_layers == 80

    def test_requires_dsv4_hybrid_variant(self):
        with pytest.raises(ValueError, match="dsv4_hybrid"):
            make_tiny_config(experimental_attention_variant=None)

    def test_requires_hyper_connections(self):
        with pytest.raises(ValueError, match="enable_hyper_connections"):
            make_tiny_config(enable_hyper_connections=False)

    def test_rejects_bad_kv_source(self):
        with pytest.raises(ValueError, match="window-only"):
            make_tiny_config(csa2_kv_source_layers=[0, 4], csa2_index_source_layers=[0, 4])

    def test_rejects_kv_source_missing_from_index_sources(self):
        with pytest.raises(ValueError, match="csa2_index_source_layers"):
            make_tiny_config(csa2_index_source_layers=[4])

    def test_rejects_candidate_not_last_kv_source(self):
        with pytest.raises(ValueError, match="last KV source"):
            make_tiny_config(csa2_candidate_source_layer=2)

    def test_rejects_mixed_ratio_sharing(self):
        kwargs = dsv41_config_kwargs_from_model_layers([0, 0, 2, 1, 1, 1])
        with pytest.raises(ValueError, match="same compress ratio"):
            make_tiny_config(**kwargs)

    def test_rejects_mtp(self):
        # Either the V4.1 check (ValueError) or the upstream ratio-length assertion fires first.
        with pytest.raises((ValueError, AssertionError)):
            make_tiny_config(mtp_num_layers=1)

    def test_m0_execution_contract(self):
        # bf16 full recompute is supported; the FP8/FP4 TE checkpoint path is not wired yet
        make_tiny_config(
            recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1
        )
        with pytest.raises(ValueError, match="qk_layernorm"):
            make_tiny_config(qk_layernorm=False)
        with pytest.raises(ValueError, match="RMSNorm"):
            make_tiny_config(normalization="LayerNorm")
        with pytest.raises(ValueError, match="indexer distillation"):
            make_tiny_config(dsa_indexer_loss_coeff=0.01)
        # Guards added after the 2026-09-12 review against the reference implementation
        with pytest.raises(ValueError, match="clamped SwiGLU"):
            make_tiny_config(activation_func_clamp_value=None)
        with pytest.raises(ValueError, match="attention_dropout=0"):
            make_tiny_config(attention_dropout=0.1)
        # layernorm_epsilon defaults to 1e-5, so the latent epsilon must differ to be rejected
        with pytest.raises(ValueError, match="attention_latent_norm_epsilon"):
            make_tiny_config(attention_latent_norm_epsilon=1e-6)
        with pytest.raises(ValueError, match="csa2_indexer_frozen"):
            make_tiny_config(csa2_indexer_frozen=False)
        with pytest.raises(ValueError, match="dsa_indexer_topk"):
            make_tiny_config(dsa_indexer_topk=None)
        with pytest.raises(ValueError, match="must be an integer"):
            make_tiny_config(csa2_kv_source_layers=[2.0, 4])
        with pytest.raises(ValueError, match="duplicates"):
            make_tiny_config(
                engram_layer_ids=[1, 1],
                engram_num_embeddings=[4096, 4096],
                engram_bucket_size=500,
                engram_n_heads=2,
                engram_head_dim=8,
                engram_compressed_vocab_size=100,
            )

    def test_v4_path_untouched(self):
        # dsv4_version defaults to v4: legacy ratio set is still enforced
        with pytest.raises(AssertionError, match="0, 4, or 128"):
            make_tiny_config(
                dsv4_version="v4",
                csa2_kv_source_layers=None,
                csa2_index_source_layers=None,
                csa2_candidate_source_layer=None,
                csa2_candidate_topk_blocks=0,
                csa2_candidate_block_size=0,
            )

    def test_engram_fields(self):
        config = make_tiny_config(
            engram_layer_ids=[1, 4],
            engram_num_embeddings=[4096, 4096],
            engram_max_ngram_size=3,
            engram_bucket_size=500,
            engram_n_heads=2,
            engram_head_dim=8,
            engram_compressed_vocab_size=100,
        )
        assert config.engram_frozen
        with pytest.raises(ValueError, match="one entry per"):
            make_tiny_config(
                engram_layer_ids=[1, 4],
                engram_num_embeddings=[4096],
                engram_bucket_size=500,
                engram_n_heads=2,
                engram_head_dim=8,
                engram_compressed_vocab_size=100,
            )
        with pytest.raises(ValueError, match="outside"):
            make_tiny_config(
                engram_layer_ids=[9],
                engram_num_embeddings=[4096],
                engram_bucket_size=500,
                engram_n_heads=2,
                engram_head_dim=8,
                engram_compressed_vocab_size=100,
            )


class TestPipelineSegmentValidation:
    def test_source_must_stay_with_consumers(self):
        from megatron.core.models.deepseek_v41.hybrid_stack import validate_dsv41_pipeline_segment

        config = make_tiny_config()
        # stage 1 of "WEWEDE|DEDEDE": model layers 3-5, layer 3 reads KV from layer 2
        with pytest.raises(ValueError, match="csa2_kv_source_layers"):
            validate_dsv41_pipeline_segment(config, list("DEDEDE"), pp_layer_offset=6)
        # "WEWEDEDE|DEDE": layers 0-3 | 4-5 keep every source with its consumers
        validate_dsv41_pipeline_segment(config, list("WEWEDEDE"), pp_layer_offset=0)
        validate_dsv41_pipeline_segment(config, list("DEDE"), pp_layer_offset=8)
