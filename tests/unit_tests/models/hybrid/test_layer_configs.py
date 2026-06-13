# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for HybridModel recipe layer config primitives."""

import dataclasses

import pytest
import torch

from megatron.core.models.hybrid.common_layer_config import CommonLayerConfig, validate_extra_kwargs
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.models.hybrid.layer_configs import (
    AttentionLayerConfig,
    DSALayerConfig,
    GDNLayerConfig,
    MambaLayerConfig,
    MLPLayerConfig,
    MoELayerConfig,
)
from megatron.core.transformer import MLATransformerConfig


def _make_common(**overrides) -> CommonLayerConfig:
    base = dict(hidden_size=256, use_cpu_initialization=True)
    base.update(overrides)
    return CommonLayerConfig(**base)


@pytest.mark.internal
class TestCommonLayerConfig:

    COMMON_FIELD_LAYER_COVERAGE = {
        "hidden_size": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "ffn_hidden_size": ("mlp", "moe"),
        "mixed_precision_dtype": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "params_dtype": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "first_last_layers_bf16": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "sequence_parallel": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "init_method_std": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "perform_initialization": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "use_cpu_initialization": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "normalization": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "layernorm_epsilon": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "layernorm_zero_centered_gamma": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "add_bias_linear": ("attention", "dsa", "mlp", "moe"),
        "gated_linear_unit": ("mlp", "moe"),
        "activation_func": ("gdn", "mlp", "moe"),
        "hidden_dropout": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "fp32_residual_connection": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "apply_rope_fusion": ("attention", "dsa"),
        "persist_layer_norm": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "transformer_impl": ("attention", "mlp", "moe"),
        "cuda_graph_impl": ("mamba", "attention", "mlp", "moe"),
        "cuda_graph_scope": ("mamba", "attention", "mlp", "moe"),
        "cuda_graph_warmup_steps": ("mamba", "attention", "mlp", "moe"),
        "apply_residual_connection_post_layernorm": (
            "mamba",
            "attention",
            "dsa",
            "gdn",
            "mlp",
            "moe",
        ),
        "fp8": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "fp8_recipe": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "fp8_param": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "fp8_margin": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "fp8_amax_history_len": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "fp8_dot_product_attention": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "fp4": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "fp4_recipe": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "fp4_param": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "recompute_granularity": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "recompute_method": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "recompute_num_layers": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
        "distribute_saved_activations": ("mamba", "attention", "dsa", "gdn", "mlp", "moe"),
    }

    NON_SHARED_FIELDS = {
        "attention_dropout",
        "attention_softmax_in_fp32",
        "apply_query_key_layer_scaling",
        "add_qkv_bias",
        "masked_softmax_fusion",
        "use_fused_weighted_squared_relu",
    }

    def test_common_fields_are_explicitly_audited_as_shared(self):
        common_fields = {f.name for f in dataclasses.fields(CommonLayerConfig) if f.name != "extra"}
        assert common_fields == set(self.COMMON_FIELD_LAYER_COVERAGE)

    def test_non_shared_fields_are_not_common_fields(self):
        common_fields = {f.name for f in dataclasses.fields(CommonLayerConfig)}
        assert common_fields.isdisjoint(self.NON_SHARED_FIELDS)

    def test_update_returns_new_instance(self):
        common = _make_common()
        updated = common.update(hidden_size=512)
        assert updated is not common
        assert common.hidden_size == 256
        assert updated.hidden_size == 512

    def test_mixed_precision_dtype_bf16_maps_to_bf16_true(self):
        common = _make_common(mixed_precision_dtype=torch.bfloat16)
        layer = MambaLayerConfig(common_config=common, head_dim=64, state_size=128, num_groups=8)
        tc = layer.to_transformer_config(num_layers=4)
        assert tc.bf16 is True
        assert tc.fp16 is False
        assert tc.params_dtype == torch.bfloat16

    def test_params_dtype_can_explicitly_differ_from_mixed_precision_dtype(self):
        common = _make_common(mixed_precision_dtype=torch.bfloat16, params_dtype=torch.float32)
        layer = MambaLayerConfig(common_config=common, head_dim=64, state_size=128, num_groups=8)
        tc = layer.to_transformer_config(num_layers=4)
        assert tc.bf16 is True
        assert tc.params_dtype == torch.float32

    def test_invalid_mixed_precision_dtype_raises(self):
        common = _make_common(mixed_precision_dtype="nope")
        with pytest.raises(TypeError, match="mixed_precision_dtype"):
            common.to_transformer_config_kwargs()

    def test_unsupported_mixed_precision_torch_dtype_raises(self):
        common = _make_common(mixed_precision_dtype=torch.float64)
        with pytest.raises(ValueError, match="mixed_precision_dtype"):
            common.to_transformer_config_kwargs()

    def test_invalid_params_dtype_raises(self):
        common = _make_common(params_dtype="bf16")
        with pytest.raises(TypeError, match="params_dtype"):
            common.to_transformer_config_kwargs()

    def test_invalid_activation_func_raises(self):
        common = _make_common(activation_func="not_a_real_activation")
        with pytest.raises(ValueError, match="activation"):
            common.to_transformer_config_kwargs()

    def test_fp8_cluster_propagates(self):
        common = _make_common(
            fp8="hybrid",
            fp8_recipe="delayed",
            fp8_param=True,
            fp8_margin=2,
            fp8_amax_history_len=1024,
            fp8_dot_product_attention=True,
        )
        layer = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        tc = layer.to_transformer_config(num_layers=2)
        assert tc.fp8 == "hybrid"
        assert tc.fp8_recipe == "delayed"
        assert tc.fp8_param is True
        assert tc.fp8_margin == 2
        assert tc.fp8_amax_history_len == 1024
        assert tc.fp8_dot_product_attention is True

    def test_recompute_cluster_propagates(self):
        common = _make_common(
            recompute_granularity="full", recompute_method="uniform", recompute_num_layers=4
        )
        layer = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        tc = layer.to_transformer_config(num_layers=8)
        assert tc.recompute_granularity == "full"
        assert tc.recompute_method == "uniform"
        assert tc.recompute_num_layers == 4


@pytest.mark.internal
class TestLayerConfigSymbols:

    def test_symbol_assignments(self):
        assert MambaLayerConfig.SYMBOL == Symbols.MAMBA
        assert GDNLayerConfig.SYMBOL == Symbols.GDN
        assert AttentionLayerConfig.SYMBOL == Symbols.ATTENTION
        assert DSALayerConfig.SYMBOL == Symbols.DS_ATTENTION
        assert MLPLayerConfig.SYMBOL == Symbols.MLP
        assert MoELayerConfig.SYMBOL == Symbols.MOE

    def test_symbols_are_valid(self):
        for cls in (
            MambaLayerConfig,
            GDNLayerConfig,
            AttentionLayerConfig,
            DSALayerConfig,
            MLPLayerConfig,
            MoELayerConfig,
        ):
            assert cls.SYMBOL in Symbols.VALID_LAYERS


@pytest.mark.internal
class TestLayerConfigToTransformerConfig:

    def test_mamba_layer_config(self):
        common = _make_common()
        layer = MambaLayerConfig(common_config=common, head_dim=64, state_size=128, num_groups=8)
        tc = layer.to_transformer_config(num_layers=4)
        assert tc.mamba_head_dim == 64
        assert tc.mamba_state_dim == 128
        assert tc.mamba_num_groups == 8
        assert tc.num_layers == 4

    def test_attention_layer_config(self):
        common = _make_common()
        layer = AttentionLayerConfig(common_config=common, num_attention_heads=32)
        tc = layer.to_transformer_config(num_layers=4)
        assert tc.num_attention_heads == 32
        assert tc.num_query_groups == 32
        assert tc.kv_channels == common.hidden_size // 32

    def test_moe_layer_config(self):
        common = _make_common()
        layer = MoELayerConfig(common_config=common, num_experts=8, top_k=2)
        tc = layer.to_transformer_config(num_layers=4)
        assert tc.num_moe_experts == 8
        assert tc.moe_router_topk == 2

    def test_moe_layer_config_accepts_multi_loss_router_fields(self):
        common = _make_common()
        layer = MoELayerConfig(
            common_config=common,
            num_experts=8,
            top_k=2,
            router_load_balancing_type=["aux_loss", "seq_aux_loss"],
            aux_loss_coeff=[0.1, 0.2],
        )
        tc = layer.to_transformer_config(num_layers=4)
        assert tc.moe_router_load_balancing_type == ["aux_loss", "seq_aux_loss"]
        assert tc.moe_aux_loss_coeff == [0.1, 0.2]

    def test_none_layer_field_does_not_clobber_common_ffn_hidden_size(self):
        common = _make_common(ffn_hidden_size=3072)
        layer = MLPLayerConfig(common_config=common)
        tc = layer.to_transformer_config(num_layers=4)
        assert tc.ffn_hidden_size == 3072

    def test_explicit_layer_field_overrides_common_field(self):
        common = _make_common(ffn_hidden_size=3072)
        layer = MLPLayerConfig(common_config=common, ffn_hidden_size=2048)
        tc = layer.to_transformer_config(num_layers=4)
        assert tc.ffn_hidden_size == 2048

    def test_none_layer_fields_do_not_clobber_common_extra_attention_fields(self):
        common = _make_common(
            extra={
                "attention_dropout": 0.25,
                "attention_softmax_in_fp32": True,
                "add_qkv_bias": True,
                "masked_softmax_fusion": False,
            }
        )
        layer = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        tc = layer.to_transformer_config(num_layers=4)
        assert tc.attention_dropout == 0.25
        assert tc.attention_softmax_in_fp32 is True
        assert tc.add_qkv_bias is True
        assert tc.masked_softmax_fusion is False

    def test_dsa_uses_mla_transformer_config(self):
        common = _make_common()
        layer = DSALayerConfig(common_config=common, num_attention_heads=4)
        tc = layer.to_transformer_config(num_layers=2)
        assert isinstance(tc, MLATransformerConfig)
        assert tc.multi_latent_attention is True
        assert tc.experimental_attention_variant == "dsa"

    def test_dsa_mla_knobs_curated(self):
        common = _make_common()
        layer = DSALayerConfig(
            common_config=common,
            num_attention_heads=128,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_head_dim=128,
            qk_pos_emb_head_dim=64,
            v_head_dim=128,
            rotary_scaling_factor=40.0,
            dsa_indexer_topk=32,
        )
        tc = layer.to_transformer_config(num_layers=2)
        assert tc.q_lora_rank == 1536
        assert tc.kv_lora_rank == 512
        assert tc.qk_head_dim == 128
        assert tc.qk_pos_emb_head_dim == 64
        assert tc.v_head_dim == 128
        assert tc.rotary_scaling_factor == 40.0
        assert tc.dsa_indexer_topk == 32


@pytest.mark.internal
class TestExtraPassthrough:

    def test_common_extra_propagates_to_layer_tc(self):
        common = _make_common(extra={"qk_layernorm": True})
        layer = MambaLayerConfig(common_config=common)
        tc = layer.to_transformer_config(num_layers=2)
        assert tc.qk_layernorm is True

    def test_layer_extra_overrides_common_extra(self):
        common = _make_common(extra={"qk_layernorm": False})
        layer = AttentionLayerConfig(
            common_config=common, num_attention_heads=4, extra={"qk_layernorm": True}
        )
        tc = layer.to_transformer_config(num_layers=2)
        assert tc.qk_layernorm is True

    def test_common_extra_unknown_field_raises(self):
        common = _make_common(extra={"definitely_not_a_real_tc_field": 7})
        with pytest.raises(ValueError, match="not TransformerConfig fields"):
            common.to_transformer_config_kwargs()

    def test_layer_extra_unknown_field_raises(self):
        common = _make_common()
        layer = MambaLayerConfig(common_config=common, extra={"another_typo_field": 1.0})
        with pytest.raises(ValueError, match="MambaLayerConfig.extra"):
            layer.to_transformer_config(num_layers=2)

    def test_extra_validation_can_target_mla_transformer_config(self):
        validate_extra_kwargs(
            {"mla_down_proj_fusion": True}, "DSALayerConfig.extra", MLATransformerConfig
        )

    def test_common_extra_cannot_shadow_curated_field(self):
        common = _make_common(extra={"hidden_size": 999})
        with pytest.raises(ValueError, match="hidden_size"):
            common.to_transformer_config_kwargs()

    def test_common_extra_cannot_shadow_curated_transformer_config_key(self):
        common = _make_common(mixed_precision_dtype=torch.bfloat16, extra={"bf16": False})
        with pytest.raises(ValueError, match="bf16"):
            common.to_transformer_config_kwargs()

    def test_layer_extra_cannot_shadow_curated_field(self):
        common = _make_common()
        layer = AttentionLayerConfig(
            common_config=common, num_attention_heads=4, extra={"num_attention_heads": 8}
        )
        with pytest.raises(ValueError, match="num_attention_heads"):
            layer.to_transformer_config(num_layers=2)

    def test_layer_extra_cannot_shadow_curated_none_field(self):
        common = _make_common()
        layer = MLPLayerConfig(common_config=common, extra={"ffn_hidden_size": 4096})
        with pytest.raises(ValueError, match="ffn_hidden_size"):
            layer.to_transformer_config(num_layers=2)

    def test_layer_extra_cannot_shadow_curated_transformer_config_key(self):
        common = _make_common()
        layer = MoELayerConfig(common_config=common, num_experts=8, extra={"num_moe_experts": 4})
        with pytest.raises(ValueError, match="num_moe_experts"):
            layer.to_transformer_config(num_layers=2)
