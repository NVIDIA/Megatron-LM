# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the HybridModel Python layer-pattern DSL primitives.

These tests exercise pure-Python composition logic and do not require a
distributed initialization or a CUDA device.
"""

import dataclasses

import pytest

from megatron.core.models.hybrid.common_layer_config import CommonLayerConfig
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.models.hybrid.hybrid_model_config import CompiledRecipe, HybridModelConfig
from megatron.core.models.hybrid.layer_configs import (
    AttentionLayerConfig,
    CrossEntropyLayerConfig,
    DSALayerConfig,
    EmbeddingLayerConfig,
    GDNLayerConfig,
    LayerConfig,
    MambaLayerConfig,
    MLPLayerConfig,
    MoELayerConfig,
    MTPLayerConfig,
    PipelineSplit,
)
from megatron.core.models.hybrid.layer_pattern import flatten_decoder_pattern


def _make_common(**overrides) -> CommonLayerConfig:
    """Build a minimal valid CommonLayerConfig for unit tests."""
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
    }

    NON_SHARED_FIELDS = {
        "attention_dropout",
        "attention_softmax_in_fp32",
        "apply_query_key_layer_scaling",
        "add_qkv_bias",
        "masked_softmax_fusion",
        "use_fused_weighted_squared_relu",
        "apply_residual_connection_post_layernorm",
    }

    def test_common_fields_are_explicitly_audited_as_shared(self):
        common_fields = {f.name for f in dataclasses.fields(CommonLayerConfig) if f.name != "extra"}
        assert common_fields == set(self.COMMON_FIELD_LAYER_COVERAGE)
        for field_name, layer_families in self.COMMON_FIELD_LAYER_COVERAGE.items():
            assert len(layer_families) >= 2, field_name

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
        common = _make_common(mixed_precision_dtype="bf16")
        layer = MambaLayerConfig(common_config=common, head_dim=64, state_size=128, num_groups=8)
        tc = layer.to_transformer_config(num_layers=4)
        assert tc.bf16 is True
        assert tc.fp16 is False

    def test_invalid_mixed_precision_dtype_raises(self):
        common = _make_common(mixed_precision_dtype="nope")
        with pytest.raises(ValueError, match="mixed_precision_dtype"):
            common.to_transformer_config_kwargs()

    def test_invalid_activation_func_raises(self):
        common = _make_common(activation_func="not_a_real_activation")
        with pytest.raises(ValueError, match="activation"):
            common.to_transformer_config_kwargs()

    def test_attention_specific_fields_live_on_attention_layer_config(self):
        common = _make_common()
        layer = AttentionLayerConfig(
            common_config=common,
            num_attention_heads=4,
            attention_dropout=0.25,
            attention_softmax_in_fp32=False,
            apply_query_key_layer_scaling=False,
            add_qkv_bias=True,
            masked_softmax_fusion=True,
        )

        tc = layer.to_transformer_config(num_layers=2)

        assert tc.attention_dropout == 0.25
        assert tc.attention_softmax_in_fp32 is False
        assert tc.apply_query_key_layer_scaling is False
        assert tc.add_qkv_bias is True
        assert tc.masked_softmax_fusion is True

    def test_moe_specific_fields_live_on_moe_layer_config(self):
        common = _make_common(activation_func="squared_relu")
        layer = MoELayerConfig(
            common_config=common, num_experts=8, top_k=2, use_fused_weighted_squared_relu=True
        )

        tc = layer.to_transformer_config(num_layers=2)

        assert tc.use_fused_weighted_squared_relu is True


@pytest.mark.internal
class TestLayerConfigSymbols:
    """Each LayerConfig subclass binds the correct hybrid-allocation symbol."""

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
        # num_query_groups auto-derived to num_attention_heads when not set.
        assert tc.num_query_groups == 32
        # kv_channels auto-derived to hidden_size // num_attention_heads.
        assert tc.kv_channels == common.hidden_size // 32
        assert tc.num_layers == 4

    def test_moe_layer_config(self):
        common = _make_common()
        layer = MoELayerConfig(common_config=common, num_experts=8, top_k=2)
        tc = layer.to_transformer_config(num_layers=4)
        assert tc.num_moe_experts == 8
        assert tc.moe_router_topk == 2
        assert tc.num_layers == 4


@pytest.mark.internal
class TestFlatten:

    def test_flat_with_repeat_block(self):
        common = _make_common()
        m = MambaLayerConfig(common_config=common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        e = MoELayerConfig(common_config=common, num_experts=4, top_k=2)
        # [Mamba, [Att, MoE] * 3] → length 1 + 2*3 = 7
        result = flatten_decoder_pattern([m, [a, e] * 3])
        assert len(result) == 7
        assert result == [m, a, e, a, e, a, e]

    def test_tuples_treated_as_lists(self):
        common = _make_common()
        m = MambaLayerConfig(common_config=common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        result = flatten_decoder_pattern([m, (a, m)])
        assert result == [m, a, m]

    def test_non_layerconfig_leaf_rejected(self):
        common = _make_common()
        m = MambaLayerConfig(common_config=common)
        with pytest.raises(TypeError, match="path"):
            flatten_decoder_pattern([m, "M", m])

    def test_embedding_in_body_rejected(self):
        common = _make_common()
        m = MambaLayerConfig(common_config=common)
        emb = EmbeddingLayerConfig(common_config=common)
        with pytest.raises(TypeError, match="start/end"):
            flatten_decoder_pattern([m, emb, m])

    def test_loss_in_body_rejected(self):
        common = _make_common()
        m = MambaLayerConfig(common_config=common)
        loss = CrossEntropyLayerConfig()
        with pytest.raises(TypeError, match="start/end"):
            flatten_decoder_pattern([m, loss, m])

    def test_pipeline_split_in_body_rejected(self):
        common = _make_common()
        m = MambaLayerConfig(common_config=common)
        with pytest.raises(TypeError):
            flatten_decoder_pattern([m, PipelineSplit(), m])


@pytest.mark.internal
class TestHybridModelConfigCompile:

    def _embedding(self, common, **overrides):
        kwargs = dict(
            common_config=common,
            vocab_size=32000,
            max_sequence_length=2048,
            position_embedding_type="rope",
        )
        kwargs.update(overrides)
        return EmbeddingLayerConfig(**kwargs)

    def test_compile_returns_compiled_recipe(self):
        common = _make_common()
        emb = self._embedding(common)
        m = MambaLayerConfig(common_config=common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        loss = CrossEntropyLayerConfig()
        recipe = HybridModelConfig(common_config=common, layer_pattern=[emb, m, a, m, a, loss])
        compiled = recipe.compile()
        assert isinstance(compiled, CompiledRecipe)
        assert compiled.layer_type_list == ["M", "*", "M", "*"]
        assert len(compiled.layer_config_list) == 4
        assert compiled.vocab_size == 32000
        assert compiled.max_sequence_length == 2048
        assert compiled.position_embedding_type == "rope"
        # Default untie=False → share=True.
        assert compiled.share_embeddings_and_output_weights is True

    def test_compile_infers_uniform_attention_metadata(self):
        common = _make_common()
        emb = self._embedding(common)
        m = MambaLayerConfig(common_config=common)
        a = AttentionLayerConfig(
            common_config=common, num_attention_heads=8, num_query_groups=2, kv_channels=16
        )
        loss = CrossEntropyLayerConfig()
        recipe = HybridModelConfig(
            common_config=common, layer_pattern=[emb, m, a, m, loss], tensor_model_parallel_size=2
        )

        compiled = recipe.compile()

        assert compiled.config.num_attention_heads == 8
        assert compiled.config.num_query_groups == 2
        assert compiled.config.kv_channels == 16
        # The Mamba layer still lowers to TransformerConfig today, but the
        # recipe author does not have to pass attention fields through
        # CommonLayerConfig.extra just to satisfy TransformerConfig invariants.
        mamba_tc = compiled.layer_config_list[0]
        assert mamba_tc.num_attention_heads == 8
        assert mamba_tc.num_query_groups == 2
        assert mamba_tc.kv_channels == 16

    def test_untie_disables_sharing(self):
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        loss = CrossEntropyLayerConfig()
        recipe = HybridModelConfig(
            common_config=common,
            layer_pattern=[emb, a, loss],
            untie_embeddings_and_output_weights=True,
        )
        compiled = recipe.compile()
        assert compiled.share_embeddings_and_output_weights is False

    def test_missing_embedding_raises(self):
        common = _make_common()
        m = MambaLayerConfig(common_config=common)
        loss = CrossEntropyLayerConfig()
        recipe = HybridModelConfig(common_config=common, layer_pattern=[m, loss])
        with pytest.raises(TypeError, match="EmbeddingLayerConfig"):
            recipe.compile()

    def test_missing_loss_raises(self):
        common = _make_common()
        emb = self._embedding(common)
        m = MambaLayerConfig(common_config=common)
        recipe = HybridModelConfig(common_config=common, layer_pattern=[emb, m])
        with pytest.raises(TypeError, match="CrossEntropyLayerConfig"):
            recipe.compile()

    def test_no_decoder_layers_raises(self):
        common = _make_common()
        emb = self._embedding(common)
        loss = CrossEntropyLayerConfig()
        recipe = HybridModelConfig(common_config=common, layer_pattern=[emb, loss])
        with pytest.raises(ValueError, match="no decoder layers"):
            recipe.compile()

    def test_attention_dsa_mix_rejected(self):
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        d = DSALayerConfig(common_config=common, num_attention_heads=4)
        loss = CrossEntropyLayerConfig()
        recipe = HybridModelConfig(common_config=common, layer_pattern=[emb, a, d, loss])
        with pytest.raises(ValueError, match="Attention.*MLA/DSA|MLA/DSA.*Attention"):
            recipe.compile()


@pytest.mark.internal
class TestMTPLayerConfig:
    """Trailing MTPLayerConfig markers compile to (mtp_layer_pattern, depths)."""

    def _embedding(self, common):
        return EmbeddingLayerConfig(
            common_config=common,
            vocab_size=32000,
            max_sequence_length=2048,
            position_embedding_type="rope",
        )

    def test_no_mtp_markers_disables_mtp(self):
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        loss = CrossEntropyLayerConfig()
        compiled = HybridModelConfig(
            common_config=common, layer_pattern=[emb, a, a, loss]
        ).compile()
        assert compiled.mtp_layer_pattern is None
        assert compiled.mtp_num_depths == 0

    def test_two_mtp_markers_compile(self):
        common = _make_common()
        emb = self._embedding(common)
        m = MambaLayerConfig(common_config=common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        mtp_body = [m, m]
        mtp = MTPLayerConfig(common_config=common, mtp_model_layer=mtp_body)
        loss = CrossEntropyLayerConfig()
        compiled = HybridModelConfig(
            common_config=common, layer_pattern=[emb, a, a, mtp, mtp, loss]
        ).compile()
        # Decoder body excludes the trailing MTP markers.
        assert compiled.layer_type_list == ["*", "*"]
        assert compiled.mtp_layer_pattern == "MM"
        assert compiled.mtp_num_depths == 2

    def test_mtp_with_nested_body_flattens(self):
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        # MTP body LayerConfigs must use only default-shape overrides — the
        # MultiTokenPredictionBlock builds layers off the stack-level config.
        a_default = AttentionLayerConfig(common_config=common)
        m_default = MambaLayerConfig(common_config=common)
        mtp = MTPLayerConfig(
            common_config=common, mtp_model_layer=[a_default, [m_default, a_default]]
        )
        loss = CrossEntropyLayerConfig()
        compiled = HybridModelConfig(
            common_config=common, layer_pattern=[emb, a, mtp, loss]
        ).compile()
        assert compiled.mtp_layer_pattern == "*M*"
        assert compiled.mtp_num_depths == 1

    def test_mtp_in_decoder_body_rejected(self):
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        mtp = MTPLayerConfig(
            common_config=common, mtp_model_layer=[AttentionLayerConfig(common_config=common)]
        )
        loss = CrossEntropyLayerConfig()
        # MTP appearing BEFORE a regular decoder layer (not in the trailing
        # contiguous run) should raise.
        recipe = HybridModelConfig(common_config=common, layer_pattern=[emb, mtp, a, loss])
        with pytest.raises(TypeError, match="trailing"):
            recipe.compile()

    def test_inconsistent_mtp_bodies_rejected(self):
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        mtp1 = MTPLayerConfig(
            common_config=common, mtp_model_layer=[MambaLayerConfig(common_config=common)]
        )
        mtp2 = MTPLayerConfig(
            common_config=common, mtp_model_layer=[AttentionLayerConfig(common_config=common)]
        )
        loss = CrossEntropyLayerConfig()
        recipe = HybridModelConfig(common_config=common, layer_pattern=[emb, a, mtp1, mtp2, loss])
        with pytest.raises(ValueError, match="identical"):
            recipe.compile()

    def test_empty_mtp_body_rejected(self):
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        mtp = MTPLayerConfig(common_config=common, mtp_model_layer=[])
        loss = CrossEntropyLayerConfig()
        recipe = HybridModelConfig(common_config=common, layer_pattern=[emb, a, mtp, loss])
        with pytest.raises(ValueError, match="at least one"):
            recipe.compile()

    def test_mtp_body_layer_with_field_override_rejected(self):
        """Per-MTP-layer config overrides are not yet plumbed through; reject
        them so the user knows the override won't take effect."""
        common = _make_common()
        emb = self._embedding(common)
        a_decoder = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        # Non-default ``head_dim`` on the MTP body Mamba would be silently
        # dropped today — the rejection makes that loud.
        mtp = MTPLayerConfig(
            common_config=common,
            mtp_model_layer=[MambaLayerConfig(common_config=common, head_dim=128)],
        )
        loss = CrossEntropyLayerConfig()
        recipe = HybridModelConfig(common_config=common, layer_pattern=[emb, a_decoder, mtp, loss])
        with pytest.raises(ValueError, match="MambaLayerConfig.head_dim"):
            recipe.compile()

    def test_mtp_marker_with_custom_common_rejected(self):
        common = _make_common()
        other_common = _make_common(hidden_size=512)
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        mtp = MTPLayerConfig(
            common_config=other_common,
            mtp_model_layer=[MambaLayerConfig(common_config=other_common)],
        )
        loss = CrossEntropyLayerConfig()
        recipe = HybridModelConfig(common_config=common, layer_pattern=[emb, a, mtp, loss])
        with pytest.raises(ValueError, match="MTPLayerConfig.common_config"):
            recipe.compile()


@pytest.mark.internal
class TestExtraPassthrough:
    """Every config exposes an ``extra: dict`` escape hatch for any
    TransformerConfig field not in the curated DSL surface."""

    def _embedding(self, common, **kwargs):
        base = dict(
            common_config=common,
            vocab_size=32000,
            max_sequence_length=2048,
            position_embedding_type="rope",
        )
        base.update(kwargs)
        return EmbeddingLayerConfig(**base)

    def test_common_extra_propagates_to_layer_tc(self):
        # qk_layernorm is a real TransformerConfig field but isn't curated
        # on CommonLayerConfig — the passthrough should set it.
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

    def test_embedding_extra_propagates_to_stack_tc(self):
        common = _make_common()
        emb = self._embedding(common, extra={"qk_layernorm": True})
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        loss = CrossEntropyLayerConfig()
        compiled = HybridModelConfig(common_config=common, layer_pattern=[emb, a, loss]).compile()
        assert compiled.config.qk_layernorm is True

    def test_loss_extra_propagates_to_stack_tc(self):
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        loss = CrossEntropyLayerConfig(extra={"qk_layernorm": True})
        compiled = HybridModelConfig(common_config=common, layer_pattern=[emb, a, loss]).compile()
        assert compiled.config.qk_layernorm is True

    def test_embedding_extra_unknown_field_raises(self):
        common = _make_common()
        emb = self._embedding(common, extra={"made_up_field": 1})
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        loss = CrossEntropyLayerConfig()
        with pytest.raises(ValueError, match="EmbeddingLayerConfig.extra"):
            HybridModelConfig(common_config=common, layer_pattern=[emb, a, loss]).compile()

    def test_loss_extra_cannot_shadow_curated_field(self):
        """``CrossEntropyLayerConfig.extra`` may not name a curated field that
        the marker already exposes (e.g. ``loss_fusion``). Such an entry would
        silently override the marker's value otherwise."""
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        loss = CrossEntropyLayerConfig(loss_fusion=True, extra={"cross_entropy_loss_fusion": False})
        with pytest.raises(ValueError, match="cross_entropy_loss_fusion"):
            HybridModelConfig(common_config=common, layer_pattern=[emb, a, loss]).compile()


@pytest.mark.internal
class TestNumLayersDerived:
    """The recipe author never sets num_layers; compile derives it."""

    def test_num_layers_from_flattened_count(self):
        common = _make_common()
        # The recipe author never sets num_layers anywhere; CommonLayerConfig
        # does not even have a num_layers field.
        assert not hasattr(common, "num_layers")
        emb = EmbeddingLayerConfig(
            common_config=common,
            vocab_size=32000,
            max_sequence_length=2048,
            position_embedding_type="rope",
        )
        m = MambaLayerConfig(common_config=common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        loss = CrossEntropyLayerConfig()
        # 5 decoder layers.
        recipe = HybridModelConfig(common_config=common, layer_pattern=[emb, m, a, m, a, m, loss])
        compiled = recipe.compile()
        assert len(compiled.layer_config_list) == 5
        for tc in compiled.layer_config_list:
            assert tc.num_layers == 5
