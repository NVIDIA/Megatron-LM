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
        # TransformerLayer-level wiring — applies wherever the per-layer
        # type is wrapped in a TransformerLayer (i.e. everywhere).
        "apply_residual_connection_post_layernorm": (
            "mamba",
            "attention",
            "dsa",
            "gdn",
            "mlp",
            "moe",
        ),
        # FP8/FP4/recompute clusters — TC-level concerns that affect every
        # layer family the recipe builds.
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

    def test_params_dtype_auto_derives_from_mixed_precision_when_unset(self):
        """Setting ``mixed_precision_dtype="bf16"`` without overriding
        ``params_dtype`` (still "fp32" by default) should produce a TC whose
        params live in bf16 — matching the legacy --bf16 CLI flag's
        behaviour."""
        import torch

        common = _make_common(mixed_precision_dtype="bf16")
        layer = MambaLayerConfig(common_config=common, head_dim=64, state_size=128, num_groups=8)
        tc = layer.to_transformer_config(num_layers=4)
        assert tc.params_dtype == torch.bfloat16

    def test_explicit_params_dtype_override_preserved(self):
        """An explicit ``params_dtype`` from the recipe author wins over the
        auto-derive. Useful when a recipe wants bf16 mixed precision but
        keeps params in fp32 (e.g. for fp32 master-weight workflows)."""
        import torch

        common = _make_common(mixed_precision_dtype="bf16", params_dtype="fp16")
        layer = MambaLayerConfig(common_config=common, head_dim=64, state_size=128, num_groups=8)
        tc = layer.to_transformer_config(num_layers=4)
        assert tc.params_dtype == torch.float16

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

    def test_fp8_cluster_propagates(self):
        """FP8 cluster on CommonLayerConfig flows through to per-layer TC."""
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

    def test_apply_residual_connection_post_layernorm_propagates(self):
        common = _make_common(apply_residual_connection_post_layernorm=True)
        layer = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        tc = layer.to_transformer_config(num_layers=2)
        assert tc.apply_residual_connection_post_layernorm is True

    def test_dsa_add_qkv_bias_curated(self):
        """``add_qkv_bias`` is curated on :class:`DSALayerConfig` (parity with
        :class:`AttentionLayerConfig`); it should not require ``extra``."""
        common = _make_common()
        layer = DSALayerConfig(common_config=common, num_attention_heads=4, add_qkv_bias=True)
        tc = layer.to_transformer_config(num_layers=2)
        assert tc.add_qkv_bias is True


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

    def test_common_config_auto_inherited_into_layers_without_explicit_common(self):
        """A recipe author who omits ``common_config=common`` on a layer
        should still get the recipe's common_config injected — otherwise the
        layer compiles with ``hidden_size=0`` and silently produces an
        invalid model."""
        common = _make_common()
        emb = EmbeddingLayerConfig(
            vocab_size=1024, max_sequence_length=512, position_embedding_type="rope"
        )
        m = MambaLayerConfig()  # No common_config — should auto-inherit.
        a = AttentionLayerConfig(num_attention_heads=4)  # Same.
        loss = CrossEntropyLayerConfig()
        compiled = HybridModelConfig(
            common_config=common, layer_pattern=[emb, m, a, loss]
        ).compile()
        # All per-layer TCs see the recipe's hidden_size, not 0.
        for tc in compiled.layer_config_list:
            assert tc.hidden_size == common.hidden_size

    def test_common_config_explicit_override_preserved(self):
        """An explicit non-default ``common_config`` on a layer wins over
        the auto-inherit path — auto-inherit only fills in defaults."""
        common = _make_common(hidden_size=256)
        other_common = _make_common(hidden_size=512)
        emb = self._embedding(common)
        # Layer carries an explicit, non-default common_config.
        m = MambaLayerConfig(common_config=other_common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        loss = CrossEntropyLayerConfig()
        compiled = HybridModelConfig(
            common_config=common, layer_pattern=[emb, m, a, loss]
        ).compile()
        # Mamba layer (idx 0) keeps its explicit hidden_size=512.
        assert compiled.layer_config_list[0].hidden_size == 512
        # Attention layer (idx 1) uses the recipe common's 256.
        assert compiled.layer_config_list[1].hidden_size == 256

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

    def test_stack_spec_default_none(self):
        """A recipe without an explicit ``stack_spec`` leaves
        :attr:`CompiledRecipe.stack_spec` ``None``; the builder is then free
        to auto-pick by ``transformer_impl``."""
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        loss = CrossEntropyLayerConfig()
        compiled = HybridModelConfig(common_config=common, layer_pattern=[emb, a, loss]).compile()
        assert compiled.stack_spec is None

    def test_stack_spec_propagates_through_compile(self):
        """An explicit ``stack_spec`` dotted path on the recipe flows through
        to :class:`CompiledRecipe`. Resolution happens later in the builder."""
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        loss = CrossEntropyLayerConfig()
        compiled = HybridModelConfig(
            common_config=common,
            layer_pattern=[emb, a, loss],
            stack_spec="my_pkg.my_module.my_stack_spec",
        ).compile()
        assert compiled.stack_spec == "my_pkg.my_module.my_stack_spec"

    def test_dsa_recipe_promotes_multi_latent_attention_to_stack(self):
        """DSA / MLA layers use their own decoupled RoPE.
        ``HybridModel.__init__`` skips global RoPE construction when
        ``self.config.multi_latent_attention`` (stack TC) is True. Compile
        must promote the flag whenever the decoder contains any DSA layer."""
        common = _make_common()
        emb = self._embedding(common)
        d = DSALayerConfig(common_config=common, num_attention_heads=4)
        loss = CrossEntropyLayerConfig()
        compiled = HybridModelConfig(common_config=common, layer_pattern=[emb, d, loss]).compile()
        assert compiled.config.multi_latent_attention is True

    def test_no_dsa_layers_keeps_multi_latent_attention_default(self):
        """A recipe without DSA must not flip the stack-level flag — that
        would silently disable global RoPE construction for ordinary
        Attention layers."""
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        loss = CrossEntropyLayerConfig()
        compiled = HybridModelConfig(common_config=common, layer_pattern=[emb, a, loss]).compile()
        assert compiled.config.multi_latent_attention is False

    def test_heterogeneous_attention_under_rope_rejected(self):
        """Two attention layers with different ``num_attention_heads`` under
        RoPE produce a single global rotary sized for a placeholder; the
        layer with the non-matching head count would hit shape mismatches
        at runtime. Reject at compile time until per-layer rotary lands."""
        common = _make_common()
        emb = self._embedding(common, position_embedding_type="rope")
        a1 = AttentionLayerConfig(common_config=common, num_attention_heads=8)
        a2 = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        loss = CrossEntropyLayerConfig()
        recipe = HybridModelConfig(common_config=common, layer_pattern=[emb, a1, a2, loss])
        with pytest.raises(NotImplementedError, match="[Hh]eterogeneous attention geometry"):
            recipe.compile()

    def test_heterogeneous_attention_under_none_position_embedding_allowed(self):
        """Without RoPE/YARN, no global rotary is built — heterogeneous
        attention geometry is fine."""
        common = _make_common()
        emb = self._embedding(common, position_embedding_type="none")
        a1 = AttentionLayerConfig(common_config=common, num_attention_heads=8)
        a2 = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        loss = CrossEntropyLayerConfig()
        compiled = HybridModelConfig(
            common_config=common, layer_pattern=[emb, a1, a2, loss]
        ).compile()
        assert compiled.layer_config_list[0].num_attention_heads == 8
        assert compiled.layer_config_list[1].num_attention_heads == 4


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

    def test_mtp_num_layers_set_on_stack_config(self):
        """Regression: ``HybridModel.forward`` gates ``process_mtp_loss`` on
        ``self.config.mtp_num_layers is not None``. If the recipe builds an
        MTP block but the stack TC has ``mtp_num_layers=None``, the auxiliary
        loss never aggregates and training silently ignores the MTP head."""
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        m = MambaLayerConfig(common_config=common)
        mtp = MTPLayerConfig(common_config=common, mtp_model_layer=[m])
        loss = CrossEntropyLayerConfig()
        compiled = HybridModelConfig(
            common_config=common, layer_pattern=[emb, a, mtp, mtp, loss]
        ).compile()
        assert compiled.mtp_num_depths == 2
        assert compiled.config.mtp_num_layers == compiled.mtp_num_depths

    def test_mtp_num_layers_none_when_no_mtp_markers(self):
        """No MTP markers → ``mtp_num_layers`` stays at the TC default (None),
        so the loss-aggregation gate stays closed."""
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        loss = CrossEntropyLayerConfig()
        compiled = HybridModelConfig(common_config=common, layer_pattern=[emb, a, loss]).compile()
        assert compiled.config.mtp_num_layers is None

    def test_mtp_loss_scaling_factor_propagates(self):
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        m = MambaLayerConfig(common_config=common)
        mtp = MTPLayerConfig(common_config=common, mtp_model_layer=[m], loss_scaling_factor=0.25)
        loss = CrossEntropyLayerConfig()
        compiled = HybridModelConfig(
            common_config=common, layer_pattern=[emb, a, mtp, loss]
        ).compile()
        assert compiled.config.mtp_loss_scaling_factor == 0.25

    def test_mtp_use_repeated_layer_propagates(self):
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        m = MambaLayerConfig(common_config=common)
        mtp = MTPLayerConfig(common_config=common, mtp_model_layer=[m], use_repeated_layer=True)
        loss = CrossEntropyLayerConfig()
        compiled = HybridModelConfig(
            common_config=common, layer_pattern=[emb, a, mtp, loss]
        ).compile()
        assert compiled.config.mtp_use_repeated_layer is True

    def test_mtp_stack_field_disagreement_rejected(self):
        """``loss_scaling_factor`` and ``use_repeated_layer`` are stack-level —
        if two MTP markers set them to different values, raise so the user
        knows only one value can take effect."""
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        m = MambaLayerConfig(common_config=common)
        mtp_a = MTPLayerConfig(common_config=common, mtp_model_layer=[m], loss_scaling_factor=0.1)
        mtp_b = MTPLayerConfig(common_config=common, mtp_model_layer=[m], loss_scaling_factor=0.5)
        loss = CrossEntropyLayerConfig()
        recipe = HybridModelConfig(common_config=common, layer_pattern=[emb, a, mtp_a, mtp_b, loss])
        with pytest.raises(ValueError, match="loss_scaling_factor"):
            recipe.compile()


@pytest.mark.internal
class TestHeterogeneousMoE:
    """Two distinct MoELayerConfig instances in one recipe must produce two
    per-layer TransformerConfigs whose MoE fields actually differ. This is
    the headline DSL capability the legacy single-character pattern can't
    express."""

    def _embedding(self, common):
        return EmbeddingLayerConfig(
            common_config=common,
            vocab_size=1024,
            max_sequence_length=512,
            position_embedding_type="rope",
        )

    def test_two_moe_configs_produce_distinct_tcs(self):
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        dense = MoELayerConfig(common_config=common, num_experts=8, top_k=4, ffn_hidden_size=256)
        sparse = MoELayerConfig(common_config=common, num_experts=4, top_k=2, ffn_hidden_size=128)
        loss = CrossEntropyLayerConfig()
        # Pattern: [emb, a, dense, a, sparse, loss] — 4 decoder layers, 2 MoE.
        compiled = HybridModelConfig(
            common_config=common,
            layer_pattern=[emb, a, dense, a, sparse, loss],
            tensor_model_parallel_size=2,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=1,
        ).compile()
        tcs = compiled.layer_config_list
        # Decoder indices: 0=att, 1=dense MoE, 2=att, 3=sparse MoE.
        dense_tc, sparse_tc = tcs[1], tcs[3]
        assert dense_tc.num_moe_experts == 8 and sparse_tc.num_moe_experts == 4
        assert dense_tc.moe_router_topk == 4 and sparse_tc.moe_router_topk == 2
        assert dense_tc.moe_ffn_hidden_size == 256 and sparse_tc.moe_ffn_hidden_size == 128
        # And the two are not the same Python object (no aliasing).
        assert dense_tc is not sparse_tc

    def test_two_moe_configs_share_global_ep(self):
        """EP is a model-wide topology fact — both MoE TCs must agree on it
        regardless of any per-layer-config differences."""
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        dense = MoELayerConfig(common_config=common, num_experts=8, top_k=2)
        sparse = MoELayerConfig(common_config=common, num_experts=4, top_k=2)
        loss = CrossEntropyLayerConfig()
        compiled = HybridModelConfig(
            common_config=common,
            layer_pattern=[emb, a, dense, a, sparse, loss],
            tensor_model_parallel_size=2,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=1,
        ).compile()
        dense_tc, sparse_tc = compiled.layer_config_list[1], compiled.layer_config_list[3]
        assert dense_tc.expert_model_parallel_size == 2
        assert sparse_tc.expert_model_parallel_size == 2

    def test_non_moe_tcs_carry_no_expert_parallelism(self):
        """The other half of the EP-only-on-MoE invariant: non-MoE TCs in a
        heterogeneous-MoE recipe default to EP=1 and ``num_moe_experts=None``,
        which is what the runtime layer construction actually consumes."""
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        dense = MoELayerConfig(common_config=common, num_experts=8, top_k=2)
        sparse = MoELayerConfig(common_config=common, num_experts=4, top_k=2)
        loss = CrossEntropyLayerConfig()
        compiled = HybridModelConfig(
            common_config=common,
            layer_pattern=[emb, a, dense, a, sparse, loss],
            tensor_model_parallel_size=2,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=1,
        ).compile()
        # Indices 0 and 2 are attention layers in the pattern above.
        att_tc_0, att_tc_2 = compiled.layer_config_list[0], compiled.layer_config_list[2]
        for tc in (att_tc_0, att_tc_2):
            assert tc.expert_model_parallel_size == 1
            assert tc.num_moe_experts is None

    def test_homogeneous_moe_stack_num_experts_matches_recipe(self):
        """Stack TC's ``num_moe_experts`` reflects the recipe's actual MoE
        configuration — not a magic placeholder. MTP block construction and
        inference capacity sizing both read this value semantically."""
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        moe = MoELayerConfig(common_config=common, num_experts=128, top_k=2)
        loss = CrossEntropyLayerConfig()
        compiled = HybridModelConfig(
            common_config=common,
            layer_pattern=[emb, a, moe, a, moe, loss],
            tensor_model_parallel_size=2,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=1,
        ).compile()
        assert compiled.config.num_moe_experts == 128

    def test_heterogeneous_moe_stack_num_experts_uses_max(self):
        """For heterogeneous MoE, the stack TC takes ``max(num_experts)`` —
        the "widest config" choice. MTP body MoE inherits the largest sane
        sizing; capacity buffers are sized for the worst case."""
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        dense = MoELayerConfig(common_config=common, num_experts=128, top_k=2)
        sparse = MoELayerConfig(common_config=common, num_experts=64, top_k=2)
        loss = CrossEntropyLayerConfig()
        compiled = HybridModelConfig(
            common_config=common,
            layer_pattern=[emb, a, dense, a, sparse, loss],
            tensor_model_parallel_size=2,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=1,
        ).compile()
        assert compiled.config.num_moe_experts == 128

    def test_stack_moe_ffn_hidden_size_derived_from_matching_layer(self):
        """The stack TC's ``moe_ffn_hidden_size`` matches the MoE layer
        that supplied ``num_moe_experts`` (rather than TC defaulting it
        from ``ffn_hidden_size``, which is the wrong size for MoE
        consumers)."""
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        # The "wide" MoE layer (most experts) supplies the stack values.
        wide = MoELayerConfig(common_config=common, num_experts=128, top_k=2, ffn_hidden_size=1024)
        narrow = MoELayerConfig(common_config=common, num_experts=64, top_k=2, ffn_hidden_size=512)
        loss = CrossEntropyLayerConfig()
        compiled = HybridModelConfig(
            common_config=common,
            layer_pattern=[emb, a, wide, a, narrow, loss],
            tensor_model_parallel_size=2,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=1,
        ).compile()
        assert compiled.config.num_moe_experts == 128
        assert compiled.config.moe_ffn_hidden_size == 1024

    def test_no_moe_no_ep_keeps_num_experts_none(self):
        """A dense recipe (no MoE, EP=1) must leave ``num_moe_experts``
        unset on the stack TC — TC defaults take over."""
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        loss = CrossEntropyLayerConfig()
        compiled = HybridModelConfig(common_config=common, layer_pattern=[emb, a, loss]).compile()
        assert compiled.config.num_moe_experts is None

    def test_ep_gt_1_without_moe_layers_rejected(self):
        """Setting ``expert_model_parallel_size > 1`` without any MoE layer
        is a recipe misconfiguration — surface it loudly rather than
        silently filling in a placeholder."""
        common = _make_common()
        emb = self._embedding(common)
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        loss = CrossEntropyLayerConfig()
        recipe = HybridModelConfig(
            common_config=common,
            layer_pattern=[emb, a, loss],
            tensor_model_parallel_size=2,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=1,
        )
        with pytest.raises(ValueError, match="requires at least one MoELayerConfig"):
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

    def test_layer_extra_cannot_shadow_parallelism(self):
        """Per-layer ``extra`` cannot override model-wide topology fields —
        process groups are global and a per-layer disagreement would be
        invalid sharding."""
        common = _make_common()
        a = AttentionLayerConfig(
            common_config=common,
            num_attention_heads=4,
            extra={"tensor_model_parallel_size": 8},  # Conflicts with TP=2 below.
        )
        emb = self._embedding(common)
        loss = CrossEntropyLayerConfig()
        recipe = HybridModelConfig(
            common_config=common, layer_pattern=[emb, a, loss], tensor_model_parallel_size=2
        )
        with pytest.raises(ValueError, match="tensor_model_parallel_size"):
            recipe.compile()

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
class TestYarnEmbedding:
    """``EmbeddingLayerConfig`` exposes curated YARN fields that
    :meth:`HybridModelConfig.compile` attaches to the stack-level
    :class:`TransformerConfig` when ``position_embedding_type == "yarn"``.
    This mirrors the existing setattr pattern in ``model_builder.py`` and
    matches the ``getattr`` lookups in :class:`HybridModel.__init__`."""

    def _attention_loss(self, common):
        a = AttentionLayerConfig(common_config=common, num_attention_heads=4)
        return a, CrossEntropyLayerConfig()

    def test_yarn_fields_attached_when_position_embedding_is_yarn(self):
        common = _make_common()
        emb = EmbeddingLayerConfig(
            common_config=common,
            vocab_size=1024,
            max_sequence_length=512,
            position_embedding_type="yarn",
            yarn_rotary_scaling_factor=32.0,
            yarn_original_max_position_embeddings=131072,
            yarn_beta_fast=32.0,
            yarn_beta_slow=1.0,
            yarn_mscale=1.0,
            yarn_mscale_all_dim=0.0,
            yarn_correction_range_round_to_int=True,
        )
        a, loss = self._attention_loss(common)
        compiled = HybridModelConfig(common_config=common, layer_pattern=[emb, a, loss]).compile()
        assert compiled.config.yarn_rotary_scaling_factor == 32.0
        assert compiled.config.yarn_original_max_position_embeddings == 131072
        assert compiled.config.yarn_beta_fast == 32.0
        assert compiled.config.yarn_beta_slow == 1.0
        assert compiled.config.yarn_mscale == 1.0
        assert compiled.config.yarn_mscale_all_dim == 0.0
        assert compiled.config.yarn_correction_range_round_to_int is True

    def test_yarn_fields_not_attached_when_position_embedding_is_rope(self):
        """When ``position_embedding_type != "yarn"``, yarn fields are silently
        unused — they're a no-op rather than an error so a recipe author can
        toggle the embedding type without removing the yarn block."""
        common = _make_common()
        emb = EmbeddingLayerConfig(
            common_config=common,
            vocab_size=1024,
            max_sequence_length=512,
            position_embedding_type="rope",
            yarn_rotary_scaling_factor=32.0,
        )
        a, loss = self._attention_loss(common)
        compiled = HybridModelConfig(common_config=common, layer_pattern=[emb, a, loss]).compile()
        assert not hasattr(compiled.config, "yarn_rotary_scaling_factor")

    def test_yarn_none_values_not_attached(self):
        """Default-None yarn fields are not stamped on the config — useful to
        avoid masking missing-required-attribute errors with stray None values
        when ``HybridModel.__init__`` calls ``getattr`` without a default."""
        common = _make_common()
        emb = EmbeddingLayerConfig(
            common_config=common,
            vocab_size=1024,
            max_sequence_length=512,
            position_embedding_type="yarn",
            yarn_rotary_scaling_factor=32.0,
            # other yarn fields left as default None
        )
        a, loss = self._attention_loss(common)
        compiled = HybridModelConfig(common_config=common, layer_pattern=[emb, a, loss]).compile()
        assert compiled.config.yarn_rotary_scaling_factor == 32.0
        assert not hasattr(compiled.config, "yarn_beta_fast")


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
