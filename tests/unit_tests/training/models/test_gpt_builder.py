# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import inspect
from unittest.mock import Mock, call, patch

import pytest
import torch

from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.heterogeneous.heterogeneous_config import (
    HeterogeneousTransformerConfig,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.models.gpt import (
    GPTModelBuilder,
    GPTModelConfig,
    default_layer_spec,
    mtp_block_spec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transformer(**kwargs):
    defaults = dict(num_layers=2, hidden_size=128, num_attention_heads=1)
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


def _make_gpt_config(**kwargs):
    defaults = dict(transformer=_make_transformer(), vocab_size=32000)
    defaults.update(kwargs)
    return GPTModelConfig(**defaults)


def _make_dispatch_config(**transformer_kwargs):
    """Build a Mock config for ``default_layer_spec`` dispatch tests.

    Defaults route execution to the ``local`` branch; tests flip individual
    flags to exercise the other branches.
    """
    config = Mock()
    config.restore_modelopt_state = False
    config.use_transformer_engine_op_fuser = False
    config.use_arbitrary_attention_mask = None

    transformer = Mock(spec=TransformerConfig)
    transformer.transformer_impl = "local"
    transformer.experimental_attention_variant = None
    transformer.num_moe_experts = None
    transformer.moe_grouped_gemm = False
    transformer.qk_layernorm = False
    transformer.qk_l2_norm = False
    transformer.multi_latent_attention = False
    transformer.normalization = "LayerNorm"
    transformer.use_kitchen = False
    transformer.use_kitchen_attention = False
    transformer.kitchen_attention_backend = None
    transformer.use_te_activation_func = False
    transformer.mla_down_proj_fusion = False
    for k, v in transformer_kwargs.items():
        setattr(transformer, k, v)
    config.transformer = transformer
    return config


# =============================================================================
# Section 1 — default_layer_spec
# =============================================================================


class TestDefaultLayerSpecDispatch:
    """Tests for ``default_layer_spec`` — branch dispatch on config flags."""

    @patch("megatron.training.models.gpt.get_gpt_modelopt_spec")
    @patch("megatron.core.parallel_state")
    def test_returns_modelopt_spec_when_restore_modelopt_state_true(self, mock_ps, mock_get_spec):
        mock_ps.get_context_parallel_world_size.return_value = 1
        spec = Mock(spec=ModuleSpec)
        mock_get_spec.return_value = spec
        config = _make_dispatch_config()
        config.restore_modelopt_state = True

        result = default_layer_spec(config, vp_stage=None)

        mock_get_spec.assert_called_once_with(
            config=config.transformer,
            local_core_attention=False,
            remap_te_layernorm=True,
            real_quant_cfg="None",
            use_arbitrary_attention_mask=True,
        )
        assert result is spec

    @patch("megatron.training.models.gpt.get_gpt_modelopt_spec")
    @patch("megatron.core.parallel_state")
    def test_modelopt_arbitrary_attention_mask_explicit_value(self, mock_ps, mock_get_spec):
        mock_ps.get_context_parallel_world_size.return_value = 4
        mock_get_spec.return_value = Mock(spec=ModuleSpec)
        config = _make_dispatch_config()
        config.restore_modelopt_state = True
        config.use_arbitrary_attention_mask = True

        default_layer_spec(config, vp_stage=None)

        assert mock_get_spec.call_args.kwargs["use_arbitrary_attention_mask"] is True

    @patch("megatron.training.models.gpt.get_gpt_modelopt_spec")
    @patch("megatron.core.parallel_state")
    def test_modelopt_arbitrary_attention_mask_derived_from_cp_world_size(
        self, mock_ps, mock_get_spec
    ):
        mock_ps.get_context_parallel_world_size.return_value = 4
        mock_get_spec.return_value = Mock(spec=ModuleSpec)
        config = _make_dispatch_config()
        config.restore_modelopt_state = True
        config.use_arbitrary_attention_mask = None

        default_layer_spec(config, vp_stage=None)

        # cp_world_size=4 → use_arbitrary_attention_mask=False
        assert mock_get_spec.call_args.kwargs["use_arbitrary_attention_mask"] is False

    @patch(
        "megatron.training.models.gpt.get_transformer_block_with_experimental_attention_variant_spec"
    )
    def test_returns_experimental_spec_when_experimental_attention_variant_set(self, mock_get_spec):
        spec = Mock(spec=ModuleSpec)
        mock_get_spec.return_value = spec
        config = _make_dispatch_config(experimental_attention_variant="variant_x")

        result = default_layer_spec(config, vp_stage=2)

        mock_get_spec.assert_called_once_with(config=config.transformer, vp_stage=2)
        assert result is spec

    @patch("megatron.training.models.gpt.get_gpt_decoder_block_spec")
    def test_returns_decoder_block_spec_when_num_moe_experts_set(self, mock_get_spec):
        spec = Mock(spec=ModuleSpec)
        mock_get_spec.return_value = spec
        config = _make_dispatch_config(num_moe_experts=4)

        # use_te flag depends on transformer_impl
        config.transformer.transformer_impl = "transformer_engine"

        result = default_layer_spec(config, vp_stage=1)

        mock_get_spec.assert_called_once_with(
            config.transformer,
            use_transformer_engine=True,
            normalization=config.transformer.normalization,
            qk_l2_norm=config.transformer.qk_l2_norm,
            vp_stage=1,
        )
        assert result is spec

    @patch("megatron.training.models.gpt.get_gpt_heterogeneous_layer_spec")
    def test_returns_heterogeneous_spec_when_heterogeneous_config(self, mock_get_spec):
        spec = Mock(spec=ModuleSpec)
        mock_get_spec.return_value = spec
        # Use spec= so isinstance(transformer, HeterogeneousTransformerConfig) is True
        transformer = Mock(spec=HeterogeneousTransformerConfig)
        transformer.transformer_impl = "transformer_engine"
        transformer.experimental_attention_variant = None
        transformer.num_moe_experts = None
        config = Mock()
        config.restore_modelopt_state = False
        config.transformer = transformer

        result = default_layer_spec(config, vp_stage=None)

        mock_get_spec.assert_called_once_with(transformer, True)
        assert result is spec

    @patch("megatron.training.models.gpt.get_gpt_layer_with_transformer_engine_spec")
    def test_returns_te_spec_when_use_te_true(self, mock_get_spec):
        mock_get_spec.__signature__ = inspect.Signature()
        spec = Mock(spec=ModuleSpec)
        mock_get_spec.return_value = spec
        config = _make_dispatch_config()
        config.transformer.transformer_impl = "transformer_engine"

        result = default_layer_spec(config, vp_stage=None)

        assert mock_get_spec.call_count == 1
        # use_te_op_fuser is not in (empty) signature, so should NOT appear in kwargs
        assert "use_te_op_fuser" not in mock_get_spec.call_args.kwargs
        assert result is spec

    @patch("megatron.training.models.gpt.get_gpt_layer_with_transformer_engine_spec")
    def test_te_spec_passes_use_te_op_fuser_when_supported(self, mock_get_spec):
        mock_get_spec.__signature__ = inspect.Signature(
            [inspect.Parameter("use_te_op_fuser", inspect.Parameter.KEYWORD_ONLY)]
        )
        mock_get_spec.return_value = Mock(spec=ModuleSpec)
        config = _make_dispatch_config()
        config.transformer.transformer_impl = "transformer_engine"
        config.use_transformer_engine_op_fuser = True

        default_layer_spec(config, vp_stage=None)

        assert mock_get_spec.call_args.kwargs.get("use_te_op_fuser") is True

    @patch("megatron.training.models.gpt.get_gpt_layer_with_inference_spec")
    def test_returns_inference_spec_when_inference_optimized(self, mock_get_spec):
        spec = Mock(spec=ModuleSpec)
        mock_get_spec.return_value = spec
        config = _make_dispatch_config()
        config.transformer.transformer_impl = "inference_optimized"

        result = default_layer_spec(config, vp_stage=None)

        mock_get_spec.assert_called_once_with(
            config.transformer.qk_layernorm,
            config.transformer.multi_latent_attention,
            qk_l2_norm=config.transformer.qk_l2_norm,
        )
        assert result is spec

    @patch("megatron.training.models.gpt.get_gpt_layer_local_spec")
    def test_returns_local_spec_as_default(self, mock_get_spec):
        spec = Mock(spec=ModuleSpec)
        mock_get_spec.return_value = spec
        config = _make_dispatch_config()

        result = default_layer_spec(config, vp_stage=None)

        mock_get_spec.assert_called_once_with(
            config.transformer.num_moe_experts,
            config.transformer.moe_grouped_gemm,
            config.transformer.qk_layernorm,
            config.transformer.multi_latent_attention,
            config.transformer.experimental_attention_variant,
            normalization=config.transformer.normalization,
            use_kitchen=config.transformer.use_kitchen,
            use_kitchen_attention=config.transformer.use_kitchen_attention,
            kitchen_attention_backend=config.transformer.kitchen_attention_backend,
        )
        assert result is spec


# =============================================================================
# Section 2 — GPTModelConfig
# =============================================================================


class TestGPTModelConfigInitialization:
    """Tests for GPTModelConfig field defaults and custom initialization."""

    def test_builder_classvar(self):
        assert GPTModelConfig.builder == "megatron.training.models.gpt.GPTModelBuilder"

    def test_default_values(self):
        config = GPTModelConfig(transformer=_make_transformer())
        assert config.transformer_layer_spec is None
        assert config.vocab_size is None
        assert config.make_vocab_size_divisible_by == 128
        assert config.should_pad_vocab is False
        assert config.seq_length == 1024
        assert config.fp16_lm_cross_entropy is False
        assert config.parallel_output is True
        assert config.share_embeddings_and_output_weights is False
        assert config.position_embedding_type == "learned_absolute"
        assert config.rotary_percent == 1.0
        assert config.rotary_base == 10000
        assert config.rope_scaling is False
        assert config.rope_scaling_factor == 8.0
        assert config.scatter_embedding_sequence_parallel is True
        assert config.seq_len_interpolation_factor is None
        assert config.tp_comm_overlap_cfg is None
        assert config.use_transformer_engine_op_fuser is False
        assert config.use_arbitrary_attention_mask is None

    def test_custom_initialization(self):
        config = GPTModelConfig(
            transformer=_make_transformer(),
            seq_length=4096,
            fp16_lm_cross_entropy=True,
            parallel_output=False,
            share_embeddings_and_output_weights=True,
            position_embedding_type="rope",
            rotary_base=500000,
            vocab_size=50000,
        )
        assert config.seq_length == 4096
        assert config.fp16_lm_cross_entropy is True
        assert config.parallel_output is False
        assert config.share_embeddings_and_output_weights is True
        assert config.position_embedding_type == "rope"
        assert config.rotary_base == 500000
        assert config.vocab_size == 50000


class TestGPTModelConfigGetAttr:
    """Tests for GPTModelConfig.__getattr__ — direct access vs. TransformerConfig proxy."""

    def setup_method(self):
        self.transformer = _make_transformer(hidden_size=256, num_layers=4)
        self.config = GPTModelConfig(transformer=self.transformer, vocab_size=32000)

    def test_own_attribute_not_proxied(self):
        assert self.config.vocab_size == 32000

    def test_proxies_transformer_attribute(self):
        assert self.config.hidden_size == 256

    def test_raises_attribute_error_for_unknown(self):
        with pytest.raises(AttributeError):
            _ = self.config.completely_unknown_attr_xyz

    def test_raises_before_transformer_init(self):
        del self.config.__dict__["transformer"]
        with pytest.raises(AttributeError):
            _ = self.config.hidden_size

    def test_error_message_contains_attr_name(self):
        attr_name = "completely_unknown_attr_xyz"
        with pytest.raises(AttributeError, match=attr_name):
            getattr(self.config, attr_name)


class TestGPTModelConfigSetAttr:
    """Tests for GPTModelConfig.__setattr__ — own-field writes vs. TransformerConfig proxy writes."""

    def setup_method(self):
        self.transformer = _make_transformer(hidden_size=256)
        self.config = GPTModelConfig(transformer=self.transformer, vocab_size=32000)

    def test_sets_own_attribute_on_self(self):
        self.config.vocab_size = 50000
        assert self.config.vocab_size == 50000
        assert self.config.__dict__.get("vocab_size") == 50000

    def test_proxies_set_to_transformer_attribute(self):
        self.config.hidden_size = 512
        assert self.transformer.hidden_size == 512

    def test_set_proxied_attr_reflects_on_transformer(self):
        self.config.hidden_size = 1024
        assert self.config.hidden_size == 1024
        assert self.transformer.hidden_size == 1024

    def test_set_before_transformer_init(self):
        del self.config.__dict__["transformer"]
        self.config.vocab_size = 42
        assert self.config.__dict__["vocab_size"] == 42

    def test_set_transformer_itself_stores_on_self(self):
        new_transformer = _make_transformer(hidden_size=512)
        self.config.transformer = new_transformer
        assert self.config.transformer is new_transformer
        assert self.config.__dict__["transformer"] is new_transformer

    def test_set_own_attr_does_not_go_to_transformer(self):
        self.config.vocab_size = 99999
        assert self.config.__dict__.get("vocab_size") == 99999
        assert not hasattr(self.config.transformer, "vocab_size")

    def test_proxied_write_does_not_shadow_on_self(self):
        self.config.hidden_size = 2048
        assert "hidden_size" not in self.config.__dict__


class TestGPTModelConfigFinalize:
    """Tests for GPTModelConfig.finalize() — validation logic."""

    def test_calls_transformer_finalize_when_present(self):
        config = _make_gpt_config()
        # TransformerConfig doesn't have finalize by default; attach a stub.
        config.transformer.finalize = Mock()
        config.finalize()
        config.transformer.finalize.assert_called_once()

    def test_no_finalize_attribute_is_ok(self):
        config = _make_gpt_config()
        # Should not raise even when transformer has no finalize()
        assert not hasattr(config.transformer, "finalize")
        config.finalize()

    def test_raises_when_cudagraph_without_te_rng_tracker(self):
        config = _make_gpt_config()
        config.transformer.cuda_graph_impl = "local"
        config.transformer.use_te_rng_tracker = False
        with pytest.raises(AssertionError, match="RNG tracker"):
            config.finalize()

    def test_no_error_when_cudagraph_with_te_rng_tracker(self):
        config = _make_gpt_config()
        config.transformer.cuda_graph_impl = "local"
        config.transformer.use_te_rng_tracker = True
        # Should not raise
        config.finalize()

    def test_vp_size_assertion_fails_on_indivisible_layers(self):
        # 6 layers, pp=2 → 3 per stage, vp=2 → 3 % 2 != 0 → AssertionError
        transformer = _make_transformer(
            num_layers=6,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
            pipeline_dtype=torch.bfloat16,
        )
        config = GPTModelConfig(transformer=transformer, vocab_size=32000)
        with pytest.raises(AssertionError, match="number of model chunks"):
            config.finalize()

    def test_vp_size_assertion_passes_on_divisible_layers(self):
        # 8 layers, pp=2 → 4 per stage, vp=2 → 4 % 2 == 0 → OK
        transformer = _make_transformer(
            num_layers=8,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
            pipeline_dtype=torch.bfloat16,
        )
        config = GPTModelConfig(transformer=transformer, vocab_size=32000)
        # Should not raise
        config.finalize()

    def test_vp_size_check_skipped_for_flexible_pp_layout(self):
        # Indivisible layers, but account_for_embedding_in_pipeline_split=True makes
        # the layout flexible, so the assertion is skipped.
        transformer = _make_transformer(
            num_layers=6,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
            account_for_embedding_in_pipeline_split=True,
            pipeline_dtype=torch.bfloat16,
        )
        config = GPTModelConfig(transformer=transformer, vocab_size=32000)
        # Should not raise
        config.finalize()


# =============================================================================
# Section 3 — GPTModelBuilder
# =============================================================================


class TestGPTModelBuilderInit:
    """Tests for GPTModelBuilder.__init__ — config storage."""

    def setup_method(self):
        self.config = _make_gpt_config()
        self.builder = GPTModelBuilder(self.config)

    def test_stores_model_config(self):
        assert self.builder._model_config is self.config


class TestGPTModelBuilderBuildModel:
    """Tests for GPTModelBuilder.build_model() — spec resolution, vocab padding,
    vp/pp-stage inference, attention-backend override, and MCoreGPTModel kwargs.
    """

    def setup_method(self):
        self.config = _make_gpt_config(vocab_size=32000)
        # Provide an explicit ModuleSpec so build_model never hits default_layer_spec.
        self._default_spec = ModuleSpec(module=object)
        self.config.__dict__["transformer_layer_spec"] = self._default_spec
        self.builder = GPTModelBuilder(self.config)
        self.pg = Mock()
        self.pg.pp = Mock()

    @patch("megatron.training.models.gpt.mtp_block_spec", return_value=None)
    @patch("megatron.training.models.gpt.calculate_padded_vocab_size")
    @patch("megatron.training.models.gpt.is_vp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_vp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.GPTModel")
    def test_raises_when_vocab_size_none(self, mock_model, *_):
        self.config.__dict__["vocab_size"] = None
        with pytest.raises(AssertionError, match="vocab_size"):
            self.builder.build_model(self.pg)

    @patch("megatron.training.models.gpt.mtp_block_spec", return_value=None)
    @patch("megatron.training.models.gpt.calculate_padded_vocab_size")
    @patch("megatron.training.models.gpt.is_vp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_vp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.GPTModel")
    def test_spec_already_module_spec_used_directly(self, mock_model, *_):
        module_spec = ModuleSpec(module=object)
        self.config.__dict__["transformer_layer_spec"] = module_spec
        self.builder.build_model(self.pg, pre_process=True, post_process=True)
        assert mock_model.call_args.kwargs["transformer_layer_spec"] is module_spec

    @patch("megatron.training.models.gpt.default_layer_spec")
    @patch("megatron.training.models.gpt.mtp_block_spec", return_value=None)
    @patch("megatron.training.models.gpt.calculate_padded_vocab_size")
    @patch("megatron.training.models.gpt.is_vp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_vp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.GPTModel")
    def test_spec_none_uses_default_layer_spec(self, mock_model, *patches):
        mock_default = patches[-1]
        default_spec = ModuleSpec(module=object)
        mock_default.return_value = default_spec
        self.config.__dict__["transformer_layer_spec"] = None

        self.builder.build_model(self.pg, pre_process=True, post_process=True, vp_stage=3)

        mock_default.assert_called_once_with(self.config, 3)
        assert mock_model.call_args.kwargs["transformer_layer_spec"] is default_spec

    @patch("megatron.training.models.gpt.mtp_block_spec", return_value=None)
    @patch("megatron.training.models.gpt.calculate_padded_vocab_size")
    @patch("megatron.training.models.gpt.is_vp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_vp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.GPTModel")
    def test_spec_callable_without_vp_stage_param_called_without_it(self, mock_model, *_):
        returned_spec = ModuleSpec(module=object)
        calls = []

        def no_vp_fn(config):
            calls.append(config)
            return returned_spec

        self.config.__dict__["transformer_layer_spec"] = no_vp_fn
        self.builder.build_model(self.pg, pre_process=True, post_process=True)
        assert calls == [self.config]
        assert mock_model.call_args.kwargs["transformer_layer_spec"] is returned_spec

    @patch("megatron.training.models.gpt.mtp_block_spec", return_value=None)
    @patch("megatron.training.models.gpt.calculate_padded_vocab_size")
    @patch("megatron.training.models.gpt.is_vp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_vp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.GPTModel")
    def test_spec_callable_with_vp_stage_param_called_with_it(self, mock_model, *_):
        returned_spec = ModuleSpec(module=object)
        received = []

        def vp_fn(config, vp_stage=None):
            received.append((config, vp_stage))
            return returned_spec

        self.config.__dict__["transformer_layer_spec"] = vp_fn
        self.builder.build_model(self.pg, pre_process=True, post_process=True, vp_stage=2)
        assert received == [(self.config, 2)]
        assert mock_model.call_args.kwargs["transformer_layer_spec"] is returned_spec

    @patch("megatron.training.models.gpt.mtp_block_spec", return_value=None)
    @patch("megatron.training.models.gpt.calculate_padded_vocab_size")
    @patch("megatron.training.models.gpt.is_vp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_vp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.GPTModel")
    def test_no_vocab_padding_uses_vocab_size_directly(self, mock_model, *patches):
        # The padding mock is the second patch from the bottom (after mtp_block_spec).
        # Fetch it by traversing in MRO-aware order isn't straightforward; instead,
        # patch it explicitly within this test for clarity.
        with patch("megatron.training.models.gpt.calculate_padded_vocab_size") as mock_pad:
            self.config.__dict__["should_pad_vocab"] = False
            self.config.__dict__["vocab_size"] = 32000
            self.builder.build_model(self.pg, pre_process=True, post_process=True)
            mock_pad.assert_not_called()
        assert mock_model.call_args.kwargs["vocab_size"] == 32000

    @patch("megatron.training.models.gpt.mtp_block_spec", return_value=None)
    @patch("megatron.training.models.gpt.is_vp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_vp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.calculate_padded_vocab_size", return_value=32128)
    @patch("megatron.training.models.gpt.GPTModel")
    def test_vocab_padding_calls_calculate_padded_vocab_size(self, mock_model, mock_pad, *_):
        self.config.__dict__["should_pad_vocab"] = True
        self.config.__dict__["vocab_size"] = 32000
        self.config.__dict__["make_vocab_size_divisible_by"] = 128
        self.config.transformer.tensor_model_parallel_size = 2
        self.builder.build_model(self.pg, pre_process=True, post_process=True)
        mock_pad.assert_called_once_with(32000, 128, 2)
        assert mock_model.call_args.kwargs["vocab_size"] == 32128

    @patch("megatron.training.models.gpt.mtp_block_spec", return_value=None)
    @patch("megatron.training.models.gpt.calculate_padded_vocab_size")
    @patch("megatron.training.models.gpt.is_vp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_vp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.GPTModel")
    def test_explicit_pre_post_process_passed_through(self, mock_model, *_):
        self.builder.build_model(self.pg, pre_process=False, post_process=True)
        kw = mock_model.call_args.kwargs
        assert kw["pre_process"] is False
        assert kw["post_process"] is True

    @patch("megatron.training.models.gpt.mtp_block_spec", return_value=None)
    @patch("megatron.training.models.gpt.calculate_padded_vocab_size")
    @patch("megatron.training.models.gpt.is_vp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_vp_first_stage", return_value=False)
    @patch("megatron.training.models.gpt.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.GPTModel")
    def test_infers_pre_process_from_vp_and_pp(
        self, mock_model, mock_pp_first, mock_pp_last, mock_vp_first, mock_vp_last, *_
    ):
        self.builder.build_model(self.pg)
        # is_vp_first_stage returns False → pre_process should be False even if pp_first is True
        assert mock_model.call_args.kwargs["pre_process"] is False

    @patch("megatron.training.models.gpt.mtp_block_spec", return_value=None)
    @patch("megatron.training.models.gpt.calculate_padded_vocab_size")
    @patch("megatron.training.models.gpt.is_vp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_vp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.GPTModel")
    def test_infers_post_process_from_vp_and_pp(self, mock_model, mock_pp_first, mock_pp_last, *_):
        self.builder.build_model(self.pg)
        mock_pp_last.assert_called_once_with(self.pg.pp)
        assert mock_model.call_args.kwargs["post_process"] is True

    @patch("megatron.training.models.gpt.mtp_block_spec", return_value=None)
    @patch("megatron.training.models.gpt.calculate_padded_vocab_size")
    @patch("megatron.training.models.gpt.is_vp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_vp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.GPTModel")
    def test_local_attn_backend_overrides_core_attention(self, mock_model, *_):
        from megatron.core.transformer.dot_product_attention import (
            DotProductAttention as MCoreDotProductAttention,
        )

        mock_spec = Mock(spec=ModuleSpec)
        mock_spec.submodules = Mock()
        mock_spec.submodules.self_attention = Mock()
        mock_spec.submodules.self_attention.submodules = Mock()
        self.config.__dict__["transformer_layer_spec"] = mock_spec
        self.config.transformer.attention_backend = AttnBackend.local

        self.builder.build_model(self.pg, pre_process=True, post_process=True)

        assert (
            mock_spec.submodules.self_attention.submodules.core_attention
            is MCoreDotProductAttention
        )

    @patch("megatron.training.models.gpt.mtp_block_spec")
    @patch("megatron.training.models.gpt.calculate_padded_vocab_size")
    @patch("megatron.training.models.gpt.is_vp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_vp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.GPTModel")
    def test_mtp_block_spec_passed_to_mcore(self, mock_model, *patches):
        mock_mtp = patches[-1]
        mtp_spec = ModuleSpec(module=object)
        mock_mtp.return_value = mtp_spec

        self.builder.build_model(self.pg, pre_process=True, post_process=True, vp_stage=1)

        # mtp_block_spec is called with (config, transformer_layer_spec, vp_stage=vp_stage)
        mock_mtp.assert_called_once_with(self.config, self._default_spec, vp_stage=1)
        assert mock_model.call_args.kwargs["mtp_block_spec"] is mtp_spec

    @patch("megatron.training.models.gpt.mtp_block_spec", return_value=None)
    @patch("megatron.training.models.gpt.calculate_padded_vocab_size")
    @patch("megatron.training.models.gpt.is_vp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_vp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.gpt.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.gpt.GPTModel")
    def test_config_params_passed_to_mcore(self, mock_model, *_):
        config = _make_gpt_config(
            vocab_size=32000,
            seq_length=4096,
            fp16_lm_cross_entropy=True,
            parallel_output=False,
            share_embeddings_and_output_weights=True,
            position_embedding_type="rope",
            rotary_percent=0.5,
            rotary_base=500000,
            rope_scaling=True,
            rope_scaling_factor=4.0,
            scatter_embedding_sequence_parallel=False,
        )
        config.__dict__["transformer_layer_spec"] = ModuleSpec(module=object)
        builder = GPTModelBuilder(config)
        pg = Mock()
        pg.pp = Mock()

        builder.build_model(pg, pre_process=True, post_process=True)

        kw = mock_model.call_args.kwargs
        assert kw["config"] is config.transformer
        assert kw["vocab_size"] == 32000
        assert kw["max_sequence_length"] == 4096
        assert kw["fp16_lm_cross_entropy"] is True
        assert kw["parallel_output"] is False
        assert kw["share_embeddings_and_output_weights"] is True
        assert kw["position_embedding_type"] == "rope"
        assert kw["rotary_percent"] == 0.5
        assert kw["rotary_base"] == 500000
        assert kw["rope_scaling"] is True
        assert kw["rope_scaling_factor"] == 4.0
        assert kw["seq_len_interpolation_factor"] is None
        assert kw["scatter_embedding_sequence_parallel"] is False
        assert kw["pre_process"] is True
        assert kw["post_process"] is True
        assert kw["pg_collection"] is pg
        assert kw["vp_stage"] is None


class TestGPTModelBuilderBuildDistributedModels:
    """Tests for GPTModelBuilder.build_distributed_models() — delegation to
    unimodal helper, hook composition, and default kwargs.
    """

    def setup_method(self):
        self.config = _make_gpt_config(vocab_size=32000)
        self.builder = GPTModelBuilder(self.config)
        self.pg = Mock()

    @patch("megatron.training.models.gpt.compose_hooks")
    @patch("megatron.training.models.gpt.unimodal_build_distributed_models")
    def test_delegates_to_unimodal_build_distributed_models(self, mock_unimodal, mock_compose):
        mock_unimodal.return_value = [Mock()]
        mock_compose.return_value = Mock(return_value=None)

        self.builder.build_distributed_models(self.pg)

        assert mock_unimodal.called

    @patch("megatron.training.models.gpt.compose_hooks")
    @patch("megatron.training.models.gpt.unimodal_build_distributed_models")
    def test_returns_model_list_from_unimodal(self, mock_unimodal, mock_compose):
        model_list = [Mock(), Mock()]
        mock_unimodal.return_value = model_list
        # post_wrap hook returns None → original list kept
        mock_compose.return_value = Mock(return_value=None)

        result = self.builder.build_distributed_models(self.pg)

        assert result is model_list

    @patch("megatron.training.models.gpt.compose_hooks")
    @patch("megatron.training.models.gpt.unimodal_build_distributed_models")
    def test_pre_wrap_hooks_composed_and_passed(self, mock_unimodal, mock_compose):
        mock_unimodal.return_value = [Mock()]
        composed_pre = Mock()
        composed_post = Mock(return_value=None)
        mock_compose.side_effect = [composed_pre, composed_post]

        hook1 = Mock()
        self.config.pre_wrap_hooks = [hook1]
        self.builder.build_distributed_models(self.pg)

        # First compose_hooks call must be with the pre_wrap_hooks list
        assert mock_compose.call_args_list[0] == call([hook1])
        # The composed pre-wrap hook is the 11th positional arg (index 10)
        unimodal_args = mock_unimodal.call_args.args
        assert unimodal_args[10] is composed_pre

    @patch("megatron.training.models.gpt.compose_hooks")
    @patch("megatron.training.models.gpt.unimodal_build_distributed_models")
    def test_post_wrap_hook_applied_to_results(self, mock_unimodal, mock_compose):
        model_list = [Mock()]
        wrapped_list = [Mock(), Mock()]
        mock_unimodal.return_value = model_list
        composed_pre = Mock()
        composed_post = Mock(return_value=wrapped_list)
        mock_compose.side_effect = [composed_pre, composed_post]

        result = self.builder.build_distributed_models(self.pg)

        composed_post.assert_called_once_with(model_list)
        assert result is wrapped_list

    @patch("megatron.training.models.gpt.compose_hooks")
    @patch("megatron.training.models.gpt.unimodal_build_distributed_models")
    def test_post_wrap_hook_returning_none_keeps_original_list(self, mock_unimodal, mock_compose):
        model_list = [Mock()]
        mock_unimodal.return_value = model_list
        mock_compose.return_value = Mock(return_value=None)

        result = self.builder.build_distributed_models(self.pg)

        assert result is model_list

    @patch("megatron.training.models.gpt.compose_hooks")
    @patch("megatron.training.models.gpt.unimodal_build_distributed_models")
    def test_default_parameters_forwarded(self, mock_unimodal, mock_compose):
        from megatron.core.enums import ModelType
        from megatron.core.transformer.module import Float16Module

        mock_unimodal.return_value = [Mock()]
        mock_compose.return_value = Mock(return_value=None)

        self.builder.build_distributed_models(self.pg)

        args = mock_unimodal.call_args.args
        assert args[0] == self.builder.build_model
        assert args[1] is self.config.transformer
        assert args[2] is self.pg
        assert args[3] is None  # ddp_config
        assert args[7] is True  # wrap_with_ddp
        assert args[8] is True  # data_parallel_random_init
        assert args[9] is Float16Module  # mixed_precision_wrapper
        assert args[11] is ModelType.encoder_or_decoder  # model_type


# =============================================================================
# Section 4 — mtp_block_spec
# =============================================================================


class TestMtpBlockSpec:
    """Tests for mtp_block_spec() helper function."""

    def _make_config(self, mtp_num_layers=None, transformer_impl="transformer_engine"):
        config = Mock()
        config.transformer.mtp_num_layers = mtp_num_layers
        config.transformer.transformer_impl = transformer_impl
        config.transformer.normalization = "LayerNorm"
        config.transformer.qk_l2_norm = False
        return config

    def test_returns_none_when_mtp_num_layers_is_none(self):
        config = self._make_config(mtp_num_layers=None)
        spec = ModuleSpec(module=object)
        result = mtp_block_spec(config, spec)
        assert result is None

    @patch("megatron.core.models.gpt.gpt_layer_specs.get_gpt_mtp_block_spec")
    def test_uses_explicit_spec_when_layer_specs_nonempty(self, mock_get_mtp):
        config = self._make_config(mtp_num_layers=1)
        spec = Mock(spec=ModuleSpec)
        spec.layer_specs = [Mock()]  # Non-empty
        mock_get_mtp.return_value = Mock(spec=ModuleSpec)

        with patch(
            "megatron.training.models.gpt.get_gpt_decoder_layer_specs"
        ) as mock_decoder_specs:
            mock_decoder_specs.return_value = [Mock(), Mock()]
            mtp_block_spec(config, spec)

        # When layer_specs is non-empty, use the last decoder spec (not the explicit spec arg)
        passed_spec = mock_get_mtp.call_args.args[1]
        assert passed_spec is mock_decoder_specs.return_value[-1]

    @patch("megatron.training.models.gpt.default_layer_spec")
    @patch("megatron.core.models.gpt.gpt_layer_specs.get_gpt_mtp_block_spec")
    def test_uses_default_layer_spec_for_empty_layer_specs(self, mock_get_mtp, mock_default):
        config = self._make_config(mtp_num_layers=1)
        spec = Mock(spec=ModuleSpec)
        spec.layer_specs = []  # Empty → falls back to default_layer_spec
        fallback_spec = Mock(spec=ModuleSpec)
        mock_default.return_value = fallback_spec
        mock_get_mtp.return_value = Mock(spec=ModuleSpec)

        mtp_block_spec(config, spec, vp_stage=4)

        mock_default.assert_called_once_with(config, 4)
        passed_spec = mock_get_mtp.call_args.args[1]
        assert passed_spec is fallback_spec

    @patch("megatron.core.models.gpt.gpt_layer_specs.get_gpt_mtp_block_spec")
    def test_passes_vp_stage_and_use_te_to_get_gpt_mtp_block_spec(self, mock_get_mtp):
        config = self._make_config(mtp_num_layers=2, transformer_impl="transformer_engine")
        spec = ModuleSpec(module=object)
        mock_get_mtp.return_value = Mock(spec=ModuleSpec)

        with patch(
            "megatron.training.models.gpt.get_gpt_decoder_layer_specs"
        ) as mock_decoder_specs:
            mock_decoder_specs.return_value = [Mock(), Mock()]
            mtp_block_spec(config, spec, vp_stage=3)

        call_kwargs = mock_get_mtp.call_args.kwargs
        assert call_kwargs["use_transformer_engine"] is True
        assert call_kwargs["vp_stage"] == 3

    @patch("megatron.core.models.gpt.gpt_layer_specs.get_gpt_mtp_block_spec")
    def test_use_transformer_engine_false_when_impl_not_te(self, mock_get_mtp):
        config = self._make_config(mtp_num_layers=2, transformer_impl="local")
        spec = ModuleSpec(module=object)
        mock_get_mtp.return_value = Mock(spec=ModuleSpec)

        with patch(
            "megatron.training.models.gpt.get_gpt_decoder_layer_specs"
        ) as mock_decoder_specs:
            mock_decoder_specs.return_value = [Mock(), Mock()]
            mtp_block_spec(config, spec)

        assert mock_get_mtp.call_args.kwargs["use_transformer_engine"] is False
