# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import Mock, call, patch

import pytest

from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.models.hybrid import HybridModelBuilder, HybridModelConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transformer(**kwargs):
    defaults = dict(num_layers=2, hidden_size=128, num_attention_heads=1)
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


def _make_hybrid_config(**kwargs):
    defaults = dict(transformer=_make_transformer(), vocab_size=32000)
    defaults.update(kwargs)
    return HybridModelConfig(**defaults)


# =============================================================================
# Section 1 — HybridModelConfig
# =============================================================================


class TestHybridModelConfigInitialization:
    """Tests for HybridModelConfig field defaults and custom initialization."""

    def test_builder_classvar(self):
        assert HybridModelConfig.builder == "megatron.training.models.hybrid.HybridModelBuilder"

    def test_default_values(self):
        config = HybridModelConfig(transformer=_make_transformer())
        assert config.fp16_lm_cross_entropy is False
        assert config.parallel_output is True
        assert config.share_embeddings_and_output_weights is False
        assert config.hybrid_layer_pattern is None
        assert config.seq_length == 8192
        assert config.position_embedding_type == "none"
        assert config.rotary_percent == 1.0
        assert config.rotary_base == 10000
        assert config.seq_len_interpolation_factor is None
        assert config.make_vocab_size_divisible_by == 128
        assert config.vocab_size is None
        assert config.should_pad_vocab is False

    def test_custom_initialization(self):
        config = HybridModelConfig(
            transformer=_make_transformer(),
            fp16_lm_cross_entropy=True,
            parallel_output=False,
            hybrid_attention_ratio=0.25,
            hybrid_mlp_ratio=0.1,
            hybrid_layer_pattern="M-M*-",
            seq_length=4096,
            vocab_size=50000,
        )
        assert config.fp16_lm_cross_entropy is True
        assert config.parallel_output is False
        assert config.hybrid_attention_ratio == 0.25
        assert config.hybrid_mlp_ratio == 0.1
        assert config.hybrid_layer_pattern == "M-M*-"
        assert config.seq_length == 4096
        assert config.vocab_size == 50000

    def test_hybrid_stack_spec_default_is_none(self):
        config = _make_hybrid_config()
        assert config.hybrid_stack_spec is None


class TestHybridModelConfigGetAttr:
    """Tests for HybridModelConfig.__getattr__ — direct access vs. TransformerConfig proxy."""

    def setup_method(self):
        self.transformer = _make_transformer(hidden_size=256, num_layers=4)
        self.config = HybridModelConfig(transformer=self.transformer, vocab_size=32000)

    def test_own_attribute_not_proxied(self):
        # vocab_size is defined on HybridModelConfig but not on TransformerConfig;
        # it is returned directly from config.__dict__, __getattr__ is never invoked.
        assert self.config.vocab_size == 32000

    def test_proxies_transformer_attribute(self):
        # hidden_size is not a field on HybridModelConfig, so __getattr__ proxies to transformer
        assert self.config.hidden_size == 256

    def test_raises_attribute_error_for_unknown(self):
        with pytest.raises(AttributeError):
            _ = self.config.completely_unknown_attr_xyz

    def test_raises_before_transformer_init(self):
        # Simulate the "transformer not yet set" path in __getattr__
        del self.config.__dict__["transformer"]
        with pytest.raises(AttributeError):
            _ = self.config.hidden_size

    def test_error_message_contains_attr_name(self):
        attr_name = "completely_unknown_attr_xyz"
        with pytest.raises(AttributeError, match=attr_name):
            getattr(self.config, attr_name)


class TestHybridModelConfigSetAttr:
    """Tests for HybridModelConfig.__setattr__ — own-field writes vs. TransformerConfig proxy writes."""

    def setup_method(self):
        self.transformer = _make_transformer(hidden_size=256)
        self.config = HybridModelConfig(transformer=self.transformer, vocab_size=32000)

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
        # Delete transformer to simulate the pre-init state in __setattr__
        del self.config.__dict__["transformer"]
        self.config.vocab_size = 42
        assert self.config.__dict__["vocab_size"] == 42

    def test_set_transformer_itself_stores_on_self(self):
        new_transformer = _make_transformer(hidden_size=512)
        self.config.transformer = new_transformer
        assert self.config.transformer is new_transformer
        assert self.config.__dict__["transformer"] is new_transformer

    def test_set_own_attr_does_not_go_to_transformer(self):
        # vocab_size is not on TransformerConfig, so the write should stay on self
        self.config.vocab_size = 99999
        assert self.config.__dict__.get("vocab_size") == 99999
        assert not hasattr(self.config.transformer, "vocab_size")

    def test_proxied_write_does_not_shadow_on_self(self):
        self.config.hidden_size = 2048
        assert "hidden_size" not in self.config.__dict__


class TestHybridModelConfigFinalize:
    """Tests for HybridModelConfig.finalize() — validation logic."""

    def test_calls_transformer_finalize(self):
        config = _make_hybrid_config()
        with patch.object(config.transformer, "finalize") as mock_finalize:
            config.finalize()
        mock_finalize.assert_called_once()


# =============================================================================
# Section 2 — HybridModelBuilder
# =============================================================================


class TestHybridModelBuilderInit:
    """Tests for HybridModelBuilder.__init__ — config storage."""

    def setup_method(self):
        self.config = _make_hybrid_config()
        self.builder = HybridModelBuilder(self.config)

    def test_stores_model_config(self):
        assert self.builder._model_config is self.config


class TestHybridModelBuilderBuildModel:
    """Tests for HybridModelBuilder.build_model() — spec resolution, vocab padding, pp-stage inference, and MCoreHybridModel kwargs."""

    def setup_method(self):
        self.config = _make_hybrid_config(vocab_size=32000)
        self.builder = HybridModelBuilder(self.config)
        self.pg = Mock()
        self.pg.pp = Mock()

    @patch("megatron.training.models.hybrid.calculate_padded_vocab_size")
    @patch("megatron.training.models.hybrid.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.hybrid.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.hybrid.HybridModel")
    def test_raises_when_vocab_size_none(self, mock_model, *_):
        self.config.vocab_size = None
        with pytest.raises(AssertionError, match="vocab_size"):
            self.builder.build_model(self.pg)

    @patch("megatron.training.models.hybrid.calculate_padded_vocab_size")
    @patch("megatron.training.models.hybrid.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.hybrid.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.hybrid.HybridModel")
    def test_spec_already_module_spec_used_directly(self, mock_model, *_):
        module_spec = ModuleSpec(module=object)
        self.config.__dict__["hybrid_stack_spec"] = module_spec
        self.builder.build_model(self.pg, pre_process=True, post_process=True)
        call_kwargs = mock_model.call_args.kwargs
        assert call_kwargs["hybrid_stack_spec"] is module_spec

    @patch("megatron.training.models.hybrid.calculate_padded_vocab_size")
    @patch("megatron.training.models.hybrid.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.hybrid.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.hybrid.HybridModel")
    def test_spec_none_uses_default_te_spec(self, mock_model, *_):
        # Default config: hybrid_stack_spec=None, restore_modelopt_state=False,
        # transformer_impl != "inference_optimized" → falls through to default TE spec.
        with patch("megatron.training.models.hybrid.default_hybrid_stack_spec") as mock_default:
            self.builder.build_model(self.pg, pre_process=True, post_process=True)
        call_kwargs = mock_model.call_args.kwargs
        assert call_kwargs["hybrid_stack_spec"] is mock_default

    @patch("megatron.training.models.hybrid.calculate_padded_vocab_size")
    @patch("megatron.training.models.hybrid.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.hybrid.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.hybrid.HybridModel")
    def test_spec_none_with_inference_optimized_uses_inference_spec(self, mock_model, *_):
        self.config.transformer.transformer_impl = "inference_optimized"
        with patch("megatron.training.models.hybrid.hybrid_inference_stack_spec") as mock_inf:
            self.builder.build_model(self.pg, pre_process=True, post_process=True)
        call_kwargs = mock_model.call_args.kwargs
        assert call_kwargs["hybrid_stack_spec"] is mock_inf

    @patch("megatron.training.models.hybrid.calculate_padded_vocab_size")
    @patch("megatron.training.models.hybrid.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.hybrid.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.hybrid.HybridModel")
    def test_spec_none_with_restore_modelopt_state_uses_modelopt_spec(self, mock_model, *_):
        self.config.restore_modelopt_state = True
        modelopt_spec = ModuleSpec(module=object)
        with patch(
            "megatron.training.models.hybrid.get_hybrid_stack_modelopt_spec",
            return_value=modelopt_spec,
        ) as mock_fn:
            self.builder.build_model(self.pg, pre_process=True, post_process=True)
        mock_fn.assert_called_once_with(local_core_attention=False, remap_te_layernorm=False)
        call_kwargs = mock_model.call_args.kwargs
        assert call_kwargs["hybrid_stack_spec"] is modelopt_spec

    @patch("megatron.training.models.hybrid.calculate_padded_vocab_size")
    @patch("megatron.training.models.hybrid.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.hybrid.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.hybrid.HybridModel")
    def test_spec_none_inference_optimized_takes_precedence_over_modelopt(self, mock_model, *_):
        self.config.transformer.transformer_impl = "inference_optimized"
        self.config.restore_modelopt_state = True
        with (
            patch("megatron.training.models.hybrid.hybrid_inference_stack_spec") as mock_inf,
            patch(
                "megatron.training.models.hybrid.get_hybrid_stack_modelopt_spec"
            ) as mock_modelopt,
        ):
            self.builder.build_model(self.pg, pre_process=True, post_process=True)
        call_kwargs = mock_model.call_args.kwargs
        assert call_kwargs["hybrid_stack_spec"] is mock_inf
        mock_modelopt.assert_not_called()

    @patch("megatron.training.models.hybrid.calculate_padded_vocab_size")
    @patch("megatron.training.models.hybrid.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.hybrid.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.hybrid.HybridModel")
    def test_no_vocab_padding_uses_vocab_size_directly(
        self, mock_model, mock_first, mock_last, mock_pad
    ):
        self.config.__dict__["should_pad_vocab"] = False
        self.config.__dict__["vocab_size"] = 32000
        self.builder.build_model(self.pg, pre_process=True, post_process=True)
        mock_pad.assert_not_called()
        assert mock_model.call_args.kwargs["vocab_size"] == 32000

    @patch("megatron.training.models.hybrid.calculate_padded_vocab_size", return_value=32128)
    @patch("megatron.training.models.hybrid.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.hybrid.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.hybrid.HybridModel")
    def test_vocab_padding_calls_calculate_padded_vocab_size(
        self, mock_model, mock_first, mock_last, mock_pad
    ):
        self.config.__dict__["should_pad_vocab"] = True
        self.config.__dict__["vocab_size"] = 32000
        self.config.__dict__["make_vocab_size_divisible_by"] = 128
        self.config.transformer.tensor_model_parallel_size = 2
        self.builder.build_model(self.pg, pre_process=True, post_process=True)
        mock_pad.assert_called_once_with(32000, 128, 2)
        assert mock_model.call_args.kwargs["vocab_size"] == 32128

    @patch("megatron.training.models.hybrid.calculate_padded_vocab_size")
    @patch("megatron.training.models.hybrid.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.hybrid.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.hybrid.HybridModel")
    def test_explicit_pre_post_process_passed_through(self, mock_model, *_):
        self.builder.build_model(self.pg, pre_process=False, post_process=True)
        call_kwargs = mock_model.call_args.kwargs
        assert call_kwargs["pre_process"] is False
        assert call_kwargs["post_process"] is True

    @patch("megatron.training.models.hybrid.calculate_padded_vocab_size")
    @patch("megatron.training.models.hybrid.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.hybrid.is_pp_first_stage", return_value=False)
    @patch("megatron.training.models.hybrid.HybridModel")
    def test_infers_pre_process_from_pg(self, mock_model, mock_first, mock_last, *_):
        self.builder.build_model(self.pg)
        mock_first.assert_called_once_with(self.pg.pp)
        assert mock_model.call_args.kwargs["pre_process"] is False

    @patch("megatron.training.models.hybrid.calculate_padded_vocab_size")
    @patch("megatron.training.models.hybrid.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.hybrid.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.hybrid.HybridModel")
    def test_infers_post_process_from_pg(self, mock_model, mock_first, mock_last, *_):
        self.builder.build_model(self.pg)
        mock_last.assert_called_once_with(self.pg.pp)
        assert mock_model.call_args.kwargs["post_process"] is True

    @patch("megatron.training.models.hybrid.calculate_padded_vocab_size")
    @patch("megatron.training.models.hybrid.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.hybrid.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.hybrid.HybridModel")
    def test_virtual_pipeline_raises(self, mock_model, *_):
        with pytest.raises(AssertionError, match="Virtual pipeline"):
            self.builder.build_model(self.pg, vp_stage=0)

    @patch("megatron.training.models.hybrid.calculate_padded_vocab_size")
    @patch("megatron.training.models.hybrid.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.hybrid.is_pp_first_stage", return_value=True)
    @patch("megatron.training.models.hybrid.HybridModel")
    def test_config_params_passed_to_mcore(self, mock_model, *_):
        config = _make_hybrid_config(
            vocab_size=32000,
            seq_length=4096,
            hybrid_layer_pattern="M-A-",
            fp16_lm_cross_entropy=True,
            parallel_output=False,
            share_embeddings_and_output_weights=True,
            position_embedding_type="rope",
        )
        builder = HybridModelBuilder(config)
        pg = Mock()
        pg.pp = Mock()

        builder.build_model(pg, pre_process=True, post_process=True)

        kw = mock_model.call_args.kwargs
        assert kw["config"] is config.transformer
        assert kw["vocab_size"] == 32000
        assert kw["max_sequence_length"] == 4096
        assert kw["hybrid_layer_pattern"] == "M-A-"
        assert kw["fp16_lm_cross_entropy"] is True
        assert kw["parallel_output"] is False
        assert kw["share_embeddings_and_output_weights"] is True
        assert kw["position_embedding_type"] == "rope"
        assert kw["rotary_percent"] == 1.0
        assert kw["rotary_base"] == 10000
        assert kw["seq_len_interpolation_factor"] is None
        assert kw["pg_collection"] is pg
        assert kw["vp_stage"] is None


class TestHybridModelBuilderBuildDistributedModels:
    """Tests for HybridModelBuilder.build_distributed_models() — delegation to unimodal helper, hook composition, and default kwargs."""

    def setup_method(self):
        self.config = _make_hybrid_config(vocab_size=32000)
        self.builder = HybridModelBuilder(self.config)
        self.pg = Mock()

    @patch("megatron.training.models.hybrid.compose_hooks")
    @patch("megatron.training.models.hybrid.unimodal_build_distributed_models")
    def test_delegates_to_unimodal_build_distributed_models(self, mock_unimodal, mock_compose):
        model_list = [Mock()]
        mock_unimodal.return_value = model_list
        mock_compose.return_value = Mock(return_value=None)

        self.builder.build_distributed_models(self.pg)

        assert mock_unimodal.called

    @patch("megatron.training.models.hybrid.compose_hooks")
    @patch("megatron.training.models.hybrid.unimodal_build_distributed_models")
    def test_returns_model_list_from_unimodal(self, mock_unimodal, mock_compose):
        model_list = [Mock(), Mock()]
        mock_unimodal.return_value = model_list
        # post_wrap hook returns None → original list is kept
        mock_compose.return_value = Mock(return_value=None)

        result = self.builder.build_distributed_models(self.pg)

        assert result is model_list

    @patch("megatron.training.models.hybrid.compose_hooks")
    @patch("megatron.training.models.hybrid.unimodal_build_distributed_models")
    def test_pre_wrap_hooks_composed_and_passed(self, mock_unimodal, mock_compose):
        model_list = [Mock()]
        mock_unimodal.return_value = model_list
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

    @patch("megatron.training.models.hybrid.compose_hooks")
    @patch("megatron.training.models.hybrid.unimodal_build_distributed_models")
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

    @patch("megatron.training.models.hybrid.compose_hooks")
    @patch("megatron.training.models.hybrid.unimodal_build_distributed_models")
    def test_post_wrap_hook_returning_none_keeps_original_list(self, mock_unimodal, mock_compose):
        model_list = [Mock()]
        mock_unimodal.return_value = model_list
        mock_compose.return_value = Mock(return_value=None)

        result = self.builder.build_distributed_models(self.pg)

        assert result is model_list

    @patch("megatron.training.models.hybrid.compose_hooks")
    @patch("megatron.training.models.hybrid.unimodal_build_distributed_models")
    def test_default_parameters_forwarded(self, mock_unimodal, mock_compose):
        from megatron.core.enums import ModelType
        from megatron.core.transformer.module import Float16Module

        model_list = [Mock()]
        mock_unimodal.return_value = model_list
        mock_compose.return_value = Mock(return_value=None)

        self.builder.build_distributed_models(self.pg)

        # unimodal_build_distributed_models is called with all positional args:
        # build_model, transformer_config, pg_collection, ddp_config,
        # overlap_param_gather_with_optimizer_step, use_megatron_fsdp, use_torch_fsdp2,
        # wrap_with_ddp, data_parallel_random_init, mixed_precision_wrapper,
        # composed_pre_wrap_hook, model_type
        args = mock_unimodal.call_args.args
        assert args[0] == self.builder.build_model
        assert args[1] is self.config.transformer
        assert args[2] is self.pg
        assert args[3] is None  # ddp_config
        assert args[7] is True  # wrap_with_ddp
        assert args[8] is False  # data_parallel_random_init
        assert args[9] is Float16Module  # mixed_precision_wrapper
        assert args[11] is ModelType.encoder_or_decoder  # model_type
