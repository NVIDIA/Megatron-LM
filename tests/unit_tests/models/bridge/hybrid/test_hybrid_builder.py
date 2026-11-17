# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/tests/unit_tests/models/hybrid/test_hybrid_builder.py
# Source license: Apache-2.0.

from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.models.common import ModelConfig
from megatron.bridge.models.hybrid.hybrid_builder import (
    get_default_hybrid_stack_spec,
    modelopt_hybrid_stack_spec,
    transformer_engine_hybrid_stack_spec,
)
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.core.transformer import ModuleSpec
from megatron.training.models.hybrid import HybridModelBuilder, HybridModelConfig


def _make_transformer(**kwargs):
    defaults = dict(num_layers=2, hidden_size=128, num_attention_heads=1)
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


def _make_hybrid_config(**kwargs):
    defaults = dict(transformer=_make_transformer(), vocab_size=32000)
    defaults.update(kwargs)
    return HybridModelConfig(**defaults)


class TestHybridStackSpecs:
    def test_transformer_engine_spec_returns_default(self):
        with patch(
            "megatron.bridge.models.hybrid.hybrid_builder.default_hybrid_stack_spec"
        ) as mock_spec:
            result = transformer_engine_hybrid_stack_spec()
        assert result is mock_spec

    def test_modelopt_spec_calls_modelopt_helper(self):
        mock_spec = Mock(spec=ModuleSpec)
        with patch(
            "megatron.bridge.models.hybrid.hybrid_builder.get_hybrid_stack_modelopt_spec",
            return_value=mock_spec,
        ) as mock_fn:
            result = modelopt_hybrid_stack_spec()
        mock_fn.assert_called_once_with(local_core_attention=False, remap_te_layernorm=True)
        assert result is mock_spec

    def test_modelopt_spec_remaps_te_layernorm_state_dict_keys(self):
        result = modelopt_hybrid_stack_spec()

        assert result.submodules.mamba_layer.submodules.sharded_state_dict_keys_map == {
            "norm.": "mixer.in_proj.layer_norm_"
        }
        assert result.submodules.attention_layer.submodules.sharded_state_dict_keys_map == {
            "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
            "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
        }

    def test_default_spec_returns_te_spec(self):
        config = _make_hybrid_config()
        with patch(
            "megatron.bridge.models.hybrid.hybrid_builder.transformer_engine_hybrid_stack_spec",
            return_value=Mock(spec=ModuleSpec),
        ) as mock_fn:
            result = get_default_hybrid_stack_spec(config)
        mock_fn.assert_called_once()
        assert result is mock_fn.return_value

    def test_default_spec_returns_modelopt_spec(self):
        config = _make_hybrid_config()
        config.restore_modelopt_state = True
        with patch(
            "megatron.bridge.models.hybrid.hybrid_builder.modelopt_hybrid_stack_spec",
            return_value=Mock(spec=ModuleSpec),
        ) as mock_fn:
            result = get_default_hybrid_stack_spec(config)
        mock_fn.assert_called_once_with(config)
        assert result is mock_fn.return_value


class TestHybridModelConfig:
    def test_rejects_mamba_stack_spec_argument(self):
        module_spec = ModuleSpec(module=object)

        with pytest.raises(TypeError, match="mamba_stack_spec"):
            HybridModelConfig(transformer=_make_transformer(), mamba_stack_spec=module_spec)

    def test_mamba_stack_spec_property_alias(self):
        module_spec = ModuleSpec(module=object)
        config = HybridModelConfig(transformer=_make_transformer())

        config.mamba_stack_spec = module_spec

        assert config.hybrid_stack_spec is module_spec
        assert config.mamba_stack_spec is module_spec

    def test_proxies_transformer_attribute(self):
        transformer = _make_transformer(hidden_size=256)
        config = HybridModelConfig(transformer=transformer, vocab_size=32000)

        assert config.hidden_size == 256
        config.hidden_size = 512
        assert transformer.hidden_size == 512

    def test_from_dict_with_hybrid_builder_target(self):
        config = HybridModelConfig(transformer=_make_transformer(), vocab_size=32000)
        data = config.as_dict()

        restored = ModelConfig.from_dict(data)

        assert isinstance(restored, HybridModelConfig)
        assert restored.get_builder_cls() is HybridModelBuilder


class TestHybridModelBuilder:
    def setup_method(self):
        self.config = _make_hybrid_config(vocab_size=32000)
        self.builder = HybridModelBuilder(self.config)
        self.pg = Mock()
        self.pg.pp = Mock()

    @patch("megatron.training.models.hybrid.HybridModel")
    def test_raises_when_vocab_size_none(self, mock_model):
        self.config.vocab_size = None
        with pytest.raises(AssertionError, match="vocab_size"):
            self.builder.build_model(self.pg)
        mock_model.assert_not_called()

    @patch("megatron.training.models.hybrid.HybridModel")
    def test_hybrid_stack_spec_used_directly(self, mock_model):
        module_spec = ModuleSpec(module=object)
        self.config.hybrid_stack_spec = module_spec

        self.builder.build_model(self.pg, pre_process=True, post_process=True)

        assert mock_model.call_args.kwargs["hybrid_stack_spec"] is module_spec

    @patch("megatron.training.models.hybrid.calculate_padded_vocab_size", return_value=32128)
    @patch("megatron.training.models.hybrid.HybridModel")
    def test_vocab_padding_calls_calculate_padded_vocab_size(self, mock_model, mock_pad):
        self.config.should_pad_vocab = True
        self.config.transformer.tensor_model_parallel_size = 2

        self.builder.build_model(self.pg, pre_process=True, post_process=True)

        mock_pad.assert_called_once_with(32000, 128, 2)
        assert mock_model.call_args.kwargs["vocab_size"] == 32128

    @patch("megatron.training.models.hybrid.is_pp_first_stage", return_value=False)
    @patch("megatron.training.models.hybrid.is_pp_last_stage", return_value=True)
    @patch("megatron.training.models.hybrid.HybridModel")
    def test_pipeline_flags_default_from_pipeline_stage(
        self, mock_model, mock_last_stage, mock_first_stage
    ):
        self.builder.build_model(self.pg)

        mock_first_stage.assert_called_once_with(self.pg.pp)
        mock_last_stage.assert_called_once_with(self.pg.pp)
        assert mock_model.call_args.kwargs["pre_process"] is False
        assert mock_model.call_args.kwargs["post_process"] is True

    @patch("megatron.training.models.hybrid.HybridModel")
    def test_explicit_pre_post_process_passed_through(self, mock_model):
        self.builder.build_model(self.pg, pre_process=False, post_process=True)

        assert mock_model.call_args.kwargs["pre_process"] is False
        assert mock_model.call_args.kwargs["post_process"] is True

    @patch("megatron.training.models.hybrid.HybridModel")
    def test_config_params_passed_to_mcore(self, mock_model):
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

        builder.build_model(self.pg, pre_process=True, post_process=True)

        kw = mock_model.call_args.kwargs
        assert kw["config"] is config.transformer
        assert kw["vocab_size"] == 32000
        assert kw["max_sequence_length"] == 4096
        assert kw["hybrid_layer_pattern"] == "M-A-"
        assert kw["fp16_lm_cross_entropy"] is True
        assert kw["parallel_output"] is False
        assert kw["share_embeddings_and_output_weights"] is True

    @patch("megatron.training.models.hybrid.HybridModel")
    def test_requested_logit_dtype_reaches_new_mcore_builder(self, mock_model):
        self.config.logit_dtype = torch.float32

        self.builder.build_model(self.pg, pre_process=True, post_process=True)

        assert mock_model.call_args.kwargs["logit_dtype"] is torch.float32

    @patch("megatron.training.models.hybrid.compose_hooks")
    @patch("megatron.training.models.hybrid.unimodal_build_distributed_models")
    def test_build_distributed_models_delegates_to_unimodal(self, mock_unimodal, mock_compose):
        model_list = [Mock()]
        mock_unimodal.return_value = model_list
        mock_compose.return_value = Mock(return_value=None)

        result = self.builder.build_distributed_models(self.pg)

        assert result is model_list
        assert mock_unimodal.called
