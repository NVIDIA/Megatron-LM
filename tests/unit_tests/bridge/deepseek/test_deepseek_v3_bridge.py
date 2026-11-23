# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c209ed397fb36272c2216053058cda4e37802f2d/tests/unit_tests/models/deepseek/test_deepseek_bridges.py
# Source license: Apache-2.0.

"""Unit tests for the builder-backed DeepSeek-V3 bridge."""

import math
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

from megatron.bridge.models import AutoBridge
from megatron.bridge.models.common.base import ModelConfig
from megatron.bridge.models.conversion import quantization_utils
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.utils import (
    carry_rope_into_rope_parameters,
    conform_config_to_reference,
)
from megatron.bridge.models.deepseek import deepseek_v3_bridge
from megatron.bridge.models.deepseek.attention import get_deepseek_decoder_block_spec
from megatron.bridge.models.deepseek.common import get_common_mapping_list
from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
from megatron.bridge.models.gpt.model_config import BridgeGPTMLAModelConfig
from megatron.bridge.models.transformer_config import MLATransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

_DS_V3_CONFIG = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "ep_size": 1,
    "first_k_dense_replace": 3,
    "hidden_act": "silu",
    "hidden_size": 7168,
    "initializer_range": 0.02,
    "intermediate_size": 18432,
    "kv_lora_rank": 512,
    "max_position_embeddings": 163840,
    "model_type": "deepseek_v3",
    "moe_intermediate_size": 2048,
    "moe_layer_freq": 1,
    "n_group": 8,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 128,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 61,
    "num_key_value_heads": 128,
    "num_nextn_predict_layers": 1,
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    },
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 40,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
    "rope_theta": 10000,
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "tie_word_embeddings": False,
    "topk_group": 4,
    "topk_method": "noaux_tc",
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "v_head_dim": 128,
    "vocab_size": 129280,
}

# Moonlight-16B-A3B and Kimi-K2 checkpoints declare the DeepseekV3ForCausalLM architecture.
_MOONLIGHT_OVERRIDES = {
    "aux_loss_alpha": 0.001,
    "first_k_dense_replace": 1,
    "n_group": 1,
    "num_nextn_predict_layers": 0,
    "q_lora_rank": None,
    "rope_scaling": None,
    "rope_theta": 50000.0,
    "topk_group": 1,
}
_KIMI_K2_OVERRIDES = {
    "aux_loss_alpha": 0.001,
    "first_k_dense_replace": 1,
    "n_group": 1,
    "n_routed_experts": 384,
    "num_nextn_predict_layers": 0,
    "rope_scaling": {
        "beta_fast": 1.0,
        "beta_slow": 1.0,
        "factor": 32.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
    "rope_theta": 50000.0,
    "topk_group": 1,
}


def _hf_config(**overrides) -> DeepseekV3Config:
    """DeepSeek-V3's Hugging Face config with optional field overrides."""
    return DeepseekV3Config.from_dict({**_DS_V3_CONFIG, **overrides})


class TestDeepSeekV3BridgeModelConfig:
    """The bridge builds DeepSeek-V3 through the builder-backed MLA model config."""

    def test_bridge_registration(self):
        assert issubclass(DeepSeekV3Bridge, MegatronModelBridge)
        assert DeepSeekV3Bridge.MODEL_CONFIG_CLASS is BridgeGPTMLAModelConfig
        assert DeepSeekV3Bridge.DEQUANTIZED_SOURCE_KEY_SUFFIXES == ("_scale_inv",)

    def test_generate_pipeline_layout_balances_uneven_decoder_layers(self):
        layout = DeepSeekV3Bridge.generate_pipeline_layout(num_layers=7, pp=3, mtp_layers=2)

        assert layout == [
            ["embedding", "decoder", "decoder", "decoder"],
            ["decoder", "decoder"],
            ["decoder", "decoder", "mtp", "mtp", "loss"],
        ]

    def test_model_config_maps_deepseek_v3(self):
        model_config = DeepSeekV3Bridge().hf_config_to_model_config(_hf_config())

        assert isinstance(model_config, BridgeGPTMLAModelConfig)
        assert isinstance(model_config.transformer, MLATransformerConfig)
        assert model_config.num_layers == 61
        assert model_config.hidden_size == 7168
        assert model_config.ffn_hidden_size == 18432
        assert model_config.vocab_size == 129280
        assert model_config.seq_length == 163840
        assert model_config.layernorm_epsilon == 1e-6
        assert model_config.multi_latent_attention is True
        assert model_config.qk_layernorm is True
        assert model_config.q_lora_rank == 1536
        assert model_config.kv_lora_rank == 512
        assert model_config.qk_head_dim == 128
        assert model_config.qk_pos_emb_head_dim == 64
        assert model_config.v_head_dim == 128
        # MLA attention applies the YaRN RoPE from its transformer config.
        assert model_config.position_embedding_type == "rope"
        assert model_config.transformer.rotary_base == 10000
        assert model_config.rotary_scaling_factor == 40
        assert model_config.original_max_position_embeddings == 4096
        assert model_config.beta_fast == 32
        assert model_config.beta_slow == 1
        assert model_config.mscale == 1.0
        assert model_config.mscale_all_dim == 1.0
        # First-k-dense MoE layers, sigmoid group-limited routing with expert bias, and MTP.
        assert model_config.num_moe_experts == 256
        assert model_config.moe_router_topk == 8
        assert model_config.moe_ffn_hidden_size == 2048
        assert model_config.moe_layer_freq == [0] * 3 + [1] * 58
        assert model_config.moe_shared_expert_intermediate_size == 2048
        assert model_config.moe_router_score_function == "sigmoid"
        assert model_config.moe_router_enable_expert_bias is True
        assert model_config.moe_router_dtype == "fp32"
        assert model_config.moe_router_num_groups == 8
        assert model_config.moe_router_group_topk == 4
        assert model_config.moe_router_topk_scaling_factor == 2.5
        assert model_config.moe_aux_loss_coeff == 0.0001
        assert model_config.mtp_num_layers == 1
        assert model_config.make_vocab_size_divisible_by == 1280
        assert model_config.share_embeddings_and_output_weights is False
        assert model_config.transformer_layer_spec.func is get_deepseek_decoder_block_spec
        assert model_config.transformer_layer_spec.keywords == {
            "use_transformer_engine": deepseek_v3_bridge.HAVE_TE
        }

    def test_model_config_kwargs_are_flat(self):
        """The MLA rope scaling fields are plain kwargs, not the base mapping's private ones."""
        kwargs = DeepSeekV3Bridge().hf_config_to_model_config_kwargs(_hf_config())

        assert "_mla_rope_params" not in kwargs
        assert "yarn_rotary_scaling_factor" not in kwargs
        assert kwargs["rotary_scaling_factor"] == 40
        assert kwargs["mscale_all_dim"] == 1.0
        assert kwargs["position_embedding_type"] == "rope"

    def test_model_config_preserves_context_length_and_aux_loss(self):
        hf_config = _hf_config(max_position_embeddings=8192, aux_loss_alpha=0.001)
        bridge = DeepSeekV3Bridge()

        model_config = bridge.hf_config_to_model_config(hf_config)
        exported = bridge.megatron_to_hf_config(model_config)

        assert model_config.seq_length == 8192
        assert model_config.moe_aux_loss_coeff == 0.001
        assert exported["max_position_embeddings"] == 8192
        assert exported["aux_loss_alpha"] == 0.001

    def test_model_config_preserves_none_q_lora_rank(self):
        bridge = DeepSeekV3Bridge()
        hf_config = _hf_config(q_lora_rank=None)

        kwargs = bridge.hf_config_to_model_config_kwargs(hf_config)
        model_config = bridge.hf_config_to_model_config(hf_config)
        exported = bridge.megatron_to_hf_config(model_config)

        assert "q_lora_rank" in kwargs and kwargs["q_lora_rank"] is None
        assert model_config.q_lora_rank is None
        assert "q_lora_rank" in exported and exported["q_lora_rank"] is None

    @pytest.mark.parametrize(
        "overrides", [_MOONLIGHT_OVERRIDES, _KIMI_K2_OVERRIDES], ids=["moonlight", "kimi-k2"]
    )
    def test_model_config_keeps_rotary_base_on_the_mla_transformer(self, overrides):
        """MLA attention reads rope_theta=50000 from the transformer config, not the outer one."""
        model_config = DeepSeekV3Bridge().hf_config_to_model_config(_hf_config(**overrides))

        assert model_config.transformer.rotary_base == 50000.0
        assert model_config.rotary_base == 50000.0
        assert model_config.mtp_num_layers is None

    def test_model_config_maps_moonlight_without_rope_scaling(self):
        model_config = DeepSeekV3Bridge().hf_config_to_model_config(
            _hf_config(**_MOONLIGHT_OVERRIDES)
        )

        assert model_config.q_lora_rank is None
        assert model_config.rotary_scaling_factor == 1.0
        assert model_config.mscale_all_dim == 1.0
        assert model_config.moe_aux_loss_coeff == 0.001
        assert model_config.moe_layer_freq == [0] + [1] * 60

    def test_model_config_round_trips(self):
        model_config = DeepSeekV3Bridge().hf_config_to_model_config(
            _hf_config(**_MOONLIGHT_OVERRIDES)
        )

        restored = ModelConfig.from_dict(model_config.as_dict())

        assert isinstance(restored, BridgeGPTMLAModelConfig)
        assert isinstance(restored.transformer, MLATransformerConfig)
        assert restored.transformer.rotary_base == 50000.0
        assert restored.q_lora_rank is None
        assert restored.moe_layer_freq == model_config.moe_layer_freq
        assert restored.transformer_layer_spec.func is get_deepseek_decoder_block_spec
        assert (
            restored.transformer_layer_spec.keywords == model_config.transformer_layer_spec.keywords
        )

    def test_megatron_to_hf_config_restores_the_moe_and_mla_layout(self):
        source = _hf_config()
        bridge = DeepSeekV3Bridge()

        exported = bridge.megatron_to_hf_config(bridge.hf_config_to_model_config(source))
        reloaded = DeepseekV3Config.from_dict(exported)

        assert exported["architectures"] == ["DeepseekV3ForCausalLM"]
        assert exported["model_type"] == "deepseek_v3"
        assert exported["first_k_dense_replace"] == 3
        assert exported["n_shared_experts"] == 1
        assert exported["num_nextn_predict_layers"] == 1
        for name in (
            "num_hidden_layers",
            "hidden_size",
            "intermediate_size",
            "moe_intermediate_size",
            "num_attention_heads",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "n_routed_experts",
            "num_experts_per_tok",
            "n_group",
            "topk_group",
            "routed_scaling_factor",
            "vocab_size",
            "max_position_embeddings",
            "rms_norm_eps",
            "tie_word_embeddings",
        ):
            assert getattr(reloaded, name) == getattr(source, name), name

    def test_megatron_to_hf_config_reports_disabled_features_as_integers(self):
        model_config = DeepSeekV3Bridge().hf_config_to_model_config(
            _hf_config(**_MOONLIGHT_OVERRIDES)
        )

        exported = DeepSeekV3Bridge.megatron_to_hf_config(model_config)

        assert exported["num_nextn_predict_layers"] == 0
        assert exported["n_group"] == 1
        assert exported["topk_group"] == 1
        assert exported["first_k_dense_replace"] == 1


class TestDeepSeekV3RopeScalingExport:
    """MLA's YaRN settings reach the exported config, not the reference model's."""

    @staticmethod
    def _exported(overrides: dict) -> dict:
        bridge = DeepSeekV3Bridge()
        return bridge.megatron_to_hf_config(
            bridge.hf_config_to_model_config(_hf_config(**overrides))
        )

    @classmethod
    def _rope_parameters_against(cls, overrides: dict, reference: DeepseekV3Config) -> dict:
        """The rope_parameters AutoBridge writes when it exports this model against ``reference``."""
        exported = cls._exported(overrides)
        conformed = conform_config_to_reference(exported, reference.to_dict())
        return carry_rope_into_rope_parameters(exported, conformed)["rope_parameters"]

    def test_yarn_scaling_is_exported(self):
        assert self._exported({})["rope_scaling"] == {
            "rope_type": "yarn",
            "factor": 40,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        }

    def test_a_model_without_scaling_exports_default_rope(self):
        assert self._exported(_MOONLIGHT_OVERRIDES)["rope_scaling"] == {"rope_type": "default"}

    def test_an_unchanged_model_keeps_the_reference_rope_parameters(self):
        reference = _hf_config()

        assert (
            self._rope_parameters_against({}, reference) == reference.to_dict()["rope_parameters"]
        )

    def test_the_model_scaling_replaces_a_different_reference_scaling(self):
        rope_scaling = dict(_DS_V3_CONFIG["rope_scaling"], factor=8, mscale_all_dim=0.5)

        rope_parameters = self._rope_parameters_against(
            {"rope_scaling": rope_scaling}, _hf_config()
        )

        assert rope_parameters["rope_type"] == "yarn"
        assert rope_parameters["factor"] == 8
        assert rope_parameters["mscale_all_dim"] == 0.5

    def test_a_model_without_scaling_drops_the_reference_scaling(self):
        rope_parameters = self._rope_parameters_against({"rope_scaling": None}, _hf_config())

        assert rope_parameters["rope_type"] == "default"
        assert "factor" not in rope_parameters
        assert rope_parameters["rope_theta"] == 10000

    def test_yarn_scaling_replaces_a_reference_without_scaling(self):
        reference = _hf_config(rope_scaling=None)

        rope_parameters = self._rope_parameters_against({}, reference)

        assert rope_parameters["rope_type"] == "yarn"
        assert rope_parameters["factor"] == 40
        assert rope_parameters["mscale_all_dim"] == 1.0


class TestDeepSeekV3FP8Import:
    """The base loader dequantizes FP8 weights with the checkpoint's declared block size."""

    @staticmethod
    def _bridge(weight_block_size: list[int] | None) -> DeepSeekV3Bridge:
        quantization_config = {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8"}
        if weight_block_size is not None:
            quantization_config["weight_block_size"] = weight_block_size
        bridge = DeepSeekV3Bridge()
        bridge.hf_config = SimpleNamespace(quantization_config=quantization_config)
        return bridge

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_non_fp8_weights_pass_through(self, dtype):
        weight = torch.randn(4, 4, dtype=dtype)

        result = self._bridge([128, 128])._load_hf_weights("layer.weight", {"layer.weight": weight})

        assert result is weight

    @pytest.mark.parametrize(
        ("weight_block_size", "block"),
        [([128, 128], 128), ([64, 64], 64), (None, 128)],
        ids=["declared_128", "declared_64", "undeclared"],
    )
    def test_each_declared_block_gets_its_own_scale(self, weight_block_size, block):
        state = {
            "layer.weight": torch.ones(2 * block, 2 * block, dtype=torch.float8_e4m3fn),
            "layer.weight_scale_inv": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        }

        result = self._bridge(weight_block_size)._load_hf_weights("layer.weight", state)

        assert torch.all(result[:block, :block] == 1.0)
        assert torch.all(result[:block, block:] == 2.0)
        assert torch.all(result[block:, :block] == 3.0)
        assert torch.all(result[block:, block:] == 4.0)

    def test_partial_trailing_blocks_use_their_block_scale(self):
        state = {
            "layer.weight": torch.ones(130, 200, dtype=torch.float8_e4m3fn),
            "layer.weight_scale_inv": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        }

        result = self._bridge([128, 128])._load_hf_weights("layer.weight", state)

        assert result.shape == (130, 200)
        assert torch.all(result[128:, 128:] == 4.0)
        assert torch.all(result[:128, 128:] == 2.0)

    def test_each_entry_of_a_compound_param_is_dequantized(self):
        up = torch.ones(64, 64, dtype=torch.bfloat16)
        state = {
            "gate.weight": torch.ones(128, 128, dtype=torch.float8_e4m3fn),
            "gate.weight_scale_inv": torch.full((1, 1), 2.0),
            "up.weight": up,
        }

        result = self._bridge([128, 128])._load_hf_weights(
            {"gate": "gate.weight", "up": "up.weight"}, state
        )

        assert torch.all(result["gate"] == 2.0)
        assert result["up"] is up


class TestDeepSeekV3MaybeModifyConvertedHFWeight:
    """Export-time aliases and persisted rotary frequencies."""

    @staticmethod
    def _bridge():
        bridge = DeepSeekV3Bridge()
        bridge.hf_config = SimpleNamespace(num_hidden_layers=61)
        return bridge

    @staticmethod
    def _task(global_name):
        return WeightConversionTask(
            param_name=global_name, global_param_name=global_name, mapping=Mock()
        )

    @pytest.mark.parametrize(
        ("global_name", "source_key", "alias_key"),
        [
            (
                "embedding.word_embeddings.weight",
                "model.embed_tokens.weight",
                "model.layers.61.embed_tokens.weight",
            ),
            ("output_layer.weight", "lm_head.weight", "model.layers.61.shared_head.head.weight"),
        ],
    )
    def test_restores_shared_mtp_alias(self, global_name, source_key, alias_key):
        weight = torch.randn(4, 4)

        result = self._bridge().maybe_modify_converted_hf_weight(
            SimpleNamespace(global_param_name=global_name),
            {source_key: weight},
            {alias_key: torch.empty(0)},
        )

        assert result[alias_key] is weight

    def test_does_not_add_mtp_alias_when_source_does_not_expect_it(self):
        converted = {"model.embed_tokens.weight": torch.randn(4, 4)}

        result = self._bridge().maybe_modify_converted_hf_weight(
            SimpleNamespace(global_param_name="embedding.word_embeddings.weight"), converted, {}
        )

        assert result is converted
        assert set(result) == {"model.embed_tokens.weight"}

    def test_preserves_the_source_inv_freq_of_the_same_layer(self):
        inv_key = "model.layers.0.self_attn.rotary_emb.inv_freq"
        source_inv_freq = torch.linspace(1.0, 0.0, 64, dtype=torch.bfloat16)
        converted = {"model.layers.0.input_layernorm.weight": torch.randn(1)}

        result = self._bridge().maybe_modify_converted_hf_weight(
            self._task("decoder.layers.0.input_layernorm.weight"),
            dict(converted),
            {inv_key: source_inv_freq},
        )

        assert torch.equal(result[inv_key], source_inv_freq)
        assert result[inv_key].dtype == source_inv_freq.dtype

    @pytest.mark.parametrize(
        ("global_name", "source"),
        [
            # Another layer's inv_freq must not be copied to layer 0.
            (
                "decoder.layers.0.input_layernorm.weight",
                {"model.layers.1.self_attn.rotary_emb.inv_freq": torch.randn(1)},
            ),
            # Only the input layernorm task carries the layer's inv_freq.
            (
                "decoder.final_layernorm.weight",
                {"model.layers.0.self_attn.rotary_emb.inv_freq": torch.randn(1)},
            ),
            # Nothing to preserve when the source does not persist it.
            ("decoder.layers.0.input_layernorm.weight", {}),
        ],
        ids=["other_layer", "not_input_layernorm", "not_in_source"],
    )
    def test_does_not_synthesize_inv_freq(self, global_name, source):
        result = self._bridge().maybe_modify_converted_hf_weight(
            self._task(global_name), {"x": torch.randn(1)}, source
        )

        assert "model.layers.0.self_attn.rotary_emb.inv_freq" not in result


class TestDeepSeekMappings:
    """Router expert_bias and MTP mappings in the shared DeepSeek-family mapping list."""

    def test_decoder_expert_bias_in_common_list(self):
        registry = MegatronMappingRegistry(*get_common_mapping_list())

        mapping = registry.megatron_to_hf_lookup("decoder.layers.5.mlp.router.expert_bias")

        assert mapping.hf_param == "model.layers.5.mlp.gate.e_score_correction_bias"

    def test_mtp_mappings_follow_the_decoder_layers(self):
        hf_config = SimpleNamespace(num_nextn_predict_layers=1, num_hidden_layers=61)
        registry = MegatronMappingRegistry(*get_common_mapping_list(hf_config=hf_config))

        expected = {
            "mtp.layers.0.mtp_model_layer.mlp.router.expert_bias": "model.layers.61.mlp.gate.e_score_correction_bias",
            "mtp.layers.0.enorm.weight": "model.layers.61.enorm.weight",
            "mtp.layers.0.hnorm.weight": "model.layers.61.hnorm.weight",
            "mtp.layers.0.eh_proj.weight": "model.layers.61.eh_proj.weight",
            "mtp.layers.0.final_layernorm.weight": "model.layers.61.shared_head.norm.weight",
        }
        for megatron_param, hf_param in expected.items():
            assert registry.megatron_to_hf_lookup(megatron_param).hf_param == hf_param

    def test_bridge_resolves_the_expert_bias(self):
        bridge = DeepSeekV3Bridge()
        bridge.hf_config = SimpleNamespace(num_nextn_predict_layers=0)

        mapping = bridge.mapping_registry().megatron_to_hf_lookup(
            "decoder.layers.0.mlp.router.expert_bias"
        )

        assert mapping.hf_param == "model.layers.0.mlp.gate.e_score_correction_bias"


def _toy_hf_config(**overrides) -> DeepseekV3Config:
    values = dict(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=3,
        first_k_dense_replace=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        vocab_size=128,
        max_position_embeddings=256,
        num_nextn_predict_layers=0,
        torch_dtype="bfloat16",
    )
    values.update(overrides)
    config = DeepseekV3Config(**values)
    config.architectures = ["DeepseekV3ForCausalLM"]
    return config


class TestDeepSeekV3AutoBridge:
    """AutoBridge selection and checkpoint import for DeepSeek-V3."""

    def test_supports_deepseek_v3_causal_lm(self):
        assert AutoBridge.supports(_toy_hf_config()) is True

    def test_import_ckpt_builds_through_model_builder(self, tmp_path):
        hf_path = tmp_path / "hf"
        _toy_hf_config(num_nextn_predict_layers=1).save_pretrained(hf_path)
        megatron_model = [Mock()]

        with (
            patch(
                "megatron.bridge.models.conversion.checkpoint.temporary_distributed_context",
                return_value=nullcontext(),
            ),
            patch.object(AutoBridge, "get_model", return_value=megatron_model) as mock_get_model,
            patch.object(AutoBridge, "save_megatron_model") as mock_save_megatron_model,
        ):
            AutoBridge.import_ckpt(hf_path, tmp_path / "megatron")

        (model_config,), build_kwargs = mock_get_model.call_args
        assert build_kwargs == {"wrap_with_ddp": False, "mixed_precision_wrapper": None}
        assert isinstance(model_config, BridgeGPTMLAModelConfig)
        assert model_config.use_cpu_initialization is True
        assert model_config.multi_latent_attention is True
        assert model_config.mtp_num_layers == 1
        assert model_config.moe_layer_freq == [0, 1, 1]
        assert mock_save_megatron_model.call_args.args[0] is megatron_model


def _fp8_quantize(weight: torch.Tensor, generator: torch.Generator):
    """Random FP8 codes and power-of-two block scales whose dequantized values are exact in BF16."""
    rows, columns = weight.shape
    codes = (torch.randn(rows, columns, generator=generator) * 2).to(torch.float8_e4m3fn)
    scale_inv = (
        2.0
        ** torch.randint(
            -8, -4, (math.ceil(rows / 128), math.ceil(columns / 128)), generator=generator
        ).float()
    )
    return codes, scale_inv


def _is_fp8_linear(name: str, tensor: torch.Tensor) -> bool:
    """DeepSeek-V3 checkpoints store every decoder linear projection in FP8, but not eh_proj."""
    return tensor.ndim == 2 and name.endswith("proj.weight") and not name.endswith("eh_proj.weight")


def _toy_checkpoint(directory, *, q_lora_rank, mtp: bool, fp8: bool):
    """Save a toy DeepSeek-V3 checkpoint; return the BF16 tensors its export must reproduce.

    The MTP layer is the extra last decoder layer of a model built with one more layer, plus the
    MTP-only norms and projection. Router expert biases are FP32, as in released checkpoints.
    Transformers saves only on rank 0 once torch.distributed is initialized, so call this on rank 0
    in that case.
    """
    num_layers = 3
    hf_config = _toy_hf_config(q_lora_rank=q_lora_rank, num_hidden_layers=num_layers + int(mtp))
    torch.manual_seed(1234)
    # Transformers fuses the experts in memory and writes per-expert tensors, like released
    # checkpoints, only when saving.
    DeepseekV3ForCausalLM(hf_config).to(torch.bfloat16).save_pretrained(directory / "saved")
    state = load_file(directory / "saved" / "model.safetensors")
    generator = torch.Generator().manual_seed(4321)
    for name in [name for name in state if name.endswith("e_score_correction_bias")]:
        state[name] = torch.randn(state[name].shape, generator=generator, dtype=torch.float32)
    if mtp:
        hidden = hf_config.hidden_size
        prefix = f"model.layers.{num_layers}."
        state[prefix + "enorm.weight"] = torch.randn(hidden, generator=generator).to(torch.bfloat16)
        state[prefix + "hnorm.weight"] = torch.randn(hidden, generator=generator).to(torch.bfloat16)
        state[prefix + "shared_head.norm.weight"] = torch.randn(hidden, generator=generator).to(
            torch.bfloat16
        )
        state[prefix + "eh_proj.weight"] = (
            torch.randn(hidden, 2 * hidden, generator=generator).to(torch.bfloat16) * 0.02
        )
        state[prefix + "embed_tokens.weight"] = state["model.embed_tokens.weight"].clone()
        state[prefix + "shared_head.head.weight"] = state["lm_head.weight"].clone()

    expected = dict(state)
    if fp8:
        for name in [name for name in expected if _is_fp8_linear(name, expected[name])]:
            codes, scale_inv = _fp8_quantize(expected[name], generator)
            state[name] = codes
            state[f"{name}_scale_inv"] = scale_inv
            expected[name] = quantization_utils.dequantize_fp8_scale_inv(codes, scale_inv).to(
                torch.bfloat16
            )
        hf_config.quantization_config = {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        }

    hf_config.num_hidden_layers = num_layers
    hf_config.num_nextn_predict_layers = int(mtp)
    checkpoint = directory / "checkpoint"
    hf_config.save_pretrained(checkpoint)
    save_file(state, checkpoint / "model.safetensors")
    return checkpoint, expected


class TestDeepSeekV3CheckpointCoverage:
    """Every tensor of a released-format checkpoint maps to a Megatron parameter."""

    @pytest.mark.parametrize("q_lora_rank", [32, None], ids=["query_lora", "without_query_lora"])
    def test_every_source_tensor_has_a_mapping(self, tmp_path, q_lora_rank):
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            pytest.skip(
                "Transformers saves the toy checkpoint only on rank 0, which runs this check."
            )
        checkpoint, _ = _toy_checkpoint(tmp_path, q_lora_rank=q_lora_rank, mtp=True, fp8=True)
        bridge = DeepSeekV3Bridge()
        bridge.hf_config = DeepseekV3Config.from_pretrained(checkpoint)
        registry = bridge.mapping_registry()
        # Dequantization consumes the scales; export writes the shared MTP aliases from the
        # embedding and output layer instead of importing them.
        not_imported = ("_scale_inv", ".embed_tokens.weight", ".shared_head.head.weight")
        source_keys = load_file(checkpoint / "model.safetensors").keys()

        unmapped = [
            key
            for key in source_keys
            if not (key.startswith("model.layers.") and key.endswith(not_imported))
            and key.endswith("_scale_inv") is False
            and registry.hf_to_megatron_lookup(key) is None
        ]

        assert unmapped == []
        assert any(key.endswith("_scale_inv") for key in source_keys)
        assert "model.layers.3.mlp.experts.0.gate_proj.weight" in source_keys


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Builds a Megatron model on GPU.")
class TestDeepSeekV3WeightRoundTrip:
    """HF -> Megatron -> HF weight round trips through Megatron-LM's GPT builder.

    The model is built like the conversion CLI builds it, without the BF16 module wrapper, so the
    FP32 router expert bias keeps its source precision.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        ("q_lora_rank", "mtp", "fp8"),
        [(32, True, False), (None, False, False), (32, True, True)],
        ids=["bf16_mtp", "bf16_without_query_lora", "fp8_mtp"],
    )
    def test_checkpoint_round_trips_exactly(self, tmp_path_dist_ckpt, q_lora_rank, mtp, fp8):
        # Transformers saves only on rank 0 once torch.distributed is initialized, so rank 0 writes
        # the checkpoint and the tensors its export must reproduce for every rank to read.
        with TempNamedDir(tmp_path_dist_ckpt / "deepseek_v3_round_trip") as directory:
            if torch.distributed.get_rank() == 0:
                _, expected = _toy_checkpoint(directory, q_lora_rank=q_lora_rank, mtp=mtp, fp8=fp8)
                save_file(expected, directory / "expected.safetensors")
            torch.distributed.barrier()
            expected = load_file(directory / "expected.safetensors")
            bridge = AutoBridge.from_hf_pretrained(directory / "checkpoint")
            models = bridge.get_model(wrap_with_ddp=False, mixed_precision_wrapper=None)
            exported = dict(bridge.export_hf_weights(models, cpu=True, show_progress=False))

        assert set(exported) == set(expected)
        for name, tensor in exported.items():
            assert tensor.dtype == expected[name].dtype, name
            torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0, msg=name)
