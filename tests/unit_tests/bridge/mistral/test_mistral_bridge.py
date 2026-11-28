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

"""Unit tests for the builder-backed Mistral bridge."""

import re
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F
from transformers import MistralConfig, MistralForCausalLM

from megatron.bridge.models import AutoBridge
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.transformers_compat import rope_theta_from_hf
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.mistral.mistral_bridge import MistralBridge
from megatron.bridge.models.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def _small_mistral_config(**kwargs) -> MistralConfig:
    values = dict(
        architectures=["MistralForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        vocab_size=128,
        sliding_window=32,
    )
    values.update(kwargs)
    return MistralConfig(**values)


def _gpt_model_config(
    *,
    seq_length: int,
    vocab_size: int,
    rotary_base: float = 10000.0,
    share_embeddings_and_output_weights: bool = False,
    **transformer_kwargs,
) -> BridgeGPTModelConfig:
    return BridgeGPTModelConfig(
        transformer=TransformerConfig(**transformer_kwargs),
        seq_length=seq_length,
        vocab_size=vocab_size,
        rotary_base=rotary_base,
        share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        position_embedding_type="rope",
        rope_scaling=False,
        rope_scaling_factor=1.0,
    )


class TestMistralBridgeModelConfig:
    """Test cases for MistralBridge's Hugging Face to builder config conversion."""

    @pytest.fixture
    def mistral_config_dict(self):
        """Create a sample Mistral configuration with sliding window and explicit head_dim."""
        return {
            "architectures": ["MistralForCausalLM"],
            "attention_dropout": 0.0,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 14336,
            "max_position_embeddings": 32768,
            "model_type": "mistral",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_theta": 1000000.0,
            "sliding_window": 4096,
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
            "vocab_size": 32768,
        }

    @pytest.fixture
    def mistral_config(self, mistral_config_dict):
        """Create a MistralConfig instance."""
        return MistralConfig(**mistral_config_dict)

    def test_bridge_registration(self):
        """Test that MistralBridge is properly registered."""
        assert issubclass(MistralBridge, MegatronModelBridge)

    def test_conversion_uses_builder_config(self):
        """Mistral converts through the builder-backed GPT config, with no provider path."""
        assert MistralBridge.MODEL_CONFIG_CLASS is BridgeGPTModelConfig
        assert "provider_bridge" not in MistralBridge.__dict__
        assert "hf_config_to_provider_kwargs" not in MistralBridge.__dict__
        assert "USE_MODEL_CONFIG_FOR_CONVERSION" not in MistralBridge.__dict__

    def test_hf_config_to_model_config_maps_architecture(self, mistral_config):
        """Architecture fields map from the Hugging Face config to the builder config."""
        result = MistralBridge().hf_config_to_model_config(mistral_config)

        assert isinstance(result, BridgeGPTModelConfig)
        assert result.num_layers == mistral_config.num_hidden_layers
        assert result.hidden_size == mistral_config.hidden_size
        assert result.ffn_hidden_size == mistral_config.intermediate_size
        assert result.num_attention_heads == mistral_config.num_attention_heads
        assert result.num_query_groups == mistral_config.num_key_value_heads
        assert result.kv_channels == mistral_config.head_dim
        assert result.seq_length == mistral_config.max_position_embeddings
        assert result.rotary_base == rope_theta_from_hf(mistral_config)
        assert result.vocab_size == mistral_config.vocab_size
        assert result.layernorm_epsilon == mistral_config.rms_norm_eps
        assert result.init_method_std == mistral_config.initializer_range
        assert result.share_embeddings_and_output_weights == mistral_config.tie_word_embeddings
        assert result.activation_func is F.silu

    def test_model_config_mistral_defaults_applied(self, mistral_config):
        """Mistral-specific settings are applied on top of the shared mapping."""
        result = MistralBridge().hf_config_to_model_config(mistral_config)

        assert result.normalization == "RMSNorm"
        assert result.gated_linear_unit is True
        assert result.position_embedding_type == "rope"
        assert result.add_bias_linear is False
        assert result.attention_dropout == 0.0
        assert result.hidden_dropout == 0.0
        assert result.masked_softmax_fusion is True
        assert result.rope_scaling is False
        assert result.rope_scaling_factor == 1.0

    @pytest.mark.parametrize(
        ("torch_dtype", "expected_dtype", "expected_bf16", "expected_fp16"),
        [("bfloat16", torch.bfloat16, True, False), ("float16", torch.float16, False, True)],
    )
    def test_model_config_dtype_handling(
        self, mistral_config_dict, torch_dtype, expected_dtype, expected_bf16, expected_fp16
    ):
        """Checkpoint dtype maps to params_dtype and precision flags."""
        config_dict = dict(mistral_config_dict, torch_dtype=torch_dtype)

        result = MistralBridge().hf_config_to_model_config(MistralConfig(**config_dict))

        assert result.params_dtype == expected_dtype
        assert result.bf16 is expected_bf16
        assert result.fp16 is expected_fp16

    def test_model_config_head_dim_independent_of_hidden_size(self):
        """Explicit head_dim maps to kv_channels even when it differs from hidden_size // num_heads."""
        config = _small_mistral_config(hidden_size=5120, num_attention_heads=32, head_dim=128)

        result = MistralBridge().hf_config_to_model_config(config)

        assert result.kv_channels == 128

    @pytest.mark.parametrize("tie_word_embeddings", [False, True])
    def test_model_config_embedding_sharing(self, tie_word_embeddings):
        """Tied embeddings follow the checkpoint."""
        config = _small_mistral_config(tie_word_embeddings=tie_word_embeddings)

        result = MistralBridge().hf_config_to_model_config(config)

        assert result.share_embeddings_and_output_weights is tie_word_embeddings

    def test_model_config_rope_theta_from_rope_parameters(self):
        """RoPE base frequency is extracted from transformers 5.0+ rope_parameters."""
        config = SimpleNamespace(
            architectures=["MistralForCausalLM"],
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=256,
            vocab_size=128,
            initializer_range=0.02,
            rms_norm_eps=1e-05,
            hidden_act="silu",
            rope_parameters={"rope_theta": 1000000.0},
            sliding_window=None,
            tie_word_embeddings=False,
            torch_dtype="bfloat16",
        )

        result = MistralBridge().hf_config_to_model_config(config)

        assert result.rotary_base == 1000000.0

    def test_model_config_sliding_window(self, mistral_config):
        """Sliding window translates to MCore's inclusive [w - 1, 0] window with a2a CP comm."""
        result = MistralBridge().hf_config_to_model_config(mistral_config)

        assert result.window_size == [4095, 0]
        assert result.cp_comm_type == "a2a"

    def test_model_config_sliding_window_none(self):
        """Disabled sliding window leaves window_size and cp_comm_type unset."""
        config = _small_mistral_config(sliding_window=None)

        result = MistralBridge().hf_config_to_model_config(config)

        assert result.window_size is None
        assert result.cp_comm_type is None

    def test_model_config_maps_yarn_to_transformer_config(self):
        """YaRN checkpoints keep YaRN on the builder path and export it back to Hugging Face."""
        config = _small_mistral_config(
            max_position_embeddings=131072,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            },
        )

        result = MistralBridge().hf_config_to_model_config(config)

        assert result.position_embedding_type == "yarn"
        assert result.rope_scaling is False
        assert result.transformer.yarn_rotary_scaling_factor == 4.0
        assert result.transformer.yarn_original_max_position_embeddings == 32768
        # Absent keys take the Hugging Face YaRN defaults rather than None.
        assert result.transformer.yarn_beta_fast == 32.0
        assert result.transformer.yarn_beta_slow == 1.0
        assert result.transformer.yarn_correction_range_round_to_int is True

        hf_config = MistralBridge.megatron_to_hf_config(result)
        assert hf_config["rope_scaling"] == {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "truncate": True,
        }

    @pytest.mark.parametrize(
        "rope_scaling",
        [
            {"rope_type": "dynamic", "factor": 2.0},
            {"rope_type": "linear", "factor": 2.0},
            {
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
        ],
        ids=["dynamic", "linear", "llama3"],
    )
    def test_rejects_rope_types_megatron_core_cannot_apply(self, rope_scaling):
        """Unsupported RoPE scaling types raise instead of silently converting to unscaled RoPE."""
        config = _small_mistral_config(max_position_embeddings=131072, rope_scaling=rope_scaling)
        rope_type = rope_scaling["rope_type"]

        with pytest.raises(ValueError, match=re.escape(f"Unsupported RoPE type {rope_type!r}.")):
            MistralBridge().hf_config_to_model_config(config)


class TestMistralBridgeMegatronToHFConfig:
    """Test cases for Megatron -> HF config conversion."""

    def test_megatron_to_hf_config_basic(self):
        """Test basic Megatron to HF config conversion."""
        model_config = _gpt_model_config(
            num_layers=32,
            hidden_size=4096,
            ffn_hidden_size=14336,
            num_attention_heads=32,
            num_query_groups=8,
            kv_channels=128,
            init_method_std=0.02,
            layernorm_epsilon=1e-05,
            seq_length=32768,
            rotary_base=1000000.0,
            vocab_size=32768,
            share_embeddings_and_output_weights=False,
            window_size=[4095, 0],
            cp_comm_type="a2a",
            bf16=True,
            fp16=False,
        )

        hf_config = MistralBridge.megatron_to_hf_config(model_config)

        assert hf_config["architectures"] == ["MistralForCausalLM"]
        assert hf_config["model_type"] == "mistral"
        assert hf_config["num_hidden_layers"] == 32
        assert hf_config["hidden_size"] == 4096
        assert hf_config["intermediate_size"] == 14336
        assert hf_config["num_attention_heads"] == 32
        assert hf_config["num_key_value_heads"] == 8
        assert hf_config["head_dim"] == 128
        assert hf_config["rms_norm_eps"] == 1e-05
        assert hf_config["rope_theta"] == 1000000.0
        assert hf_config["vocab_size"] == 32768
        assert hf_config["tie_word_embeddings"] is False
        assert hf_config["sliding_window"] == 4096
        assert hf_config["rope_scaling"] == {"rope_type": "default"}
        assert hf_config["torch_dtype"] == "bfloat16"

    def test_sliding_window_round_trip(self):
        """Sliding window round-trips through MCore's [w - 1, 0] representation."""
        source = _small_mistral_config(sliding_window=4096, rope_theta=1000000.0)
        model_config = MistralBridge().hf_config_to_model_config(source)

        assert model_config.window_size == [4095, 0]
        assert model_config.cp_comm_type == "a2a"

        exported = MistralBridge.megatron_to_hf_config(model_config)

        assert exported["sliding_window"] == 4096
        assert exported["rope_theta"] == 1000000.0

    def test_sliding_window_none_round_trip(self):
        """sliding_window=None round-trips to None without setting a window."""
        source = _small_mistral_config(sliding_window=None)
        model_config = MistralBridge().hf_config_to_model_config(source)

        assert model_config.window_size is None
        assert model_config.cp_comm_type is None

        exported = MistralBridge.megatron_to_hf_config(model_config)

        assert exported["sliding_window"] is None


class TestMistralBridgeMappingRegistry:
    """Test cases for MistralBridge.mapping_registry method."""

    @pytest.mark.parametrize(
        ("megatron_param", "hf_param"),
        [
            ("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
            ("output_layer.weight", "lm_head.weight"),
            ("decoder.final_layernorm.weight", "model.norm.weight"),
            (
                "decoder.layers.0.self_attention.linear_qkv.layer_norm_weight",
                "model.layers.0.input_layernorm.weight",
            ),
            ("decoder.layers.0.input_layernorm.weight", "model.layers.0.input_layernorm.weight"),
            (
                "decoder.layers.1.mlp.linear_fc1.layer_norm_weight",
                "model.layers.1.post_attention_layernorm.weight",
            ),
            (
                "decoder.layers.1.pre_mlp_layernorm.weight",
                "model.layers.1.post_attention_layernorm.weight",
            ),
            (
                "decoder.layers.0.self_attention.linear_proj.weight",
                "model.layers.0.self_attn.o_proj.weight",
            ),
            ("decoder.layers.1.mlp.linear_fc2.weight", "model.layers.1.mlp.down_proj.weight"),
        ],
    )
    def test_one_to_one_mappings(self, megatron_param, hf_param):
        """One-to-one parameters resolve to their Hugging Face names."""
        mapping = MistralBridge().mapping_registry().megatron_to_hf_lookup(megatron_param)

        assert mapping is not None
        assert mapping.hf_param == hf_param

    def test_fused_mappings(self):
        """QKV and the gated MLP combine their Hugging Face projections."""
        registry = MistralBridge().mapping_registry()

        qkv = registry.megatron_to_hf_lookup("decoder.layers.3.self_attention.linear_qkv.weight")
        assert qkv.hf_param == {
            "q": "model.layers.3.self_attn.q_proj.weight",
            "k": "model.layers.3.self_attn.k_proj.weight",
            "v": "model.layers.3.self_attn.v_proj.weight",
        }
        fc1 = registry.megatron_to_hf_lookup("decoder.layers.3.mlp.linear_fc1.weight")
        assert fc1.hf_param == {
            "gate": "model.layers.3.mlp.gate_proj.weight",
            "up": "model.layers.3.mlp.up_proj.weight",
        }


class TestMistralAutoBridge:
    """AutoBridge selection and checkpoint import for Mistral."""

    def test_supports_mistral_architectures(self):
        """AutoBridge supports Mistral causal LMs only."""
        assert AutoBridge.supports(_small_mistral_config()) is True

        non_causal_config = Mock()
        non_causal_config.architectures = ["MistralModel"]
        assert AutoBridge.supports(non_causal_config) is False

    def test_get_model_config_from_hf_config(self):
        """AutoBridge returns the builder-backed config for a Mistral checkpoint config."""
        config = _small_mistral_config(
            sliding_window=4096, rope_theta=1000000.0, head_dim=16, tie_word_embeddings=True
        )

        model_config = AutoBridge.from_hf_config(config).get_model_config()

        assert isinstance(model_config, BridgeGPTModelConfig)
        assert model_config.window_size == [4095, 0]
        assert model_config.cp_comm_type == "a2a"
        assert model_config.rotary_base == 1000000.0
        assert model_config.kv_channels == 16
        assert model_config.share_embeddings_and_output_weights is True

    def test_import_ckpt_builds_through_model_builder(self, tmp_path):
        """Checkpoint import builds Mistral through get_model() and keeps sliding window and tied embeddings."""
        hf_path = tmp_path / "hf"
        _small_mistral_config(
            sliding_window=4096, rope_theta=1000000.0, tie_word_embeddings=True
        ).save_pretrained(hf_path)
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
        assert isinstance(model_config, BridgeGPTModelConfig)
        assert model_config.use_cpu_initialization is True
        assert model_config.window_size == [4095, 0]
        assert model_config.cp_comm_type == "a2a"
        assert model_config.rotary_base == 1000000.0
        assert model_config.share_embeddings_and_output_weights is True
        assert mock_save_megatron_model.call_args.args[0] is megatron_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Builds a Megatron model on GPU.")
class TestMistralWeightRoundTrip:
    """HF -> Megatron -> HF weight round trip through Megatron-LM's GPT builder."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("tie_word_embeddings", [False, True], ids=["untied", "tied"])
    def test_hf_weights_round_trip_exactly(self, tmp_path_dist_ckpt, tie_word_embeddings):
        config = _small_mistral_config(
            sliding_window=32, rope_theta=1000000.0, tie_word_embeddings=tie_word_embeddings
        )
        torch.manual_seed(1234)
        hf_model = MistralForCausalLM(config).to(torch.bfloat16)
        expected = {name: tensor.detach().cpu() for name, tensor in hf_model.state_dict().items()}
        if tie_word_embeddings:
            # The tied head is the embedding; the checkpoint stores it once.
            expected.pop("lm_head.weight")

        # Transformers saves only on rank 0 once torch.distributed is initialized, so every rank
        # reads the checkpoint from one shared directory.
        with TempNamedDir(tmp_path_dist_ckpt / "mistral_round_trip") as hf_path:
            if torch.distributed.get_rank() == 0:
                hf_model.save_pretrained(hf_path)
            torch.distributed.barrier()
            bridge = AutoBridge.from_hf_pretrained(hf_path)
            models = bridge.get_model(wrap_with_ddp=False)
            exported = {
                name: tensor
                for name, tensor in bridge.export_hf_weights(models, cpu=True, show_progress=False)
            }

        assert isinstance(models[0].model_config, BridgeGPTModelConfig)
        assert models[0].model_config.window_size == [31, 0]
        assert set(exported) == set(expected)
        for name, tensor in exported.items():
            assert tensor.dtype == expected[name].dtype, name
            torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0, msg=name)
