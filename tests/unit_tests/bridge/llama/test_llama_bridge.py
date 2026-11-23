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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c209ed397fb36272c2216053058cda4e37802f2d/tests/unit_tests/models/llama/test_llama_bridge.py
# Source license: Apache-2.0.

"""Unit tests for the builder-backed Llama bridge."""

import re
from contextlib import nullcontext
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from megatron.bridge.models import AutoBridge
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.transformers_compat import rope_theta_from_hf
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.llama.llama_bridge import LlamaBridge
from megatron.bridge.models.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

_LLAMA3_ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


def _small_llama_config(**kwargs) -> LlamaConfig:
    values = dict(
        architectures=["LlamaForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        vocab_size=128,
    )
    values.update(kwargs)
    return LlamaConfig(**values)


def _gpt_model_config(
    *,
    seq_length: int,
    vocab_size: int,
    rotary_base: float = 10000,
    share_embeddings_and_output_weights: bool = False,
    rope_scaling: bool = False,
    rope_scaling_factor: float = 8.0,
    seq_len_interpolation_factor: float | None = None,
    **transformer_kwargs,
) -> BridgeGPTModelConfig:
    return BridgeGPTModelConfig(
        transformer=TransformerConfig(**transformer_kwargs),
        seq_length=seq_length,
        vocab_size=vocab_size,
        rotary_base=rotary_base,
        share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        position_embedding_type="rope",
        rope_scaling=rope_scaling,
        rope_scaling_factor=rope_scaling_factor,
        seq_len_interpolation_factor=seq_len_interpolation_factor,
    )


class TestLlamaBridgeModelConfig:
    """Test cases for LlamaBridge's Hugging Face to builder config conversion."""

    @pytest.fixture
    def llama_3_2_1b_config_dict(self):
        """Create a sample Llama 3.2 1B configuration with RoPE scaling."""
        return {
            "architectures": ["LlamaForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 128000,
            "eos_token_id": 128001,
            "head_dim": 64,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "intermediate_size": 8192,
            "max_position_embeddings": 131072,
            "mlp_bias": False,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 16,
            "num_key_value_heads": 8,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": dict(_LLAMA3_ROPE_SCALING),
            "rope_parameters": {"rope_type": "llama3", "rope_theta": 500000.0},
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
            "transformers_version": "5.0.0",
            "use_cache": True,
            "vocab_size": 128256,
        }

    @pytest.fixture
    def llama_config(self, llama_3_2_1b_config_dict):
        """Create a LlamaConfig instance for Llama 3.2 1B."""
        return LlamaConfig(**llama_3_2_1b_config_dict)

    def test_bridge_registration(self):
        """Test that LlamaBridge is properly registered."""
        assert issubclass(LlamaBridge, MegatronModelBridge)

    def test_conversion_uses_builder_config(self):
        """Llama converts through the builder-backed GPT config, with no provider path."""
        assert LlamaBridge.MODEL_CONFIG_CLASS is BridgeGPTModelConfig
        assert "provider_bridge" not in LlamaBridge.__dict__

    def test_hf_config_to_model_config_maps_architecture(self, llama_config):
        """Architecture fields map from the Hugging Face config to the builder config."""
        result = LlamaBridge().hf_config_to_model_config(llama_config)

        assert isinstance(result, BridgeGPTModelConfig)
        assert result.num_layers == llama_config.num_hidden_layers
        assert result.hidden_size == llama_config.hidden_size
        assert result.ffn_hidden_size == llama_config.intermediate_size
        assert result.num_attention_heads == llama_config.num_attention_heads
        assert result.num_query_groups == llama_config.num_key_value_heads
        assert result.seq_length == llama_config.max_position_embeddings
        assert result.rotary_base == rope_theta_from_hf(llama_config)
        assert result.vocab_size == llama_config.vocab_size
        assert result.layernorm_epsilon == llama_config.rms_norm_eps
        assert result.init_method_std == llama_config.initializer_range
        assert result.attention_dropout == llama_config.attention_dropout
        assert result.add_qkv_bias == llama_config.attention_bias
        assert result.activation_func is F.silu

    def test_model_config_llama_defaults_applied(self, llama_config):
        """Llama-specific settings are applied on top of the shared mapping."""
        result = LlamaBridge().hf_config_to_model_config(llama_config)

        assert result.normalization == "RMSNorm"
        assert result.gated_linear_unit is True
        assert result.position_embedding_type == "rope"
        assert result.hidden_dropout == 0.0
        assert result.bias_activation_fusion is True
        assert result.masked_softmax_fusion is True
        assert result.persist_layer_norm is True
        assert result.bias_dropout_fusion is True
        assert result.apply_rope_fusion is True
        assert result.rotary_percent == 1.0
        assert result.add_bias_linear is False
        assert result.attention_dropout == 0.0

    def test_model_config_dtype_handling_bfloat16(self, llama_config):
        """A bfloat16 checkpoint builds a bfloat16 model."""
        result = LlamaBridge().hf_config_to_model_config(llama_config)

        assert result.params_dtype == torch.bfloat16
        assert result.bf16 is True
        assert result.fp16 is False

    def test_model_config_dtype_handling_float16(self, llama_3_2_1b_config_dict):
        """A float16 checkpoint builds a float16 model."""
        config_dict = dict(llama_3_2_1b_config_dict, torch_dtype="float16")

        result = LlamaBridge().hf_config_to_model_config(LlamaConfig(**config_dict))

        assert result.params_dtype == torch.float16
        assert result.fp16 is True
        assert result.bf16 is False

    def test_model_config_rope_scaling_params(self, llama_config):
        """llama3 RoPE scaling maps to Megatron Core's built-in scaling."""
        result = LlamaBridge().hf_config_to_model_config(llama_config)

        assert result.rope_scaling is True
        assert result.rope_scaling_factor == 32.0
        assert result.seq_len_interpolation_factor is None
        assert result.rotary_base == rope_theta_from_hf(llama_config)

    def test_model_config_embedding_sharing(self, llama_config):
        """Tied embeddings follow the checkpoint."""
        result = LlamaBridge().hf_config_to_model_config(llama_config)

        assert result.share_embeddings_and_output_weights == llama_config.tie_word_embeddings

    def test_model_config_head_dim(self, llama_config):
        """head_dim maps to kv_channels."""
        result = LlamaBridge().hf_config_to_model_config(llama_config)

        assert result.kv_channels == 64

    def test_model_config_preserves_linear_rope_scaling(self):
        """Linear RoPE scaling maps to the builder config."""
        config = _small_llama_config(rope_scaling={"type": "linear", "factor": 8.0})

        result = LlamaBridge().hf_config_to_model_config(config)

        assert result.position_embedding_type == "rope"
        assert result.seq_len_interpolation_factor == 8.0
        assert result.rope_scaling is False
        assert result.rope_scaling_factor == 1.0

    def test_model_config_maps_yarn_to_transformer_config(self):
        """YaRN checkpoints keep YaRN on the builder path and export it back to Hugging Face."""
        config = _small_llama_config(
            max_position_embeddings=131072,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            },
        )

        result = LlamaBridge().hf_config_to_model_config(config)

        assert result.position_embedding_type == "yarn"
        assert result.rope_scaling is False
        assert result.seq_len_interpolation_factor is None
        assert result.transformer.yarn_rotary_scaling_factor == 4.0
        assert result.transformer.yarn_original_max_position_embeddings == 32768
        # Absent keys take the Hugging Face YaRN defaults rather than None.
        assert result.transformer.yarn_beta_fast == 32.0
        assert result.transformer.yarn_beta_slow == 1.0
        assert result.transformer.yarn_correction_range_round_to_int is True

        hf_config = LlamaBridge.megatron_to_hf_config(result)
        assert hf_config["rope_scaling"] == {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "truncate": True,
        }

        roundtrip = LlamaBridge().hf_config_to_model_config(LlamaConfig(**hf_config))
        assert roundtrip.position_embedding_type == "yarn"
        for field_name in (
            "yarn_rotary_scaling_factor",
            "yarn_original_max_position_embeddings",
            "yarn_beta_fast",
            "yarn_beta_slow",
            "yarn_mscale",
            "yarn_mscale_all_dim",
            "yarn_correction_range_round_to_int",
        ):
            assert getattr(roundtrip.transformer, field_name) == getattr(
                result.transformer, field_name
            ), field_name

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("low_freq_factor", 2.0),
            ("high_freq_factor", 8.0),
            ("original_max_position_embeddings", 4096),
        ],
    )
    def test_rejects_llama3_rope_scaling_megatron_core_cannot_apply(self, key, value):
        """llama3 values other than Megatron Core's fixed ones raise instead of converting silently."""
        rope_scaling = dict(_LLAMA3_ROPE_SCALING, factor=8.0)
        rope_scaling[key] = value
        config = _small_llama_config(max_position_embeddings=131072, rope_scaling=rope_scaling)
        expected = re.escape(f"Unsupported llama3 RoPE scaling values {{'{key}': {value}}}.")

        with pytest.raises(ValueError, match=expected):
            LlamaBridge().hf_config_to_model_config(config)

    def test_rejects_rope_types_megatron_core_cannot_apply(self):
        """Dynamic NTK scaling must not convert silently to unscaled RoPE."""
        config = _small_llama_config(rope_scaling={"rope_type": "dynamic", "factor": 2.0})

        with pytest.raises(ValueError, match=re.escape("Unsupported RoPE type 'dynamic'.")):
            LlamaBridge().hf_config_to_model_config(config)


class TestLlamaBridgeMegatronToHFConfig:
    """Test cases for Megatron -> HF config conversion."""

    def test_megatron_to_hf_config_basic(self):
        """Test basic Megatron to HF config conversion."""
        model_config = _gpt_model_config(
            num_layers=32,
            hidden_size=4096,
            ffn_hidden_size=11008,
            num_attention_heads=32,
            num_query_groups=32,
            init_method_std=0.02,
            layernorm_epsilon=1e-05,
            seq_length=4096,
            rotary_base=10000,
            vocab_size=32000,
            share_embeddings_and_output_weights=False,
            bf16=False,
            fp16=False,
        )

        hf_config = LlamaBridge.megatron_to_hf_config(model_config)

        assert hf_config["architectures"] == ["LlamaForCausalLM"]
        assert hf_config["model_type"] == "llama"
        assert hf_config["num_hidden_layers"] == 32
        assert hf_config["hidden_size"] == 4096
        assert hf_config["intermediate_size"] == 11008
        assert hf_config["num_attention_heads"] == 32
        assert hf_config["num_key_value_heads"] == 32
        assert hf_config["rms_norm_eps"] == 1e-05
        assert hf_config["rope_theta"] == 10000
        assert hf_config["vocab_size"] == 32000
        assert hf_config["tie_word_embeddings"] is False
        assert hf_config["torch_dtype"] == "float32"

    def test_megatron_to_hf_config_with_rope_scaling(self):
        """Test Megatron to HF config conversion with llama3 RoPE scaling."""
        model_config = _gpt_model_config(
            num_layers=16,
            hidden_size=2048,
            ffn_hidden_size=8192,
            num_attention_heads=32,
            num_query_groups=8,
            init_method_std=0.02,
            layernorm_epsilon=1e-05,
            seq_length=131072,
            rotary_base=500000,
            vocab_size=128256,
            share_embeddings_and_output_weights=True,
            bf16=True,
            rope_scaling=True,
            rope_scaling_factor=32.0,
        )

        hf_config = LlamaBridge.megatron_to_hf_config(model_config)

        assert hf_config["torch_dtype"] == "bfloat16"
        assert hf_config["tie_word_embeddings"] is True
        assert hf_config["rope_scaling"] == _LLAMA3_ROPE_SCALING

    def test_megatron_to_hf_config_with_linear_rope_scaling(self):
        """Test Megatron to HF config conversion with linear RoPE scaling."""
        model_config = _gpt_model_config(
            num_layers=2,
            hidden_size=64,
            ffn_hidden_size=128,
            num_attention_heads=4,
            num_query_groups=4,
            seq_length=256,
            vocab_size=128,
            seq_len_interpolation_factor=8.0,
        )

        hf_config = LlamaBridge.megatron_to_hf_config(model_config)

        assert hf_config["rope_scaling"] == {"rope_type": "linear", "factor": 8.0}

    def test_megatron_to_hf_config_without_rope_scaling_reports_default(self):
        """A model without RoPE scaling says so, so a config-only export can drop a reference's scaling."""
        model_config = _gpt_model_config(
            num_layers=2,
            hidden_size=64,
            ffn_hidden_size=128,
            num_attention_heads=4,
            num_query_groups=4,
            seq_length=256,
            vocab_size=128,
        )

        hf_config = LlamaBridge.megatron_to_hf_config(model_config)

        assert hf_config["rope_scaling"] == {"rope_type": "default"}

    def test_llama3_scaling_round_trips_through_the_builder_config(self):
        """A Llama 3.1 config exports the same llama3 RoPE scaling it was imported with."""
        config = _small_llama_config(
            max_position_embeddings=131072,
            rope_theta=500000.0,
            rope_scaling=dict(_LLAMA3_ROPE_SCALING, factor=8.0),
        )

        hf_config = LlamaBridge.megatron_to_hf_config(
            LlamaBridge().hf_config_to_model_config(config)
        )

        assert hf_config["rope_scaling"] == dict(_LLAMA3_ROPE_SCALING, factor=8.0)
        assert hf_config["rope_theta"] == 500000.0


class TestLlamaBridgeMappingRegistry:
    """Test cases for LlamaBridge.mapping_registry method."""

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
            (
                "decoder.layers.1.mlp.linear_fc1.layer_norm_weight",
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
        mapping = LlamaBridge().mapping_registry().megatron_to_hf_lookup(megatron_param)

        assert mapping is not None
        assert mapping.hf_param == hf_param

    def test_fused_mappings(self):
        """QKV and the gated MLP combine their Hugging Face projections."""
        registry = LlamaBridge().mapping_registry()

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

    def test_export_preserves_persisted_rotary_inv_freq(self):
        """Serialized rotary buffers are copied from the source checkpoint."""
        bridge = LlamaBridge()
        task = Mock(
            global_param_name="decoder.layers.3.self_attention.linear_qkv.layer_norm_weight"
        )
        converted = {"model.layers.3.input_layernorm.weight": torch.ones(4096)}
        inv_freq_key = "model.layers.3.self_attn.rotary_emb.inv_freq"
        source_inv_freq = torch.tensor([1.0, 0.25, 0.03125], dtype=torch.float32)

        result = bridge.maybe_modify_converted_hf_weight(
            task, converted, {inv_freq_key: source_inv_freq}
        )

        torch.testing.assert_close(result[inv_freq_key], source_inv_freq, rtol=0, atol=0)

    def test_export_does_not_add_unpersisted_rotary_inv_freq(self):
        """Checkpoints without persisted RoPE buffers are unchanged."""
        bridge = LlamaBridge()
        task = Mock(
            global_param_name="decoder.layers.3.self_attention.linear_qkv.layer_norm_weight"
        )
        converted = {"model.layers.3.input_layernorm.weight": torch.ones(4096)}

        result = bridge.maybe_modify_converted_hf_weight(task, converted, {})

        assert set(result) == {"model.layers.3.input_layernorm.weight"}


class TestLlamaAutoBridge:
    """AutoBridge selection and checkpoint import for Llama."""

    def test_supports_llama_architectures(self):
        """AutoBridge supports Llama causal LMs only."""
        assert AutoBridge.supports(_small_llama_config()) is True

        non_causal_config = Mock()
        non_causal_config.architectures = ["LlamaModel"]
        assert AutoBridge.supports(non_causal_config) is False

    def test_get_model_config_from_hf_config(self):
        """AutoBridge returns the builder-backed config for a Llama 3.1 checkpoint config."""
        config = _small_llama_config(
            max_position_embeddings=131072,
            rope_theta=500000.0,
            rope_scaling=dict(_LLAMA3_ROPE_SCALING, factor=8.0),
        )

        model_config = AutoBridge.from_hf_config(config).get_model_config()

        assert isinstance(model_config, BridgeGPTModelConfig)
        assert model_config.rope_scaling is True
        assert model_config.rope_scaling_factor == 8.0
        assert model_config.rotary_base == 500000.0

    def test_import_ckpt_builds_through_model_builder(self, tmp_path):
        """Checkpoint import builds Llama through get_model() and keeps RoPE scaling and tied embeddings."""
        hf_path = tmp_path / "hf"
        _small_llama_config(
            num_key_value_heads=2,
            max_position_embeddings=131072,
            rope_theta=500000.0,
            rope_scaling=dict(_LLAMA3_ROPE_SCALING),
            tie_word_embeddings=True,
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
        assert model_config.position_embedding_type == "rope"
        assert model_config.rope_scaling is True
        assert model_config.rope_scaling_factor == 32.0
        assert model_config.share_embeddings_and_output_weights is True
        assert mock_save_megatron_model.call_args.args[0] is megatron_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Builds a Megatron model on GPU.")
class TestLlamaWeightRoundTrip:
    """HF -> Megatron -> HF weight round trip through Megatron-LM's GPT builder."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("tie_word_embeddings", [False, True], ids=["untied", "tied"])
    def test_hf_weights_round_trip_exactly(self, tmp_path_dist_ckpt, tie_word_embeddings):
        config = _small_llama_config(
            num_key_value_heads=2,
            max_position_embeddings=131072,
            rope_theta=500000.0,
            rope_scaling=dict(_LLAMA3_ROPE_SCALING, factor=8.0),
            tie_word_embeddings=tie_word_embeddings,
        )
        torch.manual_seed(1234)
        hf_model = LlamaForCausalLM(config).to(torch.bfloat16)
        expected = {name: tensor.detach().cpu() for name, tensor in hf_model.state_dict().items()}
        if tie_word_embeddings:
            # The tied head is the embedding; the checkpoint stores it once.
            expected.pop("lm_head.weight")

        # Transformers saves only on rank 0 once torch.distributed is initialized, so every rank
        # reads the checkpoint from one shared directory.
        with TempNamedDir(tmp_path_dist_ckpt / "llama_round_trip") as hf_path:
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
        assert models[0].model_config.rope_scaling is True
        assert set(exported) == set(expected)
        for name, tensor in exported.items():
            assert tensor.dtype == expected[name].dtype, name
            torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0, msg=name)
