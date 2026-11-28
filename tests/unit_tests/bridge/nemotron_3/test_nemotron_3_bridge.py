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
# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/tests/unit_tests/models/nemotronh/test_nemotron_h_bridge.py
# Source license: Apache-2.0.
# Follow-up fixes from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/b57951ad292e43209f1cc7d4b5500768365a20ac/tests/unit_tests/models/nemotronh/test_nemotron_h_bridge.py

import dataclasses
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from transformers.configuration_utils import PretrainedConfig

from megatron.bridge.models import AutoBridge
from megatron.bridge.models.conversion.auto_bridge import _drop_readonly_config_properties
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, QKVMapping
from megatron.bridge.models.conversion.quant_mapping import AmaxFanoutMapping, AmaxMapping
from megatron.bridge.models.conversion.utils import conform_config_to_reference
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hybrid import BridgeHybridModelConfig
from megatron.bridge.models.nemotron_3.nemotron_3_bridge import (
    Nemotron3Bridge,
    _MTPFlatteningMapping,
    _MTPFlatteningQKVMapping,
    _replace_wildcards,
)
from megatron.core.activations import squared_relu


class TestNemotron3Bridge:
    """Test cases for Nemotron3Bridge class."""

    @pytest.fixture
    def active_nemotronh_config_dict(self):
        """Create a sample active NemotronH configuration."""
        return {
            "architectures": ["NemotronHForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attention_head_dim": 128,
            "auto_map": {
                "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
                "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM",
            },
            "bos_token_id": 1,
            "chunk_size": 128,
            "conv_kernel": 4,
            "eos_token_id": 2,
            "expand": 2,
            "hidden_act": "relu2",  # Required for base class activation mapping
            "hidden_dropout": 0.0,
            "hidden_size": 2688,
            "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
            "initializer_range": 0.02,
            "intermediate_size": 1856,
            "layer_norm_epsilon": 1e-05,
            "mamba_head_dim": 64,
            "mamba_hidden_act": "silu",
            "mamba_num_heads": 64,
            "mamba_proj_bias": False,
            "max_position_embeddings": 8192,
            "mlp_bias": False,
            "mlp_hidden_act": "relu2",
            "model_type": "nemotron_h",
            "n_groups": 8,
            "num_attention_heads": 32,
            "num_hidden_layers": 52,
            "num_key_value_heads": 2,
            "num_logits_to_keep": 1,
            "pad_token_id": 0,
            "rescale_prenorm_residual": True,
            "residual_in_fp32": False,
            "rms_norm_eps": 1e-05,
            "ssm_state_size": 128,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.48.0.dev0",
            "use_bias": False,
            "use_cache": True,
            "use_conv_bias": True,
            "use_mamba_kernels": True,
            "vocab_size": 131072,
            # Explicitly set to None to disable MoE; Mock objects return Mock for any attr access,
            # so hasattr() always returns True.
            "n_routed_experts": None,
        }

    @pytest.fixture
    def mock_nemotronh_config(self, active_nemotronh_config_dict):
        """Create mock config instance.

        Uses spec=[] to make getattr return None for undefined attributes
        instead of Mock objects, which would incorrectly be passed to the provider.
        """
        cfg = Mock(spec=[])
        for k, v in active_nemotronh_config_dict.items():
            setattr(cfg, k, v)
        return cfg

    @pytest.fixture
    def mock_pretrained_nemotronh(self, mock_nemotronh_config):
        """Create a mock PreTrainedCausalLM with NemotronH model."""

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_nemotronh_config
        return mock_pretrained

    def test_bridge_registration(self):
        """Test that Nemotron3Bridge is properly registered."""
        assert issubclass(Nemotron3Bridge, MegatronModelBridge)

    def test_model_config_basic(self, mock_pretrained_nemotronh, mock_nemotronh_config):
        """Test basic config conversion."""
        bridge = Nemotron3Bridge()

        # Convert the HF config
        result = bridge.hf_config_to_model_config(mock_pretrained_nemotronh.config)
        result.finalize()

        # Check that it returns a BridgeHybridModelConfig instance
        assert isinstance(result, BridgeHybridModelConfig)

        # Check basic configuration mapping
        assert result.num_layers == mock_nemotronh_config.num_hidden_layers
        assert result.hybrid_layer_pattern == mock_nemotronh_config.hybrid_override_pattern
        assert result.hidden_size == mock_nemotronh_config.hidden_size
        assert result.add_bias_linear == mock_nemotronh_config.use_bias
        assert result.num_attention_heads == mock_nemotronh_config.num_attention_heads
        assert result.seq_length == mock_nemotronh_config.max_position_embeddings

    def test_model_config_vocabulary(self, mock_pretrained_nemotronh, mock_nemotronh_config):
        """Test vocabulary size mapping."""
        bridge = Nemotron3Bridge()

        result = bridge.hf_config_to_model_config(mock_pretrained_nemotronh.config)

        # Check vocabulary configuration
        assert result.vocab_size == mock_nemotronh_config.vocab_size
        assert (
            result.share_embeddings_and_output_weights == mock_nemotronh_config.tie_word_embeddings
        )

    def test_model_config_attention_config(self, mock_pretrained_nemotronh, mock_nemotronh_config):
        """Test attention configuration mapping."""
        bridge = Nemotron3Bridge()

        result = bridge.hf_config_to_model_config(mock_pretrained_nemotronh.config)

        # Check attention configuration
        assert result.num_attention_heads == mock_nemotronh_config.num_attention_heads
        assert result.num_query_groups == mock_nemotronh_config.num_key_value_heads

    def test_model_config_mamba_config(self, mock_pretrained_nemotronh, mock_nemotronh_config):
        """Test Mamba-specific configuration mapping."""
        bridge = Nemotron3Bridge()

        result = bridge.hf_config_to_model_config(mock_pretrained_nemotronh.config)
        result.finalize()

        # Check Mamba-specific configuration
        assert result.mamba_state_dim == mock_nemotronh_config.ssm_state_size
        assert result.mamba_head_dim == mock_nemotronh_config.mamba_head_dim
        assert result.mamba_num_heads == mock_nemotronh_config.mamba_num_heads
        assert result.mamba_num_groups == mock_nemotronh_config.n_groups
        assert result.mamba_chunk_size == mock_nemotronh_config.chunk_size
        assert result.hybrid_layer_pattern == mock_nemotronh_config.hybrid_override_pattern

    def test_model_config_mlp_config(self, mock_pretrained_nemotronh, mock_nemotronh_config):
        """Test MLP configuration mapping."""
        bridge = Nemotron3Bridge()

        result = bridge.hf_config_to_model_config(mock_pretrained_nemotronh.config)

        # Check MLP configuration
        assert result.ffn_hidden_size == mock_nemotronh_config.intermediate_size
        assert result.gated_linear_unit == False  # Mamba doesn't use gated linear units

    def test_model_config_normalization(self, mock_pretrained_nemotronh, mock_nemotronh_config):
        """Test normalization configuration."""
        bridge = Nemotron3Bridge()

        result = bridge.hf_config_to_model_config(mock_pretrained_nemotronh.config)

        # Check normalization settings
        assert result.layernorm_epsilon == mock_nemotronh_config.layer_norm_epsilon

    def test_model_config_dtype_handling(self, mock_nemotronh_config):
        """Test dtype handling in config conversion."""
        # Create model with specific dtype
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_nemotronh_config.torch_dtype = "bfloat16"
        mock_pretrained.config = mock_nemotronh_config

        bridge = Nemotron3Bridge()
        result = bridge.hf_config_to_model_config(mock_pretrained.config)

        # The provider should respect the model's dtype
        assert result.params_dtype == torch.bfloat16
        assert result.bf16 == True
        assert result.fp16 == False

    def test_mapping_registry_implementation(self, mock_pretrained_nemotronh):
        """Test that mapping_registry returns a proper MegatronMappingRegistry."""
        bridge = Nemotron3Bridge()

        # Get the mapping registry
        mapping_registry = bridge.mapping_registry()

        # Check it's not None
        assert mapping_registry is not None
        assert any([isinstance(m, AutoMapping) for m in mapping_registry.mappings])
        # assert any([isinstance(m, PrunedVocabMapping) for m in mapping_registry.mappings])
        assert any([isinstance(m, QKVMapping) for m in mapping_registry.mappings])

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "hf_embedding_key", ["backbone.embedding.weight", "backbone.embeddings.weight"]
    )
    def test_embedding_mapping_preserves_checkpoint_key(self, hf_embedding_key: str) -> None:
        bridge = Nemotron3Bridge()
        bridge.hf_pretrained = SimpleNamespace(
            state=SimpleNamespace(
                source=SimpleNamespace(get_all_keys=lambda: [hf_embedding_key, "lm_head.weight"])
            )
        )

        registry = bridge.mapping_registry()
        megatron_name = "embedding.word_embeddings.weight"
        mapping = registry.megatron_to_hf_lookup(megatron_name)

        assert mapping.hf_param == hf_embedding_key
        assert registry.hf_to_megatron_lookup(hf_embedding_key).megatron_param == megatron_name
        assert not mapping.allow_hf_name_mismatch
        assert bridge._validate_conversion_mappings(registry, [megatron_name], [hf_embedding_key])

    @pytest.mark.unit
    def test_config_only_embedding_mapping_keeps_legacy_name(self) -> None:
        registry = Nemotron3Bridge().mapping_registry()

        assert (
            registry.megatron_to_hf_lookup("embedding.word_embeddings.weight").hf_param
            == "backbone.embeddings.weight"
        )

    @pytest.mark.unit
    def test_missing_embedding_remains_a_conversion_error(self) -> None:
        bridge = Nemotron3Bridge()
        bridge.hf_pretrained = SimpleNamespace(
            state=SimpleNamespace(source=SimpleNamespace(get_all_keys=lambda: ["lm_head.weight"]))
        )

        with pytest.raises(ValueError, match="Hugging Face checkpoint is missing mapped parameter"):
            bridge._validate_conversion_mappings(
                bridge.mapping_registry(), ["embedding.word_embeddings.weight"], ["lm_head.weight"]
            )

    def test_mapping_registry_contains_mamba_conv1d_compat_mappings(self):
        """Test that Mamba conv mappings support old and new Megatron-Core names."""
        registry = Nemotron3Bridge().mapping_registry()

        new_weight = registry.megatron_to_hf_lookup("decoder.layers.0.mixer.conv1d_weight")
        new_bias = registry.megatron_to_hf_lookup("decoder.layers.0.mixer.conv1d_bias")
        old_weight = registry.megatron_to_hf_lookup("decoder.layers.0.mixer.conv1d.weight")
        old_bias = registry.megatron_to_hf_lookup("decoder.layers.0.mixer.conv1d.bias")

        assert new_weight.hf_param == "backbone.layers.0.mixer.conv1d.weight"
        assert new_bias.hf_param == "backbone.layers.0.mixer.conv1d.bias"
        assert old_weight.hf_param == "backbone.layers.0.mixer.conv1d.weight"
        assert old_bias.hf_param == "backbone.layers.0.mixer.conv1d.bias"

        reverse_weight = registry.hf_to_megatron_lookup("backbone.layers.0.mixer.conv1d.weight")
        reverse_bias = registry.hf_to_megatron_lookup("backbone.layers.0.mixer.conv1d.bias")
        assert reverse_weight.megatron_param == "decoder.layers.0.mixer.conv1d_weight"
        assert reverse_bias.megatron_param == "decoder.layers.0.mixer.conv1d_bias"

    def test_model_config_fixed_settings(self, mock_pretrained_nemotronh):
        """Test fixed settings that should always be set regardless of config."""
        bridge = Nemotron3Bridge()

        result = bridge.hf_config_to_model_config(mock_pretrained_nemotronh.config)

        # These should always be set to these values for Mamba
        assert result.position_embedding_type == "none"  # Mamba doesn't use position embeddings
        assert result.rotary_percent == 1.0
        assert result.rotary_base == 10000

    def test_model_config_moe_config(self, active_nemotronh_config_dict):
        """Test MoE configuration mapping when n_routed_experts > 0."""
        # Add MoE-specific configurations to the base config
        moe_config_dict = {
            **active_nemotronh_config_dict,
            "n_routed_experts": 64,
            "moe_intermediate_size": 2048,
            "moe_shared_expert_intermediate_size": 8192,
            "num_experts_per_tok": 8,
            "n_group": 4,
            "topk_group": 2,
            "routed_scaling_factor": 2.0,
        }

        cfg = Mock(spec=[])
        for k, v in moe_config_dict.items():
            setattr(cfg, k, v)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = cfg

        bridge = Nemotron3Bridge()
        result = bridge.hf_config_to_model_config(mock_pretrained.config)

        # Check MoE configuration mappings
        assert result.num_moe_experts == cfg.n_routed_experts
        assert result.moe_ffn_hidden_size == cfg.moe_intermediate_size
        assert result.moe_shared_expert_intermediate_size == cfg.moe_shared_expert_intermediate_size
        assert result.moe_router_topk == cfg.num_experts_per_tok
        assert result.moe_router_num_groups == cfg.n_group
        assert result.moe_router_group_topk == cfg.topk_group
        assert result.moe_router_topk_scaling_factor == cfg.routed_scaling_factor

    def test_model_config_no_moe_when_n_routed_experts_zero(self, active_nemotronh_config_dict):
        """Test that MoE configs are not added when n_routed_experts is 0."""
        # Add MoE config with n_routed_experts = 0
        moe_config_dict = {
            **active_nemotronh_config_dict,
            "n_routed_experts": 0,
            "moe_intermediate_size": 2048,
            "moe_shared_expert_intermediate_size": 8192,
            "num_experts_per_tok": 8,
            "n_group": 4,
            "topk_group": 2,
            "routed_scaling_factor": 2.0,
        }

        cfg = Mock(spec=[])
        for k, v in moe_config_dict.items():
            setattr(cfg, k, v)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = cfg

        bridge = Nemotron3Bridge()
        result = bridge.hf_config_to_model_config(mock_pretrained.config)

        # When n_routed_experts is 0, num_moe_experts should be 0 or None
        assert result.num_moe_experts in (0, None)

    def test_model_config_moe_latent_size(self, active_nemotronh_config_dict):
        """Test moe_latent_size mapping for Super model."""
        moe_config_dict = {
            **active_nemotronh_config_dict,
            "n_routed_experts": 512,
            "moe_intermediate_size": 2688,
            "moe_shared_expert_intermediate_size": 5376,
            "num_experts_per_tok": 22,
            "n_group": 1,
            "topk_group": 1,
            "routed_scaling_factor": 5.0,
            "moe_latent_size": 1024,
            "moe_shared_expert_overlap": False,
        }

        cfg = Mock(spec=[])
        for k, v in moe_config_dict.items():
            setattr(cfg, k, v)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = cfg

        bridge = Nemotron3Bridge()
        result = bridge.hf_config_to_model_config(mock_pretrained.config)

        # Check Super-specific MoE configuration
        assert result.moe_latent_size == 1024
        assert result.moe_shared_expert_overlap is False

    def test_model_config_mtp_config(self, active_nemotronh_config_dict):
        """Test MTP configuration mapping for Super model."""
        mtp_config_dict = {
            **active_nemotronh_config_dict,
            "n_routed_experts": 0,
            "num_nextn_predict_layers": 2,
            "mtp_hybrid_override_pattern": "*E",
            "keep_mtp_spec_in_bf16": True,
        }

        cfg = Mock(spec=[])
        for k, v in mtp_config_dict.items():
            setattr(cfg, k, v)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = cfg

        bridge = Nemotron3Bridge()
        result = bridge.hf_config_to_model_config(mock_pretrained.config)

        # Check MTP configuration mappings
        assert result.mtp_num_layers == 2
        assert result.mtp_hybrid_override_pattern == "*E"
        assert result.mtp_use_repeated_layer is True
        assert result.mtp_loss_scaling_factor == 0.3

    @pytest.mark.parametrize("num_nextn_predict_layers", [None, 0])
    def test_model_config_disables_mtp_naturally(
        self, active_nemotronh_config_dict, num_nextn_predict_layers
    ):
        """HF configs without enabled MTP produce a normal non-MTP provider."""
        from types import SimpleNamespace

        config_dict = {
            **active_nemotronh_config_dict,
            "n_routed_experts": 0,
            "mtp_hybrid_override_pattern": "*E",
            "keep_mtp_spec_in_bf16": True,
        }
        if num_nextn_predict_layers is not None:
            config_dict["num_nextn_predict_layers"] = num_nextn_predict_layers

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = SimpleNamespace(**config_dict)

        result = Nemotron3Bridge().hf_config_to_model_config(mock_pretrained.config)

        assert result.mtp_num_layers == 0
        assert result.mtp_hybrid_override_pattern is None
        assert result.mtp_use_repeated_layer is False

    def test_model_config_rejects_incomplete_mtp_config(self, active_nemotronh_config_dict):
        """An enabled HF MTP config must describe its hybrid block."""
        from types import SimpleNamespace

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = SimpleNamespace(
            **{**active_nemotronh_config_dict, "n_routed_experts": 0, "num_nextn_predict_layers": 1}
        )

        with pytest.raises(ValueError, match="mtp_hybrid_override_pattern"):
            Nemotron3Bridge().hf_config_to_model_config(mock_pretrained.config)

    def test_model_config_no_moe_when_attribute_missing(self, active_nemotronh_config_dict):
        """Test that MoE configs are not added when n_routed_experts attribute is missing."""
        from types import SimpleNamespace

        # Create config without n_routed_experts using SimpleNamespace (hasattr returns False for missing attrs)
        config_dict = {
            k: v for k, v in active_nemotronh_config_dict.items() if k != "n_routed_experts"
        }
        cfg = SimpleNamespace(**config_dict)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = cfg

        bridge = Nemotron3Bridge()
        result = bridge.hf_config_to_model_config(mock_pretrained.config)

        # Should work without MoE configs - provider should still be created
        assert isinstance(result, BridgeHybridModelConfig)
        assert not hasattr(result, "num_moe_experts") or result.num_moe_experts is None

    def test_mapping_registry_contains_moe_mappings(self):
        """Test that mapping_registry contains MoE parameter mappings."""
        bridge = Nemotron3Bridge()
        mapping_registry = bridge.mapping_registry()

        # Get all megatron params from mappings
        megatron_params = [
            m.megatron_param for m in mapping_registry.mappings if hasattr(m, "megatron_param")
        ]

        # Check MoE router mappings exist
        assert "decoder.layers.*.mlp.router.weight" in megatron_params
        assert "decoder.layers.*.mlp.router.expert_bias" in megatron_params

        # Check MoE expert mappings exist
        assert "decoder.layers.*.mlp.experts.linear_fc1.weight*" in megatron_params
        assert "decoder.layers.*.mlp.experts.linear_fc2.weight*" in megatron_params

        # Check shared expert mappings exist
        assert "decoder.layers.*.mlp.shared_experts.linear_fc1.weight" in megatron_params
        assert "decoder.layers.*.mlp.shared_experts.linear_fc2.weight" in megatron_params

        # Check pre_mlp_layernorm mapping exists
        assert "decoder.layers.*.pre_mlp_layernorm.weight" in megatron_params

    def test_mapping_registry_contains_latent_proj_mappings(self):
        """Test that mapping_registry contains latent projection mappings for Super model."""
        bridge = Nemotron3Bridge()
        mapping_registry = bridge.mapping_registry()

        # Get all megatron params from mappings
        megatron_params = [
            m.megatron_param for m in mapping_registry.mappings if hasattr(m, "megatron_param")
        ]

        # Check latent projection mappings exist (used by Super model)
        assert "decoder.layers.*.mlp.fc1_latent_proj.weight" in megatron_params
        assert "decoder.layers.*.mlp.fc2_latent_proj.weight" in megatron_params

    def test_mapping_registry_moe_hf_params(self):
        """Test that MoE mappings have correct HF parameter names."""
        bridge = Nemotron3Bridge()
        mapping_registry = bridge.mapping_registry()

        # Create a lookup dict of megatron -> hf params
        param_map = {
            m.megatron_param: m.hf_param
            for m in mapping_registry.mappings
            if hasattr(m, "megatron_param") and hasattr(m, "hf_param")
        }

        # Check MoE HF param mappings are correct
        assert (
            param_map.get("decoder.layers.*.mlp.router.weight")
            == "backbone.layers.*.mixer.gate.weight"
        )
        assert (
            param_map.get("decoder.layers.*.mlp.router.expert_bias")
            == "backbone.layers.*.mixer.gate.e_score_correction_bias"
        )
        assert (
            param_map.get("decoder.layers.*.mlp.experts.linear_fc1.weight*")
            == "backbone.layers.*.mixer.experts.*.up_proj.weight"
        )
        assert (
            param_map.get("decoder.layers.*.mlp.experts.linear_fc2.weight*")
            == "backbone.layers.*.mixer.experts.*.down_proj.weight"
        )
        assert (
            param_map.get("decoder.layers.*.mlp.shared_experts.linear_fc1.weight")
            == "backbone.layers.*.mixer.shared_experts.up_proj.weight"
        )
        assert (
            param_map.get("decoder.layers.*.mlp.shared_experts.linear_fc2.weight")
            == "backbone.layers.*.mixer.shared_experts.down_proj.weight"
        )

    def test_mapping_registry_latent_proj_hf_params(self):
        """Test that latent projection mappings have correct HF parameter names."""
        bridge = Nemotron3Bridge()
        mapping_registry = bridge.mapping_registry()

        # Create a lookup dict of megatron -> hf params
        param_map = {
            m.megatron_param: m.hf_param
            for m in mapping_registry.mappings
            if hasattr(m, "megatron_param") and hasattr(m, "hf_param")
        }

        # Check latent projection HF param mappings
        assert (
            param_map.get("decoder.layers.*.mlp.fc1_latent_proj.weight")
            == "backbone.layers.*.mixer.fc1_latent_proj.weight"
        )
        assert (
            param_map.get("decoder.layers.*.mlp.fc2_latent_proj.weight")
            == "backbone.layers.*.mixer.fc2_latent_proj.weight"
        )


class TestNemotron3BridgeTokenizerKwargs:
    """Test get_hf_tokenizer_kwargs method."""

    def test_tokenizer_kwargs_returns_dict(self):
        """Test get_hf_tokenizer_kwargs returns a dict."""
        kwargs = Nemotron3Bridge.get_hf_tokenizer_kwargs()
        assert isinstance(kwargs, dict)

    def test_tokenizer_kwargs_use_fast(self):
        """Test get_hf_tokenizer_kwargs returns use_fast=True."""
        kwargs = Nemotron3Bridge.get_hf_tokenizer_kwargs()
        assert kwargs.get("use_fast") is True


class TestNemotron3BridgeMegatronToHFConfig:
    """Test Megatron provider config export for Nemotron-H."""

    def test_megatron_to_hf_config_splits_unified_mtp_pattern(self):
        """Export HF hybrid_override_pattern without Megatron's unified MTP separator."""
        provider = SimpleNamespace(hybrid_layer_pattern="ME|ME/*E", mtp_num_layers=1)

        hf_cfg = Nemotron3Bridge.megatron_to_hf_config(provider)

        assert hf_cfg["hybrid_override_pattern"] == "MEME"
        assert hf_cfg["mtp_hybrid_override_pattern"] == "*E"
        assert hf_cfg["num_nextn_predict_layers"] == 1

    def test_megatron_to_hf_config_keeps_repeated_identical_mtp_pattern_separate(self):
        """Collapse repeated unified MTP blocks to the HF MTP block pattern field."""
        provider = SimpleNamespace(hybrid_layer_pattern="MEME/*E/*E", mtp_num_layers=2)

        hf_cfg = Nemotron3Bridge.megatron_to_hf_config(provider)

        assert hf_cfg["hybrid_override_pattern"] == "MEME"
        assert hf_cfg["mtp_hybrid_override_pattern"] == "*E"
        assert hf_cfg["num_nextn_predict_layers"] == 2

    @pytest.mark.parametrize(
        ("mtp_use_repeated_layer", "expected_num_nextn_predict_layers"), [(True, 1), (False, 2)]
    )
    def test_megatron_to_hf_config_exports_physical_mtp_layer_count(
        self, mtp_use_repeated_layer, expected_num_nextn_predict_layers
    ):
        """Export shared and independent MTP modules at their physical depths."""
        provider = SimpleNamespace(
            hybrid_layer_pattern="MEME/*E/*E",
            mtp_num_layers=2,
            mtp_use_repeated_layer=mtp_use_repeated_layer,
        )

        hf_cfg = Nemotron3Bridge.megatron_to_hf_config(provider)

        assert hf_cfg["hybrid_override_pattern"] == "MEME"
        assert hf_cfg["mtp_hybrid_override_pattern"] == "*E"
        assert hf_cfg["num_nextn_predict_layers"] == expected_num_nextn_predict_layers

    @pytest.mark.parametrize("depth", [0, 1, 2, 3])
    @pytest.mark.parametrize("repeated", [False, True])
    @pytest.mark.parametrize("text_only", [False, True])
    def test_export_preserves_physical_mtp_block_count(self, depth, repeated, text_only):
        provider = SimpleNamespace(
            hybrid_layer_pattern="MEME" + "/*E" * depth,
            mtp_num_layers=depth,
            mtp_use_repeated_layer=repeated,
            hf_model_text_only=text_only,
        )
        config = Nemotron3Bridge.megatron_to_hf_config(provider)
        physical_blocks = 1 if depth and repeated else depth
        assert config["num_nextn_predict_layers"] == physical_blocks
        # Reimport preserves stored blocks; the training recipe selects repetitions.
        config["mtp_use_repeated_layer"] = repeated
        reimported = Nemotron3Bridge._hf_mtp_settings(SimpleNamespace(**config))
        assert reimported["mtp_num_layers"] == physical_blocks
        assert reimported["mtp_use_repeated_layer"] is bool(depth and repeated)

    def test_megatron_to_hf_config_disables_mtp_with_zero_physical_layers(self):
        """Export zero physical MTP layers when MTP is disabled."""
        provider = SimpleNamespace(
            hybrid_layer_pattern="MEME", mtp_num_layers=0, mtp_use_repeated_layer=False
        )

        hf_cfg = Nemotron3Bridge.megatron_to_hf_config(provider)

        assert hf_cfg["num_nextn_predict_layers"] == 0

    def test_megatron_to_hf_config_rejects_unknown_main_pattern_characters(self):
        """Preserve validation for unknown main hybrid_override_pattern characters."""
        provider = SimpleNamespace(hybrid_layer_pattern="MEZ", mtp_num_layers=0)

        with pytest.raises(
            ValueError, match="Unknown layer type characters in hybrid_override_pattern"
        ):
            Nemotron3Bridge.megatron_to_hf_config(provider)

    def test_megatron_to_hf_config_rejects_unknown_mtp_pattern_characters(self):
        """Validate MTP block patterns split from Megatron's unified pattern."""
        provider = SimpleNamespace(hybrid_layer_pattern="ME/*Z", mtp_num_layers=1)

        with pytest.raises(
            ValueError, match="Unknown layer type characters in mtp_hybrid_override_pattern"
        ):
            Nemotron3Bridge.megatron_to_hf_config(provider)

    def test_megatron_to_hf_config_rejects_mismatched_mtp_patterns(self):
        """Unified MTP blocks must be identical when exported to a single HF MTP pattern."""
        provider = SimpleNamespace(hybrid_layer_pattern="ME/*E/*M", mtp_num_layers=2)

        with pytest.raises(
            ValueError, match="All MTP patterns in hybrid_override_pattern must be identical"
        ):
            Nemotron3Bridge.megatron_to_hf_config(provider)


class TestAutoBridgeIntegration:
    def test_megatron_to_hf_config_dense_checkpoint_builds_hf_config(self):
        """A dense model_config without MoE sizes yields a constructible HF config with reference MoE sizes."""
        from transformers import NemotronHConfig

        reference = NemotronHConfig().to_dict()
        model_config = SimpleNamespace(
            num_moe_experts=None,
            moe_ffn_hidden_size=None,
            moe_shared_expert_intermediate_size=None,
            moe_router_topk_scaling_factor=None,
        )

        hf_cfg = Nemotron3Bridge.megatron_to_hf_config(model_config)
        config_dict = _drop_readonly_config_properties(
            conform_config_to_reference(hf_cfg, reference), NemotronHConfig
        )
        config = NemotronHConfig(**config_dict)

        assert config.n_routed_experts == reference["n_routed_experts"]
        assert config.moe_intermediate_size == reference["moe_intermediate_size"]
        assert (
            config.moe_shared_expert_intermediate_size
            == reference["moe_shared_expert_intermediate_size"]
        )
        assert config.routed_scaling_factor == reference["routed_scaling_factor"]

    def test_megatron_to_hf_config_keeps_moe_sizes(self):
        """MoE sizes set on the model_config are exported unchanged."""
        model_config = SimpleNamespace(
            num_moe_experts=16,
            moe_ffn_hidden_size=512,
            moe_shared_expert_intermediate_size=1024,
            moe_router_topk_scaling_factor=2.5,
        )

        hf_cfg = Nemotron3Bridge.megatron_to_hf_config(model_config)

        assert hf_cfg["n_routed_experts"] == 16
        assert hf_cfg["moe_intermediate_size"] == 512
        assert hf_cfg["moe_shared_expert_intermediate_size"] == 1024
        assert hf_cfg["routed_scaling_factor"] == 2.5

    """Integration tests for AutoBridge with current NemotronH models."""

    @pytest.fixture
    def nemotronh_config_dict(self):
        """Create a sample active NemotronH configuration."""
        return {
            "architectures": ["NemotronHForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attention_head_dim": 128,
            "auto_map": {
                "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
                "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM",
            },
            "bos_token_id": 1,
            "chunk_size": 128,
            "conv_kernel": 4,
            "eos_token_id": 2,
            "expand": 2,
            "hidden_act": "relu2",  # Required for base class activation mapping
            "hidden_dropout": 0.0,
            "hidden_size": 2688,
            "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
            "initializer_range": 0.02,
            "intermediate_size": 1856,
            "layer_norm_epsilon": 1e-05,
            "mamba_head_dim": 64,
            "mamba_hidden_act": "silu",
            "mamba_num_heads": 64,
            "mamba_proj_bias": False,
            "max_position_embeddings": 8192,
            "mlp_bias": False,
            "mlp_hidden_act": "relu2",
            "model_type": "nemotron_h",
            "n_groups": 8,
            "num_attention_heads": 32,
            "num_hidden_layers": 52,
            "num_key_value_heads": 2,
            "num_logits_to_keep": 1,
            "pad_token_id": 0,
            "rescale_prenorm_residual": True,
            "residual_in_fp32": False,
            "rms_norm_eps": 1e-05,
            "ssm_state_size": 128,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.48.0.dev0",
            "use_bias": False,
            "use_cache": True,
            "use_conv_bias": True,
            "use_mamba_kernels": True,
            "vocab_size": 131072,
        }

    @pytest.fixture
    def nemotronh_config(self, nemotronh_config_dict):
        """Create mock config instance."""
        cfg = Mock(spec=PretrainedConfig)
        for k, v in nemotronh_config_dict.items():
            setattr(cfg, k, v)
        return cfg

    def create_mock_model_files(self, config_dict, save_dir):
        """Create mock model files in a directory."""
        import json

        # Save config
        config_path = Path(save_dir) / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Create a dummy safetensors index file
        index_path = Path(save_dir) / "model.safetensors.index.json"
        index_data = {
            "metadata": {"total_size": 1000000},
            "weight_map": {
                "backbone.embeddings.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.0.mixer.in_proj.weight": "model-00001-of-00001.safetensors",
            },
        }
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)

        # Create tokenizer files
        tokenizer_config = {"model_max_length": config_dict["max_position_embeddings"]}
        tokenizer_path = Path(save_dir) / "tokenizer_config.json"
        with open(tokenizer_path, "w") as f:
            json.dump(tokenizer_config, f, indent=2)

        # Create dummy tokenizer.json
        tokenizer_json_path = Path(save_dir) / "tokenizer.json"
        tokenizer_data = {"version": "1.0", "model": {"type": "BPE"}}
        with open(tokenizer_json_path, "w") as f:
            json.dump(tokenizer_data, f, indent=2)

    @patch("megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained")
    @patch("transformers.AutoConfig.from_pretrained")
    def test_from_pretrained_with_temp_dir(
        self, mock_autoconfig, mock_pretrained, nemotronh_config_dict, nemotronh_config
    ):
        """Test AutoBridge.from_hf_pretrained with temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dict = nemotronh_config_dict
            self.create_mock_model_files(config_dict, temp_dir)

            # Mock the config loading
            config = nemotronh_config
            mock_autoconfig.return_value = config

            # Mock the pretrained model
            mock_model = Mock(spec=PreTrainedCausalLM)
            mock_model.config = config
            mock_model.model_name_or_path = temp_dir
            mock_pretrained.return_value = mock_model

            # Create bridge from the temp directory
            bridge = AutoBridge.from_hf_pretrained(temp_dir)

            # Verify
            assert isinstance(bridge, AutoBridge)
            assert bridge.hf_pretrained == mock_model
            mock_autoconfig.assert_called_once_with(temp_dir, trust_remote_code=False)
            mock_pretrained.assert_called_once_with(temp_dir)

    @patch("megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained")
    @patch("transformers.AutoConfig.from_pretrained")
    def test_from_pretrained_with_kwargs(
        self, mock_autoconfig, mock_pretrained, nemotronh_config_dict, nemotronh_config
    ):
        """Test AutoBridge.from_hf_pretrained with various kwargs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dict = nemotronh_config_dict
            self.create_mock_model_files(config_dict, temp_dir)

            # Mock the config loading
            config = nemotronh_config
            mock_autoconfig.return_value = config

            # Mock the pretrained model
            mock_model = Mock(spec=PreTrainedCausalLM)
            mock_model.config = config
            mock_pretrained.return_value = mock_model

            # Test with various kwargs
            kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
                "trust_remote_code": True,
            }

            _ = AutoBridge.from_hf_pretrained(temp_dir, **kwargs)

            # Verify kwargs were passed through
            mock_pretrained.assert_called_once_with(temp_dir, **kwargs)

    def test_supports_nemotronh_architectures(self, nemotronh_config):
        """Test that AutoBridge.supports correctly identifies NemotronH models."""
        config = nemotronh_config
        assert AutoBridge.supports(config) == True

        # Test non-causal LM architecture
        non_causal_config = Mock()
        non_causal_config.architectures = ["NemotronHModel"]  # Not ForCausalLM
        assert AutoBridge.supports(non_causal_config) == False


class TestReplaceWildcards:
    """Test _replace_wildcards helper function."""

    def test_single_star(self):
        result = _replace_wildcards("decoder.layers.*.weight", ("5",))
        assert result == "decoder.layers.5.weight"

    def test_two_stars(self):
        result = _replace_wildcards("mtp.layers.*.mtp_model_layer.layers.*.weight", ("0", "1"))
        assert result == "mtp.layers.0.mtp_model_layer.layers.1.weight"

    def test_double_star_then_single(self):
        result = _replace_wildcards("mtp.layers.**.experts.*.weight", ("3", "7"))
        assert result == "mtp.layers.3.experts.7.weight"

    def test_fewer_captures_than_wildcards(self):
        result = _replace_wildcards("a.*.b.*.c", ("1",))
        assert result == "a.1.b.*.c"

    def test_no_wildcards(self):
        result = _replace_wildcards("decoder.final_norm.weight", ("0",))
        assert result == "decoder.final_norm.weight"

    def test_empty_captures(self):
        result = _replace_wildcards("decoder.layers.*.weight", ())
        assert result == "decoder.layers.*.weight"


class TestMTPFlatteningMapping:
    """Test _MTPFlatteningMapping class."""

    def test_init_basic(self):
        m = _MTPFlatteningMapping(
            megatron_param="mtp.layers.*.mtp_model_layer.layers.*.mlp.weight",
            hf_param="mtp.layers.*.mixer.up_proj.weight",
            mtp_layers_per_block=2,
        )
        assert m._mtp_layers_per_block == 2
        assert m._inner_override is None
        assert m._megatron_wc == 2
        assert m._hf_wc == 1

    def test_init_with_inner_override(self):
        m = _MTPFlatteningMapping(
            megatron_param="mtp.layers.*.eh_proj.weight",
            hf_param="mtp.layers.*.eh_proj.weight",
            mtp_layers_per_block=3,
            inner_override=0,
        )
        assert m._inner_override == 0

    def test_init_invalid_mtp_layers(self):
        with pytest.raises(ValueError, match="mtp_layers_per_block must be > 0"):
            _MTPFlatteningMapping(
                megatron_param="mtp.layers.*.weight",
                hf_param="mtp.layers.*.weight",
                mtp_layers_per_block=0,
            )

    def test_resolve_megatron_nested(self):
        """Test resolve with Megatron captures (outer, inner) for nested params."""
        m = _MTPFlatteningMapping(
            megatron_param="mtp.layers.*.mtp_model_layer.layers.*.mlp.weight",
            hf_param="mtp.layers.*.mixer.up_proj.weight",
            mtp_layers_per_block=3,
        )
        result = m.resolve(("1", "2"))  # outer=1, inner=2
        assert isinstance(result, AutoMapping)
        # flat = 1*3 + 2 = 5
        assert result.megatron_param == "mtp.layers.1.mtp_model_layer.layers.2.mlp.weight"
        assert result.hf_param == "mtp.layers.5.mixer.up_proj.weight"

    def test_resolve_megatron_with_inner_override(self):
        """Test resolve with inner_override (outer-only Megatron params like eh_proj)."""
        m = _MTPFlatteningMapping(
            megatron_param="mtp.layers.*.eh_proj.weight",
            hf_param="mtp.layers.*.eh_proj.weight",
            mtp_layers_per_block=2,
            inner_override=0,
        )
        result = m.resolve(("3",))  # outer=3, inner_override=0
        assert isinstance(result, AutoMapping)
        # flat = 3*2 + 0 = 6
        assert result.megatron_param == "mtp.layers.3.eh_proj.weight"
        assert result.hf_param == "mtp.layers.6.eh_proj.weight"

    def test_resolve_megatron_last_inner_override(self):
        """Test resolve with inner_override pointing to last inner layer."""
        m = _MTPFlatteningMapping(
            megatron_param="mtp.layers.*.final_layernorm.weight",
            hf_param="mtp.layers.*.final_layernorm.weight",
            mtp_layers_per_block=4,
            inner_override=3,  # last inner = L-1
        )
        result = m.resolve(("2",))  # outer=2
        # flat = 2*4 + 3 = 11
        assert result.hf_param == "mtp.layers.11.final_layernorm.weight"

    def test_resolve_hf_path(self):
        """Test resolve when captures come from HF reverse lookup (1 capture for 1 HF wildcard)."""
        m = _MTPFlatteningMapping(
            megatron_param="mtp.layers.*.mtp_model_layer.layers.*.mlp.weight",
            hf_param="mtp.layers.*.mixer.up_proj.weight",
            mtp_layers_per_block=3,
        )
        # HF lookup: 1 capture (flat idx) != 2 megatron wildcards → treat_as_hf=True
        result = m.resolve(("7",))  # flat=7 → outer=2, inner=1
        assert isinstance(result, AutoMapping)
        assert result.hf_param == "mtp.layers.7.mixer.up_proj.weight"
        assert result.megatron_param == "mtp.layers.2.mtp_model_layer.layers.1.mlp.weight"

    def test_resolve_hf_path_with_inner_override(self):
        """Test HF resolve with inner_override."""
        m = _MTPFlatteningMapping(
            megatron_param="mtp.layers.*.eh_proj.weight",
            hf_param="mtp.layers.*.eh_proj.weight",
            mtp_layers_per_block=2,
            inner_override=0,
        )
        # Both patterns have 1 wildcard, so hf_wc == megatron_wc → treat_as_hf=False
        # This will take the Megatron path with inner_override
        result = m.resolve(("4",))  # outer=4, inner_override=0
        assert isinstance(result, AutoMapping)
        # flat = 4*2 + 0 = 8
        assert result.hf_param == "mtp.layers.8.eh_proj.weight"

    def test_resolve_megatron_with_expert_captures(self):
        """Test resolve with extra captures (e.g., expert index)."""
        m = _MTPFlatteningMapping(
            megatron_param="mtp.layers.*.mtp_model_layer.layers.*.mlp.experts.linear_fc1.weight*",
            hf_param="mtp.layers.*.mixer.experts.*.up_proj.weight",
            mtp_layers_per_block=2,
        )
        # 3 megatron wildcards vs 2 hf wildcards
        result = m.resolve(("1", "0", "5"))  # outer=1, inner=0, expert=5
        assert isinstance(result, AutoMapping)
        # flat = 1*2 + 0 = 2
        assert result.hf_param == "mtp.layers.2.mixer.experts.5.up_proj.weight"

    def test_resolve_megatron_missing_inner_raises(self):
        """Test that resolve raises when inner capture is missing for nested mapping."""
        m = _MTPFlatteningMapping(
            megatron_param="mtp.layers.*.mtp_model_layer.layers.*.mlp.weight",
            hf_param="mtp.layers.*.mixer.up_proj.weight",
            mtp_layers_per_block=2,
        )
        # 1 capture but no inner_override and needs 2 captures
        # Note: 1 capture == hf_wc (1) and hf_wc != megatron_wc (2), so this takes HF path
        # To trigger the Megatron path error, need captures count != hf_wc
        # With 0 captures and hf_wc=1, megatron_wc=2: len(captures)=0 != hf_wc=1, so Megatron path
        with pytest.raises(ValueError, match="Expected at least 1 capture"):
            m.resolve(())

    def test_hf_to_megatron_raises(self):
        m = _MTPFlatteningMapping(
            megatron_param="mtp.layers.*.weight",
            hf_param="mtp.layers.*.weight",
            mtp_layers_per_block=1,
        )
        with pytest.raises(NotImplementedError):
            m.hf_to_megatron(torch.tensor([1.0]), torch.nn.Linear(1, 1))

    def test_megatron_to_hf_raises(self):
        m = _MTPFlatteningMapping(
            megatron_param="mtp.layers.*.weight",
            hf_param="mtp.layers.*.weight",
            mtp_layers_per_block=1,
        )
        with pytest.raises(NotImplementedError):
            m.megatron_to_hf(torch.tensor([1.0]), torch.nn.Linear(1, 1))


class TestMTPFlatteningQKVMapping:
    """Test _MTPFlatteningQKVMapping class."""

    def test_init_basic(self):
        m = _MTPFlatteningQKVMapping(
            megatron_param="mtp.layers.*.mtp_model_layer.layers.*.self_attention.linear_qkv.weight",
            q="mtp.layers.*.mixer.q_proj.weight",
            k="mtp.layers.*.mixer.k_proj.weight",
            v="mtp.layers.*.mixer.v_proj.weight",
            mtp_layers_per_block=2,
        )
        assert m._mtp_layers_per_block == 2
        assert m.hf_param == {
            "q": "mtp.layers.*.mixer.q_proj.weight",
            "k": "mtp.layers.*.mixer.k_proj.weight",
            "v": "mtp.layers.*.mixer.v_proj.weight",
        }

    def test_init_invalid_mtp_layers(self):
        with pytest.raises(ValueError, match="mtp_layers_per_block must be > 0"):
            _MTPFlatteningQKVMapping(
                megatron_param="mtp.layers.*.mtp_model_layer.layers.*.weight",
                q="mtp.layers.*.q.weight",
                k="mtp.layers.*.k.weight",
                v="mtp.layers.*.v.weight",
                mtp_layers_per_block=0,
            )

    def test_resolve(self):
        m = _MTPFlatteningQKVMapping(
            megatron_param="mtp.layers.*.mtp_model_layer.layers.*.self_attention.linear_qkv.weight",
            q="mtp.layers.*.mixer.q_proj.weight",
            k="mtp.layers.*.mixer.k_proj.weight",
            v="mtp.layers.*.mixer.v_proj.weight",
            mtp_layers_per_block=3,
        )
        result = m.resolve(("2", "1"))  # outer=2, inner=1 → flat=7
        assert isinstance(result, QKVMapping)
        assert (
            result.megatron_param
            == "mtp.layers.2.mtp_model_layer.layers.1.self_attention.linear_qkv.weight"
        )
        assert result.hf_param["q"] == "mtp.layers.7.mixer.q_proj.weight"
        assert result.hf_param["k"] == "mtp.layers.7.mixer.k_proj.weight"
        assert result.hf_param["v"] == "mtp.layers.7.mixer.v_proj.weight"

    def test_resolve_hf_reverse_lookup(self):
        m = _MTPFlatteningQKVMapping(
            megatron_param="mtp.layers.*.mtp_model_layer.layers.*.weight",
            q="mtp.layers.*.q.weight",
            k="mtp.layers.*.k.weight",
            v="mtp.layers.*.v.weight",
            mtp_layers_per_block=2,
        )
        result = m.resolve(("3",))
        assert isinstance(result, QKVMapping)
        assert result.megatron_param == "mtp.layers.1.mtp_model_layer.layers.1.weight"
        assert result.hf_param["q"] == "mtp.layers.3.q.weight"

    def test_resolve_without_captures_raises(self):
        m = _MTPFlatteningQKVMapping(
            megatron_param="mtp.layers.*.mtp_model_layer.layers.*.weight",
            q="mtp.layers.*.q.weight",
            k="mtp.layers.*.k.weight",
            v="mtp.layers.*.v.weight",
            mtp_layers_per_block=2,
        )
        with pytest.raises(ValueError, match="Expected \\(outer, inner\\) captures"):
            m.resolve(())

    def test_hf_to_megatron_raises(self):
        m = _MTPFlatteningQKVMapping(
            megatron_param="mtp.layers.*.mtp_model_layer.layers.*.weight",
            q="mtp.layers.*.q.weight",
            k="mtp.layers.*.k.weight",
            v="mtp.layers.*.v.weight",
            mtp_layers_per_block=1,
        )
        with pytest.raises(NotImplementedError):
            m.hf_to_megatron({"q": torch.tensor([1.0])}, torch.nn.Linear(1, 1))

    def test_megatron_to_hf_raises(self):
        m = _MTPFlatteningQKVMapping(
            megatron_param="mtp.layers.*.mtp_model_layer.layers.*.weight",
            q="mtp.layers.*.q.weight",
            k="mtp.layers.*.k.weight",
            v="mtp.layers.*.v.weight",
            mtp_layers_per_block=1,
        )
        with pytest.raises(NotImplementedError):
            m.megatron_to_hf(torch.tensor([1.0]), torch.nn.Linear(1, 1))


class TestNemotron3BridgeMTPIntegration:
    """Test Nemotron3Bridge methods related to MTP layer handling."""

    def test_list_based_transformers_layer_schema_is_normalized(self):
        hf_config = SimpleNamespace(
            layers_block_type=["linear_attention", "moe", "full_attention", "mlp"],
            num_nextn_predict_layers=1,
            mtp_layers_block_type=["attention", "moe"],
        )

        assert Nemotron3Bridge._hf_hybrid_pattern(hf_config) == "ME*-"
        assert Nemotron3Bridge._hf_mtp_config(hf_config) == (1, "*E")

    def test_list_based_transformers_layer_schema_rejects_unknown_types(self):
        hf_config = SimpleNamespace(layers_block_type=["mamba", "unknown"])

        with pytest.raises(ValueError, match="Unknown layers_block_type entry at index 1"):
            Nemotron3Bridge._hf_hybrid_pattern(hf_config)

    @pytest.mark.parametrize(
        ("hf_config", "expected_mtp_mappings"),
        [
            ({}, False),
            ({"num_nextn_predict_layers": 0, "mtp_hybrid_override_pattern": "*E"}, False),
            ({"num_nextn_predict_layers": 1, "mtp_hybrid_override_pattern": "*E"}, True),
        ],
    )
    def test_mapping_registry_uses_hf_config(self, hf_config, expected_mtp_mappings):
        """Registry construction follows the HF config stored on the bridge."""
        bridge = Nemotron3Bridge()
        bridge.hf_config = SimpleNamespace(**hf_config)

        registry = bridge.mapping_registry()

        mtp_mappings = [
            mapping for mapping in registry.mappings if isinstance(mapping, _MTPFlatteningMapping)
        ]
        assert bool(mtp_mappings) is expected_mtp_mappings

    def test_mapping_registry_with_mtp(self):
        """Test mapping_registry includes MTP flattening mappings when mtp is configured."""
        bridge = Nemotron3Bridge()
        bridge.hf_config = SimpleNamespace(
            num_nextn_predict_layers=1, mtp_hybrid_override_pattern="*E"
        )

        registry = bridge.mapping_registry()

        # Should contain _MTPFlatteningMapping instances
        mtp_mappings = [m for m in registry.mappings if isinstance(m, _MTPFlatteningMapping)]
        assert len(mtp_mappings) > 0

        # Check inner_override=0 mappings exist (eh_proj, enorm, hnorm)
        inner0 = [m for m in mtp_mappings if m._inner_override == 0]
        assert len(inner0) == 3

        # Check inner_override=L-1 mapping exists (final_layernorm)
        inner_last = [m for m in mtp_mappings if m._inner_override == 1]  # L-1 = 2-1 = 1
        assert len(inner_last) == 1

        # Check nested MTP mappings (inner_override=None)
        nested = [m for m in mtp_mappings if m._inner_override is None]
        assert len(nested) > 0

        # Should contain _MTPFlatteningQKVMapping
        qkv_mappings = [m for m in registry.mappings if isinstance(m, _MTPFlatteningQKVMapping)]
        assert len(qkv_mappings) == 1

    def test_mapping_registry_resolves_representative_mtp_params(self):
        """Verify current MTP mappings resolve to concrete HF parameter names."""
        bridge = Nemotron3Bridge()
        bridge.hf_config = SimpleNamespace(
            num_nextn_predict_layers=1, mtp_hybrid_override_pattern="*E"
        )

        registry = bridge.mapping_registry()

        mlp_mapping = registry.megatron_to_hf_lookup(
            "mtp.layers.1.mtp_model_layer.layers.0.mlp.experts.linear_fc1.weight5"
        )
        qkv_mapping = registry.megatron_to_hf_lookup(
            "mtp.layers.1.mtp_model_layer.layers.1.self_attention.linear_qkv.weight"
        )

        assert isinstance(mlp_mapping, AutoMapping)
        assert mlp_mapping.hf_param == "mtp.layers.2.mixer.experts.5.up_proj.weight"
        assert isinstance(qkv_mapping, QKVMapping)
        assert qkv_mapping.hf_param["q"] == "mtp.layers.3.mixer.q_proj.weight"

    def test_quantized_mtp_amax_mappings_preserve_flattened_layer_indices(self):
        bridge = Nemotron3Bridge()
        bridge.hf_config = SimpleNamespace(
            num_nextn_predict_layers=1, mtp_hybrid_override_pattern="*E*M-"
        )

        registry = bridge.mapping_registry()

        mlp_mapping = registry.megatron_to_hf_lookup(
            "mtp.layers.3.mtp_model_layer.layers.2.mlp.linear_fc1.weight_quantizer._amax"
        )
        input_mapping = registry.megatron_to_hf_lookup(
            "mtp.layers.3.mtp_model_layer.layers.2.mlp.linear_fc1.input_quantizer._amax"
        )
        outer_mapping = registry.megatron_to_hf_lookup(
            "mtp.layers.3.eh_proj.weight_quantizer._amax"
        )
        reverse_mapping = registry.hf_to_megatron_lookup(
            "mtp.layers.17.mixer.up_proj.weight_quantizer._amax"
        )
        qkv_mapping = registry.megatron_to_hf_lookup(
            "mtp.layers.3.mtp_model_layer.layers.2.self_attention.linear_qkv.weight_quantizer._amax"
        )
        grouped_mapping = registry.megatron_to_hf_lookup(
            "mtp.layers.3.mtp_model_layer.layers.2.mlp.experts.linear_fc1.weight_quantizer._amax"
        )

        assert isinstance(mlp_mapping, AmaxMapping)
        assert mlp_mapping.hf_param == "mtp.layers.17.mixer.up_proj.weight_quantizer._amax"
        assert isinstance(input_mapping, AmaxMapping)
        assert input_mapping.hf_param == "mtp.layers.17.mixer.up_proj.input_quantizer._amax"
        assert isinstance(outer_mapping, AmaxMapping)
        assert outer_mapping.hf_param == "mtp.layers.15.eh_proj.weight_quantizer._amax"
        assert isinstance(reverse_mapping, AmaxMapping)
        assert (
            reverse_mapping.megatron_param
            == "mtp.layers.3.mtp_model_layer.layers.2.mlp.linear_fc1.weight_quantizer._amax"
        )
        assert isinstance(qkv_mapping, AmaxFanoutMapping)
        assert set(qkv_mapping.hf_targets) == {
            "mtp.layers.17.mixer.q_proj.weight_quantizer._amax",
            "mtp.layers.17.mixer.k_proj.weight_quantizer._amax",
            "mtp.layers.17.mixer.v_proj.weight_quantizer._amax",
        }
        assert grouped_mapping is None

    def test_mapping_registry_resolves_final_norm_aliases(self):
        """Export trunk final norm for both HybridModel and TransformerBlock key names."""
        registry = Nemotron3Bridge().mapping_registry()

        final_norm_mapping = registry.megatron_to_hf_lookup("decoder.final_norm.weight")
        final_layernorm_mapping = registry.megatron_to_hf_lookup("decoder.final_layernorm.weight")

        assert final_norm_mapping.hf_param == "backbone.norm_f.weight"
        assert final_layernorm_mapping.hf_param == "backbone.norm_f.weight"

    def test_mapping_registry_without_mtp_skips_mtp_mappings(self):
        """A non-MTP model builds its normal registry without MTP mappings."""
        bridge = Nemotron3Bridge()
        bridge.hf_config = SimpleNamespace(num_nextn_predict_layers=0)

        registry = bridge.mapping_registry()

        # Should NOT contain any MTP flattening mappings
        mtp_mappings = [m for m in registry.mappings if isinstance(m, _MTPFlatteningMapping)]
        assert len(mtp_mappings) == 0
        qkv_mappings = [m for m in registry.mappings if isinstance(m, _MTPFlatteningQKVMapping)]
        assert len(qkv_mappings) == 0


_BUILDER_BASE_CONFIG = {
    "architectures": ["NemotronHForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attention_head_dim": 128,
    "chunk_size": 128,
    "hidden_act": "relu2",
    "hidden_dropout": 0.0,
    "hidden_size": 2688,
    "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
    "initializer_range": 0.02,
    "intermediate_size": 1856,
    "layer_norm_epsilon": 1e-05,
    "mamba_head_dim": 64,
    "mamba_num_heads": 64,
    "max_position_embeddings": 8192,
    "mlp_bias": False,
    "model_type": "nemotron_h",
    "n_groups": 8,
    "num_attention_heads": 32,
    "num_hidden_layers": 52,
    "num_key_value_heads": 2,
    "residual_in_fp32": False,
    "rms_norm_eps": 1e-05,
    "ssm_state_size": 128,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_bias": False,
    "vocab_size": 131072,
}
_BUILDER_MOE = {
    "n_routed_experts": 128,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 1856,
    "moe_shared_expert_intermediate_size": 3712,
    "n_group": 1,
    "topk_group": 1,
    "routed_scaling_factor": 2.5,
}
_BUILDER_CONFIG_VARIANTS = {
    "dense": {"n_routed_experts": None},
    "moe": _BUILDER_MOE,
    "moe_mtp_latent": {
        **_BUILDER_MOE,
        "num_nextn_predict_layers": 1,
        "mtp_hybrid_override_pattern": "*E",
        "moe_latent_size": 1024,
        "moe_shared_expert_overlap": False,
    },
    "mtp_not_repeated": {
        **_BUILDER_MOE,
        "num_nextn_predict_layers": 2,
        "mtp_hybrid_override_pattern": "*E",
        "mtp_use_repeated_layer": False,
        "keep_mtp_spec_in_bf16": False,
        "mtp_loss_scaling_factor": 0.5,
    },
    "layer_types": {
        **_BUILDER_MOE,
        "hybrid_override_pattern": None,
        "layers_block_type": ["mamba", "moe", "attention", "mlp"] * 13,
        "num_nextn_predict_layers": 1,
        "mtp_layers_block_type": ["attention", "moe"],
    },
    "chunk256_fp16": {"n_routed_experts": None, "chunk_size": 256, "torch_dtype": "float16"},
    "no_moe_attributes": {},
}


def _builder_hf_config(variant: str) -> SimpleNamespace:
    values = {**_BUILDER_BASE_CONFIG, **_BUILDER_CONFIG_VARIANTS[variant]}
    if values["hybrid_override_pattern"] is None:
        del values["hybrid_override_pattern"]
    return SimpleNamespace(**values)


def _native_hf_config():
    from transformers import NemotronHConfig

    config = NemotronHConfig(
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        mamba_num_heads=8,
        mamba_head_dim=32,
        ssm_state_size=16,
        n_groups=2,
        chunk_size=256,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        moe_shared_expert_intermediate_size=256,
        moe_latent_size=64,
        vocab_size=1024,
        hybrid_override_pattern="MEM*EM-E",
        num_hidden_layers=8,
        num_nextn_predict_layers=1,
        mtp_hybrid_override_pattern="*E",
        dtype="bfloat16",
    )
    config.architectures = ["NemotronHForCausalLM"]
    return config


def _builder_field_names(model_config) -> set[str]:
    return {f.name for f in dataclasses.fields(model_config)} | {
        f.name for f in dataclasses.fields(model_config.transformer)
    }


class TestNemotronHBuilderConfig:
    """Builder-backed ModelConfig path for Nemotron-H."""

    def test_conversion_uses_builder_config(self):
        assert Nemotron3Bridge.MODEL_CONFIG_CLASS is BridgeHybridModelConfig

    def test_hf_config_to_model_config_uses_direct_mapping(self):
        bridge = Nemotron3Bridge()
        model_config = bridge.hf_config_to_model_config(_builder_hf_config("moe_mtp_latent"))

        assert isinstance(model_config, BridgeHybridModelConfig)
        assert model_config.hybrid_layer_pattern == _BUILDER_BASE_CONFIG["hybrid_override_pattern"]
        assert model_config.position_embedding_type == "none"
        assert model_config.activation_func is squared_relu
        assert model_config.normalization == "RMSNorm"
        assert model_config.is_hybrid_model is True
        assert model_config.first_last_layers_bf16 is True
        assert model_config.mamba_chunk_size == 128
        assert model_config.mamba_num_groups == 8
        assert model_config.kv_channels == 128
        assert model_config.num_moe_experts == 128
        assert model_config.moe_router_score_function == "sigmoid"
        assert model_config.moe_router_load_balancing_type == "seq_aux_loss"
        assert model_config.moe_latent_size == 1024
        assert model_config.moe_shared_expert_overlap is False
        assert model_config.mtp_num_layers == 1
        assert model_config.mtp_hybrid_override_pattern == "*E"
        assert model_config.mtp_use_repeated_layer is True
        assert model_config.mtp_loss_scaling_factor == 0.3
        assert model_config.hybrid_stack_spec is None

    @pytest.mark.parametrize("variant", [*_BUILDER_CONFIG_VARIANTS, "native"])
    def test_model_config_roundtrip(self, variant):
        hf_config = _native_hf_config() if variant == "native" else _builder_hf_config(variant)
        model_config = Nemotron3Bridge().hf_config_to_model_config(hf_config)
        restored = BridgeHybridModelConfig.from_dict(model_config.as_dict())

        assert isinstance(restored, BridgeHybridModelConfig)
        for name in sorted(_builder_field_names(model_config) - {"transformer"}):
            assert getattr(restored, name) == getattr(model_config, name), name

        model_config.finalize()
        restored.finalize()
        for name in sorted(_builder_field_names(model_config) - {"transformer"}):
            expected = getattr(model_config, name)
            if callable(expected) and not isinstance(expected, type):
                continue
            assert getattr(restored, name) == expected, name

    @pytest.mark.parametrize("variant", ["moe_mtp_latent", "mtp_not_repeated", "chunk256_fp16"])
    def test_model_config_exports_hf_architecture(self, variant):
        model_config = Nemotron3Bridge().hf_config_to_model_config(_builder_hf_config(variant))
        model_config.finalize()

        hf_config = Nemotron3Bridge.megatron_to_hf_config(model_config)

        assert (
            hf_config["hybrid_override_pattern"] == _BUILDER_BASE_CONFIG["hybrid_override_pattern"]
        )
        assert hf_config["chunk_size"] == model_config.mamba_chunk_size

    def test_unified_pattern_follows_recipe_mtp_depth(self):
        model_config = Nemotron3Bridge().hf_config_to_model_config(
            _builder_hf_config("moe_mtp_latent")
        )
        # Recipes train two prediction depths with the checkpoint's single shared MTP block.
        model_config.mtp_num_layers = 2

        model_config.finalize()

        assert model_config.hybrid_layer_pattern.endswith("/*E/*E")
        assert model_config.num_layers == 52

    def test_get_model_config_records_hf_source(self):
        hf_config = _native_hf_config()
        hf_config._name_or_path = (
            "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"  # pragma: allowlist secret
        )

        model_config = AutoBridge.from_hf_config(hf_config).get_model_config()

        assert isinstance(model_config, BridgeHybridModelConfig)
        assert model_config.extra_checkpoint_metadata == {"hf_model_id": hf_config._name_or_path}
        assert model_config.as_dict()["transformer"]["activation_func"] == "squared_relu"
