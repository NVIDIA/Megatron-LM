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
# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/0f06205924d4e19b0723aa8c33e34a1959ca17fe/tests/unit_tests/models/qwen/test_qwen35_bridge.py
# Source license: Apache-2.0.

"""
Unit tests for Qwen3.5 bridge functionality.
"""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import Qwen3_5Config, Qwen3_5MoeConfig, Qwen3_5MoeTextConfig, Qwen3_5TextConfig

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.transformers_compat import full_attention_interval_from_hf
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hybrid.model_config import BridgeHybridModelConfig
from megatron.bridge.models.qwen.modeling_qwen35.hybrid import (
    qwen35_hybrid_mappings,
    qwen35_hybrid_stack_spec,
    qwen35_pipeline_pattern,
)
from megatron.bridge.models.qwen.modeling_qwen35.model_config import Qwen35HybridModelConfig
from megatron.bridge.models.qwen.qwen35_bridge import Qwen35Bridge, Qwen35MoEBridge
from megatron.core.models.hybrid.hybrid_layer_allocation import select_pipeline_segment
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset


class TestQwen35DenseBridge:
    """Test cases for Qwen35Bridge (dense) class."""

    @pytest.fixture
    def qwen3_5_27b_config_dict(self):
        """Create a sample Qwen3.5-27B dense configuration matching the expected model structure."""
        return {
            "architectures": ["Qwen3_5ForCausalLM"],
            "attention_dropout": 0.0,
            "bos_token_id": 248045,
            "eos_token_id": 248046,
            "full_attention_interval": 4,
            "hidden_act": "silu",
            "hidden_size": 5120,
            "initializer_range": 0.02,
            "intermediate_size": 17408,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5",
            "num_attention_heads": 24,
            "num_hidden_layers": 64,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000},
            "rope_theta": 10000000,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "vocab_size": 248320,
            "attention_bias": False,
        }

    @pytest.fixture
    def mock_qwen3_5_config(self, qwen3_5_27b_config_dict):
        """Create a mock Qwen3.5 dense configuration."""
        config = Qwen3_5TextConfig(**qwen3_5_27b_config_dict)
        for key, value in qwen3_5_27b_config_dict.items():
            setattr(config, key, value)
        config.mtp_num_hidden_layers = None
        return config

    @pytest.fixture
    def mock_pretrained_qwen3_5(self, mock_qwen3_5_config):
        """Create a mock PreTrainedCausalLM with Qwen3.5 dense model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_5_config
        mock_pretrained.model = Mock()
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    def test_bridge_registration(self):
        """Test that Qwen35Bridge is properly registered."""
        assert issubclass(Qwen35Bridge, MegatronModelBridge)

    def test_model_config_basic(self, mock_pretrained_qwen3_5, mock_qwen3_5_config):
        """Test the canonical builder config mapping."""
        bridge = Qwen35Bridge()

        result = bridge.hf_config_to_model_config(mock_pretrained_qwen3_5.config)

        assert isinstance(result, Qwen35HybridModelConfig)
        assert result.num_layers == 2 * mock_qwen3_5_config.num_hidden_layers
        assert result.hidden_size == mock_qwen3_5_config.hidden_size
        assert result.num_attention_heads == mock_qwen3_5_config.num_attention_heads
        assert result.seq_length == mock_qwen3_5_config.max_position_embeddings
        assert result.rotary_base == mock_qwen3_5_config.rope_theta

    def test_model_config_vocabulary(self, mock_pretrained_qwen3_5, mock_qwen3_5_config):
        """Test vocabulary size mapping."""
        bridge = Qwen35Bridge()

        result = bridge.hf_config_to_model_config(mock_pretrained_qwen3_5.config)

        assert result.vocab_size == mock_qwen3_5_config.vocab_size
        assert result.share_embeddings_and_output_weights == mock_qwen3_5_config.tie_word_embeddings

    def test_model_config_attention_config(self, mock_pretrained_qwen3_5, mock_qwen3_5_config):
        """Test attention configuration mapping."""
        bridge = Qwen35Bridge()

        result = bridge.hf_config_to_model_config(mock_pretrained_qwen3_5.config)

        assert result.num_attention_heads == mock_qwen3_5_config.num_attention_heads
        assert result.num_query_groups == mock_qwen3_5_config.num_key_value_heads
        assert result.qk_layernorm is True
        assert result.layernorm_zero_centered_gamma is True
        assert result.attention_output_gate is True

    def test_model_config_linear_attention_config(
        self, mock_pretrained_qwen3_5, mock_qwen3_5_config
    ):
        """Test linear attention (GDN) configuration mapping."""
        bridge = Qwen35Bridge()

        result = bridge.hf_config_to_model_config(mock_pretrained_qwen3_5.config)

        expected = [int(i % 2 == 0 and (i // 2 + 1) % 4 != 0) for i in range(result.num_layers)]
        assert result.linear_attention_freq == expected
        assert result.linear_conv_kernel_dim == mock_qwen3_5_config.linear_conv_kernel_dim
        assert result.linear_key_head_dim == mock_qwen3_5_config.linear_key_head_dim
        assert result.linear_value_head_dim == mock_qwen3_5_config.linear_value_head_dim
        assert result.linear_num_key_heads == mock_qwen3_5_config.linear_num_key_heads
        assert result.linear_num_value_heads == mock_qwen3_5_config.linear_num_value_heads
        assert result.experimental_attention_variant == "gdn"

    def test_model_config_mlp_config(self, mock_pretrained_qwen3_5, mock_qwen3_5_config):
        """Test MLP configuration mapping."""
        bridge = Qwen35Bridge()

        result = bridge.hf_config_to_model_config(mock_pretrained_qwen3_5.config)

        assert result.ffn_hidden_size == mock_qwen3_5_config.intermediate_size
        assert result.gated_linear_unit is True

    def test_model_config_normalization(self, mock_pretrained_qwen3_5, mock_qwen3_5_config):
        """Test normalization configuration."""
        bridge = Qwen35Bridge()

        result = bridge.hf_config_to_model_config(mock_pretrained_qwen3_5.config)

        assert result.layernorm_epsilon == mock_qwen3_5_config.rms_norm_eps
        assert result.init_method_std == mock_qwen3_5_config.initializer_range

    def test_model_config_position_embedding(self, mock_pretrained_qwen3_5, mock_qwen3_5_config):
        """Test position embedding configuration."""
        bridge = Qwen35Bridge()

        result = bridge.hf_config_to_model_config(mock_pretrained_qwen3_5.config)

        # Check position embedding
        assert result.rotary_base == mock_qwen3_5_config.rope_theta
        assert result.rotary_percent == mock_qwen3_5_config.rope_parameters["partial_rotary_factor"]
        assert result.position_embedding_type == "rope"

    def test_model_config_mtp_config(self, mock_pretrained_qwen3_5, mock_qwen3_5_config):
        """Test MTP configuration mapping."""
        bridge = Qwen35Bridge()

        result = bridge.hf_config_to_model_config(mock_pretrained_qwen3_5.config)

        # Check MTP configuration
        assert not result.mtp_num_layers

    def test_model_config_dtype_handling(self, qwen3_5_27b_config_dict):
        """Test dtype handling in builder config conversion."""
        config = Qwen3_5TextConfig(**qwen3_5_27b_config_dict)
        for key, value in qwen3_5_27b_config_dict.items():
            setattr(config, key, value)
        config.mtp_num_hidden_layers = None
        config.torch_dtype = "bfloat16"

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config

        bridge = Qwen35Bridge()
        result = bridge.hf_config_to_model_config(mock_pretrained.config)

        assert result.bf16 is True
        assert result.fp16 is False
        assert result.params_dtype == torch.bfloat16

        config.torch_dtype = "float16"
        result = bridge.hf_config_to_model_config(mock_pretrained.config)

        assert result.fp16 is True
        assert result.bf16 is False
        assert result.params_dtype == torch.float16

    def test_model_config_tie_word_embeddings_true(self, mock_qwen3_5_config):
        """Test builder config conversion with tie_word_embeddings=True."""
        mock_qwen3_5_config.tie_word_embeddings = True

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_5_config

        bridge = Qwen35Bridge()
        result = bridge.hf_config_to_model_config(mock_pretrained.config)

        assert result.share_embeddings_and_output_weights is True

    def test_model_config_tie_word_embeddings_false(self, mock_qwen3_5_config):
        """Test builder config conversion with tie_word_embeddings=False."""
        mock_qwen3_5_config.tie_word_embeddings = False

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_5_config

        bridge = Qwen35Bridge()
        result = bridge.hf_config_to_model_config(mock_pretrained.config)

        assert result.share_embeddings_and_output_weights is False

    def test_model_config_27b_config(self, qwen3_5_27b_config_dict):
        """Test builder config conversion with Qwen3.5-27B dense configuration."""
        config = Qwen3_5TextConfig(**qwen3_5_27b_config_dict)
        for key, value in qwen3_5_27b_config_dict.items():
            setattr(config, key, value)
        config.mtp_num_hidden_layers = None

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config

        bridge = Qwen35Bridge()
        result = bridge.hf_config_to_model_config(mock_pretrained.config)

        assert result.num_layers == 128
        assert result.hidden_size == 5120
        assert result.num_attention_heads == 24
        assert result.ffn_hidden_size == 17408
        assert result.hetereogenous_dist_checkpoint is True

    def test_mapping_registry(self):
        """Test mapping_registry returns valid mappings."""
        bridge = Qwen35Bridge()

        bridge.hf_config = (
            Qwen3_5MoeTextConfig(num_hidden_layers=4)
            if isinstance(bridge, Qwen35MoEBridge)
            else Qwen3_5TextConfig(num_hidden_layers=4)
        )
        registry = bridge.mapping_registry()

        # Check that registry is not None and has mappings
        assert registry is not None
        assert len(registry.mappings) > 0

        # Check for expected mapping types
        mapping_types = [type(mapping).__name__ for mapping in registry.mappings]
        assert "AutoMapping" in mapping_types
        assert "QKVMapping" in mapping_types
        assert "GatedMLPMapping" in mapping_types
        assert "GDNLinearMappingSeparate" in mapping_types
        assert "RMSNorm2ZeroCenteredRMSNormMapping" in mapping_types

    def test_mapping_registry_parameter_mappings(self):
        """Test that mapping_registry contains expected parameter mappings."""
        bridge = Qwen35Bridge()

        bridge.hf_config = (
            Qwen3_5MoeTextConfig(num_hidden_layers=4)
            if isinstance(bridge, Qwen35MoEBridge)
            else Qwen3_5TextConfig(num_hidden_layers=4)
        )
        registry = bridge.mapping_registry()

        # Extract all AutoMapping instances
        auto_mappings = [m for m in registry.mappings if type(m).__name__ == "AutoMapping"]

        # Check for critical parameter mappings
        hf_params = [mapping.hf_param for mapping in auto_mappings]
        megatron_params = [mapping.megatron_param for mapping in auto_mappings]

        # Should have embedding mappings
        assert "model.embed_tokens.weight" in hf_params
        assert "embedding.word_embeddings.weight" in megatron_params

        # Should have output layer mappings
        assert "lm_head.weight" in hf_params
        assert "output_layer.weight" in megatron_params

        # Should have layer norm mappings
        assert "model.norm.weight" in hf_params
        assert "decoder.final_norm.weight" in megatron_params

    def test_mapping_registry_mtp_mapping(self):
        """Test that mapping_registry contains MTP mapping."""
        bridge = Qwen35Bridge()

        bridge.hf_config = (
            Qwen3_5MoeTextConfig(num_hidden_layers=4)
            if isinstance(bridge, Qwen35MoEBridge)
            else Qwen3_5TextConfig(num_hidden_layers=4)
        )
        registry = bridge.mapping_registry()

        # Extract MTP mappings
        auto_mappings = [m for m in registry.mappings if type(m).__name__ == "AutoMapping"]

        # Check for critical parameter mappings
        hf_params = [mapping.hf_param for mapping in auto_mappings]
        megatron_params = [mapping.megatron_param for mapping in auto_mappings]

        # Should have embedding and hidden projection
        assert "mtp.fc.weight" in hf_params
        assert "mtp.layers.0.eh_proj.weight" in megatron_params

        # Should have pre-fc norms for embedding and hidden
        assert "mtp.pre_fc_norm_embedding.weight" in hf_params
        assert "mtp.pre_fc_norm_hidden.weight" in hf_params
        assert "mtp.layers.0.enorm.weight" in megatron_params
        assert "mtp.layers.0.hnorm.weight" in megatron_params

        # Should have final layernorm
        assert "mtp.norm.weight" in hf_params
        assert "mtp.layers.0.final_layernorm.weight" in megatron_params

    def test_mapping_registry_qkv_mapping(self):
        """Test that mapping_registry contains QKV mapping."""
        bridge = Qwen35Bridge()

        bridge.hf_config = (
            Qwen3_5MoeTextConfig(num_hidden_layers=4)
            if isinstance(bridge, Qwen35MoEBridge)
            else Qwen3_5TextConfig(num_hidden_layers=4)
        )
        registry = bridge.mapping_registry()

        # Extract QKVMapping instances
        qkv_mappings = [m for m in registry.mappings if type(m).__name__ == "QKVMapping"]

        # Should have at least one QKV mapping
        assert len(qkv_mappings) > 0

        # Check the QKV mapping structure
        qkv_mapping = qkv_mappings[0]
        assert hasattr(qkv_mapping, "hf_param")
        assert isinstance(qkv_mapping.hf_param, dict)
        assert "q" in qkv_mapping.hf_param
        assert "k" in qkv_mapping.hf_param
        assert "v" in qkv_mapping.hf_param
        assert hasattr(qkv_mapping, "megatron_param")

    def test_mapping_registry_gdn_linear_mapping(self):
        """Test that mapping_registry contains GDN linear mapping."""
        bridge = Qwen35Bridge()

        bridge.hf_config = (
            Qwen3_5MoeTextConfig(num_hidden_layers=4)
            if isinstance(bridge, Qwen35MoEBridge)
            else Qwen3_5TextConfig(num_hidden_layers=4)
        )
        registry = bridge.mapping_registry()

        # Extract GDNLinearMappingSeparate instances
        gdn_linear_mappings = [
            m for m in registry.mappings if type(m).__name__ == "GDNLinearMappingSeparate"
        ]
        assert len(gdn_linear_mappings) > 0

        # Check the GDN linear mapping structure
        gdn_linear_mapping = gdn_linear_mappings[0]
        assert hasattr(gdn_linear_mapping, "hf_param")
        assert isinstance(gdn_linear_mapping.hf_param, dict)
        assert "qkv" in gdn_linear_mapping.hf_param
        assert "z" in gdn_linear_mapping.hf_param
        assert "b" in gdn_linear_mapping.hf_param
        assert "a" in gdn_linear_mapping.hf_param
        assert hasattr(gdn_linear_mapping, "megatron_param")

    def test_mapping_registry_no_moe_mappings(self):
        """Test that dense bridge does not contain MoE-specific mappings."""
        bridge = Qwen35Bridge()

        bridge.hf_config = (
            Qwen3_5MoeTextConfig(num_hidden_layers=4)
            if isinstance(bridge, Qwen35MoEBridge)
            else Qwen3_5TextConfig(num_hidden_layers=4)
        )
        registry = bridge.mapping_registry()

        auto_mappings = [m for m in registry.mappings if type(m).__name__ == "AutoMapping"]

        hf_params = [mapping.hf_param for mapping in auto_mappings]
        megatron_params = [mapping.megatron_param for mapping in auto_mappings]

        # Check for no expert and router mappings
        assert not any("router" in p or "experts" in p for p in hf_params + megatron_params)


class TestQwen35MoEBridge:
    """Test cases for Qwen3_5MoEBridge (MoE) class."""

    @pytest.fixture
    def qwen3_5_397b_a17b_config_dict(self):
        """Create a sample Qwen3.5-397B-A17B MoE configuration matching the expected model structure."""
        return {
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "attention_dropout": 0.0,
            "bos_token_id": 248045,
            "eos_token_id": 248046,
            "full_attention_interval": 4,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 1024,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 64,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5_moe",
            "moe_intermediate_size": 1024,
            "norm_topk_prob": True,
            "num_attention_heads": 32,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "num_hidden_layers": 60,
            "num_key_value_heads": 2,
            "output_router_logits": False,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000},
            "rope_theta": 10000000,
            "router_aux_loss_coef": 0.001,
            "shared_expert_intermediate_size": 4096,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "vocab_size": 248320,
            "attention_bias": False,
        }

    @pytest.fixture
    def mock_qwen3_5_moe_config(self, qwen3_5_397b_a17b_config_dict):
        """Create a mock Qwen3.5 MoE configuration."""
        config = Qwen3_5MoeTextConfig(**qwen3_5_397b_a17b_config_dict)
        for key, value in qwen3_5_397b_a17b_config_dict.items():
            setattr(config, key, value)
        config.mtp_num_hidden_layers = None
        return config

    @pytest.fixture
    def mock_pretrained_qwen3_5_moe(self, mock_qwen3_5_moe_config):
        """Create a mock PreTrainedCausalLM with Qwen3.5 MoE model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_5_moe_config
        mock_pretrained.model = Mock()
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    def test_bridge_registration(self):
        """Test that Qwen3_5MoEBridge is properly registered."""
        assert issubclass(Qwen35MoEBridge, MegatronModelBridge)

    def test_model_config_basic(self, mock_pretrained_qwen3_5_moe, mock_qwen3_5_moe_config):
        """Test the canonical builder config mapping."""
        bridge = Qwen35MoEBridge()

        result = bridge.hf_config_to_model_config(mock_pretrained_qwen3_5_moe.config)

        assert isinstance(result, Qwen35HybridModelConfig)
        assert result.num_layers == 2 * mock_qwen3_5_moe_config.num_hidden_layers
        assert result.hidden_size == mock_qwen3_5_moe_config.hidden_size
        assert result.num_attention_heads == mock_qwen3_5_moe_config.num_attention_heads
        assert result.seq_length == mock_qwen3_5_moe_config.max_position_embeddings
        assert result.rotary_base == mock_qwen3_5_moe_config.rope_theta

    def test_model_config_vocabulary(self, mock_pretrained_qwen3_5_moe, mock_qwen3_5_moe_config):
        """Test vocabulary size mapping."""
        bridge = Qwen35MoEBridge()

        result = bridge.hf_config_to_model_config(mock_pretrained_qwen3_5_moe.config)

        assert result.vocab_size == mock_qwen3_5_moe_config.vocab_size
        assert (
            result.share_embeddings_and_output_weights
            == mock_qwen3_5_moe_config.tie_word_embeddings
        )

    def test_model_config_attention_config(
        self, mock_pretrained_qwen3_5_moe, mock_qwen3_5_moe_config
    ):
        """Test attention configuration mapping."""
        bridge = Qwen35MoEBridge()

        result = bridge.hf_config_to_model_config(mock_pretrained_qwen3_5_moe.config)

        assert result.num_attention_heads == mock_qwen3_5_moe_config.num_attention_heads
        assert result.num_query_groups == mock_qwen3_5_moe_config.num_key_value_heads
        assert result.qk_layernorm is True
        assert result.layernorm_zero_centered_gamma is True
        assert result.attention_output_gate is True

    def test_model_config_linear_attention_config(
        self, mock_pretrained_qwen3_5_moe, mock_qwen3_5_moe_config
    ):
        """Test linear attention (GDN) configuration mapping."""
        bridge = Qwen35MoEBridge()

        result = bridge.hf_config_to_model_config(mock_pretrained_qwen3_5_moe.config)

        expected = [int(i % 2 == 0 and (i // 2 + 1) % 4 != 0) for i in range(result.num_layers)]
        assert result.linear_attention_freq == expected
        assert result.linear_conv_kernel_dim == mock_qwen3_5_moe_config.linear_conv_kernel_dim
        assert result.linear_key_head_dim == mock_qwen3_5_moe_config.linear_key_head_dim
        assert result.linear_value_head_dim == mock_qwen3_5_moe_config.linear_value_head_dim
        assert result.linear_num_key_heads == mock_qwen3_5_moe_config.linear_num_key_heads
        assert result.linear_num_value_heads == mock_qwen3_5_moe_config.linear_num_value_heads
        assert result.experimental_attention_variant == "gdn"

    def test_model_config_moe_config(self, mock_pretrained_qwen3_5_moe, mock_qwen3_5_moe_config):
        """Test MoE-specific configuration mapping."""
        bridge = Qwen35MoEBridge()

        result = bridge.hf_config_to_model_config(mock_pretrained_qwen3_5_moe.config)

        assert result.num_moe_experts == mock_qwen3_5_moe_config.num_experts
        assert result.moe_router_topk == mock_qwen3_5_moe_config.num_experts_per_tok
        assert result.moe_ffn_hidden_size == mock_qwen3_5_moe_config.moe_intermediate_size
        assert result.moe_grouped_gemm is True
        assert (
            result.moe_shared_expert_intermediate_size
            == mock_qwen3_5_moe_config.shared_expert_intermediate_size
        )
        assert result.moe_shared_expert_gate is True
        assert result.moe_router_dtype == "fp32"

    def test_model_config_normalization(self, mock_pretrained_qwen3_5_moe, mock_qwen3_5_moe_config):
        """Test normalization configuration."""
        bridge = Qwen35MoEBridge()

        result = bridge.hf_config_to_model_config(mock_pretrained_qwen3_5_moe.config)

        assert result.layernorm_epsilon == mock_qwen3_5_moe_config.rms_norm_eps
        assert result.init_method_std == mock_qwen3_5_moe_config.initializer_range

    def test_model_config_position_embedding(
        self, mock_pretrained_qwen3_5_moe, mock_qwen3_5_moe_config
    ):
        """Test position embedding configuration."""
        bridge = Qwen35MoEBridge()

        result = bridge.hf_config_to_model_config(mock_pretrained_qwen3_5_moe.config)

        assert result.rotary_base == mock_qwen3_5_moe_config.rope_theta
        assert (
            result.rotary_percent
            == mock_qwen3_5_moe_config.rope_parameters["partial_rotary_factor"]
        )
        assert result.position_embedding_type == "rope"

    def test_model_config_mtp_config(self, mock_pretrained_qwen3_5_moe, mock_qwen3_5_moe_config):
        """Test MTP configuration mapping."""
        bridge = Qwen35MoEBridge()

        result = bridge.hf_config_to_model_config(mock_pretrained_qwen3_5_moe.config)

        assert not result.mtp_num_layers

    def test_model_config_dtype_handling(self, qwen3_5_397b_a17b_config_dict):
        """Test dtype handling in builder config conversion."""
        config = Qwen3_5MoeTextConfig(**qwen3_5_397b_a17b_config_dict)
        for key, value in qwen3_5_397b_a17b_config_dict.items():
            setattr(config, key, value)
        config.mtp_num_hidden_layers = None
        config.torch_dtype = "bfloat16"

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config

        bridge = Qwen35MoEBridge()
        result = bridge.hf_config_to_model_config(mock_pretrained.config)

        assert result.bf16 is True
        assert result.fp16 is False
        assert result.params_dtype == torch.bfloat16

        config.torch_dtype = "float16"
        result = bridge.hf_config_to_model_config(mock_pretrained.config)

        assert result.fp16 is True
        assert result.bf16 is False
        assert result.params_dtype == torch.float16

    def test_model_config_tie_word_embeddings_true(self, mock_qwen3_5_moe_config):
        """Test builder config conversion with tie_word_embeddings=True."""
        mock_qwen3_5_moe_config.tie_word_embeddings = True

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_5_moe_config

        bridge = Qwen35MoEBridge()
        result = bridge.hf_config_to_model_config(mock_pretrained.config)

        assert result.share_embeddings_and_output_weights is True

    def test_model_config_tie_word_embeddings_false(self, mock_qwen3_5_moe_config):
        """Test builder config conversion with tie_word_embeddings=False."""
        mock_qwen3_5_moe_config.tie_word_embeddings = False

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_qwen3_5_moe_config

        bridge = Qwen35MoEBridge()
        result = bridge.hf_config_to_model_config(mock_pretrained.config)

        assert result.share_embeddings_and_output_weights is False

    def test_model_config_397b_a17b_config(self, qwen3_5_397b_a17b_config_dict):
        """Test builder config conversion with Qwen3.5-397B-A17B MoE configuration."""
        config = Qwen3_5MoeTextConfig(**qwen3_5_397b_a17b_config_dict)
        for key, value in qwen3_5_397b_a17b_config_dict.items():
            setattr(config, key, value)
        config.mtp_num_hidden_layers = None

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = config

        bridge = Qwen35MoEBridge()
        result = bridge.hf_config_to_model_config(mock_pretrained.config)

        assert result.num_layers == 120
        assert result.hidden_size == 4096
        assert result.num_attention_heads == 32
        assert result.moe_ffn_hidden_size == 1024
        assert result.hetereogenous_dist_checkpoint is True

    def test_mapping_registry(self):
        """Test mapping_registry returns valid mappings."""
        bridge = Qwen35MoEBridge()

        bridge.hf_config = (
            Qwen3_5MoeTextConfig(num_hidden_layers=4)
            if isinstance(bridge, Qwen35MoEBridge)
            else Qwen3_5TextConfig(num_hidden_layers=4)
        )
        registry = bridge.mapping_registry()

        assert registry is not None
        assert len(registry.mappings) > 0

        mapping_types = [type(mapping).__name__ for mapping in registry.mappings]
        assert "AutoMapping" in mapping_types
        assert "QKVMapping" in mapping_types
        assert "GatedMLPMapping" in mapping_types
        assert "GDNLinearMappingSeparate" in mapping_types
        # Routed-expert mappings default to per-expert without a checkpoint; the fused vs per-expert
        # selection is covered by test_moe_expert_mappings_fused_vs_per_expert.
        assert "ReplicatedMapping" in mapping_types
        assert "RMSNorm2ZeroCenteredRMSNormMapping" in mapping_types

    def test_mapping_registry_parameter_mappings(self):
        """Test that mapping_registry contains expected parameter mappings."""
        bridge = Qwen35MoEBridge()

        bridge.hf_config = (
            Qwen3_5MoeTextConfig(num_hidden_layers=4)
            if isinstance(bridge, Qwen35MoEBridge)
            else Qwen3_5TextConfig(num_hidden_layers=4)
        )
        registry = bridge.mapping_registry()

        auto_mappings = [m for m in registry.mappings if type(m).__name__ == "AutoMapping"]

        hf_params = [mapping.hf_param for mapping in auto_mappings]
        megatron_params = [mapping.megatron_param for mapping in auto_mappings]

        assert "model.embed_tokens.weight" in hf_params
        assert "embedding.word_embeddings.weight" in megatron_params

        assert "lm_head.weight" in hf_params
        assert "output_layer.weight" in megatron_params

        assert "model.norm.weight" in hf_params
        assert "decoder.final_norm.weight" in megatron_params

    def test_mapping_registry_mtp_mapping(self):
        """Test that mapping_registry contains MTP mapping."""
        bridge = Qwen35MoEBridge()

        bridge.hf_config = (
            Qwen3_5MoeTextConfig(num_hidden_layers=4)
            if isinstance(bridge, Qwen35MoEBridge)
            else Qwen3_5TextConfig(num_hidden_layers=4)
        )
        registry = bridge.mapping_registry()

        # Extract MTP mappings
        auto_mappings = [m for m in registry.mappings if type(m).__name__ == "AutoMapping"]

        # Check for critical parameter mappings
        hf_params = [mapping.hf_param for mapping in auto_mappings]
        megatron_params = [mapping.megatron_param for mapping in auto_mappings]

        # Should have embedding and hidden projection
        assert "mtp.fc.weight" in hf_params
        assert "mtp.layers.0.eh_proj.weight" in megatron_params

        # Should have pre-fc norms for embedding and hidden
        assert "mtp.pre_fc_norm_embedding.weight" in hf_params
        assert "mtp.pre_fc_norm_hidden.weight" in hf_params
        assert "mtp.layers.0.enorm.weight" in megatron_params
        assert "mtp.layers.0.hnorm.weight" in megatron_params

        # Should have final layernorm
        assert "mtp.norm.weight" in hf_params
        assert "mtp.layers.0.final_layernorm.weight" in megatron_params

    def test_mapping_registry_mtp_layer_spellings(self):
        """Every MTP sub-layer mapping is registered under both Megatron-Core spellings."""
        bridge = Qwen35MoEBridge()

        bridge.hf_config = (
            Qwen3_5MoeTextConfig(num_hidden_layers=4)
            if isinstance(bridge, Qwen35MoEBridge)
            else Qwen3_5TextConfig(num_hidden_layers=4)
        )
        registry = bridge.mapping_registry()

        megatron_params = [mapping.megatron_param for mapping in registry.mappings]
        for layer_attr in ("mtp_model_layer", "transformer_layer"):
            for suffix in (
                "mlp.router.weight",
                "pre_mlp_layernorm.weight",
                "self_attention.linear_qkv.layer_norm_weight",
                "self_attention.q_layernorm.weight",
                "self_attention.k_layernorm.weight",
                "self_attention.linear_proj.weight",
                "mlp.shared_experts.gate_weight",
            ):
                position = int(suffix.startswith(("mlp.", "pre_mlp_layernorm.")))
                assert f"mtp.layers.0.{layer_attr}.layers.{position}.{suffix}" in megatron_params
            for suffix in (
                "self_attention.linear_qkv.weight",
                "mlp.shared_experts.linear_fc1.weight",
                "mlp.shared_experts.linear_fc2.weight",
                "mlp.experts.linear_fc1.weight*",
                "mlp.experts.linear_fc2.weight*",
            ):
                position = int(suffix.startswith(("mlp.", "pre_mlp_layernorm.")))
                assert f"mtp.layers.*.{layer_attr}.layers.{position}.{suffix}" in megatron_params

            router = registry.megatron_to_hf_lookup(
                f"mtp.layers.0.{layer_attr}.layers.1.mlp.router.weight"
            )
            assert router.hf_param == "mtp.layers.0.mlp.gate.weight"

    def test_mapping_registry_qkv_mapping(self):
        """Test that mapping_registry contains QKV mapping."""
        bridge = Qwen35MoEBridge()

        bridge.hf_config = (
            Qwen3_5MoeTextConfig(num_hidden_layers=4)
            if isinstance(bridge, Qwen35MoEBridge)
            else Qwen3_5TextConfig(num_hidden_layers=4)
        )
        registry = bridge.mapping_registry()

        # Extract QKVMapping instances
        qkv_mappings = [m for m in registry.mappings if type(m).__name__ == "QKVMapping"]

        # Should have at least one QKV mapping
        assert len(qkv_mappings) > 0

        # Check the QKV mapping structure
        qkv_mapping = qkv_mappings[0]
        assert hasattr(qkv_mapping, "hf_param")
        assert isinstance(qkv_mapping.hf_param, dict)
        assert "q" in qkv_mapping.hf_param
        assert "k" in qkv_mapping.hf_param
        assert "v" in qkv_mapping.hf_param
        assert hasattr(qkv_mapping, "megatron_param")

    def test_mapping_registry_gdn_linear_mapping(self):
        """Test that mapping_registry contains GDN linear mapping."""
        bridge = Qwen35MoEBridge()

        bridge.hf_config = (
            Qwen3_5MoeTextConfig(num_hidden_layers=4)
            if isinstance(bridge, Qwen35MoEBridge)
            else Qwen3_5TextConfig(num_hidden_layers=4)
        )
        registry = bridge.mapping_registry()

        # Extract GDNLinearMappingSeparate instances
        gdn_linear_mappings = [
            m for m in registry.mappings if type(m).__name__ == "GDNLinearMappingSeparate"
        ]
        assert len(gdn_linear_mappings) > 0

        # Check the GDN linear mapping structure
        gdn_linear_mapping = gdn_linear_mappings[0]
        assert hasattr(gdn_linear_mapping, "hf_param")
        assert isinstance(gdn_linear_mapping.hf_param, dict)
        assert "qkv" in gdn_linear_mapping.hf_param
        assert "z" in gdn_linear_mapping.hf_param
        assert "b" in gdn_linear_mapping.hf_param
        assert "a" in gdn_linear_mapping.hf_param
        assert hasattr(gdn_linear_mapping, "megatron_param")

    def test_mapping_registry_moe_mappings(self):
        """Test that mapping_registry contains MoE-specific mappings."""
        bridge = Qwen35MoEBridge()

        bridge.hf_config = (
            Qwen3_5MoeTextConfig(num_hidden_layers=4)
            if isinstance(bridge, Qwen35MoEBridge)
            else Qwen3_5TextConfig(num_hidden_layers=4)
        )
        registry = bridge.mapping_registry()

        # Extract all mappings
        auto_mappings = [m for m in registry.mappings if type(m).__name__ == "AutoMapping"]
        replicated_mappings = [
            m for m in registry.mappings if type(m).__name__ == "ReplicatedMapping"
        ]

        # Check for MoE router mapping
        hf_params = [mapping.hf_param for mapping in auto_mappings]
        assert "model.layers.0.mlp.gate.weight" in hf_params
        # shared_expert_gate is represented via ReplicatedMapping in bridge
        replicated_hf_params = [mapping.hf_param for mapping in replicated_mappings]
        assert "model.layers.0.mlp.shared_expert_gate.weight" in replicated_hf_params

        # Routed-expert mappings exist for both the grouped-GEMM (experts.linear_fc*.weight*) and
        # SequentialMLP (local_experts.*.linear_fc*) layouts. Their fused-vs-per-expert form depends
        # on the checkpoint and is covered by test_moe_expert_mappings_fused_vs_per_expert.
        megatron_params = [getattr(m, "megatron_param", "") for m in registry.mappings]
        assert any(p.endswith("mlp.experts.linear_fc1.weight*") for p in megatron_params)
        assert any(p.endswith("local_experts.*.linear_fc1.weight") for p in megatron_params)

    def test_moe_expert_mappings_fused_vs_per_expert(self):
        """Routed-expert mappings match the detected checkpoint layout, for grouped and sequential."""

        def mapping_types(experts_packed):
            mappings = Qwen35MoEBridge._get_moe_lm_mappings(experts_packed=experts_packed)

            def _type(suffix):
                return type(
                    next(m for m in mappings if getattr(m, "megatron_param", "").endswith(suffix))
                ).__name__

            return {
                "grouped_fc1": _type("mlp.experts.linear_fc1.weight*"),
                "grouped_fc2": _type("mlp.experts.linear_fc2.weight*"),
                "seq_fc1": _type("local_experts.*.linear_fc1.weight"),
                "seq_fc2": _type("local_experts.*.linear_fc2.weight"),
            }

        # Fused checkpoint (real Qwen3.5/3.6): read the stacked experts.gate_up_proj / down_proj.
        assert mapping_types(experts_packed=True) == {
            "grouped_fc1": "FusedGatedExpertMapping",
            "grouped_fc2": "FusedExpertMapping",
            "seq_fc1": "FusedGatedExpertMapping",
            "seq_fc2": "FusedExpertMapping",
        }
        # Per-expert checkpoint (save_pretrained output): read experts.*.gate_proj / up_proj / down_proj.
        assert mapping_types(experts_packed=False) == {
            "grouped_fc1": "GatedMLPMapping",
            "grouped_fc2": "AutoMapping",
            "seq_fc1": "GatedMLPMapping",
            "seq_fc2": "AutoMapping",
        }


_SMALL_GDN_TEXT_CONFIG = dict(
    hidden_size=64,
    num_hidden_layers=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    max_position_embeddings=4096,
    vocab_size=128,
    # Released Qwen3.5 and Qwen3.6 text configs carry one MTP layer.
    mtp_num_hidden_layers=1,
    dtype="bfloat16",
)
_SMALL_MOE_CONFIG = dict(
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    num_experts=4,
    num_experts_per_tok=2,
)
# Multimodal checkpoints nest the language config and keep the mRoPE sections there.
_MULTIMODAL_ROPE_PARAMETERS = {
    "rope_type": "default",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
    "mrope_section": [11, 11, 10],
    "mrope_interleaved": True,
}
_YARN_ROPE_PARAMETERS = {
    "rope_type": "yarn",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
    "factor": 4.0,
    "original_max_position_embeddings": 1024,
}


def _qwen35_text_config(**overrides) -> Qwen3_5TextConfig:
    """Return a small Qwen3.5 dense text config with the released 3 GDN : 1 attention layer layout."""
    config = Qwen3_5TextConfig(**{**_SMALL_GDN_TEXT_CONFIG, "intermediate_size": 128, **overrides})
    config.architectures = ["Qwen3_5ForCausalLM"]
    return config


def _qwen35_moe_text_config(**overrides) -> Qwen3_5MoeTextConfig:
    """Return a small Qwen3.5 MoE text config with the released 3 GDN : 1 attention layer layout."""
    config = Qwen3_5MoeTextConfig(**{**_SMALL_GDN_TEXT_CONFIG, **_SMALL_MOE_CONFIG, **overrides})
    config.architectures = ["Qwen3_5MoeForCausalLM"]
    return config


def _text_config_from_multimodal(multimodal_config, architecture: str):
    """Extract the language config of a multimodal checkpoint, as the Qwen3.5 text recipes do."""
    text_config = deepcopy(multimodal_config.text_config)
    text_config.architectures = [architecture]
    return text_config


def _dense_from_multimodal():
    text_config = {
        **_SMALL_GDN_TEXT_CONFIG,
        "intermediate_size": 128,
        "rope_parameters": _MULTIMODAL_ROPE_PARAMETERS,
    }
    multimodal = Qwen3_5Config(text_config=text_config, tie_word_embeddings=False)
    return _text_config_from_multimodal(multimodal, "Qwen3_5ForCausalLM")


def _moe_from_multimodal():
    text_config = {
        **_SMALL_GDN_TEXT_CONFIG,
        **_SMALL_MOE_CONFIG,
        "rope_parameters": _MULTIMODAL_ROPE_PARAMETERS,
    }
    multimodal = Qwen3_5MoeConfig(text_config=text_config, tie_word_embeddings=False)
    return _text_config_from_multimodal(multimodal, "Qwen3_5MoeForCausalLM")


_DENSE_CONFIGS = {
    "default": lambda: _qwen35_text_config(),
    "tied_fp16": lambda: _qwen35_text_config(tie_word_embeddings=True, dtype="float16"),
    "no_mtp": lambda: _qwen35_text_config(mtp_num_hidden_layers=0),
    "yarn": lambda: _qwen35_text_config(rope_parameters=_YARN_ROPE_PARAMETERS),
    "from_multimodal": _dense_from_multimodal,
}
_MOE_CONFIGS = {
    "default": lambda: _qwen35_moe_text_config(),
    "tied_fp16": lambda: _qwen35_moe_text_config(tie_word_embeddings=True, dtype="float16"),
    "yarn": lambda: _qwen35_moe_text_config(rope_parameters=_YARN_ROPE_PARAMETERS),
    "from_multimodal": _moe_from_multimodal,
}
_PARITY_CASES = [
    *((Qwen35Bridge, make_config) for make_config in _DENSE_CONFIGS.values()),
    *((Qwen35MoEBridge, make_config) for make_config in _MOE_CONFIGS.values()),
]
_PARITY_IDS = [
    *(f"dense-{name}" for name in _DENSE_CONFIGS),
    *(f"moe-{name}" for name in _MOE_CONFIGS),
]


class TestQwen35BuilderConfig:
    """Builder-backed ModelConfig path for the Qwen3.5 text bridges."""

    @pytest.mark.parametrize("bridge_class", [Qwen35Bridge, Qwen35MoEBridge])
    def test_conversion_uses_builder_config(self, bridge_class):
        assert bridge_class.MODEL_CONFIG_CLASS is Qwen35HybridModelConfig

    @pytest.mark.parametrize(
        ("bridge_class", "config"),
        [(Qwen35Bridge, _qwen35_text_config()), (Qwen35MoEBridge, _qwen35_moe_text_config())],
        ids=["dense", "moe"],
    )
    def test_hf_config_to_model_config_uses_direct_mapping(self, bridge_class, config):
        bridge = bridge_class()
        model_config = bridge.hf_config_to_model_config(config)

        assert isinstance(model_config, BridgeHybridModelConfig)
        assert model_config.num_layers == 16
        assert model_config.normalization == "RMSNorm"
        assert model_config.qk_layernorm is True
        assert model_config.layernorm_zero_centered_gamma is True
        assert model_config.attention_output_gate is True
        assert model_config.experimental_attention_variant == "gdn"
        assert model_config.linear_attention_freq == [1, 0, 1, 0, 1, 0, 0, 0] * 2
        assert model_config.linear_num_key_heads == 2
        assert model_config.linear_num_value_heads == 4
        assert model_config.rotary_percent == 0.25
        assert model_config.position_embedding_type == "rope"
        assert model_config.share_embeddings_and_output_weights is False
        assert model_config.mtp_num_layers == 1
        assert model_config.mtp_loss_scaling_factor == 0.1
        assert model_config.hetereogenous_dist_checkpoint is True
        # The builder receives an explicit Hybrid stack with Qwen Q/K normalization.
        assert model_config.hybrid_stack_spec is qwen35_hybrid_stack_spec
        if bridge_class is Qwen35MoEBridge:
            assert model_config.num_moe_experts == 4
            assert model_config.moe_router_topk == 2
            assert model_config.moe_ffn_hidden_size == 32
            assert model_config.moe_shared_expert_intermediate_size == 32
            assert model_config.moe_shared_expert_gate is True
            assert model_config.moe_router_load_balancing_type == "global_aux_loss"
            assert model_config.moe_router_dtype == "fp32"
        else:
            assert model_config.num_moe_experts is None
            assert model_config.ffn_hidden_size == 128

    @pytest.mark.parametrize(("bridge_class", "make_config"), _PARITY_CASES, ids=_PARITY_IDS)
    def test_hybrid_export_preserves_hf_layer_schedule(self, bridge_class, make_config):
        hf_config = make_config()
        model_config = bridge_class().hf_config_to_model_config(hf_config)
        model_config.finalize()
        exported = bridge_class.megatron_to_hf_config(model_config)
        assert exported["num_hidden_layers"] == hf_config.num_hidden_layers
        expected = full_attention_interval_from_hf(hf_config)
        if isinstance(expected, int):
            expected = [
                int((index + 1) % expected != 0) for index in range(hf_config.num_hidden_layers)
            ]
        assert exported["layer_types"] == [
            "linear_attention" if linear else "full_attention" for linear in expected
        ]

    @pytest.mark.parametrize(
        ("bridge_class", "make_config"),
        [(Qwen35Bridge, _qwen35_text_config), (Qwen35MoEBridge, _qwen35_moe_text_config)],
        ids=["dense", "moe"],
    )
    def test_yarn_checkpoint_maps_to_yarn_embeddings(self, bridge_class, make_config):
        model_config = bridge_class().hf_config_to_model_config(
            make_config(rope_parameters=_YARN_ROPE_PARAMETERS)
        )

        assert model_config.position_embedding_type == "yarn"
        assert model_config.transformer.yarn_rotary_scaling_factor == 4.0
        assert model_config.transformer.yarn_original_max_position_embeddings == 1024

    @pytest.mark.parametrize(
        ("bridge_class", "config"),
        [(Qwen35Bridge, _qwen35_text_config()), (Qwen35MoEBridge, _qwen35_moe_text_config())],
        ids=["dense", "moe"],
    )
    def test_builder_default_layer_spec_builds_gated_delta_net_block(self, bridge_class, config):
        from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
        from megatron.core.ssm.gated_delta_net import GatedDeltaNet

        model_config = bridge_class().hf_config_to_model_config(config)
        spec = model_config.hybrid_stack_spec()
        assert spec.submodules.gdn_layer.submodules.self_attention.module is GatedDeltaNet
        attention = spec.submodules.attention_layer.submodules.self_attention.submodules
        assert attention.q_layernorm is TESpecProvider().layer_norm(for_qk=True)
        assert attention.k_layernorm is TESpecProvider().layer_norm(for_qk=True)

    @pytest.mark.parametrize(
        "bridge_class,make_config",
        [(Qwen35Bridge, _qwen35_text_config), (Qwen35MoEBridge, _qwen35_moe_text_config)],
    )
    def test_hybrid_config_export_preserves_nonperiodic_attention_schedule(
        self, bridge_class, make_config
    ):
        hf_config = make_config(
            num_hidden_layers=4,
            layer_types=[
                "full_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        )
        bridge = bridge_class()
        config = bridge.hf_config_to_model_config(hf_config)
        restored = type(hf_config)(**bridge.megatron_to_hf_config(config))
        assert restored.num_hidden_layers == hf_config.num_hidden_layers
        assert restored.layer_types == hf_config.layer_types
        assert config.num_layers == 2 * hf_config.num_hidden_layers

    @pytest.mark.parametrize(
        "make_config", [_qwen35_text_config, _qwen35_moe_text_config], ids=["dense", "moe"]
    )
    @pytest.mark.parametrize("explicit_pattern", [False, True])
    def test_text_builder_assigns_all_virtual_pipeline_stages(
        self, monkeypatch, make_config, explicit_pattern
    ):
        hf_config = make_config()
        config = AutoBridge.from_hf_config(hf_config).get_model_config()
        config.pipeline_model_parallel_size = 2
        config.virtual_pipeline_model_parallel_size = 2
        config.pipeline_dtype = torch.bfloat16
        config.mtp_num_layers = 1
        if explicit_pattern:
            pattern = config.hybrid_layer_pattern
            config.hybrid_layer_pattern = "|".join(
                pattern[i : i + 4] for i in range(0, len(pattern), 4)
            )
        config.finalize()
        original_pattern = config.hybrid_layer_pattern
        builder = config.get_builder_cls()(config)
        constructor = Mock()
        monkeypatch.setattr("megatron.training.models.hybrid.HybridModel", constructor)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        group = Mock()
        group.size.return_value = 2
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
        pg_collection = SimpleNamespace(pp=group)

        for vp_stage in range(2):
            for pp_rank in range(2):
                group.rank.return_value = pp_rank
                monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: pp_rank)
                builder.build_model(pg_collection, vp_stage=vp_stage)
                kwargs = constructor.call_args.kwargs
                main, mtp = kwargs["hybrid_layer_pattern"].split("/", 1)
                assert mtp == config.mtp_hybrid_override_pattern
                layers, offset = select_pipeline_segment(
                    main,
                    kwargs["config"],
                    pg_collection.pp,
                    vp_stage,
                    tp_group=group,
                    dp_cp_group=group,
                )
                assert len(layers) == 4
                assert offset == 4 * (vp_stage * 2 + pp_rank)
                assert kwargs["pre_process"] is (vp_stage == 0 and pp_rank == 0)
                assert kwargs["post_process"] is (vp_stage == 1 and pp_rank == 1)
        assert config.hybrid_layer_pattern == original_pattern


@pytest.mark.parametrize("moe", [False, True])
def test_split_layer_mapping_preserves_hf_block_and_expert_indices(moe):
    templates = (
        Qwen35MoEBridge._get_moe_lm_mappings(experts_packed=False)
        if moe
        else Qwen35Bridge._get_dense_lm_mappings()
    )
    registry = MegatronMappingRegistry(*qwen35_hybrid_mappings(templates, num_layers=4))
    qkv = registry.megatron_to_hf_lookup("decoder.layers.6.self_attention.linear_qkv.weight")
    assert qkv.hf_param["q"] == "model.layers.3.self_attn.q_proj.weight"
    assert registry.hf_to_megatron_lookup(qkv.hf_param["q"]).megatron_param == qkv.megatron_param

    if moe:
        mlp_name = "decoder.layers.7.mlp.experts.linear_fc1.weight12"
        hf_gate = "model.layers.3.mlp.experts.12.gate_proj.weight"
    else:
        mlp_name = "decoder.layers.7.mlp.linear_fc1.weight"
        hf_gate = "model.layers.3.mlp.gate_proj.weight"
    mlp = registry.megatron_to_hf_lookup(mlp_name)
    assert mlp.hf_param["gate"] == hf_gate
    assert registry.hf_to_megatron_lookup(hf_gate).megatron_param == mlp_name


def test_split_layer_mapping_preserves_zero_centered_gdn_conversion():
    mappings = qwen35_hybrid_mappings(Qwen35MoEBridge._get_moe_lm_mappings(), num_layers=4)
    registry = MegatronMappingRegistry(*mappings)
    norm = registry.megatron_to_hf_lookup("decoder.layers.4.self_attention.out_norm.weight")
    assert type(norm).__name__ == "RMSNorm2ZeroCenteredRMSNormMapping"
    assert norm.hf_param == "model.layers.2.linear_attn.norm.weight"


@pytest.mark.parametrize(
    "pipeline",
    [
        dict(
            pipeline_model_parallel_size=2,
            num_layers_in_first_pipeline_stage=34,
            num_layers_in_last_pipeline_stage=46,
        ),
        dict(pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2),
        dict(pipeline_model_parallel_size=2),
    ],
)
def test_pipeline_pattern_preserves_block_stage_assignment(pipeline):
    config = TransformerConfig(
        num_layers=80,
        hidden_size=64,
        num_attention_heads=4,
        pipeline_dtype=torch.bfloat16,
        **pipeline,
    )
    pattern = "".join(("*" if (i + 1) % 4 == 0 else "G") + "E" for i in range(40))
    segments = qwen35_pipeline_pattern(config, pattern).split("|")
    assert "".join(segments) == pattern

    pp_size = config.pipeline_model_parallel_size
    vp_size = config.virtual_pipeline_model_parallel_size
    for index, segment in enumerate(segments):
        vp_stage, pp_rank = divmod(index, pp_size)
        offset = get_transformer_layer_offset(config, vp_stage if vp_size else None, pp_rank)
        assert sum(map(len, segments[:index])) == offset
