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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c209ed397fb36272c2216053058cda4e37802f2d/tests/unit_tests/models/test_gpt_model_config.py
# Source license: Apache-2.0.

"""Unit tests for the builder-backed GPT model configs."""

import re
from dataclasses import fields

import pytest
import torch.nn.functional as F

from megatron.bridge.models.common.base import ModelConfig
from megatron.bridge.models.gpt.model_config import BridgeGPTMLAModelConfig, BridgeGPTModelConfig
from megatron.bridge.models.llama.llama_bridge import LlamaBridge
from megatron.bridge.models.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.training.models.gpt import GPTModelBuilder


def _make_model_config() -> BridgeGPTModelConfig:
    transformer = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        ffn_hidden_size=256,
        transformer_impl="local",
        use_cpu_initialization=True,
        activation_func=F.silu,
    )
    return BridgeGPTModelConfig(
        transformer=transformer, vocab_size=256, seq_length=128, position_embedding_type="rope"
    )


def _mla_transformer(**kwargs) -> MLATransformerConfig:
    values = dict(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        ffn_hidden_size=256,
        activation_func=F.silu,
        q_lora_rank=None,
        kv_lora_rank=32,
        qk_head_dim=16,
        qk_pos_emb_head_dim=16,
        v_head_dim=16,
    )
    values.update(kwargs)
    return MLATransformerConfig(**values)


def _make_mla_model_config(**kwargs) -> BridgeGPTMLAModelConfig:
    values = dict(
        transformer=_mla_transformer(),
        vocab_size=256,
        seq_length=128,
        position_embedding_type="rope",
        rotary_base=50_000.0,
    )
    values.update(kwargs)
    return BridgeGPTMLAModelConfig(**values)


def test_builds_through_megatron_lm_gpt_builder() -> None:
    config = _make_model_config()

    assert config.builder == "megatron.training.models.gpt.GPTModelBuilder"
    assert config.get_builder_cls() is GPTModelBuilder


def test_flat_assignment_routes_to_declared_field_owner() -> None:
    config = _make_model_config()

    config.hidden_size = 256
    config.rotary_base = 500_000

    assert config.transformer.hidden_size == 256
    assert "hidden_size" not in config.__dict__
    assert config.rotary_base == 500_000
    assert "rotary_base" in config.__dict__

    with pytest.raises(AttributeError, match="declares a field"):
        config.phantom_override = True


def test_model_config_round_trip_restores_runtime_activation() -> None:
    config = _make_model_config()

    serialized = config.as_dict()
    restored = ModelConfig.from_dict(serialized)

    assert serialized["transformer"]["activation_func"] == "silu"
    assert isinstance(restored, BridgeGPTModelConfig)
    assert restored.transformer.activation_func == "silu"
    assert restored.get_builder_cls() is GPTModelBuilder

    restored.finalize()
    assert restored.transformer.activation_func is F.silu


def test_bridge_model_config_from_dict_restores_activation_immediately() -> None:
    restored = BridgeGPTModelConfig.from_dict(_make_model_config().as_dict())

    assert restored.transformer.activation_func is F.silu


def test_from_dict_rejects_another_config_class() -> None:
    serialized = _make_model_config().as_dict()

    with pytest.raises(
        TypeError, match="Expected BridgeGPTMLAModelConfig, got BridgeGPTModelConfig"
    ):
        BridgeGPTMLAModelConfig.from_dict(serialized)


def test_model_config_rejects_unregistered_activation() -> None:
    config = _make_model_config()
    config.transformer.activation_func = lambda value: value

    with pytest.raises(ValueError, match="Cannot serialize unregistered activation"):
        config.as_dict()


def test_model_config_mapping_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="phantom_field"):
        LlamaBridge._partition_model_config_kwargs(
            {"phantom_field": True}, BridgeGPTModelConfig, TransformerConfig
        )


class TestGPTMLAModelConfig:
    def test_rotary_settings_are_stored_only_on_the_mla_transformer(self) -> None:
        """MLA attention reads rotary_base and rotary_percent from the transformer config, which alone stores them."""
        config = _make_mla_model_config(rotary_percent=0.5)

        assert BridgeGPTMLAModelConfig.transformer_config_class is MLATransformerConfig
        assert {"rotary_base", "rotary_percent"}.isdisjoint(field.name for field in fields(config))
        assert config.transformer.rotary_base == 50_000.0
        assert config.transformer.rotary_percent == 0.5

        config.rotary_base = 20_000.0
        assert config.transformer.rotary_base == 20_000.0
        assert "rotary_base" not in config.__dict__

        serialized = config.as_dict()
        assert {"rotary_base", "rotary_percent"}.isdisjoint(serialized)
        assert serialized["transformer"]["rotary_base"] == 20_000.0
        assert serialized["transformer"]["rotary_percent"] == 0.5

    def test_constructor_rejects_conflicting_rotary_values(self) -> None:
        message = re.escape("rotary_base=50000.0 conflicts with transformer.rotary_base=20000.0")

        with pytest.raises(ValueError, match=message):
            _make_mla_model_config(transformer=_mla_transformer(rotary_base=20_000.0))

    def test_round_trip_restores_the_mla_transformer(self) -> None:
        restored = BridgeGPTMLAModelConfig.from_dict(_make_mla_model_config().as_dict())

        assert isinstance(restored, BridgeGPTMLAModelConfig)
        assert isinstance(restored.transformer, MLATransformerConfig)
        assert restored.rotary_base == restored.transformer.rotary_base == 50_000.0
        assert restored.q_lora_rank is None
        assert restored.kv_lora_rank == 32
        assert restored.transformer.activation_func is F.silu
        assert restored.get_builder_cls() is GPTModelBuilder
