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
# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/tests/unit_tests/models/hybrid/test_hybrid_model_config.py
# Source license: Apache-2.0.

import re
from dataclasses import dataclass, fields
from unittest.mock import Mock, patch

import pytest
import torch.nn.functional as F
from transformers import DeepseekV3Config

from megatron.bridge.models.common.base import ModelConfig
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.hybrid.hybrid_builder import BridgeHybridModelBuilder
from megatron.bridge.models.hybrid.model_config import (
    BridgeHybridMLAModelConfig,
    BridgeHybridModelConfig,
)
from megatron.bridge.models.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.core._rank_utils import safe_get_rank
from megatron.core.activations import squared_relu
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_inference_stack_spec,
    hybrid_stack_spec,
)
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnBackend
from megatron.training.models.hybrid import HybridModelConfig


def _make_model_config(**kwargs) -> BridgeHybridModelConfig:
    transformer_kwargs = dict(
        num_layers=4,
        hidden_size=128,
        num_attention_heads=4,
        ffn_hidden_size=256,
        activation_func=squared_relu,
    )
    transformer_kwargs.update(kwargs.pop("transformer", {}))
    values = dict(
        transformer=TransformerConfig(**transformer_kwargs),
        vocab_size=256,
        hybrid_layer_pattern="ME*-",
    )
    values.update(kwargs)
    return BridgeHybridModelConfig(**values)


def _core_attention(stack_spec: ModuleSpec):
    return stack_spec.submodules.attention_layer.submodules.self_attention.submodules.core_attention


class TestStrictOverrides:
    def test_flat_assignment_routes_to_declared_field_owner(self) -> None:
        config = _make_model_config()

        config.hidden_size = 256
        config.mamba_chunk_size = 256

        assert config.transformer.hidden_size == 256
        assert "hidden_size" not in config.__dict__
        assert config.__dict__["mamba_chunk_size"] == 256

        with pytest.raises(AttributeError, match="declares a field"):
            config.keep_mtp_spec_in_bf16 = True

    def test_is_a_bridge_hybrid_config_with_its_own_builder(self) -> None:
        config = _make_model_config()

        assert isinstance(config, HybridModelConfig)
        assert BridgeHybridModelConfig.transformer_config_class is TransformerConfig
        assert config.get_builder_cls() is BridgeHybridModelBuilder

    def test_subclass_selects_nested_transformer_type(self) -> None:
        @dataclass
        class SpecializedTransformerConfig(TransformerConfig):
            specialized_field: bool = False

        @dataclass(kw_only=True)
        class SpecializedModelConfig(BridgeHybridModelConfig):
            transformer_config_class = SpecializedTransformerConfig

        config = SpecializedModelConfig(
            transformer=SpecializedTransformerConfig(
                num_layers=2, hidden_size=128, num_attention_heads=4
            ),
            vocab_size=256,
        )
        config.specialized_field = True

        assert config.transformer.specialized_field is True
        assert config.get_builder_cls() is BridgeHybridModelBuilder


class TestSerialization:
    def test_round_trip_restores_activation_and_builder(self) -> None:
        config = _make_model_config(mamba_chunk_size=256, hybrid_layer_pattern="M*E-")

        serialized = config.as_dict()
        restored = ModelConfig.from_dict(serialized)

        assert serialized["transformer"]["activation_func"] == "squared_relu"
        assert serialized["mamba_chunk_size"] == 256
        assert isinstance(restored, BridgeHybridModelConfig)
        assert restored.hybrid_layer_pattern == "M*E-"
        assert restored.mamba_chunk_size == 256
        assert restored.get_builder_cls() is BridgeHybridModelBuilder

    def test_from_dict_restores_activation_immediately(self) -> None:
        restored = BridgeHybridModelConfig.from_dict(_make_model_config().as_dict())

        assert restored.transformer.activation_func is squared_relu

    def test_rejects_unregistered_activation(self) -> None:
        config = _make_model_config()
        config.transformer.activation_func = lambda value: value

        with pytest.raises(ValueError, match="Cannot serialize unregistered activation"):
            config.as_dict()


class TestFinalize:
    @pytest.mark.parametrize(
        ("mtp_num_layers", "repeated", "expected_pattern", "expected_mtp_num_layers"),
        [
            (2, True, "ME*-/*E/*E", 2),
            (None, True, "ME*-/*E", 1),
            (1, True, "ME*-/*E", 1),
            (2, False, "ME*-/*E/*E", 2),
        ],
    )
    def test_appends_one_mtp_pattern_per_depth(
        self, mtp_num_layers, repeated, expected_pattern, expected_mtp_num_layers
    ) -> None:
        config = _make_model_config(
            transformer=dict(
                mtp_num_layers=mtp_num_layers,
                mtp_hybrid_override_pattern="*E",
                mtp_use_repeated_layer=repeated,
            )
        )

        config.finalize()

        assert config.hybrid_layer_pattern == expected_pattern
        assert config.mtp_num_layers == expected_mtp_num_layers
        assert config.num_layers == 4

    def test_mtp_pattern_replaces_an_already_combined_pattern(self) -> None:
        # Deserialized checkpoints and recipe layouts carry the combined pattern.
        config = _make_model_config(
            hybrid_layer_pattern="ME|*-/*E",
            transformer=dict(
                mtp_num_layers=2, mtp_hybrid_override_pattern="*E", mtp_use_repeated_layer=True
            ),
        )

        config.finalize()

        assert config.hybrid_layer_pattern == "ME|*-/*E/*E"
        assert config.num_layers == 4

    def test_derives_num_layers_from_the_pattern(self) -> None:
        config = _make_model_config(hybrid_layer_pattern="MEM|*E-", transformer=dict(num_layers=0))

        config.finalize()

        assert config.num_layers == 6

    def test_rejects_num_layers_that_disagree_with_the_pattern(self) -> None:
        config = _make_model_config(transformer=dict(num_layers=5))

        with pytest.raises(ValueError, match=r"num_layers \(5\) does not match the 4 layers"):
            config.finalize()

    def test_moves_deprecated_override_pattern(self) -> None:
        config = _make_model_config(hybrid_layer_pattern=None, hybrid_override_pattern="M*E-")

        # Distributed MLM tests run on every rank; only rank zero emits the warning.
        if safe_get_rank() == 0:
            with pytest.warns(DeprecationWarning, match="hybrid_override_pattern is deprecated"):
                config.finalize()
        else:
            config.finalize()

        assert config.hybrid_layer_pattern == "M*E-"
        assert config.hybrid_override_pattern is None

    @pytest.mark.parametrize("method", ["validate", "finalize"])
    def test_rejects_both_pattern_fields(self, method) -> None:
        config = _make_model_config(hybrid_override_pattern="M*E-")

        with pytest.raises(ValueError, match="cannot both be specified"):
            getattr(config, method)()

    @pytest.mark.parametrize("method", ["validate", "finalize"])
    def test_rejects_nonpositive_mamba_chunk_size(self, method) -> None:
        config = _make_model_config(mamba_chunk_size=0)

        with pytest.raises(ValueError, match="mamba_chunk_size must be at least 1"):
            getattr(config, method)()

    def test_validate_does_not_change_the_config(self) -> None:
        config = _make_model_config(
            hybrid_layer_pattern=None,
            hybrid_override_pattern="ME*-",
            transformer=dict(
                num_layers=0, mtp_hybrid_override_pattern="*E", mtp_use_repeated_layer=True
            ),
        )
        serialized = config.as_dict()

        config.validate()

        assert config.as_dict() == serialized


class TestBuilderStackSpec:
    def test_default_is_the_transformer_engine_stack(self) -> None:
        resolved = BridgeHybridModelBuilder(_make_model_config())._resolve_hybrid_stack_spec()

        assert resolved is hybrid_stack_spec

    def test_modelopt_default_remaps_fused_layer_norms(self) -> None:
        config = _make_model_config(restore_modelopt_state=True)
        modelopt_spec = ModuleSpec(module=object)

        with patch(
            "megatron.bridge.models.hybrid.hybrid_builder.get_hybrid_stack_modelopt_spec",
            return_value=modelopt_spec,
        ) as get_spec:
            resolved = BridgeHybridModelBuilder(config)._resolve_hybrid_stack_spec()

        get_spec.assert_called_once_with(local_core_attention=False, remap_te_layernorm=True)
        assert resolved is modelopt_spec

    def test_inference_optimized_default(self) -> None:
        config = _make_model_config(transformer=dict(transformer_impl="inference_optimized"))

        resolved = BridgeHybridModelBuilder(config)._resolve_hybrid_stack_spec()

        assert resolved is hybrid_inference_stack_spec

    def test_wide_residual_default_uses_megatron_core_selection(self) -> None:
        config = _make_model_config()
        config.transformer.wide_residual = Mock()
        wide_spec = ModuleSpec(module=object)

        with patch(
            "megatron.bridge.models.hybrid.hybrid_builder.resolve_hybrid_stack_spec",
            return_value=wide_spec,
        ) as resolve:
            resolved = BridgeHybridModelBuilder(config)._resolve_hybrid_stack_spec()

        resolve.assert_called_once_with(config)
        assert resolved is wide_spec

    def test_module_spec_is_used_directly(self) -> None:
        stack_spec = ModuleSpec(module=object)

        resolved = BridgeHybridModelBuilder(
            _make_model_config(hybrid_stack_spec=stack_spec)
        )._resolve_hybrid_stack_spec()

        assert resolved is stack_spec

    def test_factories_take_no_arguments_or_the_model_config(self) -> None:
        stack_spec = ModuleSpec(module=object)
        seen = []

        def config_factory(config):
            seen.append(config)
            return stack_spec

        config = _make_model_config(hybrid_stack_spec=config_factory)
        assert BridgeHybridModelBuilder(config)._resolve_hybrid_stack_spec() is stack_spec
        assert seen == [config]

        config.hybrid_stack_spec = lambda: stack_spec
        assert BridgeHybridModelBuilder(config)._resolve_hybrid_stack_spec() is stack_spec

    @pytest.mark.parametrize("attention_backend", [AttnBackend.local, "local"])
    def test_local_attention_uses_megatron_core_attention_on_a_copy(
        self, attention_backend
    ) -> None:
        config = _make_model_config(transformer=dict(attention_backend=attention_backend))

        resolved = BridgeHybridModelBuilder(config)._resolve_hybrid_stack_spec()

        assert resolved is not hybrid_stack_spec
        assert _core_attention(resolved) is DotProductAttention
        assert _core_attention(hybrid_stack_spec) is TEDotProductAttention

    def test_nondefault_mamba_chunk_size_is_applied_to_a_copy(self) -> None:
        resolved = BridgeHybridModelBuilder(
            _make_model_config(mamba_chunk_size=256)
        )._resolve_hybrid_stack_spec()

        assert resolved.submodules.mamba_layer.submodules.mixer.params["chunk_size"] == 256
        assert "chunk_size" not in hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.params

    @patch("megatron.training.models.hybrid.HybridModel")
    def test_build_model_passes_the_resolved_spec_and_restores_the_config(self, mock_model) -> None:
        def config_factory(config):
            return hybrid_stack_spec

        config = _make_model_config(hybrid_stack_spec=config_factory, mamba_chunk_size=256)
        pg_collection = Mock()

        BridgeHybridModelBuilder(config).build_model(
            pg_collection, pre_process=True, post_process=True
        )

        resolved = mock_model.call_args.kwargs["hybrid_stack_spec"]
        assert resolved.submodules.mamba_layer.submodules.mixer.params["chunk_size"] == 256
        assert mock_model.call_args.kwargs["hybrid_layer_pattern"] == "ME*-"
        assert mock_model.call_args.kwargs["config"] is config.transformer
        assert config.hybrid_stack_spec is config_factory


def _mla_transformer(**kwargs) -> MLATransformerConfig:
    values = dict(
        num_layers=4,
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


def _make_mla_model_config(**kwargs) -> BridgeHybridMLAModelConfig:
    values = dict(
        transformer=_mla_transformer(),
        vocab_size=256,
        hybrid_layer_pattern="*E*E",
        rotary_base=50_000.0,
    )
    values.update(kwargs)
    return BridgeHybridMLAModelConfig(**values)


class _HybridMLABridge(MegatronModelBridge):
    """A builder-only MLA bridge: no legacy MLA provider selects the MLA rope fields."""

    MODEL_CONFIG_CLASS = BridgeHybridMLAModelConfig

    def mapping_registry(self) -> MegatronMappingRegistry:
        return MegatronMappingRegistry()


class TestMLAConfig:
    def test_rotary_settings_live_on_the_mla_transformer(self) -> None:
        """MLA attention reads rotary_base and rotary_percent from its transformer config, which declares both."""
        config = _make_mla_model_config()

        assert BridgeHybridMLAModelConfig.transformer_config_class is MLATransformerConfig
        assert config.transformer.rotary_base == 50_000.0

        config.rotary_base = 20_000.0
        config.rotary_percent = 0.5

        assert config.transformer.rotary_base == 20_000.0
        assert config.transformer.rotary_percent == 0.5
        assert "rotary_base" not in config.__dict__

        config.transformer.rotary_base = 30_000.0
        assert config.rotary_base == 30_000.0

    def test_rotary_settings_are_stored_only_on_the_transformer(self) -> None:
        """The outer names are constructor arguments, so only the nested config is serialized and overridden."""
        config = _make_mla_model_config(rotary_percent=0.5)
        serialized = config.as_dict()

        assert {"rotary_base", "rotary_percent"}.isdisjoint(field.name for field in fields(config))
        assert {"rotary_base", "rotary_percent"}.isdisjoint(serialized)
        assert serialized["transformer"]["rotary_base"] == 50_000.0
        assert serialized["transformer"]["rotary_percent"] == 0.5

    def test_constructor_copies_a_rotary_value_the_transformer_kept_at_its_default(self) -> None:
        config = _make_mla_model_config(rotary_base=20_000.0, rotary_percent=0.5)

        assert config.transformer.rotary_base == 20_000.0
        assert config.transformer.rotary_percent == 0.5

    def test_constructor_keeps_the_transformer_rotary_value(self) -> None:
        config = BridgeHybridMLAModelConfig(
            transformer=_mla_transformer(rotary_base=20_000.0),
            vocab_size=256,
            hybrid_layer_pattern="*E*E",
        )

        assert config.rotary_base == config.transformer.rotary_base == 20_000.0

    def test_constructor_accepts_matching_rotary_values(self) -> None:
        config = _make_mla_model_config(transformer=_mla_transformer(rotary_base=50_000.0))

        assert config.transformer.rotary_base == 50_000.0

    @pytest.mark.parametrize(
        ("name", "nested", "outer"),
        [("rotary_base", 20_000.0, 50_000.0), ("rotary_percent", 0.5, 0.25)],
    )
    def test_constructor_rejects_conflicting_rotary_values(self, name, nested, outer) -> None:
        message = re.escape(f"{name}={outer!r} conflicts with transformer.{name}={nested!r}")

        with pytest.raises(ValueError, match=message):
            _make_mla_model_config(transformer=_mla_transformer(**{name: nested}), **{name: outer})

    def test_flat_kwargs_put_rotary_settings_on_the_transformer(self) -> None:
        model_kwargs, transformer_kwargs = MegatronModelBridge._partition_model_config_kwargs(
            {"rotary_base": 50_000.0, "rotary_percent": 0.5, "vocab_size": 256},
            BridgeHybridMLAModelConfig,
            MLATransformerConfig,
        )

        assert model_kwargs == {"vocab_size": 256}
        assert transformer_kwargs == {"rotary_base": 50_000.0, "rotary_percent": 0.5}

    def test_saved_config_keeps_an_edited_transformer_value(self) -> None:
        serialized = _make_mla_model_config().as_dict()
        serialized["transformer"]["rotary_base"] = 20_000.0

        restored = ModelConfig.from_dict(serialized)

        assert restored.rotary_base == restored.transformer.rotary_base == 20_000.0

    def test_saved_config_with_conflicting_rotary_values_is_rejected(self) -> None:
        """A run config that also stores the outer value is checked before the value is copied."""
        serialized = _make_mla_model_config().as_dict()
        serialized["rotary_base"] = 50_000.0
        serialized["transformer"]["rotary_base"] = 20_000.0

        with pytest.raises(ValueError, match="conflicts with transformer.rotary_base"):
            ModelConfig.from_dict(serialized)

    def test_round_trip_restores_the_mla_transformer(self) -> None:
        serialized = _make_mla_model_config().as_dict()
        restored = ModelConfig.from_dict(serialized)

        assert "rotary_base" not in serialized
        assert serialized["transformer"]["rotary_base"] == 50_000.0
        assert isinstance(restored, BridgeHybridMLAModelConfig)
        assert isinstance(restored.transformer, MLATransformerConfig)
        assert restored.rotary_base == 50_000.0
        assert restored.q_lora_rank is None
        assert restored.kv_lora_rank == 32
        assert restored.get_builder_cls() is BridgeHybridModelBuilder

    def test_finalize_derives_layers_and_keeps_mla(self) -> None:
        config = _make_mla_model_config()
        config.transformer.num_layers = None

        config.finalize()

        assert config.num_layers == 4
        assert config.multi_latent_attention is True
        assert config.rotary_base == 50_000.0

    def test_bridge_maps_mla_rope_fields_from_the_model_config(self) -> None:
        """The MLA model config, not a legacy MLA provider, selects the flat MLA rope fields."""
        hf_config = DeepseekV3Config(
            num_hidden_layers=2,
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_routed_experts=4,
            num_experts_per_tok=2,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_nope_head_dim=16,
            qk_rope_head_dim=16,
            v_head_dim=16,
            vocab_size=256,
            rope_theta=50_000.0,
            rope_scaling={
                "type": "yarn",
                "factor": 40,
                "beta_fast": 32,
                "beta_slow": 1,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 4096,
            },
        )

        kwargs = _HybridMLABridge().hf_config_to_model_config_kwargs(hf_config)

        assert "_mla_rope_params" not in kwargs
        assert kwargs["position_embedding_type"] == "rope"
        assert kwargs["rotary_scaling_factor"] == 40
        assert kwargs["original_max_position_embeddings"] == 4096
        assert "yarn_rotary_scaling_factor" not in kwargs
