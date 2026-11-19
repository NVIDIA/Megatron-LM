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
# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/0f06205924d4e19b0723aa8c33e34a1959ca17fe/tests/unit_tests/models/qwen_vl/test_qwen35_vl_bridge.py
# Source license: Apache-2.0.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import (
    Qwen35VLBridge,
    Qwen35VLMoEBridge,
    _get_vision_mappings,
)
from megatron.bridge.models.qwen_vl.qwen35_vl_config import (
    Qwen35VLModelConfig,
    Qwen35VLTransformerConfig,
)


def _make_dense_text_config():
    """Create a mock text config matching Qwen3.5-27B dense architecture."""
    cfg = Mock(spec=[])
    cfg.num_hidden_layers = 64
    cfg.hidden_size = 5120
    cfg.intermediate_size = 17408
    cfg.num_attention_heads = 24
    cfg.num_key_value_heads = 4
    cfg.initializer_range = 0.02
    cfg.rms_norm_eps = 1e-6
    cfg.vocab_size = 248320
    cfg.max_position_embeddings = 262144
    cfg.rope_theta = 10000000.0
    cfg.tie_word_embeddings = False
    cfg.hidden_act = "silu"
    cfg.attention_bias = False
    cfg.head_dim = 256
    cfg.full_attention_interval = 4
    cfg.rope_parameters = {"partial_rotary_factor": 0.25, "rope_theta": 10000000.0}
    cfg.rope_scaling = {"mrope_section": [11, 11, 10]}
    cfg.linear_conv_kernel_dim = 4
    cfg.linear_key_head_dim = 128
    cfg.linear_value_head_dim = 128
    cfg.linear_num_key_heads = 16
    cfg.linear_num_value_heads = 48
    cfg.bos_token_id = 248045
    cfg.eos_token_id = 248044
    cfg.num_nextn_predict_layers = None
    cfg.torch_dtype = "bfloat16"
    return cfg


def _make_moe_text_config():
    """Create a mock text config matching Qwen3.5-397B-A17B MoE architecture."""
    cfg = Mock(spec=[])
    cfg.num_hidden_layers = 60
    cfg.hidden_size = 4096
    cfg.intermediate_size = 1024
    cfg.num_attention_heads = 32
    cfg.num_key_value_heads = 2
    cfg.initializer_range = 0.02
    cfg.rms_norm_eps = 1e-6
    cfg.vocab_size = 248320
    cfg.max_position_embeddings = 262144
    cfg.rope_theta = 10000000.0
    cfg.tie_word_embeddings = False
    cfg.hidden_act = "silu"
    cfg.attention_bias = False
    cfg.head_dim = 256
    cfg.full_attention_interval = 4
    cfg.rope_parameters = {"partial_rotary_factor": 0.25, "rope_theta": 10000000.0}
    cfg.rope_scaling = {"mrope_section": [11, 11, 10]}
    cfg.linear_conv_kernel_dim = 4
    cfg.linear_key_head_dim = 128
    cfg.linear_value_head_dim = 128
    cfg.linear_num_key_heads = 16
    cfg.linear_num_value_heads = 64
    cfg.moe_intermediate_size = 1024
    cfg.num_experts = 512
    cfg.num_experts_per_tok = 10
    cfg.shared_expert_intermediate_size = 4096
    cfg.bos_token_id = 248045
    cfg.eos_token_id = 248046
    cfg.num_nextn_predict_layers = None
    cfg.torch_dtype = "bfloat16"
    return cfg


def _make_vision_config():
    """Create a minimal mock vision config."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

    return Qwen3_5VisionConfig()


def _make_mock_pretrained(text_config, vision_config, tie_word_embeddings=False):
    """Create a minimal VLM pretrained wrapper for provider tests."""
    pretrained = Mock(spec=PreTrainedCausalLM)
    config = Mock()
    config.text_config = text_config
    config.vision_config = vision_config
    config.tie_word_embeddings = tie_word_embeddings
    config.vision_start_token_id = 248053
    config.vision_end_token_id = 248054
    config.image_token_id = 248056
    config.video_token_id = 248057
    config.audio_token_id = 248076
    pretrained.config = config
    return pretrained


# =====================================================================
# Tests for Qwen35VLBridge (Dense)
# =====================================================================


class TestQwen35VLBridgeMappingRegistry:
    @pytest.fixture
    def bridge(self):
        bridge = Qwen35VLBridge()
        bridge.hf_config = SimpleNamespace(text_config=_make_dense_text_config())
        return bridge

    def _get_mapping_names(self, registry):
        names = []
        for mapping in registry.mappings:
            if hasattr(mapping, "megatron_param"):
                names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                names.append(hf)
        return names

    def test_mapping_registry_type(self, bridge):
        registry = bridge.mapping_registry()
        assert isinstance(registry, MegatronMappingRegistry)
        assert len(registry.mappings) > 0

    def test_mapping_registry_has_embeddings(self, bridge):
        names = self._get_mapping_names(bridge.mapping_registry())
        assert any("embed_tokens" in n or "word_embeddings" in n for n in names)

    def test_mapping_registry_has_output_layer(self, bridge):
        names = self._get_mapping_names(bridge.mapping_registry())
        assert any("lm_head" in n or "output_layer" in n for n in names)

    def test_mapping_registry_has_gdn_mappings(self, bridge):
        names = self._get_mapping_names(bridge.mapping_registry())
        assert any("in_proj" in n for n in names), "Should contain GDN in_proj mappings"
        assert any("out_proj" in n for n in names), "Should contain GDN out_proj mappings"
        assert any("A_log" in n for n in names), "Should contain GDN A_log mappings"
        assert any("conv1d" in n for n in names), "Should contain GDN conv1d mappings"
        assert any("out_norm" in n or "linear_attn.norm" in n for n in names)

    def test_mapping_registry_has_dense_mlp(self, bridge):
        names = self._get_mapping_names(bridge.mapping_registry())
        assert any("gate_proj" in n for n in names), "Should contain gate_proj for dense MLP"
        assert any("up_proj" in n for n in names), "Should contain up_proj for dense MLP"
        assert any("down_proj" in n for n in names), "Should contain down_proj for dense MLP"

    def test_mapping_registry_has_no_moe(self, bridge):
        names = self._get_mapping_names(bridge.mapping_registry())
        assert not any(
            "router" in n or "experts" in n for n in names
        ), "Dense model should not have MoE mappings"

    def test_mapping_registry_has_vision_params(self, bridge):
        names = self._get_mapping_names(bridge.mapping_registry())
        assert any("visual" in n or "vision_model" in n for n in names)

    def test_mapping_registry_has_qkv(self, bridge):
        names = self._get_mapping_names(bridge.mapping_registry())
        assert any("linear_qkv" in n for n in names)

    def test_mapping_registry_has_vision_patch_embed(self, bridge):
        names = self._get_mapping_names(bridge.mapping_registry())
        assert any("patch_embed" in n for n in names)


# =====================================================================
# Tests for Qwen35TokenClassificationBridge
# =====================================================================


# =====================================================================
# Tests for Qwen35VLMoEBridge
# =====================================================================


class TestQwen35VLMoEBridgeMappingRegistry:
    @pytest.fixture
    def bridge(self):
        bridge = Qwen35VLMoEBridge()
        bridge.hf_config = SimpleNamespace(text_config=_make_moe_text_config())
        return bridge

    def _get_mapping_names(self, registry):
        names = []
        for mapping in registry.mappings:
            if hasattr(mapping, "megatron_param"):
                names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                names.append(hf)
        return names

    def test_mapping_registry_type(self, bridge):
        registry = bridge.mapping_registry()
        assert isinstance(registry, MegatronMappingRegistry)

    def test_mapping_registry_has_moe_mappings(self, bridge):
        names = self._get_mapping_names(bridge.mapping_registry())
        assert any("router" in n or "gate.weight" in n for n in names), "Should contain MoE router"
        assert any("experts" in n for n in names), "Should contain expert MLPs"
        assert any("shared_expert" in n for n in names), "Should contain shared experts"

    def test_moe_expert_mappings_wiring(self, bridge):
        # The VL MoE bridge reuses Qwen35MoEBridge._get_moe_lm_mappings; fused-vs-per-expert layout
        # selection is covered in test_qwen35_bridge. Here we only verify the VL wiring: sequential
        # (non-grouped) routed-expert mappings, used when moe_grouped_gemm=False (e.g. ModelOpt pruning),
        # are emitted with the language_model prefix.
        mappings = bridge.mapping_registry().mappings
        seq_fc1 = next(
            m
            for m in mappings
            if getattr(m, "megatron_param", "").endswith("local_experts.*.linear_fc1.weight")
        )
        assert seq_fc1.megatron_param.startswith("language_model.decoder.")

    def test_mapping_registry_has_gdn_mappings(self, bridge):
        names = self._get_mapping_names(bridge.mapping_registry())
        assert any("in_proj" in n for n in names)
        assert any("A_log" in n for n in names)
        assert any("conv1d" in n for n in names)

    def test_mapping_registry_has_vision_params(self, bridge):
        names = self._get_mapping_names(bridge.mapping_registry())
        assert any("visual" in n or "vision_model" in n for n in names)

    def test_mapping_registry_detects_qwen36_packed_mtp_experts(self, bridge):
        source = Mock()
        source.get_all_keys.return_value = [
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            "mtp.layers.0.mlp.experts.gate_up_proj",
        ]
        bridge.hf_pretrained = SimpleNamespace(state=SimpleNamespace(source=source))

        mappings = bridge.mapping_registry().mappings

        assert any(
            getattr(mapping, "hf_param", None) == "mtp.layers.*.mlp.experts.gate_up_proj"
            for mapping in mappings
        )
        assert not any(
            isinstance(getattr(mapping, "hf_param", None), dict)
            and "mtp.layers.*.mlp.experts.*.gate_proj.weight" in mapping.hf_param.values()
            for mapping in mappings
        )


class TestQwen35VLMoEBridgeExport:
    def test_maybe_modify_converted_hf_weight_keeps_explicit_mtp_expert_keys(self, monkeypatch):
        """Preserve already-expanded MTP expert keys without extra regrouping."""
        bridge = Qwen35VLMoEBridge()
        bridge.hf_config = Mock()
        bridge.hf_config.text_config = Mock()
        bridge.hf_config.text_config.num_experts = 256

        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.parallel_state.get_expert_model_parallel_world_size",
            lambda: 8,
        )

        task = Mock()
        task.param_name = (
            "language_model.mtp.layers.0.mtp_model_layer.mlp.experts.linear_fc1.weight0"
        )
        converted = {
            "mtp.layers.0.mlp.experts.0.gate_proj.weight": torch.ones(2, 2),
            "mtp.layers.0.mlp.experts.32.gate_proj.weight": 2 * torch.ones(2, 2),
            "mtp.layers.0.mlp.experts.0.up_proj.weight": 3 * torch.ones(2, 2),
            "mtp.layers.0.mlp.experts.32.up_proj.weight": 4 * torch.ones(2, 2),
        }

        result = bridge.maybe_modify_converted_hf_weight(task, dict(converted), {})

        assert result.keys() == converted.keys()
        for key in converted:
            torch.testing.assert_close(result[key], converted[key])

    def test_stream_weights_megatron_to_hf_skips_mtp_duplicate_embedding_export(self, monkeypatch):
        """Skip the last-stage MTP embedding copy when exporting shared vocab weights."""
        bridge = Qwen35VLMoEBridge()
        calls = []

        class DummyMapping:
            def megatron_to_hf(self, weight, module):
                calls.append((weight, module))
                return {}

        config = SimpleNamespace(
            share_embeddings_and_output_weights=False, pipeline_model_parallel_size=2
        )
        stage0 = SimpleNamespace(pre_process=True, config=config)
        wrapped_stage1 = SimpleNamespace(
            module=SimpleNamespace(
                pre_process=False, config=config, language_model=SimpleNamespace(mtp_process=True)
            )
        )
        duplicate_module = Mock()
        task = WeightConversionTask(
            param_name="language_model.embedding.word_embeddings.weight",
            global_param_name="language_model.embedding.word_embeddings.weight",
            mapping=DummyMapping(),
            pp_rank=1,
            vp_stage=1,
            megatron_module=duplicate_module,
            param_weight=torch.ones(1),
        )

        monkeypatch.setattr(
            Qwen35VLMoEBridge,
            "_with_progress_tracking",
            lambda self, tasks, *_args, **_kwargs: tasks,
        )
        monkeypatch.setattr(
            Qwen35VLMoEBridge,
            "maybe_modify_converted_hf_weight",
            lambda self, *_args, **_kwargs: _args[1],
        )
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.unwrap_model",
            lambda model, *_args, **_kwargs: (
                [model[0]] if isinstance(model, list) else getattr(model, "module", model)
            ),
        )

        weights = list(
            bridge.stream_weights_megatron_to_hf(
                [stage0, wrapped_stage1],
                SimpleNamespace(),
                cpu=False,
                show_progress=False,
                conversion_tasks=[task],
            )
        )

        assert weights == []
        assert len(calls) == 1
        assert calls[0] == (None, None)

    def test_stream_weights_megatron_to_hf_keeps_pre_process_embedding_owner(self, monkeypatch):
        """Keep the first-stage embedding owner active during export."""
        bridge = Qwen35VLMoEBridge()
        calls = []
        embedding_weight = torch.ones(1)
        embedding_module = Mock()

        class DummyMapping:
            def megatron_to_hf(self, weight, module):
                calls.append((weight, module))
                return {"hf.weight": weight}

        stage0 = SimpleNamespace(
            pre_process=True,
            config=SimpleNamespace(
                share_embeddings_and_output_weights=False, pipeline_model_parallel_size=2
            ),
            language_model=SimpleNamespace(mtp_process=False),
        )
        task = WeightConversionTask(
            param_name="language_model.embedding.word_embeddings.weight",
            global_param_name="language_model.embedding.word_embeddings.weight",
            mapping=DummyMapping(),
            pp_rank=0,
            vp_stage=0,
            megatron_module=embedding_module,
            param_weight=embedding_weight,
        )

        monkeypatch.setattr(
            Qwen35VLMoEBridge,
            "_with_progress_tracking",
            lambda self, tasks, *_args, **_kwargs: tasks,
        )
        monkeypatch.setattr(
            Qwen35VLMoEBridge,
            "maybe_modify_converted_hf_weight",
            lambda self, *_args, **_kwargs: _args[1],
        )
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.unwrap_model",
            lambda model, *_args, **_kwargs: (
                [model[0]] if isinstance(model, list) else getattr(model, "module", model)
            ),
        )

        weights = list(
            bridge.stream_weights_megatron_to_hf(
                [stage0], SimpleNamespace(), cpu=False, show_progress=False, conversion_tasks=[task]
            )
        )

        assert len(weights) == 1
        assert weights[0].param_name == "hf.weight"
        assert len(calls) == 1
        assert calls[0][0] is embedding_weight
        assert calls[0][1] is embedding_module

    def test_stream_weights_megatron_to_hf_does_not_skip_mtp_embedding_without_pp(
        self, monkeypatch
    ):
        """Do not suppress the only owner when pipeline parallelism is disabled."""
        bridge = Qwen35VLMoEBridge()
        calls = []
        embedding_weight = torch.ones(1)
        embedding_module = Mock()

        class DummyMapping:
            def megatron_to_hf(self, weight, module):
                calls.append((weight, module))
                return {"hf.weight": weight}

        stage0 = SimpleNamespace(
            pre_process=False,
            config=SimpleNamespace(
                share_embeddings_and_output_weights=False, pipeline_model_parallel_size=1
            ),
            language_model=SimpleNamespace(mtp_process=True),
        )
        task = WeightConversionTask(
            param_name="language_model.embedding.word_embeddings.weight",
            global_param_name="language_model.embedding.word_embeddings.weight",
            mapping=DummyMapping(),
            pp_rank=0,
            vp_stage=0,
            megatron_module=embedding_module,
            param_weight=embedding_weight,
        )

        monkeypatch.setattr(
            Qwen35VLMoEBridge,
            "_with_progress_tracking",
            lambda self, tasks, *_args, **_kwargs: tasks,
        )
        monkeypatch.setattr(
            Qwen35VLMoEBridge,
            "maybe_modify_converted_hf_weight",
            lambda self, *_args, **_kwargs: _args[1],
        )
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.unwrap_model",
            lambda model, *_args, **_kwargs: (
                [model[0]] if isinstance(model, list) else getattr(model, "module", model)
            ),
        )

        weights = list(
            bridge.stream_weights_megatron_to_hf(
                [stage0], SimpleNamespace(), cpu=False, show_progress=False, conversion_tasks=[task]
            )
        )

        assert len(weights) == 1
        assert weights[0].param_name == "hf.weight"
        assert len(calls) == 1
        assert calls[0][0] is embedding_weight
        assert calls[0][1] is embedding_module


# =====================================================================
# Tests for the builder-backed model config
# =====================================================================

_SMALL_TEXT_CONFIG = dict(
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
    # Released Qwen3.5 and Qwen3.6 configs carry one MTP layer.
    mtp_num_hidden_layers=1,
    dtype="bfloat16",
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
        "mrope_section": [1, 1, 0],
        "mrope_interleaved": True,
    },
)
_SMALL_MOE_TEXT_CONFIG = dict(
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    num_experts=4,
    num_experts_per_tok=2,
)
_SMALL_VISION_CONFIG = dict(
    depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64
)
_BRIDGE_CASES = [(Qwen35VLBridge, False), (Qwen35VLMoEBridge, True)]
_BRIDGE_IDS = ["dense", "moe"]


def _qwen35_vl_config(*, moe: bool, tie_word_embeddings: bool = False, **text_overrides):
    """Return a small Qwen3.5 vision-language HF config with the released 3 GDN : 1 attention layout."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

    text_config = {
        **_SMALL_TEXT_CONFIG,
        **(_SMALL_MOE_TEXT_CONFIG if moe else {"intermediate_size": 128}),
    }
    text_config.update(text_overrides)
    config_class = Qwen3_5MoeConfig if moe else Qwen3_5Config
    config = config_class(
        text_config=text_config,
        vision_config=_SMALL_VISION_CONFIG,
        tie_word_embeddings=tie_word_embeddings,
    )
    config.architectures = [
        "Qwen3_5MoeForConditionalGeneration" if moe else "Qwen3_5ForConditionalGeneration"
    ]
    return config


class TestQwen35VLBridgeModelConfig:
    @pytest.mark.parametrize(("bridge_class", "moe"), _BRIDGE_CASES, ids=_BRIDGE_IDS)
    def test_export_overrides_reference_language_config(self, bridge_class, moe):
        from megatron.bridge.models.conversion.auto_bridge import _drop_readonly_config_properties
        from megatron.bridge.models.conversion.utils import (
            carry_rope_into_rope_parameters,
            conform_config_to_reference,
        )

        reference = _qwen35_vl_config(moe=moe)
        config = bridge_class().hf_config_to_model_config(reference)
        # A pruned model with an extended vocabulary and no MTP must not inherit
        # the original model's layer count, vocabulary size, or MTP head.
        config.hybrid_layer_pattern = "G" + ("E" if moe else "-") + "*" + ("E" if moe else "-")
        config.num_layers = 4
        config.vocab_size = 256
        config.mtp_num_layers = 0
        config.rotary_base = 500000.0
        exported = bridge_class.megatron_to_hf_config(config)
        projected = carry_rope_into_rope_parameters(
            exported, conform_config_to_reference(exported, reference.to_dict())
        )
        restored = type(reference)(**_drop_readonly_config_properties(projected, type(reference)))
        assert restored.text_config.num_hidden_layers == 2
        assert restored.text_config.vocab_size == 256
        assert restored.text_config.mtp_num_hidden_layers == 0
        assert restored.text_config.rope_parameters["rope_theta"] == 500000.0
        assert restored.vision_config.to_dict() == reference.vision_config.to_dict()

    @pytest.mark.parametrize(("bridge_class", "moe"), _BRIDGE_CASES, ids=_BRIDGE_IDS)
    def test_hybrid_export_preserves_language_blocks_and_attention_order(self, bridge_class, moe):
        hf_config = _qwen35_vl_config(moe=moe)
        config = bridge_class().hf_config_to_model_config(hf_config)
        config.finalize()
        exported = bridge_class.megatron_to_hf_config(config)

        assert (
            exported["text_config"]["num_hidden_layers"] == hf_config.text_config.num_hidden_layers
        )
        assert exported["text_config"]["layer_types"] == hf_config.text_config.layer_types

    @pytest.mark.parametrize("bridge_class", [Qwen35VLBridge, Qwen35VLMoEBridge])
    def test_conversion_uses_builder_config(self, bridge_class):
        assert bridge_class.MODEL_CONFIG_CLASS is Qwen35VLModelConfig

    @pytest.mark.parametrize(("bridge_class", "moe"), _BRIDGE_CASES, ids=_BRIDGE_IDS)
    def test_hf_config_to_model_config_uses_direct_mapping(self, bridge_class, moe):
        hf_config = _qwen35_vl_config(moe=moe)
        bridge = bridge_class()
        model_config = bridge.hf_config_to_model_config(hf_config)

        assert isinstance(model_config, Qwen35VLModelConfig)
        assert isinstance(model_config.transformer, Qwen35VLTransformerConfig)
        assert model_config.transformer_layer_spec is None
        # Language model: the text bridge's builder kwargs, read from text_config.
        assert model_config.num_layers == 16
        assert model_config.experimental_attention_variant == "gdn"
        assert model_config.linear_attention_freq == [1, 0, 1, 0, 1, 0, 0, 0] * 2
        assert model_config.layernorm_zero_centered_gamma is True
        assert model_config.mtp_num_layers == 1
        assert model_config.mtp_loss_scaling_factor == 0.1
        assert model_config.num_moe_experts == (4 if moe else None)
        # Vision-language fields.
        assert model_config.position_embedding_type == "mrope"
        assert model_config.mrope_section == [1, 1, 0]
        assert model_config.vision_config is hf_config.vision_config
        assert model_config.vision_config.dtype == torch.bfloat16
        assert model_config.image_token_id == hf_config.image_token_id
        assert model_config.video_token_id == hf_config.video_token_id
        assert model_config.vision_start_token_id == hf_config.vision_start_token_id
        assert model_config.vision_end_token_id == hf_config.vision_end_token_id
        assert model_config.special_token_ids == {"images": hf_config.image_token_id}
        # Settings the VL models keep where the text models differ.
        assert model_config.attention_softmax_in_fp32 is True
        assert model_config.bias_activation_fusion is True
        assert model_config.autocast_dtype is None
        assert model_config.scatter_embedding_sequence_parallel is False

    @pytest.mark.parametrize(("bridge_class", "moe"), _BRIDGE_CASES, ids=_BRIDGE_IDS)
    @pytest.mark.parametrize("tie_word_embeddings", [False, True])
    def test_tied_embeddings_come_from_top_level_config(
        self, bridge_class, moe, tie_word_embeddings
    ):
        hf_config = _qwen35_vl_config(moe=moe, tie_word_embeddings=tie_word_embeddings)
        hf_config.text_config.tie_word_embeddings = not tie_word_embeddings

        model_config = bridge_class().hf_config_to_model_config(hf_config)

        assert model_config.share_embeddings_and_output_weights is tie_word_embeddings


def test_vl_mapping_splits_only_language_blocks():
    from transformers import Qwen3_5MoeConfig

    bridge = Qwen35VLMoEBridge()
    bridge.hf_config = Qwen3_5MoeConfig(text_config={"num_hidden_layers": 4})
    registry = bridge.mapping_registry()
    language = registry.megatron_to_hf_lookup("language_model.decoder.layers.7.mlp.router.weight")
    assert language.hf_param == "model.language_model.layers.3.mlp.gate.weight"
    vision = MegatronMappingRegistry(*_get_vision_mappings())
    for template in vision.mappings:
        name = template.megatron_param.replace("*", "0")
        assert (
            registry.megatron_to_hf_lookup(name).hf_param
            == vision.megatron_to_hf_lookup(name).hf_param
        )
