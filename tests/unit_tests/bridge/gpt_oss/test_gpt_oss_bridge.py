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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c209ed397fb36272c2216053058cda4e37802f2d/tests/unit_tests/models/gpt_oss/test_gpt_oss_bridges.py
# Source license: Apache-2.0.

"""Unit tests for the builder-backed GPT-OSS bridge and MXFP4 dequantization."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from safetensors.torch import save_file
from transformers import GptOssConfig, GptOssForCausalLM

from megatron.bridge.models import AutoBridge
from megatron.bridge.models.conversion.gtp import _gather_gtp_weight, _slice_gtp_weight
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping
from megatron.bridge.models.conversion.quantization_utils import dequantize_mxfp4
from megatron.bridge.models.conversion.transformers_compat import rope_theta_from_hf
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.gpt_oss.gpt_oss_bridge import (
    GPTOSSBridge,
    GPTOSSMLPDownProjMapping,
    GPTOSSMLPGateUpProjMapping,
)
from megatron.core.fusions.fused_bias_geglu import quick_gelu
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

_EXPECTED_YARN_ROPE_SCALING = {
    "rope_type": "yarn",
    "factor": 32.0,
    "original_max_position_embeddings": 4096,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "truncate": False,
}


def _gpt_oss_config(**overrides) -> GptOssConfig:
    values = dict(
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=128,
        sliding_window=32,
        torch_dtype="bfloat16",
    )
    values.update(overrides)
    config = GptOssConfig(**values)
    config.architectures = ["GptOssForCausalLM"]
    return config


def _bridge_with_hf_dims(*, hidden_size: int, intermediate_size: int) -> GPTOSSBridge:
    bridge = GPTOSSBridge()
    bridge.hf_pretrained = SimpleNamespace(
        config=SimpleNamespace(hidden_size=hidden_size, intermediate_size=intermediate_size)
    )
    return bridge


def _reference_dequantize_mxfp4(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decode E2M1 codes (sign bit + 3-bit magnitude, low nibble first) with E8M0 scales."""
    codes = torch.stack((blocks & 0x0F, blocks >> 4), dim=-1).flatten(-2).long()
    magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])[codes & 0x7]
    signs = torch.where(codes >= 8, -1.0, 1.0)
    values = signs * magnitudes * torch.exp2(scales.float() - 127).unsqueeze(-1)
    return values.flatten(-2).to(torch.bfloat16)


class TestGPTOSSBridgeModelConfig:
    """Hugging Face to builder config conversion for GPT-OSS."""

    def test_bridge_registration(self):
        assert issubclass(GPTOSSBridge, MegatronModelBridge)
        assert GPTOSSBridge.MODEL_CONFIG_CLASS is BridgeGPTModelConfig

    def test_hf_config_to_model_config_maps_gpt_oss_architecture(self):
        config = _gpt_oss_config()

        model_config = GPTOSSBridge().hf_config_to_model_config(config)

        assert isinstance(model_config, BridgeGPTModelConfig)
        assert model_config.num_layers == 2
        assert model_config.hidden_size == 64
        assert model_config.num_attention_heads == 4
        assert model_config.num_query_groups == 2
        assert model_config.kv_channels == 16
        assert model_config.vocab_size == 128
        assert model_config.rotary_base == rope_theta_from_hf(config)
        assert model_config.num_moe_experts == config.num_local_experts
        assert model_config.moe_router_topk == config.num_experts_per_tok
        assert model_config.moe_ffn_hidden_size == config.intermediate_size
        assert model_config.moe_grouped_gemm is True
        assert model_config.moe_router_load_balancing_type == "none"
        assert model_config.activation_func is quick_gelu
        assert model_config.activation_func_clamp_value == 7.0
        assert model_config.glu_linear_offset == 1.0
        assert model_config.softmax_type == "learnable"
        assert model_config.window_size == (config.sliding_window - 1, 0)
        assert model_config.window_attn_skip_freq == 2
        assert model_config.position_embedding_type == "yarn"
        assert model_config.yarn_rotary_scaling_factor == 32.0
        assert model_config.yarn_original_max_position_embeddings == 4096
        assert model_config.yarn_beta_fast == 32.0
        assert model_config.yarn_beta_slow == 1.0
        assert model_config.yarn_correction_range_round_to_int is False
        assert model_config.add_qkv_bias is True
        assert model_config.add_bias_linear is True
        assert model_config.share_embeddings_and_output_weights is False
        assert model_config.normalization == "RMSNorm"
        assert model_config.bf16 is True
        assert model_config.params_dtype == torch.bfloat16

    @pytest.mark.parametrize(
        "rope_scaling",
        [
            "missing",
            None,
            {},
            {
                "rope_type": "yarn",
                "factor": 32.0,
                "original_max_position_embeddings": 4096,
                "truncate": False,
            },
        ],
        ids=["missing", "none", "empty", "hf_yarn"],
    )
    def test_uses_yarn_regardless_of_hf_rope_scaling(self, rope_scaling):
        config = SimpleNamespace(
            architectures=["GptOssForCausalLM"],
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_local_experts=4,
            num_experts_per_tok=2,
            vocab_size=128,
            max_position_embeddings=4096,
            sliding_window=32,
            hidden_act="silu",
            attention_bias=True,
            torch_dtype="bfloat16",
        )
        if rope_scaling != "missing":
            config.rope_scaling = rope_scaling

        model_config = GPTOSSBridge().hf_config_to_model_config(config)

        assert model_config.position_embedding_type == "yarn"
        assert model_config.yarn_rotary_scaling_factor == 32.0
        assert model_config.yarn_original_max_position_embeddings == 4096


class TestGPTOSSBridgeMegatronToHFConfig:
    """Builder config to Hugging Face config export for GPT-OSS."""

    def test_megatron_to_hf_config_from_model_config(self):
        model_config = GPTOSSBridge().hf_config_to_model_config(_gpt_oss_config())

        hf_config = GPTOSSBridge.megatron_to_hf_config(model_config)

        assert hf_config["architectures"] == ["GptOssForCausalLM"]
        assert hf_config["model_type"] == "gpt_oss"
        assert hf_config["attention_bias"] is True
        assert hf_config["num_hidden_layers"] == 2
        assert hf_config["num_local_experts"] == 4
        assert hf_config["num_experts_per_tok"] == 2
        assert hf_config["rope_scaling"] == _EXPECTED_YARN_ROPE_SCALING
        assert hf_config["torch_dtype"] == "bfloat16"

    def test_exported_config_reloads_as_the_source_architecture(self):
        source = _gpt_oss_config()
        hf_config = GPTOSSBridge.megatron_to_hf_config(
            GPTOSSBridge().hf_config_to_model_config(source)
        )

        reloaded = GptOssConfig.from_dict(hf_config)

        for name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "num_local_experts",
            "num_experts_per_tok",
            "vocab_size",
            "max_position_embeddings",
            "rms_norm_eps",
            "tie_word_embeddings",
            "attention_bias",
        ):
            assert getattr(reloaded, name) == getattr(source, name), name
        assert rope_theta_from_hf(reloaded) == rope_theta_from_hf(source)


class TestGPTOSSBridgeMappingRegistry:
    """Weight name mappings for GPT-OSS."""

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
                "decoder.layers.1.self_attention.linear_proj.weight",
                "model.layers.1.self_attn.o_proj.weight",
            ),
            (
                "decoder.layers.1.self_attention.linear_proj.bias",
                "model.layers.1.self_attn.o_proj.bias",
            ),
            (
                "decoder.layers.0.self_attention.core_attention.softmax_offset",
                "model.layers.0.self_attn.sinks",
            ),
            (
                "decoder.layers.1.pre_mlp_layernorm.weight",
                "model.layers.1.post_attention_layernorm.weight",
            ),
            ("decoder.layers.0.mlp.router.weight", "model.layers.0.mlp.router.weight"),
            ("decoder.layers.0.mlp.router.bias", "model.layers.0.mlp.router.bias"),
        ],
    )
    def test_one_to_one_mappings(self, megatron_param, hf_param):
        mapping = GPTOSSBridge().mapping_registry().megatron_to_hf_lookup(megatron_param)

        assert mapping is not None
        assert mapping.hf_param == hf_param

    @pytest.mark.parametrize("suffix", ["weight", "bias"])
    def test_qkv_mappings(self, suffix):
        mapping = (
            GPTOSSBridge()
            .mapping_registry()
            .megatron_to_hf_lookup(f"decoder.layers.3.self_attention.linear_qkv.{suffix}")
        )

        assert mapping.hf_param == {
            "q": f"model.layers.3.self_attn.q_proj.{suffix}",
            "k": f"model.layers.3.self_attn.k_proj.{suffix}",
            "v": f"model.layers.3.self_attn.v_proj.{suffix}",
        }

    @pytest.mark.parametrize(
        ("megatron_param", "mapping_class", "hf_param"),
        [
            (
                "decoder.layers.1.mlp.experts.linear_fc1.weight3",
                GPTOSSMLPGateUpProjMapping,
                "model.layers.1.mlp.experts.gate_up_proj",
            ),
            (
                "decoder.layers.1.mlp.experts.linear_fc1.bias3",
                GPTOSSMLPGateUpProjMapping,
                "model.layers.1.mlp.experts.gate_up_proj_bias",
            ),
            (
                "decoder.layers.1.mlp.experts.linear_fc2.weight3",
                GPTOSSMLPDownProjMapping,
                "model.layers.1.mlp.experts.down_proj",
            ),
            (
                "decoder.layers.1.mlp.experts.linear_fc2.bias3",
                GPTOSSMLPDownProjMapping,
                "model.layers.1.mlp.experts.down_proj_bias",
            ),
            (
                "decoder.layers.1.mlp.experts.local_experts.3.linear_fc1.weight",
                GPTOSSMLPGateUpProjMapping,
                "model.layers.1.mlp.experts.gate_up_proj",
            ),
            (
                "decoder.layers.1.mlp.experts.local_experts.3.linear_fc2.bias",
                GPTOSSMLPDownProjMapping,
                "model.layers.1.mlp.experts.down_proj_bias",
            ),
        ],
    )
    def test_expert_mappings_cover_grouped_and_sequential_layouts(
        self, megatron_param, mapping_class, hf_param
    ):
        mapping = GPTOSSBridge().mapping_registry().megatron_to_hf_lookup(megatron_param)

        assert isinstance(mapping, mapping_class)
        assert mapping.hf_param == hf_param
        assert mapping.is_grouped_export is True
        assert mapping.allow_hf_name_mismatch is True


class TestGPTOSSLoadedWeights:
    """Import-time orientation fixes and MXFP4 dequantization."""

    @pytest.mark.parametrize(
        ("hidden_size", "intermediate_size"), [(8, 4), (4, 4)], ids=["rectangular", "square"]
    )
    def test_bf16_down_proj_is_transposed_to_megatron_layout(self, hidden_size, intermediate_size):
        bridge = _bridge_with_hf_dims(hidden_size=hidden_size, intermediate_size=intermediate_size)
        down_proj = torch.randn(3, intermediate_size, hidden_size)
        name = "model.layers.0.mlp.experts.down_proj"

        loaded = bridge.maybe_modify_loaded_hf_weight(name, {name: down_proj})

        assert torch.equal(loaded, down_proj.transpose(-1, -2))

    def test_down_proj_already_in_megatron_layout_is_kept(self):
        bridge = _bridge_with_hf_dims(hidden_size=8, intermediate_size=4)
        down_proj = torch.randn(3, 8, 4)
        name = "model.layers.0.mlp.experts.down_proj"

        loaded = bridge.maybe_modify_loaded_hf_weight(name, {name: down_proj})

        assert loaded is down_proj

    @pytest.mark.parametrize(
        ("hidden_size", "intermediate_size"), [(8, 3), (8, 4)], ids=["rectangular", "square"]
    )
    def test_bf16_gate_up_proj_is_transposed_to_megatron_layout(
        self, hidden_size, intermediate_size
    ):
        """With hidden == 2 * intermediate the shape cannot tell the layouts apart."""
        bridge = _bridge_with_hf_dims(hidden_size=hidden_size, intermediate_size=intermediate_size)
        gate_up_proj = torch.randn(3, hidden_size, 2 * intermediate_size)
        name = "model.layers.0.mlp.experts.gate_up_proj"

        loaded = bridge.maybe_modify_loaded_hf_weight(name, {name: gate_up_proj})

        assert torch.equal(loaded, gate_up_proj.transpose(-1, -2))

    def test_gate_up_proj_already_in_megatron_layout_is_kept(self):
        bridge = _bridge_with_hf_dims(hidden_size=8, intermediate_size=3)
        gate_up_proj = torch.randn(3, 6, 8)
        name = "model.layers.0.mlp.experts.gate_up_proj"

        assert bridge.maybe_modify_loaded_hf_weight(name, {name: gate_up_proj}) is gate_up_proj

    def test_other_weights_are_returned_unchanged(self):
        bridge = _bridge_with_hf_dims(hidden_size=8, intermediate_size=4)
        router = torch.randn(4, 8)
        name = "model.layers.0.mlp.router.weight"

        assert bridge.maybe_modify_loaded_hf_weight(name, {name: router}) is router

    @pytest.mark.parametrize("name", ["down_proj", "gate_up_proj"])
    def test_mxfp4_experts_are_dequantized_without_transpose(self, name):
        bridge = _bridge_with_hf_dims(hidden_size=64, intermediate_size=64)
        generator = torch.Generator().manual_seed(0)
        blocks = torch.randint(0, 256, (2, 4, 2, 16), dtype=torch.uint8, generator=generator)
        scales = torch.randint(120, 134, (2, 4, 2), dtype=torch.uint8, generator=generator)
        prefix = f"model.layers.0.mlp.experts.{name}"

        loaded = bridge.maybe_modify_loaded_hf_weight(
            prefix, {f"{prefix}_blocks": blocks, f"{prefix}_scales": scales}
        )

        assert torch.equal(loaded, _reference_dequantize_mxfp4(blocks, scales))

    def test_missing_expert_weights_raise(self):
        bridge = _bridge_with_hf_dims(hidden_size=8, intermediate_size=4)

        with pytest.raises(KeyError, match="Cannot locate weights"):
            bridge.maybe_modify_loaded_hf_weight("model.layers.0.mlp.experts.down_proj", {})

    def test_multi_tensor_params_are_looked_up_by_name(self):
        bridge = GPTOSSBridge()
        state = {"q": torch.zeros(1), "k": torch.ones(1)}

        loaded = bridge.maybe_modify_loaded_hf_weight({"query": "q", "key": "k"}, state)

        assert loaded == {"query": state["q"], "key": state["k"]}


class TestDequantizeMXFP4:
    """OpenAI MXFP4 block decoding."""

    def test_decodes_nibbles_low_first_with_biased_exponents(self):
        blocks = torch.tensor([[[0x21, 0xF7]], [[0x80, 0x9A]]], dtype=torch.uint8)
        scales = torch.tensor([[128], [126]], dtype=torch.uint8)

        values = dequantize_mxfp4(blocks, scales)

        expected = torch.tensor(
            [[1.0, 2.0, 12.0, -12.0], [0.0, -0.0, -0.5, -0.25]], dtype=torch.bfloat16
        )
        assert values.dtype == torch.bfloat16
        assert torch.equal(values, expected)

    def test_matches_reference_decoding_and_chunking(self):
        generator = torch.Generator().manual_seed(1)
        blocks = torch.randint(0, 256, (2, 3, 4, 16), dtype=torch.uint8, generator=generator)
        scales = torch.randint(100, 150, (2, 3, 4), dtype=torch.uint8, generator=generator)

        values = dequantize_mxfp4(blocks, scales)

        assert values.shape == (2, 3, 4 * 16 * 2)
        assert torch.equal(values, _reference_dequantize_mxfp4(blocks, scales))
        assert torch.equal(dequantize_mxfp4(blocks, scales, rows_per_chunk=5), values)

    def test_rejects_mismatched_scales(self):
        blocks = torch.zeros(2, 3, 16, dtype=torch.uint8)
        scales = torch.zeros(2, 4, dtype=torch.uint8)

        with pytest.raises(AssertionError, match="does not match"):
            dequantize_mxfp4(blocks, scales)


class TestGPTOSSExpertMappings:
    """Per-expert slicing and HF layout restoration."""

    def test_down_proj_import_selects_the_global_expert(self, monkeypatch):
        monkeypatch.setattr(
            AutoMapping, "hf_to_megatron", lambda self, hf_weights, megatron_module: hf_weights
        )
        mapping = GPTOSSMLPDownProjMapping(
            "decoder.layers.0.mlp.experts.linear_fc2.weight2",
            "model.layers.0.mlp.experts.down_proj",
        )
        experts = torch.arange(24.0).reshape(4, 3, 2)

        assert torch.equal(mapping.hf_to_megatron(experts, None), experts[2])

    @pytest.mark.parametrize("shape", [(3, 2), (4, 3, 2)], ids=["one_expert", "ep_gathered"])
    def test_down_proj_export_transposes_weights_but_not_biases(self, monkeypatch, shape):
        monkeypatch.setattr(
            AutoMapping,
            "megatron_to_hf",
            lambda self, megatron_weights, megatron_module: {self.hf_param: megatron_weights},
        )
        weight_name = "model.layers.0.mlp.experts.down_proj"
        weight_mapping = GPTOSSMLPDownProjMapping(
            "decoder.layers.0.mlp.experts.linear_fc2.weight1", weight_name
        )
        bias_name = "model.layers.0.mlp.experts.down_proj_bias"
        bias_mapping = GPTOSSMLPDownProjMapping(
            "decoder.layers.0.mlp.experts.linear_fc2.bias1", bias_name
        )
        weight = torch.arange(float(torch.Size(shape).numel())).reshape(shape)
        bias = torch.arange(float(shape[-2]))

        assert torch.equal(
            weight_mapping.megatron_to_hf(weight, None)[weight_name], weight.transpose(-1, -2)
        )
        assert torch.equal(bias_mapping.megatron_to_hf(bias, None)[bias_name], bias)

    @pytest.mark.parametrize("hidden_size", [3, 8], ids=["rectangular", "square"])
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.parametrize("etp_size", [1, 2])
    @pytest.mark.parametrize("gtp_size", [1, 2])
    def test_gate_up_mapping_preserves_expert_rows_across_etp_and_gtp(
        self, monkeypatch, bias, etp_size, gtp_size, hidden_size
    ):
        """Distinct gate/up rows detect a transpose or a split of the fused TP matrix.

        The weights load through the bridge's import hook, as from a checkpoint. With hidden size 8
        the BF16 weight (hidden, 2 * intermediate) is square, so its shape cannot reveal the layout.
        """
        gate = (
            torch.arange(4 if bias else 4 * hidden_size)
            .reshape((4,) if bias else (4, hidden_size))
            .float()
            + 10
        )
        up = gate + 100
        interleaved = torch.empty((8,) if bias else (8, hidden_size))
        interleaved[::2], interleaved[1::2] = gate, up
        hf_expert = interleaved if bias else interleaved.t()
        hf_name = "model.layers.0.mlp.experts.gate_up_proj" + ("_bias" if bias else "")
        # Import expert 1, so extracting the wrong expert is also observable.
        bridge = _bridge_with_hf_dims(hidden_size=hidden_size, intermediate_size=4)
        hf_weights = bridge.maybe_modify_loaded_hf_weight(
            hf_name, {hf_name: torch.stack((hf_expert + 1000, hf_expert))}
        )
        param_name = "bias0" if bias else "weight0"
        megatron_name = "decoder.layers.0.mlp.experts.linear_fc1." + (
            "bias1" if bias else "weight1"
        )
        local_rows = 8 // etp_size
        mapping_path = "megatron.bridge.models.conversion.param_mapping"
        monkeypatch.setattr(
            f"{mapping_path}.get_pg_size", lambda group: group.size() if group is not None else 1
        )
        monkeypatch.setattr(
            f"{mapping_path}.get_pg_rank", lambda group: group.rank() if group is not None else 0
        )
        expected_tp = [torch.cat((g, u)) for g, u in zip(gate.chunk(etp_size), up.chunk(etp_size))]

        for etp_rank in range(etp_size):
            module = torch.nn.Module()
            padding = 2 if gtp_size > 1 else 0
            shape = ((local_rows + padding) // gtp_size, *gate.shape[1:])
            parameter = torch.nn.Parameter(torch.zeros(shape))
            if gtp_size > 1:
                parameter.is_gtp_weight_remat = True
                parameter.pad_length = padding
                parameter.group = SimpleNamespace(size=lambda: gtp_size, rank=lambda: 1)
            module.register_parameter(param_name, parameter)
            mapping = GPTOSSMLPGateUpProjMapping(megatron_name, hf_name)
            group = SimpleNamespace(size=lambda: etp_size, rank=lambda: etp_rank)
            mapping.set_process_groups_from_pg_collection(SimpleNamespace(expt_tp=group))
            delegate = mapping._gated_mapping

            def scatter(splits, output_shape, dtype, device):
                assert output_shape == expected_tp[etp_rank].shape
                if splits is not None:
                    assert all(
                        torch.equal(actual, expected)
                        for actual, expected in zip(splits, expected_tp)
                    )
                return expected_tp[etp_rank].clone()

            monkeypatch.setattr(delegate, "scatter_to_tp_ranks", scatter)
            converted = mapping.hf_to_megatron(hf_weights, module)
            assert torch.equal(converted, expected_tp[etp_rank])
            local = _slice_gtp_weight(converted, parameter)
            if gtp_size > 1:
                padded = torch.cat((expected_tp[etp_rank], torch.zeros((padding, *gate.shape[1:]))))
                assert torch.equal(local, padded[shape[0] :])
                with torch.no_grad():
                    parameter.copy_(local)
                monkeypatch.setattr(
                    torch.distributed,
                    "all_gather_into_tensor",
                    lambda output, local, **kwargs: output.copy_(padded),
                )
                converted = _gather_gtp_weight(parameter)

            monkeypatch.setattr(delegate, "broadcast_from_pp_rank", lambda weight, **kwargs: weight)
            monkeypatch.setattr(delegate, "gather_from_tp_ranks", lambda weight: expected_tp)
            exported = mapping.megatron_to_hf(converted, module)
            assert torch.equal(exported[hf_name], hf_expert)


class TestGPTOSSAutoBridge:
    """AutoBridge selection and checkpoint import for GPT-OSS."""

    def test_supports_gpt_oss_causal_lm(self):
        assert AutoBridge.supports(_gpt_oss_config()) is True

    def test_get_model_config_from_hf_config(self):
        model_config = AutoBridge.from_hf_config(_gpt_oss_config()).get_model_config()

        assert isinstance(model_config, BridgeGPTModelConfig)
        assert model_config.softmax_type == "learnable"
        assert model_config.position_embedding_type == "yarn"
        assert model_config.num_moe_experts == 4

    def test_import_ckpt_builds_through_model_builder(self, tmp_path):
        """Checkpoint import builds GPT-OSS through get_model() with its attention and MoE settings."""
        hf_path = tmp_path / "hf"
        _gpt_oss_config().save_pretrained(hf_path)
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
        assert model_config.softmax_type == "learnable"
        assert model_config.window_size == (31, 0)
        assert model_config.position_embedding_type == "yarn"
        assert model_config.num_moe_experts == 4
        assert mock_save_megatron_model.call_args.args[0] is megatron_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Builds a Megatron model on GPU.")
class TestGPTOSSWeightRoundTrip:
    """HF -> Megatron -> HF weight round trips through Megatron-LM's GPT builder."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "intermediate_size",
        [64, 96, 32],
        ids=["square_down_proj", "rectangular_down_proj", "square_gate_up"],
    )
    def test_bf16_checkpoint_round_trips_exactly(self, tmp_path_dist_ckpt, intermediate_size):
        config = _gpt_oss_config(intermediate_size=intermediate_size)
        torch.manual_seed(1234)
        hf_model = GptOssForCausalLM(config).to(torch.bfloat16)
        expected = {name: tensor.detach().cpu() for name, tensor in hf_model.state_dict().items()}

        # Transformers saves only on rank 0 once torch.distributed is initialized, so every rank
        # reads the checkpoint from one shared directory.
        with TempNamedDir(tmp_path_dist_ckpt / "gpt_oss_round_trip") as hf_path:
            if torch.distributed.get_rank() == 0:
                hf_model.save_pretrained(hf_path)
            torch.distributed.barrier()
            bridge = AutoBridge.from_hf_pretrained(hf_path)
            models = bridge.get_model(wrap_with_ddp=False)
            exported = dict(bridge.export_hf_weights(models, cpu=True, show_progress=False))

        assert models[0].model_config.softmax_type == "learnable"
        assert set(exported) == set(expected)
        for name, tensor in exported.items():
            assert tensor.dtype == expected[name].dtype, name
            torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0, msg=name)

    def test_mxfp4_checkpoint_imports_dequantized_experts(self, tmp_path):
        """Experts stored as OpenAI MXFP4 blocks import dequantized and export in the BF16 layout."""
        config = _gpt_oss_config()
        torch.manual_seed(1234)
        hf_model = GptOssForCausalLM(config).to(torch.bfloat16)
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in hf_model.state_dict().items()
        }
        expected = dict(state)
        generator = torch.Generator().manual_seed(4321)
        experts, hidden, intermediate = (
            config.num_local_experts,
            config.hidden_size,
            config.intermediate_size,
        )
        for layer in range(config.num_hidden_layers):
            for name, rows, columns in (
                ("gate_up_proj", 2 * intermediate, hidden),
                ("down_proj", hidden, intermediate),
            ):
                key = f"model.layers.{layer}.mlp.experts.{name}"
                blocks = torch.randint(
                    0,
                    256,
                    (experts, rows, columns // 32, 16),
                    dtype=torch.uint8,
                    generator=generator,
                )
                scales = torch.randint(
                    122, 128, (experts, rows, columns // 32), dtype=torch.uint8, generator=generator
                )
                del state[key]
                state[f"{key}_blocks"] = blocks
                state[f"{key}_scales"] = scales
                # MXFP4 stores [experts, rows, columns]; BF16 checkpoints store the transpose.
                expected[key] = _reference_dequantize_mxfp4(blocks, scales).transpose(-1, -2)
        config.quantization_config = {
            "quant_method": "mxfp4",
            "modules_to_not_convert": [
                "model.layers.*.self_attn",
                "model.layers.*.mlp.router",
                "model.embed_tokens",
                "lm_head",
            ],
        }
        config.save_pretrained(tmp_path)
        save_file(state, tmp_path / "model.safetensors")

        bridge = AutoBridge.from_hf_pretrained(tmp_path)
        models = bridge.get_model(wrap_with_ddp=False)
        exported = dict(bridge.export_hf_weights(models, cpu=True, show_progress=False))

        assert set(exported) == set(expected)
        for name, tensor in exported.items():
            assert tensor.dtype == torch.bfloat16, name
            torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0, msg=name)
