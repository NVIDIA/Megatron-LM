# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.quantization.quant_config import MatchContext, RecipeConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    HAVE_TE = True
    import transformer_engine as te
except ImportError:
    HAVE_TE = False

try:
    from megatron.core.extensions.kitchen import (
        KitchenColumnParallelGroupedLinear,
        KitchenColumnParallelLinear,
        KitchenLayerNormColumnParallelLinear,
        KitchenRowParallelGroupedLinear,
        KitchenRowParallelLinear,
    )

    HAVE_KITCHEN = True
except ImportError:
    HAVE_KITCHEN = False


@pytest.mark.skipif(not HAVE_KITCHEN, reason="Kitchen required for using kitchen backend.")
@pytest.mark.skipif(
    not HAVE_TE, reason="Transformer Engine required for using kitchen backend with TE layers."
)
class TestGPTModelKitchenQuantizationConfig:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_kitchen_config_resolution_dense(self) -> None:
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=False,
            gated_linear_unit=True,
            bias_activation_fusion=True,
            add_bias_linear=False,
            use_kitchen=True,
            quant_recipe=RecipeConfig.from_config_dict(
                {
                    "matchers": {
                        "keep_in_hp": {
                            "type": "glob",
                            "enabled": True,
                            "pattern": "*fc2",
                            "config": "bf16",
                        },
                        "use_fp8_cs": {
                            "type": "glob",
                            "enabled": True,
                            "pattern": "*",
                            "config": "fp8_cs",
                        },
                    },
                    "configs": {
                        "bf16": {"kitchen_config_type": "QLinearParams", "recipe_idx": 1},
                        "fp8_cs": {"kitchen_config_type": "QLinearParams", "recipe_idx": 2},
                    },
                }
            ),
        )
        transformer_layer_spec = get_gpt_decoder_block_spec(
            config=transformer_config, use_transformer_engine=True
        )
        padded_vocab_size = 512
        max_position_embeddings = 4096
        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=padded_vocab_size,
            max_sequence_length=max_position_embeddings,
        )

        expected_types = {
            "decoder.layers.0.self_attention.linear_proj": KitchenRowParallelLinear,
            "decoder.layers.1.self_attention.linear_proj": KitchenRowParallelLinear,
            "decoder.layers.0.self_attention.linear_qkv": KitchenLayerNormColumnParallelLinear,
            "decoder.layers.1.self_attention.linear_qkv": KitchenLayerNormColumnParallelLinear,
            "decoder.layers.0.mlp.linear_fc1": KitchenLayerNormColumnParallelLinear,
            "decoder.layers.1.mlp.linear_fc1": KitchenLayerNormColumnParallelLinear,
            "decoder.layers.0.mlp.linear_fc2": KitchenRowParallelLinear,
            "decoder.layers.1.mlp.linear_fc2": KitchenRowParallelLinear,
        }

        expected_match = {
            "decoder.layers.0.self_attention.linear_proj": (
                MatchContext("decoder.layers.0.self_attention.linear_proj", layer_number=0),
                "fp8_cs",
            ),
            "decoder.layers.1.self_attention.linear_proj": (
                MatchContext("decoder.layers.1.self_attention.linear_proj", layer_number=1),
                "fp8_cs",
            ),
            "decoder.layers.0.self_attention.linear_qkv": (
                MatchContext("decoder.layers.0.self_attention.linear_qkv", layer_number=0),
                "fp8_cs",
            ),
            "decoder.layers.1.self_attention.linear_qkv": (
                MatchContext("decoder.layers.1.self_attention.linear_qkv", layer_number=1),
                "fp8_cs",
            ),
            "decoder.layers.0.mlp.linear_fc1": (
                MatchContext("decoder.layers.0.mlp.linear_fc1", layer_number=0),
                "fp8_cs",
            ),
            "decoder.layers.1.mlp.linear_fc1": (
                MatchContext("decoder.layers.1.mlp.linear_fc1", layer_number=1),
                "fp8_cs",
            ),
            "decoder.layers.0.mlp.linear_fc2": (
                MatchContext("decoder.layers.0.mlp.linear_fc2", layer_number=0),
                "bf16",
            ),
            "decoder.layers.1.mlp.linear_fc2": (
                MatchContext("decoder.layers.1.mlp.linear_fc2", layer_number=1),
                "bf16",
            ),
        }

        visited_keys = set()
        for name, module in model.named_modules():
            if name in expected_types:
                assert (
                    type(module) == expected_types[name]
                ), f"Expected {name} to be {expected_types[name]}, but it is {type(module)}"
                visited_keys.add(name)
                assert hasattr(module, "kitchen_quant_params")
                assert module.kitchen_quant_params.params_config_key == expected_match[name][1]
                assert module.kitchen_quant_params.match_input == expected_match[name][0]
        assert visited_keys == set(expected_types.keys())

    def test_kitchen_config_resolution_moe(self) -> None:
        transformer_config = TransformerConfig(
            moe_layer_freq=1,
            num_moe_experts=2,
            moe_router_load_balancing_type="sinkhorn",
            moe_router_topk=1,
            moe_grouped_gemm=True,
            moe_use_legacy_grouped_gemm=False,
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=False,
            gated_linear_unit=True,
            bias_activation_fusion=True,
            add_bias_linear=False,
            use_kitchen=True,
            quant_recipe=RecipeConfig.from_config_dict(
                {
                    "matchers": {
                        "keep_in_hp": {
                            "type": "glob",
                            "enabled": True,
                            "pattern": "*fc2",
                            "config": "bf16",
                        },
                        "use_fp8_cs": {
                            "type": "glob",
                            "enabled": True,
                            "pattern": "*",
                            "config": "fp8_cs",
                        },
                    },
                    "configs": {
                        "bf16": {"kitchen_config_type": "QLinearParams", "recipe_idx": 1},
                        "fp8_cs": {"kitchen_config_type": "QLinearParams", "recipe_idx": 2},
                    },
                }
            ),
        )
        transformer_layer_spec = get_gpt_decoder_block_spec(
            config=transformer_config, use_transformer_engine=True
        )
        padded_vocab_size = 512
        max_position_embeddings = 4096
        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=padded_vocab_size,
            max_sequence_length=max_position_embeddings,
        )

        expected_types = {
            "decoder.layers.0.self_attention.linear_proj": KitchenRowParallelLinear,
            "decoder.layers.1.self_attention.linear_proj": KitchenRowParallelLinear,
            "decoder.layers.0.self_attention.linear_qkv": KitchenLayerNormColumnParallelLinear,
            "decoder.layers.1.self_attention.linear_qkv": KitchenLayerNormColumnParallelLinear,
            "decoder.layers.0.mlp.experts.linear_fc1": KitchenColumnParallelGroupedLinear,
            "decoder.layers.1.mlp.experts.linear_fc1": KitchenColumnParallelGroupedLinear,
            "decoder.layers.0.mlp.experts.linear_fc2": KitchenRowParallelGroupedLinear,
            "decoder.layers.1.mlp.experts.linear_fc2": KitchenRowParallelGroupedLinear,
        }

        expected_match = {
            "decoder.layers.0.self_attention.linear_proj": (
                MatchContext("decoder.layers.0.self_attention.linear_proj", layer_number=0),
                "fp8_cs",
            ),
            "decoder.layers.1.self_attention.linear_proj": (
                MatchContext("decoder.layers.1.self_attention.linear_proj", layer_number=1),
                "fp8_cs",
            ),
            "decoder.layers.0.self_attention.linear_qkv": (
                MatchContext("decoder.layers.0.self_attention.linear_qkv", layer_number=0),
                "fp8_cs",
            ),
            "decoder.layers.1.self_attention.linear_qkv": (
                MatchContext("decoder.layers.1.self_attention.linear_qkv", layer_number=1),
                "fp8_cs",
            ),
            "decoder.layers.0.mlp.experts.linear_fc1": (
                MatchContext("decoder.layers.0.mlp.experts.linear_fc1", layer_number=0),
                "fp8_cs",
            ),
            "decoder.layers.1.mlp.experts.linear_fc1": (
                MatchContext("decoder.layers.1.mlp.experts.linear_fc1", layer_number=1),
                "fp8_cs",
            ),
            "decoder.layers.0.mlp.experts.linear_fc2": (
                MatchContext("decoder.layers.0.mlp.experts.linear_fc2", layer_number=0),
                "bf16",
            ),
            "decoder.layers.1.mlp.experts.linear_fc2": (
                MatchContext("decoder.layers.1.mlp.experts.linear_fc2", layer_number=1),
                "bf16",
            ),
        }

        visited_keys = set()
        for name, module in model.named_modules():
            if name in expected_types:
                assert (
                    type(module) == expected_types[name]
                ), f"Expected {name} to be {expected_types[name]}, but it is {type(module)}"
                visited_keys.add(name)
                assert hasattr(module, "kitchen_quant_params")
                assert module.kitchen_quant_params.params_config_key == expected_match[name][1]
                assert module.kitchen_quant_params.match_input == expected_match[name][0]
        assert visited_keys == set(expected_types.keys())
