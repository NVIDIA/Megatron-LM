# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.quantization.quant_config import MatchContext, RecipeConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    from megatron.core.extensions.kitchen import (
        HAVE_KITCHEN,
        KitchenColumnParallelGroupedLinear,
        KitchenColumnParallelLinear,
        KitchenDotProductAttention,
        KitchenFlashAttention,
        KitchenLayerNormColumnParallelLinear,
        KitchenRowParallelGroupedLinear,
        KitchenRowParallelLinear,
    )
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

    def test_kitchen_config_resolution_dense_compound_params(self) -> None:
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=False,
            gated_linear_unit=True,
            bias_activation_fusion=True,
            add_bias_linear=False,
            use_kitchen=True,
            use_kitchen_attention=True,
            quant_recipe=RecipeConfig.from_config_dict(
                {
                    "matchers": {
                        "keep_in_fp8": {
                            "type": "glob",
                            "enabled": True,
                            "pattern": "*fc2",
                            "config": "fp8_cs",
                        },
                        "all": {"type": "glob", "enabled": True, "pattern": "*", "config": "bf16"},
                    },
                    "configs": {
                        "bf16": {
                            "kitchen_config_type": "CompoundParams",
                            "configs": [
                                {"kitchen_config_type": "QLinearParams", "recipe_idx": 1},
                                {"kitchen_config_type": "QAttentionParams", "recipe_idx": 1},
                            ],
                        },
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
            "decoder.layers.0.self_attention.core_attention": KitchenDotProductAttention,
            "decoder.layers.1.self_attention.core_attention": KitchenDotProductAttention,
        }

        expected_match = {
            "decoder.layers.0.self_attention.linear_proj": (
                MatchContext("decoder.layers.0.self_attention.linear_proj", layer_number=0),
                "bf16",
            ),
            "decoder.layers.1.self_attention.linear_proj": (
                MatchContext("decoder.layers.1.self_attention.linear_proj", layer_number=1),
                "bf16",
            ),
            "decoder.layers.0.self_attention.linear_qkv": (
                MatchContext("decoder.layers.0.self_attention.linear_qkv", layer_number=0),
                "bf16",
            ),
            "decoder.layers.1.self_attention.linear_qkv": (
                MatchContext("decoder.layers.1.self_attention.linear_qkv", layer_number=1),
                "bf16",
            ),
            "decoder.layers.0.mlp.linear_fc1": (
                MatchContext("decoder.layers.0.mlp.linear_fc1", layer_number=0),
                "bf16",
            ),
            "decoder.layers.1.mlp.linear_fc1": (
                MatchContext("decoder.layers.1.mlp.linear_fc1", layer_number=1),
                "bf16",
            ),
            "decoder.layers.0.mlp.linear_fc2": (
                MatchContext("decoder.layers.0.mlp.linear_fc2", layer_number=0),
                "fp8_cs",
            ),
            "decoder.layers.1.mlp.linear_fc2": (
                MatchContext("decoder.layers.1.mlp.linear_fc2", layer_number=1),
                "fp8_cs",
            ),
            "decoder.layers.0.self_attention.core_attention": (
                MatchContext("decoder.layers.0.self_attention.core_attention", layer_number=0),
                "bf16",
            ),
            "decoder.layers.1.self_attention.core_attention": (
                MatchContext("decoder.layers.1.self_attention.core_attention", layer_number=1),
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

    def test_kitchen_flash_attention_config_resolution(self) -> None:
        """Test GPT model with KitchenFlashAttention configuration."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=False,
            gated_linear_unit=True,
            bias_activation_fusion=True,
            add_bias_linear=False,
            use_kitchen=True,
            use_kitchen_attention=True,
            kitchen_attention_backend="fa",
            attention_dropout=0.0,
            quant_recipe=RecipeConfig.from_config_dict(
                {
                    "matchers": {
                        "attention": {
                            "type": "glob",
                            "enabled": True,
                            "pattern": "*self_attention.core_attention",
                            "config": "fa_bf16",
                        },
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
                        "fa_bf16": {
                            "kitchen_config_type": "QFlashAttentionParams",
                            "recipe_name": "triton_fa_bf16_for_all_base_2",
                        },
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
            "decoder.layers.0.self_attention.core_attention": KitchenFlashAttention,
            "decoder.layers.1.self_attention.core_attention": KitchenFlashAttention,
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
            "decoder.layers.0.self_attention.core_attention": (
                MatchContext("decoder.layers.0.self_attention.core_attention", layer_number=0),
                "fa_bf16",
            ),
            "decoder.layers.1.self_attention.core_attention": (
                MatchContext("decoder.layers.1.self_attention.core_attention", layer_number=1),
                "fa_bf16",
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

    def test_kitchen_flash_attention_with_compound_params(self) -> None:
        """Test GPT model with KitchenFlashAttention using CompoundParams configuration."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=False,
            gated_linear_unit=True,
            bias_activation_fusion=True,
            add_bias_linear=False,
            use_kitchen=True,
            use_kitchen_attention=True,
            kitchen_attention_backend="fa",
            attention_dropout=0.0,
            quant_recipe=RecipeConfig.from_config_dict(
                {
                    "matchers": {
                        "all": {"type": "glob", "enabled": True, "pattern": "*", "config": "mixed"}
                    },
                    "configs": {
                        "mixed": {
                            "kitchen_config_type": "CompoundParams",
                            "configs": [
                                {"kitchen_config_type": "QLinearParams", "recipe_idx": 2},
                                {
                                    "kitchen_config_type": "QFlashAttentionParams",
                                    "recipe_name": "triton_fa_bf16_for_all_natural",
                                },
                            ],
                        }
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
            "decoder.layers.0.self_attention.core_attention": KitchenFlashAttention,
            "decoder.layers.1.self_attention.core_attention": KitchenFlashAttention,
        }

        expected_config_key = "mixed"

        visited_keys = set()
        for name, module in model.named_modules():
            if name in expected_types:
                assert (
                    type(module) == expected_types[name]
                ), f"Expected {name} to be {expected_types[name]}, but it is {type(module)}"
                visited_keys.add(name)
                assert hasattr(module, "kitchen_quant_params")
                assert module.kitchen_quant_params.params_config_key == expected_config_key
        assert visited_keys == set(expected_types.keys())


@pytest.mark.skipif(
    not HAVE_TE, reason="Transformer Engine required for using TE backend with per-module quant."
)
class TestGPTModelTEQuantizationConfig:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.parametrize(
        "experts_path",
        ("decoder.layers.0.mlp.experts", "mtp.layers.0.mtp_model_layer.mlp.experts"),
        ids=("decoder", "mtp"),
    )
    @pytest.mark.parametrize("late_override", (False, True), ids=("constructor", "late"))
    def test_bf16_override_executes_basic_ops_under_mxfp8(
        self, monkeypatch, record_property, experts_path, late_override
    ) -> None:
        """Full GPT construction must apply exact-path overrides to real expert ops."""
        import inspect

        from transformer_engine.common.recipe import MXFP8BlockScaling
        from transformer_engine.pytorch import fp8_autocast
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

        try:
            from transformer_engine.pytorch.ops import GroupedLinear
            from transformer_engine.pytorch.ops.basic.grouped_linear import (
                is_op_fuser_grouped_tensor_path_supported,
            )
        except ImportError:
            pytest.skip("TE grouped-tensor operation fuser support is required")
        if "single_grouped_weight" not in inspect.signature(GroupedLinear.__init__).parameters:
            pytest.skip("TE operation fuser requires single_grouped_weight support")
        available, reason = FP8GlobalStateManager.is_mxfp8_available()
        if not available:
            pytest.skip(reason)
        if not is_op_fuser_grouped_tensor_path_supported(None, torch.bfloat16):
            pytest.skip("Native BF16 grouped GEMM requires a supported GPU and cuBLASLt version")

        monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
        linear_paths = [f"{experts_path}.linear_fc{idx}" for idx in (1, 2)]
        precision_recipe = RecipeConfig.from_config_dict(
            {
                "matchers": {
                    path: {"type": "glob", "enabled": True, "pattern": path, "config": "bf16"}
                    for path in linear_paths
                },
                "configs": {
                    "bf16": {
                        "transformer_engine_config_type": "TEQuantizationParams",
                        "training_recipe": {},
                    }
                },
            }
        )
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            ffn_hidden_size=128,
            moe_ffn_hidden_size=128,
            num_moe_experts=2,
            mtp_num_layers=1,
            use_cpu_initialization=False,
            add_bias_linear=False,
            gated_linear_unit=True,
            activation_func=F.silu,
            bias_activation_fusion=False,
            gradient_accumulation_fusion=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            fp8="e4m3",
            fp8_recipe=Fp8Recipe.mxfp8,
            fp8_param=False,
            moe_grouped_gemm=True,
            use_transformer_engine_op_fuser=True,
            quant_recipe=None if late_override else precision_recipe,
        )
        decoder_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True)
        model = GPTModel(
            config=config,
            transformer_layer_spec=decoder_spec,
            mtp_block_spec=get_gpt_mtp_block_spec(
                config, decoder_spec, use_transformer_engine=True
            ),
            vocab_size=512,
            max_sequence_length=64,
        ).cuda()
        modules = dict(model.named_modules())
        experts = modules[experts_path]
        if late_override:
            from megatron.core.quantization.utils import get_quant_config_or_none

            # Exercise the post-construction finish_init contract independently of
            # GPTModel's current constructor-name propagation.
            for path in linear_paths:
                assert modules[path].te_quant_params is None
                modules[path].finish_init(get_quant_config_or_none(path, precision_recipe))
        for path in linear_paths:
            assert modules[path].te_quant_params is not None
            assert not modules[path].will_execute_quantized(True)
        assert experts._with_fused_impl, "BF16 must remain on the TE ops path"
        original_params = dict(experts.named_parameters())
        # Default GPT initialization yields tiny outputs for which a fixed absolute
        # tolerance can hide accidental FP8 computation. Keep GEMM scales near one.
        torch.manual_seed(456)
        with torch.no_grad():
            for weight in original_params.values():
                weight.normal_(std=weight.shape[-1] ** -0.5)

        # Spy on real basic-op execution. A fused quantized kernel bypasses these calls,
        # while a lost override makes the recorded context True even without joint fusion.
        basic_op_contexts = []
        real_fuser_forward = GroupedLinear.fuser_forward

        def record_basic_op_context(op, *args, **kwargs):
            basic_op_contexts.append(FP8GlobalStateManager.is_fp8_enabled())
            return real_fuser_forward(op, *args, **kwargs)

        monkeypatch.setattr(GroupedLinear, "fuser_forward", record_basic_op_context)
        split_sizes = (256, 512)
        tokens_per_expert = torch.tensor(split_sizes, dtype=torch.int64, device="cuda")
        num_tokens = sum(split_sizes)
        inputs = torch.randn(
            num_tokens, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        probs = torch.rand(num_tokens, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_()
        reference_probs = probs.detach().clone().requires_grad_()
        reference_weights = {
            name: weight.detach().clone().requires_grad_()
            for name, weight in original_params.items()
        }

        # Independent BF16 reference with the same per-expert weights and router scales.
        reference_outputs = []
        offset = 0
        for expert_idx, count in enumerate(split_sizes):
            rows = slice(offset, offset + count)
            offset += count
            projected = F.linear(
                reference_inputs[rows], reference_weights[f"linear_fc1.weight{expert_idx}"]
            )
            gate, value = projected.float().chunk(2, dim=-1)
            activated = (F.silu(gate) * value * reference_probs[rows, None].float()).to(
                torch.bfloat16
            )
            reference_outputs.append(
                F.linear(activated, reference_weights[f"linear_fc2.weight{expert_idx}"])
            )
        reference_output = torch.cat(reference_outputs)

        with fp8_autocast(enabled=True, fp8_recipe=MXFP8BlockScaling()):
            output, bias = experts(inputs, tokens_per_expert, probs)
            assert FP8GlobalStateManager.is_fp8_enabled(), "The override leaked into its caller"
        assert bias is None

        def assert_bf16_close(actual, expected, label):
            actual, expected = actual.detach(), expected.detach()
            assert torch.isfinite(actual).all()
            reference_rms = expected.float().square().mean().sqrt().clamp_min(1e-12)
            relative_rms = (
                actual.float() - expected.float()
            ).square().mean().sqrt() / reference_rms
            record_property(f"{label}_relative_rms", relative_rms.item())
            assert (
                relative_rms < 1e-2
            ), f"{label}: BF16 relative RMS error {relative_rms.item():.6f}"
            torch.testing.assert_close(
                actual, expected, rtol=2e-2, atol=5e-3 * reference_rms.item()
            )

        assert_bf16_close(output, reference_output, "output")

        # Run backward outside autocast so its precision must come from the saved forward.
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        reference_output.backward(grad_output)
        assert_bf16_close(inputs.grad, reference_inputs.grad, "input_gradient")
        assert_bf16_close(probs.grad, reference_probs.grad, "probability_gradient")
        for name, weight in experts.named_parameters():
            assert weight is original_params[name]
            assert weight.grad is not None
            assert_bf16_close(weight.grad, reference_weights[name].grad, name)
        # Keep this after numerical checks: a lost override must fail numerically too.
        assert basic_op_contexts == [False, False]

    @pytest.mark.parametrize(
        ("transformer_impl", "recipe_storage"),
        [
            ("transformer_engine", {}),
            ("transformer_engine", {"inherit_model_init_context": True}),
            ("inference_optimized", {}),
        ],
    )
    def test_selective_mxfp8_parameter_storage_at_construction(
        self, transformer_impl, recipe_storage
    ):
        """Recipe names must reach constructors, before checkpoint values are quantized."""
        if torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("MXFP8 parameter initialization requires Blackwell or newer")
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

        config = TransformerConfig(
            num_layers=8,
            hidden_size=128,
            num_attention_heads=4,
            ffn_hidden_size=256,
            normalization="RMSNorm",
            num_moe_experts=2,
            moe_grouped_gemm=True,
            gated_linear_unit=True,
            activation_func=torch.nn.functional.silu,
            moe_router_dtype="fp32",
            add_bias_linear=False,
            gradient_accumulation_fusion=False,
            params_dtype=torch.bfloat16,
            bf16=True,
            fp8="e4m3",
            fp8_recipe=Fp8Recipe.mxfp8,
            fp8_param=True,
            first_last_layers_bf16=True,
            num_layers_at_start_in_bf16=2,
            num_layers_at_end_in_bf16=4,
            transformer_impl=transformer_impl,
            inference_grouped_gemm_backend="torch",
            quant_recipe=RecipeConfig.from_config_dict(
                {
                    "configs": {
                        "bf16": {
                            "transformer_engine_config_type": "TEQuantizationParams",
                            "training_recipe": {"override_quantized_autocast": True},
                        },
                        "mxfp8": {
                            "transformer_engine_config_type": "TEQuantizationParams",
                            "training_recipe": {
                                "fp8_quantization_recipe": "mxfp8",
                                **recipe_storage,
                                "override_quantized_autocast": True,
                            },
                        },
                    },
                    "matchers": {
                        "routed": {
                            "type": "glob",
                            "pattern": "*mlp.experts.linear_fc*",
                            "config": "mxfp8",
                            "enabled": True,
                        },
                        "other": {
                            "type": "glob",
                            "pattern": "*",
                            "config": "bf16",
                            "enabled": True,
                        },
                    },
                }
            ),
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_decoder_block_spec(
                config, use_transformer_engine=transformer_impl == "transformer_engine"
            ),
            vocab_size=256,
            max_sequence_length=32,
        )
        quantized_names = []
        use_mxfp8_storage = transformer_impl == "inference_optimized" or recipe_storage.get(
            "inherit_model_init_context", False
        )
        for name, parameter in model.named_parameters():
            expected_mxfp8 = (
                use_mxfp8_storage
                and name.startswith(("decoder.layers.2.", "decoder.layers.3."))
                and ".mlp.experts.linear_fc" in name
            )
            assert isinstance(parameter, MXFP8Tensor) == expected_mxfp8, name
            if expected_mxfp8:
                quantized_names.append(name)
        # Two middle layers, two projections, two experts; training defaults remain BF16.
        assert len(quantized_names) == (8 if use_mxfp8_storage else 0)

    def test_te_config_resolution_dense(self) -> None:
        from megatron.core.extensions.transformer_engine import (
            TELayerNormColumnParallelLinear,
            TERowParallelLinear,
        )

        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=False,
            gated_linear_unit=True,
            bias_activation_fusion=True,
            add_bias_linear=False,
            quant_recipe=RecipeConfig.from_config_dict(
                {
                    "matchers": {
                        "force_in_hp": {
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
                        "bf16": {
                            "transformer_engine_config_type": "TEQuantizationParams",
                            "training_recipe": {},
                        },
                        "fp8_cs": {
                            "transformer_engine_config_type": "TEQuantizationParams",
                            "training_recipe": {"fp8_quantization_recipe": "tensorwise"},
                        },
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
            "decoder.layers.0.self_attention.linear_proj": TERowParallelLinear,
            "decoder.layers.1.self_attention.linear_proj": TERowParallelLinear,
            "decoder.layers.0.self_attention.linear_qkv": TELayerNormColumnParallelLinear,
            "decoder.layers.1.self_attention.linear_qkv": TELayerNormColumnParallelLinear,
            "decoder.layers.0.mlp.linear_fc1": TELayerNormColumnParallelLinear,
            "decoder.layers.1.mlp.linear_fc1": TELayerNormColumnParallelLinear,
            "decoder.layers.0.mlp.linear_fc2": TERowParallelLinear,
            "decoder.layers.1.mlp.linear_fc2": TERowParallelLinear,
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
                assert hasattr(module, "te_quant_params")
                config_expected = expected_match[name][1]
                if config_expected == "bf16":
                    assert module.te_quant_params.training_recipe.fp8_quantization_recipe is None
                    assert module.te_quant_params.training_recipe.fp4_quantization_recipe is None
                    assert not module.te_quant_params.training_recipe.override_nonquantized_autocast
                    assert module.te_quant_params.training_recipe.override_quantized_autocast
                    assert module.te_quant_params.evaluation_recipe is None
                else:  # fp8_cs
                    assert (
                        module.te_quant_params.training_recipe.fp8_quantization_recipe
                        == Fp8Recipe.tensorwise
                    )
                    assert module.te_quant_params.training_recipe.fp4_quantization_recipe is None
                    assert module.te_quant_params.evaluation_recipe is None
        assert visited_keys == set(expected_types.keys())
