# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from copy import deepcopy
from typing import Any, Dict

import pytest

from megatron.core.extensions.transformer_engine import (
    HAVE_TE,
    TEQuantizationParams,
    TEQuantizationRecipe,
)
from megatron.core.quantization.quant_config import (
    GlobMatcher,
    MatchContext,
    QuantizationConfig,
    RecipeConfig,
)
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from megatron.core.extensions.kitchen import (
        HAVE_KITCHEN,
        AutogradFunctionImplementation,
        QAttentionParamsConfigSchema,
        QFlashAttentionParamsConfigSchema,
        QLinearParamsConfigSchema,
        QuantizeRecipe,
        QuantizeRecipeAttnBMM,
        get_qattention_params_from_predefined,
        get_qfa_params_from_recipe_name,
        get_qlinear_params_from_predefined,
    )

except ImportError:
    HAVE_KITCHEN = False


def test_recipe_config_matching() -> None:

    recipe_config = RecipeConfig(
        [
            GlobMatcher("*fc2", "fc2_cfg"),
            GlobMatcher("*fc*", "fc_cfg"),
            GlobMatcher("*", "default"),
        ],
        {"fc2_cfg": {"fc2": "foo"}, "fc_cfg": {"fc1": "bar"}, "default": {"default": "baz"}},
    )

    assert (
        recipe_config.match_to_config_key(MatchContext("decoder.1.linear_fc2", layer_number=1))
        == "fc2_cfg"
    )
    assert (
        recipe_config.match_to_config_key(MatchContext("decoder.1.linear_fc1", layer_number=1))
        == "fc_cfg"
    )
    assert (
        recipe_config.match_to_config_key(MatchContext("decoder.1.linear_qkv", layer_number=1))
        == "default"
    )


def test_recipe_config_round_trip() -> None:
    recipe_config = RecipeConfig(
        [
            GlobMatcher("*linear_fc2", "fc2_cfg"),
            GlobMatcher("*linear_fc*", "fc_cfg"),
            GlobMatcher("*", "default"),
        ],
        {
            "fc2_cfg": {"format": "nvfp4", "block_size": 16},
            "fc_cfg": {"format": "mxfp8", "enabled": True},
            "default": {"format": "bf16"},
        },
    )

    config_dict = deepcopy(recipe_config.to_cfg_dict())
    deserialized = RecipeConfig.from_config_dict(config_dict)

    assert deserialized.configs == recipe_config.configs
    assert [type(matcher) for matcher in deserialized.matchers] == [
        type(matcher) for matcher in recipe_config.matchers
    ]
    assert [vars(matcher) for matcher in deserialized.matchers] == [
        vars(matcher) for matcher in recipe_config.matchers
    ]


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine required.")
@pytest.mark.parametrize(
    ("model_overrides", "recipe_overrides", "inherits"),
    [
        ({}, {}, True),
        ({}, {"fp8_param": False}, False),
        ({}, {"fp8_param": True}, False),
        ({}, {"inherit_model_init_context": False}, False),
        ({}, {"inherit_model_init_context": True}, True),
        ({}, {"fp4_param": False}, False),
        ({}, {"fp8_quantization_recipe": None}, False),
        ({"transformer_impl": "transformer_engine"}, {}, False),
        ({"fp8_param": False}, {}, False),
        ({"fp8_recipe": "tensorwise"}, {}, False),
        ({"fp8": None, "fp8_param": False}, {}, False),
    ],
)
def test_te_inference_storage_defaults(model_overrides, recipe_overrides, inherits) -> None:
    """Resolve omission without changing bool defaults or mutating shared recipes."""
    model_kwargs = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        normalization="RMSNorm",
        add_bias_linear=False,
        transformer_impl="inference_optimized",
        fp8="e4m3",
        fp8_recipe="mxfp8",
        fp8_param=True,
    )
    model_kwargs.update(model_overrides)
    recipe = {"fp8_quantization_recipe": "mxfp8", **recipe_overrides}
    quant_config = QuantizationConfig(
        {
            "transformer_engine_config_type": "TEQuantizationParams",
            "training_recipe": recipe,
            "evaluation_recipe": recipe,
        },
        MatchContext("decoder.layers.0.mlp.experts.linear_fc1", 0),
        "mxfp8",
    )
    original = deepcopy(quant_config.config)
    parsed = TEQuantizationParams.parse_from_config(
        quant_config, model_config=TransformerConfig(**model_kwargs)
    )
    for resolved in (parsed.training_recipe, parsed.evaluation_recipe):
        assert resolved.inherit_model_init_context is inherits
        assert resolved.fp8_param is recipe_overrides.get("fp8_param", False)
    assert quant_config.config == original
    # Parsing the same recipe without an inference config keeps the old defaults.
    ordinary = TEQuantizationParams.parse_from_config(quant_config)
    assert ordinary.training_recipe.inherit_model_init_context is recipe_overrides.get(
        "inherit_model_init_context", False
    )
    assert TEQuantizationRecipe().fp8_param is False


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine required.")
@pytest.mark.parametrize("value", [None, "false", 0])
def test_te_recipe_rejects_non_boolean_fp8_param(value) -> None:
    with pytest.raises(ValueError, match="fp8_param must be a bool"):
        TEQuantizationRecipe.parse_from_config({"fp8_param": value})


@pytest.mark.skipif(not HAVE_KITCHEN, reason="Kitchen required for using kitchen backend.")
def test_parse_qlinear_params_example() -> None:
    qat_params = 2
    config = {"kitchen_config_type": "QLinearParams", "recipe_idx": qat_params}
    qlinear_params_actual = QLinearParamsConfigSchema.parse_config_dict(config).to_kitchen_qlinear()
    qlinear_params_expected = get_qlinear_params_from_predefined(QuantizeRecipe.FP8_CS)
    assert qlinear_params_actual.x_params == qlinear_params_expected.x_params
    assert qlinear_params_actual.w_params == qlinear_params_expected.w_params
    assert qlinear_params_actual.g_params == qlinear_params_expected.g_params
    assert qlinear_params_actual.mm_fprop == qlinear_params_expected.mm_fprop
    assert qlinear_params_actual.mm_dgrad == qlinear_params_expected.mm_dgrad
    assert qlinear_params_actual.mm_wgrad == qlinear_params_expected.mm_wgrad
    assert (
        qlinear_params_actual.autograd_function_implementation
        == AutogradFunctionImplementation.QUANTIZED
    )

    qat_params = 6001
    config = {"kitchen_config_type": "QAttentionParams", "recipe_idx": qat_params}
    qattention_params_actual = QAttentionParamsConfigSchema.parse_config_dict(
        config
    ).to_kitchen_qattention()
    qattention_params_expected = get_qattention_params_from_predefined(
        QuantizeRecipeAttnBMM.MXFP8_EMULATION
    )
    assert type(qattention_params_actual.quantizer_bmm1) == type(
        qattention_params_expected.quantizer_bmm1
    )
    assert type(qattention_params_actual.quantizer_bmm2) == type(
        qattention_params_expected.quantizer_bmm2
    )
    assert type(qattention_params_actual.get_quantizer(True)) == type(
        qattention_params_expected.get_quantizer(True)
    )
    assert type(qattention_params_actual.get_quantizer(False)) == type(
        qattention_params_expected.get_quantizer(False)
    )


@pytest.mark.skipif(not HAVE_KITCHEN, reason="Kitchen required for using kitchen backend.")
def test_error_from_malformed() -> None:
    qat_params = 2
    config: Dict[Any, Any] = {"recipe_idx": qat_params}
    with pytest.raises(KeyError, match="Missing required keys"):
        qlinear_params_actual = QLinearParamsConfigSchema.parse_config_dict(config)
    config = {"kitchen_config_type": "QLinearParams"}
    with pytest.raises(KeyError, match="Missing required keys"):
        qlinear_params_actual = QLinearParamsConfigSchema.parse_config_dict(config)
    config = {"kitchen_config_type": "QUnknownParams", "recipe_idx": qat_params}
    with pytest.raises(ValueError, match="Unsupported config type"):
        qlinear_params_actual = QLinearParamsConfigSchema.parse_config_dict(config)
    config = {"kitchen_config_type": "QLinearParams", "recipe_idx": "MyRecipe"}
    with pytest.raises(ValueError, match="recipe_idx must be a positive integer"):
        qlinear_params_actual = QLinearParamsConfigSchema.parse_config_dict(config)
    config = {
        "kitchen_config_type": "QLinearParams",
        "recipe_idx": qat_params,
        "extra_key": "extra_value",
    }
    with pytest.raises(KeyError, match="Unexpected keys in config"):
        qlinear_params_actual = QLinearParamsConfigSchema.parse_config_dict(config)


@pytest.mark.skipif(not HAVE_KITCHEN, reason="Kitchen required for using kitchen backend.")
def test_parse_qflash_attention_params_example() -> None:
    recipe_name = "triton_fa_bf16_for_all_base_2"
    config = {"kitchen_config_type": "QFlashAttentionParams", "recipe_name": recipe_name}
    qfa_params_actual = QFlashAttentionParamsConfigSchema.parse_config_dict(config).to_kitchen_qfa()
    qfa_params_expected = get_qfa_params_from_recipe_name(recipe_name)

    # Verify they are the same object (since recipes are cached)
    assert qfa_params_actual is qfa_params_expected
    assert qfa_params_actual.backend == "triton"
    assert qfa_params_actual.qk_dot_precisions == "bf16@bf16"
    assert qfa_params_actual.pv_dot_precisions == "bf16@bf16"
    assert qfa_params_actual.use_natural_transcendental_func is False

    # Test with natural recipe
    recipe_name = "triton_fa_bf16_for_all_natural"
    config = {"kitchen_config_type": "QFlashAttentionParams", "recipe_name": recipe_name}
    qfa_params_actual = QFlashAttentionParamsConfigSchema.parse_config_dict(config).to_kitchen_qfa()
    qfa_params_expected = get_qfa_params_from_recipe_name(recipe_name)

    assert qfa_params_actual is qfa_params_expected
    assert qfa_params_actual.backend == "triton"
    assert qfa_params_actual.use_natural_transcendental_func is True


@pytest.mark.skipif(not HAVE_KITCHEN, reason="Kitchen required for using kitchen backend.")
def test_error_from_malformed_qflash_attention_params() -> None:
    # Missing recipe_name
    config: Dict[Any, Any] = {"kitchen_config_type": "QFlashAttentionParams"}
    with pytest.raises(KeyError, match="Missing required keys"):
        qfa_params_actual = QFlashAttentionParamsConfigSchema.parse_config_dict(config)

    # Missing kitchen_config_type
    config = {"recipe_name": "triton_fa_bf16_for_all_base_2"}
    with pytest.raises(KeyError, match="Missing required keys"):
        qfa_params_actual = QFlashAttentionParamsConfigSchema.parse_config_dict(config)

    # Wrong config type
    config = {
        "kitchen_config_type": "QLinearParams",
        "recipe_name": "triton_fa_bf16_for_all_base_2",
    }
    with pytest.raises(ValueError, match="Parsing config dict of incorrect type"):
        qfa_params_actual = QFlashAttentionParamsConfigSchema.parse_config_dict(config)

    # Unsupported config type
    config = {
        "kitchen_config_type": "QUnknownParams",
        "recipe_name": "triton_fa_bf16_for_all_base_2",
    }
    with pytest.raises(ValueError, match="Unsupported config type"):
        qfa_params_actual = QFlashAttentionParamsConfigSchema.parse_config_dict(config)

    # Extra keys
    config = {
        "kitchen_config_type": "QFlashAttentionParams",
        "recipe_name": "triton_fa_bf16_for_all_base_2",
        "extra_key": "extra_value",
    }
    with pytest.raises(KeyError, match="Unexpected keys in config"):
        qfa_params_actual = QFlashAttentionParamsConfigSchema.parse_config_dict(config)

    # Invalid recipe_name (not a string)
    config = {"kitchen_config_type": "QFlashAttentionParams", "recipe_name": 123}
    with pytest.raises(ValueError, match="recipe_name must be a string"):
        qfa_params_actual = QFlashAttentionParamsConfigSchema.parse_config_dict(config)
