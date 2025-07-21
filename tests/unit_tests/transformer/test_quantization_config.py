from typing import Any, Dict

import pytest

from megatron.core.quantization.quant_config import GlobMatcher, MatchContext, RecipeConfig

try:
    import nvidia_kitchen
    from nvidia_kitchen.config import (
        AutogradFunctionImplementation,
        QuantizeRecipe,
        get_qlinear_params_from_predefined,
    )

    from megatron.core.extensions.kitchen import QLinearParamsConfigSchema

    HAVE_KITCHEN = True
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
