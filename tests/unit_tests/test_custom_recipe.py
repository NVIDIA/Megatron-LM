# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import operator
from contextlib import nullcontext
from unittest.mock import Mock, patch

import pytest

from megatron.core.enums import Fp8Recipe
from megatron.core.extensions import transformer_engine as te_extension
from megatron.core.quantization import custom_recipe, te_recipe
from megatron.core.transformer.transformer_config import TransformerConfig

RECORDED_ROLES = []


def delayed_scaling_test_factory(role):
    """Pinned-TE-compatible factory used by real MCore execution tests."""
    from transformer_engine.common.recipe import Format
    from transformer_engine.pytorch.quantization import DelayedScalingRequest

    RECORDED_ROLES.append(role)
    return DelayedScalingRequest(fp8_format=Format.HYBRID)


TEST_FACTORY_PATH = f"{__name__}.delayed_scaling_test_factory"


def test_resolve_quantizer_factory_returns_callable():
    assert custom_recipe.resolve_quantizer_factory("operator.neg") is operator.neg


@pytest.mark.parametrize(
    ("path", "error"),
    [
        ("", "must be a non-empty string"),
        ("factory", "Expected 'package.module.callable'"),
        ("missing_package.module.factory", "Failed to import module"),
        ("operator.missing_factory", "Attribute 'missing_factory' not found"),
        ("operator.__doc__", "is not callable"),
    ],
)
def test_resolve_quantizer_factory_rejects_invalid_paths(path, error):
    with pytest.raises(ValueError, match=error):
        custom_recipe.resolve_quantizer_factory(path)


@pytest.mark.skipif(custom_recipe.te_recipe is None, reason="Transformer Engine is not installed")
def test_build_custom_recipe_forwards_attention_flags():
    factory = Mock()

    recipe = custom_recipe.build_custom_recipe(factory, fp8_dpa=True, fp8_mha=True)

    assert recipe.qfactory is factory
    assert recipe.fp8_dpa is True
    assert recipe.fp8_mha is True


@pytest.mark.skipif(custom_recipe.te_recipe is None, reason="Transformer Engine is not installed")
def test_build_custom_recipe_rejects_unsupported_attention_flags():
    class RecipeWithoutAttentionFlags:

        def __init__(self, qfactory):
            self.qfactory = qfactory

    with patch.object(custom_recipe.te_recipe, "CustomRecipe", RecipeWithoutAttentionFlags):
        with pytest.raises(ValueError, match="supports the 'fp8_dpa' constructor argument"):
            custom_recipe.build_custom_recipe(Mock(), fp8_dpa=True)


@pytest.mark.skipif(custom_recipe.te_recipe is None, reason="Transformer Engine is not installed")
@pytest.mark.parametrize(
    ("config_kwargs", "recipe_getter"),
    [
        (
            {"fp8": "hybrid", "fp8_recipe": "custom", "fp8_quantizer_factory": TEST_FACTORY_PATH},
            "fp8",
        ),
        (
            {"fp4": "e2m1", "fp4_recipe": "custom", "fp4_quantizer_factory": TEST_FACTORY_PATH},
            "fp4",
        ),
    ],
)
def test_legacy_custom_recipe_paths_forward_attention_flags(config_kwargs, recipe_getter):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        fp8_dot_product_attention=True,
        fp8_multi_head_attention=True,
        **config_kwargs,
    )
    if recipe_getter == "fp8":
        from megatron.core.fp8_utils import get_fp8_recipe

        recipe = get_fp8_recipe(config)
        cached_recipe = get_fp8_recipe(config)
    else:
        from megatron.core.fp4_utils import get_fp4_recipe

        recipe = get_fp4_recipe(config)
        cached_recipe = get_fp4_recipe(config)

    assert cached_recipe is recipe
    assert recipe.qfactory is delayed_scaling_test_factory
    assert recipe.fp8_dpa is True
    assert recipe.fp8_mha is True


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"fp8": "hybrid", "fp8_recipe": "custom", "fp8_quantizer_factory": TEST_FACTORY_PATH},
        {"fp4": "e2m1", "fp4_recipe": "custom", "fp4_quantizer_factory": TEST_FACTORY_PATH},
    ],
)
def test_legacy_custom_recipe_warns_once(config_kwargs):
    mode = "fp8" if "fp8" in config_kwargs else "fp4"
    custom_recipe._WARNED_LEGACY_CUSTOM_RECIPE_MODES.discard(mode)
    try:
        with pytest.warns(
            FutureWarning, match="Pass the factory path to --custom-recipe"
        ) as warnings:
            for _ in range(2):
                TransformerConfig(
                    num_layers=1, hidden_size=128, num_attention_heads=4, **config_kwargs
                )
        custom_recipe_warnings = [
            warning
            for warning in warnings
            if issubclass(warning.category, FutureWarning)
            and f"--{mode}-recipe custom" in str(warning.message)
        ]
        assert len(custom_recipe_warnings) == 1
    finally:
        custom_recipe._WARNED_LEGACY_CUSTOM_RECIPE_MODES.discard(mode)


@pytest.mark.skipif(not te_extension.HAVE_TE, reason="Transformer Engine is not installed")
def test_per_module_custom_recipe_uses_format_neutral_autocast():
    config = te_extension.TEQuantizationRecipe(
        fp8_quantization_recipe=Fp8Recipe.custom,
        fp8_format="hybrid",
        custom_recipe_factory=TEST_FACTORY_PATH,
        override_nonquantized_autocast=True,
    )
    expected_context = nullcontext()

    with (
        patch.object(te_extension.FP8GlobalStateManager, "is_fp8_enabled", return_value=False),
        patch.object(
            te_extension.te.pytorch, "autocast", return_value=expected_context
        ) as autocast,
    ):
        context = te_extension._get_fp8_autocast_for_quant_recipe(config)

    assert context is expected_context
    assert autocast.call_args.kwargs["recipe"].qfactory is delayed_scaling_test_factory


@pytest.mark.skipif(not te_extension.HAVE_TE, reason="Transformer Engine is not installed")
def test_per_module_custom_recipe_rejects_quantized_parameter_storage():
    with pytest.raises(ValueError, match="do not yet support quantized parameter storage"):
        te_extension.TEQuantizationRecipe.parse_from_config(
            {
                "fp8_quantization_recipe": Fp8Recipe.custom,
                "custom_recipe_factory": TEST_FACTORY_PATH,
                "fp8_param": True,
            }
        )


@pytest.mark.skipif(not te_recipe.HAVE_TE, reason="Transformer Engine is not installed")
def test_get_quantization_context_uses_format_neutral_te_api():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        custom_recipe=TEST_FACTORY_PATH,
        fp8_dot_product_attention=True,
        fp8_multi_head_attention=True,
    )
    expected_context = nullcontext()

    with patch.object(te_recipe.te, "autocast", return_value=expected_context) as autocast:
        context = te_recipe.get_quantization_context(config)

    assert context is expected_context
    call_kwargs = autocast.call_args.kwargs
    assert call_kwargs["enabled"] is True
    assert call_kwargs["amax_reduction_group"] is None
    assert call_kwargs["recipe"].qfactory is delayed_scaling_test_factory
    assert call_kwargs["recipe"].fp8_dpa is True
    assert call_kwargs["recipe"].fp8_mha is True


@pytest.mark.skipif(not te_recipe.HAVE_TE, reason="Transformer Engine is not installed")
def test_custom_recipe_init_context_is_noop_without_parameter_storage():
    config = TransformerConfig(
        num_layers=1, hidden_size=128, num_attention_heads=4, custom_recipe=TEST_FACTORY_PATH
    )

    with te_recipe.get_quantization_context(config, is_init=True):
        pass


@pytest.mark.skipif(not te_extension.HAVE_TE, reason="Transformer Engine is not installed")
def test_custom_recipe_is_materialized_once_per_config():
    config = TransformerConfig(
        num_layers=1, hidden_size=128, num_attention_heads=4, custom_recipe=TEST_FACTORY_PATH
    )

    first = te_recipe.get_te_quantization_recipe(config)
    second = te_recipe.get_te_quantization_recipe(config)

    assert first is second


@pytest.mark.skipif(not te_extension.HAVE_TE, reason="Transformer Engine is not installed")
def test_custom_recipe_alignment_uses_declared_value_or_conservative_fallback():
    config = TransformerConfig(
        num_layers=1, hidden_size=128, num_attention_heads=4, custom_recipe=TEST_FACTORY_PATH
    )
    recipe = te_recipe.get_te_quantization_recipe(config)

    expected = getattr(recipe, "quantization_alignment", 128)
    assert te_recipe.get_quantization_alignment(config) == expected
    with patch.object(te_recipe, "get_te_quantization_recipe", return_value=object()):
        assert te_recipe.get_quantization_alignment(config) == 128
