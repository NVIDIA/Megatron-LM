# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import re
from typing import Any, Optional, Union

from megatron.core.enums import Fp4Recipe, Fp8Recipe

from .quant_config import GlobMatcher, MatchContext, QuantizationConfig, RecipeConfig


def is_quantization_enabled(config: Any) -> bool:
    """Return whether any global quantization mode is enabled."""
    return bool(
        getattr(config, "custom_recipe", None)
        or getattr(config, "fp8", None)
        or getattr(config, "fp4", None)
    )


def is_custom_recipe_selected(config: Any) -> bool:
    """Return whether the global quantization mode is a Transformer Engine custom recipe.

    Covers the canonical ``custom_recipe`` path as well as the deprecated
    ``fp8_recipe="custom"`` and ``fp4_recipe="custom"`` spellings.
    """
    if getattr(config, "custom_recipe", None):
        return True
    if getattr(config, "fp8", None) and getattr(config, "fp8_recipe", None) == Fp8Recipe.custom:
        return True
    if getattr(config, "fp4", None) and getattr(config, "fp4_recipe", None) == Fp4Recipe.custom:
        return True
    return False


def get_quant_config_or_none(
    module_path: Optional[str], recipe: Optional[RecipeConfig] = None
) -> Union[QuantizationConfig, None]:
    """Resolve quantization config for a layer."""
    if recipe is None or module_path is None:
        return None
    re_match = re.search(r'layers\.(\d+)', module_path)
    if re_match:
        layer_number: Optional[int] = int(re_match.group(1))
    else:
        layer_number = None
    return recipe.match(MatchContext(module_path=module_path, layer_number=layer_number))


def load_quantization_recipe(recipe_path: str) -> RecipeConfig:
    """Loads a quantization recipe from a path."""
    recipe = RecipeConfig.from_yaml_file(recipe_path)
    return recipe


def kitchen_quantization_recipe_config(recipe_idx: int) -> RecipeConfig:
    """Loads a quantization recipe that uses a QAT_PARAMS recipe for all layers."""
    recipe = RecipeConfig(
        matchers=[GlobMatcher(pattern="*", config_key="default")],
        config_dict={"default": {"kitchen_config_type": "QLinearParams", "recipe_idx": recipe_idx}},
    )
    return recipe
