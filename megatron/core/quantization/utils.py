# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import re
from typing import Optional, Union

from .quant_config import GlobMatcher, MatchContext, QuantizationConfig, RecipeConfig


def get_quant_config_or_none(
    module_path: str, recipe: Optional[RecipeConfig] = None
) -> Union[QuantizationConfig, None]:
    """Resolve quantization config for a layer."""
    re_match = re.search(r'layers\.(\d+)', module_path)
    if re_match:
        layer_number: Optional[int] = int(re_match.group(1))
    else:
        layer_number = None
    if recipe is not None:
        return recipe.match(MatchContext(module_path=module_path, layer_number=layer_number))
    else:
        return None


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
