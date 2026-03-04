# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

HAVE_KITCHEN = False

from unittest.mock import MagicMock

AutogradFunctionImplementation = MagicMock()
KitchenSpecProvider = MagicMock()

QAttentionParamsConfigSchema = MagicMock()
QFlashAttentionParamsConfigSchema = MagicMock()
QLinearParamsConfigSchema = MagicMock()
QLinearParams = MagicMock()
QuantizeRecipe = MagicMock()
QuantizeRecipeAttnBMM = MagicMock()
get_qattention_params_from_predefined = MagicMock()
get_qfa_params_from_recipe_name = MagicMock()
get_qlinear_params_from_predefined = MagicMock()
get_qlinear_params_from_qat_params = MagicMock()

KitchenColumnParallelGroupedLinear = MagicMock()
KitchenColumnParallelLinear = MagicMock()
KitchenDotProductAttention = MagicMock()
KitchenFlashAttention = MagicMock()
KitchenLayerNormColumnParallelLinear = MagicMock()
KitchenRowParallelGroupedLinear = MagicMock()
KitchenRowParallelLinear = MagicMock()

# N.B. Kitchen extension is not released publicly.
# This extension is just a stub.
