# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from megatron.core.enums import Fp8Recipe
from megatron.core.models.common.model_chunk_schedule_plan import TransformerLayerSchedulePlan
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer


def _schedule_plan(layer):
    plan = TransformerLayerSchedulePlan.__new__(TransformerLayerSchedulePlan)
    plan.layer = layer
    return plan


def test_fp4_context_is_used_for_transformer_layer():
    config = SimpleNamespace(fp8=None, fp8_recipe=Fp8Recipe.delayed, fp4="e2m1")
    layer = SimpleNamespace(config=config, layer_number=3)
    expected_context = nullcontext()

    with patch(
        "megatron.core.models.common.model_chunk_schedule_plan.get_fp4_context",
        return_value=expected_context,
    ) as get_fp4_context:
        context = _schedule_plan(layer).get_low_precision_context()

    assert context is expected_context
    get_fp4_context.assert_called_once_with(config, 2)


def test_fp4_context_is_not_used_for_mtp_layer():
    config = SimpleNamespace(fp8=None, fp8_recipe=Fp8Recipe.delayed, fp4="e2m1")
    layer = MultiTokenPredictionLayer.__new__(MultiTokenPredictionLayer)
    torch.nn.Module.__init__(layer)
    layer.config = config

    with patch(
        "megatron.core.models.common.model_chunk_schedule_plan.get_fp4_context"
    ) as get_fp4_context:
        with _schedule_plan(layer).get_low_precision_context():
            pass

    get_fp4_context.assert_not_called()
