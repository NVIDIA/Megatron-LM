# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from megatron.core.enums import Fp8Recipe
from megatron.core.models.common.model_chunk_schedule_plan import TransformerLayerSchedulePlan
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
from megatron.core.transformer.transformer_layer import TransformerLayer


def _schedule_plan(layer):
    plan = TransformerLayerSchedulePlan.__new__(TransformerLayerSchedulePlan)
    plan.layer = layer
    return plan


def test_schedule_uses_layer_quantization_context():
    expected_context = nullcontext()
    layer = SimpleNamespace(get_inner_quantization_context=Mock(return_value=expected_context))

    context = _schedule_plan(layer).get_low_precision_context()

    assert context is expected_context
    layer.get_inner_quantization_context.assert_called_once_with()


def test_transformer_layer_uses_fp4_context():
    config = SimpleNamespace(fp8=None, fp8_recipe=Fp8Recipe.delayed, fp4="e2m1")
    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.config = config
    layer.layer_number = 3
    expected_context = nullcontext()

    with patch(
        "megatron.core.fp4_utils.get_fp4_context", return_value=expected_context
    ) as get_fp4_context:
        context = layer.get_inner_quantization_context()

    assert context is expected_context
    get_fp4_context.assert_called_once_with(config, 2)


def test_transformer_layer_uses_fp8_context():
    config = SimpleNamespace(fp8="e4m3", fp8_recipe=Fp8Recipe.tensorwise, fp4=None)
    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.config = config
    layer.layer_number = 3
    expected_context = nullcontext()

    with patch(
        "megatron.core.fp8_utils.get_fp8_context", return_value=expected_context
    ) as get_fp8_context:
        context = layer.get_inner_quantization_context()

    assert context is expected_context
    get_fp8_context.assert_called_once_with(config, 2)


def test_mtp_layer_uses_global_fp8_context():
    config = SimpleNamespace(fp8="e4m3", fp8_recipe=Fp8Recipe.tensorwise, fp4=None)
    layer = MultiTokenPredictionLayer.__new__(MultiTokenPredictionLayer)
    torch.nn.Module.__init__(layer)
    layer.config = config
    layer.layer_number = 1
    expected_context = nullcontext()

    with patch(
        "megatron.core.transformer.multi_token_prediction.get_fp8_context",
        return_value=expected_context,
    ) as get_fp8_context:
        context = layer.get_inner_quantization_context()

    assert context is expected_context
    get_fp8_context.assert_called_once_with(config)


def test_mtp_layer_does_not_use_fp4_context():
    config = SimpleNamespace(fp8=None, fp8_recipe=Fp8Recipe.delayed, fp4="e2m1")
    layer = MultiTokenPredictionLayer.__new__(MultiTokenPredictionLayer)
    torch.nn.Module.__init__(layer)
    layer.config = config

    context = layer.get_inner_quantization_context()

    assert isinstance(context, nullcontext)
