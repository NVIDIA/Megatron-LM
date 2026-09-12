# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core.enums import Fp8Recipe
from megatron.core.models.common.model_chunk_schedule_plan import TransformerLayerSchedulePlan
from megatron.core.models.hybrid.model_chunk_schedule_plan import HybridStackSchedulePlan
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


@pytest.mark.parametrize("layer_type", [("*", "E"), "M", None])
def test_hybrid_schedule_runs_without_a_transformer_context_hook(layer_type):
    """The common scheduler must dispatch to the hybrid quantization hook."""
    plan = HybridStackSchedulePlan.__new__(HybridStackSchedulePlan)
    plan.layer = SimpleNamespace()
    plan.layer_type = layer_type
    visited = []
    for name in (
        "pre_dispatch_computation",
        "moe_dispatch",
        "mlp",
        "moe_combine",
        "mtp_post_process",
    ):
        setattr(
            plan,
            name,
            SimpleNamespace(forward=lambda value, name=name: visited.append(name) or value),
        )

    value = torch.ones(1)
    output, _ = TransformerLayerSchedulePlan.run(plan, None, f_input=value)

    assert output is value
    assert visited == [
        "pre_dispatch_computation",
        "moe_dispatch",
        "mlp",
        "moe_combine",
        "mtp_post_process",
    ]


def test_hybrid_schedule_preserves_plain_layer_quantization_context():
    expected_context = nullcontext()
    layer = SimpleNamespace(get_inner_quantization_context=Mock(return_value=expected_context))
    plan = HybridStackSchedulePlan.__new__(HybridStackSchedulePlan)
    plan.layer = layer
    plan.layer_type = None

    assert plan.get_low_precision_context() is expected_context
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
