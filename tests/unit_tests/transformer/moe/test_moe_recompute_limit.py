# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""moe_recompute_max_layer_number limits MoE selective recompute to the earlier layers."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig


def _layer(limit, module_enabled=True):
    layer = MoELayer.__new__(MoELayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(moe_recompute_max_layer_number=limit)
    layer._moe_recompute_module_enabled = module_enabled
    layer.router = Mock()
    return layer


@pytest.mark.parametrize(
    "limit,layer_number,expected",
    [(None, 7, True), (10, 7, True), (10, 10, True), (10, 11, False), (10, None, True)],
)
def test_limit_selects_layers(limit, layer_number, expected):
    layer = _layer(limit)
    layer.set_layer_number(layer_number)
    assert layer.moe_layer_recompute is expected
    layer.router.set_layer_number.assert_called_once_with(layer_number)


def test_limit_never_enables_recompute_without_the_moe_module():
    layer = _layer(None, module_enabled=False)
    layer.set_layer_number(3)
    assert layer.moe_layer_recompute is False


def test_config_default_checkpoints_every_moe_layer():
    assert TransformerConfig.__dataclass_fields__["moe_recompute_max_layer_number"].default is None
