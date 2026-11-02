# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""get_megatron_optimizer dispatches a MimoModel to get_mimo_optimizer."""

from unittest import mock

from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer


def test_get_megatron_optimizer_dispatches_mimo_model():
    # __new__ gives an isinstance-true MimoModel without running its heavy __init__.
    fake_mimo = MimoModel.__new__(MimoModel)
    config = OptimizerConfig(optimizer="adam")
    sentinel = object()
    with mock.patch(
        "megatron.core.models.mimo.optimizer.get_mimo_optimizer", return_value=sentinel
    ) as get_mimo:
        result = get_megatron_optimizer(config, [fake_mimo])
    get_mimo.assert_called_once_with(fake_mimo, config)
    assert result is sentinel
