# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""get_megatron_optimizer dispatches a MimoModel to get_mimo_optimizer."""

from types import SimpleNamespace
from unittest import mock

from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.optimizer import get_megatron_optimizer


def test_get_megatron_optimizer_dispatches_mimo_model():
    # __new__ gives an isinstance-true MimoModel without running its heavy __init__.
    fake_mimo = MimoModel.__new__(MimoModel)
    sentinel = object()
    with mock.patch(
        "megatron.core.models.mimo.optimizer.get_mimo_optimizer", return_value=sentinel
    ) as get_mimo:
        result = get_megatron_optimizer(SimpleNamespace(optimizer="adam"), [fake_mimo])
    get_mimo.assert_called_once()
    assert result is sentinel
