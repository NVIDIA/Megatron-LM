# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only tests for MIMO's nested DDP overlap lifecycle."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.optimizer import MimoOptimizer


def _overlap_stub(modules):
    stub = SimpleNamespace()
    stub._active_ddp_modules = lambda: iter(modules)
    for name in (
        "no_sync",
        "enable_forward_pre_hook",
        "disable_forward_pre_hook",
        "start_param_sync",
        "start_grad_sync",
        "free_overlap_buffers",
    ):
        setattr(stub, name, getattr(MimoModel, name).__get__(stub))
    return stub


def _ddp(*, grad_overlap, param_overlap, events, name):
    module = MagicMock()
    module.ddp_config = SimpleNamespace(
        overlap_grad_reduce=grad_overlap, overlap_param_gather=param_overlap
    )

    @contextmanager
    def no_sync():
        events.append(f"{name}:enter")
        try:
            yield
        finally:
            events.append(f"{name}:exit")

    module.no_sync.side_effect = no_sync
    return module


def test_nested_overlap_lifecycle_routes_only_to_enabled_modules():
    events = []
    language = _ddp(grad_overlap=True, param_overlap=True, events=events, name="language")
    encoder = _ddp(grad_overlap=True, param_overlap=False, events=events, name="encoder")
    inactive = _ddp(grad_overlap=False, param_overlap=False, events=events, name="inactive")
    model = _overlap_stub([language, encoder, inactive])

    with model.no_sync():
        events.append("body")

    assert events == ["language:enter", "encoder:enter", "body", "encoder:exit", "language:exit"]
    inactive.no_sync.assert_not_called()

    model.enable_forward_pre_hook()
    model.disable_forward_pre_hook(param_sync=False)
    model.start_param_sync(force_sync=True, force_dispatch=True)
    model.start_grad_sync()
    model.free_overlap_buffers()

    language.enable_forward_pre_hook.assert_called_once_with()
    language.disable_forward_pre_hook.assert_called_once_with(param_sync=False)
    language.start_param_sync.assert_called_once_with(force_sync=True, force_dispatch=True)
    language.start_grad_sync.assert_called_once_with()
    language.free_overlap_buffers.assert_called_once_with()

    encoder.enable_forward_pre_hook.assert_not_called()
    encoder.start_param_sync.assert_not_called()
    encoder.start_grad_sync.assert_called_once_with()
    inactive.start_grad_sync.assert_not_called()


def test_mimo_optimizer_stages_each_active_optimizer_before_param_sync():
    language_optimizer = MagicMock()
    encoder_optimizer = MagicMock()
    optimizer = SimpleNamespace(_active_optimizers=[language_optimizer, encoder_optimizer])

    MimoOptimizer.prepare_model_params_for_param_sync(optimizer)

    language_optimizer.prepare_model_params_for_param_sync.assert_called_once_with()
    encoder_optimizer.prepare_model_params_for_param_sync.assert_called_once_with()
