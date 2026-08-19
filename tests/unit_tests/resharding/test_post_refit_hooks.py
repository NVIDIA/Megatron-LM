# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import MagicMock

import torch

import megatron.core.resharding.refit as refit
from megatron.core.transformer.module import MegatronModule


class _RefitAwareModule(MegatronModule):
    def __init__(self, events: list[str]):
        super().__init__(config=MagicMock())
        self.events = events

    def post_refit(self) -> None:
        self.events.append("hook")


def test_default_post_refit_is_noop():
    module = MegatronModule(config=MagicMock())

    assert module.post_refit() is None


def test_reshard_runs_post_refit_hooks_after_transfer(monkeypatch):
    events: list[str] = []
    target_core = torch.nn.Sequential(_RefitAwareModule(events))
    plan = MagicMock()

    monkeypatch.setattr(
        refit, "_unwrap_model_cores", lambda src_model, target_model: (None, target_core, None)
    )
    monkeypatch.setattr(refit, "_build_or_get_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(refit, "_harmonize_buffer_dtypes", lambda *args, **kwargs: None)

    def execute(*args, **kwargs):
        events.append("transfer")

    monkeypatch.setattr(refit, "execute_reshard_plan", execute)

    refit.reshard_model_weights(src_model=None, target_model=target_core, service=MagicMock())

    assert events == ["transfer", "hook"]
