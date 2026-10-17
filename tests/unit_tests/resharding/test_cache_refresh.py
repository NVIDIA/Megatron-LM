# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import MagicMock

import torch

from megatron.core.resharding.execution import execute_reshard_plan, refresh_module_caches
from megatron.core.resharding.utils import ReshardPlan
from megatron.core.transformer.module import MegatronModule


class _CacheAwareModule(MegatronModule):
    def __init__(self, events: list[str]):
        super().__init__(config=MagicMock())
        self.events = events

    def refresh_cache(self) -> None:
        self.events.append("refresh")


def test_default_refresh_cache_is_noop():
    module = MegatronModule(config=MagicMock())

    assert module.refresh_cache() is None


def test_refresh_module_caches_accepts_model_lists():
    events: list[str] = []

    refresh_module_caches(None)
    refresh_module_caches([torch.nn.Sequential(_CacheAwareModule(events)), torch.nn.Sequential()])

    assert events == ["refresh"]


def test_execute_reshard_plan_refreshes_caches_before_final_sync(monkeypatch):
    events: list[str] = []
    target_core = torch.nn.Sequential(_CacheAwareModule(events))
    service = MagicMock()
    service.requires_process_group_barrier = False
    service.execute_plan.return_value = False
    service.run.side_effect = lambda: events.append("transfer")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))

    execute_reshard_plan(
        ReshardPlan(send_ops=[], recv_ops=[]),
        src_module=None,
        dst_module=target_core,
        service=service,
    )

    assert events == ["transfer", "sync", "refresh", "sync"]
