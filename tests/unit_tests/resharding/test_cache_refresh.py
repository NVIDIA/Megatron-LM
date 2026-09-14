# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import nullcontext
from unittest.mock import MagicMock

import torch

import megatron.core.resharding.execution as execution
from megatron.core.resharding.execution import execute_reshard_plan, refresh_module_caches
from megatron.core.resharding.utils import ReshardPlan, TransferOp
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


def test_execution_batches_share_one_final_device_sync(monkeypatch):
    """Multiple runs stay asynchronous until the complete plan finishes."""
    events: list[str] = []
    target_core = torch.nn.Sequential(_CacheAwareModule(events))
    service = MagicMock()
    service.supports_multiple_runs_per_plan = True
    service.requires_process_group_barrier = False
    service.execute_plan.return_value = False
    service.run.side_effect = lambda: events.append("transfer")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))

    execute_reshard_plan(
        ReshardPlan(send_ops=[], recv_ops=[], num_batches=2),
        src_module=None,
        dst_module=target_core,
        service=service,
    )

    assert events == ["transfer", "transfer", "sync", "refresh", "sync"]


def test_mxfp8_prefetch_waits_for_previous_batch_consumers(monkeypatch):
    """A later dequant batch cannot recycle storage still in use by the current stream."""
    events: list[str] = []

    class _CurrentStream:
        def wait_event(self, _event):
            events.append("consume")

    class _PrefetchStream:
        def wait_stream(self, stream):
            assert stream is current_stream
            events.append("order")

    class _Event:
        def record(self):
            events.append("produce")

    current_stream = _CurrentStream()
    prefetch_stream = _PrefetchStream()
    monkeypatch.setattr(torch.cuda, "Stream", lambda: prefetch_stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current_stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(execution, "is_mxfp8tensor", lambda _tensor: True)

    def _dequantize(param):
        events.append("dequantize")
        return param.data

    monkeypatch.setattr(execution, "_ensure_sendable", _dequantize)

    src_module = torch.nn.Module()
    src_module.register_parameter("first", torch.nn.Parameter(torch.ones(2)))
    src_module.register_parameter("second", torch.nn.Parameter(torch.ones(2)))
    full_slice = (slice(None),)
    plan = ReshardPlan(
        send_ops=[
            TransferOp("first", 1, True, full_slice, full_slice, task_id=0, batch_id=0),
            TransferOp("second", 1, True, full_slice, full_slice, task_id=1, batch_id=1),
        ],
        recv_ops=[],
        num_batches=2,
    )
    service = MagicMock()
    service.supports_multiple_runs_per_plan = True
    service.requires_process_group_barrier = False
    service.execute_plan.return_value = False
    service.run.side_effect = lambda: events.append("transfer")

    execute_reshard_plan(plan, src_module=src_module, dst_module=None, service=service)

    assert events == [
        "order",
        "dequantize",
        "produce",
        "consume",
        "transfer",
        "order",
        "dequantize",
        "produce",
        "consume",
        "transfer",
        "sync",
        "sync",
    ]
