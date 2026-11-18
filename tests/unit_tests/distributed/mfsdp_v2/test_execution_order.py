# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""A weak event source supports strict replay and independent prefetch hints."""

import gc
from unittest.mock import patch
from weakref import ref

import pytest
import torch
from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionAction,
    ExecutionEvent,
    ExecutionOrder,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.phase import Phase


def test_full_replay_and_independent_filtered_hints():
    module = nn.Module()
    order = ExecutionOrder()
    request = ExecutionEvent(module, Phase.FORWARD, unsharded_bytes=40)
    release = ExecutionEvent(module, Phase.FORWARD, ExecutionAction.RESHARD, None, None, True, 40)
    with (
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        patch.object(torch.cuda, "memory_allocated", return_value=0),
    ):
        assert order.record_event(request, device=torch.device("cuda")) == 0
    assert order.append(release) == 1
    assert order.append(request) == 2
    full = order.view()
    hints = order.view(action=ExecutionAction.UNSHARD)
    assert [event.event_id for event in full] == [0, 1, 2]
    assert order.seek(0).allocated_bytes == 0 and order.seek(0).unsharded_bytes == 40
    assert order.seek(1).rowwise_only and order.seek(1).unsharded_bytes == 40
    assert order.seek(0) == ExecutionEvent(module, Phase.FORWARD, unsharded_bytes=999)
    for index, expected in enumerate((request, release, request)):
        assert full.peek() == expected and full.position == index
        full.advance()
    assert full.position == len(full) and full.peek() is None
    with pytest.raises(RuntimeError, match="exhausted"):
        full.advance()
    assert hints.position == 0
    for _ in range(2):
        hints.advance()
    assert hints.position == 0
    full.reset()
    assert full.peek() == request
    assert order.append(ExecutionEvent(module, Phase.BACKWARD)) == 3
    assert len(full) == 3 and len(hints) == 2
    for name in ("allocated_bytes", "unsharded_bytes"):
        for bad in (-1, True, 1.5):
            with pytest.raises(ValueError, match=name):
                order.append(ExecutionEvent(module, None, **{name: bad}))
    with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
        with pytest.raises(RuntimeError, match="before CUDA graph capture"):
            order.record_event(request, device=torch.device("cuda"))
    assert len(order) == 4


def test_weak_events_keep_ids_and_distinguish_missing_and_zero_bytes():
    first, second = nn.Module(), nn.Module()
    order = ExecutionOrder(
        [
            ExecutionEvent(first, Phase.FORWARD),
            ExecutionEvent(second, Phase.FORWARD, unsharded_bytes=0),
            ExecutionEvent(first, Phase.FORWARD, ExecutionAction.RESHARD, unsharded_bytes=40),
        ]
    )
    assert order.seek(0).unsharded_bytes is None
    assert order.seek(1).unsharded_bytes == 0
    source = order.view()
    second_ref = ref(second)
    del second
    gc.collect()
    assert second_ref() is None and order.seek(1) is None
    assert len(source) == 3
    assert [event.event_id for event in source] == [0, 2]
    assert source.peek().module is first
    source.advance()
    assert source.peek() is None
    with pytest.raises(RuntimeError, match="empty"):
        ExecutionOrder().view().advance()
