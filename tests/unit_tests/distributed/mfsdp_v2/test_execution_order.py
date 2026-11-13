# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Coverage for one cursor over registration or recorded execution sequences."""

import gc
from weakref import ref

import pytest
from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionEvent,
    ExecutionOrder,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.phase import Phase


def test_cursor_wraps_across_phases():
    """One cursor follows the complete trace, including adjacent repeated calls."""
    first, second = nn.Module(), nn.Module()
    events = [
        ExecutionEvent(first, Phase.FORWARD),
        ExecutionEvent(first, Phase.FORWARD),
        ExecutionEvent(second, Phase.FORWARD),
        ExecutionEvent(first, Phase.BACKWARD),
        ExecutionEvent(second, Phase.BACKWARD),
        ExecutionEvent(first, Phase.BACKWARD),
    ]
    order = ExecutionOrder(events)
    for _ in range(2):
        for index, event in enumerate(events):
            assert order.peek() == event
            assert list(order.successors()) == events[index + 1 :]
            assert order.peek() == event
            order.advance()
        assert order.position == 0


def test_empty_order_can_be_populated():
    """Recording does not move the cursor, so replay starts at the first event."""
    order = ExecutionOrder()
    assert order.peek() is None
    with pytest.raises(RuntimeError, match="empty"):
        order.advance()
    event = ExecutionEvent(nn.Module(), Phase.FORWARD)
    order.append(event)
    assert order.peek() == event
    for _ in range(2):
        assert list(order.successors()) == []
        order.advance()
        assert order.peek() == event


def test_order_does_not_keep_modules_alive():
    """The sequence holds weak references and iteration skips dead modules."""
    first, second = nn.Module(), nn.Module()
    order = ExecutionOrder(
        [ExecutionEvent(first, Phase.FORWARD), ExecutionEvent(second, Phase.FORWARD)]
    )
    second_ref = ref(second)
    del second
    gc.collect()
    assert second_ref() is None
    assert list(order) == [ExecutionEvent(first, Phase.FORWARD)]
    assert list(order.successors()) == []
