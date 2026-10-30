# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Coverage for one cursor over registration or recorded execution sequences."""

import gc
from unittest.mock import Mock
from weakref import ref

import pytest
from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionEvent,
    ExecutionOrder,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
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


@pytest.mark.parametrize("budget", [None, 0, 2])
def test_recorded_prefetch_replays_interleaved_occurrences(budget):
    """Repeated modules and phase changes prefetch the exact recorded successor."""
    modules = [Mock(spec=FsdpModule, num_parameter_elements=1) for _ in range(3)]
    calls = [
        (0, Phase.FORWARD),
        (0, Phase.FORWARD),
        (1, Phase.FORWARD),
        (0, Phase.BACKWARD),
        (2, Phase.FORWARD),
        (1, Phase.BACKWARD),
        (0, Phase.BACKWARD),
        (2, Phase.BACKWARD),
    ]
    events = [ExecutionEvent(modules[index], phase) for index, phase in calls]
    order = ExecutionOrder(events)
    prefetched = []
    for module in modules:
        module._unshard_parameter_groups.side_effect = (
            lambda phase, target=module: prefetched.append(ExecutionEvent(target, phase))
        )
    for _ in range(2):
        for index, event in enumerate(events):
            prefetched.clear()
            event.module.phase = event.phase
            FsdpModule._prefetch_parameter_groups(event.module, order, budget)
            count = 1 if budget is None else budget
            assert prefetched == events[index + 1 : index + 1 + count]
        assert order.position == 0


@pytest.mark.parametrize("wrong_phase", [False, True])
def test_recorded_prefetch_rejects_divergence_without_consuming(wrong_phase):
    """Even unique modules must match both the expected module and phase."""
    first, second = (Mock(spec=FsdpModule, num_parameter_elements=1) for _ in range(2))
    events = [ExecutionEvent(first, Phase.FORWARD), ExecutionEvent(second, Phase.BACKWARD)]
    order = ExecutionOrder(events)
    module = first if wrong_phase else second
    phase = Phase.BACKWARD if wrong_phase else Phase.FORWARD
    module.phase = phase
    with pytest.raises(RuntimeError, match="diverged.*position 0"):
        FsdpModule._prefetch_parameter_groups(module, order, None)
    assert order.peek() == events[0]
    assert order.position == 0
    first._unshard_parameter_groups.assert_not_called()
    second._unshard_parameter_groups.assert_not_called()
    first.phase = Phase.FORWARD
    FsdpModule._prefetch_parameter_groups(first, order, 0)
    assert order.peek() == events[1]


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
