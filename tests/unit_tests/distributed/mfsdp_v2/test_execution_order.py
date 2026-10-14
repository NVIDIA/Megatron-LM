# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Coverage for occurrence-based FSDP prefetch and weak module ownership."""

import gc
from unittest.mock import Mock
from weakref import ref

import pytest
from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionOrder,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule


@pytest.mark.parametrize(
    'budget, expected',
    [
        (None, [[1], [0], [0], [2], []]),
        (0, [[], [], [], [], []]),
        (2, [[1, 0], [0, 0], [0, 2], [2], []]),
    ],
)
def test_prefetch_replays_occurrences(budget, expected):
    """Reused modules have occurrence-specific successors under every prefetch budget."""
    modules = [Mock(spec=FsdpModule, num_parameter_elements=1) for _ in range(3)]
    prefetched = []
    for module in modules:
        module._unshard_parameter_groups.side_effect = lambda target=module: prefetched.append(
            target
        )
    sequence = [modules[index] for index in (0, 1, 0, 0, 2)]
    order = ExecutionOrder()
    for module in sequence:
        order.append(module)
    # A second cycle checks cursor reset; each lookahead must leave demand calls intact.
    for _ in range(2):
        for module, targets, successor in zip(sequence, expected, sequence[1:] + [None]):
            prefetched.clear()
            FsdpModule._prefetch_parameter_groups(module, order, budget)
            assert prefetched == [modules[index] for index in targets]
            assert order.next_item() is successor


def test_execution_order_validates_calls():
    """Empty or mismatched calls raise without consuming the expected occurrence."""
    first, second = nn.Module(), nn.Module()
    empty = ExecutionOrder()
    assert empty.next_item() is None
    with pytest.raises(RuntimeError, match='diverged'):
        empty.advance(first)

    order = ExecutionOrder([first, second, first])
    with pytest.raises(RuntimeError, match='diverged'):
        order.advance(second)
    order.advance(first)
    assert order.next_item() is second
    assert order.next_item(2) is first
    assert order.next_item(3) is None
    with pytest.raises(RuntimeError, match='diverged'):
        order.advance(first)
    order.advance(second)
    assert order.next_item() is first
    order.advance(first)
    assert order.next_item() is None

    # Scheduler-selected windows stop at their boundary, including the last window.
    order.reset(2, end=3)
    order.advance(first)
    assert order.next_item() is None
    with pytest.raises(RuntimeError, match='diverged'):
        order.advance(first)
    order.reset()
    order.advance(first)
    assert order.next_item() is second
    with pytest.raises(ValueError, match='reset range'):
        order.reset(4)


def test_membership_reverse_and_weak_references():
    """Traversal preserves repeated occurrences without retaining deleted modules."""
    first, second = nn.Module(), nn.Module()
    order = ExecutionOrder([first, second, first, second])
    assert first in order
    assert second in order
    assert list(order) == [first, second, first, second]
    assert list(reversed(order)) == [second, first, second, first]

    second_ref = ref(second)
    del second
    gc.collect()
    assert second_ref() is None
    assert None not in order
    assert list(order) == [first, first]
    assert list(reversed(order)) == [first, first]
    order.advance(first)
    assert order.next_item() is None
    # Traversal skips dead references, but the demand cursor must not skip a call.
    with pytest.raises(RuntimeError, match='diverged'):
        order.advance(first)
