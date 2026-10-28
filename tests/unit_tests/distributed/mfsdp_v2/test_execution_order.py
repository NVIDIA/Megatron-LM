# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Coverage for occurrence-based FSDP prefetch and weak module ownership."""

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


@pytest.mark.parametrize(
    'budget, expected',
    [(None, [[1], [2], [2], []]), (0, [[], [], [], []]), (2, [[1, 2], [2], [2], []])],
)
@pytest.mark.parametrize("phase", [Phase.FORWARD, Phase.BACKWARD])
def test_prefetch_reuses_adjacent_shared_calls(budget, expected, phase):
    """Construction-order prefetch reuses a shared MTP block's cursor and successors."""
    modules = [Mock(spec=FsdpModule, num_parameter_elements=1) for _ in range(3)]
    prefetched = []
    for module in modules:
        module.phase = phase
        # Prefetch passes the successor's compute phase; record only which module it targets.
        module._unshard_parameter_groups.side_effect = (
            lambda _phase, target=module: prefetched.append(target)
        )
    sequence = [modules[index] for index in (0, 1, 1, 2)]
    order = ExecutionOrder(ExecutionEvent(module=module, phase=phase) for module in modules)
    # Root entry starts a new pass through the construction order.
    for _ in range(2):
        order.reset_cursor()
        for module, targets, position in zip(sequence, expected, (1, 2, 2, 3)):
            prefetched.clear()
            FsdpModule._prefetch_parameter_groups(module, order, budget)
            assert prefetched == [modules[index] for index in targets]
            assert order.position == position


def test_iteration_and_weak_references():
    """Traversal preserves repeated occurrences without retaining deleted modules."""
    first, second = nn.Module(), nn.Module()
    order = ExecutionOrder(
        ExecutionEvent(module=module, phase=Phase.FORWARD)
        for module in [first, second, first, second]
    )
    assert [event.module for event in order] == [first, second, first, second]

    second_ref = ref(second)
    del second
    gc.collect()
    assert second_ref() is None
    order.advance()
    with pytest.raises(RuntimeError, match="destroyed"):
        order[order.position]
    order.reset_cursor()
    first.phase = Phase.FORWARD
    with pytest.raises(RuntimeError, match="destroyed"):
        FsdpModule._prefetch_parameter_groups(first, order, None)
    assert order.position == 1


def test_explicit_reset_and_exhaustion():
    """Sequential traversal requires reset before replay."""
    module = nn.Module()
    a = ExecutionEvent(module=module, phase=Phase.FORWARD)
    b = ExecutionEvent(module=module, phase=Phase.BACKWARD)
    order = ExecutionOrder([a, b, a, a])
    for _ in range(2):
        order.reset_cursor()
        assert order.advance() == a
        assert order[order.position] == b
        assert order.position == 1
        assert list(order) == [a, b, a, a]
        assert order.advance() == b
        assert order.advance() == a
        assert order.advance() == a
        assert list(order)[order.position :] == []
        with pytest.raises(RuntimeError, match='exceeded'):
            order.advance()
    order.reset_cursor(1)
    assert order.advance() == b
    empty = ExecutionOrder()
    assert list(empty) == []
    with pytest.raises(RuntimeError, match='exceeded'):
        empty.advance()
