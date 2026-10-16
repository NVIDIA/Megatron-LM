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
        order.reset()
        for module, targets in zip(sequence, expected):
            prefetched.clear()
            FsdpModule._prefetch_parameter_groups(module, order, budget)
            assert prefetched == [modules[index] for index in targets]
            assert list(order.prefetch(budget)) == prefetched


def test_explicit_reset_and_exhaustion():
    """Only reset starts another pass; lookahead and iteration never advance it."""
    a, b = nn.Module(), nn.Module()
    order = ExecutionOrder([a, b, a])
    for _ in range(2):
        order.reset()
        assert order.advance() is a
        assert list(order.prefetch()) == [b]
        assert list(order.prefetch()) == [b]
        assert list(order) == [a, b, a]
        assert order.advance() is b
        assert order.advance() is a
        assert list(order.prefetch()) == []
        with pytest.raises(RuntimeError, match='exceeded'):
            order.advance()
    empty = ExecutionOrder()
    assert list(empty.prefetch()) == []
    with pytest.raises(RuntimeError, match='exceeded'):
        empty.advance()


def test_iteration_and_weak_references():
    """Traversal preserves repeated occurrences without retaining deleted modules."""
    first, second = nn.Module(), nn.Module()
    order = ExecutionOrder([first, second, first, second])
    assert list(order) == [first, second, first, second]

    second_ref = ref(second)
    del second
    gc.collect()
    assert second_ref() is None
    assert list(order) == [first, first]
    assert order.advance() is first
    assert list(order.prefetch()) == []
    # Traversal skips dead references, but the demand cursor must not skip a call.
    with pytest.raises(RuntimeError, match='destroyed'):
        order.advance()
