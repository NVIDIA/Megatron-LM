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
        order.append(ref(module))
    # A second cycle checks cursor reset; each lookahead must leave demand calls intact.
    for _ in range(2):
        order.reset()
        for module, targets, successor in zip(sequence, expected, sequence[1:] + [None]):
            prefetched.clear()
            FsdpModule._prefetch_parameter_groups(module, order, budget)
            assert prefetched == [modules[index] for index in targets]
            next_ref = order.next_item()
            assert (next_ref() if next_ref is not None else None) is successor


def test_explicit_reset_and_exhaustion():
    """Only reset starts another pass; lookahead and iteration never advance it."""
    a, b = nn.Module(), nn.Module()
    order = ExecutionOrder([a, b, a])
    for _ in range(2):
        order.reset()
        assert order.advance() is a
        assert order.next_item() is b
        assert order.next_item() is b
        assert list(order) == [a, b, a]
        assert order.advance() is b
        assert order.advance() is a
        assert order.next_item() is None
        with pytest.raises(RuntimeError, match='exceeded'):
            order.advance()
    empty = ExecutionOrder()
    assert empty.next_item() is None
    with pytest.raises(RuntimeError, match='exceeded'):
        empty.advance()


def test_iteration_and_weak_references():
    """Tuple payloads preserve occurrences while callers keep module ownership weak."""
    first, second = nn.Module(), nn.Module()
    a, b = (ref(first), 'forward'), (ref(second), 'backward')
    order = ExecutionOrder([a, b, a, b])
    assert list(order) == [a, b, a, b]
    assert order.advance() is a
    assert order.next_item() is b
    assert order.next_item(2) is a

    second_ref = ref(second)
    del second
    gc.collect()
    assert second_ref() is None
    assert order.advance() is b
    assert b[0]() is None
    assert order.position == 2 and len(order) == 4

    # Static FSDP callers detect a destroyed owner without skipping its occurrence.
    static = ExecutionOrder([ref(first), second_ref])
    FsdpModule._prefetch_parameter_groups(first, static, None)
    with pytest.raises(RuntimeError, match='destroyed'):
        FsdpModule._prefetch_parameter_groups(first, static, None)
