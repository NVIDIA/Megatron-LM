# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Weak occurrence sequences and independent execution/prefetch iterators."""

import gc
from weakref import ref

import pytest
from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionOrder,
)


def test_independent_occurrence_iterators():
    """Prefetch can advance ahead without moving or resynchronizing execution."""
    a, b = nn.Module(), nn.Module()
    order = ExecutionOrder([a, a, b, b, a])
    execution, prefetch = iter(order), iter(order)
    assert next(execution) is a
    assert [next(prefetch), next(prefetch), next(prefetch)] == [a, a, b]
    assert list(execution) == [a, b, b, a]
    assert list(prefetch) == [b, a]
    assert list(order) == [a, a, b, b, a]
    assert len(order) == 5
    assert order.position == 0
    assert order.advance() is a
    assert order.next_prefetch() is a
    assert order.next_prefetch() is b
    assert order.position == 1
    assert list(order) == [a, a, b, b, a]
    assert [order.advance() for _ in range(4)] == [a, b, b, a]
    assert order.next_prefetch() is None
    assert order.advance() is a  # a completed static pass restarts both cursors
    assert order.position == 1
    assert order.next_prefetch() is a
    order.reset()
    assert order.position == 0
    assert order.advance() is a


def test_append_during_recording_and_empty_order():
    """A recording iterator sees new occurrences appended before consumption."""
    a, b = nn.Module(), nn.Module()
    order = ExecutionOrder()
    execution = iter(order)
    order.append(a)
    assert next(execution) is a
    order.append(b)
    assert next(execution) is b
    assert list(execution) == []
    assert list(ExecutionOrder()) == []
    with pytest.raises(RuntimeError, match="empty"):
        ExecutionOrder().advance()


def test_order_does_not_retain_modules_or_skip_dead_occurrences():
    """Destroyed owners are errors, preserving occurrence positions."""
    a, b = nn.Module(), nn.Module()
    order = ExecutionOrder([a, b, a])
    b_ref = ref(b)
    del b
    gc.collect()
    assert b_ref() is None
    execution = iter(order)
    assert next(execution) is a
    with pytest.raises(RuntimeError, match="destroyed"):
        next(execution)
    # A cursor kept between static passes must not retain its last yielded owner.
    a_ref = ref(a)
    live_iterator = iter(order)
    assert next(live_iterator) is a
    del a
    gc.collect()
    assert a_ref() is None
    with pytest.raises(RuntimeError, match="destroyed"):
        order.advance()
