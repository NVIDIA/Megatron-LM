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
