# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for construction and recorded FSDP execution sequences."""

import gc
from types import SimpleNamespace
from unittest.mock import Mock
from weakref import ref

import pytest
import torch
from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.context import FsdpContext
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionOrder,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule


@pytest.mark.parametrize('budget', [None, 0, 2])
def test_prefetch_replays_occurrences(budget):
    """A reused module prefetches different successors at each recorded position."""
    modules, prefetched = [], []
    for _ in range(3):
        module = object.__new__(FsdpModule)
        parameter = SimpleNamespace(unsharded=torch.empty(1))
        module._parameter_groups = (SimpleNamespace(fsdp_parameters=[parameter]),)
        module._unshard_parameter_groups = Mock(
            side_effect=lambda target=module: prefetched.append(target)
        )
        modules.append(module)
    first, second, third = modules
    sequence = [first, second, first, third]
    order = ExecutionOrder(sequence)
    expected = [[second], [first], [third], []]
    if budget == 0:
        expected = [[], [], [], []]
    elif budget == 2:
        expected = [[second, first], [first, third], [third], []]
    for _ in range(2):
        for module, targets in zip(sequence, expected):
            prefetched.clear()
            module._prefetch_parameter_groups(order, budget)
            assert prefetched == targets
    with pytest.raises(RuntimeError, match='diverged'):
        order.advance(third)
    order.advance(first)
    assert order.next_item() is second


def test_append_preserves_repeated_occurrences():
    """Appending the same module records distinct calls with distinct successors."""
    first, second = nn.Module(), nn.Module()
    order = ExecutionOrder()
    for module in (first, second, first, first):
        order.append(module)
    assert list(order) == [first, second, first, first]
    for module, successor in ((first, second), (second, first), (first, first), (first, None)):
        order.advance(module)
        assert order.next_item() is successor


@pytest.mark.parametrize('budget', [None, 0, 2])
def test_prefetch_rejects_skipped_or_reordered_calls(budget):
    """Even with prefetch disabled, calls must follow the expected sequence."""
    first, second, third = (object.__new__(FsdpModule) for _ in range(3))
    order = ExecutionOrder([first, second, third])
    for module in (first, second, third):
        parameter = SimpleNamespace(unsharded=torch.empty(1))
        module._parameter_groups = (SimpleNamespace(fsdp_parameters=[parameter]),)
        module._unshard_parameter_groups = Mock()
    with pytest.raises(RuntimeError, match='diverged'):
        third._prefetch_parameter_groups(order, budget)
    first._prefetch_parameter_groups(order, budget)
    with pytest.raises(RuntimeError, match='diverged'):
        third._prefetch_parameter_groups(order, budget)
    second._prefetch_parameter_groups(order, budget)
    third._prefetch_parameter_groups(order, budget)
    assert order.next_item() is None


def test_lookahead_preserves_demand_cursor():
    """Repeated lookahead does not consume calls or wrap across the sequence end."""
    first, second, third = (nn.Module() for _ in range(3))
    order = ExecutionOrder([first, second, third])
    for _ in range(2):
        order.advance(first)
        assert order.next_item() is second
        assert order.next_item(2) is third
        assert order.next_item(3) is None
        assert order.next_item() is second
        order.advance(second)
        assert order.next_item() is third
        order.advance(third)
        assert order.next_item() is None


def test_empty_order_rejects_demand_calls():
    """An empty sequence has no successor and cannot consume a module call."""
    order = ExecutionOrder()
    assert order.next_item() is None
    with pytest.raises(RuntimeError, match='diverged'):
        order.advance(nn.Module())


def test_membership_reverse_and_weak_references():
    """Topology lookups and traversal preserve weak ownership and occurrences."""
    first, second = nn.Module(), nn.Module()
    sequence = [first, second, first]
    order = ExecutionOrder(sequence)
    assert first in order
    assert second in order
    assert list(reversed(order)) == list(reversed(sequence))
    sequence.clear()
    second_ref = ref(second)
    del second
    gc.collect()
    assert second_ref() is None
    assert None not in order
    assert list(order) == [first, first]
    assert list(reversed(order)) == [first, first]
    order.advance(first)
    # A deleted occurrence cannot be silently skipped by the demand cursor.
    assert order.next_item() is None
    with pytest.raises(RuntimeError, match='diverged'):
        order.advance(first)


def test_context_finalizes_construction_orders(monkeypatch):
    """Root membership, module names, and forward/backward topology remain intact."""
    monkeypatch.setattr(torch.cuda, 'Stream', Mock())
    root, first, second, independent = (nn.Module() for _ in range(4))
    root.first = first
    root.second = second
    context = FsdpContext(torch.device('cuda'))
    for module in (first, second, root, independent):
        context.register_module(module)
    context.finalize()
    assert list(context.forward_order) == [root, first, second, independent]
    assert list(context.backward_order) == [independent, root, second, first]
    assert context.is_root(root)
    assert context.is_root(independent)
    assert not context.is_root(first)
    assert not context.is_root(second)
    assert context.module_name(root) == ''
    assert context.module_name(first) == 'first'
    assert context.module_name(second) == 'second'
