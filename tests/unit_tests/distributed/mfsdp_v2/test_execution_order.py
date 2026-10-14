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
    order = ExecutionOrder(sequence, recorded=True)
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
    assert order.next_item(first) is second


def test_static_order_lookup():
    """Construction orders remain unique and independent of demand-call order."""
    first, second, third = (object.__new__(FsdpModule) for _ in range(3))
    order = ExecutionOrder()
    for module in (first, second, third):
        order.append(module)
    with pytest.raises(ValueError, match='duplicate'):
        order.append(first)
    order.advance(third)
    assert order.next_item(first) is second
    assert order.next_item(first, offset=2) is third
    assert order.next_item(third) is None


@pytest.mark.parametrize('budget', [None, 0, 2])
def test_construction_prefetch_tolerates_reordered_calls(budget):
    """Inferred order remains a prefetch hint when calls skip or reorder modules."""
    first, second, third = (object.__new__(FsdpModule) for _ in range(3))
    prefetched = []
    order = ExecutionOrder()
    for module in (first, second, third):
        parameter = SimpleNamespace(unsharded=torch.empty(1))
        module._parameter_groups = (SimpleNamespace(fsdp_parameters=[parameter]),)
        module._unshard_parameter_groups = Mock(
            side_effect=lambda target=module: prefetched.append(target)
        )
        order.append(module)
    # Start at the last module, skip the middle module, and reuse the first.
    for module in (third, first, first, second):
        prefetched.clear()
        module._prefetch_parameter_groups(order, budget)
        expected = []
        if budget != 0:
            if module is first:
                expected = [second, third] if budget == 2 else [second]
            elif module is second:
                expected = [third]
        assert prefetched == expected


@pytest.mark.parametrize('recorded', [False, True])
def test_membership_reverse_and_weak_references(recorded):
    """Topology lookups and traversal preserve weak ownership in both modes."""
    first, second = nn.Module(), nn.Module()
    sequence = [first, second, first] if recorded else [first, second]
    order = ExecutionOrder(sequence, recorded=recorded)
    assert first in order
    assert second in order
    assert list(reversed(order)) == list(reversed(sequence))
    sequence.clear()
    second_ref = ref(second)
    del second
    gc.collect()
    assert second_ref() is None
    assert None not in order
    expected = [first, first] if recorded else [first]
    assert list(order) == expected
    assert list(reversed(order)) == expected
    if not recorded:
        assert order.next_item(first) is None


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
