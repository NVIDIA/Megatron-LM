# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Coverage for occurrence-based FSDP prefetch and weak module ownership."""

import gc
from unittest.mock import Mock
from weakref import ref

import pytest
from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.context import FsdpContext
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionOrder,
    IterationOrder,
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
@pytest.mark.parametrize("recorded", [False, True], ids=["static", "captured"])
def test_prefetch_replays_occurrences(budget, expected, recorded):
    """Reused modules have occurrence-specific successors under every prefetch budget."""
    modules = [Mock(spec=FsdpModule, num_parameter_elements=1) for _ in range(3)]
    prefetched = []
    for module in modules:
        module._unshard_parameter_groups.side_effect = lambda target=module: prefetched.append(
            target
        )
    sequence = [modules[index] for index in (0, 1, 0, 0, 2)]
    order = IterationOrder()
    context = Mock(iteration_order=order)
    for module in modules:
        module.context = context
    for module in sequence:
        order.append((ref(module), "forward"))
    if recorded:
        order.mode = "record_replay"
        order.active = True
        order.captured = True
    else:
        order.backward_start = len(order)
        order.append((ref(modules[0]), "backward"))
        order.append((ref(modules[1]), "backward"))
    # A second cycle checks cursor reset; each lookahead must leave demand calls intact.
    for _ in range(2):
        order.reset_cursor()
        for index, (module, targets) in enumerate(zip(sequence, expected)):
            prefetched.clear()
            assert FsdpContext._advance_order(context, module, "forward") is order
            FsdpModule._prefetch_parameter_groups(module, order, budget)
            assert prefetched == [modules[index] for index in targets]
            assert order.position == index + 1
            end = len(order) if recorded else order.backward_start
            remaining = [module_ref() for module_ref, _ in list(order)[order.position : end]]
            assert remaining == sequence[index + 1 :]
        if not recorded:
            order.reset_cursor(order.backward_start)
            FsdpContext._advance_order(context, modules[0], "backward")
            prefetched.clear()
            FsdpModule._prefetch_parameter_groups(modules[0], order, budget)
            assert prefetched == ([] if budget == 0 else [modules[1]])
            assert order.position == order.backward_start + 1


def test_explicit_reset_and_exhaustion():
    """Only reset starts another pass; lookahead and iteration never advance it."""
    a, b = nn.Module(), nn.Module()
    order = ExecutionOrder([a, b, a])
    for _ in range(2):
        order.reset_cursor()
        assert order.advance() is a
        assert order[order.position] is b
        assert order[order.position] is b
        assert list(order) == [a, b, a]
        assert order.advance() is b
        assert order.advance() is a
        assert list(order)[order.position :] == []
        with pytest.raises(RuntimeError, match='exceeded'):
            order.advance()
    order.reset_cursor(1)
    assert order.advance() is b
    empty = ExecutionOrder()
    assert list(empty) == []
    with pytest.raises(RuntimeError, match='exceeded'):
        empty.advance()


def test_iteration_and_weak_references():
    """Tuple payloads preserve occurrences while callers keep module ownership weak."""
    first, second = nn.Module(), nn.Module()
    a, b = (ref(first), 'forward'), (ref(second), 'backward')
    order = ExecutionOrder([a, b, a, b])
    assert list(order) == [a, b, a, b]
    assert order.advance() is a
    assert order[order.position] is b
    assert order[order.position + 1] is a

    second_ref = ref(second)
    del second
    gc.collect()
    assert second_ref() is None
    assert order.advance() is b
    module_ref, _ = b
    assert module_ref() is None
    assert order.position == 2 and len(order) == 4

    # Static FSDP callers detect a destroyed owner without skipping its occurrence.
    static = IterationOrder()
    static.append((ref(first), "forward"))
    static.append((second_ref, "forward"))
    context = Mock(iteration_order=static)
    FsdpContext._advance_order(context, first, "forward")
    with pytest.raises(RuntimeError, match='destroyed'):
        FsdpContext._advance_order(context, first, "forward")
