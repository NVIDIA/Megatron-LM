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
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.phase import Phase


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
        module.phase = Phase.FORWARD
        # Prefetch passes the successor's compute phase; record only which module it targets.
        module._unshard_parameter_groups.side_effect = (
            lambda _phase, target=module: prefetched.append(target)
        )
    sequence = [modules[index] for index in (0, 1, 0, 0, 2)]
    order = ExecutionOrder()
    for module in sequence:
        order.append((module, Phase.FORWARD))
    # A second pass checks repeated occurrence matching after exhaustion.
    for _ in range(2):
        for index, (module, targets) in enumerate(zip(sequence, expected)):
            prefetched.clear()
            FsdpModule._prefetch_parameter_groups(module, order, budget)
            assert prefetched == [modules[index] for index in targets]
            assert order.position == index + 1


def test_unique_calls_resynchronize_and_repeated_calls_validate():
    """Unique VPP chunk calls select their position; ambiguous repeats stay strict."""
    chunk0, layer0, chunk1, layer1 = ((nn.Module(), Phase.FORWARD) for _ in range(4))
    order = ExecutionOrder([chunk0, layer0, chunk1, layer1])
    # Grouped VPP repeats chunk 0 before switching to chunk 1, in one shared order.
    for chunk in (chunk0, chunk0, chunk0, chunk1, chunk1, chunk0):
        layer = layer0 if chunk == chunk0 else layer1
        assert order.advance(chunk) == chunk
        assert order[order.position] == layer
        order.advance(layer)
        if chunk == chunk0:
            assert order[order.position] == chunk1
        else:
            assert order.position == len(order)

    order = ExecutionOrder([chunk0, layer0, chunk0])
    assert order.advance(chunk0) == chunk0
    assert order[order.position] == layer0
    with pytest.raises(RuntimeError, match='diverged'):
        order.advance(chunk0)
    # A rejected repeated call must not consume the expected occurrence.
    order.advance(layer0)
    assert order[order.position] == chunk0
    assert order.advance(chunk0) == chunk0
    assert order.position == len(order)
    # Unique calls remain valid anchors even when the sequence also has repeats.
    order.advance(layer0)
    assert order.advance(chunk0) == chunk0
    assert order.position == len(order)

    empty = ExecutionOrder()
    assert len(empty) == 0
    with pytest.raises(RuntimeError, match='diverged'):
        empty.advance(chunk0)


def test_iteration_and_weak_references():
    """Traversal preserves repeated occurrences without retaining deleted modules."""
    first, second = nn.Module(), nn.Module()
    order = ExecutionOrder((module, Phase.FORWARD) for module in [first, second, first, second])
    assert [module for module, _ in order] == [first, second, first, second]

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
    """Only reset starts another pass; lookahead and iteration never advance it."""
    module = nn.Module()
    a, b = (module, Phase.FORWARD), (module, Phase.BACKWARD)
    order = ExecutionOrder([a, b, a])
    for _ in range(2):
        order.reset_cursor()
        assert order.advance() == a
        assert order[order.position] == b
        assert order.position == 1
        assert list(order) == [a, b, a]
        assert order.advance() == b
        assert order.advance() == a
        assert list(order)[order.position :] == []
        with pytest.raises(RuntimeError, match='exceeded'):
            order.advance()
    order.reset_cursor(1)
    assert order.advance(b) == b
    empty = ExecutionOrder()
    assert list(empty) == []
    with pytest.raises(RuntimeError, match='exceeded'):
        empty.advance()
