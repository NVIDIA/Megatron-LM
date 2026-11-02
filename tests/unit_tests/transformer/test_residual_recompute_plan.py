# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU logic contracts for shared residual-recompute planning and its adapters."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.tensor_parallel.random import CheckpointWithoutOutputManager
from megatron.core.transformer import residual_recompute_plan
from megatron.core.transformer.hyper_connection import (
    build_mhc_recompute_layer_plan,
    finalize_mhc_recompute_layer,
)
from megatron.core.transformer.residual_recompute import (
    ResidualStreamRecomputeContext,
    build_residual_stream_recompute_plan,
)


def _make_hybrid_stack(num_layers, block_size):
    stack = SimpleNamespace(
        layers=[None] * num_layers,
        config=SimpleNamespace(mhc_recompute_layer_num=block_size),
        _mhc_block_end_plan=None,
    )
    stack._compute_mhc_block_end_plan = HybridStack._compute_mhc_block_end_plan.__get__(stack)
    return stack


def _assert_manager_partition(managers, block_ends):
    assert isinstance(managers, list)
    assert len(managers) == len(block_ends)
    assert all(isinstance(manager, CheckpointWithoutOutputManager) for manager in managers)
    assert len({id(manager) for manager in managers}) == sum(block_ends)
    for index in range(1, len(managers)):
        assert (managers[index] is not managers[index - 1]) == block_ends[index - 1]


@pytest.mark.parametrize(
    "num_layers,block_size,atomic_layer_pairs,expected",
    [
        pytest.param(0, None, (), (), id="empty"),
        pytest.param(1, None, (), (True,), id="singleton"),
        pytest.param(4, None, (), (False, False, False, True), id="whole-stack"),
        pytest.param(4, 2, (), (False, True, False, True), id="uniform"),
        pytest.param(5, 2, (), (False, True, False, True, True), id="partial-tail"),
        pytest.param(3, 1, (), (True, True, True), id="single-layer-blocks"),
        pytest.param(3, 8, (), (False, False, True), id="oversized-block"),
        pytest.param(
            4, 3, ((2, 3), (0, 1)), (False, True, False, True), id="unsorted-atomic-pairs"
        ),
        pytest.param(
            5, 1, ((1, 2), (3, 4)), (True, False, True, False, True), id="pairs-exceed-target"
        ),
        pytest.param(
            8,
            4,
            ((3, 4), (5, 6)),
            (False, False, True, False, False, False, True, True),
            id="shorten-before-pair",
        ),
        pytest.param(4, None, ((1, 2),), (False, False, False, True), id="atomic-whole-stack"),
    ],
)
def test_block_end_plan(num_layers, block_size, atomic_layer_pairs, expected):
    block_ends = residual_recompute_plan.build_recompute_block_end_plan(
        num_layers, block_size, atomic_layer_pairs=atomic_layer_pairs
    )

    assert isinstance(block_ends, tuple)
    assert block_ends == expected
    assert all(isinstance(is_end, bool) for is_end in block_ends)


@pytest.mark.parametrize("block_size", [False, True, 0, -1, 1.5])
def test_block_end_plan_rejects_invalid_block_size(block_size):
    with pytest.raises(ValueError, match="positive integer"):
        residual_recompute_plan.build_recompute_block_end_plan(3, block_size)


@pytest.mark.parametrize(
    "pairs,message",
    [
        ([(0,)], "two-item tuples"),
        ([(0, 1, 2)], "two-item tuples"),
        ([[0, 1]], "two-item tuples"),
        ([(False, 1)], "indices must be integers"),
        ([(0, True)], "indices must be integers"),
        ([(0.0, 1)], "indices must be integers"),
        ([(0, 1.0)], "indices must be integers"),
        ([(-1, 0)], "adjacent in-range"),
        ([(1, 3)], "adjacent in-range"),
        ([(2, 3)], "adjacent in-range"),
        ([(1, 0)], "adjacent in-range"),
        ([(0, 1), (1, 2)], "must not overlap"),
        ([(0, 1), (0, 1)], "must not overlap"),
    ],
)
def test_block_end_plan_rejects_malformed_pairs(pairs, message):
    with pytest.raises(ValueError, match=message):
        residual_recompute_plan.build_recompute_block_end_plan(3, 2, atomic_layer_pairs=pairs)


def test_block_end_plan_rejects_negative_layer_count():
    with pytest.raises(ValueError, match="non-negative layer count"):
        residual_recompute_plan.build_recompute_block_end_plan(-1, None)


@pytest.mark.parametrize("block_size", [False, True, 0, -1, 1.5])
def test_active_mhc_adapters_reject_sizes_disallowed_by_config(block_size):
    for num_layers in (1, 3):
        with pytest.raises(ValueError, match="positive integer"):
            build_mhc_recompute_layer_plan(num_layers, block_size, True)
        stack = _make_hybrid_stack(num_layers, block_size)
        with pytest.raises(ValueError, match="positive integer"):
            HybridStack._build_mhc_recompute_layer_plan(stack, True)


def test_active_mhc_adapter_rejects_negative_layer_count():
    with pytest.raises(ValueError, match="non-negative layer count"):
        build_mhc_recompute_layer_plan(-1, None, True)


def test_empty_residual_plan_still_short_circuits_validation(monkeypatch):
    constructor = Mock(wraps=CheckpointWithoutOutputManager)
    monkeypatch.setattr(residual_recompute_plan, "CheckpointWithoutOutputManager", constructor)

    assert build_residual_stream_recompute_plan(0, 0, atomic_layer_pairs=((0, 2),)) == []
    constructor.assert_not_called()


@pytest.mark.parametrize(
    "block_ends",
    [(), (True,), (False, False, True), [False, True, False, True, True], (True, True, True)],
    ids=["empty", "singleton", "whole-stack", "partial-tail", "single-layer-blocks"],
)
def test_managers_are_fresh_with_exactly_one_allocation_per_block(monkeypatch, block_ends):
    constructor = Mock(wraps=CheckpointWithoutOutputManager)
    monkeypatch.setattr(residual_recompute_plan, "CheckpointWithoutOutputManager", constructor)

    first = residual_recompute_plan.build_recompute_layer_managers(block_ends)
    _assert_manager_partition(first, block_ends)
    assert constructor.call_count == sum(block_ends)

    second = residual_recompute_plan.build_recompute_layer_managers(block_ends)
    _assert_manager_partition(second, block_ends)
    assert constructor.call_count == 2 * sum(block_ends)
    assert first is not second
    assert {id(manager) for manager in first}.isdisjoint(id(manager) for manager in second)


@pytest.mark.parametrize("num_layers,block_size", [(0, None), (5, None), (5, 2), (3, 1)])
def test_adapters_agree_without_atomic_pairs(num_layers, block_size):
    expected = residual_recompute_plan.build_recompute_block_end_plan(num_layers, block_size)
    stack = _make_hybrid_stack(num_layers, block_size)
    assert stack._compute_mhc_block_end_plan() == expected
    assert isinstance(stack._compute_mhc_block_end_plan(), tuple)
    prior_managers = []

    for _ in range(2):
        mhc_managers, mhc_ends = build_mhc_recompute_layer_plan(num_layers, block_size, True)
        contexts = build_residual_stream_recompute_plan(num_layers, block_size)
        hybrid_managers, hybrid_ends = HybridStack._build_mhc_recompute_layer_plan(stack, True)

        assert isinstance(contexts, list)
        assert all(isinstance(context, ResidualStreamRecomputeContext) for context in contexts)
        assert [context.is_block_end for context in contexts] == list(expected)
        assert isinstance(mhc_ends, list)
        assert isinstance(hybrid_ends, list)
        assert mhc_ends == hybrid_ends == list(expected)
        for managers in (mhc_managers, [context.manager for context in contexts], hybrid_managers):
            _assert_manager_partition(managers, expected)
            assert {id(manager) for manager in prior_managers}.isdisjoint(
                id(manager) for manager in managers
            )
            prior_managers.extend(managers)


def test_hybrid_caches_only_immutable_boundaries(monkeypatch):
    stack = _make_hybrid_stack(5, 2)
    compute = Mock(wraps=stack._compute_mhc_block_end_plan)
    monkeypatch.setattr(stack, "_compute_mhc_block_end_plan", compute)

    first_managers, first_ends = HybridStack._build_mhc_recompute_layer_plan(stack, True)
    cached = stack._mhc_block_end_plan
    assert isinstance(cached, tuple)
    assert cached == (False, True, False, True, True)
    assert isinstance(first_ends, list)
    assert first_ends == list(cached)
    _assert_manager_partition(first_managers, cached)

    first_ends[:] = [False] * 5
    first_managers[0].checkpoints.append(object())
    second_managers, second_ends = HybridStack._build_mhc_recompute_layer_plan(stack, True)

    compute.assert_called_once_with()
    assert stack._mhc_block_end_plan is cached
    assert second_ends == list(cached)
    assert second_ends is not first_ends
    _assert_manager_partition(second_managers, cached)
    assert {id(manager) for manager in first_managers}.isdisjoint(
        id(manager) for manager in second_managers
    )
    assert all(manager.checkpoints == [] for manager in second_managers)

    assert HybridStack._build_mhc_recompute_layer_plan(stack, False) == ([None] * 5, [False] * 5)
    assert stack._mhc_block_end_plan is cached
    compute.assert_called_once_with()


@pytest.mark.parametrize("num_layers,enabled", [(3, False), (0, False), (0, True)])
def test_disabled_or_empty_mhc_adapters_do_not_allocate(monkeypatch, num_layers, enabled):
    constructor = Mock(wraps=CheckpointWithoutOutputManager)
    monkeypatch.setattr(residual_recompute_plan, "CheckpointWithoutOutputManager", constructor)
    stack = _make_hybrid_stack(num_layers, 0)
    compute = Mock(wraps=stack._compute_mhc_block_end_plan)
    monkeypatch.setattr(stack, "_compute_mhc_block_end_plan", compute)

    for managers, block_ends in (
        build_mhc_recompute_layer_plan(num_layers, 0, enabled),
        HybridStack._build_mhc_recompute_layer_plan(stack, enabled),
    ):
        assert isinstance(managers, list)
        assert isinstance(block_ends, list)
        assert managers == [None] * num_layers
        assert block_ends == [False] * num_layers

    assert stack._mhc_block_end_plan is None
    compute.assert_not_called()
    constructor.assert_not_called()


@pytest.mark.parametrize(
    "finalize",
    [
        residual_recompute_plan.finalize_recompute_block,
        finalize_mhc_recompute_layer,
        HybridStack._finalize_mhc_recompute_layer,
    ],
    ids=["shared", "mhc", "hybrid"],
)
def test_finalization_dispatches_only_at_boundaries(finalize):
    hidden_states = torch.ones(2, 3, requires_grad=True)
    manager = Mock(spec=CheckpointWithoutOutputManager)
    dispatch = manager.discard_all_outputs_and_register_unified_recompute

    assert finalize(None, hidden_states, False) is None
    assert finalize(None, hidden_states, True) is None
    assert finalize(manager, hidden_states, False) is None
    dispatch.assert_not_called()

    assert finalize(manager, hidden_states, True) is None
    dispatch.assert_called_once_with(hidden_states)
    assert dispatch.call_args.args[0] is hidden_states


def test_residual_context_finalizes_each_block_with_its_own_boundary_tensor(monkeypatch):
    contexts = build_residual_stream_recompute_plan(5, 2)
    hidden_states = [torch.full((2, 3), float(index), requires_grad=True) for index in range(5)]
    dispatches = {}
    for context in contexts:
        if context.manager not in dispatches:
            dispatch = Mock()
            monkeypatch.setattr(
                context.manager, "discard_all_outputs_and_register_unified_recompute", dispatch
            )
            dispatches[context.manager] = dispatch

    for context, tensor in zip(contexts, hidden_states, strict=True):
        assert context.finalize(tensor) is None
        dispatch = dispatches[context.manager]
        if context.is_block_end:
            dispatch.assert_called_once_with(tensor)
            assert dispatch.call_args.args[0] is tensor
        else:
            dispatch.assert_not_called()

    assert sum(dispatch.call_count for dispatch in dispatches.values()) == 3
