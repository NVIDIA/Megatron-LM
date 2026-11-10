# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Regression tests for MIMO optimizer state-dict traversal over nested chains."""

from megatron.core.models.mimo.optimizer import _iter_optimizer_sub_dicts
from megatron.core.optimizer.optimizer import ChainedOptimizer


class _Leaf:
    """Stands in for a non-chained optimizer."""


def _chain(*inner):
    chained = ChainedOptimizer.__new__(ChainedOptimizer)
    chained.chained_optimizers = list(inner)
    return chained


def _leaf_sd(tag):
    return {'optimizer': {'param_groups': [tag]}}


def test_nested_chain_yields_string_keyed_leaves():
    """A chain whose child is itself a chain must be descended to the leaves.

    ChainedOptimizer.sharded_state_dict() keys its output by integer index when the
    chain holds more than one optimizer. Stopping after one level hands the caller
    that integer-keyed dict in place of a leaf state dict, which makes the save path
    skip param_groups and the load path raise AttributeError on int.startswith.
    """
    inner = _chain(_Leaf(), _Leaf())
    outer = _chain(inner, _Leaf())
    module_sd = {0: {0: _leaf_sd('a'), 1: _leaf_sd('b')}, 1: _leaf_sd('c')}

    sub_dicts = [sub_sd for sub_sd, _ in _iter_optimizer_sub_dicts(module_sd, outer)]

    assert len(sub_dicts) == 3
    for sub_sd in sub_dicts:
        assert all(isinstance(key, str) for key in sub_sd), f"non-string keys: {list(sub_sd)}"
        assert 'param_groups' in sub_sd['optimizer']


def test_single_child_chain_delegates_to_the_child():
    """A one-element chain returns the child's state dict directly, not a wrapper."""
    leaf = _Leaf()
    module_sd = _leaf_sd('solo')

    pairs = list(_iter_optimizer_sub_dicts(module_sd, _chain(leaf)))

    assert len(pairs) == 1
    assert pairs[0][0] is module_sd
    assert pairs[0][1] is leaf


def test_plain_optimizer_is_yielded_unchanged():
    leaf = _Leaf()
    module_sd = _leaf_sd('plain')

    assert list(_iter_optimizer_sub_dicts(module_sd, leaf)) == [(module_sd, leaf)]
