# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Expert CPU streams must be independent without perturbing later dense weights."""

import pytest
import torch

from megatron.core.tensor_parallel.layers import _initialize_affine_weight_cpu
from megatron.core.tensor_parallel.random import expert_cpu_init_index, expert_cpu_init_scope


def _weight():
    weight = torch.empty(16, 16)
    _initialize_affine_weight_cpu(
        weight, 16, 16, 16, 0, torch.nn.init.normal_, rank=0, world_size=1
    )
    return weight


def _experts(indices):
    torch.manual_seed(1234)
    result = {}
    with expert_cpu_init_scope():
        for index in indices:
            with expert_cpu_init_index(index):
                result[index] = (_weight(), _weight())
    return result, torch.rand(16)


def test_successive_expert_weights_use_different_draws():
    experts, _ = _experts([0, 1])
    for fc1, fc2 in experts.values():
        assert not torch.equal(fc1, fc2), "FC1 and FC2 restart the same CPU RNG stream"


def test_expert_streams_are_layout_invariant():
    reference, reference_tail = _experts(range(8))
    for indices in (range(4), range(4, 8), range(6, 8)):
        actual, tail = _experts(indices)
        torch.testing.assert_close(tail, reference_tail, rtol=0, atol=0)
        for index, weights in actual.items():
            for actual_weight, expected_weight in zip(weights, reference[index]):
                torch.testing.assert_close(actual_weight, expected_weight, rtol=0, atol=0)
    assert not torch.equal(reference[0][0], reference[1][0])


def test_expert_rng_restored_after_exception():
    torch.manual_seed(1234)
    with expert_cpu_init_scope():
        shared_state = torch.get_rng_state().clone()
        with pytest.raises(RuntimeError, match="init failed"):
            with expert_cpu_init_index(0):

                def fail(weight):
                    weight.normal_()
                    raise RuntimeError("init failed")

                _initialize_affine_weight_cpu(
                    torch.empty(16, 16), 16, 16, 16, 0, fail, rank=0, world_size=1
                )
        assert torch.equal(torch.get_rng_state(), shared_state)
    assert torch.equal(torch.get_rng_state(), shared_state)
