# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextvars import ContextVar, copy_context

import pytest
import torch

from megatron.core.models.hybrid.shared_prefix_layout import SharedPrefixLayout
from megatron.core.ssm.mamba_branch_layout import merge_mamba_branches, pack_mamba_branches
from megatron.core.tensor_observation import capture_tensor_observations, observe_tensor
from megatron.core.tensor_parallel.random import _run_recompute_with_observation_suspended
from megatron.core.transformer.moe.moe_utils import router_gating_linear, router_gating_token_blocks
from megatron.core.transformer.moe.router import _expert_bias_token_counts


@pytest.mark.parametrize("dense_indices", [False, True])
@pytest.mark.parametrize("padding", [False, True])
def test_logical_expert_counts_match_expanded_rows(dense_indices, padding):
    routes = torch.tensor([[0, 2], [1, 3], [2, 3], [0, -1]])
    multiplicities = torch.tensor([3, 1, 2, 0])
    padding_mask = torch.tensor([False, True, False, False]) if padding else None
    expected = torch.zeros(4, dtype=torch.long)
    for row, count in enumerate(multiplicities.tolist()):
        if padding and padding_mask[row]:
            continue
        for expert in routes[row].tolist():
            if expert >= 0:
                expected[expert] += count
    route_map = routes
    if not dense_indices:
        route_map = torch.zeros(4, 4, dtype=torch.bool)
        for row, experts in enumerate(routes.tolist()):
            for expert in experts:
                if expert >= 0:
                    route_map[row, expert] = True
    actual = _expert_bias_token_counts(route_map, padding_mask, multiplicities, num_experts=4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_invalid_expert_count_metadata_rejected():
    routes = torch.tensor([[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="one value per routed token"):
        _expert_bias_token_counts(routes, token_multiplicities=torch.ones(3), num_experts=3)
    with pytest.raises(ValueError, match="num_experts"):
        _expert_bias_token_counts(routes, token_multiplicities=torch.ones(2))
    with pytest.raises(ValueError, match="padding mask"):
        _expert_bias_token_counts(routes, padding_mask=torch.zeros(3, dtype=torch.bool))


@pytest.mark.parametrize(
    "requires",
    [(True, False, False), (False, True, False), (False, False, True), (True, True, True)],
)
@pytest.mark.parametrize("block_size", [None, 3])
def test_router_frozen_parameters_preserve_requested_gradients(requires, block_size):
    generator = torch.Generator().manual_seed(901)
    values = [
        torch.randn(shape, generator=generator, dtype=torch.float64)
        for shape in [(7, 1, 5), (4, 5), (4,)]
    ]
    actual_inputs = [value.clone().requires_grad_(need) for value, need in zip(values, requires)]
    reference_inputs = [value.clone().requires_grad_(need) for value, need in zip(values, requires)]
    if block_size is None:
        actual = router_gating_linear(*actual_inputs, torch.float64)
    else:
        with router_gating_token_blocks(block_size):
            actual = router_gating_linear(*actual_inputs, torch.float64)
    reference = torch.nn.functional.linear(*reference_inputs)
    cotangent = torch.randn(actual.shape, generator=generator, dtype=torch.float64)
    actual.backward(cotangent)
    reference.backward(cotangent)
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
    for value, reference_value, need in zip(actual_inputs, reference_inputs, requires):
        if need:
            torch.testing.assert_close(value.grad, reference_value.grad, rtol=1e-12, atol=1e-12)
        else:
            assert value.grad is None


@pytest.mark.parametrize("tail", [0, 2, 3])
def test_mamba_branch_layout_gradients_match_slice_reference(tail):
    lengths = (2, 0, 5)
    values = torch.randn(10, 1, 2, dtype=torch.float64, requires_grad=True)
    reference = values.detach().clone().requires_grad_()
    actual = pack_mamba_branches(values, prefix_len=3, tail_len=tail, completion_lens=lengths)
    branches = reference.new_zeros(tail + 5, 3, 2)
    offset = 3
    for branch, length in enumerate(lengths):
        branches[:tail, branch] = reference[3 - tail : 3, 0]
        branches[tail : tail + length, branch] = reference[offset : offset + length, 0]
        offset += length
    cotangent = torch.randn_like(actual)
    actual.backward(cotangent)
    branches.backward(cotangent)
    torch.testing.assert_close(actual, branches, rtol=0, atol=0)
    torch.testing.assert_close(values.grad, reference.grad, rtol=0, atol=0)
    head = torch.randn(3 - tail, 1, 2, dtype=torch.float64, requires_grad=True)
    branch_input = actual.detach().requires_grad_()
    assert torch.autograd.gradcheck(
        lambda h, b: merge_mamba_branches(h, b, tail_len=tail, completion_lens=lengths),
        (head, branch_input),
    )


def test_layout_prompt_multiplicities_equal_dense_gather_counts():
    layout = SharedPrefixLayout(3, (2, 5, 1))
    indices = torch.cat(layout.dense_branch_indices("cpu"))
    expected = torch.bincount(indices, minlength=layout.total_len).float()
    torch.testing.assert_close(
        layout.padded_token_multiplicities(layout.total_len, "cpu"), expected, rtol=0, atol=0
    )
    assert layout.position_ids("cpu").tolist() == [0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 3]


def test_recompute_restores_context_without_duplicate_observations():
    observed = []
    shared_scope = ContextVar("test_shared_scope", default=None)
    token = shared_scope.set("shared-forward")
    with capture_tensor_observations(lambda *args: observed.append(args), frozenset({"test"})):
        saved_context = copy_context()
    shared_scope.reset(token)

    def recompute():
        observe_tensor(None, "test", "test", torch.ones(1))
        return shared_scope.get()

    value = saved_context.copy().run(_run_recompute_with_observation_suspended, recompute)
    assert value == "shared-forward"
    assert observed == []
    assert shared_scope.get() is None
