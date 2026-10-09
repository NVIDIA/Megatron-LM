# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextvars import ContextVar, copy_context
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core.models.hybrid.shared_prefix_layout import SharedPrefixLayout
from megatron.core.ssm.mamba_branch_layout import merge_mamba_branches, pack_mamba_branches
from megatron.core.tensor_observation import capture_tensor_observations, observe_tensor
from megatron.core.tensor_parallel.random import _run_recompute_with_observation_suspended
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import router_gating_linear, router_gating_token_blocks
from megatron.core.transformer.moe.router import TopKRouter, _expert_bias_token_counts


@pytest.mark.parametrize("shared_prefix", [False, True])
def test_moe_route_preserves_packed_sequence_metadata_and_logical_counts(shared_prefix):
    hidden_states = torch.randn(3, 2, 4)
    padding_mask = torch.tensor([[False, False, True], [False, True, True]])
    input_ids = torch.arange(6).reshape(2, 3)
    packed_seq_params = object()
    multiplicities = torch.tensor([3, 1, 0, 2, 0, 0]) if shared_prefix else None
    captured = {}
    expected = (torch.ones(6, 2), torch.ones(6, 4, dtype=torch.bool))

    class RecordingRouter(torch.nn.Module):
        def forward(self, hidden, mask, input_ids=None, packed_seq_params=None, **kwargs):
            captured.update(
                hidden=hidden,
                mask=mask,
                input_ids=input_ids,
                packed_seq_params=packed_seq_params,
                kwargs=kwargs,
            )
            return expected

    layer = SimpleNamespace(
        router=RecordingRouter(), config=SimpleNamespace(cuda_graph_impl="none")
    )
    actual = MoELayer.route(
        layer,
        hidden_states,
        padding_mask,
        input_ids,
        packed_seq_params,
        token_multiplicities=multiplicities,
    )
    assert actual[0] is expected[0]
    assert actual[1] is expected[1]
    assert captured["hidden"] is hidden_states
    torch.testing.assert_close(captured["mask"], padding_mask.T)
    assert captured["input_ids"] is input_ids
    assert captured["packed_seq_params"] is packed_seq_params
    if shared_prefix:
        assert captured["kwargs"]["token_multiplicities"] is multiplicities
    else:
        assert captured["kwargs"] == {}


def test_moe_forward_passes_both_routing_metadata_sources():
    hidden_states = torch.randn(3, 1, 4)
    padding_mask = torch.zeros(1, 3, dtype=torch.bool)
    input_ids = torch.arange(3).unsqueeze(0)
    packed_seq_params = object()
    multiplicities = torch.tensor([2, 1, 1])
    probs, routes = torch.ones(3, 2), torch.ones(3, 4, dtype=torch.bool)
    layer = SimpleNamespace(
        training=False,
        select_token_dispatcher=Mock(),
        _shared_prefix_token_multiplicities=multiplicities,
        fwd_execution_map={"route": True},
        shared_experts_compute=Mock(return_value=None),
        route=Mock(return_value=(probs, routes)),
        preprocess=Mock(return_value=(hidden_states, probs)),
        moe_layer_recompute=False,
    )
    MoELayer.forward(
        layer,
        hidden_states,
        intermediate_tensors={},
        padding_mask=padding_mask,
        input_ids=input_ids,
        packed_seq_params=packed_seq_params,
    )
    layer.route.assert_called_once_with(
        hidden_states,
        padding_mask,
        input_ids=input_ids,
        packed_seq_params=packed_seq_params,
        token_multiplicities=multiplicities,
    )


def test_router_forward_preserves_upstream_positional_packed_sequence_argument():
    hidden_states = torch.randn(3, 1, 4)
    padding_mask = torch.zeros(3, 1, dtype=torch.bool)
    input_ids = torch.arange(3).unsqueeze(0)
    packed_seq_params = object()
    multiplicities = torch.tensor([2, 1, 1])
    expected = (torch.ones(3, 2), torch.ones(3, 4, dtype=torch.bool))
    router = SimpleNamespace(
        _maintain_float32_expert_bias=Mock(),
        apply_input_jitter=lambda value: value,
        gating=lambda value: value,
        config=SimpleNamespace(moe_router_force_load_balancing=False, moe_router_force_biased=None),
        routing=Mock(return_value=expected),
    )
    with patch("megatron.core.transformer.moe.router.is_observing_tensor", return_value=False):
        actual = TopKRouter.forward(
            router,
            hidden_states,
            padding_mask,
            input_ids,
            packed_seq_params,
            token_multiplicities=multiplicities,
        )
    assert actual[0] is expected[0]
    assert actual[1] is expected[1]
    router.routing.assert_called_once_with(
        hidden_states,
        padding_mask=padding_mask,
        input_ids=input_ids,
        packed_seq_params=packed_seq_params,
        token_multiplicities=multiplicities,
    )


def test_router_routing_preserves_packed_aux_losses_and_logical_expert_counts():
    logits = torch.randn(3, 1, 4)
    padding_mask = torch.zeros(3, 1, dtype=torch.bool)
    packed_seq_params = object()
    multiplicities = torch.tensor([2, 1, 1])
    probs, routes = torch.ones(3, 2), torch.ones(3, 4, dtype=torch.bool)
    scores = logits.view(3, 4).softmax(dim=-1)
    router = SimpleNamespace(
        training=True,
        config=SimpleNamespace(
            num_moe_experts=4,
            moe_num_hash_layers=0,
            moe_expert_capacity_factor=None,
            moe_router_pre_softmax=False,
            moe_router_num_groups=None,
            moe_router_group_topk=None,
            moe_router_topk_scaling_factor=None,
            moe_router_fusion=False,
            moe_router_aux_loss_fusion=False,
        ),
        is_hash_layer=False,
        routing_type="aux_loss",
        topk=2,
        score_function="softmax",
        expert_bias=None,
        router_replay=None,
        apply_z_loss=lambda value, **kwargs: value,
        _dense_route_indices_dtype=lambda: None,
        is_aux_loss_enabled=lambda: True,
        _apply_aux_loss=Mock(return_value=probs),
        _apply_seq_aux_loss=Mock(return_value=probs),
        _apply_global_aux_loss=Mock(return_value=probs),
        _apply_expert_bias=Mock(),
    )
    with (
        torch.enable_grad(),
        patch("megatron.core.transformer.moe.router.is_observing_tensor", return_value=False),
        patch(
            "megatron.core.transformer.moe.router.topk_routing_with_score_function",
            return_value=(probs, routes),
        ),
        patch(
            "megatron.core.transformer.moe.router.compute_routing_scores_for_aux_loss",
            return_value=(routes, scores),
        ),
    ):
        TopKRouter.routing(
            router,
            logits,
            padding_mask,
            None,
            packed_seq_params,
            token_multiplicities=multiplicities,
        )
    assert router._apply_aux_loss.call_args.kwargs["packed_seq_params"] is packed_seq_params
    assert router._apply_seq_aux_loss.call_args.kwargs["packed_seq_params"] is packed_seq_params
    torch.testing.assert_close(
        router._apply_expert_bias.call_args.kwargs["padding_mask"], padding_mask.reshape(-1)
    )
    assert router._apply_expert_bias.call_args.kwargs["token_multiplicities"] is multiplicities


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
