# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Router hooks used by shared-prefix execution."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import (
    InferenceTopKRouter,
    TopKRouter,
    _expert_bias_token_counts,
)
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _router_with_routing(routing):
    return SimpleNamespace(
        _maintain_float32_expert_bias=Mock(),
        apply_input_jitter=lambda value: value,
        gating=lambda value: value,
        config=SimpleNamespace(moe_router_force_load_balancing=False, moe_router_force_biased=None),
        routing=routing,
    )


def test_router_forward_keeps_upstream_routing_signature_without_multiplicities():
    hidden_states = torch.randn(3, 1, 4)
    expected = (torch.ones(3, 2), torch.ones(3, 4, dtype=torch.bool))
    calls = []

    def routing(logits, padding_mask=None, input_ids=None, packed_seq_params=None):
        calls.append((logits, padding_mask, input_ids, packed_seq_params))
        return expected

    router = _router_with_routing(routing)
    with patch("megatron.core.transformer.moe.router.is_observing_tensor", return_value=False):
        actual = TopKRouter.forward(router, hidden_states)
    assert actual == expected
    assert len(calls) == 1 and calls[0][0] is hidden_states


def test_router_forward_passes_multiplicities_only_when_set():
    hidden_states = torch.randn(3, 1, 4)
    multiplicities = torch.tensor([2, 1, 1])
    router = _router_with_routing(Mock(return_value=(None, None)))
    with patch("megatron.core.transformer.moe.router.is_observing_tensor", return_value=False):
        TopKRouter.forward(router, hidden_states)
        TopKRouter.forward(router, hidden_states, token_multiplicities=multiplicities)
    first, second = router.routing.call_args_list
    assert "token_multiplicities" not in first.kwargs
    assert second.kwargs["token_multiplicities"] is multiplicities


def test_inference_router_fallback_forwards_multiplicities():
    router = InferenceTopKRouter.__new__(InferenceTopKRouter)
    hidden_states = torch.randn(3, 1, 4)
    multiplicities = torch.tensor([2, 1, 1])
    parent_forward = Mock(return_value=("probs", "routes"))
    with (
        patch.object(TopKRouter, "forward", parent_forward),
        patch("megatron.core.transformer.moe.router.InferenceMode.is_active", return_value=False),
    ):
        actual = InferenceTopKRouter.forward(
            router, hidden_states, token_multiplicities=multiplicities
        )
    assert actual == ("probs", "routes")
    assert parent_forward.call_args.kwargs["token_multiplicities"] is multiplicities


@pytest.mark.parametrize("dense_indices", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_expert_bias_counts_are_exact_beyond_float32_integer_range(dense_indices, deterministic):
    # Shared-prefix layouts carry float32 multiplicities; counts must still be exact int64.
    multiplicities = torch.tensor([float(1 << 24), 1.0], dtype=torch.float32)
    routes = torch.tensor([[0, 1], [0, -1]])
    if not dense_indices:
        routes = torch.tensor([[True, True], [True, False]])
    router = SimpleNamespace(
        enable_expert_bias=True,
        local_tokens_per_expert=torch.zeros(2, dtype=torch.int64),
        config=SimpleNamespace(num_moe_experts=2),
    )
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(deterministic)
    try:
        with torch.enable_grad():
            TopKRouter._apply_expert_bias(router, routes, token_multiplicities=multiplicities)
    finally:
        torch.use_deterministic_algorithms(prev)
    expected = torch.tensor([(1 << 24) + 1, 1 << 24], dtype=torch.int64)
    assert torch.equal(router.local_tokens_per_expert, expected)


@pytest.mark.parametrize("dense_indices", [False, True])
@pytest.mark.parametrize("padding", [False, True])
def test_expert_bias_counts_accumulate_in_place(dense_indices, padding):
    routes = torch.tensor([[0, 2], [1, 3], [2, 3], [0, -1]])
    bool_routes = torch.zeros(4, 4, dtype=torch.bool)
    for row, experts in enumerate(routes.tolist()):
        for expert in experts:
            if expert >= 0:
                bool_routes[row, expert] = True
    padding_mask = torch.tensor([False, True, False, False]) if padding else None
    multiplicities = torch.tensor([3, 1, 2, 5])
    route_map = routes if dense_indices else bool_routes
    keep = torch.ones(4, dtype=torch.bool) if padding_mask is None else ~padding_mask

    counts = torch.full((4,), 7, dtype=torch.int64)
    returned = _expert_bias_token_counts(route_map, padding_mask, out=counts)
    assert returned is counts
    assert torch.equal(counts, 7 + (bool_routes & keep.unsqueeze(-1)).sum(dim=0))

    counts = torch.full((4,), 7, dtype=torch.int64)
    _expert_bias_token_counts(route_map, padding_mask, multiplicities, out=counts)
    weights = (multiplicities * keep).unsqueeze(-1)
    assert torch.equal(counts, 7 + (bool_routes * weights).sum(dim=0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("with_padding", [False, True])
def test_topk_router_expert_bias_counts_logical_tokens(with_padding):
    Utils.initialize_model_parallel(1, 1)
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=8,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="none",
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_topk=2,
            add_bias_linear=False,
        )
        submodules = get_submodules(
            get_gpt_layer_local_submodules(num_experts=8, moe_grouped_gemm=False).mlp
        )
        router = MoELayer(config, submodules).router.cuda()
        hidden_states = torch.randn(6, 2, 12, device="cuda")
        padding_mask = None
        if with_padding:
            padding_mask = torch.zeros(6, 2, dtype=torch.bool, device="cuda")
            padding_mask[4:, 1] = True
        # Float32, as produced by SharedPrefixLayout.padded_token_multiplicities.
        multiplicities = torch.tensor(
            [4, 1, 1, 3, 1, 1, 2, 1, 1, 1, 1, 1], dtype=torch.float32, device="cuda"
        )
        router.local_tokens_per_expert.zero_()
        with torch.enable_grad():
            _, routing_map = router(
                hidden_states, padding_mask, token_multiplicities=multiplicities
            )
        keep = torch.ones(12, dtype=torch.bool, device="cuda")
        if padding_mask is not None:
            keep = ~padding_mask.reshape(-1)
        weights = (multiplicities.long() * keep).unsqueeze(-1)
        expected = (routing_map.bool() * weights).sum(dim=0)
        assert router.local_tokens_per_expert.dtype == torch.int64
        assert torch.equal(router.local_tokens_per_expert, expected)
    finally:
        Utils.destroy_model_parallel()
