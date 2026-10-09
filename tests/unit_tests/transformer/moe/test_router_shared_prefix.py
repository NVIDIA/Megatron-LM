# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Router hooks used by shared-prefix execution."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core.transformer.moe.router import InferenceTopKRouter, TopKRouter


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
