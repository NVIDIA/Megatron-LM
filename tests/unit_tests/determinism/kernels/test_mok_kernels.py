# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay the MCore routing adapter without requiring the optional MOK package."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe import moe_env_flags
from megatron.core.transformer.moe.megakernel.mok.route_adapter import (
    RoutingMapToIndices,
    routing_map_to_mok_inputs,
)
from tests.unit_tests.determinism.kernels.harness import (
    assert_replays_bit_exact,
    bytes_equal,
    run_once,
    seeded,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_experts,topk", [(31, 6), (1024, 8)])
def test_routing_map_forward_and_backward_replay(dtype, num_experts, topk):
    seeded()
    num_tokens = 4096
    tokens = torch.arange(num_tokens, device="cuda")
    slots = torch.arange(topk, device="cuda")
    route_counts = tokens % (topk + 1)
    # Distinct, nonconsecutive experts; include empty and partially routed rows.
    expert_indices = (tokens[:, None] + slots[None, :] * 3) % num_experts
    valid_routes = slots[None, :] < route_counts[:, None]
    routing_map = torch.zeros((num_tokens, num_experts), dtype=torch.bool, device="cuda")
    routing_map.scatter_(1, expert_indices, valid_routes)
    probs = torch.randn((num_tokens, num_experts), device="cuda", dtype=dtype, requires_grad=True)
    upstream = torch.randn((num_tokens, topk), device="cuda", dtype=torch.float32)

    outputs, gradients = assert_replays_bit_exact(
        routing_map_to_mok_inputs,
        (probs, routing_map, topk),
        grad_outputs={"out[0]": upstream},
        replays=3,
        contention=True,
        what=f"MOK routing map[{dtype}, experts={num_experts}, topk={topk}]",
    )

    expected_indices = expert_indices.masked_fill(~valid_routes, num_experts).sort(dim=1).values
    expected_valid = expected_indices < num_experts
    safe_indices = expected_indices.clamp_max(num_experts - 1)
    expected_weights = probs.detach().float().gather(1, safe_indices)
    expected_weights.masked_fill_(~expected_valid, 0.0)
    expected_grad = torch.zeros_like(probs)
    rows = tokens[:, None].expand_as(expected_indices)
    expected_grad[rows[expected_valid], expected_indices[expected_valid]] = upstream[
        expected_valid
    ].to(dtype)

    torch.testing.assert_close(
        outputs["out[1]"], expected_indices.masked_fill(~expected_valid, -1).int(), rtol=0, atol=0
    )
    torch.testing.assert_close(outputs["out[0]"], expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(gradients["in[0]"], expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_routing_map_without_index_gradient_materialization_is_bit_exact(monkeypatch, dtype):
    """Not materializing the zero gradient of the non-differentiable indices leaves outputs and
    the probability gradient bit-identical; an absent upstream gradient gives the zeros that a
    materialized zero gradient would scatter."""
    seeded()
    num_tokens, num_experts, topk = 4096, 64, 8
    scores = torch.rand(num_tokens, num_experts, device="cuda")
    routing_map = torch.zeros((num_tokens, num_experts), dtype=torch.bool, device="cuda")
    routing_map.scatter_(1, scores.topk(topk, dim=1).indices, True)
    probs = torch.randn((num_tokens, num_experts), device="cuda", dtype=dtype, requires_grad=True)
    upstream = {"out[0]": torch.randn((num_tokens, topk), device="cuda", dtype=torch.float32)}

    def route(p):
        return routing_map_to_mok_inputs(p, routing_map, topk)

    monkeypatch.setattr(moe_env_flags, "MOK_ROUTE_NO_MATERIALIZE_GRADS", False)
    reference = run_once(route, (probs,), upstream)
    monkeypatch.setattr(moe_env_flags, "MOK_ROUTE_NO_MATERIALIZE_GRADS", True)
    outputs, gradients = assert_replays_bit_exact(
        route,
        (probs,),
        grad_outputs=upstream,
        replays=3,
        contention=True,
        what=f"MOK routing map without index-gradient materialization[{dtype}]",
    )
    for want, got in zip(reference, (outputs, gradients)):
        assert want.keys() == got.keys()
        for name in want:
            assert bytes_equal(got[name], want[name]), name

    ctx = SimpleNamespace(
        saved_tensors=(outputs["out[1]"],), num_experts=num_experts, probs_dtype=dtype
    )
    absent = RoutingMapToIndices.backward(ctx, None, None)[0]
    materialized = RoutingMapToIndices.backward(
        ctx, torch.zeros((num_tokens, topk), device="cuda"), None
    )[0]
    assert bytes_equal(absent, materialized) and not absent.any()
