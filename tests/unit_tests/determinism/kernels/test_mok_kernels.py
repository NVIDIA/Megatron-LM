# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay the MCore routing adapter without requiring the optional MOK package."""

import pytest
import torch

from megatron.core.transformer.moe.megakernel.mok.route_adapter import routing_map_to_mok_inputs
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

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
