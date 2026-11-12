"""Replay authoritative-route conversion without requiring NCCL EP."""

import torch

from megatron.core.transformer.moe.megakernel.te import routing_map_to_indices
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded


def test_te_route_conversion_replay():
    seeded()
    probs = torch.randn(128, 16, device="cuda", requires_grad=True)
    routes = torch.zeros_like(probs, dtype=torch.bool)
    routes[1:, ::4] = True
    assert_replays_bit_exact(
        routing_map_to_indices,
        (probs, routes, 4),
        grad_outputs={"out[0]": torch.randn(128, 4, device="cuda")},
        replays=3,
        contention=True,
        what="TE MoE authoritative routing and probability gradients",
    )
