# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Masked replay retains exact expert choices, values and gathered-score gradients."""

import pytest
import torch

from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_masked_router_replay_is_bit_exact(device):
    """Compare repeated executions against independently specified routes and gradients."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    source = torch.tensor([[0.1, 0.8, 0.4, 0.2], [0.2, 0.4, 0.7, 0.1], [0.9, 0.2, 0.6, 0.5]])
    expected_indices = torch.tensor([[0, 2], [2, 1], [2, 3]])
    expected_values = torch.stack(
        [source[row, indices] for row, indices in enumerate(expected_indices)]
    )
    expected_grad = torch.tensor(
        [[0.25, 0.0, 0.5, 0.0], [0.0, 0.5, 0.25, 0.0], [0.0, 0.0, 0.25, 0.5]]
    )
    original_instances = list(RouterReplay.global_router_replay_instances)

    def native(scores, topk, **kwargs):
        return scores.topk(topk, dim=-1)

    try:
        for _ in range(8):
            replay = RouterReplay()
            scores = source.to(device).clone().requires_grad_()
            replay.set_target_indices(
                torch.tensor([[0, 2], [0, 3], [2, 3]], device=device),
                torch.tensor([True, False, True], device=device),
            )
            replay.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
            values, indices = replay.get_replay_topk(scores, 2, default_compute_topk=native)
            (values * torch.tensor([0.25, 0.5], device=device)).sum().backward()
            assert torch.equal(indices.cpu(), expected_indices)
            assert torch.equal(values.cpu(), expected_values)
            assert torch.equal(scores.grad.cpu(), expected_grad)
            replay.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
            recomputed_values, recomputed_indices = replay.get_replay_topk(
                -scores.detach(), 2, default_compute_topk=native
            )
            assert torch.equal(recomputed_indices.cpu(), expected_indices)
            assert torch.equal(recomputed_values.cpu(), -expected_values)
    finally:
        RouterReplay.global_router_replay_instances[:] = original_instances
