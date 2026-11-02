# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exact local MoE summation trees; these checks do not emulate NCCL collectives."""

import pytest
import torch

from megatron.core.transformer.moe import batch_invariant


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA replay requires a GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("collective", ["ordered", "multimem"])
@pytest.mark.parametrize("weighted", [False, True])
def test_unpermute_rounding_and_backward_replay(monkeypatch, dtype, collective, weighted):
    """Cancellation exposes both within-rank and cross-rank rounding points.

    The local helper simulates three contiguous expert shards without communication.
    Token 0 cancels within one shard; token 1 cancels across three rank partials;
    token 2 has only invalid routes. Every valid input row has unit loss gradient.
    """
    monkeypatch.setattr(
        batch_invariant.parallel_state, "get_expert_model_parallel_world_size", lambda: 3
    )
    monkeypatch.setattr(batch_invariant, "get_batch_invariant_collective", lambda: collective)
    values = torch.zeros(18, 4, dtype=dtype, device="cuda")
    values[0], values[1], values[2] = 2**24, 1, -(2**24)
    values[9], values[12], values[15] = 2**24, 1, -(2**24)
    values.requires_grad_(True)
    rows = torch.cat(
        [torch.arange(18, device="cuda").reshape(2, 9), torch.full((1, 9), -1, device="cuda")]
    )
    experts = torch.arange(9, device="cuda").expand(3, -1).clone()
    experts[2] = -1
    inverse_map = torch.stack((rows, experts))
    probs = torch.ones(3, 9, dtype=torch.float32, device="cuda") if weighted else None
    expected = torch.tensor(
        [[0 if weighted else 1] * 4, [1 if collective == "multimem" else 0] * 4, [0] * 4],
        dtype=dtype,
        device="cuda",
    )

    def forward_backward():
        output = batch_invariant.unpermute(
            values, torch.Size((3, 4)), probs=probs, num_experts=9, inverse_map=inverse_map
        )
        gradient = torch.autograd.grad(output.sum(), values)[0]
        return output, gradient

    expected_gradient = torch.ones_like(values)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            eager_output, eager_gradient = forward_backward()
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(eager_output, expected)
    assert torch.equal(eager_gradient, expected_gradient)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, gradient = forward_backward()
    for _ in range(3):
        graph.replay()
        assert torch.equal(output, expected)
        assert torch.equal(gradient, expected_gradient)
