# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bitwise tests: fused batch-invariant MoE unpermute against the reference loop."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.transformer.moe import batch_invariant
from megatron.core.transformer.moe.moe_utils import permute

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


def _adversarial(shape, dtype, seed):
    """Values over a wide exponent range plus signed zeros."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    mantissa = torch.randn(shape, device="cuda", generator=gen)
    exponent = torch.randint(-40, 40, shape, device="cuda", generator=gen).float()
    values = (mantissa * torch.exp2(exponent)).to(dtype)
    values.view(-1)[::7] = -0.0
    values.view(-1)[3::11] = 0.0
    return values


def _routed(num_tokens, hidden, num_experts, topk, dtype, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand(num_tokens, num_experts, device="cuda", generator=gen)
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool, device="cuda")
    routing_map.scatter_(1, scores.topk(topk, dim=1).indices, True)
    tokens = torch.zeros(num_tokens, hidden, device="cuda", dtype=dtype)
    _, _, _, inverse_map, _ = permute(
        tokens,
        routing_map,
        num_out_tokens=num_tokens * topk,
        return_batch_invariant_inverse_map=True,
    )
    rows = _adversarial((num_tokens * topk, hidden), dtype, seed + 1)
    return rows, inverse_map


@pytest.fixture
def ep_world_size():
    def _set(size):
        parallel_state.set_expert_model_parallel_world_size(size)

    yield _set
    parallel_state.set_expert_model_parallel_world_size(None)


@pytest.mark.parametrize("collective", ["ordered", "multimem"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "num_tokens,hidden,num_experts,topk,ep_size",
    [
        (1, 8, 4, 1, 1),
        (37, 100, 8, 2, 2),
        (300, 520, 16, 4, 4),
        (1000, 2688, 128, 6, 8),
        (513, 1030, 64, 8, 8),
    ],
)
def test_fused_unpermute_matches_loop_bitwise(
    monkeypatch, ep_world_size, collective, dtype, num_tokens, hidden, num_experts, topk, ep_size
):
    monkeypatch.setattr(batch_invariant, "get_batch_invariant_collective", lambda: collective)
    ep_world_size(ep_size)
    rows, inverse_map = _routed(num_tokens, hidden, num_experts, topk, dtype, seed=num_tokens)
    restore_shape = torch.Size((num_tokens, hidden))
    grad = _adversarial((num_tokens, hidden), dtype, seed=num_tokens + 2)

    fused_rows = rows.clone().requires_grad_()
    fused = batch_invariant.unpermute(
        fused_rows, restore_shape, probs=None, num_experts=num_experts, inverse_map=inverse_map
    )
    fused.backward(grad)

    loop_rows = rows.clone().requires_grad_()
    loop = batch_invariant._unpermute_loop(
        loop_rows, restore_shape, None, num_experts, inverse_map, ep_size
    )
    loop.backward(grad)

    assert fused.dtype == loop.dtype and fused.shape == loop.shape
    assert torch.equal(_bits(fused), _bits(loop))
    assert torch.equal(_bits(fused_rows.grad), _bits(loop_rows.grad))


def test_fused_unpermute_rank_tree_rounding(ep_world_size):
    """Rank partials round to fp32 before the cross-rank sum, so 1e20 + 1 - 1e20 + 1 is 0."""
    ep_world_size(2)
    rows = torch.tensor([[1e20], [1.0], [-1e20], [1.0]], device="cuda").expand(4, 4).contiguous()
    inverse_map = torch.tensor([[[0, 1, 2, 3]], [[0, 1, 2, 3]]], device="cuda")
    out = batch_invariant.unpermute(
        rows, torch.Size((1, 4)), probs=None, num_experts=4, inverse_map=inverse_map
    )
    assert torch.equal(out, torch.zeros(1, 4, device="cuda"))


def test_unpermute_with_probs_uses_loop(ep_world_size):
    ep_world_size(2)
    num_tokens, hidden, num_experts, topk = 64, 32, 8, 2
    rows, inverse_map = _routed(num_tokens, hidden, num_experts, topk, torch.bfloat16, seed=5)
    probs = torch.rand(num_tokens, num_experts, device="cuda")
    restore_shape = torch.Size((num_tokens, hidden))
    out = batch_invariant.unpermute(
        rows, restore_shape, probs=probs, num_experts=num_experts, inverse_map=inverse_map
    )
    ref = batch_invariant._unpermute_loop(rows, restore_shape, probs, num_experts, inverse_map, 2)
    assert torch.equal(_bits(out), _bits(ref))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_unpermute_backward_is_cuda_graph_capturable(ep_world_size, dtype):
    """The fused forward and backward capture into one CUDA graph and replay bitwise.

    The engine's AllToAll combine takes this ``probs=None`` path inside captured graphs, so
    no step may depend on the inverse map's contents (no boolean indexing). The map holds
    an invalid slot and leaves rows 3 and 5 unused; their gradients must stay zero. The
    upstream gradient is non-contiguous and carries signed zeros.
    """
    ep_world_size(2)
    num_experts, hidden = 4, 40
    # Tokens x top-2 slots; slot (1, 1) is invalid, rows 3 and 5 belong to no slot.
    inverse_rows = torch.tensor([[0, 4], [1, -1], [2, 6]], device="cuda")
    inverse_experts = torch.tensor([[0, 3], [1, -1], [2, 0]], device="cuda")
    inverse_map = torch.stack((inverse_rows, inverse_experts))
    restore_shape = torch.Size((3, hidden))
    rows = _adversarial((7, hidden), dtype, seed=11).requires_grad_()
    grad = _adversarial((hidden, 3), dtype, seed=12).t()  # non-contiguous
    assert not grad.is_contiguous()

    def step():
        rows.grad = None
        out = batch_invariant.unpermute(
            rows, restore_shape, probs=None, num_experts=num_experts, inverse_map=inverse_map
        )
        out.backward(grad)
        return out

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    rows.grad = None
    with torch.cuda.graph(graph):
        out = batch_invariant.unpermute(
            rows, restore_shape, probs=None, num_experts=num_experts, inverse_map=inverse_map
        )
        out.backward(grad)
    graph.replay()
    torch.cuda.synchronize()

    loop_rows = rows.detach().clone().requires_grad_()
    loop = batch_invariant._unpermute_loop(
        loop_rows, restore_shape, None, num_experts, inverse_map, 2
    )
    loop.backward(grad)
    assert torch.equal(_bits(out), _bits(loop))
    assert torch.equal(_bits(rows.grad), _bits(loop_rows.grad))
    assert torch.equal(_bits(rows.grad[[3, 5]]), _bits(torch.zeros_like(rows.grad[[3, 5]])))
