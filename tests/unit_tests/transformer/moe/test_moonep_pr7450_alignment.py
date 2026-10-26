# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""PR #7450 (242c8dbc6) oracle for the scheduler's intentionally split planner.

The independent CPU reference below is taken from that PR's test_moe_kernels.py.
It does not read kernel intermediates to compute expected routes or placement.
"""

import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.transformer.moe.moonep_moe_scheduler import padded_rank_capacity
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.moonep_reference import (
    ReplicaPlannerWorkspace,
    plan_replica_routes,
)

pytestmark = pytest.mark.launch_on_gb200


@pytest.fixture(scope="module")
def distributed_world():
    if not torch.cuda.is_available():
        pytest.skip("MoonEP alignment tests require CUDA.")
    Utils.set_world_size(world_size=int(os.environ["WORLD_SIZE"]), rank=int(os.environ["RANK"]))
    Utils.initialize_distributed()
    return dist.group.WORLD


def _reference_plan(routes, num_experts, alignment=1):
    """CPU greedy placement and stable route assignment, using only semantic input routes."""
    ep_size, num_tokens, topk = routes.shape
    local_experts = num_experts // ep_size
    counts = torch.stack([torch.bincount(row.flatten(), minlength=num_experts) for row in routes])
    totals = counts.sum(0).tolist()
    blocks = [(count + alignment - 1) // alignment for count in totals]
    total_blocks = sum(blocks)
    targets = [total_blocks // ep_size + (rank < total_blocks % ep_size) for rank in range(ep_size)]
    loads = [
        sum(blocks[r * local_experts : (r + 1) * local_experts]) - targets[r]
        for r in range(ep_size)
    ]
    quotas = [[0] * ep_size for _ in range(ep_size)]
    while max(loads) > 0:
        sender = max(range(ep_size), key=lambda r: (loads[r], -r))
        receiver = min(range(ep_size), key=lambda r: (loads[r], r))
        # A receiver takes its entire deficit from one sender, even beyond that sender's
        # excess. This bounds the number of virtual slots by the sender's native experts.
        moved = -loads[receiver]
        quotas[sender][receiver] += moved
        loads[sender] -= moved
        loads[receiver] = 0
    allocation = [[0] * ep_size for _ in range(num_experts)]
    for expert, count in enumerate(blocks):
        allocation[expert][expert // local_experts] = count
    for sender, pending in enumerate(quotas):
        experts = range(sender * local_experts, (sender + 1) * local_experts)
        while any(pending):
            destination = max(range(ep_size), key=lambda r: (pending[r], -r))
            expert = max(experts, key=lambda e: (allocation[e][sender], -e))
            moved = min(pending[destination], allocation[expert][sender])
            assert moved > 0
            allocation[expert][sender] -= moved
            allocation[expert][destination] += moved
            pending[destination] -= moved
    allocation = [[count * alignment for count in row] for row in allocation]
    copies = []
    for destination in range(ep_size):
        remote = sorted(
            (
                e
                for e in range(num_experts)
                if e // local_experts != destination and allocation[e][destination]
            ),
            key=lambda e: (allocation[e][destination], e),
            reverse=True,
        )
        assert len(remote) <= local_experts
        copies.append(remote + [-1] * (local_experts - len(remote)))
    # Stable global order is source rank, token, top-k lane. Partition each expert's
    # positions directly by the CPU allocation, without reading kernel boundaries or slots.
    order = routes.flatten().argsort(stable=True)
    mapped = torch.empty(routes.numel(), dtype=torch.int16)
    cursor = 0
    for expert, destinations in enumerate(allocation):
        remaining = totals[expert]
        for destination, count in enumerate(destinations):
            count = min(count, remaining)
            remaining -= count
            if count:
                local = (
                    expert % local_experts
                    if destination == expert // local_experts
                    else (local_experts + copies[destination].index(expert))
                )
                mapped[order[cursor : cursor + count]] = destination * (2 * local_experts) + local
                cursor += count
    assert cursor == routes.numel()
    return (
        counts.int(),
        torch.tensor(allocation, dtype=torch.int32),
        torch.tensor(copies, dtype=torch.int32),
        mapped.view_as(routes),
    )


def _routes(ep, experts, tokens, topk, seed):
    g = torch.Generator().manual_seed(seed)
    # Distinct routes, including non-power-of-two top-k and hot-expert skew.
    scores = torch.rand(ep, tokens, experts, generator=g)
    scores[..., 0] += 0.75
    return scores.topk(topk, dim=-1).indices.sort(dim=-1).values.contiguous()


def _check_plan(routes, rank, alignment, workspace, exchange):
    counts, allocation, copies, mapped = _reference_plan(routes, workspace.num_experts, alignment)
    if not exchange:
        workspace.gathered_counts.copy_(counts)
    ids = routes[rank].cuda()
    probs = torch.linspace(0, 1, ids.numel(), device="cuda").view_as(ids).requires_grad_()
    plan, dense = plan_replica_routes(ids, probs, workspace, exchange=exchange, alignment=alignment)
    torch.testing.assert_close(workspace.field("allocation").cpu(), allocation, rtol=0, atol=0)
    torch.testing.assert_close(plan.experts_to_copy.cpu(), copies, rtol=0, atol=0)
    torch.testing.assert_close(plan.virtual_experts.cpu(), mapped[rank], rtol=0, atol=0)
    expected = torch.zeros_like(dense).scatter(1, mapped[rank].cuda().long(), probs.detach())
    torch.testing.assert_close(dense, expected, rtol=0, atol=0)
    dy = torch.arange(dense.numel(), device="cuda", dtype=torch.float32).view_as(dense)
    dense.backward(dy)
    torch.testing.assert_close(probs.grad, dy.gather(1, mapped[rank].cuda().long()), rtol=0, atol=0)
    ep = routes.shape[0]
    counts = torch.bincount(mapped.long().flatten(), minlength=2 * workspace.num_experts).reshape(
        ep, -1
    )
    padded = ((counts + alignment - 1) // alignment * alignment).sum(1)
    assert padded.max() <= padded_rank_capacity(ids.numel(), workspace.num_experts, ep, alignment)


@pytest.mark.parametrize("alignment", [1, 32, 128, 256])
@pytest.mark.parametrize(
    "ep,experts,tokens,topk", [(2, 4, 16, 1), (4, 16, 257, 3), (33, 66, 65, 2), (64, 128, 129, 2)]
)
def test_split_planner_matches_pr_oracle(distributed_world, alignment, ep, experts, tokens, topk):
    rank = ep - 1
    workspace = ReplicaPlannerWorkspace.local(
        experts, ep, torch.device("cuda", torch.cuda.current_device()), rank=rank
    )
    # Reuse scratch across changed routes to catch stale totals/slots.
    for seed in [11, 29]:
        _check_plan(_routes(ep, experts, tokens, topk, seed), rank, alignment, workspace, False)


@pytest.mark.parametrize("alignment", [1, 256])
def test_distributed_histogram_exchange(distributed_world, alignment):
    ep, rank = dist.get_world_size(), dist.get_rank()
    workspace = ReplicaPlannerWorkspace.allocate(
        num_experts=ep * 4,
        device=torch.device("cuda", torch.cuda.current_device()),
        group=distributed_world,
    )
    try:
        for tokens, seed in [(129, 7), (257, 8), (129, 9)]:
            _check_plan(_routes(ep, ep * 4, tokens, 3, seed), rank, alignment, workspace, True)
    finally:
        workspace.destroy()


def test_padding_changes_placement_and_reduces_work():
    routes = torch.tensor([[0] * 16, [0, 1] + [2] * 7 + [3] * 7]).reshape(2, 16, 1)
    real = _reference_plan(routes, 4, 1)[-1]
    aligned = _reference_plan(routes, 4, 256)[-1]

    def padded(ids):
        counts = torch.bincount(ids.long().flatten(), minlength=8).reshape(2, 4)
        return ((counts + 255) // 256 * 256).sum(1).tolist()

    assert padded(real) == [512, 768]
    assert padded(aligned) == [512, 512]


def test_split_planner_graph_replay(distributed_world):
    ep, experts, tokens, topk, alignment = 4, 16, 257, 3, 256
    workspace = ReplicaPlannerWorkspace.local(
        experts, ep, torch.device("cuda", torch.cuda.current_device())
    )
    ids = torch.empty(tokens, topk, device="cuda", dtype=torch.int64)
    probs = torch.full((tokens, topk), 1 / topk, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    routes = _routes(ep, experts, tokens, topk, 12)
    counts = _reference_plan(routes, experts, alignment)[0]
    ids.copy_(routes[0])
    workspace.gathered_counts.copy_(counts)
    with torch.cuda.stream(stream):
        plan_replica_routes(ids, probs, workspace, exchange=False, alignment=alignment)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        plan, dense = plan_replica_routes(
            ids, probs, workspace, exchange=False, alignment=alignment
        )
    for seed in [21, 38, 44]:
        routes = _routes(ep, experts, tokens, topk, seed)
        counts, _, copies, mapped = _reference_plan(routes, experts, alignment)
        ids.copy_(routes[0])
        workspace.gathered_counts.copy_(counts)
        graph.replay()
        torch.testing.assert_close(plan.virtual_experts.cpu(), mapped[0], rtol=0, atol=0)
        torch.testing.assert_close(plan.experts_to_copy.cpu(), copies, rtol=0, atol=0)
        expected = torch.zeros_like(dense).scatter(1, mapped[0].cuda().long(), probs)
        torch.testing.assert_close(dense, expected, rtol=0, atol=0)
