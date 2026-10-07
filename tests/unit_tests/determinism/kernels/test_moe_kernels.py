# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the MoE kernels in ``megatron/core/transformer/moe/``.

Operator level: token permute / unpermute (torch and TE-fused, both branches of the
``torch.are_deterministic_algorithms_enabled()`` switch), chunk sorting, top-k routing (torch
and TE-fused), group-limited routing, the load-balancing aux loss (torch and TE-fused), and
the router gating GEMM, and virtual-expert planning against a CPU oracle in eager and CUDA
graphs. Module level: ``TopKRouter``, ``TEGroupedMLP`` / ``SequentialMLP`` on
deliberately uneven expert loads (including an empty expert), and a full ``MoELayer`` through
the all-gather and all-to-all dispatchers (plus the flex/DeepEP dispatcher from
``fused_a2a.py`` when the dependency and the GPUs are available).
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.activations import squared_relu
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe import moe_utils
from megatron.core.transformer.moe.experts import (
    SequentialMLP,
    TEGroupedMLP,
    _te_supports_scaled_tanh_srelu,
)
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import _VirtualExpertHybridEPManager
from megatron.core.transformer.moe.virtual_expert_load_balancer import (
    VirtualExpertLoadBalancer,
    plan_virtual_expert_routes,
)
from megatron.core.transformer.moe.virtual_expert_triton import VirtualExpertPlannerWorkspace
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.determinism.kernels.harness import (
    assert_module_replays_bit_exact,
    assert_replays_bit_exact,
    count_differing_replays,
    deterministic_algorithms,
    seeded,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HAVE_TE = moe_utils.HAVE_TE
HAVE_TE_PERMUTE = HAVE_TE and moe_utils.fused_permute is not None
HAVE_TE_ROUTER = HAVE_TE and is_te_min_version("2.7.0")
from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP

NUM_TOKENS, HIDDEN, NUM_EXPERTS, TOPK = 16384, 2048, 64, 8


# --- virtual-expert planning ----------------------------------------------------------------


@pytest.fixture
def virtual_expert_planner_group(ep_size):
    """Run the planner independently of model parallelism and HybridEP transport."""
    Utils.initialize_distributed()
    world_size = dist.get_world_size()
    if world_size % ep_size:
        pytest.skip(f"EP={ep_size} requires a world size divisible by {ep_size}")
    groups = (
        [
            dist.new_group(list(range(start, start + ep_size)))
            for start in range(0, world_size, ep_size)
        ]
        if ep_size != world_size
        else []
    )
    group = groups[dist.get_rank() // ep_size] if groups else dist.group.WORLD
    try:
        yield group
    finally:
        torch.cuda.synchronize()
        dist.barrier(device_ids=[torch.cuda.current_device()])
        if groups:
            dist.destroy_process_group(group)


def _check_equal(actual, expected, label, errors):
    """Keep comparisons collective-safe while checking every element, including canaries."""
    try:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    except AssertionError as exc:
        errors.append(f"{label}: {exc}")


def _report(errors, group):
    """Fail on every rank if any rank saw a mismatch."""
    gathered = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(gathered, errors, group=group)
    combined = [error for rank_errors in gathered for error in rank_errors]
    assert not combined, "\n".join(combined)


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


def _routes_for_skew(ep_size, num_experts, num_tokens, topk, skew):
    """Generate distinct top-k choices with exact balance, ties or controlled load skew."""
    if skew == "boundary":
        assert topk == 1
        counts = (31, 32, 33, 127, 128, 129, 255, 256, 257)
        flat = torch.cat([torch.full((count,), i % num_experts) for i, count in enumerate(counts)])
        size = ep_size * num_tokens
        return flat.repeat((size + flat.numel() - 1) // flat.numel())[:size].reshape(
            ep_size, num_tokens, 1
        )
    if skew == "ties":
        return torch.arange(9).repeat_interleave(4).reshape(4, 9, 1)
    rows = torch.arange(num_tokens)[:, None] + torch.arange(topk)
    local_experts = num_experts // ep_size
    if skew == "rank_ties":
        return (rows % (num_experts // 2)).repeat(ep_size, 1, 1)
    if skew in ("balanced", "local", "remote"):
        return torch.stack(
            [
                (
                    (rows + rank * local_experts) % num_experts
                    if skew == "balanced"
                    else rows % local_experts
                    + ((rank + (skew == "remote")) % ep_size) * local_experts
                )
                for rank in range(ep_size)
            ]
        )
    if skew == "hot_expert" and topk == 1:
        return torch.zeros((ep_size, num_tokens, topk), dtype=torch.int64)
    weights = torch.ones(num_experts)
    if skew == "hot_expert":
        weights[0] = 20
    elif skew == "hot_rank":
        weights[:local_experts] = 12
    elif skew == "concentrated":
        weights[topk:] = 0
    else:
        assert skew == "random"
    return torch.multinomial(
        weights.expand(ep_size * num_tokens, -1),
        topk,
        replacement=False,
        generator=torch.Generator().manual_seed(1234),
    ).reshape(ep_size, num_tokens, topk)


@pytest.mark.internal
def test_virtual_expert_planner_reference_tie_breaks():
    """Hand-computed over-migration exercises rank/expert ties and reversed slot ties."""
    # Two equally overloaded senders and two equally empty receivers pair in rank order.
    _, _, copies, mapped = _reference_plan(_routes_for_skew(4, 8, 4, 1, "rank_ties"), 8)
    assert copies.tolist() == [[-1, -1], [-1, -1], [0, -1], [2, -1]]
    assert mapped.flatten().tolist() == [10, 1, 14, 5] * 4
    routes = _routes_for_skew(4, 12, 9, 1, "ties")
    _, allocation, copies, mapped = _reference_plan(routes, 12)
    assert allocation.tolist() == [
        [0, 0, 0, 4],
        [0, 0, 0, 4],
        [3, 0, 0, 1],
        [4, 0, 0, 0],
        [2, 2, 0, 0],
        [0, 4, 0, 0],
        [0, 3, 1, 0],
        [0, 0, 4, 0],
        [0, 0, 4, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert copies.tolist() == [[3, 4, -1], [6, -1, -1], [-1, -1, -1], [1, 0, 2]]
    expected = [22] * 4 + [21] * 4 + [2] * 3 + [23] + [3] * 4 + [4] * 2 + [7] * 2
    expected += [8] * 4 + [9] * 3 + [12] + [13] * 4 + [14] * 4
    assert mapped.flatten().tolist() == expected


def _check_planner_reference(group, num_experts, num_tokens, topk, skews, alignments=(1,)):
    ep_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    device = torch.device("cuda", torch.cuda.current_device())
    local_experts = num_experts // ep_size
    workspace = VirtualExpertPlannerWorkspace(num_experts=num_experts, device=device, group=group)
    errors = []
    try:
        cases = [(skew, alignment) for alignment in alignments for skew in skews.split()]
        for iteration, (skew, alignment) in enumerate(cases):
            routes = _routes_for_skew(ep_size, num_experts, num_tokens, topk, skew)
            counts, allocation, copies, mapped = _reference_plan(routes, num_experts, alignment)
            dtype = torch.int32 if iteration % 2 else torch.int64
            own = routes[rank].to(device=device, dtype=dtype)
            if iteration % 3 and num_tokens > 1:
                # A nonzero offset and poisoned gaps catch a wrapper that forgets contiguous().
                storage = torch.full(
                    (num_tokens, 2 * topk + 1), num_experts + 7, device=device, dtype=dtype
                )
                storage[:, 1::2] = own
                own = storage[:, 1::2]
                assert not own.is_contiguous()
            elif num_tokens:
                # A contiguous view need not have a 16-byte-aligned base. Launcher reuse must
                # not borrow the compiler's stronger pointer alignment from another input.
                storage = torch.empty(own.numel() + 1, device=device, dtype=dtype)
                storage[1:].copy_(own.flatten())
                own = storage[1:].view_as(own)
                assert own.is_contiguous() and own.data_ptr() % 16 != 0
            plan = plan_virtual_expert_routes(own, workspace, alignment)
            actual = plan.virtual_experts.cpu()
            actual_copies = plan.experts_to_copy.cpu()
            for label, value, expected in (
                ("histogram", workspace.gathered_counts.cpu(), counts),
                ("allocation", workspace.field("allocation").cpu(), allocation),
                ("copies", actual_copies, copies),
                ("routes", actual, mapped[rank]),
            ):
                _check_equal(value, expected, f"{skew} {label}", errors)
            # Independently decode the public outputs and count real routes, rather than
            # accepting a self-consistent allocation/slot table with the wrong expert owners.
            try:
                assert actual.shape == (num_tokens, topk) and actual.dtype == torch.int16
                assert ((actual >= 0) & (actual < 2 * num_experts)).all()
                destination, local = actual.long() // (2 * local_experts), actual.long() % (
                    2 * local_experts
                )
                semantic = destination * local_experts + local
                remote = local >= local_experts
                semantic[remote] = actual_copies[
                    destination[remote], local[remote] - local_experts
                ].long()
                torch.testing.assert_close(semantic, routes[rank], rtol=0, atol=0)
                if alignment == 1 and skew in ("balanced", "local", "remote"):
                    assert (actual_copies == -1).all()
                if alignment == 1 and skew in ("local", "remote"):
                    assert (
                        (destination == rank) if skew == "local" else (destination != rank)
                    ).all()
                observed = torch.bincount(
                    (routes[rank] * ep_size + destination).flatten(),
                    minlength=num_experts * ep_size,
                ).to(device=device, dtype=torch.int32)
            except (AssertionError, IndexError) as exc:
                errors.append(f"{skew} semantic routes: {exc}")
                observed = torch.zeros(num_experts * ep_size, device=device, dtype=torch.int32)
            dist.all_reduce(observed, group=group)
            real_counts = observed.cpu().reshape(num_experts, ep_size)
            padded_counts = ((real_counts + alignment - 1) // alignment) * alignment
            _check_equal(padded_counts, allocation, f"{skew} route counts", errors)
            _check_equal(
                padded_counts.sum(0),
                torch.tensor(
                    [
                        (
                            int(allocation.sum()) // alignment // ep_size
                            + (r < int(allocation.sum()) // alignment % ep_size)
                        )
                        * alignment
                        for r in range(ep_size)
                    ]
                ),
                f"{skew} balanced load",
                errors,
            )
            assert int(real_counts.sum()) == routes.numel()
            capacity_owner = object.__new__(VirtualExpertLoadBalancer)
            capacity_owner.ep_size = ep_size
            capacity_owner.num_owned_experts = local_experts
            capacity_owner.router_topk = topk
            capacity_owner._alignment = alignment
            assert int(padded_counts.sum(0).max()) <= capacity_owner._compute_rank_capacity(
                num_tokens
            )
            # Every nonempty replica on a receiver belongs to a single owner.
            for row in copies:
                assert len(set((row[row >= 0] // local_experts).tolist())) <= 1
            # The real dispatcher provides this cross-rank ordering between planner launches.
            dist.barrier(group=group, device_ids=[device.index])
        _report(errors, dist.group.WORLD)
    finally:
        workspace.destroy()


@pytest.mark.internal
@pytest.mark.parametrize(
    "ep_size,num_experts,num_tokens,topk,skews",
    [
        (1, 8, 17, 3, "balanced concentrated balanced"),
        (2, 8, 257, 3, "random concentrated balanced"),
        (2, 8, 0, 1, "balanced"),
        (4, 8, 17, 3, "balanced concentrated balanced"),
        (4, 8, 1, 1, "hot_expert random"),
        (4, 8, 0, 1, "balanced"),
        (4, 8, 257, 1, "hot_expert random balanced boundary"),
        (4, 12, 9, 1, "ties local remote"),
        (4, 512, 33, 10, "random concentrated balanced"),
        (8, 16, 17, 3, "rank_ties random balanced"),
    ],
)
def test_virtual_expert_planner_matches_reference(
    virtual_expert_planner_group, ep_size, num_experts, num_tokens, topk, skews
):
    """Aligned remapping preserves identity and balances loads, including reused workspace."""
    _check_planner_reference(
        virtual_expert_planner_group,
        num_experts,
        num_tokens,
        topk,
        skews,
        alignments=(1, 32, 128, 256, 1, 256),
    )


@pytest.mark.internal
@pytest.mark.parametrize("ep_size", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_virtual_expert_planner_graph_replay(virtual_expert_planner_group, ep_size, dtype):
    """Changing routes and cached shapes/alignments must agree with the CPU oracle."""
    group = virtual_expert_planner_group
    rank = dist.get_rank(group)
    device = torch.device("cuda", torch.cuda.current_device())
    workspace = VirtualExpertPlannerWorkspace(num_experts=8, device=device, group=group)
    errors = []
    try:
        for num_tokens, topk, alignment in (
            (17, 1, 1),
            (257, 3, 32),
            (257, 3, 128),
            (33, 1, 256),
            (257, 3, 32),
            (17, 1, 1),
        ):
            routes = _routes_for_skew(ep_size, 8, num_tokens, topk, "balanced")
            own = routes[rank].to(device=device, dtype=dtype)
            # Compile before capture, including a cached launcher on the same workspace.
            plan_virtual_expert_routes(own, workspace, alignment)
            dist.barrier(group=group)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = plan_virtual_expert_routes(own, workspace, alignment)
            dist.barrier(group=group)
            for skew in ("concentrated", "balanced", "random", "balanced"):
                routes = _routes_for_skew(ep_size, 8, num_tokens, topk, skew)
                counts, allocation, copies, mapped = _reference_plan(routes, 8, alignment)
                own.copy_(routes[rank])
                for replay in range(3):
                    graph.replay()
                    dist.barrier(group=group)
                    for label, value, expected in (
                        ("histogram", workspace.gathered_counts.cpu(), counts),
                        ("allocation", workspace.field("allocation").cpu(), allocation),
                        ("copies", captured.experts_to_copy.cpu(), copies),
                        ("routes", captured.virtual_experts.cpu(), mapped[rank]),
                    ):
                        _check_equal(
                            value,
                            expected,
                            f"graph {num_tokens=} {topk=} {alignment=} {skew=} {replay=} {label}",
                            errors,
                        )
        _report(errors, dist.group.WORLD)
    finally:
        workspace.destroy()


def _routing(num_tokens=NUM_TOKENS, num_experts=NUM_EXPERTS, topk=TOPK):
    logits = torch.randn(num_tokens, num_experts, device="cuda")
    probs_full = torch.softmax(logits, dim=-1)
    idx = probs_full.topk(topk, dim=-1).indices
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool, device="cuda")
    routing_map.scatter_(1, idx, True)
    probs = torch.zeros_like(probs_full).scatter(1, idx, probs_full.gather(1, idx))
    return routing_map, probs


@pytest.mark.parametrize("dtype", [torch.int16, torch.int64])
@pytest.mark.parametrize("compiled", [False, True])
def test_virtual_expert_routing_adapter_replay(dtype, compiled):
    """Compact router outputs retain their pairing, zero scores, padding and gradients."""
    Utils.initialize_distributed()
    seeded()
    _, full_probs = _routing(num_tokens=258, num_experts=512, topk=10)
    ids = full_probs.topk(10, dim=1).indices
    probs = full_probs.gather(1, ids)
    probs[0].zero_()
    # Exercise noncontiguous IDs and scores without changing selected-route order.
    ids = ids.to(dtype).t().contiguous().t()
    probs = probs.t().contiguous().t().requires_grad_()
    padding_mask = torch.zeros((2, 129), dtype=torch.bool, device="cuda")
    padding_mask[0, 3::7] = True
    flat_padding = padding_mask.t().reshape(-1)
    manager = object.__new__(_VirtualExpertHybridEPManager)
    manager.router_topk = 10
    planned = []
    manager.plan_dispatch = planned.append

    def adapt(probs):
        planned.clear()
        manager.setup_metadata(ids, probs, padding_mask)
        return planned[0], manager.token_probs

    if compiled:
        adapt = torch.compile(adapt)
    indices, selected_probs = adapt(probs)
    torch.testing.assert_close(indices, ids.long(), rtol=0, atol=0)
    expected_probs = probs.masked_fill(flat_padding[:, None], 0)
    torch.testing.assert_close(selected_probs, expected_probs, rtol=0, atol=0)
    (gradient,) = torch.autograd.grad(selected_probs.sum(), probs)
    expected_gradient = torch.ones_like(probs).masked_fill(flat_padding[:, None], 0)
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    assert_replays_bit_exact(
        adapt,
        (probs,),
        replays=3,
        contention=True,
        what=f"virtual-expert routing adapter[{dtype=}, {compiled=}]",
    )


@pytest.mark.parametrize("invalid", ["bool-map", "wrong-topk", "full-width-probs"])
def test_virtual_expert_routing_adapter_rejects_invalid_layout(invalid):
    """Reject incompatible router metadata before starting the planner."""
    manager = object.__new__(_VirtualExpertHybridEPManager)
    manager.router_topk = 2
    ids = torch.zeros((3, 1 if invalid == "wrong-topk" else 2), dtype=torch.int64)
    if invalid == "bool-map":
        ids = ids.bool()
    probs = torch.zeros((3, 4)) if invalid == "full-width-probs" else torch.zeros_like(ids).float()
    with pytest.raises(ValueError, match="expert IDs and probabilities"):
        manager.setup_metadata(ids, probs)


# --- permute / unpermute --------------------------------------------------------------------


def _permute_roundtrip(fused, with_probs):
    def fn(tokens, routing_map, probs):
        permuted, permuted_probs, sorted_indices, *_ = moe_utils.permute(
            tokens,
            routing_map,
            probs=probs if with_probs else None,
            num_out_tokens=NUM_TOKENS * TOPK,
            fused=fused,
        )
        # The experts apply the routed probabilities before the combine; every production
        # caller then unpermutes with ``probs=None``.
        if with_probs:
            permuted = permuted * permuted_probs.unsqueeze(-1).to(permuted.dtype)
        else:
            permuted = permuted * 1.001
        return moe_utils.unpermute(
            permuted, sorted_indices, tokens.shape, routing_map=routing_map, fused=fused
        )

    return fn


@pytest.mark.parametrize(
    "fused",
    [
        False,
        pytest.param(
            True, marks=pytest.mark.skipif(not HAVE_TE_PERMUTE, reason="TE permute missing")
        ),
    ],
)
@pytest.mark.parametrize("with_probs", [False, True])
def test_permute_unpermute_replay_under_deterministic_algorithms(fused, with_probs):
    """Deterministic branch: ``index_add_`` combine, ``index_put_`` gathers -- bit-exact."""
    seeded()
    routing_map, probs = _routing()
    tokens = torch.randn(
        NUM_TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    probs.requires_grad_(with_probs)
    with deterministic_algorithms(True):
        assert_replays_bit_exact(
            _permute_roundtrip(fused, with_probs),
            (tokens, routing_map, probs),
            replays=3,
            what=f"permute/unpermute[fused={fused}, probs={with_probs}]",
        )


@pytest.mark.xfail(
    strict=False,
    reason="Negative control. On GB300 the bf16 scatter_add_ combine replayed bit-exactly 8 "
    "times (2026-09-04), so the race is hardware/shape dependent; recorded, not gated.",
)
def test_default_unpermute_is_the_racy_path():
    """Negative control: without deterministic algorithms the torch combine uses
    ``scatter_add_`` (atomic bf16 accumulation of 8 rows per token over 16k tokens). When it
    races visibly, the deterministic assertion above is known to be sensitive."""
    seeded()
    routing_map, probs = _routing()
    tokens = torch.randn(
        NUM_TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    with deterministic_algorithms(False):
        differing = count_differing_replays(
            _permute_roundtrip(False, False), (tokens, routing_map, probs), replays=8
        )
    assert differing > 0, (
        "the scatter_add_ combine replayed bit-exactly 7 times; either torch made it deterministic "
        "(drop this control) or the shape no longer contends"
    )


@pytest.mark.parametrize(
    "fused",
    [
        False,
        pytest.param(True, marks=pytest.mark.skipif(not HAVE_TE_PERMUTE, reason="TE sort missing")),
    ],
)
@pytest.mark.parametrize("with_probs", [False, True])
def test_sort_chunks_by_idxs_replays(fused, with_probs):
    seeded()
    num_chunks = 32
    sizes = torch.randint(100, 2000, (num_chunks,))
    sizes[3] = 0
    rows = int(sizes.sum())
    split_sizes = sizes.to("cuda") if fused else sizes
    sorted_idxs = torch.randperm(num_chunks).to("cuda") if fused else torch.randperm(num_chunks)
    x = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    probs = torch.rand(rows, device="cuda", requires_grad=with_probs)

    def fn(x, probs):
        out, out_probs = moe_utils.sort_chunks_by_idxs(
            x, split_sizes, sorted_idxs, probs=probs if with_probs else None, fused=fused
        )
        return (out, out_probs) if with_probs else out

    assert_replays_bit_exact(fn, (x, probs), replays=3, what=f"sort_chunks_by_idxs[fused={fused}]")


# --- routing --------------------------------------------------------------------------------

ROUTING_CASES = {
    "softmax_topk8": dict(topk=8, score_function="softmax"),
    "softmax_pre_softmax": dict(topk=8, use_pre_softmax=True, score_function="softmax"),
    "sigmoid_scaled": dict(topk=8, score_function="sigmoid", scaling_factor=2.5),
    "group_limited": dict(topk=8, score_function="sigmoid", num_groups=8, group_topk=4),
    "topk2": dict(topk=2, score_function="softmax"),
}


@pytest.mark.parametrize("case", sorted(ROUTING_CASES))
@pytest.mark.parametrize(
    "fused",
    [
        False,
        pytest.param(True, marks=pytest.mark.skipif(not HAVE_TE_ROUTER, reason="TE>=2.7 needed")),
    ],
)
@pytest.mark.parametrize("det_algos", [True, False], ids=["det-branch", "default-branch"])
def test_topk_routing_replays(case, fused, det_algos):
    seeded()
    kwargs = dict(ROUTING_CASES[case])
    logits = torch.randn(8192, 256, device="cuda", dtype=torch.float32, requires_grad=True)
    expert_bias = (
        torch.randn(256, device="cuda") * 0.01
        if kwargs.get("score_function") == "sigmoid"
        else None
    )

    def fn(logits):
        probs, routing_map = moe_utils.topk_routing_with_score_function(
            logits, fused=fused, expert_bias=expert_bias, **kwargs
        )
        return probs, routing_map

    with deterministic_algorithms(det_algos):
        assert_replays_bit_exact(
            fn, (logits,), replays=3, what=f"topk routing[{case}, fused={fused}]"
        )


def test_group_limited_topk_replays():
    seeded()
    scores = torch.rand(8192, 256, device="cuda")
    assert_replays_bit_exact(
        lambda s: moe_utils.group_limited_topk(s, 8, 8192, 256, 8, 4),
        (scores,),
        backward=False,
        what="group_limited_topk",
    )


@pytest.mark.parametrize(
    "fused",
    [
        False,
        pytest.param(
            True,
            marks=[
                pytest.mark.skipif(not HAVE_TE_ROUTER, reason="TE>=2.7 needed"),
                pytest.mark.xfail(
                    strict=False,
                    reason="TE fused_moe_aux_loss reduces with atomicAdd (open gap, recorded not gated)",
                ),
            ],
        ),
    ],
)
def test_switch_load_balancing_loss_replays(fused):
    seeded()
    routing_map, probs = _routing(num_tokens=65536, num_experts=256, topk=8)
    probs = probs.detach().requires_grad_(True)
    tokens_per_expert = routing_map.sum(dim=0)

    def fn(probs):
        return moe_utils.switch_load_balancing_loss_func(
            probs, tokens_per_expert, 65536, 8, 256, 1e-2, fused=fused
        )

    assert_replays_bit_exact(fn, (probs,), replays=4, what=f"aux loss[fused={fused}]")


@pytest.mark.parametrize("router_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("with_bias", [False, True])
def test_router_gating_linear_replays(router_dtype, with_bias):
    seeded()
    inp = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(256, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True) * 0.02
    weight = weight.detach().requires_grad_(True)
    bias = (
        torch.randn(256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        if with_bias
        else None
    )
    assert_replays_bit_exact(
        lambda i, w, b: moe_utils.router_gating_linear(i, w, b, router_dtype),
        (inp, weight, bias),
        replays=3,
        contention=True,
        what="router_gating_linear",
    )


# --- modules --------------------------------------------------------------------------------


def _moe_config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=1024,
        ffn_hidden_size=2048,
        num_attention_heads=8,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.01,
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_activation_fusion=True,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        deterministic_mode=True,
        sequence_parallel=False,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


class _InQuantizationContext(torch.nn.Module):
    """Run the wrapped module inside MCore's FP8 autocast, as TransformerBlock does."""

    def __init__(self, module, config):
        super().__init__()
        self.module = module
        self.config = config

    def forward(self, *args):
        with get_fp8_context(self.config):
            return self.module(*args)


class TestMoEModules:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _init(self, ep=1):
        Utils.initialize_model_parallel(expert_model_parallel_size=ep)
        model_parallel_cuda_manual_seed(123)

    @pytest.mark.parametrize(
        "balancing,score,expert_bias,hash_routing",
        [
            ("aux_loss", "softmax", False, False),
            ("seq_aux_loss", "sigmoid", True, False),
            ("sinkhorn", "softmax", False, False),
            ("none", "sigmoid", False, True),
            ("none", "softmax", False, True),
        ],
        ids=["aux_loss", "seq_aux_loss+sigmoid+bias", "sinkhorn", "hash+sigmoid", "hash+softmax"],
    )
    def test_topk_router_replays(self, balancing, score, expert_bias, hash_routing):
        self._init()
        seeded()
        config = _moe_config(
            num_moe_experts=64,
            moe_router_topk=8 if balancing != "sinkhorn" else 1,
            moe_router_load_balancing_type=balancing,
            moe_router_score_function=score,
            moe_router_enable_expert_bias=expert_bias,
            moe_router_pre_softmax=balancing == "sinkhorn",
            moe_aux_loss_coeff=0.0 if balancing in ("sinkhorn", "none") else 0.01,
            moe_num_hash_layers=int(hash_routing),
            hash_moe_vocab_size=128 if hash_routing else None,
        )
        router = TopKRouter(
            config, pg_collection=ProcessGroupCollection.use_mpu_process_groups()
        ).cuda()
        router.set_layer_number(0)
        hidden = torch.randn(2048, 4, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        inputs = (hidden,)
        if expert_bias:
            padding_mask = torch.arange(8192, device="cuda").reshape(2048, 4) % 3 == 0
            inputs = {"input": hidden, "padding_mask": padding_mask}
        grad_output = None
        if hash_routing:
            assert router.is_hash_layer
            # Repeated token IDs exercise the fixed table with many tokens per expert.
            input_ids = torch.arange(8192, device="cuda").view(4, 2048) % config.hash_moe_vocab_size
            inputs = {"input": hidden, "input_ids": input_ids}
            # Summing normalized routing probabilities would mask their gradients.
            grad_output = torch.randn(8192, config.num_moe_experts, device="cuda")
        assert_module_replays_bit_exact(
            router,
            inputs,
            replays=3,
            grad_output=grad_output,
            contention=hash_routing,
            what=f"TopKRouter[{balancing}, hash={hash_routing}]",
        )

    @pytest.mark.skipif(
        not (HAVE_TE_ROUTER and moe_utils.fused_topk_with_score_function_supports_topk_indices),
        reason="TE dense fused router output is not available",
    )
    @pytest.mark.parametrize("backend", ["deepep", "ncclep"])
    @pytest.mark.parametrize("expert_bias", [False, True], ids=["no_bias", "bias"])
    def test_topk_router_dense_indices_replays(self, backend, expert_bias):
        """With TE dense fused output, flex deepep/ncclep routers return dense int64
        [tokens, topk] indices next to the full-width [tokens, num_experts] probs (the dispatcher
        selects the weights) and count expert loads from the indices."""
        self._init()
        seeded()
        config = _moe_config(
            num_moe_experts=64,
            moe_router_topk=8,
            moe_router_load_balancing_type="aux_loss",
            # Expert bias is only permitted with the sigmoid / sqrtsoftplus score functions.
            moe_router_score_function="sigmoid" if expert_bias else "softmax",
            moe_router_enable_expert_bias=expert_bias,
            moe_router_fusion=True,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend=backend,
        )
        router = TopKRouter(
            config, pg_collection=ProcessGroupCollection.use_mpu_process_groups()
        ).cuda()
        router.set_layer_number(0)
        hidden = torch.randn(2048, 4, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        probs, routing_map = router(hidden)
        assert routing_map.dtype == torch.int64 and routing_map.shape == (8192, 8)
        assert probs.shape == (8192, 64)
        assert_module_replays_bit_exact(
            router, (hidden,), replays=3, what=f"TopKRouter[flex-{backend}-dense]"
        )

    @pytest.mark.skipif(not HAVE_TE, reason="TE grouped MLP needs Transformer Engine")
    def test_te_grouped_mlp_replays_on_uneven_experts(self):
        self._init()
        seeded()
        config = _moe_config(hidden_size=2048, ffn_hidden_size=4096)
        spec = get_gpt_layer_with_transformer_engine_spec(num_experts=8, moe_grouped_gemm=True)
        experts = get_submodules(spec.submodules.mlp).experts(
            num_local_experts=8,
            config=config,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        assert isinstance(experts, TEGroupedMLP)
        experts = experts.cuda()
        tokens_per_expert = torch.tensor([4096, 17, 0, 2048, 1, 8191, 33, 1998], dtype=torch.int64)
        rows = int(tokens_per_expert.sum())
        hidden = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        probs = torch.rand(rows, device="cuda", requires_grad=True)
        assert_module_replays_bit_exact(
            experts,
            (hidden, tokens_per_expert, probs),
            replays=3,
            contention=True,
            what="TEGroupedMLP",
        )

    @pytest.mark.skipif(
        not hasattr(torch, "float8_e8m0fnu") or torch.cuda.get_device_capability()[0] < 10,
        reason="MXFP8 parameter storage needs Blackwell",
    )
    def test_vllm_mxfp8_weight_stacking_replays(self):
        """The mixed-precision vLLM path builds canonical expert stacks bit-exactly."""
        from types import SimpleNamespace

        from megatron.core.inference.moe import InferenceGroupedGemmBackend
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
        from megatron.core.transformer.moe.experts import InferenceGroupedMLP

        class GroupedMLPStub:
            _stack_mxfp8_linear_weight = InferenceGroupedMLP._stack_mxfp8_linear_weight

        seeded()
        weights = tuple(torch.randn(64, 128, device="cuda", dtype=torch.bfloat16) for _ in range(4))

        def build_stacks(*bf16_weights):
            grouped = GroupedMLPStub()
            grouped.num_local_experts = 2
            grouped.inference_grouped_gemm_backend = InferenceGroupedGemmBackend.VLLM
            grouped.linear_fc1 = SimpleNamespace()
            grouped.linear_fc2 = SimpleNamespace()
            for linear, offset in ((grouped.linear_fc1, 0), (grouped.linear_fc2, 2)):
                for expert in range(2):
                    setattr(
                        linear,
                        f"weight{expert}",
                        MXFP8Tensor.from_bf16(bf16_weights[offset + expert], backend="triton"),
                    )
            InferenceGroupedMLP._build_concatenated_mxfp8_weights(grouped)
            return (
                grouped._fc1_weight.data.view(torch.uint8),
                grouped._fc1_weight.scale.view(torch.uint8),
                grouped._fc2_weight.data.view(torch.uint8),
                grouped._fc2_weight.scale.view(torch.uint8),
            )

        assert_replays_bit_exact(
            build_stacks, weights, backward=False, what="vLLM MXFP8 expert-weight stacking"
        )

    @pytest.mark.skipif(not HAVE_TE, reason="TE grouped MLP needs Transformer Engine")
    @pytest.mark.parametrize("op_fuser", [False, True], ids=["unfused", "op-fuser-mxfp8"])
    def test_te_grouped_mlp_tanh_clamp_replays_on_uneven_experts(self, op_fuser):
        """Squared ReLU with the tanh soft clamp: the unfused weighted_squared_relu_impl path, and
        the fused cuDNN srelu_tanh grouped GEMM (op fuser + MXFP8; SM100 and ScaledTanhSReLU)."""
        if op_fuser:
            if torch.cuda.get_device_capability()[0] < 10:
                pytest.skip("fused grouped MLP under MXFP8 needs SM100+")
            if not _te_supports_scaled_tanh_srelu():
                pytest.skip("installed TE has no ScaledTanhSReLU")
        self._init()
        seeded()
        config = _moe_config(
            hidden_size=2048,
            ffn_hidden_size=4096,
            gated_linear_unit=False,
            activation_func=squared_relu,
            bias_activation_fusion=False,
            use_fused_weighted_squared_relu=True,
            activation_func_tanh_clamp_scale=16.0,
            use_transformer_engine_op_fuser=op_fuser,
            **({"fp8": "e4m3", "fp8_recipe": "mxfp8"} if op_fuser else {}),
        )
        spec = get_gpt_layer_with_transformer_engine_spec(num_experts=8, moe_grouped_gemm=True)
        with get_fp8_context(config, is_init=True):
            experts = get_submodules(spec.submodules.mlp).experts(
                num_local_experts=8,
                config=config,
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            )
        assert isinstance(experts, TEGroupedMLP)
        assert experts._with_fused_impl is op_fuser
        experts = experts.cuda()
        tokens_per_expert = torch.tensor([4096, 17, 0, 2048, 1, 8191, 33, 1998], dtype=torch.int64)
        rows = int(tokens_per_expert.sum())
        hidden = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        probs = torch.rand(rows, device="cuda", requires_grad=True)
        module = _InQuantizationContext(experts, config) if op_fuser else experts
        assert_module_replays_bit_exact(
            module,
            (hidden, tokens_per_expert, probs),
            replays=3,
            contention=True,
            what=f"TEGroupedMLP[tanh clamp, {'op fuser' if op_fuser else 'unfused'}]",
        )

    def test_sequential_mlp_replays_on_uneven_experts(self):
        self._init()
        seeded()
        config = _moe_config(hidden_size=2048, ffn_hidden_size=4096, moe_grouped_gemm=False)
        submodules = get_gpt_layer_local_submodules(num_experts=8, moe_grouped_gemm=False)
        experts = get_submodules(submodules.mlp).experts(
            num_local_experts=8,
            config=config,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        assert isinstance(experts, SequentialMLP)
        experts = experts.cuda()
        tokens_per_expert = torch.tensor([4096, 17, 0, 2048, 1, 8191, 33, 1998], dtype=torch.int64)
        rows = int(tokens_per_expert.sum())
        hidden = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        probs = torch.rand(rows, device="cuda", requires_grad=True)
        assert_module_replays_bit_exact(
            experts,
            (hidden, tokens_per_expert, probs),
            replays=3,
            contention=True,
            what="SequentialMLP",
        )

    @pytest.mark.parametrize(
        "dispatcher,ep,extra",
        [
            ("allgather", 1, {}),
            ("alltoall", 1, {}),
            pytest.param(
                "alltoall",
                1,
                {
                    "moe_num_hash_layers": 1,
                    "hash_moe_vocab_size": 128,
                    "moe_router_score_function": "sigmoid",
                    "moe_router_load_balancing_type": "none",
                    "moe_aux_loss_coeff": 0.0,
                },
                id="alltoall-hash",
            ),
            (
                "alltoall",
                1,
                {"moe_expert_capacity_factor": 0.5, "moe_pad_expert_input_to_capacity": True},
            ),
            pytest.param("alltoall", 2, {"moe_permute_fusion": HAVE_TE_PERMUTE}, id="alltoall-ep2"),
            pytest.param(
                "flex",
                2,
                {"moe_flex_dispatcher_backend": "deepep"},
                id="flex-deepep-ep2",
                marks=pytest.mark.skipif(not HAVE_DEEP_EP, reason="DeepEP not installed"),
            ),
        ],
    )
    def test_moe_layer_replays(self, dispatcher, ep, extra):
        if Utils.world_size % ep != 0 or (ep > 1 and Utils.world_size < ep):
            pytest.skip(f"needs a world size divisible by EP={ep}")
        self._init(ep=ep)
        seeded()
        config = _moe_config(
            moe_token_dispatcher_type=dispatcher, expert_model_parallel_size=ep, **extra
        )
        if not HAVE_TE:
            config.moe_grouped_gemm = False
            mlp_spec = get_gpt_layer_local_submodules(num_experts=8, moe_grouped_gemm=False).mlp
        else:
            mlp_spec = get_gpt_layer_with_transformer_engine_spec(
                num_experts=8, moe_grouped_gemm=True
            ).submodules.mlp
        layer = MoELayer(
            config,
            get_submodules(mlp_spec),
            hash_moe_layer_threshold=1 if config.moe_num_hash_layers else None,
        ).cuda()
        layer.set_layer_number(0)
        hidden = torch.randn(2048, 2, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        inputs = (hidden,)
        if config.moe_num_hash_layers:
            assert layer.router.is_hash_layer
            input_ids = torch.arange(4096, device="cuda").view(2, 2048) % config.hash_moe_vocab_size
            inputs = {"hidden_states": hidden, "input_ids": input_ids}
        with deterministic_algorithms(True):
            assert_module_replays_bit_exact(
                layer, inputs, replays=3, contention=True, what=f"MoELayer[{dispatcher}, ep={ep}]"
            )
