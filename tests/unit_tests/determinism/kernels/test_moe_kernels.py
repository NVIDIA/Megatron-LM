# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the MoE kernels in ``megatron/core/transformer/moe/``.

Operator level: token permute / unpermute (torch and TE-fused, both branches of the
``torch.are_deterministic_algorithms_enabled()`` switch), chunk sorting, top-k routing (torch
and TE-fused), group-limited routing, the load-balancing aux loss (torch and TE-fused), and
the router gating GEMM, and virtual-expert planning against a CPU oracle in eager and CUDA
graphs. Module level: ``TopKRouter``, ``TEGroupedMLP`` / ``SequentialMLP`` on
deliberately uneven expert loads (including an empty expert), and a full ``MoELayer`` through
the all-gather and all-to-all dispatchers (plus the flex/DeepEP dispatcher from
``fused_a2a.py`` when the dependency and the GPUs are available), and the inference
``MoELayer`` through FlashInfer per-token NVFP4 experts on Blackwell.
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import megatron.core.extensions.transformer_engine as te_ext
from megatron.core import parallel_state
from megatron.core.activations import squared_relu
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface,
    PipelineOffloadManager,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe import batch_invariant, moe_utils
from megatron.core.transformer.moe.experts import (
    SequentialMLP,
    TEGroupedMLP,
    _te_supports_scaled_tanh_srelu,
)
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.ops.virtual_expert_triton import VirtualExpertPlannerWorkspace
from megatron.core.transformer.moe.router import InferenceTopKRouter, TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    _NCCLEPManager,
    _VirtualExpertHybridEPManager,
)
from megatron.core.transformer.moe.virtual_expert_load_balancer import (
    VirtualExpertLoadBalancer,
    plan_virtual_expert_routes,
)
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.determinism.kernels.harness import (
    assert_module_replays_bit_exact,
    assert_replays_bit_exact,
    bytes_equal,
    count_differing_replays,
    deterministic_algorithms,
    seeded,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HAVE_TE = moe_utils.HAVE_TE
HAVE_TE_PERMUTE = HAVE_TE and moe_utils.fused_permute is not None
HAVE_TE_ROUTER = HAVE_TE and is_te_min_version("2.7.0")
from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP, HAVE_DEEP_EP_V2

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

    # Invalid metadata must fail before planning; valid inputs below reuse this manager.
    for invalid_ids, invalid_probs in (
        (ids.bool(), probs),
        (ids[:, :-1], probs[:, :-1]),
        (ids, full_probs),
    ):
        with pytest.raises(ValueError, match="expert IDs and probabilities"):
            manager.setup_metadata(invalid_ids, invalid_probs)
        assert not planned, "invalid metadata must not reach the planner"

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


@pytest.mark.parametrize("routing", ["bool", "indices"])
def test_ncclep_padding_mask_metadata_replays(routing):
    """NCCL EP routing metadata under a padding mask: padded rows become -1 routes with zero
    weight (the kernels skip them and combine writes zeros back), valid rows keep their ids and
    weights, and the packed-sequence flag is recorded; bit-exact on replay."""
    Utils.initialize_distributed()
    seeded()
    num_tokens, num_experts, topk = 258, 64, 8
    routing_map, full_probs = _routing(num_tokens=num_tokens, num_experts=num_experts, topk=topk)
    expected_ids = full_probs.topk(topk, dim=1).indices
    if routing == "indices":
        routing_map = expected_ids.to(torch.int32)
    padding_mask = torch.zeros((2, 129), dtype=torch.bool, device="cuda")
    padding_mask[0, 3::7] = True
    flat_padding = padding_mask.t().reshape(-1)
    manager = object.__new__(_NCCLEPManager)
    manager.router_topk = topk

    def adapt(probs):
        manager.setup_metadata(
            routing_map,
            probs,
            padding_mask=padding_mask,
            packed_seq_params=PackedSeqParams(qkv_format="thd"),
        )
        return manager.token_indices, manager.token_probs

    indices, selected_probs = adapt(full_probs)
    assert manager._is_thd and manager.num_local_tokens == num_tokens
    expected_ids = expected_ids.masked_fill(flat_padding[:, None], -1)
    torch.testing.assert_close(indices, expected_ids, rtol=0, atol=0)
    expected_probs = full_probs.gather(1, expected_ids.clamp_min(0))
    expected_probs = expected_probs.masked_fill(flat_padding[:, None], 0)
    torch.testing.assert_close(selected_probs, expected_probs, rtol=0, atol=0)
    assert_replays_bit_exact(
        adapt,
        (full_probs,),
        replays=3,
        contention=True,
        what=f"ncclep padding-mask metadata[{routing=}]",
    )


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


@pytest.mark.parametrize("det_algos", [True, False])
def test_quantile_balancing_histogram_and_bias_replay(det_algos):
    """Replay accumulated integer histograms, routing gradients, and the batch update."""
    seeded()
    num_tokens, num_experts, num_bins = 8192, 64, 128
    logits = torch.randn(num_tokens, num_experts, device="cuda", requires_grad=True)
    bias = torch.linspace(-0.1, 0.1, num_experts, device="cuda")
    bounds = torch.tensor([-1.0, 1.0], device="cuda")
    histogram = torch.zeros(num_experts, num_bins, device="cuda", dtype=torch.int32)

    def fn(logits, histogram):
        # The harness clones the caller-owned histogram before every replay.
        outputs = []
        for microbatch in logits.chunk(2):
            outputs.append(
                moe_utils.topk_routing_with_score_function(
                    microbatch,
                    topk=8,
                    score_function="sigmoid",
                    expert_bias=bias,
                    qb_histogram=histogram,
                    qb_bin_bounds=bounds,
                )
            )
        probs = torch.cat([output[0] for output in outputs])
        routing_map = torch.cat([output[1] for output in outputs])
        updated_bias, updated_bounds = moe_utils.get_updated_expert_bias_with_quantile(
            histogram, bounds, bias, topk=8
        )
        return probs, routing_map, histogram, updated_bias, updated_bounds

    with deterministic_algorithms(det_algos):
        outputs, grads = assert_replays_bit_exact(
            fn,
            (logits, histogram),
            grad_outputs={"out[0]": torch.randn_like(logits)},
            replays=3,
            contention=True,
            what="global-batch QB histogram and bias update",
        )
    assert torch.equal(
        outputs["out[2]"].sum(dim=-1),
        torch.full((num_experts,), num_tokens, device="cuda", dtype=torch.int64),
    )
    assert torch.all(outputs["out[1]"].sum(dim=-1) == 8)
    assert torch.isfinite(outputs["out[3]"]).all()
    assert torch.isfinite(outputs["out[4]"]).all()
    assert grads["in[0]"].abs().sum() > 0


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
@pytest.mark.parametrize("all_padding", [False, True])
def test_switch_load_balancing_loss_replays(fused, all_padding):
    seeded()
    routing_map, probs = _routing(num_tokens=65536, num_experts=256, topk=8)
    if all_padding:
        routing_map.zero_()
        probs.zero_()
    probs = probs.detach().requires_grad_(True)
    tokens_per_expert = routing_map.sum(dim=0)
    total_num_tokens = tokens_per_expert.sum() // 8 if all_padding else 65536

    def fn(probs):
        return moe_utils.switch_load_balancing_loss_func(
            probs, tokens_per_expert, total_num_tokens, 8, 256, 1e-2, fused=fused
        )

    assert_replays_bit_exact(
        fn, (probs,), replays=4, what=f"aux loss[fused={fused}, all_padding={all_padding}]"
    )


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

    @pytest.mark.parametrize("dispatcher", ["allgather", "alltoall"])
    @pytest.mark.parametrize("balancing", ["aux_loss", "seq_aux_loss"])
    @pytest.mark.parametrize("per_token_loss", [False, True])
    def test_runtime_cp_moe_layer_replays(self, dispatcher, balancing, per_token_loss, monkeypatch):
        """Replay real routing, collectives, experts and backward as runtime CP changes.

        CP2/CP4 include a rank with no valid tokens and uneven peer counts. CP1 is
        repeated after the larger groups to catch stale per-microbatch routing state.
        """
        if Utils.world_size < 4 or Utils.world_size % 4:
            pytest.skip("runtime CP1/CP2/CP4 needs a world size divisible by four")
        Utils.initialize_model_parallel(dynamic_context_parallel=True)
        model_parallel_cuda_manual_seed(123)
        seeded()
        config = _moe_config(
            moe_grouped_gemm=False,
            moe_token_dispatcher_type=dispatcher,
            moe_router_load_balancing_type=balancing,
            calculate_per_token_loss=per_token_loss,
        )
        mlp_spec = get_gpt_layer_local_submodules(num_experts=8, moe_grouped_gemm=False).mlp
        layer = MoELayer(config, get_submodules(mlp_spec)).cuda().train()
        layer.set_layer_number(1)
        monkeypatch.setattr(
            moe_utils.MoEAuxLossAutoScaler,
            "main_loss_backward_scale",
            torch.tensor(1.0, device="cuda"),
        )
        hidden = torch.randn(2048, 1, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        cp1_reference = None
        for cp_size in (1, 2, 4, 1):
            cp_group = parallel_state.get_dynamic_data_context_parallel_groups(group_size=cp_size)
            packed = PackedSeqParams(qkv_format="thd", local_cp_size=cp_size, cp_group=cp_group)
            valid_tokens = 1537 if cp_size == 1 else cp_group.rank() * 511
            padding_mask = torch.arange(2048, device="cuda").unsqueeze(0) >= valid_tokens

            # Integer counts must retain their local value through in-place all-reduces.
            routing_map = torch.zeros(2048, 8, device="cuda", dtype=torch.bool)
            routing_map[:valid_tokens, :2] = True
            counts, local_count, total_count = moe_utils.get_tokens_per_expert_and_token_count(
                routing_map, (cp_group, layer.router.tp_group), 2, True
            )
            expected_total = 1537 if cp_size == 1 else 511 * cp_size * (cp_size - 1) // 2
            assert local_count.item() == valid_tokens
            assert total_count.item() == expected_total
            torch.testing.assert_close(counts[:2], torch.full_like(counts[:2], expected_total))
            assert counts[2:].count_nonzero().item() == 0

            with deterministic_algorithms(True):
                outputs, grads = assert_module_replays_bit_exact(
                    layer,
                    {
                        "hidden_states": hidden,
                        "padding_mask": padding_mask,
                        "packed_seq_params": packed,
                    },
                    replays=3,
                    contention=True,
                    what=f"runtime CP{cp_size} MoELayer[{dispatcher}, {balancing}, per-token={per_token_loss}]",
                )
            assert grads, "backward must produce parameter and input gradients"
            assert any("router" in name for name in grads)
            for tensor in (*outputs.values(), *grads.values()):
                assert torch.isfinite(tensor).all()
            if cp_size == 1:
                if cp1_reference is None:
                    cp1_reference = (outputs, grads)
                else:
                    for reference, current in zip(cp1_reference, (outputs, grads)):
                        assert reference.keys() == current.keys()
                        for name in reference:
                            torch.testing.assert_close(
                                reference[name], current[name], rtol=0, atol=0
                            )
        moe_utils.get_moe_metrics_tracker().clear()

    @pytest.mark.parametrize(
        "balancing,score,expert_bias,hash_routing,per_token_loss",
        [
            ("aux_loss", "softmax", False, False, False),
            ("seq_aux_loss", "sigmoid", True, False, False),
            ("sinkhorn", "softmax", False, False, False),
            ("none", "sigmoid", False, True, False),
            ("none", "softmax", False, True, False),
            ("seq_aux_loss", "softmax", False, False, True),
        ],
        ids=[
            "aux_loss",
            "seq_aux_loss+sigmoid+bias",
            "sinkhorn",
            "hash+sigmoid",
            "hash+softmax",
            "seq_aux_loss+per_token_loss",
        ],
    )
    def test_topk_router_replays(
        self, balancing, score, expert_bias, hash_routing, per_token_loss, monkeypatch
    ):
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
            calculate_per_token_loss=per_token_loss,
        )
        router = TopKRouter(
            config, pg_collection=ProcessGroupCollection.use_mpu_process_groups()
        ).cuda()
        router.set_layer_number(0)
        if per_token_loss:
            # Without a padding mask, seq aux loss uses the Python-int token-count branch.
            monkeypatch.setattr(
                moe_utils.MoEAuxLossAutoScaler,
                "main_loss_backward_scale",
                torch.ones((), device="cuda"),
            )
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

    def test_topk_router_per_token_loss_replays_and_captures(self):
        """seq_aux_loss with per-token-loss scaling replays bit-exactly and its scale tensor can
        be created inside CUDA graph capture (a pageable H2D copy there fails the capture)."""
        self._init()
        seeded()
        config = _moe_config(
            num_moe_experts=64,
            moe_router_topk=8,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_router_score_function="sigmoid",
            moe_aux_loss_coeff=0.01,
            calculate_per_token_loss=True,
        )
        router = TopKRouter(
            config, pg_collection=ProcessGroupCollection.use_mpu_process_groups()
        ).cuda()
        router.set_layer_number(0)
        hidden = torch.randn(2048, 4, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        assert_module_replays_bit_exact(
            router, (hidden,), replays=3, what="TopKRouter[seq_aux_loss, per_token_loss]"
        )
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            router(hidden)
        torch.cuda.current_stream().wait_stream(side_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            router(hidden)
        graph.replay()
        torch.cuda.synchronize()
        moe_utils.get_moe_metrics_tracker().clear()

    @pytest.mark.parametrize(
        "num_experts,topk,scaling_factor",
        [(128, 6, 2.5), (512, 22, 5.0)],
        ids=["nano", "super-ultra"],
    )
    @pytest.mark.parametrize("num_tokens", [2, 512])
    def test_inference_topk_router_replays(self, num_experts, topk, scaling_factor, num_tokens):
        """Inference router replay through the fused sigmoid top-k kernel."""
        self._init()
        seeded()
        config = _moe_config(
            num_moe_experts=num_experts,
            moe_router_topk=topk,
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0.0,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_topk_scaling_factor=scaling_factor,
            inference_moe_router_backend="triton",
        )
        router = InferenceTopKRouter(
            config, pg_collection=ProcessGroupCollection.use_mpu_process_groups()
        ).cuda()
        router.set_layer_number(0)
        with torch.no_grad():
            router.expert_bias.uniform_(-0.05, 0.05)
        hidden = torch.randn(num_tokens, 1, 1024, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            assert router._uses_sigmoid_topk_kernel(router.gating(hidden).squeeze(1), None)

        def route(h):
            with InferenceMode.active(), torch.inference_mode():
                return router(h)

        assert_replays_bit_exact(
            route,
            (hidden,),
            replays=3,
            backward=False,
            what=f"InferenceTopKRouter[sigmoid+bias, experts={num_experts}, topk={topk}]",
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

    @pytest.mark.parametrize("accumulate", [True, False])
    def test_global_batch_quantile_router_replays(self, accumulate):
        """Include the router's nonpersistent histogram in the replay comparison."""
        self._init()
        seeded()
        config = _moe_config(
            num_moe_experts=64,
            moe_router_topk=8,
            moe_router_load_balancing_type="quantile_balancing",
            moe_router_quantile_balancing_estimation_scope="global_batch",
            moe_router_score_function="sigmoid",
            moe_aux_loss_coeff=0.0,
        )
        router = TopKRouter(
            config, pg_collection=ProcessGroupCollection.use_mpu_process_groups()
        ).cuda()
        router.train(accumulate)
        with torch.no_grad():
            router.expert_bias.copy_(torch.linspace(-0.1, 0.1, 64, device="cuda"))
        logits = torch.randn(8192, 64, device="cuda", requires_grad=True)

        def fn(logits):
            router.qb_histogram.zero_()
            probs, routing_map = router.quantile_balancing(logits)
            return probs, routing_map, router.qb_histogram.clone()

        outputs, grads = assert_replays_bit_exact(
            fn,
            (logits,),
            grad_outputs={"out[0]": torch.randn_like(logits)},
            replays=3,
            contention=True,
            what=f"global-batch TopKRouter[accumulate={accumulate}]",
        )
        assert torch.all(outputs["out[2]"].sum(dim=-1) == (8192 if accumulate else 0))
        assert grads["in[0]"].abs().sum() > 0

    @pytest.mark.skipif(not HAVE_TE, reason="TE grouped MLP needs Transformer Engine")
    @pytest.mark.parametrize(
        "use_grouped_tensor",
        [
            False,
            pytest.param(
                True,
                marks=pytest.mark.skipif(
                    not te_ext._TE_GROUPED_LINEAR_SUPPORTS_GROUPED_TENSOR,
                    reason="Installed TE GroupedLinear does not expose use_grouped_tensor",
                ),
            ),
        ],
        ids=["multi-stream", "device-initiated"],
    )
    def test_te_grouped_mlp_replays_on_uneven_experts(self, use_grouped_tensor):
        self._init()
        seeded()
        config = _moe_config(
            hidden_size=2048, ffn_hidden_size=4096, moe_use_grouped_tensor=use_grouped_tensor
        )
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
            what=f"TEGroupedMLP[grouped_tensor={use_grouped_tensor}]",
        )

    @pytest.mark.parametrize("gated", [False, True], ids=["squared_relu", "swiglu"])
    def test_inference_nvfp4_moe_layer_replays(self, gated):
        """Inference MoELayer through FlashInfer per-token NVFP4 squared-ReLU or SwiGLU experts."""
        from megatron.core.inference.moe.flashinfer_nvfp4 import HAVE_FLASHINFER_ROUTED_NVFP4
        from megatron.core.inference.utils import InferenceMode
        from megatron.core.models.gpt.moe_module_specs import get_inference_optimized_moe_spec
        from megatron.core.transformer.moe.token_dispatcher_inference import NCCLAllGatherDispatcher

        if not HAVE_FLASHINFER_ROUTED_NVFP4 or torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("per-token NVFP4 MoE needs FlashInfer >= 0.7.0 and Blackwell")

        class _InInferenceMode(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, hidden):
                with InferenceMode.active():
                    return self.module(hidden)

        self._init(ep=1)
        seeded()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=1024,
            num_attention_heads=8,
            num_moe_experts=8,
            moe_ffn_hidden_size=1024,
            moe_router_topk=2,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_dtype="fp32",
            moe_grouped_gemm=True,
            activation_func=F.silu if gated else squared_relu,
            gated_linear_unit=gated,
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_cpu_initialization=True,
            normalization="RMSNorm",
            transformer_impl="inference_optimized",
            inference_grouped_gemm_backend="flashinfer",
            inference_moe_token_dispatcher_type="nccl",
            inference_moe_nvfp4_recipe="per_token",
        )
        NCCLAllGatherDispatcher.allocate_buffers()
        layer = get_inference_optimized_moe_spec()(config=config).cuda().eval()
        hidden = torch.randn(512, 1, 1024, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad(), deterministic_algorithms(True):
            assert_module_replays_bit_exact(
                _InInferenceMode(layer),
                (hidden,),
                replays=3,
                backward=False,
                contention=True,
                what="MoELayer[inference, flashinfer per-token NVFP4]",
            )
        assert layer.experts._nvfp4_weights is not None

    @pytest.mark.skipif(not HAVE_TE, reason="TE grouped MLP needs Transformer Engine")
    @pytest.mark.parametrize(
        "offload_modules,offloaded_group",
        [
            (["expert_fc1", "moe_act"], "fused_group_mlp"),
            (["expert_fc1"], "expert_fc1"),
            (["moe_act"], "moe_act"),
        ],
        ids=["fc1+act->one_group", "fc1", "act"],
    )
    def test_te_grouped_mlp_replays_under_fine_grained_offloading(
        self, offload_modules, offloaded_group
    ):
        """Unfused ``TEGroupedMLP`` with its activations really offloaded to CPU and reloaded.

        Two experts modules run back to back because the offload manager always keeps the
        last group of each name resident; the first module's group is the one that travels.
        With both unfused knobs set, fc1 input and activation input go as ONE group named like
        the fused path's (``fused_group_mlp``); a single knob keeps its own group.
        """
        self._init()
        seeded()
        PipelineOffloadManager.reset_instance()
        config = _moe_config(
            hidden_size=2048,
            ffn_hidden_size=4096,
            fine_grained_activation_offloading=True,
            offload_modules=offload_modules,
            min_offloaded_tensor_size=1024,
        )
        spec = get_gpt_layer_with_transformer_engine_spec(num_experts=8, moe_grouped_gemm=True)

        def build():
            experts = get_submodules(spec.submodules.mlp).experts(
                num_local_experts=8,
                config=config,
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            )
            assert isinstance(experts, TEGroupedMLP)
            assert not experts._with_fused_impl
            return experts.cuda()

        stack = torch.nn.ModuleList([build(), build()])
        tokens_per_expert = torch.tensor([4096, 17, 0, 2048, 1, 8191, 33, 1998], dtype=torch.int64)
        rows = int(tokens_per_expert.sum())
        hidden = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16)
        probs = torch.rand(rows, device="cuda")

        def run():
            FineGrainedActivationOffloadingInterface.init_chunk_handler(
                pp_rank=0,
                vp_size=None,
                vp_stage=None,
                min_offloaded_tensor_size=1024,
                delta_offload_bytes_across_pp_ranks=0,
                activation_offload_fraction=1.0,
            )
            for p in stack.parameters():
                p.grad = None
            h = hidden.clone().requires_grad_(True)
            pr = probs.clone().requires_grad_(True)
            out, _ = stack[0](h, tokens_per_expert, pr)
            out, _ = stack[1](out, tokens_per_expert, pr)
            out.backward(torch.ones_like(out))
            torch.cuda.synchronize()
            FineGrainedActivationOffloadingInterface.reset()
            return [out.detach().clone(), h.grad.clone(), pr.grad.clone()] + [
                p.grad.detach().clone() for p in stack.parameters()
            ]

        try:
            # The warmup iteration records the groups; the extra reset applies the
            # post-warmup offload decisions.
            run()
            FineGrainedActivationOffloadingInterface.reset()
            offloaded = PipelineOffloadManager.get_instance().offload_summary_bytes
            assert (
                offloaded.get(offloaded_group, 0) > 0
            ), f"nothing offloaded under group {offloaded_group!r}; summary: {dict(offloaded)}"
            assert set(offloaded) == {offloaded_group}, dict(offloaded)
            names = ["out", "grad_hidden", "grad_probs"] + [
                f"grad[{n}]" for n, _ in stack.named_parameters()
            ]
            ref = run()
            for i in range(1, 4):
                for name, a, b in zip(names, ref, run()):
                    assert bytes_equal(a, b), f"TEGroupedMLP[offload] replay {i}: {name} differs"
        finally:
            PipelineOffloadManager.reset_instance()

    @pytest.mark.skipif(
        not hasattr(torch, "float8_e8m0fnu") or torch.cuda.get_device_capability()[0] < 10,
        reason="MXFP8 parameter storage needs Blackwell",
    )
    @pytest.mark.parametrize("backend", ["vllm", "flashinfer"])
    def test_mxfp8_weight_stacking_replays(self, backend):
        """Canonical and CUTLASS execution expert stacks replay bit-exactly."""
        from types import SimpleNamespace

        from megatron.core.inference.moe import InferenceGroupedGemmBackend
        from megatron.core.inference.moe.flashinfer_mxfp8 import HAVE_FLASHINFER_CUTLASS_MXFP8
        from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
        from megatron.core.transformer.moe.experts import InferenceGroupedMLP

        if backend == "flashinfer" and not HAVE_FLASHINFER_CUTLASS_MXFP8:
            pytest.skip("requires FlashInfer CUTLASS MXFP8")

        class GroupedMLPStub:
            _stack_mxfp8_linear_weight = InferenceGroupedMLP._stack_mxfp8_linear_weight

        seeded()
        weights = tuple(
            torch.randn(128, 128, device="cuda", dtype=torch.bfloat16) for _ in range(4)
        )

        def build_stacks(*bf16_weights):
            grouped = GroupedMLPStub()
            grouped.num_local_experts = 2
            grouped.inference_grouped_gemm_backend = InferenceGroupedGemmBackend.from_config(
                backend
            )
            grouped._activation_clamp_scale = 16.0
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
            build_stacks, weights, backward=False, what=f"{backend} MXFP8 expert-weight stacking"
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
                {"moe_shared_expert_intermediate_size": 2048},
                id="alltoall-native-shared-expert",
            ),
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
            pytest.param(
                "flex",
                2,
                {"moe_flex_dispatcher_backend": "deepepv2"},
                id="flex-deepepv2-ep2",
                marks=pytest.mark.skipif(not HAVE_DEEP_EP_V2, reason="DeepEP v2 not installed"),
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
            prev_fill = torch.utils.deterministic.fill_uninitialized_memory
            try:
                if extra.get("moe_flex_dispatcher_backend") in ("deepep", "deepepv2"):
                    # PyTorch memory-fill kernels can race with DeepEP communication.
                    torch.utils.deterministic.fill_uninitialized_memory = False
                assert_module_replays_bit_exact(
                    layer,
                    inputs,
                    replays=3,
                    contention=True,
                    what=f"MoELayer[{dispatcher}, ep={ep}]",
                )
            finally:
                torch.utils.deterministic.fill_uninitialized_memory = prev_fill

    def test_alltoall_dispatcher_releases_router_probs_after_forward(self):
        """combine_preprocess drops ``self.probs`` so it does not pin the recompute graph.

        The all-to-all dispatcher sets ``self.probs`` during dispatch and clears it at the end
        of ``combine_preprocess`` (the last consumer of router probs). After a forward pass the
        reference must be gone, otherwise the recomputed MoE forward graph is retained until the
        next iteration.
        """
        self._init(ep=1)
        seeded()
        config = _moe_config(moe_token_dispatcher_type="alltoall", expert_model_parallel_size=1)
        if not HAVE_TE:
            config.moe_grouped_gemm = False
            mlp_spec = get_gpt_layer_local_submodules(num_experts=8, moe_grouped_gemm=False).mlp
        else:
            mlp_spec = get_gpt_layer_with_transformer_engine_spec(
                num_experts=8, moe_grouped_gemm=True
            ).submodules.mlp
        layer = MoELayer(config, get_submodules(mlp_spec)).cuda()
        layer.set_layer_number(0)
        hidden = torch.randn(2048, 2, 1024, device="cuda", dtype=torch.bfloat16)
        with deterministic_algorithms(True):
            layer(hidden)
        assert layer.token_dispatcher.probs is None


@pytest.mark.parametrize("collective", ["ordered", "multimem"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_batch_invariant_unpermute_replays(monkeypatch, collective, dtype):
    """The fused batch-invariant unpermute (``probs=None``) and its backward replay bit-exactly."""
    if not batch_invariant.HAVE_TRITON:
        pytest.skip("requires Triton")
    seeded()
    monkeypatch.setattr(batch_invariant, "get_batch_invariant_collective", lambda: collective)
    monkeypatch.setattr(parallel_state, "get_expert_model_parallel_world_size", lambda: 2)
    num_tokens, hidden = 4096, 520
    rows = torch.randn(2 * num_tokens, hidden, device="cuda", dtype=dtype, requires_grad=True)
    # Two experts on two EP ranks, one slot each: every token sums one row per rank.
    inverse_map = torch.stack(
        (
            torch.arange(2 * num_tokens, device="cuda").view(2, num_tokens).T,
            torch.tensor([0, 1], device="cuda").expand(num_tokens, 2),
        )
    )
    grad = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype)
    grad.view(-1)[::7] = -0.0

    def combine(x):
        return batch_invariant.unpermute(
            x, torch.Size((num_tokens, hidden)), probs=None, num_experts=2, inverse_map=inverse_map
        )

    assert_replays_bit_exact(
        combine,
        (rows,),
        replays=3,
        contention=True,
        grad_outputs={"out": grad},
        what=f"batch-invariant unpermute[{collective=}, {dtype=}]",
    )
