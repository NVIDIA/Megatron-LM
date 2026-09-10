# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Cross-rank correctness for the virtual-expert weight-transfer kernels.

Run on one four-GPU NVLink node::

    uv run python -m torch.distributed.run --nproc-per-node 4 -m pytest -q \
      tests/unit_tests/transformer/moe/test_virtual_expert_triton.py

These tests cover what the kernels put on the wire. Wire *bandwidth* is not measured here;
``bench_virtual_expert_weight_sol.py`` is the benchmark of record and sweeps plan occupancy,
which is what actually moves the number.
"""

import gc
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.transformer.moe.virtual_expert_load_balancer import plan_virtual_expert_routes
from megatron.core.transformer.moe.virtual_expert_triton import (
    MAX_VIRTUAL_EXPERT_WEIGHT_SMS,
    VirtualExpertPlannerWorkspace,
    _transport_tile,
    _validate_transport_shape,
    launch_virtual_expert_grad_reduce,
    launch_virtual_expert_weight_prefetch,
)
from tests.unit_tests.test_utilities import Utils

# The GB200 CI bucket launches marked files with four ranks, which these tests need.
pytestmark = pytest.mark.launch_on_gb200

NUM_SMS = 4
requires_four_ranks = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4 or not torch.cuda.is_available(),
    reason="Virtual-expert transport coverage requires a 4-rank torchrun launch on CUDA",
)


def test_virtual_expert_transport_shape_guards():
    """Reject launches the transport kernels cannot serve."""
    with pytest.raises(ValueError, match="limited to 32 SMs"):
        _validate_transport_shape(
            world_size=4, num_local_experts=32, num_sms=MAX_VIRTUAL_EXPERT_WEIGHT_SMS + 1
        )
    # Both FC layers share one row-aligned tile, so an odd member breaks it.
    with pytest.raises(ValueError, match="256-aligned"):
        _transport_tile(32768 // torch.bfloat16.itemsize, 16384, 16385)


def _allocate_symmetric(numel, dtype, group):
    """Allocate and rendezvous one native NCCL symmetric-memory tensor."""
    import torch.distributed._symmetric_memory as symm_mem

    device = torch.device("cuda", torch.cuda.current_device())
    dist.barrier(group=group, device_ids=[device.index])
    if not group._get_backend(device)._comm_ptr():
        raise RuntimeError("NCCL communicator is unavailable for symmetric memory.")
    if symm_mem.get_backend(device) != "NCCL":
        symm_mem.set_backend("NCCL")
    tensor = symm_mem.empty(numel, dtype=dtype, device=device)
    return tensor, symm_mem.rendezvous(tensor, group)


def _pointer_table(members):
    """Return the ``int64`` per-expert base-address table the kernels consume."""
    return torch.tensor(
        [members[index].data_ptr() for index in range(members.shape[0])],
        dtype=torch.int64,
        device=members.device,
    )


def _arena_view(arena, member_numels, scale_numels, fc_layer, num_local_experts):
    """Return the ``[num_local_experts, numel]`` data and scale views of one FC layer."""
    stride = (
        member_numels
        if scale_numels is None
        else tuple(member + scale for member, scale in zip(member_numels, scale_numels))
    )
    offset = num_local_experts * sum(stride[:fc_layer])
    member = member_numels[fc_layer]
    data = arena.narrow(0, offset, num_local_experts * member).view(num_local_experts, member)
    if scale_numels is None:
        return data, None
    scale = arena.narrow(
        0, offset + num_local_experts * member, num_local_experts * scale_numels[fc_layer]
    ).view(num_local_experts, scale_numels[fc_layer])
    return data, scale


def _check_ends(view, expected_per_slot, label, errors):
    """Compare the first and last element of every slot row against its expectation."""
    for slot, expected in enumerate(expected_per_slot):
        for column, end in ((0, "head"), (-1, "tail")):
            actual = view[slot, column].item()
            if actual != expected:
                errors.append(f"{label} slot={slot} {end}: got {actual}, expected {expected}")


def _report(errors, group):
    """Fail on every rank if any rank saw a mismatch."""
    gathered = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(gathered, errors, group=group)
    combined = [error for rank_errors in gathered for error in rank_errors]
    assert not combined, "\n".join(combined)


def _make_plan(placement, slots, world_size, num_local_experts, device):
    """Build one ``[world_size, num_local_experts]`` virtual-expert assignment."""
    plan = torch.full((world_size, num_local_experts), -1, dtype=torch.int32, device=device)
    if placement == "asymmetric":
        # Only two ranks receive anything, and only into slot 0.
        plan[0, slots[0]] = num_local_experts
        plan[1, slots[0]] = 2 * num_local_experts
        return plan
    for destination in range(world_size):
        peers = [peer for peer in range(world_size) if peer != destination]
        for ordinal, slot in enumerate(slots):
            owner = (
                (destination + 1) % world_size
                if placement == "ring"
                else peers[ordinal % len(peers)]
            )
            plan[destination, slot] = owner * num_local_experts + slot
    return plan


@pytest.mark.internal
@requires_four_ranks
@pytest.mark.parametrize(
    "grad_dtype", [torch.float32, torch.bfloat16], ids=["fp32-grad", "bf16-grad"]
)
def test_virtual_expert_weight_transport(grad_dtype):
    """Push BF16 weights and reduce virtual-expert gradients over full, sparse and empty plans."""
    Utils.initialize_distributed()
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.device("cuda", torch.cuda.current_device())
    num_local_experts = 8
    # Keep the test compact while using the same 8-KiB-aligned transactions as
    # the production 2048x640 expert FC layers.
    member_numels = (262144, 524288)
    arena_numel = num_local_experts * sum(member_numels)
    weight_arena, weight_handle = _allocate_symmetric(arena_numel, torch.bfloat16, group)
    grad_arena, grad_handle = _allocate_symmetric(arena_numel, grad_dtype, group)
    sources = tuple(
        torch.empty(num_local_experts, member, dtype=torch.bfloat16, device=device)
        for member in member_numels
    )
    for fc_layer, source in enumerate(sources):
        source.copy_(
            (
                torch.arange(num_local_experts, dtype=torch.bfloat16, device=device)
                + rank * num_local_experts
                + fc_layer * 1000
            )[:, None]
        )
    main_grads = tuple(
        torch.empty(num_local_experts, member, dtype=grad_dtype, device=device)
        for member in member_numels
    )
    workspace = SimpleNamespace(
        weight_arena=weight_arena,
        weight_handle=weight_handle,
        weight_grid_barrier=torch.zeros(1, dtype=torch.int32, device=device),
        grad_arena=grad_arena,
        grad_handle=grad_handle,
        grad_grid_barrier=torch.zeros(1, dtype=torch.int32, device=device),
        rank=rank,
        world_size=world_size,
        num_local_experts=num_local_experts,
        member_numels=member_numels,
        num_sms=NUM_SMS,
    )

    cases = (
        ("all-peers", tuple(range(num_local_experts))),
        ("ring", tuple(range(num_local_experts))),
        ("all-peers", tuple()),
        ("all-peers", (0, 3, 7)),
        ("asymmetric", (0,)),
    )
    errors = []

    def scalar(value):
        """Round a reference value the same way the kernel's store does."""
        return torch.tensor(value, dtype=grad_dtype).item()

    try:
        for placement, slots in cases:
            case = f"{placement}/{slots}"
            plan = _make_plan(placement, slots, world_size, num_local_experts, device)
            rows = plan.tolist()
            local_slots = tuple(slot for slot in range(num_local_experts) if rows[rank][slot] >= 0)

            weight_arena.fill_(-123)
            torch.cuda.synchronize(device)
            dist.barrier(group=group, device_ids=[device.index])
            launch_virtual_expert_weight_prefetch(
                workspace,
                sources=tuple(_pointer_table(source) for source in sources),
                experts_to_copy=plan,
            )
            torch.cuda.synchronize(device)

            for fc_layer in range(len(member_numels)):
                view, _ = _arena_view(
                    weight_arena, member_numels, None, fc_layer, num_local_experts
                )
                _check_ends(
                    view,
                    [
                        torch.tensor(
                            -123 if rows[rank][slot] < 0 else fc_layer * 1000 + rows[rank][slot],
                            dtype=torch.bfloat16,
                        ).item()
                        for slot in range(num_local_experts)
                    ],
                    f"{case} p{fc_layer} weight",
                    errors,
                )

            grad_arena.fill_(-77)
            for fc_layer in range(len(member_numels)):
                view, _ = _arena_view(grad_arena, member_numels, None, fc_layer, num_local_experts)
                for slot in local_slots:
                    view[slot].fill_(fc_layer * 1000 + rank * 100 + slot + 1)
                main_grads[fc_layer].fill_(fc_layer + 5)
            torch.cuda.synchronize(device)
            dist.barrier(group=group, device_ids=[device.index])
            launch_virtual_expert_grad_reduce(
                workspace,
                native_grads=tuple(_pointer_table(grad) for grad in main_grads),
                experts_to_copy=plan,
            )
            torch.cuda.synchronize(device)

            for fc_layer in range(len(member_numels)):
                # BF16 partials accumulate in FP32 registers and round once on
                # the final store, so build the reference the same way.
                expected = torch.full(
                    (num_local_experts,), scalar(fc_layer + 5), dtype=torch.float32, device=device
                )
                for destination in range(world_size):
                    for slot in range(num_local_experts):
                        expert = rows[destination][slot]
                        if expert // num_local_experts == rank and expert >= 0:
                            expected[expert % num_local_experts] += scalar(
                                fc_layer * 1000 + destination * 100 + slot + 1
                            )
                try:
                    torch.testing.assert_close(
                        main_grads[fc_layer][:, 0], expected.to(grad_dtype), rtol=0, atol=0
                    )
                except AssertionError as exc:
                    errors.append(f"{case} p{fc_layer} main_grad: {exc}")

                # The reduction reads the slots and leaves them as they were;
                # TE's overwriting wgrad GEMM refreshes them next backward.
                view, _ = _arena_view(grad_arena, member_numels, None, fc_layer, num_local_experts)
                _check_ends(
                    view,
                    [
                        scalar(
                            fc_layer * 1000 + rank * 100 + slot + 1 if slot in local_slots else -77
                        )
                        for slot in range(num_local_experts)
                    ],
                    f"{case} p{fc_layer} grad",
                    errors,
                )
        _report(errors, group)
    finally:
        dist.barrier(group=group, device_ids=[device.index])
        del weight_arena, grad_arena, weight_handle, grad_handle
        gc.collect()
        Utils.destroy_model_parallel()


@pytest.mark.internal
@requires_four_ranks
def test_virtual_expert_mxfp8_transport_moves_one_orientation_at_a_time():
    """Copy MXFP8 bytes and scales exactly without touching the other GEMM orientation."""
    Utils.initialize_distributed()
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.device("cuda", torch.cuda.current_device())
    num_local_experts = 4
    member_numels = (16384, 32768)
    scale_numels = tuple(member // 32 for member in member_numels)
    arena_numel = num_local_experts * sum(
        member + scale for member, scale in zip(member_numels, scale_numels)
    )

    arenas = {}
    handles = {}
    for orientation in ("rowwise", "columnwise"):
        arenas[orientation], handles[orientation] = _allocate_symmetric(
            arena_numel, torch.uint8, group
        )
    # Distinct byte ranges per orientation and component: any crossed wire shows
    # up as a wrong value rather than a coincidental match.
    bases = {("rowwise", "data"): 1, ("rowwise", "scale"): 65}
    bases.update({("columnwise", "data"): 129, ("columnwise", "scale"): 193})
    sources = {}
    for (orientation, kind), base in bases.items():
        numels = member_numels if kind == "data" else scale_numels
        tensors = tuple(
            torch.empty(num_local_experts, numel, dtype=torch.uint8, device=device)
            for numel in numels
        )
        for fc_layer, tensor in enumerate(tensors):
            for expert in range(num_local_experts):
                tensor[expert].fill_(base + rank * num_local_experts + expert + 20 * fc_layer)
        sources[(orientation, kind)] = tensors

    # Every rank materializes its right-hand neighbour's whole expert set.
    plan = torch.empty((world_size, num_local_experts), dtype=torch.int32, device=device)
    for destination in range(world_size):
        owner = (destination + 1) % world_size
        plan[destination] = torch.arange(
            owner * num_local_experts,
            (owner + 1) * num_local_experts,
            dtype=torch.int32,
            device=device,
        )
    workspaces = {
        orientation: SimpleNamespace(
            weight_arena=arenas[orientation],
            weight_handle=handles[orientation],
            weight_grid_barrier=torch.zeros(1, dtype=torch.int32, device=device),
            rank=rank,
            world_size=world_size,
            num_local_experts=num_local_experts,
            member_numels=member_numels,
            num_sms=NUM_SMS,
        )
        for orientation in arenas
    }

    def launch(orientation):
        dist.barrier(group=group, device_ids=[device.index])
        launch_virtual_expert_weight_prefetch(
            workspaces[orientation],
            sources=tuple(_pointer_table(s) for s in sources[(orientation, "data")]),
            scale_sources=tuple(_pointer_table(s) for s in sources[(orientation, "scale")]),
            experts_to_copy=plan,
        )
        torch.cuda.synchronize(device)

    def verify(orientation):
        owner = (rank + 1) % world_size
        for fc_layer in range(len(member_numels)):
            data, scale = _arena_view(
                arenas[orientation], member_numels, scale_numels, fc_layer, num_local_experts
            )
            experts = torch.arange(
                owner * num_local_experts,
                (owner + 1) * num_local_experts,
                dtype=torch.int64,
                device=device,
            )
            for view, kind in ((data, "data"), (scale, "scale")):
                expected = (experts + bases[(orientation, kind)] + 20 * fc_layer).to(torch.uint8)
                for column, label in ((0, "head"), (-1, "tail")):
                    torch.testing.assert_close(
                        view[:, column],
                        expected,
                        rtol=0,
                        atol=0,
                        msg=lambda msg: f"{orientation} p{fc_layer} {kind} {label}: {msg}",
                    )

    try:
        arenas["rowwise"].fill_(17)
        arenas["columnwise"].fill_(23)
        launch("rowwise")
        verify("rowwise")
        # Forward pushes the rowwise orientation only; the backward arena must
        # still hold its fill.
        torch.testing.assert_close(
            arenas["columnwise"], torch.full_like(arenas["columnwise"], 23), rtol=0, atol=0
        )

        rowwise_snapshot = arenas["rowwise"].clone()
        launch("columnwise")
        verify("columnwise")
        torch.testing.assert_close(arenas["rowwise"], rowwise_snapshot, rtol=0, atol=0)
    finally:
        dist.barrier(group=group, device_ids=[device.index])
        arenas.clear()
        handles.clear()
        gc.collect()
        Utils.destroy_model_parallel()


@requires_four_ranks
def test_virtual_expert_histogram_exchange_matches_all_gather():
    """The planner's symmetric-memory histogram exchange gathers what an all-gather gathers,
    launch after launch (the flags self-reset)."""
    Utils.initialize_distributed()
    group = dist.group.WORLD
    rank, world_size = dist.get_rank(group), dist.get_world_size(group)
    device = torch.device("cuda", torch.cuda.current_device())
    num_tokens, topk, num_experts = 512, 3, 4 * world_size
    workspace = VirtualExpertPlannerWorkspace(num_experts=num_experts, device=device, group=group)
    generator = torch.Generator(device=device).manual_seed(77 + rank)
    try:
        for launch in range(3):
            weights = torch.rand(num_experts, device=device, generator=generator) + 0.05
            weights[(rank + launch) % num_experts] = 10.0  # a hot expert per rank migrates
            indices = torch.multinomial(
                weights.expand(num_tokens, num_experts), topk, generator=generator
            )
            probs = torch.rand((num_tokens, topk), device=device, generator=generator)
            plan, runtime_probs = plan_virtual_expert_routes(indices, probs, workspace)
            # Snapshot the window in stream order: a peer may publish its next histogram into
            # it as soon as this rank's placement has read it (the all-gather below is what
            # keeps the peers from getting further ahead than that).
            gathered = workspace.gathered_counts.clone()
            histogram = torch.bincount(indices.reshape(-1), minlength=num_experts).int()
            expected = torch.empty_like(gathered)
            dist.all_gather_into_tensor(expected.view(-1), histogram, group=group)
            torch.cuda.synchronize(device)
            assert torch.equal(gathered, expected)
            # Every route lands on some rank's runtime expert; the mapping's exactness against
            # a torch reference is tier 1's job.
            assert plan.virtual_experts.shape == (num_tokens, topk)
            assert (
                0 <= int(plan.virtual_experts.min())
                and int(plan.virtual_experts.max()) < 2 * num_experts
            )
            assert torch.equal(runtime_probs.gather(1, plan.virtual_experts.long()), probs)
    finally:
        workspace.destroy()
        dist.barrier(group=group, device_ids=[device.index])


# --------------------------------------------------------------------------------------
# Planner: placement and route mapping through the real histogram exchange
# --------------------------------------------------------------------------------------

# One fixed planner shape drives every placement test: four ranks owning two
# experts each, sixteen tokens picking three distinct experts.
EP_SIZE = 4
NUM_LOCAL_EXPERTS = 2
NUM_EXPERTS = EP_SIZE * NUM_LOCAL_EXPERTS
NUM_TOKENS = 16
ROUTER_TOPK = 3
NUM_ROUTES = NUM_TOKENS * ROUTER_TOPK


def _routes_for_skew(skew: str, num_tokens: int = NUM_TOKENS) -> torch.Tensor:
    """Return ``[ep_size, num_tokens, router_topk]`` semantic routes for one load skew."""
    weights = torch.ones(NUM_EXPERTS)
    if skew == "hot_expert":
        weights[0] = 20.0
    elif skew == "hot_rank":
        weights[:NUM_LOCAL_EXPERTS] = 12.0
    elif skew == "two_ranks_own_everything":
        # Only ranks 0 and 1 hold routed experts, so ranks 2 and 3 must receive
        # a full capacity each from a single sender: the tightest slot fit the
        # single-sender migration rule allows.
        weights[ROUTER_TOPK:] = 0.0
    elif skew != "balanced":
        raise ValueError(f"Unknown skew {skew!r}.")
    generator = torch.Generator().manual_seed(1234)
    return torch.stack(
        [
            torch.stack(
                [
                    torch.multinomial(weights, ROUTER_TOPK, replacement=False, generator=generator)
                    for _ in range(num_tokens)
                ]
            )
            for _ in range(EP_SIZE)
        ]
    )


def _histogram(routes: torch.Tensor) -> torch.Tensor:
    """Return the ``[ep_size, num_experts]`` route histogram of every rank."""
    return torch.stack(
        [torch.bincount(rank.reshape(-1), minlength=NUM_EXPERTS) for rank in routes]
    ).to(torch.int32)


def _reference_map_routes(
    routes: torch.Tensor, workspace: VirtualExpertPlannerWorkspace
) -> torch.Tensor:
    """Torch oracle for the planner's route mapping: stable sort by expert, one global array of
    segment ends, then the runtime id of every route."""
    ep_size, num_experts = workspace.gathered_counts.shape
    num_local_experts = num_experts // ep_size
    flat = routes.reshape(-1)
    experts, order = torch.sort(flat, stable=True)
    tokens_per_expert = torch.bincount(flat, minlength=num_experts)
    bucket_start = torch.cumsum(tokens_per_expert, 0) - tokens_per_expert
    boundaries = (
        workspace.field("destination_boundaries")[:, :ep_size]
        .clamp(min=0)
        .minimum(tokens_per_expert[:, None])
    )
    ends = (bucket_start[:, None] + boundaries).reshape(-1)
    positions = torch.arange(flat.numel(), device=flat.device, dtype=torch.int64)
    destination = torch.searchsorted(ends, positions, right=True) - experts * ep_size
    slot = workspace.field("virtual_expert_slots").view(-1)[experts * ep_size + destination]
    runtime_local = torch.where(
        destination == experts // num_local_experts,
        experts % num_local_experts,
        num_local_experts + slot,
    )
    virtual = torch.empty_like(positions)
    virtual[order] = destination * (2 * num_local_experts) + runtime_local
    return virtual.view(routes.shape)


def _plan_on_this_rank(routes, probs=None):
    """Plan this rank's routes of a complete four-rank group through the real histogram
    exchange, and check what a torch oracle can reproduce: the local histogram, the exchanged
    window and the route mapping. The caller destroys the returned workspace."""
    Utils.initialize_distributed()
    group = dist.group.WORLD
    assert dist.get_world_size(group) == EP_SIZE
    rank = dist.get_rank(group)
    device = torch.device("cuda", torch.cuda.current_device())
    workspace = VirtualExpertPlannerWorkspace(num_experts=NUM_EXPERTS, device=device, group=group)
    own = routes[rank].to(device=device, dtype=torch.int64)
    if probs is None:
        probs = torch.rand(own.shape, device=device)
    plan, runtime_probs = plan_virtual_expert_routes(own, probs, workspace)
    torch.cuda.synchronize(device)
    counts = _histogram(routes).to(device)
    assert torch.equal(workspace.gathered_counts, counts)
    assert torch.equal(plan.virtual_experts.long(), _reference_map_routes(own, workspace))
    return workspace, plan, runtime_probs, rank, group


def _release(workspace, group):
    workspace.destroy()
    dist.barrier(group=group, device_ids=[torch.cuda.current_device()])


@pytest.mark.internal
@requires_four_ranks
@pytest.mark.parametrize("skew", ["balanced", "hot_expert", "hot_rank", "two_ranks_own_everything"])
def test_virtual_expert_placement_balances_every_destination(skew):
    """Equalize route load across ranks without over-subscribing virtual-expert slots."""
    routes = _routes_for_skew(skew)
    counts = _histogram(routes)
    workspace, plan, _, _, group = _plan_on_this_rank(routes)
    try:
        allocation = workspace.field("allocation").cpu()
        experts_to_copy = plan.experts_to_copy.cpu()
        global_per_expert = counts.sum(0).to(torch.int32)

        # Every rank executes exactly its own route capacity. The dispatcher's
        # dropless rank capacity is derived from this equality, so a placement that
        # merely reduced the imbalance would silently drop real routes.
        torch.testing.assert_close(
            allocation.sum(0).to(torch.int32),
            torch.full((EP_SIZE,), NUM_ROUTES, dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        # Migration moves routes; it never invents or loses them.
        torch.testing.assert_close(
            allocation.sum(1).to(torch.int32), global_per_expert, rtol=0, atol=0
        )
        assert (allocation >= 0).all()
        torch.testing.assert_close(
            workspace.field("balance").cpu(),
            global_per_expert.view(EP_SIZE, NUM_LOCAL_EXPERTS).sum(1).to(torch.int32) - NUM_ROUTES,
            rtol=0,
            atol=0,
        )
        for destination in range(EP_SIZE):
            slots = experts_to_copy[destination].tolist()
            migrated = {
                expert
                for expert in range(NUM_EXPERTS)
                if expert // NUM_LOCAL_EXPERTS != destination
                and allocation[expert, destination] > 0
            }
            filled = [expert for expert in slots if expert >= 0]
            # A remote expert that receives routes but owns no virtual-expert slot would
            # execute against a stale slot's weights. The single-sender rule bounds
            # this set by num_local_experts; the device assert covers the rest.
            assert (
                set(filled) == migrated
            ), f"rank {destination} slots {slots} vs migrated {migrated}"
            assert len(filled) == len(set(filled))
            assert all(expert // NUM_LOCAL_EXPERTS != destination for expert in filled)
            # The slot table is written only for assigned slots; check exactly those.
            for slot, expert in enumerate(slots):
                if expert >= 0:
                    assert int(workspace.field("virtual_expert_slots")[expert, destination]) == slot

        # Every rank computes the placement independently and must agree exactly.
        for tensor in (
            workspace.field("balance"),
            workspace.field("allocation"),
            plan.experts_to_copy,
        ):
            tensor = tensor.contiguous()
            gathered = [torch.empty_like(tensor) for _ in range(EP_SIZE)]
            dist.all_gather(gathered, tensor, group=group)
            assert all(torch.equal(peer, tensor) for peer in gathered)
    finally:
        _release(workspace, group)


@pytest.mark.internal
@requires_four_ranks
@pytest.mark.parametrize("skew", ["hot_expert", "two_ranks_own_everything"])
def test_virtual_expert_planner_maps_every_route_to_the_expert_it_selected(skew):
    """Decode each virtual route back to the semantic expert and destination it was given."""
    routes = _routes_for_skew(skew)
    workspace, plan, _, rank, group = _plan_on_this_rank(routes)
    try:
        allocation = workspace.field("allocation").cpu()
        experts_to_copy = plan.experts_to_copy.cpu()
        observed = torch.zeros((NUM_EXPERTS, EP_SIZE), dtype=torch.int32)
        for route, virtual in zip(
            routes[rank].reshape(-1).tolist(), plan.virtual_experts.cpu().reshape(-1).tolist()
        ):
            destination, runtime_local = divmod(virtual, 2 * NUM_LOCAL_EXPERTS)
            if runtime_local < NUM_LOCAL_EXPERTS:
                assert (destination, runtime_local) == divmod(
                    route, NUM_LOCAL_EXPERTS
                ), f"route to expert {route} became native id {virtual}"
            else:
                slot = runtime_local - NUM_LOCAL_EXPERTS
                assert int(experts_to_copy[destination][slot]) == route, (
                    f"route to expert {route} became virtual-expert slot {slot} on rank "
                    f"{destination}, which holds expert "
                    f"{int(experts_to_copy[destination][slot])}"
                )
            observed[route, destination] += 1
        # The allocation names how many routes each destination owes each expert;
        # the routes every rank mapped must reproduce it exactly, not merely in aggregate.
        observed = observed.cuda()
        dist.all_reduce(observed, group=group)
        torch.testing.assert_close(observed.cpu(), allocation, rtol=0, atol=0)
    finally:
        _release(workspace, group)


@pytest.mark.internal
@requires_four_ranks
@pytest.mark.parametrize("skew", ["balanced", "hot_expert", "two_ranks_own_everything"])
@pytest.mark.parametrize("num_tokens", [NUM_TOKENS, 1000])
def test_virtual_expert_planner_writes_hybridep_inputs_with_probability_gradients(skew, num_tokens):
    """The planner's dense runtime probabilities match a torch scatter of the router's compact
    probabilities at the mapped runtime ids, and their gradient flows back to those entries."""
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device="cuda").manual_seed(4321)
    routes = _routes_for_skew(skew, num_tokens)
    probs = torch.rand((num_tokens, ROUTER_TOPK), device=device, generator=generator)
    probs = probs.requires_grad_(True)
    workspace, plan, runtime_probs, _, group = _plan_on_this_rank(routes, probs)
    try:
        expected = torch.zeros((num_tokens, 2 * NUM_EXPERTS), device=device)
        expected = expected.scatter(1, plan.virtual_experts.long(), probs)
        torch.testing.assert_close(runtime_probs, expected, rtol=0, atol=0)
        assert runtime_probs.requires_grad and not plan.virtual_experts.requires_grad
        assert plan.virtual_experts.dtype == torch.int16

        grad = torch.rand(runtime_probs.shape, device=device, generator=generator)
        (actual_grad,) = torch.autograd.grad(runtime_probs, probs, grad)
        (expected_grad,) = torch.autograd.grad(expected, probs, grad)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    finally:
        _release(workspace, group)
