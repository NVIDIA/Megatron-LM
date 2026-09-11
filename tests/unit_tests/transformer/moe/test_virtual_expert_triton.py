# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Cross-rank correctness for the virtual-expert weight-transfer kernels.

Run on one four-GPU NVLink node::

    uv run python -m torch.distributed.run --nproc-per-node 4 -m pytest -q \
      tests/unit_tests/transformer/moe/test_virtual_expert_triton.py

These tests cover transport correctness; they do not impose hardware-dependent bandwidth
thresholds or depend on external benchmark scripts.
"""

import gc
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.transformer.moe.virtual_expert_load_balancer import plan_virtual_expert_routes
from megatron.core.transformer.moe.virtual_expert_triton import (
    VirtualExpertPlannerWorkspace,
    _transport_tile,
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


def test_virtual_expert_transport_tile_alignment():
    """Both FC layers must support the same TMA row-aligned tile."""
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
    "grad_dtype,num_sms",
    [(torch.float32, 1), (torch.bfloat16, 32)],
    ids=["fp32-grad-1sm", "bf16-grad-32sm"],
)
def test_virtual_expert_weight_transport(grad_dtype, num_sms):
    """Check all transport tiles, signed reduction, split FC scheduling and untouched storage."""
    Utils.initialize_distributed()
    group = dist.group.WORLD
    rank, world_size = dist.get_rank(group), dist.get_world_size(group)
    device = torch.device("cuda", torch.cuda.current_device())
    num_local_experts = 8
    # Two/four BF16 transport tiles per member; smaller than the old endpoint-only fixture.
    member_numels = (32768, 65536)
    arena_numel = num_local_experts * sum(member_numels)
    guard = 256
    weight_arena, weight_handle = _allocate_symmetric(arena_numel + guard, torch.bfloat16, group)
    grad_arena, grad_handle = _allocate_symmetric(arena_numel + guard, grad_dtype, group)
    rng = torch.Generator(device=device).manual_seed(900 + rank)
    sources = tuple(
        (torch.randint(-128, 128, (num_local_experts, n), device=device, generator=rng) / 32).to(
            torch.bfloat16
        )
        for n in member_numels
    )
    # Each row has a guard after its last tile, including the final native expert.
    native_storage = tuple(
        torch.empty(num_local_experts, n + guard, dtype=grad_dtype, device=device)
        for n in member_numels
    )
    main_grads = tuple(storage[:, :n] for storage, n in zip(native_storage, member_numels))
    weight_tables = tuple(_pointer_table(source) for source in sources)
    grad_tables = tuple(_pointer_table(grad) for grad in main_grads)
    all_weights = []
    for source in sources:
        gathered = [torch.empty_like(source) for _ in range(world_size)]
        dist.all_gather(gathered, source, group=group)
        all_weights.append(torch.cat(gathered))
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
        num_sms=num_sms,
    )
    communication = torch.cuda.Stream()
    compute = torch.cuda.current_stream()
    cases = (
        ("all-peers", tuple(range(num_local_experts))),
        ("ring", tuple(range(num_local_experts))),
        ("all-peers", tuple()),
        ("all-peers", (0, 3, 7)),
        ("asymmetric", (0,)),
    )
    errors = []
    try:
        for placement, slots in cases:
            case = f"{placement}/{slots}"
            plan = _make_plan(placement, slots, world_size, num_local_experts, device)
            rows = plan.tolist()
            weight_arena.fill_(-123)
            expected_weights = weight_arena.clone()
            for fc, weights in enumerate(all_weights):
                view, _ = _arena_view(expected_weights, member_numels, None, fc, num_local_experts)
                for slot, expert in enumerate(rows[rank]):
                    if expert >= 0:
                        view[slot].copy_(weights[expert])
            # Push has an exit barrier only: all destinations must finish their canary fill.
            torch.cuda.synchronize(device)
            dist.barrier(group=group, device_ids=[device.index])
            communication.wait_stream(compute)
            with torch.cuda.stream(communication):
                launch_virtual_expert_weight_prefetch(
                    workspace, sources=weight_tables, experts_to_copy=plan
                )
            compute.wait_stream(communication)
            _check_equal(weight_arena, expected_weights, f"{case} weights", errors)

            grad_arena.fill_(-77)
            initials = []
            for fc, n in enumerate(member_numels):
                view, _ = _arena_view(grad_arena, member_numels, None, fc, num_local_experts)
                for slot, expert in enumerate(rows[rank]):
                    if expert >= 0:
                        view[slot].copy_(
                            torch.randint(-64, 65, (n,), device=device, generator=rng) / 8
                        )
                        # Rank 0/expert 0 receives 256 + 1 - 256 in the all-peer plan.
                        # With native 0.5 the answer is 1.5; intermediate BF16 rounding loses it.
                        view[slot, 0] = (0, 256, 1, -256)[rank]
                initial = (
                    torch.randint(-16, 17, (num_local_experts, n), device=device, generator=rng) / 8
                ).to(grad_dtype)
                initial[:, 0] = 0.5
                initials.append(initial)
            slot_snapshot = grad_arena.clone()
            partials = [torch.empty_like(grad_arena) for _ in range(world_size)]
            dist.all_gather(partials, grad_arena, group=group)
            expected = [initial.float().clone() for initial in initials]
            for peer, row in enumerate(rows):
                for slot, expert in enumerate(row):
                    if expert >= 0 and expert // num_local_experts == rank:
                        for fc in range(2):
                            view, _ = _arena_view(
                                partials[peer], member_numels, None, fc, num_local_experts
                            )
                            expected[fc][expert % num_local_experts].add_(view[slot].float())
            expected = [value.to(grad_dtype) for value in expected]
            for schedule in (((0, 1),), ((1,), (0,))):
                for storage, grad, initial in zip(native_storage, main_grads, initials):
                    storage.fill_(-99)
                    grad.copy_(initial)
                completed = set()
                for fc_layers in schedule:
                    communication.wait_stream(compute)
                    with torch.cuda.stream(communication):
                        launch_virtual_expert_grad_reduce(
                            workspace,
                            native_grads=grad_tables,
                            experts_to_copy=plan,
                            fc_layers=fc_layers,
                        )
                    compute.wait_stream(communication)
                    completed.update(fc_layers)
                    for fc, (storage, n) in enumerate(zip(native_storage, member_numels)):
                        target = torch.full_like(storage, -99)
                        target[:, :n] = expected[fc] if fc in completed else initials[fc]
                        _check_equal(storage, target, f"{case} {schedule} FC{fc + 1}", errors)
                    _check_equal(grad_arena, slot_snapshot, f"{case} unchanged partials", errors)
        _report(errors, group)
    finally:
        torch.cuda.synchronize(device)
        dist.barrier(group=group, device_ids=[device.index])
        del workspace, weight_arena, grad_arena, weight_handle, grad_handle
        gc.collect()
        Utils.destroy_model_parallel()


@pytest.mark.internal
@requires_four_ranks
def test_virtual_expert_mxfp8_transport_moves_one_orientation_at_a_time():
    """Overwrite one shared arena with each orientation, checking all bytes and inactive slots."""
    Utils.initialize_distributed()
    group = dist.group.WORLD
    rank, world_size = dist.get_rank(group), dist.get_world_size(group)
    device = torch.device("cuda", torch.cuda.current_device())
    num_local_experts = 4
    member_numels = (65536, 131072)
    scale_numels = tuple(member // 32 for member in member_numels)
    arena_numel = num_local_experts * sum(
        member + scale for member, scale in zip(member_numels, scale_numels)
    )
    arena, handle = _allocate_symmetric(arena_numel + 256, torch.uint8, group)
    workspace = SimpleNamespace(
        weight_arena=arena,
        weight_handle=handle,
        weight_grid_barrier=torch.zeros(1, dtype=torch.int32, device=device),
        rank=rank,
        world_size=world_size,
        num_local_experts=num_local_experts,
        member_numels=member_numels,
        num_sms=NUM_SMS,
    )
    # Seeded bytes vary across experts, components, orientations, tiles and offsets.
    rng = torch.Generator(device=device).manual_seed(1500 + rank)
    sources, tables, gathered_sources = {}, {}, {}
    for orientation in ("rowwise", "columnwise"):
        for kind, numels in (("data", member_numels), ("scale", scale_numels)):
            key = (orientation, kind)
            sources[key] = tuple(
                torch.randint(
                    0, 256, (num_local_experts, n), dtype=torch.uint8, device=device, generator=rng
                )
                for n in numels
            )
            tables[key] = tuple(_pointer_table(source) for source in sources[key])
            gathered_sources[key] = []
            for source in sources[key]:
                gathered = [torch.empty_like(source) for _ in range(world_size)]
                dist.all_gather(gathered, source, group=group)
                gathered_sources[key].append(torch.cat(gathered))
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    compute = torch.cuda.current_stream()
    errors = []
    try:
        arena.fill_(17)
        expected = arena.clone()
        # Sparse and empty pushes must preserve the preceding orientation in untouched slots.
        for index, (orientation, slots) in enumerate(
            (
                ("rowwise", (0, 1, 2, 3)),
                ("columnwise", (0, 2)),
                ("rowwise", ()),
                ("rowwise", (1, 3)),
                ("columnwise", (0, 1, 2, 3)),
            )
        ):
            plan = _make_plan("ring", slots, world_size, num_local_experts, device)
            for fc in range(2):
                data, scale = _arena_view(
                    expected, member_numels, scale_numels, fc, num_local_experts
                )
                for slot, expert in enumerate(plan[rank].tolist()):
                    if expert >= 0:
                        for view, kind in ((data, "data"), (scale, "scale")):
                            view[slot].copy_(gathered_sources[(orientation, kind)][fc][expert])
            torch.cuda.synchronize(device)
            dist.barrier(group=group, device_ids=[device.index])
            stream = streams[index % 2]
            stream.wait_stream(compute)
            with torch.cuda.stream(stream):
                launch_virtual_expert_weight_prefetch(
                    workspace,
                    sources=tables[(orientation, "data")],
                    scale_sources=tables[(orientation, "scale")],
                    experts_to_copy=plan,
                )
            compute.wait_stream(stream)
            _check_equal(arena, expected, f"{orientation}/{slots}", errors)
        _report(errors, group)
    finally:
        torch.cuda.synchronize(device)
        dist.barrier(group=group, device_ids=[device.index])
        del workspace, arena, handle
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
            plan = plan_virtual_expert_routes(indices, workspace)
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


def _plan_on_this_rank(routes):
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
    plan = plan_virtual_expert_routes(own, workspace)
    torch.cuda.synchronize(device)
    counts = _histogram(routes).to(device)
    assert torch.equal(workspace.gathered_counts, counts)
    assert torch.equal(plan.virtual_experts.long(), _reference_map_routes(own, workspace))
    return workspace, plan, rank, group


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
    workspace, plan, _, group = _plan_on_this_rank(routes)
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
    workspace, plan, rank, group = _plan_on_this_rank(routes)
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
