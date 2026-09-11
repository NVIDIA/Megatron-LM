# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Cross-rank correctness for virtual-expert planning and weight-transfer kernels.

Run on one four-GPU NVLink node::

    uv run python -m torch.distributed.run --nproc-per-node 4 -m pytest -q \
      tests/unit_tests/transformer/moe/test_virtual_expert_triton.py

These tests cover planning and transport correctness; they do not impose hardware-dependent bandwidth
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


def _reference_plan(routes, num_experts):
    """CPU greedy placement and stable route assignment, using only semantic input routes."""
    ep_size, num_tokens, topk = routes.shape
    local_experts, capacity = num_experts // ep_size, num_tokens * topk
    counts = torch.stack([torch.bincount(row.flatten(), minlength=num_experts) for row in routes])
    totals = counts.sum(0).tolist()
    loads = counts.sum(0).reshape(ep_size, local_experts).sum(1).tolist()
    quotas = [[0] * ep_size for _ in range(ep_size)]
    while max(loads) > capacity:
        sender = max(range(ep_size), key=lambda r: (loads[r], -r))
        receiver = min(range(ep_size), key=lambda r: (loads[r], r))
        # A receiver takes its entire deficit from one sender, even beyond that sender's
        # excess. This bounds the number of virtual slots by the sender's native experts.
        moved = capacity - loads[receiver]
        quotas[sender][receiver] += moved
        loads[sender] -= moved
        loads[receiver] = capacity
    allocation = [[0] * ep_size for _ in range(num_experts)]
    for expert, count in enumerate(totals):
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
        for destination, count in enumerate(destinations):
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


@pytest.mark.internal
@requires_four_ranks
@pytest.mark.parametrize(
    "ep_size,num_experts,num_tokens,topk,skews",
    [
        (4, 8, 16, 1, "balanced local remote rank_ties hot_expert balanced"),
        (4, 8, 16, 3, "random hot_expert hot_rank concentrated balanced"),
        (2, 8, 257, 3, "random hot_expert concentrated"),
        (4, 12, 9, 1, "ties local remote ties"),
        (4, 512, 8192, 10, "random hot_rank balanced"),
        (4, 32, 1025, 22, "random concentrated"),
        (2, 64, 513, 32, "random concentrated"),
    ],
    ids=[
        "ep4-k1",
        "ep4-k3",
        "ep2-strided",
        "non-power-of-two-ties",
        "production",
        "k22",
        "ep2-k32",
    ],
)
def test_virtual_expert_planner_matches_independent_reference(
    ep_size, num_experts, num_tokens, topk, skews
):
    """Exact placement and routes across shapes and changing plans in the same workspace."""
    Utils.initialize_distributed()
    groups = (
        [dist.new_group(list(range(start, start + ep_size))) for start in range(0, 4, ep_size)]
        if ep_size == 2
        else []
    )
    group = groups[dist.get_rank() // ep_size] if groups else dist.group.WORLD
    rank = dist.get_rank(group)
    device = torch.device("cuda", torch.cuda.current_device())
    local_experts = num_experts // ep_size
    workspace = VirtualExpertPlannerWorkspace(num_experts=num_experts, device=device, group=group)
    errors = []
    try:
        for iteration, skew in enumerate(skews.split()):
            routes = _routes_for_skew(ep_size, num_experts, num_tokens, topk, skew)
            counts, allocation, copies, mapped = _reference_plan(routes, num_experts)
            dtype = torch.int32 if iteration % 2 else torch.int64
            own = routes[rank].to(device=device, dtype=dtype)
            if iteration % 3:
                # A nonzero offset and poisoned gaps catch a wrapper that forgets contiguous().
                storage = torch.full(
                    (num_tokens, 2 * topk + 1), num_experts + 7, device=device, dtype=dtype
                )
                storage[:, 1::2] = own
                own = storage[:, 1::2]
                assert not own.is_contiguous()
            plan = plan_virtual_expert_routes(own, workspace)
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
                if skew in ("balanced", "local", "remote"):
                    assert (actual_copies == -1).all()
                if skew in ("local", "remote"):
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
            _check_equal(
                observed.cpu().reshape(num_experts, ep_size),
                allocation,
                f"{skew} route counts",
                errors,
            )
            _check_equal(
                observed.reshape(num_experts, ep_size).sum(0).cpu(),
                torch.full((ep_size,), num_tokens * topk, dtype=torch.int64),
                f"{skew} balanced load",
                errors,
            )
            # The real dispatcher provides this cross-rank ordering between planner launches.
            dist.barrier(group=group, device_ids=[device.index])
        _report(errors, dist.group.WORLD)
    finally:
        torch.cuda.synchronize(device)
        workspace.destroy()
        dist.barrier(device_ids=[device.index])
        if groups:
            dist.destroy_process_group(group)
