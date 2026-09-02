# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Triton kernels for replica planning and intra-node replica transport.

The planner kernels recover compact semantic routes, compute deterministic
replica placement, and map routes to native or replica runtime experts. They
share the cooperative-grid synchronization used by the weight transport.

Only virtual weights and gradients occupy PyTorch native symmetric memory. Source
weights remain in parameter or GTP-gather storage: each owner pushes them directly
into the destination virtual slots of its peers, and pulls the resulting replica
gradients back into native wgrad staging, clearing the local slots behind the
transport. No activation transport is involved.

Both directions are pure wire movement, so they are bound by NVLink bandwidth
rather than by arithmetic, and within the reserved SM budget only the TMA unit can
saturate the link. ``tl.make_tensor_descriptor`` reaches that bandwidth from a peer
base address resolved at runtime, and Triton's loop pipeliner supplies the
multi-stage schedule that the transfer needs.
"""

import functools
import math

import torch
import triton
import triton.language as tl

MAX_REPLICA_WEIGHT_SMS = 32
MAX_REPLICA_EP_RANKS = 64

# Constants a kernel reads must be ``tl.constexpr`` objects; ``.value`` recovers
# the plain integer for host-side arithmetic.
#
# A tiled TMA descriptor caps its innermost box dimension at 256 elements, so a
# flat stream is addressed as ``[rows, _ROW]``. Every transport offset is a
# multiple of the tile size and therefore already row aligned. The push moves
# opaque bytes and so views a row as 256 bytes; the reduction has to do
# arithmetic and views it as 256 gradient elements.
_ROW = tl.constexpr(256)
# Measured on GB300 at 32 SMs: 32 KiB tiles over four pipeline stages sustain the
# peer write bandwidth. Halving the tile or moving to three stages costs about
# 3%, and eight stages exceed the shared-memory budget.
_MAX_TILE_BYTES = 32768
_MAX_SCALE_TILE_BYTES = 8192
_NUM_WARPS = 4
_NUM_STAGES = tl.constexpr(4)
_THREADS = 32 * _NUM_WARPS

# Toggling one high bit lets a block detect completion from its own pre-arrival
# value, so the barrier resets itself and needs no separate clearing pass.
_GRID_SYNC_TAG = tl.constexpr(0x40000000)
# One int32 word per ordered rank pair inside the symmetric-memory signal pad.
_SIGNAL_STRIDE = tl.constexpr(4)
_BARRIER_TIMEOUT_NS = tl.constexpr(100_000_000_000)

# Persistent planner grids must remain fully resident at their cooperative
# barrier. This width is safe on the validated GB300 workloads.
_MAX_PLANNER_PROGRAMS = 128


def planner_route_partition_count(num_routes: int) -> int:
    """Return the shared route-ranking and route-mapping grid width."""
    return min(_MAX_PLANNER_PROGRAMS, num_routes)


@triton.jit
def _emit_on_every_thread(ASM: tl.constexpr, THREADS: tl.constexpr):
    """Run one side-effecting PTX instruction on every thread of the block."""
    tl.inline_asm_elementwise(
        ASM, "=r,r", [tl.zeros([THREADS], tl.int32)], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _grid_sync(grid_barrier, TAG: tl.constexpr, NUM_SMS: tl.constexpr):
    """Self-resetting cooperative-grid barrier."""
    tl.debug_barrier()
    increment = tl.where(tl.program_id(0) == 0, TAG - (NUM_SMS - 1), 1)
    previous = tl.atomic_add(grid_barrier, increment, sem="release", scope="gpu")
    complete = False
    while not complete:
        current = tl.atomic_add(grid_barrier, 0, sem="acquire", scope="gpu")
        complete = ((current ^ previous) & TAG) != 0
    tl.debug_barrier()


@triton.jit
def _plan_replica_placement_kernel(
    gathered_tokens_per_expert,
    rank_load_balance,
    expert_rank_allocations,
    destination_boundaries,
    experts_to_copy,
    expert_replica_slots,
    grid_sync,
    RANK_ROUTE_CAPACITY: tl.constexpr,
    SOURCE_EP_RANK: tl.constexpr,
    EP_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_PER_GPU: tl.constexpr,
    BLOCK_EP_SIZE: tl.constexpr,
    BLOCK_NUM_EXPERTS_PER_GPU: tl.constexpr,
    BLOCK_NUM_EXPERTS: tl.constexpr,
):
    """Compute deterministic replica placement in one cooperative launch.

    One program owns each EP rank. It computes native expert totals, replays
    the quota greedy, assigns quotas across its experts, and finally fills its
    replica slots. Quotas and the local allocation tile remain in registers.
    Equal rank ties choose the lowest rank, expert allocation ties choose the
    lowest expert, and replica-slot ties choose the highest expert.
    """
    rank = tl.program_id(0)
    ranks = tl.arange(0, BLOCK_EP_SIZE)
    valid_ranks = ranks < EP_SIZE
    local_experts = tl.arange(0, BLOCK_NUM_EXPERTS_PER_GPU)
    valid_local_experts = local_experts < NUM_EXPERTS_PER_GPU
    native_experts = rank * NUM_EXPERTS_PER_GPU + local_experts

    source_counts = tl.load(
        gathered_tokens_per_expert + ranks[:, None] * NUM_EXPERTS + native_experts[None, :],
        mask=valid_ranks[:, None] & valid_local_experts[None, :],
        other=0,
    )
    native_totals = tl.sum(source_counts, axis=0).to(tl.int32)
    routes_before_source = tl.sum(
        tl.where(ranks[:, None] < SOURCE_EP_RANK, source_counts, 0), axis=0
    ).to(tl.int32)
    tl.store(
        rank_load_balance + rank, tl.sum(native_totals, axis=0).to(tl.int32) - RANK_ROUTE_CAPACITY
    )

    _grid_sync(grid_sync, _GRID_SYNC_TAG, EP_SIZE)

    balances = tl.load(rank_load_balance + ranks, mask=valid_ranks, other=0)
    quotas = tl.zeros((BLOCK_EP_SIZE,), dtype=tl.int32)
    for _ in tl.range(0, EP_SIZE, 1, loop_unroll_factor=1):
        maximum = tl.max(tl.where(valid_ranks, balances, -2147483648), axis=0)
        minimum = tl.min(tl.where(valid_ranks, balances, 2147483647), axis=0)
        overloaded = tl.min(
            tl.where(valid_ranks & (balances == maximum), ranks, BLOCK_EP_SIZE), axis=0
        )
        receiver = tl.min(
            tl.where(valid_ranks & (balances == minimum), ranks, BLOCK_EP_SIZE), axis=0
        )
        active = maximum > 0
        moved = tl.where(active, -minimum, 0).to(tl.int32)
        quotas = tl.where(active & (overloaded == rank) & (ranks == receiver), moved, quotas)
        balances = tl.where(active & (ranks == overloaded), balances - moved, balances)
        balances = tl.where(active & (ranks == receiver), 0, balances)
    remaining = native_totals
    allocations = tl.where(ranks[None, :] == rank, native_totals[:, None], 0)
    for _ in tl.range(0, EP_SIZE + NUM_EXPERTS_PER_GPU, 1, loop_unroll_factor=1):
        max_quota = tl.max(tl.where(valid_ranks, quotas, -1), axis=0)
        destination = tl.min(
            tl.where(valid_ranks & (quotas == max_quota), ranks, BLOCK_EP_SIZE), axis=0
        )
        max_remaining = tl.max(tl.where(valid_local_experts, remaining, -1), axis=0)
        local_expert = tl.min(
            tl.where(
                valid_local_experts & (remaining == max_remaining),
                local_experts,
                BLOCK_NUM_EXPERTS_PER_GPU,
            ),
            axis=0,
        )
        active = max_quota > 0
        moved = tl.where(active, tl.minimum(max_quota, max_remaining), 0).to(tl.int32)
        transfer = tl.where(
            ranks[None, :] == destination, moved, tl.where(ranks[None, :] == rank, -moved, 0)
        )
        allocations += tl.where((local_experts[:, None] == local_expert) & active, transfer, 0)
        remaining = tl.where(active & (local_experts == local_expert), remaining - moved, remaining)
        quotas = tl.where(active & (ranks == destination), quotas - moved, quotas)
    tl.store(
        expert_rank_allocations + native_experts[:, None] * EP_SIZE + ranks[None, :],
        allocations,
        mask=valid_local_experts[:, None] & valid_ranks[None, :],
    )
    tl.store(
        destination_boundaries + native_experts[:, None] * BLOCK_EP_SIZE + ranks[None, :],
        tl.cumsum(allocations, axis=1) - routes_before_source[:, None],
        mask=valid_local_experts[:, None],
    )

    _grid_sync(grid_sync, _GRID_SYNC_TAG, EP_SIZE)

    experts = tl.arange(0, BLOCK_NUM_EXPERTS)
    owner = experts // NUM_EXPERTS_PER_GPU
    valid_remote = (experts < NUM_EXPERTS) & (owner != rank)
    counts = tl.load(
        expert_rank_allocations + experts * EP_SIZE + rank, mask=valid_remote, other=-1
    )
    for slot in tl.range(0, NUM_EXPERTS_PER_GPU, 1, loop_unroll_factor=1):
        maximum = tl.max(tl.where(valid_remote, counts, -1), axis=0)
        expert = tl.max(tl.where(valid_remote & (counts == maximum), experts, -1), axis=0)
        selected = tl.where(maximum > 0, expert, -1).to(tl.int32)
        tl.store(experts_to_copy + rank * NUM_EXPERTS_PER_GPU + slot, selected)
        tl.store(expert_replica_slots + selected * EP_SIZE + rank, slot, mask=selected >= 0)
        counts = tl.where(experts == expert, -1, counts)


@triton.jit
def _rank_routes_within_experts_kernel(
    flat_topk_indices,
    route_metadata,
    partition_counts,
    grid_sync,
    NUM_ROUTES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_NUM_EXPERTS: tl.constexpr,
    BLOCK_NUM_ROUTES: tl.constexpr,
    BLOCK_SCAN_PARTITIONS: tl.constexpr,
    NUM_SCAN_EXPERTS: tl.constexpr,
):
    """Give each route its stable ordinal within its expert's local stream."""
    partition = tl.program_id(0)
    num_partitions = tl.num_programs(0)
    expert_offsets = tl.arange(0, BLOCK_NUM_EXPERTS)
    valid_experts = expert_offsets < NUM_EXPERTS
    routes_per_partition = tl.cdiv(NUM_ROUTES, num_partitions)
    partition_start = partition * routes_per_partition
    partition_end = tl.minimum(partition_start + routes_per_partition, NUM_ROUTES)
    partition_histogram = tl.zeros((BLOCK_NUM_EXPERTS,), dtype=tl.int32)
    tile_offsets = tl.arange(0, BLOCK_NUM_ROUTES)

    for route_start in tl.range(
        partition_start, partition_end, BLOCK_NUM_ROUTES, loop_unroll_factor=1
    ):
        route_positions = route_start + tile_offsets
        valid_routes = route_positions < partition_end
        route_experts = tl.load(
            flat_topk_indices + route_positions, mask=valid_routes, other=NUM_EXPERTS + tile_offsets
        ).to(tl.int32)
        ranks_in_tile = tl.inline_asm_elementwise(
            asm="""
            {
                .reg .b32 matching_lanes;
                .reg .b32 lower_lanes;
                match.sync.any.b32 matching_lanes, $1, 0xffffffff;
                mov.u32 lower_lanes, %lanemask_lt;
                and.b32 matching_lanes, matching_lanes, lower_lanes;
                popc.b32 $0, matching_lanes;
            }
            """,
            constraints="=r,r",
            args=[route_experts],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        safe_route_experts = tl.where(valid_routes, route_experts, 0)
        first_warp_counts = tl.histogram(
            route_experts, BLOCK_NUM_EXPERTS, mask=valid_routes & (tile_offsets < 32)
        )
        second_warp_counts = tl.histogram(
            route_experts,
            BLOCK_NUM_EXPERTS,
            mask=valid_routes & (tile_offsets >= 32) & (tile_offsets < 64),
        )
        preceding_warp_counts = tl.gather(first_warp_counts, safe_route_experts, axis=0)
        ranks_in_tile += tl.where(tile_offsets >= 32, preceding_warp_counts, 0)
        ordinals_before_tile = tl.gather(partition_histogram, safe_route_experts, axis=0)
        local_ordinals = ordinals_before_tile + ranks_in_tile
        tl.store(
            route_metadata + route_positions,
            local_ordinals * BLOCK_NUM_EXPERTS + route_experts,
            mask=valid_routes,
        )
        partition_histogram += first_warp_counts + second_warp_counts

    tl.store(
        partition_counts + partition * NUM_EXPERTS + expert_offsets,
        partition_histogram,
        mask=valid_experts,
    )

    _grid_sync(grid_sync, _GRID_SYNC_TAG, num_partitions)

    partition_offsets = tl.arange(0, BLOCK_SCAN_PARTITIONS)
    valid_partitions = partition_offsets < num_partitions
    for scan_expert_offset in tl.static_range(0, NUM_SCAN_EXPERTS):
        scan_expert = partition + scan_expert_offset * num_partitions
        valid_scan = valid_partitions & (scan_expert < NUM_EXPERTS)
        counts_for_expert = tl.load(
            partition_counts + partition_offsets * NUM_EXPERTS + scan_expert,
            mask=valid_scan,
            other=0,
        )
        tl.store(
            partition_counts + partition_offsets * NUM_EXPERTS + scan_expert,
            tl.cumsum(counts_for_expert, axis=0) - counts_for_expert,
            mask=valid_scan,
        )


@triton.jit
def _map_virtual_experts_kernel(
    route_metadata,
    partition_counts,
    destination_boundaries,
    expert_replica_slots,
    virtual_experts,
    NUM_ROUTES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_PER_GPU: tl.constexpr,
    EP_SIZE: tl.constexpr,
    BLOCK_NUM_EXPERTS: tl.constexpr,
    BLOCK_NUM_ROUTES: tl.constexpr,
    BLOCK_EP_SIZE: tl.constexpr,
    LOG2_BLOCK_EP_SIZE: tl.constexpr,
):
    """Map local route ordinals to rank-major native or replica expert ids."""
    partition = tl.program_id(0)
    num_partitions = tl.num_programs(0)
    routes_per_partition = tl.cdiv(NUM_ROUTES, num_partitions)
    partition_start = partition * routes_per_partition
    partition_end = tl.minimum(partition_start + routes_per_partition, NUM_ROUTES)
    expert_offsets = tl.arange(0, BLOCK_NUM_EXPERTS)
    routes_before_partition = tl.load(
        partition_counts + partition * NUM_EXPERTS + expert_offsets,
        mask=expert_offsets < NUM_EXPERTS,
        other=0,
    )
    tile_offsets = tl.arange(0, BLOCK_NUM_ROUTES)

    for route_start in tl.range(
        partition_start, partition_end, BLOCK_NUM_ROUTES, loop_unroll_factor=1
    ):
        route_positions = route_start + tile_offsets
        valid_routes = route_positions < partition_end
        packed_metadata = tl.load(route_metadata + route_positions, mask=valid_routes, other=0).to(
            tl.int32
        )
        experts = packed_metadata % BLOCK_NUM_EXPERTS
        ordinals_in_partition = packed_metadata // BLOCK_NUM_EXPERTS
        safe_experts = tl.where(valid_routes, experts, 0)
        local_ordinal = (
            tl.gather(routes_before_partition, safe_experts, axis=0) + ordinals_in_partition
        )

        boundary_base = destination_boundaries + safe_experts * BLOCK_EP_SIZE
        destination = tl.zeros((BLOCK_NUM_ROUTES,), dtype=tl.int32)
        for step in tl.static_range(0, LOG2_BLOCK_EP_SIZE):
            candidate = destination + (BLOCK_EP_SIZE >> (step + 1))
            boundary = tl.load(boundary_base + candidate - 1)
            destination = tl.where(local_ordinal >= boundary, candidate, destination)

        owner = experts // NUM_EXPERTS_PER_GPU
        owned_local = experts % NUM_EXPERTS_PER_GPU
        replica_slot = tl.load(
            expert_replica_slots + safe_experts * EP_SIZE + destination,
            mask=valid_routes & (destination != owner),
            other=-1,
        )
        runtime_local = tl.where(
            destination == owner, owned_local, NUM_EXPERTS_PER_GPU + replica_slot
        )
        virtual = destination.to(tl.int64) * (2 * NUM_EXPERTS_PER_GPU) + runtime_local
        tl.store(virtual_experts + route_positions, virtual, mask=valid_routes)


@triton.jit
def _compact_routing_map_kernel(
    routing_map,
    token_indices,
    tokens_per_expert,
    num_tokens,
    ROUTER_TOPK: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_NUM_EXPERTS: tl.constexpr,
):
    """Compact a dense routing map and accumulate its expert histogram."""
    program = tl.program_id(0)
    experts = tl.arange(0, BLOCK_NUM_EXPERTS)
    valid_experts = experts < NUM_EXPERTS
    token_offsets = tl.arange(0, BLOCK_TOKENS)
    histogram = tl.zeros((BLOCK_NUM_EXPERTS,), dtype=tl.int32)
    tokens_per_program = tl.cdiv(num_tokens, tl.num_programs(0))
    program_start = program * tokens_per_program
    program_end = tl.minimum(program_start + tokens_per_program, num_tokens)

    for token_start in tl.range(program_start, program_end, BLOCK_TOKENS, loop_unroll_factor=1):
        tokens = token_start + token_offsets
        valid = (tokens[:, None] < program_end) & valid_experts[None, :]
        selected = tl.load(
            routing_map + tokens[:, None] * NUM_EXPERTS + experts[None, :], mask=valid, other=0
        ).to(tl.int32)
        slots = tl.cumsum(selected, axis=1) - selected
        tl.store(
            token_indices + tokens[:, None] * ROUTER_TOPK + slots,
            tl.broadcast_to(experts[None, :], (BLOCK_TOKENS, BLOCK_NUM_EXPERTS)),
            mask=(selected != 0) & (slots < ROUTER_TOPK),
        )
        histogram += tl.sum(selected, axis=0)

    tl.atomic_add(tokens_per_expert + experts, histogram, mask=valid_experts)


def launch_replica_route_ranking(
    flat_topk_indices: torch.Tensor,
    route_metadata: torch.Tensor,
    partition_counts: torch.Tensor,
    grid_sync: torch.Tensor,
    *,
    num_experts: int,
    num_routes: int,
) -> None:
    """Launch one-kernel stable per-expert route ranking."""
    num_programs = planner_route_partition_count(num_routes)
    _rank_routes_within_experts_kernel[(num_programs,)](
        flat_topk_indices,
        route_metadata,
        partition_counts,
        grid_sync,
        NUM_ROUTES=num_routes,
        NUM_EXPERTS=num_experts,
        BLOCK_NUM_EXPERTS=triton.next_power_of_2(num_experts),
        BLOCK_NUM_ROUTES=64,
        BLOCK_SCAN_PARTITIONS=triton.next_power_of_2(num_programs),
        NUM_SCAN_EXPERTS=triton.cdiv(num_experts, num_programs),
        launch_cooperative_grid=True,
        num_warps=2,
    )


def launch_replica_placement(
    gathered_counts: torch.Tensor,
    balance: torch.Tensor,
    allocation: torch.Tensor,
    destination_boundaries: torch.Tensor,
    experts_to_copy: torch.Tensor,
    expert_replica_slots: torch.Tensor,
    grid_sync: torch.Tensor,
    *,
    rank_route_capacity: int,
    source_rank: int,
    ep_size: int,
    num_experts: int,
    num_local_experts: int,
) -> None:
    """Launch deterministic single-kernel replica placement."""
    _plan_replica_placement_kernel[(ep_size,)](
        gathered_counts,
        balance,
        allocation,
        destination_boundaries,
        experts_to_copy,
        expert_replica_slots,
        grid_sync,
        RANK_ROUTE_CAPACITY=rank_route_capacity,
        SOURCE_EP_RANK=source_rank,
        EP_SIZE=ep_size,
        NUM_EXPERTS=num_experts,
        NUM_EXPERTS_PER_GPU=num_local_experts,
        BLOCK_EP_SIZE=triton.next_power_of_2(ep_size),
        BLOCK_NUM_EXPERTS_PER_GPU=triton.next_power_of_2(num_local_experts),
        BLOCK_NUM_EXPERTS=triton.next_power_of_2(num_experts),
        launch_cooperative_grid=True,
        num_warps=1,
    )


def launch_replica_route_mapping(
    route_metadata: torch.Tensor,
    partition_counts: torch.Tensor,
    destination_boundaries: torch.Tensor,
    expert_replica_slots: torch.Tensor,
    virtual_experts: torch.Tensor,
    *,
    ep_size: int,
    num_experts: int,
    num_local_experts: int,
    num_routes: int,
) -> None:
    """Map ranked routes to native-or-replica ids."""
    block_ep_size = triton.next_power_of_2(ep_size)
    _map_virtual_experts_kernel[(planner_route_partition_count(num_routes),)](
        route_metadata,
        partition_counts,
        destination_boundaries,
        expert_replica_slots,
        virtual_experts,
        NUM_ROUTES=num_routes,
        NUM_EXPERTS=num_experts,
        NUM_EXPERTS_PER_GPU=num_local_experts,
        EP_SIZE=ep_size,
        BLOCK_NUM_EXPERTS=triton.next_power_of_2(num_experts),
        BLOCK_NUM_ROUTES=256,
        BLOCK_EP_SIZE=block_ep_size,
        LOG2_BLOCK_EP_SIZE=block_ep_size.bit_length() - 1,
        num_warps=8,
    )


def launch_compact_routing_map(
    routing_map: torch.Tensor,
    token_indices: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    *,
    num_tokens: int,
    router_topk: int,
    num_experts: int,
) -> None:
    """Compact selected semantic experts and accumulate their histogram."""
    block_num_experts = triton.next_power_of_2(num_experts)
    block_tokens = min(32, max(1, 16384 // block_num_experts))
    num_programs = min(_MAX_PLANNER_PROGRAMS, triton.cdiv(num_tokens, block_tokens))
    _compact_routing_map_kernel[(num_programs,)](
        routing_map,
        token_indices,
        tokens_per_expert,
        num_tokens,
        ROUTER_TOPK=router_topk,
        NUM_EXPERTS=num_experts,
        BLOCK_TOKENS=block_tokens,
        BLOCK_NUM_EXPERTS=block_num_experts,
        num_warps=8,
    )


@triton.jit
def _handshake(
    address,
    pending,
    dummy,
    rank,
    COMPARE: tl.constexpr,
    VALUE: tl.constexpr,
    SEM: tl.constexpr,
    LABEL: tl.constexpr,
    TIMEOUT_NS: tl.constexpr,
):
    """Flip one self-resetting signal per rank pair until every peer has flipped.

    Every pair owns an independent word, so the system-scope atomics issue in
    parallel instead of serializing one NVLink round trip per peer. Lanes that
    have already flipped retarget a scratch word, because ``tl.atomic_cas`` takes
    no mask and re-flipping a live signal would forge a second arrival.
    """
    compare = tl.full(address.shape, COMPARE, tl.int32)
    flipped = tl.full(address.shape, VALUE, tl.int32)
    start = tl.extra.cuda.globaltimer()
    while tl.sum(pending.to(tl.int32), 0) > 0:
        target = tl.where(pending, address, dummy.to(tl.int64)).to(tl.pointer_type(tl.int32))
        previous = tl.atomic_cas(target, compare, flipped, sem=SEM, scope="sys")
        pending = pending & (previous != compare)
        if tl.extra.cuda.globaltimer() - start >= TIMEOUT_NS:
            tl.device_print(LABEL, rank)
            _emit_on_every_thread("trap; mov.u32 $0, 0;", THREADS=1)


@triton.jit
def _cross_rank_barrier(
    signal_bases,
    grid_barrier,
    dummy_signal,
    rank,
    WORLD: tl.constexpr,
    WORLD_POW2: tl.constexpr,
    NUM_SMS: tl.constexpr,
    THREADS: tl.constexpr,
):
    """Publish preceding stores and acquire peer stores entirely on device."""
    # Native symmetric memory exposes the same allocation through a local and one
    # or more peer VMM aliases. Publish stores through the alias proxy before the
    # system-scope release signals make them available to readers.
    _emit_on_every_thread("fence.proxy.alias; mov.u32 $0, 0;", THREADS)
    _grid_sync(grid_barrier, _GRID_SYNC_TAG, NUM_SMS)
    if tl.program_id(0) == 0:
        signals = signal_bases.to(tl.pointer_type(tl.int64))
        peer = tl.arange(0, WORLD_POW2)
        valid = peer < WORLD
        # Raise this rank's word in every peer's pad, then consume the word each
        # peer raised in ours.
        _handshake(
            tl.load(signals + peer, mask=valid, other=0) + rank * _SIGNAL_STRIDE,
            valid,
            dummy_signal,
            rank,
            COMPARE=0,
            VALUE=1,
            SEM="release",
            LABEL="replica push send stalled on rank",
            TIMEOUT_NS=_BARRIER_TIMEOUT_NS,
        )
        _handshake(
            tl.load(signals + rank) + peer * _SIGNAL_STRIDE,
            valid,
            dummy_signal,
            rank,
            COMPARE=1,
            VALUE=0,
            SEM="acquire",
            LABEL="replica push receive stalled on rank",
            TIMEOUT_NS=_BARRIER_TIMEOUT_NS,
        )
    _grid_sync(grid_barrier, _GRID_SYNC_TAG, NUM_SMS)
    # The system-scope acquire above published peer writes through the generic
    # proxy. Bridge that visibility before a following asynchronous transaction.
    _emit_on_every_thread("fence.proxy.async.global; mov.u32 $0, 0;", THREADS)


@triton.jit
def _push_projection(
    bases,
    plan,
    peer_bases,
    entry,
    mine,
    ordinal,
    active,
    block,
    rank,
    MEMBER_BYTES: tl.constexpr,
    ARENA_BYTES: tl.constexpr,
    TILE_BYTES: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Push this block's share of one projection into every replica slot.

    Both descriptors span a whole projection and stay fixed across the tile loop.
    That is a requirement, not a convenience: Triton cannot predicate a descriptor
    construction, so one built inside the pipelined loop would refuse to compile.
    """
    ROWS: tl.constexpr = MEMBER_BYTES // _ROW
    TILE_ROWS: tl.constexpr = TILE_BYTES // _ROW
    TILES: tl.constexpr = ROWS // TILE_ROWS
    # Cut each replica into as many segments as it takes to occupy the grid, and
    # give every block one contiguous run. Striping every replica across every
    # block instead would refill the copy pipeline once per replica, which costs
    # more than it saves once a projection is small - as the MXFP8 scales are.
    segments = tl.maximum(NUM_SMS // tl.maximum(active, 1), 1)
    for unit in tl.range(block, active * segments, NUM_SMS, num_stages=1):
        # Vary the replica fastest. Consecutive replicas have different
        # destinations, so this keeps the blocks running at any instant spread
        # over the peers instead of queued behind the one peer being swept.
        replica = unit % active
        segment = unit // active
        chosen = tl.sum(tl.where(mine & (ordinal == replica), entry, 0), 0)
        destination = chosen // NUM_LOCAL_EXPERTS
        slot = (chosen - destination * NUM_LOCAL_EXPERTS).to(tl.int64)
        expert = tl.load(plan + chosen) - rank * NUM_LOCAL_EXPERTS
        arena = tl.load(peer_bases.to(tl.pointer_type(tl.int64)) + destination)
        source = tl.make_tensor_descriptor(
            tl.load(bases + expert).to(tl.pointer_type(tl.uint8)),
            [ROWS, _ROW],
            [_ROW, 1],
            [TILE_ROWS, _ROW],
        )
        replica_slot = tl.make_tensor_descriptor(
            (arena + ARENA_BYTES + slot * MEMBER_BYTES).to(tl.pointer_type(tl.uint8)),
            [ROWS, _ROW],
            [_ROW, 1],
            [TILE_ROWS, _ROW],
        )
        for tile in tl.range(
            segment * TILES // segments,
            (segment + 1) * TILES // segments,
            1,
            num_stages=_NUM_STAGES,
        ):
            row = tile * TILE_ROWS
            replica_slot.store([row, 0], source.load([row, 0]))


# ``rank`` must not be specialized: Triton would otherwise compile a separate
# kernel per rank value, and the ahead-of-time warmup could not cover them all.
@triton.jit(do_not_specialize=["rank"])
def _replica_weight_push_kernel(
    fc1_bases,
    fc2_bases,
    peer_bases,
    signal_bases,
    plan,
    grid_barrier,
    dummy_signal,
    rank,
    FC1_BYTES: tl.constexpr,
    FC2_BYTES: tl.constexpr,
    FC1_ARENA_BYTES: tl.constexpr,
    FC2_ARENA_BYTES: tl.constexpr,
    TILE_BYTES: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    WORLD: tl.constexpr,
    WORLD_POW2: tl.constexpr,
    PLAN_POW2: tl.constexpr,
    NUM_SMS: tl.constexpr,
    THREADS: tl.constexpr,
    BARRIER: tl.constexpr,
):
    """Push one component of every owner-local expert into its replica slots.

    ``plan`` holds the destination-major ``[world, num_local_experts]`` table of
    globally numbered experts the planner wants materialized, so the entries this
    rank owns are a sparse subset of it.  Compacting them into a dense ordinal
    keeps the sweep free of idle iterations even when a rank owns 8 of 512 slots,
    and recovering each plan entry with one masked reduction avoids staging the
    compacted table through memory.
    """
    entry = tl.arange(0, PLAN_POW2)
    planned = entry < WORLD * NUM_LOCAL_EXPERTS
    owner_expert = tl.load(plan + entry, mask=planned, other=-1) - rank * NUM_LOCAL_EXPERTS
    mine = planned & (owner_expert >= 0) & (owner_expert < NUM_LOCAL_EXPERTS)
    ordinal = tl.cumsum(mine.to(tl.int32), 0) - 1
    active = tl.sum(mine.to(tl.int32), 0)

    block = tl.program_id(0)
    _push_projection(
        fc1_bases,
        plan,
        peer_bases,
        entry,
        mine,
        ordinal,
        active,
        block,
        rank,
        FC1_BYTES,
        FC1_ARENA_BYTES,
        TILE_BYTES,
        NUM_LOCAL_EXPERTS,
        NUM_SMS,
    )
    _push_projection(
        fc2_bases,
        plan,
        peer_bases,
        entry,
        mine,
        ordinal,
        active,
        block,
        rank,
        FC2_BYTES,
        FC2_ARENA_BYTES,
        TILE_BYTES,
        NUM_LOCAL_EXPERTS,
        NUM_SMS,
    )

    if BARRIER:
        _cross_rank_barrier(
            signal_bases, grid_barrier, dummy_signal, rank, WORLD, WORLD_POW2, NUM_SMS, THREADS
        )


# The reduction holds a whole tile in registers, so the tile is bounded by the
# register file rather than by the pipeline's shared-memory budget. Measured on
# GB300 at 32 SMs: 32 KiB over four stages sustains the peer read bandwidth, and
# three stages give up 20% of it.
_MAX_GRAD_TILE_BYTES = 32768
_GRAD_NUM_WARPS = 8
_GRAD_STAGES = tl.constexpr(4)
# Transport the tiles in several passes over disjoint ranges. The cross-rank
# barrier closing a pass proves every owner has read that range, so the next
# pass can zero the local slots behind it while it waits on the wire, and only
# the trailing range stays exposed. More passes shrink that range but shorten
# the sweep each one pipelines; three is the measured optimum.
_TRANSPORT_PASSES = tl.constexpr(3)


@triton.jit
def _member_row(
    member,
    tile,
    FC1_ROWS: tl.constexpr,
    FC2_ROWS: tl.constexpr,
    FC2_BASE_ROW: tl.constexpr,
    TILE_ROWS: tl.constexpr,
):
    """Row of one transport tile in an arena that stores every FC1 member first."""
    FC1_TILES: tl.constexpr = FC1_ROWS // TILE_ROWS
    second = tile >= FC1_TILES
    member_rows = tl.where(second, FC2_ROWS, FC1_ROWS)
    return (
        tl.where(second, FC2_BASE_ROW, 0)
        + member * member_rows
        + (tile - tl.where(second, FC1_TILES, 0)) * TILE_ROWS
    )


@triton.jit
def _retire_tile(
    arena,
    hosted,
    ordinal,
    slot,
    low,
    high,
    index,
    retiring,
    FC1_ROWS: tl.constexpr,
    FC2_ROWS: tl.constexpr,
    FC2_BASE_ROW: tl.constexpr,
    TILE_ROWS: tl.constexpr,
):
    """Zero item ``index`` of the tile range ``[low, high)`` of every hosted slot.

    Ordinary vector stores rather than a bulk copy: this runs beside the peer
    loads, and a bulk store would queue behind them on the same asynchronous
    copy engine. Being plain stores is also what lets the zero fill hide, since
    it is issued from the transport loop and drains while the next tile is on
    the wire - which is also why ``retiring`` masks the store instead of guarding
    the call: a block-wide reduction inside a conditional would stop the
    pipeliner from prefetching anything else in the loop.
    """
    tiles = tl.maximum(high - low, 1)
    member = tl.sum(tl.where(hosted & (ordinal == index // tiles), slot, 0), 0)
    row = _member_row(member, low + index % tiles, FC1_ROWS, FC2_ROWS, FC2_BASE_ROW, TILE_ROWS)
    offset = (row + tl.arange(0, TILE_ROWS)[:, None]).to(tl.int64) * _ROW + tl.arange(0, _ROW)
    tl.store(arena + offset, tl.zeros([TILE_ROWS, _ROW], arena.dtype.element_ty), mask=retiring)


@triton.jit
def _staging_pointer(arena, address, ELEMENT_BYTES: tl.constexpr):
    """Return one native wgrad base as a pointer whose alignment Triton knows.

    Reached as an offset from the arena pointer rather than by casting the
    address itself: a pointer cast from an integer carries no alignment, and
    Triton will not prefetch a load through one. The staging read then stops
    overlapping the peer transport and costs a third of the bandwidth. Staging
    is 16-byte aligned, which is what the transport has always assumed of it.
    """
    offset = (tl.load(address) - arena.to(tl.int64)) // ELEMENT_BYTES
    return arena + tl.multiple_of(offset, 16 // ELEMENT_BYTES)


@triton.jit
def _tile_offset(TILE_ROWS: tl.constexpr):
    """Offsets of one transport tile, shaped like a tile the descriptor returns."""
    return tl.arange(0, TILE_ROWS)[None, :, None].to(tl.int64) * _ROW + tl.arange(0, _ROW)


@triton.jit
def _symmetric_window(
    arena,
    peer_bases,
    rank,
    ARENA_ROWS: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    ELEMENT_BYTES: tl.constexpr,
    WORLD: tl.constexpr,
    WORLD_POW2: tl.constexpr,
):
    """Return one descriptor whose outermost index selects the peer to read.

    Summing an expert's sources in a single FP32 accumulator puts the source
    loop inside the pipelined tile loop, and Triton cannot build a descriptor
    there, so one descriptor has to reach every peer. That is possible because
    the symmetric allocator maps each rank's window at a fixed virtual stride,
    which is an allocator invariant rather than a documented guarantee. Check it
    here and trap, rather than silently reading a wrong address.
    """
    bases = peer_bases.to(tl.pointer_type(tl.int64))
    base = tl.load(bases)
    stride = tl.load(bases + 1) - base
    peer = tl.arange(0, WORLD_POW2)
    mapped = tl.load(bases + peer, mask=peer < WORLD, other=0)
    strided = (mapped == base + peer.to(tl.int64) * stride) | (peer >= WORLD)
    if tl.sum((~strided).to(tl.int32), 0) != 0:
        tl.device_print("replica symmetric window is not uniformly strided on rank", rank)
        _emit_on_every_thread("trap; mov.u32 $0, 0;", THREADS=1)
    return tl.make_tensor_descriptor(
        base.to(tl.pointer_type(arena.dtype.element_ty)),
        [WORLD, ARENA_ROWS, _ROW],
        [stride // ELEMENT_BYTES, _ROW, 1],
        [1, TILE_ROWS, _ROW],
    )


# ``rank`` must not be specialized: Triton would otherwise compile a separate
# kernel per rank value, and the ahead-of-time warmup could not cover them all.
@triton.jit(do_not_specialize=["rank"])
def _replica_grad_reduce_kernel(
    arena,
    fc1_bases,
    fc2_bases,
    peer_bases,
    signal_bases,
    plan,
    sources,
    grid_barrier,
    dummy_signal,
    rank,
    FC1_ROWS: tl.constexpr,
    FC2_ROWS: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    ELEMENT_BYTES: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    LOCAL_POW2: tl.constexpr,
    WORLD: tl.constexpr,
    WORLD_POW2: tl.constexpr,
    PLAN_POW2: tl.constexpr,
    NUM_SMS: tl.constexpr,
    THREADS: tl.constexpr,
):
    """Reduce every peer's replica gradients into native wgrad staging.

    ``plan`` holds the destination-major ``[world, num_local_experts]`` table of
    globally numbered experts the planner materialized, so the sources of one
    owner-local expert are the entries naming it, one per peer that hosts it.
    Every block sweeps every expert but only its own contiguous slice of the
    tiles, which splits the payload evenly however sparse the plan is, and each
    block starts at a different expert so the blocks running at any instant are
    spread over the peers instead of queued behind one of them. The slots this
    rank hosts for its peers are zeroed from the same sweep, one tile at a time,
    a whole transport pass behind the barrier that proved they had been read.
    """
    FC1_TILES: tl.constexpr = FC1_ROWS // TILE_ROWS
    TILES: tl.constexpr = FC1_TILES + FC2_ROWS // TILE_ROWS
    FC2_BASE_ROW: tl.constexpr = NUM_LOCAL_EXPERTS * FC1_ROWS
    ARENA_ROWS: tl.constexpr = NUM_LOCAL_EXPERTS * (FC1_ROWS + FC2_ROWS)
    block = tl.program_id(0)

    entry = tl.arange(0, PLAN_POW2)
    planned = entry < WORLD * NUM_LOCAL_EXPERTS
    owner_expert = tl.load(plan + entry, mask=planned, other=-1) - rank * NUM_LOCAL_EXPERTS
    mine = planned & (owner_expert >= 0) & (owner_expert < NUM_LOCAL_EXPERTS)

    # Slots this rank hosts for a peer; they are retired behind the transport.
    slot = tl.arange(0, LOCAL_POW2)
    local = slot < NUM_LOCAL_EXPERTS
    hosted = local & (tl.load(plan + rank * NUM_LOCAL_EXPERTS + slot, mask=local, other=-1) >= 0)
    hosted_ordinal = tl.cumsum(hosted.to(tl.int32), 0) - 1
    active = tl.sum(hosted.to(tl.int32), 0)

    # Compact the experts some peer replicated, and each of their sources, into
    # a table the transport reads with scalar loads. Compacting also keeps the
    # sweep below, which hands block ``b`` the expert ``step + b``, from leaving
    # every block on the same expert - and so on the same peer - whenever the
    # plan is sparse. Recovering a source inside the transport with a masked
    # reduction instead costs 6% of the wire, because the reduction is block
    # wide and its two barriers land in the middle of the pipelined loop. The
    # grid synchronizing inside the rendezvous below publishes the table.
    replicas = sources + NUM_LOCAL_EXPERTS * WORLD
    if block == 0:
        found = 0
        for expert in tl.range(0, NUM_LOCAL_EXPERTS, num_stages=1):
            source = mine & (owner_expert == expert)
            tl.store(
                sources + expert * WORLD + tl.cumsum(source.to(tl.int32), 0) - 1, entry, mask=source
            )
            tl.store(replicas + found, expert)
            found += tl.minimum(tl.sum(source.to(tl.int32), 0), 1)
        tl.store(replicas + NUM_LOCAL_EXPERTS, found)

    window = _symmetric_window(
        arena, peer_bases, rank, ARENA_ROWS, TILE_ROWS, ELEMENT_BYTES, WORLD, WORLD_POW2
    )
    _cross_rank_barrier(
        signal_bases, grid_barrier, dummy_signal, rank, WORLD, WORLD_POW2, NUM_SMS, THREADS
    )
    replicated = tl.load(replicas + NUM_LOCAL_EXPERTS)

    # One pass beyond the transport ranges retires the trailing range, which no
    # barrier has covered yet; it is the same loop body with nothing to move.
    for transport_pass in tl.static_range(_TRANSPORT_PASSES + 1):
        first = TILES * transport_pass // _TRANSPORT_PASSES
        last = TILES * min(transport_pass + 1, _TRANSPORT_PASSES) // _TRANSPORT_PASSES
        retired = TILES * max(transport_pass - 1, 0) // _TRANSPORT_PASSES
        low = first + block * (last - first) // NUM_SMS
        high = first + (block + 1) * (last - first) // NUM_SMS
        clear_low = retired + block * (first - retired) // NUM_SMS
        clear_high = retired + (block + 1) * (first - retired) // NUM_SMS
        pending = active * (clear_high - clear_low)
        done = 0
        for step in tl.range(0, replicated, num_stages=1):
            expert = tl.load(replicas + (step + block) % tl.maximum(replicated, 1))
            count = tl.sum((mine & (owner_expert == expert)).to(tl.int32), 0)
            fc1 = _staging_pointer(arena, fc1_bases + expert, ELEMENT_BYTES)
            fc2 = _staging_pointer(arena, fc2_bases + expert, ELEMENT_BYTES)
            partial = tl.zeros([1, TILE_ROWS, _ROW], tl.float32)
            for work in tl.range(0, (high - low) * count, num_stages=_GRAD_STAGES):
                tile = low + work // count
                index = work - (tile - low) * count
                # Retire one tile of the previous pass per tile transported,
                # ahead of the peer read so the stores drain while that tile is
                # on the wire. That ordering is worth 4% of the bandwidth, and
                # it is the only overlap available without warp specialization.
                _retire_tile(
                    arena,
                    hosted,
                    hosted_ordinal,
                    slot,
                    clear_low,
                    clear_high,
                    done + work,
                    done + work < pending,
                    FC1_ROWS,
                    FC2_ROWS,
                    FC2_BASE_ROW,
                    TILE_ROWS,
                )
                chosen = tl.load(sources + expert * WORLD + index)
                destination = chosen // NUM_LOCAL_EXPERTS
                row = _member_row(
                    chosen - destination * NUM_LOCAL_EXPERTS,
                    tile,
                    FC1_ROWS,
                    FC2_ROWS,
                    FC2_BASE_ROW,
                    TILE_ROWS,
                )
                second = tile >= FC1_TILES
                native = (
                    tl.where(second, fc2, fc1)
                    + (tile - tl.where(second, FC1_TILES, 0)) * TILE_ROWS * _ROW
                )
                # Peer traffic and persistent storage stay in the gradient
                # dtype while the partials are summed in FP32, so a BF16
                # gradient rounds once, on the last source. The staging is read
                # on every source and written only on the last: a load inside a
                # conditional is one the pipeliner will not prefetch, and
                # seeding the accumulator with it keeps the summation order the
                # one every caller has always seen.
                offset = _tile_offset(TILE_ROWS)
                staged = tl.load(native + offset).to(tl.float32)
                partial = tl.where(index == 0, staged, partial) + window.load(
                    [destination, row, 0]
                ).to(tl.float32)
                tl.store(
                    native + offset, partial.to(arena.dtype.element_ty), mask=index == count - 1
                )
            done += (high - low) * count
        # A rank that hosts more slots than it owns replicas runs out of
        # transport to hide behind before the range is retired.
        for index in tl.range(done, pending, num_stages=1):
            _retire_tile(
                arena,
                hosted,
                hosted_ordinal,
                slot,
                clear_low,
                clear_high,
                index,
                True,
                FC1_ROWS,
                FC2_ROWS,
                FC2_BASE_ROW,
                TILE_ROWS,
            )
        if transport_pass < _TRANSPORT_PASSES:
            _cross_rank_barrier(
                signal_bases, grid_barrier, dummy_signal, rank, WORLD, WORLD_POW2, NUM_SMS, THREADS
            )


def _tile_bytes(limit: int, *component_bytes: int) -> int:
    """Return the largest transport tile that divides every component."""
    tile = functools.reduce(math.gcd, component_bytes, limit)
    if tile % _ROW.value:
        raise ValueError(
            f"Replica weight components must share a {_ROW.value}-byte aligned "
            f"transport tile, got {component_bytes} yielding {tile}."
        )
    return tile


def _validate_transport_shape(world_size: int, num_local_experts: int, num_sms: int) -> None:
    if world_size <= 0 or num_local_experts <= 0 or num_sms <= 0:
        raise ValueError("Replica weight launch dimensions must be positive.")
    if num_sms > MAX_REPLICA_WEIGHT_SMS:
        raise ValueError(
            f"Replica weight kernels are limited to {MAX_REPLICA_WEIGHT_SMS} SMs, got {num_sms}."
        )
    if world_size > MAX_REPLICA_EP_RANKS:
        raise ValueError(
            f"Replica transport supports at most {MAX_REPLICA_EP_RANKS} EP ranks, "
            f"got {world_size}."
        )


def _validate_scale_shape(
    member_numels: tuple[int, int], scale_numels: tuple[int, int], orientation: str
) -> None:
    """Validate the aligned native MXFP8 byte layout the push kernel assumes."""
    for projection, (member_numel, scale_numel) in enumerate(zip(member_numels, scale_numels)):
        if scale_numel <= 0 or scale_numel % 2:
            raise ValueError(
                "Replica MXFP8 scales must contain a positive even number of bytes; "
                f"{orientation} projection {projection} has {scale_numel}."
            )
        if scale_numel * 32 != member_numel:
            raise ValueError(
                "Replica MXFP8 requires one E8M0 scale byte per 32 weight bytes; "
                f"{orientation} projection {projection} has weight_bytes={member_numel}, "
                f"scale_bytes={scale_numel}."
            )


@functools.lru_cache(maxsize=None)
def _barrier_scratch(device_index: int) -> torch.Tensor:
    """Return the inert word that retargets peer lanes which already signalled."""
    return torch.zeros(1, dtype=torch.int32, device=torch.device("cuda", device_index))


@functools.lru_cache(maxsize=None)
def _source_scratch(device_index: int, entries: int) -> torch.Tensor:
    """Return the table the reduction compacts each expert's sources into.

    One buffer per device and shape, so its address is stable enough to be
    captured into a CUDA graph. The kernel fills it before its own rendezvous,
    which is what keeps concurrent launches on the same device from reading a
    table another launch is still writing.
    """
    return torch.empty(entries, dtype=torch.int32, device=torch.device("cuda", device_index))


@functools.lru_cache(maxsize=None)
def _descriptor_scratch(device_index: int) -> torch.Tensor:
    """Return the global buffer Triton fills with device-built TMA descriptors.

    One persistent buffer per device keeps the descriptors at a stable address,
    which is what lets the launch be captured into a CUDA graph and replayed.
    """
    return torch.empty(1 << 20, dtype=torch.int8, device=torch.device("cuda", device_index))


class _DescriptorAllocator:
    """Scope Triton's process-wide workspace allocator to one launch.

    ``triton.set_allocator`` writes a process-global ``ContextVar`` that any other
    Triton user shares, so the previous allocator is restored on the way out
    rather than left overwritten.
    """

    def __init__(self, device_index: int) -> None:
        self._scratch = _descriptor_scratch(device_index)
        self._previous = None

    def _allocate(self, size: int, alignment: int, stream) -> torch.Tensor:
        if size > self._scratch.numel():
            raise RuntimeError(
                f"Replica weight push needs {size} descriptor bytes but only "
                f"{self._scratch.numel()} are reserved."
            )
        return self._scratch[:size]

    def __enter__(self) -> None:
        from triton.runtime import _allocation

        self._previous = _allocation._allocator.get()
        triton.set_allocator(self._allocate)

    def __exit__(self, *exc_info) -> None:
        triton.set_allocator(self._previous)


def as_pointer_table(
    tensor: torch.Tensor, num_local_experts: int, *, dtype: torch.dtype
) -> torch.Tensor:
    """Return a stable device table containing one data pointer per local expert.

    The public kernel helpers historically accepted one contiguous ``[expert, ...]``
    tensor.  Replica bridges can now pass an ``int64`` pointer table instead, which
    also represents TE's independently allocated ``weight0..weightN`` parameters.
    """
    if tensor.dtype == torch.int64:
        if (
            tensor.device.type != "cuda"
            or tensor.ndim != 1
            or tensor.numel() != num_local_experts
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "Replica pointer tables must be contiguous CUDA int64 tensors "
                f"with {num_local_experts} entries."
            )
        return tensor
    if (
        tensor.device.type != "cuda"
        or tensor.dtype != dtype
        or tensor.ndim < 2
        or tensor.shape[0] != num_local_experts
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            "Replica sources and native grads must be contiguous CUDA tensors "
            f"with shape [{num_local_experts}, ...] and dtype {dtype}."
        )
    return torch.tensor(
        [tensor[index].data_ptr() for index in range(num_local_experts)],
        dtype=torch.int64,
        device=tensor.device,
    )


def _address(table: torch.Tensor | int) -> int:
    """Return the device address of a peer or signal table.

    Symmetric-memory handles expose ``buffer_ptrs_dev`` and ``signal_pad_ptrs_dev``
    as raw device addresses rather than tensors, so both forms are accepted.
    """
    return table.data_ptr() if isinstance(table, torch.Tensor) else int(table)


def _push_components(
    member_bytes: tuple[int, int], scale_bytes: tuple[int, int] | None
) -> list[dict]:
    """Describe each transport phase and where its arena section starts.

    The arena interleaves the sections as ``fc1 data, fc1 scales, fc2 data,
    fc2 scales``, so a phase only needs the byte offset of each projection.
    """
    fc1_scale, fc2_scale = scale_bytes if scale_bytes is not None else (0, 0)
    phases = [
        {
            "member_bytes": member_bytes,
            "arena_bytes": (0, member_bytes[0] + fc1_scale),
            "tile_bytes": _tile_bytes(_MAX_TILE_BYTES, *member_bytes),
        }
    ]
    if scale_bytes is not None:
        phases.append(
            {
                "member_bytes": scale_bytes,
                "arena_bytes": (member_bytes[0], member_bytes[0] + fc1_scale + member_bytes[1]),
                "tile_bytes": _tile_bytes(_MAX_SCALE_TILE_BYTES, *scale_bytes),
            }
        )
    return phases


def _launch_arguments(
    phase: dict, *, world_size: int, num_local_experts: int, num_sms: int
) -> dict:
    return dict(
        FC1_BYTES=phase["member_bytes"][0],
        FC2_BYTES=phase["member_bytes"][1],
        FC1_ARENA_BYTES=num_local_experts * phase["arena_bytes"][0],
        FC2_ARENA_BYTES=num_local_experts * phase["arena_bytes"][1],
        TILE_BYTES=phase["tile_bytes"],
        NUM_LOCAL_EXPERTS=num_local_experts,
        WORLD=world_size,
        WORLD_POW2=triton.next_power_of_2(world_size),
        PLAN_POW2=triton.next_power_of_2(world_size * num_local_experts),
        NUM_SMS=num_sms,
        THREADS=_THREADS,
        num_warps=_NUM_WARPS,
    )


def compile_replica_weight_push(
    *,
    world_size: int,
    num_local_experts: int,
    member_numels: tuple[int, int],
    num_sms: int,
    device_index: int,
    rowwise_scale_numels: tuple[int, int] | None = None,
    columnwise_scale_numels: tuple[int, int] | None = None,
) -> None:
    """Compile every push specialization the configured weight format can launch.

    Compiling ahead of the first transport keeps a cold Triton cache out of the
    device-side rendezvous, where one slow rank would stall every peer.
    """
    _validate_transport_shape(world_size, num_local_experts, num_sms)
    if rowwise_scale_numels is None and columnwise_scale_numels is None:
        formats = [(tuple(2 * numel for numel in member_numels), None)]
    elif rowwise_scale_numels is None or columnwise_scale_numels is None:
        raise ValueError("MXFP8 compilation requires both rowwise and columnwise scale shapes.")
    else:
        _validate_scale_shape(member_numels, rowwise_scale_numels, "rowwise")
        _validate_scale_shape(member_numels, columnwise_scale_numels, "columnwise")
        formats = [(member_numels, rowwise_scale_numels), (member_numels, columnwise_scale_numels)]
    with _DescriptorAllocator(device_index), torch.cuda.device(device_index):
        placeholder_i64 = torch.zeros(world_size, dtype=torch.int64, device="cuda")
        placeholder_i32 = torch.zeros(
            world_size * num_local_experts, dtype=torch.int32, device="cuda"
        )
        address = placeholder_i64.data_ptr()
        for member_bytes, scale_bytes in formats:
            phases = _push_components(member_bytes, scale_bytes)
            for index, phase in enumerate(phases):
                _replica_weight_push_kernel.warmup(
                    placeholder_i64,
                    placeholder_i64,
                    address,
                    address,
                    placeholder_i32,
                    placeholder_i32,
                    placeholder_i32,
                    0,
                    BARRIER=index == len(phases) - 1,
                    grid=(num_sms,),
                    **_launch_arguments(
                        phase,
                        world_size=world_size,
                        num_local_experts=num_local_experts,
                        num_sms=num_sms,
                    ),
                )


def launch_replica_weight_prefetch(
    *,
    sources: tuple[torch.Tensor, torch.Tensor],
    arena: torch.Tensor,
    peer_bases: torch.Tensor,
    signal_bases: torch.Tensor,
    experts_to_copy: torch.Tensor,
    grid_barrier: torch.Tensor,
    rank: int,
    world_size: int,
    num_local_experts: int,
    member_numels: tuple[int, int],
    num_sms: int,
    scale_sources: tuple[torch.Tensor, torch.Tensor] | None = None,
    rowwise_scale_numels: tuple[int, int] | None = None,
    columnwise_scale_numels: tuple[int, int] | None = None,
    orientation: str | None = None,
) -> None:
    """Launch a BF16 or MXFP8 owner-push into destination virtual slots."""
    device_index = arena.device.index
    if device_index is None:
        raise ValueError("Replica weight arena must be a CUDA tensor.")
    _validate_transport_shape(world_size, num_local_experts, num_sms)
    if scale_sources is None:
        if any(
            value is not None
            for value in (rowwise_scale_numels, columnwise_scale_numels, orientation)
        ):
            raise ValueError("BF16 prefetch does not accept MXFP8 scale metadata.")
        if arena.dtype != torch.bfloat16:
            raise ValueError(f"Replica BF16 arena must use torch.bfloat16, got {arena.dtype}.")
        member_bytes = tuple(2 * numel for numel in member_numels)
        scale_bytes = None
        source_dtype = torch.bfloat16
    else:
        if rowwise_scale_numels is None or columnwise_scale_numels is None:
            raise ValueError("MXFP8 prefetch requires rowwise and columnwise scale shapes.")
        if orientation not in ("rowwise", "columnwise"):
            raise ValueError(
                "Replica MXFP8 orientation must be 'rowwise' or 'columnwise', "
                f"got {orientation!r}."
            )
        if arena.dtype != torch.uint8:
            raise ValueError(f"Replica MXFP8 arena must use torch.uint8, got {arena.dtype}.")
        member_bytes = member_numels
        scale_bytes = rowwise_scale_numels if orientation == "rowwise" else columnwise_scale_numels
        _validate_scale_shape(member_bytes, scale_bytes, orientation)
        source_dtype = torch.uint8

    tables = [as_pointer_table(source, num_local_experts, dtype=source_dtype) for source in sources]
    if scale_sources is not None:
        tables += [
            as_pointer_table(source, num_local_experts, dtype=torch.uint8)
            for source in scale_sources
        ]
    phases = _push_components(member_bytes, scale_bytes)
    dummy_signal = _barrier_scratch(device_index)
    # One bulk-copy engine per CTA serves both phases, so the much smaller scale
    # transfer follows the data rather than competing with it, and only the final
    # phase closes the cross-rank rendezvous.
    with _DescriptorAllocator(device_index):
        for index, phase in enumerate(phases):
            _replica_weight_push_kernel[(num_sms,)](
                tables[2 * index],
                tables[2 * index + 1],
                _address(peer_bases),
                _address(signal_bases),
                experts_to_copy,
                grid_barrier,
                dummy_signal,
                rank,
                BARRIER=index == len(phases) - 1,
                **_launch_arguments(
                    phase,
                    world_size=world_size,
                    num_local_experts=num_local_experts,
                    num_sms=num_sms,
                ),
            )


def _grad_tile_elements(member_numels: tuple[int, int], element_bytes: int) -> int:
    """Return the largest gradient transport tile that divides both projections."""
    tile = functools.reduce(math.gcd, member_numels, _MAX_GRAD_TILE_BYTES // element_bytes)
    if tile % _ROW.value:
        raise ValueError(
            f"Replica gradient projections must share a {_ROW.value}-element aligned "
            f"transport tile, got {member_numels} yielding {tile}."
        )
    return tile


def _grad_launch_arguments(
    member_numels: tuple[int, int],
    grad_dtype: torch.dtype,
    *,
    world_size: int,
    num_local_experts: int,
    num_sms: int,
) -> dict:
    tile = _grad_tile_elements(member_numels, grad_dtype.itemsize)
    return dict(
        FC1_ROWS=member_numels[0] // _ROW.value,
        FC2_ROWS=member_numels[1] // _ROW.value,
        TILE_ROWS=tile // _ROW.value,
        ELEMENT_BYTES=grad_dtype.itemsize,
        NUM_LOCAL_EXPERTS=num_local_experts,
        LOCAL_POW2=triton.next_power_of_2(num_local_experts),
        WORLD=world_size,
        WORLD_POW2=triton.next_power_of_2(world_size),
        PLAN_POW2=triton.next_power_of_2(world_size * num_local_experts),
        NUM_SMS=num_sms,
        THREADS=32 * _GRAD_NUM_WARPS,
        num_warps=_GRAD_NUM_WARPS,
    )


def _validate_grad_dtype(grad_dtype: torch.dtype) -> None:
    if grad_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            "Replica gradients must use torch.float32 or torch.bfloat16, " f"got {grad_dtype}."
        )


def compile_replica_weight_kernels(
    *,
    world_size: int,
    num_local_experts: int,
    member_numels: tuple[int, int],
    num_sms: int,
    device_index: int,
    grad_dtype: torch.dtype = torch.float32,
    rowwise_scale_numels: tuple[int, int] | None = None,
    columnwise_scale_numels: tuple[int, int] | None = None,
) -> None:
    """Compile the shared gradient reduction and every format-specific push."""
    _validate_transport_shape(world_size, num_local_experts, num_sms)
    _validate_grad_dtype(grad_dtype)
    with _DescriptorAllocator(device_index), torch.cuda.device(device_index):
        placeholder_i64 = torch.zeros(world_size, dtype=torch.int64, device="cuda")
        placeholder_i32 = torch.zeros(
            world_size * num_local_experts, dtype=torch.int32, device="cuda"
        )
        _replica_grad_reduce_kernel.warmup(
            torch.zeros(1, dtype=grad_dtype, device="cuda"),
            placeholder_i64,
            placeholder_i64,
            placeholder_i64.data_ptr(),
            placeholder_i64.data_ptr(),
            placeholder_i32,
            placeholder_i32,
            placeholder_i32,
            placeholder_i32,
            0,
            grid=(num_sms,),
            **_grad_launch_arguments(
                member_numels,
                grad_dtype,
                world_size=world_size,
                num_local_experts=num_local_experts,
                num_sms=num_sms,
            ),
        )
    compile_replica_weight_push(
        world_size=world_size,
        num_local_experts=num_local_experts,
        member_numels=member_numels,
        num_sms=num_sms,
        device_index=device_index,
        rowwise_scale_numels=rowwise_scale_numels,
        columnwise_scale_numels=columnwise_scale_numels,
    )


def launch_replica_grad_reduce(
    *,
    arena: torch.Tensor,
    native_grads: tuple[torch.Tensor, torch.Tensor],
    peer_bases: torch.Tensor,
    signal_bases: torch.Tensor,
    experts_to_copy: torch.Tensor,
    grid_barrier: torch.Tensor,
    rank: int,
    world_size: int,
    num_local_experts: int,
    member_numels: tuple[int, int],
    num_sms: int,
) -> None:
    """Accumulate virtual gradients into native wgrad staging and clear used slots."""
    device_index = arena.device.index
    if device_index is None:
        raise ValueError("Replica gradient arena must be a CUDA tensor.")
    _validate_transport_shape(world_size, num_local_experts, num_sms)
    _validate_grad_dtype(arena.dtype)
    tables = [
        as_pointer_table(native_grad, num_local_experts, dtype=arena.dtype)
        for native_grad in native_grads
    ]
    with _DescriptorAllocator(device_index):
        _replica_grad_reduce_kernel[(num_sms,)](
            arena,
            tables[0],
            tables[1],
            _address(peer_bases),
            _address(signal_bases),
            experts_to_copy,
            _source_scratch(device_index, (world_size + 1) * num_local_experts + 1),
            grid_barrier,
            _barrier_scratch(device_index),
            rank,
            **_grad_launch_arguments(
                member_numels,
                arena.dtype,
                world_size=world_size,
                num_local_experts=num_local_experts,
                num_sms=num_sms,
            ),
        )
