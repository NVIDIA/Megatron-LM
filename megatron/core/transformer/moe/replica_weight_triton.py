# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Triton kernels for replica planning and intra-node replica transport.

The planner kernels compact semantic routes, compute deterministic replica placement
and map routes to native or replica runtime experts. The transport kernels move only
weights and gradients: owners push weights straight into their peers' replica slots
in symmetric memory and pull the replica gradients back into native wgrad staging.
Both are pure wire movement, and within the reserved SM budget only TMA saturates
NVLink, so they are built on ``tl.make_tensor_descriptor`` over runtime peer addresses
plus Triton's loop pipeliner.

Ordering against the expert GEMMs is stream order plus the collectives already in the
layer: the planner's histogram all-gather precedes every push, and the reduction's
device-side rendezvous brackets every gradient exchange. Replica gradient slots are
never cleared; the runtime parameters carry ``overwrite_main_grad`` so TE's wgrad GEMM
rewrites them on every backward.
"""

import functools
import math

import torch
import triton
import triton.language as tl

MAX_REPLICA_WEIGHT_SMS = 32
MAX_REPLICA_EP_RANKS = 64

# Constants a kernel reads must be ``tl.constexpr`` objects (``.value`` on the host).
# A tiled TMA descriptor caps its innermost box at 256 elements, so a flat stream is
# addressed as ``[rows, _ROW]``: bytes for the push, gradient elements for the reduction.
_ROW = tl.constexpr(256)
# Measured on GB300 at 32 SMs: 32 KiB tiles over four pipeline stages sustain the peer
# bandwidth both ways; smaller tiles or three stages cost 3-20%, eight stages exceed SMEM.
_MAX_TILE_BYTES = 32768
_MAX_SCALE_TILE_BYTES = 8192
_NUM_STAGES = tl.constexpr(4)
_PUSH_NUM_WARPS = 4
_GRAD_NUM_WARPS = 8

# The grid barrier toggles one high bit, so a block detects completion from its own
# pre-arrival value and the barrier resets itself.
_GRID_SYNC_TAG = tl.constexpr(0x40000000)
# One int32 word per ordered rank pair inside the symmetric-memory signal pad.
_SIGNAL_STRIDE = tl.constexpr(4)
_BARRIER_TIMEOUT_NS = tl.constexpr(100_000_000_000)
# Persistent planner grids must stay fully resident at their cooperative barrier.
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


@triton.jit(do_not_specialize=["source_rank"])
def _plan_replica_placement_kernel(
    gathered_tokens_per_expert,
    rank_load_balance,
    expert_rank_allocations,
    destination_boundaries,
    experts_to_copy,
    expert_replica_slots,
    grid_sync,
    source_rank,
    RANK_ROUTE_CAPACITY: tl.constexpr,
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
        tl.where(ranks[:, None] < source_rank, source_counts, 0), axis=0
    ).to(tl.int32)
    # Every rank must contribute exactly RANK_ROUTE_CAPACITY routes: the
    # capacity math and the compaction's slot count both assume router_topk
    # selections per token. This kernel runs eagerly, so the trap always fires.
    source_total = tl.sum(
        tl.load(
            gathered_tokens_per_expert + rank * NUM_EXPERTS + tl.arange(0, BLOCK_NUM_EXPERTS),
            mask=tl.arange(0, BLOCK_NUM_EXPERTS) < NUM_EXPERTS,
            other=0,
        ),
        axis=0,
    )
    if source_total != RANK_ROUTE_CAPACITY:
        tl.device_print(
            "replica planner: a rank's route count differs from tokens * topk on rank", rank
        )
        _emit_on_every_thread("trap; mov.u32 $0, 0;", THREADS=1)
    tl.store(
        rank_load_balance + rank, tl.sum(native_totals, axis=0).to(tl.int32) - RANK_ROUTE_CAPACITY
    )

    _grid_sync(grid_sync, _GRID_SYNC_TAG, EP_SIZE)

    # Pair the most overloaded rank with the emptiest one and move the
    # receiver's whole deficit from that single sender. This can send more than
    # the sender's excess, but it gives every receiver exactly one sender, and
    # a sender owns NUM_EXPERTS_PER_GPU experts, so a receiver never needs more
    # replica slots than it has. Moving only min(excess, deficit) would cut
    # traffic but let a receiver draw on several senders and overflow the slots.
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
    # Allocations name the destination of every route, so an expert allocated
    # here without a slot would map its routes to a stale slot id. The
    # single-sender rule above makes this unreachable; keep it loud anyway.
    if tl.max(tl.where(valid_remote, counts, -1), axis=0) > 0:
        tl.device_print("replica placement needs more replica slots than experts on rank", rank)
        _emit_on_every_thread("trap; mov.u32 $0, 0;", THREADS=1)


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
    """Give each route its stable ordinal within its expert's local stream.

    The intra-tile ordinal comes from ``match.sync`` lane masks plus a first-warp
    histogram, so ``BLOCK_NUM_ROUTES=64`` with two warps is a correctness
    constraint, not a tuning choice.
    """
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


# ``debug=True`` keeps the device assert below alive for eager launches, and a
# device assert is the only trap torch.compile can analyze: the flex dispatcher's
# preprocess launches this kernel under torch.compile, where Inductor re-emits
# the kernel without the flag. The placement kernel's route-total check covers
# that path.
@triton.jit(debug=True)
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
        # The planner sizes every rank's capacity from ROUTER_TOPK routes per token
        # and fills exactly that many slots; a token with a different count would
        # silently misroute or drop routes downstream, so fail loudly instead.
        selections = tl.sum(selected, axis=1)
        tl.device_assert(
            (selections == ROUTER_TOPK) | (tokens >= program_end),
            "replica planner requires exactly router_topk routes per token",
        )
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
        source_rank,
        RANK_ROUTE_CAPACITY=rank_route_capacity,
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
            LABEL="replica transport send stalled on rank",
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
            LABEL="replica transport receive stalled on rank",
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
    """Push this block's share of one component into every replica slot.

    Both descriptors span a whole member and stay fixed across the tile loop.
    That is a requirement, not a convenience: Triton cannot predicate a descriptor
    construction, so one built inside the pipelined loop would refuse to compile.
    """
    ROWS: tl.constexpr = MEMBER_BYTES // _ROW
    TILE_ROWS: tl.constexpr = TILE_BYTES // _ROW
    TILES: tl.constexpr = ROWS // TILE_ROWS
    # Cut each replica into as many segments as it takes to occupy the grid, and
    # give every block one contiguous run. Striping every replica across every
    # block instead would refill the copy pipeline once per replica, which costs
    # more than it saves once a member is small - as the MXFP8 scales are.
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
    fc1_scale_bases,
    fc2_scale_bases,
    peer_bases,
    signal_bases,
    plan,
    grid_barrier,
    dummy_signal,
    rank,
    FC1_BYTES: tl.constexpr,
    FC2_BYTES: tl.constexpr,
    FC1_SCALE_BYTES: tl.constexpr,
    FC2_SCALE_BYTES: tl.constexpr,
    TILE_BYTES: tl.constexpr,
    SCALE_TILE_BYTES: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    WORLD: tl.constexpr,
    WORLD_POW2: tl.constexpr,
    PLAN_POW2: tl.constexpr,
    NUM_SMS: tl.constexpr,
    THREADS: tl.constexpr,
):
    """Push every owner-local expert into its replica slots and rendezvous.

    ``plan`` holds the destination-major ``[world, num_local_experts]`` table of
    globally numbered experts the planner wants materialized, so the entries this
    rank owns are a sparse subset of it. Compacting them into a dense ordinal
    keeps the sweep free of idle iterations even when a rank owns 8 of 512 slots,
    and recovering each plan entry with one masked reduction avoids staging the
    compacted table through memory. The arena holds ``fc1 data, fc1 scales, fc2
    data, fc2 scales``, each section ``NUM_LOCAL_EXPERTS`` members long; the two
    scale sections are empty for BF16 weights.
    """
    FC1_SCALE_ARENA: tl.constexpr = NUM_LOCAL_EXPERTS * FC1_BYTES
    FC2_ARENA: tl.constexpr = FC1_SCALE_ARENA + NUM_LOCAL_EXPERTS * FC1_SCALE_BYTES
    FC2_SCALE_ARENA: tl.constexpr = FC2_ARENA + NUM_LOCAL_EXPERTS * FC2_BYTES
    entry = tl.arange(0, PLAN_POW2)
    planned = entry < WORLD * NUM_LOCAL_EXPERTS
    owner_expert = tl.load(plan + entry, mask=planned, other=-1) - rank * NUM_LOCAL_EXPERTS
    mine = planned & (owner_expert >= 0) & (owner_expert < NUM_LOCAL_EXPERTS)
    ordinal = tl.cumsum(mine.to(tl.int32), 0) - 1
    active = tl.sum(mine.to(tl.int32), 0)
    block = tl.program_id(0)

    # One bulk-copy engine per block serves every component, so the much smaller
    # scale transfers follow the data rather than competing with it.
    # fmt: off
    _push_projection(
        fc1_bases, plan, peer_bases, entry, mine, ordinal, active, block, rank,
        FC1_BYTES, 0, TILE_BYTES, NUM_LOCAL_EXPERTS, NUM_SMS,
    )
    _push_projection(
        fc2_bases, plan, peer_bases, entry, mine, ordinal, active, block, rank,
        FC2_BYTES, FC2_ARENA, TILE_BYTES, NUM_LOCAL_EXPERTS, NUM_SMS,
    )
    if FC1_SCALE_BYTES > 0:
        _push_projection(
            fc1_scale_bases, plan, peer_bases, entry, mine, ordinal, active, block, rank,
            FC1_SCALE_BYTES, FC1_SCALE_ARENA, SCALE_TILE_BYTES, NUM_LOCAL_EXPERTS, NUM_SMS,
        )
        _push_projection(
            fc2_scale_bases, plan, peer_bases, entry, mine, ordinal, active, block, rank,
            FC2_SCALE_BYTES, FC2_SCALE_ARENA, SCALE_TILE_BYTES, NUM_LOCAL_EXPERTS, NUM_SMS,
        )
    # fmt: on
    _cross_rank_barrier(
        signal_bases, grid_barrier, dummy_signal, rank, WORLD, WORLD_POW2, NUM_SMS, THREADS
    )


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
    TILE_BEGIN: tl.constexpr,
    TILE_END: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    WORLD: tl.constexpr,
    WORLD_POW2: tl.constexpr,
    PLAN_POW2: tl.constexpr,
    NUM_SMS: tl.constexpr,
    THREADS: tl.constexpr,
):
    """Reduce every peer's replica gradients into native wgrad staging.

    ``[TILE_BEGIN, TILE_END)`` selects the transport tiles of one launch: the FC1
    members, the FC2 members, or both. The bridge launches the projections
    separately so each can start as soon as its wgrad GEMM has finished.

    ``plan`` holds the destination-major ``[world, num_local_experts]`` table of
    globally numbered experts the planner materialized, so the sources of one
    owner-local expert are the entries naming it, one per peer that hosts it.
    Every block sweeps every replicated expert but only its own contiguous slice
    of the tiles, which splits the payload evenly however sparse the plan is,
    and each block starts at a different expert so the blocks running at any
    instant are spread over the peers instead of queued behind one of them.

    The entry rendezvous proves every peer's wgrad GEMM has finished writing;
    the exit rendezvous proves every owner has read, so a peer may rewrite its
    slots on the next backward.
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

    # Compact the experts some peer replicated, and each of their sources, into
    # a table the transport reads with scalar loads. Recovering a source inside
    # the transport with a masked reduction instead costs 6% of the wire,
    # because the reduction is block wide and its two barriers land in the
    # middle of the pipelined loop. Compacting the experts as well keeps a
    # sparse plan from starting every block on the same peer. The grid sync
    # inside the rendezvous below publishes the table.
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

    low = TILE_BEGIN + block * (TILE_END - TILE_BEGIN) // NUM_SMS
    high = TILE_BEGIN + (block + 1) * (TILE_END - TILE_BEGIN) // NUM_SMS
    for step in tl.range(0, replicated, num_stages=1):
        expert = tl.load(replicas + (step + block) % tl.maximum(replicated, 1))
        count = tl.sum((mine & (owner_expert == expert)).to(tl.int32), 0)
        fc1 = _staging_pointer(arena, fc1_bases + expert, ELEMENT_BYTES)
        fc2 = _staging_pointer(arena, fc2_bases + expert, ELEMENT_BYTES)
        partial = tl.zeros([1, TILE_ROWS, _ROW], tl.float32)
        for work in tl.range(0, (high - low) * count, num_stages=_NUM_STAGES):
            tile = low + work // count
            index = work - (tile - low) * count
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
            # Peer traffic and persistent storage stay in the gradient dtype
            # while the partials are summed in FP32, so a BF16 gradient rounds
            # once, on the last source. The staging is read on every source and
            # written only on the last: a load inside a conditional is one the
            # pipeliner will not prefetch, and seeding the accumulator with it
            # keeps the summation order ``native + s0 + s1 + ...``.
            offset = tl.arange(0, TILE_ROWS)[None, :, None].to(tl.int64) * _ROW + tl.arange(0, _ROW)
            staged = tl.load(native + offset).to(tl.float32)
            partial = tl.where(index == 0, staged, partial) + window.load([destination, row, 0]).to(
                tl.float32
            )
            tl.store(native + offset, partial.to(arena.dtype.element_ty), mask=index == count - 1)
    _cross_rank_barrier(
        signal_bases, grid_barrier, dummy_signal, rank, WORLD, WORLD_POW2, NUM_SMS, THREADS
    )


def _transport_tile(limit: int, *components: int) -> int:
    """Return the largest transport tile (in the components' unit) dividing every component."""
    tile = functools.reduce(math.gcd, components, limit)
    if tile % _ROW.value:
        raise ValueError(
            f"Replica transport components must share a {_ROW.value}-aligned tile, "
            f"got {components} yielding {tile}."
        )
    return tile


def _validate_transport_shape(world_size: int, num_local_experts: int, num_sms: int) -> None:
    if not 0 < num_sms <= MAX_REPLICA_WEIGHT_SMS:
        raise ValueError(
            f"Replica weight kernels are limited to {MAX_REPLICA_WEIGHT_SMS} SMs, got {num_sms}."
        )
    if not 0 < world_size <= MAX_REPLICA_EP_RANKS or num_local_experts <= 0:
        raise ValueError(
            f"Replica transport supports 1..{MAX_REPLICA_EP_RANKS} EP ranks with a positive "
            f"expert count, got world_size={world_size}, num_local_experts={num_local_experts}."
        )


def _validate_grad_dtype(grad_dtype: torch.dtype) -> None:
    if grad_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"Replica gradients must use torch.float32 or torch.bfloat16, got {grad_dtype}."
        )


def _pointer_table(table: torch.Tensor, num_local_experts: int) -> torch.Tensor:
    """Validate one ``int64`` device table holding one base address per local expert."""
    if (
        table.dtype != torch.int64
        or table.device.type != "cuda"
        or tuple(table.shape) != (num_local_experts,)
        or not table.is_contiguous()
    ):
        raise ValueError(
            "Replica pointer tables must be contiguous CUDA int64 tensors "
            f"with {num_local_experts} entries."
        )
    return table


def _plan_table(experts_to_copy: torch.Tensor, world_size: int, num_local_experts: int):
    if (
        experts_to_copy.dtype != torch.int32
        or tuple(experts_to_copy.shape) != (world_size, num_local_experts)
        or not experts_to_copy.is_contiguous()
    ):
        raise ValueError(
            f"Replica plans must be contiguous int32 [{world_size}, {num_local_experts}] tensors."
        )
    return experts_to_copy


@functools.lru_cache(maxsize=None)
def _barrier_scratch(device_index: int) -> torch.Tensor:
    """The inert word that retargets peer lanes which already signalled."""
    return torch.zeros(1, dtype=torch.int32, device=torch.device("cuda", device_index))


@functools.lru_cache(maxsize=None)
def _source_scratch(device_index: int, entries: int) -> torch.Tensor:
    """The table the reduction compacts each expert's sources into; one per device and
    shape so its address is stable under CUDA-graph capture. The kernel fills it before
    its own rendezvous, so concurrent launches on one device never read a half-written table."""
    return torch.empty(entries, dtype=torch.int32, device=torch.device("cuda", device_index))


@functools.lru_cache(maxsize=None)
def _descriptor_scratch(device_index: int) -> torch.Tensor:
    """The buffer Triton fills with device-built TMA descriptors; persistent per device so
    captured launches replay against a stable address."""
    return torch.empty(1 << 20, dtype=torch.int8, device=torch.device("cuda", device_index))


class _DescriptorAllocator:
    """Scope Triton's process-wide workspace allocator (a shared ``ContextVar``) to one launch."""

    def __init__(self, device_index: int) -> None:
        self._scratch = _descriptor_scratch(device_index)

    def _allocate(self, size: int, alignment: int, stream) -> torch.Tensor:
        if size > self._scratch.numel():
            raise RuntimeError(f"Replica transport needs {size} descriptor bytes; reserved less.")
        return self._scratch[:size]

    def __enter__(self) -> None:
        from triton.runtime import _allocation

        self._previous = _allocation._allocator.get()
        triton.set_allocator(self._allocate)

    def __exit__(self, *exc_info) -> None:
        triton.set_allocator(self._previous)


def _push_arguments(
    member_numels: tuple[int, int],
    *,
    mxfp8: bool,
    world_size: int,
    num_local_experts: int,
    num_sms: int,
) -> dict:
    """Return the push specialization for BF16 or native MXFP8 weights.

    MXFP8 members are one byte per element plus one E8M0 scale byte per 32
    elements in either orientation, so the arena layout depends only on the
    member shapes.
    """
    member_bytes = tuple(numel if mxfp8 else 2 * numel for numel in member_numels)
    scale_bytes = tuple(numel // 32 for numel in member_numels) if mxfp8 else (0, 0)
    return dict(
        FC1_BYTES=member_bytes[0],
        FC2_BYTES=member_bytes[1],
        FC1_SCALE_BYTES=scale_bytes[0],
        FC2_SCALE_BYTES=scale_bytes[1],
        TILE_BYTES=_transport_tile(_MAX_TILE_BYTES, *member_bytes),
        SCALE_TILE_BYTES=_transport_tile(_MAX_SCALE_TILE_BYTES, *scale_bytes) if mxfp8 else 0,
        NUM_LOCAL_EXPERTS=num_local_experts,
        WORLD=world_size,
        WORLD_POW2=triton.next_power_of_2(world_size),
        PLAN_POW2=triton.next_power_of_2(world_size * num_local_experts),
        NUM_SMS=num_sms,
        THREADS=32 * _PUSH_NUM_WARPS,
        num_warps=_PUSH_NUM_WARPS,
        launch_cooperative_grid=True,
    )


def _grad_arguments(
    member_numels: tuple[int, int],
    grad_dtype: torch.dtype,
    *,
    world_size: int,
    num_local_experts: int,
    num_sms: int,
    projections: tuple[int, ...] = (0, 1),
) -> dict:
    tile = _transport_tile(_MAX_TILE_BYTES // grad_dtype.itemsize, *member_numels)
    fc1_tiles, fc2_tiles = (numel // tile for numel in member_numels)
    if not projections or any(projection not in (0, 1) for projection in projections):
        raise ValueError(
            f"Replica gradient projections must be a subset of (0, 1), got {projections}."
        )
    return dict(
        FC1_ROWS=member_numels[0] // _ROW.value,
        FC2_ROWS=member_numels[1] // _ROW.value,
        TILE_ROWS=tile // _ROW.value,
        ELEMENT_BYTES=grad_dtype.itemsize,
        TILE_BEGIN=0 if 0 in projections else fc1_tiles,
        TILE_END=fc1_tiles + fc2_tiles if 1 in projections else fc1_tiles,
        NUM_LOCAL_EXPERTS=num_local_experts,
        WORLD=world_size,
        WORLD_POW2=triton.next_power_of_2(world_size),
        PLAN_POW2=triton.next_power_of_2(world_size * num_local_experts),
        NUM_SMS=num_sms,
        THREADS=32 * _GRAD_NUM_WARPS,
        num_warps=_GRAD_NUM_WARPS,
        launch_cooperative_grid=True,
    )


def compile_replica_weight_kernels(
    *,
    world_size: int,
    num_local_experts: int,
    member_numels: tuple[int, int],
    num_sms: int,
    device_index: int,
    grad_dtype: torch.dtype = torch.float32,
    mxfp8: bool = False,
) -> None:
    """Compile the push and the gradient reduction for one weight layout.

    Compiling ahead of the first transport keeps a cold Triton cache out of the
    device-side rendezvous, where one slow rank would stall every peer.
    """
    _validate_transport_shape(world_size, num_local_experts, num_sms)
    _validate_grad_dtype(grad_dtype)
    shape = dict(world_size=world_size, num_local_experts=num_local_experts, num_sms=num_sms)
    with _DescriptorAllocator(device_index), torch.cuda.device(device_index):
        table = torch.zeros(world_size * num_local_experts, dtype=torch.int64, device="cuda")
        plan = torch.zeros(world_size * num_local_experts, dtype=torch.int32, device="cuda")
        _replica_weight_push_kernel.warmup(
            table,
            table,
            table,
            table,
            table.data_ptr(),
            table.data_ptr(),
            plan,
            plan,
            plan,
            0,
            grid=(num_sms,),
            **_push_arguments(member_numels, mxfp8=mxfp8, **shape),
        )
        # The bridge reduces FC2 and FC1 in separate launches; tests and benches
        # reduce both at once.
        for projections in ((1,), (0,), (0, 1)):
            _replica_grad_reduce_kernel.warmup(
                torch.zeros(1, dtype=grad_dtype, device="cuda"),
                table,
                table,
                table.data_ptr(),
                table.data_ptr(),
                plan,
                plan,
                plan,
                plan,
                0,
                grid=(num_sms,),
                **_grad_arguments(member_numels, grad_dtype, projections=projections, **shape),
            )


def launch_replica_weight_prefetch(
    *,
    sources: tuple[torch.Tensor, torch.Tensor],
    arena: torch.Tensor,
    peer_bases: int,
    signal_bases: int,
    experts_to_copy: torch.Tensor,
    grid_barrier: torch.Tensor,
    rank: int,
    world_size: int,
    num_local_experts: int,
    member_numels: tuple[int, int],
    num_sms: int,
    scale_sources: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Push BF16 or native MXFP8 owner weights into destination virtual slots.

    ``sources`` and ``scale_sources`` are ``int64`` pointer tables with one FC1
    or FC2 member base per local expert; ``peer_bases`` and ``signal_bases`` are
    the raw device addresses a symmetric-memory handle exposes. A ``uint8``
    arena selects the MXFP8 layout and requires the matching orientation's
    scale tables.
    """
    _validate_transport_shape(world_size, num_local_experts, num_sms)
    mxfp8 = arena.dtype == torch.uint8
    if not mxfp8 and arena.dtype != torch.bfloat16:
        raise ValueError(f"Replica weight arena must be uint8 or bfloat16, got {arena.dtype}.")
    if mxfp8 != (scale_sources is not None):
        raise ValueError("Replica MXFP8 weights require scale tables; BF16 weights forbid them.")
    tables = [_pointer_table(table, num_local_experts) for table in sources]
    tables += [_pointer_table(table, num_local_experts) for table in scale_sources or tables]
    with _DescriptorAllocator(arena.device.index):
        _replica_weight_push_kernel[(num_sms,)](
            *tables,
            int(peer_bases),
            int(signal_bases),
            _plan_table(experts_to_copy, world_size, num_local_experts),
            grid_barrier,
            _barrier_scratch(arena.device.index),
            rank,
            **_push_arguments(
                member_numels,
                mxfp8=mxfp8,
                world_size=world_size,
                num_local_experts=num_local_experts,
                num_sms=num_sms,
            ),
        )


def launch_replica_grad_reduce(
    *,
    arena: torch.Tensor,
    native_grads: tuple[torch.Tensor, torch.Tensor],
    peer_bases: int,
    signal_bases: int,
    experts_to_copy: torch.Tensor,
    grid_barrier: torch.Tensor,
    rank: int,
    world_size: int,
    num_local_experts: int,
    member_numels: tuple[int, int],
    num_sms: int,
    projections: tuple[int, ...] = (0, 1),
) -> None:
    """Accumulate every peer's replica gradients into native wgrad staging.

    ``native_grads`` are ``int64`` pointer tables with one FC1 or FC2 staging
    base per local expert; ``projections`` selects which of the two this launch
    reduces. Used replica slots are left holding their partials; the next wgrad
    GEMM overwrites them.
    """
    _validate_transport_shape(world_size, num_local_experts, num_sms)
    _validate_grad_dtype(arena.dtype)
    device_index = arena.device.index
    with _DescriptorAllocator(device_index):
        _replica_grad_reduce_kernel[(num_sms,)](
            arena,
            *(_pointer_table(table, num_local_experts) for table in native_grads),
            int(peer_bases),
            int(signal_bases),
            _plan_table(experts_to_copy, world_size, num_local_experts),
            _source_scratch(device_index, (world_size + 1) * num_local_experts + 1),
            grid_barrier,
            _barrier_scratch(device_index),
            rank,
            **_grad_arguments(
                member_numels,
                arena.dtype,
                world_size=world_size,
                num_local_experts=num_local_experts,
                num_sms=num_sms,
                projections=projections,
            ),
        )
