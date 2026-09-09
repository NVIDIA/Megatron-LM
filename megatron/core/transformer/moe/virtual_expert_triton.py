# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Triton kernels for virtual-expert planning and intra-node virtual-expert transport.

The planner kernel histograms the router's routes, exchanges the histograms between ranks,
computes the deterministic virtual-expert placement and maps every route to its runtime expert,
writing HybridEP's inputs in one cooperative launch. The transport kernels move only weights and
gradients: owners push weights straight into their peers' virtual-expert slots in symmetric memory
and pull the virtual-expert gradients back into native wgrad staging. Within the reserved SM budget
only TMA saturates NVLink, so both are ``tl.make_tensor_descriptor`` copies over runtime peer
addresses under Triton's loop pipeliner. Ordering against the expert GEMMs is stream order plus the
layer's collectives: the planner's exchange precedes every push and the reduction's rendezvous
brackets every gradient exchange. Virtual-expert gradient slots are never cleared; the runtime
parameters carry ``overwrite_main_grad`` so TE's wgrad GEMM rewrites them on every backward.
"""

import functools
import math

import torch
import triton
import triton.language as tl

MAX_VIRTUAL_EXPERT_WEIGHT_SMS = 32
MAX_VIRTUAL_EXPERT_EP_RANKS = 64

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
# Grid width of the planner kernel (a cooperative launch; the first EP_SIZE programs also place).
# Measured at EP 4, 8192 tokens x top-k 10 x 512 experts: 128 programs x 4 warps beat 64 x 8
# by 75 us.
PLANNER_PROGRAMS = 128
_PLANNER_PROGRAMS = tl.constexpr(PLANNER_PROGRAMS)
# One 128-byte line per flag word of the planner's scratch arena.
_FLAG_STRIDE = tl.constexpr(32)


@triton.jit
def _emit_on_every_thread(ASM: tl.constexpr, THREADS: tl.constexpr):
    """Run side-effecting PTX (proxy fences, bulk-group waits) on every thread of the block."""
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
def _argmax_lowest(values, ids, valid, SENTINEL: tl.constexpr):
    """``(max, id)`` over the valid entries; ties take the lowest id (``SENTINEL`` if none)."""
    best = tl.max(tl.where(valid, values, -2147483648), axis=0)
    return best, tl.min(tl.where(valid & (values == best), ids, SENTINEL), axis=0)


@triton.jit
def _argmax_highest(values, ids, valid):
    """``(max, id)`` over the valid entries; ties take the highest id (``-1`` if none)."""
    best = tl.max(tl.where(valid, values, -1), axis=0)
    return best, tl.max(tl.where(valid & (values == best), ids, -1), axis=0)


# The route-count, exchange-timeout and slot asserts compile in only under ``TRITON_DEBUG=1``;
# production runs without them.
@triton.jit(do_not_specialize=["source_rank", "num_tokens"])
def _plan_virtual_expert_routes_kernel(
    top_indices,
    probs,
    virtual_experts,
    runtime_probs,
    experts_to_copy,
    scratch,
    counts_sym_mem,
    source_rank,
    num_tokens,
    peer_bases,
    signal_bases,
    ROUTER_TOPK: tl.constexpr,
    EP_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
):
    """Plan one layer's virtual-expert routes in one cooperative launch.

    Phase 1: every program histograms its token range of the router's ``[num_tokens, topk]``
    ids into its row and atomically into this rank's total. Phase 2: program ``r < EP_SIZE``
    publishes this rank's histogram into row ``source_rank`` of peer ``r``'s ``counts_sym_mem``
    and waits for that peer's row in ours (the transport kernels' self-resetting handshake, one
    peer per program), so the buffer completes without a collective; it then replays the quota
    greedy for rank ``r``, assigns quotas across its experts and fills its virtual-expert slots,
    all in registers (rank ties take the lowest rank, expert ties the lowest expert, slot ties
    the highest expert) while the other programs wait at the grid barrier. Consecutive launches
    never race on the rows: HybridEP's dispatch sits between them, and that collective completes
    only once every peer has finished its own planner.
    Phase 3: every program maps its routes: the stable ordinal among this rank's routes to the
    same expert (earlier rows + running count + rank in the tile) against the placement's
    segment ends picks the destination, remote destinations take the slot the placement
    assigned, and the pass writes the int16 runtime ids and the dense
    ``[num_tokens, 2 * num_experts]`` runtime probabilities HybridEP consumes.
    """
    NUM_EXPERTS_PER_GPU: tl.constexpr = NUM_EXPERTS // EP_SIZE
    NUM_RUNTIME_EXPERTS: tl.constexpr = 2 * NUM_EXPERTS
    BLOCK_EP_SIZE: tl.constexpr = 1 << (EP_SIZE - 1).bit_length()
    BLOCK_NUM_EXPERTS_PER_GPU: tl.constexpr = 1 << (NUM_EXPERTS_PER_GPU - 1).bit_length()
    BLOCK_NUM_EXPERTS: tl.constexpr = 1 << (NUM_EXPERTS - 1).bit_length()
    BLOCK_TOPK: tl.constexpr = 1 << (ROUTER_TOPK - 1).bit_length()
    BLOCK_TOKENS: tl.constexpr = 128 // BLOCK_TOPK
    BLOCK_RUNTIME_EXPERTS: tl.constexpr = min(256, 2 * BLOCK_NUM_EXPERTS)
    HISTOGRAM_TILE: tl.constexpr = min(16, 8192 // BLOCK_NUM_EXPERTS)
    # Scratch arena fields, mirroring ``_scratch_layout`` on the host. The two flag words lead,
    # each on its own 128-byte line, so the programs spinning on the grid barrier do not contend
    # with the placing programs' barrier and data.
    placement_sync = scratch
    grid_sync = scratch + _FLAG_STRIDE
    balance = scratch + 2 * _FLAG_STRIDE
    allocation = balance + EP_SIZE
    boundaries = allocation + NUM_EXPERTS * EP_SIZE
    slots = boundaries + NUM_EXPERTS * BLOCK_EP_SIZE
    histogram_rows = slots + NUM_EXPERTS * EP_SIZE
    running_counts = histogram_rows + _PLANNER_PROGRAMS * NUM_EXPERTS
    totals = running_counts + _PLANNER_PROGRAMS * NUM_EXPERTS
    program = tl.program_id(0)
    num_routes = num_tokens * ROUTER_TOPK
    experts = tl.arange(0, BLOCK_NUM_EXPERTS)
    valid_experts = experts < NUM_EXPERTS
    ranks = tl.arange(0, BLOCK_EP_SIZE)
    valid_ranks = ranks < EP_SIZE
    tokens_per_program = tl.cdiv(num_tokens, _PLANNER_PROGRAMS)
    program_start = program * tokens_per_program
    program_end = tl.minimum(program_start + tokens_per_program, num_tokens)
    # One tile holds BLOCK_TOKENS tokens' routes in token-major order, flat index
    # local_token * BLOCK_TOPK + k, so ordering by flat index is the routes' order.
    flat = tl.arange(0, BLOCK_TOKENS * BLOCK_TOPK)
    tile_tokens = flat // BLOCK_TOPK
    tile_slots = flat % BLOCK_TOPK

    # Phase 1: this program's histogram row.
    row = tl.zeros((BLOCK_NUM_EXPERTS,), dtype=tl.int32)
    for token_start in tl.range(program_start, program_end, BLOCK_TOKENS, loop_unroll_factor=1):
        tokens = token_start + tile_tokens
        valid = (tokens < program_end) & (tile_slots < ROUTER_TOPK)
        ids = tl.load(top_indices + tokens * ROUTER_TOPK + tile_slots, mask=valid, other=0)
        row += tl.histogram(ids.to(tl.int32), BLOCK_NUM_EXPERTS, mask=valid)
    tl.store(histogram_rows + program * NUM_EXPERTS + experts, row, mask=valid_experts)
    tl.atomic_add(totals + experts, row, mask=valid_experts)
    _grid_sync(grid_sync, _GRID_SYNC_TAG, _PLANNER_PROGRAMS)

    # Phase 2: histogram exchange and placement; program ``r`` acts for rank ``r``.
    if program < EP_SIZE:
        local_experts = tl.arange(0, BLOCK_NUM_EXPERTS_PER_GPU)
        valid_local_experts = local_experts < NUM_EXPERTS_PER_GPU
        native_experts = program * NUM_EXPERTS_PER_GPU + local_experts

        histogram = tl.load(totals + experts, mask=valid_experts, other=0)
        # Our own row goes through the buffer's own address: the symmetric table maps even the
        # local buffer at a second virtual address, and same-GPU accesses through two aliases
        # are not ordered by release/acquire alone.
        peer_counts_sym_mem = tl.load(peer_bases.to(tl.pointer_type(tl.int64)) + program)
        counts_base = tl.where(
            program == source_rank, counts_sym_mem.to(tl.int64), peer_counts_sym_mem
        )
        peer_row = counts_base.to(tl.pointer_type(tl.int32)) + source_rank * NUM_EXPERTS
        tl.store(peer_row + experts, histogram, mask=valid_experts)
        # Peer ``program`` is this program's only counterpart, so its one lane is never
        # retargeted; the dummy is a word only this program touches, and only later.
        peers = program + tl.zeros((1,), dtype=tl.int32)
        _rendezvous(
            signal_bases,
            source_rank,
            peers,
            peers == program,
            running_counts + program * NUM_EXPERTS,
            LABEL="virtual-expert planner: histogram exchange stalled",
        )
        # Every program has acquired its peer's row; the barrier hands them all to everyone.
        _grid_sync(placement_sync, _GRID_SYNC_TAG, EP_SIZE)
        if program == 0:
            # Every placing program has read the histogram; clear it for the next launch's atomics.
            tl.store(
                totals + experts, tl.zeros((BLOCK_NUM_EXPERTS,), dtype=tl.int32), mask=valid_experts
            )

        source_counts = tl.load(
            counts_sym_mem + ranks[:, None] * NUM_EXPERTS + native_experts[None, :],
            mask=valid_ranks[:, None] & valid_local_experts[None, :],
            other=0,
        )
        native_totals = tl.sum(source_counts, axis=0).to(tl.int32)

        routes_before_source = tl.sum(
            tl.where(ranks[:, None] < source_rank, source_counts, 0), axis=0
        ).to(tl.int32)
        # Every rank must contribute exactly num_routes routes: the capacity math and the
        # compaction's slot count both assume router_topk selections per token. The assert
        # compiles in under ``TRITON_DEBUG=1``.
        source_total = tl.sum(
            tl.load(counts_sym_mem + program * NUM_EXPERTS + experts, mask=valid_experts, other=0),
            axis=0,
        )
        tl.device_assert(
            source_total == num_routes,
            "virtual-expert planner: a rank's route count differs from tokens * topk",
        )
        # The balance is per rank, so reduce native_totals, which is per expert.
        tl.store(balance + program, tl.sum(native_totals, axis=0).to(tl.int32) - num_routes)
        _grid_sync(placement_sync, _GRID_SYNC_TAG, EP_SIZE)

        # Pair the most overloaded rank with the emptiest one and move the receiver's whole deficit
        # from that single sender. This can send more than the sender's excess, but it gives every
        # receiver exactly one sender, and a sender owns NUM_EXPERTS_PER_GPU experts, so a receiver
        # never needs more virtual-expert slots than it has. Moving only min(excess, deficit) would
        # cut traffic but let a receiver draw on several senders and overflow the slots.
        balances = tl.load(balance + ranks, mask=valid_ranks, other=0)
        # quotas[BLOCK_EP_SIZE], quotas[d] is how many routes this rank must transfer to rank d
        quotas = tl.zeros((BLOCK_EP_SIZE,), dtype=tl.int32)
        for _ in tl.range(0, EP_SIZE, 1, loop_unroll_factor=1):
            maximum, overloaded = _argmax_lowest(balances, ranks, valid_ranks, BLOCK_EP_SIZE)
            deficit, receiver = _argmax_lowest(-balances, ranks, valid_ranks, BLOCK_EP_SIZE)
            active = maximum > 0
            moved = tl.where(active, deficit, 0).to(tl.int32)
            quotas = tl.where(active & (overloaded == program) & (ranks == receiver), moved, quotas)
            balances = tl.where(active & (ranks == overloaded), balances - moved, balances)
            balances = tl.where(active & (ranks == receiver), 0, balances)
        # allocations[local_expert, destination_rank]
        allocations = tl.where(ranks[None, :] == program, native_totals[:, None], 0)
        remaining = native_totals
        for _ in tl.range(0, EP_SIZE + NUM_EXPERTS_PER_GPU, 1, loop_unroll_factor=1):
            max_quota, destination = _argmax_lowest(quotas, ranks, valid_ranks, BLOCK_EP_SIZE)
            max_remaining, local_expert = _argmax_lowest(
                remaining, local_experts, valid_local_experts, BLOCK_NUM_EXPERTS_PER_GPU
            )
            active = max_quota > 0
            moved = tl.where(active, tl.minimum(max_quota, max_remaining), 0).to(tl.int32)
            transfer = tl.where(
                ranks[None, :] == destination, moved, tl.where(ranks[None, :] == program, -moved, 0)
            )
            allocations += tl.where((local_experts[:, None] == local_expert) & active, transfer, 0)
            remaining = tl.where(
                active & (local_experts == local_expert), remaining - moved, remaining
            )
            quotas = tl.where(active & (ranks == destination), quotas - moved, quotas)
        tl.store(
            allocation + native_experts[:, None] * EP_SIZE + ranks[None, :],
            allocations,
            mask=valid_local_experts[:, None] & valid_ranks[None, :],
        )
        tl.store(
            boundaries + native_experts[:, None] * BLOCK_EP_SIZE + ranks[None, :],
            tl.cumsum(allocations, axis=1) - routes_before_source[:, None],
            mask=valid_local_experts[:, None],
        )

        _grid_sync(placement_sync, _GRID_SYNC_TAG, EP_SIZE)

        owner = experts // NUM_EXPERTS_PER_GPU
        valid_remote = (experts < NUM_EXPERTS) & (owner != program)
        counts = tl.load(allocation + experts * EP_SIZE + program, mask=valid_remote, other=-1)
        for slot in tl.range(0, NUM_EXPERTS_PER_GPU, 1, loop_unroll_factor=1):
            maximum, expert = _argmax_highest(counts, experts, valid_remote)
            selected = tl.where(maximum > 0, expert, -1).to(tl.int32)
            tl.store(experts_to_copy + program * NUM_EXPERTS_PER_GPU + slot, selected)
            tl.store(slots + selected * EP_SIZE + program, slot, mask=selected >= 0)
            counts = tl.where(experts == expert, -1, counts)

        tl.device_assert(
            tl.max(tl.where(valid_remote, counts, -1), axis=0) <= 0,
            "virtual-expert placement needs more virtual-expert slots than experts",
        )
    _grid_sync(grid_sync, _GRID_SYNC_TAG, _PLANNER_PROGRAMS)

    # Phase 3: map this program's routes. Routes of the same expert issued by earlier programs
    # come first in the ordinal space.
    running = tl.zeros((BLOCK_NUM_EXPERTS,), dtype=tl.int32)
    rows_tile = tl.arange(0, HISTOGRAM_TILE)
    for row_start in tl.range(0, program, HISTOGRAM_TILE):
        rows = row_start + rows_tile
        running += tl.sum(
            tl.load(
                histogram_rows + rows[:, None] * NUM_EXPERTS + experts[None, :],
                mask=(rows[:, None] < program) & valid_experts[None, :],
                other=0,
            ),
            axis=0,
        )
    tile_rows = tl.arange(0, BLOCK_TOKENS)
    runtime_columns = tl.arange(0, BLOCK_RUNTIME_EXPERTS)
    for token_start in tl.range(program_start, program_end, BLOCK_TOKENS, loop_unroll_factor=1):
        # The running counts go through memory so every route can gather its expert's count.
        tl.store(running_counts + program * NUM_EXPERTS + experts, running, mask=valid_experts)
        tl.debug_barrier()
        tokens = token_start + tile_tokens
        valid = (tokens < program_end) & (tile_slots < ROUTER_TOPK)
        route_offsets = tokens * ROUTER_TOPK + tile_slots
        ids = tl.load(top_indices + route_offsets, mask=valid, other=0).to(tl.int32)
        earlier = tl.sum(
            ((ids[None, :] == ids[:, None]) & (flat[None, :] < flat[:, None]) & valid[None, :]).to(
                tl.int32
            ),
            axis=1,
        )
        ordinal = tl.load(running_counts + program * NUM_EXPERTS + ids, mask=valid, other=0)
        ordinal += earlier
        # Destination = number of segment ends at or below the ordinal. Segment ends are the
        # placement's cumulative allocations in this rank's ordinal space, clipped to the
        # routes this rank actually holds.
        local_routes = tl.load(
            counts_sym_mem + source_rank * NUM_EXPERTS + ids, mask=valid, other=0
        )
        segment_ends = tl.load(
            boundaries + ids[:, None] * BLOCK_EP_SIZE + ranks[None, :],
            mask=valid[:, None] & valid_ranks[None, :],
            other=0,
        )
        segment_ends = tl.minimum(tl.maximum(segment_ends, 0), local_routes[:, None])
        destination = tl.sum(
            ((segment_ends <= ordinal[:, None]) & valid_ranks[None, :]).to(tl.int32), axis=1
        )
        remote = valid & (destination != ids // NUM_EXPERTS_PER_GPU)
        slot = tl.load(slots + ids * EP_SIZE + destination, mask=remote, other=0)
        runtime = destination * (2 * NUM_EXPERTS_PER_GPU) + tl.where(
            remote, NUM_EXPERTS_PER_GPU + slot, ids % NUM_EXPERTS_PER_GPU
        )
        tl.store(virtual_experts + route_offsets, runtime.to(tl.int16), mask=valid)
        # Dense runtime probabilities: clear the tile's rows, then scatter the routes into them.
        rows = token_start + tile_rows
        valid_rows = rows < program_end
        for column_start in tl.range(0, NUM_RUNTIME_EXPERTS, BLOCK_RUNTIME_EXPERTS):
            columns = column_start + runtime_columns
            tl.store(
                runtime_probs + rows[:, None] * NUM_RUNTIME_EXPERTS + columns[None, :],
                tl.zeros((BLOCK_TOKENS, BLOCK_RUNTIME_EXPERTS), dtype=tl.float32),
                mask=valid_rows[:, None] & (columns[None, :] < NUM_RUNTIME_EXPERTS),
            )
        tl.debug_barrier()
        prob = tl.load(probs + route_offsets, mask=valid, other=0.0)
        tl.store(
            runtime_probs + tokens * NUM_RUNTIME_EXPERTS + runtime, prob.to(tl.float32), mask=valid
        )
        running += tl.histogram(ids, BLOCK_NUM_EXPERTS, mask=valid)


def launch_virtual_expert_planner(
    top_indices: torch.Tensor, probs: torch.Tensor, workspace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Plan one layer's routes in one cooperative launch.

    ``top_indices`` / ``probs`` are the router's ``[num_tokens, topk]`` expert ids and
    probabilities, ``workspace`` the planner scratch (``VirtualExpertPlannerWorkspace``) whose
    ``gathered_counts`` is this rank's symmetric buffer of every rank's histogram. Returns the
    int16 ``[num_tokens, topk]`` runtime ids, the float32 ``[num_tokens, 2 * num_experts]``
    runtime probabilities and the int32 ``[ep_size, num_local_experts]`` slot table.
    """
    num_tokens, router_topk = top_indices.shape
    ep_size, num_experts = workspace.ep_size, workspace.num_experts
    if ep_size > PLANNER_PROGRAMS or num_experts > 8192:
        raise ValueError(
            f"Virtual-expert planner supports at most {PLANNER_PROGRAMS} EP ranks and 8192 experts."
        )
    empty = functools.partial(torch.empty, device=top_indices.device)
    virtual_experts = empty((num_tokens, router_topk), dtype=torch.int16)
    runtime_probs = empty((num_tokens, 2 * num_experts), dtype=torch.float32)
    experts_to_copy = empty((ep_size, num_experts // ep_size), dtype=torch.int32)
    _plan_virtual_expert_routes_kernel[(PLANNER_PROGRAMS,)](
        top_indices,
        probs,
        virtual_experts,
        runtime_probs,
        experts_to_copy,
        workspace.scratch,
        workspace.gathered_counts,
        workspace.rank,
        num_tokens,
        int(workspace.histogram_handle.buffer_ptrs_dev),
        int(workspace.histogram_handle.signal_pad_ptrs_dev),
        ROUTER_TOPK=router_topk,
        EP_SIZE=ep_size,
        NUM_EXPERTS=num_experts,
        launch_cooperative_grid=True,
        num_warps=4,
    )
    return virtual_experts, runtime_probs, experts_to_copy


@triton.jit
def _handshake(
    address,
    pending,
    dummy,
    COMPARE: tl.constexpr,
    VALUE: tl.constexpr,
    SEM: tl.constexpr,
    LABEL: tl.constexpr,
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
        # Compiled out unless ``TRITON_DEBUG=1``: the assert's call site alone slows these
        # kernels about 2x, so in production a peer that never arrives hangs here.
        tl.device_assert(tl.extra.cuda.globaltimer() - start < _BARRIER_TIMEOUT_NS, LABEL)


@triton.jit
def _rendezvous(signal_bases, rank, peers, valid, dummy, LABEL: tl.constexpr):
    """Raise this rank's word in every peer's pad, then consume the word each peer raised in
    ours: a self-resetting release/acquire handshake over one int32 word per ordered rank pair.
    ``peers`` holds the peer ranks, ``valid`` the live lanes."""
    # A release covers only the stores its thread has observed; the block barrier lets one
    # thread's release cover every warp's preceding stores.
    tl.debug_barrier()
    signals = signal_bases.to(tl.pointer_type(tl.int64))
    _handshake(
        tl.load(signals + peers, mask=valid, other=0) + rank * _SIGNAL_STRIDE,
        valid,
        dummy,
        COMPARE=0,
        VALUE=1,
        SEM="release",
        LABEL=LABEL,
    )
    _handshake(
        tl.load(signals + rank) + peers * _SIGNAL_STRIDE,
        valid,
        dummy,
        COMPARE=1,
        VALUE=0,
        SEM="acquire",
        LABEL=LABEL,
    )


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
    # TMA stores complete asynchronously and Triton only waits for their shared-memory reads:
    # wait for this thread's bulk groups to be performed and order the async proxy before the
    # generic release below.
    _emit_on_every_thread(
        "cp.async.bulk.wait_group 0; fence.proxy.async.global; mov.u32 $0, 0;", THREADS
    )
    _grid_sync(grid_barrier, _GRID_SYNC_TAG, NUM_SMS)
    if tl.program_id(0) == 0:
        peers = tl.arange(0, WORLD_POW2)
        _rendezvous(
            signal_bases,
            rank,
            peers,
            peers < WORLD,
            dummy_signal,
            LABEL="virtual-expert transport rendezvous stalled",
        )
    _grid_sync(grid_barrier, _GRID_SYNC_TAG, NUM_SMS)
    # The system-scope acquire above published peer writes through the generic
    # proxy. Bridge that visibility before a following asynchronous transaction.
    _emit_on_every_thread("fence.proxy.async.global; mov.u32 $0, 0;", THREADS)


@triton.jit
def _push_fc_layer(
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
    """Push this block's share of one component into every virtual-expert slot.

    Both descriptors span a whole member and stay fixed across the tile loop.
    That is a requirement, not a convenience: Triton cannot predicate a descriptor
    construction, so one built inside the pipelined loop would refuse to compile.
    """
    ROWS: tl.constexpr = MEMBER_BYTES // _ROW
    TILE_ROWS: tl.constexpr = TILE_BYTES // _ROW
    TILES: tl.constexpr = ROWS // TILE_ROWS
    # Cut each virtual-expert into as many segments as it takes to occupy the grid, and
    # give every block one contiguous run. Striping every virtual-expert across every
    # block instead would refill the copy pipeline once per virtual-expert, which costs
    # more than it saves once a member is small - as the MXFP8 scales are.
    segments = tl.maximum(NUM_SMS // tl.maximum(active, 1), 1)
    for unit in tl.range(block, active * segments, NUM_SMS, num_stages=1):
        # Vary the virtual-expert fastest. Consecutive virtual experts have different
        # destinations, so this keeps the blocks running at any instant spread
        # over the peers instead of queued behind the one peer being swept.
        virtual_expert = unit % active
        segment = unit // active
        chosen = tl.sum(tl.where(mine & (ordinal == virtual_expert), entry, 0), 0)
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
        virtual_expert_slot = tl.make_tensor_descriptor(
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
            virtual_expert_slot.store([row, 0], source.load([row, 0]))


# ``rank`` must not be specialized: Triton would otherwise compile one kernel per rank value.
@triton.jit(do_not_specialize=["rank"])
def _virtual_expert_weight_push_kernel(
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
    """Push every owner-local expert into its virtual-expert slots and rendezvous.

    ``plan`` holds the destination-major ``[world, num_local_experts]`` table of globally numbered
    experts the planner wants materialized, so the entries this rank owns are a sparse subset of it.
    Compacting them into a dense ordinal keeps the sweep free of idle iterations even when a rank
    owns 8 of 512 slots, and recovering each plan entry with one masked reduction avoids staging the
    compacted table through memory. The arena holds ``fc1 data, fc1 scales, fc2 data, fc2 scales``,
    each section ``NUM_LOCAL_EXPERTS`` members long; the two scale sections are empty for BF16
    weights.
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
    _push_fc_layer(
        fc1_bases, plan, peer_bases, entry, mine, ordinal, active, block, rank,
        FC1_BYTES, 0, TILE_BYTES, NUM_LOCAL_EXPERTS, NUM_SMS,
    )
    _push_fc_layer(
        fc2_bases, plan, peer_bases, entry, mine, ordinal, active, block, rank,
        FC2_BYTES, FC2_ARENA, TILE_BYTES, NUM_LOCAL_EXPERTS, NUM_SMS,
    )
    if FC1_SCALE_BYTES > 0:
        _push_fc_layer(
            fc1_scale_bases, plan, peer_bases, entry, mine, ordinal, active, block, rank,
            FC1_SCALE_BYTES, FC1_SCALE_ARENA, SCALE_TILE_BYTES, NUM_LOCAL_EXPERTS, NUM_SMS,
        )
        _push_fc_layer(
            fc2_scale_bases, plan, peer_bases, entry, mine, ordinal, active, block, rank,
            FC2_SCALE_BYTES, FC2_SCALE_ARENA, SCALE_TILE_BYTES, NUM_LOCAL_EXPERTS, NUM_SMS,
        )
    # fmt: on
    _cross_rank_barrier(
        signal_bases, grid_barrier, dummy_signal, rank, WORLD, WORLD_POW2, NUM_SMS, THREADS
    )


@triton.jit
def _staging_pointer(arena, address, ELEMENT_BYTES: tl.constexpr):
    """Return one native wgrad base as a pointer whose alignment Triton knows.

    Reached as an offset from the arena pointer rather than by casting the address itself: a pointer
    cast from an integer carries no alignment, and Triton will not prefetch a load through one. The
    staging read then stops overlapping the peer transport and costs a third of the bandwidth.
    Staging is 16-byte aligned, which is what the transport has always assumed of it.
    """
    offset = (tl.load(address) - arena.to(tl.int64)) // ELEMENT_BYTES
    return arena + tl.multiple_of(offset, 16 // ELEMENT_BYTES)


@triton.jit(do_not_specialize=["rank"])
def _virtual_expert_grad_reduce_kernel(
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
    """Reduce every peer's virtual-expert gradients into native wgrad staging.

    ``[TILE_BEGIN, TILE_END)`` selects the transport tiles of one launch: the FC1 members, the FC2
    members, or both. The host launches the FC layers separately so each can start as soon as
    its wgrad GEMM has finished.

    ``plan`` holds the destination-major ``[world, num_local_experts]`` table of globally numbered
    experts the planner materialized, so the sources of one owner-local expert are the entries
    naming it, one per peer that hosts it. Every block sweeps every materialized expert but only its
    own contiguous slice of the tiles, which splits the payload evenly however sparse the plan is,
    and each block starts at a different expert so the blocks running at any instant are spread over
    the peers instead of queued behind one of them.

    The entry rendezvous proves every peer's wgrad GEMM has finished writing; the exit rendezvous
    proves every owner has read, so a peer may rewrite its slots on the next backward.
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

    # Compact the experts some peer materialized, and each of their sources, into a table the
    # transport reads with scalar loads. Recovering a source inside the transport with a masked
    # reduction instead costs 6% of the wire, because the reduction is block wide and its two
    # barriers land in the middle of the pipelined loop. Compacting the experts as well keeps a
    # sparse plan from starting every block on the same peer. The grid sync inside the rendezvous
    # below publishes the table.
    virtual_experts = sources + NUM_LOCAL_EXPERTS * WORLD
    if block == 0:
        found = 0
        for expert in tl.range(0, NUM_LOCAL_EXPERTS, num_stages=1):
            source = mine & (owner_expert == expert)
            tl.store(
                sources + expert * WORLD + tl.cumsum(source.to(tl.int32), 0) - 1, entry, mask=source
            )
            tl.store(virtual_experts + found, expert)
            found += tl.minimum(tl.sum(source.to(tl.int32), 0), 1)
        tl.store(virtual_experts + NUM_LOCAL_EXPERTS, found)

    # One descriptor whose outermost index selects the peer: the source loop sits inside the
    # pipelined tile loop, where Triton cannot build a descriptor, so one has to reach every peer.
    # The symmetric allocator maps each rank's arena at a fixed virtual stride; the host workspace
    # verifies that once.
    bases = peer_bases.to(tl.pointer_type(tl.int64))
    base = tl.load(bases)
    window = tl.make_tensor_descriptor(
        base.to(tl.pointer_type(arena.dtype.element_ty)),
        [WORLD, ARENA_ROWS, _ROW],
        [(tl.load(bases + 1) - base) // ELEMENT_BYTES, _ROW, 1],
        [1, TILE_ROWS, _ROW],
    )
    _cross_rank_barrier(
        signal_bases, grid_barrier, dummy_signal, rank, WORLD, WORLD_POW2, NUM_SMS, THREADS
    )
    materialized = tl.load(virtual_experts + NUM_LOCAL_EXPERTS)

    low = TILE_BEGIN + block * (TILE_END - TILE_BEGIN) // NUM_SMS
    high = TILE_BEGIN + (block + 1) * (TILE_END - TILE_BEGIN) // NUM_SMS
    for step in tl.range(0, materialized, num_stages=1):
        expert = tl.load(virtual_experts + (step + block) % tl.maximum(materialized, 1))
        count = tl.sum((mine & (owner_expert == expert)).to(tl.int32), 0)
        fc1 = _staging_pointer(arena, fc1_bases + expert, ELEMENT_BYTES)
        fc2 = _staging_pointer(arena, fc2_bases + expert, ELEMENT_BYTES)
        partial = tl.zeros([1, TILE_ROWS, _ROW], tl.float32)
        for work in tl.range(0, (high - low) * count, num_stages=_NUM_STAGES):
            tile = low + work // count
            index = work - (tile - low) * count
            chosen = tl.load(sources + expert * WORLD + index)
            destination = chosen // NUM_LOCAL_EXPERTS
            second = tile >= FC1_TILES
            tile_row = (tile - tl.where(second, FC1_TILES, 0)) * TILE_ROWS
            # The peer's arena stores every FC1 member first, then every FC2 member.
            row = (
                tl.where(second, FC2_BASE_ROW, 0)
                + (chosen - destination * NUM_LOCAL_EXPERTS) * tl.where(second, FC2_ROWS, FC1_ROWS)
                + tile_row
            )
            native = tl.where(second, fc2, fc1) + tile_row * _ROW
            # Peer traffic and persistent storage stay in the gradient dtype while the partials are
            # summed in FP32, so a BF16 gradient rounds once, on the last source. The staging is
            # read on every source and written only on the last: a load inside a conditional is one
            # the pipeliner will not prefetch, and seeding the accumulator with it keeps the
            # summation order ``native + s0 + s1 + ...``.
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
            f"Virtual-expert transport components must share a {_ROW.value}-aligned tile, "
            f"got {components} yielding {tile}."
        )
    return tile


def _validate_transport_shape(world_size: int, num_local_experts: int, num_sms: int) -> None:
    if not 0 < num_sms <= MAX_VIRTUAL_EXPERT_WEIGHT_SMS:
        raise ValueError(
            f"Virtual-expert weight kernels are limited to {MAX_VIRTUAL_EXPERT_WEIGHT_SMS} SMs, "
            f"got {num_sms}."
        )
    if not 0 < world_size <= MAX_VIRTUAL_EXPERT_EP_RANKS or num_local_experts <= 0:
        raise ValueError(
            f"Virtual-expert transport supports 1..{MAX_VIRTUAL_EXPERT_EP_RANKS} EP ranks with a "
            f"positive expert count, got world_size={world_size}, "
            f"num_local_experts={num_local_experts}."
        )


def _check_table(tensor: torch.Tensor, dtype: torch.dtype, shape: tuple, what: str):
    """Validate a kernel input table (pointer tables are int64 ``[L]``, plans int32 ``[W, L]``)."""
    if tensor.dtype != dtype or tuple(tensor.shape) != shape or not tensor.is_contiguous():
        raise ValueError(
            f"Virtual-expert {what} must be a contiguous {dtype} tensor of shape {shape}."
        )
    return tensor


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


def _allocate_descriptor_scratch(size: int, alignment: int, stream) -> torch.Tensor:
    """Triton's device-descriptor allocator: the per-launch ``torch.empty`` Inductor installs."""
    return torch.empty(size, dtype=torch.int8, device="cuda")


def launch_virtual_expert_weight_prefetch(
    workspace,
    *,
    sources: tuple[torch.Tensor, torch.Tensor],
    experts_to_copy: torch.Tensor,
    scale_sources: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Push BF16 or native MXFP8 owner weights into destination virtual slots.

    ``workspace`` is the host's transport workspace: ``weight_arena`` with its symmetric-memory
    ``weight_handle`` and ``weight_grid_barrier``, plus ``rank``, ``world_size``,
    ``num_local_experts``, ``member_numels`` and ``num_sms``. ``sources`` and ``scale_sources``
    are ``int64`` pointer tables with one FC1 or FC2 member base per local expert. A ``uint8``
    arena selects the MXFP8 layout and requires the matching orientation's scale tables.
    """
    world_size, num_local_experts = workspace.world_size, workspace.num_local_experts
    member_numels, num_sms = workspace.member_numels, workspace.num_sms
    arena = workspace.weight_arena
    _validate_transport_shape(world_size, num_local_experts, num_sms)
    mxfp8 = arena.dtype == torch.uint8
    if arena.dtype not in (torch.uint8, torch.bfloat16) or mxfp8 != (scale_sources is not None):
        raise ValueError(
            "Virtual-expert weight arena must be uint8 (MXFP8, with scale tables) or bfloat16 "
            f"(without), got {arena.dtype}."
        )
    tables = [
        _check_table(table, torch.int64, (num_local_experts,), "pointer tables")
        for table in (*sources, *(scale_sources or sources))
    ]
    # MXFP8 members are one byte per element plus one E8M0 scale byte per 32 elements in
    # either orientation, so the arena layout depends only on the member shapes.
    member_bytes = tuple(numel if mxfp8 else 2 * numel for numel in member_numels)
    scale_bytes = tuple(numel // 32 for numel in member_numels) if mxfp8 else (0, 0)
    triton.set_allocator(_allocate_descriptor_scratch)
    _virtual_expert_weight_push_kernel[(num_sms,)](
        *tables,
        int(workspace.weight_handle.buffer_ptrs_dev),
        int(workspace.weight_handle.signal_pad_ptrs_dev),
        _check_table(experts_to_copy, torch.int32, (world_size, num_local_experts), "plans"),
        workspace.weight_grid_barrier,
        _barrier_scratch(arena.device.index),
        workspace.rank,
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


def launch_virtual_expert_grad_reduce(
    workspace,
    *,
    native_grads: tuple[torch.Tensor, torch.Tensor],
    experts_to_copy: torch.Tensor,
    fc_layers: tuple[int, ...] = (0, 1),
) -> None:
    """Accumulate every peer's virtual-expert gradients into native wgrad staging.

    ``workspace`` is the host's transport workspace: ``grad_arena`` with its symmetric-memory
    ``grad_handle`` and ``grad_grid_barrier``, plus ``rank``, ``world_size``,
    ``num_local_experts``, ``member_numels`` and ``num_sms``. ``native_grads`` are ``int64``
    pointer tables with one FC1 or FC2 staging base per local expert; ``fc_layers`` selects
    which of the two this launch reduces. Used virtual-expert slots are left holding their
    partials; the next wgrad GEMM overwrites them.
    """
    world_size, num_local_experts = workspace.world_size, workspace.num_local_experts
    member_numels, num_sms = workspace.member_numels, workspace.num_sms
    arena = workspace.grad_arena
    _validate_transport_shape(world_size, num_local_experts, num_sms)
    if arena.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"Virtual-expert gradients must use torch.float32 or torch.bfloat16, got {arena.dtype}."
        )
    device_index = arena.device.index
    tile = _transport_tile(_MAX_TILE_BYTES // arena.dtype.itemsize, *member_numels)
    fc1_tiles, fc2_tiles = (numel // tile for numel in member_numels)
    triton.set_allocator(_allocate_descriptor_scratch)
    _virtual_expert_grad_reduce_kernel[(num_sms,)](
        arena,
        *(
            _check_table(table, torch.int64, (num_local_experts,), "pointer tables")
            for table in native_grads
        ),
        int(workspace.grad_handle.buffer_ptrs_dev),
        int(workspace.grad_handle.signal_pad_ptrs_dev),
        _check_table(experts_to_copy, torch.int32, (world_size, num_local_experts), "plans"),
        _source_scratch(device_index, (world_size + 1) * num_local_experts + 1),
        workspace.grad_grid_barrier,
        _barrier_scratch(device_index),
        workspace.rank,
        FC1_ROWS=member_numels[0] // _ROW.value,
        FC2_ROWS=member_numels[1] // _ROW.value,
        TILE_ROWS=tile // _ROW.value,
        ELEMENT_BYTES=arena.dtype.itemsize,
        TILE_BEGIN=0 if 0 in fc_layers else fc1_tiles,
        TILE_END=fc1_tiles + fc2_tiles if 1 in fc_layers else fc1_tiles,
        NUM_LOCAL_EXPERTS=num_local_experts,
        WORLD=world_size,
        WORLD_POW2=triton.next_power_of_2(world_size),
        PLAN_POW2=triton.next_power_of_2(world_size * num_local_experts),
        NUM_SMS=num_sms,
        THREADS=32 * _GRAD_NUM_WARPS,
        num_warps=_GRAD_NUM_WARPS,
        launch_cooperative_grid=True,
    )
