# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoonEP virtual-expert placement and reroute kernels adapted from PR #7450 (242c8dbc6).

The cooperative placement kernel histograms compact router routes, exchanges
the histograms through an NCCL symmetric-memory window, and computes the
deterministic replica placement. A second kernel maps tokens to that placement,
allowing replica-weight dispatch to overlap token rerouting.
"""

from __future__ import annotations

import functools

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAVE_TRITON = False

MAX_REPLICA_EP_RANKS = 64
PLANNER_PROGRAMS = 32


def _require_triton() -> None:
    if not HAVE_TRITON:
        raise RuntimeError("MoonEP virtual-expert planning requires Triton.")


if HAVE_TRITON:
    _GRID_SYNC_TAG = tl.constexpr(0x40000000)
    _SIGNAL_STRIDE = tl.constexpr(4)
    _BARRIER_TIMEOUT_NS = tl.constexpr(100_000_000_000)
    _PLANNER_PROGRAMS = tl.constexpr(PLANNER_PROGRAMS)
    _ROUTE_TILE = tl.constexpr(512)
    _FLAG_STRIDE = tl.constexpr(32)

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

    @triton.jit
    def _prefix_maximum(a, b):
        return tl.maximum(a, b)

    @triton.jit
    def _planner_fields(scratch, EP_SIZE: tl.constexpr, NUM_EXPERTS: tl.constexpr):
        """Split the planner's int32 scratch arena into its fields (mirrors ``_scratch_layout`` on
        the host). Three flag words lead, each on its own 128-byte line so the programs spinning on
        the grid barrier do not contend with the placing programs' barrier and data."""
        BLOCK_EP_SIZE: tl.constexpr = 1 << (EP_SIZE - 1).bit_length()
        balance = scratch + 3 * _FLAG_STRIDE
        allocation = balance + EP_SIZE
        boundaries = allocation + NUM_EXPERTS * EP_SIZE
        slots = boundaries + NUM_EXPERTS * BLOCK_EP_SIZE
        histogram = slots + NUM_EXPERTS * EP_SIZE
        running = histogram + _PLANNER_PROGRAMS * NUM_EXPERTS
        totals = running + _PLANNER_PROGRAMS * NUM_EXPERTS
        return balance, allocation, boundaries, slots, histogram, running, totals

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
        while (tl.sum(pending.to(tl.int32), 0) > 0) & (
            tl.extra.cuda.globaltimer() - start < _BARRIER_TIMEOUT_NS
        ):
            target = tl.where(pending, address, dummy.to(tl.int64)).to(tl.pointer_type(tl.int32))
            previous = tl.atomic_cas(target, compare, flipped, sem=SEM, scope="sys")
            pending = pending & (previous != compare)
        tl.device_assert(tl.sum(pending.to(tl.int32), 0) == 0, LABEL)

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

    @triton.jit(debug=True, do_not_specialize=["source_rank"])
    def _update_virtual_expert_placement_kernel(
        top_indices,
        experts_to_copy,
        scratch,
        counts_sym_mem,
        source_rank,
        peer_bases,
        signal_bases,
        NUM_TOKENS: tl.constexpr,
        ROUTER_TOPK: tl.constexpr,
        EP_SIZE: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        ALIGNMENT: tl.constexpr,
        EXCHANGE: tl.constexpr,
    ):
        """PR #7450 histogram/block placement, retaining the separate reroute phase."""
        PLACEMENT_PROGRAMS: tl.constexpr = min(EP_SIZE, _PLANNER_PROGRAMS)
        RANKS_PER_PROGRAM: tl.constexpr = tl.cdiv(EP_SIZE, _PLANNER_PROGRAMS)
        ROUTE_TILE: tl.constexpr = _ROUTE_TILE
        NUM_EXPERTS_PER_GPU: tl.constexpr = NUM_EXPERTS // EP_SIZE
        BLOCK_EP_SIZE: tl.constexpr = 1 << (EP_SIZE - 1).bit_length()
        BLOCK_NUM_EXPERTS_PER_GPU: tl.constexpr = 1 << (NUM_EXPERTS_PER_GPU - 1).bit_length()
        BLOCK_NUM_EXPERTS: tl.constexpr = 1 << (NUM_EXPERTS - 1).bit_length()
        # Keep cooperative barrier words on separate cache lines, apart from their data.
        placement_sync = scratch
        grid_sync = scratch + _FLAG_STRIDE
        balance = scratch + 3 * _FLAG_STRIDE
        allocation = balance + EP_SIZE
        boundaries = allocation + NUM_EXPERTS * EP_SIZE
        slots = boundaries + NUM_EXPERTS * BLOCK_EP_SIZE
        histogram_rows = slots + NUM_EXPERTS * EP_SIZE
        running_counts = histogram_rows + _PLANNER_PROGRAMS * NUM_EXPERTS
        totals = running_counts + _PLANNER_PROGRAMS * NUM_EXPERTS
        program = tl.program_id(0)
        NUM_ROUTES: tl.constexpr = NUM_TOKENS * ROUTER_TOPK
        experts = tl.arange(0, BLOCK_NUM_EXPERTS)
        valid_experts = experts < NUM_EXPERTS
        ranks = tl.arange(0, BLOCK_EP_SIZE)
        valid_ranks = ranks < EP_SIZE
        TOKENS_PER_PROGRAM: tl.constexpr = tl.cdiv(NUM_TOKENS, _PLANNER_PROGRAMS)
        program_start = program * TOKENS_PER_PROGRAM * ROUTER_TOPK
        program_end = tl.minimum((program + 1) * TOKENS_PER_PROGRAM, NUM_TOKENS) * ROUTER_TOPK
        # Contiguous route tiles preserve token-major order without padding top-k slots.
        flat = tl.arange(0, ROUTE_TILE)

        # Phase 1: this program's histogram row.
        row = tl.zeros((BLOCK_NUM_EXPERTS,), dtype=tl.int32)
        for route_start in tl.range(program_start, program_end, ROUTE_TILE, loop_unroll_factor=1):
            offsets = route_start + flat
            valid = offsets < program_end
            ids = tl.load(top_indices + offsets, mask=valid, other=0)
            row += tl.histogram(ids.to(tl.int32), BLOCK_NUM_EXPERTS, mask=valid)
        tl.store(histogram_rows + program * NUM_EXPERTS + experts, row, mask=valid_experts)
        tl.atomic_add(totals + experts, row, mask=valid_experts)
        _grid_sync(grid_sync, _GRID_SYNC_TAG, _PLANNER_PROGRAMS)

        # Phase 2: each block handles ranks program, program + grid size, ... .
        if program < PLACEMENT_PROGRAMS:
            owned_ranks = program + tl.arange(0, RANKS_PER_PROGRAM) * _PLANNER_PROGRAMS
            valid_owned_ranks = owned_ranks < EP_SIZE
            if EXCHANGE:
                histogram = tl.load(totals + experts, mask=valid_experts, other=0)
                # Use the original local address rather than the symmetric-memory alias so the
                # release/acquire ordering also covers same-GPU loads of our own histogram.
                peer_counts = tl.load(
                    peer_bases.to(tl.pointer_type(tl.int64)) + owned_ranks,
                    mask=valid_owned_ranks,
                    other=0,
                )
                counts_base = tl.where(
                    owned_ranks == source_rank, counts_sym_mem.to(tl.int64), peer_counts
                )
                peer_rows = counts_base.to(tl.pointer_type(tl.int32)) + source_rank * NUM_EXPERTS
                tl.store(
                    peer_rows[:, None] + experts[None, :],
                    histogram[None, :],
                    mask=valid_owned_ranks[:, None] & valid_experts[None, :],
                )
                # Exchange both peers together: no rank can block waiting for an unscheduled peer.
                # Completed/inactive lanes can retarget this block's scratch word until remapping.
                _rendezvous(
                    signal_bases,
                    source_rank,
                    owned_ranks,
                    valid_owned_ranks,
                    running_counts + program * NUM_EXPERTS,
                    LABEL="virtual-expert planner: histogram exchange stalled",
                )
                _grid_sync(placement_sync, _GRID_SYNC_TAG, PLACEMENT_PROGRAMS)
            if program == 0:
                tl.store(
                    totals + experts,
                    tl.zeros((BLOCK_NUM_EXPERTS,), dtype=tl.int32),
                    mask=valid_experts,
                )

            # Keep every owned rank's expert totals across the balance barrier, avoiding a
            # second read of the gathered histogram when a block handles two ranks.
            owned_experts = tl.arange(0, RANKS_PER_PROGRAM * BLOCK_NUM_EXPERTS_PER_GPU)
            owners = program + (owned_experts // BLOCK_NUM_EXPERTS_PER_GPU) * _PLANNER_PROGRAMS
            local_ids = owned_experts % BLOCK_NUM_EXPERTS_PER_GPU
            native_ids = owners * NUM_EXPERTS_PER_GPU + local_ids
            source_counts = tl.load(
                counts_sym_mem + ranks[:, None] * NUM_EXPERTS + native_ids[None, :],
                mask=valid_ranks[:, None]
                & (owners[None, :] < EP_SIZE)
                & (local_ids[None, :] < NUM_EXPERTS_PER_GPU),
                other=0,
            )
            owned_totals = tl.sum(source_counts, axis=0).to(tl.int32)
            owned_prefix = tl.sum(
                tl.where(ranks[:, None] < source_rank, source_counts, 0), axis=0
            ).to(tl.int32)
            # Round each global logical expert exactly once. Placement operates in blocks;
            # ordinals remain real routes, with conceptual padding only at each expert's end.
            owned_totals = tl.cdiv(owned_totals, ALIGNMENT)

            source_total = tl.sum(
                tl.load(
                    counts_sym_mem + owned_ranks[:, None] * NUM_EXPERTS + experts[None, :],
                    mask=valid_owned_ranks[:, None] & valid_experts[None, :],
                    other=0,
                ),
                axis=1,
            )
            tl.device_assert(
                (source_total == NUM_ROUTES) | ~valid_owned_ranks,
                "virtual-expert planner: a rank's route count differs from tokens * topk",
            )
            tl.store(
                balance + owned_ranks,
                tl.sum(
                    tl.reshape(owned_totals, (RANKS_PER_PROGRAM, BLOCK_NUM_EXPERTS_PER_GPU)), axis=1
                ),
                mask=valid_owned_ranks,
            )
            _grid_sync(placement_sync, _GRID_SYNC_TAG, PLACEMENT_PROGRAMS)

            local_experts = tl.arange(0, BLOCK_NUM_EXPERTS_PER_GPU)
            valid_local_experts = local_experts < NUM_EXPERTS_PER_GPU
            for owned in tl.static_range(RANKS_PER_PROGRAM):
                placing_rank = program + owned * _PLANNER_PROGRAMS
                if placing_rank < EP_SIZE:
                    native_experts = placing_rank * NUM_EXPERTS_PER_GPU + local_experts
                    indices = owned * BLOCK_NUM_EXPERTS_PER_GPU + local_experts
                    native_totals = tl.gather(owned_totals, indices, axis=0)
                    routes_before_source = tl.gather(owned_prefix, indices, axis=0)
                    # Pair the most overloaded rank with the emptiest one and move the receiver's
                    # whole deficit from that sender. This can exceed the sender's excess, but each
                    # receiver gets exactly one sender's experts and therefore fits its available
                    # virtual-expert slots. Moving only min(excess, deficit) could draw from several
                    # senders and overflow those slots.
                    loads = tl.load(balance + ranks, mask=valid_ranks, other=0)
                    if ALIGNMENT == 1:
                        targets = NUM_ROUTES
                    else:
                        total_blocks = tl.sum(loads, 0)
                        targets = total_blocks // EP_SIZE + (ranks < total_blocks % EP_SIZE)
                    balances = tl.where(valid_ranks, loads - targets, 0)
                    # quotas[d] counts placement units sent to d: rows at A=1, blocks otherwise.
                    quotas = tl.zeros((BLOCK_EP_SIZE,), dtype=tl.int32)
                    quota_step = 0
                    while (quota_step < EP_SIZE) & (tl.max(balances, 0) > 0):
                        quota_step += 1
                        _, overloaded = _argmax_lowest(balances, ranks, valid_ranks, BLOCK_EP_SIZE)
                        moved, receiver = _argmax_lowest(
                            -balances, ranks, valid_ranks, BLOCK_EP_SIZE
                        )
                        quotas = tl.where(
                            (overloaded == placing_rank) & (ranks == receiver), moved, quotas
                        )
                        balances = tl.where(ranks == overloaded, balances - moved, balances)
                        balances = tl.where(ranks == receiver, 0, balances)
                    # allocations[local_expert, destination_rank]
                    allocations = tl.where(
                        ranks[None, :] == placing_rank, native_totals[:, None], 0
                    )
                    remaining = native_totals
                    placement_step = 0
                    while (placement_step < EP_SIZE + NUM_EXPERTS_PER_GPU) & (
                        tl.max(quotas, 0) > 0
                    ):
                        placement_step += 1
                        max_quota, destination = _argmax_lowest(
                            quotas, ranks, valid_ranks, BLOCK_EP_SIZE
                        )
                        max_remaining, local_expert = _argmax_lowest(
                            remaining, local_experts, valid_local_experts, BLOCK_NUM_EXPERTS_PER_GPU
                        )
                        moved = tl.minimum(max_quota, max_remaining).to(tl.int32)
                        transfer = tl.where(
                            ranks[None, :] == destination,
                            moved,
                            tl.where(ranks[None, :] == placing_rank, -moved, 0),
                        )
                        allocations += tl.where(local_experts[:, None] == local_expert, transfer, 0)
                        remaining = tl.where(
                            local_experts == local_expert, remaining - moved, remaining
                        )
                        quotas = tl.where(ranks == destination, quotas - moved, quotas)
                    allocations *= ALIGNMENT
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

            _grid_sync(placement_sync, _GRID_SYNC_TAG, PLACEMENT_PROGRAMS)

            for owned in tl.static_range(RANKS_PER_PROGRAM):
                placing_rank = program + owned * _PLANNER_PROGRAMS
                if placing_rank < EP_SIZE:
                    owner = experts // NUM_EXPERTS_PER_GPU
                    valid_remote = (experts < NUM_EXPERTS) & (owner != placing_rank)
                    counts = tl.load(
                        allocation + experts * EP_SIZE + placing_rank, mask=valid_remote, other=-1
                    )
                    for slot in tl.range(0, NUM_EXPERTS_PER_GPU, 1, loop_unroll_factor=1):
                        maximum, expert = _argmax_highest(counts, experts, valid_remote)
                        selected = tl.where(maximum > 0, expert, -1).to(tl.int32)
                        tl.store(
                            experts_to_copy + placing_rank * NUM_EXPERTS_PER_GPU + slot, selected
                        )
                        tl.store(
                            slots + selected * EP_SIZE + placing_rank, slot, mask=selected >= 0
                        )
                        counts = tl.where(experts == expert, -1, counts)

                    tl.device_assert(
                        tl.max(tl.where(valid_remote, counts, -1), axis=0) <= 0,
                        "virtual-expert placement needs more virtual-expert slots than experts",
                    )

        _grid_sync(grid_sync, _GRID_SYNC_TAG, _PLANNER_PROGRAMS)

    @triton.jit
    def _reroute_virtual_expert_routes_kernel(
        top_indices,
        probs,
        virtual_experts,
        runtime_probs,
        scratch,
        NUM_TOKENS: tl.constexpr,
        ROUTER_TOPK: tl.constexpr,
        EP_SIZE: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
    ):
        """PR #7450 stable sort/scan remap, emitting the existing dense scheduler IR."""
        ROUTE_TILE: tl.constexpr = _ROUTE_TILE
        GATHER_RUNNING: tl.constexpr = NUM_EXPERTS <= 512
        NUM_EXPERTS_PER_GPU: tl.constexpr = NUM_EXPERTS // EP_SIZE
        BLOCK_EP_SIZE: tl.constexpr = 1 << (EP_SIZE - 1).bit_length()
        BLOCK_NUM_EXPERTS: tl.constexpr = 1 << (NUM_EXPERTS - 1).bit_length()
        HISTOGRAM_TILE: tl.constexpr = min(16, 8192 // BLOCK_NUM_EXPERTS)
        _, _, boundaries, slots, histogram_rows, running_counts, _ = _planner_fields(
            scratch, EP_SIZE, NUM_EXPERTS
        )
        program = tl.program_id(0)
        experts = tl.arange(0, BLOCK_NUM_EXPERTS)
        valid_experts = experts < NUM_EXPERTS
        TOKENS_PER_PROGRAM: tl.constexpr = tl.cdiv(NUM_TOKENS, _PLANNER_PROGRAMS)
        program_start = program * TOKENS_PER_PROGRAM * ROUTER_TOPK
        program_end = tl.minimum((program + 1) * TOKENS_PER_PROGRAM, NUM_TOKENS) * ROUTER_TOPK
        flat = tl.arange(0, ROUTE_TILE)
        # Each program owns whole token rows, so dense clearing and scattering
        # stay in this kernel without a cross-program race or an extra memset.
        for row_start in tl.range(
            program * TOKENS_PER_PROGRAM,
            tl.minimum((program + 1) * TOKENS_PER_PROGRAM, NUM_TOKENS),
            16,
        ):
            rows = row_start + tl.arange(0, 16)
            for col_start in tl.range(0, 2 * NUM_EXPERTS, 256):
                columns = col_start + tl.arange(0, 256)
                tl.store(
                    runtime_probs + rows[:, None] * (2 * NUM_EXPERTS) + columns[None, :],
                    0.0,
                    mask=(rows[:, None] < program_end // ROUTER_TOPK)
                    & (columns[None, :] < 2 * NUM_EXPERTS),
                )
        tl.debug_barrier()
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
        for route_start in tl.range(program_start, program_end, ROUTE_TILE, loop_unroll_factor=1):
            # Small histograms stay in registers; larger layouts retain the memory fallback.
            if not GATHER_RUNNING:
                tl.store(
                    running_counts + program * NUM_EXPERTS + experts, running, mask=valid_experts
                )
                tl.debug_barrier()
            route_offsets = route_start + flat
            valid = route_offsets < program_end
            ids = tl.load(top_indices + route_offsets, mask=valid, other=0).to(tl.int32)
            packed = tl.sort(
                tl.where(valid, ids, NUM_EXPERTS) * ROUTE_TILE + flat, descending=False
            )
            sorted_ids = packed // ROUTE_TILE
            previous_ids = tl.gather(sorted_ids, tl.maximum(flat - 1, 0), axis=0)
            first = (flat == 0) | (sorted_ids != previous_ids)
            starts = tl.associative_scan(tl.where(first, flat, 0), 0, _prefix_maximum)
            earlier = flat - starts
            valid = sorted_ids < NUM_EXPERTS
            ids = tl.where(valid, sorted_ids, 0)
            original = packed % ROUTE_TILE
            route_offsets = route_start + original
            if GATHER_RUNNING:
                ordinal = tl.gather(running, ids, axis=0)
            else:
                ordinal = tl.load(running_counts + program * NUM_EXPERTS + ids, mask=valid, other=0)
            ordinal += earlier
            # Find the first cumulative allocation above this route's ordinal.
            # The final boundary exceeds every valid local ordinal, so skip it.
            destination = tl.zeros((ROUTE_TILE,), dtype=tl.int32)
            for bit in tl.static_range((EP_SIZE - 1).bit_length() - 1, -1, -1):
                probe = destination + (1 << bit)
                end = tl.load(
                    boundaries + ids * BLOCK_EP_SIZE + probe - 1,
                    mask=valid & (probe < EP_SIZE),
                    other=2147483647,
                )
                destination = tl.where(end <= ordinal, probe, destination)
            remote = valid & (destination != ids // NUM_EXPERTS_PER_GPU)
            slot = tl.load(slots + ids * EP_SIZE + destination, mask=remote, other=0)
            runtime = destination * (2 * NUM_EXPERTS_PER_GPU) + tl.where(
                remote, NUM_EXPERTS_PER_GPU + slot, ids % NUM_EXPERTS_PER_GPU
            )
            tl.store(virtual_experts + route_offsets, runtime.to(tl.int16), mask=valid)
            probability = tl.load(probs + route_offsets, mask=valid, other=0.0)
            tl.store(
                runtime_probs + (route_offsets // ROUTER_TOPK) * (2 * NUM_EXPERTS) + runtime,
                probability.to(tl.float32),
                mask=valid,
            )
            running += tl.histogram(ids, BLOCK_NUM_EXPERTS, mask=valid)


def _validate_launch(top_indices: torch.Tensor, workspace) -> tuple[int, int, int, int]:
    """Validate common placement/reroute launch dimensions."""
    _require_triton()
    if top_indices.dim() != 2 or top_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("MoonEP virtual-expert planning requires 2D integer top_indices.")
    num_tokens, router_topk = top_indices.shape
    ep_size, num_experts = workspace.ep_size, workspace.num_experts
    if ep_size > MAX_REPLICA_EP_RANKS or num_experts > 8192:
        raise ValueError(
            f"Virtual-expert planner supports at most {MAX_REPLICA_EP_RANKS} EP ranks "
            "and 8192 experts."
        )
    return num_tokens, router_topk, ep_size, num_experts


def launch_virtual_expert_placement(
    top_indices: torch.Tensor, workspace, *, exchange: bool = True, alignment: int = 1
) -> torch.Tensor:
    """Compute placement and return its semantic expert ids per replica slot.

    Without ``exchange`` (the process-local test seam), the workspace window
    must already contain every rank's histogram.
    """
    num_tokens, router_topk, ep_size, num_experts = _validate_launch(top_indices, workspace)
    empty = functools.partial(torch.empty, device=top_indices.device)
    experts_to_copy = empty((ep_size, num_experts // ep_size), dtype=torch.int32)
    if not isinstance(alignment, int) or isinstance(alignment, bool) or alignment < 1:
        raise ValueError("MoonEP alignment must be a positive integer.")
    handle = workspace.histogram_handle
    arguments = (
        top_indices,
        experts_to_copy,
        workspace.scratch,
        workspace.gathered_counts,
        workspace.rank,
        int(handle.buffer_ptrs_dev) if exchange else 0,
        int(handle.signal_pad_ptrs_dev) if exchange else 0,
    )
    key = (
        num_tokens,
        router_topk,
        alignment,
        exchange,
        top_indices.dtype,
        top_indices.data_ptr() % 16,
    )
    kernel = workspace.placement_kernels.get(key)
    if kernel is not None:
        kernel[(PLANNER_PROGRAMS, 1, 1)](*arguments)
        return experts_to_copy
    workspace.placement_kernels[key] = _update_virtual_expert_placement_kernel[(PLANNER_PROGRAMS,)](
        *arguments,
        NUM_TOKENS=num_tokens,
        ROUTER_TOPK=router_topk,
        EP_SIZE=ep_size,
        NUM_EXPERTS=num_experts,
        ALIGNMENT=alignment,
        EXCHANGE=exchange,
        launch_cooperative_grid=True,
        num_warps=4,
    )
    return experts_to_copy


def launch_virtual_expert_reroute(
    top_indices: torch.Tensor, probs: torch.Tensor, workspace
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map tokens to a previously computed placement."""
    num_tokens, router_topk, ep_size, num_experts = _validate_launch(top_indices, workspace)
    if probs.shape != top_indices.shape:
        raise ValueError("MoonEP reroute takes matching [num_tokens, topk] ids and probs.")
    empty = functools.partial(torch.empty, device=top_indices.device)
    virtual_experts = empty((num_tokens, router_topk), dtype=torch.int16)
    runtime_probs = empty((num_tokens, 2 * num_experts), dtype=torch.float32)
    arguments = (top_indices, probs, virtual_experts, runtime_probs, workspace.scratch)
    key = (
        num_tokens,
        router_topk,
        top_indices.dtype,
        probs.dtype,
        top_indices.data_ptr() % 16,
        probs.data_ptr() % 16,
    )
    kernel = workspace.reroute_kernels.get(key)
    if kernel is not None:
        kernel[(PLANNER_PROGRAMS, 1, 1)](*arguments)
        return virtual_experts, runtime_probs
    workspace.reroute_kernels[key] = _reroute_virtual_expert_routes_kernel[(PLANNER_PROGRAMS,)](
        *arguments,
        NUM_TOKENS=num_tokens,
        ROUTER_TOPK=router_topk,
        EP_SIZE=ep_size,
        NUM_EXPERTS=num_experts,
        num_warps=4,
    )
    return virtual_experts, runtime_probs


def launch_virtual_expert_planner(
    top_indices: torch.Tensor,
    probs: torch.Tensor,
    workspace,
    *,
    exchange: bool = True,
    alignment: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compatibility wrapper that runs placement followed by rerouting."""
    experts_to_copy = launch_virtual_expert_placement(
        top_indices, workspace, exchange=exchange, alignment=alignment
    )
    virtual_experts, runtime_probs = launch_virtual_expert_reroute(top_indices, probs, workspace)
    return virtual_experts, runtime_probs, experts_to_copy
