# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `symm_mem_sync` (megatron/core/inference/communication/torch_symm_triton/barrier.py:65),
the cross-rank barrier every NVLS collective uses on entry and exit.

Contract (from its docstring and its use in the collectives): blocks with the same block id on
every rank of a group meet. With hasPreviousMemAccess=True a block's earlier stores (all its
threads', ordered by a bar.sync before the call) are released; with hasSubsequentMemAccess=True
the peers' released stores are visible to the block's later loads (after a bar.sync). With both
False (the entry barrier of the reduce-scatter, fused and reduce-scatter-v kernels) it only
waits: the data it protects was written by earlier kernels. Each (block, peer) pair owns one
int32 slot per rank's signal pad, at block_id * ws + rank; a completed barrier leaves every
slot at 0 again (no epoch, so it is CUDA-graph safe), and the pad holds the slots of the largest
grid (128 blocks; kernel_config.py). ws must be a power of two (tl.arange).

No library barrier exposes this device-level contract, so the oracle is a property test: small
Triton kernels use the barrier the way the collectives do and pass messages through it.
  * publish/read: every block stores a per-round token into its slot of every peer's buffer
    (P2P stores), barriers (release/acquire), and reads every peer's token from its own
    buffer. Rounds alternate between two slot sets, so a rank can be at most one round ahead.
  * entry/exit: each rank writes its token with a torch kernel; the probe enters with the
    relaxed barrier the reduce-scatter kernels use, reads every peer's token over P2P, and
    leaves with the acq_rel barrier, as the fused kernel does.
  * blocks meet their own block id: publish/read with every block of every rank delayed by its
    own random time before publishing, so a peer's other blocks signal long before the matching
    block has published. A barrier that matched blocks by count instead of by block id (any
    block's signal releasing a waiter) would let a block read a slot its peer block has not
    written yet. Delaying whole ranks cannot show this: all blocks of a rank arrive together.
  Every read must return this round's token: a barrier that lets a block through early returns
  the previous round's. Many rounds back to back, a random rank delayed before each, eager and
  replayed from a CUDA graph; grids of 1, 2, 127 and 128 blocks (the slot range), ws in
  {2, 4, 8}. Afterwards every slot is 0 and the pad holds 128 * ws slots.

Tolerance class: exact (integer tokens). Covers main and #8868 (t3-r04f3), which masks the
barrier's peer loads for padded world sizes but leaves existing callers' behaviour unchanged.
"""

import pytest
import torch

from megatron.core.inference.communication.torch_symm_triton.barrier import symm_mem_sync
from megatron.core.inference.communication.torch_symm_triton.kernel_config import (
    SYMM_MEM_KERNEL_CONFIG,
)
from megatron.core.inference.communication.torch_symm_triton.utils import sync_threads
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.nvls import nvls_oracle_utils as nu

try:
    import triton
    import triton.language as tl
except ImportError:  # collected without Triton; the GPU tests skip before running
    triton = None

MAX_NUM_BLOCKS = SYMM_MEM_KERNEL_CONFIG.max_num_blocks
GRIDS = [1, 2, MAX_NUM_BLOCKS - 1, MAX_NUM_BLOCKS]
ROUNDS = 40
BLOCK_DELAY_MAX_NS = 20_000
"""Upper bound of the random per-block delay before a block publishes: well above the NVLink
signal round trip and the launch skew between the blocks of one grid (a few microseconds)."""


def gpu_test(fn):
    """NVLS needs Hopper or Blackwell; GB200 CI launches the file through launch_on_gb200."""
    return pytest.mark.launch_on_gb200(ou.requires_sm(ou.HOPPER, ou.BLACKWELL)(fn))


if triton is not None:

    @triton.jit
    def _global_timer_ns():
        """The GPU's global nanosecond timer."""
        return tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
        )

    @triton.jit
    def _spin_ns(ns):
        """Busy-wait at least ``ns`` nanoseconds."""
        start = _global_timer_ns()
        now = start
        while now - start < ns:
            now = _global_timer_ns()

    @triton.jit
    def _publish_then_read_kernel(
        buffer_ptrs,
        signal_pad_ptrs,
        out_ptr,
        token_ptr,
        delay_ptr,
        parity,
        RANK: tl.constexpr,
        WORLD_SIZE: tl.constexpr,
        BLOCK_DELAYS: tl.constexpr,
    ):
        """Store this rank's token in every peer's slot [parity, block, RANK], barrier, then
        read slots [parity, block, :] of this rank's buffer. With BLOCK_DELAYS, block ``b``
        first waits ``delay_ptr[b]`` ns."""
        pid = tl.program_id(0)
        if BLOCK_DELAYS:
            _spin_ns(tl.load(delay_ptr + pid).to(tl.int64))
        peers = tl.arange(0, WORLD_SIZE)
        token = tl.load(token_ptr)
        ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
        slot = (parity * tl.num_programs(0) + pid) * WORLD_SIZE
        peer_bases = tl.load(ptrs + peers).to(tl.pointer_type(tl.int32))
        tl.store(peer_bases + slot + RANK, token + tl.zeros([WORLD_SIZE], dtype=tl.int32))
        sync_threads()
        symm_mem_sync(
            signal_pad_ptrs,
            None,
            RANK,
            WORLD_SIZE,
            hasPreviousMemAccess=True,
            hasSubsequentMemAccess=True,
        )
        sync_threads()
        local = tl.load(ptrs + RANK).to(tl.pointer_type(tl.int32))
        values = tl.load(local + slot + peers)
        tl.store(out_ptr + pid * WORLD_SIZE + peers, values)

    @triton.jit
    def _entry_read_exit_kernel(
        buffer_ptrs, signal_pad_ptrs, out_ptr, RANK: tl.constexpr, WORLD_SIZE: tl.constexpr
    ):
        """Relaxed entry barrier (as the reduce-scatter kernels), read every peer's slot [block]
        over P2P, then the acq_rel exit barrier (as the fused and all-gather kernels)."""
        pid = tl.program_id(0)
        symm_mem_sync(
            signal_pad_ptrs,
            None,
            RANK,
            WORLD_SIZE,
            hasPreviousMemAccess=False,
            hasSubsequentMemAccess=False,
        )
        sync_threads()
        peers = tl.arange(0, WORLD_SIZE)
        ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
        peer_bases = tl.load(ptrs + peers).to(tl.pointer_type(tl.int32))
        values = tl.load(peer_bases + pid)
        tl.store(out_ptr + pid * WORLD_SIZE + peers, values)
        sync_threads()
        symm_mem_sync(
            signal_pad_ptrs,
            None,
            RANK,
            WORLD_SIZE,
            hasPreviousMemAccess=True,
            hasSubsequentMemAccess=True,
        )


def token(round_index, rank):
    """The value rank `rank` publishes in a round (never 0 or -1, the initial fills)."""
    return 1000 * (round_index + 1) + rank + 1


def expected_tokens(round_index, grid, ws):
    row = torch.tensor([token(round_index, q) for q in range(ws)], dtype=torch.int32)
    return row.expand(grid, ws).cuda()


def _signal_pad_slots(hdl, ws):
    return hdl.get_signal_pad(hdl.rank, (MAX_NUM_BLOCKS * ws,), torch.int32)


def check_signal_pad(g, slots_before):
    """Every slot the collectives use is back at 0, and the pad holds all of them."""
    ou.assert_exact(slots_before, torch.zeros_like(slots_before), what="barrier slots after use")
    pad_bytes = getattr(g.buffer().hdl, "signal_pad_size", None)
    if pad_bytes is not None:
        needed = SYMM_MEM_KERNEL_CONFIG.signal_pad_bytes(g.ws)
        assert pad_bytes >= needed, (
            f"signal pad of {pad_bytes} bytes; a {MAX_NUM_BLOCKS}-block barrier at ws={g.ws}"
            f" uses {needed}"
        )


@gpu_test
@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("mode", ["eager", "graph"])
@pytest.mark.parametrize("ws", nu.group_size_params())
def test_publish_then_read_sees_this_round(nvls_groups, ws, mode, grid):
    """Release/acquire barrier: every block reads every peer's token of this round."""
    g = nvls_groups.get(ws)
    symm = g.buffer()
    hdl = symm.hdl
    slots = symm.view((2 * grid * ws,), torch.int32)
    slots.fill_(-1)
    tok = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.zeros(grid, ws, dtype=torch.int32, device="cuda")
    g.sync()

    def launch(parity):
        _publish_then_read_kernel[(grid,)](
            hdl.buffer_ptrs_dev,
            hdl.signal_pad_ptrs_dev,
            out,
            tok,
            tok,  # no per-block delays
            parity,
            RANK=hdl.rank,
            WORLD_SIZE=ws,
            BLOCK_DELAYS=False,
            num_warps=1,
        )

    graphs = {}
    if mode == "graph":
        for parity in (0, 1):
            tok.fill_(token(-1, g.rank))
            launch(parity)  # compile and warm up
            torch.cuda.synchronize()
            graphs[parity] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[parity]):
                launch(parity)
        nu.sync_world()
    shared = g.shared_generator(nu.case_seed("publish", ws, mode, grid))
    results = []
    for r in range(ROUNDS):
        tok.fill_(token(r, g.rank))
        g.maybe_sleep(shared)
        if mode == "graph":
            graphs[r % 2].replay()
        else:
            launch(r % 2)
        results.append(out.clone())
    nu.sync_world()
    pad = _signal_pad_slots(hdl, ws).clone()

    def check():
        for r, got in enumerate(results):
            ou.assert_exact(got, expected_tokens(r, grid, ws), what=f"round {r}: tokens read")
        check_signal_pad(g, pad)

    nu.agree(check, f"publish/read barrier ({mode}, {grid} blocks)")


@gpu_test
@pytest.mark.parametrize("grid", [g for g in GRIDS if g > 1])
@pytest.mark.parametrize("ws", nu.group_size_params())
def test_blocks_meet_their_own_block_id(nvls_groups, ws, grid):
    """Release/acquire barrier with every block delayed by its own random time (different on
    every rank, block and round) before it publishes: block ``b`` must still wait for block
    ``b`` of every peer, so it reads every peer's token of this round."""
    g = nvls_groups.get(ws)
    symm = g.buffer()
    hdl = symm.hdl
    slots = symm.view((2 * grid * ws,), torch.int32)
    slots.fill_(-1)
    tok = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.zeros(grid, ws, dtype=torch.int32, device="cuda")
    delays = torch.zeros(grid, dtype=torch.int64, device="cuda")
    own = g.rank_generator(nu.case_seed("block-delays", ws, grid))
    g.sync()
    results = []
    for r in range(ROUNDS):
        tok.fill_(token(r, g.rank))
        delays.random_(0, BLOCK_DELAY_MAX_NS, generator=own)
        _publish_then_read_kernel[(grid,)](
            hdl.buffer_ptrs_dev,
            hdl.signal_pad_ptrs_dev,
            out,
            tok,
            delays,
            r % 2,
            RANK=hdl.rank,
            WORLD_SIZE=ws,
            BLOCK_DELAYS=True,
            num_warps=1,
        )
        results.append(out.clone())
    nu.sync_world()
    pad = _signal_pad_slots(hdl, ws).clone()

    def check():
        for r, got in enumerate(results):
            ou.assert_exact(got, expected_tokens(r, grid, ws), what=f"round {r}: tokens read")
        check_signal_pad(g, pad)

    nu.agree(check, f"publish/read barrier with per-block delays ({grid} blocks)")


@gpu_test
@pytest.mark.parametrize(
    "grid", [pytest.param(g, marks=[pytest.mark.sanitizer] if g == 2 else []) for g in GRIDS]
)
@pytest.mark.parametrize("ws", nu.group_size_params())
def test_relaxed_entry_barrier_sees_previous_kernel_writes(nvls_groups, ws, grid):
    """The reduce-scatter kernels' relaxed entry barrier, after each rank's own token write (a
    torch kernel), lets every block read every peer's token of this round; the acq_rel exit
    barrier keeps the next round's write from landing before peers have read."""
    g = nvls_groups.get(ws)
    symm = g.buffer()
    hdl = symm.hdl
    slots = symm.view((grid,), torch.int32)
    out = torch.zeros(grid, ws, dtype=torch.int32, device="cuda")
    shared = g.shared_generator(nu.case_seed("entry", ws, grid))
    results = []
    for r in range(ROUNDS):
        slots.fill_(token(r, g.rank))
        g.maybe_sleep(shared)
        _entry_read_exit_kernel[(grid,)](
            hdl.buffer_ptrs_dev,
            hdl.signal_pad_ptrs_dev,
            out,
            RANK=hdl.rank,
            WORLD_SIZE=ws,
            num_warps=1,
        )
        results.append(out.clone())
    nu.sync_world()
    pad = _signal_pad_slots(hdl, ws).clone()

    def check():
        for r, got in enumerate(results):
            ou.assert_exact(got, expected_tokens(r, grid, ws), what=f"round {r}: tokens read")
        check_signal_pad(g, pad)

    nu.agree(check, f"entry/exit barriers ({grid} blocks)")
