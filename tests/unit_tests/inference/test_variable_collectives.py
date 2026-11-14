# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Row offsets of the variable-count NVLS collectives used by the inference MoE dispatcher.

``multimem_all_gatherv_3tensor`` gathers the dispatched tokens into symmetric buffers that hold
every EP rank's rows, and ``multimem_reduce_scatter_v`` (or, in batch-invariant mode,
``ordered_reduce_scatter_v``) reduces the expert outputs back from such a buffer. Each rank
addresses its rows at ``rank_token_offset``, so the row offsets grow with the whole step's
tokens. Each case splits ``rows`` tokens of hidden size 7168 (DeepSeek-V3) over the ranks, fills
the buffers with closed-form values (exact under any summation order) and checks every row on
every rank. The ``large`` cases (opt-in, ``-m large``; 16 GiB symmetric buffers) put the last
rows where the offsets reach 2**31 in the unit the kernel indexes: 8-byte units of the 16-byte
lanes for the multimem kernels, fp32 elements for the ordered reduce-scatter.

``TestAllGatherVZeroTokenStep`` checks that a dispatch in which no rank has tokens still
synchronizes the group, which the dispatcher's metadata buffer relies on.
"""

import time

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference.communication.torch_symm_triton import (
    is_device_nvls_capable,
    multimem_all_gatherv_3tensor,
    multimem_reduce_scatter_v,
)
from megatron.core.inference.moe.batch_invariant import ordered_reduce_scatter_v
from megatron.core.inference.symmetric_memory import SymmetricMemoryBuffer
from tests.unit_tests.test_utilities import Utils

HIDDEN = 7168
TOPK = 8
SMALL_ROWS = 2 * 129 + 5
"""More rows per rank than the 128-block grid at two ranks, so the grid-stride loop runs."""


def _limit_rows(row_units):
    """The fewest rows, ``row_units`` units each, whose last row starts at or past 2**31 units:
    the row before it crosses 2**31."""
    return -(-(2**31) // row_units) + 1


# Units per row as each kernel indexes them: 16-byte lanes times 2 (8-byte units) for the
# multimem kernels, fp32 elements for the ordered reduce-scatter.
COMBINE_ROWS = {
    "multimem": _limit_rows(HIDDEN * 4 // 16 * 2),  # 599188 fp32 rows, 16 GiB
    "ordered": _limit_rows(HIDDEN),  # 299595 fp32 rows, 8 GiB
}
GATHER_ROWS = _limit_rows(HIDDEN * 2 // 16 * 2)  # 1198374 bf16 rows, 16 GiB
COMBINES = {"multimem": multimem_reduce_scatter_v, "ordered": ordered_reduce_scatter_v}


def _values(owner, first, count, width, dtype):
    """Rows ``[first, first + count)`` of rank ``owner``'s data: multiples of 1/16 below 2 in
    magnitude, so their fp32 sums over the ranks are exact in any order, and different between
    neighbouring rows, columns and ranks."""
    r = torch.arange(first, first + count, device="cuda", dtype=torch.int64)[:, None]
    c = torch.arange(width, device="cuda", dtype=torch.int64)[None, :]
    v = ((r % 61) * 3 + (r // 61) % 17 + 7 * c + 29 * owner) % 61 - 30
    return (v.to(torch.float32) / 16).to(dtype)


def _chunks(start, stop, width, elements=1 << 25):
    """``(first, count)`` chunks of rows ``[start, stop)`` of at most ``elements`` elements."""
    step = max(1, elements // width)
    for first in range(start, stop, step):
        yield first, min(step, stop - first)


def _mismatch(actual, expected, what):
    """A description of where ``actual`` differs from ``expected``, or None."""
    differ = (actual != expected).nonzero()
    if differ.numel() == 0:
        return None
    return f"{what}: {differ.shape[0]} elements differ, first at {tuple(differ[0].tolist())}"


def _agree(errors, what):
    """Fail on every rank if the check failed on any rank."""
    failed = torch.tensor([len(errors)], device="cuda")
    dist.all_reduce(failed)
    assert not errors, "; ".join(errors)
    assert int(failed) == 0, f"{what} failed on another rank"


class TestVariableCollectiveRowOffsets:

    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def _setup(self, rows, nbytes, local_row_bytes):
        """The ranks' row counts, this rank's first row and row count, and symmetric buffers of
        ``nbytes`` each; skips on every rank unless every rank has NVLS multicast and the memory
        (the buffers, ``local_row_bytes`` per local row and room for the checks)."""
        world = dist.get_world_size() if dist.is_initialized() else 1
        if world < 2:
            pytest.skip("needs at least two ranks")
        if not is_device_nvls_capable(torch.device("cuda", torch.cuda.current_device())):
            pytest.skip("NVLS multicast needs a Hopper or newer GPU")
        rank = dist.get_rank()
        counts = [rows // world + (q < rows % world) for q in range(world)]
        first, mine = sum(counts[:rank]), counts[rank]
        free = torch.tensor([torch.cuda.mem_get_info()[0]], dtype=torch.int64, device="cuda")
        dist.all_reduce(free, op=dist.ReduceOp.MIN)
        need = sum(nbytes) + mine * local_row_bytes + (2 << 30)
        if int(free) < need:
            pytest.skip(f"needs {need} bytes free on every rank, the least is {int(free)}")
        buffers = [SymmetricMemoryBuffer(-(-n // 2**20), dist.group.WORLD) for n in nbytes]
        problems = [b.init_failure_reason for b in buffers if b.symm_mem_hdl is None]
        problems += [
            "no NVLink multicast"
            for b in buffers
            if b.symm_mem_hdl is not None and not b.symm_mem_hdl.multicast_ptr
        ]
        anywhere = torch.tensor([len(problems)], device="cuda")
        dist.all_reduce(anywhere)
        if int(anywhere):
            pytest.skip(f"symmetric memory with multicast unavailable: {problems or 'on a peer'}")
        return counts, first, mine, buffers

    def _run_combine(self, combine, rows):
        counts, first, mine, (buf,) = self._setup(rows, [rows * HIDDEN * 4], HIDDEN * 4)
        rank = dist.get_rank()
        # Each rank's partial sums of every row, as the expert GEMMs leave them.
        partials = buf.symm_buffer[: rows * HIDDEN * 4].view(torch.float32).view(rows, HIDDEN)
        for start, count in _chunks(0, rows, HIDDEN):
            partials[start : start + count] = _values(rank, start, count, HIDDEN, torch.float32)
        out = torch.full((mine, HIDDEN), float("nan"), dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        dist.barrier()
        COMBINES[combine](
            out,
            partials,
            buf.symm_mem_hdl,
            rank_token_offset=torch.tensor([first], dtype=torch.int32, device="cuda"),
            ep_max_tokens=torch.tensor([max(counts)], dtype=torch.int32, device="cuda"),
            per_rank_max_tokens=max(counts),
        )
        errors = []
        for start, count in _chunks(0, mine, HIDDEN):
            expected = sum(
                _values(q, first + start, count, HIDDEN, torch.float32) for q in range(len(counts))
            )
            what = f"{combine} rows {first + start}-{first + start + count - 1} on rank {rank}"
            error = _mismatch(out[start : start + count], expected, what)
            if error:
                errors.append(error)
                break
        torch.cuda.synchronize()
        _agree(errors, f"{combine} reduce-scatter-v of {rows} rows")

    def _run_gather(self, rows):
        specs = [(torch.bfloat16, HIDDEN), (torch.int64, TOPK), (torch.float32, TOPK)]
        row_bytes = [width * dtype.itemsize for dtype, width in specs]
        counts, first, mine, buffers = self._setup(
            rows, [rows * n for n in row_bytes], sum(row_bytes)
        )
        rank = dist.get_rank()
        outputs = []
        for (dtype, width), buf in zip(specs, buffers):
            out = buf.symm_buffer[: rows * width * dtype.itemsize].view(dtype).view(rows, width)
            out.fill_(float("nan") if dtype.is_floating_point else -1)
            outputs.append(out)
        hidden = torch.empty((mine, HIDDEN), dtype=torch.bfloat16, device="cuda")
        for start, count in _chunks(0, mine, HIDDEN):
            hidden[start : start + count] = _values(
                rank, first + start, count, HIDDEN, torch.bfloat16
            )
        routing = torch.arange(first * TOPK, (first + mine) * TOPK, device="cuda").view(mine, TOPK)
        probs = _values(rank, first, mine, TOPK, torch.float32)
        # The kernel has no entry barrier: every rank's fill before any peer's stores.
        torch.cuda.synchronize()
        dist.barrier()
        multimem_all_gatherv_3tensor(
            *outputs,
            hidden,
            routing,
            probs,
            *(buf.symm_mem_hdl for buf in buffers),
            rank_token_offset=torch.tensor([first], dtype=torch.int32, device="cuda"),
            ep_max_tokens=torch.tensor([max(counts)], dtype=torch.int32, device="cuda"),
            per_rank_max_tokens=max(counts),
        )
        errors = []
        out_hidden, out_routing, out_probs = outputs
        for q, count in enumerate(counts):
            lo = sum(counts[:q])
            for start, n in _chunks(lo, lo + count, HIDDEN):
                expected = _values(q, start, n, HIDDEN, torch.bfloat16)
                what = f"hidden rows {start}-{start + n - 1} of rank {q}, read on rank {rank}"
                errors.append(_mismatch(out_hidden[start : start + n], expected, what))
            expected = torch.arange(lo * TOPK, (lo + count) * TOPK, device="cuda")
            what = f"routing rows of rank {q}, read on rank {rank}"
            errors.append(_mismatch(out_routing[lo : lo + count], expected.view(count, TOPK), what))
            expected = _values(q, lo, count, TOPK, torch.float32)
            what = f"probabilities of rank {q}, read on rank {rank}"
            errors.append(_mismatch(out_probs[lo : lo + count], expected, what))
        torch.cuda.synchronize()
        _agree([e for e in errors if e], f"all-gather-v of {rows} rows")

    @pytest.mark.parametrize("combine", ["multimem", "ordered"])
    def test_reduce_scatter_v(self, combine):
        self._run_combine(combine, SMALL_ROWS)

    def test_all_gatherv_3tensor(self):
        self._run_gather(SMALL_ROWS)

    @pytest.mark.large
    @pytest.mark.parametrize("combine", ["multimem", "ordered"])
    def test_reduce_scatter_v_past_int32_offsets(self, combine):
        """The last two rows of the combine buffer cross and start past 2**31 units: 8 GiB of
        fp32 rows for the ordered kernel, 16 GiB for the multimem one."""
        self._run_combine(combine, COMBINE_ROWS[combine])

    @pytest.mark.large
    def test_all_gatherv_3tensor_past_int32_offsets(self):
        """The last two hidden-state rows of the gather buffer cross and start past 16 GiB."""
        self._run_gather(GATHER_ROWS)


class TestAllGatherVZeroTokenStep:
    """A dispatch step in which no rank has tokens (``ep_max_tokens`` 0) must still synchronize
    the EP group. ``fused_metadata_update`` multicasts each rank's token count into every peer's
    buffer before its barrier, and nothing after its read of the counts orders the next step's
    store but the dispatch all-gather's barrier. If that all-gather returned without waiting for
    the peers, a rank could run its next step and overwrite a peer's counts before the peer had
    read them, so the ranks would disagree on the token offsets and on which blocks barrier.

    Rank 0 launches the dispatch all-gather of a zero-token step while every peer holds its launch
    back until rank 0 has watched its own for ``WINDOW_S`` seconds; the all-gather must not finish
    on rank 0 within that window."""

    WINDOW_S = 2.0

    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_zero_token_step_waits_for_every_rank(self):
        world = dist.get_world_size() if dist.is_initialized() else 1
        if world < 2:
            pytest.skip("needs at least two ranks")
        if not is_device_nvls_capable(torch.device("cuda", torch.cuda.current_device())):
            pytest.skip("NVLS multicast needs a Hopper or newer GPU")
        rank = dist.get_rank()
        specs = [(torch.bfloat16, HIDDEN), (torch.int64, TOPK), (torch.float32, TOPK)]
        rows = world * 4
        buffers = [SymmetricMemoryBuffer(1, dist.group.WORLD) for _ in specs]
        problems = [b.init_failure_reason for b in buffers if b.symm_mem_hdl is None]
        problems += [
            "no NVLink multicast"
            for b in buffers
            if b.symm_mem_hdl is not None and not b.symm_mem_hdl.multicast_ptr
        ]
        anywhere = torch.tensor([len(problems)], device="cuda")
        dist.all_reduce(anywhere)
        if int(anywhere):
            pytest.skip(f"symmetric memory with multicast unavailable: {problems or 'on a peer'}")
        # A CPU group, so rank 0 can hold its peers back while its own stream is blocked.
        host_group = dist.new_group(backend="gloo")

        outputs = [
            buf.symm_buffer[: rows * width * dtype.itemsize].view(dtype).view(rows, width)
            for (dtype, width), buf in zip(specs, buffers)
        ]
        # No tokens on any rank.
        inputs = [torch.zeros(1, width, dtype=dtype, device="cuda")[:0] for dtype, width in specs]
        step = dict(
            rank_token_offset=torch.zeros(1, dtype=torch.int32, device="cuda"),
            ep_max_tokens=torch.zeros(1, dtype=torch.int32, device="cuda"),
            per_rank_max_tokens=4,
        )

        def dispatch():
            multimem_all_gatherv_3tensor(
                *outputs, *inputs, *(buf.symm_mem_hdl for buf in buffers), **step
            )

        torch.cuda.synchronize()
        dist.barrier()
        finished_early = False
        if rank == 0:
            dispatch()
            done = torch.cuda.Event()
            done.record()
            deadline = time.monotonic() + self.WINDOW_S
            while not finished_early and time.monotonic() < deadline:
                finished_early = done.query()
                time.sleep(0.01)
        dist.barrier(group=host_group)
        if rank != 0:
            dispatch()
        torch.cuda.synchronize()
        early = torch.tensor([int(finished_early)], device="cuda")
        dist.all_reduce(early)
        dist.destroy_process_group(host_group)
        assert int(early) == 0, (
            "the all-gather-v of a step with no tokens finished on rank 0 before any peer "
            "launched it, so it did not synchronize the group"
        )
