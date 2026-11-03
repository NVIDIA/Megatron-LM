# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared pieces of the NVLS collective oracles (multimem symmetric-memory Triton kernels).

The kernels run on a group of ``ws`` ranks that share symmetric buffers with NVLink multicast.
The oracles run every group size that divides the job (2, 4 and 8 on an 8-GPU H100 node; 2 and
4 on a 4-GPU GB200 node): the job is split into subgroups of ``ws`` ranks, each with its own
symmetric buffers, and every subgroup runs the same case at the same time with its own data.

References are built with NCCL (``torch.distributed``) on the same group: an exact
``all_gather_into_tensor`` of every rank's inputs feeds the fp64 reference, and NCCL's own
reduce-scatter / all-reduce serves as the trusted implementation for bounded comparisons.

Every rank must issue the same collectives in the same order, or the job hangs. So each test
runs its kernels and NCCL calls first and checks afterwards, through ``agree``: a check that
fails on one rank fails the test on every rank, and the job stays in step.
"""

import math
import os
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import pytest
import torch
import torch.distributed as dist

from tests.unit_tests.inference.oracles import oracle_utils as ou

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
"""Ranks in the job (``torch.distributed.run`` sets ``WORLD_SIZE``)."""

GROUP_SIZES: List[int] = [ws for ws in (2, 4, 8) if ws <= WORLD_SIZE and WORLD_SIZE % ws == 0] or [
    2
]
"""Group sizes the oracles run: the powers of two (the barrier's ``tl.arange``) that divide the
job. Outside a multi-rank job this is ``[2]``, and the cases fail or skip at setup (see
``NvlsGroups.get``) instead of disappearing from collection."""

BUFFER_MB = {"main": 80, "aux1": 8, "aux2": 8}
"""Symmetric buffers per group. ``main`` holds the largest case of any module (64 MiB of data
plus sentinels); ``aux1`` and ``aux2`` are the second and third output buffers of the
three-tensor all-gather-v (the dispatcher allocates one buffer per gathered tensor)."""

SENTINEL_BYTES = 64 * 1024
"""Bytes after (and, with a byte offset, before) a kernel's region that must survive it."""

SENTINEL_BYTE = 0xA5

MAX_SLEEP_CYCLES = 200_000
"""Upper bound of the random per-rank delay before a collective (~0.1 ms on H100). A delay on
one rank makes a missing or misplaced barrier observable: its peers run ahead into data the
delayed rank has not written yet, or overwrite data it has not read yet."""


def group_size_params() -> List[Any]:
    """``GROUP_SIZES`` as ``pytest.param`` objects with ids ``ws2``, ``ws4``, ``ws8``."""
    return [pytest.param(ws, id=f"ws{ws}") for ws in GROUP_SIZES]


def case_seed(*parts: Any) -> int:
    """A stable seed for a case, from its parameters (identical on every rank)."""
    return (ou.DEFAULT_SEED + zlib.crc32(repr(parts).encode())) % (2**31)


def rank_seed(seed: int, rank: int) -> int:
    """The seed of group rank ``rank``'s own inputs (distinct per rank)."""
    return seed + 7919 * (rank + 1)


def target_hardware() -> bool:
    """Whether this run declares NVLS hardware (``MCORE_ORACLE_TARGET_SM`` or ``-m``)."""
    return ou.session_target_sm() in (ou.HOPPER, ou.BLACKWELL)


def unavailable(reason: str) -> None:
    """Fail when the run targets NVLS hardware (a skip there is not evidence), else skip."""
    if target_hardware():
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


# ---------------------------------------------------------------------------------------------
# Groups and symmetric buffers
# ---------------------------------------------------------------------------------------------


class Sentinel:
    """``SENTINEL_BYTE`` written over a byte range of a buffer; ``check`` asserts it survived."""

    def __init__(self, raw: torch.Tensor, start: int, nbytes: int = SENTINEL_BYTES):
        start = max(0, start)
        stop = min(raw.numel(), start + nbytes)
        self.start = start
        self.region = raw[start:stop]
        self.region.fill_(SENTINEL_BYTE)

    def check(self, what: str = "symmetric buffer") -> None:
        """Raise if any sentinel byte changed: the kernel wrote outside its region."""
        changed = (self.region != SENTINEL_BYTE).nonzero().reshape(-1)
        if changed.numel():
            first, last = int(changed[0]) + self.start, int(changed[-1]) + self.start
            raise AssertionError(
                f"{what}: {changed.numel()} sentinel bytes changed between buffer bytes {first}"
                f" and {last} (outside the region the collective may write)"
            )


@dataclass
class SymmBuffer:
    """One symmetric buffer of a group (``SymmetricMemoryBuffer``: uint8 tensor and handle)."""

    raw: torch.Tensor
    hdl: Any

    def view(self, shape: Sequence[int], dtype: torch.dtype, byte_offset: int = 0) -> torch.Tensor:
        """The tensor of ``shape``/``dtype`` at ``byte_offset`` in this rank's buffer."""
        nbytes = math.prod(shape) * dtype.itemsize
        if byte_offset % 16 or byte_offset + nbytes > self.raw.numel():
            raise ValueError(
                f"view of {nbytes} bytes at offset {byte_offset} does not fit the"
                f" {self.raw.numel()}-byte symmetric buffer (or is not 16-byte aligned)"
            )
        return self.raw[byte_offset : byte_offset + nbytes].view(dtype).view(tuple(shape))

    def sentinel_after(self, end: int) -> Sentinel:
        """Fill ``SENTINEL_BYTES`` past byte ``end``."""
        return Sentinel(self.raw, end)

    def sentinel_before(self, start: int) -> Optional[Sentinel]:
        """Fill up to ``SENTINEL_BYTES`` before byte ``start`` (None when ``start`` is 0)."""
        if start == 0:
            return None
        return Sentinel(self.raw, start - SENTINEL_BYTES, min(start, SENTINEL_BYTES))


@dataclass
class NvlsGroup:
    """This rank's subgroup of ``ws`` ranks, its symmetric buffers and NCCL helpers."""

    ws: int
    group: Any
    rank: int
    buffers: Dict[str, SymmBuffer] = field(default_factory=dict)

    def buffer(self, name: str = "main") -> SymmBuffer:
        return self.buffers[name]

    # NCCL references -------------------------------------------------------------------------

    def gather(self, t: torch.Tensor) -> torch.Tensor:
        """``[ws, *t.shape]``: every rank's ``t`` in group-rank order (NCCL all-gather, exact)."""
        t = t.contiguous()
        out = torch.empty((self.ws, *t.shape), dtype=t.dtype, device=t.device)
        dist.all_gather_into_tensor(out, t, group=self.group)
        return out

    def reduce_scatter_fp32(self, t: torch.Tensor) -> torch.Tensor:
        """NCCL reduce-scatter of ``t`` (split on dim 0) with fp32 accumulation."""
        t = t.to(torch.float32).contiguous()
        out = torch.empty((t.shape[0] // self.ws, *t.shape[1:]), dtype=t.dtype, device=t.device)
        dist.reduce_scatter_tensor(out, t, op=dist.ReduceOp.SUM, group=self.group)
        return out

    def all_reduce_fp32(self, t: torch.Tensor) -> torch.Tensor:
        """NCCL all-reduce of ``t`` with fp32 accumulation."""
        t = t.to(torch.float32).clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=self.group)
        return t

    def sync(self) -> None:
        """Order every rank's earlier stream work before every rank's later work (NCCL)."""
        flag = torch.zeros(1, device="cuda")
        dist.all_reduce(flag, group=self.group)

    # Inputs ----------------------------------------------------------------------------------

    def rank_generator(self, seed: int) -> torch.Generator:
        """A CUDA generator with a different stream on every rank of the group."""
        return ou.make_generator(rank_seed(seed, self.rank), device="cuda")

    def shared_generator(self, seed: int, device: str = "cpu") -> torch.Generator:
        """A generator with the same stream on every rank of the group."""
        return ou.make_generator(seed, device=device)

    def maybe_sleep(self, shared: torch.Generator) -> None:
        """Delay one rank, picked by ``shared`` (the same draw on every rank), by a random time."""
        sleeper = int(torch.randint(0, self.ws, (), generator=shared))
        cycles = int(torch.randint(0, MAX_SLEEP_CYCLES, (), generator=shared))
        if sleeper == self.rank:
            torch.cuda._sleep(cycles)


class NvlsGroups:
    """Creates, on first use, one subgroup and its symmetric buffers per group size."""

    def __init__(self):
        self._groups: Optional[Dict[int, NvlsGroup]] = None
        self._error: Optional[str] = None

    def get(self, ws: int) -> NvlsGroup:
        """This rank's group of ``ws`` ranks; fails (on target hardware) or skips otherwise."""
        ou.require_sm(ou.HOPPER, ou.BLACKWELL)
        if WORLD_SIZE < ws or WORLD_SIZE % ws:
            unavailable(f"needs a job of a multiple of {ws} ranks, got WORLD_SIZE={WORLD_SIZE}")
        if self._groups is None and self._error is None:
            self._build()
        if self._error is not None:
            unavailable(self._error)
        return self._groups[ws]

    def _build(self) -> None:
        from megatron.core.inference.communication.torch_symm_triton.utils import (
            is_device_nvls_capable,
        )
        from megatron.core.inference.symmetric_memory import SymmetricMemoryBuffer
        from tests.unit_tests.test_utilities import Utils

        Utils.initialize_distributed()
        if not is_device_nvls_capable(torch.device("cuda", torch.cuda.current_device())):
            self._error = "NVLS multicast needs a Hopper or newer GPU"
            return
        groups = {}
        problems = []
        # Every rank creates every subgroup and buffer in the same order (collective calls).
        for ws in GROUP_SIZES:
            group, _ = dist.new_subgroups(group_size=ws)
            entry = NvlsGroup(ws=ws, group=group, rank=dist.get_rank(group))
            for name, size_mb in BUFFER_MB.items():
                buf = SymmetricMemoryBuffer(size_in_mb=size_mb, process_group=group)
                if buf.symm_mem_hdl is None:
                    problems.append(f"ws={ws} {name}: {buf.init_failure_reason}")
                    continue
                if not buf.symm_mem_hdl.multicast_ptr:
                    problems.append(f"ws={ws} {name}: no NVLink multicast (multicast_ptr is 0)")
                entry.buffers[name] = SymmBuffer(raw=buf.symm_buffer, hdl=buf.symm_mem_hdl)
            groups[ws] = entry
        # Agree on success: a rank that cannot allocate must not leave its peers waiting.
        problems_anywhere = torch.tensor([len(problems)], device="cuda")
        dist.all_reduce(problems_anywhere)
        if int(problems_anywhere):
            self._error = "symmetric memory with multicast unavailable: " + (
                "; ".join(problems) or "on another rank"
            )
            return
        self._groups = groups


# ---------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------


def agree(check: Callable[[], Any], what: str) -> Any:
    """Run ``check`` (local work only, no collectives) on every rank; fail everywhere together.

    A rank whose check raised re-raises its own error; every other rank fails with the list of
    failing ranks. So one rank's failure never leaves the others waiting in a later collective.
    """
    error: Optional[BaseException] = None
    result = None
    try:
        result = check()
    except Exception as exc:  # pylint: disable=broad-except
        error = exc
    flags = torch.zeros(dist.get_world_size(), dtype=torch.int32, device="cuda")
    mine = torch.tensor([0 if error is None else 1], dtype=torch.int32, device="cuda")
    dist.all_gather_into_tensor(flags, mine)
    if error is not None:
        raise error
    failed = flags.nonzero().reshape(-1).tolist()
    if failed:
        pytest.fail(f"{what}: failed on job ranks {failed}; see their logs", pytrace=False)
    return result


def sync_world() -> None:
    """Finish every rank's work before the next test touches the symmetric buffers."""
    if dist.is_available() and dist.is_initialized() and torch.cuda.is_available():
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()


def random_bits(
    shape: Sequence[int], dtype: torch.dtype, generator: torch.Generator
) -> torch.Tensor:
    """Every bit pattern of ``dtype`` equally likely (NaN payloads, infinities, -0.0, subnormals).

    For copy kernels, which must move bit patterns, not values.
    """
    nbytes = math.prod(shape) * dtype.itemsize
    raw = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device="cuda", generator=generator)
    return raw.view(dtype).view(tuple(shape))


def poison_value(dtype: torch.dtype) -> float:
    """The scalar ``ou.poison_`` writes for ``dtype``."""
    if dtype.is_floating_point:
        return float("nan")
    return torch.iinfo(dtype).min if torch.iinfo(dtype).min < 0 else torch.iinfo(dtype).max


def offsets_from_counts(counts: Sequence[int]) -> List[int]:
    """Exclusive prefix sum: rank q's first row in the variable-count global layout."""
    out, running = [], 0
    for c in counts:
        out.append(running)
        running += c
    return out


PER_RANK_MAX = [1, 127, 128, 129, 300]
"""per_rank_max_tokens of the variable-count collectives: their grid is min(per_rank_max, 128)
blocks of one token each, with a grid-stride loop past 128 tokens per rank."""

COUNTS = ["uneven", "full", "one-rank", "empty"]

DYADIC = dict(mantissa_bits=8, exponents=(-8, -6))
"""Dyadic inputs for exact sums: |x| < 4 in steps of 2**-8 with up to 8 significant bits, so a
sum of ws <= 8 of them needs rounding to bf16 but is exact in fp32 in any order."""


def token_counts(pattern: str, ws: int, per_rank_max: int, generator: torch.Generator) -> List[int]:
    """Per-rank token counts of a variable-count collective (the same on every rank).

    ``uneven``: random, with rank 0 at ``per_rank_max`` and the last rank at 0; ``full``: all at
    ``per_rank_max``; ``one-rank``: only rank ws // 2 has tokens; ``empty``: all 0.
    """
    if pattern == "full":
        return [per_rank_max] * ws
    if pattern == "empty":
        return [0] * ws
    if pattern == "one-rank":
        counts = [0] * ws
        counts[ws // 2] = per_rank_max
        return counts
    counts = [int(c) for c in torch.randint(0, per_rank_max + 1, (ws,), generator=generator)]
    counts[0], counts[-1] = per_rank_max, 0
    return counts


def concat_rows(gathered: torch.Tensor, counts: Sequence[int]) -> torch.Tensor:
    """Rows ``[0, counts[q])`` of each rank's padded tensor ``gathered[q]``, in rank order."""
    return torch.cat([gathered[q, :c] for q, c in enumerate(counts)], dim=0)
