# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""References and inputs shared by the MoE routing / permute oracle tests.

Every reference here is written from the contract of the inference MoE dispatch path
(``megatron/core/inference/moe/permute.py`` launchers and their callers), in plain Python or
CPU integer PyTorch, and shares no index math with the Triton kernels:

* ``local_counts``: per local expert, the number of ``(token, slot)`` pairs among the first
  ``min(valid, rows)`` routing-map rows whose global expert id falls in the local slice
  (``torch.bincount`` on CPU).
* ``aligned_offsets``: each count rounded up to the alignment (zero stays zero), then the
  exclusive and inclusive running sums (a Python loop).
* ``expected_expert_rows``: per local expert, the ``(token, prob, hidden row)`` records the
  permutation must place in that expert's block (Python grouping).

Inputs: routing maps are built as the head of a larger *canary* buffer whose tail rows route
every slot to a local expert, so a kernel that reads past the routing map's rows counts them
and fails the exact comparison.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou

# Kernel constants (megatron/core/inference/moe/permute.py), named here so the boundary sizes
# below are derived from the kernel's structure.
COUNT_BLOCK = 1024
"""Pairs per program of both count kernels (``BLOCK = 1024`` in compute_local_tokens_per_expert)."""
INIT_BLOCK = 1024
"""Entries per program of _init_permutation_map_kernel (``BLOCK_SIZE = 1024``)."""
MAX_GRID = 512
"""Cap on the grid of the permute / zero-padding / zero-output / unpermute kernels
(``NUM_BLOCKS = min(n, 512)``)."""
MAX_BLOCK_H = 1024
"""Cap on the hidden-dim tile (``BLOCK_H = min(next_pow2(H), 1024)``)."""

INT32_MAX = torch.iinfo(torch.int32).max


def device() -> str:
    """The device the oracle runs on: CUDA, or CPU under the Triton interpreter.

    Without either, skip (or fail when the run targets H100, so a misplaced runner cannot pass
    by skipping).
    """
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("TRITON_INTERPRET") == "1":
        return "cpu"
    ou.require_sm(ou.HOPPER)
    raise AssertionError("unreachable: require_sm skips or fails without a CUDA device")


def require_cuda() -> str:
    """For CUDA-graph tests: a CUDA device, or skip/fail as ``device()`` does without one."""
    if not torch.cuda.is_available():
        ou.require_sm(ou.HOPPER)
    return "cuda"


def valid_tensor(n: int, dev: str) -> torch.Tensor:
    """The ``[1]`` int32 device scalar the dispatcher hands the kernels as ``valid_tokens``."""
    return torch.tensor([n], dtype=torch.int32, device=dev)


def rows_around_pairs(block: int, topk: int, multiples: Sequence[int] = (1, 2)) -> List[int]:
    """Row counts whose pair count ``rows * topk`` lies just below, at and just above each
    multiple of ``block`` (the count kernels tile the flattened ``[rows, topk]`` map by pairs)."""
    rows = set()
    for m in multiples:
        b = m * block
        rows.add((b - 1) // topk)  # rows * topk <= b - 1
        rows.add(-(-b // topk))  # first rows with rows * topk >= b
        rows.add(-(-(b + 1) // topk))  # first rows with rows * topk >= b + 1
    return sorted(rows)


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


def with_canary_tail(
    head: torch.Tensor, tail_rows: int, tail_expert: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(view, buffer)``: ``head`` copied into the first rows of a larger buffer.

    The ``tail_rows`` rows after it route every slot to ``tail_expert`` (pick a local expert),
    so any read past the view's rows lands on a local expert and changes a count. The view is
    contiguous, like the router's output and the AGV buffer slice the kernels receive.
    """
    rows, topk = head.shape
    buf = torch.full((rows + tail_rows, topk), tail_expert, dtype=head.dtype, device=head.device)
    buf[:rows] = head
    return buf[:rows], buf


def boundary_expert_ids(
    rows: int,
    topk: int,
    num_experts: int,
    start: int,
    num_local: int,
    *,
    dtype: torch.dtype = torch.int64,
    generator: torch.Generator,
    dev: str,
) -> torch.Tensor:
    """A ``[rows, topk]`` map that stresses the local-slice test.

    Half the entries are uniform over the global experts; the rest are drawn from the ids at
    the slice edges (``start - 1``, ``start``, ``start + num_local - 1``, ``start + num_local``)
    and ``-1`` (masked padding). Ids outside ``[-1, num_experts)`` are dropped (the router never
    produces them). Duplicates within a token are allowed: the count contract counts pairs.
    """
    edges = [-1, start - 1, start, start + num_local - 1, start + num_local]
    edges = torch.tensor(sorted({e for e in edges if -1 <= e < num_experts}), dtype=torch.int64)
    uniform = torch.randint(0, num_experts, (rows, topk), generator=generator)
    pick = torch.randint(0, edges.numel(), (rows, topk), generator=generator)
    use_edge = torch.rand(rows, topk, generator=generator) < 0.5
    ids = torch.where(use_edge, edges[pick], uniform)
    return ids.to(dtype=dtype, device=dev)


def topk_distinct_ids(
    rows: int,
    topk: int,
    num_experts: int,
    *,
    masked_rows: int = 0,
    dtype: torch.dtype = torch.int64,
    generator: torch.Generator,
    dev: str,
) -> torch.Tensor:
    """A router-shaped ``[rows, topk]`` map: ``topk`` distinct experts per token.

    The last ``masked_rows`` rows are all ``-1``, as the routing-padding mask leaves CUDA-graph
    padding rows.
    """
    ids = torch.rand(rows, num_experts, generator=generator).argsort(dim=1)[:, :topk]
    if masked_rows:
        ids[rows - masked_rows :] = -1
    return ids.to(dtype=dtype, device=dev)


# ---------------------------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------------------------


def local_counts(routing_map: torch.Tensor, valid: int, start: int, num_local: int) -> torch.Tensor:
    """int64 ``[num_local]``: pairs of the first ``min(valid, rows)`` rows on each local expert."""
    live = routing_map[: max(0, min(valid, routing_map.shape[0]))]
    local = live.reshape(-1).to(device="cpu", dtype=torch.int64) - start
    local = local[(local >= 0) & (local < num_local)]
    return torch.bincount(local, minlength=num_local)


def aligned_offsets(counts: Sequence[int], alignment: int) -> Tuple[List[int], List[int]]:
    """(exclusive, inclusive) running sums of the counts rounded up to ``alignment``."""
    exclusive, inclusive, total = [], [], 0
    for c in (int(c) for c in counts):
        padded = 0 if c == 0 else ((c + alignment - 1) // alignment) * alignment
        exclusive.append(total)
        total += padded
        inclusive.append(total)
    return exclusive, inclusive


def expected_expert_rows(
    routing_map: torch.Tensor, valid: int, start: int, num_local: int
) -> Dict[int, List[Tuple[int, int]]]:
    """Per local expert: the ``(token, slot)`` pairs of live rows routed to it."""
    rows = routing_map.shape[0]
    ids = routing_map[: max(0, min(valid, rows))].to(device="cpu", dtype=torch.int64).tolist()
    out: Dict[int, List[Tuple[int, int]]] = {e: [] for e in range(num_local)}
    for t, row in enumerate(ids):
        for k, gid in enumerate(row):
            if start <= gid < start + num_local:
                out[gid - start].append((t, k))
    return out


def bits_i64(t: torch.Tensor) -> torch.Tensor:
    """Bit patterns of a float tensor widened to int64 (for row-multiset keys)."""
    itemsize = t.dtype.itemsize
    width = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[itemsize]
    return t.view(width).to(torch.int64)


def record_rows(tokens: torch.Tensor, probs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """int64 rows ``[token, prob bits, hidden bits...]`` identifying a permuted row exactly."""
    return torch.cat(
        [
            tokens.to(torch.int64).reshape(-1, 1),
            bits_i64(probs.contiguous()).reshape(-1, 1),
            bits_i64(hidden.contiguous()).reshape(hidden.shape[0], -1),
        ],
        dim=1,
    ).cpu()


def counting_slices(num_local: int, ep: int = 4) -> List[Tuple[str, int]]:
    """Local slices of a ``num_local * ep``-expert model: first, second and last rank."""
    return [("first", 0), ("second", num_local), ("last", (ep - 1) * num_local)]


def poisoned_outputs():
    """Context manager: ``torch.empty`` fills NaN (floats) / ``INT32_MAX`` (int32) meanwhile.

    The permute and unpermute launchers allocate their outputs with ``torch.empty``; with
    PyTorch's ``fill_uninitialized_memory`` those allocations start poisoned, so an element the
    kernels never write is recognisable afterwards.
    """
    return _PoisonedOutputs()


class _PoisonedOutputs:
    def __enter__(self):
        self._mode = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        self._fill = torch.utils.deterministic.fill_uninitialized_memory
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        return self

    def __exit__(self, *exc):
        torch.use_deterministic_algorithms(self._mode[0], warn_only=self._mode[1])
        torch.utils.deterministic.fill_uninitialized_memory = self._fill
        return False


def assert_untouched(t: torch.Tensor, *, what: str, poison: Optional[float] = None) -> None:
    """Assert a ``poisoned_outputs`` allocation was never written: NaN or ``INT32_MAX``."""
    if t.numel() == 0:
        return
    if t.is_floating_point():
        bad = ~torch.isnan(t)
    else:
        bad = t.to(torch.int64) != (INT32_MAX if poison is None else int(poison))
    count = int(bad.sum())
    if count:
        first = bad.reshape(-1).nonzero()[:8].reshape(-1).tolist()
        raise AssertionError(
            f"{what}: {count} of {t.numel()} elements were written (must be untouched);"
            f" first flat indices {first}"
        )
