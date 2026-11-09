# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_scatter_token_indices_kernel``
(``megatron/core/inference/moe/vllm_fused_moe.py:361``), launched directly as its launcher
(``_moe_align_block_size_cuda_graphable``) launches it.

Contract
--------
Written from the alignment's contract (each local expert's live pairs occupy the first slots
of its segment of the padded table, in any order) and the launcher, before reading the body.

* **Computes**: for each live pair ``p = t * topk + j`` (``t < valid_tokens``) whose expert
  ``routing_map[t, j]`` lies in ``[start, start + E_local)``, with ``e = routing_map[t, j] -
  start``, it claims the next position of expert ``e`` from ``counters[e]`` and writes ``p``
  there. On entry ``counters`` holds each expert's segment start (the launcher passes the
  exclusive aligned offsets); afterwards it holds ``start_e + n_e``, ``n_e`` the expert's live
  pair count. So ``sorted_token_ids[start_e : start_e + n_e]`` holds exactly expert ``e``'s
  pairs, each once, in an unspecified order.
* **Inputs**: ``routing_map`` ``[max_tokens, topk]`` int32 or int64, contiguous, global ids or
  ``-1``; ``valid_tokens`` an int32 device scalar ``<= max_tokens``; ``max_pairs = max_tokens *
  topk``; ``topk`` and ``E_local`` (constexprs) and ``start``.
* **Outputs**: no other entry of ``sorted_token_ids`` is written; rows ``>= valid_tokens`` of the
  map are never read (their pairs are never written); the map is read-only.
* **Degenerate**: ``valid_tokens = 0`` writes nothing and leaves the counters; nothing local;
  every slot ``-1``; every slot on one expert; ``max_tokens = 0`` (an empty grid).
* **``valid_tokens > max_tokens``** (finding F1, fix #8563): only the map's ``max_tokens`` rows
  exist, so only their pairs may be scattered. ``main`` reads past the map; #8563 clamps the
  pair count to ``max_pairs``.
* **Tolerance class**: exact. The segment order is unspecified, so each segment is compared
  sorted (``align_reference.canonical_segments``) and every other entry bit for bit.

Variants and boundaries (from the kernel, after the contract)
-------------------------------------------------------------
* ``BLOCK_SIZE = 256`` pairs per program (``:485``), grid ``ceil(max_pairs / 256)``; programs
  past ``valid_pairs`` return early. Live pair counts 1, 255, 256, 257, 511, 512, 513 and 1500,
  with 0, 1 or 300 rows past ``valid_tokens`` (the early-exit programs).
* Constexprs ``topk`` 1, 2, 3, 6, 8, 10 and ``E_local`` 1, 2, 7, 16, 128.
* Locality: ranks first, middle and last of EP 1 and 4; ids at and next to the slice edges.
* int32 and int64 maps. Counters aligned to every ``BLOCK_SIZE_M`` (16 to 128).
* int32: pair indices are below ``max_pairs``, which the int32 table bounds below 2**31; every
  offset the kernel forms is then below 2**31 (pair counts at 2**31 are O1-N1, #8962). No
  ``large`` case.

Why a direct test
-----------------
The entry oracle sees the table only through the MoE output, and the old unit tests check set
membership only. A pair written twice (a second copy in a padding slot of its own segment) or
the counters' final values change no output; nor does a pair placed in another expert's
segment when that expert has the same weights. Here the table itself is checked.

Hardware: H100 (integer atomics; nothing architecture-specific).
"""

import dataclasses
import importlib
from typing import List, Optional

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_vllm import align_reference as ar

vfm = importlib.import_module("megatron.core.inference.moe.vllm_fused_moe")

DEVICE = torch.device("cuda")

SCATTER_BLOCK = 256  # pairs per program: the launcher's SCATTER_BLOCK (vllm_fused_moe.py:485)


@dataclasses.dataclass(frozen=True)
class Case:
    pairs: int = 300  # target live pair count; valid_tokens = ceil(pairs / topk)
    topk: int = 2
    num_local: int = 4
    ep: int = 1
    rank: str = "first"  # "first" | "middle" | "last"
    pad: int = 3  # rows past valid_tokens
    kind: str = "random"
    dtype: torch.dtype = torch.int64
    block_m: int = 16  # the counters' segment alignment

    @property
    def valid(self) -> int:
        return -(-self.pairs // self.topk)

    @property
    def start(self) -> int:
        index = {"first": 0, "middle": self.ep // 2, "last": self.ep - 1}[self.rank]
        return index * self.num_local

    def id(self) -> str:
        dtype = "i32" if self.dtype == torch.int32 else "i64"
        return (
            f"pairs{self.pairs}-k{self.topk}-el{self.num_local}-ep{self.ep}{self.rank}"
            f"-pad{self.pad}-{self.kind}-{dtype}-bm{self.block_m}"
        )


def make_map(case: Case, g: torch.Generator) -> torch.Tensor:
    live = ar.make_routing(
        case.kind,
        case.valid,
        case.topk,
        case.ep * case.num_local,
        case.start,
        case.num_local,
        g,
        DEVICE,
        block_m=case.block_m,
    )
    return ar.full_routing(live, case.valid + case.pad, case.start, case.num_local, case.dtype, g)


def run_scatter(
    routing: torch.Tensor,
    valid: int,
    start: int,
    num_local: int,
    block_m: int,
    what: str,
    rows: Optional[int] = None,
) -> None:
    """Launch with the launcher's grid and buffer sizes and check the table and the counters.

    ``routing`` is the map (contiguous). ``rows`` (default: all of them) is how many rows the
    launcher is told about; rows past it must not be read even when ``valid`` exceeds it.
    """
    rows = routing.shape[0] if rows is None else rows
    topk = routing.shape[1]
    max_pairs = rows * topk
    live_rows = min(valid, rows)
    pairs = ar.live_pairs_per_expert(routing[:rows], live_rows, start, num_local)
    counts = [p.numel() for p in pairs]
    exclusive, _ = ar.aligned_offsets(counts, block_m)
    max_sorted = max_pairs + block_m * (num_local + 1)  # the launcher's table length (:443)

    routing_buf = ou.guarded_like(routing)
    counters = ou.guarded_like(torch.tensor(exclusive, dtype=torch.int32, device=DEVICE))
    sorted_ids = ou.guarded_empty(
        (max_sorted,), torch.int32, DEVICE, guard_bytes=4 * SCATTER_BLOCK * 4
    )
    valid_t = torch.tensor([valid], dtype=torch.int32, device=DEVICE)
    map_before = routing_buf.tensor.clone()

    grid = -(-max_pairs // SCATTER_BLOCK)
    vfm._scatter_token_indices_kernel[(grid,)](
        routing_buf.tensor,
        sorted_ids.tensor,
        counters.tensor,
        valid_t,
        topk,
        start,
        num_local,
        max_pairs,
        BLOCK_SIZE=SCATTER_BLOCK,
    )

    got = ar.canonical_segments(sorted_ids.tensor, exclusive, counts)
    expected = ar.expected_sorted_ids(pairs, exclusive, max_sorted, ar.INT32_MIN)
    ou.assert_exact(got, expected, what=f"{what}: sorted_token_ids (segments sorted)")
    ou.assert_exact(
        counters.tensor,
        torch.tensor([s + n for s, n in zip(exclusive, counts)]),
        what=f"{what}: counters end at segment start + live pairs",
    )
    ou.check_guards(sorted_ids=sorted_ids, counters=counters, routing=routing_buf)
    ou.assert_exact(routing_buf.tensor, map_before, what=f"{what}: routing_map is read-only")
    ou.assert_exact(valid_t, torch.tensor([valid]), what=f"{what}: valid_tokens is read-only")


def run_case(case: Case, seed: int = 0) -> None:
    g = ou.make_generator(ou.DEFAULT_SEED + seed, device=DEVICE)
    routing = make_map(case, g)
    run_scatter(routing, case.valid, case.start, case.num_local, case.block_m, case.id())


KINDS = ["random", "edges", "one_hot", "ends_empty", "nonlocal", "masked", "block_counts"]

PAIRWISE = [
    Case(**c)
    for c in ou.pairwise_cases(
        {
            "pairs": [1, 255, 256, 257, 511, 512, 513, 1500],
            "topk": [1, 2, 3, 6, 8, 10],
            "num_local": [1, 2, 7, 16, 128],
            "kind": KINDS,
            "pad": [0, 1, 300],
            "ep": [1, 4],
            "rank": ["first", "middle", "last"],
            "dtype": [torch.int64, torch.int32],
            "block_m": [16, 32, 64, 128],
        }
    )
]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", [pytest.param(c, id=c.id()) for c in PAIRWISE])
def test_scatters_each_live_local_pair_once(case):
    """Each expert's segment holds exactly its live pairs; nothing else is written."""
    run_case(case)


DEGENERATE = {
    "valid0": Case(pairs=0, pad=40),
    "valid0-one-row": Case(pairs=0, pad=1, topk=8),
    "max_tokens0": Case(pairs=0, pad=0, topk=8, num_local=16),
    "single-pair": Case(pairs=1, topk=1, num_local=1, pad=0),
    "one-expert-all-local": Case(pairs=1000, topk=1, num_local=1, kind="one_hot"),
    "nothing-local": Case(pairs=600, ep=4, rank="middle", kind="nonlocal"),
    "all-masked": Case(pairs=600, kind="masked", dtype=torch.int32),
}


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("name", list(DEGENERATE))
def test_degenerate(name):
    """No live rows, no rows at all, one pair, everything on one expert, nothing local."""
    run_case(DEGENERATE[name])


PRODUCTION = {
    # Nemotron 3 Nano, EP 8 prefill: 8192 tokens gathered per rank, top-6, 16 local experts.
    "nemotron-nano-ep8-prefill": Case(
        pairs=8192 * 6, topk=6, num_local=16, ep=8, rank="middle", pad=0, block_m=128
    ),
    # DeepSeek-V3 at EP 1: 256 experts, top-8, 4096 tokens, a padded graph batch.
    "deepseek-v3-ep1": Case(pairs=4096 * 8, topk=8, num_local=128, pad=512, block_m=128),
    # Qwen3-30B-A3B, EP 8 decode: 512 gathered tokens, top-8, 16 local experts.
    "qwen3-30b-a3b-ep8-decode": Case(
        pairs=512 * 8, topk=8, num_local=16, ep=8, rank="last", pad=64, block_m=64
    ),
}


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("name", list(PRODUCTION))
def test_production_sizes(name):
    """Deployed top-k, expert counts and batch sizes."""
    run_case(PRODUCTION[name])


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("extra", [1, 2])
@pytest.mark.parametrize("dtype", [torch.int64, torch.int32])
def test_valid_above_rows_scatters_only_the_map_rows(dtype, extra):
    """``valid_tokens = max_tokens + extra``: only the map's rows are scattered (finding F1).

    The map is the head of a longer buffer whose next rows route to local experts, so reading
    past ``max_tokens`` claims extra slots and writes pair ids of rows that do not exist. Fails
    on ``main`` (the kernel bounds the pairs by ``valid_tokens`` alone); passes with #8563, which
    clamps the pair count to ``max_pairs``. No out-of-bounds access: the longer buffer holds the
    rows read.
    """
    rows, topk, num_local, start = 40, 2, 4, 4
    g = ou.make_generator(device=DEVICE)
    live = ar.make_routing("random", rows, topk, 12, start, num_local, g, DEVICE)
    longer = ar.full_routing(live, rows + extra, start, num_local, dtype, g)
    run_scatter(longer, rows + extra, start, num_local, 16, f"valid = rows + {extra}", rows=rows)


@pytest.mark.sanitizer
@ou.requires_sm(ou.HOPPER)
def test_sanitizer():
    """A small case for compute-sanitizer memcheck (also run normally)."""
    run_case(Case(pairs=257, topk=3, num_local=7, ep=4, rank="middle", pad=5, kind="edges"))


def all_cases() -> List[Case]:
    """Every case of the parametrized tests (for scratch checks outside pytest)."""
    return list(PAIRWISE) + list(DEGENERATE.values()) + list(PRODUCTION.values())
