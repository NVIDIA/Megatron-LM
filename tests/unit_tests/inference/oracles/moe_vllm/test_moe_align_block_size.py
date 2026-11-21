# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the vLLM backend's block alignment, ``_moe_align_block_size_cuda_graphable``
(``megatron/core/inference/moe/vllm_fused_moe.py:415``): the init, count, prefix-sum,
``_fill_expert_block_ids_kernel`` and ``_scatter_token_indices_kernel`` launches together, as
``vllm_fused_moe`` runs them. ``test_fill_expert_block_ids_kernel.py`` and
``test_scatter_token_indices_kernel.py`` test the two kernels alone; this module checks the
tables they build in production, at every ``BLOCK_SIZE_M``, in the persistent pool and under
CUDA-graph replay.

Contract
--------
Written from the grouped GEMM's use of the tables (``_fused_moe_kernel``: block ``b`` of
``BLOCK_SIZE_M`` slots belongs to expert ``expert_ids[b]``; slot ids ``>= max_tokens * topk`` are
padding) and the function's docstring, before reading the kernels.

* **Computes**, for the live pairs ``p = t * topk + j`` (``t < valid_tokens``) routed to local
  expert ``e = routing_map[t, j] - start`` in ``[0, E_local)``, with ``n_e`` their count and
  ``a_e`` it rounded up to ``BLOCK_SIZE_M`` (0 stays 0):

  - ``num_tokens_post_padded = sum_e a_e``, int32 ``[1]``;
  - segments ``[s_e, s_e + a_e)`` with ``s_0 = 0`` and ``s_{e+1} = s_e + a_e``;
  - ``sorted_token_ids[s_e : s_e + n_e]`` holds expert ``e``'s pairs, each once, in an
    unspecified order; every other entry is the sentinel ``max_tokens * topk``;
  - ``expert_ids[b] = e`` for the blocks of segment ``e``; every other entry is ``-1``.
* **Inputs**: ``routing_map`` ``[max_tokens, topk]`` int32 or int64, contiguous; ``valid_tokens`` an
  int32 device scalar ``<= max_tokens``. Rows past ``valid_tokens`` are never read; the map is
  read-only.
* **Outputs**: int32 tables at least as long as the worst case (``max_tokens * topk + E_local *
  (BLOCK_SIZE_M - 1)`` slots), every returned entry as above, so no stale value from an earlier
  call (the persistent pool is shared by every layer) survives. Pool memory past the returned
  views is not written.
* **CUDA graphs**: one capture, replayed with new routing and ``valid_tokens``, gives each
  replay's tables.
* **Tolerance class**: exact (segments compared sorted).

Variants and boundaries (from the launcher and its kernels, after the contract)
-------------------------------------------------------------------------------
* ``BLOCK_SIZE_M`` 16, 32, 64, 128; ``topk`` 1, 2, 6, 8; ``E_local`` 1, 2, 7, 16, 128.
* Per-expert counts ``BM - 1``, ``BM``, ``BM + 1``, 0, 1, ``2 BM - 1``, ``2 BM``, ``2 BM + 1``
  (``block_counts``); a hot expert with 127, 128, 129 and 257 blocks (the fill kernel's 128-block
  store loop); live pairs past the scatter's 256 and the count kernel's 1024 blocks.
* Ranks first, middle and last of EP 1 and 4; ``-1`` and slice-edge ids; int32 and int64 maps.
* The pool: absent (fresh ``torch.empty``) or allocated and poisoned, and reused across calls
  with other sizes and ``BLOCK_SIZE_M``.

Why a direct test
-----------------
The entry oracle sees the tables only through the MoE output: a stale or extra entry past
``num_tokens_post_padded``, a duplicated pair, or a block id past the last segment changes no
output. The old unit tests check set membership, ``expert_ids`` ranges and eager-vs-replay
equality only.

Hardware: H100 (integer kernels; nothing architecture-specific).
"""

import dataclasses
import importlib
from typing import List

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_vllm import align_reference as ar

vfm = importlib.import_module("megatron.core.inference.moe.vllm_fused_moe")

DEVICE = torch.device("cuda")

BLOCK_MS = (16, 32, 64, 128)  # every BLOCK_SIZE_M of _get_default_config


@dataclasses.dataclass(frozen=True)
class Case:
    block_m: int = 16
    topk: int = 2
    num_local: int = 4
    ep: int = 1
    rank: str = "first"
    tokens: int = 100  # live tokens
    pad: int = 3  # rows past valid_tokens
    kind: str = "random"
    dtype: torch.dtype = torch.int64
    pool: bool = False

    @property
    def start(self) -> int:
        index = {"first": 0, "middle": self.ep // 2, "last": self.ep - 1}[self.rank]
        return index * self.num_local

    def id(self) -> str:
        dtype = "i32" if self.dtype == torch.int32 else "i64"
        return (
            f"bm{self.block_m}-k{self.topk}-el{self.num_local}-ep{self.ep}{self.rank}"
            f"-t{self.tokens}-pad{self.pad}-{self.kind}-{dtype}-{'pool' if self.pool else 'nopool'}"
        )


def make_map(case: Case, g: torch.Generator) -> torch.Tensor:
    live = ar.make_routing(
        case.kind,
        case.tokens,
        case.topk,
        case.ep * case.num_local,
        case.start,
        case.num_local,
        g,
        DEVICE,
        block_m=case.block_m,
    )
    return ar.full_routing(live, case.tokens + case.pad, case.start, case.num_local, case.dtype, g)


def poisoned_pool(max_tokens: int, topk: int, num_local: int) -> dict:
    """The persistent pool at the sizes ``allocate_buffers`` picks, NaN / INT_MIN poisoned and
    guarded. Install it with ``monkeypatch.setattr(VllmFusedMoeBuffers, "_buffers", ...)``."""
    saved = vfm.VllmFusedMoeBuffers._buffers
    try:
        vfm.VllmFusedMoeBuffers.allocate_buffers(max_tokens, topk, 16, 16, num_local, device=DEVICE)
        sizes = vfm.VllmFusedMoeBuffers._buffers
        return {name: ou.guarded_empty(b.shape, b.dtype, DEVICE) for name, b in sizes.items()}
    finally:
        vfm.VllmFusedMoeBuffers._buffers = saved


def check_tables(result, routing, valid, start, num_local, block_m, what) -> None:
    """Check the three returned tables, every entry, against the reference."""
    sorted_ids, expert_ids, npp = result
    max_tokens, topk = routing.shape
    pairs = ar.live_pairs_per_expert(routing, valid, start, num_local)
    counts = [p.numel() for p in pairs]
    exclusive, inclusive = ar.aligned_offsets(counts, block_m)
    worst = max_tokens * topk + num_local * (block_m - 1)
    for name, t, need in (
        ("sorted_token_ids", sorted_ids, worst),
        ("expert_ids", expert_ids, -(-worst // block_m)),
        ("num_tokens_post_padded", npp, 1),
    ):
        assert t.dtype == torch.int32, f"{what}: {name} is {t.dtype}, not int32"
        assert t.numel() >= need, f"{what}: {name} has {t.numel()} entries, needs {need}"
    ou.assert_exact(npp, torch.tensor([inclusive[-1]]), what=f"{what}: num_tokens_post_padded")
    ou.assert_exact(
        ar.canonical_segments(sorted_ids, exclusive, counts),
        ar.expected_sorted_ids(pairs, exclusive, sorted_ids.numel(), max_tokens * topk),
        what=f"{what}: sorted_token_ids (segments sorted)",
    )
    ou.assert_exact(
        expert_ids,
        ar.expected_expert_ids(exclusive, inclusive, block_m, expert_ids.numel(), -1),
        what=f"{what}: expert_ids",
    )


def check_pool(pool: dict, used_sorted: int, used_blocks: int, what: str) -> None:
    """Guards intact; pool entries past the longest views requested so far still poisoned."""
    ou.check_guards(**pool)
    sorted_pool, expert_pool = pool["sorted_token_ids"].tensor, pool["expert_ids"].tensor
    ou.assert_poisoned(sorted_pool[used_sorted:], what=f"{what}: pool past sorted_token_ids")
    ou.assert_poisoned(expert_pool[used_blocks:], what=f"{what}: pool past expert_ids")
    for name in ("intermediate1", "intermediate3", "moe_sum_out"):
        ou.assert_poisoned(pool[name].tensor, what=f"{what}: pool {name} is not the align's")


def run_align(case: Case, monkeypatch, seed: int = 0, pool=None):
    """Build the tables for ``case`` and check them; return the lengths of the returned views.

    With ``case.pool`` and no ``pool`` given, a fresh poisoned pool is used and checked; a
    caller that passes its own ``pool`` checks it (``check_pool``) after its sequence of calls.
    """
    g = ou.make_generator(ou.DEFAULT_SEED + seed, device=DEVICE)
    routing = make_map(case, g)
    max_tokens = routing.shape[0]
    fresh_pool = pool is None and case.pool
    if fresh_pool:
        pool = poisoned_pool(max_tokens, case.topk, case.num_local)
    monkeypatch.setattr(
        vfm.VllmFusedMoeBuffers,
        "_buffers",
        None if pool is None else {name: gb.tensor for name, gb in pool.items()},
    )
    routing_buf = ou.guarded_like(routing)
    valid_t = torch.tensor([case.tokens], dtype=torch.int32, device=DEVICE)
    result = vfm._moe_align_block_size_cuda_graphable(
        routing_buf.tensor, case.block_m, case.num_local, case.start, valid_t
    )
    what = case.id()
    check_tables(result, routing, case.tokens, case.start, case.num_local, case.block_m, what)
    ou.check_guards(routing=routing_buf)
    ou.assert_exact(routing_buf.tensor, routing, what=f"{what}: routing_map is read-only")
    if pool is not None:
        assert result[0].data_ptr() == pool["sorted_token_ids"].tensor.data_ptr(), "not pooled"
        assert result[1].data_ptr() == pool["expert_ids"].tensor.data_ptr(), "not pooled"
    if fresh_pool:
        check_pool(pool, result[0].numel(), result[1].numel(), what)
    return result[0].numel(), result[1].numel()


KINDS = ["random", "edges", "one_hot", "ends_empty", "nonlocal", "masked", "block_counts"]

PAIRWISE = [
    Case(**c)
    for c in ou.pairwise_cases(
        {
            "block_m": list(BLOCK_MS),
            "kind": KINDS,
            "topk": [1, 2, 6, 8],
            "num_local": [1, 2, 7, 16, 128],
            "tokens": [0, 1, 43, 200, 700],
            "pad": [0, 1, 129],
            "ep": [1, 4],
            "rank": ["first", "middle", "last"],
            "dtype": [torch.int64, torch.int32],
            "pool": [False, True],
        }
    )
]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", [pytest.param(c, id=c.id()) for c in PAIRWISE])
def test_tables_match_the_reference(case, monkeypatch):
    """Segments, padding sentinels, block ids and the padded count, exactly."""
    run_align(case, monkeypatch)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("fill", ["full", "one-pair-in-last-block"])
@pytest.mark.parametrize("block_m", [16, 128])
@pytest.mark.parametrize("blocks", [127, 128, 129, 257])
def test_hot_expert_around_the_fill_loop(blocks, block_m, fill, monkeypatch):
    """One local expert holds every live pair: 127 to 257 blocks, around the fill's 128."""
    pairs = blocks * block_m if fill == "full" else (blocks - 1) * block_m + 1
    case = Case(
        block_m=block_m, topk=1, num_local=4, ep=2, rank="last", tokens=pairs, kind="one_hot"
    )
    run_align(case, monkeypatch)


@ou.requires_sm(ou.HOPPER)
def test_pool_reused_across_calls(monkeypatch):
    """One poisoned pool, then calls with other sizes, BLOCK_SIZE_M and routing: no stale entry."""
    first = Case(block_m=16, topk=8, num_local=16, ep=4, rank="middle", tokens=500, pad=12)
    pool = poisoned_pool(first.tokens + first.pad, first.topk, first.num_local)
    sequence = [
        first,
        dataclasses.replace(first, block_m=128, tokens=37, pad=475, kind="block_counts"),
        dataclasses.replace(first, block_m=32, tokens=0, pad=512),
        dataclasses.replace(first, block_m=64, tokens=512, pad=0, kind="ends_empty"),
        dataclasses.replace(first, block_m=16, tokens=3, pad=509, kind="one_hot"),
    ]
    used_sorted = used_blocks = 0
    for i, case in enumerate(sequence):
        lengths = run_align(case, monkeypatch, seed=i, pool=pool)
        used_sorted, used_blocks = max(used_sorted, lengths[0]), max(used_blocks, lengths[1])
        check_pool(pool, used_sorted, used_blocks, case.id())


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("block_m", BLOCK_MS)
def test_cuda_graph_replay(block_m, monkeypatch):
    """One capture replayed with new routing and valid_tokens, each replay's tables exact."""
    max_tokens, topk, num_local, ep = 200, 6, 8, 4
    start = num_local  # rank 1 of 4
    pool = poisoned_pool(max_tokens, topk, num_local)
    monkeypatch.setattr(
        vfm.VllmFusedMoeBuffers, "_buffers", {name: gb.tensor for name, gb in pool.items()}
    )
    routing = torch.full((max_tokens, topk), -1, dtype=torch.int64, device=DEVICE)
    valid_t = torch.tensor([max_tokens], dtype=torch.int32, device=DEVICE)

    def call():
        return vfm._moe_align_block_size_cuda_graphable(routing, block_m, num_local, start, valid_t)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        call()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = call()

    kinds = ["random", "block_counts", "one_hot", "nonlocal", "ends_empty"]
    for i, valid in enumerate([max_tokens, 0, 1, block_m - 1, block_m, block_m + 1, 171, 43]):
        valid = min(valid, max_tokens)
        g = ou.make_generator(ou.DEFAULT_SEED + 1 + i, device=DEVICE)
        live = ar.make_routing(
            kinds[i % len(kinds)], valid, topk, ep * num_local, start, num_local, g, DEVICE, block_m
        )
        routing.copy_(ar.full_routing(live, max_tokens, start, num_local, torch.int64, g))
        valid_t.fill_(valid)
        graph.replay()
        torch.cuda.synchronize()
        what = f"replay {i} (valid {valid})"
        check_tables(result, routing, valid, start, num_local, block_m, what)
        check_pool(pool, result[0].numel(), result[1].numel(), what)


PRODUCTION = {
    # Nemotron 3 Nano, EP 8 prefill: 8192 tokens, top-6, 16 local experts (BLOCK_SIZE_M 128).
    "nemotron-nano-ep8-prefill": Case(
        block_m=128, topk=6, num_local=16, ep=8, rank="middle", tokens=8192, pad=0, pool=True
    ),
    # DeepSeek-V3 at EP 1: 256 experts, top-8, 4096 tokens (BLOCK_SIZE_M 128).
    "deepseek-v3-ep1": Case(block_m=128, topk=8, num_local=256, tokens=4096, pad=100),
    # Qwen3-30B-A3B, EP 8 decode: 512 gathered tokens, top-8, 16 local experts (BLOCK_SIZE_M 64).
    "qwen3-30b-a3b-ep8-decode": Case(
        block_m=64, topk=8, num_local=16, ep=8, rank="last", tokens=500, pad=12, pool=True
    ),
}


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("name", list(PRODUCTION))
def test_production_sizes(name, monkeypatch):
    """Deployed top-k, expert counts and batch sizes."""
    run_align(PRODUCTION[name], monkeypatch)


@pytest.mark.sanitizer
@ou.requires_sm(ou.HOPPER)
def test_sanitizer(monkeypatch):
    """A small case for compute-sanitizer memcheck (also run normally)."""
    case = Case(block_m=32, topk=3, num_local=7, ep=4, rank="last", tokens=90, pad=4, pool=True)
    run_align(case, monkeypatch)


def all_cases() -> List[Case]:
    """Every case of the pairwise and production tests (for scratch checks outside pytest)."""
    return list(PAIRWISE) + list(PRODUCTION.values())
