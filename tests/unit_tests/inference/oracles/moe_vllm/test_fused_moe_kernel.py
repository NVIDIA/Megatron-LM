# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_fused_moe_kernel`` (``megatron/core/inference/moe/vllm_fused_moe.py:106``), the
vLLM backend's grouped GEMM, launched directly through its launcher ``_invoke_fused_moe_kernel``
(``:508``) on hand-built indirection tables.

Contract
--------
Written from the alignment's contract (the tables), the two call sites in ``vllm_fused_moe``
(FC1 and FC2) and the launcher's docstring, before reading the kernel body.

* **Computes**, for every slot ``s < num_tokens_post_padded`` whose block ``b = s // BLOCK_SIZE_M``
  has an expert ``e = expert_ids[b] != -1`` and whose pair ``p = sorted_token_ids[s]`` is below
  ``M * top_k`` (``M = A.size(0)``; larger ids are padding)::

      C[p, n] = bf16( w_p * act( sum_k A[p // top_k, k] * B[e, n, k] ) )     for n < N = B.size(1)

  The sum is accumulated in fp32 from exact products of the bf16 operands. ``act`` runs in fp32:
  the identity, ``relu(h)**2`` with ``FUSE_SQUARED_RELU``, or ``(s * tanh(relu(h) / s))**2`` with a
  clamp scale ``s`` as well. ``w_p = topk_weights[p]`` (fp32) with ``MUL_ROUTED_WEIGHT``, else 1,
  applied after the activation. There is one rounding to bf16.
* **Inputs**: ``A`` ``[M, K]``, ``B`` ``[E, N, K]`` and ``C`` ``[M * top_k, N]`` bf16 with any
  strides; ``topk_weights`` fp32 ``[M * top_k]``; ``sorted_token_ids`` int32 (at least ``npp``
  long); ``expert_ids`` int32 (at least ``npp / BLOCK_SIZE_M``); ``num_tokens_post_padded``
  (``npp``) int32 ``[1]``, a multiple of ``BLOCK_SIZE_M``; a launch config and ``grid_size >= 1``.
* **Outputs**: only the rows ``C[p, :N]`` of pairs in live slots are written. Every other element
  of ``C`` (pairs absent from the table, pairs in blocks of expert ``-1``, columns past ``N`` of a
  wider buffer) keeps its value. ``A``, ``B``, the weights and the tables are read-only; rows of
  ``A``, experts of ``B`` and weights that no live slot uses are never read.
* **Independence**: the result does not depend on ``grid_size``, ``GROUP_SIZE_M``, the order of
  the blocks, where a pair or a padding slot sits in its block, ``BLOCK_SIZE_M``,
  ``BLOCK_SIZE_N``, ``num_warps`` or ``num_stages``. Only ``BLOCK_SIZE_K`` groups the fp32 sum, so
  at a fixed ``BLOCK_SIZE_K`` the bits are fixed: batch-invariant mode pins only that
  (``vllm_fused_moe.py:764-773``), and the atomic scatter places pairs at random positions.
* **Degenerate**: ``npp = 0``; every block of expert ``-1``; blocks of padding only; one pair;
  ``K = 1``; ``N = 1``.
* **Tolerance class**: exact by construction on small integers (``A``, ``B`` in {-1, 0, 1},
  ``w = j / 16``: every sum, square and weight product is exact in fp32, checked per case), so
  ``C`` equals the spec bit for bit. Gaussian data: floating point, per row at most 2x the error
  of a trusted torch implementation (fp32 matmul with TF32 off on the same device, the same
  activation and weight order) plus one ulp. Bitwise for the independence properties.

Variants and boundaries (from the kernel and launcher, after the contract)
--------------------------------------------------------------------------
* Constexprs ``MUL_ROUTED_WEIGHT`` x ``FUSE_SQUARED_RELU`` x ``CLAMP_ACTIVATION``: all six
  combinations. The call sites compile four (FC1: plain, squared ReLU, clamp; FC2: weighted or
  plain); ``MUL_ROUTED_WEIGHT`` with ``FUSE_SQUARED_RELU`` is a valid launcher input no call site
  uses. ``top_k`` 1 (FC2), 2, 3, 6 and 8 (FC1), so weights are indexed by pair, not token.
* Every launch config ``_get_default_config`` returns, with batch-invariant mode's
  ``BLOCK_SIZE_K = 64``: BM 16/32/64/128, BN 64/128, BK 64/128, GROUP_SIZE_M 1/16, 4/8 warps,
  3/4 stages (10 configs).
* Grid 1, 7, tiles - 1, tiles and tiles + 5: CTAs that stride over many tiles, one tile short
  of a full grid, idle CTAs. GROUP_SIZE_M 16 over 33 M blocks (groups of 16, 16 and 1).
* ``K`` and ``N`` at 1, 63/64/65, 127/128/129 and 300 (the BK and BN edges, partial tiles, the
  ``% N`` wrap of the B columns), and 1000/1025 for gaussian data.
* Strides: ``A`` with a row stride past ``K`` and column-major; ``B`` stored ``[E, K, N]``
  (transposed) and as a window of a larger buffer (expert stride past ``N * K``, NaN outside the
  window, finite weights just before expert 0); ``C`` with a row stride past ``N`` and
  column-major.
* Tables: the align's layout (experts ascending, padding at block tails) and shuffled block and
  slot orders, padding slots anywhere in a block, blocks of expert ``-1`` holding decoy pair ids
  of pairs that must not be computed, and blocks of padding only.
* Per-expert pair counts ``BM - 1``, ``BM``, ``BM + 1``, 0, 1 and ``2 BM + 1``.
* Production: Nemotron 3 Nano FC1 and FC2 (2688 / 1856, top-6), DeepSeek-V2-Lite SwiGLU FC1
  (2048 -> 2 x 1408), Mixtral 8x7B FC2 (14336 -> 4096).
* int32 limits (``large``): an expert's ``B`` offset ``e * stride_be`` past 2**31. The ``A`` and
  ``C`` row offsets past 2**31 are ``test_vllm_fused_moe.py::test_int32_offsets``.

Why a direct test
-----------------
Through ``vllm_fused_moe`` the tables come from the align kernels: experts ascending, padding at
block tails, no ``-1`` block below ``num_tokens_post_padded``. FC1 never weights, FC2 always has
``top_k = 1``, ``B`` and ``C`` are contiguous, the grid follows the hint, and no two configs ever
run on the same table, so the bit-for-bit independence of the tile shape and of a pair's place in
its tile (which batch invariance and determinism rest on) is seen only through whole batches.

Hardware: H100. The exact sweep and the bitwise independence test also run on GB200
(``launch_on_gb200``), where ``tl.dot`` lowers to other tensor-core instructions (``tcgen05``)
and the vLLM backend is the default too.
"""

import contextlib
import dataclasses
import importlib
import math
from typing import Dict, List, Optional

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou

vfm = importlib.import_module("megatron.core.inference.moe.vllm_fused_moe")

DEVICE = torch.device("cuda")


def _config_id(c: Dict[str, int]) -> str:
    return (
        f"bm{c['BLOCK_SIZE_M']}-bn{c['BLOCK_SIZE_N']}-bk{c['BLOCK_SIZE_K']}-g{c['GROUP_SIZE_M']}"
        f"-w{c['num_warps']}-s{c['num_stages']}"
    )


def _launch_configs() -> Dict[str, Dict[str, int]]:
    """Every config ``vllm_fused_moe`` can launch: the table at each threshold, both groupings,
    and batch-invariant mode's BLOCK_SIZE_K pin (vllm_fused_moe.py:763-773)."""
    configs = {}
    for m in (1, 33, 65, 97, 129, 513):
        for num_local in (1, 64):
            for pin_k in (False, True):
                c = dict(vfm._get_default_config(m, num_local, 2))
                if pin_k:
                    c["BLOCK_SIZE_K"] = 64
                configs[_config_id(c)] = c
    return dict(sorted(configs.items()))


CONFIGS = _launch_configs()

# (MUL_ROUTED_WEIGHT, FUSE_SQUARED_RELU, clamp scale): every compiled combination.
VARIANTS = {
    "plain": (False, False, None),  # FC1 for SwiGLU / batch-invariant mode, FC2 in BI mode
    "relu2": (False, True, None),  # FC1, squared ReLU
    "weighted": (True, False, None),  # FC2, default mode
    "weighted-relu2": (True, True, None),  # no call site
}
CLAMP_VARIANTS = {
    "clamp16": (False, True, 16.0),  # FC1, Nemotron's activation_func_tanh_clamp_scale
    "clamp0.5": (False, True, 0.5),  # mostly saturated
    "weighted-clamp16": (True, True, 16.0),  # no call site
}


@dataclasses.dataclass(frozen=True)
class Case:
    config: str
    variant: str = "plain"
    top_k: int = 2
    k: int = 128
    n: int = 128
    load: str = "light"  # per-expert pair counts, see counts_for
    table: str = "packed"  # see build_table
    strides: str = "contiguous"  # see place_operands
    grid: str = "tiles"  # see grid_size
    data: str = "int"  # "int": exact spec; "gauss": fp64 reference + trusted bound

    def id(self) -> str:
        return (
            f"{self.config}-{self.variant}-k{self.top_k}-K{self.k}-N{self.n}-{self.load}"
            f"-{self.table}-{self.strides}-grid{self.grid}-{self.data}"
        )


def counts_for(load: str, bm: int) -> List[int]:
    """Live pairs per expert."""
    if load == "light":  # around one and two M blocks, an empty expert, a single pair
        return [bm - 1, bm, bm + 1, 0, 1, 2 * bm + 1]
    if load == "groups":  # 10 + 15 + 8 = 33 M blocks: GROUP_SIZE_M 16 groups of 16, 16 and 1
        return [9 * bm + 1, 15 * bm, 8 * bm - 1]
    if load == "one_expert":
        return [bm + 1]
    if load == "one_pair":
        return [0, 1]
    raise ValueError(load)


# ---------------------------------------------------------------------------------------------
# Problems, tables and operands
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Problem:
    a: torch.Tensor  # [M, K] bf16; NaN rows for tokens without a live pair
    b: torch.Tensor  # [E, N, K] bf16; NaN for experts without a live pair
    w: torch.Tensor  # [M * top_k] fp32; NaN for pairs without a live slot
    assign: torch.Tensor  # [M * top_k] int64 (host): expert of each live pair, -1 otherwise
    top_k: int


def make_problem(
    counts: List[int], top_k: int, k: int, n: int, data: str, gain: float, g: torch.Generator
) -> Problem:
    """Pairs assigned to experts by ``counts``, a third more pairs left without a live slot."""
    live = sum(counts)
    tokens = max(-(-(live + live // 3 + 1) // top_k), 1)
    pairs = tokens * top_k
    order = torch.randperm(pairs, generator=g, device=DEVICE).cpu()
    assign = torch.full((pairs,), -1, dtype=torch.int64)
    position = 0
    for e, c in enumerate(counts):
        assign[order[position : position + c]] = e
        position += c
    if data == "int":
        a = torch.randint(-1, 2, (tokens, k), generator=g, device=DEVICE).to(torch.bfloat16)
        b = torch.randint(-1, 2, (len(counts), n, k), generator=g, device=DEVICE).to(torch.bfloat16)
        w = torch.randint(1, 17, (pairs,), generator=g, device=DEVICE) / 16.0
    else:
        a = torch.randn(tokens, k, generator=g, device=DEVICE).to(torch.bfloat16)
        b = torch.randn(len(counts), n, k, generator=g, device=DEVICE) * (gain / math.sqrt(k))
        b = b.to(torch.bfloat16)
        w = torch.rand(pairs, generator=g, device=DEVICE)
    w = w.float()
    used_tokens = torch.zeros(tokens, dtype=torch.bool)
    used_tokens[torch.nonzero(assign >= 0).flatten() // top_k] = True
    a[~used_tokens.to(DEVICE)] = float("nan")
    b[torch.tensor(counts, device=DEVICE) == 0] = float("nan")
    w[(assign < 0).to(DEVICE)] = float("nan")
    return Problem(a=a, b=b, w=w, assign=assign, top_k=top_k)


def build_table(assign: torch.Tensor, num_experts: int, bm: int, layout: str, g: torch.Generator):
    """``(sorted_token_ids, expert_ids)`` on the host for the live pairs in ``assign``.

    ``packed``: the align's layout (experts ascending, pairs ascending, padding at block tails).
    ``shuffled``: blocks and the slots inside each block in random order. ``holes``: each block
    holds 1 to BM pairs, padding slots anywhere. ``dead``: as shuffled, plus blocks of expert -1
    holding decoy ids of pairs without a live slot. ``padding_blocks``: as shuffled, plus blocks
    of a real expert holding padding only.
    """
    sentinel = assign.numel()

    def perm(n: int) -> torch.Tensor:
        return torch.randperm(n, generator=g, device=DEVICE).cpu()

    def block(ids: torch.Tensor, expert: int, shuffle: bool):
        slots = torch.cat([ids, torch.full((bm - ids.numel(),), sentinel, dtype=torch.int64)])
        return (expert, slots[perm(bm)] if shuffle else slots)

    blocks = []
    for e in range(num_experts):
        ids = torch.nonzero(assign == e).flatten()
        if layout != "packed":
            ids = ids[perm(ids.numel())]
        if layout == "holes":
            while ids.numel():
                take = int(torch.randint(1, bm + 1, (1,), generator=g, device=DEVICE))
                blocks.append(block(ids[:take], e, True))
                ids = ids[take:]
        else:
            for lo in range(0, ids.numel(), bm):
                blocks.append(block(ids[lo : lo + bm], e, layout != "packed"))
    if layout == "dead":
        decoys = torch.nonzero(assign < 0).flatten()
        for lo in range(0, decoys.numel(), bm):
            blocks.append(block(decoys[lo : lo + bm], -1, True))
        blocks.append(block(decoys[:0], -1, True))
    if layout == "padding_blocks":
        for e in range(num_experts):
            blocks.append(block(assign[:0], e, False))
    if layout != "packed":
        blocks = [blocks[i] for i in perm(len(blocks)).tolist()]
    if not blocks:
        return torch.zeros(0, dtype=torch.int32), torch.zeros(0, dtype=torch.int32)
    sorted_ids = torch.cat([slots for _, slots in blocks]).to(torch.int32)
    expert_ids = torch.tensor([e for e, _ in blocks], dtype=torch.int32)
    return sorted_ids, expert_ids


@dataclasses.dataclass
class Operands:
    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor  # the logical [M * top_k, N] view the kernel writes
    buffers: Dict[str, ou.GuardedBuffer]
    c_outside: Optional[torch.Tensor]  # columns of a wider C buffer past N, or None


def place_operands(p: Problem, layout: str) -> Operands:
    """``A``, ``B`` and a poisoned ``C`` in guarded buffers, with the strides of ``layout``."""
    tokens, k = p.a.shape
    num_experts, n, _ = p.b.shape
    pairs = p.assign.numel()
    buffers, c_outside = {}, None
    if layout == "a_strided":
        wide = torch.full((tokens, k + 24), float("nan"), device=DEVICE).to(torch.bfloat16)
        wide[:, :k] = p.a
        buffers["a"] = ou.guarded_like(wide)
        a = buffers["a"].tensor[:, :k]
    elif layout == "a_column_major":
        buffers["a"] = ou.guarded_like(p.a.t().contiguous())
        a = buffers["a"].tensor.t()
    else:
        buffers["a"] = ou.guarded_like(p.a)
        a = buffers["a"].tensor
    if layout == "b_transposed":
        buffers["b"] = ou.guarded_like(p.b.transpose(1, 2).contiguous())
        b = buffers["b"].tensor.transpose(1, 2)
    elif layout == "b_window":
        # Expert stride past N * K; NaN around the window, finite weights just before expert 0
        # (what an expert id of -1 would read), so a computed -1 block writes finite rows.
        store = torch.full((num_experts + 2, n + 8, k + 8), float("nan"), device=DEVICE)
        store = store.to(torch.bfloat16)
        store[0] = 1.0
        store[1 : num_experts + 1, :n, :k] = p.b
        buffers["b"] = ou.guarded_like(store)
        b = buffers["b"].tensor[1 : num_experts + 1, :n, :k]
    else:
        buffers["b"] = ou.guarded_like(p.b)
        b = buffers["b"].tensor
    if layout == "c_strided":
        buffers["c"] = ou.guarded_empty((pairs, n + 40), torch.bfloat16, DEVICE)
        c = buffers["c"].tensor[:, :n]
        c_outside = buffers["c"].tensor[:, n:]
    elif layout == "c_column_major":
        buffers["c"] = ou.guarded_empty((n, pairs), torch.bfloat16, DEVICE)
        c = buffers["c"].tensor.t()
    else:
        buffers["c"] = ou.guarded_empty((pairs, n), torch.bfloat16, DEVICE)
        c = buffers["c"].tensor
    buffers["w"] = ou.guarded_like(p.w)
    return Operands(a=a, b=b, c=c, buffers=buffers, c_outside=c_outside)


def grid_size(mode: str, num_blocks: int, n: int, block_n: int) -> int:
    tiles = num_blocks * -(-n // block_n)
    return {
        "1": 1,
        "7": 7,
        "tiles-1": max(tiles - 1, 1),
        "tiles": max(tiles, 1),
        "tiles+5": tiles + 5,
    }[mode]


def launch(
    p: Problem,
    ops: Operands,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    config: Dict[str, int],
    variant: str,
    grid: str,
) -> Dict[str, object]:
    """Launch through ``_invoke_fused_moe_kernel`` with guarded tables; return them for checks."""
    mul, fuse, clamp = {**VARIANTS, **CLAMP_VARIANTS}[variant]
    bm = config["BLOCK_SIZE_M"]
    # An empty table still gets one (never read) element, as a pool view would have.
    if not sorted_ids.numel():
        sorted_ids = torch.zeros(1, dtype=torch.int32)
        expert_ids = torch.full((1,), -1, dtype=torch.int32)
        num_blocks = 0
    else:
        num_blocks = expert_ids.numel()
    tables = {
        "sorted_ids": ou.guarded_like(sorted_ids.to(DEVICE)),
        "expert_ids": ou.guarded_like(expert_ids.to(DEVICE)),
    }
    npp = torch.tensor([num_blocks * bm], dtype=torch.int32, device=DEVICE)
    before = {name: t.tensor.clone() for name, t in tables.items()}
    before.update({name: b.tensor.clone() for name, b in ops.buffers.items() if name != "c"})
    vfm._invoke_fused_moe_kernel(
        ops.a,
        ops.b,
        ops.c,
        ops.buffers["w"].tensor,
        tables["sorted_ids"].tensor,
        tables["expert_ids"].tensor,
        npp,
        mul_routed_weight=mul,
        top_k=p.top_k,
        config=dict(config),
        grid_size=grid_size(grid, num_blocks, p.b.shape[1], config["BLOCK_SIZE_N"]),
        fuse_squared_relu=fuse,
        activation_clamp_scale=clamp,
    )
    return dict(tables=tables, npp=npp, before=before, num_blocks=num_blocks, bm=bm)


# ---------------------------------------------------------------------------------------------
# References and checks
# ---------------------------------------------------------------------------------------------


def _activation(h: torch.Tensor, fuse: bool, clamp: Optional[float]) -> torch.Tensor:
    if not fuse:
        return h
    r = torch.relu(h)
    if clamp is not None:
        r = clamp * torch.tanh(r / clamp)
    return r * r


def reference_rows(a, b, w, assign, top_k: int, variant: str) -> torch.Tensor:
    """The contract per live pair, one expert at a time (no tables, tiles or padding)."""
    mul, fuse, clamp = {**VARIANTS, **CLAMP_VARIANTS}[variant]
    out = torch.zeros(assign.numel(), b.shape[1], dtype=a.dtype, device=a.device)
    for e in range(b.shape[0]):
        pairs = torch.nonzero(assign == e).flatten().to(a.device)
        if pairs.numel() == 0:
            continue
        # + 0.0: an exact zero sum is the +0.0 of an accumulator that starts at +0.0, whatever
        # the signs of its zero products.
        rows = _activation(a[pairs // top_k] @ b[e].t() + 0.0, fuse, clamp)
        out[pairs] = rows * w[pairs, None] if mul else rows
    return out


@contextlib.contextmanager
def _ieee_fp32_matmul():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")  # no TF32
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _require_exact(p: Problem, variant: str) -> None:
    """Test precondition for the integer spec: every value the kernel forms is exact in fp32.

    ``|A @ B.T| <= K``; the square is at most ``K**2``; a weight ``j / 16`` adds 4 bits.
    """
    mul, fuse, _ = {**VARIANTS, **CLAMP_VARIANTS}[variant]
    k = p.a.shape[1]
    bound = (k * k if fuse else k) * (16 if mul else 1)
    if bound > 2**24:
        raise ValueError(f"test precondition: K = {k} is too large for the exact spec")


def check(case_id: str, p: Problem, ops: Operands, run: Dict[str, object], variant: str, data):
    """Live rows against the oracle, every other element of C untouched, inputs read-only."""
    live = torch.nonzero(p.assign >= 0).flatten().to(DEVICE)
    dead = torch.nonzero(p.assign < 0).flatten().to(DEVICE)
    got = ops.c[live]
    if data == "int":
        _require_exact(p, variant)
        spec = ou.fp64_reference(reference_rows, p.a, p.b, p.w, p.assign, p.top_k, variant)
        ou.assert_exact(got, spec[live].to(torch.bfloat16), what=f"{case_id}: live rows")
    else:
        reference = ou.fp64_reference(reference_rows, p.a, p.b, p.w, p.assign, p.top_k, variant)
        with _ieee_fp32_matmul():
            trusted = reference_rows(p.a.float(), p.b.float(), p.w, p.assign, p.top_k, variant)
        ou.assert_close_to_reference(
            got, reference[live], trusted[live].to(torch.bfloat16), dim=-1, what=case_id
        )
    ou.assert_poisoned(ops.c[dead], what=f"{case_id}: rows of pairs without a live slot")
    if ops.c_outside is not None:
        ou.assert_poisoned(ops.c_outside, what=f"{case_id}: C columns past N")
    ou.check_guards(**ops.buffers, **run["tables"])
    for name, t in run["before"].items():
        now = run["tables"][name].tensor if name in run["tables"] else ops.buffers[name].tensor
        ou.assert_exact(now, t, what=f"{case_id}: {name} is read-only")
    ou.assert_exact(
        run["npp"], torch.tensor([run["num_blocks"] * run["bm"]]), what="npp is read-only"
    )


def gain_for(variant: str) -> float:
    """FC1 weight gain that puts relu(h) / s around 1 for the clamp (as the entry oracle)."""
    clamp = {**VARIANTS, **CLAMP_VARIANTS}[variant][2]
    return clamp if clamp is not None and clamp > 1 else 1.0


def run_case(case: Case, seed: int = 0):
    """Build, launch and check one case; return the problem and the C rows of its live pairs."""
    g = ou.make_generator(ou.DEFAULT_SEED + seed, device=DEVICE)
    config = CONFIGS[case.config]
    counts = counts_for(case.load, config["BLOCK_SIZE_M"])
    p = make_problem(counts, case.top_k, case.k, case.n, case.data, gain_for(case.variant), g)
    sorted_ids, expert_ids = build_table(
        p.assign, len(counts), config["BLOCK_SIZE_M"], case.table, g
    )
    ops = place_operands(p, case.strides)
    run = launch(p, ops, sorted_ids, expert_ids, config, case.variant, case.grid)
    check(case.id(), p, ops, run, case.variant, case.data)
    return p, ops


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------

TABLES = ["packed", "shuffled", "holes", "dead", "padding_blocks"]
STRIDES = [
    "contiguous",
    "a_strided",
    "a_column_major",
    "b_transposed",
    "b_window",
    "c_strided",
    "c_column_major",
]
GRIDS = ["1", "7", "tiles-1", "tiles", "tiles+5"]

EXACT_CASES = [
    Case(data="int", **c)
    for c in ou.pairwise_cases(
        {
            "config": list(CONFIGS),
            "variant": list(VARIANTS),
            "k": [1, 63, 64, 65, 127, 128, 129, 300],
            "n": [1, 63, 64, 65, 127, 128, 129, 300],
            "top_k": [1, 2, 3, 6, 8],
            "table": TABLES,
            "strides": STRIDES,
            "grid": GRIDS,
            "load": ["light", "groups", "one_expert"],
        }
    )
]


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.parametrize("case", [pytest.param(c, id=c.id()) for c in EXACT_CASES])
def test_exact_variants_configs_and_tables(case):
    """Small-integer data: every constexpr variant, launch config, table layout, stride layout
    and grid gives the spec bit for bit, and writes nothing else."""
    run_case(case)


# One config per BLOCK_SIZE_M, plus a GROUP_SIZE_M 16 one; the exact sweep covers all configs.
TOLERANCE_CONFIGS = [
    "bm16-bn64-bk128-g1-w4-s4",
    "bm32-bn64-bk64-g1-w4-s3",
    "bm32-bn128-bk64-g1-w4-s3",
    "bm64-bn128-bk64-g16-w8-s3",
    "bm128-bn128-bk64-g1-w8-s3",
]
assert set(TOLERANCE_CONFIGS) <= set(CONFIGS), (TOLERANCE_CONFIGS, list(CONFIGS))

TOLERANCE_CASES = [
    Case(data="gauss", **c)
    for c in ou.pairwise_cases(
        {
            "variant": list(VARIANTS) + list(CLAMP_VARIANTS),
            "config": TOLERANCE_CONFIGS,
            "variant": list(VARIANTS) + list(CLAMP_VARIANTS),
            "top_k": [1, 2, 6],
            "k": [64, 129, 1000],
            "n": [65, 128, 1025],
            "table": ["packed", "holes", "dead"],
            "strides": ["contiguous", "b_transposed", "c_strided"],
            "grid": ["7", "tiles"],
        }
    )
]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", [pytest.param(c, id=c.id()) for c in TOLERANCE_CASES])
def test_matches_fp64_reference(case):
    """Gaussian data, every variant including the tanh clamp, against the fp64 reference."""
    run_case(case)


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.parametrize("variant", ["plain", "relu2", "weighted"])
@pytest.mark.parametrize("block_k", [64, 128])
def test_bits_fixed_by_block_k(block_k, variant):
    """At a fixed BLOCK_SIZE_K, C's bits do not depend on anything else.

    Gaussian data with K = 1000 (a partial last K block), one problem, launched with every
    config of that BLOCK_SIZE_K (BM 16 to 128, BN 64/128, 4/8 warps, 3/4 stages, GROUP_SIZE_M
    1/16), table layouts that move pairs and padding within and across blocks, and grids from 1
    to past the tile count. Every live row must equal the first launch's bit for bit; that first
    launch also matches the fp64 reference.
    """
    configs = {name: c for name, c in CONFIGS.items() if c["BLOCK_SIZE_K"] == block_k}
    g = ou.make_generator(device=DEVICE)
    # Fixed counts, so every BLOCK_SIZE_M sees the same problem (around 16, 32, 64 and 128).
    p = make_problem([37, 70, 0, 129, 1, 200], 2, 1000, 200, "gauss", 1.0, g)
    live = torch.nonzero(p.assign >= 0).flatten().to(DEVICE)
    first = None
    for name, config in configs.items():
        for table in ("packed", "shuffled", "holes", "dead"):
            for grid in ("1", "tiles+5"):
                tables = build_table(p.assign, 6, config["BLOCK_SIZE_M"], table, g)
                ops = place_operands(p, "contiguous")
                run = launch(p, ops, *tables, config, variant, grid)
                what = f"{name}-{table}-grid{grid}"
                if first is None:
                    check(what, p, ops, run, variant, "gauss")
                    first = ops.c[live].clone()
                    continue
                ou.assert_exact(ops.c[live], first, what=f"{what}: bits vs the first launch")
                dead = torch.nonzero(p.assign < 0).flatten().to(DEVICE)
                ou.assert_poisoned(ops.c[dead], what=f"{what}: rows without a live slot")
                ou.check_guards(**ops.buffers, **run["tables"])


DEGENERATE = {
    "one-pair": Case(config="bm128-bn128-bk64-g16-w8-s3", load="one_pair", variant="weighted"),
    "k1": Case(config="bm32-bn64-bk64-g1-w4-s3", k=1, variant="relu2", load="one_expert"),
    "n1": Case(config="bm64-bn128-bk64-g1-w8-s3", n=1, variant="weighted-relu2", top_k=3),
    "k1-n1": Case(config="bm16-bn64-bk128-g1-w4-s4", k=1, n=1, table="holes", top_k=8),
}


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("name", list(DEGENERATE))
def test_degenerate(name):
    """A single live pair, K = 1, N = 1."""
    run_case(DEGENERATE[name])


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("table", ["dead", "packed"])
def test_no_live_slot_writes_nothing(table):
    """No live pair: only blocks of expert -1 (holding a decoy pair), or ``npp = 0``."""
    g = ou.make_generator(device=DEVICE)
    p = make_problem([0, 0], 1, 64, 64, "int", 1.0, g)
    sorted_ids, expert_ids = build_table(p.assign, 2, 16, table, g)
    ops = place_operands(p, "b_window")
    run = launch(p, ops, sorted_ids, expert_ids, CONFIGS["bm16-bn64-bk128-g1-w4-s4"], "plain", "7")
    check(f"no live slot ({table})", p, ops, run, "plain", "int")


PRODUCTION = {
    # Nemotron 3 Nano, EP 8 prefill (hint 1024 -> BM 128): FC1 2688 -> 1856 with the fused
    # squared ReLU, top-6, 16 local experts; then FC2 1856 -> 2688, weighted.
    "nemotron-nano-fc1": ("bm128-bn128-bk64-g1-w8-s3", "relu2", 6, 2688, 1856, 16, 1024 * 6),
    "nemotron-nano-fc2": ("bm128-bn128-bk64-g1-w8-s3", "weighted", 1, 1856, 2688, 16, 1024 * 6),
    # Nemotron 3 Nano decode with the tanh clamp 16 (hint 64 -> BM 32, BN 64, BK 128).
    "nemotron-nano-fc1-decode-clamp16": (
        "bm32-bn64-bk128-g1-w4-s3",
        "clamp16",
        6,
        2688,
        1856,
        16,
        64 * 6,
    ),
    # DeepSeek-V2-Lite SwiGLU FC1, 2048 -> 2 x 1408, top-6, 8 local experts (hint 256 -> BM 64).
    "deepseek-v2-lite-fc1": ("bm64-bn128-bk64-g1-w8-s3", "plain", 6, 2048, 2816, 8, 256 * 6),
    # Mixtral 8x7B FC2 at EP 8, 14336 -> 4096, one local expert (hint 96 -> BM 32, BN 128).
    "mixtral-8x7b-fc2": ("bm32-bn128-bk64-g1-w4-s3", "weighted", 1, 14336, 4096, 1, 96 * 2),
}


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("name", list(PRODUCTION))
def test_production_sizes(name):
    """Deployed FC1 and FC2 shapes at the config their hint selects (gaussian data)."""
    config, variant, top_k, k, n, num_experts, pairs = PRODUCTION[name]
    g = ou.make_generator(device=DEVICE)
    weights = torch.rand(num_experts, generator=g, device=DEVICE)
    draws = torch.multinomial(weights, pairs, replacement=True, generator=g)
    counts = torch.bincount(draws, minlength=num_experts).tolist()
    p = make_problem(counts, top_k, k, n, "gauss", gain_for(variant), g)
    tables = build_table(p.assign, num_experts, CONFIGS[config]["BLOCK_SIZE_M"], "shuffled", g)
    ops = place_operands(p, "contiguous")
    run = launch(p, ops, *tables, CONFIGS[config], variant, "tiles")
    check(name, p, ops, run, variant, "gauss")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "expert",
    [
        pytest.param(1, id="offset-below-2^31", marks=pytest.mark.large),
        pytest.param(2, id="offset-past-2^31", marks=pytest.mark.large),
    ],
)
def test_int32_expert_offsets(expert):
    """``B`` as a 128 x 128 window of a ``[3, 32769, 32768]`` buffer: expert ``e`` starts at
    element ``e * (2**30 + 2**15)``, past 2**31 for expert 2. Its pairs must read its weights."""
    ou.require_free_memory(8 * 2**30)
    rows, cols, k, n = 2**15 + 1, 2**15, 128, 128
    g = ou.make_generator(device=DEVICE)
    counts = [0, 0, 0]
    counts[expert] = 100
    counts[0] = 30
    p = make_problem(counts, 2, k, n, "int", 1.0, g)
    store = torch.full((3, rows, cols), float("nan"), dtype=torch.bfloat16, device=DEVICE)
    store[:, :n, :k] = p.b
    ops = place_operands(p, "contiguous")
    ops.b = store[:, :n, :k]
    ops.buffers.pop("b")  # the 6 GiB window store is not guarded or compared
    assert (ops.b.stride(0) * expert >= 2**31) == (expert == 2)
    config = CONFIGS["bm64-bn128-bk64-g1-w8-s3"]
    sorted_ids, expert_ids = build_table(p.assign, 3, config["BLOCK_SIZE_M"], "packed", g)
    run = launch(p, ops, sorted_ids, expert_ids, config, "weighted", "tiles")
    check(f"expert {expert}", p, ops, run, "weighted", "int")


@pytest.mark.sanitizer
@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "case",
    [
        Case(config="bm16-bn64-bk128-g1-w4-s4", variant="relu2", k=65, n=63, table="dead"),
        Case(
            config="bm64-bn128-bk64-g16-w8-s3",
            variant="weighted",
            top_k=3,
            k=129,
            n=127,
            table="holes",
            strides="c_column_major",
            grid="7",
        ),
    ],
    ids=lambda c: c.id(),
)
def test_sanitizer(case):
    """Small cases for compute-sanitizer memcheck (also run normally)."""
    run_case(case)


def all_cases() -> List[Case]:
    """Every case of the parametrized ``run_case`` tests (for scratch checks outside pytest)."""
    return list(EXACT_CASES) + list(TOLERANCE_CASES)
