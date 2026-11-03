# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the vLLM fused-MoE backend, ``vllm_fused_moe``
(``megatron/core/inference/moe/vllm_fused_moe.py:706``), with its block alignment
(``_moe_align_block_size_cuda_graphable`` and its init / count / prefix-sum / fill / scatter
kernels), launch-config selection (``_get_default_config``), persistent buffers
(``VllmFusedMoeBuffers``) and activation glue (``bounded_silu_mul``, the batch-invariant
weighted activations, ``_moe_sum``).

Contract
--------
Written from the call site (``InferenceGroupedMLP._vllm_forward``, ``experts.py``), the training
expert math (``TEGroupedMLP``: ``squared_relu``, the tanh clamp, gated SiLU,
``moe_apply_probs_on_input``) and the function's documented API, before reading the kernels.

* **Computes**, for each live token ``t < valid_tokens``, with ``g = routing_map[t, j]`` and the
  rank's experts ``[start, start + E_local)``::

      out[t] = sum over slots j with start <= g < start + E_local of
               p[t, j] * W2[e] @ act(W1[e] @ x[t]),        e = g - start

  ``W1 = fc1_weight[e]`` is ``[N1, H]`` and ``W2 = fc2_weight[e]`` is ``[H, F]``.
  ``act`` is ``relu(h)**2`` (SQUARED_RELU, ``F = N1``); with ``activation_clamp_scale = s``,
  ``(s * tanh(relu(h) / s))**2``; or SwiGLU ``silu(gate) * up`` with ``gate`` the first half of
  ``h`` (SWIGLU, ``N1 = 2F``). Slots on other ranks' experts or ``-1`` contribute nothing; a
  token without a local slot gets an exact zero row.
* **Probabilities on the input** (``moe_apply_probs_on_input``, training's
  ``_unfused_forward``, top-k 1): ``out[t] = sum_j W2[e] @ act(W1[e] @ bf16(p[t, j] * x[t]))``,
  with no probability applied to the output. On ``main`` the backend has no such option (the
  layer ignores the config; audit finding D2). #8566 adds ``apply_probs_on_input``.
* **Inputs**: ``hidden_states`` ``[max_tokens, H]`` bf16, any row stride; ``probs``
  ``[max_tokens, topk]`` fp32, contiguous; ``routing_map`` ``[max_tokens, topk]`` int32 or
  int64, contiguous; ``fc1_weight`` ``[E_local, N1, H]`` and ``fc2_weight`` ``[E_local, H, F]``
  bf16; ``valid_tokens`` an int32 device scalar; ``num_tokens_hint`` a host int or None
  (performance only: no result may depend on it); ``out`` an optional ``[max_tokens, H]`` fp32
  or bf16 buffer. Rows past ``valid_tokens`` of every input are garbage and must not be read.
* **Outputs**: ``out`` (or a new fp32 ``[max_tokens, H]``) with rows ``< valid_tokens``
  written and the rest untouched. The inputs are read-only.
* **Numerics** (the rounding points; every GEMM multiplies bf16 operands exactly and
  accumulates in fp32):

  - squared ReLU, default mode: ``a = bf16(act(h))`` with the activation on the fp32
    accumulator; ``y = bf16(p * (W2 @ a))``, the probability on the fp32 accumulator (#4743);
    ``out = sum_j y`` in fp32;
  - SwiGLU, default mode: ``h = bf16(W1 @ x)``; ``a = bf16(silu(gate) * up)`` in fp32; ``y``
    and ``out`` as above;
  - batch-invariant mode (``batch_invariant.enabled()``): ``h = bf16(W1 @ x)``;
    ``a = bf16(act(h) * p)`` in fp32, as training's weighted activations; ``y = bf16(W2 @ a)``;
    ``out = fp32(sum_j y)`` summed in fp64. A token's output bits must not depend on the other
    tokens, ``valid_tokens``, ``max_tokens`` or ``num_tokens_hint``;
  - probabilities on the input: ``x' = bf16(p * x)``, then as above with no other probability.
* **Degenerate inputs**: ``valid_tokens = 0`` writes nothing; ``max_tokens = 0`` returns an
  empty output; experts with no tokens; every token on one expert; ``top-k = E_local``.
* **Special values**: a NaN in a live token's hidden row makes that token's output row NaN
  (torch's relu, silu and tanh propagate NaN, as in training) and changes no other token; a
  NaN token with no local slot still gets a zero row.
* **Tolerance class**: floating point. Per output row, the max-abs and rms error against the
  fp64 reference are at most 2x those of a trusted torch implementation with the same rounding
  points (fp32 matmuls on the same device), plus a one-ulp floor (``assert_close_to_reference``,
  ``dim=-1``). Exact by construction: with small-integer inputs every accumulation is exact
  (checked per case), so squared-ReLU output must equal the rounding-point spec bit for bit.
  Batch invariance: bitwise. Zero rows and untouched rows: exact.

Variants and dispatch (from the kernels, after the contract)
------------------------------------------------------------
* ``_fused_moe_kernel`` constexprs: ``FUSE_SQUARED_RELU`` (default-mode squared ReLU, FC1),
  ``CLAMP_ACTIVATION``, ``MUL_ROUTED_WEIGHT`` (default-mode FC2), ``top_k`` (router top-k at FC1,
  1 at FC2), and the launch config. The config is a table on the hint ``M`` (or ``max_tokens``)
  and ``E_local`` (``vllm_fused_moe.py:48-102``): ``M <= 32`` BM16/BN64/BK128/4 warps/4 stages;
  ``<= 64`` BM32/BN64/BK128; ``<= 96`` BM32/BN128/BK64; ``<= 128`` BM64/BN128/BK64/4 warps;
  ``<= 512`` BM64/8 warps; above BM128; GROUP_SIZE_M 16 iff ``M // E_local > 128``. Batch-invariant
  mode pins BK64 (``:773``). Covered: every row, both groupings, both modes (``TUNER``,
  ``test_tuner_thresholds``), checked against the launches the kernel actually made.
* Grid ``ceil((hint * topk + BM * E_local) / BM) * ceil(N / BN)`` (``:788-794``); CTAs stride
  when the real tile count is larger (``test_grid_stride``, with a multi-group swizzle).
* Alignment: per-expert counts padded to BM, so per-expert counts BM-1/BM/BM+1/2BM+1, empty
  experts, a single hot expert (more than 128 BM-blocks: the fill kernel's ``BLOCK=128`` loop,
  ``:481``), scatter ``BLOCK 256`` (``:485``), init ``BLOCK 1024`` (``:453``).
* Glue: ``bounded_silu_mul`` / the batch-invariant activations (``BLOCK <= 1024``, at most 512
  programs: widths 1025 and production, more than 512 rows); ``_moe_sum`` ``BLOCK_K <= 1024``
  (``:679``, hidden 1025 and production sizes); fp32 and fp64 accumulation; fp32, bf16 and
  allocated outputs; the persistent pool (poisoned, reused across layers, too small).
* Tile boundaries: H and F at BK-1/BK/BK+1 and BN-1/BN/BN+1 for BK, BN in {64, 128}; strided
  hidden states; int32 and int64 routing maps; offsets past 2**31 (``large``).

Open PRs
--------
* #8566 (d2) adds ``apply_probs_on_input`` (and the GLU options and gated squared ReLU, which this
  file does not cover). ``test_probs_on_input`` fails on ``main`` with a ``TypeError`` (the
  option does not exist; the layer silently ignores ``moe_apply_probs_on_input``) and must pass
  with #8566. Every other test has the same expected result with #8566 (default options).
* #8562 (d4-d9) changes the router, not this file: it makes top-1 routing a dense
  ``[tokens, 1]`` map, the layout every test here feeds. No expected result changes.

Hardware: H100. ``test_batch_invariance_bitwise`` also runs on GB200, where the tensor-core
instructions differ (Blackwell ``tcgen05``), as the step 3 plan R06 asks.
"""

import contextlib
import dataclasses
import importlib
import math
from typing import Dict, List, Optional, Union

import pytest
import torch
import torch.nn.functional as F

from megatron.core.inference.moe.fused_moe import ActivationType
from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode
from tests.unit_tests.inference.oracles import oracle_utils as ou

# The package re-exports the function under the module's name, so import the module itself.
vfm = importlib.import_module("megatron.core.inference.moe.vllm_fused_moe")

DEVICE = torch.device("cuda")

SWIGLU = "swiglu"
SQRELU = "squared_relu"
CLAMP16 = "clamp16"  # the production clamp scale (activation_func_tanh_clamp_scale=16)
CLAMP_HALF = "clamp0.5"  # saturates almost everywhere

# activation name -> (ActivationType, clamp scale, FC1 weight gain). The gain puts relu(h) / s
# around 1, so the clamp is neither linear nor saturated everywhere.
ACTIVATIONS = {
    SWIGLU: (ActivationType.SWIGLU, None, 1.0),
    SQRELU: (ActivationType.SQUARED_RELU, None, 1.0),
    CLAMP16: (ActivationType.SQUARED_RELU, 16.0, 16.0),
    CLAMP_HALF: (ActivationType.SQUARED_RELU, 0.5, 1.0),
}

# One representative (num_tokens_hint, E_local) per row of the launch-config table
# (vllm_fused_moe.py:48-102), both GROUP_SIZE_M values for the rows that can pick 16.
TUNER = {
    "bm16": (16, 4),
    "bm32-bn64": (48, 4),
    "bm32-bn128": (80, 4),
    "bm64-w4": (112, 4),
    "bm64-w8": (256, 4),
    "bm64-g16": (512, 2),
    "bm128": (1024, 8),
    "bm128-g16": (1024, 2),
}


def spec_config(m: int, num_local: int, batch_invariant: bool) -> Dict[str, int]:
    """The launch-config table (vllm_fused_moe.py:48-102; BK pinned in BI mode, :773)."""
    return {
        "BLOCK_SIZE_M": 16 if m <= 32 else 32 if m <= 96 else 64 if m <= 512 else 128,
        "BLOCK_SIZE_N": 64 if m <= 64 else 128,
        "BLOCK_SIZE_K": 64 if batch_invariant or m > 64 else 128,
        "GROUP_SIZE_M": 16 if m // max(num_local, 1) > 128 else 1,
        "num_warps": 4 if m <= 128 else 8,
        "num_stages": 4 if m <= 32 else 3,
    }


# ---------------------------------------------------------------------------------------------
# Cases and inputs
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Case:
    """One call of ``vllm_fused_moe``."""

    act: str = SQRELU
    data: str = "gauss"  # "gauss": fp64 reference + trusted bound; "int": exact spec
    bi: bool = False
    probs_on_input: bool = False
    topk: int = 2
    hint: Optional[int] = 16  # num_tokens_hint; None: the config follows max_tokens
    num_local: int = 4
    ep: int = 1  # expert-parallel size; this rank is ep // 2
    tokens: Union[int, str] = "bm"  # live tokens, or relative to the config's BLOCK_SIZE_M
    pad: int = 3  # rows past valid_tokens
    routing: str = "random"
    hidden: int = 128
    ffn: int = 128
    out: str = "fp32"  # "fp32" / "bf16": a guarded buffer; "none": the function allocates
    routing_dtype: torch.dtype = torch.int64
    strided: bool = False  # hidden_states with a row stride larger than H

    @property
    def config_m(self) -> int:
        return self.hint if self.hint is not None else self.max_tokens

    @property
    def block_m(self) -> int:
        if self.hint is None:
            raise ValueError("block_m of a hint-less case depends on max_tokens")
        return spec_config(self.hint, self.num_local, self.bi)["BLOCK_SIZE_M"]

    @property
    def max_tokens(self) -> int:
        return self.valid + self.pad

    @property
    def valid(self) -> int:
        if isinstance(self.tokens, int):
            return self.tokens
        bm = self.block_m
        return {"bm-1": bm - 1, "bm": bm, "bm+1": bm + 1, "2bm+1": 2 * bm + 1}[self.tokens]

    def id(self) -> str:
        parts = [self.act, self.data, "bi" if self.bi else "nobi", f"k{self.topk}"]
        parts += ["poi"] if self.probs_on_input else []
        parts += [f"hint{self.hint}", f"el{self.num_local}", f"ep{self.ep}", f"t{self.tokens}"]
        parts += [f"pad{self.pad}", self.routing, f"h{self.hidden}", f"f{self.ffn}", self.out]
        parts += ["i32"] if self.routing_dtype == torch.int32 else []
        parts += ["strided"] if self.strided else []
        return "-".join(str(p) for p in parts)


def params(cases: List[Case], marks=()) -> List[object]:
    return [pytest.param(c, id=c.id(), marks=marks) for c in cases]


def with_tuner(factors: dict) -> dict:
    """Replace a pairwise ``hint_el`` factor (a TUNER row) by its hint and E_local."""
    factors = dict(factors)
    factors["hint"], factors["num_local"] = TUNER[factors.pop("hint_el")]
    return factors


@dataclasses.dataclass
class Inputs:
    hidden: torch.Tensor
    probs: torch.Tensor
    routing: torch.Tensor
    fc1: torch.Tensor
    fc2: torch.Tensor
    valid: int
    start: int
    num_local: int
    buffers: Dict[str, ou.GuardedBuffer]


def _from_counts(
    counts: Dict[int, int], tokens: int, topk: int, g: torch.Generator
) -> torch.Tensor:
    """Routing in which expert ``e`` holds ``counts[e]`` slots (``-1`` holds the rest).

    Slots are filled column by column from the expert-sorted list, so an expert with at most
    ``tokens`` slots never appears twice in one token; then tokens and slots are shuffled.
    """
    ids = torch.cat(
        [
            torch.full((c,), e, dtype=torch.int64)
            for e, c in sorted(counts.items(), key=lambda i: i[0] < 0)
            if c
        ]
        or [torch.empty(0, dtype=torch.int64)]
    ).to(DEVICE)
    assert ids.numel() == tokens * topk, (ids.numel(), tokens, topk)
    routing = ids.view(topk, tokens).t()
    routing = routing[torch.randperm(tokens, generator=g, device=DEVICE)]
    order = torch.rand(tokens, topk, generator=g, device=DEVICE).argsort(dim=1)
    return torch.gather(routing, 1, order).contiguous()


def make_routing(case: Case, g: torch.Generator, num_experts: int, start: int) -> torch.Tensor:
    """Live rows of the routing map (global expert ids, -1 for a masked slot)."""
    k, num_local = case.topk, case.num_local
    local = list(range(start, start + num_local))
    nonlocal_ids = [e for e in range(num_experts) if not start <= e < start + num_local]

    if case.routing == "block_counts":
        # Per-expert counts around BLOCK_SIZE_M (and empty experts); the rest goes elsewhere.
        bm = case.block_m
        cycle = [bm - 1, bm, bm + 1, 0, 1, 2 * bm + 1, 2 * bm, 2 * bm - 1]
        counts = {e: cycle[i % len(cycle)] for i, e in enumerate(local)}
        total = sum(counts.values())
        tokens = max(max(counts.values()), -(-total // k), 1)
        rest = tokens * k - total
        for e in nonlocal_ids:
            take = min(rest, tokens)
            counts[e], rest = take, rest - take
        counts[-1] = rest
        return _from_counts(counts, tokens, k, g)

    tokens = case.valid
    scores = torch.rand(tokens, num_experts, generator=g, device=DEVICE)
    if case.routing == "one_expert":
        scores[:, start + num_local - 1] = 2.0  # every token on the last local expert
    elif case.routing == "empty_experts":
        # The first and last local experts, and every other one, receive nothing (with two
        # local experts, the first).
        used = [e for i, e in enumerate(local) if i % 2 == 1 and i != num_local - 1]
        used = used or local[1:]
        scores[:, [e for e in local if e not in used]] = -1.0
    elif case.routing == "nonlocal_only":
        scores[:, local] = -1.0
    elif case.routing not in ("random", "masked", "all_masked"):
        raise ValueError(case.routing)
    take = min(k, num_experts)
    order = scores.argsort(dim=1, descending=True)[:, :take]
    picked = torch.gather(scores, 1, order)
    routing = torch.where(picked >= 0, order, torch.full_like(order, -1))
    if take < k:  # more slots than experts: the extra slots repeat experts
        extra = torch.randint(0, num_experts, (tokens, k - take), generator=g, device=DEVICE)
        routing = torch.cat([routing, extra], dim=1)
    if case.routing == "masked":
        routing[torch.rand(tokens, k, generator=g, device=DEVICE) < 0.25] = -1
    elif case.routing == "all_masked":
        routing.fill_(-1)
    return routing.contiguous()


def make_inputs(case: Case, g: torch.Generator) -> Inputs:
    act, _, gain = ACTIVATIONS[case.act]
    num_experts, start = case.ep * case.num_local, (case.ep // 2) * case.num_local
    routing_live = make_routing(case, g, num_experts, start)
    valid = routing_live.shape[0]
    max_tokens = valid + case.pad
    k, hidden, ffn = case.topk, case.hidden, case.ffn
    n1 = 2 * ffn if act == ActivationType.SWIGLU else ffn

    if case.data == "int":
        # Small integers (and probabilities j/16): every GEMM and sum is exact in fp32.
        def ints(*shape):
            return torch.randint(-1, 2, shape, generator=g, device=DEVICE).to(torch.bfloat16)

        x, w1, w2 = (
            ints(max_tokens, hidden),
            ints(case.num_local, n1, hidden),
            ints(case.num_local, hidden, ffn),
        )
        p = torch.randint(1, 17, (max_tokens, k), generator=g, device=DEVICE) / 16.0
    else:
        x = torch.randn(max_tokens, hidden, generator=g, device=DEVICE).to(torch.bfloat16)
        w1 = torch.randn(case.num_local, n1, hidden, generator=g, device=DEVICE)
        w1 = (w1 * (gain / math.sqrt(hidden))).to(torch.bfloat16)
        w2 = torch.randn(case.num_local, hidden, ffn, generator=g, device=DEVICE)
        w2 = (w2 / math.sqrt(ffn)).to(torch.bfloat16)
        p = torch.rand(max_tokens, k, generator=g, device=DEVICE)
    p = p.float()
    # Rows past valid_tokens: NaN data routed to local experts, so reading them shows.
    x[valid:] = float("nan")
    p[valid:] = float("nan")
    routing = torch.randint(
        start, start + case.num_local, (max_tokens, k), generator=g, device=DEVICE
    )
    routing[:valid] = routing_live
    routing = routing.to(case.routing_dtype)

    # Guard bands with NaN sentinels: an out-of-bounds read of a float input becomes a NaN.
    buffers = {}
    if case.strided:
        wide = torch.full((max_tokens, hidden + 16), float("nan"), device=DEVICE)
        wide = wide.to(torch.bfloat16)
        wide[:, :hidden] = x
        buffers["hidden"] = ou.guarded_like(wide)
        hidden_states = buffers["hidden"].tensor[:, :hidden]
    else:
        buffers["hidden"] = ou.guarded_like(x)
        hidden_states = buffers["hidden"].tensor
    for name, t in (("probs", p), ("fc1", w1), ("fc2", w2), ("routing", routing)):
        buffers[name] = ou.guarded_like(t)
    return Inputs(
        hidden=hidden_states,
        probs=buffers["probs"].tensor,
        routing=buffers["routing"].tensor,
        fc1=buffers["fc1"].tensor,
        fc2=buffers["fc2"].tensor,
        valid=valid,
        start=start,
        num_local=case.num_local,
        buffers=buffers,
    )


# ---------------------------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------------------------


def _activation(h: torch.Tensor, act: str) -> torch.Tensor:
    kind, clamp, _ = ACTIVATIONS[act]
    if kind == ActivationType.SWIGLU:
        gate, up = h.chunk(2, dim=-1)
        return F.silu(gate) * up
    r = torch.relu(h)
    if clamp is not None:
        r = clamp * torch.tanh(r / clamp)
    return r * r


def reference_moe(x, probs, fc1, fc2, routing, *, act, start, probs_on_input):
    """fp64 MoE from the contract: each token's sum over its local slots, no padding or tiles.

    Evaluated one expert at a time (each row of an expert's matmul is one token).
    """
    tokens, topk = routing.shape
    y = torch.zeros(tokens, topk, x.shape[1], dtype=x.dtype, device=x.device)
    for e in range(fc1.shape[0]):
        tok, slot = torch.nonzero(routing == start + e, as_tuple=True)
        if tok.numel() == 0:
            continue
        p = probs[tok, slot].unsqueeze(1)
        inp = x[tok] * p if probs_on_input else x[tok]
        out = _activation(inp @ fc1[e].t(), act) @ fc2[e].t()
        y[tok, slot] = out if probs_on_input else out * p
    return y.sum(dim=1)


@contextlib.contextmanager
def _ieee_fp32_matmul():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")  # no TF32
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _quantum(t: torch.Tensor) -> float:
    """The largest power of two dividing every element of ``t`` (inf when all are zero)."""
    v = t.detach().double().abs().flatten()
    v = v[(v != 0) & torch.isfinite(v)]
    if v.numel() == 0:
        return math.inf
    mantissa, exponent = torch.frexp(v)
    integer = (mantissa * 2.0**53).to(torch.int64)
    lowest_bit = torch.log2((integer & -integer).double()).round()
    return 2.0 ** float((exponent.double() - 53 + lowest_bit).min())


def _require_exact(bound: torch.Tensor, quantum: float, what: str) -> None:
    """Test precondition: sums of multiples of ``quantum`` below ``bound`` are exact in fp32."""
    if bound.numel() and float(bound.max()) / quantum > 2.0**24:
        raise ValueError(
            f"test precondition: {what} is not exact in fp32 (bound {float(bound.max())},"
            f" quantum {quantum}); the integer inputs are too large for this size"
        )


def _gemm(a: torch.Tensor, w: torch.Tensor, exact: bool, what: str) -> torch.Tensor:
    """``a @ w.T`` of bf16 operands into an fp32 accumulator (exact products)."""
    if not exact:
        return a.float() @ w.float().t()
    _require_exact(a.double().abs() @ w.double().abs().t(), _quantum(a) * _quantum(w), what)
    # The sum is exact in fp32 in any order; + 0.0 turns a -0.0 into the +0.0 an fp32
    # accumulator starting at +0.0 produces.
    return (a.double() @ w.double().t() + 0.0).float()


def rounded_moe(inp: Inputs, case: Case, exact: bool) -> torch.Tensor:
    """The contract's arithmetic with its rounding points, in torch (fp32 out, valid rows).

    ``exact=False``: the trusted implementation (fp32 matmuls, TF32 off). ``exact=True``: the
    rounding-point spec for small-integer inputs (exact accumulation, checked).
    """
    v, k, start = inp.valid, case.topk, inp.start
    x, p, routing = inp.hidden[:v], inp.probs[:v], inp.routing[:v].long()
    kind = ACTIVATIONS[case.act][0]
    y = torch.zeros(v, k, x.shape[1], dtype=torch.float32, device=x.device)
    with _ieee_fp32_matmul():
        for e in range(inp.num_local):
            tok, slot = torch.nonzero(routing == start + e, as_tuple=True)
            if tok.numel() == 0:
                continue
            pe = p[tok, slot].unsqueeze(1)
            xe = x[tok]
            if case.probs_on_input:
                xe = (xe.float() * pe).to(torch.bfloat16)
            h = _gemm(xe, inp.fc1[e], exact, "FC1")
            if kind == ActivationType.SWIGLU or case.bi:
                h = h.to(torch.bfloat16).float()  # FC1 output in bf16
            a = _activation(h, case.act)  # fp32
            if case.bi and not case.probs_on_input:
                a = a * pe  # batch-invariant: the probability at the activation
            a = a.to(torch.bfloat16)
            acc = _gemm(a, inp.fc2[e], exact, "FC2")
            if not case.bi and not case.probs_on_input:
                acc = acc * pe  # default: the probability on FC2's fp32 accumulator
            y[tok, slot] = acc.to(torch.bfloat16).float()
    if case.bi:
        return y.double().sum(dim=1).float()
    if exact:
        _require_exact(y.double().abs().sum(dim=1), _quantum(y), "top-k sum")
    total = torch.zeros(v, x.shape[1], dtype=torch.float32, device=x.device)
    for j in range(k):
        total = total + y[:, j]
    return total


# ---------------------------------------------------------------------------------------------
# Running and checking
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def launches(monkeypatch) -> List[dict]:
    """Records the launch config and grid of every ``_fused_moe_kernel`` launch."""
    log: List[dict] = []
    original = vfm._invoke_fused_moe_kernel

    def spy(*args, **kwargs):
        log.append(
            dict(
                config=dict(kwargs["config"]),
                grid=kwargs["grid_size"],
                n=args[1].shape[1],
                topk=kwargs["top_k"],
            )
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(vfm, "_invoke_fused_moe_kernel", spy)
    return log


def run(case: Case, inp: Inputs, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    kind, clamp, _ = ACTIVATIONS[case.act]
    kwargs = {}
    if case.probs_on_input:
        # Not a parameter on main: this raises TypeError there (see the module docstring).
        kwargs["apply_probs_on_input"] = True
    mode = set_batch_invariant_mode(True, backend="triton") if case.bi else contextlib.nullcontext()
    with mode:
        return vfm.vllm_fused_moe(
            inp.hidden,
            inp.probs,
            inp.fc1,
            inp.fc2,
            activation_type=kind,
            num_local_experts=inp.num_local,
            local_expert_start=inp.start,
            valid_tokens=torch.tensor([inp.valid], dtype=torch.int32, device=DEVICE),
            routing_map=inp.routing,
            out=out,
            num_tokens_hint=case.hint,
            activation_clamp_scale=clamp,
            **kwargs,
        )


def no_local_slot(inp: Inputs) -> torch.Tensor:
    routing = inp.routing[: inp.valid].long()
    local = (routing >= inp.start) & (routing < inp.start + inp.num_local)
    return ~local.any(dim=1)


def check_output(result: torch.Tensor, case: Case, inp: Inputs, what: str) -> None:
    """Live rows against the oracle; rows with no local slot exactly zero."""
    live = result[: inp.valid]
    if case.data == "int":
        spec = rounded_moe(inp, case, exact=True).to(live.dtype)
        ou.assert_exact(live, spec, what=f"{what}: output vs the rounding-point spec")
    else:
        reference = ou.fp64_reference(
            reference_moe,
            inp.hidden[: inp.valid],
            inp.probs[: inp.valid],
            inp.fc1,
            inp.fc2,
            inp.routing[: inp.valid].long(),
            act=case.act,
            start=inp.start,
            probs_on_input=case.probs_on_input,
        )
        trusted = rounded_moe(inp, case, exact=False).to(live.dtype)
        ou.assert_close_to_reference(live, reference, trusted, dim=-1, what=what)
    zero_rows = no_local_slot(inp)
    if bool(zero_rows.any()):
        ou.assert_exact(
            live[zero_rows], torch.zeros_like(live[zero_rows]), what=f"{what}: no-local-slot rows"
        )


def run_and_check(case: Case, launches: Optional[List[dict]] = None, seed: int = 0) -> Inputs:
    """Run one case with guarded, poisoned buffers and check every output and invariant."""
    g = ou.make_generator(ou.DEFAULT_SEED + seed, device=DEVICE)
    inp = make_inputs(case, g)
    before = {name: buf.tensor.clone() for name, buf in inp.buffers.items()}
    out_buf = None
    if case.out != "none":
        dtype = torch.float32 if case.out == "fp32" else torch.bfloat16
        # A whole BLOCK_SIZE_M tile of rows on either side.
        guard = 128 * case.hidden * dtype.itemsize
        out_buf = ou.guarded_empty(
            (inp.hidden.shape[0], case.hidden), dtype, DEVICE, guard_bytes=guard
        )
    if launches is not None:
        launches.clear()

    result = run(case, inp, out_buf.tensor if out_buf is not None else None)

    what = case.id()
    assert result.shape == (inp.hidden.shape[0], case.hidden), result.shape
    if out_buf is not None:
        assert result.data_ptr() == out_buf.tensor.data_ptr(), "out was not used"
        ou.check_guards(out=out_buf)
        ou.assert_poisoned(out_buf.tensor[inp.valid :], what=f"{what}: rows past valid_tokens")
    else:
        assert result.dtype == torch.float32
    check_output(result, case, inp, what)
    ou.check_guards(**inp.buffers)
    for name, buf in inp.buffers.items():
        ou.assert_exact(buf.tensor, before[name], what=f"{what}: input {name} is read-only")
    if launches is not None:
        expected = spec_config(case.config_m, case.num_local, case.bi)
        assert len(launches) == 2, launches
        for launch in launches:
            assert launch["config"] == expected, (launch["config"], expected)
    return inp


def tiles(inp: Inputs, block_m: int, block_n: int, n: int) -> int:
    """Tiles the kernel must visit: per-expert M-blocks times N-blocks."""
    routing = inp.routing[: inp.valid].long().flatten()
    local = routing[(routing >= inp.start) & (routing < inp.start + inp.num_local)]
    counts = torch.bincount(local - inp.start, minlength=inp.num_local)
    return int((-(-counts // block_m)).sum()) * -(-n // block_n)


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------

EXACT_CASES = [
    Case(data="int", act=SQRELU, **with_tuner(c))
    for c in ou.pairwise_cases(
        {
            "hint_el": list(TUNER),
            "tokens": ["bm-1", "bm", "bm+1", "2bm+1"],
            "topk": [1, 2, 8],
            "routing": ["one_expert", "random", "empty_experts", "block_counts", "masked"],
            "bi": [False, True],
            "hidden": [63, 64, 65, 127, 128, 129],
            "ffn": [63, 64, 65, 127, 128, 129],
            "ep": [1, 4],
            "out": ["fp32", "bf16", "none"],
            "routing_dtype": [torch.int64, torch.int32],
            "strided": [False, True],
        }
    )
]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", params(EXACT_CASES))
def test_exact_squared_relu(case, launches):
    """Squared ReLU on small-integer inputs equals the rounding-point spec bit for bit.

    Every launch-config row, per-expert counts around BLOCK_SIZE_M, both modes, H and F at the
    BK and BN tile edges. Any wrong token, expert, slot, column, K-tail term, probability or
    rounding point changes bits.
    """
    run_and_check(case, launches)


TOLERANCE_CASES = [
    Case(data="gauss", **with_tuner(c))
    for c in ou.pairwise_cases(
        {
            "act": [SWIGLU, SQRELU, CLAMP16, CLAMP_HALF],
            "bi": [False, True],
            "hint_el": list(TUNER),
            "topk": [1, 2, 8],
            "routing": ["random", "one_expert", "empty_experts", "block_counts", "masked"],
            "tokens": ["bm-1", "bm+1", "2bm+1"],
            "hidden": [64, 96, 129, 1025],
            "ffn": [64, 97, 128, 1025],
            "ep": [1, 4],
            "out": ["fp32", "bf16", "none"],
        }
    )
]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", params(TOLERANCE_CASES))
def test_matches_fp64_reference(case, launches):
    """Every activation and mode against the fp64 MoE, bounded by the trusted torch MoE."""
    run_and_check(case, launches)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("hint_mode", ["hint", "max_tokens"])
@pytest.mark.parametrize("m", [1, 32, 33, 64, 65, 96, 97, 128, 129, 512, 513])
def test_tuner_thresholds(m, hint_mode, launches):
    """The config switches at each threshold of the table; both sides give the spec bits.

    ``hint``: num_tokens_hint = m, m live tokens. ``max_tokens``: no hint, so the config follows
    max_tokens = m (with one padding row).
    """
    common = dict(
        data="int", num_local=2, ep=2, hidden=72, ffn=80, topk=(1, 2, 8)[m % 3], bi=m % 2 == 1
    )
    if hint_mode == "hint":
        case = Case(hint=m, tokens=m, pad=0, **common)
    else:
        case = Case(hint=None, tokens=max(m - 1, 0), pad=min(m, 1), **common)
    run_and_check(case, launches)


BLOCK_M_CASES = [
    Case(
        data="int",
        routing="one_expert",
        tokens=tokens,
        bi=bi,
        topk=(1, 2, 8)[i % 3],
        ep=2,
        hint=TUNER[name][0],
        num_local=TUNER[name][1],
    )
    for i, (name, tokens, bi) in enumerate(
        (name, tokens, bi)
        for name in TUNER
        for tokens in ("bm-1", "bm", "bm+1")
        for bi in (False, True)
    )
]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", params(BLOCK_M_CASES))
def test_block_m_boundaries(case, launches):
    """Every config the tuner can pick, in both modes, with one expert holding BM-1, BM and
    BM+1 rows: one M-tile partly padded, exactly full, and one row into a second tile."""
    run_and_check(case, launches)


GRID_STRIDE_CASES = [
    # hint 1: a BM16 grid of a few tiles; 300 tokens need many more.
    Case(data="int", hint=1, num_local=4, tokens=300, topk=2),
    # One hot expert with 2100 rows at BM16: 132 blocks, past the fill kernel's 128-block loop.
    Case(data="int", hint=16, num_local=4, tokens=2100, topk=1, routing="one_expert", ep=2),
    # GROUP_SIZE_M 16 over 46 M-tiles: groups of 16, 16 and a partial 14.
    Case(data="int", hint=513, num_local=2, tokens=2900, topk=2, ep=1, bi=True),
    Case(
        data="gauss",
        act=SWIGLU,
        hint=129,
        num_local=1,
        tokens=700,
        topk=1,
        ep=8,
        ffn=96,
        routing="one_expert",
    ),
    Case(data="gauss", act=CLAMP16, hint=32, num_local=8, tokens=500, topk=8, ep=1),
]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", params(GRID_STRIDE_CASES))
def test_grid_stride(case, launches):
    """More tiles than the hint-sized grid: the CTAs stride, and the result is unchanged."""
    inp = run_and_check(case, launches)
    fc1, fc2 = launches
    config = fc1["config"]
    need = tiles(inp, config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"], fc1["n"])
    assert fc1["grid"] < need, f"grid {fc1['grid']} covers all {need} tiles: no striding"


DEGENERATE_CASES = {
    "valid0": Case(data="int", tokens=0, pad=20),
    "valid0-bi-swiglu": Case(data="gauss", act=SWIGLU, bi=True, tokens=0, pad=20),
    "max_tokens0": Case(data="int", tokens=0, pad=0, out="none"),
    "max_tokens0-swiglu": Case(data="gauss", act=SWIGLU, tokens=0, pad=0, out="none"),
    "nonlocal_only": Case(data="int", ep=4, tokens=40, routing="nonlocal_only"),
    "all_masked": Case(data="gauss", act=SWIGLU, tokens=40, routing="all_masked"),
    "topk_eq_local": Case(data="int", num_local=4, topk=4, tokens=33, hint=33),
    "one_local_expert": Case(data="gauss", act=CLAMP16, num_local=1, ep=8, topk=2, tokens=50),
    "single_token": Case(data="gauss", act=SWIGLU, tokens=1, pad=0, hint=1, topk=8, ep=4),
}


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("name", list(DEGENERATE_CASES))
def test_degenerate(name, launches):
    """No live tokens, no rows at all, nothing local, every expert per token, one expert."""
    run_and_check(DEGENERATE_CASES[name], launches)


PRODUCTION_CASES = {
    # DeepSeek-V2-Lite: 64 experts, top-6, H 2048, F 1408, SwiGLU; EP 8.
    "deepseek-v2-lite-ep8": Case(
        act=SWIGLU, hidden=2048, ffn=1408, num_local=8, ep=8, topk=6, tokens=256, hint=256
    ),
    "deepseek-v2-lite-ep8-bi": Case(
        act=SWIGLU, bi=True, hidden=2048, ffn=1408, num_local=8, ep=8, topk=6, tokens=100, hint=8
    ),
    # Nemotron 3 Nano: 128 experts, top-6, H 2688, F 1856, squared ReLU; EP 8.
    "nemotron-nano-ep8-prefill": Case(
        act=SQRELU, hidden=2688, ffn=1856, num_local=16, ep=8, topk=6, tokens=1024, hint=1024
    ),
    "nemotron-nano-ep8-decode-clamp16": Case(
        act=CLAMP16, hidden=2688, ffn=1856, num_local=16, ep=8, topk=6, tokens=64, hint=64
    ),
    "nemotron-nano-ep8-bi": Case(
        act=SQRELU, bi=True, hidden=2688, ffn=1856, num_local=16, ep=8, topk=6, tokens=300, hint=300
    ),
    # Qwen3-30B-A3B: 128 experts, top-8, H 2048, F 768, SwiGLU; EP 8.
    "qwen3-30b-a3b-ep8": Case(
        act=SWIGLU, hidden=2048, ffn=768, num_local=16, ep=8, topk=8, tokens=600, hint=600
    ),
    # Mixtral 8x7B: 8 experts, top-2, H 4096, F 14336, SwiGLU; EP 8 (one local expert).
    "mixtral-8x7b-ep8": Case(
        act=SWIGLU, hidden=4096, ffn=14336, num_local=1, ep=8, topk=2, tokens=96, hint=96
    ),
}


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("name", list(PRODUCTION_CASES))
def test_production_sizes(name, launches):
    """Deployed model shapes (hidden past _moe_sum's 1024-wide block, F past 1024)."""
    run_and_check(PRODUCTION_CASES[name], launches)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("act", [SWIGLU, SQRELU, CLAMP16])
@pytest.mark.parametrize("bi", [False, True])
def test_nan_hidden_row_propagates_to_its_token_only(act, bi):
    """NaN in a live token's hidden row: that row is NaN, every other row is unchanged.

    The token's other copy with no local slot still gets a zero row.
    """
    case = Case(act=act, bi=bi, ep=4, topk=2, tokens=40, hint=48)
    g = ou.make_generator(device=DEVICE)
    inp = make_inputs(case, g)
    routing = inp.routing[: inp.valid].long()
    local = (routing >= inp.start) & (routing < inp.start + inp.num_local)
    with_local = torch.nonzero(local.any(dim=1)).flatten()
    without_local = torch.nonzero(~local.any(dim=1)).flatten()
    assert with_local.numel() and without_local.numel(), "routing lacks the needed tokens"
    inp.hidden[int(with_local[0])] = float("nan")
    inp.hidden[int(without_local[0])] = float("nan")
    out = ou.guarded_empty((inp.hidden.shape[0], case.hidden), torch.float32, DEVICE)
    result = run(case, inp, out.tensor)
    ou.check_guards(out=out)
    check_output(result, case, inp, case.id())
    assert bool(torch.isnan(result[int(with_local[0])]).all()), "NaN token's row is not NaN"


PROBS_ON_INPUT_CASES = [
    Case(probs_on_input=True, topk=1, **with_tuner(c))
    for c in ou.pairwise_cases(
        {
            "act": [SWIGLU, SQRELU, CLAMP16],
            "bi": [False, True],
            "hint_el": ["bm16", "bm32-bn128", "bm64-w8", "bm128-g16"],
            "tokens": ["bm-1", "2bm+1"],
            "routing": ["random", "block_counts", "masked"],
            "ep": [1, 4],
        }
    )
] + [
    Case(data="int", probs_on_input=True, topk=1, bi=bi, hint=h, num_local=4, ep=2, ffn=65)
    for bi in (False, True)
    for h in (16, 112)
]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", params(PROBS_ON_INPUT_CASES))
def test_probs_on_input(case, launches):
    """moe_apply_probs_on_input: the probability scales the expert input (bf16), not its output.

    Fails on main: ``vllm_fused_moe`` has no ``apply_probs_on_input`` (TypeError), and the
    layer silently ignores ``moe_apply_probs_on_input`` (audit finding D2). Passes with #8566.
    Top-k 1, as training requires.
    """
    run_and_check(case, launches)


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.parametrize("act", [SWIGLU, SQRELU, CLAMP16])
@pytest.mark.parametrize("num_local", [2, 8])
def test_batch_invariance_bitwise(act, num_local, launches):
    """Batch-invariant mode: a token's output bits do not depend on the batch around it.

    Six probe tokens run alone and inside batches whose size, padding and hint select every
    row of the launch-config table (BM 16 to 128, BN 64/128, 4/8 warps, GROUP_SIZE_M 1/16, a
    striding grid), at different positions. BLOCK_SIZE_K is pinned, and the rest of the tile
    shape must not change any accumulation. The alone run also matches the fp64 reference.
    K (1000 and 520, with partial K-blocks) is long enough that a changed fp32 summation order
    flips bf16 roundings among the probes' ~12k FC1 outputs: with K 96, a CPU run of an
    unpinned BLOCK_SIZE_K kept every probe bit.
    """
    probes, topk, hidden, ffn, ep = 6, 2, 1000, 520, 4
    base = Case(
        act=act,
        bi=True,
        topk=topk,
        num_local=num_local,
        ep=ep,
        hidden=hidden,
        ffn=ffn,
        tokens=probes,
        pad=0,
        hint=None,
    )
    g = ou.make_generator(device=DEVICE)
    probe = make_inputs(base, g)
    px, pp, pr = (t[:probes].clone() for t in (probe.hidden, probe.probs, probe.routing))
    alone = run(base, probe)[:probes].clone()
    check_output(alone, base, probe, "probes alone")

    configs = set()
    # (live tokens, padding rows, num_tokens_hint)
    for batch, pad, hint in [
        (6, 0, 1024),
        (20, 5, 20),
        (48, 0, 48),
        (80, 16, 80),
        (112, 0, 112),
        (200, 9, 200),
        (300, 0, 300),
        (520, 0, 520),
        (700, 3, 700),
        (700, 0, 1),
        (1100, 0, 1100),
    ]:
        case = dataclasses.replace(base, tokens=batch, pad=pad, hint=hint)
        filler = make_inputs(case, ou.make_generator(ou.DEFAULT_SEED + batch + hint, DEVICE))
        filler.fc1, filler.fc2 = probe.fc1, probe.fc2  # the same layer
        positions = torch.randperm(batch, generator=g, device=DEVICE)[:probes]
        filler.hidden[positions] = px
        filler.probs[positions] = pp
        filler.routing[positions] = pr.to(filler.routing.dtype)
        launches.clear()
        out = run(case, filler)
        ou.assert_exact(
            out[positions], alone, what=f"probe rows in a batch of {batch} (hint {hint})"
        )
        for launch in launches:
            assert launch["config"]["BLOCK_SIZE_K"] == 64, launch
            configs.add(tuple(sorted(launch["config"].items())))
    assert len(configs) >= 6, f"only {len(configs)} distinct launch configs exercised"


@contextlib.contextmanager
def poisoned_pool(max_tokens, topk, fc1_out, hidden, num_local):
    """The persistent pool, NaN / INT_MIN poisoned and guarded."""
    vfm.VllmFusedMoeBuffers.allocate_buffers(
        max_tokens, topk, fc1_out, hidden, num_local, device=DEVICE
    )
    guarded = {
        name: ou.guarded_empty(buf.shape, buf.dtype, DEVICE)
        for name, buf in vfm.VllmFusedMoeBuffers._buffers.items()
    }
    vfm.VllmFusedMoeBuffers._buffers = {name: gb.tensor for name, gb in guarded.items()}
    try:
        yield guarded
    finally:
        vfm.VllmFusedMoeBuffers._delete_buffers()


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("bi", [False, True])
def test_pooled_buffers(bi):
    """The shared buffer pool: poisoned contents never leak, nor a previous layer's.

    Layer A (SwiGLU) then layer B (squared ReLU, fewer tokens, other experts on this rank) use
    one poisoned pool with experts on other ranks; then a call too large for the pool falls
    back to fresh buffers. The pool's unrequested tails stay poisoned and its guards intact.
    """
    hidden, ffn, topk, num_local, max_tokens = 128, 96, 4, 4, 160
    layer_a = Case(
        act=SWIGLU,
        bi=bi,
        tokens=150,
        pad=10,
        hint=150,
        topk=topk,
        num_local=num_local,
        ep=4,
        hidden=hidden,
        ffn=ffn,
        out="none",
    )
    layer_b = dataclasses.replace(layer_a, act=SQRELU, tokens=37, pad=123, hint=37)
    too_big = dataclasses.replace(layer_a, act=CLAMP16, tokens=300, pad=0, hint=300)
    with poisoned_pool(max_tokens, topk, 2 * ffn, hidden, num_local) as pool:
        sorted_ids_used = expert_ids_used = 0
        for i, case in enumerate((layer_a, layer_b)):
            inp = make_inputs(case, ou.make_generator(ou.DEFAULT_SEED + i, DEVICE))
            result = run(case, inp)
            check_output(result, case, inp, f"pooled layer {'AB'[i]} ({case.act})")
            assert result.data_ptr() == pool["moe_sum_out"].tensor.data_ptr()
            ou.assert_poisoned(result[inp.valid :], what="pooled output rows past valid")
            result.fill_(float("nan"))  # the pool's output view: re-poison for the next layer
            bm = spec_config(case.hint, num_local, bi)["BLOCK_SIZE_M"]
            max_sorted = case.max_tokens * topk + bm * (num_local + 1)
            sorted_ids_used = max(sorted_ids_used, max_sorted)
            expert_ids_used = max(expert_ids_used, -(-max_sorted // bm))
        ou.assert_poisoned(
            pool["sorted_token_ids"].tensor[sorted_ids_used:], what="sorted_token_ids tail"
        )
        ou.assert_poisoned(pool["expert_ids"].tensor[expert_ids_used:], what="expert_ids tail")
        inp = make_inputs(too_big, ou.make_generator(ou.DEFAULT_SEED + 2, DEVICE))
        check_output(run(too_big, inp), too_big, inp, "call larger than the pool")
        ou.check_guards(**pool)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "act,bi", [(SWIGLU, False), (SQRELU, False), (CLAMP16, False), (SWIGLU, True), (SQRELU, True)]
)
def test_cuda_graph_replay(act, bi):
    """One capture, replayed with new inputs, routing and valid_tokens, against the fp64 MoE.

    valid_tokens takes values on both sides of BLOCK_SIZE_M and above the captured hint, so
    replays both shrink and grow the tile count; the pool and the output start poisoned.
    """
    hidden, ffn, topk, num_local, max_tokens, hint = 128, 96, 2, 4, 128, 48
    base = Case(
        act=act,
        bi=bi,
        topk=topk,
        num_local=num_local,
        ep=4,
        hidden=hidden,
        ffn=ffn,
        tokens=max_tokens,
        pad=0,
        hint=hint,
    )
    fc1_out = 2 * ffn if act == SWIGLU else ffn
    with poisoned_pool(max_tokens, topk, fc1_out, hidden, num_local) as pool:
        static = make_inputs(base, ou.make_generator(device=DEVICE))
        valid_t = torch.tensor([max_tokens], dtype=torch.int32, device=DEVICE)
        out = ou.guarded_empty((max_tokens, hidden), torch.float32, DEVICE)
        kind, clamp, _ = ACTIVATIONS[act]

        def call():
            return vfm.vllm_fused_moe(
                static.hidden,
                static.probs,
                static.fc1,
                static.fc2,
                activation_type=kind,
                num_local_experts=num_local,
                local_expert_start=static.start,
                valid_tokens=valid_t,
                routing_map=static.routing,
                out=out.tensor,
                num_tokens_hint=hint,
                activation_clamp_scale=clamp,
            )

        mode = set_batch_invariant_mode(True, backend="triton") if bi else contextlib.nullcontext()
        with mode:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                call()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()

        for i, valid in enumerate([128, 0, 1, 31, 32, 33, 47, 97, 64]):
            case = dataclasses.replace(base, tokens=valid, pad=max_tokens - valid)
            fresh = make_inputs(case, ou.make_generator(ou.DEFAULT_SEED + 1 + i, DEVICE))
            for name in ("hidden", "probs", "routing"):
                getattr(static, name).copy_(getattr(fresh, name))
            static.valid = valid
            valid_t.fill_(valid)
            ou.poison_(out.tensor)
            graph.replay()
            torch.cuda.synchronize()
            check_output(out.tensor, case, static, f"replay {i} (valid {valid})")
            ou.assert_poisoned(out.tensor[valid:], what=f"replay {i}: rows past valid")
            ou.check_guards(out=out, **pool)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("act", [SQRELU, SWIGLU])
@pytest.mark.parametrize("pairs", ou.int32_overflow_params(8192))
def test_int32_offsets(act, pairs):
    """FC1's output rows (8192 wide) at element offsets past 2**31; sampled rows checked."""
    ou.require_free_memory(14 * 2**30)
    ffn = 8192 if act == SQRELU else 4096
    case = Case(
        act=act,
        topk=1,
        num_local=4,
        hidden=128,
        ffn=ffn,
        tokens=pairs,
        pad=0,
        hint=pairs,
        out="none",
    )
    g = ou.make_generator(device=DEVICE)
    inp = make_inputs(case, g)
    result = run(case, inp)
    assert not bool(torch.isnan(result).any()), "NaN in the output: a row was read or lost"
    limit = 2**31 // 8192
    rows = (
        torch.cat(
            [
                torch.tensor([0, 1, limit - 2, limit - 1, pairs - 2, pairs - 1], device=DEVICE),
                torch.randint(0, pairs, (58,), generator=g, device=DEVICE),
            ]
        )
        .clamp(max=pairs - 1)
        .unique()
    )
    sample = dataclasses.replace(
        inp,
        hidden=inp.hidden[rows],
        probs=inp.probs[rows],
        routing=inp.routing[rows],
        valid=rows.numel(),
    )
    check_output(result[rows], case, sample, f"sampled rows of {pairs}")


@pytest.mark.sanitizer
@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "case",
    params(
        [
            Case(data="int", topk=2, num_local=4, ep=4, tokens=17, hidden=65, ffn=63, hint=16),
            Case(act=SWIGLU, topk=2, num_local=4, tokens=33, hidden=64, ffn=72, hint=40),
        ]
    ),
)
def test_sanitizer(case):
    """Small cases for compute-sanitizer memcheck (also run normally)."""
    run_and_check(case)
