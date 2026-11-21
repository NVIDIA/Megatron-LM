# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_moe_sum_kernel`` (``megatron/core/inference/moe/vllm_fused_moe.py:577``), launched
through its launcher ``_moe_sum`` (``:644``), the top-k reduction of the vLLM backend.

Contract
--------
Written from the call site (``vllm_fused_moe``: the reduction of FC2's per-slot rows, after the
probabilities were applied in FC2's epilogue or at the activation) and the launcher's docstring,
before reading the kernel body.

* **Computes**, for each token ``t < valid_tokens``::

      out[t, :] = sum over slots j with start <= routing_map[t, j] < start + E_local of
                  input[t * topk + j, :] * (probs[t, j] if apply_weights else 1)

  ``acc_fp64=False``: accumulated in fp32 (the order of the slots is not part of the contract).
  ``acc_fp64=True``: accumulated in fp64 (each product of a bf16 value and an fp32 weight is exact
  there) and rounded once to fp32. The fp32 value is stored in ``out``'s dtype (fp32, or bf16
  rounded to nearest even). A token without a local slot gets an exact ``+0.0`` row.
* **Inputs**: ``input`` ``[max_tokens * topk, K]`` bf16, contiguous; ``probs`` ``[max_tokens,
  topk]`` fp32, contiguous; ``routing_map`` ``[max_tokens, topk]`` int32 or int64, contiguous,
  global ids or ``-1``; ``valid_tokens`` an int32 device scalar ``<= max_tokens``; ``out`` an
  optional ``[max_tokens, K]`` fp32 or bf16 buffer (``None``: the pool's ``moe_sum_out`` view,
  or a fresh fp32 tensor).
* **Outputs**: rows ``< valid_tokens`` of ``out``; rows past it are not written. The input rows
  and probabilities of non-local or ``-1`` slots, and every row of tokens past ``valid_tokens``,
  are never read. The inputs are read-only.
* **``valid_tokens > max_tokens``** (finding F1): only the ``max_tokens`` rows exist, so only
  they are summed and written. ``main`` reads and writes past them; the case is #8563's
  regression test ``TestVllmFusedMoe::test_valid_tokens_past_input_rows``
  (``tests/unit_tests/inference/test_vllm_fused_moe.py``).
* **Special values**: IEEE arithmetic. An Inf or NaN in a summed slot propagates; ``Inf - Inf``
  gives NaN; a sum past fp32's range is Inf; a sum of ``-0.0`` slots is ``+0.0``.
* **Batch-invariant mode** (``vllm_fused_moe``): the sum is the fp64 one, as training's
  within-rank accumulation (``test_batch_invariant_mode_sums_in_fp64``).
* **Tolerance class**: exact by construction (dyadic inputs and probabilities ``j / 16``: every
  product and partial sum is exact in fp32, checked per case), so the output equals the fp64 sum
  rounded once to ``out``'s dtype. Gaussian inputs: floating point, per row at most 2x the error
  of a trusted torch reduction on the same device plus one ulp (``assert_close_to_reference``).

Variants and boundaries (from the kernel and launcher, after the contract)
--------------------------------------------------------------------------
* Constexprs ``APPLY_WEIGHTS`` x ``ACC_FP64``: all four. Production uses ``(False, False)``
  (default mode) and ``(False, True)`` (batch-invariant mode); ``_moe_sum``'s default
  ``apply_weights=True`` gives the other two, which the entry never compiles.
* ``BLOCK_M = _get_num_sms(device)`` programs, each striding over the tokens: the SM count is
  monkeypatched to 1 and 3 (many tokens per program) and kept at the device's (132 on H100);
  tokens 0, 1, ``SMs - 1``, ``SMs``, ``SMs + 1`` and ``2 SMs + 1``.
* ``BLOCK_K = min(next_pow2(K), 1024)`` and ``NUM_K_BLOCKS = ceil(K / BLOCK_K)``: K 1, 2, 31, 32,
  33, 1023, 1024, 1025 and 2049 (one to three K blocks, partial ones), and production 2048, 2688,
  4096 and 7168.
* Constexprs ``topk`` 1, 2, 3, 6, 8 and ``E_local`` 1, 2, 7, 16, 32, 128; ranks first, middle and
  last of EP 1, 4 and 8; int32 and int64 maps; outputs fp32, bf16, allocated and pooled.
* int32 limits (``large``): input element offsets ``(t * topk + j) * K`` past 2**31 (top-8,
  K 4096: 65537 tokens).

Why a direct test
-----------------
Through ``vllm_fused_moe`` only ``APPLY_WEIGHTS=False`` is reached, the SM count is the device's,
and the inputs' magnitudes never let fp32 and fp64 accumulation differ, so the accumulation
precision of either mode is invisible there (mutant o3-msum-004 survived). Here every variant is
launched, the precision is pinned by sums that fp32 rounds and fp64 does not, and the
batch-invariant dispatch is checked end to end on such a sum.

Hardware: H100 (no tensor cores; nothing architecture-specific).
"""

import dataclasses
import importlib
import math
from typing import List, Optional, Union

import pytest
import torch

from megatron.core.inference.moe.fused_moe import ActivationType
from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_vllm import align_reference as ar

vfm = importlib.import_module("megatron.core.inference.moe.vllm_fused_moe")

DEVICE = torch.device("cuda")

VARIANTS = [(False, False), (False, True), (True, False), (True, True)]  # (apply, fp64)


@dataclasses.dataclass(frozen=True)
class Case:
    apply_weights: bool = False
    acc_fp64: bool = False
    data: str = "dyadic"  # "dyadic": exact spec; "gauss": fp64 reference + trusted bound
    topk: int = 2
    hidden: int = 128  # K
    tokens: Union[int, str] = 40  # live tokens, or "sms-1" / "sms" / "sms+1" / "2sms+1"
    pad: int = 3  # rows past valid_tokens
    num_sms: Optional[int] = 3  # _get_num_sms, i.e. BLOCK_M; None: the device's
    num_local: int = 4
    ep: int = 1
    rank: str = "first"
    kind: str = "random"
    out: str = "fp32"  # "fp32" / "bf16": a guarded buffer; "none": allocated; "pool": pooled
    dtype: torch.dtype = torch.int64

    @property
    def start(self) -> int:
        index = {"first": 0, "middle": self.ep // 2, "last": self.ep - 1}[self.rank]
        return index * self.num_local

    def valid(self, sms: int) -> int:
        if isinstance(self.tokens, int):
            return self.tokens
        return {"sms-1": sms - 1, "sms": sms, "sms+1": sms + 1, "2sms+1": 2 * sms + 1}[self.tokens]

    def id(self) -> str:
        dtype = "i32" if self.dtype == torch.int32 else "i64"
        return (
            f"{'w' if self.apply_weights else 'now'}-{'fp64' if self.acc_fp64 else 'fp32'}"
            f"-{self.data}-k{self.topk}-h{self.hidden}-t{self.tokens}-pad{self.pad}"
            f"-sms{self.num_sms or 'dev'}-el{self.num_local}-ep{self.ep}{self.rank}-{self.kind}"
            f"-{self.out}-{dtype}"
        )


def with_variant(factors: dict) -> dict:
    factors = dict(factors)
    factors["apply_weights"], factors["acc_fp64"] = factors.pop("variant")
    return factors


@dataclasses.dataclass
class Inputs:
    input: ou.GuardedBuffer
    probs: ou.GuardedBuffer
    routing: ou.GuardedBuffer
    valid: int
    local: torch.Tensor  # [valid, topk] bool: the slots that are summed


def make_inputs(case: Case, sms: int, g: torch.Generator) -> Inputs:
    valid, k, h = case.valid(sms), case.topk, case.hidden
    max_tokens = valid + case.pad
    live = ar.make_routing(
        case.kind, valid, k, case.ep * case.num_local, case.start, case.num_local, g, DEVICE
    )
    routing = ar.full_routing(live, max_tokens, case.start, case.num_local, case.dtype, g)
    local = (live >= case.start) & (live < case.start + case.num_local)
    if case.data == "dyadic":
        x = ou.dyadic_tensor(
            (max_tokens, k, h), mantissa_bits=4, exponents=(-2, 2), device=DEVICE, generator=g
        )
        p = torch.randint(1, 17, (max_tokens, k), generator=g, device=DEVICE) / 16.0
    else:
        x = torch.randn(max_tokens, k, h, generator=g, device=DEVICE).to(torch.bfloat16)
        p = torch.rand(max_tokens, k, generator=g, device=DEVICE)
    p = p.float()
    # Slots that must not be read: non-local or -1 slots of live tokens, and every slot of the
    # tokens past valid_tokens (whose routing is local, so a read of them adds a NaN term).
    unread = torch.ones(max_tokens, k, dtype=torch.bool, device=DEVICE)
    unread[:valid] = ~local
    x[unread] = float("nan")
    p[unread] = float("nan")
    return Inputs(
        input=ou.guarded_like(x.view(max_tokens * k, h)),
        probs=ou.guarded_like(p),
        routing=ou.guarded_like(routing),
        valid=valid,
        local=local,
    )


def reference_sum(x: torch.Tensor, p: torch.Tensor, local: torch.Tensor, apply_weights: bool):
    """fp64 sum over the local slots, slot by slot from +0.0 (no reduction kernel involved)."""
    valid, k = local.shape
    total = torch.zeros(valid, x.shape[-1], dtype=x.dtype, device=x.device)
    for j in range(k):
        term = x[:, j] * p[:, j, None] if apply_weights else x[:, j]
        total = total + torch.where(local[:, j, None], term, torch.zeros_like(term))
    return total


def trusted_sum(x, p, local, apply_weights: bool, acc_fp64: bool) -> torch.Tensor:
    """torch on the same device with the contract's precision: fp32 slot by slot, or fp64."""
    if acc_fp64:
        return reference_sum(x.double(), p.double(), local, apply_weights).float()
    return reference_sum(x.float(), p.float(), local, apply_weights)


def _quantum(t: torch.Tensor) -> float:
    """The largest power of two dividing every finite nonzero element (inf if there is none)."""
    v = t.detach().double().abs().flatten()
    v = v[(v != 0) & torch.isfinite(v)]
    if v.numel() == 0:
        return math.inf
    mantissa, exponent = torch.frexp(v)
    integer = (mantissa * 2.0**53).to(torch.int64)
    lowest_bit = torch.log2((integer & -integer).double()).round()
    return 2.0 ** float((exponent.double() - 53 + lowest_bit).min())


def require_exact_in_fp32(x, p, local, apply_weights: bool) -> None:
    """Test precondition: every product and partial sum of the case is exact in fp32."""
    terms = reference_sum(x.double().abs(), p.double().abs(), local, apply_weights)
    quantum = _quantum(x.double()) * (_quantum(p.double()) if apply_weights else 1.0)
    if terms.numel() and float(terms.max()) / quantum > 2.0**24:
        raise ValueError("test precondition: the dyadic sums are not exact in fp32")


def set_num_sms(monkeypatch, n: Optional[int]) -> int:
    """Make ``_moe_sum`` launch ``n`` programs (its ``BLOCK_M``); None keeps the device's count."""
    if n is not None:
        monkeypatch.setattr(vfm, "_get_num_sms", lambda device: n)
    return vfm._get_num_sms(DEVICE)


def run_case(case: Case, monkeypatch, seed: int = 0) -> None:
    """Run ``_moe_sum`` on ``case`` with guarded, poisoned buffers and check every output."""
    sms = set_num_sms(monkeypatch, case.num_sms)
    g = ou.make_generator(ou.DEFAULT_SEED + seed, device=DEVICE)
    inp = make_inputs(case, sms, g)
    max_tokens = inp.probs.tensor.shape[0]
    out_buf = pool = None
    if case.out in ("fp32", "bf16"):
        dtype = torch.float32 if case.out == "fp32" else torch.bfloat16
        out_buf = ou.guarded_empty((max_tokens, case.hidden), dtype, DEVICE)
    if case.out == "pool":
        # A pool longer than the request: its tail must stay poisoned.
        pool = ou.guarded_empty((max_tokens * case.hidden + 1000,), torch.float32, DEVICE)
        monkeypatch.setattr(vfm.VllmFusedMoeBuffers, "_buffers", {"moe_sum_out": pool.tensor})
    else:
        monkeypatch.setattr(vfm.VllmFusedMoeBuffers, "_buffers", None)
    buffers = dict(input=inp.input, probs=inp.probs, routing=inp.routing)
    before = {name: buf.tensor.clone() for name, buf in buffers.items()}
    valid_t = torch.tensor([inp.valid], dtype=torch.int32, device=DEVICE)

    result = vfm._moe_sum(
        inp.input.tensor,
        inp.probs.tensor,
        max_tokens,
        case.topk,
        case.hidden,
        valid_t,
        inp.routing.tensor,
        case.start,
        case.num_local,
        out=None if out_buf is None else out_buf.tensor,
        apply_weights=case.apply_weights,
        acc_fp64=case.acc_fp64,
    )

    check_result(result, case, inp, max_tokens, out_buf, pool)
    for name, buf in buffers.items():
        ou.assert_exact(buf.tensor, before[name], what=f"{case.id()}: {name} is read-only")
    ou.check_guards(**buffers)
    ou.assert_exact(valid_t, torch.tensor([inp.valid]), what="valid_tokens is read-only")


def check_result(result, case: Case, inp: Inputs, max_tokens: int, out_buf, pool) -> None:
    what = case.id()
    assert result.shape == (max_tokens, case.hidden), (what, result.shape)
    if out_buf is not None:
        assert result.data_ptr() == out_buf.tensor.data_ptr(), f"{what}: out was not used"
        ou.check_guards(out=out_buf)
    else:
        assert result.dtype == torch.float32, f"{what}: allocated output is {result.dtype}"
    if pool is not None:
        assert result.data_ptr() == pool.tensor.data_ptr(), f"{what}: the pool was not used"
        ou.assert_poisoned(pool.tensor[max_tokens * case.hidden :], what=f"{what}: pool tail")
        ou.check_guards(pool=pool)
    if out_buf is not None or pool is not None:
        ou.assert_poisoned(result[inp.valid :], what=f"{what}: rows past valid_tokens")

    v, k = inp.valid, case.topk
    x = inp.input.tensor.view(max_tokens, k, case.hidden)[:v]
    p = inp.probs.tensor[:v]
    live = result[:v]
    if case.data == "dyadic":
        require_exact_in_fp32(x, p, inp.local, case.apply_weights)
        spec = ou.fp64_reference(reference_sum, x, p, inp.local, case.apply_weights)
        ou.assert_exact(live, spec.float().to(live.dtype), what=f"{what}: vs the exact sum")
    else:
        reference = ou.fp64_reference(reference_sum, x, p, inp.local, case.apply_weights)
        trusted = trusted_sum(x, p, inp.local, case.apply_weights, case.acc_fp64)
        ou.assert_close_to_reference(live, reference, trusted.to(live.dtype), dim=-1, what=what)
    no_local = ~inp.local.any(dim=1)
    if bool(no_local.any()):
        zeros = torch.zeros_like(live[no_local])
        ou.assert_exact(live[no_local], zeros, what=f"{what}: rows without a local slot")


KINDS = ["random", "edges", "nonlocal", "masked", "ends_empty", "one_hot"]

EXACT_CASES = [
    Case(data="dyadic", **with_variant(c))
    for c in ou.pairwise_cases(
        {
            "variant": VARIANTS,
            "hidden": [1, 2, 31, 32, 33, 1023, 1024, 1025, 2049],
            "tokens": [0, 1, "sms-1", "sms", "sms+1", "2sms+1"],
            "topk": [1, 2, 3, 6, 8],
            "num_sms": [1, 3, None],
            "out": ["fp32", "bf16", "none", "pool"],
            "num_local": [1, 2, 7, 128],
            "kind": KINDS,
            "ep": [1, 4],
            "rank": ["first", "middle", "last"],
            "dtype": [torch.int64, torch.int32],
        }
    )
]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", [pytest.param(c, id=c.id()) for c in EXACT_CASES])
def test_exact_dyadic(case, monkeypatch):
    """Every variant, K-block count, SM count and slot locality, bit for bit."""
    run_case(case, monkeypatch)


TOLERANCE_CASES = [
    Case(data="gauss", **with_variant(c))
    for c in ou.pairwise_cases(
        {
            "variant": VARIANTS,
            "topk": [1, 6, 8],
            "hidden": [64, 1025, 2688],
            "tokens": [1, "sms+1"],
            "num_sms": [None, 2],
            "out": ["fp32", "bf16"],
            "kind": ["random", "edges"],
        }
    )
]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", [pytest.param(c, id=c.id()) for c in TOLERANCE_CASES])
def test_matches_fp64_reference(case, monkeypatch):
    """Gaussian inputs and probabilities, bounded by torch's reduction at the same precision."""
    run_case(case, monkeypatch)


# Per column, three local slots whose fp32 sum (in slot order) rounds and whose fp64 sum does
# not: 2**25 + 1 - 2**25 is 0 in fp32 and 1 in fp64; 2**24 + 1 + 1 is 2**24 in fp32 and 2**24 + 2
# (exact in fp32) in fp64; the third column has no rounding at all.
FP64_COLUMNS = [(2.0**25, 1.0, -(2.0**25)), (2.0**24, 1.0, 1.0), (3.0, 0.5, 0.25)]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("apply_weights", [False, True])
def test_fp64_accumulation_is_exact(apply_weights, monkeypatch):
    """``acc_fp64``: the sum is the exact one rounded once to fp32, where fp32 would round."""
    tokens, k = 5, 3
    x = torch.tensor(FP64_COLUMNS, dtype=torch.float64).t()  # [slot, column]
    x = x.unsqueeze(0).expand(tokens, k, len(FP64_COLUMNS)).to(torch.bfloat16).to(DEVICE)
    p = torch.ones(tokens, k, device=DEVICE)
    routing = torch.tensor([[0, 1, 2]] * tokens, dtype=torch.int64, device=DEVICE)
    out = ou.guarded_empty((tokens, len(FP64_COLUMNS)), torch.float32, DEVICE)
    vfm._moe_sum(
        x.reshape(tokens * k, -1).contiguous(),
        p,
        tokens,
        k,
        len(FP64_COLUMNS),
        torch.tensor([tokens], dtype=torch.int32, device=DEVICE),
        routing,
        0,
        3,
        out=out.tensor,
        apply_weights=apply_weights,
        acc_fp64=True,
    )
    exact = torch.tensor([sum(c) for c in FP64_COLUMNS], dtype=torch.float64).float()
    ou.assert_exact(out.tensor, exact.expand(tokens, -1).to(DEVICE), what="fp64 sums")
    ou.check_guards(out=out)


# (slot values, slots that are local): the contract's IEEE results.
SPECIAL_TOKENS = [
    ((math.inf, 1.0), (True, True)),  # inf
    ((math.inf, -math.inf), (True, True)),  # inf - inf = nan
    ((math.nan, 1.0), (True, True)),  # nan propagates
    ((3.3895313892515355e38, 3.3895313892515355e38), (True, True)),  # past fp32's range: inf
    ((-3.3895313892515355e38, -3.3895313892515355e38), (True, True)),  # -inf
    ((-0.0, -0.0), (True, True)),  # +0.0
    ((1.5, math.nan), (True, False)),  # the NaN slot is not local: never read
    ((math.inf, -2.0), (False, True)),  # the inf slot is not local: never read
]


def assert_exact_nan_aware(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    """Bitwise equal, except that a NaN may have any payload (the same positions must be NaN)."""
    expected = expected.to(actual.device)
    nan_actual, nan_expected = torch.isnan(actual), torch.isnan(expected)
    ou.assert_exact(nan_actual, nan_expected, what=f"{what}: NaN positions")
    zero = torch.zeros((), dtype=actual.dtype, device=actual.device)
    ou.assert_exact(
        torch.where(nan_actual, zero, actual), torch.where(nan_expected, zero, expected), what=what
    )


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("apply_weights,acc_fp64", VARIANTS)
def test_special_values(apply_weights, acc_fp64, out_dtype, monkeypatch):
    """Inf, NaN, overflow past fp32 and -0.0 follow IEEE; non-local slots are never read."""
    k, hidden = 2, 40
    tokens = len(SPECIAL_TOKENS)
    values = torch.tensor([v for v, _ in SPECIAL_TOKENS], dtype=torch.float64)
    is_local = torch.tensor([m for _, m in SPECIAL_TOKENS], device=DEVICE)
    x = values[:, :, None].expand(tokens, k, hidden).to(torch.bfloat16).to(DEVICE)
    p = torch.ones(tokens, k, device=DEVICE)
    p[~is_local] = float("nan")
    routing = torch.where(is_local, 1, 7).to(torch.int64)  # local slice [0, 4) on rank 0 of 2
    out = ou.guarded_empty((tokens, hidden), out_dtype, DEVICE)
    vfm._moe_sum(
        x.reshape(tokens * k, hidden).contiguous(),
        p,
        tokens,
        k,
        hidden,
        torch.tensor([tokens], dtype=torch.int32, device=DEVICE),
        routing,
        0,
        4,
        out=out.tensor,
        apply_weights=apply_weights,
        acc_fp64=acc_fp64,
    )
    expected = reference_sum(x.double(), p.double(), is_local, apply_weights)
    assert_exact_nan_aware(out.tensor, expected.float().to(out_dtype), "special values")
    ou.check_guards(out=out)


@ou.requires_sm(ou.HOPPER)
def test_batch_invariant_mode_sums_in_fp64():
    """``vllm_fused_moe`` in batch-invariant mode reduces the top-k slots in fp64.

    One token, three local slots whose FC2 rows are exact powers of two: column 0 is
    ``2**25 + 1 - 2**25`` and column 1 ``2**24 + 1 + 1``. Every GEMM and activation is exact
    (``x = e_0``; FC1 gives ``h = c_e`` in column 0, squared ReLU ``c_e**2`` at probability 1; FC2
    scales it by ``d``), so only the top-k sum can round: the contract's fp64 sum gives 1 and
    ``2**24 + 2``; an fp32 sum in slot order gives 0 and ``2**24``.
    """
    hidden = ffn = 64
    c = (2.0**6, 1.0, 2.0**6)  # FC1: h = c_e, so the activation is c_e**2 = 2**12, 1, 2**12
    d0 = (2.0**13, 1.0, -(2.0**13))  # FC2 column 0: y = 2**25, 1, -2**25
    d1 = (2.0**12, 1.0, 2.0**-12)  # FC2 column 1: y = 2**24, 1, 1
    x = torch.zeros(1, hidden, dtype=torch.bfloat16, device=DEVICE)
    x[0, 0] = 1
    fc1 = torch.zeros(3, ffn, hidden, dtype=torch.bfloat16, device=DEVICE)
    fc2 = torch.zeros(3, hidden, ffn, dtype=torch.bfloat16, device=DEVICE)
    for e in range(3):
        fc1[e, 0, 0], fc2[e, 0, 0], fc2[e, 1, 0] = c[e], d0[e], d1[e]
    out = ou.guarded_empty((1, hidden), torch.float32, DEVICE)
    with set_batch_invariant_mode(True, backend="triton"):
        vfm.vllm_fused_moe(
            x,
            torch.ones(1, 3, device=DEVICE),
            fc1,
            fc2,
            activation_type=ActivationType.SQUARED_RELU,
            num_local_experts=3,
            local_expert_start=0,
            valid_tokens=torch.tensor([1], dtype=torch.int32, device=DEVICE),
            routing_map=torch.tensor([[0, 1, 2]], dtype=torch.int64, device=DEVICE),
            out=out.tensor,
        )
    expected = torch.zeros(1, hidden)
    expected[0, 0], expected[0, 1] = 1.0, 2.0**24 + 2
    ou.assert_exact(out.tensor, expected, what="batch-invariant top-k sum")
    ou.check_guards(out=out)


PRODUCTION = {
    # Nemotron 3 Nano: H 2688, top-6, 16 local experts at EP 8; prefill, default and BI.
    "nemotron-nano-ep8-prefill": Case(
        topk=6, hidden=2688, tokens=8192, num_local=16, ep=8, rank="middle", num_sms=None
    ),
    "nemotron-nano-ep8-prefill-bi": Case(
        acc_fp64=True, topk=6, hidden=2688, tokens=8192, num_local=16, ep=8, num_sms=None
    ),
    # DeepSeek-V3: H 7168 (seven K blocks), top-8, 32 local experts at EP 8.
    "deepseek-v3-ep8": Case(
        topk=8, hidden=7168, tokens=2048, num_local=32, ep=8, rank="last", num_sms=None
    ),
    "deepseek-v3-ep8-bi-gauss": Case(
        data="gauss",
        acc_fp64=True,
        topk=8,
        hidden=7168,
        tokens=512,
        num_local=32,
        ep=8,
        num_sms=None,
        out="pool",
    ),
    # Mixtral 8x7B: H 4096, top-2, one local expert at EP 8.
    "mixtral-8x7b-ep8": Case(topk=2, hidden=4096, tokens=4096, num_local=1, ep=8, num_sms=None),
    # Qwen3-30B-A3B decode: H 2048, top-8, 16 local experts, 512 gathered tokens, bf16 output.
    "qwen3-30b-a3b-ep8-decode": Case(
        topk=8, hidden=2048, tokens=512, pad=64, num_local=16, ep=8, num_sms=None, out="bf16"
    ),
}


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("name", list(PRODUCTION))
def test_production_sizes(name, monkeypatch):
    """Deployed hidden sizes (one to seven K blocks), top-k, expert counts and batch sizes."""
    run_case(PRODUCTION[name], monkeypatch)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("apply_weights,acc_fp64", [(False, False), (True, True)])
@pytest.mark.parametrize("tokens", ou.int32_overflow_params(8 * 4096))
def test_int32_offsets(tokens, apply_weights, acc_fp64):
    """Input offsets ``(t * topk + j) * K`` past 2**31 (top-8, K 4096); sampled rows exact."""
    k, hidden, num_local = 8, 4096, 8
    ou.require_free_memory(8 * 2**30)
    g = ou.make_generator(device=DEVICE)
    routing = torch.randint(-1, 2 * num_local, (tokens, k), generator=g, device=DEVICE)
    local = (routing >= 0) & (routing < num_local)
    row = torch.randint(-2, 3, (tokens, k), generator=g, device=DEVICE).double()
    col = torch.randint(-3, 4, (hidden,), generator=g, device=DEVICE).double()
    p = (torch.randint(1, 17, (tokens, k), generator=g, device=DEVICE) / 16.0).float()
    x = torch.empty(tokens, k, hidden, dtype=torch.bfloat16, device=DEVICE)
    for lo in range(0, tokens, 4096):  # rank-one fill, in chunks
        x[lo : lo + 4096] = (row[lo : lo + 4096, :, None] * col).to(torch.bfloat16)
    x[~local] = float("nan")
    out = torch.full((tokens, hidden), float("nan"), device=DEVICE)
    vfm._moe_sum(
        x.view(tokens * k, hidden),
        p,
        tokens,
        k,
        hidden,
        torch.tensor([tokens], dtype=torch.int32, device=DEVICE),
        routing,
        0,
        num_local,
        out=out,
        apply_weights=apply_weights,
        acc_fp64=acc_fp64,
    )
    limit = 2**31 // (k * hidden)
    sample = torch.tensor([0, 1, limit - 1, limit, tokens - 1], device=DEVICE)
    sample = torch.cat([sample, torch.randint(0, tokens, (27,), generator=g, device=DEVICE)])
    sample = sample.clamp(max=tokens - 1).unique()
    # The slot-by-slot sum from +0.0, as in the other exact cases: a sum that cancels exactly is
    # +0.0 (the closed form (row . weights) * col would give -0.0 there for col < 0).
    xs, ps, ls = x[sample], p[sample], local[sample]
    require_exact_in_fp32(xs, ps, ls, apply_weights)
    spec = ou.fp64_reference(reference_sum, xs, ps, ls, apply_weights)
    ou.assert_exact(out[sample], spec.float(), what=f"sampled rows of {tokens} tokens")


@pytest.mark.sanitizer
@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("apply_weights,acc_fp64", VARIANTS)
def test_sanitizer(apply_weights, acc_fp64, monkeypatch):
    """Small cases for compute-sanitizer memcheck, one per variant (also run normally)."""
    case = Case(
        apply_weights=apply_weights,
        acc_fp64=acc_fp64,
        topk=3,
        hidden=1025,
        tokens=7,
        pad=2,
        num_local=7,
        ep=4,
        rank="middle",
        kind="edges",
    )
    run_case(case, monkeypatch)


def all_cases() -> List[Case]:
    """Every case of the parametrized ``run_case`` tests (for scratch checks outside pytest)."""
    return list(EXACT_CASES) + list(TOLERANCE_CASES) + list(PRODUCTION.values())
