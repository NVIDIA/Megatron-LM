# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `_sigmoid_topk_kernel`, the Triton sigmoid top-k routing kernel.

Kernel: `megatron/core/inference/moe/sigmoid_topk.py` (`_sigmoid_topk_kernel`, launched by
`sigmoid_topk_routing`). Inference reaches it twice: `InferenceTopKRouter._forward` calls it
directly on the 'triton' router backend (`router.py`, `_uses_sigmoid_topk_kernel`), and in
batch-invariant mode `topk_routing_with_score_function` calls it with `batch_invariant=True`
(`moe_utils.py`, `_use_batch_invariant_sigmoid_topk`, since #8602). Every sigmoid MoE layer with
an expert bias (DeepSeek-V3-style aux-loss-free routing) routes through it.
`test_inference_topk_router.py` reaches it only through the router, which cannot select its
block size, warp count, batch-invariant launch, strided bias, tie rule or slot order.

Contract (written from `sigmoid_topk_routing`'s documentation and the routing semantics of
`topk_routing_with_score_function` it replaces):
- Inputs: fp32 logits z [T, E], contiguous; an fp32 expert bias b [E] on the same device, any
  1-D stride (strided and broadcast views are documented as supported); 1 < k <= min(32, E),
  E <= 512; an optional scaling factor (None means unscaled). Anything else is rejected with an
  AssertionError before the launch.
- Selection: per token, the k experts with the largest routing score r = sigmoid(z) + b,
  computed in fp32 (the bias steers selection only). Ties at the k-th place go to the lower
  expert index. Output order (documented; it is the order of aten `topk(sorted=False)`, which
  the compiled router produces): the experts whose score is above the k-th score, in ascending
  expert order, then the selected experts tied with it, in ascending order.
- Probabilities: p_j = s_j / (sum of the k selected s + 1e-20) * scaling, with s = sigmoid(z)
  (no bias), in the slot of its expert. Outputs are new contiguous tensors: probs [T, k] fp32
  and indices [T, k] int64; T = 0 gives empty outputs. Inputs are read only.
- Batch invariance: a token's indices depend only on its own row (exact integer selection), at
  every launch. With `batch_invariant=True` its probabilities' bits do not depend on T or on its
  position either (documented); without it they may differ in the last bit for k >= 8 once
  T >= 1024 (documented).
- Dispatch in batch-invariant mode (`_use_batch_invariant_sigmoid_topk`): sigmoid scores with an
  expert bias, k > 1, fp32 CUDA logits, a supported shape and no group-limited routing,
  pre-softmax, router replay, precomputed indices, quantile balancing or opt-out run this kernel
  with `batch_invariant=True` (once per call); everything else runs the reference routing.
  Group-limited routing is not supported by the kernel, so it must take the reference routing.

Reference: fp64 routing written from the contract: a stable descending sort of the fp64 scores
(ties by index) gives the selected set, regrouped into the documented slot order; probabilities
from fp64 sigmoid. Logits are bf16-exact values with a selection margin (the k-th and (k+1)-th
fp64 scores differ by more than 1e-4 of the k-th), so every correct fp32 implementation selects
the same experts; ties are built exactly (same logit, same bias). Trusted implementation: the
same probabilities computed with torch's fp32 ops on the device. Group-limited routing has its
own fp64 reference (`group_limited_reference`, from `group_limited_topk`'s documented rule: groups
ranked by the sum of their top k / group_topk scores). Tolerance class: exact for indices, slot
order, shapes, dtypes, dispatch, batch invariance and read-only inputs; floating point for probs,
bounded over the tensor with factor 4 (PROB_FACTOR: Triton's `tl.sigmoid` lowers fp32 exp to
`ex2.approx` after a log2(e) prescale, about 2**-22 relative, while torch's `expf` is within 1-2
ulp; a wrong normalization, a dropped expert or a missing scale errs by 1e-3 or more). On exactly
tied rows the probs are exact fp32 values, so the tie test bounds each row with a floor of 2 ulp
per prob for the kernel's `div.full.f32` division (DIVISION_ULPS).

Structure (read after the contract): one program per `BLOCK_T` tokens (`block_tokens`: 1 for
k >= 8 or E <= 128, else 4), `EXPERTS_POW2` columns (padding columns masked and given the
smallest key), keys from the fp32 bits with a separate rule for negative scores, the k-th key by
`tl.topk` (k >= 8 and Triton has it) or a k-step max loop (k < 8), slots by prefix sums, the
probability sum as a left fold in slot order (k < 8) or a tree sum (k >= 8), `num_warps` 1 or 4
(1 for k >= 8 at T >= 1024 unless `batch_invariant`, 1 for k < 8 at E <= 128), int64 row
offsets. Covered: k in {2, 3, 6, 7, 8, 9, 16, 22, 31, 32} and k = E; E in {2, 8, 32, 64, 127,
128, 129, 160, 256, 384, 512}; T in {0, 1, 3, 4, 5, 9, 129, 1023, 1024, 1025} (4-row blocks and
the 1024-token warp switch) and production sizes; block_tokens None/2/8; scaling; contiguous,
strided and broadcast bias; both `batch_invariant` settings; negative scores; ties at, above and
across the k-th place, and all-equal rows; row offsets past 2**31 elements (`large`).

Main vs open PRs: all cases are expected to pass on main. #8562 does not touch the kernel.

Hardware: H100 (1 GPU). The kernel has no architecture-specific variant.
"""

import math

import pytest
import torch

from megatron.core.inference.moe import sigmoid_topk as st
from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from megatron.core.transformer.moe import moe_utils
from tests.unit_tests.inference.oracles import oracle_utils as ou

pytestmark = [ou.requires_sm(ou.HOPPER)]

_DEVICE = "cuda"

MAX_TOPK = 32  # sigmoid_topk.py: is_sigmoid_topk_supported
MAX_EXPERTS = 512
TL_TOPK_FROM = 8  # USE_TL_TOPK and the tree-sum probabilities for k >= 8
WIDE_ROWS_FROM = 129  # block_tokens = 4 for k < 8 when E > 128
BLOCK_T_WIDE = 4
WARP_SWITCH_TOKENS = 1024  # num_warps = 1 for k >= 8 at T >= 1024 unless batch_invariant
PROB_FACTOR = 4.0
MARGIN = 1e-4
DIVISION_ULPS = 2.0  # Triton's fp32 `/` is `div.full.f32`: at most 2 ulp (PTX ISA)


@pytest.fixture(autouse=True)
def _batch_invariant_mode_off():
    """Each test sets the mode it needs; none may leak it."""
    assert not bik.is_batch_invariant_mode_enabled(), "batch-invariant mode leaked into the test"
    yield
    assert not bik.is_batch_invariant_mode_enabled(), "the test left batch-invariant mode on"


# ---- reference ---------------------------------------------------------------------------------


def documented_order(keys: torch.Tensor, topk: int) -> torch.Tensor:
    """Selected experts in the documented slot order, from routing scores [T, E] (any dtype).

    A stable descending sort keeps tied scores in ascending expert order, so its first k entries
    are the k largest scores with ties at the k-th place going to the lower index. They are then
    regrouped: those above the k-th score first, then those equal to it, each in ascending order.
    """
    num_experts = keys.shape[-1]
    if keys.shape[0] == 0:
        return torch.empty(0, topk, dtype=torch.int64, device=keys.device)
    chosen = torch.sort(keys, dim=-1, descending=True, stable=True).indices[:, :topk]
    chosen_keys = keys.gather(-1, chosen)
    tied = chosen_keys == chosen_keys[:, -1:]
    slot_key = tied.to(torch.int64) * num_experts + chosen
    return chosen.gather(-1, slot_key.argsort(dim=-1))


def routing_reference(logits, bias, topk, scaling):
    """(probs, indices) from the contract; fp64 under ``ou.fp64_reference``."""
    scores = torch.sigmoid(logits)
    indices = documented_order(scores + bias, topk)
    picked = scores.gather(-1, indices)
    probs = picked / (picked.sum(dim=-1, keepdim=True) + 1e-20)
    return (probs * scaling if scaling else probs), indices


def probs_at(logits, indices, scaling):
    """Probabilities of the given experts (the trusted fp32 form, or fp64)."""
    picked = torch.sigmoid(logits).gather(-1, indices)
    probs = picked / (picked.sum(dim=-1, keepdim=True) + 1e-20)
    return probs * scaling if scaling else probs


def group_margin(bias: torch.Tensor, topk: int, num_groups: int, group_topk: int):
    """``accept`` for `make_logits`: margins on the kept groups and on the k-th expert in them."""
    bias64 = bias.detach().double().cpu()

    def accept(z):
        keys = torch.sigmoid(z) + bias64
        grouped = keys.view(z.shape[0], num_groups, -1)
        group_score = grouped.topk(topk // group_topk, dim=-1).values.sum(dim=-1)
        kept = torch.zeros_like(group_score, dtype=torch.bool)
        kept.scatter_(1, group_score.topk(group_topk, dim=-1).indices, True)
        mask = kept.unsqueeze(-1).expand_as(grouped).reshape(z.shape)
        return has_margin(group_score, group_topk) & has_margin(keys.masked_fill(~mask, -1e9), topk)

    return accept


def group_limited_reference(logits, bias, topk, num_groups, group_topk, scaling):
    """Group-limited sigmoid routing (`group_limited_topk`'s documented rule), as sets.

    Experts form ``num_groups`` equal groups; each token keeps the ``group_topk`` groups with the
    largest sum of their top ``topk // group_topk`` routing scores, then the k largest routing
    scores among the kept groups. Returns probs and experts sorted by expert id.
    """
    tokens, num_experts = logits.shape
    scores = torch.sigmoid(logits)
    keys = scores + bias
    grouped = keys.view(tokens, num_groups, num_experts // num_groups)
    group_score = grouped.topk(topk // group_topk, dim=-1).values.sum(dim=-1)
    kept = torch.zeros_like(group_score, dtype=torch.bool)
    kept.scatter_(1, group_score.topk(group_topk, dim=-1).indices, True)
    mask = kept.unsqueeze(-1).expand_as(grouped).reshape(tokens, num_experts)
    masked = keys.masked_fill(~mask, float("-inf"))
    experts = masked.topk(topk, dim=-1).indices.sort(dim=-1).values
    return probs_at(logits, experts, scaling), experts


# ---- inputs ------------------------------------------------------------------------------------


def bf16_pool(span: float = 8.0) -> torch.Tensor:
    """Every bf16 value v with 2**-4 <= |v| <= span / 2, both signs (fp64, CPU)."""
    bits = torch.arange(0, 0x7F80, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16).double()
    values = values[(values >= 2.0**-4) & (values <= span / 2)]
    return torch.cat([-values.flip(0), values])


def make_bias(num_experts: int, seed: int = 5, positive: int = None) -> torch.Tensor:
    """Expert biases, multiples of 1/64 (exact in bf16 and fp32), fp32 on the device.

    Default: in [-1/4, 1/4]. With ``positive``: only that many experts keep such a bias and all
    others get one in [-2, -1], so at most ``positive`` routing scores are positive; with
    ``positive < k`` the k-th score of every token is negative (the kernel's keys treat negative
    floats separately).
    """
    g = ou.make_generator(seed)
    bias = torch.randint(-16, 17, (num_experts,), generator=g).double() / 64
    if positive is not None:
        low = torch.randint(-128, -63, (num_experts,), generator=g).double() / 64
        keep = torch.randperm(num_experts, generator=g)[:positive]
        low[keep] = bias[keep]
        bias = low
    return bias.float().to(_DEVICE)


def has_margin(keys: torch.Tensor, topk: int) -> torch.Tensor:
    """Rows whose k-th and (k+1)-th largest keys differ by more than ``MARGIN`` of the k-th."""
    if topk >= keys.shape[-1]:
        return torch.ones(keys.shape[0], dtype=torch.bool)
    top = keys.topk(topk + 1, dim=-1).values
    gap = top[:, topk - 1] - top[:, topk]
    return gap > MARGIN * top[:, topk - 1].abs().clamp(min=1.0)


def make_logits(tokens, num_experts, topk, bias, seed, accept=None):
    """bf16-exact fp32 logits [T, E] with a selection margin, a different row for every token.

    Each row is E distinct bf16 values drawn from `bf16_pool`; rows whose k-th and (k+1)-th fp64
    scores sigmoid(z) + b are closer than ``MARGIN`` of the k-th (or that ``accept`` rejects) are
    redrawn, so every correct fp32 implementation selects the same experts.
    """
    pool = bf16_pool()
    assert pool.numel() >= num_experts
    bias64 = bias.detach().double().cpu()
    accept = accept or (lambda z: has_margin(torch.sigmoid(z) + bias64, topk))
    g = ou.make_generator(seed)
    out = torch.empty(tokens, num_experts, dtype=torch.float64)
    todo = torch.arange(tokens)
    for _ in range(200):
        if todo.numel() == 0:
            break
        draw = torch.rand(todo.numel(), pool.numel(), generator=g).argsort(dim=-1)
        rows = pool[draw[:, :num_experts]]
        ok = accept(rows)
        out[todo[ok]] = rows[ok]
        todo = todo[~ok]
    assert todo.numel() == 0, "could not build rows with a selection margin"
    logits = out.float()
    assert torch.equal(logits.to(torch.bfloat16).double(), out), "logits must be bf16-exact"
    return logits.to(_DEVICE)


def place_bias(bias: torch.Tensor, layout: str) -> torch.Tensor:
    """The bias values as a contiguous, strided (stride 2, offset 1) or broadcast view.

    Unused storage holds NaN, so a load that ignores the stride or offset reads NaN.
    """
    if layout == "contiguous":
        return bias.clone()
    if layout == "strided":
        storage = torch.full((2 * bias.numel() + 1,), float("nan"), device=bias.device)
        view = storage[1::2]
        view.copy_(bias)
        return view
    if layout == "broadcast":
        storage = torch.full((bias.numel(),), float("nan"), device=bias.device)
        storage[0] = bias[0]
        return storage[:1].expand(bias.numel())
    raise ValueError(layout)


def logical_bias(bias: torch.Tensor) -> torch.Tensor:
    return bias.contiguous().clone()


# ---- checks ------------------------------------------------------------------------------------


def run_kernel(logits, bias, topk, scaling=None, **kwargs):
    """Run the launcher on a guarded copy of the logits; inputs must come back unchanged."""
    buf = ou.guarded_like(logits)
    bias_before = bias.clone()
    probs, indices = st.sigmoid_topk_routing(buf.tensor, bias, topk, scaling, **kwargs)
    ou.check_guards(logits=buf)
    ou.assert_exact(buf.tensor, logits, what="logits are read-only")
    ou.assert_exact(bias, bias_before, what="expert bias is read-only")
    return probs, indices


def check_routing(probs, indices, logits, bias, topk, scaling, what, **bound):
    """Exact indices in the documented order; probs bounded against fp64 (``bound``: extra
    arguments for `ou.assert_close_to_reference`)."""
    tokens = logits.shape[0]
    assert tuple(probs.shape) == (tokens, topk), f"{what}: probs {tuple(probs.shape)}"
    assert tuple(indices.shape) == (tokens, topk), f"{what}: indices {tuple(indices.shape)}"
    assert probs.dtype == torch.float32 and indices.dtype == torch.int64, what
    assert probs.is_contiguous() and indices.is_contiguous(), f"{what}: outputs not contiguous"
    ref_probs, ref_indices = ou.fp64_reference(
        routing_reference, logits, logical_bias(bias), topk, scaling
    )
    ou.assert_exact(indices, ref_indices, what=f"{what}: indices (documented order)")
    if tokens == 0:
        return
    trusted = probs_at(logits, ref_indices, scaling)
    ou.assert_close_to_reference(
        probs, ref_probs, trusted, factor=PROB_FACTOR, what=f"{what}: probs", **bound
    )


def _id(case):
    return "-".join(f"{k}{v}" for k, v in case.items())


# ---- numerics over the variants -----------------------------------------------------------------

_CASES = ou.pairwise_cases(
    {
        "e": (8, 64, 128, 129, 160, 512),
        "k": (2, 6, 7, 8, 9, 22, 32),
        "t": (0, 1, 4, 5, 1023, 1024, 1025),
        "bias": ("contiguous", "strided", "broadcast"),
        "bi": (False, True),
        "scaling": (None, 2.5),
    }
)


@pytest.mark.parametrize("case", _CASES, ids=[_id(c) for c in _CASES])
def test_routing_matches_fp64(case):
    """Every k variant (loop / tl.topk, left fold / tree sum), E around the 128-column switch and
    non-powers of two, T around the 4-row blocks and the 1024-token warp switch, bias layouts,
    both launches and scaling: exact indices in the documented order, probs against fp64."""
    num_experts, topk = case["e"], min(case["k"], case["e"])
    bias = place_bias(make_bias(num_experts), case["bias"])
    logits = make_logits(case["t"], num_experts, topk, logical_bias(bias), seed=case["t"] + 1)
    probs, indices = run_kernel(logits, bias, topk, case["scaling"], batch_invariant=case["bi"])
    check_routing(probs, indices, logits, bias, topk, case["scaling"], _id(case))


_K_EQUALS_E = [(2, 2), (8, 8), (9, 9), (32, 32)]


@pytest.mark.parametrize("num_experts,topk", _K_EQUALS_E, ids=[f"e{e}" for e, _ in _K_EQUALS_E])
def test_every_expert_selected(num_experts, topk):
    """k = E (the smallest E, and E at the 8 and 32 boundaries): every expert, in index order."""
    bias = make_bias(num_experts)
    logits = make_logits(33, num_experts, topk, bias, seed=num_experts)
    probs, indices = run_kernel(logits, bias, topk, 2.5)
    check_routing(probs, indices, logits, bias, topk, 2.5, f"k = E = {num_experts}")


_BLOCKS = [(2, 6, 64), (8, 6, 64), (2, 9, 160), (8, 6, 256)]


@pytest.mark.parametrize(
    "block_tokens,topk,num_experts", _BLOCKS, ids=[f"bt{b}-k{k}-e{e}" for b, k, e in _BLOCKS]
)
@pytest.mark.parametrize("tokens", [7, 9])
def test_explicit_block_tokens(block_tokens, topk, num_experts, tokens):
    """The launcher's `block_tokens` override (rows per program, a power of two) with partial
    last blocks: the same contract."""
    bias = make_bias(num_experts)
    logits = make_logits(tokens, num_experts, topk, bias, seed=tokens)
    probs, indices = run_kernel(logits, bias, topk, None, block_tokens=block_tokens)
    check_routing(probs, indices, logits, bias, topk, None, f"block_tokens {block_tokens}")


_NEG = [(64, 2, False), (160, 6, True), (256, 8, False), (512, 22, True), (512, 32, False)]


@pytest.mark.parametrize(
    "num_experts,topk,bi", _NEG, ids=[f"e{e}-k{k}-bi{int(b)}" for e, k, b in _NEG]
)
def test_negative_routing_scores(num_experts, topk, bi):
    """Expert biases push the k-th score (and most of the selected ones) below zero: selection
    and slot order still follow the values (negative keys are ordered by their own rule)."""
    bias = make_bias(num_experts, positive=topk // 2)
    logits = make_logits(129, num_experts, topk, bias, seed=topk)
    keys = torch.sigmoid(logits.double()) + bias.double()
    assert bool((keys.topk(topk, dim=-1).values[:, -1] < 0).all()), "a k-th score is not < 0"
    probs, indices = run_kernel(logits, bias, topk, None, batch_invariant=bi)
    check_routing(probs, indices, logits, bias, topk, None, f"negative E={num_experts} k={topk}")


# ---- ties --------------------------------------------------------------------------------------

GRID = torch.arange(-256, 256, dtype=torch.float64) / 64  # 512 bf16-exact, well-separated logits


def tie_rows(num_experts, topk, kind, tokens, seed):
    """Rows with exact ties, under a uniform bias (so equal logits give equal fp32 scores).

    Each row is a random permutation of distinct grid logits (consecutive grid values give
    scores at least 2.7e-4 apart), then: ``at_kth`` copies the k-th largest logit to 1-5 other
    experts below the top k; ``across`` copies the (k-1)-th largest logit to the k-th and
    (k+1)-th; ``above`` copies the largest to the second largest; ``all_equal`` sets every logit
    to the same value; ``underflow`` gives all but k - 1 experts the same very negative logit
    (z = -100, whose sigmoid is 0 or a subnormal in fp32 and 4e-44 in fp64, the same for all of
    them), so the k-th place is a tie of E - k + 1 experts.
    """
    g = ou.make_generator(seed)
    rows = []
    for _ in range(tokens):
        z = GRID[torch.randperm(GRID.numel(), generator=g)[:num_experts]].clone()
        order = z.argsort(descending=True).tolist()
        level = [float(z[i]) for i in order]  # descending logit at each rank
        if kind == "at_kth" and topk < num_experts:
            extra = torch.tensor(order[topk:])[torch.randperm(num_experts - topk, generator=g)]
            z[extra[: int(torch.randint(1, 6, (1,), generator=g))]] = level[topk - 1]
        elif kind == "across" and topk < num_experts:
            z[order[topk - 1]] = level[topk - 2]
            z[order[topk]] = level[topk - 2]
        elif kind == "above":
            z[order[1]] = level[0]
        elif kind == "all_equal":
            z[:] = 0.5
        elif kind == "underflow":
            z[torch.tensor(order[topk - 1 :])] = -100.0
        rows.append(z)
    logits = torch.stack(rows).float()
    assert torch.equal(logits.to(torch.bfloat16).float(), logits)
    return logits.to(_DEVICE)


_TIE_KINDS = ("at_kth", "across", "above", "all_equal", "underflow")
_TIE_SHAPES = [(8, 2), (64, 7), (160, 6), (160, 9), (512, 32)]


@pytest.mark.parametrize("kind", _TIE_KINDS)
@pytest.mark.parametrize(
    "num_experts,topk", _TIE_SHAPES, ids=[f"e{e}-k{k}" for e, k in _TIE_SHAPES]
)
def test_ties_follow_the_documented_rule(kind, num_experts, topk):
    """Exactly tied scores: the k-th place goes to the lowest indices, ties above it are all
    selected, and slots keep the documented order, under both launches."""
    bias = torch.full((num_experts,), -0.25, device=_DEVICE)
    logits = tie_rows(num_experts, topk, kind, 33, seed=topk * 7 + len(kind))
    # Tied scores cancel exactly in s_j / sum(s): a row whose selected experts are all tied has
    # probs 1/k * scaling, exact in fp32, which torch's IEEE division returns exactly. The kernel
    # divides with Triton's fp32 `/` (`div.full.f32`, up to 2 ulp, PTX ISA), so every prob of
    # such a row may be off by 2 ulp while the trusted error is 0 (pdx 7986825: 1 ulp on the
    # k = 2 rows). The bound is per row, with a floor admitting 2 ulp on all k probs.
    bound = dict(dim=-1, floor_ulps=DIVISION_ULPS * math.sqrt(topk))
    for bi in (False, True):
        probs, indices = run_kernel(logits, bias, topk, 2.5, batch_invariant=bi)
        check_routing(probs, indices, logits, bias, topk, 2.5, f"ties {kind} bi={bi}", **bound)


# ---- batch invariance --------------------------------------------------------------------------

_INVARIANCE = [(64, 6), (256, 6), (64, 8), (384, 9), (512, 32)]


@pytest.mark.parametrize(
    "num_experts,topk", _INVARIANCE, ids=[f"e{e}-k{k}" for e, k in _INVARIANCE]
)
@pytest.mark.parametrize("bi", [False, True], ids=["launch", "bi"])
def test_rows_do_not_depend_on_the_batch(num_experts, topk, bi):
    """A probe block of 16 tokens, placed at the start, in the middle (off any 4-row boundary)
    and at the end of batches of 16 ... 4099 tokens (both sides of the 1024-token warp switch):
    its indices are the same bits everywhere; so are its probs with `batch_invariant=True`, and
    for k < 8 at every launch (the warp count depends only on E there)."""
    bias = make_bias(num_experts)
    probe = make_logits(16, num_experts, topk, bias, seed=99)
    filler = make_logits(4099, num_experts, topk, bias, seed=98)
    base_probs, base_indices = st.sigmoid_topk_routing(probe, bias, topk, 2.5, batch_invariant=bi)
    check_routing(base_probs, base_indices, probe, bias, topk, 2.5, "probe block alone")
    for tokens in (17, 23, 1023, 1024, 1025, 4099):
        for start in sorted({0, min((tokens - 16) // 2 | 1, tokens - 16), tokens - 16}):
            batch = filler[:tokens].clone()
            batch[start : start + 16] = probe
            probs, indices = st.sigmoid_topk_routing(batch, bias, topk, 2.5, batch_invariant=bi)
            what = f"T={tokens} start={start} bi={bi}"
            ou.assert_exact(indices[start : start + 16], base_indices, what=f"{what}: indices")
            if bi or topk < TL_TOPK_FROM:
                ou.assert_exact(probs[start : start + 16], base_probs, what=f"{what}: probs")


# ---- launcher preconditions --------------------------------------------------------------------


def _reject_cases():
    logits = lambda t, e, dtype=torch.float32: torch.zeros(t, e, device=_DEVICE, dtype=dtype)
    bias = lambda e, **kw: torch.zeros(e, device=_DEVICE, **kw)
    return {
        "k1": (logits(4, 64), bias(64), 1),
        "k33": (logits(4, 64), bias(64), MAX_TOPK + 1),
        "k_above_e": (logits(4, 16), bias(16), 17),
        "e513": (logits(4, MAX_EXPERTS + 1), bias(MAX_EXPERTS + 1), 8),
        "bf16_logits": (logits(4, 64, torch.bfloat16), bias(64), 6),
        "non_contiguous_logits": (logits(64, 4).t(), bias(64), 6),
        "bias_2d": (logits(4, 64), bias(64).view(64, 1), 6),
        "bias_length": (logits(4, 64), bias(63), 6),
        "bias_bf16": (logits(4, 64), bias(64, dtype=torch.bfloat16), 6),
        "bias_cpu": (logits(4, 64), torch.zeros(64), 6),
    }


@pytest.mark.parametrize("name", sorted(_reject_cases()))
def test_unsupported_inputs_are_rejected(name):
    """Shapes and layouts outside the documented range raise before any launch."""
    logits, bias, topk = _reject_cases()[name]
    with pytest.raises(AssertionError):
        st.sigmoid_topk_routing(logits, bias, topk, 2.5)


# ---- dispatch in batch-invariant mode ----------------------------------------------------------


class _KernelSpy:
    """Counts the launcher calls `topk_routing_with_score_function` makes, with their flag."""

    def __init__(self, monkeypatch):
        self.calls = []
        launcher = moe_utils.sigmoid_topk_routing

        def spy(*args, **kwargs):
            self.calls.append(kwargs.get("batch_invariant", False))
            return launcher(*args, **kwargs)

        monkeypatch.setattr(moe_utils, "sigmoid_topk_routing", spy)


_DISPATCH = {
    # name: (routing kwargs overrides, logits dtype, expected kernel calls)
    "sigmoid_bias": ({}, torch.float32, [True]),
    "sigmoid_bias_k32": ({"topk": 32}, torch.float32, [True]),
    "sigmoid_bias_e512": ({"num_experts": 512, "topk": 8}, torch.float32, [True]),
    "top1": ({"topk": 1}, torch.float32, []),
    "k33": ({"topk": 33}, torch.float32, []),
    "e1024": ({"num_experts": 1024, "topk": 8}, torch.float32, []),
    "no_bias": ({"expert_bias": None}, torch.float32, []),
    "softmax": ({"score_function": "softmax", "expert_bias": None}, torch.float32, []),
    "pre_softmax": ({"use_pre_softmax": True}, torch.float32, []),
    "groups": ({"num_groups": 8, "group_topk": 2}, torch.float32, []),
    "precomputed": ({"precomputed": True}, torch.float32, []),
    "opt_out": ({"allow_batch_invariant_sigmoid_topk": False}, torch.float32, []),
    "bf16_logits": ({}, torch.bfloat16, []),
}


@pytest.mark.parametrize("name", list(_DISPATCH))
def test_batch_invariant_dispatch(name, monkeypatch):
    """In batch-invariant mode the routing runs this kernel, once and with the batch-invariant
    launch, exactly where the documented conditions hold; outside the mode it never does. Where
    it runs, the result is the kernel's contract; group-limited routing is checked against its
    own fp64 reference."""
    overrides, dtype, expected = _DISPATCH[name]
    overrides = dict(overrides)
    num_experts = overrides.pop("num_experts", 64)
    topk = overrides.pop("topk", 6)
    bias = make_bias(num_experts)
    accept = None
    if "num_groups" in overrides:
        accept = group_margin(bias, topk, overrides["num_groups"], overrides["group_topk"])
    logits = make_logits(65, num_experts, min(topk, num_experts), bias, 3, accept).to(dtype)
    kwargs = dict(topk=topk, score_function="sigmoid", expert_bias=bias, dense_output=True)
    kwargs.update(scaling_factor=2.5)
    if overrides.pop("precomputed", False):
        kwargs["precomputed_indices"] = documented_order(logits.float(), topk)
    kwargs.update(overrides)
    spy = _KernelSpy(monkeypatch)
    with torch.no_grad():
        outside = moe_utils.topk_routing_with_score_function(logits, **kwargs)
        assert spy.calls == [], f"{name}: the kernel ran outside batch-invariant mode"
        with bik.set_batch_invariant_mode(True, backend="te_native"):
            probs, indices = moe_utils.topk_routing_with_score_function(logits, **kwargs)
    assert spy.calls == expected, f"{name}: kernel calls {spy.calls}, expected {expected}"
    if expected:
        check_routing(probs, indices, logits, bias, topk, 2.5, f"dispatch {name}")
        direct, direct_idx = st.sigmoid_topk_routing(logits, bias, topk, 2.5, batch_invariant=True)
        ou.assert_exact(indices, direct_idx, what=f"{name}: same indices as the direct launch")
        ou.assert_exact(probs, direct, what=f"{name}: same probs as the direct launch")
    elif name == "groups":
        ref_probs, ref_experts = ou.fp64_reference(
            group_limited_reference, logits, bias, topk, 8, 2, 2.5
        )
        order = indices.argsort(dim=-1)
        ou.assert_exact(indices.gather(-1, order), ref_experts, what="group-limited experts")
        ou.assert_close_to_reference(
            probs.gather(-1, order),
            ref_probs,
            probs_at(logits, ref_experts, 2.5),
            factor=PROB_FACTOR,
            what="group-limited probs",
        )
    assert tuple(outside[1].shape) == tuple(indices.shape), f"{name}: shapes differ by mode"


def test_batch_invariant_autograd_forward_is_the_kernel(monkeypatch):
    """With gradients (batch-invariant training and RL recompute) the forward is the same
    kernel launch, so it returns the inference bits."""
    num_experts, topk = 128, 6
    bias = make_bias(num_experts)
    logits = make_logits(65, num_experts, topk, bias, seed=4)
    spy = _KernelSpy(monkeypatch)
    with bik.set_batch_invariant_mode(True, backend="te_native"):
        probs, indices = moe_utils.topk_routing_with_score_function(
            logits.clone().requires_grad_(),
            topk,
            score_function="sigmoid",
            expert_bias=bias,
            scaling_factor=2.5,
            dense_output=True,
        )
    assert spy.calls == [True], f"kernel calls {spy.calls}"
    direct, direct_idx = st.sigmoid_topk_routing(logits, bias, topk, 2.5, batch_invariant=True)
    ou.assert_exact(indices, direct_idx, what="indices with gradients")
    ou.assert_exact(probs.detach(), direct, what="probs with gradients")


# ---- production sizes, sanitizer, int32 limits -------------------------------------------------

_PRODUCTION = [
    pytest.param(128, 6, 2.5, 4096, False, id="nano-e128-k6"),
    pytest.param(160, 8, 2.5, 4096, True, id="e160-k8-bi"),
    pytest.param(256, 8, 2.5, 8192, False, id="e256-k8"),
    pytest.param(384, 8, 2.827, 8192, True, id="e384-k8-bi"),
    pytest.param(512, 22, 5.0, 4096, False, id="super-e512-k22"),
]


@pytest.mark.parametrize("num_experts,topk,scaling,tokens,bi", _PRODUCTION)
def test_production_sizes(num_experts, topk, scaling, tokens, bi):
    """Deployed routing shapes (prefill-sized batches) against fp64."""
    bias = make_bias(num_experts)
    logits = make_logits(tokens, num_experts, topk, bias, seed=tokens + topk)
    probs, indices = run_kernel(logits, bias, topk, scaling, batch_invariant=bi)
    check_routing(probs, indices, logits, bias, topk, scaling, f"production E={num_experts}")


@pytest.mark.sanitizer
@pytest.mark.parametrize(
    "num_experts,topk,tokens", [(160, 6, 7), (160, 9, 5)], ids=["k6-4row-blocks", "k9-tl-topk"]
)
def test_sanitizer_case(num_experts, topk, tokens):
    """Small case for compute-sanitizer: 160 experts (96 masked columns of 256), a strided bias,
    k = 6 with 4-row blocks ending in a partial block, and k = 9 on the tl.topk path."""
    bias = place_bias(make_bias(num_experts), "strided")
    logits = make_logits(tokens, num_experts, topk, logical_bias(bias), seed=tokens)
    probs, indices = run_kernel(logits, bias, topk, 2.5, batch_invariant=True)
    check_routing(probs, indices, logits, bias, topk, 2.5, "sanitizer")


@pytest.mark.parametrize("tokens", ou.int32_overflow_params(MAX_EXPERTS))
def test_row_offsets_past_int32(tokens):
    """T x 512 logits whose element offsets reach 2**31: rows past the boundary still route
    their own logits (int64 row offsets). The rows are a repeated block with a margin; the
    reference is checked on the boundary rows and the last rows."""
    num_experts, topk = MAX_EXPERTS, 8
    ou.require_free_memory(tokens * num_experts * 4 + tokens * topk * 12 + 2**30)
    bias = make_bias(num_experts)
    block = make_logits(1024, num_experts, topk, bias, seed=11)
    logits = block.repeat(-(-tokens // 1024), 1)[:tokens].contiguous()
    probs, indices = st.sigmoid_topk_routing(logits, bias, topk, 2.5, batch_invariant=True)
    boundary = 2**31 // num_experts
    rows = sorted({0, boundary - 1, min(boundary, tokens - 1), tokens - 2, tokens - 1})
    rows_t = torch.tensor(rows, device=_DEVICE)
    check_routing(probs[rows_t], indices[rows_t], logits[rows_t], bias, topk, 2.5, f"rows {rows}")
