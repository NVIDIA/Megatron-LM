# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `fused_recurrent_gated_delta_rule_fwd_kernel` (megatron/core/ssm/ops/gdp/fused_recurrent.py),
launched by `fused_recurrent_gated_delta_rule_update`: the GDP and GDN decode step, with speculative
snapshots.

Contract (written from the gated delta rule, the launcher's documented arguments and its call sites
`GatedDeltaProductMixer.ssm_decode` (megatron/core/ssm/gated_delta_product.py, after `gdp_decode_prepare`)
and `GatedDeltaNet.ssm_decode` (megatron/core/ssm/gated_delta_net/gdn.py), before reading the kernel):

* For each request n (slot s_n = state_indices[n], or n without indices) and value head hv (query/key head
  hv // (HV / H)), from h = state[s_n, hv] (K x V, carried in fp32), for t = 0 .. T-1:
  q_t, k_t L2-normalised (x / sqrt(|x|^2 + 1e-6)) if `use_qk_l2norm_in_kernel`; beta_t = sigmoid(beta_t) if
  `use_beta_sigmoid_in_kernel`; the log decay d_t = g_t, or -exp(A_log[hv]) * softplus(g_t + dt_bias[hv])
  when `A_log` is given (GDN's in-kernel gate; softplus is the identity above 20);
  h <- exp(d_t) h;  u = beta_t (v_t - h^T k_t);  h <- h + k_t u^T;  o_t = scale * h^T q_t
  (scale defaults to K^-0.5). With `compact_output`, o holds only the output of every `steps_per_token`-th
  step (the last of each group). With `intermediate_states`, snapshot[s_n, i] = h after step
  (i + 1) * steps_per_token - 1 (after draft token i's M Householder updates). After the last step,
  state[s_n, hv] = h, rounded once to the state dtype; o in v's dtype; snapshots in their buffer's dtype.
* GDP folds its M Householder copies into the step dimension: T = S * M for S draft tokens, the query on
  the last copy and the decay on the first (zeros elsewhere), `steps_per_token = M`, compact output.
* Padding: a request with s_n < 0 gets o = +0 for every output and reads and writes no state and no
  snapshot, whatever its inputs hold (NaN). Slots no request maps to are unchanged bit for bit. q, k, v, g,
  beta, A_log and dt_bias are read-only.
* Preconditions (asserted by the launcher): state [slots, HV, K, V] with contiguous (K, V); snapshots
  [slots, T / M, HV, K, V] with contiguous (K, V); `state_indices` requires `state` and no `cu_seqlens`;
  `compact_output` requires T % M == 0. Unchecked: live slots unique; the dense state of static batching
  (no indices) contiguous per request.

Tolerance class: floating point. The fp64 reference is `delta_rule.delta_rule`, a step-by-step loop; the
trusted implementation is the same loop in torch fp32 on the same device, rounded to each output's dtype
(bound factor 2; 3 for fp32 states and snapshots, `FP32_STATE_FACTOR`: Triton's approximate exp). Padding outputs, untouched slots and snapshot rows, read-only inputs: exact.

Variants and dispatch (read from the kernel after the contract): heuristics USE_G, USE_INITIAL_STATE,
STORE_FINAL_STATE, HAS_STATE_INDICES, STORE_INTERMEDIATE, IS_VARLEN, USE_GATE_IN_KERNEL, HAS_DT_BIAS;
constexprs USE_QK_L2NORM_IN_KERNEL, IS_BETA_HEADWISE, APPLY_BETA_SIGMOID, COMPACT_OUTPUT, STEPS_PER_TOKEN and
PREFETCH_NEXT (= T > steps_per_token: the speculative path that prefetches the next step; #7754, newer than
the audit record); BK = next_pow2(K) (masked when K is not a power of two), BV = min(8, next_pow2(V)), grid
(cdiv(V, BV), N * HV); the `slot < 0` early exit; the in-kernel softplus (inline PTX on NVIDIA, identity
above 20). Covered: GDP decode (M 1, 2, 3; S 1, 2, 4 with and without snapshots; int64, int32 and no
indices), GDN decode (in-kernel gate, beta sigmoid, grouped heads, L2 norm on and off, dt_bias fp32 / bf16 /
none, A_log fp32 / bf16), and the launcher's other variants (no decay, per-channel beta, no initial state,
dense initial and final state, varlen `cu_seqlens`, explicit scale). Sizes: K 16, 64, 96 (masked BK), 128
and 256; V 7, 8, 9 (around BV = 8), 64 and 128; HV 1 to 32; decays from the mixers' gate in three regimes
(`delta_rule.G_REGIMES`: production, #8926's clamp on, and off with decays down to -1e4); the softplus
threshold (19.5, 20, 20.5, 90, 1000). Production: the GDP 800m functional config (32 heads, K 128, V 64,
M 3; tests/functional_tests/test_cases/hybrid/hybrid_gdp_dynamic_inference_*_800m) at 64 requests with S 1
and S 4 (snapshots), and the GDN defaults of TransformerConfig (16 key heads, 32 value heads, K = V = 128,
the Qwen3-Next shape) at 64 requests.

Memory: state and snapshot buffers in guard bands; unmapped slots, unwritten snapshot rows and padding
requests' inputs NaN-poisoned; `sanitizer` on a small GDP case with padding and snapshots and a GDN case.
int32: dynamic slot offsets are int64 by design; `test_state_offsets_past_int32` (`large`) puts a request
at a slot whose state and snapshot offsets cross 2^31 elements. Not covered: the static (no-index) path's
`i_nh * K * V` offset at 2^31 (a static batch of thousands of requests, not realistic).

Suspected finding O7b-M1 (GDN decode, no fix PR): with `A_log` the kernel forms softplus as
log(1 + exp(x)) in fp32 (inline PTX `ex2.approx`/`lg2.approx` on NVIDIA, `tl.math.log(1 + tl.math.exp(x))`
elsewhere), whose absolute error of ~2^-24 is a relative error of ~1e-4 at production gate arguments
(x = a + dt_bias ~ -7); exp(A_log) up to 16 carries it into the decay. GDN prefill forms the same gate with
torch's softplus. With an fp32 state that is 9.5-11.7x torch's fp32 state error on H100 (pdx 7987035; ~8x
in the Triton interpreter), so `test_gdn_decode_fp32_state` and `test_production_decode_shapes[gdn-fp32_state]` are
expected to fail on main on the state check; their outputs (bf16) and every bf16-state case pass. It is the
GDN analog of O7-N1 (Mamba decode softplus).
"""

from dataclasses import dataclass, field, replace
from typing import Optional

import pytest
import torch

from megatron.core.ssm.ops.gdp.fused_recurrent import fused_recurrent_gated_delta_rule_update
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.gdp import delta_rule as dr

DEVICE = "cuda"
BV_MAX = 8
"""The launcher's value tile: BV = min(8, next_pow2(V))."""

# Rows of the indexed cases: row 2 is CUDA-graph padding (-1); slots 1, 3, 4 and 7 (the last slot, where a
# host gather of -1 lands) belong to no row.
INDEXED_SLOTS = (5, 2, -1, 0, 6)
SLOT_COUNT = 8

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

FP32_STATE_FACTOR = 3.0
"""Bound factor for fp32 states and snapshots (bf16 ones keep 2: their final rounding dominates).

The kernel decays the whole carried state each step by `exp(g)`, and Triton's `tl.exp` is the
approximate `ex2.approx.f32` (up to 2 ulp), where the trusted torch fp32 loop's `exp` is accurate. When a
large carried state dominates the step's update, that approximation sets the fp32 state's error: in
`test_dense_initial_and_final_state[h0-dense]` (initial states N(0, 1), four steps) the kernel measured
2.08x the trusted rms error on H100 (pdx 7987035), every other fp32-state case <= 2x. In the Triton
interpreter (IEEE exp) the same case measures 1.6x, and 3.1-3.7x with a 1-ulp-rms error injected into
exp (1.3-1.75x into sqrt and division instead). Wiring errors are O(1); O7b-M1's softplus error measured
9.5-11.7x on H100, so its cases still fail at this factor."""


@dataclass
class Case:
    """One decode launch."""

    mode: str = "gdp"  # gdp | gdn | plain
    M: int = 3  # steps per token (GDP Householder copies)
    S: int = 1  # draft tokens per request
    H: int = 4  # query/key heads
    HV: int = 4  # value heads
    K: int = 64
    V: int = 64
    indices: str = "int64"  # int64 | int32 | none
    snapshots: bool = False
    state: str = "fp32"
    io: str = "bf16"
    regime: str = "production"
    l2norm: bool = True
    A_log: str = "fp32"  # gdn: fp32 | bf16
    dt_bias: Optional[str] = "fp32"  # gdn: fp32 | bf16 | None
    rows: tuple = INDEXED_SLOTS
    slot_count: int = SLOT_COUNT
    nan_row: Optional[int] = None  # live row whose inputs hold a NaN
    extra: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return self.S * self.M


def _params(cases):
    return [pytest.param(c, id=cid) for cid, c in cases]


def _gdp_id(c):
    snap = "-snap" if c.snapshots else ""
    return (
        f"M{c.M}-S{c.S}-HV{c.HV}-K{c.K}-V{c.V}-{c.indices}{snap}-{c.state}state-{c.io}-{c.regime}"
    )


def _gdp_cases():
    factors = {
        "M": [1, 2, 3],
        "S": [1, 2, 4],
        "K": [16, 64, 96, 128, 256],
        "V": [7, 8, 9, 64],
        "HV": [1, 4],
        "slots": ["int64+snap", "int64", "int32+snap", "none"],
        "state": ["fp32", "bf16"],
        "io": ["bf16", "fp16"],
        "regime": list(dr.G_REGIMES),
    }
    out = []
    for f in ou.pairwise_cases(factors):
        indices, _, snap = f.pop("slots").partition("+")
        c = Case(mode="gdp", H=f["HV"], indices=indices, snapshots=bool(snap), **f)
        out.append((_gdp_id(c), c))
    return out


GDP_CASES = _gdp_cases()


def _gdn_cases():
    factors = {
        "heads": [(1, 1), (2, 4), (1, 8)],
        "K": [64, 128],
        "V": [9, 64, 128],
        "l2norm": [True, False],
        "dt_bias": ["fp32", "bf16", None],
        "A_log": ["fp32", "bf16"],
        "io": ["bf16", "fp16"],
        "indices": ["int64", "int32"],
    }
    out = []
    for f in ou.pairwise_cases(factors):
        H, HV = f.pop("heads")
        c = Case(mode="gdn", M=1, S=1, H=H, HV=HV, state="bf16", regime="production", **f)
        cid = (
            f"H{H}-HV{HV}-K{c.K}-V{c.V}-{'l2' if c.l2norm else 'nol2'}-dtb_{c.dt_bias}-Alog_{c.A_log}"
            f"-{c.io}-{c.indices}"
        )
        out.append((cid, c))
    return out


GDN_CASES = _gdn_cases()


# ---------------------------------------------------------------------------------------------
# One launch, checked against the contract
# ---------------------------------------------------------------------------------------------


def _inputs(c: Case, n: int, g):
    """Kernel inputs for `n` requests, as the mixers hand them over (see the module docstring)."""
    io = DTYPES[c.io]

    def rand(*shape, dtype=io):
        return torch.randn(*shape, generator=g, device=DEVICE).to(dtype)

    x = {}
    if c.mode == "gdp":
        q_tok = rand(n, c.S, c.H, c.K)
        k = rand(n, c.T, c.H, c.K)
        v = rand(n, c.T, c.HV, c.V)
        beta = torch.sigmoid(torch.randn(n, c.T, c.HV, generator=g, device=DEVICE)).to(io)
        g_tok = dr.log_decays((n, c.S, c.HV), c.regime, generator=g, device=DEVICE)
        q, _, _, decay, _ = dr.fold_householder(q_tok, k, v, g_tok, beta, c.M)
        x.update(q=q.contiguous(), k=k, v=v, g=decay.contiguous(), beta=beta)
    elif c.mode == "gdn":
        A_log, dt_bias = dr.gate_parameters(c.HV, generator=g, device=DEVICE)
        x.update(
            q=rand(n, c.T, c.H, c.K),
            k=rand(n, c.T, c.H, c.K),
            v=rand(n, c.T, c.HV, c.V),
            g=dr.gate_inputs((n, c.T, c.HV), c.regime, generator=g, device=DEVICE).to(io),
            beta=rand(n, c.T, c.HV),
            A_log=A_log.to(DTYPES[c.A_log]),
            dt_bias=None if c.dt_bias is None else dt_bias.to(DTYPES[c.dt_bias]),
        )
    elif c.mode == "plain":  # q on every step, optional per-channel beta, optional decay
        beta_shape = (n, c.T, c.HV, c.V) if c.extra.get("beta_per_channel") else (n, c.T, c.HV)
        x.update(
            q=rand(n, c.T, c.H, c.K),
            k=rand(n, c.T, c.H, c.K),
            v=rand(n, c.T, c.HV, c.V),
            g=(
                None
                if c.extra.get("no_decay")
                else dr.log_decays((n, c.T, c.HV), c.regime, generator=g, device=DEVICE)
            ),
            beta=torch.sigmoid(torch.randn(*beta_shape, generator=g, device=DEVICE)).to(io),
        )
    if not c.l2norm:
        # Without the in-kernel L2 norm, keys of unit expected norm: the delta rule contracts only for
        # beta |k|^2 <= 2, which is what a layer that turns the norm off feeds it.
        for name in ("q", "k"):
            x[name] = (x[name].float() * c.K**-0.5).to(x[name].dtype)
    if c.extra.get("qk_scale"):
        for name in ("q", "k"):
            x[name] = (x[name].float() * c.extra["qk_scale"]).to(x[name].dtype)
    return x


def _reference_fn(c: Case, scale):
    """The contract as a function of the inputs (dtype-generic: fp64 reference or fp32 trusted)."""
    record = [c.M * (i + 1) for i in range(c.S)]

    def fn(q, k, v, g, beta, h0, A_log=None, dt_bias=None):
        if c.mode == "gdn":
            decay, beta = dr.gate(g, A_log, dt_bias), torch.sigmoid(beta)
        else:
            decay = g
        return dr.delta_rule(
            q, k, v, beta, decay, h0, scale=scale, l2norm_qk=c.l2norm, record=record
        )

    return fn


def run_decode(c: Case, *, check=("o", "state", "snapshots"), g=None):
    """Launch the kernel for case `c` and check every output against the contract."""
    g = ou.make_generator(device=DEVICE) if g is None else g
    state_dtype = DTYPES[c.state]
    if c.indices == "none":
        rows = tuple(range(len([r for r in c.rows if r >= 0])))
        slot_count = len(rows)
    else:
        rows, slot_count = c.rows, c.slot_count
    n = len(rows)
    x = _inputs(c, n, g)
    live = [i for i, s in enumerate(rows) if s >= 0]
    pad = [i for i, s in enumerate(rows) if s < 0]
    for name in ("q", "k", "v", "g", "beta"):
        for i in pad:
            if x.get(name) is not None:
                ou.poison_(x[name][i])
    if c.nan_row is not None:
        x["v"][c.nan_row, 0, 0, 0] = float("nan")
    before = {k: t.clone() for k, t in x.items() if t is not None}

    init = torch.randn(slot_count, c.HV, c.K, c.V, generator=g, device=DEVICE).to(state_dtype)
    mapped = sorted({s for s in rows if s >= 0})
    unmapped = [s for s in range(slot_count) if s not in mapped]
    if unmapped:
        for s in unmapped:
            ou.poison_(init[s])
    state = ou.guarded_like(init)
    snaps = None
    if c.snapshots:
        snaps = ou.guarded_empty((slot_count, c.S, c.HV, c.K, c.V), state_dtype, DEVICE)
    indices = None
    if c.indices != "none":
        indices = torch.tensor(rows, dtype=getattr(torch, c.indices), device=DEVICE)

    scale = c.extra.get("scale")
    o, final = fused_recurrent_gated_delta_rule_update(
        x["q"],
        x["k"],
        x["v"],
        g=x.get("g"),
        beta=x["beta"],
        scale=scale,
        state=state.tensor,
        state_indices=indices,
        use_qk_l2norm_in_kernel=c.l2norm,
        intermediate_states=None if snaps is None else snaps.tensor,
        steps_per_token=c.M,
        compact_output=c.mode == "gdp",
        A_log=x.get("A_log"),
        dt_bias=x.get("dt_bias"),
        use_beta_sigmoid_in_kernel=c.mode == "gdn",
    )
    ou.check_guards(state=state, **({} if snaps is None else {"snapshots": snaps}))
    assert final is state.tensor, "with a state cache, the returned final state is that cache"
    for name, t in before.items():
        ou.assert_exact(x[name], t, what=f"{name} is read-only")

    scale = c.K**-0.5 if scale is None else scale
    fn = _reference_fn(c, scale)
    live_slots = [rows[i] for i in live]
    args = [None if x.get(k_) is None else x[k_][live] for k_ in ("q", "k", "v", "g", "beta")]
    args.append(init[live_slots])
    kw = {}
    if c.mode == "gdn":
        kw = dict(A_log=x["A_log"], dt_bias=x["dt_bias"])
    ref = ou.fp64_reference(fn, *args, **kw)
    trusted = dr.trusted_fp32(fn, *args, **kw)
    sel = slice(c.M - 1, None, c.M) if c.mode == "gdp" else slice(None)
    out_dtype = x["v"].dtype
    label = f"{c.mode} decode"

    ou.assert_exact(o[pad], torch.zeros_like(o[pad]), what="padding requests' outputs (+0)")
    if "o" in check:
        ou.assert_close_to_reference(
            o[live], ref[0][:, sel], trusted[0][:, sel].to(out_dtype), what=f"{label} output o"
        )
    for s in unmapped:
        ou.assert_poisoned(state.tensor[s], what=f"unmapped slot {s}")
        if snaps is not None:
            ou.assert_poisoned(snaps.tensor[s], what=f"snapshots of unmapped slot {s}")
    if "state" in check:
        ou.assert_close_to_reference(
            state.tensor[live_slots],
            ref[1],
            trusted[1].to(state_dtype),
            factor=FP32_STATE_FACTOR if state_dtype == torch.float32 else ou.DEFAULT_FACTOR,
            what=f"{label} final state",
        )
    if snaps is not None and "snapshots" in check:
        for i in range(c.S):
            step = c.M * (i + 1)
            ou.assert_close_to_reference(
                snaps.tensor[live_slots, i],
                ref[2][step],
                trusted[2][step].to(state_dtype),
                factor=FP32_STATE_FACTOR if state_dtype == torch.float32 else ou.DEFAULT_FACTOR,
                what=f"{label} snapshot after draft token {i}",
            )
    return o, state.tensor, ref


# ---------------------------------------------------------------------------------------------
# GDP decode
# ---------------------------------------------------------------------------------------------


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", _params(GDP_CASES))
def test_gdp_decode_matches_fp64(case):
    """GDP decode as `ssm_decode` launches it (folded copies, compact output, L2 norm in the kernel),
    pairwise over M, S, K, V, heads, index dtype and snapshots, state and activation dtypes, decay regime.
    """
    run_decode(case)


def test_gdp_case_list_covers_every_branch():
    """The pairwise list reaches every branch the GDP call site can select (CPU only)."""
    seen = {
        "prefetch": {c.T > c.M for _, c in GDP_CASES},
        "masked BK": {c.K & (c.K - 1) != 0 for _, c in GDP_CASES},
        "V tiles": {(c.V + BV_MAX - 1) // BV_MAX > 1 for _, c in GDP_CASES},
        "V masked": {c.V % min(BV_MAX, 1 << (c.V - 1).bit_length()) != 0 for _, c in GDP_CASES},
        "indices": {c.indices for _, c in GDP_CASES},
        "snapshots x S>1": {c.snapshots and c.S > 1 for _, c in GDP_CASES},
        "regimes": {c.regime for _, c in GDP_CASES},
        "state": {c.state for _, c in GDP_CASES},
    }
    assert seen["prefetch"] == {True, False}
    assert seen["masked BK"] == {True, False}
    assert seen["V tiles"] == {True, False}
    assert seen["V masked"] == {True, False}
    assert seen["indices"] == {"int64", "int32", "none"}
    assert True in seen["snapshots x S>1"]
    assert seen["regimes"] == set(dr.G_REGIMES)
    assert seen["state"] == {"fp32", "bf16"}


# ---------------------------------------------------------------------------------------------
# GDN decode (in-kernel gate and beta sigmoid)
# ---------------------------------------------------------------------------------------------


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", _params(GDN_CASES))
def test_gdn_decode_matches_fp64(case):
    """GDN decode as `GatedDeltaNet.ssm_decode` launches it, with a bf16 state (the inference default):
    grouped heads, L2 norm on and off, dt_bias and A_log dtypes, index dtypes."""
    run_decode(case)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "case",
    _params(
        [
            ("H2-HV4-K128-V128", Case(mode="gdn", M=1, S=1, H=2, HV=4, K=128, V=128, state="fp32")),
            (
                "H1-HV8-K64-V9-nol2",
                Case(mode="gdn", M=1, S=1, H=1, HV=8, K=64, V=9, l2norm=False, state="fp32"),
            ),
        ]
    ),
)
def test_gdn_decode_fp32_state(case):
    """GDN decode with an fp32 state (`--mamba-inference-ssm-states-dtype fp32`, used by the functional tests).

    Expected to fail on main at the final-state check: suspected finding O7b-M1 (the in-kernel softplus's
    absolute error, see the module docstring). The bf16 output is checked first and passes.
    """
    run_decode(case)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("x", [19.5, 20.0, 20.5, 90.0, 1000.0], ids=lambda x: f"x{x:g}")
def test_gdn_gate_saturates_above_20(x):
    """The in-kernel softplus is the identity above 20, so a large gate argument decays by exp(-A x), not
    inf/NaN. A = exp(A_log) is small (1e-3) so the decay stays visible in the state."""
    c = Case(mode="gdn", M=1, S=1, H=1, HV=2, K=64, V=64, state="bf16", indices="int64")
    g = ou.make_generator(device=DEVICE)
    n = len(c.rows)
    x_in = _inputs(c, n, g)
    x_in["A_log"] = torch.full_like(x_in["A_log"], -6.907755)  # A = 1e-3
    x_in["dt_bias"] = torch.zeros_like(x_in["dt_bias"])
    x_in["g"] = torch.full_like(x_in["g"], x)
    live = [i for i, s in enumerate(c.rows) if s >= 0]
    pad = [i for i, s in enumerate(c.rows) if s < 0]
    init = torch.randn(SLOT_COUNT, c.HV, c.K, c.V, generator=g, device=DEVICE).bfloat16()
    state = init.clone()
    indices = torch.tensor(c.rows, dtype=torch.int64, device=DEVICE)
    o, _ = fused_recurrent_gated_delta_rule_update(
        x_in["q"],
        x_in["k"],
        x_in["v"],
        g=x_in["g"],
        beta=x_in["beta"],
        state=state,
        state_indices=indices,
        use_qk_l2norm_in_kernel=True,
        A_log=x_in["A_log"],
        dt_bias=x_in["dt_bias"],
        use_beta_sigmoid_in_kernel=True,
    )
    fn = _reference_fn(c, c.K**-0.5)
    slots = [c.rows[i] for i in live]
    args = [x_in[k][live] for k in ("q", "k", "v", "g", "beta")] + [init[slots]]
    kw = dict(A_log=x_in["A_log"], dt_bias=x_in["dt_bias"])
    ref = ou.fp64_reference(fn, *args, **kw)
    trusted = dr.trusted_fp32(fn, *args, **kw)
    ou.assert_exact(o[pad], torch.zeros_like(o[pad]), what="padding output")
    ou.assert_close_to_reference(
        o[live], ref[0], trusted[0].bfloat16(), what=f"o at gate argument {x}"
    )
    ou.assert_close_to_reference(
        state[slots], ref[1], trusted[1].bfloat16(), what=f"state at gate argument {x}"
    )


# ---------------------------------------------------------------------------------------------
# The launcher's other variants
# ---------------------------------------------------------------------------------------------


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "case",
    _params(
        [
            ("no_decay", Case(mode="plain", M=1, S=3, extra={"no_decay": True})),
            (
                "beta_per_channel",
                Case(mode="plain", M=1, S=3, V=9, extra={"beta_per_channel": True}),
            ),
            (
                "gqa_plain_S5",
                Case(mode="plain", M=1, S=5, H=2, HV=8, K=32, V=16, regime="clamp_off"),
            ),
            ("explicit_scale", Case(mode="plain", M=1, S=2, extra={"scale": 0.37})),
            ("no_l2norm", Case(mode="plain", M=1, S=2, l2norm=False, state="bf16")),
        ]
    ),
)
def test_plain_recurrence_variants(case):
    """Variants the decode call sites do not select: no decay (USE_G off), per-channel beta, an output at
    every step with the query on every step, an explicit scale, no L2 norm."""
    run_decode(case)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "case",
    _params(
        [
            (
                "gdn-S1",
                Case(
                    mode="gdn",
                    M=1,
                    S=1,
                    H=2,
                    HV=4,
                    K=128,
                    V=16,
                    state="bf16",
                    extra={"qk_scale": 1e-3},
                ),
            ),
            (
                "gdp-S1",
                Case(mode="gdp", M=3, S=1, H=2, HV=2, K=128, V=16, extra={"qk_scale": 1e-3}),
            ),
            (
                "gdp-S2-prefetch",
                Case(mode="gdp", M=3, S=2, H=2, HV=2, K=128, V=16, extra={"qk_scale": 1e-3}),
            ),
        ]
    ),
)
def test_small_query_and_key_norms(case):
    """q and k a thousand times smaller (|x|^2 ~ 1e-4), where the L2 norm's epsilon (1e-6) is about 1% of
    the denominator: the epsilon must be the contract's, on the single-step and the prefetching loops.
    """
    run_decode(case)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("varlen", [False, True], ids=["dense", "varlen"])
@pytest.mark.parametrize("with_h0", [False, True], ids=["zeros", "h0"])
def test_dense_initial_and_final_state(varlen, with_h0):
    """The gather/scatter API (`initial_state`, `output_final_state`) and `cu_seqlens` varlen input, which
    the decode call sites never use: final_state [N, HV, K, V] fp32, from zeros when no initial state.
    """
    g = ou.make_generator(device=DEVICE)
    H, HV, K, V = 2, 4, 64, 16
    lens = [3, 1, 6] if varlen else [4, 4, 4]
    N = len(lens)
    T = sum(lens) if varlen else lens[0]
    B = 1 if varlen else N
    q = torch.randn(B, T, H, K, generator=g, device=DEVICE).bfloat16()
    k = torch.randn(B, T, H, K, generator=g, device=DEVICE).bfloat16()
    v = torch.randn(B, T, HV, V, generator=g, device=DEVICE).bfloat16()
    decay = dr.log_decays((B, T, HV), "production", generator=g, device=DEVICE)
    beta = torch.sigmoid(torch.randn(B, T, HV, generator=g, device=DEVICE)).bfloat16()
    h0 = torch.randn(N, HV, K, V, generator=g, device=DEVICE) if with_h0 else None
    cu = None
    if varlen:
        cu = torch.tensor(
            [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int64, device=DEVICE
        )
    o, final = fused_recurrent_gated_delta_rule_update(
        q,
        k,
        v,
        g=decay,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
    )
    assert final.dtype == torch.float32 and final.shape == (N, HV, K, V)
    starts = [0, *torch.tensor(lens).cumsum(0).tolist()]
    for n in range(N):
        if varlen:
            sl = (slice(0, 1), slice(starts[n], starts[n + 1]))
        else:
            sl = (slice(n, n + 1), slice(None))
        init = torch.zeros(1, HV, K, V, device=DEVICE) if h0 is None else h0[n : n + 1]

        def fn(q, k, v, d, b, h):
            return dr.delta_rule(q, k, v, b, d, h, scale=K**-0.5, l2norm_qk=True)

        args = (q[sl], k[sl], v[sl], decay[sl], beta[sl], init)
        ref = ou.fp64_reference(fn, *args)
        trusted = dr.trusted_fp32(fn, *args)
        ou.assert_close_to_reference(
            o[sl], ref[0], trusted[0].bfloat16(), what=f"o of sequence {n}"
        )
        ou.assert_close_to_reference(
            final[n : n + 1], ref[1], trusted[1], factor=FP32_STATE_FACTOR, what=f"final state {n}"
        )


# ---------------------------------------------------------------------------------------------
# Padding, isolation, production, memory
# ---------------------------------------------------------------------------------------------


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("mode", ["gdp", "gdn"])
def test_nan_in_one_request_stays_in_that_request(mode):
    """A NaN in one live request's value makes that request's output and state NaN, and nothing else."""
    if mode == "gdp":
        c = Case(mode="gdp", M=3, S=2, H=2, HV=2, K=64, V=16, snapshots=True, nan_row=0)
    else:
        c = Case(mode="gdn", M=1, S=1, H=1, HV=2, K=64, V=16, state="bf16", nan_row=0)
    o, state, ref = run_decode(c, check=())
    live = [i for i, s in enumerate(c.rows) if s >= 0]
    assert bool(torch.isnan(state[c.rows[0]]).any()), "the NaN request's state must be NaN"
    for i in live[1:]:
        assert bool(torch.isfinite(o[i]).all()), f"request {i} picked up another request's NaN"
        assert bool(torch.isfinite(state[c.rows[i]]).all()), f"slot {c.rows[i]} picked up a NaN"


PRODUCTION = {
    # GDP 800m functional config: hidden 1024, 32 heads x 64, d_state 128, 8 groups, M 3 (default).
    "gdp-S1": Case(mode="gdp", M=3, S=1, H=32, HV=32, K=128, V=64, state="fp32"),
    "gdp-S4-snap": Case(
        mode="gdp", M=3, S=4, H=32, HV=32, K=128, V=64, state="fp32", snapshots=True
    ),
    "gdp-S1-bf16_state": Case(mode="gdp", M=3, S=1, H=32, HV=32, K=128, V=64, state="bf16"),
    # GDN: TransformerConfig defaults (Qwen3-Next): 16 key heads, 32 value heads, K = V = 128.
    "gdn-bf16_state": Case(mode="gdn", M=1, S=1, H=16, HV=32, K=128, V=128, state="bf16"),
    "gdn-fp32_state": Case(mode="gdn", M=1, S=1, H=16, HV=32, K=128, V=128, state="fp32"),
}


def _production_rows(n_live=64, n_pad=8, slots=96):
    """64 live requests on scattered slots of 96, then 8 CUDA-graph padding rows."""
    perm = torch.randperm(slots, generator=ou.make_generator())[:n_live].tolist()
    return tuple(perm) + (-1,) * n_pad, slots


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("name", list(PRODUCTION))
def test_production_decode_shapes(name):
    """Production decode batches: 64 requests (and 8 padding rows) at the GDP 800m and GDN default shapes.
    `[gdn-fp32_state]` is expected to fail on main at the state check (O7b-M1)."""
    rows, slots = _production_rows()
    run_decode(replace(PRODUCTION[name], rows=rows, slot_count=slots))


@ou.requires_sm(ou.HOPPER)
@pytest.mark.sanitizer
@pytest.mark.parametrize("mode", ["gdp", "gdn"])
def test_sanitizer_small_case(mode):
    """Small padded cases for the compute-sanitizer pass: GDP with M 3, two draft tokens and snapshots
    (the prefetching loop); GDN with grouped heads and the in-kernel gate."""
    if mode == "gdp":
        run_decode(Case(mode="gdp", M=3, S=2, H=2, HV=2, K=96, V=9, snapshots=True))
    else:
        run_decode(Case(mode="gdn", M=1, S=1, H=1, HV=4, K=64, V=9, state="bf16"))


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("slots", ou.int32_overflow_params(128 * 128))
def test_state_offsets_past_int32(slots):
    """One request at the last of `slots` slots (HV 1, K = V = 128), so its state and snapshot offsets
    cross 2^31 elements: the slot offsets are int64 (bf16 state and snapshots, ~4 GiB each)."""
    ou.require_free_memory(2 * slots * 128 * 128 * 2 + 2**30)
    g = ou.make_generator(device=DEVICE)
    c = Case(mode="gdp", M=3, S=1, H=1, HV=1, K=128, V=128, state="bf16", snapshots=True)
    n = 2
    x = _inputs(c, n, g)
    state = torch.zeros(slots, 1, 128, 128, dtype=torch.bfloat16, device=DEVICE)
    snaps = torch.zeros(slots, 1, 1, 128, 128, dtype=torch.bfloat16, device=DEVICE)
    rows = [slots - 1, 1]
    init = torch.randn(2, 1, 128, 128, generator=g, device=DEVICE).bfloat16()
    state[rows] = init
    o, _ = fused_recurrent_gated_delta_rule_update(
        x["q"],
        x["k"],
        x["v"],
        g=x["g"],
        beta=x["beta"],
        state=state,
        state_indices=torch.tensor(rows, dtype=torch.int64, device=DEVICE),
        use_qk_l2norm_in_kernel=True,
        intermediate_states=snaps,
        steps_per_token=3,
        compact_output=True,
    )
    fn = _reference_fn(c, 128**-0.5)
    args = [x[k] for k in ("q", "k", "v", "g", "beta")] + [init]
    ref = ou.fp64_reference(fn, *args)
    trusted = dr.trusted_fp32(fn, *args)
    ou.assert_close_to_reference(o, ref[0][:, 2::3], trusted[0][:, 2::3].bfloat16(), what="o")
    ou.assert_close_to_reference(state[rows], ref[1], trusted[1].bfloat16(), what="state past 2^31")
    ou.assert_close_to_reference(
        snaps[rows, 0], ref[2][3], trusted[2][3].bfloat16(), what="snapshot past 2^31"
    )
    untouched = state[slots - 2]
    ou.assert_exact(untouched, torch.zeros_like(untouched), what="the slot before the last")
