# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the fork's `causal_conv1d_update_kernel` (megatron/core/ssm/ops/common/causal_conv1d_triton.py),
the decode-time depthwise causal conv of Mamba, GDN and GDP, through its launcher `causal_conv1d_update`,
including the speculative-decoding snapshots.

Before this module the kernel had only the unit tests of its own PR family (fp32, one shape) and end-to-end
checks; the static-decode oracle (test_mamba_static_decode.py) reaches it only without indices, at S = 1,
through the mixer.

Contract (written from the callers — `MambaMixer.ssm_decode`, `GatedDeltaProductMixer._decode_conv`,
`GatedDeltaNet.ssm_decode` — and the launcher's signature, before reading the kernel body):

* Inputs: x (B, S, D) — or (B, D), one token — in any layout (Mamba passes a split view of the projection);
  conv_state (slots, D, L), L >= W - 1, any strides; weight (D, W), bias (D,) or None; `silu_activation`,
  annotated `bool` (True: SiLU), and passed by every caller as the activation's name ("silu");
  conv_state_indices (B,) int32 or int64 (row b uses slot idx[b]; -1 marks a padding row), or None (row b
  uses slot b); intermediate_conv_states (slots, S, D, L), indexed by slot, or None.
* For a row b with slot s >= 0 and each token t = 0..S-1 in order:

      window_t   = the last L columns of [conv_state[s], x[b, 0], ..., x[b, t]]   (shift left, append)
      out[b, t]  = act(bias + sum_{j < W} weight[:, j] * window_t[:, L - W + j])
      snapshot[s, t] = window_t                                     (with intermediate_conv_states)

  and conv_state[s] = window_{S-1} afterwards. Appended values are x in the state's dtype (callers pass x
  in the state dtype, or narrower); `out` has x's dtype and shape. A padding row gets out[b] = 0 and
  writes no state or snapshot; slots no row maps to are untouched; x, weight, bias and the indices are
  read-only.
* Numerics: fp32 accumulation of at most W products, SiLU in fp32, one rounding to x's dtype.
* Tolerance class: floating point for `out` (reference: fp64; trusted: the same math in torch fp32 on the
  device); exact for the state, the snapshots, padding rows and untouched slots (copies), and for `out`
  with dyadic inputs and no activation (every product and sum exact in fp32).

Variants and dispatch (from the kernel, read after the contract): constexpr WIDTH 2, 3, 4 (production 4);
HAS_BIAS; HAS_STATE_INDICES (none, int32, int64 — Mamba/GDN/GDP decode pass int64 outside batch-invariant
mode); HAS_INT_STATE; SILU_ACTIVATION; the runtime branch `state_len == WIDTH` (the register fast path,
production) vs the generic shift loop (L = W - 1 and L = W + 3); BLOCK_DIM = 64 over D: D 63, 64, 65, 129
and production 10240 (the Nemotron-H 8B conv dim), 4096 + 2 * 8 * 128; S 1, 2, 4 (speculative tokens);
2-D x; x as a strided split view; a strided conv_state view. bf16, fp32, and bf16 x with an fp32 state.

Finding O7-M2: on main the launcher computes `SILU_ACTIVATION = silu_activation == "silu"`
(causal_conv1d_triton.py:319) although the parameter is annotated `bool`: `silu_activation=True` silently
applies no activation (the determinism test test_ssm_kernels.py passes True). Every production caller passes
"silu", which works, and so does every case here. #9308 (fix/conv1d-update-silu-flag) accepts a bool or the
activation's name and carries `True` as its regression test
(tests/unit_tests/ssm/ops/common/test_causal_conv1d_triton.py,
`TestCausalConv1dUpdate::test_activation_forms`).

Main vs open PRs: #9308 (above) changes how the launcher reads the flag; "silu" and None keep their meaning.
#9320 (fix/mamba-decode-int64-indices) forms the kernel's batch and slot offsets in int64. #8646 (on main)
only changes what the Mamba call site passes (cached FP32 weight and bias for an FP32 conv state); the
kernel's contract is unchanged.
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.ssm.ops.common.causal_conv1d_triton import causal_conv1d_update
from tests.unit_tests.inference.oracles import oracle_utils as ou

DEVICE = "cuda"
BLOCK_DIM = 64  # causal_conv1d_triton.py launcher
PRODUCTION_DIM = (
    4096 * 2 + 2 * 8 * 128
)  # Nemotron-H 8B at TP 1: d_inner 8192 + 2 * ngroups * dstate


# ---------------------------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------------------------


def conv_update_reference(x, conv_state, weight, bias, slots, *, silu, state_dtype=None):
    """The contract for every row, in the inputs' dtype: (out (B, S, D), new state per row (B, D, L),
    snapshots (B, S, D, L)); padding rows (slot < 0) give zeros and no states (None entries).

    `state_dtype`: the dtype appended values are stored in (the conv state's), applied to the window.
    """
    batch, seq_len, dim = x.shape
    width = weight.shape[1]
    out = torch.zeros_like(x)
    states, snaps = [], []
    for b in range(batch):
        s = int(slots[b])
        if s < 0:
            states.append(None)
            snaps.append(None)
            continue
        window = conv_state[s]
        row_snaps = []
        for t in range(seq_len):
            token = x[b, t]
            if state_dtype is not None:
                token = token.to(state_dtype).to(x.dtype)
            full = torch.cat([window, token[:, None]], dim=1)  # L + 1 columns
            taps = full[:, -width:]  # the last W inputs (L = W - 1 needs the oldest column too)
            window = full[:, 1:]
            acc = (taps * weight).sum(-1) + (0 if bias is None else bias)
            out[b, t] = F.silu(acc) if silu else acc
            row_snaps.append(window)
        states.append(window)
        snaps.append(torch.stack(row_snaps))
    return out, states, snaps


# ---------------------------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------------------------


def run_update(
    *,
    batch,
    seq_len,
    dim,
    width,
    state_len,
    x_dtype=torch.bfloat16,
    state_dtype=torch.bfloat16,
    bias=True,
    indices="int64",
    snapshots=False,
    silu=True,
    two_d=False,
    strided_x=False,
    strided_state=False,
    padding=(1,),
    extra_slots=3,
    exact=False,
):
    """One launch of `causal_conv1d_update` on random (or dyadic) inputs, checked against the contract."""
    g = ou.make_generator(device=DEVICE)
    nslots = batch + extra_slots
    if indices is None:
        slots = torch.arange(batch, device=DEVICE)
        index_tensor = None
    else:
        slots = torch.randperm(nslots, device="cpu", generator=torch.Generator().manual_seed(3))[
            :batch
        ]
        slots = slots.to(DEVICE)
        for p in padding:
            if p < batch:
                slots[p] = -1
        index_tensor = slots.to(torch.int32 if indices == "int32" else torch.int64)

    def values(*shape, dtype):
        if exact:
            return ou.dyadic_tensor(shape, dtype=dtype, device=DEVICE, generator=g)
        return torch.randn(*shape, device=DEVICE, generator=g).to(dtype)

    if strided_x:
        # x as Mamba passes it: the xBC columns of a (B, S, z + xBC + dt) projection.
        proj = values(batch, seq_len, 2 * dim + 7, dtype=x_dtype)
        x = proj[..., dim + 3 : 2 * dim + 3]
    else:
        x = values(batch, seq_len, dim, dtype=x_dtype)
    weight = values(dim, width, dtype=state_dtype)
    bias_t = values(dim, dtype=state_dtype) if bias else None
    state0 = values(nslots, dim, state_len, dtype=state_dtype)
    # The cache in a guard band; a strided view of it when requested (state stored (slots, L, D)).
    if strided_state:
        cache = ou.guarded_like(
            state0.transpose(1, 2).contiguous(), guard_bytes=dim * state_len * 8
        )
        conv_state = cache.tensor.transpose(1, 2)
    else:
        cache = ou.guarded_like(state0, guard_bytes=dim * state_len * 8)
        conv_state = cache.tensor
    snaps = None
    if snapshots:
        snaps = ou.guarded_empty((nslots, seq_len, dim, state_len), state_dtype, DEVICE)
    x_in = x[:, 0] if two_d else x
    if two_d:
        assert seq_len == 1
    inputs = dict(x=x_in, weight=weight)
    if bias_t is not None:
        inputs["bias"] = bias_t
    if index_tensor is not None:
        inputs["indices"] = index_tensor
    before = {k: v.clone() for k, v in inputs.items()}

    out = causal_conv1d_update(
        x_in,
        conv_state,
        weight,
        bias_t,
        "silu" if silu else None,
        index_tensor,
        intermediate_conv_states=None if snaps is None else snaps.tensor,
    )

    ou.check_guards(conv_state=cache, **({} if snaps is None else {"snapshots": snaps}))
    for name, value in before.items():
        ou.assert_exact(inputs[name], value, what=f"{name} is read-only")
    assert out.shape == x_in.shape and out.dtype == x_dtype
    out3 = out[:, None] if two_d else out

    ref_out, ref_states, ref_snaps = conv_update_reference(
        x.double(),
        state0.double(),
        weight.double(),
        None if bias_t is None else bias_t.double(),
        slots,
        silu=silu,
        state_dtype=state_dtype,
    )
    live = [b for b in range(batch) if int(slots[b]) >= 0]
    dead = [b for b in range(batch) if b not in live]
    if dead:
        ou.assert_exact(out3[dead], torch.zeros_like(out3[dead]), what="padding rows' output")
    # States and snapshots are copies: exact.
    mapped = [int(slots[b]) for b in live]
    for b in live:
        s = int(slots[b])
        ou.assert_exact(
            conv_state[s], ref_states[b].to(state_dtype), what=f"conv state of slot {s} (row {b})"
        )
        if snaps is not None:
            ou.assert_exact(
                snaps.tensor[s], ref_snaps[b].to(state_dtype), what=f"snapshots of slot {s}"
            )
    unmapped = [s for s in range(nslots) if s not in mapped]
    ou.assert_exact(conv_state[unmapped], state0[unmapped], what="slots no row maps to")
    if snaps is not None:
        ou.assert_poisoned(snaps.tensor[unmapped], what="snapshots of slots no row maps to")
    if not live:
        return out
    got, ref = out3[live], ref_out[live]
    if exact:
        # + 0.0: a zero sum's sign carries no meaning.
        ou.assert_exact(got + 0.0, ref.to(x_dtype) + 0.0, what="conv output (exact inputs)")
        return out
    t_out, _, _ = conv_update_reference(
        x.float(),
        state0.float(),
        weight.float(),
        None if bias_t is None else bias_t.float(),
        slots,
        silu=silu,
        state_dtype=state_dtype,
    )
    ou.assert_close_to_reference(
        got.flatten(0, 1),
        ref.flatten(0, 1),
        t_out[live].to(x_dtype).flatten(0, 1),
        what="conv output",
    )
    return out


# ---------------------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------------------


def _cases():
    factors = {
        "width": (4, 2, 3),
        "state_len": ("W", "W-1", "W+3"),  # fast path, generic loop at its two extremes
        "seq_len": (1, 2, 4),
        "dim": (BLOCK_DIM, BLOCK_DIM - 1, BLOCK_DIM + 1, 2 * BLOCK_DIM + 1),
        "dtypes": ("bf16", "fp32", "bf16_x_fp32_state"),
        "bias": (True, False),
        "indices": ("int64", "int32", None),
        "snapshots": (False, True),
        "silu": (True, False),
        "layout": ("contiguous", "strided_x", "strided_state"),
    }
    params = []
    for case in ou.pairwise_cases(factors):
        case_id = "-".join(
            [
                f"W{case['width']}",
                f"L{case['state_len']}",
                f"S{case['seq_len']}",
                f"D{case['dim']}",
                case["dtypes"],
                "bias" if case["bias"] else "nobias",
                f"idx_{case['indices']}",
                "snap" if case["snapshots"] else "nosnap",
                "silu" if case["silu"] else "linear",
                case["layout"],
            ]
        )
        params.append(pytest.param(case, id=case_id))
    return params


DTYPES = {
    "bf16": (torch.bfloat16, torch.bfloat16),
    "fp32": (torch.float32, torch.float32),
    "bf16_x_fp32_state": (torch.bfloat16, torch.float32),
}


def _kwargs(case):
    width = case["width"]
    state_len = {"W": width, "W-1": width - 1, "W+3": width + 3}[case["state_len"]]
    x_dtype, state_dtype = DTYPES[case["dtypes"]]
    return dict(
        batch=5,
        seq_len=case["seq_len"],
        dim=case["dim"],
        width=width,
        state_len=state_len,
        x_dtype=x_dtype,
        state_dtype=state_dtype,
        bias=case["bias"],
        indices=case["indices"],
        snapshots=case["snapshots"],
        silu=case["silu"],
        strided_x=case["layout"] == "strided_x",
        strided_state=case["layout"] == "strided_state",
    )


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", _cases())
def test_conv_update_matches_fp64(case):
    """Pairwise over every compiled variant, the fast-path / generic branch, sizes around BLOCK_DIM,
    dtypes and layouts; padding rows and unmapped slots whenever there are indices."""
    run_update(**_kwargs(case))


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("state_len_kind", ["W", "W+3"])
def test_conv_update_exact_on_dyadic_inputs(width, state_len_kind):
    """No activation and dyadic inputs: every output is the exact sum (at most 5 terms of 8 significant
    bits), so each tap's position, the shift and the snapshots are checked bit for bit, over 4 tokens.
    """
    state_len = width if state_len_kind == "W" else width + 3
    run_update(
        batch=6,
        seq_len=4,
        dim=BLOCK_DIM + 1,
        width=width,
        state_len=state_len,
        x_dtype=torch.float32,
        state_dtype=torch.float32,
        snapshots=True,
        silu=False,
        exact=True,
    )


@ou.requires_sm(ou.HOPPER)
def test_one_token_two_d_input():
    """2-D x (one token per row, the GDN/Mamba S = 1 layout) keeps its shape."""
    run_update(batch=7, seq_len=1, dim=129, width=4, state_len=4, two_d=True, padding=(0, 6))


@ou.requires_sm(ou.HOPPER)
def test_all_rows_padding():
    """A decode graph whose rows are all padding: zero output, no state written."""
    run_update(
        batch=4, seq_len=2, dim=65, width=4, state_len=4, padding=(0, 1, 2, 3), snapshots=True
    )


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "seq_len,snapshots,state_dtype",
    [
        pytest.param(1, False, torch.bfloat16, id="decode-bf16_state"),
        pytest.param(1, False, torch.float32, id="decode-fp32_state"),
        pytest.param(4, True, torch.bfloat16, id="mtp3-snapshots"),
    ],
)
def test_production_shapes(seq_len, snapshots, state_dtype):
    """Production: the Nemotron-H 8B layer's conv (D 10240, W 4, L 4, bias, SiLU) at a decode batch of
    64 with int64 slot indices and padding rows, x a split view of the projection; one token, and four
    (three MTP drafts) with snapshots. An fp32 conv state takes x, weight and bias in fp32 (#8646's
    cached FP32 parameters)."""
    run_update(
        batch=64,
        seq_len=seq_len,
        dim=PRODUCTION_DIM,
        width=4,
        state_len=4,
        x_dtype=state_dtype,
        state_dtype=state_dtype,
        snapshots=snapshots,
        strided_x=True,
        padding=(5, 63),
        extra_slots=64,
    )


@pytest.mark.parametrize("slots_total", ou.int32_overflow_params(PRODUCTION_DIM * 4))
@ou.requires_sm(ou.HOPPER)
def test_state_offsets_past_int32(slots_total):
    """int64 slot indices (production) whose conv-state offsets (slot * D * L) reach 2^31 elements: the
    last slot of a 10240 x 4 bf16 cache of `slots_total` slots (4.3 GB) is read and written correctly.
    """
    ou.require_free_memory(16 * 2**30)
    g = ou.make_generator(device=DEVICE)
    dim, width = PRODUCTION_DIM, 4
    cache = torch.randn(slots_total, dim, width, device=DEVICE, generator=g).bfloat16()
    slots = torch.tensor([slots_total - 1, 0, -1, slots_total - 2], device=DEVICE)
    x = torch.randn(4, 1, dim, device=DEVICE, generator=g).bfloat16()
    weight = torch.randn(dim, width, device=DEVICE, generator=g).bfloat16()
    bias = torch.randn(dim, device=DEVICE, generator=g).bfloat16()
    live = [0, 1, 3]
    before = cache[slots[live]].clone()
    out = causal_conv1d_update(x, cache, weight, bias, "silu", slots)
    ref_out, ref_states, _ = conv_update_reference(
        x.double(),
        before.double(),
        weight.double(),
        bias.double(),
        torch.tensor([0, 1, -1, 2]),
        silu=True,
        state_dtype=torch.bfloat16,
    )
    trusted, _, _ = conv_update_reference(
        x.float(),
        before.float(),
        weight.float(),
        bias.float(),
        torch.tensor([0, 1, -1, 2]),
        silu=True,
        state_dtype=torch.bfloat16,
    )
    for b in live:
        ou.assert_exact(
            cache[int(slots[b])], ref_states[b].bfloat16(), what=f"slot {int(slots[b])}"
        )
    ou.assert_exact(out[2], torch.zeros_like(out[2]), what="padding row")
    ou.assert_close_to_reference(
        out[live].flatten(0, 1),
        ref_out[live].flatten(0, 1),
        trusted[live].bfloat16().flatten(0, 1),
        what="output",
    )


@pytest.mark.sanitizer
@ou.requires_sm(ou.HOPPER)
def test_sanitizer_small_case():
    """Small case for compute-sanitizer: D 65 (a partial block), the generic shift (L = W + 3), int64
    indices with a padding row, snapshots, two tokens."""
    run_update(batch=4, seq_len=2, dim=65, width=4, state_len=7, snapshots=True)


def test_case_list_covers_every_variant():
    """The pairwise sweep reaches every width, branch, index dtype and layout (CPU check)."""
    cases = [p.values[0] for p in _cases()]
    for factor in ("width", "seq_len", "dim", "dtypes", "bias", "silu", "layout"):
        assert len({str(c[factor]) for c in cases}) > 1
    assert {c["state_len"] for c in cases} == {"W", "W-1", "W+3"}
    assert {c["indices"] for c in cases} == {"int64", "int32", None}
    assert {c["snapshots"] for c in cases} == {False, True}
