# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `_selective_scan_update_kernel` (megatron/core/ssm/ops/mamba2/mamba_ssm.py:64), launched by
`selective_state_update` (:285): one Mamba-2 decode step per request, with speculative-decoding snapshots.

Contract (written from the Mamba-2 recurrence, the launcher's documented API and its call sites
`MambaMixer.ssm_decode`, megatron/core/ssm/mamba_mixer.py:1271-1297, and `mamba_state_selective_copy`,
megatron/core/inference/text_generation_controllers/mtp_utils_triton.py:419, before reading the kernel body):

* For each batch row r with slot s_r = state_batch_indices[r] (s_r = r without indices), each head h with
  group g(h) = h // (nheads / ngroups), and the row's tokens t = 1..S in order, starting from
  h_0 = state[s_r, h] (shape (dim, dstate)):

      dt'_t[p]   = softplus(dt[r,t,h,p] + dt_bias[h,p])     (dt + dt_bias without dt_softplus)
      h_t[p,n]   = exp(A[h,p,n] * dt'_t[p]) * h_{t-1}[p,n] + dt'_t[p] * B[r,t,g(h),n] * x[r,t,h,p]
      y_t[p]     = sum_n C[r,t,g(h),n] * h_t[p,n] + D[h,p] * x[r,t,h,p]
      out[r,t,h,p] = y_t[p] * z * sigmoid(z)                 (z = z[r,t,h,p]; no gate without z)
      intermediate_ssm_states[s_r, t, h] = h_t               (if the snapshot buffer is given)

  After the last token, state[s_r, h] = h_S. All of it in fp32 or better.
* Inputs: state (slots, nheads, dim, dstate) or (slots, dim, dstate); x, dt, z (batch, S, nheads, dim) or
  (batch, nheads, dim) / (batch, S, dim) / (batch, dim); A (nheads, dim, dstate) or (dim, dstate); B, C
  (batch, S, ngroups, dstate) or the matching reduced forms; D, dt_bias (nheads, dim) or (dim,);
  state_batch_indices (batch,) int32 or int64 (int64 in production non-batch-invariant decode);
  intermediate_ssm_states (slots, S, nheads, dim, dstate), indexed by slot. Any strides: `ssm_decode`
  passes split views of one projection, and A, dt, dt_bias as stride-0 expands of per-head values.
* Outputs / in-place effects: `out` (shape and dtype of x), state[s_r] updated in place (rounded once to the
  state dtype), snapshots rounded to their dtype. Read-only: every other input (regression for #6598, which
  corrupted x[0, 0, 0, 0]).
* Padding: a row with s_r < 0 gets out = 0 for all S tokens and writes no state and no snapshot. Slots no row
  maps to are unchanged bit for bit.
* Speculative rollback: after the step, `mamba_state_selective_copy` with accepted index a copies
  snapshot[s, a] into state[s] (an exact copy; prefill rows are left alone).
* Preconditions on the caller: nheads % ngroups == 0; slots unique within a launch (both unchecked); D shaped
  (nheads, dim) (asserted by #8569, s10).
* Tolerance class: floating point, bounded by a trusted implementation: the same recurrence in torch fp32 on
  the same device (`_recurrence` on fp32 copies of the kernel's inputs, cast to each output's dtype). Padding
  rows, untouched slots and snapshot rows, read-only inputs and the rollback copy: exact.

Variants and dispatch paths (from the kernel, read after the contract): heuristics HAS_DT_BIAS, HAS_D, HAS_Z,
HAS_STATE_BATCH_INDICES, HAS_INT_STATE and BLOCK_SIZE_DSTATE = next_pow2(dstate) (:55-62); constexpr
DT_SOFTPLUS and TIE_HDIM (A, dt, dt_bias stride-0 over (dim, dstate), launcher :399-404, the production
layout); the softplus branch dt + dt_bias <= 20 (:236, :243); the `slot < 0` early exit (:148-156); launcher
reshapes for 3-D state and 2-/3-D x (:316-353); and (BLOCK_SIZE_M, num_warps) chosen by dstate and
architecture (:381-397, `DECODE_TILES` below), with grid (cdiv(dim, BLOCK_SIZE_M), batch, nheads). Sizes
are B-1/B/B+1 around every dstate class edge (16, 32, 64, 128) and around BLOCK_SIZE_M for head dim, plus
production head dim 64 and dstate 128/256; S in {1, 2, 4} (1 + num_speculative_tokens). The compute
capability >= 10 tile (32, 8) for dstate > 64 runs only on GB200 (`launch_on_gb200`).

Main vs open PRs: #8569 (s10) asserts D is (nheads, dim); every call here passes that shape, so the module
runs the same on main and with s10.

Suspected finding O7-N1 (no fix PR yet): the kernel's softplus is log(exp(x) + 1) in fp32 (:34-36), whose
absolute error of ~2**-24 is a relative error up to ~5e-5 at production arguments (dt' ~ 1e-3); exp(A dt')
with |A| up to 16 carries it into the state. With an fp32 state or output that is ~3x torch's fp32 rms
error, so cases of `test_decode_step_fp32_outputs_match_fp64`, its Blackwell twin and
`test_production_shapes[fp32_state]` fail on main. bf16 outputs and states are rounding-dominated and pass.
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.inference.text_generation_controllers.mtp_utils_triton import (
    mamba_state_selective_copy,
)
from megatron.core.ssm.ops.mamba2.mamba_ssm import selective_state_update
from tests.unit_tests.inference.oracles import oracle_utils as ou

DEVICE = "cuda"

# (BLOCK_SIZE_M, num_warps) by dstate: launcher, megatron/core/ssm/ops/mamba2/mamba_ssm.py:381-397.
DSTATE_CLASS_EDGES = (16, 32, 64, 128)


def decode_tile(dstate, blackwell):
    """The decode kernel's (BLOCK_SIZE_M, num_warps) for `dstate` (mamba_ssm.py:381-397)."""
    if dstate <= 16:
        return 32, 4
    if dstate <= 32:
        return 16, 4
    if dstate <= 64:
        return 8, 4
    if blackwell:
        return 32, 8
    if dstate <= 128:
        return 4, 4
    return 4, 8


PRODUCTION_HEADDIM = 64
PRODUCTION_DSTATES = (128, 256)
DSTATES = sorted(
    set(ou.boundary_sizes(*DSTATE_CLASS_EDGES, multiples=(), degenerate=False))
    | {1}
    | set(PRODUCTION_DSTATES)
)

# Rows and slots of the indexed cases: row 2 is CUDA-graph padding (-1), slots 1, 3, 4 and 7 (the last
# slot, where a host gather of -1 lands) belong to no row.
INDEXED_SLOTS = (5, 2, -1, 0, 6)
INDEXED_SLOT_COUNT = 8


# ---------------------------------------------------------------------------------------------
# fp64 reference
# ---------------------------------------------------------------------------------------------


def _recurrence(state0, x, dt, A, B, C, D, z, dt_bias, dt_softplus):
    """The Mamba-2 decode recurrence for live rows, written from the contract.

    state0 (R, H, P, N): each row's slot state at entry; x, dt, z (R, S, H, P); A (H, P, N); B, C
    (R, S, G, N); D, dt_bias (H, P) or None. Returns (out (R, S, H, P), states after each token
    (R, S, H, P, N)).
    """
    heads, groups = x.shape[2], B.shape[2]
    group_of_head = torch.arange(heads, device=x.device) // (heads // groups)
    B = B[:, :, group_of_head]
    C = C[:, :, group_of_head]
    h = state0
    outs, states = [], []
    for t in range(x.shape[1]):
        delta = dt[:, t] if dt_bias is None else dt[:, t] + dt_bias
        if dt_softplus:
            delta = F.softplus(delta)
        decay = torch.exp(A * delta[..., None])
        h = decay * h + (delta * x[:, t])[..., None] * B[:, t, :, None, :]
        y = (h * C[:, t, :, None, :]).sum(-1)
        if D is not None:
            y = y + D * x[:, t]
        if z is not None:
            y = y * z[:, t] * torch.sigmoid(z[:, t])
        outs.append(y)
        states.append(h)
    return torch.stack(outs, 1), torch.stack(states, 1)


def _trusted(state0, x, dt, A, B, C, D, z, dt_bias, dt_softplus):
    """`_recurrence` in fp32 on the kernel's (rounded) inputs: the trusted implementation."""

    def f32(t):
        return None if t is None else t.float()

    return _recurrence(
        f32(state0),
        f32(x),
        f32(dt),
        f32(A),
        f32(B),
        f32(C),
        f32(D),
        f32(z),
        f32(dt_bias),
        dt_softplus,
    )


# ---------------------------------------------------------------------------------------------
# Inputs in the production layout
# ---------------------------------------------------------------------------------------------


def _decode_inputs(
    *,
    rows,
    seq_len,
    heads,
    groups,
    headdim,
    dstate,
    io_dtype,
    layout,
    d_kind,
    with_z,
    with_dt_bias,
    dt_softplus,
    generator,
    padding_rows=(),
    edit_values=None,
):
    """Kernel inputs as `ssm_decode` builds them: x, z, B, C, dt are views of one (rows, S, proj) tensor.

    layout: "tie" (A, dt, dt_bias stride-0 expands of per-head values, the production TIE_HDIM path),
    "materialized" (the same values, contiguous: the general path) or "general" (values vary over
    (dim, dstate)). d_kind: None, "per_head" (stride-0 expand, as on main) or "per_channel".
    Padding rows are filled with NaN. `edit_values(proj=, a_heads=, bias_heads=)` may overwrite the
    per-head values in place before the layout is formed.
    """
    g = generator
    d_inner = heads * headdim
    proj = torch.randn(
        rows, seq_len, 2 * d_inner + 2 * groups * dstate + heads, device=DEVICE, generator=g
    )
    if not dt_softplus:
        # Without softplus dt' = dt + dt_bias must stay positive for a decaying state.
        proj[..., -heads:] = proj[..., -heads:].abs() * 0.1
    proj = proj.to(io_dtype)
    for r in padding_rows:
        proj[r] = float("nan")
    a_heads = -torch.empty(heads, device=DEVICE).uniform_(1.0, 16.0, generator=g)
    bias_lo, bias_hi = (-4.0, 1.0) if dt_softplus else (0.0, 0.1)
    bias_heads = torch.empty(heads, device=DEVICE).uniform_(bias_lo, bias_hi, generator=g)
    if edit_values is not None:
        edit_values(proj=proj, a_heads=a_heads, bias_heads=bias_heads)

    z, x, B, C, dt_heads = proj.split(
        [d_inner, d_inner, groups * dstate, groups * dstate, heads], -1
    )
    x = x.unflatten(-1, (heads, headdim))
    z = z.unflatten(-1, (heads, headdim)) if with_z else None
    B = B.unflatten(-1, (groups, dstate))
    C = C.unflatten(-1, (groups, dstate))
    if layout == "general":
        A = -torch.empty(heads, headdim, dstate, device=DEVICE).uniform_(1.0, 16.0, generator=g)
        dt = torch.randn(rows, seq_len, heads, headdim, device=DEVICE, generator=g)
        if not dt_softplus:
            dt = dt.abs() * 0.1
        dt = dt.to(io_dtype)
        for r in padding_rows:
            dt[r] = float("nan")
        dt_bias = torch.empty(heads, headdim, device=DEVICE).uniform_(bias_lo, bias_hi, generator=g)
    else:
        A = a_heads[:, None, None].expand(heads, headdim, dstate)
        dt = dt_heads[..., None].expand(rows, seq_len, heads, headdim)
        dt_bias = bias_heads[:, None].expand(heads, headdim)
        if layout == "materialized":
            A, dt, dt_bias = A.contiguous(), dt.contiguous(), dt_bias.contiguous()
    if not with_dt_bias:
        dt_bias = None
    if d_kind == "per_head":
        D = torch.randn(heads, device=DEVICE, generator=g)[:, None].expand(heads, headdim)
    elif d_kind == "per_channel":
        D = torch.randn(heads, headdim, device=DEVICE, generator=g)
    else:
        D = None
    return dict(proj=proj, x=x, dt=dt, A=A, B=B, C=C, D=D, z=z, dt_bias=dt_bias)


def _slots(indices, rows):
    """(row slots as a list, slot count, index tensor or None) for an index mode."""
    if indices == "none":
        return list(range(rows)), rows, None
    dtype = torch.int64 if indices == "int64" else torch.int32
    return (
        list(INDEXED_SLOTS),
        INDEXED_SLOT_COUNT,
        torch.tensor(INDEXED_SLOTS, dtype=dtype, device=DEVICE),
    )


def run_decode_case(
    *,
    dstate,
    headdim,
    seq_len,
    heads_groups,
    layout="tie",
    io_dtype=torch.bfloat16,
    state_dtype=torch.float32,
    with_z=False,
    indices="int64",
    snapshots=True,
    d_kind="per_head",
    with_dt_bias=True,
    dt_softplus=True,
    edit_values=None,
):
    """Run one decode step and check every output against the contract.

    `edit_values(proj=, a_heads=, bias_heads=)` may set special values before the layout is formed.
    """
    heads, groups = heads_groups
    g = ou.make_generator(device=DEVICE)
    rows = len(INDEXED_SLOTS) if indices != "none" else 3
    slots, slot_count, index_tensor = _slots(indices, rows)
    live = [r for r, s in enumerate(slots) if s >= 0]
    padding = [r for r, s in enumerate(slots) if s < 0]
    live_slots = [slots[r] for r in live]
    untouched = [s for s in range(slot_count) if s not in live_slots]
    inputs = _decode_inputs(
        rows=rows,
        seq_len=seq_len,
        heads=heads,
        groups=groups,
        headdim=headdim,
        dstate=dstate,
        io_dtype=io_dtype,
        layout=layout,
        d_kind=d_kind,
        with_z=with_z,
        with_dt_bias=with_dt_bias,
        dt_softplus=dt_softplus,
        generator=g,
        padding_rows=padding,  # NaN: nothing of a padding row may be read
        edit_values=edit_values,
    )

    state0 = torch.randn(slot_count, heads, headdim, dstate, device=DEVICE, generator=g)
    state0 = state0.to(state_dtype)
    state0[untouched] = float("nan")  # a read of a slot no row maps to shows up as NaN
    state = ou.guarded_like(state0)
    snap = (
        ou.guarded_empty((slot_count, seq_len, heads, headdim, dstate), state_dtype, DEVICE)
        if snapshots
        else None
    )
    read_only = {
        name: t.clone()
        for name, t in inputs.items()
        if t is not None and name not in ("x", "z", "B", "C")  # views of proj
    }
    index_before = None if index_tensor is None else index_tensor.clone()

    out = selective_state_update(
        state.tensor,
        inputs["x"],
        inputs["dt"],
        inputs["A"],
        inputs["B"],
        inputs["C"],
        D=inputs["D"],
        z=inputs["z"],
        dt_bias=inputs["dt_bias"],
        dt_softplus=dt_softplus,
        state_batch_indices=index_tensor,
        intermediate_ssm_states=None if snap is None else snap.tensor,
    )

    guards = {"state": state} if snap is None else {"state": state, "snapshots": snap}
    ou.check_guards(**guards)
    for name, before in read_only.items():
        ou.assert_exact(inputs[name], before, what=f"{name} is read-only")
    if index_before is not None:
        ou.assert_exact(index_tensor, index_before, what="state_batch_indices is read-only")
    assert out.shape == inputs["x"].shape and out.dtype == inputs["x"].dtype

    if padding:
        ou.assert_exact(
            out[padding], torch.zeros_like(out[padding]), what="out of padding (-1) rows"
        )
    ou.assert_exact(state.tensor[untouched], state0[untouched], what="slots no row maps to")
    if snap is not None and untouched:
        ou.assert_poisoned(snap.tensor[untouched], what="snapshots of slots no row maps to")

    args = (
        state0[live_slots],
        inputs["x"][live],
        inputs["dt"][live],
        inputs["A"],
        inputs["B"][live],
        inputs["C"][live],
        inputs["D"],
        None if inputs["z"] is None else inputs["z"][live],
        inputs["dt_bias"],
        dt_softplus,
    )
    ref_out, ref_states = ou.fp64_reference(_recurrence, *args)
    trusted_out, trusted_states = _trusted(*args)
    ou.assert_close_to_reference(out[live], ref_out, trusted_out.to(out.dtype), what="out")
    ou.assert_close_to_reference(
        state.tensor[live_slots],
        ref_states[:, -1],
        trusted_states[:, -1].to(state_dtype),
        what="final state",
    )
    if snap is not None:
        ou.assert_close_to_reference(
            snap.tensor[live_slots], ref_states, trusted_states.to(state_dtype), what="snapshots"
        )
    return dict(out=out, state=state.tensor, snapshots=None if snap is None else snap.tensor)


# ---------------------------------------------------------------------------------------------
# Structure-derived cases
# ---------------------------------------------------------------------------------------------

HEADDIM_OFFSETS = ("-1", "0", "+1", "production")


def _headdim(offset, dstate, blackwell):
    block_m = decode_tile(dstate, blackwell)[0]
    if offset == "production":
        return PRODUCTION_HEADDIM
    return max(1, block_m + int(offset))


def _decode_cases(blackwell, fp32_outputs):
    """Pairwise cases for one architecture's tile table; `fp32_outputs` picks the cases with an fp32 out or
    state (where the kernel's own fp32 error, not output rounding, meets the bound) or those without.
    """
    factors = {
        "dstate": DSTATES,
        "headdim": HEADDIM_OFFSETS,
        "seq_len": (1, 2, 4),
        "heads_groups": ((1, 1), (8, 1), (8, 2), (8, 8)),
        "layout": ("tie", "materialized", "general"),
        # (io dtype, state dtype)
        "dtypes": (
            (
                (torch.bfloat16, torch.float32),
                (torch.float32, torch.float32),
                (torch.float32, torch.bfloat16),
            )
            if fp32_outputs
            else ((torch.bfloat16, torch.bfloat16),)
        ),
        "with_z": (False, True),
        "indices": ("int64", "int32", "none"),
        "snapshots": (True, False),
        "d_kind": ("per_head", "per_channel", None),
        "variant": ("default", "no_dt_bias", "no_softplus"),
    }
    params = []
    for case in ou.pairwise_cases(factors):
        case["io_dtype"], case["state_dtype"] = case.pop("dtypes")
        case["headdim"] = _headdim(case["headdim"], case["dstate"], blackwell)
        variant = case.pop("variant")
        case["with_dt_bias"] = variant != "no_dt_bias"
        case["dt_softplus"] = variant != "no_softplus"
        block_m, warps = decode_tile(case["dstate"], blackwell)
        case_id = "-".join(
            [
                f"n{case['dstate']}",
                f"p{case['headdim']}[M{block_m}w{warps}]",
                f"s{case['seq_len']}",
                "h{}g{}".format(*case["heads_groups"]),
                case["layout"],
                str(case["io_dtype"]).replace("torch.", ""),
                "state_" + str(case["state_dtype"]).replace("torch.", ""),
                "z" if case["with_z"] else "noz",
                case["indices"],
                "snap" if case["snapshots"] else "nosnap",
                f"D_{case['d_kind']}",
                variant,
            ]
        )
        params.append(pytest.param(case, id=case_id))
    return params


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", _decode_cases(blackwell=False, fp32_outputs=False))
def test_decode_step_matches_fp64(case):
    """Every tile class, variant and layout on Hopper against the fp64 recurrence (bf16 out and state)."""
    run_decode_case(**case)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", _decode_cases(blackwell=False, fp32_outputs=True))
def test_decode_step_fp32_outputs_match_fp64(case):
    """The sweep's cases with an fp32 out or state (production: fp32 states). Cases fail on main for the
    suspected finding O7-N1 (module docstring): the kernel's fp32 softplus costs about 3x torch's fp32 rms
    error in the state (2x allowed)."""
    run_decode_case(**case)


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("case", _decode_cases(blackwell=True, fp32_outputs=False))
def test_decode_step_matches_fp64_blackwell(case):
    """The same sweep on Blackwell, where dstate > 64 runs the (32, 8) tile (mamba_ssm.py:392-395)."""
    run_decode_case(**case)


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("case", _decode_cases(blackwell=True, fp32_outputs=True))
def test_decode_step_fp32_outputs_match_fp64_blackwell(case):
    """Blackwell's fp32-output cases; see `test_decode_step_fp32_outputs_match_fp64` (O7-N1)."""
    run_decode_case(**case)


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.parametrize(
    "state_dtype", [torch.bfloat16, torch.float32], ids=["bf16_state", "fp32_state"]
)
def test_production_shapes(state_dtype):
    """Production decode: 8 local heads, 1 group, head dim 64, dstate 128, bf16 activations, int64 slots,
    with and without three speculative tokens. fp32 states (the default) fail on main for O7-N1."""
    for seq_len, snapshots in ((1, False), (4, True)):
        run_decode_case(
            dstate=128,
            headdim=64,
            seq_len=seq_len,
            heads_groups=(8, 1),
            state_dtype=state_dtype,
            snapshots=snapshots,
        )


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
def test_softplus_threshold_and_extreme_decay():
    """dt + dt_bias on both sides of softplus's identity threshold (20) and far below it; A from -1e-3 to
    -16. Values are exact in bf16, so the threshold is crossed exactly where intended."""
    values = torch.tensor([-40.0, -20.5, 0.0, 19.5, 20.0, 20.5, 40.0, 100.0])
    decays = torch.tensor([-1e-3, -0.5, -1.0, -2.0, -4.0, -8.0, -12.0, -16.0])

    def edit(proj, a_heads, bias_heads):
        proj[..., -values.numel() :] = values.to(DEVICE, proj.dtype)
        a_heads.copy_(decays)
        bias_heads.zero_()

    for layout in ("tie", "materialized"):
        run_decode_case(
            dstate=16,
            headdim=8,
            seq_len=2,
            heads_groups=(8, 2),
            layout=layout,
            state_dtype=torch.bfloat16,
            edit_values=edit,
        )


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
def test_nan_in_one_request_stays_in_that_request():
    """A NaN in one request's x reaches only that request's output, state and snapshots."""

    heads, headdim = 4, 16

    def edit(proj, a_heads, bias_heads):
        proj[0, 0, heads * headdim + 2 * headdim + 1] = float("nan")  # x[row 0, t 0, head 2, p 1]

    result = run_decode_case(
        dstate=64,
        headdim=headdim,
        seq_len=2,
        heads_groups=(heads, 2),
        state_dtype=torch.bfloat16,
        with_z=True,
        edit_values=edit,
    )
    assert torch.isnan(result["out"][0, :, 2, 1]).all()
    others = [s for s in INDEXED_SLOTS[1:] if s >= 0]
    assert torch.isfinite(result["state"][others]).all()


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.parametrize("form", ["state3d_x2d", "state3d_x3d", "state4d_x3d"])
def test_reduced_launcher_layouts(form):
    """The launcher's reshapes: a head-less 3-D state with x (b, dim) or (b, S, dim), and x (b, nheads, dim)
    without a sequence dim. Each must equal the canonical 4-D call's contract."""
    g = ou.make_generator(device=DEVICE)
    rows, dim, dstate = 3, 24, 32
    heads, seq_len = (
        (1, 1) if form == "state3d_x2d" else ((1, 3) if form == "state3d_x3d" else (4, 1))
    )
    x = torch.randn(rows, seq_len, heads, dim, device=DEVICE, generator=g).bfloat16()
    z = torch.randn(rows, seq_len, heads, dim, device=DEVICE, generator=g).bfloat16()
    dt = torch.randn(rows, seq_len, heads, dim, device=DEVICE, generator=g).bfloat16()
    B = torch.randn(rows, seq_len, 1 if heads == 1 else 2, dstate, device=DEVICE, generator=g)
    B = B.bfloat16()
    C = torch.randn(B.shape, device=DEVICE, generator=g).bfloat16()
    A = -torch.empty(heads, dim, dstate, device=DEVICE).uniform_(1.0, 16.0, generator=g)
    D = torch.randn(heads, dim, device=DEVICE, generator=g)
    dt_bias = torch.empty(heads, dim, device=DEVICE).uniform_(-4.0, 1.0, generator=g)
    state0 = torch.randn(rows, heads, dim, dstate, device=DEVICE, generator=g).bfloat16()
    snap0 = ou.poisoned_empty((rows, seq_len, heads, dim, dstate), torch.bfloat16, DEVICE)

    if form == "state4d_x3d":
        state = state0.clone()
        call = (x[:, 0], dt[:, 0], A, B[:, 0], C[:, 0], D, z[:, 0], dt_bias)
        snap = None
    else:
        state = state0[:, 0].clone()
        head = (lambda t: t[:, :, 0]) if form == "state3d_x3d" else (lambda t: t[:, 0, 0])
        call = (head(x), head(dt), A[0], head(B), head(C), D[0], head(z), dt_bias[0])
        snap = snap0[:, :, 0].clone() if form == "state3d_x3d" else None
    out = selective_state_update(
        state,
        call[0],
        call[1],
        call[2],
        call[3],
        call[4],
        D=call[5],
        z=call[6],
        dt_bias=call[7],
        dt_softplus=True,
        intermediate_ssm_states=snap,
    )
    args = (state0, x, dt, A, B, C, D, z, dt_bias, True)
    ref_out, ref_states = ou.fp64_reference(_recurrence, *args)
    trusted_out, trusted_states = _trusted(*args)
    assert out.shape == call[0].shape
    ou.assert_close_to_reference(
        out.reshape(ref_out.shape), ref_out, trusted_out.bfloat16(), what="out"
    )
    ou.assert_close_to_reference(
        state.reshape(state0.shape),
        ref_states[:, -1],
        trusted_states[:, -1].bfloat16(),
        what="final state",
    )
    if snap is not None:
        ou.assert_close_to_reference(
            snap.reshape(ref_states.shape), ref_states, trusted_states.bfloat16(), what="snapshots"
        )


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_speculative_rollback_restores_accepted_prefix(state_dtype):
    """Speculative decoding: S = 4 tokens with snapshots, then `mamba_state_selective_copy` with each
    request's accepted index a. The live state must be snapshot[slot, a] bit for bit and the fp64 state after
    a + 1 tokens within the bound; a prefill request's slot is left as the decode step wrote it."""
    seq_len, heads, headdim, dstate = 4, 4, 16, 64
    result = run_decode_case(
        dstate=dstate,
        headdim=headdim,
        seq_len=seq_len,
        heads_groups=(heads, 1),
        state_dtype=state_dtype,
    )
    live_slots = [s for s in INDEXED_SLOTS if s >= 0]
    accepted = [3, 0, 1, 2]
    prefill = [0, 0, 1, 0]
    state = result["state"]
    after_step = state.clone()
    # The production buffers carry a leading layer dim: (L, slots, S, ...) and (L, slots, ...).
    mamba_state_selective_copy(
        intermediate_states=result["snapshots"].unsqueeze(0),
        current_states=state.unsqueeze(0),
        prefill_status=torch.tensor(prefill, dtype=torch.int32, device=DEVICE),
        state_idx=torch.tensor(live_slots, dtype=torch.int32, device=DEVICE),
        accepted_counts=torch.tensor(accepted, dtype=torch.int32, device=DEVICE),
        num_layers=1,
    )
    for slot, a, is_prefill in zip(live_slots, accepted, prefill):
        expected = after_step[slot] if is_prefill else result["snapshots"][slot, a]
        ou.assert_exact(state[slot], expected, what=f"live state of slot {slot} after rollback")
    untouched = [s for s in range(INDEXED_SLOT_COUNT) if s not in live_slots]
    ou.assert_exact(state[untouched], after_step[untouched], what="slots no request maps to")


@pytest.mark.sanitizer
@ou.requires_sm(ou.HOPPER)
def test_sanitizer_small_case():
    """Small case for compute-sanitizer: padded dstate (17 -> 32), head dim past BLOCK_SIZE_M (16 + 1),
    a -1 row, untouched slots, snapshots and z."""
    run_decode_case(
        dstate=17,
        headdim=17,
        seq_len=2,
        heads_groups=(2, 1),
        state_dtype=torch.bfloat16,
        with_z=True,
    )


INT32_ROW = (1, 64, 128)  # one slot of a one-head state: H, P, N


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("slot_count", ou.int32_overflow_params(INT32_ROW[0] * 64 * 128))
def test_state_offsets_past_int32(slot_count):
    """int64 slot indices whose state offsets reach 2**31 elements (bf16 state, 4 GiB)."""
    heads, headdim, dstate = INT32_ROW
    ou.require_free_memory(int(slot_count * headdim * dstate * 2 * 1.1) + (1 << 30))
    g = ou.make_generator(device=DEVICE)
    slots = [slot_count - 1, slot_count // 2, 0, -1]
    state = torch.zeros(slot_count, heads, headdim, dstate, dtype=torch.bfloat16, device=DEVICE)
    live = [s for s in slots if s >= 0]
    state0 = torch.randn(len(live), heads, headdim, dstate, device=DEVICE, generator=g).bfloat16()
    state[live] = state0
    sentinel = slot_count - 2
    ou.poison_(state[sentinel])
    rows = len(slots)
    x = torch.randn(rows, 1, heads, headdim, device=DEVICE, generator=g).bfloat16()
    dt = torch.randn(rows, 1, heads, device=DEVICE, generator=g).bfloat16()
    dt = dt[..., None].expand(rows, 1, heads, headdim)
    A = -torch.empty(heads, device=DEVICE).uniform_(1.0, 16.0, generator=g)
    A = A[:, None, None].expand(heads, headdim, dstate)
    B = torch.randn(rows, 1, 1, dstate, device=DEVICE, generator=g).bfloat16()
    C = torch.randn(rows, 1, 1, dstate, device=DEVICE, generator=g).bfloat16()
    D = torch.randn(heads, headdim, device=DEVICE, generator=g)
    dt_bias = torch.zeros(heads, device=DEVICE)[:, None].expand(heads, headdim)
    out = selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=torch.tensor(slots, dtype=torch.int64, device=DEVICE),
    )
    ou.assert_poisoned(state[sentinel], what="slot next to the last one")
    ou.assert_exact(out[3], torch.zeros_like(out[3]), what="out of the -1 row")
    args = (state0, x[:3], dt[:3], A, B[:3], C[:3], D, None, dt_bias, True)
    ref_out, ref_states = ou.fp64_reference(_recurrence, *args)
    trusted_out, trusted_states = _trusted(*args)
    ou.assert_close_to_reference(out[:3], ref_out, trusted_out.bfloat16(), what="out")
    ou.assert_close_to_reference(
        state[live], ref_states[:, -1], trusted_states[:, -1].bfloat16(), what="final state"
    )


def test_case_lists_cover_every_tile_class():
    """The sweeps reach every (BLOCK_SIZE_M, num_warps) class and every dstate padding class (CPU check)."""
    for blackwell in (False, True):
        cases = [p.values[0] for f in (False, True) for p in _decode_cases(blackwell, f)]
        tiles = {decode_tile(c["dstate"], blackwell) for c in cases}
        expected = {decode_tile(n, blackwell) for n in DSTATES}
        assert tiles == expected
        pads = {1 << max(0, (c["dstate"] - 1).bit_length()) for c in cases}
        assert pads == {1 << max(0, (n - 1).bit_length()) for n in DSTATES}
        for factor in ("seq_len", "layout", "indices", "io_dtype", "state_dtype"):
            assert len({str(c[factor]) for c in cases}) > 1
        for fp32_outputs in (False, True):
            subset = [p.values[0] for p in _decode_cases(blackwell, fp32_outputs)]
            assert {decode_tile(c["dstate"], blackwell) for c in subset} == expected
