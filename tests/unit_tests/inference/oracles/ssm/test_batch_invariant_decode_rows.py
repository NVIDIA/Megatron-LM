# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the batch-invariant Mamba decode row kernels (megatron/core/ssm/ops/mamba2/
batch_invariant_decode_rows.py): `_chunk_scan_row_kernel` (`chunk_scan_rows`), `_bmm_row_kernel`
(`bmm_rows`) and `_chunk_state_passing_rows_kernel` (`chunk_state_passing_rows`), launched by
`_mamba_chunk_scan_decode_rows` when the training scan kernels are pinned (batch-invariant mode).

Before this module these kernels were checked only against the training tile kernels (bitwise, co-derived)
and, in bf16 at one shape, end to end against the training scan (tests/unit_tests/ssm/ops/mamba2/
test_batch_invariant_decode.py); never against an independent reference, in fp32 against anything but the
tile kernels, with per-channel D against a reference, at production shapes, or on GB200.

Contract (written from the batch-invariant decode design, `BatchInvariantDecodeBuffers` and the launcher
docstrings, before reading the kernel bodies). One decode step of E entries over the replay buffers of a
slot-indexed cache, chunk size c:

* Entry e owns slot s_e (-1: a padding entry) and the window [w_e, w_e + c) of the replay buffers x
  (rows, nheads, headdim), B and C (rows, ngroups, dstate), w_e = s_e * c. Its target row r_e (-1 for a
  padding entry) is the current token's position in the window: rows 0..r_e hold the slot's tokens since
  its last chunk boundary (rows past r_e hold finite filler, `seed`). initial_states[s] (fp32, nslots,
  nheads, headdim, dstate) is slot s's state at that boundary. dt and dA_cumsum (nheads, E, c) fp32 are
  the windows' processed dt and cumulative decay (`_chunk_cumsum_fwd` in target-rows mode). Head h reads
  group g(h) = h // (nheads / ngroups).
* `bmm_rows` -> CB (E, ngroups, c, c) fp32 with, for every live entry, row r_e:

      CB[e, g, r_e, j] = sum_n C[w_e + r_e, g, n] * B[w_e + j, g, n]          for j <= r_e

  every other entry unspecified.
* `chunk_scan_rows` -> out[e] (E, nheads, headdim), the Mamba-2 output of the current token:

      y[h, p] = exp(dA[h, e, r]) * sum_n C[w+r, g, n] * S0[s, h, p, n]
                + sum_{k <= r} CB[e, g, r, k] * exp(dA[h, e, r] - dA[h, e, k]) * dt[h, e, k] * x[w+k, h, p]
                + D[h(, p)] * x[w + r, h, p]                                        (no term without D)
      out[e, h, p] = y * z[e, h, p] * sigmoid(z[e, h, p])                          (y without z)

  in x's dtype; a padding entry's row is not written. Only CB[e, g, r, j <= r] is read.
* `chunk_state_passing_rows`: for every entry with chunk_flags[e] != 0 (its window is full: r = c - 1),
  the slot's state advances over the window:

      S0[s] <- exp(dA[h, e, c-1]) * S0[s] + sum_{k < c} x[w+k] (outer) B[w+k, g] * dt[h, e, k]
                                                         * exp(dA[h, e, c-1] - dA[h, e, k])

  in place, fp32; every other slot (unflagged, padding, unmapped) is untouched.
* Batch invariance (the reason these kernels exist): with the training kernels pinned, the row kernels'
  outputs and committed states equal, bit for bit, those of the training kernels in target-rows /
  chunk-flags / dst-states mode (`use_row_kernels=False`). This is the production contract of the module
  and no fp64 reference can express it, so it is checked as a metamorphic property
  (`test_row_kernels_reproduce_the_training_tiles_bitwise`), on H100 and GB200: the selection rule
  (`row_kernel_configs`) is a heuristic on the tile shape whose premise (`mma.sync` on both sides) was
  measured on GB200.
* Tolerance class: floating point. Reference: the formulas above in fp64. Trusted: for CB, the training
  scan's CB kernel (pip mamba_ssm, same tensor-core accumulation; test_bmm_chunk.py says why); for the
  output and the states, the same formulas with the dot operands as the training scan's tensor cores
  read them (C and the state rounded to C's dtype; the decayed CB * dt weights and the decayed B rounded
  to x's dtype; fp32 operands as TF32), whose rounding dominates the error, products and sums in fp64,
  the rest in fp32 (`ssd_utils`). Exact: CB for dyadic inputs, padding rows, untouched slots, guard
  bands, read-only inputs.

Variants and dispatch (from the kernels, read after the contract): each launcher takes the training
kernel's config; every candidate the training autotuners list is passed in turn (the chunk scan's
selectable ones: `row_kernel_configs` declines num_warps >= 4 with BLOCK_SIZE_M >= 64); the 16-row MMA tile
(ROW_BLOCK_SIZE_M) over the target row and the training tile's causal bound K_MAX: target rows 0, 15, 16,
17, 31, 32, 33, 63, 64, 65, c - 1; HAS_D (per head, per channel D_HAS_HDIM), HAS_Z; x/C dtype bf16 (the
fma for 16-bit x) and fp32; dstate 16, 64, 100, 128 (the row path requires next_pow2(dstate) <= 128;
BLOCK_SIZE_DSTATE >= 16); headdim 64 and 80 (partial BLOCK_SIZE_N tiles); CHUNKS_PER_PROGRAM 1/2/4/16 at
31, 32, 33, 64, 257 entries (the `c < nchunks` tail); production: the Nemotron-H 8B layer (128 heads of
64, 8 groups, dstate 128, chunk 128) at decode batches 64 and 256; int64 offsets past 2^31 buffer
elements (`large`).

Main vs open PRs: no open PR changes these kernels.
"""

import contextlib

import pytest
import torch
from triton.runtime.errors import OutOfResources

from megatron.core.ssm.ops.mamba2 import (
    batch_invariant_decode_rows,
    ssd_bmm,
    ssd_chunk_scan,
    ssd_chunk_state,
    ssd_state_passing,
)
from megatron.core.ssm.ops.mamba2.batch_invariant_decode import _mamba_chunk_scan_decode_rows
from megatron.core.ssm.ops.mamba2.batch_invariant_decode_rows import (
    ROW_BLOCK_SIZE_M,
    bmm_rows,
    chunk_scan_rows,
    chunk_state_passing_rows,
    row_kernel_configs,
)
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.ssm import ssd_utils as su
from tests.unit_tests.inference.oracles.ssm.test_bmm_chunk import (
    bmm_reference,
    required_mask,
    trusted_bmm,
)
from tests.unit_tests.inference.oracles.ssm.test_chunk_cumsum import (
    cumsum_reference,
    production_parameters,
)
from tests.unit_tests.inference.oracles.ssm.test_chunk_state import (
    chunk_state_reference,
    trusted_chunk_state,
)

DEVICE = "cuda"
SCAN, BMM, STATE = (
    ssd_chunk_scan._chunk_scan_fwd_kernel,
    ssd_bmm._bmm_chunk_fwd_kernel,
    ssd_chunk_state._chunk_state_fwd_kernel,
)
TRAINING_KERNELS = (
    ssd_chunk_state._chunk_cumsum_fwd_kernel,
    STATE,
    BMM,
    SCAN,
    ssd_state_passing._state_passing_fwd_kernel,
)
ROWS = (
    0,
    15,
    16,
    17,
    31,
    32,
    33,
    63,
    64,
    65,
)  # around the 16-row MMA tile and the training M tiles


def selectable_scan_configs(chunk_size):
    """The chunk-scan candidates `row_kernel_configs` accepts (it declines warpgroup-sized tiles)."""
    return [
        c
        for c in su.kernel_configs(SCAN)
        if not (c.num_warps >= 4 and c.kwargs["BLOCK_SIZE_M"] >= 64)
        and chunk_size % c.kwargs["BLOCK_SIZE_M"] == 0
    ]


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


def decode_step(
    entries,
    heads,
    groups,
    headdim,
    dstate,
    dtype,
    g,
    *,
    chunk_size=su.PRODUCTION_CHUNK,
    padding=(3,),
    rows=None,
    extra_slots=2,
):
    """One decode step's inputs: replay buffers for `entries + extra_slots` slots (the extra ones are
    mapped by no entry), metadata (`ssd_utils.decode_windows`), dt / dA_cumsum of the fixed windows (the
    contract computed in fp64 and rounded to fp32), the slots' boundary states, z and both D layouts.
    """
    rows = list(rows) if rows is not None else [*ROWS, chunk_size - 1]
    rows = [r for r in rows if r < chunk_size]
    win = su.decode_windows(entries, chunk_size, padding=padding, rows=rows, device=DEVICE)
    nslots = entries + extra_slots
    nrows = nslots * chunk_size
    x = torch.randn(nrows, heads, headdim, device=DEVICE, generator=g).to(dtype)
    B = torch.randn(nrows, groups, dstate, device=DEVICE, generator=g).to(dtype)
    C = torch.randn(nrows, groups, dstate, device=DEVICE, generator=g).to(dtype)
    A, bias = production_parameters(heads, g)
    dt_raw = torch.randn(nrows, heads, device=DEVICE, generator=g)
    windows = su.fixed_windows(win["chunk_starts"].clamp(min=0), chunk_size)
    dA, dt = ou.fp64_reference(
        cumsum_reference,
        dt_raw,
        A,
        bias,
        windows,
        chunk_size,
        softplus=True,
        dt_limit=(0.0, float("inf")),
    )
    states = 0.1 * torch.randn(nslots, heads, headdim, dstate, device=DEVICE, generator=g)
    return dict(
        x=x,
        B=B,
        C=C,
        dt=dt.float(),
        dA=dA.float(),
        z=torch.randn(entries, heads, headdim, device=DEVICE, generator=g).to(dtype),
        D_head=torch.randn(heads, device=DEVICE, generator=g),
        D_channel=torch.randn(heads, headdim, device=DEVICE, generator=g),
        states=states,
        windows=windows,
        chunk_size=chunk_size,
        # raw inputs, for the end-to-end pipeline
        dt_raw=dt_raw.to(dtype),
        A=A,
        dt_bias=bias,
        **win,
    )


def live_entries(d):
    return [e for e in range(d["slots"].numel()) if int(d["target_rows"][e]) >= 0]


def compact(d, entries):
    """The windows of `entries` gathered into a small buffer (window i at rows [i * c, i * c + c)) with
    their slots' states, so the fp64 references never copy the whole replay buffer or cache."""
    c = d["chunk_size"]
    rows = torch.cat(
        [torch.arange(d["windows"][e][0], d["windows"][e][0] + c, device=DEVICE) for e in entries]
    )
    index = torch.tensor(entries, device=DEVICE, dtype=torch.long)
    return dict(
        x=d["x"][rows],
        B=d["B"][rows],
        C=d["C"][rows],
        S0=d["states"][[int(d["slots"][e]) for e in entries]],
        dt=d["dt"][:, index],
        dA=d["dA"][:, index],
        target_rows=d["target_rows"][index],
        z=None if d.get("z") is None else d["z"][index],
        windows=[(i * c, c) for i in range(len(entries))],
    )


# ---------------------------------------------------------------------------------------------
# References and trusted implementations (on compacted entries: window i at rows i * c, state S0[i])
# ---------------------------------------------------------------------------------------------


def scan_rows_reference(cb, x, z, dt, dA, C, S0, D, target_rows, *, op=None):
    """The current token's output of each entry (E, H, P), in the inputs' dtype.

    `op(t, dtype)` is applied to each dot operand (the trusted implementation's tensor-core rounding);
    without it every term is exact (the fp64 reference).
    """
    heads, c = x.shape[1], dt.shape[-1]
    goh = su.group_of_head(heads, C.shape[1], x.device)
    outs = []
    for e in range(S0.shape[0]):
        r, w = int(target_rows[e]), e * c
        Cr, S, xs = C[w + r][goh], S0[e], x[w : w + r + 1]
        weight = cb[e][goh][:, r, : r + 1]
        decay = dA[:, e, r : r + 1] - dA[:, e, : r + 1]
        if op is None:
            weight = weight * torch.exp(decay) * dt[:, e, : r + 1]
            y = torch.exp(dA[:, e, r])[:, None] * torch.einsum("hn,hpn->hp", Cr, S)
            y = y + torch.einsum("hk,khp->hp", weight, xs)
        else:
            weight = weight * torch.exp(decay.clamp(max=0.0)) * dt[:, e, : r + 1]
            prod = torch.einsum("hn,hpn->hp", op(Cr, C.dtype), op(S, C.dtype)).float()
            y = torch.exp(dA[:, e, r])[:, None] * prod
            y = y + torch.einsum("hk,khp->hp", op(weight, x.dtype), op(xs, x.dtype)).float()
            xs = xs.float()
        if D is not None:
            y = y + (D[:, None] if D.dim() == 1 else D) * xs[-1].to(y.dtype)
        if z is not None:
            gate = z[e].to(y.dtype)
            y = y * gate * torch.sigmoid(gate)
        outs.append(y)
    return torch.stack(outs)


def state_rows_reference(x, B, dt, dA, S0, windows, chunk_size):
    """The slot state after a crossing, per compacted entry (E, H, P, N), in the inputs' dtype."""
    new = chunk_state_reference(x, B, dt, dA, windows, chunk_size)
    return torch.exp(dA[:, :, -1]).T[:, :, None, None] * S0 + new


def trusted_state_rows(x, B, dt, dA, S0, windows, chunk_size):
    new = trusted_chunk_state(x, B, dt, dA, windows, chunk_size, torch.float32)
    return torch.exp(dA[:, :, -1]).T[:, :, None, None] * S0 + new


# ---------------------------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------------------------


def run_bmm_rows(d, config, *, exact=False):
    """`bmm_rows` with poisoned output: row r_e, columns j <= r_e of every live entry."""
    with su.poisoned_allocations(batch_invariant_decode_rows) as buffers:
        cb = bmm_rows(d["C"], d["B"], d["chunk_size"], d["target_rows"], d["chunk_starts"], config)
    su.check_all(buffers, "CB rows")
    live = live_entries(d)
    k = compact(d, live)
    c = d["chunk_size"]
    mask = required_mask(k["windows"], c, target_rows=k["target_rows"], device=DEVICE)
    mask = mask.expand(len(live), cb.shape[1], c, c)
    got = cb[torch.tensor(live, device=DEVICE)]
    ref = ou.fp64_reference(bmm_reference, k["C"], k["B"], k["windows"], c)
    if exact:
        su.assert_equal(got[mask], ref[mask], what="CB target rows (exact inputs)")
    else:
        trusted = trusted_bmm(k["C"], k["B"], k["windows"], c, torch.float32)
        ou.assert_close_to_reference(got[mask], ref[mask], trusted[mask], what="CB target rows")
    return cb


def run_scan_rows(d, cb, config, *, D=None, gate=True):
    """`chunk_scan_rows` into a poisoned, guarded `out`; padding rows stay poisoned. `cb` is NaN outside
    the entries `bmm_rows` writes, so a read of any other CB entry shows up in the output."""
    entries = d["slots"].numel()
    out = ou.guarded_empty((entries,) + tuple(d["x"].shape[1:]), d["x"].dtype, DEVICE)
    z = d["z"] if gate else None
    inputs = dict(x=d["x"], z=z, dt=d["dt"], dA=d["dA"], C=d["C"], D=D)
    before = {name: v.clone() for name, v in inputs.items() if v is not None}
    chunk_scan_rows(
        cb,
        d["x"],
        z,
        d["dt"],
        d["dA"],
        d["C"],
        out.tensor,
        d["slots"],
        d["target_rows"],
        d["chunk_starts"],
        d["states"],
        D,
        config,
    )
    ou.check_guards(out=out)
    for name, value in before.items():
        ou.assert_exact(inputs[name], value, what=f"{name} is read-only")
    live = live_entries(d)
    dead = [e for e in range(entries) if e not in live]
    if dead:
        ou.assert_poisoned(out.tensor[dead], what="output rows of padding entries")
    k = compact(dict(d, z=z), live)
    cb_live = cb[torch.tensor(live, device=DEVICE)]
    args = (cb_live, k["x"], k["z"], k["dt"], k["dA"], k["C"], k["S0"], D, k["target_rows"])
    ref = ou.fp64_reference(scan_rows_reference, *args)
    trusted = scan_rows_reference(*args, op=su.as_operand)
    ou.assert_close_to_reference(
        out.tensor[live].flatten(1),
        ref.flatten(1),
        trusted.to(out.tensor.dtype).flatten(1),
        dim=-1,
        what="current-token output",
    )


def run_state_rows(d, config, flags=None):
    """`chunk_state_passing_rows` on guarded slot states: flagged slots advanced, the rest untouched."""
    flags = d["chunk_flags"] if flags is None else flags
    states = ou.guarded_like(d["states"])
    inputs = dict(x=d["x"], B=d["B"], dt=d["dt"], dA=d["dA"], flags=flags, slots=d["slots"])
    kept = {name: v.clone() for name, v in inputs.items()}
    chunk_state_passing_rows(
        d["x"],
        d["B"],
        d["dt"],
        d["dA"],
        d["chunk_starts"],
        flags,
        d["slots"],
        states.tensor,
        config,
    )
    ou.check_guards(states=states)
    for name, value in kept.items():
        ou.assert_exact(inputs[name], value, what=f"{name} is read-only")
    crossing = [e for e in range(flags.numel()) if int(flags[e]) != 0]
    committed = [int(d["slots"][e]) for e in crossing]
    # Every slot but the committed ones keeps its bits (compared in place, without copying the cache).
    after = states.tensor
    new_values = after[committed].clone()
    after[committed] = d["states"][committed]
    ou.assert_exact(after, d["states"], what="slots no crossing entry owns")
    after[committed] = new_values
    if not crossing:
        return
    k = compact(d, crossing)
    args = (k["x"], k["B"], k["dt"], k["dA"], k["S0"], k["windows"], d["chunk_size"])
    ref = ou.fp64_reference(state_rows_reference, *args)
    trusted = trusted_state_rows(*args)
    ou.assert_close_to_reference(
        new_values.flatten(1),
        ref.flatten(1),
        trusted.flatten(1),
        dim=-1,
        what="committed slot states",
    )


# ---------------------------------------------------------------------------------------------
# Per-kernel oracle sweeps
# ---------------------------------------------------------------------------------------------


def _shape_cases():
    factors = {
        "dtype": (torch.bfloat16, torch.float32),
        "heads_groups": ((8, 2), (8, 1), (8, 8)),
        "headdim": (64, 80),
        "dstate": (128, 16, 64, 100),
        "D": ("head", "channel", None),
        "gate": (True, False),
    }
    params = []
    for case in ou.pairwise_cases(factors):
        case_id = "-".join(
            [
                str(case["dtype"]).replace("torch.", ""),
                "h{}g{}".format(*case["heads_groups"]),
                f"p{case['headdim']}",
                f"n{case['dstate']}",
                f"D_{case['D']}",
                "z" if case["gate"] else "noz",
            ]
        )
        params.append(pytest.param(case, id=case_id))
    return params


def _inputs_for(case, entries=24, **kw):
    g = ou.make_generator(device=DEVICE)
    heads, groups = case["heads_groups"]
    return decode_step(
        entries, heads, groups, case["headdim"], case["dstate"], case["dtype"], g, **kw
    )


def _D(d, kind):
    return {"head": d["D_head"], "channel": d["D_channel"], None: None}[kind]


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.parametrize("case", _shape_cases())
def test_row_kernels_match_fp64(case):
    """All three row kernels on one decode step (every target-row class, a padding entry, two unmapped
    slots, crossings) at the configs batch-invariant mode pins, against the fp64 contract."""
    d = _inputs_for(case)
    configs = row_kernel_configs(d["C"], d["chunk_size"]) or (
        su.default_config(SCAN),
        su.default_config(BMM),
        su.default_config(STATE),
    )
    scan_config, bmm_config, state_config = configs
    cb = run_bmm_rows(d, bmm_config)
    run_scan_rows(d, cb, scan_config, D=_D(d, case["D"]), gate=case["gate"])
    run_state_rows(d, state_config)


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_bmm_rows_exact_on_dyadic_inputs(dtype):
    """Dyadic C and B: every guaranteed CB entry is exact, for every bmm candidate (each target-row
    class against its M and N tiles)."""
    g = ou.make_generator(device=DEVICE)
    d = decode_step(24, 8, 2, 64, 128, dtype, g)
    shape = d["C"].shape
    for name in ("C", "B"):
        d[name] = ou.dyadic_tensor(
            shape, exponents=(-1, 1), dtype=torch.float32, device=DEVICE, generator=g
        ).to(dtype)
    for config in su.kernel_configs(BMM):
        try:
            run_bmm_rows(d, config, exact=True)
        except OutOfResources as err:
            assert err.name == "shared memory", f"{su.tile_id(config)}: {err}"


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_every_row_config(dtype):
    """Each launcher with every candidate config it can be given: the chunk scan's selectable configs
    (TILE_SIZE_M 32/64/128 sets K_MAX), every bmm and chunk-state config (the training kernels' tiles,
    which the row kernels reuse). A config the device cannot launch is skipped only on a shared-memory
    OutOfResources."""
    d = decode_step(24, 8, 2, 64, 128, dtype, ou.make_generator(device=DEVICE))
    cb = run_bmm_rows(d, su.default_config(BMM))
    launched = {"scan": 0, "bmm": 0, "state": 0}
    for kind, configs in (
        ("scan", selectable_scan_configs(d["chunk_size"])),
        ("bmm", su.kernel_configs(BMM)),
        ("state", su.kernel_configs(STATE)),
    ):
        for config in configs:
            try:
                if kind == "scan":
                    run_scan_rows(d, cb, config, D=d["D_channel"])
                elif kind == "bmm":
                    run_bmm_rows(d, config)
                else:
                    run_state_rows(d, config)
                launched[kind] += 1
            except OutOfResources as err:
                assert err.name == "shared memory", f"{kind} {su.tile_id(config)}: {err}"
    assert all(launched.values()), f"no config launched for some kernel: {launched}"


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.parametrize("entries", [31, 32, 33, 64, 257])
def test_chunks_per_program_edges(entries):
    """CHUNKS_PER_PROGRAM steps (1 below 32 entries, 2 at 32, 4 at 64, 16 from 256) and the `c < nchunks`
    tail of the last program: crossings at every third entry, padding entries, the rest untouched.
    """
    g = ou.make_generator(device=DEVICE)
    d = decode_step(entries, 4, 1, 64, 64, torch.bfloat16, g, padding=(0, entries - 1))
    flags = torch.zeros_like(d["chunk_flags"])
    flags[1::3] = 1
    flags[0] = flags[entries - 1] = 0
    # The flagged entries' windows are full (r = c - 1), as at a crossing.
    d["target_rows"] = torch.where(flags != 0, d["chunk_size"] - 1, d["target_rows"]).to(
        torch.int32
    )
    run_state_rows(d, su.default_config(STATE), flags=flags)


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.parametrize("entries", [64, 256])
def test_production_decode_batch(entries):
    """Production: the Nemotron-H 8B layer at TP 1 (128 heads of 64, 8 groups, dstate 128, chunk 128),
    bf16, per-head D and the z gate, decode batches of 64 and 256 with padding entries."""
    g = ou.make_generator(device=DEVICE)
    d = decode_step(entries, 128, 8, 64, 128, torch.bfloat16, g, padding=(3, entries - 2))
    scan_config, bmm_config, state_config = (
        su.default_config(SCAN),
        su.default_config(BMM),
        su.default_config(STATE),
    )
    cb = run_bmm_rows(d, bmm_config)
    run_scan_rows(d, cb, scan_config, D=d["D_head"])
    run_state_rows(d, state_config)


# ---------------------------------------------------------------------------------------------
# Batch invariance: the row kernels reproduce the training tiles
# ---------------------------------------------------------------------------------------------


def _pipeline(d, D, gate, use_row_kernels):
    out = torch.zeros(d["slots"].numel(), *d["x"].shape[1:], device=DEVICE, dtype=d["x"].dtype)
    state = d["states"].clone()
    _mamba_chunk_scan_decode_rows(
        d["x"],
        d["z"] if gate else None,
        d["dt_raw"],
        d["A"],
        d["B"],
        d["C"],
        d["chunk_size"],
        chunk_starts=d["chunk_starts"],
        slots=d["slots"],
        target_rows=d["target_rows"],
        chunk_flags=d["chunk_flags"],
        initial_states=state,
        out=out,
        D=D,
        dt_bias=d["dt_bias"],
        dt_softplus=True,
        use_row_kernels=use_row_kernels,
    )
    return out, state


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("D_kind", ["head", "channel"])
def test_row_kernels_reproduce_the_training_tiles_bitwise(dtype, D_kind):
    """With the five training kernels pinned (batch-invariant mode's tiles), the decode step through the
    row kernels equals, bit for bit, the training kernels in target-rows mode: outputs and committed
    states, at the production layer shape (128 heads of 64, 8 groups, dstate 128) and 64 entries."""
    g = ou.make_generator(device=DEVICE)
    d = decode_step(64, 128, 8, 64, 128, dtype, g, padding=(3, 40))
    D = _D(d, D_kind)
    with contextlib.ExitStack() as stack:
        for kernel in TRAINING_KERNELS:
            stack.enter_context(su.pinned(kernel, su.default_config(kernel)))
        assert row_kernel_configs(d["C"], d["chunk_size"]) is not None, "row kernels not selected"
        for gate in (True, False):
            tile_out, tile_state = _pipeline(d, D, gate, use_row_kernels=False)
            row_out, row_state = _pipeline(d, D, gate, use_row_kernels=True)
            assert torch.isfinite(tile_out).all()
            ou.assert_exact(row_out, tile_out, what=f"outputs (z={gate})")
            ou.assert_exact(row_state, tile_state, what=f"states (z={gate})")


@ou.requires_sm(ou.HOPPER)
def test_selection_rule_declines_what_the_rows_cannot_reproduce():
    """`row_kernel_configs` (the dispatch): None while any of the three training kernels is unpinned,
    for a warpgroup-sized chunk-scan tile, for dstate above 128, and for a chunk size the scan tile does
    not divide; the pinned configs otherwise."""
    C = torch.empty(1, 1, 128, device=DEVICE)
    defaults = [su.default_config(k) for k in (SCAN, BMM, STATE)]
    for unpinned in (SCAN, BMM, STATE):
        with contextlib.ExitStack() as stack:
            for kernel, config in zip((SCAN, BMM, STATE), defaults):
                if kernel is unpinned:
                    others = [c for c in su.kernel_configs(kernel) if c is not config][:1]
                    stack.enter_context(su.with_configs(kernel, [config, *others]))
                else:
                    stack.enter_context(su.pinned(kernel, config))
            assert row_kernel_configs(C, 128) is None, "an unpinned kernel must keep the tile path"
    with su.pinned(SCAN, defaults[0]), su.pinned(BMM, defaults[1]), su.pinned(STATE, defaults[2]):
        assert row_kernel_configs(C, 128) == tuple(defaults)
        assert row_kernel_configs(torch.empty(1, 1, 129, device=DEVICE), 128) is None
        scan_m = defaults[0].kwargs["BLOCK_SIZE_M"]
        assert row_kernel_configs(C, scan_m // 2) is None
    for config in su.kernel_configs(SCAN):
        with su.pinned(SCAN, config), su.pinned(BMM, defaults[1]), su.pinned(STATE, defaults[2]):
            declined = config.num_warps >= 4 and config.kwargs["BLOCK_SIZE_M"] >= 64
            assert (row_kernel_configs(C, 128) is None) == declined, su.tile_id(config)


@pytest.mark.large
@ou.requires_sm(ou.HOPPER)
def test_replay_buffer_offsets_past_int32():
    """Replay buffers of 2049 slots at the 8B layer (x: 2049 * 128 * 8192 elements > 2^31) and slot
    states of 2^31 + 2^20 elements: entries on slots 2047 and 2048 read and commit past 2^31 elements.
    """
    heads, groups, headdim, dstate, c = 128, 8, 64, 128, su.PRODUCTION_CHUNK
    nslots = 2049
    ou.require_free_memory(30 * 2**30)
    g = ou.make_generator(device=DEVICE)
    x = torch.randn(nslots * c, heads, headdim, device=DEVICE, dtype=torch.bfloat16, generator=g)
    B = torch.randn(nslots * c, groups, dstate, device=DEVICE, dtype=torch.bfloat16, generator=g)
    C = torch.randn(nslots * c, groups, dstate, device=DEVICE, dtype=torch.bfloat16, generator=g)
    slots = torch.tensor([2048, 0, 2047, -1], dtype=torch.int32, device=DEVICE)
    target_rows = torch.tensor([c - 1, 17, 64, -1], dtype=torch.int32, device=DEVICE)
    chunk_starts = (slots * c).to(torch.int32)
    flags = (target_rows == c - 1).to(torch.int32)
    windows = su.fixed_windows(chunk_starts.clamp(min=0), c)
    A, bias = production_parameters(heads, g)
    dt_raw = torch.zeros(nslots * c, heads, device=DEVICE)
    for start, _ in windows:
        dt_raw[start : start + c] = torch.randn(c, heads, device=DEVICE, generator=g)
    dA, dt = ou.fp64_reference(
        cumsum_reference, dt_raw, A, bias, windows, c, softplus=True, dt_limit=(0.0, float("inf"))
    )
    states = torch.zeros(nslots, heads, headdim, dstate, device=DEVICE)
    for s in (0, 2047, 2048):
        states[s] = 0.1 * torch.randn(heads, headdim, dstate, device=DEVICE, generator=g)
    d = dict(
        x=x,
        B=B,
        C=C,
        dt=dt.float(),
        dA=dA.float(),
        z=torch.randn(4, heads, headdim, device=DEVICE, generator=g).bfloat16(),
        D_head=torch.randn(heads, device=DEVICE, generator=g),
        states=states,
        windows=windows,
        chunk_size=c,
        slots=slots,
        target_rows=target_rows,
        chunk_starts=chunk_starts,
        chunk_flags=flags,
    )
    cb = run_bmm_rows(d, su.default_config(BMM))
    run_scan_rows(d, cb, su.default_config(SCAN), D=d["D_head"])
    run_state_rows(d, su.default_config(STATE))


@pytest.mark.sanitizer
@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
def test_sanitizer_small_case():
    """Small case for compute-sanitizer: 6 entries (a padding entry, a crossing), headdim 80 and dstate
    100 (partial N and dstate tiles), per-channel D and the gate."""
    g = ou.make_generator(device=DEVICE)
    d = decode_step(6, 2, 1, 80, 100, torch.bfloat16, g, padding=(2,), rows=[127, 0, 5, 16, 33])
    cb = run_bmm_rows(d, su.default_config(BMM))
    run_scan_rows(d, cb, su.default_config(SCAN), D=d["D_channel"])
    run_state_rows(d, su.default_config(STATE))


def test_case_list_covers_every_factor():
    """Every dtype, layout, dstate, D kind and gate setting occurs in the pairwise sweep, and the target
    rows straddle the 16-row MMA tile (CPU check)."""
    cases = [p.values[0] for p in _shape_cases()]
    for factor in ("dtype", "heads_groups", "headdim", "dstate", "D", "gate"):
        assert len({str(c[factor]) for c in cases}) > 1
    assert {ROW_BLOCK_SIZE_M - 1, ROW_BLOCK_SIZE_M, ROW_BLOCK_SIZE_M + 1} <= set(ROWS)
