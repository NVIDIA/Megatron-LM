# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `_state_passing_fwd_kernel` (megatron/core/ssm/ops/mamba2/ssd_state_passing.py), the
inter-chunk recurrence of the Mamba-2 chunked scan, through its launcher `_state_passing_fwd`, in varlen
mode (prefill: chunk-exit states, with each sequence's initial state) and dst-states mode (the
batch-invariant decode tile path: flagged entries commit their boundary state into the slot's state).

Before this module the kernel was reached only through the varlen scan oracle (test_chunk_scan_varlen.py),
which sees its chunk-exit states but never the dst-states mode, a bf16 output with the chunk structure
exposed, or the per-config tiles.

Contract (written from the Mamba-2 recurrence and the launcher's callers, `_mamba_chunk_scan_combined_fwd`
and `_mamba_chunk_scan_decode_rows`, before reading the kernel body):

* Inputs: states (nchunks, nheads, dim) fp32, each chunk's contribution from a zero entry state (dim =
  headdim * dstate, flattened); dA_cumsum (nheads, nchunks, chunk_size) fp32, of which a chunk's total
  decay is the value at chunk_size - 1 (`_chunk_cumsum_fwd`'s interface); seq_idx (nchunks,) the chunk's
  sequence (varlen) or slot (dst mode); initial_states (nsequences or nslots, nheads, dim) or None.
* Varlen mode: a running state S (fp32), restarted from initial_states[seq_idx[c]] (zeros without
  initial states) at chunk 0 and whenever seq_idx[c] differs from seq_idx[c - 1]; then

      S = exp(dA_cumsum[h, c, chunk_size - 1]) * S + states[c];   out[c] = S rounded to out_dtype

  out (nchunks, nheads, dim) holds every chunk-exit state; the carry itself is not rounded.
* dst mode (dst_states, dst_indices, dst_flags): only chunks with dst_flags[c] != 0, each from
  initial_states[seq_idx[c]]: dst_states[dst_indices[c]] = exp(dA total) * initial + states[c]. Every other
  dst row is untouched, nothing else is written, and the launcher returns None. In production
  initial_states and dst_states are the same fp32 SSM state tensor and seq_idx = dst_indices = the slots.
* Inputs are read-only (except dst_states).
* Tolerance class: floating point. Reference: the recurrence in fp64. Trusted: the recurrence in torch fp32
  on the device (exp, multiply, add). Exact: dyadic inputs with a zero decay (exp(0) = 1, every sum
  exact), the restart rule (a NaN-poisoned initial state of a padded request reaches exactly its own
  chunk), untouched dst rows, guard bands.

Finding O7-M3 (H100, pdx 7986854): dst mode updates `dst_states` in place, and production passes the same
tensor as `initial_states`, so the update is not idempotent; on main the kernel's `@triton.autotune` has no
`restore_value`, so timed autotuning (a cold cache at a new key, with more than one candidate) runs the
update once per benchmark launch. Batch-invariant mode pins the autotuner (`enable_batch_invariant_mode`
-> `_pin_mamba_autotuners`), which avoids it; a batch-invariant mixer run without that pin does not.
#9309 (fix/state-passing-autotune-restore) restores the dst states around each benchmark launch and
carries the cold-cache case as its regression test
(tests/unit_tests/ssm/ops/mamba2/test_ssd_state_passing.py,
`test_dst_states_commit_once_under_timed_autotuning`); the dst-mode cases here pin the autotuner.

Variants and dispatch (from the kernel, read after the contract): autotune BLOCK_SIZE in {64, ..., 2048}
keyed on dim, each pinned in turn at dim BLOCK_SIZE - 1, BLOCK_SIZE, BLOCK_SIZE + 1; constexpr
HAS_INITSTATES (with and without initial states) and HAS_DST_STATES; out_dtype fp32 and bf16 (an SSM
cache in bf16); production dim 64 x 128 = 8192. Offsets are int64 (`tl.int64` strides and sequence ids):
no `large` case.

Main vs open PRs: #9309 (above) restores the dst states while the autotuner times the kernel; launches
outside timed autotuning are unchanged.
"""

import pytest
import torch

from megatron.core.ssm.ops.mamba2 import ssd_state_passing
from megatron.core.ssm.ops.mamba2.ssd_state_passing import _state_passing_fwd
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.ssm import ssd_utils as su
from tests.unit_tests.inference.oracles.ssm.test_chunk_cumsum import (
    cumsum_reference,
    production_parameters,
)

DEVICE = "cuda"
KERNEL = ssd_state_passing._state_passing_fwd_kernel


# ---------------------------------------------------------------------------------------------
# Reference and trusted implementation
# ---------------------------------------------------------------------------------------------


def passing_reference(states, dA, seq_idx, initial_states, out_dtype=None):
    """Varlen mode: every chunk-exit state (nchunks, nheads, dim), in the inputs' dtype (fp64 for the
    reference; fp32 for the trusted implementation, which then rounds to `out_dtype`)."""
    nchunks = states.shape[0]
    decay = torch.exp(dA[:, :, -1])  # (H, nchunks): each chunk's total decay
    owners = seq_idx.tolist()
    out = torch.empty_like(states)
    S = None
    for c in range(nchunks):
        if c == 0 or owners[c] != owners[c - 1]:
            S = (
                torch.zeros_like(states[0])
                if initial_states is None
                else initial_states[owners[c]].to(states.dtype)
            )
        S = decay[:, c, None] * S + states[c]
        out[c] = S
    return out if out_dtype is None else out.to(out_dtype)


def dst_reference(states, dA, seq_idx, initial_states, dst_indices, dst_flags, dst_before):
    """dst mode: the dst tensor after the launch (flagged rows replaced, every other row as before)."""
    out = dst_before.clone()
    decay = torch.exp(dA[:, :, -1])
    for c in range(states.shape[0]):
        if int(dst_flags[c]) != 0:
            start = initial_states[int(seq_idx[c])].to(states.dtype)
            out[int(dst_indices[c])] = decay[:, c, None] * start + states[c]
    return out


# ---------------------------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------------------------


def run_varlen(states, dA, seq_idx, initial_states, out_dtype, *, exact=False):
    """Launch `_state_passing_fwd` in varlen mode with a poisoned, guarded output and check it."""
    inputs = dict(states=states, dA=dA, seq_idx=seq_idx)
    if initial_states is not None:
        inputs["initial_states"] = initial_states
    before = {k: v.clone() for k, v in inputs.items()}
    with su.poisoned_allocations(ssd_state_passing) as buffers:
        out = _state_passing_fwd(
            states, dA, None, seq_idx, initial_states=initial_states, out_dtype=out_dtype
        )
    su.check_all(buffers, "chunk-exit states")
    for name, value in before.items():
        ou.assert_exact(inputs[name], value, what=f"{name} is read-only")
    assert out.shape == states.shape and out.dtype == (out_dtype or states.dtype)
    ref = ou.fp64_reference(passing_reference, states, dA, seq_idx, initial_states)
    if exact:
        su.assert_equal(out, ref, what="chunk-exit states (exact inputs)")
        return out
    trusted = passing_reference(
        states.float(), dA.float(), seq_idx, initial_states, out_dtype=out.dtype
    )
    ou.assert_close_to_reference(out, ref, trusted, what="chunk-exit states")
    return out


def run_dst(states, dA, slots, flags, initial_states, dst, *, exact=False):
    """Launch `_state_passing_fwd` in dst mode (`dst` may be `initial_states` itself, as in production)
    on a guarded dst and check every dst row."""
    guarded = ou.guarded_like(dst)
    aliased = dst is initial_states
    init = guarded.tensor if aliased else initial_states
    dst_before = dst.clone()
    init_before = init.clone()
    inputs = dict(states=states, dA=dA, slots=slots, flags=flags)
    before = {k: v.clone() for k, v in inputs.items()}
    with su.poisoned_allocations(ssd_state_passing) as buffers:
        result = _state_passing_fwd(
            states,
            dA,
            None,
            slots,
            initial_states=init,
            dst_states=guarded.tensor,
            dst_indices=slots,
            dst_flags=flags,
        )
    assert result is None, "dst mode returns nothing"
    assert not buffers, "dst mode allocates no output"
    ou.check_guards(dst=guarded)
    for name, value in before.items():
        ou.assert_exact(inputs[name], value, what=f"{name} is read-only")
    if not aliased:
        ou.assert_exact(initial_states, init_before, what="initial_states is read-only")
    ref = ou.fp64_reference(dst_reference, states, dA, slots, init_before, slots, flags, dst_before)
    flagged = sorted({int(slots[c]) for c in range(len(slots)) if int(flags[c]) != 0})
    others = [s for s in range(dst.shape[0]) if s not in flagged]
    ou.assert_exact(guarded.tensor[others], dst_before[others], what="dst rows of no flagged entry")
    if exact:
        su.assert_equal(guarded.tensor, ref, what="dst states (exact inputs)")
        return
    trusted = dst_reference(
        states.float(), dA.float(), slots, init_before.float(), slots, flags, dst_before.float()
    ).to(dst.dtype)
    ou.assert_close_to_reference(
        guarded.tensor[flagged], ref[flagged], trusted[flagged], what="committed slot states"
    )


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


def varlen_inputs(lengths, padded_requests, tail, chunk_size, heads, dim, g, *, init=True):
    """Chunk contributions and decays for a pack with production metadata: random fp32 contributions,
    the cumsum of production-range dt and A as dA_cumsum; padded requests' initial states NaN."""
    meta = su.production_metadata(lengths, padded_requests, tail, chunk_size, DEVICE)
    windows = su.varlen_windows(meta["cu_chunk_seqlens"])
    nchunks = len(windows)
    states = torch.randn(nchunks, heads, dim, device=DEVICE, generator=g)
    dt_raw = torch.randn(meta["padded_tokens"], heads, device=DEVICE, generator=g)
    A, bias = production_parameters(heads, g)
    dA, _ = ou.fp64_reference(
        cumsum_reference,
        dt_raw,
        A,
        bias,
        windows,
        chunk_size,
        softplus=True,
        dt_limit=(0.0, float("inf")),
    )
    nseq = len(lengths) + padded_requests
    initial = None
    if init:
        initial = torch.randn(nseq, heads, dim, device=DEVICE, generator=g)
        initial[len(lengths) :] = float(
            "nan"
        )  # padded requests gather an unused slot in production
    return states, dA.float(), meta["seq_idx"], initial, meta


def _cases():
    factors = {
        "pack": ("boundaries", "short", "single"),
        "chunk_size": (su.PRODUCTION_CHUNK, 64, 256),
        "heads": (8, 1, 3),
        "dim": (8192, 6192, 65, 2049),
        "init": (True, False),
        "out_dtype": (torch.float32, torch.bfloat16),
    }
    params = []
    for case in ou.pairwise_cases(factors):
        case_id = "-".join(
            [
                case["pack"],
                f"c{case['chunk_size']}",
                f"h{case['heads']}",
                f"dim{case['dim']}",
                "init" if case["init"] else "noinit",
                str(case["out_dtype"]).replace("torch.", ""),
            ]
        )
        params.append(pytest.param(case, id=case_id))
    return params


PACKS = {
    "boundaries": (lambda c: [1, c - 1, c, c + 1, 2 * c, 2 * c + 3], 2, 5),
    "short": (lambda c: [3, 1, c - 1], 1, 0),
    "single": (lambda c: [5 * c + 5], 0, 0),
}


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", _cases())
def test_varlen_state_passing_matches_fp64(case):
    """Production pseudo-chunks: the running state restarts at each request (its initial state, NaN for
    padded requests, which must reach only their own chunk), pairwise over sizes and output dtype.
    """
    g = ou.make_generator(device=DEVICE)
    make, padded, tail = PACKS[case["pack"]]
    states, dA, seq_idx, initial, _ = varlen_inputs(
        make(case["chunk_size"]),
        padded,
        tail,
        case["chunk_size"],
        case["heads"],
        case["dim"],
        g,
        init=case["init"],
    )
    with su.pinned(KERNEL, su.default_config(KERNEL)):
        run_varlen(states, dA, seq_idx, initial, case["out_dtype"])


def _block_cases():
    params = []
    for index, config in enumerate(su.kernel_configs(KERNEL)):
        block = config.kwargs["BLOCK_SIZE"]
        for dim in (block - 1, block, block + 1):
            params.append(pytest.param(index, dim, id=f"{su.tile_id(config)}-dim{dim}"))
    return params


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("config_index,dim", _block_cases())
def test_every_block_size_at_its_edges(config_index, dim):
    """Each BLOCK_SIZE candidate pinned, at dim BLOCK_SIZE - 1, BLOCK_SIZE and + 1, varlen with initial
    states (dyadic contributions and zero decay: exact) and dst mode (random: bounded)."""
    g = ou.make_generator(device=DEVICE)
    c, heads = 64, 2
    meta = su.production_metadata([1, c, 2 * c + 1, 3], 1, 2, c, DEVICE)
    nchunks = meta["cu_chunk_seqlens"].numel() - 1
    states = ou.dyadic_tensor(
        (nchunks, heads, dim), dtype=torch.float32, device=DEVICE, generator=g
    )
    initial = ou.dyadic_tensor((5, heads, dim), dtype=torch.float32, device=DEVICE, generator=g)
    dA = torch.zeros(heads, nchunks, c, device=DEVICE)
    with su.pinned(KERNEL, su.kernel_configs(KERNEL)[config_index]):
        run_varlen(states, dA, meta["seq_idx"], initial, None, exact=True)
        entries = 7
        win = su.decode_windows(entries, c, padding=(2,), device=DEVICE)
        flags = win["chunk_flags"].clone()
        flags[1::2] = 1
        flags[2] = 0
        states = torch.randn(entries, heads, dim, device=DEVICE, generator=g)
        dA = -torch.rand(heads, entries, c, device=DEVICE, generator=g).cumsum(-1)
        ssm = torch.randn(entries, heads, dim, device=DEVICE, generator=g)
        run_dst(states, dA, win["slots"], flags, ssm, ssm)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("aliased", [True, False], ids=["in_place", "separate_dst"])
@pytest.mark.parametrize("dyadic", [False, True], ids=["random", "exact"])
def test_dst_states_commit_flagged_entries(aliased, dyadic):
    """dst mode as batch-invariant decode runs it: shuffled slots, a padding entry (slot -1, flag 0),
    flagged entries commit exp(total) * slot state + contribution into their slot; every other slot row
    is untouched. In place (initial_states is dst_states, production) and into a separate dst. The
    autotuner is pinned, as batch-invariant mode pins it (`_pin_mamba_autotuners`); timed autotuning is
    #9309's `test_dst_states_commit_once_under_timed_autotuning` (finding O7-M3)."""
    with su.pinned(KERNEL, su.default_config(KERNEL)):
        run_dst(*dst_case(dyadic, aliased), exact=dyadic)


def dst_case(dyadic, aliased, seed=ou.DEFAULT_SEED):
    """A decode step's dst-mode inputs: (states, dA, slots, flags, initial_states, dst)."""
    g = ou.make_generator(seed, device=DEVICE)
    entries, heads, dim, c = 29, 4, 64 * 128, su.PRODUCTION_CHUNK
    win = su.decode_windows(entries, c, padding=(0, 13), device=DEVICE)
    flags = win["chunk_flags"].clone()
    flags[3::4] = 1
    flags[0] = flags[13] = 0
    slots_total = entries + 3  # slots no entry maps to stay untouched
    if dyadic:
        states = ou.dyadic_tensor(
            (entries, heads, dim), dtype=torch.float32, device=DEVICE, generator=g
        )
        ssm = ou.dyadic_tensor(
            (slots_total, heads, dim), dtype=torch.float32, device=DEVICE, generator=g
        )
        dA = torch.zeros(heads, entries, c, device=DEVICE)
    else:
        states = torch.randn(entries, heads, dim, device=DEVICE, generator=g)
        ssm = torch.randn(slots_total, heads, dim, device=DEVICE, generator=g)
        dA = -(torch.rand(heads, entries, c, device=DEVICE, generator=g) * 0.05).cumsum(-1)
    dst = ssm if aliased else torch.randn_like(ssm)
    return states, dA, win["slots"], flags, ssm, dst


@ou.requires_sm(ou.HOPPER)
def test_production_prefill_pack():
    """Production: the Nemotron-H 8B layer at TP 1 (128 heads, dim 64 x 128), chunk 128, an 8193-token
    pack of mixed lengths with cached initial states, fp32 out (the default SSM cache) and bf16."""
    g = ou.make_generator(device=DEVICE)
    lengths = [4096, 2049, 1000, 128, 127, 1, 792]
    states, dA, seq_idx, initial, _ = varlen_inputs(lengths, 0, 0, 128, 128, 64 * 128, g)
    for out_dtype in (torch.float32, torch.bfloat16):
        run_varlen(states, dA, seq_idx, initial, out_dtype)


@pytest.mark.sanitizer
@ou.requires_sm(ou.HOPPER)
def test_sanitizer_small_case():
    """Small case for compute-sanitizer: a padded pack (dim 65, a partial block) and dst mode in place."""
    g = ou.make_generator(device=DEVICE)
    make, padded, tail = PACKS["boundaries"]
    states, dA, seq_idx, initial, _ = varlen_inputs(make(64), padded, tail, 64, 2, 65, g)
    with su.pinned(KERNEL, su.default_config(KERNEL)):
        run_varlen(states, dA, seq_idx, initial, torch.bfloat16)
        win = su.decode_windows(5, 64, padding=(1,), device=DEVICE)
        flags = torch.tensor([1, 0, 1, 0, 1], dtype=torch.int32, device=DEVICE)
        ssm = torch.randn(5, 2, 65, device=DEVICE, generator=g)
        run_dst(states[:5], dA[:, :5].contiguous(), win["slots"], flags, ssm, ssm)


def test_case_list_covers_every_factor():
    """Every pack, size, init mode and output dtype occurs in the pairwise sweep (CPU check)."""
    cases = [p.values[0] for p in _cases()]
    for factor in ("pack", "chunk_size", "heads", "dim", "init", "out_dtype"):
        assert len({str(c[factor]) for c in cases}) > 1
