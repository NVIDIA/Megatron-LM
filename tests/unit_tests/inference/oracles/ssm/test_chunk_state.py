# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `_chunk_state_fwd_kernel` (megatron/core/ssm/ops/mamba2/ssd_chunk_state.py), the per-chunk
state contribution of the Mamba-2 chunked scan, through its launcher `_chunk_state_fwd`, in varlen mode
(prefill) and chunk-flags mode (the batch-invariant decode tile path).

Before this module the kernel was reached only through the varlen scan oracle (test_chunk_scan_varlen.py),
which sees it only through the scan's states and never in chunk-flags mode.

Contract (written from the Mamba-2 state recurrence and the launcher's callers, `_mamba_chunk_scan_combined_fwd`
and `_mamba_chunk_scan_decode_rows`, before reading the kernel body):

* Inputs: x (T, nheads, headdim) and B (T, ngroups, dstate) in the activation dtype (head h reads group
  h // (nheads / ngroups)); dt and dA_cumsum (nheads, nchunks, chunk_size) fp32, as `_chunk_cumsum_fwd`
  produces them (dt zero and dA_cumsum equal to the window's total past each window's length); the chunk
  windows: varlen, [cu[c], cu[c+1]); chunk-flags mode, [chunk_starts[c], chunk_starts[c] + chunk_size),
  computed only where chunk_flags[c] != 0.
* Output: states (nchunks, nheads, headdim, dstate), fp32 (`states_in_fp32`, every caller) or B's dtype:

      states[c, h, p, n] = sum_{k < len(c)} x[s_c + k, h, p] * B[s_c + k, g(h), n] * dt[h, c, k]
                                            * exp(dA_cumsum[h, c, len(c) - 1] - dA_cumsum[h, c, k])

  the state a chunk contributes from a zero entry state; an empty window gives 0. Unflagged entries are
  not written. Inputs are read-only.
* Numerics (the training scan's, documented by the scan oracle): fp32 accumulation on tensor cores, with
  the decayed B (B * exp(..) * dt, fp32) rounded to x's dtype before the dot; fp32 operands are read as
  TF32.
* Tolerance class: floating point. Reference: the sum above in fp64. Trusted: the same sum with the dot
  operands rounded as the tensor cores read them (decayed B rounded to x's dtype in fp32, TF32 for fp32),
  products and sums in fp64, the result rounded to the output dtype (`ssd_utils`). Exact: unflagged
  entries (NaN-poisoned through the launcher's own allocation), guard bands, read-only inputs, and every
  output for dyadic inputs with A = 0 (decay 1, every product and sum exact in fp32).

Variants and dispatch (from the kernel, read after the contract): 9 autotune configs (BLOCK_SIZE_M over
headdim 32/64/128, BLOCK_SIZE_N over dstate 32/64/128/256, BLOCK_SIZE_K over the window 32/64), each
pinned in turn on bf16 and fp32 inputs (`test_every_chunk_state_tile`, H100 and GB200, whose tensor-core
instructions differ); constexpr HAS_CHUNK_FLAGS; the K loop over the window (lengths 1, 31, 32, 33, c-1,
c); headdim 64 (production), 80 (Nemotron Nano 9B v2), 48, 33; dstate 128 (production), 64, 16, 129,
256; (nheads, ngroups) (8, 1), (8, 2), (1, 1), (8, 8); output in fp32 and in B's dtype; x/B as contiguous
tensors and as strided views. Offsets are int64 (`tl.int64` strides, int64 chunk id): no `large` case.

Main vs open PRs: no open PR changes this kernel.
"""

import pytest
import torch
from triton.runtime.errors import OutOfResources

from megatron.core.ssm.ops.mamba2 import ssd_chunk_state
from megatron.core.ssm.ops.mamba2.ssd_chunk_state import _chunk_state_fwd
from megatron.core.tuning.selection import cheapest
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.ssm import ssd_utils as su
from tests.unit_tests.inference.oracles.ssm.test_chunk_cumsum import (
    cumsum_reference,
    production_parameters,
)

DEVICE = "cuda"
KERNEL = ssd_chunk_state._chunk_state_fwd_kernel


# ---------------------------------------------------------------------------------------------
# Reference and trusted implementation
# ---------------------------------------------------------------------------------------------


def _window_terms(x, B, dt, dA, windows, chunk_size):
    """Per window: x and B rows (nc, chunk, H, .), the decay to the window end and dt (nc, chunk, H)."""
    heads, groups = x.shape[1], B.shape[1]
    goh = su.group_of_head(heads, groups, x.device)
    xw = su.gather_windows(x, windows, chunk_size)
    Bw = su.gather_windows(B, windows, chunk_size)[:, :, goh]
    lengths = torch.tensor([n for _, n in windows], device=x.device)
    last = (lengths - 1).clamp(min=0)[None, :, None].expand(heads, -1, 1)
    total = dA.gather(2, last)  # (H, nc, 1): the decay of the whole window
    decay = (total - dA).permute(1, 2, 0)
    return xw, Bw, decay, dt.permute(1, 2, 0)


def chunk_state_reference(x, B, dt, dA, windows, chunk_size):
    """The contract in the inputs' dtype (fp64 for the reference): states (nc, H, P, N)."""
    xw, Bw, decay, dtw = _window_terms(x, B, dt, dA, windows, chunk_size)
    weighted = xw * (torch.exp(decay) * dtw)[..., None]
    return torch.einsum("ckhp,ckhn->chpn", weighted, Bw)


def trusted_chunk_state(x, B, dt, dA, windows, chunk_size, out_dtype):
    """The contract with the dot operands as the tensor cores read them (see the module docstring)."""
    xw, Bw, decay, dtw = _window_terms(x, B, dt.float(), dA.float(), windows, chunk_size)
    scale = torch.exp(decay.clamp(max=0.0)) * dtw  # fp32, the decayed weight of each token
    b = su.as_operand(Bw.float() * scale[..., None], x.dtype)
    states = torch.einsum("ckhp,ckhn->chpn", su.as_operand(xw, x.dtype), b)
    return states.to(out_dtype)


def scan_inputs(dt_raw, A, dt_bias, windows, chunk_size):
    """dt and dA_cumsum as `_chunk_cumsum_fwd` produces them (computed here from the contract in fp64,
    rounded once to fp32), so this oracle does not depend on the cumsum kernel."""
    dA, dt = ou.fp64_reference(
        cumsum_reference,
        dt_raw,
        A,
        dt_bias,
        windows,
        chunk_size,
        softplus=True,
        dt_limit=(0.0, float("inf")),
    )
    return dt.float(), dA.float()


def run_chunk_state(
    x,
    B,
    dt,
    dA,
    chunk_size,
    *,
    cu_chunk_seqlens=None,
    chunk_starts=None,
    chunk_flags=None,
    states_in_fp32=True,
    exact=False,
):
    """Launch `_chunk_state_fwd` with a poisoned, guarded output and check it against the contract."""
    if chunk_flags is None:
        windows = su.varlen_windows(cu_chunk_seqlens)
        live = list(range(len(windows)))
    else:
        windows = su.fixed_windows(chunk_starts, chunk_size)
        live = [c for c in range(len(windows)) if int(chunk_flags[c]) != 0]
    inputs = dict(x=x, B=B, dt=dt, dA=dA)
    before = {k: v.clone() for k, v in inputs.items()}
    with su.poisoned_allocations(ssd_chunk_state) as buffers:
        states = _chunk_state_fwd(
            B,
            x,
            dt,
            dA,
            cu_chunk_seqlens,
            states_in_fp32=states_in_fp32,
            chunk_flags=chunk_flags,
            chunk_starts=chunk_starts,
        )
    su.check_all(buffers, "chunk states")
    for name, value in before.items():
        ou.assert_exact(inputs[name], value, what=f"{name} is read-only")
    out_dtype = torch.float32 if states_in_fp32 else B.dtype
    assert states.dtype == out_dtype
    assert states.shape == (len(windows), x.shape[1], x.shape[2], B.shape[2])
    skipped = [c for c in range(len(windows)) if c not in live]
    if skipped:
        ou.assert_poisoned(states[skipped], what="states of unflagged entries")
    live_windows = [windows[c] for c in live]
    # The kernel reads a window's total decay at chunk_size - 1, the interface the cumsum keeps.
    sliced = dict(dt=dt[:, live], dA=dA[:, live])
    ref = ou.fp64_reference(
        chunk_state_reference, x, B, sliced["dt"], sliced["dA"], live_windows, chunk_size
    )
    if exact:
        su.assert_equal(states[live], ref, what="chunk states (exact inputs)")
        return states
    trusted = trusted_chunk_state(
        x, B, sliced["dt"], sliced["dA"], live_windows, chunk_size, out_dtype
    )
    ou.assert_close_to_reference(states[live], ref, trusted, what="chunk states")
    return states


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


PACKS = {
    # name: (request lengths in terms of the chunk size c, padded requests, padded tail tokens)
    "boundaries": (lambda c: [1, c - 1, c, c + 1, 2 * c, 2 * c + 3], 2, 5),
    "kblock": (lambda c: [31, 32, 33, 65, c - 1], 0, 3),  # around BLOCK_SIZE_K = 32 and 64
    "single": (lambda c: [3 * c + 5], 0, 0),
}


def make_inputs(
    lengths_or_pack, chunk_size, heads, groups, headdim, dstate, dtype, g, *, strided=False
):
    """A prefill pack with production metadata: x, B, dt, dA_cumsum and the metadata. Padded tail
    tokens are NaN (no window covers them)."""
    if isinstance(lengths_or_pack, str):
        make, padded_requests, tail = PACKS[lengths_or_pack]
        lengths = make(chunk_size)
    else:
        lengths, padded_requests, tail = lengths_or_pack, 0, 0
    meta = su.production_metadata(lengths, padded_requests, tail, chunk_size, DEVICE)
    total = meta["padded_tokens"]
    if strided:
        # x and B as views into one projection row (the mixer's split of xBC), row stride > width.
        width = heads * headdim + groups * dstate + 7
        proj = torch.randn(total, width, device=DEVICE, generator=g).to(dtype)
        x = proj[:, : heads * headdim].view(total, heads, headdim)
        B = proj[:, heads * headdim : heads * headdim + groups * dstate].view(total, groups, dstate)
    else:
        x = torch.randn(total, heads, headdim, device=DEVICE, generator=g).to(dtype)
        B = torch.randn(total, groups, dstate, device=DEVICE, generator=g).to(dtype)
    x[meta["real_tokens"] :] = float("nan")
    B[meta["real_tokens"] :] = float("nan")
    dt_raw = torch.randn(total, heads, device=DEVICE, generator=g)
    A, bias = production_parameters(heads, g)
    windows = su.varlen_windows(meta["cu_chunk_seqlens"])
    dt, dA = scan_inputs(dt_raw, A, bias, windows, chunk_size)
    return x, B, dt, dA, meta


def _cases():
    factors = {
        "pack": ("boundaries", "kblock", "single"),
        "chunk_size": (su.PRODUCTION_CHUNK, 64, 256),
        "heads_groups": ((8, 1), (8, 2), (1, 1), (8, 8)),
        "headdim": (64, 80, 48, 33),
        "dstate": (128, 64, 16, 129, 256),
        "dtype": (torch.bfloat16, torch.float32),
        "states_in_fp32": (True, False),
        "strided": (False, True),
    }
    params = []
    for case in ou.pairwise_cases(factors):
        case_id = "-".join(
            [
                case["pack"],
                f"c{case['chunk_size']}",
                "h{}g{}".format(*case["heads_groups"]),
                f"p{case['headdim']}",
                f"n{case['dstate']}",
                str(case["dtype"]).replace("torch.", ""),
                "fp32_states" if case["states_in_fp32"] else "B_dtype_states",
                "strided" if case["strided"] else "contiguous",
            ]
        )
        params.append(pytest.param(case, id=case_id))
    return params


def _run_case(case):
    g = ou.make_generator(device=DEVICE)
    heads, groups = case["heads_groups"]
    x, B, dt, dA, meta = make_inputs(
        case["pack"],
        case["chunk_size"],
        heads,
        groups,
        case["headdim"],
        case["dstate"],
        case["dtype"],
        g,
        strided=case["strided"],
    )
    run_chunk_state(
        x,
        B,
        dt,
        dA,
        case["chunk_size"],
        cu_chunk_seqlens=meta["cu_chunk_seqlens"],
        states_in_fp32=case["states_in_fp32"],
    )


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", _cases())
def test_varlen_chunk_state_matches_fp64(case):
    """Production pseudo-chunks against the fp64 contract, pairwise over the shapes, dtypes, output
    dtype and layouts, on the tile batch-invariant mode pins (the tile sweep covers the others)."""
    with su.pinned(KERNEL, su.default_config(KERNEL)):
        _run_case(case)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("chunk_size", [64, su.PRODUCTION_CHUNK])
def test_chunk_state_exact_on_dyadic_inputs(chunk_size, dtype):
    """With A = 0 (decay 1) and dyadic x, B and dt (3 significant bits each, B * dt exact in bf16 and in
    TF32), every product and partial sum is exact in fp32: windows, masks, the head-to-group map and the
    K loop are checked bit for bit, in varlen and chunk-flags mode."""
    g = ou.make_generator(device=DEVICE)
    heads, groups, headdim, dstate = 8, 2, 48, 129
    meta = su.production_metadata(
        [1, 31, 33, chunk_size, 2 * chunk_size + 3], 1, 2, chunk_size, DEVICE
    )
    total = meta["padded_tokens"]

    def dyadic(*shape, exponents=(0, 0)):
        return ou.dyadic_tensor(
            shape,
            mantissa_bits=3,
            exponents=exponents,
            dtype=torch.float32,
            device=DEVICE,
            generator=g,
        )

    x = dyadic(total, heads, headdim).to(dtype)
    B = dyadic(total, groups, dstate).to(dtype)
    windows = su.varlen_windows(meta["cu_chunk_seqlens"])
    dt_tokens = dyadic(total, heads, exponents=(-1, 0)).abs()
    dt = su.gather_windows(dt_tokens, windows, chunk_size).permute(2, 0, 1).contiguous()
    dA = torch.zeros_like(dt)
    run_chunk_state(x, B, dt, dA, chunk_size, cu_chunk_seqlens=meta["cu_chunk_seqlens"], exact=True)

    entries = 9
    win = su.decode_windows(entries, chunk_size, padding=(4,), device=DEVICE)
    x = dyadic(entries * chunk_size, heads, headdim).to(dtype)
    B = dyadic(entries * chunk_size, groups, dstate).to(dtype)
    dt = dyadic(heads, entries, chunk_size, exponents=(-1, 0)).abs()
    flags = win["chunk_flags"].clone()
    flags[1::2] = 1  # more crossings than the decode pattern gives
    flags[4] = 0  # the padding entry never crosses
    run_chunk_state(
        x,
        B,
        dt,
        torch.zeros_like(dt),
        chunk_size,
        chunk_starts=win["chunk_starts"],
        chunk_flags=flags,
        exact=True,
    )


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_chunk_flags_windows(dtype):
    """Chunk-flags mode (batch-invariant decode): fixed windows [slot * c, slot * c + c) of the replay
    buffer, only flagged entries computed, unflagged and padding entries (chunk start -c) untouched.
    """
    g = ou.make_generator(device=DEVICE)
    c, heads, groups, headdim, dstate, entries = su.PRODUCTION_CHUNK, 8, 2, 64, 128, 23
    win = su.decode_windows(entries, c, padding=(5,), device=DEVICE)
    flags = win["chunk_flags"].clone()
    flags[2::3] = 1
    flags[5] = 0
    x = torch.randn(entries * c, heads, headdim, device=DEVICE, generator=g).to(dtype)
    B = torch.randn(entries * c, groups, dstate, device=DEVICE, generator=g).to(dtype)
    A, bias = production_parameters(heads, g)
    dt_raw = torch.randn(entries * c, heads, device=DEVICE, generator=g)
    windows = su.fixed_windows(win["chunk_starts"].clamp(min=0), c)
    dt, dA = scan_inputs(dt_raw, A, bias, windows, c)
    run_chunk_state(x, B, dt, dA, c, chunk_starts=win["chunk_starts"], chunk_flags=flags)


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_every_chunk_state_tile(dtype):
    """Every autotune candidate pinned in turn on the boundary pack at production head dim and dstate
    (headdim 64, dstate 128, chunk 128, 8 heads in 2 groups), varlen and chunk-flags mode. A candidate
    the device cannot launch (shared memory) is dropped only after that error, and then neither the
    pinned policy's cheapest fallback nor timed autotuning may land on it."""
    configs = su.kernel_configs(KERNEL)
    too_big, failures = [], []
    case = dict(
        pack="kblock",
        chunk_size=su.PRODUCTION_CHUNK,
        heads_groups=(8, 2),
        headdim=64,
        dstate=128,
        dtype=dtype,
        states_in_fp32=True,
        strided=False,
    )
    for config in configs:
        with su.pinned(KERNEL, config):
            try:
                _run_case(case)
                _run_case(dict(case, pack="boundaries", dstate=129, headdim=80))
            except OutOfResources as err:
                assert err.name == "shared memory", f"{su.tile_id(config)}: {err}"
                too_big.append(config)
            except AssertionError as err:
                failures.append(f"{su.tile_id(config)}: {err}")
    assert not failures, "pinned chunk-state configs disagree with the contract:\n" + "\n".join(
        failures
    )
    assert len(too_big) < len(configs), "no chunk-state config fits this GPU"
    assert cheapest(configs) not in too_big, "the pinned policy's fallback cannot launch here"
    tuner = su.autotuner(KERNEL)
    saved = tuner.cache
    tuner.cache = {}
    try:
        _run_case(case)
        chosen = tuner.best_config
    finally:
        tuner.cache = saved
    assert (
        chosen not in too_big
    ), f"timed autotuning chose {su.tile_id(chosen)}, which cannot launch"


@ou.requires_sm(ou.HOPPER)
def test_production_prefill_pack():
    """Production: the Nemotron-H 8B layer at TP 1 (128 heads of 64, 8 groups, dstate 128, chunk 128),
    an 8193-token bf16 pack of mixed lengths."""
    g = ou.make_generator(device=DEVICE)
    lengths = [4096, 2049, 1000, 128, 127, 1, 792]
    x, B, dt, dA, meta = make_inputs(lengths, 128, 128, 8, 64, 128, torch.bfloat16, g)
    with su.pinned(KERNEL, su.default_config(KERNEL)):
        run_chunk_state(x, B, dt, dA, 128, cu_chunk_seqlens=meta["cu_chunk_seqlens"])


@pytest.mark.sanitizer
@ou.requires_sm(ou.HOPPER)
def test_sanitizer_small_case():
    """Small case for compute-sanitizer: a padded pack at chunk 64 with headdim 33 and dstate 129 (partial
    M and N tiles), then chunk-flags mode with a padding entry."""
    g = ou.make_generator(device=DEVICE)
    x, B, dt, dA, meta = make_inputs("boundaries", 64, 2, 1, 33, 129, torch.bfloat16, g)
    with su.pinned(KERNEL, su.default_config(KERNEL)):
        run_chunk_state(x, B, dt, dA, 64, cu_chunk_seqlens=meta["cu_chunk_seqlens"])
        win = su.decode_windows(4, 64, padding=(1,), device=DEVICE)
        x = torch.randn(4 * 64, 2, 33, device=DEVICE, generator=g).bfloat16()
        B = torch.randn(4 * 64, 1, 129, device=DEVICE, generator=g).bfloat16()
        flags = torch.tensor([1, 0, 1, 0], dtype=torch.int32, device=DEVICE)
        dt, dA = dt[:, :4].contiguous(), dA[:, :4].contiguous()
        run_chunk_state(x, B, dt, dA, 64, chunk_starts=win["chunk_starts"], chunk_flags=flags)


def test_case_list_covers_every_factor():
    """Every pack, shape and layout value occurs in the pairwise sweep (CPU check)."""
    cases = [p.values[0] for p in _cases()]
    for factor in ("pack", "chunk_size", "heads_groups", "headdim", "dstate", "dtype", "strided"):
        assert len({str(c[factor]) for c in cases}) > 1
