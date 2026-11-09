# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `_chunk_cumsum_fwd_kernel` (megatron/core/ssm/ops/mamba2/ssd_chunk_state.py), the first stage
of the Mamba-2 chunked scan, through its launcher `_chunk_cumsum_fwd`, in both of its modes: varlen
(prefill, `mamba_chunk_scan_combined_varlen`) and fixed windows with target rows (the batch-invariant
decode tile path, `_mamba_chunk_scan_decode_rows`).

Before this module the kernel was reached only through the varlen scan oracle (test_chunk_scan_varlen.py),
which sees its outputs only through `y` and the states, and never in target-rows mode.

Contract (written from the Mamba-2 discretization, the launcher's docstring and its two callers before
reading the kernel body):

* Inputs: dt (T, nheads) in the activation dtype, A (nheads,) fp32 (A = -exp(A_log)), dt_bias (nheads,)
  or None, `chunk_size` a power of two, and the chunk windows: varlen, window c = [cu[c], cu[c+1]) from
  `cu_chunk_seqlens` (production pseudo-chunks, `MambaMetadata.compute_cpu_metadata`, at most chunk_size
  long, zero-length for padded requests); target-rows mode, window c = [chunk_starts[c], chunk_starts[c] +
  chunk_size), skipped entirely when target_rows[c] < 0 (a padding entry).
* Outputs, fp32, (nheads, nchunks, chunk_size), for every window c and head h, positions k < len(c):

      v           = dt[start_c + k, h] + dt_bias[h]                       (no bias without dt_bias)
      dt_out      = clamp(softplus(v) if dt_softplus else v, dt_min, dt_max)
      dA_cumsum   = sum_{i <= k} dt_out[h, c, i] * A[h]

  and for positions len(c) <= k < chunk_size: dt_out = 0 and dA_cumsum = the window's total (zero for an
  empty window). The chunk-state and state-passing kernels read the window's total decay at position
  chunk_size - 1, so this padding is part of the interface. The total there is the prefix sum at that
  position, which adds the same terms as position len(c) - 1 but, in a parallel scan, possibly grouped
  differently: equal to dA_cumsum[len(c) - 1] only up to fp32 rounding (H100, pdx 7986854: 1-2 ulp in
  29 of the cases), exactly when every partial sum is exact (the dyadic cases). A skipped
  target-rows entry writes nothing. Tokens outside every window (padded tail tokens) are never read; dt
  is read-only.
* softplus(v): log(1 + exp(v)) for v <= 20, v above. The kernel and the training scan (mamba_ssm) evaluate
  it in fp32 as log(exp(v) + 1), which has an absolute error of up to 2^-24 instead of a relative one
  (the decode kernel's O7-N1). #8960 (s4f-mamba-softplus) deliberately keeps this formula here so that
  batch-invariant decode matches the training scan bit for bit; the contract for this kernel is therefore
  that formula in fp32, and the trusted implementation evaluates the same formula.
* Tolerance class: floating point. Reference: the formulas above in fp64 (exact softplus, no chunking of
  the sum). Trusted: torch fp32 on the same device (the training formula for softplus; the prefix sum
  accumulated in fp32 in order, which is no more accurate than the kernel's scan).
  Exact: dt_out's zeros past each window, skipped entries (still NaN-poisoned through the
  launcher's own allocation), and every output for dyadic inputs without softplus (each product and
  partial sum is then exact in fp32).

Variants and dispatch (from the kernel, read after the contract): autotune BLOCK_SIZE_H in {2, 4, 8, 16, 32,
64} keyed on (chunk_size, nheads) — every one pinned in turn at nheads BLOCK_SIZE_H - 1, BLOCK_SIZE_H,
BLOCK_SIZE_H + 1 (the head mask); constexpr HAS_TARGET_ROWS, DT_SOFTPLUS, HAS_DT_BIAS, dt_min/dt_max
(production (0, inf) and a finite clamp); BLOCK_SIZE_CHUNK = chunk_size (64, 128 production, 256); the
`v <= 20` softplus threshold. Window lengths 1, c-1, c, c+1, 2c, 2c+3 (pseudo-chunks of length 1, c-1, c),
zero-length padded requests, NaN padded tail tokens. dt in bf16 and fp32. Offsets: the kernel declares
its token and chunk strides `tl.int64` and casts the chunk id to int64, so no int32 product exists to
cross 2^31 (no `large` case).

Main vs open PRs: no open PR changes this kernel (#8960 changes only the decode kernel's softplus).
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.ssm.ops.mamba2 import ssd_chunk_state
from megatron.core.ssm.ops.mamba2.ssd_chunk_state import _chunk_cumsum_fwd
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.ssm import ssd_utils as su

DEVICE = "cuda"
KERNEL = ssd_chunk_state._chunk_cumsum_fwd_kernel


# ---------------------------------------------------------------------------------------------
# Reference and trusted implementation
# ---------------------------------------------------------------------------------------------


def _sequential_cumsum(t):
    """Prefix sums along the last dim, accumulated in t's dtype in order (torch's CPU cumsum accumulates
    fp32 in fp64, so it would not be an fp32 implementation)."""
    out = torch.empty_like(t)
    acc = torch.zeros_like(t[..., 0])
    for k in range(t.shape[-1]):
        acc = acc + t[..., k]
        out[..., k] = acc
    return out


def cumsum_reference(dt, A, dt_bias, windows, chunk_size, *, softplus, dt_limit, softplus_fn=None):
    """The contract's dt_out and dA_cumsum, (nheads, nchunks, chunk_size), in the dtype of `dt`.

    `softplus_fn` defaults to the exact softplus (the fp64 reference); the trusted implementation passes
    the training formula. The prefix sum accumulates in order, in the inputs' dtype.
    """
    softplus_fn = softplus_fn or (lambda v: F.softplus(v))
    v = su.gather_windows(dt, windows, chunk_size)  # (nchunks, chunk, H)
    if dt_bias is not None:
        v = v + dt_bias
    if softplus:
        v = softplus_fn(v)
    v = v.clamp(*dt_limit)
    mask = su.window_mask(windows, chunk_size, dt.device)[..., None]
    dt_out = torch.where(mask, v, torch.zeros_like(v)).permute(2, 0, 1)
    dA = _sequential_cumsum(dt_out * A[:, None, None])
    return dA.contiguous(), dt_out.contiguous()


def trusted_cumsum(dt, A, dt_bias, windows, chunk_size, *, softplus, dt_limit):
    """The contract in torch fp32 on the device, softplus by the training formula."""
    return cumsum_reference(
        dt.float(),
        A.float(),
        None if dt_bias is None else dt_bias.float(),
        windows,
        chunk_size,
        softplus=softplus,
        dt_limit=dt_limit,
        softplus_fn=su.softplus_training_formula,
    )


def run_cumsum(
    dt,
    A,
    dt_bias,
    chunk_size,
    *,
    cu_chunk_seqlens=None,
    chunk_starts=None,
    target_rows=None,
    softplus=True,
    dt_limit=(0.0, float("inf")),
    exact=False,
):
    """Launch `_chunk_cumsum_fwd` with poisoned, guarded outputs and check them against the contract."""
    if target_rows is None:
        windows = su.varlen_windows(cu_chunk_seqlens)
    else:
        windows = su.fixed_windows(chunk_starts, chunk_size)
    live = [c for c in range(len(windows)) if target_rows is None or int(target_rows[c]) >= 0]
    before = dt.clone()
    with su.poisoned_allocations(ssd_chunk_state) as buffers:
        dA, dt_out = _chunk_cumsum_fwd(
            dt,
            A,
            chunk_size,
            cu_chunk_seqlens,
            dt_bias=dt_bias,
            dt_softplus=softplus,
            dt_limit=dt_limit,
            chunk_starts=chunk_starts,
            target_rows=target_rows,
        )
    su.check_all(buffers, "cumsum outputs")
    ou.assert_exact(dt, before, what="dt is read-only")
    assert dA.shape == dt_out.shape == (dt.shape[1], len(windows), chunk_size)
    assert dA.dtype == dt_out.dtype == torch.float32
    skipped = [c for c in range(len(windows)) if c not in live]
    if skipped:
        ou.assert_poisoned(dt_out[:, skipped], what="dt_out of skipped target-rows entries")
        ou.assert_poisoned(dA[:, skipped], what="dA_cumsum of skipped target-rows entries")
    live_windows = [windows[c] for c in live]
    dA, dt_out = dA[:, live], dt_out[:, live]

    kw = dict(softplus=softplus, dt_limit=dt_limit)
    ref_dA, ref_dt = ou.fp64_reference(
        cumsum_reference, dt, A, dt_bias, live_windows, chunk_size, **kw
    )
    # Past each window the reference holds dt 0 and the window's total, so every check below covers the
    # padding as well; dt_out's zeros are exact whatever the inputs.
    mask = su.window_mask(live_windows, chunk_size, dt.device)[None].expand_as(dt_out)
    su.assert_equal(dt_out[~mask], torch.zeros_like(dt_out[~mask]), what="dt_out past the window")
    if exact:
        su.assert_equal(dt_out, ref_dt, what="dt_out (exact inputs)")
        su.assert_equal(
            dA, ref_dA, what="dA_cumsum (exact inputs, incl. the totals past each window)"
        )
        return dA, dt_out
    t_dA, t_dt = trusted_cumsum(dt, A, dt_bias, live_windows, chunk_size, **kw)
    ou.assert_close_to_reference(dt_out, ref_dt, t_dt, what="dt_out")
    ou.assert_close_to_reference(
        dA, ref_dA, t_dA, what="dA_cumsum (incl. the totals past each window)"
    )
    return dA, dt_out


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


PACKS = {
    # name: (request lengths in terms of the chunk size c, padded requests, padded tail tokens)
    "boundaries": (lambda c: [1, c - 1, c, c + 1, 2 * c, 2 * c + 3], 2, 5),
    "short": (lambda c: [3, 1, c - 1], 1, 0),
}


def production_parameters(heads, g):
    """A and dt_bias as MambaMixer initializes them: A_log ~ log U(1, 16); softplus(dt_bias) ~ U(1e-3,
    1e-1) (log-uniform in the mixer; uniform here covers the same range)."""
    A = -torch.empty(heads, device=DEVICE).uniform_(1.0, 16.0, generator=g)
    dt0 = torch.empty(heads, device=DEVICE).uniform_(1e-3, 1e-1, generator=g)
    return A, dt0 + torch.log(-torch.expm1(-dt0))


def varlen_case(pack, chunk_size, heads, dtype, g, lengths=None):
    """dt for a pack with production metadata; padded tail tokens are NaN (they belong to no window)."""
    if lengths is None:
        make, padded_requests, tail = PACKS[pack]
        lengths = make(chunk_size)
    else:
        padded_requests, tail = 0, 0
    meta = su.production_metadata(lengths, padded_requests, tail, chunk_size, DEVICE)
    dt = torch.randn(meta["padded_tokens"], heads, device=DEVICE, generator=g).to(dtype)
    dt[meta["real_tokens"] :] = float("nan")
    return dt, meta


def _cases():
    factors = {
        "pack": ("boundaries", "short"),
        "chunk_size": (su.PRODUCTION_CHUNK, 64, 256),
        "heads": (8, 1, 9, 65, 128),
        "dtype": (torch.bfloat16, torch.float32),
        "bias": (True, False),
        "softplus": (True, False),
        "dt_limit": ((0.0, float("inf")), (1e-3, 0.1)),
        "config": tuple(range(len(su.kernel_configs(KERNEL)))),
    }
    params = []
    for case in ou.pairwise_cases(factors):
        config = su.kernel_configs(KERNEL)[case["config"]]
        case_id = "-".join(
            [
                case["pack"],
                f"c{case['chunk_size']}",
                f"h{case['heads']}",
                str(case["dtype"]).replace("torch.", ""),
                "bias" if case["bias"] else "nobias",
                "softplus" if case["softplus"] else "raw",
                "limit_none" if case["dt_limit"][1] == float("inf") else "limit",
                su.tile_id(config),
            ]
        )
        params.append(pytest.param(case, id=case_id))
    return params


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", _cases())
def test_varlen_cumsum_matches_fp64(case):
    """Production pseudo-chunks (padded requests, NaN tail tokens) against the fp64 contract, each
    BLOCK_SIZE_H pinned in turn (pairwise with the head count, dtype and modes)."""
    g = ou.make_generator(device=DEVICE)
    dt, meta = varlen_case(case["pack"], case["chunk_size"], case["heads"], case["dtype"], g)
    A, dt_bias = production_parameters(case["heads"], g)
    with su.pinned(KERNEL, su.kernel_configs(KERNEL)[case["config"]]):
        run_cumsum(
            dt,
            A,
            dt_bias if case["bias"] else None,
            case["chunk_size"],
            cu_chunk_seqlens=meta["cu_chunk_seqlens"],
            softplus=case["softplus"],
            dt_limit=case["dt_limit"],
        )


def _block_h_cases():
    params = []
    for index, config in enumerate(su.kernel_configs(KERNEL)):
        block = config.kwargs["BLOCK_SIZE_H"]
        for heads in sorted({max(block - 1, 1), block, block + 1}):
            params.append(pytest.param(index, heads, id=f"{su.tile_id(config)}-h{heads}"))
    return params


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("config_index,heads", _block_h_cases())
def test_every_block_size_h_at_its_head_edges(config_index, heads):
    """Each BLOCK_SIZE_H candidate pinned, at nheads BLOCK_SIZE_H - 1, BLOCK_SIZE_H and + 1 (the head
    mask), on dyadic inputs without softplus: every output exact."""
    g = ou.make_generator(device=DEVICE)
    c = 64
    meta = su.production_metadata([1, c - 1, c + 1, 2 * c + 3], 1, 3, c, DEVICE)
    dt = ou.dyadic_tensor(
        (meta["padded_tokens"], heads),
        exponents=(-2, 1),
        dtype=torch.float32,
        device=DEVICE,
        generator=g,
    )
    dt[meta["real_tokens"] :] = float("nan")
    A = -ou.dyadic_tensor(
        (heads,), exponents=(-2, 1), dtype=torch.float32, device=DEVICE, generator=g
    )
    bias = ou.dyadic_tensor(
        (heads,), exponents=(-2, 1), dtype=torch.float32, device=DEVICE, generator=g
    )
    with su.pinned(KERNEL, su.kernel_configs(KERNEL)[config_index]):
        run_cumsum(
            dt, A, bias, c, cu_chunk_seqlens=meta["cu_chunk_seqlens"], softplus=False, exact=True
        )


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("chunk_size", [64, su.PRODUCTION_CHUNK, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cumsum_exact_on_dyadic_inputs(chunk_size, dtype):
    """Without softplus and with dyadic dt, bias and A, every product and partial sum is exact in fp32
    (at most 8 significant bits per product, 2^8 of them): the windows, the masks and the padding are
    checked bit for bit, including negative v clamped to 0."""
    g = ou.make_generator(device=DEVICE)
    heads = 9
    dt, meta = varlen_case("boundaries", chunk_size, heads, torch.float32, g)
    real = meta["real_tokens"]
    dt[:real] = ou.dyadic_tensor(
        (real, heads), exponents=(-2, 1), dtype=dtype, device=DEVICE, generator=g
    ).float()
    dt = dt.to(dtype)
    A = -ou.dyadic_tensor(
        (heads,), exponents=(-2, 1), dtype=torch.float32, device=DEVICE, generator=g
    )
    bias = ou.dyadic_tensor(
        (heads,), exponents=(-2, 1), dtype=torch.float32, device=DEVICE, generator=g
    )
    run_cumsum(
        dt,
        A,
        bias,
        chunk_size,
        cu_chunk_seqlens=meta["cu_chunk_seqlens"],
        softplus=False,
        exact=True,
    )


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("exact", [False, True], ids=["softplus", "dyadic"])
@pytest.mark.parametrize("chunk_size", [64, su.PRODUCTION_CHUNK])
def test_target_rows_windows(chunk_size, exact):
    """Fixed windows (the batch-invariant decode tile path): each entry reads [slot * c, slot * c + c)
    of the replay buffer whatever its target row; padding entries (target row -1, chunk start -c) write
    nothing. Random production-range inputs with softplus, and dyadic inputs without (exact)."""
    g = ou.make_generator(device=DEVICE)
    entries, heads = 37, 8
    win = su.decode_windows(entries, chunk_size, padding=(3, 20), device=DEVICE)
    if exact:
        dt = ou.dyadic_tensor(
            (entries * chunk_size, heads),
            exponents=(-2, 1),
            dtype=torch.float32,
            device=DEVICE,
            generator=g,
        ).bfloat16()
        A = -ou.dyadic_tensor(
            (heads,), exponents=(-2, 1), dtype=torch.float32, device=DEVICE, generator=g
        )
        bias = None
    else:
        dt = torch.randn(entries * chunk_size, heads, device=DEVICE, generator=g).bfloat16()
        A, bias = production_parameters(heads, g)
    run_cumsum(
        dt,
        A,
        bias,
        chunk_size,
        chunk_starts=win["chunk_starts"],
        target_rows=win["target_rows"],
        softplus=not exact,
        exact=exact,
    )


@ou.requires_sm(ou.HOPPER)
def test_softplus_threshold_and_extremes():
    """dt + dt_bias at -100, -40, -20, -7, 0, 19.5, 20, 20.5, 40, 100: below the threshold the training
    formula (bounded), above it the identity (v itself, exact in fp32 like the trusted sum)."""
    values = torch.tensor([-100.0, -40.0, -20.0, -7.0, 0.0, 19.5, 20.0, 20.5, 40.0, 100.0])
    heads, c = 4, 64
    tokens = values.numel() * 3
    dt = values.repeat(3)[:, None].repeat(1, heads).to(DEVICE)
    dt = dt + torch.linspace(-0.25, 0.25, tokens, device=DEVICE)[:, None]
    A = -torch.tensor([1.0, 2.0, 8.0, 16.0], device=DEVICE)
    bias = torch.zeros(heads, device=DEVICE)
    meta = su.production_metadata([tokens], 0, 0, c, DEVICE)
    run_cumsum(dt, A, bias, c, cu_chunk_seqlens=meta["cu_chunk_seqlens"])


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("heads", [128, 256], ids=["nemotron_h_8b", "nemotron_h_56b"])
def test_production_prefill_pack(heads):
    """Production: chunk 128, the 8B (128 heads) and 56B (256 heads) layers at TP 1, an 8193-token pack
    of mixed lengths (pseudo-chunks of every length class), bf16 dt, mixer-initialized A and dt_bias.
    """
    g = ou.make_generator(device=DEVICE)
    c = su.PRODUCTION_CHUNK
    lengths = [4096, 2049, 1000, 128, 127, 1, 792]
    dt, meta = varlen_case(None, c, heads, torch.bfloat16, g, lengths=lengths)
    A, bias = production_parameters(heads, g)
    run_cumsum(dt, A, bias, c, cu_chunk_seqlens=meta["cu_chunk_seqlens"])


@pytest.mark.sanitizer
@ou.requires_sm(ou.HOPPER)
def test_sanitizer_small_case():
    """Small case for compute-sanitizer: a padded pack at chunk 64 with 9 heads (a partial head block),
    then the target-rows mode with a padding entry."""
    g = ou.make_generator(device=DEVICE)
    dt, meta = varlen_case("short", 64, 9, torch.bfloat16, g)
    A, bias = production_parameters(9, g)
    run_cumsum(dt, A, bias, 64, cu_chunk_seqlens=meta["cu_chunk_seqlens"])
    win = su.decode_windows(6, 64, padding=(2,), device=DEVICE)
    dt = torch.randn(6 * 64, 9, device=DEVICE, generator=g).bfloat16()
    run_cumsum(dt, A, bias, 64, chunk_starts=win["chunk_starts"], target_rows=win["target_rows"])


def test_case_list_covers_every_block_size_and_mode():
    """The pairwise sweep pins every BLOCK_SIZE_H and covers each mode flag both ways (CPU check)."""
    cases = [p.values[0] for p in _cases()]
    assert {c["config"] for c in cases} == set(range(len(su.kernel_configs(KERNEL))))
    for factor in ("softplus", "bias", "dt_limit", "chunk_size", "heads", "dtype"):
        assert len({str(c[factor]) for c in cases}) > 1
