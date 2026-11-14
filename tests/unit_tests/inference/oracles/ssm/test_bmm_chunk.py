# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `_bmm_chunk_fwd_kernel` (megatron/core/ssm/ops/mamba2/ssd_bmm.py), the intra-chunk C B^T
products of the Mamba-2 chunked scan, through its launcher `_bmm_chunk_fwd`, in varlen mode (prefill) and
target-rows mode (the batch-invariant decode tile path).

Before this module the kernel was reached only through the varlen scan oracle (test_chunk_scan_varlen.py),
which sees CB only through `y`, and never in target-rows mode.

Contract (written from the Mamba-2 attention form and the launcher's docstring and callers before reading the
kernel body):

* Inputs: a = C and b = B, (T, ngroups, K) (K = dstate), in the activation dtype; the chunk windows:
  varlen, [cu[c], cu[c+1]); target-rows mode, [chunk_starts[c], chunk_starts[c] + chunk_size) with one
  target row per window (target_rows[c] < 0: a padding entry, nothing needed).
* Output (nchunks, ngroups, chunk_size, chunk_size) in `output_dtype` (fp32 for every caller) or a's dtype:

      out[c, g, i, j] = sum_k a[s_c + i, g, k] * b[s_c + j, g, k]     for i, j < len(c)
      out[c, g, i, j] = 0                                             if i or j >= len(c) (varlen)

  With `causal` only the entries j <= i are guaranteed (the launcher's docstring says i <= j; the scan
  consumes CB[i, j <= i], and the kernel skips tiles above the diagonal: the docstring has the indices
  swapped). In target-rows mode only row target_rows[c], columns j <= target_rows[c], is guaranteed; the
  rest is unspecified. Inputs are read-only; nothing is written outside the output.
* Numerics: fp32 accumulation on tensor cores; bf16/fp16 operands as stored, fp32 as TF32.
* Tolerance class: floating point. Reference: the sum in fp64. Trusted: the training scan's CB kernel (pip mamba_ssm `_bmm_chunk_fwd`) on each window, on the same
  device: with exact 16-bit operands the error is the tensor cores' fp32 accumulation alone, so the
  trusted implementation must accumulate the same way. Exact: every guaranteed entry for dyadic inputs
  (each product and sum exact in fp32), the zeros past a window, guard bands.

Variants and dispatch (from the kernel, read after the contract): autotune configs (BLOCK_SIZE_M/N over
the chunk 16..128, BLOCK_SIZE_K over dstate 32/64), each pinned in turn on bf16 and fp32 inputs
(`test_every_bmm_tile`, H100 and GB200); constexpr IS_CAUSAL (no in-tree caller passes causal=True; one
case), HAS_TARGET_ROWS, and dot_dtype bf16 / fp16 / fp32; the launcher's layout branch (operands passed by
stride when the last or first dim is unit-stride, copied otherwise). K (dstate) 16, 64, 128 (production),
129, 256; ngroups 1, 2, 8; window lengths 1, c-1, c and padded requests. Offsets are int64: no `large`
case.

Main vs open PRs: no open PR changes this kernel.
"""

import pytest
import torch
from triton.runtime.errors import OutOfResources

from megatron.core.ssm.ops.mamba2 import ssd_bmm
from megatron.core.ssm.ops.mamba2.ssd_bmm import _bmm_chunk_fwd
from megatron.core.tuning.selection import cheapest
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.ssm import ssd_utils as su

DEVICE = "cuda"
KERNEL = ssd_bmm._bmm_chunk_fwd_kernel


# ---------------------------------------------------------------------------------------------
# Reference and trusted implementation
# ---------------------------------------------------------------------------------------------


def bmm_reference(a, b, windows, chunk_size):
    """The contract in the inputs' dtype (fp64 for the reference): (nchunks, G, chunk, chunk)."""
    aw = su.gather_windows(a, windows, chunk_size)  # (nc, chunk, G, K), zero past the window
    bw = su.gather_windows(b, windows, chunk_size)
    return torch.einsum("cigk,cjgk->cgij", aw, bw)


def _pip_bmm():
    """pip mamba_ssm's `_bmm_chunk_fwd` (the training scan's CB kernel), or a failure without it."""
    try:
        from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_fwd as pip_bmm_chunk_fwd
    except ImportError:
        pytest.fail("mamba_ssm (the trusted implementation) is not installed", pytrace=False)
    return pip_bmm_chunk_fwd


def trusted_bmm(a, b, windows, chunk_size, out_dtype):
    """The training scan's CB kernel (pip mamba_ssm) on each window as a one-chunk sequence (zero rows
    past a window's length, as the kernel masks them), on the same device.

    With exact 16-bit operands CB's error is the tensor cores' fp32 accumulation alone, which is not a
    round-to-nearest sum; a CUDA-core fp32 sum would be the more accurate one, so the trusted
    implementation is the training kernel, with the same tensor-core arithmetic (as the scan oracle).
    """
    aw = su.gather_windows(a, windows, chunk_size)  # (nc, chunk, G, K)
    bw = su.gather_windows(b, windows, chunk_size)
    out = _pip_bmm()(aw, bw, chunk_size, output_dtype=out_dtype)  # (nc, 1, G, chunk, chunk)
    return out[:, 0]


def required_mask(windows, chunk_size, *, causal=False, target_rows=None, device="cuda"):
    """The guaranteed entries (nchunks, 1, chunk, chunk)."""
    i = torch.arange(chunk_size, device=device)
    full = torch.ones(len(windows), 1, chunk_size, chunk_size, dtype=torch.bool, device=device)
    if causal:
        full &= (i[None, :] <= i[:, None])[None, None]
    if target_rows is not None:
        tr = target_rows.to(device=device, dtype=torch.long)
        row = i[None, :] == tr[:, None]  # (nc, chunk)
        col = i[None, :] <= tr[:, None]
        full &= (row[:, :, None] & col[:, None, :])[:, None]
    return full


def run_bmm(
    a,
    b,
    chunk_size,
    *,
    cu_chunk_seqlens=None,
    chunk_starts=None,
    target_rows=None,
    causal=False,
    output_dtype=torch.float32,
    exact=False,
):
    """Launch `_bmm_chunk_fwd` with a poisoned, guarded output and check every guaranteed entry."""
    if target_rows is None:
        windows = su.varlen_windows(cu_chunk_seqlens)
    else:
        windows = su.fixed_windows(chunk_starts.clamp(min=0), chunk_size)
    before = a.clone(), b.clone()
    with su.poisoned_allocations(ssd_bmm) as buffers:
        out = _bmm_chunk_fwd(
            a,
            b,
            chunk_size,
            cu_chunk_seqlens,
            causal=causal,
            output_dtype=output_dtype,
            target_rows=target_rows,
            chunk_starts=chunk_starts,
        )
    su.check_all(buffers, "CB")
    ou.assert_exact(a, before[0], what="C is read-only")
    ou.assert_exact(b, before[1], what="B is read-only")
    out_dtype = output_dtype or a.dtype
    assert out.dtype == out_dtype
    assert out.shape == (len(windows), a.shape[1], chunk_size, chunk_size)
    mask = required_mask(
        windows, chunk_size, causal=causal, target_rows=target_rows, device=a.device
    )
    mask = mask.expand_as(out)
    if target_rows is not None:
        mask = mask & (target_rows >= 0).to(a.device)[:, None, None, None]
    ref = ou.fp64_reference(bmm_reference, a, b, windows, chunk_size)
    if exact:
        su.assert_equal(out[mask], ref[mask], what="CB (exact inputs)")
        return out
    trusted = trusted_bmm(a, b, windows, chunk_size, out_dtype)
    ou.assert_close_to_reference(out[mask], ref[mask], trusted[mask], what="CB")
    if target_rows is None and not causal:
        # Past a window every entry is an exact zero (masked loads), not just "unspecified".
        outside = ~su.window_mask(windows, chunk_size, a.device)
        rows_out = outside[:, None, :, None].expand_as(out)
        cols_out = outside[:, None, None, :].expand_as(out)
        zeros = out[rows_out | cols_out]
        su.assert_equal(zeros, torch.zeros_like(zeros), what="CB past the window")
    return out


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


PACKS = {
    "boundaries": (lambda c: [1, c - 1, c, c + 1, 2 * c, 2 * c + 3], 2, 5),
    "short": (lambda c: [3, 1, c - 1], 1, 0),
    "single": (lambda c: [3 * c + 5], 0, 0),
}


def make_pack(pack, chunk_size, groups, k, dtype, g, *, layout="contiguous", dyadic=False):
    """C and B for a pack with production metadata; padded tail tokens NaN (in no window)."""
    if isinstance(pack, str):
        make, padded, tail = PACKS[pack]
        lengths = make(chunk_size)
    else:
        lengths, padded, tail = pack, 0, 0
    meta = su.production_metadata(lengths, padded, tail, chunk_size, DEVICE)
    total = meta["padded_tokens"]

    def one():
        if dyadic:
            t = ou.dyadic_tensor(
                (total, groups, k),
                exponents=(-1, 1),
                dtype=torch.float32,
                device=DEVICE,
                generator=g,
            )
        else:
            t = torch.randn(total, groups, k, device=DEVICE, generator=g)
        t = t.to(dtype)
        if layout == "strided":  # row stride > groups * k, unit last stride: passed by stride
            wide = torch.full((total, groups * k + 5), 7.0, dtype=dtype, device=DEVICE)
            wide[:, : groups * k] = t.view(total, -1)
            t = wide[:, : groups * k].view(total, groups, k)
        elif layout == "token_major":  # stride(0) == 1: passed by stride
            t = t.permute(2, 1, 0).contiguous().permute(2, 1, 0)
        elif layout == "copied":  # neither the last nor the first stride is 1: the launcher copies
            wide = torch.zeros(total, groups, 2 * k, dtype=dtype, device=DEVICE)
            wide[..., ::2] = t
            t = wide[..., ::2]
        t[meta["real_tokens"] :] = float("nan")
        return t

    return one(), one(), meta


def _cases():
    factors = {
        "pack": ("boundaries", "short", "single"),
        "chunk_size": (su.PRODUCTION_CHUNK, 64, 256),
        "groups": (1, 2, 8),
        "k": (128, 16, 64, 129, 256),
        "dtype": (torch.bfloat16, torch.float32, torch.float16),
        "output_dtype": (torch.float32, None),
        "layout": ("contiguous", "strided", "token_major", "copied"),
    }
    params = []
    for case in ou.pairwise_cases(factors):
        case_id = "-".join(
            [
                case["pack"],
                f"c{case['chunk_size']}",
                f"g{case['groups']}",
                f"k{case['k']}",
                str(case["dtype"]).replace("torch.", ""),
                "out_fp32" if case["output_dtype"] else "out_input_dtype",
                case["layout"],
            ]
        )
        params.append(pytest.param(case, id=case_id))
    return params


def _run_case(case, **kw):
    g = ou.make_generator(device=DEVICE)
    C, B, meta = make_pack(
        case["pack"],
        case["chunk_size"],
        case["groups"],
        case["k"],
        case["dtype"],
        g,
        layout=case["layout"],
        dyadic=kw.get("exact", False),
    )
    return run_bmm(
        C,
        B,
        case["chunk_size"],
        cu_chunk_seqlens=meta["cu_chunk_seqlens"],
        output_dtype=case["output_dtype"],
        **kw,
    )


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", _cases())
def test_varlen_bmm_matches_fp64(case):
    """Production pseudo-chunks (padded requests, NaN tail tokens) against the fp64 contract, pairwise
    over shapes, the three dot dtypes, the output dtype and the launcher's layout branches."""
    with su.pinned(KERNEL, su.default_config(KERNEL)):
        _run_case(case)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_bmm_exact_on_dyadic_inputs(dtype, causal):
    """Dyadic C and B (4 significant bits, exponents -1..1): every product and sum is exact in fp32 and
    in TF32 operands, so each guaranteed entry equals the fp64 product, causal or not."""
    case = dict(
        pack="boundaries",
        chunk_size=64,
        groups=2,
        k=129,
        dtype=dtype,
        output_dtype=torch.float32,
        layout="contiguous",
    )
    with su.pinned(KERNEL, su.default_config(KERNEL)):
        _run_case(case, exact=True, causal=causal)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("exact", [False, True], ids=["random", "dyadic"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_target_rows_windows(dtype, exact):
    """Target-rows mode (batch-invariant decode tile path): fixed windows of the replay buffer, row
    target_rows[c] and its columns j <= row are exact (dyadic) / bounded (random), for every row class
    (0, 15, 16, 17, 63, 64, 65, chunk - 1) and padding entries (row -1, chunk start -c)."""
    g = ou.make_generator(device=DEVICE)
    c, groups, k, entries = su.PRODUCTION_CHUNK, 2, 128, 24
    rows = [0, 15, 16, 17, 63, 64, 65, c - 1]
    win = su.decode_windows(entries, c, padding=(5, 18), rows=rows, device=DEVICE)
    shape = (entries * c, groups, k)
    if exact:
        C = ou.dyadic_tensor(
            shape, exponents=(-1, 1), dtype=torch.float32, device=DEVICE, generator=g
        )
        B = ou.dyadic_tensor(
            shape, exponents=(-1, 1), dtype=torch.float32, device=DEVICE, generator=g
        )
    else:
        C = torch.randn(shape, device=DEVICE, generator=g)
        B = torch.randn(shape, device=DEVICE, generator=g)
    with su.pinned(KERNEL, su.default_config(KERNEL)):
        run_bmm(
            C.to(dtype),
            B.to(dtype),
            c,
            chunk_starts=win["chunk_starts"],
            target_rows=win["target_rows"],
            exact=exact,
        )


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.HOPPER, ou.BLACKWELL)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_every_bmm_tile(dtype):
    """Every autotune candidate pinned in turn: varlen on the boundary pack at production dstate (K 128,
    8 groups) and K 129, and target-rows mode. A candidate the device cannot launch (shared memory) is
    dropped only after that error; then neither the cheapest fallback nor timed autotuning may pick it.
    """
    configs = su.kernel_configs(KERNEL)
    too_big, failures = [], []
    base = dict(
        pack="boundaries",
        chunk_size=su.PRODUCTION_CHUNK,
        groups=8,
        k=128,
        dtype=dtype,
        output_dtype=torch.float32,
        layout="contiguous",
    )
    g = ou.make_generator(device=DEVICE)
    win = su.decode_windows(
        9, su.PRODUCTION_CHUNK, padding=(2,), rows=[0, 17, 64, 127], device=DEVICE
    )
    C = torch.randn(9 * su.PRODUCTION_CHUNK, 8, 128, device=DEVICE, generator=g).to(dtype)
    B = torch.randn(9 * su.PRODUCTION_CHUNK, 8, 128, device=DEVICE, generator=g).to(dtype)
    for config in configs:
        with su.pinned(KERNEL, config):
            try:
                _run_case(base)
                _run_case(dict(base, k=129, groups=1, pack="short"))
                run_bmm(
                    C,
                    B,
                    su.PRODUCTION_CHUNK,
                    chunk_starts=win["chunk_starts"],
                    target_rows=win["target_rows"],
                )
            except OutOfResources as err:
                assert err.name == "shared memory", f"{su.tile_id(config)}: {err}"
                too_big.append(config)
            except AssertionError as err:
                failures.append(f"{su.tile_id(config)}: {err}")
    assert not failures, "pinned bmm configs disagree with the contract:\n" + "\n".join(failures)
    assert len(too_big) < len(configs), "no bmm config fits this GPU"
    assert cheapest(configs) not in too_big, "the pinned policy's fallback cannot launch here"
    tuner = su.autotuner(KERNEL)
    saved = tuner.cache
    tuner.cache = {}
    try:
        _run_case(base)
        chosen = tuner.best_config
    finally:
        tuner.cache = saved
    assert (
        chosen not in too_big
    ), f"timed autotuning chose {su.tile_id(chosen)}, which cannot launch"


@ou.requires_sm(ou.HOPPER)
def test_production_prefill_pack():
    """Production: the Nemotron-H 8B layer at TP 1 (8 groups, dstate 128), chunk 128, an 8193-token bf16
    pack of mixed lengths, fp32 CB."""
    g = ou.make_generator(device=DEVICE)
    lengths = [4096, 2049, 1000, 128, 127, 1, 792]
    C, B, meta = make_pack(lengths, 128, 8, 128, torch.bfloat16, g)
    with su.pinned(KERNEL, su.default_config(KERNEL)):
        run_bmm(C, B, 128, cu_chunk_seqlens=meta["cu_chunk_seqlens"])


@pytest.mark.sanitizer
@ou.requires_sm(ou.HOPPER)
def test_sanitizer_small_case():
    """Small case for compute-sanitizer: a padded pack at chunk 64 with K 129 (a partial K tile) and
    target-rows mode with a padding entry."""
    g = ou.make_generator(device=DEVICE)
    C, B, meta = make_pack("boundaries", 64, 2, 129, torch.bfloat16, g)
    with su.pinned(KERNEL, su.default_config(KERNEL)):
        run_bmm(C, B, 64, cu_chunk_seqlens=meta["cu_chunk_seqlens"])
        win = su.decode_windows(4, 64, padding=(1,), rows=[63, 0, 31], device=DEVICE)
        C = torch.randn(4 * 64, 2, 129, device=DEVICE, generator=g).bfloat16()
        B = torch.randn(4 * 64, 2, 129, device=DEVICE, generator=g).bfloat16()
        run_bmm(C, B, 64, chunk_starts=win["chunk_starts"], target_rows=win["target_rows"])


def test_case_list_covers_every_factor():
    """Every pack, size, dtype and layout branch occurs in the pairwise sweep (CPU check)."""
    cases = [p.values[0] for p in _cases()]
    for factor in ("pack", "chunk_size", "groups", "k", "dtype", "output_dtype", "layout"):
        assert len({str(c[factor]) for c in cases}) > 1
