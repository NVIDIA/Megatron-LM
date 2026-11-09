# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `mean_kernel` / `mean_batch_invariant`, the batch-invariant `aten::mean.dim`.

Kernel: `megatron/core/transformer/custom_layers/batch_invariant_kernels.py` (`mean_kernel`,
launched by `mean_dim`; `mean_batch_invariant` is installed on `aten::mean.dim` by
`enable_batch_invariant_mode` for every backend). Inference reaches it through every eager
`x.mean(dim)` of a batch-invariant model: the batch-invariant RMSNorm's statistic
(`BatchInvariantRMSNormFn`, `_te_apply_normalization_patched`, the inference-optimized
linears' norm on the triton/deepgemm backends), the mHC stream average (`.float().mean(dim=-2)`
in `hybrid_block.py` and `multi_token_prediction.py`), and other eager means.

Contract (written from `aten::mean.dim`, which the override replaces, and the backend's promise):
- Under batch-invariant mode, `torch.mean(x, dim, keepdim=False, dtype=None)` for a floating x
  returns the arithmetic mean over `dim` (one dim, several, or all for an empty list), with
  aten's output dtype (`dtype` if given, else x's dtype) and shape (`keepdim` keeps size-1
  dims). The sum is accumulated in fp32 (at least: aten accumulates bf16/fp16 in fp32) and the
  result rounded once, so on dyadic inputs, whose sums are exact in fp32, the result is the fp64
  mean rounded once (N a power of two). An empty reduction gives NaN; an empty output is empty.
  The input is read only.
- Batch invariance: each output element's bits depend only on the elements it averages: not on
  the number of other rows or on their values.

Reference: fp64 `x.mean(dim)`. Trusted implementation: `torch.mean` on the same device outside
batch-invariant mode (shape, dtype, the production bound). Tolerance class: on dyadic inputs,
whose sums are exact in fp32, each fp32 mean within 2 ulp of the fp64 mean (the kernel's final
`div.full.f32` division, PTX ISA) and exact for N a power of two; floating point bounded over the
output (default factor 2) for the random production shapes; exact for dtypes, shapes, dispatch
and batch invariance.

Structure (read after the contract): `mean_dim` views x as (M, N, K) around the reduced dim
(copying when the view needs it), launches one program per output element, sums N in blocks of
1024 (`tl.sum` of the loaded block, in the input dtype, then into an fp32 scalar), divides by N
and stores in the output dtype; element offsets are int32. Integer inputs are cast to fp32.
`mean_batch_invariant` sends one-dim reductions to `mean_dim` and several dims to
`torch.sum(dtype=float32) / n`. Covered: N at 1, 2 and B-1/B/B+1 of 1024 and 2048 plus 7168 and
8192; the reduced dim last (K = 1), in the middle (M, K > 1) and first (M = 1); strided inputs;
keepdim; 1 to 513 rows; fp32, bf16, fp16 and fp64; one and several dims; empty reductions and
outputs; production shapes; offsets past 2**31 elements (`large`).

Main vs open PRs (each failing case is deselected and run as an expect-fail task):
- `test_dyadic_half_means_accumulate_in_fp32` fails on main (finding O8a-M4): `tl.sum` keeps a
  bf16/fp16 block in its own dtype (Triton's `_pick_sum_dtype` widens only integers), so each
  1024-element partial sum is rounded to 8/11 bits. No fix PR yet.
- `test_output_dtype_follows_aten_for_dtype_and_half_inputs` fails on main (finding O8a-M5): the
  one-dim path drops `dtype=` (bf16 in, `dtype=torch.float32`, bf16 out), the several-dim path
  returns fp32 for bf16/fp16 inputs without `dtype`, and fp64 inputs raise (an assertion on
  several dims; on one dim, predicted, an fp32 loop-carried accumulator). No fix PR yet.
- `test_several_dims_do_not_depend_on_the_batch` fails on main (finding O8a-M7, pdx 7986825): the
  several-dim path's `torch.sum` is not batch invariant. No fix PR yet.
- `test_rows_past_int32_offsets` (`large`) is predicted to fail on main (finding O8a-M6, by
  reading): `mean_kernel` computes element offsets in int32. No fix PR yet.
All other cases are expected to pass on main.

Hardware: H100 (1 GPU).
"""

import pytest
import torch

from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from tests.unit_tests.inference.oracles import oracle_utils as ou

pytestmark = [ou.requires_sm(ou.HOPPER)]

_DEVICE = "cuda"

BLOCK = 1024  # mean_dim: BLOCK_SIZE
# Triton's fp32 `/` is `div.full.f32`, at most 2 ulp from the exact quotient (PTX ISA).
DIVISION_ULPS = 2.0
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
BACKENDS = ("te_native", "triton", "deepgemm")


@pytest.fixture(autouse=True)
def _batch_invariant_mode_off():
    """References and trusted outputs use torch's own ops: the mode must be off around them."""
    assert not bik.is_batch_invariant_mode_enabled(), "batch-invariant mode leaked into the test"
    yield
    assert not bik.is_batch_invariant_mode_enabled(), "the test left batch-invariant mode on"


def bi_mode(backend: str = "te_native"):
    if backend == "deepgemm":
        assert bik.HAVE_DEEPGEMM_BF16, "the deepgemm backend needs DeepGEMM's bf16 bindings"
    return bik.set_batch_invariant_mode(True, backend=backend)


def bi_mean(x, dim, keepdim=False, **kwargs):
    """`torch.mean` under batch-invariant mode; the input must come back unchanged."""
    before = x.clone()
    with bi_mode():
        out = torch.mean(x, dim=dim, keepdim=keepdim, **kwargs)
    ou.assert_exact(x, before, what="input is read-only")
    return out


def mean_reference(x, dim, keepdim):
    return x.mean(dim=dim, keepdim=keepdim)


def shaped(n_reduce: int, rows: int, where: str, dtype, seed: int, layout: str = "contiguous"):
    """(x, dim): ``rows`` independent slices of ``n_reduce`` dyadic elements (`dyadic`: every sum
    is exact in fp32), the reduced dim at ``where`` (last: [rows, N]; middle: [rows // 3 or 1, N,
    3]; first: [N, rows]); ``layout`` ``strided`` makes x a transposed view of the same values.

    With exact sums the only error left is the final division by N, so the check sees a wrong
    mask, offset or block count (an error of whole elements), not the summation order: with
    random fp32 data a mean of 7168 terms is dominated by cancellation, and a single output's
    error against fp64 measured 2.3x torch's on a correct kernel (pdx 7986825)."""
    if where == "last":
        shape, dim = (rows, n_reduce), -1
    elif where == "middle":
        shape, dim = (max(rows // 3, 1), n_reduce, 3), 1
    else:
        shape, dim = (n_reduce, rows), 0
    x = dyadic(shape, dtype, seed)
    if layout == "strided":
        x = x.transpose(0, -1).contiguous().transpose(0, -1)
    return x, dim


def _id(case):
    return "-".join(f"{k}{v}" for k, v in case.items())


_CASES = ou.pairwise_cases(
    {
        "n": (1, 2, BLOCK - 1, BLOCK, BLOCK + 1, 2 * BLOCK + 1, 7168),
        "rows": (1, 3, 257),
        "where": ("last", "middle", "first"),
        "keepdim": (False, True),
        "layout": ("contiguous", "strided"),
    }
)


@pytest.mark.parametrize("case", _CASES, ids=[_id(c) for c in _CASES])
def test_mean_matches_fp64(case):
    """fp32 means over the last, a middle and the first dim, N around the 1024-element block,
    contiguous and strided inputs, keepdim: shape and dtype as aten, values against fp64.

    The sums are exact (`shaped`), so the kernel's only error is its final fp32 division by N,
    which Triton lowers to `div.full.f32`: each mean must be within DIVISION_ULPS of the fp64
    mean (exact for N a power of two). A wrong mask, offset or block count is off by whole
    elements, far beyond that."""
    x, dim = shaped(case["n"], case["rows"], case["where"], torch.float32, seed=case["n"])
    out = bi_mean(x, dim, case["keepdim"])
    trusted = torch.mean(x, dim=dim, keepdim=case["keepdim"])
    assert out.shape == trusted.shape and out.dtype == trusted.dtype, _id(case)
    assert_within_division_ulps(out, ou.fp64_reference(mean_reference, x, dim, case["keepdim"]))


def assert_within_division_ulps(out, ref, what="mean"):
    """Each fp32 mean within DIVISION_ULPS ulp of the fp64 mean of an exact sum."""
    err = (out.double() - ref).abs()
    allowed = DIVISION_ULPS * ou.ulp(ref, torch.float32)
    bad = ~(err <= allowed)
    if bool(bad.any()):
        i = int(bad.reshape(-1).nonzero()[0])
        raise AssertionError(
            f"{what}: {int(bad.sum())} of {out.numel()} means are more than {DIVISION_ULPS} ulp"
            f" from the fp64 mean; first at flat index {i}: got {out.reshape(-1)[i].item()!r},"
            f" fp64 {ref.reshape(-1)[i].item()!r}"
        )


def dyadic(shape, dtype, seed):
    """4-bit mantissas, exponents -3..3: exact in bf16/fp16, and any sum of up to 8192 of them
    is exact in fp32 (ou.dyadic_tensor), but not in bf16/fp16 once it grows past their 8/11
    significant bits."""
    g = ou.make_generator(seed, device=_DEVICE)
    return ou.dyadic_tensor(
        shape, mantissa_bits=4, exponents=(-3, 3), dtype=dtype, device=_DEVICE, generator=g
    )


def randn(shape, dtype, seed):
    """Random values whose sums depend on the summation order (for the invariance checks)."""
    g = ou.make_generator(seed, device=_DEVICE)
    return torch.randn(*shape, device=_DEVICE, generator=g).to(dtype)


def check_dyadic_mean(dtype, n, where):
    """A mean with an fp32-exact sum and N a power of two is the fp64 mean rounded once."""
    shape = (5, n) if where == "last" else (2, n, 3)
    dim = -1 if where == "last" else 1
    x = dyadic(shape, dtype, seed=n)
    out = bi_mean(x, dim)
    ref = ou.fp64_reference(mean_reference, x, dim, False)
    ou.assert_exact(torch.mean(x, dim=dim), ref.to(dtype), what="trusted torch.mean is exact")
    ou.assert_exact(out, ref.to(dtype), what=f"{dtype} mean of {n} ({where} dim)")


_DYADIC_SIZES = [(BLOCK, "last"), (2 * BLOCK, "middle"), (8192, "last")]


@pytest.mark.parametrize("n,where", _DYADIC_SIZES, ids=[f"n{n}-{w}" for n, w in _DYADIC_SIZES])
def test_dyadic_fp32_means_are_exact(n, where):
    """fp32: one, two and eight 1024-element blocks, last and middle dims."""
    check_dyadic_mean(torch.float32, n, where)


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("n,where", _DYADIC_SIZES, ids=[f"n{n}-{w}" for n, w in _DYADIC_SIZES])
def test_dyadic_half_means_accumulate_in_fp32(n, where, dtype):
    """bf16 / fp16 inputs must be accumulated in fp32 (as aten does). Fails on main (O8a-M4):
    each 1024-element block is summed in the input dtype (`tl.sum` keeps bf16/fp16)."""
    check_dyadic_mean(DTYPES[dtype], n, where)


_DTYPE_CASES = {
    # name: (input dtype, dims, dtype kwarg)
    "bf16-last": (torch.bfloat16, [-1], None),
    "fp32-dtype-fp32": (torch.float32, [-1], torch.float32),
    "fp32-two-dims": (torch.float32, [1, 2], None),
    "bf16-last-dtype-fp32": (torch.bfloat16, [-1], torch.float32),
    "bf16-two-dims": (torch.bfloat16, [1, 2], None),
    "fp16-all-dims": (torch.float16, [], None),
    "fp64-last": (torch.float64, [-1], None),
    "fp64-two-dims": (torch.float64, [1, 2], None),
}
_DTYPE_CONTROLS = ["bf16-last", "fp32-dtype-fp32", "fp32-two-dims"]


def check_output_dtype(name):
    """The output dtype is aten's (``dtype`` if given, else the input's) and the value is the
    mean. The values are +-1, so every partial sum is exact in every dtype and only the dtype
    handling is checked."""
    in_dtype, dims, out_dtype = _DTYPE_CASES[name]
    g = ou.make_generator(3, device=_DEVICE)
    signs = torch.randint(0, 2, (4, 64, 32), device=_DEVICE, generator=g) * 2 - 1
    x = signs.to(in_dtype)
    kwargs = {} if out_dtype is None else {"dtype": out_dtype}
    out = bi_mean(x, dims, **kwargs)
    want_dtype = out_dtype or in_dtype
    assert out.dtype == want_dtype, f"{name}: dtype {out.dtype}, aten gives {want_dtype}"
    ref = ou.fp64_reference(mean_reference, x, dims or [0, 1, 2], False)
    ou.assert_exact(out, ref.to(want_dtype), what=f"{name}: value")


@pytest.mark.parametrize("name", _DTYPE_CONTROLS)
def test_output_dtype_follows_aten(name):
    """One dim without ``dtype`` and with ``dtype`` equal to the input's, several fp32 dims."""
    check_output_dtype(name)


@pytest.mark.parametrize("name", [n for n in _DTYPE_CASES if n not in _DTYPE_CONTROLS])
def test_output_dtype_follows_aten_for_dtype_and_half_inputs(name):
    """``dtype=torch.float32`` on bf16, several and all dims of bf16/fp16, fp64 inputs. Fails on
    main (O8a-M5): the one-dim path drops ``dtype``, the several-dim path returns fp32 for half
    inputs and asserts on fp64, and fp64 on one dim is predicted to fail Triton's loop-carried
    type check (an fp32 scalar accumulator)."""
    check_output_dtype(name)


_INVARIANCE = [("fp32", "last", 4096), ("fp32", "middle", 4), ("bf16", "last", 3000)]


@pytest.mark.parametrize(
    "dtype,where,n", _INVARIANCE, ids=[f"{d}-{w}-n{n}" for d, w, n in _INVARIANCE]
)
def test_rows_do_not_depend_on_the_batch(dtype, where, n):
    """Five probe slices placed among 5 ... 1029 slices (at the start, an odd middle offset and
    the end): their means are the same bits everywhere (one-dim path)."""
    dtype = DTYPES[dtype]
    if where == "last":
        probe, filler, dim = randn((5, n), dtype, 1), randn((1029, n), dtype, 2), -1
    else:  # the mHC stream average: [tokens, streams, hidden], mean over the streams
        probe, filler, dim = randn((5, n, 64), dtype, 1), randn((1029, n, 64), dtype, 2), 1
    base = bi_mean(probe, dim)
    for rows in (5, 6, 129, 1029):
        for start in sorted({0, min((rows - 5) // 2 | 1, rows - 5), rows - 5}):
            x = filler[:rows].clone()
            x[start : start + 5] = probe
            out = bi_mean(x, dim)
            ou.assert_exact(out[start : start + 5], base, what=f"rows={rows} start={start}")


def test_several_dims_do_not_depend_on_the_batch():
    """The several-dim path (`torch.sum` in fp32): a [5, 33, 160] probe's means over dims (1, 2)
    are the same bits among 5 ... 1029 slices of the batch.

    Fails on main (finding O8a-M7, pdx 7986825): `torch.sum(dim=[1, 2])` reduces each slice in an
    order that depends on the number of slices; at 129 slices all five probe means moved by 1-5
    fp32 ulp against 5 slices. No fix PR yet."""
    probe = randn((5, 33, 160), torch.float32, 4)
    filler = randn((1029, 33, 160), torch.float32, 5)
    base = bi_mean(probe, [1, 2])
    for rows in (5, 6, 129, 1029):
        x = filler[:rows].clone()
        x[rows - 5 :] = probe
        out = bi_mean(x, [1, 2])
        ou.assert_exact(out[rows - 5 :], base, what=f"rows={rows}")


@pytest.mark.parametrize("backend", BACKENDS)
def test_batch_invariant_mode_dispatches_mean(backend, monkeypatch):
    """Every backend installs the override: one-dim means (any spelling) run `mean_dim` and
    return its bits; full and several-dim means do not run it."""
    x = randn((6, 3000), torch.float32, 6)
    expected = bik.mean_dim(x, -1)
    calls = []
    launcher = bik.mean_dim

    def spy(input, dim, keepdim=False, dtype=None):
        calls.append(dim)
        return launcher(input, dim, keepdim=keepdim, dtype=dtype)

    monkeypatch.setattr(bik, "mean_dim", spy)
    with bi_mode(backend):
        outs = [
            torch.mean(x, dim=-1),
            x.mean(1),
            x.mean(dim=[1]),
            torch.mean(x.view(6, 1, 3000), dim=2).view(6),
            torch.mean(x, dim=1, keepdim=True).view(6),
        ]
        assert len(calls) == len(outs), f"{backend}: mean_dim ran {len(calls)} of {len(outs)}"
        x.mean()
        x.view(6, 30, 100).mean(dim=(1, 2))
    assert len(calls) == len(outs), f"{backend}: several-dim means ran mean_dim"
    for i, out in enumerate(outs):
        ou.assert_exact(out, expected, what=f"entry point {i} under {backend}")


_EMPTY = [((0, 7), -1), ((3, 0), -1), ((2, 0, 4), 1), ((0, 5, 3), 1), ((4, 0), 0)]


@pytest.mark.parametrize("shape,dim", _EMPTY, ids=["x".join(map(str, s)) for s, _ in _EMPTY])
def test_empty_inputs(shape, dim):
    """Empty outputs are empty; an empty reduction gives NaN, as aten."""
    x = torch.randn(*shape, device=_DEVICE)
    out = bi_mean(x, dim)
    want = torch.mean(x, dim=dim)
    assert out.shape == want.shape and out.dtype == want.dtype, f"{shape}: {tuple(out.shape)}"
    assert bool(torch.isnan(want).all()), "aten's empty reduction is NaN"
    assert bool(torch.isnan(out).all()), f"empty {shape} dim {dim}: not NaN"


_PRODUCTION = [
    pytest.param((8192, 8192), -1, id="rmsnorm-statistic-8192x8192"),
    pytest.param((4096, 4, 7168), 1, id="mhc-stream-average-4x7168"),
]


@pytest.mark.parametrize("shape,dim", _PRODUCTION)
def test_production_sizes(shape, dim):
    """The RMSNorm statistic of a prefill (fp32 x * x) and the mHC stream average."""
    g = ou.make_generator(7, device=_DEVICE)
    x = torch.randn(*shape, device=_DEVICE, generator=g)
    if dim == -1:
        x = x * x
    out = bi_mean(x, dim)
    ou.assert_close_to_reference(
        out,
        ou.fp64_reference(mean_reference, x, dim, False),
        torch.mean(x, dim=dim),
        what=f"production {shape}",
    )


@pytest.mark.sanitizer
def test_sanitizer_case():
    """Small case for compute-sanitizer: a middle-dim mean of 1025 elements (a full block and a
    masked one-element block) with a stride of 3, keepdim."""
    x, dim = shaped(BLOCK + 1, 6, "middle", torch.float32, seed=8)
    out = bi_mean(x, dim, keepdim=True)
    assert_within_division_ulps(out, ou.fp64_reference(mean_reference, x, dim, True), "sanitizer")


@pytest.mark.parametrize("rows", ou.int32_overflow_params(4096))
def test_rows_past_int32_offsets(rows):
    """[rows, 4096] fp32 whose element offsets reach 2**31: the rows around the boundary and the
    last row against fp64. Predicted to fail on main (O8a-M6): int32 offsets."""
    n = 4096
    ou.require_free_memory(rows * n * 4 + 2**30)
    block = dyadic((4096, n), torch.float32, 9)
    x = torch.empty(rows, n, device=_DEVICE)
    for start in range(0, rows, 4096):
        x[start : start + 4096] = block[: min(4096, rows - start)]
    with bi_mode():
        out = torch.mean(x, dim=-1)
    boundary = 2**31 // n
    picked = torch.tensor(sorted({0, boundary - 1, min(boundary, rows - 1), rows - 1}))
    picked = picked.to(_DEVICE)
    ref = ou.fp64_reference(mean_reference, x[picked], -1, False)
    ou.assert_exact(out[picked], ref.float(), what=f"rows {picked.tolist()}")
