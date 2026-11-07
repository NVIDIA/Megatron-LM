# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for DeepGEMM `bf16_gemm_nn` behind `_mm_deepgemm`, the `deepgemm` batch-invariant GEMM.

Kernel: `megatron/core/transformer/custom_layers/batch_invariant_kernels.py` (`_mm_deepgemm`,
called by `mm_batch_invariant` / `addmm_batch_invariant`, which `enable_batch_invariant_mode`
installs on `aten::mm` / `aten::addmm` for `backend="deepgemm"`). Inference reaches it through
every dense GEMM of a batch-invariant model on that backend: `torch.matmul(x, w.t())` in the
inference-optimized linears, `F.linear`, and TE linears through the patched `general_gemm`.

Contract (written from the aten ops it replaces and the backend's promise of batch invariance):
- With batch-invariant mode on and `backend="deepgemm"`, `torch.mm(a, b)` for bf16 a [M, K] and
  b [K, N] (any strides and storage offsets torch accepts: the backend replaces aten::mm for
  every caller in the process) returns D[m, n] = bf16(sum_k a[m, k] b[k, n]), accumulated
  in fp32 and rounded once, as a new contiguous [M, N] bf16 tensor. `torch.addmm(bias, a, b)`
  returns D + bias; the backend adds the bias to the rounded product in bf16 (a second rounding,
  documented; the bound below still holds). Degenerate shapes follow aten: M = 0 or N = 0 give an
  empty result, K = 0 gives zeros (plus the bias). Inputs are read only.
- Dispatch (exact): under `deepgemm`, a bf16 x bf16 mm/addmm runs DeepGEMM's `bf16_gemm_nn`
  once, and other dtypes (the fp32 router GEMM, fp16) run the Triton `matmul_persistent`; under
  `triton` every dtype runs `matmul_persistent`; `te_native` replaces neither op. Whichever GEMM
  runs, mm and addmm keep aten's semantics (addmm adds the bias) in every dtype; the GEMMs
  accumulate in fp32, so on inputs whose products and sums are exact in fp32 (dyadic values) a
  GEMM that rounds once returns the exact result rounded once to the dtype.
- Batch invariance: each output row's bits depend only on that row of a and on b: not on M, the
  row's position among the rows, the other rows, nor on a copy of the same operand placed at
  another storage offset or with a padded outer stride (same major-ness). Rows are checked for
  b = w.t() (the layout every linear passes) and for a row-major b.

Reference: fp64 `a @ b` (+ bias). Trusted implementation: `torch.mm` / `torch.addmm` on the same
device outside batch-invariant mode (cuBLAS, bf16 output). Tolerance class: floating point,
bounded per row (default factor); bit-exact for the batch-invariance properties and dispatch, and
for the dyadic mm/addmm of `test_mm_and_addmm_on_every_dispatch_path` on `matmul_persistent`
(DeepGEMM's addmm rounds twice there, so DeepGEMM stays bounded).

Structure (read after the contract): `_mm_deepgemm` copies an operand whose outer stride is not
16-byte aligned into a padded row-major buffer, allocates D with a padded stride and returns
`D.contiguous()`. DeepGEMM (pin ea2b7805) picks tiles per (M, N, K): on SM90 block_m in
{64, 128, 256} plus 16 if M <= 16 and 32 if M <= 32, block_n multiples of 16, block_k = 64,
clusters of 1 or 2 (`heuristics/sm90.hpp`); on SM100 block_m = 32 / 64 / 128 for M <= 32 / <= 64
/ larger without swap_ab, block_m multiples of 16 with swap_ab, block_n up to 128 when K <= 256
(`heuristics/sm100.hpp`). M is therefore covered at B-1, B, B+1 of 16, 32, 64, 128 and 256 plus
production sizes, K around block_k = 64 and unaligned (8, 60, 63, 65), N small, unaligned (50,
60, 129) and production, and int32-overflowing A offsets (`large`).

Main vs open PRs:
- An operand whose data pointer is not 16-byte aligned (a view at an odd storage offset) makes
  `_mm_deepgemm` raise on main: it checks only the outer stride and hands the pointer to
  DeepGEMM, whose cuBLASLt dispatch picks an algorithm for aligned buffers and fails with
  CUBLAS_STATUS_NOT_SUPPORTED (finding O8a-N5; `test_misaligned_operands_match_fp64`). #8867
  copies such operands to an aligned buffer and passes.
- At the pinned DeepGEMM, `bf16_gemm_nt`/`nn` forward to cuBLASLt's heuristic algorithm unless
  `deep_gemm.use_deterministic_algorithms(True)` is set (`csrc/apis/gemm.hpp:498`), and main never
  sets it. On main the batch-invariance tests therefore run cuBLASLt and are expected to fail
  where it changes algorithm with M (e.g. K = N = 4096 on H100); #8867 (t3-deepgemm) turns the
  switch on with the backend and makes them pass. The numeric tests pass on both.
- #8867 also passes `w.t()` to DeepGEMM without copying it (K-major), so the transposed-weight
  rows are checked for invariance within that layout, not for equality with a row-major copy
  (DeepGEMM runs a different kernel per major-ness; equality across them is not required).

Hardware: H100 and GB200 (different heuristics). Needs DeepGEMM with bf16 bindings (the
`batch_invariant` extra); a missing DeepGEMM fails the test on the target hardware.
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from tests.unit_tests.inference.oracles import oracle_utils as ou

pytestmark = [pytest.mark.launch_on_gb200, ou.requires_sm(ou.HOPPER, ou.BLACKWELL)]

_DEVICE = "cuda"

BLOCK_K = 64  # DeepGEMM: block_k = 128 * 8 / 16 bits for bf16 (heuristics/sm90.hpp, sm100.hpp)
SM90_BLOCK_M = (16, 32, 64, 128, 256)
SM100_BLOCK_M = (32, 64, 128)
M_SIZES = ou.boundary_sizes(
    *sorted(set(SM90_BLOCK_M + SM100_BLOCK_M)), multiples=(), production=(1000, 4096, 8192)
)
K_SIZES = (8, 60, BLOCK_K - 1, BLOCK_K, BLOCK_K + 1, 128, 4096, 16384)
N_SIZES = (8, 50, 60, 64, 128, 129, 4096)
LAYOUTS = ("row_major", "weight_t", "padded")  # misaligned bases: test_misaligned_operands_*


@pytest.fixture(autouse=True)
def _batch_invariant_mode_off():
    """Trusted outputs come from torch's own ops: the mode must be off around each test."""
    assert not bik.is_batch_invariant_mode_enabled(), "batch-invariant mode leaked into the test"
    yield
    assert not bik.is_batch_invariant_mode_enabled(), "the test left batch-invariant mode on"


def _require_deepgemm():
    assert bik.HAVE_DEEPGEMM_BF16, (
        "DeepGEMM with bf16 bindings is required on the target hardware (pip extra"
        " 'batch_invariant'); the deepgemm backend cannot run without it"
    )


def deepgemm_mode():
    _require_deepgemm()
    return bik.set_batch_invariant_mode(True, backend="deepgemm")


def _randn(shape, seed, scale=1.0):
    g = ou.make_generator(seed, device=_DEVICE)
    return (torch.randn(*shape, device=_DEVICE, generator=g) * scale).to(torch.bfloat16)


def place(t: torch.Tensor, layout: str, offset: int = 1) -> torch.Tensor:
    """The values of the 2-D tensor ``t`` in the requested memory layout.

    row_major: contiguous. weight_t: the transpose of a contiguous [cols, rows] tensor, as
    ``w.t()`` reaches aten::mm from a linear. padded: rows with an outer stride padded by 3
    elements (not 16-byte aligned). offset / weight_t_offset: row_major / weight_t, starting
    ``offset`` elements into a fresh allocation (a base address that is not 16-byte aligned for
    offset % 8 != 0).
    """
    rows, cols = t.shape
    if layout == "row_major":
        return t.contiguous()
    if layout == "weight_t":
        return t.t().contiguous().t()
    if layout == "padded":
        buf = torch.zeros(rows, cols + 3, device=t.device, dtype=t.dtype)
        buf[:, :cols] = t
        return buf[:, :cols]
    if layout in ("offset", "weight_t_offset"):
        flat = torch.zeros(rows * cols + offset, device=t.device, dtype=t.dtype)
        if layout == "offset":
            view = flat[offset:].view(rows, cols)
            view.copy_(t)
            return view
        view = flat[offset:].view(cols, rows)
        view.copy_(t.t())
        return view.t()
    raise ValueError(layout)


def _gemm_reference(a, b):
    return a @ b


def _addmm_reference(bias, a, b):
    return a @ b + bias


def _check_mm(out, a, b, what):
    assert out.shape == (a.shape[0], b.shape[1]) and out.dtype == torch.bfloat16, what
    assert out.is_contiguous(), f"{what}: output is not contiguous"
    ou.assert_close_to_reference(
        out, ou.fp64_reference(_gemm_reference, a, b), torch.mm(a, b), dim=-1, what=what
    )


def _id(case):
    return "-".join(f"{k}{v}" for k, v in case.items())


_MM_CASES = ou.pairwise_cases(
    {"m": M_SIZES, "k": K_SIZES, "n": N_SIZES, "a": ("row_major", "padded"), "b": LAYOUTS}
)


@pytest.mark.parametrize("case", _MM_CASES, ids=[_id(c) for c in _MM_CASES])
def test_mm_matches_fp64(case):
    """torch.mm under the deepgemm backend against fp64, bounded by cuBLAS's bf16 GEMM."""
    m, k, n = case["m"], case["k"], case["n"]
    a = place(_randn((m, k), seed=1), case["a"])
    b = place(_randn((k, n), seed=2), case["b"])
    a_before, b_before = a.clone(), b.clone()
    with deepgemm_mode():
        out = torch.mm(a, b)
    ou.assert_exact(a, a_before, what="a is read-only")
    ou.assert_exact(b, b_before, what="b is read-only")
    _check_mm(out, a, b, f"mm {_id(case)}")


_MISALIGNED = [
    # (operand, layout of that operand, element offset, m, k, n)
    ("a", "offset", 1, 64, 64, 64),
    ("a", "offset", 4, 17, 4096, 64),
    ("b", "offset", 1, 64, 64, 64),
    ("b", "weight_t_offset", 1, 1, 4096, 4096),
    ("b", "weight_t_offset", 3, 129, 4096, 136),
    ("both", "offset", 7, 33, 128, 256),
]


@pytest.mark.parametrize(
    "operand,layout,offset,m,k,n",
    _MISALIGNED,
    ids=[f"{o}-{lay}{off}-m{m}-k{k}-n{n}" for o, lay, off, m, k, n in _MISALIGNED],
)
def test_misaligned_operands_match_fp64(operand, layout, offset, m, k, n):
    """Operands whose data pointer is not 16-byte aligned: the result aten::mm gives, in bf16.

    Finding O8a-N5: fails on main, where `_mm_deepgemm` passes the pointer to DeepGEMM and its
    cuBLASLt dispatch raises CUBLAS_STATUS_NOT_SUPPORTED (H100 and GB200, run dfw 21220462 /
    hsg 8571400); passes with #8867, which copies such operands to an aligned buffer. The
    `weight_t_offset` cases pass on main too (main copies every transposed weight); they guard
    #8867, which reads `w.t()` in place.
    """
    a, b = _randn((m, k), 1), _randn((k, n), 2)
    a_layout = layout if operand in ("a", "both") else "row_major"
    b_layout = layout if operand in ("b", "both") else "row_major"
    a = place(a, a_layout.replace("weight_t_", ""), offset)
    b = place(b, b_layout, offset)
    with deepgemm_mode():
        out = torch.mm(a, b)
    _check_mm(out, a, b, f"misaligned {operand} ({layout}, {offset} elements)")


@pytest.mark.parametrize(
    "m,k,n", [(1, 64, 128), (17, 4096, 4096), (129, 60, 50), (1000, 4096, 129)]
)
def test_addmm_and_linear_match_fp64(m, k, n):
    """addmm (bias) and F.linear(x, w) (w.t() reaches aten::mm) against fp64."""
    a, w, bias = _randn((m, k), 1), _randn((n, k), 2), _randn((n,), 3)
    with deepgemm_mode():
        out = torch.addmm(bias, a, w.t())
        lin = F.linear(a, w)
        lin_bias = F.linear(a, w, bias)
    ref = ou.fp64_reference(_addmm_reference, bias, a, w.t())
    ou.assert_close_to_reference(out, ref, torch.addmm(bias, a, w.t()), dim=-1, what="addmm")
    _check_mm(lin, a, w.t(), "F.linear")
    ou.assert_close_to_reference(
        lin_bias, ref, F.linear(a, w, bias), dim=-1, what="F.linear with bias"
    )


@pytest.mark.parametrize("m,k,n", [(0, 64, 128), (5, 64, 0), (0, 0, 0), (7, 0, 33)])
def test_degenerate_shapes_follow_aten(m, k, n):
    """M = 0 or N = 0: empty; K = 0: zeros, and the bias for addmm."""
    a, b, bias = _randn((m, k), 1), _randn((k, n), 2), _randn((n,), 3)
    with deepgemm_mode():
        out = torch.mm(a, b)
        out_bias = torch.addmm(bias, a, b)
    ou.assert_exact(out, torch.zeros(m, n, device=_DEVICE, dtype=torch.bfloat16), what="mm")
    ou.assert_exact(out_bias, bias.expand(m, n).contiguous(), what="addmm")


# (K, N): small and unaligned K, square projection (where cuBLASLt changes algorithm with M on
# H100), an FFN up-projection, and a long-K GEMM.
_INVARIANCE_SHAPES = [(64, 128), (60, 136), (4096, 4096), (4096, 14336), (16384, 64)]


@pytest.mark.parametrize("b_layout", ["weight_t", "row_major"])
@pytest.mark.parametrize(
    "k,n", _INVARIANCE_SHAPES, ids=[f"k{k}-n{n}" for k, n in _INVARIANCE_SHAPES]
)
def test_rows_do_not_depend_on_m(k, n, b_layout):
    """Rows of A[:M] @ B equal the first M rows of the largest GEMM, for M across every block_m.

    Fails on main where cuBLASLt (which DeepGEMM forwards to on main) picks an M-dependent
    algorithm; passes with #8867.
    """
    m_max = max(M_SIZES)
    a = _randn((m_max, k), 1)
    b = place(_randn((k, n), 2), b_layout)
    with deepgemm_mode():
        full = torch.mm(a, b)
        for m in M_SIZES:
            if m == 0:
                continue
            out = torch.mm(a[:m], b)
            ou.assert_exact(out, full[:m], what=f"rows of M={m} vs M={m_max} ({b_layout})")


@pytest.mark.parametrize("k,n", [(64, 128), (4096, 4096)], ids=["k64-n128", "k4096-n4096"])
def test_rows_do_not_depend_on_neighbours(k, n):
    """Probe rows give the same bits alone and at any position among random rows.

    Expected to fail on main for K = N = 4096 (cuBLASLt); passes with #8867.
    """
    probes = _randn((3, k), 7)
    w = _randn((n, k), 2)
    g = ou.make_generator(5, device=_DEVICE)
    with deepgemm_mode():
        alone = torch.mm(probes, w.t())
        for i in range(3):
            ou.assert_exact(torch.mm(probes[i : i + 1], w.t()), alone[i : i + 1], what="1 row")
        for batch in (16, 33, 129, 257):
            a = _randn((batch, k), batch)
            pos = torch.randperm(batch, device=_DEVICE, generator=g)[:3]
            a[pos] = probes
            ou.assert_exact(torch.mm(a, w.t())[pos], alone, what=f"probes among {batch} rows")


@pytest.mark.parametrize("operand", ["a", "b"])
@pytest.mark.parametrize("m,k,n", [(1, 4096, 4096), (129, 64, 128), (300, 1024, 2048)])
def test_rows_do_not_depend_on_operand_placement(m, k, n, operand):
    """The same row-major values at another storage offset or with a padded stride: same bits.

    On main the misaligned operand reaches cuBLASLt as is, which may choose another algorithm
    for it (expected to fail there); #8867 copies it to an aligned buffer first.
    """
    a, b = _randn((m, k), 1), _randn((k, n), 2)
    with deepgemm_mode():
        expected = torch.mm(a, b)
        for layout in ("padded", "offset"):
            if operand == "a":
                out = torch.mm(place(a, layout), b)
            else:
                out = torch.mm(a, place(b, layout))
            ou.assert_exact(out, expected, what=f"{operand} {layout}")


@pytest.mark.parametrize("backend", ["deepgemm", "triton", "te_native"])
def test_dispatch_decisions(backend, monkeypatch):
    """Which GEMM runs under each backend, by dtype (exact), and that its result is right."""
    if backend == "deepgemm":
        _require_deepgemm()
    calls = []
    if bik.deep_gemm is not None:
        real_dg = bik.deep_gemm.bf16_gemm_nn

        def dg_spy(*args, **kwargs):
            calls.append(("deepgemm", args[0].dtype))
            return real_dg(*args, **kwargs)

        monkeypatch.setattr(bik.deep_gemm, "bf16_gemm_nn", dg_spy)
    real_mp = bik.matmul_persistent

    def mp_spy(a, b, bias=None, *args, **kwargs):
        calls.append(("triton", a.dtype))
        return real_mp(a, b, bias, *args, **kwargs)

    monkeypatch.setattr(bik, "matmul_persistent", mp_spy)
    a, w, bias = _randn((33, 256), 1), _randn((128, 256), 2), _randn((128,), 3)
    expected = {
        "deepgemm": {torch.bfloat16: "deepgemm", torch.float16: "triton", torch.float32: "triton"},
        "triton": {torch.bfloat16: "triton", torch.float16: "triton", torch.float32: "triton"},
        "te_native": {torch.bfloat16: None, torch.float16: None, torch.float32: None},
    }[backend]
    for dtype, target in expected.items():
        x, wt, bb = a.to(dtype), w.to(dtype).t(), bias.to(dtype)
        for op in ("mm", "addmm"):
            calls.clear()
            with bik.set_batch_invariant_mode(True, backend=backend):
                out = torch.mm(x, wt) if op == "mm" else torch.addmm(bb, x, wt)
            want = [] if target is None else [(target, dtype)]
            assert calls == want, f"{backend} {op} {dtype}: ran {calls}, expected {want}"
            if target == "deepgemm":
                # matmul_persistent's results, bias included, are checked exactly by
                # test_mm_and_addmm_on_every_dispatch_path.
                ref = (
                    ou.fp64_reference(_gemm_reference, x, wt)
                    if op == "mm"
                    else ou.fp64_reference(_addmm_reference, bb, x, wt)
                )
                trusted = torch.mm(x, wt) if op == "mm" else torch.addmm(bb, x, wt)
                ou.assert_close_to_reference(out, ref, trusted, dim=-1, what=f"{op} {dtype}")


# Dyadic values m * 2**e with |m| < 8 and e in [-1, 1]: products are multiples of 2**-2 below
# 196, so for K <= 256 every partial sum of x @ w^T + bias is exact in fp32 (|sum| < 2**16, 18
# significant bits) and finite in fp16 (< 65504), whatever the accumulation order. The 3-bit
# mantissas are also exact in TF32, which matmul_persistent uses for fp32 operands (O8a-N4).
_DYADIC = dict(mantissa_bits=3, exponents=(-1, 1))
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@pytest.mark.parametrize("m,k,n", [(33, 256, 128), (129, 64, 50)])
@pytest.mark.parametrize("dtype", list(_DTYPES), ids=list(_DTYPES))
@pytest.mark.parametrize("backend", ["deepgemm", "triton"])
def test_mm_and_addmm_on_every_dispatch_path(backend, dtype, m, k, n):
    """mm and addmm keep aten's semantics on whichever GEMM the backend dispatches them to
    (`test_dispatch_decisions` pins which): the product, plus the bias for addmm, in every dtype.

    On dyadic inputs the result is exact in fp32, so `matmul_persistent` (fp32 accumulation, the
    bias added in fp32, one rounding) must return the fp64 result rounded once, bit for bit.
    DeepGEMM's bf16 path (deepgemm backend, bf16) adds the bias after rounding the product (a
    second rounding, documented above) and is bounded against fp64 like the other DeepGEMM cases.
    """
    if backend == "deepgemm":
        _require_deepgemm()
    dtype = _DTYPES[dtype]
    g = ou.make_generator(11, device=_DEVICE)
    x = ou.dyadic_tensor((m, k), dtype=dtype, device=_DEVICE, generator=g, **_DYADIC)
    w = ou.dyadic_tensor((n, k), dtype=dtype, device=_DEVICE, generator=g, **_DYADIC)
    bias = ou.dyadic_tensor((n,), dtype=dtype, device=_DEVICE, generator=g, **_DYADIC)
    with bik.set_batch_invariant_mode(True, backend=backend):
        out = torch.mm(x, w.t())
        out_bias = torch.addmm(bias, x, w.t())
    ref = ou.fp64_reference(_gemm_reference, x, w.t())
    ref_bias = ou.fp64_reference(_addmm_reference, bias, x, w.t())
    if backend == "deepgemm" and dtype == torch.bfloat16:
        _check_mm(out, x, w.t(), "mm (DeepGEMM)")
        ou.assert_close_to_reference(
            out_bias, ref_bias, torch.addmm(bias, x, w.t()), dim=-1, what="addmm (DeepGEMM)"
        )
    else:
        ou.assert_exact(out, ref.to(dtype), what=f"mm ({backend}, {dtype})")
        ou.assert_exact(out_bias, ref_bias.to(dtype), what=f"addmm ({backend}, {dtype})")


_LARGE_K = 4096


@pytest.mark.parametrize("m", ou.int32_overflow_params(_LARGE_K))
def test_a_offsets_past_int32(m):
    """A [M, 4096] whose element offsets reach 2**31: the last rows still match fp64."""
    ou.require_free_memory(m * _LARGE_K * 2 + (2 << 30))
    a = torch.empty(m, _LARGE_K, device=_DEVICE, dtype=torch.bfloat16)
    for start in range(0, m, 65536):
        stop = min(m, start + 65536)
        a[start:stop] = _randn((stop - start, _LARGE_K), seed=start)
    w = _randn((128, _LARGE_K), 2)
    with deepgemm_mode():
        out = torch.mm(a, w.t())
    boundary = (2**31) // _LARGE_K
    rows = [r for r in sorted({0, boundary - 1, boundary, boundary + 1, m - 1}) if r < m]
    _check_mm(out[rows], a[rows], w.t(), f"rows {rows} of {m}")


@pytest.mark.sanitizer
def test_sanitizer_case():
    """Small case for compute-sanitizer: unaligned K and N (padded copies) and a partial tile."""
    a, b = _randn((17, 60), 1), _randn((60, 50), 2)
    with deepgemm_mode():
        out = torch.mm(a, b)
    _check_mm(out, a, b, "sanitizer")
