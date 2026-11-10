# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `matmul_kernel_persistent`, the Triton batch-invariant GEMM (`matmul_persistent`).

Kernel: `megatron/core/transformer/custom_layers/batch_invariant_kernels.py`
(`matmul_kernel_persistent`, launched by `matmul_persistent`, which `mm_batch_invariant` /
`addmm_batch_invariant` install on `aten::mm` / `aten::addmm` under
`enable_batch_invariant_mode(backend="triton")`, and for non-bf16 operands under `"deepgemm"`).
Inference reaches it through every dense GEMM of a batch-invariant model on the `triton` backend
(`torch.matmul(x, w.t())` in the inference-optimized linears, `F.linear`, TE linears through the
patched `general_gemm`) and through the fp32 GEMMs of the `deepgemm` backend.
`test_deepgemm_dense.py` checks which GEMM each backend dispatches to and mm/addmm on dyadic
inputs; this module checks the kernel itself.

Contract (written from the aten ops it replaces and the backend's promise of batch invariance):
- `matmul_persistent(a, b, bias)` for a [M, K], b [K, N] of one dtype (bf16, fp16, fp32), any
  strides and storage offsets, returns a new contiguous [M, N] tensor of that dtype with
  C[m, n] = round(sum_k a[m, k] b[k, n] + bias[n]): fp32 accumulation, the bias added in fp32,
  one rounding. M = 0 or N = 0 give an empty result, K = 0 gives the bias (or zeros). Inputs are
  read only.
- Under the backend, `torch.mm` / `torch.addmm` / `F.linear` keep aten's semantics: `addmm(bias,
  a, b, beta=, alpha=)` returns beta * bias + alpha * (a @ b) for any bias broadcastable to
  [M, N] (a 1-D [N] view of any stride, [1, N], [M, N]); fp32 operands follow PyTorch's fp32
  matmul precision setting (`torch.set_float32_matmul_precision`: "highest" means no TF32).
- Batch invariance: each output row's bits depend only on that row of a and on b: not on M, the
  row's position, the other rows.
- Element offsets: the kernel must address every element of a strided operand, so an operand
  whose element offsets reach 2**31 must be launched with 64-bit offsets (`*_LARGE`).

Reference: fp64 `a @ b + bias`. Trusted implementation: `torch.mm` / `torch.addmm` on the same
device outside batch-invariant mode (cuBLAS). fp32 numerics use dyadic operands whose products
and sums are exact in fp32 (with or without TF32), so a correct fp32 GEMM returns the fp64 result
exactly in any summation order (with fp32 outputs a per-row bound would compare summation orders,
not correctness); `test_fp32_mm_honours_highest_precision` checks the precision setting
separately. Tolerance class: floating point bounded per row (default factor 2) for bf16 / fp16;
exact for fp32 numerics, degenerate shapes, batch invariance and the launch flags; exact for
addmm's bias semantics on dyadic inputs.

Structure (read after the contract): a persistent grid of min(#SMs, tiles) programs walks the
tiles in groups of GROUP_SIZE_M = 8 M-tiles; per-dtype tiles BLOCK_M x BLOCK_N x BLOCK_K = bf16
128 x 128 x 64, fp16 128 x 256 x 64, fp32 128 x 128 x 32; tail rows and columns load row/col 0
(clamped) and are masked on store; the K tail is masked with zeros; the bias is added to the fp32
accumulator; int32 offsets unless A_LARGE / B_LARGE / C_LARGE (set from numel > 2**31). Covered:
M around 128 and 9 M-tiles (one more than GROUP_SIZE_M), N around 128 and 256, K around 32 and 64,
more tiles than SMs, row-major / transposed / padded / offset operands, bias, every dtype, M = 0,
N = 0, K = 0, production shapes, and the 2**31 offset limit (launch flags; `large` launches).

Main vs open PRs (each failing case is deselected and run as an expect-fail task):
- `test_fp32_mm_honours_highest_precision` fails on main: the kernel's `tl.dot` uses Triton's
  default TF32 for fp32 operands (finding O8a-N4). No fix PR yet.
- `test_addmm_strided_vector_bias` fails on main: the kernel reads the bias as contiguous
  (finding O8a-M1); `test_addmm_broadcast_bias_and_scalars` fails: a 2-D bias asserts and
  beta/alpha raise TypeError in `addmm_batch_invariant` (finding O8a-M2). No fix PR yet.
- `test_launch_uses_64_bit_offsets_for_strided_views` fails on main: A_LARGE / B_LARGE come from
  numel, so a strided view whose offsets cross 2**31 with fewer elements runs on int32 offsets
  (finding O8a-M3; checked on the launch, never launched). No fix PR yet.
All other cases pass on main and on every open PR touching the file (#8867, #8957, #8865, #8869).

Hardware: H100 and GB200 (Triton lowers `tl.dot` to wgmma on SM90 and tcgen05 on SM100).
"""

import contextlib

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from tests.unit_tests.inference.oracles import oracle_utils as ou

pytestmark = [pytest.mark.launch_on_gb200, ou.requires_sm(ou.HOPPER, ou.BLACKWELL)]

_DEVICE = "cuda"

# matmul_persistent's per-dtype configs
BLOCK_M = 128
BLOCK_N = {torch.bfloat16: 128, torch.float16: 256, torch.float32: 128}
BLOCK_K = {torch.bfloat16: 64, torch.float16: 64, torch.float32: 32}
GROUP_SIZE_M = 8
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
INT32_LIMIT = 2**31


@pytest.fixture(autouse=True)
def _batch_invariant_mode_off():
    """Trusted outputs come from torch's own ops: the mode must be off around each test."""
    assert not bik.is_batch_invariant_mode_enabled(), "batch-invariant mode leaked into the test"
    yield
    assert not bik.is_batch_invariant_mode_enabled(), "the test left batch-invariant mode on"


def gemm_reference(a, b, bias=None):
    out = a @ b
    return out if bias is None else out + bias


def operand(shape, dtype, seed, scale=1.0):
    """bf16 / fp16: random values (exact in bf16). fp32: dyadic values (3-bit mantissas,
    exponents -1..1; ``scale`` ignored), whose dot products up to K = 8192 plus a bias are exact
    in fp32 (`ou.dyadic_tensor`)."""
    g = ou.make_generator(seed, device=_DEVICE)
    if dtype == torch.float32:
        return ou.dyadic_tensor(
            shape, mantissa_bits=3, exponents=(-1, 1), dtype=dtype, device=_DEVICE, generator=g
        )
    x = torch.randn(*shape, device=_DEVICE, generator=g) * scale
    return x.to(torch.bfloat16).to(dtype)


def noisy(shape, dtype, seed, scale=1.0):
    """Random values whose sums depend on the summation order (for the invariance checks)."""
    g = ou.make_generator(seed, device=_DEVICE)
    return (torch.randn(*shape, device=_DEVICE, generator=g) * scale).to(dtype)


def place(t: torch.Tensor, layout: str) -> torch.Tensor:
    """The values of the 2-D tensor ``t`` in a memory layout, with NaN in every unused element.

    row_major: contiguous. transposed: the transpose of a contiguous [cols, rows] tensor (as
    ``w.t()`` reaches aten::mm). padded: rows with an outer stride 5 elements longer. offset:
    row-major, starting 3 elements into its allocation. A load outside the view reads NaN.
    """
    rows, cols = t.shape
    if layout == "row_major":
        return t.contiguous()
    if layout == "transposed":
        return t.t().contiguous().t()
    if layout == "padded":
        buf = torch.full((rows, cols + 5), float("nan"), device=t.device, dtype=t.dtype)
        buf[:, :cols] = t
        return buf[:, :cols]
    if layout == "offset":
        flat = torch.full((rows * cols + 3,), float("nan"), device=t.device, dtype=t.dtype)
        view = flat[3:].view(rows, cols)
        view.copy_(t)
        return view
    raise ValueError(layout)


def check_gemm(out, a, b, bias, what):
    """Shape, dtype, contiguity; per-row bound against fp64, trusted cuBLAS."""
    m, n = a.shape[0], b.shape[1]
    assert tuple(out.shape) == (m, n), f"{what}: shape {tuple(out.shape)}"
    assert out.dtype == a.dtype, f"{what}: dtype {out.dtype}"
    assert out.is_contiguous(), f"{what}: output is not contiguous"
    if out.numel() == 0:
        return
    ref = ou.fp64_reference(gemm_reference, a, b, bias)
    if out.dtype == torch.float32:
        ou.assert_exact(out, ref.to(out.dtype), what=f"{what} (dyadic fp32: exact)")
        return
    trusted = torch.mm(a, b) if bias is None else torch.addmm(bias, a, b)
    ou.assert_close_to_reference(out, ref, trusted, dim=-1, what=what)


def run_launcher(a, b, bias=None):
    """`matmul_persistent` on copies of the inputs; the inputs must come back unchanged."""
    before = [t.clone() for t in (a, b) + (() if bias is None else (bias,))]
    out = bik.matmul_persistent(a, b, bias)
    for t, saved in zip((a, b, bias), before):
        ou.assert_exact(t, saved, what="inputs are read-only")
    return out


def _id(case):
    return "-".join(f"{k}{v}" for k, v in case.items())


_CASES = ou.pairwise_cases(
    {
        "dtype": tuple(DTYPES),
        "m": (1, 127, 128, 129, 9 * BLOCK_M - 127, 4096),
        "n": (1, 127, 128, 129, 255, 256, 257, 4096),
        "k": (31, 32, 33, 63, 64, 65, 4096),
        "a": ("row_major", "transposed", "padded", "offset"),
        "b": ("transposed", "row_major", "padded"),
        "bias": (False, True),
    }
)


@pytest.mark.parametrize("case", _CASES, ids=[_id(c) for c in _CASES])
def test_matmul_matches_fp64(case):
    """Every dtype config, M/N/K at B-1/B/B+1 of their tiles (BLOCK_N 256 for fp16, BLOCK_K 32
    for fp32), 9 M-tiles (a partial GROUP_SIZE_M group), more tiles than SMs, every operand layout
    (unused elements are NaN), with and without bias: per row against fp64, bounded by cuBLAS."""
    dtype = DTYPES[case["dtype"]]
    m, n, k = case["m"], case["n"], case["k"]
    a = place(operand((m, k), dtype, seed=m + k), case["a"])
    b = place(operand((k, n), dtype, seed=n + 7, scale=k**-0.5), case["b"])
    bias = operand((n,), dtype, seed=n) if case["bias"] else None
    out = run_launcher(a, b, bias)
    check_gemm(out, a, b, bias, _id(case))


@pytest.mark.parametrize("dtype", list(DTYPES.values()), ids=list(DTYPES))
@pytest.mark.parametrize("m,n,k", [(0, 64, 64), (64, 0, 64), (64, 96, 0), (0, 0, 0)])
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "bias"])
def test_degenerate_shapes(dtype, m, n, k, with_bias):
    """M = 0 / N = 0: empty; K = 0: the bias (or zeros), exactly."""
    a, b = operand((m, k), dtype, seed=1), operand((k, n), dtype, seed=2)
    bias = operand((n,), dtype, seed=3) if with_bias else None
    out = run_launcher(a, b, bias)
    assert tuple(out.shape) == (m, n) and out.dtype == dtype and out.is_contiguous()
    want = torch.zeros(m, n, device=_DEVICE, dtype=dtype)
    if bias is not None:
        want = want + bias
    ou.assert_exact(out, want, what=f"degenerate {m}x{n}x{k}")


_INVARIANCE = [
    ("bf16", 4096, 4096),
    ("bf16", 65, 129),
    ("fp16", 1000, 257),
    ("fp32", 7168, 256),
    ("fp32", 33, 129),
]


@pytest.mark.parametrize(
    "dtype,k,n", _INVARIANCE, ids=[f"{d}-k{k}-n{n}" for d, k, n in _INVARIANCE]
)
def test_rows_do_not_depend_on_m(dtype, k, n):
    """Seven probe rows at the start, at an odd middle offset and at the end of batches of 7 ...
    8197 rows (one to more than 64 M-tiles, past the SM count and across GROUP_SIZE_M groups):
    the same bits everywhere, with row-major and transposed b."""
    dtype = DTYPES[dtype]
    probe = noisy((7, k), dtype, seed=5)
    filler = noisy((8197, k), dtype, seed=6)
    w = noisy((n, k), dtype, seed=7, scale=k**-0.5)
    for b in (w.t(), w.t().contiguous()):
        base = bik.matmul_persistent(probe, b)
        for m in (7, 128, 129, 9 * BLOCK_M - 127, 8197):
            for start in sorted({0, min((m - 7) // 2 | 1, m - 7), m - 7}):
                a = filler[:m].clone()
                a[start : start + 7] = probe
                out = bik.matmul_persistent(a, b)
                ou.assert_exact(
                    out[start : start + 7], base, what=f"M={m} start={start} {tuple(b.stride())}"
                )


@contextlib.contextmanager
def float32_matmul_precision(precision: str):
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)


@pytest.mark.parametrize("backend", ["triton", "deepgemm"])
def test_fp32_mm_honours_highest_precision(backend):
    """With fp32 matmul precision "highest" (PyTorch's default), an fp32 `torch.mm` under the
    backend is an fp32 GEMM: within 2x cuBLAS's fp32 error against fp64 per row.

    Fails on main (finding O8a-N4): `tl.dot` runs fp32 operands at Triton's default TF32, so the
    inputs lose 13 mantissa bits (errors ~1e3 times cuBLAS's). The deepgemm backend sends fp32 to
    the same kernel ("intentional higher-precision operations such as the fp32 MoE router").
    """
    if backend == "deepgemm":
        assert bik.HAVE_DEEPGEMM_BF16, "the deepgemm backend needs DeepGEMM's bf16 bindings"
    g = ou.make_generator(8, device=_DEVICE)
    a = torch.randn(129, 1024, device=_DEVICE, generator=g)
    b = torch.randn(1024, 256, device=_DEVICE, generator=g) / 32
    with float32_matmul_precision("highest"):
        trusted = torch.mm(a, b)
        with bik.set_batch_invariant_mode(True, backend=backend):
            out = torch.mm(a, b)
    ou.assert_close_to_reference(
        out, ou.fp64_reference(gemm_reference, a, b), trusted, dim=-1, what=f"fp32 mm {backend}"
    )


# aten::addmm(self, mat1, mat2, *, beta=1, alpha=1): bias forms aten accepts.
def _addmm_call(form, bias_values, a, b):
    """(callable, expected fp64) for one aten addmm form, from dyadic values."""
    m, n = a.shape[0], b.shape[1]
    exact = a.double() @ b.double()
    if form == "vector":
        bias = bias_values[0].clone()
        return (lambda: torch.addmm(bias, a, b)), exact + bias.double()
    if form == "vector_strided":
        storage = torch.full((2 * n,), float("nan"), device=_DEVICE, dtype=a.dtype)
        storage[::2] = bias_values[0]
        bias = storage[::2]
        return (lambda: torch.addmm(bias, a, b)), exact + bias.double()
    if form == "row":
        bias = bias_values[0].view(1, n).clone()
        return (lambda: torch.addmm(bias, a, b)), exact + bias.double()
    if form == "matrix":
        bias = bias_values[:m].clone()
        return (lambda: torch.addmm(bias, a, b)), exact + bias.double()
    bias = bias_values[0].clone()
    if form == "beta":
        return (lambda: torch.addmm(bias, a, b, beta=0.5)), exact + 0.5 * bias.double()
    return (lambda: torch.addmm(bias, a, b, alpha=2.0)), 2.0 * exact + bias.double()


_ADDMM_BACKENDS = dict(
    argnames="backend,dtype",
    argvalues=[("triton", "bf16"), ("deepgemm", "fp32")],
    ids=["triton-bf16", "dg-fp32"],
)


def check_addmm(backend, dtype, form):
    """`torch.addmm` in one of aten's forms under the backend, exactly.

    Dyadic operands (3-bit mantissas, K = 96) make every product, sum and scaling exact in fp32,
    so the result must be the fp64 value rounded once.
    """
    if backend == "deepgemm":
        assert bik.HAVE_DEEPGEMM_BF16, "the deepgemm backend needs DeepGEMM's bf16 bindings"
    dtype = DTYPES[dtype]
    g = ou.make_generator(9, device=_DEVICE)
    dyadic = dict(mantissa_bits=3, exponents=(-1, 1), dtype=dtype, device=_DEVICE, generator=g)
    a = ou.dyadic_tensor((33, 96), **dyadic)
    b = ou.dyadic_tensor((96, 50), **dyadic)
    bias_values = ou.dyadic_tensor((33, 50), **dyadic)
    call, expected = _addmm_call(form, bias_values, a, b)
    with bik.set_batch_invariant_mode(True, backend=backend):
        out = call()
    assert tuple(out.shape) == (33, 50) and out.dtype == dtype, f"{form}: {tuple(out.shape)}"
    ou.assert_exact(out, expected.to(dtype), what=f"addmm {form} under {backend}")


@pytest.mark.parametrize(**_ADDMM_BACKENDS)
def test_addmm_vector_bias(backend, dtype):
    """`torch.addmm(bias, a, b)` with a contiguous [N] bias, the form `F.linear` uses."""
    check_addmm(backend, dtype, "vector")


@pytest.mark.parametrize(**_ADDMM_BACKENDS)
def test_addmm_strided_vector_bias(backend, dtype):
    """A strided [N] bias view (stride 2, NaN in between). Fails on main (O8a-M1): the kernel
    loads ``bias_ptr + n``, ignoring the stride."""
    check_addmm(backend, dtype, "vector_strided")


@pytest.mark.parametrize("form", ["row", "matrix", "beta", "alpha"])
@pytest.mark.parametrize(**_ADDMM_BACKENDS)
def test_addmm_broadcast_bias_and_scalars(backend, dtype, form):
    """[1, N] and [M, N] biases, beta and alpha. Fails on main (O8a-M2): `addmm_batch_invariant`
    asserts a 1-D bias and its signature takes no beta or alpha (TypeError)."""
    check_addmm(backend, dtype, form)


class _LaunchSpy:
    """Stands in for the kernel: records the launch flags, launches nothing."""

    def __init__(self):
        self.flags = None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.flags = {k: kwargs[k] for k in ("A_LARGE", "B_LARGE", "C_LARGE")}

        return launch


def max_offset(t: torch.Tensor) -> int:
    """The largest element offset a kernel walking ``t`` adds to its data pointer."""
    return sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()) if size)


def _flags_for(a, b, monkeypatch):
    spy = _LaunchSpy()
    monkeypatch.setattr(bik, "matmul_kernel_persistent", spy)
    bik.matmul_persistent(a, b)
    return spy.flags


def _meta(shape, stride=None):
    if stride is None:
        return torch.empty(shape, device="meta", dtype=torch.bfloat16)
    return torch.empty_strided(shape, stride, device="meta", dtype=torch.bfloat16)


_CONTIGUOUS = {
    # name: (a, b): numel at and one past 2**31 for each operand
    "a_fits": ((INT32_LIMIT // 4096, 4096), (4096, 8)),
    "a_crosses": ((INT32_LIMIT // 4096 + 1, 4096), (4096, 8)),
    "b_fits": ((8, 32768), (32768, INT32_LIMIT // 32768)),
    "b_crosses": ((8, 32768), (32768, INT32_LIMIT // 32768 + 1)),
}


@pytest.mark.parametrize("name", list(_CONTIGUOUS))
def test_launch_uses_64_bit_offsets_for_large_operands(name, monkeypatch):
    """Contiguous operands with 2**31 and 2**31 + 1 elements: 64-bit offsets exactly when an
    element offset reaches 2**31 (checked on the launch; meta tensors, nothing runs)."""
    a_shape, b_shape = _CONTIGUOUS[name]
    a, b = _meta(a_shape), _meta(b_shape)
    flags = _flags_for(a, b, monkeypatch)
    for key, t in (("A_LARGE", a), ("B_LARGE", b)):
        assert flags[key] == (max_offset(t) >= INT32_LIMIT), f"{name}: {key}={flags[key]}"


_STRIDED = {
    # name: (a shape, a stride, b shape, b stride): views spanning more than 2**31 elements
    "a_padded_rows": ((INT32_LIMIT // 8192, 4096), (8192 + 64, 1), (4096, 8), (8, 1)),
    "a_column_slice": ((300000, 2048), (8192, 1), (2048, 8), (8, 1)),
    "b_padded_rows": ((8, 32768), (32768, 1), (32768, 40000), (70000, 1)),
    "b_transposed_slice": ((8, 4096), (4096, 1), (4096, 300000), (1, 8192)),
}


@pytest.mark.parametrize("name", list(_STRIDED))
def test_launch_uses_64_bit_offsets_for_strided_views(name, monkeypatch):
    """A strided view (a column slice, padded rows, a slice of a transposed weight) whose element
    offsets reach 2**31 while it holds fewer than 2**31 elements must still get 64-bit offsets.

    Fails on main (finding O8a-M3): the flags come from ``numel() > 2**31``, so the kernel would
    compute these offsets in int32 and wrap. Checked on the launch with meta tensors: launching
    would read out of bounds (CI.md rule 10).
    """
    a_shape, a_stride, b_shape, b_stride = _STRIDED[name]
    a, b = _meta(a_shape, a_stride), _meta(b_shape, b_stride)
    assert max(max_offset(a), max_offset(b)) >= INT32_LIMIT, "the case does not cross 2**31"
    assert a.numel() <= INT32_LIMIT and b.numel() <= INT32_LIMIT
    flags = _flags_for(a, b, monkeypatch)
    for key, t in (("A_LARGE", a), ("B_LARGE", b)):
        if max_offset(t) >= INT32_LIMIT:
            assert flags[key], f"{name}: {key} is False for offsets up to {max_offset(t)}"


@pytest.mark.parametrize("m", ou.int32_overflow_params(4096))
def test_rows_past_int32_offsets(m):
    """A [M, 4096] bf16 operand whose element offsets reach 2**31 (A_LARGE): the rows around the
    boundary and the last rows against fp64."""
    k, n = 4096, 64
    ou.require_free_memory(m * k * 2 + m * n * 2 + 2**30)
    a = torch.empty(m, k, device=_DEVICE, dtype=torch.bfloat16)
    block = operand((4096, k), torch.bfloat16, seed=10)
    for start in range(0, m, 4096):
        a[start : start + 4096] = block[: min(4096, m - start)]
    b = operand((k, n), torch.bfloat16, seed=11, scale=k**-0.5)
    out = bik.matmul_persistent(a, b)
    boundary = INT32_LIMIT // k
    rows = torch.tensor(sorted({0, boundary - 1, min(boundary, m - 1), m - 1}), device=_DEVICE)
    check_gemm(out[rows], a[rows], b, None, f"rows {rows.tolist()}")


_PRODUCTION = [
    pytest.param("bf16", 8192, 4096, 14336, id="bf16-fc1-llama8b"),
    pytest.param("bf16", 4096, 7168, 2048, id="bf16-proj-dsv3"),
    pytest.param("fp32", 4096, 7168, 256, id="fp32-router-dsv3"),
]


@pytest.mark.parametrize("dtype,m,k,n", _PRODUCTION)
def test_production_sizes(dtype, m, k, n):
    """Deployed shapes with ``w.t()`` weights (as linears pass them): per row against fp64."""
    dtype = DTYPES[dtype]
    a = operand((m, k), dtype, seed=12)
    b = operand((n, k), dtype, seed=13, scale=k**-0.5).t()
    check_gemm(run_launcher(a, b), a, b, None, f"production {m}x{k}x{n}")


@pytest.mark.sanitizer
@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_sanitizer_case(dtype):
    """Small case for compute-sanitizer: two M-tiles (the second with 2 rows), a partial N tile,
    a partial K tile, a transposed weight and a bias."""
    dtype = DTYPES[dtype]
    m, n, k = BLOCK_M + 2, BLOCK_N[dtype] + 1, BLOCK_K[dtype] + 1
    a = operand((m, k), dtype, seed=14)
    b = operand((n, k), dtype, seed=15, scale=k**-0.5).t()
    bias = operand((n,), dtype, seed=16)
    check_gemm(run_launcher(a, b, bias), a, b, bias, "sanitizer")


def test_linear_entry_points_run_the_kernel(monkeypatch):
    """F.linear (3-D input without bias, 2-D input with bias: aten's mm and addmm) and
    torch.matmul on a 3-D input run the launcher under the triton backend and return its bits."""
    calls = []
    launcher = bik.matmul_persistent

    def spy(a, b, bias=None, **kwargs):
        # Forward any keywords: open PRs add some (#8957 out_dtype=, the addmm fix beta=/alpha=).
        calls.append(bias is not None)
        return launcher(a, b, bias, **kwargs)

    x = operand((3, 5, 256), torch.bfloat16, seed=17)
    w = operand((96, 256), torch.bfloat16, seed=18, scale=256**-0.5)
    bias = operand((96,), torch.bfloat16, seed=19)
    flat = x.reshape(15, 256)
    want = launcher(flat, w.t()).view(3, 5, 96)
    want_bias = launcher(flat, w.t(), bias).view(3, 5, 96)
    monkeypatch.setattr(bik, "matmul_persistent", spy)
    with bik.set_batch_invariant_mode(True, backend="triton"):
        outs = {
            "linear 3-D": (F.linear(x, w), want),
            "linear 2-D bias": (F.linear(flat, w, bias), want_bias.view(15, 96)),
            "matmul 3-D": (torch.matmul(x, w.t()), want),
        }
    assert len(calls) == len(outs), f"the launcher ran {len(calls)} times for {len(outs)} calls"
    for name, (out, expected) in outs.items():
        ou.assert_exact(out, expected, what=name)
