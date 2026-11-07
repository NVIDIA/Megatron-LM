# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `_log_softmax_kernel`, the batch-invariant log-softmax.

Kernel: `megatron/core/transformer/custom_layers/batch_invariant_kernels.py` (`_log_softmax_kernel`,
launched by `log_softmax`, installed on `aten::_log_softmax` by `enable_batch_invariant_mode` under
every backend). Inference reaches it through every `log_softmax` of the log-prob paths (text
generation controller, dynamic context, both samplers) whenever batch-invariant mode is on.

Contract (written from the aten op it replaces, `torch.log_softmax(x, dim=-1)`):
- For each row r of the input viewed as [-1, n]: y[r, j] = x[r, j] - log(sum_i exp(x[r, i])),
  computed in fp32 and rounded once to the input dtype (bf16, fp16, fp32). The output is a new
  contiguous tensor of the input's shape and dtype; the input is read only.
- Special values as in torch: a row holding a NaN or a +inf is all NaN; a row of only -inf is
  all NaN; a -inf entry of any other row gives -inf; a one-column row gives 0. Empty inputs give
  empty outputs.
- Only the last dimension: any other dim raises. `F.log_softmax(fp16, dtype=torch.float32)` on
  CUDA (aten's `half_to_float=True`) is not implemented and must raise, never return a wrong
  value or dtype.
- Batch invariance: a row's output bits depend only on that row's values: not on the number of
  rows, the row's position, the other rows, the input's storage offset (base alignment) or its
  contiguity.
- Under batch-invariant mode, with any backend (te_native, triton, deepgemm), `torch.log_softmax`
  along the last dim returns exactly this kernel's bits.

Reference: fp64 `x - logsumexp(x)` with torch's special-value rule (`log_softmax_reference`).
Trusted implementation: `torch.log_softmax` on the same device, outside batch-invariant mode.
Tolerance class: floating point, bounded per row (`dim=-1`; default factor for bf16/fp16, 4 for
fp32 outputs, see FP32_FACTOR); bit-exact for every
metamorphic property; exact for rows whose arithmetic is exact (`test_overflowing_rows_are_exact`).

Structure (read after the contract was written): one program per row, three passes over the row in
blocks of BLOCK_SIZE = 1024 columns (max, sum of exp, store), fp32 max and sum, int64 row offsets.
One compiled variant per input dtype and per Triton specialization of n_cols, the row stride and
the base pointer (integers and pointers divisible by 16 are specialized), so the alignment case
compares launches whose loads vectorize differently. Covered: n_cols around 1024 and 2048 (0, 1,
B-1, B, B+1), production vocabularies (32000, 151936, 152064, 262144), rows 0..1000 and
int32-overflowing element offsets (`large`), bf16/fp16/fp32, every special value above, every
backend's aten registration, and one `sanitizer` case.

Hardware: H100 and GB200 (Blackwell lowers `tl.sum`/`tl.exp` differently).
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from tests.unit_tests.inference.oracles import oracle_utils as ou

pytestmark = [pytest.mark.launch_on_gb200, ou.requires_sm(ou.HOPPER, ou.BLACKWELL)]

_DEVICE = "cuda"

BLOCK_SIZE = 1024  # batch_invariant_kernels.py, log_softmax(): BLOCK_SIZE = 1024
PRODUCTION_VOCABS = (32000, 151936, 152064, 262144)
DTYPES = (torch.bfloat16, torch.float16, torch.float32)
# n_cols = 0 has its own test (test_empty_inputs).
N_COLS = ou.boundary_sizes(BLOCK_SIZE, multiples=(2,), production=PRODUCTION_VOCABS, minimum=1)
N_ROWS = (0, 1, 2, 3, 31, 32, 33, 1000)


@pytest.fixture(autouse=True)
def _batch_invariant_mode_off():
    """Trusted outputs are computed with torch's own op: the mode must be off around each test."""
    assert not bik.is_batch_invariant_mode_enabled(), "batch-invariant mode leaked into the test"
    yield
    assert not bik.is_batch_invariant_mode_enabled(), "the test left batch-invariant mode on"


def log_softmax_reference(x: torch.Tensor) -> torch.Tensor:
    """fp64 log-softmax over the last dim, with torch's special-value rule (from the contract)."""
    y = x - torch.logsumexp(x, dim=-1, keepdim=True)
    poisoned = (torch.isnan(x) | (x == float("inf"))).any(dim=-1, keepdim=True)
    return torch.where(poisoned, torch.full_like(y, float("nan")), y)


def _rows(n_rows: int, n_cols: int, dtype: torch.dtype, seed: int = ou.DEFAULT_SEED):
    """Random logits, each row with its own scale (2**-2 .. 2**3) so per-row bounds matter."""
    g = ou.make_generator(seed, device=_DEVICE)
    x = torch.randn(n_rows, n_cols, device=_DEVICE, generator=g, dtype=torch.float32)
    scale = torch.pow(2.0, torch.randint(-2, 4, (n_rows, 1), device=_DEVICE, generator=g))
    return (x * scale).to(dtype)


# fp32 outputs: the kernel's tl.exp / tl.log lower to the approximate PTX instructions
# (ex2.approx after a log2(e) prescale, documented to about 2**-22 relative), while torch's CUDA
# expf / logf are within 1-2 ulp. Where |y| < 1 (rows of a few columns), a log-sum-exp error of
# 1e-7 is several ulps of y, so the kernel may measure up to ~4x torch's error while correct.
# A dropped term or a wrong maximum errs by orders of magnitude more. bf16 / fp16 outputs are
# dominated by the final rounding and keep the default factor.
FP32_FACTOR = 4.0


def _check(out: torch.Tensor, x: torch.Tensor, what: str) -> None:
    assert out.shape == x.shape, f"{what}: shape {tuple(out.shape)} != {tuple(x.shape)}"
    assert out.dtype == x.dtype, f"{what}: dtype {out.dtype} != {x.dtype}"
    assert out.is_contiguous(), f"{what}: output is not contiguous"
    ou.assert_close_to_reference(
        out,
        ou.fp64_reference(log_softmax_reference, x),
        torch.log_softmax(x, dim=-1),
        factor=FP32_FACTOR if x.dtype == torch.float32 else ou.DEFAULT_FACTOR,
        dim=-1,
        what=what,
    )


def _case_id(case):
    return f"{case['n_rows']}x{case['n_cols']}-{str(case['dtype']).split('.')[-1]}"


_NUMERIC_CASES = ou.pairwise_cases({"n_cols": N_COLS, "n_rows": N_ROWS, "dtype": DTYPES})


@pytest.mark.parametrize("case", _NUMERIC_CASES, ids=[_case_id(c) for c in _NUMERIC_CASES])
def test_log_softmax_matches_fp64(case):
    """Every n_cols boundary, row count and dtype against fp64, bounded by torch.log_softmax.

    The input sits in a guarded buffer whose guards are NaN sentinels: a load past a row's end
    that the mask fails to stop turns that row's outputs NaN.
    """
    x = _rows(case["n_rows"], case["n_cols"], case["dtype"])
    buf = ou.guarded_like(x)
    out = bik.log_softmax(buf.tensor)
    ou.check_guards(x=buf)
    ou.assert_exact(buf.tensor, x, what="input is read-only")
    _check(out, x, f"log_softmax {_case_id(case)}")


_EMPTY_SHAPES = [(0, 5), (3, 0), (0, 0), (2, 0, 7), (2, 3, 0)]


@pytest.mark.parametrize("shape", _EMPTY_SHAPES, ids=["x".join(map(str, s)) for s in _EMPTY_SHAPES])
def test_empty_inputs(shape):
    """Empty inputs, including an empty last dim, give empty outputs of the same shape (as torch).

    Fails on main for an empty last dim: the launcher's ``reshape(-1, 0)`` raises (finding
    O8a-N2; unreachable from inference, where the last dim is the vocabulary).
    """
    x = torch.empty(shape, device=_DEVICE, dtype=torch.bfloat16)
    out = bik.log_softmax(x)
    assert out.shape == x.shape and out.dtype == x.dtype, f"{tuple(out.shape)} {out.dtype}"


def _special_rows(n: int, dtype: torch.dtype) -> torch.Tensor:
    """One row per special value of the contract, n columns each."""
    inf, nan = float("inf"), float("nan")
    g = ou.make_generator(device=_DEVICE)
    rows = _rows(9, n, torch.float32)
    pick = torch.randint(0, n, (9,), device=_DEVICE, generator=g).tolist()
    rows[0] = -inf  # all -inf: NaN
    rows[1] = -inf
    rows[1, pick[1]] = 1.5  # a single finite entry: 0 there, -inf elsewhere
    rows[2, pick[2]] = -inf  # one -inf in a finite row
    rows[3, pick[3]] = nan  # NaN poisons the row
    rows[4, pick[4]] = inf  # +inf poisons the row
    rows[5] = 0.75  # constant: -log(n) everywhere
    rows[6] = -100.0
    rows[6, pick[6]] = 0.0  # exp(-100) underflows fp32: the max entry is -log(1 + tiny)
    tiny = torch.finfo(dtype).tiny
    rows[7] = rows[7].sign() * tiny / 4  # subnormal inputs
    rows[8] = rows[8] * 1e-3  # nearly constant: y close to -log(n)
    return rows.to(dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("n_cols", [1, 2, 3, BLOCK_SIZE, BLOCK_SIZE + 1, 2 * BLOCK_SIZE + 1])
def test_special_values_follow_torch(n_cols, dtype):
    """NaN, +-inf, constant, underflowing and subnormal rows, in and across blocks."""
    x = _special_rows(n_cols, dtype)
    _check(bik.log_softmax(x), x, f"special rows n={n_cols} {dtype}")


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("n_cols", [2, 3, BLOCK_SIZE + 1])
def test_overflowing_rows_are_exact(n_cols, dtype):
    """Rows [M, -M, 0, ...] with M near the dtype's maximum: every step is exact.

    The row maximum is M, the sum of exponentials is exactly 1 (the others underflow), so the
    log-sum-exp is M and y = x - M: 0, -2M (overflows to -inf) and -M. The fp64 reference
    rounded to the dtype gives the same values, so the output must match it bit for bit.
    """
    big = {torch.float16: 6.0e4, torch.bfloat16: 3.0e38, torch.float32: 3.0e38}[dtype]
    x = torch.zeros(2, n_cols, device=_DEVICE, dtype=torch.float32)
    x[:, 0] = big
    x[0, 1] = -big
    x[1, -1] = -big
    x = x.to(dtype)
    out = bik.log_softmax(x)
    ou.assert_exact(out, ou.fp64_reference(log_softmax_reference, x).to(dtype), what="overflow")


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("n_cols", [1, BLOCK_SIZE - 1, BLOCK_SIZE + 1, 32000])
def test_row_bits_do_not_depend_on_the_batch(n_cols, dtype):
    """A row gives the same bits alone, at any position of any batch, and under permutation."""
    probes = _rows(3, n_cols, dtype, seed=7)
    alone = bik.log_softmax(probes)
    for i in range(3):
        ou.assert_exact(bik.log_softmax(probes[i : i + 1]), alone[i : i + 1], what=f"probe {i}")
    g = ou.make_generator(11, device=_DEVICE)
    for batch in (7, 64, 513):
        x = _rows(batch, n_cols, dtype, seed=batch)
        positions = torch.randperm(batch, device=_DEVICE, generator=g)[:3]
        x[positions] = probes
        out = bik.log_softmax(x)
        ou.assert_exact(out[positions], alone, what=f"probes in a batch of {batch}")
    x = _rows(64, n_cols, dtype, seed=3)
    perm = torch.randperm(64, device=_DEVICE, generator=g)
    ou.assert_exact(bik.log_softmax(x[perm]), bik.log_softmax(x)[perm], what="row permutation")


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("n_cols", [BLOCK_SIZE, BLOCK_SIZE + 1, 32000, 32001])
def test_row_bits_do_not_depend_on_storage_offset(n_cols, dtype):
    """The same rows at every element offset within 16 bytes give the bits of an aligned copy.

    n_cols = 1024 and 32000 make the row stride divisible by 16 elements, so an aligned base lets
    Triton vectorize the loads, which an offset base does not; 1025 and 32001 never vectorize.
    """
    rows = 4
    x = _rows(rows, n_cols, dtype)
    expected = bik.log_softmax(x)
    per_16_bytes = 16 // x.element_size()
    flat = torch.empty(rows * n_cols + 2 * per_16_bytes, device=_DEVICE, dtype=dtype)
    for offset in range(1, per_16_bytes + 1):
        view = flat[offset : offset + rows * n_cols].view(rows, n_cols)
        view.copy_(x)
        ou.assert_exact(bik.log_softmax(view), expected, what=f"storage offset {offset}")


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_layout_and_rank_do_not_change_bits(dtype):
    """Strided views and a 3-D input give the bits of the contiguous 2-D input."""
    x = _rows(6, 1500, dtype)
    expected = bik.log_softmax(x)
    wide = torch.zeros(6, 1600, device=_DEVICE, dtype=dtype)
    wide[:, :1500] = x
    ou.assert_exact(bik.log_softmax(wide[:, :1500]), expected, what="row-strided view")
    column_major = x.t().contiguous().t()
    ou.assert_exact(bik.log_softmax(column_major), expected, what="column-major view")
    out3 = bik.log_softmax(x.view(2, 3, 1500))
    assert out3.shape == (2, 3, 1500)
    ou.assert_exact(out3.reshape(6, 1500), expected, what="3-D input")


_BACKENDS = ("te_native", "triton", "deepgemm")


@pytest.mark.parametrize("backend", _BACKENDS)
def test_batch_invariant_mode_dispatches_log_softmax(backend, monkeypatch):
    """Under every backend, aten's log-softmax entry points run this kernel (exact).

    The kernel is identified twice: a spy on the launcher counts the calls, and the outputs are
    the launcher's bits.
    """
    if backend == "deepgemm":
        assert bik.HAVE_DEEPGEMM_BF16, "the deepgemm backend needs DeepGEMM's bf16 bindings"
    x = _rows(5, 3000, torch.bfloat16)
    x32 = _rows(5, 3000, torch.float32)
    expected, expected32 = bik.log_softmax(x), bik.log_softmax(x32)
    calls = []
    launcher = bik.log_softmax

    def spy(input, dim=-1):
        calls.append(input.dtype)
        return launcher(input, dim=dim)

    monkeypatch.setattr(bik, "log_softmax", spy)
    with bik.set_batch_invariant_mode(True, backend=backend):
        outs = [
            torch.log_softmax(x, dim=-1),
            F.log_softmax(x, dim=1),
            x.log_softmax(-1),
            torch.log_softmax(x.view(5, 1, 3000), dim=2).view(5, 3000),
            torch.log_softmax(x32, dim=-1),
        ]
    for i, out in enumerate(outs[:4]):
        ou.assert_exact(out, expected, what=f"bf16 entry point {i} under {backend}")
    ou.assert_exact(outs[4], expected32, what=f"fp32 under {backend}")
    assert len(calls) == len(outs), f"{backend}: the kernel ran {len(calls)} of {len(outs)} times"


def test_unsupported_calls_raise():
    """A non-last dim and fp16 with dtype=float32 raise instead of returning a wrong result."""
    x = _rows(6, 40, torch.float32).view(2, 3, 40)
    with pytest.raises(ValueError):
        bik.log_softmax(x, dim=1)
    with bik.set_batch_invariant_mode(True, backend="triton"):
        with pytest.raises((ValueError, RuntimeError)):
            torch.log_softmax(x, dim=1)
        with pytest.raises((AssertionError, NotImplementedError, RuntimeError)):
            F.log_softmax(x.half(), dim=-1, dtype=torch.float32)


_LARGE_N_COLS = PRODUCTION_VOCABS[0]


@pytest.mark.parametrize("n_rows", ou.int32_overflow_params(_LARGE_N_COLS))
def test_rows_past_int32_element_offsets(n_rows):
    """Rows whose element offsets reach 2**31 (bf16, 32000 columns): the last rows still match."""
    n_cols, dtype = _LARGE_N_COLS, torch.bfloat16
    ou.require_free_memory(2 * n_rows * n_cols * 2 + (1 << 30))
    x = torch.empty(n_rows, n_cols, device=_DEVICE, dtype=dtype)
    chunk = 4096
    for start in range(0, n_rows, chunk):
        stop = min(n_rows, start + chunk)
        x[start:stop] = _rows(stop - start, n_cols, dtype, seed=start)
    out = bik.log_softmax(x)
    boundary = (2**31) // n_cols
    rows = sorted({0, boundary - 1, boundary, min(boundary + 1, n_rows - 1), n_rows - 1})
    rows = [r for r in rows if r < n_rows]
    _check(out[rows], x[rows], f"rows {rows} of {n_rows}")


@pytest.mark.sanitizer
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_sanitizer_case(dtype):
    """Small case for compute-sanitizer: tails of a partial block, one and three rows."""
    for n_rows in (1, 3):
        x = _rows(n_rows, BLOCK_SIZE + 3, dtype)
        _check(bik.log_softmax(x), x, f"sanitizer {n_rows} rows")
