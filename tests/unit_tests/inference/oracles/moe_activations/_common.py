# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared pieces of the MoE activation and MXFP8 oracle modules (not a test module).

* Device selection. On a GPU the tests run on CUDA under the harness's hardware markers. With
  ``TRITON_INTERPRET=1`` they run the Triton kernels on CPU tensors instead, so the contract
  checks can run without a GPU. That mode drops the hardware markers, and the tests that need
  CUDA itself (CUDA graphs, cuBLAS, ``torch.compile`` parity with the GPU training ops) skip.
* ``captured_allocations``: hands a kernel launcher guarded, NaN-poisoned output buffers in
  place of its internal ``torch.empty`` / ``torch.zeros``, so the tests can check rows the
  kernel must not write and guard bands around launcher-allocated outputs.
* Exact rounding of fp64 values to bf16 (torch's own fp64 -> bf16 cast rounds twice, through
  fp32) and the per-element check of "fp32 compute, one bf16 rounding"
  (``assert_rounded_once``).
"""

import contextlib
import os
from typing import Any, Iterator, List, Optional, Sequence
from unittest import mock

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
"""The Triton interpreter runs the kernels on CPU tensors (``TRITON_INTERPRET=1``)."""

DEVICE = "cpu" if INTERPRET else "cuda"


def _interpreter_range_yields_tensors() -> None:
    """Compiled Triton gives ``tl.range`` loops an int32 tensor induction variable, and the
    kernels here call ``.to(tl.int64)`` on it; the interpreter (Triton 3.8) yields Python ints.
    Make the interpreter's ``tl.range`` yield int32 tensors as well (CPU checks only)."""
    from triton.runtime import interpreter

    patch_core = interpreter._patch_lang_core
    if getattr(patch_core, "_yields_tensors", False):
        return

    def patched(lang, scope):
        patch_core(lang, scope)

        def tensor_range(arg1, arg2=None, step=None, **_):
            start, end = (0, arg1) if arg2 is None else (arg1, arg2)
            for i in range(start, end, 1 if step is None else step):
                yield interpreter._implicit_cvt(i)

        scope.set_attr(lang, "range", tensor_range)

    patched._yields_tensors = True
    interpreter._patch_lang_core = patched


def _interpreter_float_casts_like_compiled() -> None:
    """The Triton 3.8 interpreter truncates fp32 -> bf16 casts, misreads bf16 subnormals and
    turns NaN into a finite E4M3 code; compiled Triton rounds to nearest even (fp32 -> E4M3
    with saturation, ``cvt.rn.satfinite``) and converts bf16 exactly. Make the interpreter's
    casts that involve bf16 or E4M3 follow the compiled semantics (CPU checks only)."""
    import numpy as np
    import triton.language as tl
    from triton.runtime import interpreter

    builder = interpreter.InterpreterBuilder
    if getattr(builder.cast_impl, "_like_compiled", False):
        return
    torch_dtype = {
        tl.float64: torch.float64,
        tl.float32: torch.float32,
        tl.float16: torch.float16,
        tl.bfloat16: torch.bfloat16,
        tl.float8e4nv: torch.float8_e4m3fn,
    }
    stored_as = {tl.bfloat16: torch.uint16, tl.float8e4nv: torch.uint8}

    def handled(src, dst):
        return src in torch_dtype and dst in torch_dtype and bool({src, dst} & set(stored_as))

    def convert(data, src, dst):
        t = torch.from_numpy(np.ascontiguousarray(data))
        t = t.view(torch_dtype[src]) if src in stored_as else t
        if dst == tl.float8e4nv:
            t = t.to(torch.float32).clamp(-448.0, 448.0)  # satfinite; clamp keeps NaN
        out = t.to(torch_dtype[dst])
        return (out.view(stored_as[dst]) if dst in stored_as else out).numpy()

    original_cast, original_fp_to_fp = builder.cast_impl, builder.create_fp_to_fp

    def cast_impl(self, src, dst_type):
        s, d = src.dtype.scalar, dst_type.scalar
        if handled(s, d):
            return interpreter.TensorHandle(convert(src.data, s, d), d)
        return original_cast(self, src, dst_type)

    def create_fp_to_fp(self, src, dst_type, rounding_mode):
        s, d = src.dtype.scalar, dst_type.scalar
        if handled(s, d):
            return interpreter.TensorHandle(convert(src.data, s, d), d)
        return original_fp_to_fp(self, src, dst_type, rounding_mode)

    cast_impl._like_compiled = True
    builder.cast_impl, builder.create_fp_to_fp = cast_impl, create_fp_to_fp


def _interpreter_libdevice() -> None:
    """The interpreter has no ``libdevice``: its ``exp`` and ``tanh`` (used by these kernels)
    return None. Evaluate them with numpy in the input's precision (CPU checks only)."""
    import numpy as np
    import triton.language as tl
    from triton.language.extra import libdevice
    from triton.runtime import interpreter

    def unary(fn):
        def impl(arg0, _semantic=None):
            data = arg0.handle.data
            out = fn(data).astype(data.dtype)
            return tl.tensor(interpreter.TensorHandle(out, arg0.handle.dtype), arg0.type)

        return impl

    libdevice.exp = unary(np.exp)
    libdevice.tanh = unary(np.tanh)


if INTERPRET:
    _interpreter_range_yields_tensors()
    _interpreter_float_casts_like_compiled()
    _interpreter_libdevice()


def hardware(*majors: int) -> Any:
    """``ou.requires_sm(*majors)`` on a GPU; no marker under the Triton interpreter (CPU)."""
    if INTERPRET:
        return pytest.mark.skipif(False, reason="Triton interpreter on CPU")
    return ou.requires_sm(*majors)


needs_cuda = pytest.mark.skipif(
    INTERPRET,
    reason="needs CUDA itself (CUDA graphs, cuBLAS or the GPU training ops), not the Triton"
    " interpreter",
)


def device_scalar(value: int, dtype: torch.dtype = torch.int32) -> torch.Tensor:
    """A 1-element device tensor, as the MoE path passes live-row counts (``offs[-1:]``)."""
    return torch.tensor([value], dtype=dtype, device=DEVICE)


@contextlib.contextmanager
def launcher_on_cpu() -> Iterator[None]:
    """Under the interpreter, let launchers that assert ``x.is_cuda`` accept CPU tensors."""
    if not INTERPRET:
        yield
        return
    with mock.patch.object(torch.Tensor, "is_cuda", property(lambda self: True)):
        yield


class _GuardedTorch:
    """Stands in for a kernel module's ``torch``: output allocations come back guarded."""

    def __init__(self, guard_bytes: Optional[int]):
        self.guard_bytes = guard_bytes
        self.buffers: List[ou.GuardedBuffer] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(torch, name)

    def _alloc(self, size: Sequence[Any], dtype: Optional[torch.dtype], device: Any, fill):
        if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)):
            size = size[0]
        shape = tuple(int(s) for s in size)
        dtype = dtype or torch.get_default_dtype()
        buf = ou.guarded_empty(
            shape, dtype, device or DEVICE, guard_bytes=self.guard_bytes, poison=fill is None
        )
        if fill is not None:
            buf.tensor.view(torch.uint8).fill_(fill)
        self.buffers.append(buf)
        return buf.tensor

    def empty(self, *size: Any, dtype=None, device=None, **_: Any) -> torch.Tensor:
        return self._alloc(size, dtype, device, None)

    def zeros(self, *size: Any, dtype=None, device=None, **_: Any) -> torch.Tensor:
        return self._alloc(size, dtype, device, 0)

    def empty_like(self, t: torch.Tensor, dtype=None, device=None, **_: Any) -> torch.Tensor:
        return self._alloc(t.shape, dtype or t.dtype, device or t.device, None)

    def check(self, what: str) -> None:
        """Every guarded output's guard bands are intact."""
        ou.check_guards(**{f"{what}[{i}]": buf for i, buf in enumerate(self.buffers)})


@contextlib.contextmanager
def captured_allocations(module: Any, guard_bytes: Optional[int] = None) -> Iterator[_GuardedTorch]:
    """Within the block, ``module``'s ``torch.empty`` / ``zeros`` / ``empty_like`` return guarded
    buffers: ``empty`` ones start NaN-poisoned (``ou.poison_``), ``zeros`` ones zero-filled.

    The launcher still decides the shapes and dtypes; the buffers are recorded in allocation
    order (``.buffers``) for guard and poison checks after the kernel ran.
    """
    stand_in = _GuardedTorch(guard_bytes)
    with mock.patch.object(module, "torch", stand_in):
        yield stand_in


# ---------------------------------------------------------------------------------------------
# Exact rounding to bf16 / E4M3 of fp64 values
# ---------------------------------------------------------------------------------------------

BF16_MANTISSA_BITS, BF16_EMIN = 7, -126
BF16_MAX = (2 - 2**-7) * 2.0**127


def format_quantum(x: torch.Tensor, mantissa_bits: int, emin: int) -> torch.Tensor:
    """Spacing of a binary floating-point format with ``mantissa_bits`` fraction bits and
    minimum normal exponent ``emin`` at ``|x|`` (fp64 in and out; subnormal spacing below
    ``2**emin``)."""
    _, exponent = torch.frexp(x.abs())  # |x| = m * 2**exponent, m in [0.5, 1)
    binade = (exponent - 1).clamp(min=emin)  # floor(log2 |x|), at least emin
    return torch.ldexp(torch.ones_like(x), (binade - mantissa_bits).to(torch.float64))


def round_to_format(x: torch.Tensor, mantissa_bits: int, emin: int) -> torch.Tensor:
    """Round fp64 values to the nearest value of the format, ties to even (no overflow
    handling; NaN and infinities pass through)."""
    q = format_quantum(x, mantissa_bits, emin)
    # x / q is exact (q is a power of two); torch.round rounds half to even.
    rounded = torch.round(x / q) * q
    # Keep the sign of zero: a negative value that rounds to zero is -0.
    rounded = torch.where(rounded == 0, torch.copysign(torch.zeros_like(x), x), rounded)
    return torch.where(torch.isfinite(x), rounded, x)


def rne_bf16(x: torch.Tensor) -> torch.Tensor:
    """fp64 -> bf16, correctly rounded (ties to even), as an fp64 tensor of bf16 values."""
    x = x.to(torch.float64)
    rounded = round_to_format(x, BF16_MANTISSA_BITS, BF16_EMIN)
    overflow = rounded.abs() > BF16_MAX
    return torch.where(overflow, torch.copysign(torch.full_like(x, float("inf")), x), rounded)


def bf16_neighbours(exact: torch.Tensor) -> tuple:
    """The largest bf16 value <= ``exact`` and the smallest >= it (equal when representable)."""
    exact = exact.to(torch.float64)
    q = format_quantum(exact, BF16_MANTISSA_BITS, BF16_EMIN)
    return torch.floor(exact / q) * q, torch.ceil(exact / q) * q


FP32_TINY = 2.0**-126
"""Smallest normal fp32 (and bf16) magnitude."""


def assert_same_nonfinite(actual: torch.Tensor, expected: torch.Tensor, *, what: str) -> None:
    """NaN exactly where ``expected`` is NaN and the same infinities (sign included)."""
    expected = expected.to(actual.device)
    a, e = actual.to(torch.float64), expected.to(torch.float64)
    ou.assert_exact(torch.isnan(a), torch.isnan(e), what=f"{what}: NaN positions")
    a_inf = torch.where(torch.isinf(a), a, torch.zeros_like(a))
    e_inf = torch.where(torch.isinf(e), e, torch.zeros_like(e))
    ou.assert_exact(a_inf, e_inf, what=f"{what}: infinities")


def assert_exact_except_nan_payload(
    actual: torch.Tensor, expected: torch.Tensor, *, what: str
) -> None:
    """Bitwise equality, except that any NaN matches any NaN (payloads and NaN signs differ
    between torch and Triton)."""
    expected = expected.to(actual.device)
    ou.assert_exact(torch.isnan(actual), torch.isnan(expected), what=f"{what}: NaN positions")
    keep = ~torch.isnan(actual)
    ou.assert_exact(actual[keep], expected[keep], what=what)


FP32_EVALUATION_WINDOW = 2.0**-16
"""Relative distance from a bf16 rounding midpoint within which an fp32 evaluation of the
GLU formula may round either way. fp32 SiLU errs by about ``|g| * 2**-24`` relative (the
``exp`` argument ``g * log2(e)`` is rounded) plus a few fp32 ulps: below ``2**-17`` for the
gates in these tests (|g| <= 100), so the window has a margin of 2 or more. It is 2**-8 of a
bf16 ulp: outside it, a single correct rounding of an fp32 evaluation is the correctly
rounded value."""


def assert_rounded_once(
    actual: torch.Tensor,
    exact: torch.Tensor,
    *,
    what: str,
    may_flush: Optional[torch.Tensor] = None,
    max_report: int = 8,
) -> None:
    """Assert bf16 ``actual`` is what evaluating the formula in fp32 and rounding once to bf16
    gives: the correctly rounded value of the fp64 ``exact``, or either bf16 neighbour where
    ``exact`` lies within ``FP32_EVALUATION_WINDOW`` (relative) of a rounding midpoint.

    This is the precision class of "fp32 compute, one bf16 rounding" element by element: a
    bf16 intermediate (two roundings), truncation instead of round-to-nearest-even, or a wrong
    formula fails it. Only finite ``exact`` elements are checked; the caller checks NaN and
    infinities.

    ``may_flush`` marks elements whose fp32 evaluation passes through a subnormal (a subnormal
    input, or a sigmoid below the fp32 normal range): GPU fp32 arithmetic may flush those to
    zero, so there ``±0`` is accepted as well.
    """
    exact = exact.to(actual.device, torch.float64)
    a = actual.to(torch.float64)
    nearest = rne_bf16(exact)
    lo, hi = bf16_neighbours(exact)
    near_tie = (exact - (lo + hi) / 2).abs() <= FP32_EVALUATION_WINDOW * exact.abs()
    ok = (a == nearest) | (near_tie & ((a == lo) | (a == hi)))
    if may_flush is not None:
        ok = ok | (may_flush.to(a.device) & (a == 0))
    bad = torch.isfinite(exact) & torch.isfinite(nearest) & ~ok
    count = int(bad.sum())
    if count:
        idx = bad.reshape(-1).nonzero().reshape(-1)[:max_report].tolist()
        flat_a, flat_e, flat_n = a.reshape(-1), exact.reshape(-1), nearest.reshape(-1)
        lines = [
            f"  flat {i}: got {flat_a[i].item()!r}, exact {flat_e[i].item()!r}, correctly"
            f" rounded {flat_n[i].item()!r}"
            for i in idx
        ]
        raise AssertionError(
            f"{what}: {count} of {bad.numel()} elements are not the single bf16 rounding of an"
            f" fp32 evaluation (correctly rounded, or a neighbour within"
            f" {FP32_EVALUATION_WINDOW:.1e} relative of a tie):\n" + "\n".join(lines)
        )
