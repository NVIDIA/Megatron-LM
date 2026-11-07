# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared helpers for the inference kernel oracle tests.

Each oracle test checks one kernel against a reference that does not share its algorithm. This
module holds the parts every such test needs (see README.md in this directory for the
conventions):

* fp64 references and the tolerance policy: ``fp64_reference``, ``assert_close_to_reference``
  (bounded by a trusted implementation's error, never a hand-picked atol), ``assert_exact``,
  ``assert_is_permutation``, ``assert_rows_equal_unordered`` and ``assert_quantized_bit_exact``;
* structure-derived inputs: ``boundary_sizes`` / ``size_params``, ``int32_overflow_rows`` /
  ``int32_overflow_params`` (opt-in by the ``large`` marker), ``pairwise_cases`` and
  ``dyadic_tensor``;
* memory safety: ``guarded_empty`` / ``guarded_like`` / ``check_guards`` (guard bands),
  ``poison_`` / ``poisoned_empty`` / ``assert_poisoned`` (NaN-poisoned unused memory), and the
  ``sanitizer`` marker;
* hardware: ``requires_sm`` / ``require_sm`` (skip off-target, fail when the run targets the
  hardware) and ``use_local_rank_device`` (one GPU per rank under ``torch.distributed.run``);
* samplers: ``assert_matches_distribution`` (chi-square), ``assert_counts_within_binomial_bounds``
  (exact binomial bounds), with a stated false-failure rate and a fixed seed.
"""

import contextlib
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pytest
import torch

from tests.unit_tests.find_test_cases import PLATFORM_MARKERS

# ---------------------------------------------------------------------------------------------
# Seeds and fp64 references
# ---------------------------------------------------------------------------------------------

DEFAULT_SEED = 1234
"""Seed for every oracle test's inputs. A fixed seed makes each statistical test deterministic:
its false-failure rate is the chance that a correct kernel fails for a seed picked at random."""


def make_generator(seed: int = DEFAULT_SEED, device: Any = "cpu") -> torch.Generator:
    """A ``torch.Generator`` on ``device`` seeded with ``seed``; pass it to ``torch.randn`` etc."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


@contextlib.contextmanager
def seeded(seed: int = DEFAULT_SEED) -> Iterator[None]:
    """Seed torch's global CPU and CUDA generators for the block, then restore their state.

    Use it around kernels that draw from the global generator (samplers); use
    ``make_generator`` for inputs.
    """
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def to_fp64(obj: Any) -> Any:
    """Upcast every floating-point tensor in ``obj`` (nested lists, tuples, dicts) to float64.

    The upcast is exact, so the reference sees the same rounded inputs the kernel saw (for
    example bf16 values), not the higher-precision values they were rounded from. Integer and
    bool tensors and non-tensors are returned unchanged.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(torch.float64) if obj.is_floating_point() else obj
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(to_fp64(x) for x in obj))
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_fp64(x) for x in obj)
    if isinstance(obj, dict):
        return {k: to_fp64(v) for k, v in obj.items()}
    return obj


def _floating_tensors(obj: Any) -> Iterator[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            yield obj
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _floating_tensors(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _floating_tensors(x)


def fp64_reference(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run the reference ``fn`` on fp64 copies of the kernel's floating-point inputs.

    ``fn`` is plain PyTorch written from the kernel's contract. Raises ``TypeError`` if any
    floating-point output is not float64, which means the reference rounded somewhere.
    """
    out = fn(*to_fp64(args), **to_fp64(kwargs))
    for tensor in _floating_tensors(out):
        if tensor.dtype != torch.float64:
            raise TypeError(f"fp64 reference produced a {tensor.dtype} output; it must stay fp64")
    return out


def dyadic_tensor(
    shape: Sequence[int],
    *,
    mantissa_bits: int = 4,
    exponents: Tuple[int, int] = (-2, 2),
    dtype: torch.dtype = torch.bfloat16,
    device: Any = "cpu",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Random values ``m * 2**e`` with integer ``|m| < 2**mantissa_bits`` and ``e`` in ``exponents``.

    The values are exact in ``dtype``, and every sum of them is a multiple of
    ``2**exponents[0]``. A sum of ``n`` of them is therefore exact in fp32, in any order, when
    ``n * 2**(mantissa_bits + exponents[1] - exponents[0]) <= 2**24``; a dot product of length
    ``k`` is exact when ``k * 2**(2 * (mantissa_bits + exponents[1] - exponents[0])) <= 2**24``.
    Under those bounds a reduction kernel with fp32 accumulation, whatever its blocking or
    reduction order, must return the fp64 reference rounded once to its output dtype:
    ``assert_exact(out, reference.to(out.dtype))``.
    """
    significant_bits = round(-math.log2(torch.finfo(dtype).eps)) + 1
    if not 1 <= mantissa_bits <= significant_bits:
        raise ValueError(f"mantissa_bits must be in [1, {significant_bits}] for {dtype}")
    lo, hi = exponents
    m = torch.randint(
        -(2**mantissa_bits) + 1,
        2**mantissa_bits,
        tuple(shape),
        generator=generator,
        device=device,
        dtype=torch.int64,
    )
    e = torch.randint(lo, hi + 1, tuple(shape), generator=generator, device=device)
    return (m.to(torch.float64) * pow2(e)).to(dtype)


def pow2(e: torch.Tensor) -> torch.Tensor:
    """Exactly ``2**e`` in fp64 for integer ``e``, clamped to the normal range [-1022, 1023].

    Built from the exponent bits: ``torch.ldexp`` on CUDA multiplies by ``pow(2, e)``, which is
    not always an exact power of two, so a grid built with it can be one ulp off.
    """
    e = e.to(torch.int64).clamp(-1022, 1023)
    return ((e + 1023) << 52).view(torch.float64)


# ---------------------------------------------------------------------------------------------
# Tolerance policy: floating point bounded by a trusted implementation
# ---------------------------------------------------------------------------------------------

DEFAULT_FACTOR = 2.0
"""The kernel may be at most this many times less accurate than the trusted implementation.

With bf16/fp16 outputs the final rounding dominates and correct implementations measure within
a few percent of each other (ratio ~1.0); a bf16 accumulator over 1024 terms measures ~3x. With
fp32 outputs of long reductions the summation order dominates, and correct implementations can
differ by up to ~3x (fp32 tiles of 64 summed in sequence vs. a blocked CPU GEMM, K=8192): take
the trusted implementation from the same device (cuBLAS for a GPU GEMM), or raise the factor
with the reason written in the test."""

DEFAULT_FLOOR_ULPS = 1.0
"""The floor admits this many ulps of error (ulps of the slice's largest reference magnitude) at
one element. It matters only when the trusted error is (near) zero, e.g. exactly representable
results, where ``factor * 0`` would demand bit equality; it is too small to hide a systematic
loss of precision."""


def ulp(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The spacing of ``dtype`` values at ``|x|`` (fp64 in, fp64 out; 0 where ``x`` is 0)."""
    info = torch.finfo(dtype)
    x = x.to(torch.float64).abs()
    _, exponent = torch.frexp(x)
    spacing = info.eps * pow2(exponent - 1)
    return torch.where(x > 0, spacing.clamp(min=info.tiny * info.eps), torch.zeros_like(x))


@dataclass
class ErrorReport:
    """Errors against the fp64 reference, for the slice where the bound is tightest."""

    kernel_max_abs: float
    trusted_max_abs: float
    bound_max_abs: float
    kernel_rms: float
    trusted_rms: float
    bound_rms: float
    factor: float
    slice_index: Optional[Tuple[int, ...]] = None

    def __str__(self) -> str:
        where = "" if self.slice_index is None else f" (slice {self.slice_index})"
        return (
            f"max-abs error: kernel {self.kernel_max_abs:.3e}, trusted {self.trusted_max_abs:.3e},"
            f" bound {self.bound_max_abs:.3e}{where}\n"
            f"rms error:     kernel {self.kernel_rms:.3e}, trusted {self.trusted_rms:.3e},"
            f" bound {self.bound_rms:.3e}{where}"
        )


def _index(flat: int, shape: Sequence[int]) -> Tuple[int, ...]:
    out = []
    for size in reversed(list(shape)):
        out.append(flat % size if size else 0)
        flat = flat // size if size else 0
    return tuple(reversed(out))


def _first_indices(mask: torch.Tensor, limit: int) -> List[Tuple[int, ...]]:
    flat = mask.reshape(-1).nonzero().reshape(-1)[:limit].tolist()
    return [_index(i, mask.shape) for i in flat]


def assert_close_to_reference(
    actual: torch.Tensor,
    reference: torch.Tensor,
    trusted: torch.Tensor,
    *,
    factor: float = DEFAULT_FACTOR,
    floor_ulps: float = DEFAULT_FLOOR_ULPS,
    dim: Optional[int] = None,
    what: str = "output",
) -> ErrorReport:
    """Assert the kernel is at most ``factor`` times less accurate than a trusted implementation.

    Args:
        actual: the kernel's output.
        reference: the fp64 reference (``fp64_reference``) for the same inputs.
        trusted: an independent implementation's output in the same dtype as ``actual``, for
            example torch's own bf16 op on the same inputs.
        factor: allowed ratio of the kernel's error to the trusted error.
        floor_ulps: ulp-scale floor, see ``DEFAULT_FLOOR_ULPS``.
        dim: if given, the bound is applied to each slice along ``dim`` separately (e.g.
            ``dim=-1`` for per-row bounds when rows have very different scales); by default to
            the whole tensor.
        what: name used in failure messages.

    Both the max-abs error (localized bugs) and the rms error (systematic precision loss) must
    satisfy ``kernel <= factor * trusted + floor``. The floor is ``floor_ulps`` ulps (of the
    largest reference magnitude, in the output dtype) at a single element: that much for
    max-abs, and that divided by ``sqrt(n)`` for rms. Where the reference is NaN or +-inf the
    kernel must produce the same; where it is finite the kernel must be finite.

    Norms cannot see an error smaller than the trusted implementation's largest error, e.g. a
    few ulps on one small element; use ``dim`` for row-scaled data, and dyadic inputs with
    ``assert_exact`` where the result can be made exact.

    Returns:
        The ``ErrorReport`` of the tightest slice, for logging.
    """
    if reference.dtype != torch.float64:
        raise TypeError(f"{what}: reference must be float64, got {reference.dtype}")
    if not actual.is_floating_point() or trusted.dtype != actual.dtype:
        raise TypeError(
            f"{what}: actual and trusted must share a floating dtype, got {actual.dtype} and "
            f"{trusted.dtype}; compare implementations at the same output precision"
        )
    if trusted.shape != reference.shape:
        raise ValueError(f"{what}: trusted shape {tuple(trusted.shape)} != reference shape")
    if actual is trusted or (
        actual.numel()
        and actual.data_ptr() == trusted.data_ptr()
        and actual.device == trusted.device
    ):
        raise ValueError(f"{what}: actual and trusted are the same tensor")
    if actual.shape != reference.shape:
        raise AssertionError(
            f"{what}: shape {tuple(actual.shape)} != reference shape {tuple(reference.shape)}"
        )

    device = actual.device
    a = actual.to(torch.float64)
    t = trusted.to(device=device, dtype=torch.float64)
    r = reference.to(device)
    finite = torch.isfinite(r)
    same_special = (torch.isnan(a) & torch.isnan(r)) | (a == r)
    bad_special = ~finite & ~same_special
    bad_finite = finite & ~torch.isfinite(a)
    if bool(bad_special.any()) or bool(bad_finite.any()):
        mask = bad_special | bad_finite
        lines = [
            f"  {i}: kernel {a[i].item()!r}, reference {r[i].item()!r}"
            for i in _first_indices(mask, 8)
        ]
        raise AssertionError(
            f"{what}: {int(mask.sum())} of {mask.numel()} elements disagree with the reference"
            f" on NaN/inf (non-finite where the reference is finite, or a different non-finite"
            f" value):\n" + "\n".join(lines)
        )
    if bool((finite & ~torch.isfinite(t)).any()):
        raise ValueError(
            f"{what}: the trusted implementation is non-finite where the reference is finite;"
            " choose another trusted implementation or input range"
        )

    zero = torch.zeros((), dtype=torch.float64, device=device)
    err_a = torch.where(finite, (a - r).abs(), zero)
    err_t = torch.where(finite, (t - r).abs(), zero)
    scale = torch.where(finite, r.abs(), zero)
    in_slice, reduce_dim = finite, dim
    if dim is None:
        err_a, err_t, scale, in_slice = (x.reshape(-1) for x in (err_a, err_t, scale, finite))
        reduce_dim = 0
    count = in_slice.sum(reduce_dim).clamp(min=1).to(torch.float64)

    def max_abs(x: torch.Tensor) -> torch.Tensor:
        return x.amax(reduce_dim) if x.numel() else torch.zeros_like(count)

    def rms(x: torch.Tensor) -> torch.Tensor:
        return (x.square().sum(reduce_dim) / count).sqrt()

    k_max, t_max, k_rms, t_rms = max_abs(err_a), max_abs(err_t), rms(err_a), rms(err_t)
    floor = floor_ulps * ulp(max_abs(scale), actual.dtype)
    # A relative 1e-12 absorbs fp64 rounding in the norms (an error equal to the floor passes).
    b_max = (factor * t_max + floor) * (1 + 1e-12)
    b_rms = (factor * t_rms + floor / count.sqrt()) * (1 + 1e-12)
    # Tightest slice: the largest kernel/bound ratio over both metrics.
    ratio = torch.maximum(k_max / b_max.clamp(min=1e-300), k_rms / b_rms.clamp(min=1e-300))
    worst = int(ratio.reshape(-1).argmax()) if ratio.numel() else 0

    def pick(x: torch.Tensor) -> float:
        return float(x.reshape(-1)[worst]) if x.numel() else 0.0

    report = ErrorReport(
        kernel_max_abs=pick(k_max),
        trusted_max_abs=pick(t_max),
        bound_max_abs=pick(b_max),
        kernel_rms=pick(k_rms),
        trusted_rms=pick(t_rms),
        bound_rms=pick(b_rms),
        factor=factor,
        slice_index=None if dim is None else _index(worst, ratio.shape),
    )
    failed = (k_max > b_max) | (k_rms > b_rms)
    if bool(failed.any()):
        full_err = torch.where(finite, (a - r).abs(), zero)
        i = _index(int(full_err.reshape(-1).argmax()), full_err.shape)
        slices = "" if dim is None else f" in {int(failed.sum())} of {failed.numel()} slices"
        raise AssertionError(
            f"{what}: error against the fp64 reference exceeds {factor}x the trusted"
            f" implementation's error{slices} (plus a floor of {floor_ulps} ulp at one element;"
            f" dtype {actual.dtype})\n{report}\n"
            f"largest kernel error at {i}: kernel {a[i].item()!r}, trusted {t[i].item()!r},"
            f" reference {r[i].item()!r}"
        )
    return report


# ---------------------------------------------------------------------------------------------
# Exact assertions: integer, permutation, copy and quantized outputs
# ---------------------------------------------------------------------------------------------

_INT_OF_SIZE = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _bits(t: torch.Tensor) -> torch.Tensor:
    """The tensor's raw bit patterns as an integer tensor of the same element size."""
    if t.dtype == torch.bool:
        return t.to(torch.uint8)
    if t.is_floating_point():
        return t.view(_INT_OF_SIZE[t.dtype.itemsize])
    return t


def _show(t: torch.Tensor, index: Tuple[int, ...]) -> str:
    value = t[index]
    if t.is_floating_point():
        bits = int(_bits(t)[index].item()) & ((1 << (8 * t.dtype.itemsize)) - 1)
        try:
            shown = repr(value.to(torch.float64).item())
        except (RuntimeError, TypeError):
            shown = "?"
        return f"{shown} (0x{bits:0{2 * t.dtype.itemsize}x})"
    return repr(value.item())


def assert_exact(
    actual: torch.Tensor, expected: torch.Tensor, *, what: str = "output", max_report: int = 8
) -> None:
    """Assert ``actual`` equals ``expected`` exactly, listing the first mismatches.

    Integer and bool tensors compare by value (their dtypes may differ). Floating-point tensors
    must share a dtype and compare by bit pattern, so ``-0.0 != 0.0`` and NaNs must carry the
    same bits: the contract of a copy, gather or permutation kernel.
    """
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{what}: shape {tuple(actual.shape)} != expected {tuple(expected.shape)}"
        )
    expected = expected.to(actual.device)
    if actual.is_floating_point() or expected.is_floating_point():
        if actual.dtype != expected.dtype:
            raise TypeError(f"{what}: dtypes differ ({actual.dtype} vs {expected.dtype})")
        mismatch = _bits(actual) != _bits(expected)
    else:
        mismatch = actual.to(torch.int64) != expected.to(torch.int64)
    count = int(mismatch.sum())
    if count:
        lines = [
            f"  {i}: got {_show(actual, i)}, expected {_show(expected, i)}"
            for i in _first_indices(mismatch, max_report)
        ]
        raise AssertionError(
            f"{what}: {count} of {mismatch.numel()} elements differ; first {len(lines)}:\n"
            + "\n".join(lines)
        )


def assert_is_permutation(
    perm: torch.Tensor, n: Optional[int] = None, *, what: str = "perm"
) -> None:
    """Assert the 1-D integer tensor ``perm`` holds each of ``0 .. n-1`` exactly once."""
    if perm.dim() != 1 or perm.is_floating_point():
        raise TypeError(
            f"{what}: expected a 1-D integer tensor, got {perm.dtype} {tuple(perm.shape)}"
        )
    n = perm.numel() if n is None else n
    values = perm.to(torch.int64).cpu()
    problems = []
    out_of_range = (values < 0) | (values >= n)
    if bool(out_of_range.any()):
        pos = out_of_range.nonzero().reshape(-1)[:8].tolist()
        problems.append(f"out of range [0, {n}) at {pos}: {values[pos].tolist()}")
    counts = torch.bincount(values[~out_of_range], minlength=n)
    missing = (counts == 0).nonzero().reshape(-1)
    repeated = (counts > 1).nonzero().reshape(-1)
    if missing.numel():
        problems.append(f"{missing.numel()} values missing, first {missing[:8].tolist()}")
    if repeated.numel():
        problems.append(f"{repeated.numel()} values repeated, first {repeated[:8].tolist()}")
    if perm.numel() != n:
        problems.append(f"length {perm.numel()} != {n}")
    if problems:
        raise AssertionError(f"{what} is not a permutation of 0..{n - 1}: " + "; ".join(problems))


def assert_rows_equal_unordered(
    actual: torch.Tensor, expected: torch.Tensor, *, what: str = "rows", max_report: int = 8
) -> None:
    """Assert two tensors hold the same multiset of rows (dim 0), bit for bit, in any order.

    For kernels whose output order within a group is unspecified, e.g. tokens placed by atomics
    inside an expert's block: compare each group's rows with this, and the group boundaries
    with ``assert_exact``.
    """
    if actual.shape[1:] != expected.shape[1:] or actual.dtype != expected.dtype:
        raise TypeError(
            f"{what}: row shape/dtype differ: {tuple(actual.shape)} {actual.dtype} vs"
            f" {tuple(expected.shape)} {expected.dtype}"
        )
    width = math.prod(actual.shape[1:])
    a = _bits(actual).reshape(actual.shape[0], width).to(torch.int64).cpu()
    e = _bits(expected).reshape(expected.shape[0], width).to(torch.int64).cpu()
    rows, inverse = torch.unique(torch.cat([a, e]), dim=0, return_inverse=True)
    count_a = torch.bincount(inverse[: a.shape[0]], minlength=rows.shape[0])
    count_e = torch.bincount(inverse[a.shape[0] :], minlength=rows.shape[0])
    differ = (count_a != count_e).nonzero().reshape(-1)
    if differ.numel():
        lines = []
        for u in differ[:max_report].tolist():
            row = rows[u]
            shown = row[:8].tolist()
            more = ", ..." if row.numel() > 8 else ""
            lines.append(
                f"  row bits {shown}{more}: {int(count_a[u])}x in actual, {int(count_e[u])}x"
                " expected"
            )
        raise AssertionError(
            f"{what}: {differ.numel()} distinct rows occur a different number of times"
            f" ({a.shape[0]} actual rows, {e.shape[0]} expected); first {len(lines)} (raw bits"
            " of the first 8 elements):\n" + "\n".join(lines)
        )


def assert_quantized_bit_exact(
    codes: torch.Tensor,
    scales: torch.Tensor,
    ref_codes: torch.Tensor,
    ref_scales: torch.Tensor,
    *,
    elems_per_scale: Optional[int] = None,
    what: str = "quantized",
    max_report: int = 8,
) -> None:
    """Assert quantized codes and scales equal the spec reference bit for bit, reported apart.

    Args:
        codes, ref_codes: element codes (e.g. ``float8_e4m3fn``, or ``uint8`` with two packed
            FP4 codes), same shape and element size.
        scales, ref_scales: block scales (e.g. ``float8_e8m0fnu``/``uint8`` E8M0 for MXFP8,
            ``float8_e4m3fn`` for NVFP4) in the same, unswizzled layout.
        elems_per_scale: code storage elements per scale along the last dim (MXFP8: 32;
            NVFP4 packed in bytes: 8). If given, code mismatches are split into those inside a
            block whose scale also differs (a scale bug) and those under a matching scale (a
            rounding or encoding bug).

    Any per-tensor scale (NVFP4's global fp32 scale) is checked separately with
    ``assert_exact``.
    """
    sections = []
    for name, got, ref in (("scales", scales, ref_scales), ("codes", codes, ref_codes)):
        if got.shape != ref.shape or got.dtype.itemsize != ref.dtype.itemsize:
            raise AssertionError(
                f"{what}: {name} shape/element size {tuple(got.shape)}/{got.dtype} != reference"
                f" {tuple(ref.shape)}/{ref.dtype}"
            )
    scale_bad = _bits(scales) != _bits(ref_scales.to(scales.device))
    code_bad = _bits(codes) != _bits(ref_codes.to(codes.device))
    if bool(scale_bad.any()):
        lines = [
            f"  block {i}: got {_show(scales, i)}, expected {_show(ref_scales, i)}"
            for i in _first_indices(scale_bad, max_report)
        ]
        sections.append(
            f"scales: {int(scale_bad.sum())} of {scale_bad.numel()} differ:\n" + "\n".join(lines)
        )
    if bool(code_bad.any()):
        split = ""
        if elems_per_scale is not None:
            expected_blocks = -(-codes.shape[-1] // elems_per_scale)
            if scales.shape[:-1] != codes.shape[:-1] or scales.shape[-1] != expected_blocks:
                raise ValueError(
                    f"{what}: scales {tuple(scales.shape)} do not tile codes"
                    f" {tuple(codes.shape)} by {elems_per_scale} along the last dim"
                )
            under_bad_scale = scale_bad.repeat_interleave(elems_per_scale, dim=-1)[
                ..., : codes.shape[-1]
            ]
            in_bad = int((code_bad & under_bad_scale).sum())
            split = (
                f" ({in_bad} in blocks whose scale differs, {int(code_bad.sum()) - in_bad} under"
                " a matching scale)"
            )
        lines = [
            f"  {i}: got {_show(codes, i)}, expected {_show(ref_codes, i)}"
            for i in _first_indices(code_bad, max_report)
        ]
        sections.append(
            f"codes: {int(code_bad.sum())} of {code_bad.numel()} differ{split}:\n"
            + "\n".join(lines)
        )
    if sections:
        raise AssertionError(f"{what}: not bit-exact against the spec\n" + "\n".join(sections))


# ---------------------------------------------------------------------------------------------
# Structure-derived sizes
# ---------------------------------------------------------------------------------------------


def _size_cases(
    blocks: Iterable[int],
    multiples: Iterable[int],
    degenerate: bool,
    production: Iterable[int],
    minimum: int,
    maximum: Optional[int],
) -> Dict[int, str]:
    cases: Dict[int, str] = {}

    def add(n: int, why: str) -> None:
        if n >= minimum and (maximum is None or n <= maximum):
            cases.setdefault(n, why)

    if degenerate:
        add(0, "empty")
        add(1, "single")
    for block in blocks:
        if block < 1:
            raise ValueError(f"block sizes must be positive, got {block}")
        for m in (1, *multiples):
            label = f"{block}" if m == 1 else f"{m}x{block}"
            add(m * block - 1, f"{label}-1")
            add(m * block, label)
            add(m * block + 1, f"{label}+1")
    for n in production:
        add(n, "production")
    return dict(sorted(cases.items()))


def boundary_sizes(
    *blocks: int,
    multiples: Iterable[int] = (2,),
    degenerate: bool = True,
    production: Iterable[int] = (),
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> List[int]:
    """Sizes around a kernel's tile, chunk or window sizes, sorted and de-duplicated.

    For each block size ``B`` (and ``m * B`` for each of ``multiples``): ``B - 1``, ``B``,
    ``B + 1``. With ``degenerate``: 0 and 1. ``production`` adds the sizes the kernel sees in
    deployed models. Sizes outside ``[minimum, maximum]`` are dropped. Take ``B`` from the
    kernel's own constants (``BLOCK_M``, chunk size, window), not from a guess.
    """
    return list(_size_cases(blocks, tuple(multiples), degenerate, production, minimum, maximum))


def size_params(
    *blocks: int,
    multiples: Iterable[int] = (2,),
    degenerate: bool = True,
    production: Iterable[int] = (),
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> List[Any]:
    """``boundary_sizes`` as ``pytest.param`` objects whose ids say why each size is there."""
    cases = _size_cases(blocks, tuple(multiples), degenerate, production, minimum, maximum)
    return [pytest.param(n, id=f"{n}[{why}]") for n, why in cases.items()]


INT32_LIMIT = 2**31


def int32_overflow_rows(row_elems: int, *, limit: int = INT32_LIMIT) -> List[int]:
    """Row counts whose element offsets ``row * row_elems + col`` reach ``limit`` (2**31).

    Returns the largest row count whose offsets all fit below ``limit``, the next one (the last
    row's offsets cross it), and the first count with a row that starts at or past it (these
    coincide when ``row_elems`` divides ``limit``). For byte offsets pass
    ``limit=2**31 // itemsize``. Tests at these sizes need tens of GB: use
    ``int32_overflow_params``, which marks them ``large``.
    """
    if row_elems < 1:
        raise ValueError(f"row_elems must be positive, got {row_elems}")
    fits = limit // row_elems
    starts_past = -(-limit // row_elems) + 1
    return sorted({fits, fits + 1, starts_past})


def int32_overflow_params(row_elems: int, *, limit: int = INT32_LIMIT) -> List[Any]:
    """``int32_overflow_rows`` as ``pytest.param`` objects marked ``large`` (opt-in)."""
    fits = limit // row_elems
    labels = {fits: "fits", fits + 1: "crosses"}
    return [
        pytest.param(
            n, id=f"{n}[int32-{labels.get(n, 'row-starts-past')}]", marks=pytest.mark.large
        )
        for n in int32_overflow_rows(row_elems, limit=limit)
    ]


def require_free_memory(nbytes: int, device: Any = None) -> None:
    """Skip, with the numbers, when the CUDA device has less than ``nbytes`` free."""
    free, _ = torch.cuda.mem_get_info(device)
    if free < nbytes:
        pytest.skip(f"needs {nbytes / 2**30:.1f} GiB of device memory, {free / 2**30:.1f} free")


def pairwise_cases(factors: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """A small set of cases in which every pair of values of two factors occurs at least once.

    Deterministic (greedy). Use it to combine variants, dtypes and sizes when the full product
    is too large; add any combination the kernel treats specially as an explicit case.
    """
    names = list(factors)
    values = [list(factors[name]) for name in names]
    if any(not v for v in values):
        raise ValueError("every factor needs at least one value")
    if len(names) < 2:
        return [{name: v} for name in names for v in values[0]] or [{}]
    uncovered = {
        (i, a, j, b)
        for i in range(len(names))
        for j in range(i + 1, len(names))
        for a in range(len(values[i]))
        for b in range(len(values[j]))
    }
    cases = []
    while uncovered:
        i, a, j, b = min(uncovered)
        choice: List[Optional[int]] = [None] * len(names)
        choice[i], choice[j] = a, b
        for f in range(len(names)):
            if choice[f] is not None:
                continue

            def gain(v: int) -> int:
                return sum(
                    ((g, choice[g], f, v) if g < f else (f, v, g, choice[g])) in uncovered
                    for g in range(len(names))
                    if choice[g] is not None
                )

            choice[f] = max(range(len(values[f])), key=lambda v: (gain(v), -v))
        uncovered -= {
            (p, choice[p], q, choice[q])
            for p in range(len(names))
            for q in range(p + 1, len(names))
        }
        cases.append({name: values[f][choice[f]] for f, name in enumerate(names)})
    return cases


# ---------------------------------------------------------------------------------------------
# Memory safety: NaN poison and guard bands
# ---------------------------------------------------------------------------------------------


def poison_(t: torch.Tensor) -> torch.Tensor:
    """Fill ``t`` in place with its dtype's poison and return it.

    Floating point: NaN (an unwritten output or a read of padding shows up as NaN downstream).
    Signed integers: the minimum value; unsigned: the maximum (out of range as an index).
    Dtypes without a NaN that torch cannot fill (packed FP4) get all-ones bytes.
    """
    if t.dtype == torch.bool:
        raise TypeError("bool tensors have no poison value")
    if t.is_floating_point():
        try:
            return t.fill_(float("nan"))
        except (RuntimeError, NotImplementedError):
            t.view(torch.uint8).fill_(0xFF)
            return t
    info = torch.iinfo(t.dtype)
    return t.fill_(info.min if info.min < 0 else info.max)


def poisoned_empty(shape: Sequence[int], dtype: torch.dtype, device: Any = "cuda") -> torch.Tensor:
    """``torch.empty`` filled with poison; use it for outputs and for padding/unused regions."""
    return poison_(torch.empty(tuple(shape), dtype=dtype, device=device))


def assert_poisoned(t: torch.Tensor, *, what: str = "region") -> None:
    """Assert every element of ``t`` still holds the poison: the kernel never wrote it."""
    assert_exact(
        t, poison_(torch.empty_like(t)), what=f"{what} (must be untouched, still poisoned)"
    )


# Sentinel bit patterns for guard bands: NaNs with a payload no kernel produces (so an
# out-of-bounds read turns an output into NaN, and a write changes the bits), or 0xA5 bytes.
_GUARD_BITS = {
    torch.float16: 0x7DA5,
    torch.bfloat16: 0x7FA5,
    torch.float32: 0x7FA5A5A5,
    torch.float64: 0x7FF5A5A5A5A5A5A5,
    torch.float8_e4m3fn: 0x7F,
    torch.float8_e5m2: 0x7D,
}
if hasattr(torch, "float8_e8m0fnu"):
    _GUARD_BITS[torch.float8_e8m0fnu] = 0xFF
_GUARD_ALIGN = 256
_MIN_GUARD_BYTES = 4096


def _guard_pattern(dtype: torch.dtype) -> List[int]:
    size = dtype.itemsize
    bits = _GUARD_BITS.get(dtype, int.from_bytes(b"\xa5" * size, "little"))
    return list(bits.to_bytes(size, "little"))


class GuardedBuffer:
    """A tensor with sentinel-filled guard bands before and after it in the same allocation.

    The tensor (``.tensor``) is contiguous and 256-byte aligned within the allocation. After the
    kernel runs, ``check()`` raises if any guard byte changed, i.e. the kernel wrote out of
    bounds, and says where. Float guards are NaN, so an out-of-bounds read shows up as NaN in
    the output too. Pick ``guard_bytes`` at least the kernel's largest possible overrun (e.g. a
    whole tile of rows); the default is one leading-dim row, and at least 4 KiB.
    """

    def __init__(
        self,
        shape: Sequence[int],
        dtype: torch.dtype,
        device: Any = "cuda",
        *,
        guard_bytes: Optional[int] = None,
        poison: bool = True,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        itemsize = dtype.itemsize
        numel = math.prod(self.shape)
        row_bytes = math.prod(self.shape[1:]) * itemsize if len(self.shape) > 1 else itemsize
        guard = max(_MIN_GUARD_BYTES, row_bytes) if guard_bytes is None else guard_bytes
        self.guard_bytes = -(-guard // _GUARD_ALIGN) * _GUARD_ALIGN
        self.nbytes = numel * itemsize
        self._raw = torch.empty(
            2 * self.guard_bytes + self.nbytes, dtype=torch.uint8, device=device
        )
        pattern = torch.tensor(_guard_pattern(dtype), dtype=torch.uint8, device=device)
        self._expected = pattern.repeat(self.guard_bytes // itemsize)
        self._front().copy_(self._expected)
        self._back().copy_(self._expected)
        inner = self._raw[self.guard_bytes : self.guard_bytes + self.nbytes]
        self.tensor = inner.view(dtype).view(self.shape)
        if poison and dtype != torch.bool:
            poison_(self.tensor)

    def _front(self) -> torch.Tensor:
        return self._raw[: self.guard_bytes]

    def _back(self) -> torch.Tensor:
        return self._raw[self.guard_bytes + self.nbytes :]

    def violations(self) -> List[str]:
        """Descriptions of changed guard bytes (empty if intact)."""
        out = []
        itemsize = self.dtype.itemsize
        front = (self._front() != self._expected).nonzero().reshape(-1)
        if front.numel():
            first = self.guard_bytes - int(front[0])
            last = self.guard_bytes - int(front[-1])
            out.append(
                f"{front.numel()} guard bytes changed before the start, from {first} to {last}"
                f" bytes before it (elements -{-(-first // itemsize)} to -{-(-last // itemsize)})"
            )
        back = (self._back() != self._expected).nonzero().reshape(-1)
        if back.numel():
            first, last = int(back[0]), int(back[-1])
            out.append(
                f"{back.numel()} guard bytes changed past the end, from byte +{first} to"
                f" +{last} (elements +{first // itemsize} to +{last // itemsize} past the end)"
            )
        return out

    def check(self, what: str = "buffer") -> None:
        """Raise ``AssertionError`` if the kernel wrote into either guard band."""
        problems = self.violations()
        if problems:
            raise AssertionError(
                f"{what} {self.shape} {self.dtype}: out-of-bounds write; " + "; ".join(problems)
            )


def guarded_empty(
    shape: Sequence[int],
    dtype: torch.dtype,
    device: Any = "cuda",
    *,
    guard_bytes: Optional[int] = None,
    poison: bool = True,
) -> GuardedBuffer:
    """A ``GuardedBuffer`` for a kernel output; its contents start poisoned (NaN for floats)."""
    return GuardedBuffer(shape, dtype, device, guard_bytes=guard_bytes, poison=poison)


def guarded_like(src: torch.Tensor, *, guard_bytes: Optional[int] = None) -> GuardedBuffer:
    """A ``GuardedBuffer`` holding a copy of ``src``, for kernel inputs and in-place buffers."""
    buf = GuardedBuffer(src.shape, src.dtype, src.device, guard_bytes=guard_bytes, poison=False)
    buf.tensor.copy_(src)
    return buf


def check_guards(**buffers: GuardedBuffer) -> None:
    """Check several guarded buffers, named by keyword, and report every violation at once."""
    problems = [
        f"{name}: " + "; ".join(v) for name, buf in buffers.items() if (v := buf.violations())
    ]
    if problems:
        raise AssertionError("out-of-bounds writes:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------------------------------------
# Hardware markers
# ---------------------------------------------------------------------------------------------

HOPPER = 9
BLACKWELL = 10
TARGET_SM_ENV = "MCORE_ORACLE_TARGET_SM"
"""Set to a compute-capability major (``9``, ``10``) by a runner whose node has that hardware:
tests requiring it then fail, not skip, when the device is something else."""

# The CI marker that launches tests on each architecture (only GB200 is marker-driven; H100 CI
# runs every test). Mirrors tests/unit_tests/find_test_cases.py.
LAUNCH_MARKERS = {BLACKWELL: PLATFORM_MARKERS["gb200"]}

_session: Dict[str, str] = {"markexpr": ""}


def configure_session(markexpr: Optional[str]) -> None:
    """Record the run's ``-m`` expression (called by this directory's conftest)."""
    _session["markexpr"] = markexpr or ""


def session_target_sm(markexpr: Optional[str] = None, environ: Any = None) -> Optional[int]:
    """The compute-capability major this run is declared to run on, or None if undeclared.

    ``MCORE_ORACLE_TARGET_SM`` wins; otherwise a ``-m`` expression that selects a launch marker
    (GB200 CI runs ``-m "launch_on_gb200 and ..."``) declares that marker's architecture.
    """
    environ = os.environ if environ is None else environ
    if environ.get(TARGET_SM_ENV):
        return int(environ[TARGET_SM_ENV])
    expr = _session["markexpr"] if markexpr is None else markexpr
    for major, marker in LAUNCH_MARKERS.items():
        selects = re.search(rf"\b{marker}\b", expr)
        if selects and not re.search(rf"\bnot\s+{marker}\b", expr):
            return major
    return None


def device_sm_major() -> Optional[int]:
    """The current CUDA device's compute-capability major, or None without a CUDA device."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()[0]


def use_local_rank_device(environ: Any = None) -> Optional[int]:
    """Make ``LOCAL_RANK`` (modulo the visible devices) this process's CUDA device; return it.

    CI runs each unit-test bucket under ``torch.distributed.run`` on every GPU of the node, and
    every rank runs every test. Most oracles never set up process groups (which would pick the
    rank's device), so without this every rank would run on device 0. A no-op outside
    ``torch.distributed.run`` or without CUDA (returns None).
    """
    environ = os.environ if environ is None else environ
    if "LOCAL_RANK" not in environ or not torch.cuda.is_available():
        return None
    index = int(environ["LOCAL_RANK"]) % torch.cuda.device_count()
    torch.cuda.set_device(index)
    return index


def requires_sm(*majors: int) -> Any:
    """Marker: the test runs only on these compute-capability majors (``HOPPER``, ``BLACKWELL``).

    Off-target it skips with a reason; it fails instead when the run targets one of these
    architectures (``session_target_sm``), so a misplaced runner cannot pass by skipping. A
    test that includes ``BLACKWELL`` must also be marked ``@pytest.mark.launch_on_gb200`` in its
    file, or GB200 CI never launches it; the conftest fails it otherwise.
    """
    if not majors:
        raise ValueError("requires_sm needs at least one compute-capability major")
    return pytest.mark.requires_sm(*majors)


def hardware_decision(
    required: Sequence[int],
    *,
    device_major: Optional[int],
    target_sm: Optional[int],
    launch_markers: Iterable[str] = (),
    file_has_markers: Iterable[str] = (),
) -> Tuple[str, str]:
    """Decide ``("run" | "skip" | "fail", reason)`` for a test requiring ``required`` majors.

    ``launch_markers``: markers on the test item; ``file_has_markers``: markers whose name
    appears in the test file (what CI's file selection looks for).
    """
    names = ", ".join(f"sm_{m}x" for m in required)
    for major in required:
        marker = LAUNCH_MARKERS.get(major)
        if marker and (marker not in launch_markers or marker not in file_has_markers):
            return "fail", (
                f"requires_sm({major}) without @pytest.mark.{marker} in the test file: CI"
                f" launches sm_{major}x tests only by that marker (find_test_cases.py)"
            )
    if device_major is not None and device_major in required:
        return "run", ""
    have = "no CUDA device" if device_major is None else f"sm_{device_major}x"
    if target_sm is not None and target_sm in required:
        return "fail", f"needs {names}; this run targets sm_{target_sm}x but the device is {have}"
    hints = [f" (runs in CI via {LAUNCH_MARKERS[m]})" for m in required if m in LAUNCH_MARKERS]
    return "skip", f"needs {names}, device is {have}" + "".join(hints)


def require_sm(*majors: int) -> None:
    """In a test body: skip or fail, as ``requires_sm``, unless the device is one of ``majors``.

    Prefer the ``requires_sm`` marker; this form cannot check the launch marker.
    """
    action, reason = hardware_decision(
        majors,
        device_major=device_sm_major(),
        target_sm=session_target_sm(),
        launch_markers=LAUNCH_MARKERS.values(),
        file_has_markers=LAUNCH_MARKERS.values(),
    )
    if action == "skip":
        pytest.skip(reason)
    if action == "fail":
        pytest.fail(reason, pytrace=False)


def large_selected(markexpr: Optional[str] = None) -> bool:
    """True when the run opts into ``large`` tests: its ``-m`` expression names ``large``."""
    expr = _session["markexpr"] if markexpr is None else markexpr
    return re.search(r"\blarge\b", expr) is not None


# ---------------------------------------------------------------------------------------------
# Statistical tests for samplers
# ---------------------------------------------------------------------------------------------

DEFAULT_ALPHA = 1e-6
"""False-failure rate of each statistical assertion: the probability that a correct sampler
fails it, over seeds. With the seed fixed, a test's outcome is deterministic."""

MIN_EXPECTED = 5.0
"""Bins with a smaller expected count are pooled before the chi-square test."""


def chi_square_sf(statistic: float, dof: int) -> float:
    """P(X >= statistic) for X ~ chi-square with ``dof`` degrees of freedom."""
    a = torch.tensor(dof / 2.0, dtype=torch.float64)
    x = torch.tensor(statistic / 2.0, dtype=torch.float64)
    return float(torch.special.gammaincc(a, x))


@dataclass
class ChiSquareResult:
    """Outcome of a chi-square goodness-of-fit test."""

    statistic: float
    dof: int
    p_value: float
    observed: torch.Tensor
    expected: torch.Tensor


def chi_square_gof(counts: torch.Tensor, probs: torch.Tensor) -> ChiSquareResult:
    """Pearson chi-square test of 1-D ``counts`` against category probabilities ``probs``.

    Categories with zero probability are left out (``assert_matches_distribution`` checks that
    they are empty); categories with expected counts below ``MIN_EXPECTED`` are pooled.
    """
    counts = counts.detach().to("cpu", torch.float64).reshape(-1)
    probs = probs.detach().to("cpu", torch.float64).reshape(-1)
    if counts.shape != probs.shape:
        raise ValueError(f"counts {tuple(counts.shape)} and probs {tuple(probs.shape)} differ")
    if not bool(torch.isfinite(probs).all()) or bool((probs < 0).any()) or probs.sum() <= 0:
        raise ValueError("probs must be finite, non-negative and not all zero")
    support = probs > 0
    observed = counts[support]
    n = observed.sum()
    expected = probs[support] / probs.sum() * n
    order = torch.argsort(expected)
    expected, observed = expected[order], observed[order]
    small = int((expected < MIN_EXPECTED).sum())
    # Pool the small bins (and, if the pool is still small, the next-smallest bins).
    while small < expected.numel() and 0 < small and expected[:small].sum() < MIN_EXPECTED:
        small += 1
    if small:
        expected = torch.cat([expected[:small].sum().reshape(1), expected[small:]])
        observed = torch.cat([observed[:small].sum().reshape(1), observed[small:]])
    if expected.numel() < 2 or bool((expected < MIN_EXPECTED).any()):
        raise ValueError(
            f"too few samples ({int(n)}) for a chi-square test: need at least {MIN_EXPECTED}"
            " expected per bin in two or more bins"
        )
    statistic = float(((observed - expected) ** 2 / expected).sum())
    dof = expected.numel() - 1
    return ChiSquareResult(statistic, dof, chi_square_sf(statistic, dof), observed, expected)


def _rows(counts: torch.Tensor, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if counts.shape != probs.shape or counts.dim() not in (1, 2):
        raise ValueError(
            f"counts {tuple(counts.shape)} and probs {tuple(probs.shape)} must be equal 1-D or 2-D"
        )
    counts = counts.detach().to("cpu", torch.float64)
    probs = probs.detach().to("cpu", torch.float64)
    return (counts, probs) if counts.dim() == 2 else (counts[None], probs[None])


def _check_support(counts: torch.Tensor, probs: torch.Tensor, what: str) -> None:
    outside = (probs == 0) & (counts > 0)
    if bool(outside.any()):
        cells = _first_indices(outside, 8)
        shown = ", ".join(f"{c}: {int(counts[c])}" for c in cells)
        raise AssertionError(
            f"{what}: {int(counts[outside].sum())} samples fall on categories with probability 0"
            f" (row, category: count): {shown}"
        )


def assert_matches_distribution(
    counts: torch.Tensor,
    probs: torch.Tensor,
    *,
    alpha: float = DEFAULT_ALPHA,
    what: str = "sampler",
) -> List[ChiSquareResult]:
    """Assert sample counts fit the expected distribution; false-failure rate ``alpha``.

    Args:
        counts: samples per category, ``[V]`` or ``[B, V]`` (one independent distribution per
            row), e.g. ``torch.bincount(samples, minlength=V)``.
        probs: the reference distribution (from the fp64 oracle), same shape; rows need not be
            normalized.
        alpha: probability that a correct sampler fails, split evenly over the rows
            (Bonferroni).

    Any sample on a zero-probability category fails exactly (no false failures). The rest is a
    chi-square goodness-of-fit test per row at level ``alpha / B``.
    """
    counts, probs = _rows(counts, probs)
    _check_support(counts, probs, what)
    level = alpha / counts.shape[0]
    results = [chi_square_gof(c, p) for c, p in zip(counts, probs)]
    failed = [(row, r) for row, r in enumerate(results) if r.p_value < level]
    if failed:
        lines = []
        for row, r in failed[:4]:
            contrib = (r.observed - r.expected) ** 2 / r.expected
            top = torch.argsort(contrib, descending=True)[:4].tolist()
            bins = ", ".join(
                f"observed {int(r.observed[b])} vs expected {float(r.expected[b]):.1f}" for b in top
            )
            lines.append(
                f"  row {row}: chi2 {r.statistic:.1f} on {r.dof} dof, p = {r.p_value:.3e};"
                f" largest deviations: {bins}"
            )
        raise AssertionError(
            f"{what}: samples do not fit the reference distribution in {len(failed)} of"
            f" {len(results)} rows (chi-square, level {level:.1e} per row; a correct sampler"
            f" fails with probability {alpha:.1e})\n" + "\n".join(lines)
        )
    return results


def binomial_bounds(n: int, p: float, alpha: float = DEFAULT_ALPHA) -> Tuple[int, int]:
    """Exact bounds ``(lo, hi)`` with ``P(X < lo) + P(X > hi) <= alpha`` for ``X ~ Bin(n, p)``.

    Each tail has probability at most ``alpha / 2``; computed from the binomial pmf in fp64.
    """
    if not 0.0 <= p <= 1.0 or n < 0:
        raise ValueError(f"need n >= 0 and 0 <= p <= 1, got n={n}, p={p}")
    if p == 0.0 or n == 0:
        return 0, 0
    if p == 1.0:
        return n, n
    mean, sd = n * p, math.sqrt(n * p * (1 - p))
    # The pmf outside this window is far below any alpha used in tests (Chernoff bound).
    lo_k = max(0, int(mean - 40 * sd - 100))
    hi_k = min(n, int(mean + 40 * sd + 100))
    k = torch.arange(lo_k, hi_k + 1, dtype=torch.float64)
    log_pmf = (
        math.lgamma(n + 1)
        - torch.lgamma(k + 1)
        - torch.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )
    pmf = log_pmf.exp()
    below = torch.cumsum(pmf, 0)  # P(X <= k)
    above = torch.flip(torch.cumsum(torch.flip(pmf, [0]), 0), [0])  # P(X >= k)
    lo = lo_k + int((below > alpha / 2).nonzero()[0])  # P(X < lo) <= alpha / 2
    hi = lo_k + int((above > alpha / 2).nonzero()[-1])  # P(X > hi) <= alpha / 2
    return lo, hi


def assert_counts_within_binomial_bounds(
    counts: torch.Tensor,
    probs: torch.Tensor,
    *,
    alpha: float = DEFAULT_ALPHA,
    what: str = "sampler",
) -> None:
    """Assert each category's count lies in its exact binomial bounds; false-failure rate ``alpha``.

    Same arguments as ``assert_matches_distribution``. Checks every category separately (each
    count is ``Bin(n, p_i)``), splitting ``alpha`` over all categories with ``p_i > 0``
    (Bonferroni), so it pinpoints which categories are off. Zero-probability categories must be
    empty.
    """
    counts, probs = _rows(counts, probs)
    _check_support(counts, probs, what)
    probs = probs / probs.sum(-1, keepdim=True)
    level = alpha / max(1, int((probs > 0).sum()))
    bad = []
    for row in range(counts.shape[0]):
        n = int(counts[row].sum())
        for cat in (probs[row] > 0).nonzero().reshape(-1).tolist():
            lo, hi = binomial_bounds(n, float(probs[row, cat]), level)
            c = int(counts[row, cat])
            if not lo <= c <= hi:
                bad.append(f"  ({row}, {cat}): count {c} outside [{lo}, {hi}] of n={n}")
    if bad:
        raise AssertionError(
            f"{what}: {len(bad)} category counts outside their exact binomial bounds (level"
            f" {level:.1e} each; a correct sampler fails with probability {alpha:.1e}):\n"
            + "\n".join(bad[:8])
        )


def samples_to_detect(
    p: float, delta: float, *, alpha: float = DEFAULT_ALPHA, power: float = 0.99
) -> int:
    """Samples needed to detect a category probability of ``p + delta`` instead of ``p``.

    Two-sided normal approximation at level ``alpha`` and the given power, for the per-category
    test (``assert_counts_within_binomial_bounds``; divide ``alpha`` by the number of
    categories it checks). A chi-square test over many categories needs a few times more. Use
    it to size a sampler test so it can see the bug class it targets.
    """
    z = lambda q: float(torch.special.ndtri(torch.tensor(q, dtype=torch.float64)))  # noqa: E731
    q = min(max(p + delta, 0.0), 1.0)
    num = z(1 - alpha / 2) * math.sqrt(p * (1 - p)) + z(power) * math.sqrt(q * (1 - q))
    return math.ceil((num / abs(delta)) ** 2)
