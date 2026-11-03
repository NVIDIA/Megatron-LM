# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for FlashInfer's ``mxfp8_quantize`` at Megatron's call sites: the routed-MoE input,
``mxfp8_quantize(x, False)`` (linear scales) in
``megatron/core/inference/moe/flashinfer_mxfp8.py:quantize_routed_mxfp8_input``, and the swizzled
variant ``MXFP8Tensor.from_bf16(x, backend="flashinfer")``
(``megatron/core/inference/quantization/mxfp8_tensor.py``), which feeds ``mm_mxfp8``.

Contract (OCP MX v1.0 element and scale encodings, NVIDIA's RCEIL scale rule; see ``spec``):

* Input ``x [rows, H]`` BF16, contiguous. ``quantize_routed_mxfp8_input(x, Hp)`` zero-pads to
  ``Hp >= H`` columns (``Hp < H`` raises ValueError) and returns E4M3 codes ``[rows, Hp]`` and
  E8M0 scale bytes ``[rows, Hp / 32]`` (uint8, linear layout: row-major, one per 32 columns).
* Each block of 32 elements gets the smallest power-of-two scale ``X`` with ``amax <= 448 X``
  (clamped to E8M0's range; an all-zero block gets ``2**-127``, byte 0), and each element the
  E4M3 value nearest ``x / X`` with ties to even; the sign of zero is kept. Blocks and rows are
  independent, so padding columns are zero codes and rows never affect one another.
* The swizzled variant returns the same codes and the same scales in the cuBLAS 128x4 layout
  (1-D, rows padded to 128, scale columns to 4).
* Non-finite input (never produced by a healthy model) must stay visible: a NaN element decodes
  to NaN; an infinite element decodes to NaN or to the largest MXFP8 magnitude ``448 * 2**127``
  with its sign; every other block and row is still bit-exact.
* Inputs are read only.

Oracle: ``spec.mxfp8_quantize_spec`` (exact fp64 arithmetic: the scale exponent is found by exact
comparisons with ``448 * 2**e``). Tolerance class: bit-exact against the spec, codes and scales
reported apart. Not covered: blocks whose amax is below about ``2**-100``. There the block
scale nears the E8M0 floor (``2**-127``) and elements fall below FP32's normal range, where a
flush-to-zero implementation differs from the exact spec; hidden states never get there.

Structure (FlashInfer ``invokeMxFP8Quantization``): 16 elements per thread, 4 threads per scale;
the "two-load" kernel when ``Hp / 16`` is a multiple of 64 (``Hp % 1024 == 0``), else 128-thread
blocks (2048 columns per pass); a grid-stride loop over rows (grid = min(rows, SMs x blocks per
SM)). Inputs cover both kernels, column counts around 32, 1024 and 2048, rows around 128 (the
swizzled layout's row tile), rows beyond one grid pass, production sizes (H = 2688, 4096,
7168), and (opt-in ``large``) element offsets past 2**31. Hardware: Blackwell (FlashInfer's
quantizer is SM100-only).
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.flashinfer_moe import spec

ROUTED_ALIGN = 128  # quantize_routed_mxfp8_input pads to fc1_weight.padded_cols (128-aligned)

# FlashInfer's NaN and -inf handling (O4b-N1, O4b-N2) has no Megatron fix: those cases are strict
# xfails up to the FlashInfer the CI image pins (flashinfer-python in uv.lock).
FLASHINFER_IN_CI = "0.7.2.post1"
XFAIL_O4B_N1 = ou.xfail_upstream(
    "O4b-N1",
    "flashinfer-python",
    fixed_after=FLASHINFER_IN_CI,
    reason="mxfp8_quantize's amax and saturating clamp turn NaN into +448; drafted patch"
    " audit/fixes/upstream/flashinfer-mxfp8-quantize-nan.diff",
    raises=AssertionError,
)
XFAIL_O4B_N2 = ou.xfail_upstream(
    "O4b-N2",
    "flashinfer-python",
    fixed_after=FLASHINFER_IN_CI,
    reason="mxfp8_quantize decodes -inf as +448 * 2**127; drafted patch"
    " audit/fixes/upstream/flashinfer-mxfp8-quantize-neg-inf.diff",
    raises=AssertionError,
)


def _hidden(rows: int, cols: int, generator: torch.Generator) -> torch.Tensor:
    """BF16 activations, Gaussian with row magnitudes spread over 12 decades (CPU)."""
    x = torch.randn(rows, cols, generator=generator, dtype=torch.float64)
    row_exp = torch.randint(-20, 21, (rows, 1), generator=generator)
    return (x * spec.pow2(row_exp)).to(torch.bfloat16)


def _quantize_routed(x: torch.Tensor, padded: int):
    from megatron.core.inference.moe.flashinfer_mxfp8 import quantize_routed_mxfp8_input

    before = x.clone()
    codes, scales = quantize_routed_mxfp8_input(x, padded)
    ou.assert_exact(x, before, what="input is read-only")
    assert codes.dtype == torch.float8_e4m3fn and codes.shape == (x.shape[0], padded)
    assert scales.dtype == torch.uint8 and scales.shape == (x.shape[0], padded // 32)
    return spec.as_bytes(codes), scales


def _expected(x: torch.Tensor, padded: int):
    xp = torch.nn.functional.pad(x.double(), (0, padded - x.shape[1]))
    return spec.mxfp8_quantize_spec(xp)


def _check_linear(x: torch.Tensor, padded: int, what: str):
    codes, scales = _quantize_routed(x, padded)
    ref_codes, ref_scales = _expected(x, padded)
    ou.assert_quantized_bit_exact(
        codes, scales, ref_codes, ref_scales, elems_per_scale=32, what=what
    )


def test_spec_quantizer_round_trips_exact_values():
    """CPU check of the oracle: MX-representable values quantize losslessly, and the decoded
    values of random data are within half an E4M3 step of the input."""
    g = ou.make_generator()
    exact = spec.mx_exact_values(64, 256, generator=g)
    ou.assert_exact(spec.mxfp8_roundtrip(exact), exact, what="lossless round trip")
    x = _hidden(64, 256, g).double()
    codes, scales = spec.mxfp8_quantize_spec(x)
    back = spec.mxfp8_decode(codes, scales)
    # Half an E4M3 step: 2**-4 relative (normal codes), 2**-10 of the scale (subnormal codes).
    scale = spec.e8m0_value(scales).repeat_interleave(32, -1)
    assert bool(((back - x).abs() <= x.abs() * 2.0**-4 + scale * 2.0**-10).all())
    amax = x.reshape(64, 8, 32).abs().amax(-1)
    assert bool((amax <= 448 * spec.e8m0_value(scales)).all()), "a block overflows its scale"
    assert bool((amax > 224 * spec.e8m0_value(scales)).all()), "a scale is not the smallest"


# Hidden sizes: one scale, around the 1024 two-load switch and the 2048-column pass, production;
# the routed call pads to 128.
HIDDEN = [32, 96, 128, 160, 992, 1024, 1056, 2016, 2048, 2080, 2688, 4096, 7168]
ROWS = ou.boundary_sizes(128, multiples=(), degenerate=True, production=(3000,), minimum=1)


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=f"rows{c['rows']}-H{c['hidden']}")
        for c in ou.pairwise_cases({"rows": ROWS, "hidden": HIDDEN})
    ],
)
def test_routed_input_matches_spec(case):
    spec.require_flashinfer()
    rows, hidden = case["rows"], case["hidden"]
    x = _hidden(rows, hidden, ou.make_generator()).cuda()
    _check_linear(x, spec._ceil_to(hidden, ROUTED_ALIGN), what=f"mxfp8_quantize[{rows}x{hidden}]")


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize(
    "rows,hidden", [(4096, 2688), (8192, 7168)], ids=["4096x2688", "8192x7168"]
)
def test_routed_input_production_sizes(rows, hidden):
    spec.require_flashinfer()
    x = _hidden(rows, hidden, ou.make_generator()).cuda()
    _check_linear(x, spec._ceil_to(hidden, ROUTED_ALIGN), what="production")


LARGE_HIDDEN = 4096


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("rows", ou.int32_overflow_params(LARGE_HIDDEN))
def test_element_offsets_past_int32(rows):
    """Rows whose element offsets ``row * 4096 + col`` reach 2**31 (opt-in ``large``): the
    output is compared chunk by chunk, so the reference needs no fp64 copy of the whole input."""
    spec.require_flashinfer()
    ou.require_free_memory(rows * LARGE_HIDDEN * 4 + 2**33)
    g = ou.make_generator(device="cuda")
    x = torch.randn(rows, LARGE_HIDDEN, device="cuda", generator=g, dtype=torch.bfloat16)
    codes, scales = _quantize_routed(x, LARGE_HIDDEN)
    chunk = 2**16
    for start in range(0, rows, chunk):
        ref_codes, ref_scales = spec.mxfp8_quantize_spec(x[start : start + chunk].double())
        ou.assert_quantized_bit_exact(
            codes[start : start + chunk],
            scales[start : start + chunk],
            ref_codes,
            ref_scales,
            elems_per_scale=32,
            what=f"rows {start}..{start + chunk} of {rows}",
        )


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
def test_padding_wider_than_one_block_and_bad_padding():
    """Explicit padded widths beyond the next 128 (zero blocks) and a too-small width."""
    spec.require_flashinfer()
    from megatron.core.inference.moe.flashinfer_mxfp8 import quantize_routed_mxfp8_input

    x = _hidden(33, 96, ou.make_generator()).cuda()
    for padded in (96, 128, 256, 1024):
        _check_linear(x, padded, what=f"padded to {padded}")
    with pytest.raises(ValueError, match="smaller than input hidden size"):
        quantize_routed_mxfp8_input(x, 64)


def _blocks_with(values: torch.Tensor, rows: int, generator: torch.Generator) -> torch.Tensor:
    """``rows`` x (len(values) blocks): block ``b`` of each row is filled from ``values[b]``
    (a ``[32]`` pattern per block), randomly permuted inside the block and signed at random."""
    nblocks = values.shape[0]
    out = values[None].expand(rows, nblocks, 32).clone()
    perm = torch.rand(rows, nblocks, 32, generator=generator).argsort(-1)
    out = out.gather(-1, perm) * spec.random_signs((rows, nblocks, 32), generator)
    return out.reshape(rows, nblocks * 32).to(torch.bfloat16)


def _grid():
    return spec.e4m3_values()[:127].double()


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
def test_round_half_to_even_and_subnormal_codes():
    """Elements exactly halfway between E4M3 neighbours (normal and subnormal range) round to the
    even code; the block amax (256 * 2**s) fixes the scale, so the ties are exact in BF16."""
    spec.require_flashinfer()
    g = ou.make_generator()
    grid = _grid()
    mids = (grid[:-1] + grid[1:]) / 2  # 126 midpoints, from 2**-10 up to 432
    mids = mids[mids < 256]
    blocks = []
    for start in range(0, mids.numel(), 31):
        chunk = mids[start : start + 31]
        chunk = torch.cat([chunk, mids[: 31 - chunk.numel()]])
        blocks.append(torch.cat([torch.tensor([256.0], dtype=torch.float64), chunk]))
    pattern = torch.stack(blocks)  # [nb, 32]
    scale_exp = torch.randint(-20, 20, (pattern.shape[0], 1), generator=g)
    pattern = pattern * spec.pow2(scale_exp)
    assert torch.equal(pattern.to(torch.bfloat16).double(), pattern), "ties are not exact in BF16"
    x = _blocks_with(pattern, 16, g)
    _check_linear(x.cuda(), spec._ceil_to(x.shape[1], ROUTED_ALIGN), what="ties")


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("k", [-90, -8, 0, 8, 100, 119])
def test_amax_at_and_around_448_times_a_power_of_two(k):
    """A block whose amax is exactly ``448 * 2**k`` takes scale ``2**k`` (code 448); its BF16
    neighbours ``446 * 2**k`` and ``450 * 2**k`` take ``2**k`` and ``2**(k+1)``. This is the RCEIL
    boundary, where an approximate 1/448 in the scale computation picks the next scale up."""
    spec.require_flashinfer()
    g = ou.make_generator()
    patterns = []
    for amax in (448.0 * 2.0**k, 446.0 * 2.0**k, 450.0 * 2.0**k):
        rest = torch.rand(31, generator=g, dtype=torch.float64) * amax
        patterns.append(torch.cat([torch.tensor([amax], dtype=torch.float64), rest]))
    x = _blocks_with(torch.stack(patterns), 8, g)
    assert float(x.double().abs().amax()) == 450.0 * 2.0**k, "the amax pattern is not exact in BF16"
    _check_linear(x.cuda(), spec._ceil_to(x.shape[1], ROUTED_ALIGN), what=f"amax near 448*2**{k}")


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
def test_zero_blocks_signed_zeros_and_bf16_subnormals():
    """All-zero blocks (scale byte 0, codes 0), -0.0 (code 0x80), and BF16 subnormal elements in
    a normal block (they round to a signed zero)."""
    spec.require_flashinfer()
    g = ou.make_generator()
    tiny = 2.0**-130
    patterns = torch.stack(
        [
            torch.zeros(32, dtype=torch.float64),
            torch.cat([torch.tensor([1.0]), torch.full((31,), tiny)]).double(),
            torch.cat([torch.tensor([2.0**-100]), torch.full((31,), 2.0**-110)]).double(),
            torch.cat([torch.tensor([2.0**120]), torch.rand(31, generator=g).double() * 2.0**120]),
        ]
    )
    x = _blocks_with(patterns, 4, g)
    x[0, 32:64] = -0.0  # a block of negative zeros
    _check_linear(x.cuda(), 256, what="zeros and subnormals")


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.sanitizer
def test_input_is_a_prefix_of_a_nan_filled_buffer():
    """The kernel reads only its ``rows`` rows: the routed call passes ``hidden[:active_rows]``
    of a larger buffer; rows past it are NaN here and must not leak into any output."""
    spec.require_flashinfer()
    buffer = torch.full((300, 160), float("nan"), dtype=torch.bfloat16, device="cuda")
    buffer[:129] = _hidden(129, 160, ou.make_generator()).cuda()
    _check_linear(buffer[:129], 256, what="prefix of a NaN buffer")


def _decode_routed(x: torch.Tensor, padded: int) -> torch.Tensor:
    codes, scales = _quantize_routed(x, padded)
    return spec.mxfp8_decode(codes.cpu(), scales.cpu())


def _assert_other_blocks_exact(x: torch.Tensor, row: int, block: int, what: str):
    """Every block but ``(row, block)`` equals the spec, so the special value stayed in its own
    block (the routed call pads to 128 columns: 4 blocks per row here)."""
    keep = torch.ones(x.shape[0], 4, dtype=torch.bool)
    keep[row, block] = False  # the special value's own block is otherwise unspecified
    finite = x.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    ref_codes, ref_scales = _expected(finite, 128)
    codes, scales = _quantize_routed(x.cuda(), 128)
    mask = keep.repeat_interleave(32, -1)
    ou.assert_exact(codes.cpu()[mask], ref_codes[mask], what=f"codes outside the {what} block")
    ou.assert_exact(scales.cpu()[keep], ref_scales[keep], what=f"scales outside the {what} block")


@XFAIL_O4B_N1
@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
def test_nan_element_stays_nan_and_is_isolated():
    """A NaN element must decode to NaN (E4M3 0x7F/0xFF or an E8M0 0xFF scale), as the OCP FP8
    conversion and torch's float8 cast keep NaN. FlashInfer 0.7.2.post1 gives a finite value
    (finding O4b-N1, GB200): its amax ignores NaN (``__hmax2``) and its saturating clamp
    (``fminf``/``fmaxf``) turns the NaN into +448. No Megatron change can fix it: a strict xfail
    up to the FlashInfer the CI image pins (``XFAIL_O4B_N1``)."""
    spec.require_flashinfer()
    x = _hidden(4, 128, ou.make_generator())
    x[1, 37] = float("nan")
    _assert_other_blocks_exact(x, 1, 1, "NaN")
    decoded = _decode_routed(x.cuda(), 128)
    assert torch.isnan(decoded[1, 37]), (
        f"a NaN input element decoded to the finite value {float(decoded[1, 37])!r}: MXFP8 has"
        " NaN encodings (E4M3 0x7F/0xFF, E8M0 0xFF) and NaN must not be masked"
    )


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize(
    "sign", [pytest.param(1.0, id="+inf"), pytest.param(-1.0, id="-inf", marks=XFAIL_O4B_N2)]
)
def test_infinite_element_saturates_with_its_sign_and_is_isolated(sign):
    """An infinite element must decode to NaN or to the largest MXFP8 magnitude with its sign
    (saturating conversion; FlashInfer's own comment: "inf / overflow map to the max finite
    value"). FlashInfer 0.7.2.post1 turns -inf into +448 * 2**127 (finding O4b-N2, GB200): the
    scale saturates to 2**127, ``rcp.approx.ftz`` of it flushes to 0, ``-inf * 0`` is NaN and the
    clamp makes it +448. No Megatron change can fix it: ``[-inf]`` is a strict xfail up to the
    FlashInfer the CI image pins (``XFAIL_O4B_N2``)."""
    spec.require_flashinfer()
    x = _hidden(4, 128, ou.make_generator())
    x[2, 70] = sign * float("inf")
    _assert_other_blocks_exact(x, 2, 2, "inf")
    decoded = _decode_routed(x.cuda(), 128)
    got = float(decoded[2, 70])
    largest = 448.0 * 2.0**127
    assert got != got or got == sign * largest, (
        f"{sign * float('inf')} decoded to {got!r}; expected NaN or {sign * largest!r} (the"
        " largest MXFP8 magnitude, with the input's sign)"
    )


SWIZZLED = ou.pairwise_cases({"rows": [1, 127, 128, 129, 300], "hidden": [32, 160, 2688, 4096]})


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize(
    "case", [pytest.param(c, id=f"rows{c['rows']}-H{c['hidden']}") for c in SWIZZLED]
)
def test_swizzled_variant_via_mxfp8_tensor(case):
    """``MXFP8Tensor.from_bf16(x, backend="flashinfer")``: same codes, scales in the 128x4
    layout (the logical entries; padding entries are not part of the contract)."""
    spec.require_flashinfer()
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

    rows, hidden = case["rows"], case["hidden"]
    x = _hidden(rows, hidden, ou.make_generator()).cuda()
    before = x.clone()
    t = MXFP8Tensor.from_bf16(x, backend="flashinfer")
    ou.assert_exact(x, before, what="input is read-only")
    assert t.backend == "flashinfer" and t.scale.dtype == torch.uint8 and t.scale.dim() == 1
    assert t.scale.numel() == spec._ceil_to(rows, 128) * spec._ceil_to(hidden // 32, 4)
    ref_codes, ref_scales = spec.mxfp8_quantize_spec(x.double())
    scales = spec.unswizzle_128x4(t.scale.cpu(), rows, hidden // 32)
    ou.assert_quantized_bit_exact(
        spec.as_bytes(t.data).cpu(),
        scales,
        ref_codes,
        ref_scales,
        elems_per_scale=32,
        what="swizzled mxfp8_quantize",
    )
