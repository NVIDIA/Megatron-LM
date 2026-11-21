# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for FlashInfer's ``nvfp4_quantize(x, 1 / 2688, sfLayout=layout_linear,
per_token_activation=True)`` at Megatron's call site ``quantize_routed_nvfp4_input``
(``megatron/core/inference/moe/flashinfer_nvfp4.py``), which quantizes every per-token NVFP4 MoE
input.

Contract (from the NVFP4 definition, the call site and FlashInfer's documented API; written
before reading the kernel):

* Input BF16 ``x [T, H]`` (any strides); ``padded >= H`` (``padded < H`` raises ValueError); the
  row is zero-padded to ``padded`` columns. Rows are independent.
* Per token: decode scale ``s_t = amax_t / 2688`` (``2688 = 448 * 6``; Megatron passes
  ``a_global_sf = 1 / 2688``), returned in FP32 ``[T]`` (0 for an all-zero row). Block of 16:
  scale ``s_b = E4M3(amax_b / (6 s_t)) = E4M3(448 amax_b / amax_t)`` (nearest even); codes
  ``E2M1(x / (s_b s_t))`` (nearest even, saturating at 6, the sign of zero kept). A block whose
  nonzero amax rounds to a zero scale decodes to 0 whatever its codes; an all-zero block has
  zero codes.
* Output: codes two per byte ``[T, padded / 2]`` (uint8, the even element in the low nibble),
  E4M3 scale bytes ``[T, padded / 16]`` in the linear layout (row-major), ``s_t`` ``[T]`` FP32.
  Padding columns are zero codes and zero scales. Inputs are read only.
* Non-finite input must stay visible: a NaN element decodes to NaN, and a row holding an
  infinity (whose amax, hence ``s_t``, is infinite) decodes the infinity to NaN or to an
  infinity of its sign; other rows are unaffected.

Oracle: ``spec.nvfp4_quantize_spec`` with the exact per-token encode ``2688 / amax_t``;
tolerance class bit-exact against the spec, codes and scales reported apart. The FP32 ``s_t``
must be an FP32 evaluation of ``amax_t / 2688`` with at most four roundings
(``spec.assert_fp32_evaluation``; FlashInfer computes ``amax * fp32(1/2688)`` by default and
``1 / fp32(2688 / amax)`` in IEEE mode, up to 1.24 ulp off on BF16 amaxes). As for the weights
(``test_nvfp4_quantize_expert_weights.py``), the random rows are redrawn until every block value
and scaled element is ``spec.NVFP4_MARGIN`` from its rounding boundary, and round-half-to-even
is tested at exact ties with power-of-two per-token and block scales.

Structure (FlashInfer 0.7.2.post1 ``nvfp4QuantAndPerTokenScaleKernel``, read afterwards): one
128-thread block per row (grid = T), each thread 16 elements per step over ``padded / 16``
vectors; the row amax is a block reduction; compiled with fast math (``rcp.approx.ftz``,
default) and without (``FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH=1``, read at every launch); the
binding requires a contiguous input. Inputs cover both variants, ``padded / 16`` below, at and
above the 128 threads (1792, 2048, 2304), one vector per thread or fewer (256), H around the
256 padding (255, 256, 257, 2047, 2049), rows 1, 2, 127/128/129, 1000 and production (4096 x
2688, 8192 x 7168), row magnitudes from 2**-40 to 2**40, zero rows, signed zeros, BF16
subnormals and blocks whose scale is an E4M3 subnormal or zero, non-contiguous inputs, and a
prefix of a NaN-filled buffer (sanitizer). Not covered: row amaxes below about 2**-60 or above
2**60 (FP32 flush-to-zero and overflow of the intermediate scales; hidden states never get
there), and int32 offsets: the kernel's vector offsets are 32-bit, which only overflow past
2**36 elements of input. Hardware: Blackwell (SM100-only conversion).

Finding O4b-M1 (suspected from the source, FlashInfer upstream; strict xfails until the GB200 run
confirms them): the row and block maxima ignore NaN (``__hmax2``, ``fmaxf``) and E2M1 has no NaN,
so a NaN element is quantized to a finite code under a finite scale; in IEEE mode an infinite row
amax falls back to ``s_t = 1`` (``rowEncodeScale == 0 ? 1``), so an infinity decodes to a finite
``+-2688``.
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.flashinfer_moe import spec

ALIGNMENT = spec.NVFP4_MOE_ALIGNMENT
DEVICE = "cuda"

FLASHINFER_IN_CI = "0.7.2.post1"
XFAIL_O4B_M1 = ou.xfail_upstream(
    "O4b-M1",
    "flashinfer-python",
    fixed_after=FLASHINFER_IN_CI,
    reason="per-token nvfp4_quantize ignores NaN in its row and block maxima and saturates it to"
    " a finite E2M1 code; in IEEE mode an infinite row amax falls back to a per-token scale of 1"
    " (suspected from the source, to be confirmed on GB200)",
    raises=AssertionError,
)


def _setup(monkeypatch, mode="fast"):
    spec.require_flashinfer()
    spec.clear_nvfp4_recipe_env(monkeypatch)
    spec.set_fp4_fast_math(monkeypatch, mode)


def _quantize(x: torch.Tensor, padded: int):
    from megatron.core.inference.moe.flashinfer_nvfp4 import quantize_routed_nvfp4_input

    before = x.clone()
    data, scales, per_token = quantize_routed_nvfp4_input(x, padded)
    ou.assert_exact(x, before, what="input is read-only")
    rows = x.shape[0]
    assert data.dtype == torch.uint8 and tuple(data.shape) == (rows, padded // 2), data.shape
    assert scales.dtype == torch.uint8 and tuple(scales.shape) == (rows, padded // 16)
    assert per_token.dtype == torch.float32 and tuple(per_token.shape) == (rows,)
    return spec.unpack_fp4(data), scales, per_token


def _padded_input(x: torch.Tensor, padded: int) -> torch.Tensor:
    return torch.nn.functional.pad(x.double(), (0, padded - x.shape[1]))


def check_per_token(x: torch.Tensor, padded: int, what: str):
    """Quantize BF16 ``x`` through Megatron's call site and compare with the spec."""
    codes, scales, per_token = _quantize(x, padded)
    xp = _padded_input(x, padded)
    encode = spec.nvfp4_per_token_encode(xp)
    ref_codes, ref_scales, defined, _, _ = spec.nvfp4_quantize_spec(xp, encode)
    ou.assert_quantized_bit_exact(
        torch.where(defined, codes, 0),
        scales,
        torch.where(defined, ref_codes, 0),
        ref_scales,
        elems_per_scale=spec.NVFP4_BLOCK,
        what=what,
    )
    spec.assert_fp32_evaluation(
        per_token,
        xp.abs().amax(-1) / spec.NVFP4_PER_TOKEN_DIVISOR,
        roundings=spec.NVFP4_PER_TOKEN_SCALE_ROUNDINGS,
        what=f"{what}: per-token scale amax / 2688",
    )
    return codes, scales, per_token


def _tokens(rows, hidden, generator, *, row_exponents=(-40, 40), zero_rows=()):
    """BF16 tokens ``[rows, hidden]`` away from every rounding boundary (CUDA)."""
    padded = spec._ceil_to(hidden, ALIGNMENT)
    x = spec.nvfp4_spread_values(rows, padded, generator=generator, row_exponents=row_exponents)
    x[:, hidden:] = 0
    x[list(zero_rows)] = 0
    x = spec.nvfp4_away_from_ties(x.to(DEVICE), spec.nvfp4_per_token_encode, generator=generator)
    return x[:, :hidden].to(torch.bfloat16).contiguous()


def test_per_token_spec_and_inputs():
    """CPU check of the oracle: per-token NVFP4-exact rows round-trip through the spec; the
    redrawn rows keep every margin and their amaxes; both FP32 recipes for ``s_t`` (FlashInfer's
    fast and IEEE modes, evaluated in torch FP32 on the CPU) stay within the four-rounding bound
    for every BF16 amax from 2**-60 to 2**60, and give ``2**-k`` exactly for ``2688 * 2**-k``;
    tie rows have power-of-two per-token scales."""
    g = ou.make_generator()
    x = spec.nvfp4_exact_values(64, 256, generator=g, row_exponents=(-13, 13))
    encode = spec.nvfp4_per_token_encode(x)
    codes, scales, defined, _, _ = spec.nvfp4_quantize_spec(x, encode)
    assert bool(defined.all())
    ou.assert_exact(spec.nvfp4_decode(codes, scales) / encode, x, what="exact round trip")
    raw = spec.nvfp4_spread_values(64, 512, generator=g, row_exponents=(-40, 40))
    raw[5] = 0
    y = spec.nvfp4_away_from_ties(raw, spec.nvfp4_per_token_encode, generator=g)
    _, _, _, block_margin, element_margin = spec.nvfp4_quantize_spec(
        y, spec.nvfp4_per_token_encode(y)
    )
    assert bool((block_margin >= spec.NVFP4_MARGIN).all())
    assert bool((element_margin >= spec.NVFP4_MARGIN).all())
    assert torch.equal(y.abs().amax(-1), raw.abs().amax(-1)) and not bool(y[5].any())
    bits = torch.arange(1, 0x7F80, dtype=torch.int32).to(torch.int16)
    amax = bits.view(torch.bfloat16).float()
    amax = amax[(amax >= 2.0**-60) & (amax <= 2.0**60)]
    inv = torch.tensor(1.0 / spec.NVFP4_PER_TOKEN_DIVISOR, dtype=torch.float32)
    fast = amax * inv
    ieee = torch.reciprocal(torch.tensor(1.0, dtype=torch.float32) / inv / amax)
    exact = amax.double() / spec.NVFP4_PER_TOKEN_DIVISOR
    for name, value in (("fast", fast), ("ieee", ieee)):
        spec.assert_fp32_evaluation(
            value, exact, roundings=spec.NVFP4_PER_TOKEN_SCALE_ROUNDINGS, what=name
        )
    for k in (-40, -12, 0, 12, 40):
        row = spec.nvfp4_tie_blocks(k, g)
        a = row.abs().amax().float()
        assert (
            float(a * inv) == 2.0 ** -k and float(spec.nvfp4_per_token_encode(row[None])) == 2.0**k
        )


ROWS = [1, 2, 127, 128, 129, 1000]
HIDDEN = [255, 256, 257, 1792, 2047, 2048, 2049, 2304, 4096]
CASES = ou.pairwise_cases(
    {"rows": ROWS, "hidden": HIDDEN, "math": ["fast", "ieee"], "zero_rows": [False, True]}
)


def _case_id(c):
    return f"T{c['rows']}-H{c['hidden']}-{c['math']}{'-zero-rows' if c['zero_rows'] else ''}"


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("case", [pytest.param(c, id=_case_id(c)) for c in CASES])
def test_quantize_routed_nvfp4_input_matches_spec(case, monkeypatch):
    _setup(monkeypatch, case["math"])
    rows, hidden = case["rows"], case["hidden"]
    zero_rows = range(0, rows, 3) if case["zero_rows"] else ()
    x = _tokens(rows, hidden, ou.make_generator(), zero_rows=zero_rows)
    _, scales, per_token = check_per_token(x, spec._ceil_to(hidden, ALIGNMENT), _case_id(case))
    for r in zero_rows:
        assert float(per_token[r]) == 0.0 and not bool(scales[r].any()), f"zero row {r}"


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize(
    "rows,hidden", [(4096, 2688), (8192, 7168)], ids=["4096x2688", "8192x7168"]
)
def test_production_sizes(rows, hidden, monkeypatch):
    _setup(monkeypatch)
    x = _tokens(rows, hidden, ou.make_generator(), row_exponents=(-8, 8))
    check_per_token(x, spec._ceil_to(hidden, ALIGNMENT), what=f"production {rows}x{hidden}")


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("mode", ["fast", "ieee"])
def test_round_half_to_even_at_exact_ties(mode, monkeypatch):
    """Rows of ``spec.nvfp4_tie_blocks`` with per-token scales ``2**-k`` (k from -30 to 30) and
    block scales ``2**-9 .. 2**8``: 270 elements per row exactly halfway between E2M1 values must
    round to the even code (assumes ``rcp.approx.ftz`` is exact on powers of two in fast mode)."""
    _setup(monkeypatch, mode)
    g = ou.make_generator()
    rows = torch.stack([spec.nvfp4_tie_blocks(k, g) for k in range(-30, 31, 3)])
    check_per_token(rows.to(torch.bfloat16).to(DEVICE), 512, what=f"ties ({mode})")


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("mode", ["fast", "ieee"])
def test_zero_rows_signed_zeros_subnormals_and_underflowing_blocks(mode, monkeypatch):
    """An all-zero row (scale 0, zero codes and scales), a row of -0.0 (codes 0x8), BF16
    subnormal elements in normal blocks (signed zero codes), and blocks far below their row's
    amax whose scale is an E4M3 subnormal or zero."""
    _setup(monkeypatch, mode)
    g = ou.make_generator()
    x = _tokens(8, 512, g, row_exponents=(-4, 4)).double()
    x[1] = 0.0
    x[2] = -0.0
    x[3, 17] = -(2.0**-130)
    x[3, 18] = 2.0**-133
    x[4, 32:48] = x[4, 32:48] * 2.0**-24
    x[4, 48:64] = x[4, 48:64] * 2.0**-14
    x = spec.nvfp4_away_from_ties(x, spec.nvfp4_per_token_encode, generator=g)
    codes, scales, per_token = check_per_token(x.to(torch.bfloat16), 512, what=f"zeros ({mode})")
    assert float(per_token[1]) == 0.0 and float(per_token[2]) == 0.0
    assert bool((codes[2] == 8).all()) and not bool(codes[1].any())
    assert int(scales[4, 2]) == 0, "the 2**-24 block's scale underflows"


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("layout", ["transpose", "column-slice", "row-slice"])
def test_non_contiguous_input(layout, monkeypatch):
    """Megatron makes the input contiguous (FlashInfer's binding requires it) when no padding
    copy does; H is a multiple of 256 here, so nothing else would."""
    _setup(monkeypatch)
    g = ou.make_generator()
    base = _tokens(512, 512, g, row_exponents=(-4, 4))
    if layout == "transpose":
        x = base[:, :256].t()  # [256, 512] with strides (1, 512)
    elif layout == "column-slice":
        x = base[:256, 256:]  # [256, 256], row stride 512
    else:
        x = base[::2]  # [256, 512], row stride 1024
    assert not x.is_contiguous()
    xc = x.double().to(torch.bfloat16).contiguous()
    xc = spec.nvfp4_away_from_ties(xc.double(), spec.nvfp4_per_token_encode, generator=g)
    x.copy_(xc.to(torch.bfloat16))  # the view now holds margin-checked rows
    check_per_token(x, x.shape[1], what=f"{layout} input")


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
def test_padding_wider_than_one_alignment(monkeypatch):
    """Explicit padded widths beyond the next 256 (zero blocks)."""
    _setup(monkeypatch)
    x = _tokens(33, 320, ou.make_generator(), row_exponents=(-4, 4))
    for padded in (320, 512, 1024):
        check_per_token(x, padded, what=f"padded to {padded}")


def test_padded_width_smaller_than_hidden_raises():
    """The width check runs before any kernel (CPU)."""
    from megatron.core.inference.moe.flashinfer_nvfp4 import quantize_routed_nvfp4_input

    with pytest.raises(ValueError, match="smaller than input hidden size"):
        quantize_routed_nvfp4_input(torch.zeros(4, 320, dtype=torch.bfloat16), 256)


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.sanitizer
def test_input_is_a_prefix_of_a_nan_filled_buffer(monkeypatch):
    """The bounded call passes ``hidden[:rows]`` of a larger buffer: rows past it are NaN here
    and must not leak; H 320 takes Megatron's padding path."""
    _setup(monkeypatch)
    buffer = torch.full((300, 320), float("nan"), dtype=torch.bfloat16, device="cuda")
    buffer[:129] = _tokens(129, 320, ou.make_generator(), row_exponents=(-8, 8))
    check_per_token(buffer[:129], 512, what="prefix of a NaN buffer")


def _decoded(codes, scales, per_token):
    return spec.nvfp4_decode(codes.cpu(), scales.cpu()) * per_token.double().cpu()[:, None]


def _other_rows_exact(x, row, padded, what):
    keep = [r for r in range(x.shape[0]) if r != row]
    check_per_token(x[keep], padded, what=f"rows other than the {what} row")


@XFAIL_O4B_M1
@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize("mode", ["fast", "ieee"])
def test_nan_element_stays_visible(mode, monkeypatch):
    """A NaN element must decode to NaN (an E4M3 NaN scale, or a NaN per-token scale): E2M1 has
    no NaN, so the scale has to carry it. Suspected O4b-M1: FlashInfer's maxima ignore NaN and
    the NaN element gets a finite code under a finite scale."""
    _setup(monkeypatch, mode)
    x = _tokens(4, 256, ou.make_generator(), row_exponents=(-4, 4))
    x[1, 37] = float("nan")
    _other_rows_exact(x, 1, 256, "NaN")
    value = _decoded(*_quantize(x, 256))[1, 37]
    assert torch.isnan(value), f"a NaN input element decoded to the finite value {float(value)!r}"


@pytest.mark.launch_on_gb200
@ou.requires_sm(ou.BLACKWELL)
@pytest.mark.parametrize(
    "mode,sign",
    [
        pytest.param("fast", 1.0, id="fast-+inf"),
        pytest.param("fast", -1.0, id="fast--inf"),
        pytest.param("ieee", 1.0, id="ieee-+inf", marks=XFAIL_O4B_M1),
        pytest.param("ieee", -1.0, id="ieee--inf", marks=XFAIL_O4B_M1),
    ],
)
def test_infinite_element_stays_visible(mode, sign, monkeypatch):
    """A row holding an infinity has an infinite amax, so no finite per-token scale represents
    it: the infinity must decode to NaN or to an infinity of its sign; other rows stay exact.
    In IEEE mode FlashInfer falls back to a per-token scale of 1 (suspected O4b-M1)."""
    _setup(monkeypatch, mode)
    x = _tokens(4, 256, ou.make_generator(), row_exponents=(-4, 4))
    x[2, 70] = sign * float("inf")
    _other_rows_exact(x, 2, 256, "infinite")
    value = float(_decoded(*_quantize(x, 256))[2, 70])
    assert value != value or value == sign * float(
        "inf"
    ), f"{sign * float('inf')} decoded to the finite value {value!r}"
