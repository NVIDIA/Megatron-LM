# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MXFP8 (E4M3 elements, E8M0 scales, 32-element blocks) from the OCP specifications, for the
MXFP8 quantization oracles (not a test module). Written from the specs, not from the kernels.

OCP Microscaling Formats (MX) v1.0 (Sep 2023):

* §5.2 Table 1: MXFP8 = block size k = 32, element FP8 (E4M3 here), scale E8M0 (8 bits).
* §5.1: a block decodes to ``v_i = X * P_i``; when the scale ``X`` is NaN every ``v_i`` is NaN; an
  element that is Inf or NaN decodes to itself.
* §5.3.1 Table 2 (reproduced from the OCP 8-bit Floating Point Specification): E4M3 has exponent
  bias 7, no infinities, NaN ``S.1111.111``, max normal ``S.1111.110 = 448``, min normal
  ``S.0001.000 = 2**-6``, subnormals ``S.0000.mmm = mmm * 2**-9``; normal numbers are
  ``(-1)**S * 2**(E-7) * (1 + M/8)``. Table 3: conversions saturate (SAT: out of range -> ±448,
  NaN -> NaN) or overflow (OVF: -> NaN).
* §5.4.1 Table 7: E8M0 is an unsigned biased exponent, bias 127, exponents -127..127, NaN
  ``0xFF``, no zero and no infinities.
* §6.3: conversion must support roundTiesToEven for the elements. The scale computation is out
  of scope (§4: "Detailed algorithms for computing the block scale"); the semantics that
  "should" be supported set ``X = 2**(floor(log2 amax) - 8)`` (largest power of two <= amax over
  the largest power of two of E4M3, 256) and clamp elements above 448, and "other algorithms
  may be supported".

The Megatron quantizers use one of those other algorithms, the round-up scale of the training
recipe (Mishra et al., "Recipes for Pre-training LLMs with MXFP8", which the kernel's docstring
cites; NVIDIA's ``cvt.rp.satfinite.ue8m0x2.f32`` and Transformer Engine compute the same): the
smallest E8M0 power of two ``X`` with ``amax / X <= 448``,

    X = 2**clamp(ceil(log2(amax / 448)), -127, 127),   X = 2**-127 for an all-zero block,

so no element ever saturates. ``floor_scale_exponent`` is the §6.3 rule, for comparison.

The swizzled scale layout is cuBLAS's "1D block scaling factors layout" (also
``torch.nn.functional.SwizzleType.SWIZZLE_32_4_4``): the [rows, K/32] scale matrix, zero-padded to
multiples of 128 rows and 4 columns, is cut into 128 x 4 tiles stored row-tile-major as 512
contiguous bytes each; inside a tile, byte ``16 * (r % 32) + 4 * (r // 32) + c`` holds row ``r``,
column ``c``.
"""

from typing import Optional

import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_activations import _common

BLOCK = 32
E4M3_MAX = 448.0
E4M3_MANTISSA_BITS, E4M3_EMIN, E4M3_BIAS = 3, -6, 7
E8M0_BIAS, E8M0_EMIN, E8M0_EMAX, E8M0_NAN = 127, -127, 127, 0xFF
E4M3_NAN_MAGNITUDE = 0x7F
TILE_ROWS, TILE_COLS = 128, 4


# ---------------------------------------------------------------------------------------------
# Element and scale encodings
# ---------------------------------------------------------------------------------------------


def e4m3_round(q: torch.Tensor) -> torch.Tensor:
    """fp64 values to the nearest E4M3 value, ties to even, saturating at ±448 (SAT)."""
    v = _common.round_to_format(q.to(torch.float64), E4M3_MANTISSA_BITS, E4M3_EMIN)
    return torch.where(v.abs() > E4M3_MAX, torch.copysign(torch.full_like(v, E4M3_MAX), v), v)


def e4m3_encode(v: torch.Tensor) -> torch.Tensor:
    """E4M3 values (fp64, already representable, or NaN) to their 8-bit codes (uint8)."""
    v = v.to(torch.float64)
    sign = torch.signbit(v).to(torch.int64) << 7
    mag = v.abs()
    _, exponent = torch.frexp(mag)
    binade = exponent - 1  # floor(log2 |v|) for v != 0
    normal = mag >= 2.0**E4M3_EMIN
    e_field = torch.where(normal, binade + E4M3_BIAS, torch.zeros_like(binade))
    unit = _common.pow2(torch.where(normal, binade, E4M3_EMIN) - 3)
    m_field = (mag / unit).round().to(torch.int64) - torch.where(normal, 8, 0)
    code = sign | (e_field.to(torch.int64) << 3) | m_field
    code = torch.where(mag == 0, sign, code)
    code = torch.where(torch.isnan(v), torch.full_like(code, E4M3_NAN_MAGNITUDE), code)
    return code.to(torch.uint8)


def e4m3_decode(codes: torch.Tensor) -> torch.Tensor:
    """8-bit E4M3 codes (uint8 or float8_e4m3fn storage) to fp64 values (Table 2)."""
    c = codes.view(torch.uint8).to(torch.int64)
    sign = torch.where((c >> 7) == 1, -1.0, 1.0).to(torch.float64)
    e = (c >> 3) & 0xF
    m = (c & 0x7).to(torch.float64)
    normal = (1.0 + m / 8.0) * _common.pow2(e - E4M3_BIAS)
    subnormal = m * 2.0 ** (E4M3_EMIN - 3)
    value = sign * torch.where(e == 0, subnormal, normal)
    return torch.where((c & 0x7F) == E4M3_NAN_MAGNITUDE, float("nan"), value)


def e8m0_decode(scales: torch.Tensor) -> torch.Tensor:
    """E8M0 bytes to fp64 scales ``2**(byte - 127)``, NaN for 0xFF (Table 7)."""
    b = scales.view(torch.uint8).to(torch.int64)
    value = _common.pow2(b - E8M0_BIAS)
    return torch.where(b == E8M0_NAN, float("nan"), value)


def ceil_log2(d: torch.Tensor) -> torch.Tensor:
    """``ceil(log2(d))`` exactly, for positive finite fp64 ``d`` (int64)."""
    mantissa, exponent = torch.frexp(d)  # d = mantissa * 2**exponent, mantissa in [0.5, 1)
    return torch.where(mantissa == 0.5, exponent - 1, exponent).to(torch.int64)


def rceil_scale_exponent(amax: torch.Tensor) -> torch.Tensor:
    """The round-up scale exponent: smallest ``e`` in [-127, 127] with ``amax / 2**e <= 448``.

    ``amax / 448`` is computed in fp64, where it is exact at the powers of two that matter (amax
    a power of two times 448) and cannot cross a power of two otherwise for fp32 inputs.
    """
    amax = amax.to(torch.float64)
    positive = amax > 0
    d = torch.where(positive, amax / E4M3_MAX, torch.ones_like(amax))
    e = torch.where(positive, ceil_log2(d), torch.full_like(d, E8M0_EMIN, dtype=torch.int64))
    return e.clamp(E8M0_EMIN, E8M0_EMAX)


def floor_scale_exponent(amax: torch.Tensor) -> torch.Tensor:
    """OCP MX §6.3's suggested exponent ``floor(log2 amax) - 8`` (for comparison only)."""
    _, exponent = torch.frexp(amax.to(torch.float64))
    return (exponent - 1 - 8).clamp(E8M0_EMIN, E8M0_EMAX)


# ---------------------------------------------------------------------------------------------
# Block quantization (finite inputs)
# ---------------------------------------------------------------------------------------------


def quantize(x: torch.Tensor) -> tuple:
    """The spec MXFP8 quantization of finite ``x`` [rows, K] (K a multiple of 32).

    Returns ``(codes, scales)``: uint8 E4M3 codes [rows, K] and uint8 E8M0 scale bytes
    [rows, K/32] (row-major, unswizzled). Elements are ``RNE(x / X)`` with the round-up scale
    ``X``; ``x / X`` is exact in fp64 (X is a power of two), so this is one rounding. A negative
    value that rounds to zero keeps its sign (code 0x80).
    """
    rows, k = x.shape
    blocks = x.to(torch.float64).reshape(rows, k // BLOCK, BLOCK)
    exponent = rceil_scale_exponent(blocks.abs().amax(dim=-1))
    scale = _common.pow2(exponent)
    elements = e4m3_round(blocks / scale[..., None])
    codes = e4m3_encode(elements).reshape(rows, k)
    return codes, (exponent + E8M0_BIAS).to(torch.uint8)


def dequantize(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """§5.1 decode: ``v = X * P`` per element; NaN wherever the block scale is NaN (fp64)."""
    rows, k = codes.shape
    p = e4m3_decode(codes).reshape(rows, k // BLOCK, BLOCK)
    x = e8m0_decode(scales)[..., None]
    return (x * p).reshape(rows, k)


# ---------------------------------------------------------------------------------------------
# cuBLAS 128 x 4 scale-factor layout
# ---------------------------------------------------------------------------------------------


def padded_scale_shape(rows: int, scale_cols: int) -> tuple:
    """The scale matrix padded to whole 128 x 4 tiles."""
    return (-(-rows // TILE_ROWS) * TILE_ROWS, -(-scale_cols // TILE_COLS) * TILE_COLS)


def swizzle(scales: torch.Tensor) -> torch.Tensor:
    """[rows, C] scale bytes to the 1-D cuBLAS tile layout, padding entries zero."""
    rows, cols = scales.shape
    pr, pc = padded_scale_shape(rows, cols)
    padded = torch.zeros(pr, pc, dtype=torch.uint8, device=scales.device)
    padded[:rows, :cols] = scales.view(torch.uint8)
    # [row tile, row group of 32 (r // 32), row in group (r % 32), column tile, column]
    tiles = padded.reshape(pr // TILE_ROWS, 4, 32, pc // TILE_COLS, TILE_COLS)
    # stored as [row tile, column tile, r % 32, r // 32, column]
    return tiles.permute(0, 3, 2, 1, 4).reshape(-1).contiguous()


def unswizzle(buffer: torch.Tensor, rows: int, cols: int) -> tuple:
    """1-D tile layout to ``(scales [rows, cols], padding)``; ``padding`` holds the bytes of the
    padded positions (rows past ``rows``, columns past ``cols``), flattened."""
    pr, pc = padded_scale_shape(rows, cols)
    tiles = buffer.view(torch.uint8).reshape(pr // TILE_ROWS, pc // TILE_COLS, 32, 4, TILE_COLS)
    padded = tiles.permute(0, 3, 2, 1, 4).reshape(pr, pc)
    pad_mask = torch.ones(pr, pc, dtype=torch.bool, device=buffer.device)
    pad_mask[:rows, :cols] = False
    return padded[:rows, :cols].contiguous(), padded[pad_mask]


def scale_bytes(rows: int, k: int) -> int:
    """Size of the swizzled scale buffer for an [rows, k] tensor."""
    pr, pc = padded_scale_shape(rows, k // BLOCK)
    return pr * pc


# ---------------------------------------------------------------------------------------------
# Kernel outputs against the spec
# ---------------------------------------------------------------------------------------------


def assert_matches_spec(
    codes: torch.Tensor,
    scale_buffer: torch.Tensor,
    x: torch.Tensor,
    *,
    what: str,
    rows: Optional[torch.Tensor] = None,
) -> None:
    """A kernel's MXFP8 output against the spec quantization of ``x`` [M, K].

    ``codes`` [M, K] (E4M3 storage) and the 1-D swizzled ``scale_buffer``: the rows selected by
    the bool mask ``rows`` (default all) are bit-exact, codes and E8M0 scales reported apart;
    the scale bytes of the other rows and every padding byte of the tile layout are 0x00.
    """
    m, k = x.shape
    assert scale_buffer.numel() == scale_bytes(m, k), (
        f"{what}: scale buffer has {scale_buffer.numel()} bytes, the 128 x 4 tile layout of"
        f" [{m}, {k // BLOCK}] needs {scale_bytes(m, k)}"
    )
    rows = torch.ones(m, dtype=torch.bool) if rows is None else rows.cpu()
    ref_codes, ref_scales = quantize(x[rows.to(x.device)].cpu())
    scales, padding = unswizzle(scale_buffer.cpu(), m, k // BLOCK)
    ou.assert_quantized_bit_exact(
        codes.cpu().view(torch.uint8)[rows],
        scales[rows],
        ref_codes,
        ref_scales,
        elems_per_scale=BLOCK,
        what=what,
    )
    ou.assert_exact(
        scales[~rows], torch.zeros_like(scales[~rows]), what=f"{what}: scales of skipped rows"
    )
    ou.assert_exact(padding, torch.zeros_like(padding), what=f"{what}: tile padding bytes")


def mx_inputs(rows: int, k: int, dtype: torch.dtype, kind: str, generator: torch.Generator):
    """Quantizer inputs on the CPU (move them to the device), exactly representable in ``dtype``.

    ``randn``: standard normal times 3. ``wide``: each 32-block at its own random binade, so the
    scales span a wide range. ``boundary``: each block's amax is ``448 * 2**k`` (the round-up
    tie), or its next value up or down in ``dtype``; the other elements land on E4M3 rounding
    midpoints after scaling (normal and subnormal), or anywhere below the amax. ``zeros``:
    all-zero blocks, signed zeros, and negative values too small for the block's scale (which
    must encode as -0).
    """
    g = generator
    blocks = rows * (k // BLOCK)
    small_range = dtype == torch.float16
    if kind == "randn":
        x = torch.randn(rows, k, generator=g) * 3
    elif kind == "wide":
        lo, hi = (-14, 10) if small_range else (-100, 100)
        exps = torch.randint(lo, hi, (blocks, 1), generator=g).float()
        x = (torch.randn(blocks, BLOCK, generator=g) * torch.exp2(exps)).reshape(rows, k)
    elif kind == "boundary":
        lo, hi = (-10, 6) if small_range else (-40, 40)
        scale = torch.exp2(torch.randint(lo, hi, (blocks, 1), generator=g).double())
        # E4M3 rounding midpoints: normal (2m + 1) * 2**(e - 4), m in 8..14, e in -6..8;
        # subnormal (2m + 1) * 2**-10, m in 0..6.
        m = torch.randint(8, 15, (blocks, BLOCK), generator=g).double()
        e = torch.randint(-6, 9, (blocks, BLOCK), generator=g).double()
        ties = (2 * m + 1) * torch.exp2(e - 4)
        sub = (2 * torch.randint(0, 7, (blocks, BLOCK), generator=g).double() + 1) * 2.0**-10
        anywhere = torch.rand(blocks, BLOCK, generator=g, dtype=torch.float64) * E4M3_MAX
        pick = torch.randint(0, 3, (blocks, BLOCK), generator=g)
        q = torch.where(pick == 0, ties, torch.where(pick == 1, sub, anywhere))
        q = q.clamp(max=E4M3_MAX) * torch.where(torch.rand(blocks, BLOCK, generator=g) < 0.5, -1, 1)
        x = (q * scale).to(dtype)
        amax = (E4M3_MAX * scale).to(dtype).reshape(-1)
        step = torch.randint(-1, 2, (blocks,), generator=g)
        up = torch.nextafter(amax, torch.full_like(amax, float("inf")))
        down = torch.nextafter(amax, torch.zeros_like(amax))
        amax = torch.where(step > 0, up, torch.where(step < 0, down, amax))
        col = torch.randint(0, BLOCK, (blocks,), generator=g)
        x[torch.arange(blocks), col] = amax * torch.where(col % 2 == 0, 1, -1).to(dtype)
        return x.reshape(rows, k).contiguous()
    elif kind == "zeros":
        x = torch.randn(blocks, BLOCK, generator=g) * 100
        x[0::3] = 0.0
        x[1::3, ::2] = -0.0
        x[2::3, 1:] = -(2.0**-30)  # below E4M3's smallest subnormal at the block scale
        x = x.reshape(rows, k)
    else:
        raise ValueError(kind)
    return x.to(dtype).contiguous()


VALUE_KINDS = ("randn", "wide", "boundary", "zeros")
