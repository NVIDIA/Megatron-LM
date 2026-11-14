# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MXFP8 encoding from the OCP specifications, for the fused permute + quantize oracle (not a
test module). Written from the specifications, not from the kernels.

OCP Microscaling Formats (MX) v1.0:

* MXFP8 (Table 1): 32-element blocks along a row, E4M3 elements, one E8M0 scale per block; a
  block decodes to ``X * P_i``.
* E4M3 (Table 2 of the OCP 8-bit floating point specification): bias 7, ``S.EEEE.MMM``, max
  normal ``448 = S.1111.110``, min normal ``2**-6``, subnormals ``M * 2**-9``, NaN
  ``S.1111.111``, no infinities. Elements round to nearest, ties to even.
* E8M0: an unsigned biased exponent, ``X = 2**(byte - 127)`` for bytes 0..254, NaN for 0xFF.
* The scale algorithm is left to the implementation. The Megatron quantizers document the
  round-up scale of the MXFP8 training recipe: the smallest power of two ``X`` in E8M0's range
  with ``amax / X <= 448``, so no element saturates, and ``2**-127`` (byte 0x00) for an all-zero
  block.

Scale layout: cuBLAS's 1-D block-scaling-factor layout (``SWIZZLE_32_4_4``), which
``torch.nn.functional.scaled_grouped_mm`` reads. The ``[rows, K/32]`` scale matrix, zero-padded
to multiples of 128 rows and 4 columns, is cut into 128 x 4 tiles stored row-tile-major, 512
contiguous bytes each; inside a tile, row ``r`` and column ``c`` sit at byte
``16 * (r % 32) + 4 * (r // 32) + c``.
"""

from typing import Tuple

import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou

BLOCK = 32
E4M3_MAX = 448.0
E4M3_EMIN = -6
E4M3_MANTISSA_BITS = 3
E4M3_BIAS = 7
E4M3_NAN = 0x7F
E8M0_BIAS = 127
E8M0_EMIN, E8M0_EMAX = -127, 127
E8M0_NAN = 0xFF
TILE_ROWS, TILE_COLS = 128, 4


def e4m3_round(q: torch.Tensor) -> torch.Tensor:
    """fp64 values rounded to the E4M3 grid (ties to even), saturating at +-448.

    The quantum of a value's binade is ``2**(floor(log2|q|) - 3)`` for normal values and
    ``2**-9`` below ``2**-6``; ``q / quantum`` is exact, so ``torch.round`` (half to even) is the
    single rounding. A negative value that rounds to zero keeps its sign.
    """
    q = q.to(torch.float64)
    _, exponent = torch.frexp(q.abs())  # |q| = m * 2**exponent, m in [0.5, 1)
    binade = (exponent - 1).clamp(min=E4M3_EMIN)
    quantum = ou.pow2(binade - E4M3_MANTISSA_BITS)
    v = torch.round(q / quantum) * quantum
    return torch.where(v.abs() > E4M3_MAX, torch.copysign(torch.full_like(v, E4M3_MAX), v), v)


def e4m3_encode(v: torch.Tensor) -> torch.Tensor:
    """E4M3 values (fp64, on the grid, or NaN) to their 8-bit codes (uint8)."""
    v = v.to(torch.float64)
    sign = torch.signbit(v).to(torch.int64) << 7
    mag = v.abs()
    _, exponent = torch.frexp(mag)
    binade = exponent - 1
    normal = mag >= 2.0**E4M3_EMIN
    e_field = torch.where(normal, binade + E4M3_BIAS, torch.zeros_like(binade))
    quantum = ou.pow2(torch.where(normal, binade, torch.full_like(binade, E4M3_EMIN)) - 3)
    m_field = (mag / quantum).round().to(torch.int64) - torch.where(normal, 8, 0)
    code = sign | (e_field.to(torch.int64) << 3) | m_field
    code = torch.where(mag == 0, sign, code)
    code = torch.where(torch.isnan(v), torch.full_like(code, E4M3_NAN), code)
    return code.to(torch.uint8)


def e4m3_decode(codes: torch.Tensor) -> torch.Tensor:
    """8-bit E4M3 codes (uint8 or float8_e4m3fn storage) to fp64 values."""
    c = codes.view(torch.uint8).to(torch.int64)
    sign = torch.where((c >> 7) == 1, -1.0, 1.0).to(torch.float64)
    e = (c >> 3) & 0xF
    m = (c & 0x7).to(torch.float64)
    normal = (1.0 + m / 8.0) * ou.pow2(e - E4M3_BIAS)
    subnormal = m * 2.0 ** (E4M3_EMIN - 3)
    value = sign * torch.where(e == 0, subnormal, normal)
    return torch.where((c & 0x7F) == E4M3_NAN, float("nan"), value)


def e8m0_decode(scales: torch.Tensor) -> torch.Tensor:
    """E8M0 bytes to fp64 scales ``2**(byte - 127)``; NaN for 0xFF."""
    b = scales.view(torch.uint8).to(torch.int64)
    return torch.where(b == E8M0_NAN, float("nan"), ou.pow2(b - E8M0_BIAS))


def scale_exponent(amax: torch.Tensor) -> torch.Tensor:
    """Smallest ``e`` in [-127, 127] with ``amax / 2**e <= 448`` (int64); -127 for ``amax = 0``.

    ``amax / 448`` in fp64 is a power of two exactly when ``amax`` is 448 times one, and cannot
    round onto a power of two otherwise for bf16 / fp32 ``amax``.
    """
    amax = amax.to(torch.float64)
    positive = amax > 0
    d = torch.where(positive, amax / E4M3_MAX, torch.ones_like(amax))
    mantissa, exponent = torch.frexp(d)  # d = mantissa * 2**exponent, mantissa in [0.5, 1)
    ceil_log2 = torch.where(mantissa == 0.5, exponent - 1, exponent).to(torch.int64)
    e = torch.where(positive, ceil_log2, torch.full_like(ceil_log2, E8M0_EMIN))
    return e.clamp(E8M0_EMIN, E8M0_EMAX)


def quantize_rows(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The MXFP8 encoding of finite ``x`` ``[n, K]`` (K a multiple of 32), on ``x``'s device.

    Returns ``(codes, scales)``: uint8 E4M3 codes ``[n, K]`` and uint8 E8M0 bytes ``[n, K/32]``
    (unswizzled). Elements are ``RNE(x / X)``; ``x / X`` is exact (X a power of two).
    """
    n, k = x.shape
    blocks = x.to(torch.float64).reshape(n, k // BLOCK, BLOCK)
    exponent = scale_exponent(blocks.abs().amax(dim=-1))
    elements = e4m3_round(blocks / ou.pow2(exponent)[..., None])
    return e4m3_encode(elements).reshape(n, k), (exponent + E8M0_BIAS).to(torch.uint8)


def dequantize_rows(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """``X * P`` per element (fp64); NaN wherever the code or the block scale is NaN."""
    n, k = codes.shape
    p = e4m3_decode(codes).reshape(n, k // BLOCK, BLOCK)
    return (e8m0_decode(scales)[..., None] * p).reshape(n, k)


def padded_scale_shape(rows: int, scale_cols: int) -> Tuple[int, int]:
    """The ``[rows, scale_cols]`` scale matrix padded to whole 128 x 4 tiles."""
    return -(-rows // TILE_ROWS) * TILE_ROWS, -(-scale_cols // TILE_COLS) * TILE_COLS


def scale_buffer_bytes(rows: int, k: int) -> int:
    """Bytes of the swizzled scale buffer of an ``[rows, k]`` MXFP8 tensor."""
    pr, pc = padded_scale_shape(rows, k // BLOCK)
    return pr * pc


def unswizzle(buffer: torch.Tensor, rows: int, cols: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """The 1-D tile layout back to ``(scales [rows, cols], padding)`` as uint8; ``padding``
    holds the bytes of the padded positions (rows past ``rows``, columns past ``cols``)."""
    pr, pc = padded_scale_shape(rows, cols)
    # Stored as [row tile, column tile, r % 32, (r // 32) % 4, c % 4].
    tiles = buffer.view(torch.uint8).reshape(pr // TILE_ROWS, pc // TILE_COLS, 32, 4, TILE_COLS)
    padded = tiles.permute(0, 3, 2, 1, 4).reshape(pr, pc)
    pad = torch.ones(pr, pc, dtype=torch.bool, device=buffer.device)
    pad[:rows, :cols] = False
    return padded[:rows, :cols].contiguous(), padded[pad]


def value_kind_inputs(rows: int, k: int, kind: str, generator: torch.Generator) -> torch.Tensor:
    """bf16 ``[rows, k]`` on the CPU, exactly representable, for the scale and rounding cases.

    * ``randn``: standard normal times 3.
    * ``wide``: each 32-block at its own binade in ``[2**-100, 2**100]``, so scales span E8M0.
    * ``boundary``: each block's amax is ``448 * 2**j`` (where the round-up scale changes) or the
      neighbouring bf16 value above or below; other elements land on E4M3 rounding midpoints
      (normal and subnormal) after scaling, or anywhere below the amax.
    * ``zeros``: all-zero blocks, blocks of signed zeros, and negative values too small for the
      block's scale, which must encode as -0 (code 0x80).
    """
    g = generator
    blocks = rows * (k // BLOCK)
    if kind == "randn":
        x = torch.randn(rows, k, generator=g) * 3
    elif kind == "wide":
        exps = torch.randint(-100, 100, (blocks, 1), generator=g).double()
        x = (torch.randn(blocks, BLOCK, generator=g, dtype=torch.float64) * ou.pow2(exps)).reshape(
            rows, k
        )
    elif kind == "boundary":
        scale = ou.pow2(torch.randint(-40, 40, (blocks, 1), generator=g))
        m = torch.randint(8, 15, (blocks, BLOCK), generator=g).double()
        e = torch.randint(-6, 8, (blocks, BLOCK), generator=g)
        ties = (2 * m + 1) * ou.pow2(e - 4)  # midway between two E4M3 normals (bf16-exact)
        sub = (2 * torch.randint(0, 7, (blocks, BLOCK), generator=g).double() + 1) * 2.0**-10
        anywhere = torch.rand(blocks, BLOCK, generator=g, dtype=torch.float64) * E4M3_MAX
        pick = torch.randint(0, 3, (blocks, BLOCK), generator=g)
        q = torch.where(pick == 0, ties, torch.where(pick == 1, sub, anywhere))
        sign = torch.where(torch.rand(blocks, BLOCK, generator=g) < 0.5, -1.0, 1.0)
        x = (q.clamp(max=E4M3_MAX) * sign * scale).to(torch.bfloat16)
        amax = (E4M3_MAX * scale).to(torch.bfloat16).reshape(-1)
        step = torch.randint(-1, 2, (blocks,), generator=g)
        up = torch.nextafter(amax, torch.full_like(amax, float("inf")))
        down = torch.nextafter(amax, torch.zeros_like(amax))
        amax = torch.where(step > 0, up, torch.where(step < 0, down, amax))
        col = torch.randint(0, BLOCK, (blocks,), generator=g)
        x[torch.arange(blocks), col] = amax * torch.where(col % 2 == 0, 1, -1).to(torch.bfloat16)
        return x.reshape(rows, k).contiguous()
    elif kind == "zeros":
        x = torch.randn(blocks, BLOCK, generator=g) * 100
        x[0::3] = 0.0
        x[1::3, ::2] = -0.0
        x[2::3, 1:] = -(2.0**-30)  # below E4M3's smallest subnormal at the block's scale
        x = x.reshape(rows, k)
    else:
        raise ValueError(kind)
    return x.to(torch.bfloat16).contiguous()


VALUE_KINDS = ("randn", "wide", "boundary", "zeros")
