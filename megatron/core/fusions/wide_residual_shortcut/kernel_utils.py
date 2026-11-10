# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared tensor contracts, numerical primitives and storage utilities."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, NVMMASharedLayout
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

# Training shapes (rows, hidden) admitted for the automatic MXFP8 handoff.
WIDTHS = tuple(range(2048, 16385, 1024))
SHAPES = tuple(
    [(m, h) for m in range(2048, 16385, 1024) for h in WIDTHS]
    + [(m, h) for m in range(17408, 65537, 1024) for h in WIDTHS[:3]]
)
# Nano/Super/Ultra (rows, hidden) with tuned backward knobs.
FLAGSHIPS = ((16384, 3072), (8192, 4096), (8192, 10240))


@dataclass(frozen=True)
class SavedWrite:
    """Detached write inputs and reader statistics needed by replay and backward.

    ``coefficients`` is [write, retain, read0, read1, their derivative terms] x streams.
    ``has_retention`` False is the static identity retention, never inferred from values.
    ``update_count`` counts the branch updates whose BF16 sum is ``update``.
    """

    update: torch.Tensor
    coefficients: torch.Tensor
    gammas: tuple
    rstd: tuple
    controller_sizes: tuple
    num_streams: int = 3
    has_retention: bool = True
    update_count: int = 1

    @property
    def reader_count(self):
        """Number of readers produced by this write."""
        return len(self.gammas)


def validate_num_streams(num_streams):
    """Validate the residual stream count (1-6)."""
    if type(num_streams) is not int or not 1 <= num_streams <= 6:
        raise ValueError("num_streams must be an integer from 1 through 6")


def qualified_device(properties):
    """The measured GB300 residency is not portable to every SM103 device."""
    return (
        "GB300" in properties.name
        and properties.major == 10
        and properties.minor == 3
        and properties.multi_processor_count == 152
    )


def _tensor(t, shape, dtype, device, name):
    if (
        not isinstance(t, torch.Tensor)
        or tuple(t.shape) != tuple(shape)
        or t.dtype != dtype
        or t.device != device
        or not t.is_contiguous()
    ):
        raise ValueError(f"{name} must be contiguous {dtype} {tuple(shape)} on {device}")


def allocate_operand(shape, device, bf16=False, *, columnwise=False):
    """Allocate a BF16 reader operand or an MXFP8 one of the requested TE orientation."""
    if bf16:
        return torch.empty(shape, dtype=torch.bfloat16, device=device)
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    from transformer_engine_torch import DType  # constants does not re-export it in every TE

    q = MXFP8Quantizer(DType.kFloat8E4M3, rowwise=not columnwise, columnwise=columnwise)
    q.optimize_for_gemm = True
    return q.make_empty(shape, dtype=torch.bfloat16, device=device)


def operand_buffers(operands, *, columnwise=True):
    """``(data, scales)`` of BF16 (scales None), MXFP8 and absent operands."""
    axis = "columnwise" if columnwise else "rowwise"
    pairs = []
    for operand in operands:
        pair = (operand, None)
        if operand is not None and type(operand) is not torch.Tensor:
            pair = (getattr(operand, f"_{axis}_data"), getattr(operand, f"_{axis}_scale_inv"))
        pairs.append(pair)
    return tuple(zip(*pairs)) if pairs else ((), ())


def _descriptor(tensor, tile_shape):
    if tensor is None:
        return None
    bits = tensor.element_size() * 8
    layout = NVMMASharedLayout(swizzle_byte_width=bits * 4, element_bitwidth=bits, rank=2)
    return TensorDescriptor.from_tensor(tensor, list(tile_shape), layout)


@g.jit
def _coefficient(table, INDEX: gl.constexpr):
    """One coefficient-table entry (registers or shared memory) as a scalar."""
    indices = gl.full(
        (1,), INDEX, gl.int32, gl.constexpr(BlockedLayout([1], [32], [gl.num_warps()], [0]))
    )
    return gl.sum(table.gather(indices, 0), 0)


@g.jit
def _emit_mxfp8(y, data, scale, coordinates, rows=None, AXIS: gl.constexpr = 1):
    """Store MXFP8 data and TE-swizzled E8M0 scales of 32-value blocks along ``AXIS``.

    ``rows`` zeroes padded-row scales.

    Inline PTX: Gluon has no E8M0 conversion. ``cvt.rp.satfinite.ue8m0x2.f32`` produces the
    scale exponent (rounded toward +inf, finite values saturated to 254, NaN kept as 255) in one
    instruction; the scales match TE's MXFP8 quantizer bit for bit (tested against it).
    """
    exponent = (
        gl.inline_asm_elementwise(
            "cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;",
            "=h,f",
            [gl.max(gl.abs(y), AXIS) * (1.0 / 448.0)],
            dtype=gl.uint16,
            is_pure=True,
            pack=1,
        ).to(gl.uint32)
        & 255
    )
    # TE's E8M0 exponent; 254 has a subnormal reciprocal and 255 preserves NaN encoding.
    bits = gl.where(exponent == 254, 0x00400000, (254 - exponent) << 23)
    bits = gl.where(exponent == 255, 0x7FFFFFFF, bits)
    outer, inner, width = coordinates
    # Left-associated, so constant per-thread offsets fold into store immediates.
    index = ((outer // 128) * gl.cdiv(width, 128) + inner // 4) * 512 + (outer % 32) * 16
    index = index + (outer % 128 // 32) * 4 + inner % 4
    scale_pointer, scale_mask = scale
    data_pointer, data_mask = data
    if rows is not None:
        for offset in gl.static_range(32, 128, 32):
            padding = (
                (outer // 128 == (rows - 1) // 128) & (outer % 128 < 32) & (outer + offset >= rows)
            )
            gl.store(scale_pointer + index + offset // 32 * 4, 0, scale_mask & padding)
    q = y * gl.expand_dims(bits.to(gl.float32, bitcast=True), AXIS)
    q = q.to(gl.float8e4nv).to(gl.uint8, bitcast=True)
    gl.store(data_pointer, gl.reshape(q, data_pointer.shape) if AXIS == 1 else q, data_mask)
    gl.store(scale_pointer + index, exponent.to(gl.uint8), scale_mask)


# Packed FP32 (f32x2) arithmetic for the backward kernel. Inline PTX: Blackwell's
# fma.rn.f32x2 / mul.rn.f32x2 issue two FP32 operations per instruction, but Gluon's gl.fma and *
# leave most of them unpacked (2080 -> 426 packed FMAs in the backward) and, under the backward's
# FP contraction, round some products differently; measured ~5% slower and not bit-identical.


@g.jit
def _fma2(a, b, c):
    """Two independent RN FP32 FMAs ``a * b + c`` as one packed f32x2 instruction."""
    return gl.inline_asm_elementwise(
        """
        {
            .reg .b64 a,b,c,r;
            mov.b64 a,{$2,$3};
            mov.b64 b,{$4,$5};
            mov.b64 c,{$6,$7};
            fma.rn.f32x2 r,a,b,c;
            mov.b64 {$0,$1},r;
        }
        """,
        "=f,=f,f,f,f,f,f,f",
        [a, b, c],
        dtype=gl.float32,
        is_pure=True,
        pack=2,
    )


@g.jit
def _mul2(a, b):
    """Two independent RN FP32 products as one packed f32x2 instruction."""
    return gl.inline_asm_elementwise(
        """
        {
            .reg .b64 a,b,r;
            mov.b64 a,{$2,$3};
            mov.b64 b,{$4,$5};
            mul.rn.f32x2 r,a,b;
            mov.b64 {$0,$1},r;
        }
        """,
        "=f,=f,f,f,f,f",
        [a, b],
        dtype=gl.float32,
        is_pure=True,
        pack=2,
    )


# Uncontracted IEEE FP32 multiply/add for replay. Inline PTX: replay must reproduce the
# forward's FP32 rounding exactly, but the compiler still contracts a plain ``*`` / ``+`` into an
# FMA even with enable_fp_fusion=False (measured: replay no longer bit-identical), and
# libdevice.mul_rn flushes subnormals under reflect-FTZ.


@g.jit
def _mul_rn(a, b):
    """``a * b`` rounded to nearest, never fused into an FMA."""
    return gl.inline_asm_elementwise(
        "mul.rn.f32 $0,$1,$2;", "=f,f,f", [a, b], dtype=gl.float32, is_pure=True, pack=1
    )


@g.jit
def _add_rn(a, b):
    """``a + b`` rounded to nearest, never fused into an FMA."""
    return gl.inline_asm_elementwise(
        "add.rn.f32 $0,$1,$2;", "=f,f,f", [a, b], dtype=gl.float32, is_pure=True, pack=1
    )


@g.jit
def _forward_value(previous, update, write, retain, rows, stream):
    """The forward's residual write with its explicit fused/separate FP32 rounding."""
    retained = _mul_rn(retain, previous)
    separate = _add_rn(_mul_rn(write, update), retained)
    contracted = (stream == 0) | ((stream == 1) & (rows % 32 < 12))
    return gl.where(contracted, gl.fma(write, update, retained), separate)


@triton.jit
def _reduce_jobs(JOBS):
    """Reduce disjoint matrix tiles in one launch, preserving each axis's sum tree."""
    for job in tl.static_range(len(JOBS)):
        if tl.program_id(0) >= JOBS[job][3][6] and tl.program_id(0) < JOBS[job][3][7]:
            source, output, factor = JOBS[job][:3]
            spec: tl.constexpr = JOBS[job][3]
            width: tl.constexpr = spec[0]
            height: tl.constexpr = spec[1]
            tile: tl.constexpr = spec[2]
            chunk: tl.constexpr = spec[3]
            axis: tl.constexpr = spec[4]
            size: tl.constexpr = spec[5]
            chunks: tl.constexpr = (height + chunk - 1) // chunk
            tiles: tl.constexpr = (width + tile - 1) // tile
            bid = tl.program_id(0) - spec[6]
            rows = bid % tiles * tile + tl.arange(0, tile)
            cols = bid // tiles * chunk + tl.arange(0, chunk)
            i, j = (cols[:, None], rows[None, :]) if axis == 0 else (rows[:, None], cols[None, :])
            h, w = (height, width) if axis == 0 else (width, height)
            total = tl.sum(tl.load(source + i * w + j, (i < h) & (j < w), 0), axis)
            if JOBS[job][4] is not None:
                if len(JOBS[job][4]) == 3:
                    # Read controller: t * a + w * b.
                    b_source, writes, retains = JOBS[job][4]
                    b_total = tl.sum(tl.load(b_source + cols, cols < height, 0), 0)
                    w = tl.load(writes + rows, rows < width, 0)
                    t = tl.load(retains + rows, rows < width, 0) if retains is not None else 1.0
                    total = t * total + w * b_total
                else:
                    # Write (raw b rows, 5 entries) or retention (a planes) controller: plus
                    # each reader's beta * (its b or a).
                    for k in tl.static_range(2):
                        extra = JOBS[job][4][2 * k]
                        if extra is not None:
                            if len(JOBS[job][4]) == 5:
                                part = tl.sum(tl.load(extra + cols, cols < height, 0), 0)
                            else:
                                part = tl.sum(
                                    tl.load(extra + i * w + j, (i < h) & (j < w), 0), axis
                                )
                            total += tl.load(JOBS[job][4][2 * k + 1] + rows, rows < width, 0) * part
            if factor is not None:
                total *= tl.load(factor + rows, rows < width, 0)
            tl.store(output + rows * chunks + bid // tiles, total, rows < width)
            if size > width and bid == 0:
                padding = tl.arange(0, triton.next_power_of_2(size))
                tl.store(output + padding, 0, (padding >= width) & (padding < size))


def reduce_jobs(jobs):
    """Pack ``(source, output, axis, tile, chunk, factor[, combo])`` reductions into one launch."""
    packed, grid = [], 0
    for job in jobs:
        source, output, axis, tile, chunk, factor = job[:6]
        combo = job[6] if len(job) == 7 else None
        width, height = source.shape[1 - axis], source.shape[axis]
        chunk = chunk or triton.next_power_of_2(height)
        size = output.numel() if output.ndim == 1 else width
        end = grid + triton.cdiv(width, tile) * triton.cdiv(height, chunk)
        spec = (width, height, tile, chunk, axis, size, grid, end)
        packed.append((source, output, factor, tl.constexpr(spec), combo))
        grid = end
    if packed:
        _reduce_jobs[(grid,)](tuple(packed), num_warps=4, enable_fp_fusion=False)
