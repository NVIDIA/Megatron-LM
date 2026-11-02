# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared tensor contracts, numerical primitives, and storage utilities.

Algorithm stages belong in forward_kernel or backward_kernel. This module
provides common validation, encoding, reductions, and packed arithmetic.
SavedWrite contains detached update/controller/norm state, never residual storage.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import NVMMASharedLayout
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

# Shared domain and tensor contracts.


WIDTHS = tuple(range(2048, 16385, 1024))


SHAPES = tuple(
    [(m, h) for m in range(2048, 16385, 1024) for h in WIDTHS]
    + [(m, h) for m in range(17408, 65537, 1024) for h in WIDTHS[:3]]
)


FLAGSHIPS = ((16384, 3072), (8192, 4096), (8192, 10240))


@dataclass(frozen=True)
class SavedWrite:
    """Original write inputs and reader statistics needed by replay and backward."""

    update: torch.Tensor
    coefficients: torch.Tensor  # [write, retain, read0, read1, derivatives] x num_streams
    gammas: tuple
    rstd: tuple
    controller_sizes: tuple
    num_streams: int = 3
    # None is a structural mode, never inferred from a learned coefficient value.
    has_retention: bool = True
    # Branch updates summed into ``update`` (two for a shortcut add); None means one per reader.
    num_updates: int | None = None

    @property
    def reader_count(self):
        """Return the number of readers produced by this write."""
        return len(self.gammas)

    @property
    def update_count(self):
        """Return the number of branch updates whose (BF16) sum is ``update``."""
        return self.reader_count if self.num_updates is None else self.num_updates


def validate_num_streams(num_streams):
    """Validate and return the supported residual stream count."""
    if type(num_streams) is not int or not 1 <= num_streams <= 6:
        raise ValueError("num_streams must be an integer from 1 through 6")
    return num_streams


def qualified_device(properties):
    """The measured GB300 residency is not portable to every SM103 device."""
    return (
        "GB300" in properties.name
        and properties.major == 10
        and properties.minor == 3
        and properties.multi_processor_count == 152
    )


# Shared MXFP8 stores.


@g.jit
def _e8m0(amax):
    """Return TE's exact E8M0 scale for a block amax and its FP32 reciprocal's bits."""
    exponent = (
        gl.inline_asm_elementwise(
            """
            cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1; // Round the scale upward into E8M0.
            """,
            "=h,f",
            [amax * (1.0 / 448.0)],
            dtype=gl.uint16,
            is_pure=True,
            pack=1,
        ).to(gl.uint32)
        & 255
    )
    # Exponent 254 has a subnormal reciprocal; 255 preserves NaN encoding.
    bits = gl.where(exponent == 254, 0x00400000, (254 - exponent) << 23)
    bits = gl.where(exponent == 255, 0x7FFFFFFF, bits)
    return exponent, bits


@g.jit
def _emit_mxfp8(y, data, scale, coordinates, rows=None):
    """Store either MXFP8 orientation with TE's exact E8M0 scale and FP32 reciprocal."""
    exponent, bits = _e8m0(gl.max(gl.abs(y), 1))
    outer, inner, width = coordinates
    index = ((outer // 128) * gl.cdiv(width, 128) + inner // 4) * 512
    index += (outer % 32) * 16 + (outer % 128 // 32) * 4 + inner % 4
    scale_pointer, scale_mask = scale
    data_pointer, data_mask = data
    if rows is not None:
        for offset in gl.static_range(32, 128, 32):
            padding = (
                (outer // 128 == (rows - 1) // 128) & (outer % 128 < 32) & (outer + offset >= rows)
            )
            gl.store(scale_pointer + index + offset // 32 * 4, 0, scale_mask & padding)
    q = y * gl.expand_dims(bits.to(gl.float32, bitcast=True), 1)
    q = q.to(gl.float8e4nv).to(gl.uint8, bitcast=True)
    gl.store(data_pointer, gl.reshape(q, data_pointer.shape), data_mask)
    gl.store(scale_pointer + index, exponent.to(gl.uint8), scale_mask)


# Shared operand allocation and generic reductions.


def allocate_operand(shape, device, bf16=False, *, columnwise=False):
    """Allocate a normalized reader operand with the requested TE orientation."""
    if bf16:
        return torch.empty(shape, dtype=torch.bfloat16, device=device)
    from transformer_engine.pytorch.constants import DType
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    q = MXFP8Quantizer(DType.kFloat8E4M3, rowwise=not columnwise, columnwise=columnwise)
    q.optimize_for_gemm = True
    return q.make_empty(shape, dtype=torch.bfloat16, device=device)


def operand_buffers(operands, *, columnwise=True):
    """Expose data and scales for BF16, MXFP8 and absent operands."""
    axis = "columnwise" if columnwise else "rowwise"
    pairs = []
    for operand in operands:
        pair = (operand, None)
        if operand is not None and type(operand) is not torch.Tensor:
            pair = (getattr(operand, f"_{axis}_data"), getattr(operand, f"_{axis}_scale_inv"))
        pairs.append(pair)
    return tuple(zip(*pairs)) if pairs else ((), ())


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
                    b_source, writes, retains = JOBS[job][4]
                    b_total = tl.sum(tl.load(b_source + cols, cols < height, 0), 0)
                    w = tl.load(writes + rows, rows < width, 0)
                    t = tl.load(retains + rows, rows < width, 0) if retains is not None else 1.0
                    total = t * total + w * b_total
                elif len(JOBS[job][4]) == 5:
                    b0, beta0, b1, beta1, raw_write = JOBS[job][4]
                    b0_total = tl.sum(tl.load(b0 + cols, cols < height, 0), 0)
                    total += tl.load(beta0 + rows, rows < width, 0) * b0_total
                    if b1 is not None:
                        b1_total = tl.sum(tl.load(b1 + cols, cols < height, 0), 0)
                        total += tl.load(beta1 + rows, rows < width, 0) * b1_total
                else:
                    a0, beta0, a1, beta1 = JOBS[job][4]
                    a0_total = tl.sum(tl.load(a0 + i * w + j, (i < h) & (j < w), 0), axis)
                    total += tl.load(beta0 + rows, rows < width, 0) * a0_total
                    if a1 is not None:
                        a1_total = tl.sum(tl.load(a1 + i * w + j, (i < h) & (j < w), 0), axis)
                        total += tl.load(beta1 + rows, rows < width, 0) * a1_total
            if factor is not None:
                total *= tl.load(factor + rows, rows < width, 0)
            tl.store(output + rows * chunks + bid // tiles, total, rows < width)
            if size > width and bid == 0:
                padding = tl.arange(0, triton.next_power_of_2(size))
                tl.store(output + padding, 0, (padding >= width) & (padding < size))


def reduce_jobs(jobs):
    """Pack reductions directly into disjoint CTA ranges, without equal-shape grouping."""
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


def _tensor(t, shape, dtype, device, name):
    if (
        not isinstance(t, torch.Tensor)
        or tuple(t.shape) != tuple(shape)
        or t.dtype != dtype
        or t.device != device
        or not t.is_contiguous()
    ):
        raise ValueError(f"{name} must be contiguous {dtype} {tuple(shape)} on {device}")


@g.jit
def _fma2(a, b, c):
    # Two independent RN FP32 FMAs (a * b + c), issued as one packed f32x2 instruction.
    return gl.inline_asm_elementwise(
        """
        { // Begin paired-FMA register scope.
            .reg .b64 a,b,c,r; // Declare packed FP32 operand and result registers.
            mov.b64 a,{$2,$3}; // Pack the first pair of multiplicands.
            mov.b64 b,{$4,$5}; // Pack the second pair of multiplicands.
            mov.b64 c,{$6,$7}; // Pack the two addends.
            fma.rn.f32x2 r,a,b,c; // Compute two round-to-nearest fused multiply-adds.
            mov.b64 {$0,$1},r; // Unpack the two FP32 results.
        } // End paired-FMA register scope.
        """,
        "=f,=f,f,f,f,f,f,f",
        [a, b, c],
        dtype=gl.float32,
        is_pure=True,
        pack=2,
    )


def _descriptor(tensor, tile_shape):
    if tensor is None:
        return None
    bits = tensor.element_size() * 8
    layout = NVMMASharedLayout(swizzle_byte_width=bits * 4, element_bitwidth=bits, rank=2)
    return TensorDescriptor.from_tensor(tensor, list(tile_shape), layout)


@g.jit
def _mul_rn(a, b):
    # Uncontracted IEEE multiply; libdevice.mul_rn flushes subnormals under reflect-FTZ.
    return gl.inline_asm_elementwise(
        "mul.rn.f32 $0,$1,$2;", "=f,f,f", [a, b], dtype=gl.float32, is_pure=True, pack=1
    )


@g.jit
def _add_rn(a, b):
    return gl.inline_asm_elementwise(
        "add.rn.f32 $0,$1,$2;", "=f,f,f", [a, b], dtype=gl.float32, is_pure=True, pack=1
    )


@g.jit
def _forward_value(previous, update, write, retain, rows, stream):
    """Match forward's explicit fused/separate FP32 contraction boundaries."""
    retained = _mul_rn(retain, previous)
    separate = _add_rn(_mul_rn(write, update), retained)
    contracted = (stream == 0) | ((stream == 1) & (rows % 32 < 12))
    return gl.where(contracted, gl.fma(write, update, retained), separate)
