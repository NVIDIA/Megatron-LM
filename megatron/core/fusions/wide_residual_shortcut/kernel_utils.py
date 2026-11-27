# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared tensor contracts, numerical primitives and storage utilities."""

import types
from dataclasses import dataclass
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, NVMMASharedLayout
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

# A reader output or update gradient can emit both formats from the same values.
OUTPUT_FORMATS = ("bf16", "mxfp8", "both")

# Host forms of Triton's integer helpers: their constexpr-function wrappers cost ~1 us a call.
cdiv, next_power_of_2 = triton.cdiv.fn, triton.next_power_of_2.fn


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


def validate_num_streams(num_streams):
    """Validate the residual stream count (1-6)."""
    if type(num_streams) is not int or not 1 <= num_streams <= 6:
        raise ValueError("num_streams must be an integer from 1 through 6")


def _tensor(t, shape, dtype, device, name):
    if (
        not isinstance(t, torch.Tensor)
        or tuple(t.shape) != tuple(shape)
        or t.dtype != dtype
        or t.device != device
        or not t.is_contiguous()
    ):
        raise ValueError(f"{name} must be contiguous {dtype} {tuple(shape)} on {device}")


def allocate_operand(shape, device, *, rowwise=True, columnwise=False):
    """Allocate an MXFP8 operand with the requested TE orientations (its own
    quantizer: TE may change a tensor's quantizer usage)."""
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    from transformer_engine_torch import DType  # constants does not re-export it in every TE

    quantizer = MXFP8Quantizer(DType.kFloat8E4M3, rowwise=rowwise, columnwise=columnwise)
    quantizer.optimize_for_gemm = True
    return quantizer.make_empty(shape, dtype=torch.bfloat16, device=device)


def _descriptor(tensor, tile_shape=(16, 128)):
    """Describe a backward TMA copy, normally one column half of a 16-row tile."""
    if tensor is None:
        return None
    bits = tensor.element_size() * 8
    layout = NVMMASharedLayout(swizzle_byte_width=bits * 4, element_bitwidth=bits, rank=2)
    return TensorDescriptor.from_tensor(tensor, list(tile_shape), layout)


@g.jit
def _volatile_asm(asm: gl.constexpr, constraints: gl.constexpr, args):
    """One volatile PTX invocation per element, including side-effect-only instructions."""
    return gl.inline_asm_elementwise(asm, constraints, args, dtype=gl.int32, is_pure=False, pack=1)


@g.jit
def _coefficient(table, INDEX: gl.constexpr):
    """One coefficient-table entry (registers or shared memory) as a scalar."""
    indices = gl.full(
        (1,), INDEX, gl.int32, gl.constexpr(BlockedLayout([1], [32], [gl.num_warps()], [0]))
    )
    return gl.sum(table.gather(indices, 0), 0)


@g.jit
def _e8m0_scale(amax):
    """TE's E8M0 exponent of each block amax and the FP32 reciprocal scale it encodes.

    Inline PTX: Gluon has no E8M0 conversion. ``cvt.rp.satfinite.ue8m0x2.f32`` produces the
    scale exponent (rounded toward +inf, finite values saturated to 254, NaN kept as 255) in one
    instruction; the scales match TE's MXFP8 quantizer bit for bit (tested against it).
    """
    exponent = (
        gl.inline_asm_elementwise(
            "cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;",
            "=h,f",
            [amax * (1.0 / 448.0)],
            dtype=gl.uint16,
            is_pure=True,
            pack=1,
        ).to(gl.uint32)
        & 255
    )
    # TE's E8M0 exponent; 254 has a subnormal reciprocal and 255 preserves NaN encoding.
    bits = gl.where(exponent == 254, 0x00400000, (254 - exponent) << 23)
    bits = gl.where(exponent == 255, 0x7FFFFFFF, bits)
    return exponent, bits


@g.jit
def _swizzled_scale_index(outer, inner, width):
    """Offset of the E8M0 scale of 32-value block ``inner`` of row ``outer`` in TE's GEMM-ready
    layout (128 x 4 scale tiles, ``width`` values per row)."""
    # Left-associated, so constant per-thread offsets fold into store immediates.
    index = ((outer // 128) * gl.cdiv(width, 128) + inner // 4) * 512 + (outer % 32) * 16
    return index + (outer % 128 // 32) * 4 + inner % 4


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


@g.jit
def _round_bf16(x):
    """FP32 -> BF16, round to nearest even, two values per ``cvt.rn.bf16x2.f32`` (the scalar
    conversion runs on the quarter-rate conversion pipe)."""
    return gl.inline_asm_elementwise(
        "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f", [x], dtype=gl.bfloat16, is_pure=True, pack=2
    )


@g.jit
def _store_bf16x4(pointer, x):
    """Store FP32 ``x`` RN-rounded to BF16, four contiguous 8-byte-aligned values per store, and
    return the stored values widened back to FP32 (exact). Inline PTX: packing in the conversion
    and widening from the packed words avoids the compiler's half-word repacking."""
    return gl.inline_asm_elementwise(
        """
        {
            .reg .b32 p, q;
            cvt.rn.bf16x2.f32 p, $5, $4;
            cvt.rn.bf16x2.f32 q, $7, $6;
            st.global.v2.b32 [$8], {p, q};
            shl.b32 $0, p, 16;
            and.b32 $1, p, 0xffff0000;
            shl.b32 $2, q, 16;
            and.b32 $3, q, 0xffff0000;
        }
        """,
        "=f,=f,=f,=f,f,f,f,f,l,l,l,l",
        [x, pointer.to(gl.int64, bitcast=True)],
        dtype=gl.float32,
        is_pure=False,
        pack=4,
    )


@g.jit
def _load_f32x4(pointer):
    """Four FP32 scalars from a 16-byte-aligned address in one vector load (one L1 wavefront
    instead of four uniform scalar loads)."""
    return gl.inline_asm_elementwise(
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        [pointer.to(gl.int64, bitcast=True)],
        dtype=(gl.float32, gl.float32, gl.float32, gl.float32),
        is_pure=True,
        pack=1,
    )


@g.jit
def _widen_bf16(x):
    """Exact widening to FP32; BF16 bits become the upper half of the FP32 word. Inline PTX on
    the packed pair: one shift or mask per value, without unpacking the halves first."""
    if x.dtype == gl.bfloat16:
        return gl.inline_asm_elementwise(
            """
            shl.b32 $0, $2, 16;
            and.b32 $1, $2, 0xffff0000;
            """,
            "=f,=f,r",
            [x],
            dtype=gl.float32,
            is_pure=True,
            pack=2,
        )
    return x.to(gl.float32)


# Uncontracted IEEE FP32 multiply for replay. Inline PTX: replay must reproduce the forward's
# FP32 rounding exactly, but the compiler still contracts a plain ``*`` into a neighboring FMA
# even with enable_fp_fusion=False (measured: replay no longer bit-identical), and
# libdevice.mul_rn flushes subnormals under reflect-FTZ.


@g.jit
def _mul_rn(a, b):
    """``a * b`` rounded to nearest, never fused into an FMA."""
    return gl.inline_asm_elementwise(
        "mul.rn.f32 $0,$1,$2;", "=f,f,f", [a, b], dtype=gl.float32, is_pure=True, pack=1
    )


@g.jit
def _forward_value(previous, update, write, retain):
    """The forward's residual write: the rounded retain product, then one fused write."""
    return gl.fma(write, update, _mul_rn(retain, previous))


def named_jit(fn, name):
    """Gluon-jit ``fn`` under ``name``, the kernel name profilers show: one kernel body appears
    under a descriptive name per variant (each name compiles only its own specializations)."""
    copy = types.FunctionType(fn.__code__, fn.__globals__, name, fn.__defaults__, fn.__closure__)
    copy.__qualname__, copy.__module__, copy.__doc__ = name, fn.__module__, fn.__doc__
    copy.__annotations__, copy.__kwdefaults__ = dict(fn.__annotations__), fn.__kwdefaults__
    return g.jit(copy)


class _Reduction(NamedTuple):
    width: int
    height: int
    tile: int
    chunk: int
    axis: int
    output_size: int
    begin: int
    end: int
    term_ranks: tuple
    accumulate: int  # 0 store, 1 add, 2 round to BF16 then add


@triton.jit
def _reduce_tile(source, output, factor, spec: tl.constexpr, terms, base_weight):
    if tl.program_id(0) >= spec.begin and tl.program_id(0) < spec.end:
        width: tl.constexpr = spec.width
        height: tl.constexpr = spec.height
        chunks: tl.constexpr = tl.cdiv(height, spec.chunk)
        tiles: tl.constexpr = tl.cdiv(width, spec.tile)
        bid = tl.program_id(0) - spec.begin
        rows = bid % tiles * spec.tile + tl.arange(0, spec.tile)
        cols = bid // tiles * spec.chunk + tl.arange(0, spec.chunk)
        i, j = (cols[:, None], rows[None, :]) if spec.axis == 0 else (rows[:, None], cols[None, :])
        h, w = (height, width) if spec.axis == 0 else (width, height)
        total = tl.sum(tl.load(source + i * w + j, (i < h) & (j < w), 0), spec.axis)
        # A reader contributes one shared B row to the write/read derivative and
        # one A row per stream to retention. Keep each reduction's original sum tree.
        for k in tl.static_range(len(terms)):
            partial, weight = terms[k]
            if spec.term_ranks[k] == 2:
                part = tl.sum(tl.load(partial + i * w + j, (i < h) & (j < w), 0), spec.axis)
            else:
                part = tl.sum(tl.load(partial + cols, cols < height, 0), 0)
            w_k = tl.load(weight + rows, rows < width, 0)
            if k == 0 and base_weight is not None:
                total *= tl.load(base_weight + rows, rows < width, 0)
            total += w_k * part
        if factor is not None:
            total *= tl.load(factor + rows, rows < width, 0)
        if spec.accumulate:
            if spec.accumulate == 2:
                total = total.to(tl.bfloat16).to(tl.float32)
            total += tl.load(output + rows, rows < width, 0).to(tl.float32)
        tl.store(output + rows * chunks + bid // tiles, total, rows < width)
        if spec.output_size > width and bid == 0 and not spec.accumulate:
            padding = tl.arange(0, triton.next_power_of_2(spec.output_size))
            tl.store(output + padding, 0, (padding >= width) & (padding < spec.output_size))


@triton.jit
def bwd_wr_writeread_reduce(JOBS):
    """Reduce disjoint matrix tiles in one launch, preserving each axis's sum tree. An
    accumulating job adds its result, rounded to the parameter dtype first, into the output (a
    parameter's main_grad): autograd's cast of the returned gradient, then the DDP hook's add."""
    for job in tl.static_range(len(JOBS)):
        _reduce_tile(*JOBS[job])


def reduce_jobs(jobs):
    """Pack ``(source, output, axis, tile, chunk, factor, combo, accumulate)`` reductions into one
    launch (``accumulate`` as in bwd_wr_writeread_reduce; a single chunk only)."""
    packed, grid = [], 0
    for job in jobs:
        source, output, axis, tile, chunk, factor, combo, accumulate = job
        width, height = source.shape[1 - axis], source.shape[axis]
        chunk = chunk or next_power_of_2(height)
        size = output.numel() if output.ndim == 1 else width
        end = grid + cdiv(width, tile) * cdiv(height, chunk)
        terms, base_weight = combo or ((), None)
        ranks = tuple(partial.ndim for partial, _ in terms)
        spec = _Reduction(width, height, tile, chunk, axis, size, grid, end, ranks, accumulate)
        packed.append((source, output, factor, tl.constexpr(spec), terms, base_weight))
        grid = end
    if packed:
        bwd_wr_writeread_reduce[(grid,)](tuple(packed), num_warps=4, enable_fp_fusion=False)
