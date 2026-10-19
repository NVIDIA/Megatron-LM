# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Standalone MXFP8 quantization kernel with fused scale swizzle.

One block per token. Quantizes BF16 → FP8 e4m3 and writes scales directly
in cuBLAS 2D blocked (swizzled) layout. No FP4, no triton_kernels dependency.

Usage:
    from megatron.core.inference.quantization.mxfp8_quantize import mxfp8_quantize
    data, swizzled_scales, total_scale_bytes = mxfp8_quantize(x_bf16)
    # data: [M, K] float8_e4m3fn
    # swizzled_scales: 1D uint8 in cuBLAS blocked layout
"""

import torch

MXFP8_BLOCK_SIZE = 32
MXFP8_SCALE_ROW_BLOCK = 128
MXFP8_SCALE_COL_BLOCK = 4

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    from unittest.mock import MagicMock

    from megatron.core.utils import null_decorator

    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False


def _ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def _mxfp8_quantize_groups(x, BLOCK_GROUPS: tl.constexpr):
    """Quantize one row of BLOCK_GROUPS * 32 fp32 values to e4m3 with per-32 e8m0 scales.

    Returns the e4m3 values ([BLOCK_GROUPS * 32]) and the scale exponents ([BLOCK_GROUPS] uint8).
    The scale rounds up to the next power of two (Mishra et al., https://arxiv.org/pdf/2506.08027).
    """
    x_grouped = tl.reshape(x, [BLOCK_GROUPS, 32])
    max_vals = tl.max(tl.abs(x_grouped), axis=1)

    # 448 is the max representable value in FP8 e4m3.
    # dequant_scale = min scale s.t. max_val / scale <= 448.
    dequant_scale = max_vals / 448.0
    # Round up to next power of 2 via integer bit manipulation:
    # Adding 0x007FFFFF (mantissa mask) before masking with 0x7F800000
    # (exponent-only mask) bumps the exponent if any mantissa bits are set.
    # Result: 2^ceil(log2(max/448)) as a uint32-encoded float.
    dequant_exp = (dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000
    # Reinterpret uint32 back as float32 — now a power-of-2 dequantization scale.
    dequant_rounded = dequant_exp.to(tl.float32, bitcast=True)
    # Quantization scale is the reciprocal; guard against div-by-zero for all-zero groups.
    quant_scale = tl.where(dequant_rounded == 0, 0.0, 1.0 / dequant_rounded)

    quantized = tl.reshape(x_grouped * quant_scale[:, None], [BLOCK_GROUPS * 32])
    return quantized.to(tl.float8e4nv), (dequant_exp >> 23).to(tl.uint8)


@triton.jit
def _mxfp8_quant_swizzle_row(
    row,
    out_ptr,
    scale_ptr,
    src_ptr,
    K,
    n_col_blocks,
    REAL_GROUPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    """Quantize one row (int64 index) to FP8 e4m3 and store its swizzled scales."""
    src_row = src_ptr + row * K
    out_row = out_ptr + row * K

    offs = tl.arange(0, BLOCK_K)
    mask = offs < K

    # Load full row
    x = tl.load(src_row + offs, mask=mask, other=0.0).to(tl.float32)
    out_fp8, scale_exp = _mxfp8_quantize_groups(x, BLOCK_GROUPS)

    # Store FP8 data
    tl.store(out_row + offs, out_fp8, mask=mask)

    # Store swizzled scales
    col_offs = tl.arange(0, BLOCK_GROUPS)
    col_mask = col_offs < REAL_GROUPS

    # Compute swizzled offsets for each scale element.
    #
    # The scale matrix [M, K//32] is divided into 128×4 macro-tiles.
    # Within each tile, rows are split into 4 groups of 32 (group = local_row // 32).
    # Rather than flattening row-major, the layout interleaves groups so that
    # rows 32 apart are adjacent in memory:
    #
    #   offset = tile_idx * 512 + sub_row * 16 + group * 4 + local_col
    macro_row_block = row // 128
    macro_col_block = col_offs // 4
    local_row = row % 128
    local_col = col_offs % 4
    group = local_row // 32
    sub_row = local_row % 32
    tile_idx = macro_row_block * n_col_blocks + macro_col_block
    swizzled_offs = tile_idx * 512 + sub_row * 16 + group * 4 + local_col

    tl.store(scale_ptr + swizzled_offs, scale_exp, mask=col_mask)


@triton.jit
def _mxfp8_quant_swizzle_kernel(
    out_ptr,  # [M, K] output buffer for float8_e4m3fn quantized data
    scale_ptr,  # 1D output buffer for swizzled uint8 scales (e8m0 exponents)
    src_ptr,  # [M, K] input tensor in bf16/fp16/fp32
    K,  # number of columns in the input (must be divisible by 32)
    n_col_blocks,  # ceil(K/32 / 4) — number of macro-tile columns in the swizzle layout
    REAL_GROUPS: tl.constexpr,  # actual number of scale groups per row (K // 32)
    BLOCK_K: tl.constexpr,  # next_power_of_2(K) — padded column count for tl.reshape
    BLOCK_GROUPS: tl.constexpr,  # BLOCK_K // 32 — padded group count (must be power of 2)
):
    """Each triton block quantizes one row → FP8 e4m3, write scales directly in swizzled layout.

    We use round up in scale calculation. see: Mishra et al.,
    Recipes for Pre-training LLMs with MXFP8 (https://arxiv.org/pdf/2506.08027)

    The implementation borrows code from the triton upstream MXFP downcast kernel:
    https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/numerics_details/mxfp_details/_downcast_to_mxfp.py

    Note on swizzled scale layout (torch.nn.functional.SwizzleType.SWIZZLE_32_4_4):

        Background: In MXFP8, every group of 32 elements shares one 1-byte scale
        (an e8m0 exponent). For an [M, K] matrix, this gives an [M, K//32] scale
        matrix. cuBLAS doesn't read these scales in simple row-major order — it
        expects a "swizzled" layout optimized for its internal access patterns.

        Step 1 — Divide into macro-tiles:
            The scale matrix is partitioned into 128-row x 4-col macro-tiles.
            Each tile is stored as a contiguous 512-byte (128 x 4) block.

        Step 2 — Interleave within each tile:
            Within a macro-tile, the 128 rows are NOT stored sequentially.
            Instead, they are split into 4 groups of 32 rows:
                group 0: rows   0- 31
                group 1: rows  32- 63
                group 2: rows  64- 95
                group 3: rows  96-127

            Rows with the same position within their group (same "sub_row")
            are placed next to each other. So the memory layout is:

            Concretely, for sub_row=0:
                byte 0:  row  0, col 0
                byte 1:  row  0, col 1
                byte 2:  row  0, col 2
                byte 3:  row  0, col 3
                byte 4:  row 32, col 0
                byte 5:  row 32, col 1
                byte 6:  row 32, col 2
                byte 7:  row 32, col 3
                byte 8:  row 64, col 0
                ...
                byte 15: row 96, col 3

        The formula to map logical (row, col) → byte offset:
            tile_idx = (row // 128) * n_col_blocks + (col // 4)
            sub_row  = row % 32
            group    = (row % 128) // 32
            local_col = col % 4
            offset   = tile_idx * 512 + sub_row * 16 + group * 4 + local_col

    """
    row = tl.program_id(0).to(tl.int64)
    _mxfp8_quant_swizzle_row(
        row, out_ptr, scale_ptr, src_ptr, K, n_col_blocks, REAL_GROUPS, BLOCK_K, BLOCK_GROUPS
    )


def mxfp8_quantize_into(x: torch.Tensor, out_data: torch.Tensor, out_scale: torch.Tensor) -> None:
    """Quantize a 2D tensor into existing MXFP8 data and scale storage.

    Args:
        x: [M, K] tensor in bf16/fp16/fp32. K must be divisible by 32.
        out_data: [M, K] contiguous float8_e4m3fn output tensor.
        out_scale: Contiguous 1D scale storage in cuBLAS blocked layout. The
            dtype may be uint8 or float8_e8m0fnu.
    """
    assert x.is_cuda and x.dim() == 2
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    M, K = x.shape
    assert K % MXFP8_BLOCK_SIZE == 0, f"K ({K}) must be divisible by {MXFP8_BLOCK_SIZE}"
    assert out_data.is_cuda and out_data.device == x.device
    assert out_data.shape == x.shape and out_data.dtype == torch.float8_e4m3fn
    assert out_data.is_contiguous()
    assert out_scale.is_cuda and out_scale.device == x.device
    assert out_scale.dim() == 1 and out_scale.is_contiguous()
    assert out_scale.dtype in (torch.uint8, torch.float8_e8m0fnu)

    scale_cols = K // MXFP8_BLOCK_SIZE
    n_row_blocks = _ceil_div(M, MXFP8_SCALE_ROW_BLOCK)
    n_col_blocks = _ceil_div(scale_cols, MXFP8_SCALE_COL_BLOCK)
    total_scale_bytes = n_row_blocks * n_col_blocks * MXFP8_SCALE_ROW_BLOCK * MXFP8_SCALE_COL_BLOCK
    assert out_scale.numel() == total_scale_bytes

    # The quantization kernel only writes logical scale entries. Clear the
    # padded rows and columns so reused storage matches freshly quantized output.
    out_scale_bytes = out_scale.view(torch.uint8)
    out_scale_bytes.zero_()

    BLOCK_K = triton.next_power_of_2(K)
    BLOCK_GROUPS = BLOCK_K // MXFP8_BLOCK_SIZE

    _mxfp8_quant_swizzle_kernel[(M,)](
        out_data,
        out_scale_bytes,
        x,
        K,
        n_col_blocks,
        REAL_GROUPS=scale_cols,
        BLOCK_K=BLOCK_K,
        BLOCK_GROUPS=BLOCK_GROUPS,
    )


@triton.jit
def _mxfp8_quant_swizzle_rows_kernel(
    out_ptr,
    scale_ptr,
    src_ptr,
    n_rows_ptr,  # scalar int32/int64 CUDA tensor: rows [0, n_rows) are quantized
    K,
    n_col_blocks,
    REAL_GROUPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
):
    """Persistent variant: a fixed grid quantizes only the first ``n_rows`` rows.

    The row count is read on device, so the launch is CUDA-graph safe while the work
    scales with the live rows instead of the worst-case buffer. Each row runs the same
    code as ``_mxfp8_quant_swizzle_kernel``, so the quantized rows are bitwise equal.
    """
    pid = tl.program_id(0)
    n_rows = tl.load(n_rows_ptr).to(tl.int64)
    for row in tl.range(pid, n_rows, NUM_PROGRAMS):
        _mxfp8_quant_swizzle_row(
            row.to(tl.int64),
            out_ptr,
            scale_ptr,
            src_ptr,
            K,
            n_col_blocks,
            REAL_GROUPS,
            BLOCK_K,
            BLOCK_GROUPS,
        )


def mxfp8_quantize_rows(
    x: torch.Tensor, n_rows: torch.Tensor, max_programs: int = 4096
) -> tuple[torch.Tensor, torch.Tensor]:
    """Like ``mxfp8_quantize`` but only rows ``[0, n_rows)`` are quantized.

    ``n_rows`` is a 1-element CUDA tensor (device-side count, CUDA-graph safe). Data rows
    at or beyond ``n_rows`` are left uninitialized and their scales are zero; callers
    must not read them (grouped GEMMs bounded by expert offsets ending at ``n_rows``).
    Quantized rows are bitwise identical to ``mxfp8_quantize``.
    """
    assert x.is_cuda and x.dim() == 2
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert n_rows.is_cuda and n_rows.numel() == 1
    M, K = x.shape
    assert K % MXFP8_BLOCK_SIZE == 0, f"K ({K}) must be divisible by {MXFP8_BLOCK_SIZE}"
    scale_cols = K // MXFP8_BLOCK_SIZE
    n_row_blocks = _ceil_div(M, MXFP8_SCALE_ROW_BLOCK)
    n_col_blocks = _ceil_div(scale_cols, MXFP8_SCALE_COL_BLOCK)
    total_scale_bytes = n_row_blocks * n_col_blocks * MXFP8_SCALE_ROW_BLOCK * MXFP8_SCALE_COL_BLOCK
    out_data = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
    out_scale = torch.zeros(total_scale_bytes, dtype=torch.uint8, device=x.device)
    BLOCK_K = triton.next_power_of_2(K)
    num_programs = max(1, min(M, max_programs))
    _mxfp8_quant_swizzle_rows_kernel[(num_programs,)](
        out_data,
        out_scale,
        x,
        n_rows,
        K,
        n_col_blocks,
        REAL_GROUPS=scale_cols,
        BLOCK_K=BLOCK_K,
        BLOCK_GROUPS=BLOCK_K // MXFP8_BLOCK_SIZE,
        NUM_PROGRAMS=num_programs,
    )
    return out_data, out_scale.view(torch.float8_e8m0fnu)


def mxfp8_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to MXFP8 with fused scale swizzle.

    Args:
        x: [M, K] tensor in bf16/fp16/fp32. K must be divisible by 32.

    Returns:
        (data, swizzled_scales):
            data: [M, K] float8_e4m3fn
            swizzled_scales: 1D tensor in cuBLAS blocked layout (uint8/e8m0)
    """
    assert x.is_cuda and x.dim() == 2
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    M, K = x.shape
    assert K % MXFP8_BLOCK_SIZE == 0, f"K ({K}) must be divisible by {MXFP8_BLOCK_SIZE}"
    scale_cols = K // MXFP8_BLOCK_SIZE
    n_row_blocks = _ceil_div(M, MXFP8_SCALE_ROW_BLOCK)
    n_col_blocks = _ceil_div(scale_cols, MXFP8_SCALE_COL_BLOCK)
    total_scale_bytes = n_row_blocks * n_col_blocks * MXFP8_SCALE_ROW_BLOCK * MXFP8_SCALE_COL_BLOCK

    out_data = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
    out_scale = torch.empty(total_scale_bytes, dtype=torch.uint8, device=x.device)
    mxfp8_quantize_into(x, out_data, out_scale)

    return out_data, out_scale.view(torch.float8_e8m0fnu)
