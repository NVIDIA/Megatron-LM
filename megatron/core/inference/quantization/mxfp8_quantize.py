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
import triton
import triton.language as tl


def _ceil_div(a, b):
    return (a + b - 1) // b


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
    row = tl.program_id(0)
    src_row = src_ptr + row * K
    out_row = out_ptr + row * K

    offs = tl.arange(0, BLOCK_K)
    mask = offs < K

    # Load full row
    x = tl.load(src_row + offs, mask=mask, other=0.0).to(tl.float32)

    # Per-group-of-32 max
    x_grouped = tl.reshape(x, [BLOCK_GROUPS, 32])
    abs_grouped = tl.abs(x_grouped)
    max_vals = tl.max(abs_grouped, axis=1)

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

    # Quantize
    quantized = x_grouped * quant_scale[:, None]
    quantized_flat = tl.reshape(quantized, [BLOCK_K])
    out_fp8 = quantized_flat.to(tl.float8e4nv)

    # Store FP8 data
    tl.store(out_row + offs, out_fp8, mask=mask)

    # Store swizzled scales
    scale_exp = (dequant_exp >> 23).to(tl.uint8)
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
    assert K % 32 == 0, f"K ({K}) must be divisible by 32"

    scale_cols = K // 32
    n_row_blocks = _ceil_div(M, 128)
    n_col_blocks = _ceil_div(scale_cols, 4)
    total_scale_bytes = n_row_blocks * n_col_blocks * 512

    out_data = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
    out_scale = torch.zeros(total_scale_bytes, dtype=torch.uint8, device=x.device)

    BLOCK_K = triton.next_power_of_2(K)
    BLOCK_GROUPS = BLOCK_K // 32

    _mxfp8_quant_swizzle_kernel[(M,)](
        out_data,
        out_scale,
        x,
        K,
        n_col_blocks,
        REAL_GROUPS=scale_cols,
        BLOCK_K=BLOCK_K,
        BLOCK_GROUPS=BLOCK_GROUPS,
    )

    return out_data, out_scale.view(torch.float8_e8m0fnu)
