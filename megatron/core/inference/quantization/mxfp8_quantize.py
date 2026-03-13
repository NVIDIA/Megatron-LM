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
    out_ptr, scale_ptr, src_ptr,
    K,
    n_col_blocks,
    REAL_GROUPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    """Quantize one row → FP8 e4m3, write scales directly in swizzled layout.

    Grid: (M,) — one program per token/row.

    Swizzle layout (cuBLAS 2D blocked):
      Each 128×4 macro tile is stored as 512 bytes.
      For scale at (row, col):
        macro_row_block = row // 128
        macro_col_block = col // 4
        local_row = row % 128
        local_col = col % 4
        group = local_row // 32
        sub_row = local_row % 32
        tile_idx = macro_row_block * n_col_blocks + macro_col_block
        offset = tile_idx * 512 + sub_row * 16 + group * 4 + local_col
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

    # Scale = 2^ceil(log2(max / 448.0))
    dequant_scale = max_vals / 448.0
    dequant_exp = (dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000
    dequant_rounded = dequant_exp.to(tl.float32, bitcast=True)
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

    # Compute swizzled offsets for each scale element
    # divide scales into 128x4 macro tiles
    # then flatten each tile.
    macro_row_block = row // 128
    macro_col_block = col_offs // 4
    local_row = row % 128
    local_col = col_offs % 4
    # each group of 32 elements shares the same scale
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
        out_data, out_scale, x,
        K, n_col_blocks,
        REAL_GROUPS=scale_cols,
        BLOCK_K=BLOCK_K,
        BLOCK_GROUPS=BLOCK_GROUPS,
    )

    return out_data, out_scale.view(torch.float8_e8m0fnu)