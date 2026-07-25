# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Padding-aware activation kernels for fused MoE.

These kernels skip padding rows (where permutation_map == -1) to avoid
wasted computation on aligned-but-empty expert slots.
"""

from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()


def _ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def _squared_relu_kernel(
    input_ptr,
    output_ptr,
    src_idx_ptr,
    n_used_ptr,
    N,
    max_rows,  # output_size (fixed for CG)
    BLOCK_N: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,  # grid size (fixed for CG)
):
    """Squared ReLU that skips rows beyond n_used and alignment-padding rows (perm_map == -1).

    Grid: fixed NUM_BLOCKS CTAs, each iterating over multiple rows.
    n_used_ptr gates how many rows are processed — required for CUDA graph compatibility.
    """
    pid = tl.program_id(0)
    n_used = tl.load(n_used_ptr)
    if pid >= n_used:
        return
    for row in tl.range(pid, max_rows, NUM_BLOCKS):
        if row < n_used:
            if tl.load(src_idx_ptr + row) >= 0:
                for n in tl.range(0, N, BLOCK_N):
                    o = n + tl.arange(0, BLOCK_N)
                    m = o < N
                    x = tl.load(input_ptr + row * N + o, mask=m).to(tl.float32)
                    r = tl.maximum(x, 0.0)
                    tl.store(output_ptr + row * N + o, (r * r).to(tl.bfloat16), mask=m)


def padded_squared_relu(
    x: torch.Tensor, permutation_map: torch.Tensor, n_used: torch.Tensor
) -> torch.Tensor:
    """Squared ReLU activation that skips rows beyond n_used and alignment-padding rows.

    Args:
        x: [output_size, ffn_hidden] BF16 FC1 output.
        permutation_map: [output_size] int32, original token index or -1 for padding.
        n_used: scalar int32 CUDA tensor = inclusive_expert_offsets[-1].
    """
    M, N = x.shape
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    BLOCK_N = min(triton.next_power_of_2(N), 1024)
    NUM_BLOCKS = min(M, 512)
    _squared_relu_kernel[(NUM_BLOCKS,)](
        x, out, permutation_map, n_used, N, M, BLOCK_N=BLOCK_N, NUM_BLOCKS=NUM_BLOCKS
    )
    return out


@triton.jit
def _squared_relu_quantize_kernel(
    input_ptr,
    out_fp8_ptr,
    out_scale_ptr,
    src_idx_ptr,
    n_used_ptr,  # pointer to inclusive_expert_offsets[-1]: number of used rows this iteration
    K,
    n_col_blocks,
    max_rows,  # output_size (fixed for CG)
    REAL_GROUPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,  # grid size (fixed for CG)
):
    """Fused squared ReLU + MXFP8 quantize + swizzle in one kernel.

    Grid: fixed NUM_BLOCKS CTAs, each iterating over multiple rows.
    Rows beyond n_used and alignment-padding rows (perm_map == -1) are skipped.
    """
    pid = tl.program_id(0)
    n_used = tl.load(n_used_ptr)
    if pid >= n_used:
        return
    for row in tl.range(pid, max_rows, NUM_BLOCKS):
        if row < n_used:
            if tl.load(src_idx_ptr + row) >= 0:
                offs = tl.arange(0, BLOCK_K)
                mask = offs < K

                # Load and apply squared ReLU
                x = tl.load(input_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
                relu = tl.maximum(x, 0.0)
                activated = relu * relu

                # Per-group-of-32 quantization
                x_grouped = tl.reshape(activated, [BLOCK_GROUPS, 32])
                abs_grouped = tl.abs(x_grouped)
                max_vals = tl.max(abs_grouped, axis=1)

                dequant_scale = max_vals / 448.0
                dequant_exp = (dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000
                dequant_rounded = dequant_exp.to(tl.float32, bitcast=True)
                quant_scale = tl.where(dequant_rounded == 0, 0.0, 1.0 / dequant_rounded)

                quantized = x_grouped * quant_scale[:, None]
                quantized_flat = tl.reshape(quantized, [BLOCK_K])
                out_fp8 = quantized_flat.to(tl.float8e4nv)

                # Store FP8 data
                tl.store(out_fp8_ptr + row * K + offs, out_fp8, mask=mask)

                # Store swizzled scales
                scale_exp = (dequant_exp >> 23).to(tl.uint8)
                col_offs = tl.arange(0, BLOCK_GROUPS)
                col_mask = col_offs < REAL_GROUPS

                macro_row_block = row // 128
                macro_col_block = col_offs // 4
                local_row = row % 128
                local_col = col_offs % 4
                group = local_row // 32
                sub_row = local_row % 32
                tile_idx = macro_row_block * n_col_blocks + macro_col_block
                swizzled_offs = tile_idx * 512 + sub_row * 16 + group * 4 + local_col

                tl.store(out_scale_ptr + swizzled_offs, scale_exp, mask=col_mask)


def squared_relu_and_quantize_mxfp8(
    x: torch.Tensor, permutation_map: torch.Tensor, n_used: torch.Tensor
):
    """Fused squared ReLU + MXFP8 quantize + swizzle.

    Reads BF16 FC1 output, applies squared ReLU, quantizes to FP8 with
    swizzled scales. Single kernel replaces padded_squared_relu + mxfp8_quantize.

    Args:
        x: [output_size, K] BF16 FC1 output.
        permutation_map: [output_size] int32, original token index or -1 for padding.
        n_used: scalar int32 CUDA tensor = inclusive_expert_offsets[-1]. Rows beyond
            this are skipped before even checking the permutation_map.

    Returns:
        MXFP8Tensor with .data [output_size, K] float8_e4m3fn and .scale (swizzled e8m0).
    """
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

    M, K = x.shape
    assert K % 32 == 0

    scale_cols = K // 32
    n_row_blocks = _ceil_div(M, 128)
    n_col_blocks = _ceil_div(scale_cols, 4)
    total_scale_bytes = n_row_blocks * n_col_blocks * 512

    out_fp8 = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
    out_scale = torch.zeros(total_scale_bytes, dtype=torch.uint8, device=x.device)

    BLOCK_K = triton.next_power_of_2(K)
    BLOCK_GROUPS = BLOCK_K // 32
    NUM_BLOCKS = min(M, 512)

    _squared_relu_quantize_kernel[(NUM_BLOCKS,)](
        x,
        out_fp8,
        out_scale,
        permutation_map,
        n_used,
        K,
        n_col_blocks,
        M,
        REAL_GROUPS=scale_cols,
        BLOCK_K=BLOCK_K,
        BLOCK_GROUPS=BLOCK_GROUPS,
        NUM_BLOCKS=NUM_BLOCKS,
    )

    return MXFP8Tensor(data=out_fp8, scale=out_scale.view(torch.float8_e8m0fnu), backend="triton")
