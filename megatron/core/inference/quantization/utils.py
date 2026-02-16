# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch
import triton
import triton.language as tl

try:
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

    HAVE_TE = True
except ImportError:
    MXFP8Tensor = None

    HAVE_TE = False

@triton.jit
def _quantize_mxfp8_kernel(
    X_ptr,
    Y_ptr,  # E4M3FN Data Output
    S_ptr,  # E8M0FNU Scale Output
    stride_x_row,
    stride_x_col,
    stride_y_row,
    stride_y_col,
    stride_s_row,
    stride_s_col,
    K,
    BLOCK_SIZE: tl.constexpr,
):
    # Map program ID to specific block in tensor (M, K/32)
    pid = tl.program_id(0)
    num_blocks_k = tl.cdiv(K, BLOCK_SIZE)

    row = pid // num_blocks_k
    col_block = pid % num_blocks_k
    col_offset = col_block * BLOCK_SIZE

    # Offsets for loading/storing
    cols = col_offset + tl.arange(0, BLOCK_SIZE)
    mask = cols < K

    # 1. Load BF16/FP16 Block
    #    Address = base + row * stride_row + col * stride_col
    x_ptrs = X_ptr + (row * stride_x_row) + (cols * stride_x_col)
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # 2. Calculate Scale (Block-wise Max)
    #    Formula: scale_exponent = floor(log2(max(abs(x)))) - 8
    #    This maps the max value into the E4M3 range [-448, 448]
    abs_x = tl.abs(x)
    max_val = tl.max(abs_x, axis=0)

    # Avoid log2(0) by clamping to min small value (scale becomes min representable)
    # If max_val is 0, we can effectively use any scale, but 1.0 (exp 0) is safe.
    max_val = tl.where(max_val == 0, 1.0, max_val)

    # Compute exponent
    log2_val = tl.log2(max_val)
    exponent = tl.floor(log2_val)

    # Shift exponent to normalize to E4M3 range (largest power of 2 is 8)
    # scale_unbiased = 2^(exponent - 8)
    target_exponent = exponent - 8.0
    scale_float = tl.exp2(target_exponent)

    # 3. Quantize Data
    #    q = x / scale
    y = x / scale_float
    #    Clamp to E4M3FN range [-448, 448]
    y = tl.clamp(y, -448.0, 448.0)

    # 4. Store Data (cast to fp8e4nv which corresponds to e4m3fn on NVIDIA GPUs)
    y_ptrs = Y_ptr + (row * stride_y_row) + (cols * stride_y_col)
    tl.store(y_ptrs, y.to(tl.float8e4nv), mask=mask)

    # 5. Store Scale (Format E8M0FNU)
    #    E8M0 is basically an 8-bit exponent with bias 127.
    #    We computed `target_exponent` (unbiased).
    #    We add bias 127 and store as int8/uint8 bit pattern.
    scale_biased = target_exponent + 127.0
    scale_biased = tl.clamp(scale_biased, 0.0, 255.0)  # Safety clamp
    scale_bits = scale_biased.to(tl.int8)

    #    Scale tensor shape is (M, K/32)
    s_ptr = S_ptr + (row * stride_s_row) + (col_block * stride_s_col)
    tl.store(s_ptr, scale_bits)


def quantize_to_mxfp8(x: torch.Tensor, group_size: int = 32):
    """
    Quantize BF16/FP16 tensor to MXFP8 format using Triton.
    Returns:
        x_fp8: Tensor of type float8_e4m3fn (M, K)
        x_scale: Tensor of type float8_e8m0fnu (M, K // 32)
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dim() == 2, "Input must be 2D [M, K]"
    M, K = x.shape
    assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"

    # Allocate outputs
    # Data: E4M3FN
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)

    # Scales: E8M0FNU
    # Note: Torch might not expose 'float8_e8m0fnu' in older versions.
    # If available, use it. If not, use uint8 and view/reinterpret as needed by scaled_mm.
    # scaled_mm expects the scale tensor to be a Float8 type.
    scale_dtype = getattr(torch, 'float8_e8m0fnu', torch.uint8)
    s = torch.empty((M, K // group_size), device=x.device, dtype=scale_dtype)
    s_view = s.view(torch.uint8)

    # Grid: One program per block of 32
    grid = (M * (K // group_size),)

    _quantize_mxfp8_kernel[grid](
        x,
        y,
        s_view,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        s_view.stride(0),
        s_view.stride(1),
        K,
        BLOCK_SIZE=group_size,
    )

    return y, s

def translate_te_mxfp8_tensor(tensor: MXFP8Tensor):
    """
    Helper to extract MXFP8 data and scales from a TE MXFP8Tensor.

    TE internally allocates _rowwise_scale_inv with padded dimensions
    (rows rounded up to 128, cols rounded up to 4) for its own cuBLAS GEMM
    path. When using torch.nn.functional.scaled_mm instead, the scales must
    be trimmed back to the logical (unpadded) shape so that the strides match
    what scaled_mm / cuBLAS expects.

    Returns (tensor_fp8, tensor_scale) with tensor_scale trimmed to match
    the data dimensions.
    """
    assert HAVE_TE and isinstance(tensor, MXFP8Tensor)

    data = tensor._rowwise_data
    scale = tensor._rowwise_scale_inv

    if data.dtype == torch.uint8:
        data = data.view(torch.float8_e4m3fn)

    if scale.dtype == torch.uint8:
        scale = scale.view(torch.float8_e8m0fnu)

    # Trim padded scale to match data dimensions.
    # TE allocates scale as [round_up(N, 128), round_up(K/32, 4)] but
    # scaled_mm expects [N, K/32] with contiguous strides (K/32, 1).
    n, k = data.shape
    expected_scale_rows = n
    expected_scale_cols = k // 32
    if scale.shape[0] != expected_scale_rows or scale.shape[1] != expected_scale_cols:
        scale = scale[:expected_scale_rows, :expected_scale_cols].contiguous()

    return data, scale
