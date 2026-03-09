# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused MoE: permute -> FC1 -> activation -> FC2 -> unpermute.

Supports BF16 and MXFP8 weight/activation dtypes. All permutation logic
lives here — callers (e.g. experts.py) invoke a single function.

Usage:
    from megatron.core.inference.kernels.fused_moe import (
        fused_moe, padded_squared_relu, padded_swiglu,
    )
    output = fused_moe(
        hidden_states, routing_map, probs,
        fc1_weight, fc2_weight,
        padded_squared_relu,
        num_local_experts, local_expert_start,
    )
"""

from unittest.mock import MagicMock

import torch
from packaging import version

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    if version.parse(triton.__version__) < version.parse("3.4.0") and not torch.cuda.is_available():
        HAVE_TRITON = False
    else:
        HAVE_TRITON = tl.constexpr(version.parse(triton.__version__) >= version.parse("2.0.0"))
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()

try:
    from torch.nn.functional import scaled_grouped_mm, ScalingType, SwizzleType

    _SWIZZLE = SwizzleType.SWIZZLE_32_4_4
    _RECIPE = ScalingType.BlockWise1x32
    HAVE_SCALED_GMM = True
except ImportError:
    HAVE_SCALED_GMM = False

try:
    from torch.nn.functional import grouped_mm

    HAVE_GROUPED_MM = True
except ImportError:
    HAVE_GROUPED_MM = False

from enum import Enum
from typing import Callable

from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor, _ceil_div


class ActivationType(Enum):
    """Activation functions supported by mcore_fused_moe."""
    SQUARED_RELU = "squared_relu"
    SWIGLU = "swiglu"


# =========================================================================== #
# Token count kernel
# =========================================================================== #
@triton.jit
def _count_local_tokens_kernel(
    routing_map_ptr, tokens_per_expert_ptr, total_pairs,
    local_expert_start, num_local_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Count tokens assigned to each local expert."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_pairs
    expert_ids = tl.load(routing_map_ptr + offsets, mask=mask, other=-1)
    local_ids = expert_ids - local_expert_start
    is_local = (local_ids >= 0) & (local_ids < num_local_experts) & mask
    tl.atomic_add(tokens_per_expert_ptr + local_ids, 1, mask=is_local)


def compute_local_tokens_per_expert(
    routing_map: torch.Tensor, local_expert_start: int, num_local_experts: int,
) -> torch.Tensor:
    """Count tokens routed to each local expert."""
    total_pairs = routing_map.numel()
    tokens_per_expert = torch.zeros(
        num_local_experts, dtype=torch.int32, device=routing_map.device,
    )
    BLOCK = 256
    _count_local_tokens_kernel[(_ceil_div(total_pairs, BLOCK),)](
        routing_map, tokens_per_expert, total_pairs,
        local_expert_start, num_local_experts, BLOCK_SIZE=BLOCK,
    )
    return tokens_per_expert


# =========================================================================== #
# Expert offset computation
# =========================================================================== #
@triton.jit
def _prefix_sum_kernel(
    tokens_per_expert_ptr, exclusive_offsets_ptr, inclusive_offsets_ptr,
    num_local_experts, alignment: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    """Exclusive and inclusive prefix sums of aligned token counts."""
    r = tl.arange(0, BLOCK_SIZE)
    mask = r < num_local_experts
    h = tl.load(tokens_per_expert_ptr + r, mask=mask, other=0)
    if alignment > 1:
        h = tl.where(h > 0, ((h + alignment - 1) // alignment) * alignment, h)
    inc = tl.cumsum(h, axis=0)
    tl.store(exclusive_offsets_ptr + r, inc - h, mask=mask)
    tl.store(inclusive_offsets_ptr + r, inc, mask=mask)


def compute_expert_offsets(
    tokens_per_expert: torch.Tensor, alignment: int = 1,
) -> tuple:
    """Compute exclusive and inclusive prefix sums of aligned token counts."""
    n = tokens_per_expert.shape[0]
    exc = torch.empty_like(tokens_per_expert)
    inc = torch.empty_like(tokens_per_expert)
    _prefix_sum_kernel[(1,)](
        tokens_per_expert, exc, inc, n, alignment,
        BLOCK_SIZE=triton.next_power_of_2(n),
    )
    return exc, inc


# =========================================================================== #
# Fused permute + MXFP8 quantize kernel
# =========================================================================== #
@triton.jit
def _permute_quantize_kernel(
    hidden_ptr, probs_ptr, routing_map_ptr,
    out_fp8_ptr, out_scale_ptr, out_probs_ptr, out_src_idx_ptr,
    counters_ptr,
    num_tokens, K,
    n_col_blocks,
    topk: tl.constexpr,
    local_expert_start,
    num_local_experts: tl.constexpr,
    REAL_GROUPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    """Fused permute + MXFP8 quantize + swizzle in one kernel.

    Grid: (num_tokens * topk,) — one program per (token, k) pair.
    Reads BF16 from source token, quantizes to FP8 e4m3, writes FP8 data +
    swizzled e8m0 scales to the permuted write position.
    """
    pair = tl.program_id(0)
    tok = pair // topk
    k = pair % topk
    if tok >= num_tokens:
        return
    eid = tl.load(routing_map_ptr + tok * topk + k)
    lid = eid - local_expert_start
    if lid < 0 or lid >= num_local_experts:
        return

    pos = tl.atomic_add(counters_ptr + lid, 1)

    # Load full row from source token
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    x = tl.load(hidden_ptr + tok * K + offs, mask=mask, other=0.0).to(tl.float32)

    # Per-group-of-32 quantization
    x_grouped = tl.reshape(x, [BLOCK_GROUPS, 32])
    abs_grouped = tl.abs(x_grouped)
    max_vals = tl.max(abs_grouped, axis=1)

    dequant_scale = max_vals / 448.0
    dequant_exp = (dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000
    dequant_rounded = dequant_exp.to(tl.float32, bitcast=True)
    quant_scale = tl.where(dequant_rounded == 0, 0.0, 1.0 / dequant_rounded)

    quantized = x_grouped * quant_scale[:, None]
    quantized_flat = tl.reshape(quantized, [BLOCK_K])
    out_fp8 = quantized_flat.to(tl.float8e4nv)

    # Store FP8 data at permuted position
    tl.store(out_fp8_ptr + pos * K + offs, out_fp8, mask=mask)

    # Store swizzled scales at permuted position
    scale_exp = (dequant_exp >> 23).to(tl.uint8)
    col_offs = tl.arange(0, BLOCK_GROUPS)
    col_mask = col_offs < REAL_GROUPS

    macro_row_block = pos // 128
    macro_col_block = col_offs // 4
    local_row = pos % 128
    local_col = col_offs % 4
    group = local_row // 32
    sub_row = local_row % 32
    tile_idx = macro_row_block * n_col_blocks + macro_col_block
    swizzled_offs = tile_idx * 512 + sub_row * 16 + group * 4 + local_col

    tl.store(out_scale_ptr + swizzled_offs, scale_exp, mask=col_mask)

    # Store prob and source index
    tl.store(out_probs_ptr + pos, tl.load(probs_ptr + tok * topk + k))
    tl.store(out_src_idx_ptr + pos, tok)


def permute_and_quantize(
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    expert_offsets: torch.Tensor,
    local_expert_start: int,
    num_local_experts: int,
    output_size: int,
) -> tuple:
    """Fused permute + MXFP8 quantize + swizzle.

    Reads BF16 from source tokens, quantizes to FP8, writes directly to
    permuted positions with swizzled scales. Single kernel launch replaces
    permute_tokens + mxfp8_quantize.

    NOTE: expert_offsets is mutated in-place via atomic adds.

    Returns:
        (out_fp8, out_scale, out_probs, out_src_idx):
            out_fp8: [output_size, K] float8_e4m3fn
            out_scale: 1D swizzled e8m0 scales
            out_probs: [output_size] routing probs
            out_src_idx: [output_size] source token indices (-1 for padding)
    """
    num_tokens, K = hidden_states.shape
    topk = probs.shape[1]
    assert K % 32 == 0

    scale_cols = K // 32
    n_row_blocks = _ceil_div(output_size, 128)
    n_col_blocks = _ceil_div(scale_cols, 4)
    total_scale_bytes = n_row_blocks * n_col_blocks * 512

    out_fp8 = torch.empty(output_size, K, dtype=torch.float8_e4m3fn, device=hidden_states.device)
    out_scale = torch.zeros(total_scale_bytes, dtype=torch.uint8, device=hidden_states.device)
    out_probs = torch.empty(output_size, dtype=probs.dtype, device=probs.device)
    out_src_idx = torch.full((output_size,), -1, dtype=torch.int32, device=probs.device)

    BLOCK_K = triton.next_power_of_2(K)
    BLOCK_GROUPS = BLOCK_K // 32

    _permute_quantize_kernel[(num_tokens * topk,)](
        hidden_states, probs, routing_map,
        out_fp8, out_scale, out_probs, out_src_idx,
        expert_offsets,
        num_tokens, K, n_col_blocks,
        topk, local_expert_start, num_local_experts,
        REAL_GROUPS=scale_cols,
        BLOCK_K=BLOCK_K,
        BLOCK_GROUPS=BLOCK_GROUPS,
    )

    return out_fp8, out_scale.view(torch.float8_e8m0fnu), out_probs, out_src_idx


# =========================================================================== #
# Permute / unpermute kernels (BF16 path)
# =========================================================================== #
@triton.jit
def _permute_tokens_kernel(
    hidden_ptr, probs_ptr, routing_map_ptr,
    out_hidden_ptr, out_probs_ptr, out_src_idx_ptr, counters_ptr,
    num_tokens, hidden_dim, topk: tl.constexpr,
    local_expert_start, num_local_experts: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Permute tokens into expert-grouped order."""
    pair = tl.program_id(0)
    tok = pair // topk
    k = pair % topk
    if tok >= num_tokens:
        return
    eid = tl.load(routing_map_ptr + tok * topk + k)
    lid = eid - local_expert_start
    if lid < 0 or lid >= num_local_experts:
        return
    pos = tl.atomic_add(counters_ptr + lid, 1)
    for h in tl.range(0, hidden_dim, BLOCK_H):
        o = h + tl.arange(0, BLOCK_H)
        m = o < hidden_dim
        tl.store(out_hidden_ptr + pos * hidden_dim + o,
                 tl.load(hidden_ptr + tok * hidden_dim + o, mask=m), mask=m)
    tl.store(out_probs_ptr + pos, tl.load(probs_ptr + tok * topk + k))
    tl.store(out_src_idx_ptr + pos, tok)


def permute_tokens(
    hidden_states: torch.Tensor, probs: torch.Tensor,
    routing_map: torch.Tensor, expert_offsets: torch.Tensor,
    local_expert_start: int, num_local_experts: int,
    output_size: int = 0,
) -> tuple:
    """Permute tokens into expert-grouped order.

    NOTE: expert_offsets is mutated in-place via atomic adds.
    """
    num_tokens, hidden_dim = hidden_states.shape
    topk = probs.shape[1]
    if output_size <= 0:
        output_size = num_tokens * min(topk, num_local_experts)
    out_h = torch.empty(output_size, hidden_dim, dtype=hidden_states.dtype, device=hidden_states.device)
    out_p = torch.empty(output_size, dtype=probs.dtype, device=probs.device)
    out_s = torch.full((output_size,), -1, dtype=torch.int32, device=probs.device)
    BLOCK_H = min(triton.next_power_of_2(hidden_dim), 1024)
    _permute_tokens_kernel[(num_tokens * topk,)](
        hidden_states, probs, routing_map, out_h, out_p, out_s,
        expert_offsets, num_tokens, hidden_dim, topk,
        local_expert_start, num_local_experts, BLOCK_H=BLOCK_H,
    )
    return out_h, out_p, out_s


@triton.jit
def _unpermute_tokens_kernel(
    expert_out_ptr, probs_ptr, src_idx_ptr, output_ptr,
    hidden_dim, BLOCK_H: tl.constexpr,
):
    """Accumulate weighted expert outputs back to original token positions."""
    row = tl.program_id(0)
    tok = tl.load(src_idx_ptr + row)
    if tok < 0:
        return
    prob = tl.load(probs_ptr + row)
    for h in tl.range(0, hidden_dim, BLOCK_H):
        o = h + tl.arange(0, BLOCK_H)
        m = o < hidden_dim
        v = tl.load(expert_out_ptr + row * hidden_dim + o, mask=m)
        tl.atomic_add(output_ptr + tok * hidden_dim + o, v * prob, mask=m)


def unpermute_tokens(
    expert_output: torch.Tensor, permuted_probs: torch.Tensor,
    source_token_indices: torch.Tensor, num_tokens: int,
) -> torch.Tensor:
    """Unpermute expert outputs back to original token order."""
    output_size, hidden_dim = expert_output.shape
    output = torch.zeros(num_tokens, hidden_dim, dtype=expert_output.dtype, device=expert_output.device)
    BLOCK_H = min(triton.next_power_of_2(hidden_dim), 1024)
    _unpermute_tokens_kernel[(output_size,)](
        expert_output, permuted_probs, source_token_indices,
        output, hidden_dim, BLOCK_H=BLOCK_H,
    )
    return output


# =========================================================================== #
# Padding-aware activation kernels
# =========================================================================== #
@triton.jit
def _squared_relu_kernel(
    input_ptr, output_ptr, src_idx_ptr, M, N,
    skip_padding: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Squared ReLU that optionally skips padding rows (source_indices == -1)."""
    row = tl.program_id(0)
    if skip_padding:
        if tl.load(src_idx_ptr + row) < 0:
            return
    for n in tl.range(0, N, BLOCK_N):
        o = n + tl.arange(0, BLOCK_N)
        m = o < N
        x = tl.load(input_ptr + row * N + o, mask=m)
        r = tl.maximum(x, 0.0)
        tl.store(output_ptr + row * N + o, r * r, mask=m)


def padded_squared_relu(
    x: torch.Tensor, source_indices: torch.Tensor, skip_padding: bool = True,
) -> torch.Tensor:
    """Squared ReLU activation.

    Args:
        skip_padding: If True (default), skip rows where source_indices == -1.
            Set to False to run on all rows including padding.
    """
    M, N = x.shape
    out = torch.zeros(M, N, dtype=x.dtype, device=x.device)
    BLOCK_N = min(triton.next_power_of_2(N), 1024)
    _squared_relu_kernel[(M,)](
        x, out, source_indices, M, N, skip_padding, BLOCK_N=BLOCK_N,
    )
    return out


@triton.jit
def _swiglu_kernel(
    input_ptr, output_ptr, src_idx_ptr, M, N,
    skip_padding: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """SwiGLU activation that optionally skips padding rows (source_indices == -1)."""
    row = tl.program_id(0)
    if skip_padding:
        if tl.load(src_idx_ptr + row) < 0:
            return
    for n in tl.range(0, N, BLOCK_N):
        o = n + tl.arange(0, BLOCK_N)
        m = o < N
        gate = tl.load(input_ptr + row * 2 * N + o, mask=m)
        up = tl.load(input_ptr + row * 2 * N + N + o, mask=m)
        tl.store(output_ptr + row * N + o,
                 tl.sigmoid(gate.to(tl.float32)).to(gate.dtype) * gate * up, mask=m)


def padded_swiglu(
    x: torch.Tensor, source_indices: torch.Tensor, skip_padding: bool = True,
) -> torch.Tensor:
    """SwiGLU activation.

    Args:
        skip_padding: If True (default), skip rows where source_indices == -1.
            Set to False to run on all rows including padding.
    """
    M = x.shape[0]
    N = x.shape[1] // 2
    out = torch.zeros(M, N, dtype=x.dtype, device=x.device)
    BLOCK_N = min(triton.next_power_of_2(N), 1024)
    _swiglu_kernel[(M,)](
        x, out, source_indices, M, N, skip_padding, BLOCK_N=BLOCK_N,
    )
    return out


# =========================================================================== #
# Fused activation + MXFP8 quantize kernels
# =========================================================================== #
@triton.jit
def _squared_relu_quantize_kernel(
    input_ptr, out_fp8_ptr, out_scale_ptr, src_idx_ptr,
    K,
    n_col_blocks,
    skip_padding: tl.constexpr,
    REAL_GROUPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    """Fused squared ReLU + MXFP8 quantize + swizzle in one kernel.

    Grid: (M,) — one program per row.
    Reads BF16 FC1 output, applies squared ReLU, quantizes to FP8,
    writes FP8 data + swizzled scales in place.
    """
    row = tl.program_id(0)
    if skip_padding:
        if tl.load(src_idx_ptr + row) < 0:
            return

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


def squared_relu_and_quantize(
    x: torch.Tensor,
    source_indices: torch.Tensor,
    skip_padding: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused squared ReLU + MXFP8 quantize + swizzle.

    Reads BF16 FC1 output, applies squared ReLU, quantizes to FP8 with
    swizzled scales. Single kernel replaces padded_squared_relu + mxfp8_quantize.

    Returns:
        (out_fp8, out_scale):
            out_fp8: [M, K] float8_e4m3fn
            out_scale: 1D swizzled e8m0 scales
    """
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

    _squared_relu_quantize_kernel[(M,)](
        x, out_fp8, out_scale, source_indices,
        K, n_col_blocks,
        skip_padding,
        REAL_GROUPS=scale_cols,
        BLOCK_K=BLOCK_K,
        BLOCK_GROUPS=BLOCK_GROUPS,
    )

    return out_fp8, out_scale.view(torch.float8_e8m0fnu)


# =========================================================================== #
# MXFP8 helpers
# =========================================================================== #
def _mxfp8_grouped_gemm(
    act: MXFP8Tensor, weight: MXFP8Tensor, offs: torch.Tensor,
) -> torch.Tensor:
    """MXFP8 scaled_grouped_mm with pre-quantized activations and weights."""
    return scaled_grouped_mm(
        act.data, weight.data.transpose(1, 2),
        act.scale_2d(), _RECIPE,
        weight.scale, _RECIPE,
        swizzle_a=_SWIZZLE, swizzle_b=_SWIZZLE,
        offs=offs, output_dtype=torch.bfloat16,
    )


def _bf16_grouped_mm(
    x_bf16: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor,
) -> torch.Tensor:
    """BF16 grouped GEMM using torch.nn.functional.grouped_mm."""
    assert x_bf16.dtype == torch.bfloat16, f"Expected bf16 input, got {x_bf16.dtype}"
    return grouped_mm(x_bf16, weight.transpose(1, 2), offs=offs)


# =========================================================================== #
# Fused MoE API
# =========================================================================== #
def _get_activation_func(
    activation_type: ActivationType, skip_padding: bool,
) -> Callable:
    """Resolve ActivationType enum to a concrete kernel with skip_padding bound."""
    if activation_type == ActivationType.SWIGLU:
        return lambda x, si: padded_swiglu(x, si, skip_padding=skip_padding)
    elif activation_type == ActivationType.SQUARED_RELU:
        return lambda x, si: padded_squared_relu(x, si, skip_padding=skip_padding)
    else:
        raise ValueError(f"Unsupported activation type: {activation_type}")


def mcore_fused_moe(
    hidden_states: torch.Tensor,
    routing_map: torch.Tensor,
    probs: torch.Tensor,
    fc1_weight,
    fc2_weight,
    activation_type: ActivationType,
    num_local_experts: int,
    local_expert_start: int,
    expert_alignment: int = 128,
    skip_padding: bool = True,
    fuse_quant: bool = False,
) -> torch.Tensor:
    """Fused MoE: permute -> FC1 -> activation -> FC2 -> unpermute.

    Supports both BF16 weights (grouped_mm) and MXFP8 weights
    (scaled_grouped_mm with runtime activation quantization).

    Args:
        hidden_states: [num_tokens, hidden_size] BF16 input.
        routing_map: [num_tokens, topk] int expert assignments.
        probs: [num_tokens, topk] float32 routing probabilities.
        fc1_weight: stacked weight for FC1. Either:
            - torch.Tensor [num_experts, out_features, hidden_size] for BF16
            - MXFP8Tensor for MXFP8
        fc2_weight: stacked weight for FC2. Same type as fc1_weight.
        activation_type: ActivationType enum (SWIGLU or SQUARED_RELU).
        num_local_experts: number of experts on this rank.
        local_expert_start: first global expert index on this rank.
        expert_alignment: per-expert token alignment (default 128).
        skip_padding: if True (default), activation kernels skip rows where
            source_indices == -1. Set to False to run on all rows.
        fuse_quant: if True, fuse MXFP8 quantization with permute (FC1 input)
            and with activation (FC2 input, squared_relu only). MXFP8 only.

    Returns:
        [num_tokens, hidden_size] BF16 output.
    """
    assert hidden_states.dtype == torch.bfloat16, (
        f"mcore_fused_moe requires bf16 input, got {hidden_states.dtype}"
    )

    num_tokens = hidden_states.shape[0]
    use_mxfp8 = isinstance(fc1_weight, MXFP8Tensor)

    activation_func = _get_activation_func(activation_type, skip_padding)

    # --- Common: compute expert offsets ---
    tokens_per_expert = compute_local_tokens_per_expert(
        routing_map, local_expert_start, num_local_experts,
    )
    padded_exc, padded_inc = compute_expert_offsets(
        tokens_per_expert, alignment=expert_alignment,
    )
    topk = probs.shape[1]
    max_output_size = (
        num_tokens * min(topk, num_local_experts)
        + expert_alignment * num_local_experts
    )
    offs = padded_inc

    if use_mxfp8:
        assert HAVE_SCALED_GMM, "torch.nn.functional.scaled_grouped_mm not available"

        # --- FC1 input: permute + quantize ---
        if fuse_quant:
            permuted_fp8, permuted_scale, permuted_probs, source_indices = (
                permute_and_quantize(
                    hidden_states, probs, routing_map, padded_exc,
                    local_expert_start, num_local_experts,
                    output_size=max_output_size,
                )
            )
            if permuted_fp8.nelement() == 0:
                return torch.zeros_like(hidden_states)
            permuted_act = MXFP8Tensor(data=permuted_fp8, scale=permuted_scale)
        else:
            permuted_hidden, permuted_probs, source_indices = permute_tokens(
                hidden_states, probs, routing_map, padded_exc,
                local_expert_start, num_local_experts,
                output_size=max_output_size,
            )
            if permuted_hidden.nelement() == 0:
                return torch.zeros_like(hidden_states)
            permuted_act = MXFP8Tensor.from_bf16_torch(permuted_hidden)

        # --- FC1 ---
        fc1_output = _mxfp8_grouped_gemm(permuted_act, fc1_weight, offs)

        # --- FC2 input: activation + quantize ---
        if fuse_quant and activation_type == ActivationType.SQUARED_RELU:
            fc2_fp8, fc2_scale = squared_relu_and_quantize(
                fc1_output, source_indices, skip_padding=skip_padding,
            )
            fc2_act = MXFP8Tensor(data=fc2_fp8, scale=fc2_scale)
        else:
            activated = activation_func(fc1_output, source_indices)
            fc2_act = MXFP8Tensor.from_bf16_torch(activated)

        # --- FC2 ---
        fc2_output = _mxfp8_grouped_gemm(fc2_act, fc2_weight, offs)

    else:
        assert HAVE_GROUPED_MM, "torch.nn.functional.grouped_mm not available"

        # --- BF16 path: permute + grouped_mm ---
        permuted_hidden, permuted_probs, source_indices = permute_tokens(
            hidden_states, probs, routing_map, padded_exc,
            local_expert_start, num_local_experts,
            output_size=max_output_size,
        )

        if permuted_hidden.nelement() == 0:
            return torch.zeros_like(hidden_states)

        fc1_output = _bf16_grouped_mm(permuted_hidden, fc1_weight, offs)
        activated = activation_func(fc1_output, source_indices)
        fc2_output = _bf16_grouped_mm(activated, fc2_weight, offs)

    # --- Unpermute ---
    return unpermute_tokens(fc2_output, permuted_probs, source_indices, num_tokens)
