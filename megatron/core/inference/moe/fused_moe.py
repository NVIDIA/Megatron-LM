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
    from torch.nn.functional import grouped_mm

    HAVE_GROUPED_MM = True
except ImportError:
    HAVE_GROUPED_MM = False

def _ceil_div(a, b):
    return (a + b - 1) // b

from enum import Enum
from typing import Callable


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
    BLOCK_N: tl.constexpr,
):
    """Squared ReLU that optionally skips padding rows (source_indices == -1)."""
    row = tl.program_id(0)
    if tl.load(src_idx_ptr + row) < 0:
        return
    for n in tl.range(0, N, BLOCK_N):
        o = n + tl.arange(0, BLOCK_N)
        m = o < N
        x = tl.load(input_ptr + row * N + o, mask=m)
        r = tl.maximum(x, 0.0)
        tl.store(output_ptr + row * N + o, r * r, mask=m)


def padded_squared_relu(
    x: torch.Tensor, source_indices: torch.Tensor
) -> torch.Tensor:
    """Squared ReLU activation.

    Args:
    """
    M, N = x.shape
    out = torch.zeros(M, N, dtype=x.dtype, device=x.device)
    BLOCK_N = min(triton.next_power_of_2(N), 1024)
    _squared_relu_kernel[(M,)](
        x, out, source_indices, M, N, BLOCK_N=BLOCK_N,
    )
    return out


@triton.jit
def _swiglu_kernel(
    input_ptr, output_ptr, src_idx_ptr, M, N,
    BLOCK_N: tl.constexpr,
):
    """SwiGLU activation that optionally skips padding rows (source_indices == -1)."""
    row = tl.program_id(0)
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
    x: torch.Tensor, source_indices: torch.Tensor
) -> torch.Tensor:
    """SwiGLU activation.
    Args:
    """
    M = x.shape[0]
    N = x.shape[1] // 2
    out = torch.zeros(M, N, dtype=x.dtype, device=x.device)
    BLOCK_N = min(triton.next_power_of_2(N), 1024)
    _swiglu_kernel[(M,)](
        x, out, source_indices, M, N, BLOCK_N=BLOCK_N,
    )
    return out


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
    activation_type: ActivationType
) -> Callable:
    """Resolve ActivationType enum to a concrete kernel."""
    if activation_type == ActivationType.SWIGLU:
        return padded_swiglu
    elif activation_type == ActivationType.SQUARED_RELU:
        return padded_squared_relu
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
    expert_alignment: int = 32,
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

    Returns:
        [num_tokens, hidden_size] BF16 output.
    """
    assert hidden_states.dtype == torch.bfloat16, (
        f"mcore_fused_moe requires bf16 input, got {hidden_states.dtype}"
    )

    num_tokens = hidden_states.shape[0]
   

    activation_func = _get_activation_func(activation_type)

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