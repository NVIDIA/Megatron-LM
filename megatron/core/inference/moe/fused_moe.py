# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused MoE: permute -> FC1 -> activation -> FC2 -> unpermute.

Supports BF16 weights with torch.nn.functional.grouped_mm.
All permutation logic is handled internally — callers invoke a single function.
"""

from enum import Enum
from typing import Callable

import torch

from megatron.core.inference.moe.activations import padded_squared_relu, padded_swiglu
from megatron.core.inference.moe.permute import permute_tokens, unpermute_tokens
from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

try:
    from torch.nn.functional import grouped_mm

    HAVE_GROUPED_MM = True
except ImportError:
    HAVE_GROUPED_MM = False

try:
    from torch.nn.functional import scaled_grouped_mm, ScalingType, SwizzleType
    HAVE_SCALED_GMM = True
except ImportError:
    HAVE_SCALED_GMM = False


class ActivationType(Enum):
    """Activation functions supported by mcore_fused_moe."""
    SQUARED_RELU = "squared_relu"
    SWIGLU = "swiglu"


def _bf16_grouped_mm(
    x_bf16: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor,
) -> torch.Tensor:
    """BF16 grouped GEMM using torch.nn.functional.grouped_mm."""
    assert x_bf16.dtype == torch.bfloat16, f"Expected bf16 input, got {x_bf16.dtype}"
    return grouped_mm(x_bf16, weight.transpose(1, 2), offs=offs)


def _mxfp8_grouped_mm(
    act: MXFP8Tensor, weight: MXFP8Tensor, offs: torch.Tensor,
) -> torch.Tensor:
    """MXFP8 scaled_grouped_mm with pre-quantized activations and weights."""
    return scaled_grouped_mm(
        act.data, weight.data.transpose(1, 2),
        act.scale_2d(), ScalingType.BlockWise1x32,
        weight.scale, ScalingType.BlockWise1x32,
        swizzle_a=SwizzleType.SWIZZLE_32_4_4, swizzle_b=SwizzleType.SWIZZLE_32_4_4,
        offs=offs, output_dtype=torch.bfloat16,
    )

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
) -> torch.Tensor:
    """Fused MoE: permute -> FC1 -> activation -> FC2 -> unpermute.

    Args:
        hidden_states: [num_tokens, hidden_size] BF16 input.
        routing_map: [num_tokens, topk] int expert assignments.
        probs: [num_tokens, topk] float32 routing probabilities.
        fc1_weight: stacked weight for FC1 (torch.Tensor for BF16, MXFP8Tensor for MXFP8).
        fc2_weight: stacked weight for FC2 (same type as fc1_weight).
        activation_type: ActivationType enum (SWIGLU or SQUARED_RELU).
        num_local_experts: number of experts on this rank.
        local_expert_start: first global expert index on this rank.

    Returns:
        [num_tokens, hidden_size] BF16 output.
    """
    assert hidden_states.dtype == torch.bfloat16, (
        f"mcore_fused_moe requires bf16 input, got {hidden_states.dtype}"
    )

    num_tokens = hidden_states.shape[0]
    use_mxfp8 = isinstance(fc1_weight, MXFP8Tensor)
    activation_func = _get_activation_func(activation_type)

    if use_mxfp8:
        assert HAVE_SCALED_GMM, "torch.nn.functional.scaled_grouped_mm not available"
        mm_fn = _mxfp8_grouped_mm
        # scaled_grouped_mm requires each expert's token count aligned to 32,
        # but swizzled MXFP8 scales require alignment to 128. Use 128 to
        # satisfy both constraints.
        expert_alignment = 128
    else:
        assert HAVE_GROUPED_MM, "torch.nn.functional.grouped_mm not available"
        mm_fn = _bf16_grouped_mm
        expert_alignment = 16

    # --- Permute ---
    permuted_hidden, permuted_probs, permutation_map, offs = permute_tokens(
        hidden_states, probs, routing_map,
        local_expert_start, num_local_experts,
        alignment=expert_alignment,
    )

    # --- FC1 -> activation -> FC2 ---
    if use_mxfp8: 
        permuted_hidden = MXFP8Tensor.from_bf16(permuted_hidden, backend="triton")
    fc1_output = mm_fn(permuted_hidden, fc1_weight, offs)
    activated = activation_func(fc1_output, permutation_map)
    if use_mxfp8: 
        activated = MXFP8Tensor.from_bf16(activated, backend="triton")
    fc2_output = mm_fn(activated, fc2_weight, offs)

    # --- Unpermute ---
    return unpermute_tokens(fc2_output, permuted_probs, permutation_map, num_tokens)
