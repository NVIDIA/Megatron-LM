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

try:
    from torch.nn.functional import grouped_mm

    HAVE_GROUPED_MM = True
except ImportError:
    HAVE_GROUPED_MM = False


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

    Args:
        hidden_states: [num_tokens, hidden_size] BF16 input.
        routing_map: [num_tokens, topk] int expert assignments.
        probs: [num_tokens, topk] float32 routing probabilities.
        fc1_weight: [num_experts, out_features, hidden_size] BF16 stacked weight for FC1.
        fc2_weight: [num_experts, hidden_size, ffn_hidden] BF16 stacked weight for FC2.
        activation_type: ActivationType enum (SWIGLU or SQUARED_RELU).
        num_local_experts: number of experts on this rank.
        local_expert_start: first global expert index on this rank.
        expert_alignment: per-expert token alignment (default 32).

    Returns:
        [num_tokens, hidden_size] BF16 output.
    """
    assert hidden_states.dtype == torch.bfloat16, (
        f"mcore_fused_moe requires bf16 input, got {hidden_states.dtype}"
    )
    assert HAVE_GROUPED_MM, "torch.nn.functional.grouped_mm not available"

    num_tokens = hidden_states.shape[0]
    activation_func = _get_activation_func(activation_type)

    # --- Permute ---
    permuted_hidden, permuted_probs, permutation_map, offs = permute_tokens(
        hidden_states, probs, routing_map,
        local_expert_start, num_local_experts,
        alignment=expert_alignment,
    )

    if permuted_hidden.nelement() == 0:
        return torch.zeros_like(hidden_states)

    # --- FC1 -> activation -> FC2 ---
    fc1_output = _bf16_grouped_mm(permuted_hidden, fc1_weight, offs)
    activated = activation_func(fc1_output, permutation_map)
    fc2_output = _bf16_grouped_mm(activated, fc2_weight, offs)

    # --- Unpermute ---
    return unpermute_tokens(fc2_output, permuted_probs, permutation_map, num_tokens)
