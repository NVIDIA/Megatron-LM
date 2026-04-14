# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused MoE: permute -> FC1 -> activation -> FC2 -> unpermute.

Supports BF16 weights with torch.nn.functional.grouped_mm.
All permutation logic is handled internally — callers invoke a single function.
"""

from enum import Enum
from typing import Callable, Optional

import torch

from megatron.core.inference.moe.activations import (
    padded_squared_relu,
    squared_relu_and_quantize_mxfp8,
)
from megatron.core.inference.moe.pad import pad_to_alignment, unpad_from_alignment
from megatron.core.inference.moe.permute import (
    permute_and_quantize_mxfp8,
    permute_tokens,
    unpermute_tokens,
)
from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

try:
    from torch.nn.functional import grouped_mm

    HAVE_GROUPED_MM = True
except ImportError:
    HAVE_GROUPED_MM = False

try:
    from torch.nn.functional import ScalingType, SwizzleType, scaled_grouped_mm

    HAVE_SCALED_GMM = True
except ImportError:
    HAVE_SCALED_GMM = False


class ActivationType(Enum):
    """Activation functions supported by mcore_fused_moe."""

    SQUARED_RELU = "squared_relu"


def _bf16_grouped_mm(
    x_bf16: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    """BF16 grouped GEMM using torch.nn.functional.grouped_mm."""
    assert x_bf16.dtype == torch.bfloat16, f"Expected bf16 input, got {x_bf16.dtype}"
    return grouped_mm(x_bf16, weight.transpose(1, 2), offs=offs)


def _mxfp8_grouped_mm(act: MXFP8Tensor, weight: MXFP8Tensor, offs: torch.Tensor) -> torch.Tensor:
    """MXFP8 scaled_grouped_mm with pre-quantized activations and weights."""
    return scaled_grouped_mm(
        act.data,
        weight.data.transpose(1, 2),
        act.scale_2d(),
        ScalingType.BlockWise1x32,
        weight.scale,
        ScalingType.BlockWise1x32,
        swizzle_a=SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=SwizzleType.SWIZZLE_32_4_4,
        offs=offs,
        output_dtype=torch.bfloat16,
    )


def _get_activation_func(activation_type: ActivationType, fused_quant: bool = False) -> Callable:
    """Resolve ActivationType enum to a concrete kernel.

    If fused_quant=True, returns the fused activation + MXFP8 quantize kernel.
    """
    if activation_type == ActivationType.SQUARED_RELU:
        return squared_relu_and_quantize_mxfp8 if fused_quant else padded_squared_relu
    else:
        raise ValueError(f"Unsupported activation type: {activation_type}")


def mcore_fused_moe(
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
    fc1_weight,
    fc2_weight,
    activation_type: ActivationType,
    num_local_experts: int,
    local_expert_start: int,
    routing_map: Optional[torch.Tensor] = None,
    tokens_per_expert: Optional[torch.Tensor] = None,
    skip_permute: bool = False,
    disable_fused_quant_kernels: bool = False,
    load_balance: bool = False,
    num_experts: Optional[int] = None,
) -> torch.Tensor:
    """Fused MoE: [permute ->] pad -> FC1 -> activation -> FC2 -> unpad [-> unpermute].

    Two modes:
    - skip_permute=False (default): tokens are unpermuted. Requires routing_map.
      Performs full permute -> compute -> unpermute.
    - skip_permute=True: tokens are already permuted by the dispatcher. Requires
      tokens_per_expert. Pads to alignment, computes, then unpads. Probs are
      applied during unpad.

    Unless disable_fused_quant_kernels=True, when weights are MXFP8, uses fused
    kernels that combine permute/activation with MXFP8 quantization into single
    kernel launches.

    Args:
        hidden_states: [num_tokens, hidden_size] BF16 input.
        probs: routing probabilities. Shape is [num_tokens, topk] when
            skip_permute=False, or [num_tokens] (already gathered) when
            skip_permute=True.
        fc1_weight: stacked weight for FC1 (torch.Tensor for BF16, MXFP8Tensor for MXFP8).
        fc2_weight: stacked weight for FC2 (same type as fc1_weight).
        activation_type: ActivationType enum (SQUARED_RELU).
        num_local_experts: number of experts on this rank.
        local_expert_start: first global expert index on this rank.
        routing_map: [num_tokens, topk] int expert assignments. Required when skip_permute=False.
        tokens_per_expert: [num_local_experts] int32 token counts. Required when skip_permute=True.
        skip_permute: if True, skip permute/unpermute (tokens already in expert order).
        disable_fused_quant_kernels: if True, disable fused permute+quantize and
            activation+quantize kernels for MXFP8, using separate launches instead.
            Useful for debugging. Ignored when weights are BF16.

    Returns:
        [num_tokens, hidden_size] BF16 output.
    """
    assert (
        hidden_states.dtype == torch.bfloat16
    ), f"mcore_fused_moe requires bf16 input, got {hidden_states.dtype}"

    num_tokens = hidden_states.shape[0]
    use_mxfp8 = isinstance(fc1_weight, MXFP8Tensor)
    # Fused quant kernels only apply to MXFP8 path
    use_fused_quant = use_mxfp8 and not disable_fused_quant_kernels

    if use_mxfp8:
        assert (
            HAVE_SCALED_GMM
        ), "torch.nn.functional.scaled_grouped_mm not available. Install PyTorch 2.10+."
        mm_fn = _mxfp8_grouped_mm
        # scaled_grouped_mm requires each expert's token count aligned to 32,
        # but swizzled MXFP8 scales require alignment to 128. Use 128 to
        # satisfy both constraints.
        expert_alignment = 128
    else:
        assert (
            HAVE_GROUPED_MM
        ), "torch.nn.functional.grouped_mm not available. Install PyTorch 2.10+."
        mm_fn = _bf16_grouped_mm
        expert_alignment = 16

    activation_func = _get_activation_func(activation_type, fused_quant=use_fused_quant)

    # --- Pre-processing: permute or pad ---
    if skip_permute:
        assert tokens_per_expert is not None, "tokens_per_expert is required when skip_permute=True"
        tokens_per_expert = tokens_per_expert.cuda().int()
        assert routing_map is None, "routing_map must be None when skip_permute=True"
        hidden_states, permutation_map, offs = pad_to_alignment(
            hidden_states, tokens_per_expert, expert_alignment
        )
        permuted_probs = None

    else:
        assert routing_map is not None, "routing_map is required when skip_permute=False"
        if use_fused_quant:
            # Fused permute + MXFP8 quantize: single kernel produces MXFP8Tensor
            hidden_states, permuted_probs, permutation_map, offs = permute_and_quantize_mxfp8(
                hidden_states,
                probs,
                routing_map,
                local_expert_start,
                num_local_experts,
                alignment=expert_alignment,
                load_balance=load_balance,
                num_experts=num_experts,
            )
        else:
            hidden_states, permuted_probs, permutation_map, offs = permute_tokens(
                hidden_states,
                probs,
                routing_map,
                local_expert_start,
                num_local_experts,
                alignment=expert_alignment,
                load_balance=load_balance,
                num_experts=num_experts,
            )

    # --- FC1 -> activation -> FC2 ---
    # Quantize if MXFP8 path and hidden_states not already quantized (fused permute+quant
    # produces MXFP8Tensor directly; skip_permute path always needs separate quant).
    needs_quant = use_mxfp8 and not isinstance(hidden_states, MXFP8Tensor)
    if needs_quant:
        hidden_states = MXFP8Tensor.from_bf16(hidden_states, backend="triton")
    fc1_output = mm_fn(hidden_states, fc1_weight, offs)

    activation_out = activation_func(fc1_output, permutation_map)
    # Fused activation+quant returns MXFP8Tensor; otherwise quantize separately.
    if use_mxfp8 and not isinstance(activation_out, MXFP8Tensor):
        activation_out = MXFP8Tensor.from_bf16(activation_out, backend="triton")
    fc2_output = mm_fn(activation_out, fc2_weight, offs)
    # --- Post-processing: unpermute or unpad ---
    if skip_permute:
        probs_1d = probs.squeeze(-1) if probs.dim() > 1 else probs
        return unpad_from_alignment(fc2_output, permutation_map, num_tokens, probs=probs_1d)
    else:
        return unpermute_tokens(fc2_output, permuted_probs, permutation_map, num_tokens)
