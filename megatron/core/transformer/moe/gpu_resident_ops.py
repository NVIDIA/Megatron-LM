# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
GPU-resident operations for CUDA-graph compatible MoE inference.

This module provides GPU-resident implementations of AlltoAll and GroupedGEMM
operations that accept device tensors for split sizes, eliminating host
synchronization points required for CUDA graph compatibility.
"""

from typing import Optional

import torch


def gpu_resident_all_to_all(
    process_group,
    input_tensor: torch.Tensor,
    output_split_sizes: torch.Tensor,
    input_split_sizes: torch.Tensor,
) -> torch.Tensor:
    """
    GPU-resident AlltoAll that accepts device tensors for split sizes.

    This function eliminates the host synchronization bottleneck present in
    the standard torch.distributed.all_to_all by accepting split sizes as
    GPU tensors instead of CPU lists.

    Args:
        process_group: The process group for communication
        input_tensor: [sum(input_split_sizes), ...] tensor to send
        output_split_sizes: [world_size] GPU tensor - number of elements to receive from each rank
        input_split_sizes: [world_size] GPU tensor - number of elements to send to each rank

    Returns:
        output_tensor: [sum(output_split_sizes), ...] received tensor

    Example:
        >>> # Instead of CPU lists:
        >>> # output_splits = [100, 200, 150]  # CPU list
        >>> # input_splits = [80, 120, 200]    # CPU list
        >>> # output = all_to_all(group, input, output_splits, input_splits)
        >>>
        >>> # Use GPU tensors:
        >>> output_splits = torch.tensor([100, 200, 150], device='cuda')  # GPU
        >>> input_splits = torch.tensor([80, 120, 200], device='cuda')    # GPU
        >>> output = gpu_resident_all_to_all(group, input, output_splits, input_splits)

    Implementation notes:
        - This is a placeholder for your GPU-resident AlltoAll implementation
        - The actual implementation should avoid any .item(), .tolist(), or .cpu() calls
        - Split sizes must remain on GPU throughout the operation
        - Should support CUDA graph capture
    """
    # TODO: Replace with actual GPU-resident AlltoAll implementation
    # For now, this is a placeholder showing the expected interface
    raise NotImplementedError(
        "gpu_resident_all_to_all requires a custom implementation. "
        "This placeholder shows the expected API: accepts GPU tensors for split sizes."
    )


def gpu_resident_grouped_gemm(
    input: torch.Tensor,
    weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    use_fp8: bool = False,
) -> torch.Tensor:
    """
    GPU-resident GroupedGEMM that accepts device tensor for expert splits.

    This function provides a CUDA-graph compatible grouped GEMM by accepting
    tokens_per_expert as a GPU tensor and computing offsets on-device.

    Args:
        input: [total_tokens, K] input tensor
        weights: [num_experts, K, N] or [num_experts*K, N] weight tensor
        tokens_per_expert: [num_experts] GPU tensor - token count per expert
        use_fp8: Whether to use FP8 computation (if available)

    Returns:
        output: [total_tokens, N] output tensor

    Example:
        >>> # Instead of CPU tokens_per_expert:
        >>> # tokens_per_expert_cpu = tokens_per_expert.cpu()  # Sync!
        >>> # offs = tokens_per_expert_cpu.cumsum(0).cuda()    # Another sync!
        >>> # output = torch._grouped_mm(input, weights, offs=offs)
        >>>
        >>> # Use GPU-resident version:
        >>> output = gpu_resident_grouped_gemm(input, weights, tokens_per_expert)

    Implementation notes:
        - This is a placeholder for your GPU-resident GroupedGEMM implementation
        - Should compute cumsum(tokens_per_expert) on GPU without host sync
        - Must keep all tensors GPU-resident throughout
        - Should support CUDA graph capture
        - Can wrap torch._grouped_mm or use custom kernel
    """
    # TODO: Replace with actual GPU-resident GroupedGEMM implementation
    # For now, this is a placeholder showing the expected interface

    # Example of what the implementation might look like:
    # offs = tokens_per_expert.cumsum(0).to(torch.int32)  # No .cuda() needed!
    # return torch._grouped_mm(input, weights, offs=offs)

    raise NotImplementedError(
        "gpu_resident_grouped_gemm requires a custom implementation. "
        "This placeholder shows the expected API: accepts GPU tensor for tokens_per_expert."
    )
