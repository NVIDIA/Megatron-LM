# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch
import triton
import triton.language as tl
from megatron.core.transformer.moe.moe_utils import topk_softmax_with_capacity

@triton.jit
def fused_topk_softmax_forward_kernel_with_group(
    # temp
    scores_temp_ptr,
    group_view_temp_ptr,
    top_values_temp_ptr,
    top_indices_temp2_ptr,
    group_mask_temp_ptr,
    # input
    logits_ptr,
    expert_bias_ptr,
    has_expert_bias: tl.constexpr,
    # output
    topk_masked_gates_ptr,
    topk_map_ptr,
    top_indices_ptr,
    top_scores_ptr,
    # params
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    num_groups: tl.constexpr,
    experts_per_group: tl.constexpr,
    topk: tl.constexpr,
    group_topk: tl.constexpr,
    topk_per_group: tl.constexpr,
    scaling_factor: tl.constexpr,
    score_function: tl.constexpr,
    use_pre_softmax: tl.constexpr,
    # block size
    BLOCK_SIZE_NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_NUM_GROUPS: tl.constexpr,
    BLOCK_SIZE_EXPERTS_PER_GROUP: tl.constexpr,
    BLOCK_SIZE_TOPK: tl.constexpr,
    BLOCK_SIZE_GROUP_TOPK: tl.constexpr,
    BLOCK_SIZE_TOPK_PER_GROUP: tl.constexpr
):
    pid = tl.program_id(axis=0)

    # Initialize offsets and masks for different dimensions
    expert_offs = tl.arange(0, BLOCK_SIZE_NUM_EXPERTS)
    expert_mask = expert_offs < num_experts

    group_offs = tl.arange(0, BLOCK_SIZE_NUM_GROUPS)

    group_topk_offs = tl.arange(0, BLOCK_SIZE_GROUP_TOPK)
    group_topk_mask = group_topk_offs < group_topk

    topk_offs = tl.arange(0, BLOCK_SIZE_TOPK)
    topk_mask = topk_offs < topk

    # Load input logits
    logits = tl.load(logits_ptr + pid * num_experts + expert_offs, mask=expert_mask)

    # Apply sigmoid first
    logits_fp32 = logits.to(tl.float32)
    logits_fp32 = tl.sigmoid(logits_fp32)
    scores = logits_fp32.to(logits.dtype)
    tl.store(scores_temp_ptr + pid * num_experts + expert_offs, scores, mask=expert_mask)
    if has_expert_bias:
        expert_bias = tl.load(expert_bias_ptr + expert_offs, mask=expert_mask)
        scores_for_routing = scores + expert_bias
    else:
        scores_for_routing = scores
    # Then find topk 
    tl.store(group_view_temp_ptr + pid * num_experts + expert_offs, scores_for_routing, mask=expert_mask)
    tl.debug_barrier()

    # Reshape scores into groups
    offs_m = tl.arange(0, BLOCK_SIZE_NUM_GROUPS)[:, None]
    offs_n = tl.arange(0, BLOCK_SIZE_EXPERTS_PER_GROUP)[None, :]
    indices = offs_m * experts_per_group + offs_n
    mask = (offs_m < num_groups) & (offs_n < experts_per_group)
    group_view = tl.load(group_view_temp_ptr + pid * num_experts + indices, mask=mask)

    # Sort within each group
    data = group_view
    for i in range(topk_per_group):
        max_idx = tl.argmax(data, axis=1)
        max_val = tl.max(data, axis=1)
        max_idx_reshaped = max_idx[:, None]
        max_val_reshaped = max_val[:, None]
        index = pid * num_groups * topk_per_group + offs_m * topk_per_group + i
        tl.store(top_values_temp_ptr + index, max_val_reshaped)
        data_mask = (offs_n == max_idx[:, None])
        data = tl.where(data_mask, -float('inf'), data)

    tl.debug_barrier()

    # Calculate group scores
    offs_n2 = tl.arange(0, BLOCK_SIZE_TOPK_PER_GROUP)[None, :]
    top_values_offs = pid * num_groups * topk_per_group + offs_m * topk_per_group + offs_n2
    top_values_mask = (offs_m < num_groups) & (offs_n2 < topk_per_group)
    top_values = tl.load(top_values_temp_ptr + top_values_offs, mask=top_values_mask)

    group_scores = tl.sum(top_values, axis=-1)

    # Select top groups
    data = group_scores
    for i in range(group_topk):
        max_idx = tl.argmax(data, axis=0)
        tl.store(top_indices_temp2_ptr + pid * group_topk + i, max_idx)
        data = tl.where(group_offs == max_idx, -float('inf'), data)

    tl.debug_barrier()

    group_idx = tl.load(top_indices_temp2_ptr + pid * group_topk + group_topk_offs, mask=group_topk_mask)
    
    # Create group mask
    ones = tl.full([BLOCK_SIZE_GROUP_TOPK], 1, logits_ptr.dtype.element_ty)
    tl.store(group_mask_temp_ptr + pid * num_groups + group_idx, ones, mask=group_topk_mask)

    tl.debug_barrier()

    # Apply group mask to scores
    expert_group_idx = expert_offs // experts_per_group
    score_mask = tl.load(group_mask_temp_ptr + pid * num_groups + expert_group_idx, mask=expert_mask)

    score_mask_bool = score_mask != 0
    masked_scores = tl.where(score_mask_bool, scores_for_routing, -float('inf'))

    # Find topk experts
    data = masked_scores
    for i in range(topk):
        max_idx = tl.argmax(data, axis=0)
        tl.store(top_indices_ptr + pid * topk + i, max_idx)
        data = tl.where(expert_offs == max_idx, -float('inf'), data)

    tl.debug_barrier()

    top_indices = tl.load(top_indices_ptr + pid * topk + topk_offs, mask=topk_mask)
    top_scores = tl.load(scores_temp_ptr + pid * num_experts + top_indices, mask=topk_mask)

    tl.store(top_scores_ptr + pid * topk + topk_offs, top_scores, mask=topk_mask)

    # Normalize probabilities
    if topk > 1:
        sum_top_scores = tl.sum(top_scores, axis=0)
        probs = top_scores / (sum_top_scores + 1e-20)
    else:
        probs = top_scores

    # Apply scaling factor if specified
    if scaling_factor != 0.0:
        probs = probs * scaling_factor

    # Store final results
    tl.store(topk_masked_gates_ptr + pid * num_experts + top_indices, probs, mask=topk_mask)
    ones = tl.full([BLOCK_SIZE_TOPK], 1, logits_ptr.dtype.element_ty)
    tl.store(topk_map_ptr + pid * num_experts + top_indices, ones, mask=topk_mask)

@triton.jit
def fused_topk_softmax_forward_kernel_without_group(
    # temp
    scores_temp_ptr,
    # input
    logits_ptr,
    expert_bias_ptr,
    has_expert_bias: tl.constexpr,
    # output
    topk_masked_gates_ptr,
    topk_map_ptr,
    top_indices_ptr,
    top_scores_ptr,
    # params
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    scaling_factor: tl.constexpr,
    score_function: tl.constexpr,
    use_pre_softmax: tl.constexpr,
    # block size
    BLOCK_SIZE_NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_TOPK: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # Initialize offsets and masks
    expert_offs = tl.arange(0, BLOCK_SIZE_NUM_EXPERTS)
    expert_mask = expert_offs < num_experts

    topk_offs = tl.arange(0, BLOCK_SIZE_TOPK)
    topk_mask = topk_offs < topk

    # Load input logits
    logits = tl.load(logits_ptr + pid * num_experts + expert_offs, mask=expert_mask)

    # Apply sigmoid first
    logits_fp32 = logits.to(tl.float32)
    logits_fp32 = tl.sigmoid(logits_fp32)
    scores = logits_fp32.to(logits.dtype)
    tl.store(scores_temp_ptr + pid * num_experts + expert_offs, scores, mask=expert_mask)
    tl.debug_barrier()

    if has_expert_bias:
        expert_bias = tl.load(expert_bias_ptr + expert_offs, mask=expert_mask)
        scores_for_routing = scores + expert_bias
        
    else:
        scores_for_routing = scores
        
    # topk logits (num_experts -> topk)
    data = scores_for_routing
    for i in range(topk):
        max_idx = tl.argmax(data, axis=0)
        tl.store(top_indices_ptr + pid * topk + i, max_idx)
        data = tl.where(expert_offs == max_idx, -float('inf'), data)

    tl.debug_barrier()
    top_indices = tl.load(top_indices_ptr + pid * topk + topk_offs, mask=topk_mask)
    scores = tl.load(scores_temp_ptr + pid * num_experts + top_indices, mask=topk_mask)

    tl.store(top_scores_ptr + pid * topk + topk_offs, scores, mask=topk_mask)

    # compute probs
    if topk > 1:
        sum_scores = tl.sum(scores) + 1e-20
        probs = scores / sum_scores
    else:
        probs = scores

    # Apply scaling factor if specified
    if scaling_factor != 0.0:
        probs = probs * scaling_factor

    # Store final results
    tl.store(topk_masked_gates_ptr + pid * num_experts + top_indices, probs, mask=topk_mask)
    ones = tl.full([BLOCK_SIZE_TOPK], 1, logits_ptr.dtype.element_ty)
    tl.store(topk_map_ptr + pid * num_experts + top_indices, ones, mask=topk_mask)

@triton.jit
def fused_topk_softmax_backward_kernel(
    # input
    logits_ptr,
    top_indices_ptr,
    top_scores_ptr,
    grad_topk_masked_gates_ptr,
    # output
    grad_logits_ptr,
    # params
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    scaling_factor: tl.constexpr,
    BLOCK_SIZE_NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_TOPK: tl.constexpr
):
    pid = tl.program_id(axis=0)

    # offs & mask
    expert_offs = tl.arange(0, BLOCK_SIZE_NUM_EXPERTS)
    expert_mask = expert_offs < num_experts

    topk_offs = tl.arange(0, BLOCK_SIZE_TOPK)
    topk_mask = topk_offs < topk

    # load top_indices
    top_indices = tl.load(top_indices_ptr + pid * topk + topk_offs, mask=topk_mask)

    # compute grad_probs
    grad_probs = tl.load(grad_topk_masked_gates_ptr + pid * num_experts + top_indices, mask=topk_mask)

    # scaling_factor grad_probs
    if scaling_factor != 0.0:
        grad_probs = grad_probs * scaling_factor

    # compute grad_scores
    if topk > 1:
        top_scores = tl.load(top_scores_ptr + pid * topk + topk_offs, mask=topk_mask)
        sum_top_scores = tl.sum(top_scores, -1) + 1e-20
        grad_scores = (grad_probs * sum_top_scores - tl.sum((top_scores * grad_probs), -1)) / (sum_top_scores * sum_top_scores)
    else:
        grad_scores = grad_probs

    # compute grad_logits(1)
    tl.store(grad_logits_ptr + pid * num_experts + top_indices, grad_scores, mask=topk_mask)

    # compute sig
    logits = tl.load(logits_ptr + pid * num_experts + expert_offs, mask=expert_mask)
    logits_fp32 = logits.to(tl.float32)
    sig = tl.sigmoid(logits_fp32)

    # compute grad_logits(2)
    tl.debug_barrier()
    grad_logits = tl.load(grad_logits_ptr + pid * num_experts + expert_offs, mask=expert_mask)
    grad_logits *= sig * (1 - sig)

    tl.store(grad_logits_ptr + pid * num_experts + expert_offs, grad_logits, mask=expert_mask)


class TopkSoftmax(torch.autograd.Function):
    """A custom autograd function for top-k gating with optional capacity constraints."""

    @staticmethod
    def forward(
        ctx,
        logits,
        topk,
        capacity_factor,
        pad_to_capacity,
        drop_policy,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        deterministic_mode,
        score_function,
        expert_bias,
    ):
        """
        Forward pass of the top-k gating function.

        Args:
            ctx: Context object for storing tensors needed in backward pass
            logits: Input logits tensor
            topk: Number of experts to select for each token
            capacity_factor: Capacity factor for each expert
            pad_to_capacity: Whether to pad to expert capacity
            drop_policy: Policy for dropping tokens
            use_pre_softmax: Whether to apply softmax before top-k selection
            num_groups: Number of expert groups
            group_topk: Number of groups to select for each token
            scaling_factor: Scaling factor for routing scores
            deterministic_mode: Whether to use deterministic mode
            score_function: Score function to use ("softmax" or "sigmoid")
            expert_bias: Optional bias for experts

        Returns:
            Tuple of (routing probabilities, routing map, tokens per expert)
        """
        # Store tensors needed in backward pass
        ctx.topk = topk
        ctx.scaling_factor = scaling_factor if scaling_factor is not None else 0.0
        ctx.score_function = score_function
        ctx.use_pre_softmax = use_pre_softmax
        
        if score_function == "softmax":
            assert use_pre_softmax == True, "use_pre_softmax must be True for softmax"

        # Get input dimensions
        num_tokens, num_experts = logits.shape

        # Initialize output tensors
        topk_masked_gates = torch.zeros_like(logits)
        topk_map = torch.zeros_like(logits, dtype=torch.bool)
        top_indices = torch.zeros((num_tokens, topk), dtype=torch.int64, device=logits.device)
        top_scores = torch.zeros((num_tokens, topk), dtype=logits.dtype, device=logits.device)

        # Define block sizes
        BLOCK_SIZE_NUM_EXPERTS = triton.next_power_of_2(num_experts)
        BLOCK_SIZE_TOPK = triton.next_power_of_2(topk)

        # Initialize temporary tensors
        scores_temp = torch.empty_like(logits)

        # Convert None scaling_factor to 0.0
        scaling_factor = 0.0 if scaling_factor is None else scaling_factor

        # Launch kernel
        if num_groups:
            # Calculate group-related parameters
            experts_per_group = num_experts // num_groups
            topk_per_group = topk // group_topk

            # Define group-related block sizes
            BLOCK_SIZE_NUM_GROUPS = triton.next_power_of_2(num_groups)
            BLOCK_SIZE_EXPERTS_PER_GROUP = triton.next_power_of_2(experts_per_group)
            BLOCK_SIZE_GROUP_TOPK = triton.next_power_of_2(group_topk)
            BLOCK_SIZE_TOPK_PER_GROUP = triton.next_power_of_2(topk_per_group)

            # Initialize group-related temporary tensors
            group_view_temp = torch.empty_like(logits)
            top_values_temp = torch.empty((num_tokens, num_groups, experts_per_group), dtype=logits.dtype, device=logits.device)
            top_indices_temp2 = torch.zeros((num_tokens, group_topk), dtype=torch.int64, device=logits.device)
            group_mask_temp = torch.zeros((num_tokens, num_groups), dtype=logits.dtype, device=logits.device)

            fused_topk_softmax_forward_kernel_with_group[(num_tokens,)](
                scores_temp,
                group_view_temp,
                top_values_temp,
                top_indices_temp2,
                group_mask_temp,
                logits,
                expert_bias if expert_bias is not None else torch.empty(0, device=logits.device),
                expert_bias is not None,
                topk_masked_gates,
                topk_map,
                top_indices,
                top_scores,
                num_tokens,
                num_experts,
                num_groups,
                experts_per_group,
                topk,
                group_topk,
                topk_per_group,
                scaling_factor,
                score_function,
                use_pre_softmax,
                BLOCK_SIZE_NUM_EXPERTS,
                BLOCK_SIZE_NUM_GROUPS,
                BLOCK_SIZE_EXPERTS_PER_GROUP,
                BLOCK_SIZE_TOPK,
                BLOCK_SIZE_GROUP_TOPK,
                BLOCK_SIZE_TOPK_PER_GROUP,
            )
        else:
            fused_topk_softmax_forward_kernel_without_group[(num_tokens,)](
                scores_temp,
                logits,
                expert_bias if expert_bias is not None else torch.empty(0, device=logits.device),
                expert_bias is not None,
                topk_masked_gates,
                topk_map,
                top_indices,
                top_scores,
                num_tokens,
                num_experts,
                topk,
                scaling_factor,
                score_function,
                use_pre_softmax,
                BLOCK_SIZE_NUM_EXPERTS,
                BLOCK_SIZE_TOPK,
            )

        # save tensors for backward
        ctx.save_for_backward(logits, top_indices, top_scores)    

        # Compute tokens per expert
        tokens_per_expert = topk_map.sum(dim=0)

        return topk_masked_gates, topk_map, tokens_per_expert

    @staticmethod
    def backward(ctx, grad_topk_masked_gates, grad_topk_map, grad_tokens_per_expert):
        """
        Backward pass of the top-k gating function.

        Args:
            ctx: Context object containing saved tensors
            grad_topk_masked_gates: Gradient of routing probabilities
            grad_topk_map: Gradient of routing map
            grad_tokens_per_expert: Gradient of tokens per expert

        Returns:
            Tuple of gradients for each input
        """
        # Load saved tensors
        logits, top_indices, top_scores = ctx.saved_tensors
        topk = ctx.topk
        scaling_factor = ctx.scaling_factor
        score_function = ctx.score_function
        use_pre_softmax = ctx.use_pre_softmax

        # Get input dimensions
        num_tokens, num_experts = logits.shape

        # Initialize gradient tensor
        grad_logits = torch.zeros_like(logits)

        # Define block sizes
        BLOCK_SIZE_NUM_EXPERTS = triton.next_power_of_2(num_experts)
        BLOCK_SIZE_TOPK = triton.next_power_of_2(topk)

        # Launch kernel
        fused_topk_softmax_backward_kernel[(num_tokens,)](
            logits,
            top_indices,
            top_scores,
            grad_topk_masked_gates,
            grad_logits,
            num_experts,
            topk,
            scaling_factor,
            BLOCK_SIZE_NUM_EXPERTS,
            BLOCK_SIZE_TOPK,
        )

        return grad_logits, None, None, None, None, None, None, None, None, None, None, None


def fused_topk_softmax_without_capacity(
    logits, 
    topk,
    capacity_factor,
    pad_to_capacity,
    drop_policy,
    use_pre_softmax,
    num_groups,
    group_topk,
    scaling_factor,
    deterministic_mode,
    score_function,
    expert_bias,
):
    """
    Fused top-k gating function without capacity constraints.

    Args:
        logits: Input logits tensor
        topk: Number of experts to select for each token
        capacity_factor: Capacity factor for each expert
        pad_to_capacity: Whether to pad to expert capacity
        drop_policy: Policy for dropping tokens
        use_pre_softmax: Whether to apply softmax before top-k selection
        num_groups: Number of expert groups
        group_topk: Number of groups to select for each token
        scaling_factor: Scaling factor for routing scores
        deterministic_mode: Whether to use deterministic mode
        score_function: Score function to use ("softmax" or "sigmoid")
        expert_bias: Optional bias for experts

    Returns:
        Tuple of (routing probabilities, routing map, tokens per expert)
    """

    if score_function == "softmax":
        return topk_softmax_with_capacity(
            logits,
            topk,
            capacity_factor,
            pad_to_capacity,
            drop_policy,
            use_pre_softmax,
            num_groups,
            group_topk,
            scaling_factor,
            deterministic_mode,
            score_function,
            expert_bias,
        )
    else:
        return TopkSoftmax.apply(
            logits,
            topk,
            capacity_factor,
            pad_to_capacity,
            drop_policy,
            use_pre_softmax,
            num_groups,
            group_topk,
            scaling_factor,
            deterministic_mode,
            score_function,
            expert_bias,
        )