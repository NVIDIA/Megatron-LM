# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from megatron.core import parallel_state
from megatron.core.fp4_utils import get_fp4_align_size
from megatron.core.fp8_utils import get_fp8_align_size
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import get_cuda_rng_tracker, get_expert_parallel_rng_tracker_name
from megatron.core.transformer.cuda_graphs import is_graph_capturing
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import internal_api

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import (
        fused_compute_score_for_moe_aux_loss,
        fused_moe_aux_loss,
        fused_permute,
        fused_permute_with_probs,
        fused_sort_chunks_by_index,
        fused_sort_chunks_by_index_with_probs,
        fused_topk_with_score_function,
        fused_unpermute,
        te_general_gemm,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


# MOE logging
_MOE_LAYER_WISE_LOGGING_TRACKER = {}


def switch_load_balancing_loss_func(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_num_tokens: int,
    topk: int,
    num_experts: int,
    moe_aux_loss_coeff: float,
    fused: bool = False,
):
    """Calculate the auxiliary loss for load balancing.
    Refer to the Switch Transformer (https://arxiv.org/abs/2101.03961)
    and Global Load Balancing Loss(https://arxiv.org/abs/2501.11873) for details.

    ### Detailed explanation of the auxiliary loss #######

    The formula for the auxiliary loss is:
        loss = E * Σ_{i=1}^{E} (f_i * P_i)
    where:
        f_i = 1 / (T * topk) * Σ_{x∈B} routing_map(x, i)
             (fraction of tokens dispatched to expert i)
        P_i = 1 / T * Σ_{x∈B} probs(x, i)
             (averaged router probability allocated for expert i)
        E is the number of experts
        T is the total number of tokens in the batch B

    For distributed training with sequence or context parallelism, each rank can
    process a subset of the batch.
        loss = E * Σ_{i=1}^{E} (f_i * Σ_{j=1}^{N} P_ij)
             = E * Σ_{i=1}^{E} Σ_{j=1}^{N} (f_i * P_ij)
             = Σ_{j=1}^{N} E * (Σ_{i=1}^{E} f_i * P_ij)

    where:
        f_i = 1 / (T * topk) * Σ_{x∈B} routing_map(x, i)
             (fraction of tokens dispatched to expert i in the global batch)
        P_ij = 1 / T * Σ_{x∈B_j} probs(x, i)
              (averaged router probability allocated for expert i in local batch of the j-th rank)
        N is the number of ranks
        B_j is the batch of tokens in the j-th rank
        T is the total number of tokens in the global batch B

    Note:
    To calculate the auxiliary loss at different levels (micro-batch or global batch):
    - probs: Should always be from the local batch being processed
    - tokens_per_expert: Should represent token counts at the desired level
      (either micro-batch or global batch)
    - total_num_tokens: Should match the total token count at the same level as tokens_per_expert

    #########################################################

    Args:
        probs (torch.Tensor): Softmax probabilities output by the router for each token.
                              Shape in [num_tokens, num_experts].
        tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert in the batch.
                                          Shape in [num_experts]
        total_num_tokens (int): Total number of tokens in the batch.
        topk (int): The number of experts selected for each token.
        num_experts (int): The number of experts.
        moe_aux_loss_coeff (float): The coefficient for the auxiliary loss.
    Returns:
        torch.Tensor: The auxiliary loss for load balancing.
    """
    if fused:
        if not HAVE_TE or fused_moe_aux_loss is None:
            raise ValueError("fused_moe_aux_loss is not available. Please install TE >= 2.7.0.")
        return fused_moe_aux_loss(
            probs=probs,
            tokens_per_expert=tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=topk,
            num_experts=num_experts,
            coeff=moe_aux_loss_coeff,
        )

    aggregated_probs_per_expert = probs.sum(dim=0)
    aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (topk * total_num_tokens * total_num_tokens)
    )
    return aux_loss


def z_loss_func(logits, z_loss_coeff):
    """Encourages the router's logits to remain small to enhance stability.
    Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

    Args:
        logits (torch.Tensor): The logits of the router.

    Returns:
        torch.Tensor: The logits after applying the z-loss.
    """

    z_loss = torch.mean(torch.square(torch.logsumexp(logits, dim=-1))) * z_loss_coeff
    return z_loss


def sinkhorn(cost: torch.Tensor, tol: float = 0.0001):
    """Sinkhorn based MoE routing function"""
    cost = torch.exp(cost)
    d0 = torch.ones(cost.size(0), device=cost.device, dtype=cost.dtype)
    d1 = torch.ones(cost.size(1), device=cost.device, dtype=cost.dtype)

    eps = 0.00000001
    error = 1e9
    d1_old = d1
    while error > tol:
        d0 = (1 / d0.size(0)) * 1 / (torch.sum(d1 * cost, 1) + eps)
        d1 = (1 / d1.size(0)) * 1 / (torch.sum(d0.unsqueeze(1) * cost, 0) + eps)
        error = torch.mean(torch.abs(d1_old - d1))
        d1_old = d1
    return d1 * cost * d0.unsqueeze(1)


def get_capacity(num_tokens: int, num_experts: int, capacity_factor: float, min_capacity=None):
    """
    Calculate the capacity of each expert.

    Args:
        num_tokens (int): num of the input tokens.
        num_experts (int): num of the experts.
        capacity_factor (float): Capacity factor.
        min_capacity (int, optional): Minimum capacity. Defaults to None.

    Returns:
        Tensor: Capacity of each expert.
    """
    capacity = math.ceil((num_tokens / num_experts) * capacity_factor)
    if min_capacity is not None and capacity < min_capacity:
        capacity = min_capacity
    return capacity


class MoEAuxLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for auxiliary loss."""

    main_loss_backward_scale: torch.Tensor = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        """Preserve the aux_loss by storing it in the context to avoid garbage collection.

        Args:
            output (torch.Tensor): The output tensor.
            aux_loss (torch.Tensor): The auxiliary loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for auxiliary loss..

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled auxiliary loss
                                               gradient.
        """
        (aux_loss,) = ctx.saved_tensors
        if MoEAuxLossAutoScaler.main_loss_backward_scale is None:
            MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(
                1.0, device=aux_loss.device
            )
        aux_loss_backward_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
        scaled_aux_loss_grad = torch.ones_like(aux_loss) * aux_loss_backward_scale
        return grad_output, scaled_aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the aux loss.

        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in
                                  matches the scale of the main_loss.
        """
        if MoEAuxLossAutoScaler.main_loss_backward_scale is None:
            MoEAuxLossAutoScaler.main_loss_backward_scale = scale
        else:
            MoEAuxLossAutoScaler.main_loss_backward_scale.copy_(scale)


def permute(
    tokens,
    routing_map,
    probs: Optional[torch.Tensor] = None,
    num_out_tokens: Optional[int] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.

    When drop_and_pad=True, in routing_map, the number of non-zeros in each column equals to
    expert capacity. This function exploits this feature to use ops that support cuda graph.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
        probs (torch.Tensor, optional): The probs tensor, [num_tokens, num_experts].
        num_out_tokens (int, optional): The number of output tokens. If None, it's set to
                                        the number of input tokens.
        fused (bool, optional): Whether use the fused permute function.
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.
                                       If set to true, routing_map has a fixed number of non-zeros
                                       in each column.
    """
    if fused and probs is None:
        if not HAVE_TE or fused_permute is None:
            raise ValueError("fused_permute is not available. Please install TE >= 2.1.0.")
        permuted_input, sorted_indices = fused_permute(
            tokens, routing_map, num_out_tokens=num_out_tokens
        )
        return permuted_input, None, sorted_indices

    if fused and probs is not None:
        if not HAVE_TE or fused_permute_with_probs is None:
            raise ValueError(
                "fused_permute_with_probs is not available. Please install TE >= 2.1.0."
            )
        return fused_permute_with_probs(tokens, probs, routing_map, num_out_tokens=num_out_tokens)

    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]
    permuted_probs = None
    if drop_and_pad and not (num_out_tokens is None):
        capacity = num_out_tokens // num_experts
        assert not routing_map.requires_grad
        # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
        routing_map = routing_map.to(dtype=torch.int8).T.contiguous()
        # use argsort to put indices of all non-zeros in the beginning of list
        # and keep the first `capacity` number of indices
        sorted_indices = routing_map.argsort(dim=-1, descending=True, stable=True)[
            :, :capacity
        ].contiguous()
        # flatten from [num_experts, capacity] to 1D
        sorted_indices = sorted_indices.view(-1)

        if probs is not None:
            # [num_tokens, num_experts] -> num_experts * num_tokens
            probs_T_1D = probs.T.contiguous().view(-1)
            # get 1D indices of the probs selected by routing_map
            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1D = (indices_dim0 * num_tokens + indices_dim1).view(-1)
            # get probs from indices
            permuted_probs = probs_T_1D.index_select(0, indices_1D)
    else:
        # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
        routing_map = routing_map.bool().T.contiguous()

        # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
        token_indices = (
            torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
        )
        sorted_indices = token_indices.masked_select(routing_map)

        if probs is not None:
            permuted_probs = probs.T.contiguous().masked_select(routing_map)

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, permuted_probs, sorted_indices


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: torch.Tensor = None,
    routing_map: torch.Tensor = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    """
    Restore the original order of tokens after permutation. If probs are provided, it
    will also apply them to the tokens before restoring the order.

    When drop_and_pad=True, the tensors will have the following properties:
      - In routing_map, the number of non-zeros in each column equals to expert capacity
      - The size of sorted_indices equals to num_experts * capacity, each split of `capacity`
        contains the indices of tokens routed to an expert.
    This function exploits these features to use ops that support cuda graph.

    Args:
        permuted_tokens (torch.Tensor): The permuted token tensor.
        sorted_indices (torch.Tensor): The indices used to sort the tokens.
        restore_shape (torch.Size): The shape of the unpermuted tensor.
        probs (torch.Tensor, optional): The unpermuted probs tensor,
        routing_map (torch.Tensor, optional): Token to expert mapping, shape
            [num_tokens, num_experts].
        fused (bool, optional): Whether use the fused unpermute function.
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.

    Returns:
        torch.Tensor: The tokens restored to their original order.
    """
    if fused:
        if not HAVE_TE or fused_unpermute is None:
            raise ValueError("fused_unpermute is not available. Please install TE >= 2.1.0.")
        return fused_unpermute(
            permuted_tokens, sorted_indices, merging_probs=probs, restore_shape=restore_shape
        )

    _, hidden = restore_shape
    input_dtype = permuted_tokens.dtype

    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        if drop_and_pad:
            num_experts = routing_map.size(1)
            num_permuted_tokens = sorted_indices.size(0)
            capacity = num_permuted_tokens // num_experts
            num_unpermuted_tokens = probs.size(0)

            # [num_unpermuted_tokens, num_experts] -> num_experts * num_unpermuted_tokens
            probs_T_1D = probs.T.contiguous().view(-1)

            # get 1D indices of the probs selected by routing_map
            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1D = (indices_dim0 * num_unpermuted_tokens + indices_dim1).view(-1)

            # get probs from indices
            permuted_probs = probs_T_1D.index_select(0, indices_1D)
        else:
            permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        # Here may promote permuted_tokens to higher precision (fp32/fp64) if probs is in
        # higher precision due to moe_router_dtype being enabled. This can lead to
        # additional GPU memory usage. Use --moe-permute-fusion flag to avoid this extra memory
        # allocation.
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    # Create an output tensor filled with zeros
    output_tokens = torch.zeros(
        restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device
    )
    if torch.are_deterministic_algorithms_enabled():
        # Use index_add which is deterministic when deterministic algorithms are enabled
        # and is CUDA graph compatible
        output_tokens = torch.zeros(
            restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device
        )
        # index_add is deterministic when torch.use_deterministic_algorithms(True) is set
        # and is CUDA graph compatible unlike scatter_add
        output_tokens.index_add_(0, sorted_indices, permuted_tokens)
    else:
        # Scatter add the permuted_input back to the original positions
        output_tokens.scatter_add_(
            0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens
        )
    return output_tokens.to(dtype=input_dtype)


def sort_chunks_by_idxs(
    input: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_idxs: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    fused: bool = False,
):
    """Split and sort the input tensor based on the split_sizes and sorted indices."""
    if fused and probs is None:
        if not HAVE_TE or fused_sort_chunks_by_index is None:
            raise ValueError(
                "fused_sort_chunks_by_index is not available. Please install TE >= 2.1.0."
            )
        return fused_sort_chunks_by_index(input, split_sizes, sorted_idxs), None

    if fused and probs is not None:
        if not HAVE_TE or fused_sort_chunks_by_index_with_probs is None:
            raise ValueError(
                "fused_sort_chunks_by_index_with_probs is not available. "
                "Please install TE >= 2.1.0."
            )
        return fused_sort_chunks_by_index_with_probs(input, probs, split_sizes, sorted_idxs)

    input = torch.split(input, split_sizes.tolist(), dim=0)
    output = torch.cat([input[i] for i in sorted_idxs.tolist()], dim=0)
    if probs is not None:
        probs = torch.split(probs, split_sizes.tolist(), dim=0)
        permuted_probs = torch.cat([probs[i] for i in sorted_idxs.tolist()], dim=0)
    else:
        permuted_probs = None
    return output, permuted_probs


def group_limited_topk(
    scores: torch.Tensor,
    topk: int,
    num_tokens: int,
    num_experts: int,
    num_groups: int,
    group_topk: int,
):
    """Perform top-k routing on a subset of expert groups.

    When using group-limited routing:
    1. Experts are divided into 'moe_router_num_groups' equal-sized groups
    2. For each token, 'moe_router_group_topk' groups are selected based on routing scores
       (specifically, the sum of top-2 expert scores within each group)
    3. From these selected groups, 'moe_router_topk' individual experts are chosen

    Two common use cases:
    - Device-limited routing: Set 'moe_router_num_groups' equal to expert parallel size (EP)
      to limit each token to experts on a subset of devices
      (See DeepSeek-V2: https://arxiv.org/pdf/2405.04434)

    - Node-limited routing: Set 'moe_router_num_groups' equal to number of nodes in EP group
      to limit each token to experts on a subset of nodes
      (See DeepSeek-V3: https://arxiv.org/pdf/2412.19437)

    Args:
        scores (torch.Tensor): Softmax scores generated by the router.
        topk (int): The number of experts to select for each token.
        num_tokens (int): The number of tokens.
        num_experts (int): The number of experts.
        num_groups (int): Number of groups for routed experts.
        group_topk (int): Number of groups selected for each token.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Probs and indices tensor.
    """
    # Organize the experts into groups
    # Select groups based on sum of top-(topk/group_topk) routing scores within each group
    group_scores = (
        scores.view(num_tokens, num_groups, -1).topk(topk // group_topk, dim=-1)[0].sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)

    # Mask the experts based on selection groups
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_groups, num_experts // num_groups)
        .reshape(num_tokens, -1)
    )

    masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
    probs, top_indices = torch.topk(masked_scores, k=topk, dim=-1)

    return probs, top_indices


def pad_routing_map(routing_map: torch.Tensor, pad_multiple: int) -> torch.Tensor:
    """Pad the routing map to ensure each expert has a multiple of pad_multiple tokens.

    This function ensures that each expert has a number of tokens that is a multiple of
    pad_multiple by converting some 0s to 1s in the routing map. The padding is done by
    selecting the first N zero elements in each row, where N is the number needed to reach
    the next multiple of pad_multiple.

    Args:
        routing_map (torch.Tensor): A boolean or integer tensor of shape [num_tokens,
            num_experts] indicating which tokens are routed to which experts.
        pad_multiple (int): The multiple to pad each expert's token count to.

    Returns:
        torch.Tensor: The padded routing map of shape [num_tokens, num_experts].
    """
    # Transpose to [num_experts, num_tokens] for easier row-wise operations
    routing_map = routing_map.transpose(0, 1)  # [num_experts, num_tokens]

    # Calculate how many tokens need to be padded for each expert
    num_ones = routing_map.sum(dim=1)
    num_to_pad = (-num_ones) % pad_multiple

    # Find the positions of zeros in each row and their ranks
    is_zero = routing_map == 0
    zero_ranks = torch.cumsum(is_zero.int(), dim=1)

    # Create mask for elements that need to be padded (converted from 0 to 1)
    mask = zero_ranks <= num_to_pad.unsqueeze(1)
    routing_map[mask] = 1

    routing_map = routing_map.transpose(0, 1)
    return routing_map


def topk_routing_with_score_function(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool = False,
    num_groups: Optional[int] = None,
    group_topk: Optional[int] = None,
    scaling_factor: Optional[float] = None,
    score_function: str = "softmax",
    expert_bias: Optional[torch.Tensor] = None,
    fused: bool = False,
):
    """Compute the routing probabilities and map for top-k selection with score function.
    Args:
        logits (torch.Tensor): Logits tensor.
        topk (int): The number of experts to select for each token.
        use_pre_softmax (bool): Whether to apply softmax or sigmoid before top-k selection.
        num_groups (int): Number of groups for routed experts.
        group_topk (int): Number of selected groups for each token.
        scaling_factor (float): Scaling factor of routing score in top-k selection.
        score_function (str): The score function to use. Can be either "softmax" or "sigmoid".
        expert_bias (torch.Tensor): The bias added to logits for expert routing.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - routing_probs (torch.Tensor): A tensor of shape [num_tokens, num_experts] containing
              the routing probabilities for each token to each expert.
            - routing_map (torch.Tensor): A mask tensor of shape [num_tokens, num_experts]
              indicating which experts were selected for each token. True values represent
              the selected experts.
    """
    assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."
    num_tokens, num_experts = logits.shape
    if fused:
        if not HAVE_TE or fused_topk_with_score_function is None:
            raise ValueError(
                "fused_topk_with_score_function is not available. Please install TE >= 2.6.0."
            )
        return fused_topk_with_score_function(
            logits=logits,
            topk=topk,
            use_pre_softmax=use_pre_softmax,
            num_groups=num_groups,
            group_topk=group_topk,
            scaling_factor=scaling_factor,
            score_function=score_function,
            expert_bias=expert_bias,
        )

    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if group_topk:
            return group_limited_topk(
                scores=scores,
                topk=topk,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_groups=num_groups,
                group_topk=group_topk,
            )
        else:
            return torch.topk(scores, k=topk, dim=1)

    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            probs, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        else:
            scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float()).type_as(logits)
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
            scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
        else:
            scores, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    if scaling_factor:
        probs = probs * scaling_factor

    if torch.are_deterministic_algorithms_enabled():
        # build [num_tokens, num_experts] from [num_tokens, topk]
        routing_probs = torch.zeros_like(logits)
        rows = torch.arange(num_tokens, device=logits.device).unsqueeze(1)
        routing_probs.index_put_((rows, top_indices), probs, accumulate=False)

        routing_map = torch.zeros_like(logits, dtype=logits.dtype)
        routing_map.index_put_(
            (rows, top_indices), torch.ones_like(probs, dtype=routing_map.dtype), accumulate=False
        )
        routing_map = routing_map.bool()
    else:
        # TODO Try using element-wise operations instead of scatter?
        routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
        routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()

    return routing_probs, routing_map


def compute_routing_scores_for_aux_loss(
    logits: torch.Tensor, topk: int, score_function: str, fused: bool = False
):
    """Compute routing scores based on the score function.

    Args:
        logits (torch.Tensor): The logits tensor after gating, shape: [num_tokens, num_experts].

    Returns:
        torch.Tensor: The normalized routing scores.
    """
    if fused:
        if not HAVE_TE or fused_compute_score_for_moe_aux_loss is None:
            raise ValueError(
                "fused_compute_score_for_moe_aux_loss is not available. Please install TE >= 2.6.0."
            )
        return fused_compute_score_for_moe_aux_loss(
            logits=logits, topk=topk, score_function=score_function
        )

    if score_function == "softmax":
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits)
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    _, top_indices = torch.topk(scores, k=topk, dim=1)
    routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    return routing_map, scores


def apply_router_token_dropping(
    routing_probs: torch.Tensor,
    routing_map: torch.Tensor,
    router_topk: int,
    capacity_factor: float,
    drop_policy: str = "probs",
    pad_to_capacity: bool = False,
):
    """Apply token dropping to top-k expert selection.

    This function enforces expert capacity limits by dropping tokens that exceed
    the capacity and optionally padding to capacity.

    Args:
        routing_probs (torch.Tensor): Tensor of shape [num_tokens, num_experts]
            containing the routing probabilities for selected experts.
        routing_map (torch.Tensor): Boolean tensor of shape [num_tokens, num_experts]
            indicating which experts were selected for each token.
        router_topk (int): Number of experts selected per token.
        capacity_factor (float): The capacity factor of each expert.
        drop_policy (str): Policy to drop tokens - "probs" or "position".
        pad_to_capacity (bool): Whether to pad to capacity.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - final_probs: Routing probabilities after applying capacity constraints
            - final_map: Boolean mask after applying capacity constraints
    """
    assert routing_probs.ndim == 2 and routing_map.ndim == 2
    num_tokens, num_experts = routing_probs.shape
    # Calculate expert capacity
    expert_capacity = get_capacity(
        num_tokens=num_tokens * router_topk,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )

    # Create capacity mask based on drop policy
    if expert_capacity > num_tokens:
        # No need to drop tokens if capacity exceeds the number of tokens
        capacity_mask = torch.ones_like(routing_probs).bool()
    else:
        if drop_policy == "probs":
            _, capacity_indices = torch.topk(routing_probs, k=expert_capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(routing_probs).scatter(0, capacity_indices, 1).bool()
        elif drop_policy == "position":
            _, capacity_indices = torch.topk(
                routing_map.int(), k=expert_capacity, dim=0, sorted=False
            )
            capacity_mask = torch.zeros_like(routing_probs).scatter(0, capacity_indices, 1).bool()
        else:
            raise ValueError(f"Invalid drop_policy: {drop_policy}")

    # Apply capacity constraints
    if pad_to_capacity:
        final_map = capacity_mask
        final_probs = routing_probs * final_map
    else:
        # Get exceed mask and maskout exceeded probs and indices
        final_map = torch.logical_and(routing_map, capacity_mask)
        final_probs = routing_probs * final_map

    return final_probs, final_map


def save_to_aux_losses_tracker(
    name: str,
    loss: torch.Tensor,
    layer_number: int,
    num_layers: int,
    reduce_group: torch.distributed.ProcessGroup = None,
    avg_group: torch.distributed.ProcessGroup = None,
    reduce_group_has_dp: bool = False,
):
    """Save the auxiliary loss for logging.
    Args:
        name (str): The name of the loss.
        loss (torch.Tensor): The loss tensor.
        layer_number (int): Layer index of the loss.
        num_layers (int): The number of total layers.
        reduce_group (torch.distributed.ProcessGroup): The group for reducing the loss.
        avg_group (torch.distributed.ProcessGroup): The group for averaging the loss.
        reduce_group_has_dp (bool): Whether the reduce group has data parallel ranks.
            Set this to True if the reduce group has data parallel ranks. This flag is used to
            ensure the correct reduction in aux loss tracking.
    """
    # Skip aux loss logging if layer_number is None.
    if layer_number is None:
        return

    tracker = get_moe_layer_wise_logging_tracker()
    if name not in tracker:
        tracker[name] = {}
        tracker[name]["values"] = torch.zeros(num_layers, device=loss.device)
    tracker[name]["values"][layer_number - 1] += loss.detach()  # Aggregate the loss for the layer.
    tracker[name]["reduce_group"] = reduce_group
    tracker[name]["avg_group"] = avg_group
    tracker[name]["reduce_group_has_dp"] = reduce_group_has_dp


def clear_aux_losses_tracker():
    """Clear the auxiliary losses."""
    tracker = get_moe_layer_wise_logging_tracker()
    for name in tracker:
        tracker[name]["values"].zero_()


def reduce_aux_losses_tracker_across_ranks(
    track_names: Optional[List[str]] = None, pg_collection: Optional[ProcessGroupCollection] = None
):
    """Collect and reduce the auxiliary losses across ranks."""
    tracker = get_moe_layer_wise_logging_tracker()
    if track_names is None:
        track_names = tracker.keys()

    if pg_collection is None:
        # Use parallel_state groups
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        dp_group = parallel_state.get_data_parallel_group(
            with_context_parallel=False, partial_data_parallel=False
        )
    else:
        pp_group = pg_collection.pp
        dp_group = pg_collection.dp

    for name in track_names:
        values = tracker[name]["values"]
        # TODO(Hepteract): delete the usage of the global parallel_state.
        # Collect aux losses across PP.
        torch.distributed.all_reduce(values, group=pp_group)
        # Reduce aux losses across ranks.
        if tracker[name].get('reduce_group') is not None:
            torch.distributed.all_reduce(values, group=tracker[name].get('reduce_group'))
            # Need to conduct reduction across data parallel ranks. When the reduce_group
            # does not have 'dp' attribute, do it manually.
            if not tracker[name].get('reduce_group_has_dp', False):
                torch.distributed.all_reduce(
                    values,
                    group=parallel_state.get_data_parallel_group(with_context_parallel=False),
                    op=torch.distributed.ReduceOp.AVG,
                )
        if tracker[name].get('avg_group') is not None:
            torch.distributed.all_reduce(
                values, group=tracker[name]['avg_group'], op=torch.distributed.ReduceOp.AVG
            )
        # Average aux losses across data parallel ranks.
        # The `global_load_balancing_loss` already uses `tp_dp_cp_group` in `reduce_group`,
        # so we don't need to reduce it again. Others use `tp_cp_group` in `reduce_group`.
        if name != "global_load_balancing_loss":
            torch.distributed.all_reduce(values, group=dp_group, op=torch.distributed.ReduceOp.AVG)


def track_moe_metrics(
    loss_scale: float,
    iteration: int,
    writer,
    wandb_writer=None,
    total_loss_dict=None,
    per_layer_logging=False,
    force_initialize: bool = False,
    track_names: Optional[List[str]] = None,
    num_layers: Optional[int] = None,
    moe_layer_freq: Optional[Union[int, List[int]]] = None,
    mtp_num_layers: Optional[int] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
):
    """Track the MoE metrics for logging."""
    # Aux loss logging
    tracker = get_moe_layer_wise_logging_tracker()
    # Initialize the tracker if force_initialize is True
    if force_initialize:
        if track_names is not None:
            for key in track_names:
                if key not in tracker:
                    tracker[key] = {}
                    tracker[key]["values"] = torch.zeros(num_layers, device="cuda")
                    tracker[key]["reduce_group"] = None
                    tracker[key]["avg_group"] = None
                    tracker[key]["reduce_group_has_dp"] = False

    reduce_aux_losses_tracker_across_ranks(track_names, pg_collection=pg_collection)

    # Get number of MoE layers
    if moe_layer_freq is None:
        num_moe_layers = num_layers
    elif isinstance(moe_layer_freq, int):
        assert isinstance(num_layers, int)
        moe_layer_pattern = [1 if (i % moe_layer_freq == 0) else 0 for i in range(num_layers)]
        num_moe_layers = sum(moe_layer_pattern)
    elif isinstance(moe_layer_freq, list):
        num_moe_layers = sum(moe_layer_freq)
    else:
        raise ValueError(f"Invalid moe_layer_freq: {moe_layer_freq}")

    if mtp_num_layers is not None:
        num_moe_layers += mtp_num_layers

    aux_losses = {k: v['values'].float() * loss_scale for k, v in tracker.items()}
    for name, loss_list in aux_losses.items():
        if total_loss_dict is not None:
            if name not in total_loss_dict:
                total_loss_dict[name] = loss_list.sum() / num_moe_layers
            else:
                total_loss_dict[name] += loss_list.sum() / num_moe_layers
        if writer is not None:
            # currently when using add_scalars,
            # torch.utils.add_scalars makes each timer its own run, which
            # polutes the runs list, so we just add each as a scalar
            writer.add_scalar(name, loss_list.sum() / num_moe_layers, iteration)
            if per_layer_logging:
                for i, loss in enumerate(loss_list.tolist()):
                    writer.add_scalar(f"moe/{name}_layer_{i}", loss, iteration)

            # W&B logging lacks support for logging multiple scalars simultaneously.
            # As a workaround, we log each scalar individually first, then we can create
            # a custom panel to manually group them to a single plot.
            if wandb_writer:
                wandb_writer.log({f"{name}": loss_list.sum() / num_moe_layers}, iteration)
                if per_layer_logging:
                    wandb_writer.log(
                        {
                            f"moe/{name}_layer_{i}": loss
                            for i, loss in enumerate(loss_list.tolist())
                        },
                        iteration,
                    )

    clear_aux_losses_tracker()


def get_updated_expert_bias(tokens_per_expert, expert_bias, expert_bias_update_rate):
    """Update expert bias for biased expert routing. See https://arxiv.org/abs/2408.15664v1#

    Args:
        tokens_per_expert (torch.Tensor): The number of tokens assigned to each expert.
        expert_bias (torch.Tensor): The bias for each expert.
        expert_bias_udpate_rate (float): The update rate for the expert bias.
    """
    with torch.no_grad():
        # All Reduce Across TPxCPxDP group
        torch.distributed.all_reduce(
            tokens_per_expert,
            # TODO(Hepteract): delete the usage of the global parallel_state.
            group=parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True),
        )
        average_tokens = tokens_per_expert.sum(dim=-1, keepdim=True) / tokens_per_expert.shape[-1]
        offset = average_tokens - tokens_per_expert
        updated_expert_bias = expert_bias + torch.sign(offset) * expert_bias_update_rate
        return updated_expert_bias


def maybe_move_tensor_to_cpu(tensor, as_numpy=False, record_stream=False):
    """Move a tensor to CPU if it is on GPU.
    Args:
        tensor (torch.Tensor or None): The tensor to move to CPU.
        as_numpy (bool): Whether to convert the tensor to a numpy array.
        record_stream (bool): Whether to record the stream of the tensor, to prevent memory leak
                              when the DtoH data transfer is on a side stream.
    """
    if torch.is_tensor(tensor) and tensor.is_cuda:
        cpu_tensor = tensor.to(torch.device("cpu"), non_blocking=True)
        if as_numpy:
            cpu_tensor = cpu_tensor.numpy()
        if record_stream:
            tensor.record_stream(torch.cuda.current_stream())
        tensor = cpu_tensor
    return tensor


def get_moe_layer_wise_logging_tracker():
    """Return the moe layer wise tracker."""
    global _MOE_LAYER_WISE_LOGGING_TRACKER
    return _MOE_LAYER_WISE_LOGGING_TRACKER


@internal_api
class RandomSTE(torch.autograd.Function):
    """
    Straight-Through Estimator(STE) function that returns random values
    with different seed for each rank.

    This is used to generate random logits of router for load-balanced benchmark.
    """

    @staticmethod
    def forward(ctx, logits):
        """
        Forward pass returns random logits with rank-specific seed.
        """
        with get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name()):
            random_logits = logits.clone().normal_()
        return random_logits

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass propagates the gradient for logits.
        """
        return grad_output


def apply_random_logits(logits):
    """
    Apply the RandomSTE function to the logits.
    """
    return RandomSTE.apply(logits)


class RouterGatingLinearFunction(torch.autograd.Function):
    """
    Autograd function for router gating linear.
    """

    @staticmethod
    def forward(
        ctx, inp: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, router_dtype: torch.dtype
    ):
        """
        Forward pass of the RouterGatingLinearFunction function.
        """
        ctx.save_for_backward(inp, weight, bias)
        ctx.router_dtype = router_dtype
        ctx.input_dtype = inp.dtype
        ctx.weight_dtype = weight.dtype
        inp_shape = inp.shape
        inp = inp.view(-1, inp_shape[-1])

        if te_general_gemm is not None and router_dtype != torch.float64:
            output = te_general_gemm(weight, inp, router_dtype, layout="TN", bias=bias)
            output = output[0]
        elif bias is None:
            output = torch.mm(inp.to(router_dtype), weight.to(router_dtype).t())
        else:
            output = torch.addmm(
                bias.to(router_dtype), inp.to(router_dtype), weight.to(router_dtype).t()
            )

        output = output.view(*inp_shape[:-1], -1)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass of the RouterGatingLinearFunction function.
        """
        inp, weight, bias = ctx.saved_tensors
        inp_shape = inp.shape
        grad_shape = grad_output.shape
        inp = inp.view(-1, inp_shape[-1])
        grad_output = grad_output.view(-1, grad_shape[-1])

        if te_general_gemm is not None and ctx.router_dtype != torch.float64:
            grad_input = te_general_gemm(
                weight.to(ctx.router_dtype), grad_output, ctx.router_dtype, layout="NN", grad=True
            )
            grad_weight = te_general_gemm(
                inp.to(ctx.router_dtype), grad_output, ctx.router_dtype, layout="NT", grad=True
            )
            grad_input = grad_input[0].to(ctx.input_dtype)
            grad_weight = grad_weight[0].to(ctx.weight_dtype)
        else:
            grad_input = torch.mm(grad_output, weight.to(ctx.router_dtype)).to(ctx.input_dtype)
            grad_weight = torch.mm(grad_output.t(), inp.to(ctx.router_dtype)).to(ctx.weight_dtype)

        grad_bias = grad_output.sum(dim=0).to(ctx.weight_dtype) if bias is not None else None
        grad_input = grad_input.view(*inp_shape)
        return grad_input, grad_weight, grad_bias, None


def router_gating_linear(
    inp: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, router_dtype: torch.dtype
):
    """
    Customized linear layer for router gating.
    This linear layer accepts bfloat16 input and weight, and can return output with router_dtype.
    It can reduce the memory usage by avoiding saving the intermediate high precision tensors.
    """
    return RouterGatingLinearFunction.apply(inp, weight, bias, router_dtype)


def get_align_size_for_quantization(config: TransformerConfig):
    """Get the alignment size for quantization."""
    if config.fp8:
        return get_fp8_align_size(config.fp8_recipe)
    elif config.fp4:
        return get_fp4_align_size(config.fp4_recipe)
    return 16


# TODO(Hepteract): delete the usage of the global parallel_state.
# Initialize process groups with the global parallel_state.
def get_default_pg_collection():
    """Get the default process groups for MoE.

    Returns:
        ProcessGroupCollection: The default process groups for MoE.
    """
    pg_collection = ProcessGroupCollection()
    pg_collection.ep = parallel_state.get_expert_model_parallel_group()
    pg_collection.tp = parallel_state.get_tensor_model_parallel_group()
    pg_collection.cp = parallel_state.get_context_parallel_group()
    pg_collection.expt_tp = parallel_state.get_expert_tensor_parallel_group()
    pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()
    pg_collection.tp_ep = parallel_state.get_expert_tensor_and_model_parallel_group()
    pg_collection.tp_cp = parallel_state.get_tensor_and_context_parallel_group()
    pg_collection.tp_dp_cp = parallel_state.get_tensor_and_data_parallel_group(
        with_context_parallel=True
    )
    return pg_collection


class MoECudaGraphPartialCaptureSignal(Exception):
    """
    Used to early-return from a MoE layer forward pass in CUDA graph capture.
    This signal is raised when we are partially capturing the CUDA graph of the MoE layer,
    and the related intermediate tensors are recorded in self.kwargs.
    Call self.get_early_return_outputs() to collect the CUDA graph outputs.
    """

    def __init__(self, moe_layer, return_step: str, **kwargs):
        self.moe_layer = moe_layer
        self.return_step = return_step
        self.kwargs = kwargs

    def get_early_return_outputs(
        self, hidden_states: torch.Tensor, shared_expert_output: torch.Tensor
    ):
        """
        Get the CUDA graph early return outputs for the MoE layer, including the intermediate
        tensors and the intermediate attributes of the token dispatcher.
        """
        if self.return_step == "route":
            # Capturing the router step returns three intermediate tensors:
            # hidden states, routing probabilities, and routing map.
            outputs = [hidden_states, self.kwargs['probs'], self.kwargs['routing_map']]
        elif self.return_step == "preprocess":
            # Capturing the preprocess step returns three intermediate tensors:
            # hidden states, routing probabilities, and residual connection.
            # It also returns the intermediate attributes of the token dispatcher, recorded in
            # "token_dispatcher.cudagraph_attrs".
            outputs = [self.kwargs['hidden_states'], self.kwargs['probs'], self.kwargs['residual']]
            valid_cudagraph_attrs = []
            for attr_name in self.moe_layer.token_dispatcher.cudagraph_attrs:
                hier_attr_name = attr_name.split('.')
                attr = self.moe_layer.token_dispatcher
                for name in hier_attr_name:
                    attr = getattr(attr, name, None)
                    if attr is None:
                        break
                if isinstance(attr, torch.Tensor):
                    outputs.append(attr)
                    valid_cudagraph_attrs.append(attr_name)
            if self.moe_layer.token_dispatcher.valid_cudagraph_attrs is None:
                self.moe_layer.token_dispatcher.valid_cudagraph_attrs = valid_cudagraph_attrs
            else:
                assert (
                    self.moe_layer.token_dispatcher.valid_cudagraph_attrs == valid_cudagraph_attrs
                ), (
                    "valid_cudagraph_attrs mismatch: "
                    f"{self.moe_layer.token_dispatcher.valid_cudagraph_attrs} != "
                    f"{valid_cudagraph_attrs}"
                )
        # Also return the shared expert output, if it is not None.
        if shared_expert_output is not None:
            outputs.append(shared_expert_output)
        return outputs


@dataclass
class MoECudaGraphTensorStore:
    """Storage for tensors used in CUDA graph replay for MoE layers.

    This dataclass stores intermediate tensors computed during CUDA graph replay
    that need to be resumed from the end of the CUDA graph scope to skip redundant computations.

    Attributes:
        hidden_states (Optional[torch.Tensor]): The hidden states output from the CUDA graph replay.
        probs (Optional[torch.Tensor]): The routing probabilities for each token-expert pair.
        routing_map (Optional[torch.Tensor]): The sparse mapping indicating which experts
            were selected for each token. Used to skip the normal router step.
        residual (Optional[torch.Tensor]): The residual connection tensor before routing.
            Used to skip the normal preprocess step.
        shared_expert_output (Optional[torch.Tensor]): The output from shared experts
            computation. Used to skip the normal shared expert computation step.
    """

    hidden_states: Optional[torch.Tensor] = None
    probs: Optional[torch.Tensor] = None
    routing_map: Optional[torch.Tensor] = None
    residual: Optional[torch.Tensor] = None
    shared_expert_output: Optional[torch.Tensor] = None

    def is_empty(self) -> bool:
        """Check if the store has any non-None tensors.

        Returns:
            bool: True if all fields are None, False otherwise.
        """
        return all(
            getattr(self, field_name) is None
            for field_name in [
                'hidden_states',
                'probs',
                'routing_map',
                'residual',
                'shared_expert_output',
            ]
        )

    def set(self, **kwargs):
        """Set the tensors in the store from keyword arguments."""
        for field_name, value in kwargs.items():
            assert field_name in [
                'hidden_states',
                'probs',
                'routing_map',
                'residual',
                'shared_expert_output',
            ], f"Invalid field name: {field_name}"
            if value is not None:
                assert isinstance(
                    value, torch.Tensor
                ), f"Value must be a torch.Tensor, got {type(value)} for field {field_name}"
                setattr(self, field_name, value)

    def clear(self):
        """Reset all stored tensors to None."""
        for field_name in [
            'hidden_states',
            'probs',
            'routing_map',
            'residual',
            'shared_expert_output',
        ]:
            setattr(self, field_name, None)


def maybe_skip_or_early_return_by_cudagraph(step_condition):
    """
    Decorator to skip certain codepaths in the MoE layer forward pass in CUDA graph replay,
    or early return from the MoE layer forward pass in CUDA graph capture.

    Args:
        step_condition: The step condition to check. Can be "shared_experts_compute", "route",
        or "preprocess". If "shared_experts_compute", the shared experts computation will be
        skipped in replay if it is in the CUDA graph scope. If "route" or "preprocess", the
        router or preprocess will be skipped in replay if it is in the CUDA graph scope, or
        early return from the MoE layer forward pass if it is in CUDA graph capturing mode.

    Returns:
        A decorator function that wraps the MoE layer forward pass.
    """

    def maybe_raise_signal(moe_layer, **kwargs):
        """
        Check if the MoE layer should early return for CUDA graph capture.
        If so, raise a MoECudaGraphPartialCaptureSignal.
        """
        if (
            moe_layer.config.cuda_graph_impl == "transformer_engine"
            and moe_layer.training
            and is_graph_capturing()
        ):
            if (
                step_condition == "route"
                and CudaGraphScope.moe_router in moe_layer.config.cuda_graph_scope
                and CudaGraphScope.moe_preprocess not in moe_layer.config.cuda_graph_scope
            ):
                raise MoECudaGraphPartialCaptureSignal(moe_layer, "route", **kwargs)
            elif (
                step_condition == "preprocess"
                and CudaGraphScope.moe_preprocess in moe_layer.config.cuda_graph_scope
            ):
                raise MoECudaGraphPartialCaptureSignal(moe_layer, "preprocess", **kwargs)

    def decorator(func):
        def wrapped_func(moe_layer, *args, **kwargs):
            """
            Check if we should skip executing the original function based on the current
            step condition and the tensor store status. If the tensor can be found in the store,
            it indicates that it is already computed by the CUDA graph replay, so we can skip it.
            Otherwise, we execute the original function and check if we should raise a signal to
            early return in CUDA graph capture.
            """
            # The non-cudagraph codepath just calls the original function.
            if not is_graph_capturing() and moe_layer.cudagraph_tensor_store.is_empty():
                return func(moe_layer, *args, **kwargs)

            assert (
                not is_graph_capturing() or moe_layer.cudagraph_tensor_store.is_empty()
            ), "cudagraph_tensor_store cannot be used when it is capturing cuda graph."
            if step_condition == "shared_experts_compute":
                if moe_layer.cudagraph_tensor_store.shared_expert_output is None:
                    # Don't skip the shared expert computation.
                    shared_expert_output = func(moe_layer, *args, **kwargs)
                else:
                    # Skip the shared expert computation and get value from store.
                    shared_expert_output = moe_layer.cudagraph_tensor_store.shared_expert_output
                return shared_expert_output
            elif step_condition == "route":
                if moe_layer.cudagraph_tensor_store.probs is None:
                    # Don't skip the router.
                    assert (
                        moe_layer.cudagraph_tensor_store.routing_map is None
                        and moe_layer.cudagraph_tensor_store.residual is None
                    ), "both routing_map and residual must be None if probs is None"
                    probs, routing_map = func(moe_layer, *args, **kwargs)

                    # Maybe early return after the router.
                    maybe_raise_signal(moe_layer, probs=probs, routing_map=routing_map)
                else:
                    # Skip the router and get value from store.
                    assert (
                        moe_layer.cudagraph_tensor_store.routing_map is not None
                        or moe_layer.cudagraph_tensor_store.residual is not None
                    ), "either routing_map or residual must be given if probs is given"
                    probs, routing_map = (
                        moe_layer.cudagraph_tensor_store.probs,
                        moe_layer.cudagraph_tensor_store.routing_map,
                    )
                return probs, routing_map
            elif step_condition == "preprocess":
                if moe_layer.cudagraph_tensor_store.residual is None:
                    # Don't skip the preprocess.
                    hidden_states, probs, residual = func(moe_layer, *args, **kwargs)

                    # Maybe early return after the preprocess.
                    maybe_raise_signal(
                        moe_layer, hidden_states=hidden_states, probs=probs, residual=residual
                    )
                else:
                    # Skip the preprocess and get value from store.
                    assert (
                        moe_layer.cudagraph_tensor_store.probs is not None
                    ), "probs must not be None if residual is not None"
                    assert (
                        moe_layer.cudagraph_tensor_store.routing_map is None
                    ), "routing_map must be None if residual is not None"
                    hidden_states, probs, residual = (
                        moe_layer.cudagraph_tensor_store.hidden_states,
                        moe_layer.cudagraph_tensor_store.probs,
                        moe_layer.cudagraph_tensor_store.residual,
                    )
                return hidden_states, probs, residual

        return wrapped_func

    return decorator
