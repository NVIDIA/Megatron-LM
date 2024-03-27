# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch


def switch_load_balancing_loss_func(gates, mask, moe_aux_loss_coeff):
    """Calculate the auxiliary loss for better load balacing. 
    Please refer to the Switch Transformer paper (https://arxiv.org/abs/2101.03961) for details.

    Args:
        gates (torch.Tensor): The gates tensor representing the routing probabilities for each expert.
        mask (torch.Tensor): The 2D mask tensor indicating which experts are selected.

    Returns:
        torch.Tensor: The auxiliary loss for load balancing.
    """
    num_experts = mask.size(-1)
    gates_mean = gates.mean(dim=0)
    top_k = mask[0].count_nonzero()
    selection_mean = mask.float().mean(dim=0) / top_k
    aux_loss = torch.sum(gates_mean * selection_mean) * num_experts
    aux_loss *= moe_aux_loss_coeff
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


class MoEAuxLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that compute and scales the grad for auxiliary loss.

    """

    main_loss_backward_scale: torch.Tensor = torch.tensor(1.0)

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
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled auxiliary loss gradient.
        """
        (aux_loss,) = ctx.saved_tensors
        aux_loss_backward_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
        scaled_aux_loss_grad = torch.ones_like(aux_loss) * aux_loss_backward_scale
        return grad_output, scaled_aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the aux loss.
        
        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in matches the scale of the main_loss.
        """
        MoEAuxLossAutoScaler.main_loss_backward_scale = scale


def permute(tokens, indices, topk: int = 1):
    """Permute the tokens based on the indices. Token with the same index will be grouped together.

    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): The token to expert indices tensor, should have a shape of [num_tokens, topk].
        topk (int, optional): The topk value. Defaults to 1.

    Returns:
        torch.Tensor: The permuted tensor.
    """
    if topk > 1:
        assert indices.size(1) == topk
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices


def unpermute(permuted_tokens, sorted_indices, probs: torch.Tensor = None, topk: int = 1):
    """Unpermute a tensor of permuted tokens based on sorted indices, and optionally merge the tokens with their corresponding probabilities.

    Args:
        permuted_tokens (torch.Tensor): The tensor of permuted tokens to be unpermuted.
        sorted_indices (torch.Tensor): The tensor of sorted indices used to unpermute the tokens.
        probs (torch.Tensor, optional): The tensor of probabilities corresponding to the permuted tokens. If provided, the unpermuted tokens will be merged with their respective probabilities.
        topk (int, optional): The number of top tokens to consider for merging with probabilities. Defaults to 1.
    """
    if topk > 1:
        assert probs is not None
        assert (
            probs.size(0) == permuted_tokens.size(0) // topk
        ), f"{probs.size()} {permuted_tokens.size()}"
    if probs is not None:
        assert probs.size(0) == permuted_tokens.size(0) // topk
        assert probs.size(1) == topk, f"probs size {probs.size()} merge_factor {topk}"

    unpermuted_tokens = torch.zeros_like(permuted_tokens)
    unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)

    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))

    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)

    unpermuted_tokens = unpermuted_tokens.sum(dim=1)

    return unpermuted_tokens
