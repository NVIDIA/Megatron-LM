# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import math

import torch

from megatron.core import parallel_state


def switch_load_balancing_loss_func(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    moe_aux_loss_coeff: float,
    sequence_partition_group=None,
):
    """Calculate the auxiliary loss for load balancing.
    Refer to the Switch Transformer paper (https://arxiv.org/abs/2101.03961) for details.

    Args:
        probs (torch.Tensor): Softmax probabilities output by the router for each token.
                              Shape in [num_tokens, num_experts].
        tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert.
                                          Shape in [num_experts]
        topk (int): The number of experts selected for each token.
        moe_aux_loss_coeff (float): The coefficient for the auxiliary loss.
        sequence_partition_group (optional): The parallel group over which the sequence is
                                             partitioned. If None, no partitioning is applied.
                                             Defaults to None.

    Returns:
        torch.Tensor: The auxiliary loss for load balancing.
    """
    num_sub_sequence = 1

    # If the sequence is partitioned by certain parallelism strategies like Sequence Parallelism
    # or Context Parallelism, compute the gradient of the auxiliary loss with respect to the full
    # sequence.
    if sequence_partition_group is not None:
        # We can keep `aggregated_probs_per_expert` local since we don't need the gradient for
        # `tokens_per_expert`, saving one allreduce operation for `aggregated_probs_per_expert`.
        num_sub_sequence = torch.distributed.get_world_size(sequence_partition_group)
        torch.distributed.all_reduce(tokens_per_expert, group=sequence_partition_group)

    num_tokens = probs.shape[0] * num_sub_sequence
    num_experts = probs.shape[1]

    # The formula of aux_loss: aux_loss = sum((probs_per_expert/num_tokens) *
    # (tokens_per_expert/(num_tokens*topk))) * num_experts * moe_aux_loss_coeff.
    # This can be simplified to fuse the division and multiplication operations.
    aggregated_probs_per_expert = probs.sum(dim=0)
    aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (num_tokens * num_tokens * topk)
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
    """An AutoScaler that compute and scales the grad for auxiliary loss."""

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
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled auxiliary loss
                                               gradient.
        """
        (aux_loss,) = ctx.saved_tensors
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
        MoEAuxLossAutoScaler.main_loss_backward_scale = scale


def permute(tokens, routing_map, num_out_tokens: int = None):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
        num_out_tokens (int, optional): The number of output tokens. If None, it's set to
                                        the number of input tokens.
    """
    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]

    # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
    routing_map = routing_map.bool().T.contiguous()

    # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
    token_indices = (
        torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    )
    sorted_indices = token_indices.masked_select(routing_map)

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, sorted_indices


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: torch.Tensor = None,
    routing_map: torch.Tensor = None,
):
    """
    Restore the original order of tokens after permutation. If probs are provided, it
    will also apply them to the tokens before restoring the order.

    Args:
        permuted_tokens (torch.Tensor): The permuted token tensor.
        sorted_indices (torch.Tensor): The indices used to sort the tokens.
        restore_shape (torch.Size): The shape of the unpermuted tensor.
        probs (torch.Tensor, optional): The unpermuted probs tensor,
        routing_map (torch.Tensor, optional): Token to expert mapping, shape
            [num_tokens, num_experts].

    Returns:
        torch.Tensor: The tokens restored to their original order.
    """
    _, hidden = restore_shape

    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    # Create an output tensor filled with zeros
    output_tokens = torch.zeros(
        restore_shape, device=permuted_tokens.device, dtype=permuted_tokens.dtype
    )
    # Scatter add the permuted_input back to the original positions
    output_tokens.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)
    return output_tokens


def sort_chunks_by_idxs(input: torch.Tensor, split_sizes: torch.Tensor, sorted_idxs: torch.Tensor):
    """Split and sort the input tensor based on the split_sizes and sorted indices."""
    input = torch.split(input, split_sizes.tolist(), dim=0)
    output = torch.cat([input[i] for i in sorted_idxs], dim=0)
    return output


def topk_softmax_with_capacity(
    logits: torch.Tensor,
    topk: int,
    capacity_factor: float = None,
    pad_to_capacity: bool = False,
    drop_policy: str = "probs",
    use_pre_softmax: bool = False,
    deterministic_mode: bool = False,
):
    """Apply capacity and padding to the top-k selection.
    Args:
        logits (torch.Tensor): Logits tensor.
        topk (int): The number of experts to select for each token.
        capacity_factor (int): The capacity factor of each expert. Will drop tokens if the number
                               of tokens exceeds the capacity.
        pad_to_capacity (bool): Whether to need padding in token drop mode.
        drop_policy (str): The policy to drop tokens. Can be either "prob" or "position".
                           If "prob", the tokens with the lowest probabilities will be dropped.
                           If "position", tokens at the end of each batch will be dropped.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - routing_probs (torch.Tensor): A tensor of shape [num_tokens, num_experts] containing
              the routing probabilities for each token to each expert.
            - routing_map (torch.Tensor): A mask tensor of shape [num_tokens, num_experts]
              indicating which experts were selected for each token. True values represent
              the selected experts.
            - tokens_per_expert (torch.Tensor): A tensor of shape [num_experts] containing
              the number of local tokens assigned to each expert.
    """
    assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."
    num_tokens = logits.shape[0]
    num_experts = logits.shape[1]
    if use_pre_softmax:
        # Pre softmax
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
        probs, top_indices = torch.topk(scores, k=topk, dim=1)
    else:
        # Post softmax
        if topk == 1:
            # Requires applying softmax before selecting the top-k when k is 1,
            # since softmax on a [num_tokens, 1] would yield a zero gradient.
            raise ValueError("Please use --moe-router-pre-softmax when topk is 1.")
        scores, top_indices = torch.topk(logits, k=topk, dim=1)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)

    # TODO Try using element-wise operations instead of scatter?
    topk_masked_gates = torch.zeros_like(logits).scatter(1, top_indices, probs)
    topk_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    tokens_per_expert = topk_map.sum(dim=0)

    if capacity_factor is None:
        # TopK without capacity
        return topk_masked_gates, topk_map, tokens_per_expert
    else:
        # TopK with capacity
        expert_capacity = get_capacity(
            num_tokens=num_tokens * topk, num_experts=num_experts, capacity_factor=capacity_factor
        )

        # Maskout exceeded tokens
        if drop_policy == "probs":
            _, capacity_indices = torch.topk(
                topk_masked_gates, k=expert_capacity, dim=0, sorted=False
            )
            capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1).bool()
        elif drop_policy == "position":
            _, capacity_indices = torch.topk(topk_map.int(), k=expert_capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1).bool()
        else:
            raise ValueError(f"Invalid drop_policy: {drop_policy}")

        if pad_to_capacity:
            final_map = capacity_mask
            final_probs = topk_masked_gates * final_map
        else:
            # Get exceed mask and maskout exceeded probs and indices
            final_map = torch.logical_and(topk_map, capacity_mask)
            final_probs = topk_masked_gates * final_map
        return final_probs, final_map, tokens_per_expert


def save_to_aux_losses_tracker(
    name: str,
    loss: torch.Tensor,
    layer_number: int,
    num_layers: int,
    reduce_group: torch.distributed.ProcessGroup = None,
    avg_group: torch.distributed.ProcessGroup = None,
):
    """Save the auxiliary loss for logging.
    Args:
        name (str): The name of the loss.
        loss (torch.Tensor): The loss tensor.
        layer_number (int): Layer index of the loss.
        num_layers (int): The number of total layers.
        reduce_group (torch.distributed.ProcessGroup): The group for reducing the loss.
        mean_group (torch.distributed.ProcessGroup): The group for averaging the loss.
    """
    # Skip aux loss logging if layer_number is None.
    if layer_number is None:
        return

    tracker = parallel_state.get_moe_layer_wise_logging_tracker()
    if name not in tracker:
        tracker[name] = {}
        tracker[name]["values"] = torch.zeros(num_layers, device=loss.device)
    tracker[name]["values"][layer_number - 1] += loss.detach()  # Aggregate the loss for the layer.
    tracker[name]["reduce_group"] = reduce_group
    tracker[name]["avg_group"] = avg_group


def clear_aux_losses_tracker():
    """Clear the auxiliary losses."""
    tracker = parallel_state.get_moe_layer_wise_logging_tracker()
    for name in tracker:
        tracker[name]["values"].zero_()
        tracker[name]["reduce_group"] = None
        tracker[name]["avg_group"] = None


def reduce_aux_losses_tracker_across_ranks():
    """Collect and reduce the auxiliary losses across ranks."""
    tracker = parallel_state.get_moe_layer_wise_logging_tracker()
    for name in tracker:
        values = tracker[name]["values"]
        # Collect aux losses across PP.
        torch.distributed.all_reduce(
            values, group=parallel_state.get_pipeline_model_parallel_group()
        )
        # Reduce aux losses across ranks.
        if tracker[name].get('reduce_group') is not None:
            torch.distributed.all_reduce(values, group=tracker[name].get('reduce_group'))
        if tracker[name].get('avg_group') is not None:
            torch.distributed.all_reduce(
                values, group=tracker[name]['avg_group'], op=torch.distributed.ReduceOp.AVG
            )


def track_moe_metrics(
    loss_scale, iteration, writer, wandb_writer=None, total_loss_dict=None, per_layer_logging=False
):
    """Track the MoE metrics for logging."""
    # Aux loss logging
    reduce_aux_losses_tracker_across_ranks()
    tracker = parallel_state.get_moe_layer_wise_logging_tracker()
    if writer is not None:
        aux_losses = {k: v['values'].float() * loss_scale for k, v in tracker.items()}
        for name, loss_list in aux_losses.items():
            if total_loss_dict is not None:
                if name not in total_loss_dict:
                    total_loss_dict[name] = loss_list.mean()
                else:
                    total_loss_dict[name] += loss_list.mean()

            # currently when using add_scalars,
            # torch.utils.add_scalars makes each timer its own run, which
            # polutes the runs list, so we just add each as a scalar
            writer.add_scalar(name, loss_list.mean(), iteration)
            if per_layer_logging:
                for i, loss in enumerate(loss_list.tolist()):
                    writer.add_scalar(f"moe/{name}_layer_{i}", loss, iteration)

            # W&B logging lacks support for logging multiple scalars simultaneously.
            # As a workaround, we log each scalar individually first, then we can create
            # a custom panel to manually group them to a single plot.
            if wandb_writer:
                wandb_writer.log({f"{name}": loss_list.mean()}, iteration)
                if per_layer_logging:
                    wandb_writer.log(
                        {
                            f"moe/{name}_layer_{i}": loss
                            for i, loss in enumerate(loss_list.tolist())
                        },
                        iteration,
                    )

    clear_aux_losses_tracker()
