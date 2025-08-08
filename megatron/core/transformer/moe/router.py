# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Optional, Union

import torch
import triton
import triton.language as tl

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    ModelCommProcessGroups,
    MoEAuxLossAutoScaler,
    apply_random_logits,
    router_gating_linear,
    save_to_aux_losses_tracker,
    sequence_load_balancing_loss_func,
    sinkhorn,
    switch_load_balancing_loss_func,
    topk_softmax_with_capacity,
    z_loss_func,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class Router(ABC, MegatronModule):
    """Base Router class"""

    def __init__(
        self, config: TransformerConfig, model_comm_pgs: Optional[ModelCommProcessGroups] = None
    ) -> None:
        """
        Initialize the Router module.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
            model_comm_pgs (ModelCommProcessGroups, optional): Process groups for MoE operations.
        """
        super().__init__(config)
        self.config = config
        self.num_experts = self.config.num_moe_experts
        self.moe_aux_loss_func = None
        self.layer_number = None
        self.tp_group = model_comm_pgs.tp
        self.cp_group = model_comm_pgs.cp
        self.tp_cp_group = model_comm_pgs.tp_cp

        # Initialize the gate weights.
        # TODO: Add support for GPU initialization, which requires updating the golden values.
        self.weight = torch.nn.Parameter(
            torch.empty((self.config.num_moe_experts, self.config.hidden_size), dtype=torch.float32)
        )
        # If calculate per token loss, we need to scale up moe aux loss by the number of tokens.
        # So we need to know if the model is configured to calculate per token loss.
        self.calculate_per_token_loss = self.config.calculate_per_token_loss
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the router parameters."""
        if self.config.perform_initialization:
            self.config.init_method(self.weight)
        self.weight.data = self.weight.data.to(dtype=self.config.params_dtype)
        setattr(self.weight, 'sequence_parallel', self.config.sequence_parallel)

    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        if self.weight.device.type == 'cpu':
            # move weights to GPU
            self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
        # Convert to specified datatype for routing computation if enabled
        router_dtype = input.dtype
        if self.config.moe_router_dtype == 'fp32':
            router_dtype = torch.float32
        elif self.config.moe_router_dtype == 'fp64':
            router_dtype = torch.float64
        logits = router_gating_linear(input, self.weight, router_dtype)
        return logits

    @abstractmethod
    def routing(self, logits: torch.Tensor):
        """Routing function.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mapping.
        """
        raise NotImplementedError("Routing function not implemented.")

    @abstractmethod
    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        raise NotImplementedError("Forward function not implemented.")

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the router."""
        self.layer_number = layer_number


class TopKRouter(Router):
    """Route each token to the top-k experts."""

    def __init__(
        self, config: TransformerConfig, model_comm_pgs: Optional[ModelCommProcessGroups] = None
    ) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            model_comm_pgs (ModelCommProcessGroups, optional): Process groups for MoE operations.
        """
        super().__init__(config=config, model_comm_pgs=model_comm_pgs)
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.score_function = self.config.moe_router_score_function
        self.input_jitter = None

        self.enable_expert_bias = self.config.moe_router_enable_expert_bias
        if self.enable_expert_bias:
            self.register_buffer(
                'local_tokens_per_expert',
                torch.zeros(self.config.num_moe_experts, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                'expert_bias', torch.zeros(self.config.num_moe_experts, dtype=torch.float32)
            )
        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None

    def _maintain_float32_expert_bias(self):
        """
        Maintain the expert bias in float32.

        When using bf16/fp16, the expert bias gets converted to lower precision in Float16Module.
        We keep it in float32 to avoid routing errors when updating the expert_bias.
        """
        if hasattr(self, 'expert_bias') and self.expert_bias is not None:
            if self.expert_bias.dtype != torch.float32:
                self.expert_bias.data = self.expert_bias.data.to(torch.float32)

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mask.
        """

        def _sinkhorn_activation(logits):
            if self.topk == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            return logits

        assert self.config.moe_aux_loss_coeff == 0, "Sinkhorn routing does not support aux loss."
        if self.training:
            with torch.no_grad():
                norm_logits = sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.topk, dim=1)
            logits = _sinkhorn_activation(logits)
        else:
            logits = _sinkhorn_activation(logits)
            _, indices = torch.topk(logits, k=self.topk, dim=1)
        map = torch.zeros_like(logits).int().scatter(1, indices, 1).bool()
        scores = logits * map
        return scores, map

    def compute_routing_scores_for_aux_loss(self, logits: torch.Tensor):
        """Compute routing scores based on the score function.

        Args:
            logits (torch.Tensor): The logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            torch.Tensor: The normalized routing scores.
        """
        if self.score_function == "softmax":
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
        elif self.score_function == "sigmoid":
            scores = torch.sigmoid(logits)
            scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
        else:
            raise ValueError(f"Invalid score_function: {self.score_function}")

        _, top_indices = torch.topk(scores, k=self.topk, dim=1)
        topk_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()

        return scores, topk_map

    def aux_loss_load_balancing(self, logits: torch.Tensor):
        """Apply auxiliary loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mask of token to experts assignment.
        """
        probs, routing_map, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            deterministic_mode=self.config.deterministic_mode,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
        )

        if self.training and torch.is_grad_enabled():
            # Apply auxiliary load balancing loss
            # Skip auxiliary loss calculations when using torch.no_grad() or checkpointing.
            scores, loss_routing_map = self.compute_routing_scores_for_aux_loss(logits)
            aux_loss_func = partial(
                switch_load_balancing_loss_func,
                probs=scores,
                tokens_per_expert=loss_routing_map.sum(dim=0),
                topk=self.topk,
            )
            probs = self.apply_load_balancing_loss(
                activation=probs, load_balancing_loss_func=aux_loss_func
            )
        return probs, routing_map

    def seq_aux_loss_load_balancing(self, logits: torch.Tensor, bsz: int, seq_length: int):
        """Apply sequence-auxiliary loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor after gating, shape: [num_tokens, num_experts].
            bsz (int): The batch size.
            seq_length (int): The sequence length.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mask of token to experts assignment.
        """

        probs, routing_map, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            deterministic_mode=self.config.deterministic_mode,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
        )

        if self.training and torch.is_grad_enabled():
            # Apply sequence-auxiliary load balancing loss
            scores, loss_routing_map = self.compute_routing_scores_for_aux_loss(logits)
            aux_loss_func = partial(
                sequence_load_balancing_loss_func,
                probs=scores,
                routing_map=loss_routing_map,
                batch_size=bsz,
                seq_length=seq_length,
                topk=self.topk,
            )
            probs = self.apply_load_balancing_loss(
                activation=probs, load_balancing_loss_func=aux_loss_func
            )

        return probs, routing_map

    def apply_load_balancing_loss(
        self, activation: torch.Tensor, load_balancing_loss_func: Callable
    ):
        """Calculate auxiliary loss, attach gradient function to activation and add to logging."""
        moe_aux_loss_coeff = self.config.moe_aux_loss_coeff
        if moe_aux_loss_coeff == 0:
            return activation

        sequence_partition_group = None
        if self.tp_cp_group.size() > 1:
            sequence_partition_group = self.tp_cp_group

        aux_loss = load_balancing_loss_func(
            moe_aux_loss_coeff=moe_aux_loss_coeff, sequence_partition_group=sequence_partition_group
        )
        # TODO (zijiey): fix the per_layer_logging for MTP, currently it will incorrectly
        # add the aux loss logging value to other layer's since it is difficult to get the
        # correct layer_number for MTP. It does not affect the correctness of the calculation
        # results and the reduced load_balancing_loss logging value.
        num_layers = self.config.num_layers
        if self.config.mtp_num_layers is not None:
            num_layers += self.config.mtp_num_layers
        save_to_aux_losses_tracker(
            "load_balancing_loss",
            aux_loss / moe_aux_loss_coeff,
            self.layer_number,
            num_layers,
            reduce_group=sequence_partition_group,
        )
        if self.calculate_per_token_loss:
            # Scale the aux_loss by the number of tokens.
            # The expected final scaling for aux_loss gradients is 1/(num_micro_batches * dp_size).
            # After commit 02648000, Megatron started using the number of total tokens to scale
            # gradients under the argument of calculate_per_token_loss,
            # which scales both the main_loss gradient and aux_loss gradient by
            # 1/(num_local_tokens * dp_size * num_micro_batches) in finalize_model_grads function.
            # To correct this scaling, we need to scale the aux_loss by num_local_tokens here.
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss * activation.shape[0])
        else:
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.moe_z_loss_coeff is not None and self.training and torch.is_grad_enabled():
            # Skip Z loss calculations when using torch.no_grad() or checkpointing.
            moe_z_loss_coeff = self.config.moe_z_loss_coeff / self.tp_cp_group.size()
            z_loss = z_loss_func(logits, moe_z_loss_coeff)
            scale_up = 1.0
            if self.calculate_per_token_loss:
                # The expected final scaling for z_loss gradients is
                # 1/(num_micro_batches * dp_size).
                # After commit 02648000, Megatron started using the number of total tokens
                # to scale gradients under the argument of calculate_per_token_loss,
                # which scales both the main_loss gradient and z_loss gradient by
                # 1/(num_local_tokens * dp_size * num_micro_batches) in finalize_model_grads().
                # To correct this scaling, we need to scale the z_loss by num_local_tokens here.
                logits = MoEAuxLossAutoScaler.apply(logits, z_loss * logits.shape[0])
            else:
                logits = MoEAuxLossAutoScaler.apply(logits, z_loss)

            num_layers = self.config.num_layers
            if self.config.mtp_num_layers is not None:
                num_layers += self.config.mtp_num_layers
            save_to_aux_losses_tracker(
                "z_loss", z_loss / moe_z_loss_coeff, self.layer_number, num_layers
            )
        return logits

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """
        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        if self.routing_type == "sinkhorn":
            scores, routing_map = self.sinkhorn_load_balancing(logits)
        elif self.routing_type == "aux_loss":
            scores, routing_map = self.aux_loss_load_balancing(logits)
        elif self.routing_type == "seq_aux_loss":
            scores, routing_map = self.seq_aux_loss_load_balancing(logits, bsz, seq_length)
        elif self.routing_type == "none":
            # A naive top-k routing without load balancing
            scores, routing_map, _ = topk_softmax_with_capacity(
                logits,
                self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
                drop_policy=self.config.moe_token_drop_policy,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                num_groups=self.config.moe_router_num_groups,
                group_topk=self.config.moe_router_group_topk,
                scaling_factor=self.config.moe_router_topk_scaling_factor,
                deterministic_mode=self.config.deterministic_mode,
                score_function=self.score_function,
                expert_bias=self.expert_bias,
            )
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")
        # Prevent extra local tokens accumulation on evaluation or activation recomputation
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert += routing_map.sum(dim=0)

        return scores, routing_map

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        self._maintain_float32_expert_bias()

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)

        if self.config.moe_router_force_load_balancing:
            # Apply force load balancing with random logits for benchmark
            logits = apply_random_logits(logits)

        scores, routing_map = self.routing(logits)

        return scores, routing_map

    def _load_from_state_dict(self, *args, **kwargs):
        """Load the state dict of the router."""
        self._maintain_float32_expert_bias()  # switch to float32 before loading
        return super()._load_from_state_dict(*args, **kwargs)

    def _save_to_state_dict(self, *args, **kwargs):
        """Save the state dict of the router."""
        self._maintain_float32_expert_bias()  # switch to float32 before saving
        return super()._save_to_state_dict(*args, **kwargs)

def one_shot_greedy_assignment(
    token_chunks: torch.Tensor,
    buckets: torch.Tensor
) -> torch.Tensor:
    """
    Perform one-shot greedy assignment between token chunks and buckets.

    This function implements a greedy algorithm that assigns token chunks to buckets
    in a single pass. It calculates the overlap between cumulative token chunks and
    cumulative bucket capacities to determine an assignment strategy.

    The algorithm works by:
    1. Computing cumulative sums for both token chunks and buckets
    2. Finding overlapping ranges between token chunk intervals and bucket intervals
    3. Calculating the overlap amount as the assignment

    Args:
        token_chunks: Token chunks to be assigned
            Shape: [num_token_chunks]
            Type: torch.Tensor (int)
            Description: Each element represents the number of tokens in a chunk
                       that needs to be assigned to buckets

        buckets: Bucket capacities available for assignment
            Shape: [num_buckets]
            Type: torch.Tensor (int)
            Description: Each element represents the capacity of a bucket that can
                       receive token chunks

    Returns:
        overlap: Assignment matrix showing token chunk to bucket assignments
            Shape: [num_token_chunks, num_buckets]
            Type: torch.Tensor (same dtype as inputs)
            Description: Element [i, j] represents how many tokens from chunk i
                       are assigned to bucket j. Zero if no assignment.
    """
    token_chunks_cumsum = torch.cumsum(token_chunks, dim=0)
    buckets_cumsum = torch.cumsum(buckets, dim=0)
    token_chunks_start = token_chunks_cumsum - token_chunks
    buckets_start = buckets_cumsum - buckets
    token_chunks_start = token_chunks_start.unsqueeze(1)  # (num_token_chunks, 1)
    token_chunks_end = token_chunks_cumsum.unsqueeze(1)   # (num_token_chunks, 1)
    buckets_start = buckets_start.unsqueeze(0)  # (1, num_buckets)
    buckets_end = buckets_cumsum.unsqueeze(0)   # (1, num_buckets)
    overlap_start = torch.maximum(token_chunks_start, buckets_start)
    overlap_end = torch.minimum(token_chunks_end, buckets_end)
    overlap = (overlap_end - overlap_start).clamp(min=0)
    return overlap

def reclaim_spare_experts(
    num_token_to_ep_rank: torch.Tensor,
    avg_token_to_ep_rank: float,
    matched_assignment: torch.Tensor,
    threshold_multiplier: float
) -> torch.Tensor:
    """
    Reclaim spare experts by selectively disabling some spare experts.

    This function implements a threshold-based selection algorithm to reduce the
    number of active spare experts by disabling some offloading assignments based 
    on load balancing criteria. It helps prevent over-allocation of spare experts
    when the load is already well-balanced.

    The algorithm works by:
    1. Calculating current token loads per EP rank after offloading
    2. Determining a threshold based on average load and multiplier
    3. Sorting offloading assignments by size for each EP rank
    4. Using binary search to find which assignments can be safely disabled
    5. Creating a mask to disable assignments that exceed capacity thresholds

    Args:
        num_token_to_ep_rank: Current token distribution per EP rank
            Shape: [num_ep_ranks]
            Type: torch.Tensor (int)
            Description: Number of tokens currently assigned to each EP rank

        avg_token_to_ep_rank: Average tokens per EP rank
            Type: int
            Description: Average token load across all EP ranks, used as baseline
                       for threshold calculations

        matched_assignment: Current offloading assignment matrix
            Shape: [num_home_experts, num_spare_experts]
            Type: torch.Tensor (int)
            Description: Matrix where [i, j] indicates tokens offloaded from home
                       expert i to spare expert j

        threshold_multiplier: Multiplier for threshold calculation
            Type: float
            Description: Controls how aggressive the filtering is. Lower values
                       result in more spare experts being disabled. If <= 0,
                       no filtering is applied.

    Returns:
        matched_assignment: Updated assignment matrix with some assignments disabled
            Shape: [num_home_experts, num_spare_experts]
            Type: torch.Tensor (int)
            Description: Same shape as input but with some offloading assignments
                       set to zero based on threshold criteria

    """
    # Calculate current total tokens per home expert (without offloading)
    ep = num_token_to_ep_rank.shape[0]
    num_home_expert, num_spare_expert = matched_assignment.shape
    current_tokens_home_rank = num_token_to_ep_rank.view(ep, -1).sum(dim=1)
    threshold = threshold_multiplier * avg_token_to_ep_rank
    max_allowed_load = avg_token_to_ep_rank + threshold
    
    # Calculate current home expert loads after offloading
    current_tokens_after_offloading = current_tokens_home_rank - matched_assignment.sum(dim=1).view(ep, -1).sum(dim=1)  
    
    # Sort each row in ascending order (zeros will naturally be at the beginning)
    matched_assignment_ep_rank_view = matched_assignment.view(ep, num_home_expert//ep, -1).sum(dim=1)
    sorted_tokens, sorted_indices = torch.sort(matched_assignment_ep_rank_view, dim=1)
    cumulative_tokens = torch.cumsum(sorted_tokens, dim=1)
    
    # Use searchsorted to find where cumulative tokens exceed capacity for each home expert
    # We need to find where cumulative_tokens > (max_allowed_load - current_tokens_after_offloading)
    # safe_steps contains the last spare expert index needed for each EP rank (after sorting)
    capacity_remaining = max_allowed_load - current_tokens_after_offloading 
    safe_steps = torch.searchsorted(cumulative_tokens, capacity_remaining.unsqueeze(1), side='right').squeeze(1)
    
    # Create a boolean mask based on safe_steps with the same shape as sorted_tokens
    # True indicates positions that need to be enabled (after the boundary)
    step_indices = torch.arange(sorted_tokens.shape[1], device=matched_assignment.device).unsqueeze(0).expand(sorted_tokens.shape[0], -1)
    safe_steps_expanded = safe_steps.unsqueeze(1).expand(-1, sorted_tokens.shape[1])
    sorted_mask = step_indices >= safe_steps_expanded  
    
    # Recover the mask to the original order using sorted_indices
    # Use scatter to map sorted_mask back to original order
    original_order_mask = torch.zeros_like(sorted_mask)
    original_order_mask.scatter_(1, sorted_indices, sorted_mask)
    
    # Do OR operation along dim=0 to get a 1D tensor indicating which columns to mask
    column_mask = original_order_mask.any(dim=0) 
    
    # Apply the column mask directly to matched_assignment
    matched_assignment = torch.where(column_mask.unsqueeze(0), matched_assignment, torch.zeros_like(matched_assignment))
    return matched_assignment

def breadth_first_allocation(transport, reroute_map, device='cuda'):
    """
    Ultra-compact loop-free version with over-allocation protection
    """
    transport, reroute_map = transport.to(device).float(), reroute_map.to(device).float()
    # Initial allocation
    supplier = reroute_map.argmax(0)
    active = (reroute_map > 0).sum(0) > 0
    capacity = reroute_map[supplier, torch.arange(reroute_map.shape[1], device=device)]
    
    x_rel = transport[:, supplier]
    # Use safe division to avoid NaN and division by zero
    denominator = x_rel.sum(0, keepdim=True)
    props = torch.where(denominator > 0, x_rel / denominator, torch.zeros_like(x_rel))
    ideal = props * capacity
    floors = torch.floor(ideal).int()*active

    offloaded = torch.zeros_like(transport, dtype=torch.int32)
    supplier_expanded = supplier.unsqueeze(0).expand(transport.shape[0], -1)  # (num_src, num_new_dst)
    
    # For each new destination k, add t[:, k] to the reduction for old destination supplier[k]
    offloaded.scatter_add_(1, supplier_expanded, floors)
    transport_rerouted = transport - offloaded
    leftover_spare_space = reroute_map.clone()
    leftover_spare_space[supplier, torch.arange(reroute_map.shape[1], device=device)] -= floors.sum(dim=0)
    return floors, transport_rerouted, leftover_spare_space

def depth_first_allocation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Vectorized batched version of one_shot_greedy_assignment for token routing composition.
    """
    S, J = x.shape
    J2, K = y.shape
    assert J == J2, "Dimension mismatch"
    
    # For each old destination j, x[:,j] are the token chunks
    # Compute cumulative intervals for token chunks along source dimension
    x_cumsum = torch.cumsum(x, dim=0)  # (S, J)
    x_start = x_cumsum - x             # (S, J) - start positions for each chunk
    x_end = x_cumsum                   # (S, J) - end positions for each chunk
    
    # Compute bucket intervals for each old destination
    y_cumsum = torch.cumsum(y, dim=1)  # (J, K)
    y_start = y_cumsum - y             # (J, K) - bucket start positions
    y_end = y_cumsum                   # (J, K) - bucket end positions
    
    # Expand dimensions for broadcasting
    x_start = x_start.unsqueeze(2)     # (S, J, 1)
    x_end = x_end.unsqueeze(2)         # (S, J, 1)
    y_start = y_start.unsqueeze(0)     # (1, J, K)
    y_end = y_end.unsqueeze(0)         # (1, J, K)
    
    # Compute overlaps for all (source, old_dst, new_dst) combinations
    overlap_start = torch.maximum(x_start, y_start)  # (S, J, K)
    overlap_end = torch.minimum(x_end, y_end)        # (S, J, K)
    overlap = (overlap_end - overlap_start).clamp(min=0)  # (S, J, K)
    
    # Sum over intermediate destinations to get final source->new_dst flows
    z = overlap.sum(dim=1)  # (S, K)
    supplier = y.argmax(0)
    supplier_expanded = supplier.unsqueeze(0).expand(x.shape[0], -1)  # (num_src, num_new_dst)
    x_rerouted = x.scatter_add(1, supplier_expanded, -z)
    return z, x_rerouted

@triton.jit
def reroute_tokens_kernel(
    x_indices_sorted_ptr, 
    expert_for_offload_ptr,
    num_tokens_to_route_ptr,
    cumulative_offsets_ptr,
    y_ptr,
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    num_offloading_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offload_expert_id = tl.program_id(0)
    
    source_expert_id = tl.load(expert_for_offload_ptr + offload_expert_id).to(tl.int64)
    if source_expert_id < 0:
        return
    num_tokens_to_route = tl.load(num_tokens_to_route_ptr + offload_expert_id).to(tl.int64)
    offset = tl.load(cumulative_offsets_ptr + offload_expert_id).to(tl.int64)
    
    token_positions = tl.arange(0, BLOCK_SIZE)
    valid_mask = (token_positions < num_tokens_to_route)
    
    # No conditional check needed - masked operations handle empty cases
    base_offset = source_expert_id * num_tokens + offset
    token_indices = tl.load(x_indices_sorted_ptr + base_offset + token_positions, 
                           mask=valid_mask, other=0)
    
    total_experts = num_experts + num_offloading_experts
    
    # These stores are safe even with all-False masks
    x_flat_idx = token_indices * total_experts + source_expert_id
    tl.store(y_ptr + x_flat_idx, False, mask=valid_mask)
    
    offload_col = num_experts + offload_expert_id
    y_flat_idx = token_indices * total_experts + offload_col
    tl.store(y_ptr + y_flat_idx, True, mask=valid_mask)


def reroute_tokens_triton(x, num_offloading_from, num_offloading_to, reroute_map):
    """
    Triton-based token rerouting with static shapes for CUDA graph compatibility.
    """
    device = x.device
    num_tokens, num_experts = x.shape
    num_offloading_experts = num_offloading_to.shape[0]
    num_offloading_to = num_offloading_to.to(torch.int64)
    # Step 1: Generate reroute_map2
    reroute_map2 = reroute_map * num_offloading_to.unsqueeze(0)
    
    # Step 2: Compute cumulative offsets
    cumsum_routes = torch.cumsum(reroute_map2, dim=1)
    cumulative_starts = cumsum_routes - reroute_map2
    
    # Step 3: Extract expert mapping 
    has_routing = reroute_map.any(dim=0)
    expert_indices = torch.argmax(reroute_map.float(), dim=0)
    expert_for_offload = torch.where(has_routing, expert_indices, -1)
    oe_indices = torch.arange(num_offloading_experts, device=device)
    
    # Clamp expert indices to valid range for safe indexing (invalid entries will be ignored anyway)
    safe_expert_indices = torch.clamp(expert_for_offload, 0, num_experts - 1)
    # Extract offsets for ALL offloading experts using static indexing
    all_offsets = cumulative_starts[safe_expert_indices, oe_indices]
    
    # Use torch.where to zero out offsets for invalid offloading experts
    cumulative_offsets = torch.where(expert_for_offload >= 0, all_offsets, 0)
    
    # Convert expert_for_offload to float for Triton compatibility
    expert_for_offload = expert_for_offload
    
    # Step 5: Create sorted indices
    x_indices_sorted = x.argsort(dim=0, descending=True).transpose(0, 1).contiguous()
    
    # Step 6: Initialize output tensor
    y = torch.zeros(num_tokens, num_experts + num_offloading_experts, 
                    dtype=torch.bool, device=device)
    y[:, :num_experts] = x.clone()
    
    # Step 7: Launch Triton kernel
    max_tokens = num_tokens
    BLOCK_SIZE = triton.next_power_of_2(max_tokens)
    grid = (num_offloading_experts,)
    
    reroute_tokens_kernel[grid](
        x_indices_sorted, expert_for_offload, num_offloading_to,
        cumulative_offsets, y, num_tokens, num_experts, num_offloading_experts, BLOCK_SIZE
    )
    return y

def gen_offloading_plan(
    routing_map: torch.Tensor,
    tokens_per_expert_from_ep_rank: torch.Tensor,
    ep_rank: Union[torch.Tensor, int],
    ep: int,
    spare_expert_per_ep_rank: int = 1,
    threshold_multiplier: float = 0.0,
    index_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate an offloading plan to redistribute tokens from overloaded experts to spare experts.

    This function implements a balanced routing algorithm that redistributes tokens
    from overloaded experts to spare experts to achieve better load balancing.
    The algorithm works in phases:
    1. Calculate spillover tokens that need to be offloaded from each expert
    2. Use greedy assignment to match spillover tokens to spare expert capacity
    3. Apply threshold-based filtering to disable some offloading assignments
    4. Execute the offloading plan using breadth-first and depth-first allocation
    5. Generate the final rerouting map with token movements

    Args:
        routing_map: Binary routing map for current EP rank
            Shape: [num_tokens_in_ep_rank, num_experts]
            Type: torch.Tensor (bool)
            Description: Binary tensor indicating which tokens are routed to which
                       experts within the current EP rank

        tokens_per_expert_from_ep_rank: Token distribution per expert from all EP ranks
            Shape: [num_ep_ranks, num_experts_per_ep_rank]
            Type: torch.Tensor (int)
            Description: Number of tokens assigned to each expert across all EP ranks

        ep_rank: Current EP rank index
            Type: Union[torch.Tensor, int]
            Description: Index of the current expert parallel rank (0 to ep-1)

        ep: Number of expert parallel ranks
            Type: int
            Description: Total number of expert parallel ranks

        spare_expert_per_ep_rank: Number of spare experts per EP rank
            Type: int
            Default: 1
            Description: How many spare experts are available for offloading per
                       EP rank

        threshold_multiplier: Multiplier for threshold-based expert selection
            Type: float
            Default: 0.0
            Description: If > 0, enables threshold-based selection to disable
                       some spare experts based on average token load. Lower
                       values = more aggressive filtering.

        index_dtype: Data type for index tensors
            Type: torch.dtype
            Default: torch.int32
            Description: Data type used for index tensors in the computation

    Returns:
        tuple containing:
            - rerouting_map: Updated routing map after offloading
                Shape: [num_tokens_in_ep_rank, num_experts + num_spare_experts]
                Type: torch.Tensor (bool)
                Description: Binary tensor showing final token assignments including
                           offloaded tokens to spare experts

            - expert_offloading_map: Mapping of home experts to spare experts
                Shape: [num_home_experts, num_spare_experts]
                Type: torch.Tensor (bool)
                Description: Boolean matrix where [i,j] = True indicates home expert
                           i offloads tokens to spare expert j
    """
    # Phase 1: calculate how many tokens need to be offloaded from home experts
    device = routing_map.device
    num_tokens_to_expert = tokens_per_expert_from_ep_rank.sum(dim=0).to(index_dtype)
    num_token_to_ep_rank = num_tokens_to_expert.view(ep, -1).sum(dim=1)
    avg_token_to_ep_rank = num_token_to_ep_rank.sum() // ep
    deviation = num_token_to_ep_rank - avg_token_to_ep_rank
    spare_space = torch.relu(-deviation)

    # sort the local experts by token count and place smaller token chunk first
    local_exp_sorted_token_count, local_exp_sorted_idx = num_tokens_to_expert.view(ep, -1).sort(dim=1)
    spillover_tokens_per_exp_sorted_cumsum = (local_exp_sorted_token_count.cumsum(dim=1) - avg_token_to_ep_rank).clamp(min=0)
    spillover_tokens_per_exp_sorted = torch.cat([spillover_tokens_per_exp_sorted_cumsum[:, :1], torch.diff(spillover_tokens_per_exp_sorted_cumsum, dim=1)], dim=1)
    spillover_tokens_per_exp = torch.scatter(torch.empty_like(spillover_tokens_per_exp_sorted), 1, local_exp_sorted_idx, spillover_tokens_per_exp_sorted).view(-1)
    
    # [num_home_experts, num_spare_experts]
    # assignment[i][j] indicates how many tokens are offloaded from home expert i to spare expert j
    assignment = one_shot_greedy_assignment(spillover_tokens_per_exp, spare_space)

    # Find top spare_expert_per_ep_rank token chunks for each EP rank
    spare_bucket_max, spare_bucket_max_index = torch.topk(assignment, k=spare_expert_per_ep_rank, dim=0)
    num_columns = assignment.shape[1]
    matched_assignment = torch.zeros(assignment.shape[0], num_columns * spare_expert_per_ep_rank, device=device, dtype=assignment.dtype)
    row_indices = spare_bucket_max_index.transpose(0, 1).flatten()
    col_indices = torch.arange(num_columns, device=device).repeat_interleave(spare_expert_per_ep_rank) \
                  * spare_expert_per_ep_rank \
                  + torch.arange(spare_expert_per_ep_rank, device=device).repeat(num_columns)
    values = spare_bucket_max.transpose(0, 1).flatten()
    # [num_home_experts, num_spare_experts]
    # matched_assignment[i][j] indicates how many tokens are offloaded from home expert i to spare expert j
    matched_assignment[row_indices, col_indices] = values
    # import pdb; pdb.set_trace()    
    # Apply threshold-based offloading expert selection
    if threshold_multiplier > 0:
        matched_assignment = reclaim_spare_experts(
            num_token_to_ep_rank, avg_token_to_ep_rank, matched_assignment, threshold_multiplier
        )
    
    # Create expert_offloading_map with shape (num_home_experts, num_spare_experts)
    # expert_offloading_map[i][j] indicates if home expert i is offloaded to spare expert j
    export_offloading_map = matched_assignment > 0
    
    offloaded_tokens, token_dist_after_offloading, leftover_spare_space = breadth_first_allocation(tokens_per_expert_from_ep_rank, matched_assignment)
    offloaded_tokens2, token_dist_after_offloading = depth_first_allocation(token_dist_after_offloading, leftover_spare_space)
    rerouting_map = reroute_tokens_triton(routing_map, 
                                        (tokens_per_expert_from_ep_rank-token_dist_after_offloading)[ep_rank].int(), 
                                        (offloaded_tokens+offloaded_tokens2)[ep_rank].int().squeeze(), 
                                        export_offloading_map)
    return rerouting_map, export_offloading_map