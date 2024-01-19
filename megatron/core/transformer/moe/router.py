# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import math
from abc import ABC, abstractmethod
from typing import Callable, List

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.parallel_state import get_tensor_and_expert_parallel_group
from megatron.core.tensor_parallel import get_cuda_rng_tracker, get_data_parallel_rng_tracker_name
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    sinkhorn,
    switch_load_balancing_loss_func,
    z_loss_func,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class Router(ABC, MegatronModule):
    """Base Router class"""

    def __init__(self, config: TransformerConfig) -> None:
        """
        Initialize the Router module.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
        """
        super().__init__(config)
        self.config = config
        self.num_experts = self.config.num_moe_experts
        # Token dispatcher for exchange tokens between experts.
        self.token_dispatcher = None
        self.moe_aux_loss_func = None

        # Initialize the gate weights.
        self.weight = torch.nn.Parameter(
            torch.empty((self.config.num_moe_experts, self.config.hidden_size))
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
            config.init_method(self.weight)
        setattr(self.weight, 'sequence_parallel', config.sequence_parallel)

    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        logits = torch.nn.functional.linear(input, self.weight)
        return logits

    @abstractmethod
    def routing(self, logits: torch.Tensor):
        """Routing function.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of tensors representing max probs and the indices.
        """
        raise NotImplementedError("Routing function not implemented.")

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: scores and indices.
        """
        self.hidden = input.shape[-1]

        logits = self.gating(input)
        logits = logits.view(-1, self.config.num_moe_experts)

        scores, indices = self.routing(logits)

        return scores, indices

    def apply_aux_loss(
        self,
        loss_func: Callable,
        probs: torch.Tensor,
        indices: torch.Tensor,
        activation: torch.Tensor,
    ):
        """Applies auxiliary loss to the MoE layer.

        Args:
            loss_func (callable): The loss function to be used.
            probs (torch.Tensor): The probabilities output by the MoE layer.
            indices (torch.Tensor): The indices of the selected experts.
            activation (torch.Tensor): The activation tensor to attach the gradient function to.

        Returns:
            torch.Tensor: The activation tensor with the attached gradient function.
        """
        mask = torch.nn.functional.one_hot(indices, num_classes=self.num_experts).sum(dim=1)
        aux_loss = loss_func(probs, mask, self.config.moe_aux_loss_coeff)
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
        if self.config.moe_z_loss_coeff is not None:
            z_loss = z_loss_func(logits)
            logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
        return logits


class MoETokenDispatcher:
    """
    MoE Token Dispatcher
    """

    def __init__(self, config: TransformerConfig) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        self.config = config

    @abstractmethod
    def dispatch(
        self, tokens: torch.Tensor, indices: torch.Tensor,
    ):
        """Dispatch tokens to experts.

        Args:
            tokens (torch.Tensor): Input tokens.
            indices (torch.Tensor): indices tensor.

        Returns:
            torch.Tensor: Tokens tensor.
        """
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def restore(
        self, expert_output: torch.Tensor, scores: torch.Tensor, indices: torch.Tensor,
    ):
        """Restores the expert output to its original ordering.

        Args:
            expert_output (torch.Tensor): The output tensor from the expert models.
            scores (torch.Tensor): Each token's score with each expert.
            indices (torch.Tensor): The indices used to reorder the expert output.

        Returns: 
            (torch.Tensor, torch.Tensor): Unpermuted activation and optional bias.            
        """
        raise NotImplementedError("Restore function not implemented.")


class MoEDroplessTokenDispatcher(MoETokenDispatcher):
    """
    Token dispatcher without token dropping.
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        k: int,
        config: TransformerConfig,
    ) -> None:
        """
        Initialize the zero token dropping router.
        """
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        self.local_expert_indices = local_expert_indices
        self.k = k
        self.add_bias = config.add_bias_linear

    def gather_indices(self, local_indices: torch.Tensor):
        """ Gather tensors and concatenate along the first dimension."""
        group = get_tensor_and_expert_parallel_group()
        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return local_indices

        dim_size = list(local_indices.size())
        dim_size[0] = dim_size[0] * world_size

        # TODO pre allocate memory
        output = torch.empty(
            dim_size, dtype=local_indices.dtype, device=torch.cuda.current_device()
        )
        torch.distributed._all_gather_base(output, local_indices.contiguous(), group=group)
        return output

    def dispatch(self, hidden_states: torch.Tensor, max_prob: torch.Tensor, max_ind: torch.Tensor):
        """Dispatch tokens to local experts. It's composed of two stages:
        (1) Permute the tokens across the expert parallel devices. After this stage,
        each device receives all of the tokens assigned to its local set of experts
        in its local HBM.
        (2) Permute the tokens locally so that they are grouped by their expert
        assignment. After the stage (1), the tokens are grouped by which device
        they came from. We re-order them locally for subsequent efficient computation.

        Args:
            hidden_states: input tokens of shape [SeqLen/TP, MBS, HiddenSize]

        Returns:
            permuted_local_hidden_states: Permutation of tokens to local experts group.
            tokens_per_expert: the number of tokens each local expert to process.
            indices: The indices of `local_indices` (which holds the un-sorted expert
            indices of tokens that local expert can process) that give its sorted order along dim 0.
            global_local_map (optional): 2D tensor. A mask of mapping between global and local tokens where each
            element is True if it's between the local_expert_indices. Only useful
            when cross device token permutation is enabled and **AllGahter** is performed.
        """
        self.hidden_shape = hidden_states.shape
        # [S/TP, B, H] -> [S*B/TP, H]
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # Permute the tokens across the expert parallel devices.
        if self.config.sequence_parallel or (self.config.expert_model_parallel_size > 1):
            # [S*B/TP, H] -> [S*B, H]
            global_hidden_states = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
                hidden_states
            )
            with torch.no_grad():
                global_indices = self.gather_indices(max_ind)
                # Create a mask of mapping between global and local tokens where each
                # element is True if it's between the local_expert_indices
                global_local_map = (global_indices >= self.local_expert_indices[0]) & (
                    global_indices <= self.local_expert_indices[-1]
                )
                local_indices = global_indices.masked_select(global_local_map)
                if self.k > 1:  # k > 1
                    global_probs = self.gather_indices(max_prob)
                    local_probs = global_probs.masked_select(global_local_map)
                else:
                    local_probs = max_prob
                # Reshape global_local_map to be compatible with Tensor.gather
                global_local_map = global_local_map.nonzero()[:, 0]
                global_local_map = global_local_map.view(-1, 1).expand(-1, hidden_states.shape[-1])
            local_hidden_states = torch.gather(global_hidden_states, 0, global_local_map)
        else:
            if self.k > 1:
                global_local_map = torch.ones_like(max_ind).bool()
                local_indices = max_ind.masked_select(global_local_map)
                local_probs = max_prob.masked_select(global_local_map)
                global_local_map = global_local_map.nonzero()[:, 0]
                global_local_map = global_local_map.view(-1, 1).expand(-1, hidden_states.shape[-1])
                local_hidden_states = torch.gather(hidden_states, 0, global_local_map)
            else:
                local_indices = max_ind
                local_probs = max_prob
                local_hidden_states = hidden_states
                global_local_map = None

        with torch.no_grad():
            # The indices of local_indices that give its sorted order along dim 0.
            indices = torch.argsort(local_indices, dim=0)
            tokens_per_expert = torch.histc(
                local_indices,
                bins=self.num_local_experts,
                min=self.local_expert_indices[0],
                max=self.local_expert_indices[-1],
            )
            tokens_per_expert = tokens_per_expert.cpu().to(torch.long)

        # Stage2: permute the tokens locally so that they are grouped by their expert assignment
        # Reshape indices to be compatible with Tensor.gather
        indices = indices.view(-1, 1).expand(-1, hidden_states.shape[-1])
        permuted_local_hidden_states = torch.gather(local_hidden_states, 0, indices)
        return (
            permuted_local_hidden_states,
            tokens_per_expert,
            local_probs,
            indices,
            global_local_map,
        )

    def restore(
        self,
        hidden_states: torch.Tensor,
        scores: torch.Tensor,
        indices: torch.Tensor,
        global_local_map: torch.Tensor = None,
        bias: torch.Tensor = None,
    ):
        """
        Reverse process of `dispatch()` which permutes the ouput of local
        experts locallay and across expert parallel rank into the original order to
        produce the final output.

        Args:
            hidden_states: 2D tensor of shape [sum_tokens_of_all_local_experts, HiddenSize],
            ouput of local experts.
            indices: 2D tensor of the indices of `local_indices` (which holds the un-sorted expert
            indices of tokens that local expert can process) that give its sorted order along dim 0.
            global_local_map (optional): 2D tensor, a mask of mapping between global and local tokens where each
            element is True if it's between the local_expert_indices. Only useful
            when cross device token permutation is enabled and **AllGahter** is performed.

        Returns:
            output_total: un-permuted updated hidden states output from all local experts
            with shape of [SeqLen/TP, MBS, HiddenSize]
        """
        # Stage1: unpermute the tokens and bias locally respectively.
        scores = scores.to(dtype=hidden_states.dtype)
        unpermuted_local_hidden = torch.zeros_like(hidden_states)
        assert indices.shape == hidden_states.shape
        unpermuted_local_hidden = unpermuted_local_hidden.scatter(0, indices, hidden_states)

        # Scale the expert output prior to reduction and subsequent to local unpermutation if k > 1.
        if self.k > 1:
            unpermuted_local_hidden = unpermuted_local_hidden * scores.view(-1, 1)

        unpermuted_local_bias = None
        if self.add_bias:
            assert bias is not None
            unpermuted_local_bias = torch.zeros_like(hidden_states)
            assert indices.shape == bias.shape
            unpermuted_local_bias = unpermuted_local_bias.scatter(0, indices, bias)
            if self.k > 1:
                unpermuted_local_bias = unpermuted_local_bias * scores.view(-1, 1)

        output_total = unpermuted_local_hidden
        output_bias_total = unpermuted_local_bias

        # Unpermute the tokens across expert parallel devices.
        if self.config.sequence_parallel or (self.config.expert_model_parallel_size > 1):
            assert global_local_map is not None, "global_local_map is necessary for `AllGather`."
            ep_group_size = parallel_state.get_tensor_and_expert_parallel_world_size()
            # hidden_shape: [SeqLen/TP, MBS, HiddenSize], glboal_num_tokens = SeqLen/TP*MBS*(TP*EP)
            global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1] * ep_group_size
            global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
            unpermuted_global_hidden = torch.zeros(
                global_hidden_shape, dtype=hidden_states.dtype, device=torch.cuda.current_device()
            )
            # Reshape global_local_map to be compatible with Tensor.scatter
            assert global_local_map.shape == unpermuted_local_hidden.shape
            unpermuted_global_hidden = unpermuted_global_hidden.scatter_add(
                0, global_local_map, unpermuted_local_hidden
            )
            output_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
                unpermuted_global_hidden
            )
            if self.add_bias:
                # Unpermute the bias across expert parallel devices.
                unpermuted_global_bias = torch.zeros_like(unpermuted_global_hidden)
                unpermuted_global_bias = unpermuted_global_bias.scatter_add(
                    0, global_local_map, unpermuted_local_bias
                )
                output_bias_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
                    unpermuted_global_bias
                )
                # bias is duplicated across tensor parallelism ranks;
                # reduce scatter reduces bias across tensor parallel_ranks
                output_bias_total = (
                    output_bias_total / parallel_state.get_tensor_model_parallel_world_size()
                )
        else:
            if self.k > 1:
                global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1]
                global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
                unpermuted_global_hidden = torch.zeros(
                    global_hidden_shape,
                    dtype=hidden_states.dtype,
                    device=torch.cuda.current_device(),
                )
                output_total = unpermuted_global_hidden.scatter_add(
                    0, global_local_map, unpermuted_local_hidden
                )
                if self.add_bias:
                    unpermuted_global_bias = torch.zeros_like(unpermuted_global_hidden)
                    output_bias_total = unpermuted_global_bias.scatter_add(
                        0, global_local_map, unpermuted_local_bias
                    )

        if self.k == 1:
            output_total = output_total * scores
        output_total = output_total.view(self.hidden_shape)
        if self.add_bias:
            assert output_bias_total is not None
            if self.k == 1:
                output_bias_total = output_bias_total * scores
            output_bias_total = output_bias_total.view(self.hidden_shape)
        else:
            output_bias_total = None

        return output_total, output_bias_total


class TopKRouter(Router):
    """Route each token to the top-k experts."""

    def __init__(
        self, num_local_experts: int, local_expert_indices: List[int], config: TransformerConfig,
    ) -> None:
        """Initialize the zero token dropping router.

        Args:
            num_local_experts (int): The number of local experts.
            local_expert_indices (List[int]): The indices of the local experts.
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__(config=config)
        assert config.moe_token_dropping == False
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.token_dispatcher = MoEDroplessTokenDispatcher(
            num_local_experts, local_expert_indices, self.topk, config
        )
        self.moe_aux_loss_func = switch_load_balancing_loss_func

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            torch.Tensor: The logits tensor after applying sinkhorn routing.
        """
        assert self.config.moe_aux_loss_coeff == 0, "Sinkhorn routing does not support aux loss."
        router_activation = torch.sigmoid
        if self.training:
            with torch.no_grad():
                norm_logits = sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.topk, dim=1)
            logits = router_activation(logits)
            scores = torch.gather(logits, 1, indices)
        else:
            logits = router_activation(logits)
            scores, indices = torch.topk(logits, k=self.topk, dim=1)
        return scores, indices

    def aux_loss_load_balancing(self, logits: torch.Tensor):
        """Apply loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The scores and the indices tensor after applying load balancing.
        """
        top_logits, indices = torch.topk(logits, k=self.topk, dim=1)
        scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32).type_as(logits)
        # Apply load balancing loss
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        scores = self.apply_aux_loss(self.moe_aux_loss_func, probs, indices, activation=scores)
        return scores, indices

    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Probs and the indices tensor.
        """
        logits = logits.view(-1, self.config.num_moe_experts)
        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        if self.routing_type == "sinkhorn":
            scores, indices = self.sinkhorn_load_balancing(logits)
        elif self.routing_type == "aux_loss":
            scores, indices = self.aux_loss_load_balancing(logits)
        elif self.routing_type is None:
            # A naive top-k routing without load balancing
            top_logits, indices = torch.topk(logits, k=self.k, dim=1)
            scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32).type_as(logits)

        return scores, indices


class MoEAuxLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that compute and scales the grad for auxiliary loss.

    """

    main_loss_backward_scale: int = 1

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
    def set_loss_scale(scale: int):
        """set the scale of the aux loss.
        
        Args:
            scale (int): The scale value to set. Please ensure that the scale passed in matches the scale of the main_loss.
        """
        MoEAuxLossAutoScaler.main_loss_backward_scale = scale
