# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.parallel_state import get_tensor_and_expert_parallel_group
from megatron.core.tensor_parallel import get_cuda_rng_tracker, get_data_parallel_rng_tracker_name
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


def sinkhorn(cost, tol=0.0001):
    "Sinkhorn based MoE routing function"
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


def get_router_linear_layer(config):
    router = torch.nn.Linear(config.hidden_size, config.num_moe_experts, bias=False, dtype=config.params_dtype)
    with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
        config.init_method(router.weight)
    setattr(router.weight, 'sequence_parallel', config.sequence_parallel)
    return router


class BaseMoELayer(ABC, MegatronModule):
    """
    Basic MoE layer.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        self.router = get_router_linear_layer(self.config)
        self.add_bias = config.add_bias_linear
        self.sequence_parallel = config.sequence_parallel
        self.route_algo = sinkhorn
        self.router_activation = torch.sigmoid
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()

        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )
        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        self.k = 1  # TODO: self.config.top_k

    def gather_indices(self, local_indices):
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

    def token_permutation(self, hidden_states):
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
        route = self.router(hidden_states)
        route = route.view(-1, self.config.num_moe_experts)

        if self.training:
            with torch.no_grad():
                norm_route = self.route_algo(
                    route.detach().to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, max_ind = torch.topk(norm_route, k=self.k, dim=1)
            route = self.router_activation(route)
            # max_ind = max_ind.view(-1)
            max_prob = torch.gather(route, 1, max_ind)
        else:
            route = self.router_activation(route)
            max_prob, max_ind = torch.topk(route, k=self.k, dim=1)
        # [S/TP, B, H] -> [S*B/TP, H]
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # Stage1: permute the tokens across the expert parallel devices.
        if self.sequence_parallel or (self.expert_parallel_size > 1):
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
            local_indices = max_ind
            local_probs = max_prob
            local_hidden_states = hidden_states
            global_local_map = None
        self.max_prob = local_probs

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

        return permuted_local_hidden_states, tokens_per_expert, indices, global_local_map

    def token_unpermutation(self, hidden_states, indices, global_local_map=None, bias=None):
        """Reverse process of `token_permutation()` which permutes the ouput of local
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
            bias: bias if self.add_bias is enabled.

        Returns:
            output_total: un-permuted updated hidden states output from all local experts
            with shape of [SeqLen/TP, MBS, HiddenSize]
            output_bias_total: un-permuted bias output from all local experts if
            self.add_bias is enabled.
        """
        # Stage1: unpermute the tokens and bias locally respectively.
        unpermuted_local_hidden = torch.zeros_like(hidden_states)
        assert indices.shape == hidden_states.shape
        unpermuted_local_hidden = unpermuted_local_hidden.scatter(0, indices, hidden_states)

        # Scale the expert output prior to reduction and subsequent to local unpermutation if k > 1.
        if self.k > 1:
            unpermuted_local_hidden = unpermuted_local_hidden * self.max_prob.view(-1, 1)

        unpermuted_local_bias = None
        if self.add_bias:
            assert bias is not None
            unpermuted_local_bias = torch.zeros_like(hidden_states)
            assert indices.shape == bias.shape
            unpermuted_local_bias = unpermuted_local_bias.scatter(0, indices, bias)
            if self.k > 1:
                unpermuted_local_bias = unpermuted_local_bias * self.max_prob.view(-1, 1)

        output_total = unpermuted_local_hidden
        output_bias_total = unpermuted_local_bias

        # Stage2: unpermute the tokens across expert parallel devices.
        if self.sequence_parallel or (self.expert_parallel_size > 1):
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
        if self.k == 1:
            output_total = output_total * self.max_prob.view(-1, 1)
        output_total = output_total.view(self.hidden_shape)
        if self.add_bias:
            assert output_bias_total is not None
            if self.k == 1:
                output_bias_total = output_bias_total * self.max_prob.view(-1, 1)
            output_bias_total = output_bias_total.view(self.hidden_shape)
        else:
            output_bias_total = None

        return output_total, output_bias_total

    @abstractmethod
    def forward(self, hidden_states):
        """Forward computation of MoE layer.

        Args:
            hidden_states: input activation of shape [SeqLen, MBS, HiddenSize]

        """
        pass
