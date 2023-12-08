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
    router = torch.nn.Linear(config.hidden_size, config.num_moe_experts, bias=False)
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
        """
        self.hidden_shape = hidden_states.shape
        route = self.router(hidden_states)
        route = route.view(-1, self.config.num_moe_experts)

        if self.training:
            with torch.no_grad():
                norm_route = self.route_algo(
                    route.detach().to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, max_ind = torch.max(norm_route, dim=1)
            route = self.router_activation(route)
            max_prob = route[torch.arange(route.size(0)), max_ind]
        else:
            route = self.router_activation(route)
            max_prob, max_ind = torch.max(route, dim=1)

        self.max_prob = torch.unsqueeze(max_prob, 1)
        # [S/TP, B, H] -> [S*B/TP, H]
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # Permute the tokens across the expert parallel devices.
        if self.sequence_parallel or (self.expert_parallel_size > 1):
            # [S*B/TP, H] -> [S*B, H]
            global_hidden_states = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
                hidden_states
            )
            global_indices = self.gather_indices(max_ind)
            self.ghs_shape = global_hidden_states.shape
            # Create a mask where each element is True if it's between the local_expert_indices
            self.mask = (global_indices >= self.local_expert_indices[0]) & (
                global_indices <= self.local_expert_indices[-1]
            )
            self.local_indices = global_indices[self.mask]
            local_hidden_states = global_hidden_states[self.mask, :]
        else:
            self.ghs_shape = hidden_states.shape
            self.local_indices = max_ind
            local_hidden_states = hidden_states

        # Permute the tokens locally so that they are grouped by their expert assignment
        with torch.no_grad():
            self.permuted_indices = torch.argsort(self.local_indices)
            # Permutation of tokens to each expert group.
            permuted_local_hidden_states = local_hidden_states[self.permuted_indices]
            tokens_per_expert = torch.histc(
                self.local_indices,
                bins=self.num_local_experts,
                min=self.local_expert_indices[0],
                max=self.local_expert_indices[-1],
            )
            tokens_per_expert = tokens_per_expert.cpu().to(torch.long)

        return permuted_local_hidden_states, tokens_per_expert

    def token_unpermutation(self, hidden_states, bias=None):
        """Reverse process of 'token_permutation' which permutes the ouput of local
        experts into the original order to produce the final output.

        Args:
            hidden_states: 2D tensor of shape [sum_tokens_of_all_local_experts, HiddenSize],
            ouput of local experts.
            bias: bias if self.add_bias is enabled.

        Returns:
            output_total: un-permuted updated hidden states output from all local experts
            with shape of [SeqLen/TP, MBS, HiddenSize]
        """
        # Unpermute the tokens locally.
        original_order_lhs = torch.zeros_like(hidden_states)
        original_order_lhs[self.permuted_indices] = hidden_states
        output_total = original_order_lhs
        output_bias_total = bias

        # Unpermute the tokens across expert parallel devices.
        if self.sequence_parallel or (self.expert_parallel_size > 1):
            original_order_ghs = torch.zeros(
                self.ghs_shape, dtype=hidden_states.dtype, device=torch.cuda.current_device()
            )
            global_local_map = torch.squeeze(self.mask.nonzero().contiguous())
            original_order_ghs[global_local_map] = original_order_lhs
            output_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
                original_order_ghs
            )
            if self.add_bias:
                assert bias is not None
                original_order_bias = torch.zeros_like(original_order_ghs)
                original_order_bias[global_local_map] = bias
                output_bias_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
                    original_order_bias
                )
                # bias is duplicated across tensor parallelism ranks;
                # reduce scatter reduces bias across tensor parallel_ranks
                output_bias_total = (
                    output_bias_total / parallel_state.get_tensor_model_parallel_world_size()
                )

        output_total = output_total * self.max_prob
        output_total = output_total.view(self.hidden_shape)
        if self.add_bias:
            assert output_bias_total is not None
            output_bias_total = output_bias_total * self.max_prob
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
