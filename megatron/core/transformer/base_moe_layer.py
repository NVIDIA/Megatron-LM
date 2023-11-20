# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

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


class BaseMoELayer(MegatronModule):
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
        self.hidden_shape = hidden_states.shape
        route = self.router(hidden_states)
        # print(self.router.weight)
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
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        if self.sequence_parallel or (self.expert_parallel_size > 1):
            global_hidden_states = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
                hidden_states
            )
            global_indices = self.gather_indices(max_ind)
        else:
            global_hidden_states = hidden_states
            global_indices = max_ind

        return global_hidden_states, global_indices

    def token_unpermutation(self, output_total, output_bias_total=None):
        if self.sequence_parallel or (self.expert_parallel_size > 1):
            output_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
                output_total
            )
            if self.add_bias:
                assert output_bias_total is not None
                output_bias_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
                    output_bias_total
                )
                # bias is duplicated across tensor parallelism ranks;
                # reduce scatter reduces bias across tensor parallel_ranks
                output_bias_total = (
                    output_bias_total / parallel_state.get_tensor_model_parallel_world_size()
                )

        output_total = output_total * self.max_prob
        output_total = output_total.view(self.hidden_shape)
        if self.add_bias:
            output_bias_total = output_bias_total * self.max_prob
            output_bias_total = output_bias_total.view(self.hidden_shape)
        else:
            output_bias_total = None

        return output_total, output_bias_total
