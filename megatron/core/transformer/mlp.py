# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch
import torch.nn.functional as F

from megatron.core import parallel_state, tensor_parallel
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.transformer.custom_layers.transformer_engine import (
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.parallel_state import get_tensor_and_data_parallel_group


class MLP(MegatronModule):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.
    If config.add_bias_linear is False, the bias returned is None.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(self, config: TransformerConfig, is_expert: bool = False):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        # If this is a gated linear unit we double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        ffn_hidden_size = self.config.ffn_hidden_size
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        # TODO: revert this to TE; need to think of configurability
        self.linear_fc1 = tensor_parallel.ColumnParallelLinear(
            self.config.hidden_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert
        )

        if self.config.gated_linear_unit:

            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return self.config.activation_func(x[0]) * x[1]

            self.activation_func = glu
        else:
            self.activation_func = self.config.activation_func

        self.linear_fc2 = tensor_parallel.RowParallelLinear(
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert
        )

    def forward(self, hidden_states):

        # [s, b, 4 * h/p]
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)

        if self.config.bias_gelu_fusion:
            assert self.config.add_bias_linear is True
            assert self.activation_func == F.gelu
            intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)

        # [s, b, h]
        output, output_bias = self.linear_fc2(intermediate_parallel)
        return output, output_bias


class SwitchMLP(MegatronModule):
    """
    Top-1 Mixture of Experts Layer. Routes input to one of N MLP "experts"
    Curently supports Sinkhorn based expert routing.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        self.router = TERowParallelLinear(
            self.config.hidden_size,
            self.config.num_moe_experts,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
        )
        self.add_bias = config.add_bias_linear
        self.expert_parallel = config.expert_parallel
        self.sequence_parallel = config.sequence_parallel
        self.route_algo = SwitchMLP.sinkhorn

        if self.expert_parallel:
            assert self.config.num_moe_experts % parallel_state.get_data_parallel_world_size() == 0
            self.num_local_experts = self.config.num_moe_experts // parallel_state.get_data_parallel_world_size()
            local_expert_indices_offset = parallel_state.get_data_parallel_rank() * self.num_local_experts
            self.local_expert_indices = [local_expert_indices_offset + i for i in range(self.num_local_experts)]
        else:
            self.num_local_experts = self.config.num_moe_experts
            self.local_expert_indices = [i for i in range(self.num_local_experts)]

        self.local_experts = torch.nn.ModuleList()
        for _ in range(self.num_local_experts):
            expert = MLP(self.config, is_expert=True)
            self.local_experts.append(expert)
    
    def gather_indices(self, local_indices):
        """ Gather tensors and concatinate along the first dimension."""
        group = get_tensor_and_data_parallel_group()
        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return local_indices

        dim_size = list(local_indices.size())
        dim_size[0] = dim_size[0] * world_size

        # TODO pre allocate memory
        output = torch.empty(dim_size, dtype=local_indices.dtype,
                             device=torch.cuda.current_device())
        torch.distributed._all_gather_base(
            output, local_indices.contiguous(), group=group)
        return output
    
    @classmethod
    def sinkhorn(cls, cost, tol=0.0001):
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

    def forward(self, hidden_states):
        hidden_shape = hidden_states.shape
        route, _ = self.router(hidden_states)
        route = route.view(-1, self.config.num_moe_experts)

        if self.training:
            with torch.no_grad():
                norm_route = self.route_algo(
                    route.detach().to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, max_ind = torch.max(norm_route, dim=1)
            route = torch.sigmoid(route)
            max_prob = route[torch.arange(route.size(0)), max_ind]
        else:
            route = torch.sigmoid(route)
            max_prob, max_ind = torch.max(route, dim=1)
        
        max_prob = torch.unsqueeze(max_prob, 1)
        hidden_states = hidden_states.view(-1, hidden_shape[-1])

        if self.sequence_parallel or self.expert_parallel:
            global_hidden_states = \
                tensor_parallel.gather_from_sequence_parallel_region_to_moe(hidden_states)
            global_indices = self.gather_indices(max_ind)
        else:
            global_hidden_states = hidden_states
            global_indices = max_ind
        
        output_total = torch.zeros_like(global_hidden_states)
        if self.add_bias:
            output_bias_total = torch.zeros_like(global_hidden_states)

        for expert_num, expert in enumerate(self.local_experts):
            local_expert_index = self.local_expert_indices[expert_num]
            local_indices = (global_indices == local_expert_index).nonzero()
            hidden = global_hidden_states[local_indices, :]
            output, output_bias = expert(hidden)

            output_total[local_indices, :] = output
            if self.add_bias:
                output_bias = output_bias.expand_as(output)
                output_bias_total[local_indices, :] = output_bias

        if self.sequence_parallel or self.expert_parallel:
            output_total = \
                tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(output_total)
            if self.add_bias:
                output_bias_total = \
                    tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(output_bias_total)
                # bias is duplicated across tensor parallelism ranks;
                # reduce scatter reduces bias across tensor parallel_ranks
                output_bias_total = \
                    output_bias_total/parallel_state.get_tensor_model_parallel_world_size()

        output_total = output_total*max_prob
        output_total = output_total.view(hidden_shape)
        if self.add_bias:
            output_bias_total = output_bias_total*max_prob
            output_bias_total = output_bias_total.view(hidden_shape)
        else:
            output_bias_total = None

        return output_total, output_bias_total
