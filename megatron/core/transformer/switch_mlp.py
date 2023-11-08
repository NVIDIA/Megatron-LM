# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import numpy as np
import torch
from torch.nn.parameter import Parameter

from megatron.core import parallel_state, tensor_parallel
from megatron.core.parallel_state import (
    get_tensor_and_expert_parallel_group,
    get_tensor_model_parallel_group,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker, get_data_parallel_rng_tracker_name
from megatron.core.transformer import grouped_gemm_util as gg
from megatron.core.tensor_parallel.layers import _initialize_affine_weight_gpu
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer import grouped_gemm_util as gg
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

from .mlp import MLP, MLPSubmodules


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

class ScaleGradient(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        return grad * ctx.scale, None
scale_gradient = ScaleGradient.apply

def get_router_linear_layer(config):
    router = torch.nn.Linear(config.hidden_size, config.num_moe_experts, bias=False)
    with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
        config.init_method(router.weight)
    setattr(router.weight, 'sequence_parallel', config.sequence_parallel)
    return router


class SwitchMLP(MegatronModule):
    """
    Top-1 Mixture of Experts Layer. Routes input to one of N MLP "experts"
    Curently supports Sinkhorn based expert routing.
    """

    def __init__(self, config: TransformerConfig, submodules: MLPSubmodules):
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

        if not self.config.moe_grouped_gemm:
            self.local_experts = torch.nn.ModuleList()
            for _ in range(self.num_local_experts):
                expert = MLP(self.config, submodules, is_expert=True)
                self.local_experts.append(expert)
        else:
            self.expert_parallel = config.expert_model_parallel_size > 1
            self.gradient_scale = 1 / parallel_state.get_tensor_and_expert_parallel_world_size()
            if self.config.gated_linear_unit:
                def glu(x):
                    x = torch.chunk(x, 2, dim=-1)
                    return self.config.activation_func(x[0]) * x[1]

                self.activation_func = glu
            else:
                self.activation_func = self.config.activation_func

            assert not config.use_cpu_initialization
            # How many feature each rank holds
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            ffn_hs_per_expert_per_partition = divide(self.config.ffn_hidden_size, tp_size)
            output_size_per_partition = self.num_local_experts * ffn_hs_per_expert_per_partition
            fc1_output_size_per_partition = output_size_per_partition
            if config.gated_linear_unit:
                fc1_output_size_per_partition *= 2

            self.weight1 = Parameter(
                torch.empty(
                    fc1_output_size_per_partition,
                    self.config.hidden_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    output_size_per_partition,
                    self.config.hidden_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight1,
                    config.init_method,
                    partition_dim=0,
                    expert_parallel=self.expert_parallel,
                )
                _initialize_affine_weight_gpu(
                    self.weight2,
                    config.output_layer_init_method,
                    partition_dim=0,
                    expert_parallel=self.expert_parallel,
                )
            setattr(self.weight1, 'allreduce', not self.expert_parallel)
            setattr(self.weight2, 'allreduce', not self.expert_parallel)

    def scale_grad(self, w):
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)

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

    def forward(self, hidden_states):
        hidden_shape = hidden_states.shape
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

        max_prob = torch.unsqueeze(max_prob, 1)
        hidden_states = hidden_states.view(-1, hidden_shape[-1])

        if self.sequence_parallel or (self.expert_parallel_size > 1):
            global_hidden_states = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
                hidden_states
            )
            global_indices = self.gather_indices(max_ind)
        else:
            global_hidden_states = hidden_states
            global_indices = max_ind

        if not self.config.moe_grouped_gemm:
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
        else:
            with torch.no_grad():
                sorted, indices = torch.sort(global_indices, stable=True)
                # Permutation of tokens
                sorted_global_hidden_states = global_hidden_states[indices]
                # Histogram the expert ids to identify the number of tokens routed to each expert
                # Note that for np.histogram, all but the last (righthand-most) bin is half-open.
                tokens_per_expert, bin_edges = np.histogram(
                    sorted.cpu(),
                    bins=np.arange(self.config.num_moe_experts + 1))
                tokens_per_expert = torch.tensor(tokens_per_expert).to(torch.long)
                reverse_indices = indices.argsort()

            w1, w2 = (self.scale_grad(self.weight1), self.scale_grad(self.weight2))
            # Reshape the weights for the grouped GEMMs.
            w1 = w1.view(self.num_local_experts, -1, self.config.hidden_size)
            w2 = w2.view(self.num_local_experts, -1, self.config.hidden_size)

            fc1_output = gg.ops.gmm(
                sorted_global_hidden_states,
                w1,
                tokens_per_expert,
                trans_b=True)

            intermediate_parallel = self.activation_func(fc1_output)

            fc2_output = gg.ops.gmm(
                intermediate_parallel,
                w2,
                tokens_per_expert,
                trans_b=False)
            # Un-permutation of tokens
            output_total = fc2_output[reverse_indices]

        if self.sequence_parallel or (self.expert_parallel_size > 1):
            output_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
                output_total
            )
            if self.add_bias:
                output_bias_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
                    output_bias_total
                )
                # bias is duplicated across tensor parallelism ranks;
                # reduce scatter reduces bias across tensor parallel_ranks
                output_bias_total = (
                    output_bias_total / parallel_state.get_tensor_model_parallel_world_size()
                )

        output_total = output_total * max_prob
        output_total = output_total.view(hidden_shape)
        if self.add_bias:
            output_bias_total = output_bias_total * max_prob
            output_bias_total = output_bias_total.view(hidden_shape)
        else:
            output_bias_total = None

        return output_total, output_bias_total
