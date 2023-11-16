# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import numpy as np
import torch
from torch.nn.parameter import Parameter

from megatron.core import parallel_state

from megatron.core.tensor_parallel.layers import _initialize_affine_weight_gpu
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer import grouped_gemm_util as gg
from megatron.core.transformer.transformer_config import TransformerConfig

from .base_moe_layer import BaseMoELayer
from .mlp import MLPSubmodules

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

class GroupedMLP(BaseMoELayer):
    """
    Top-1 Mixture of Experts Layer with Grouped GEMM. Routes input to one of N MLP "experts"
    Curently supports Sinkhorn based expert routing.
    """

    def __init__(self, config: TransformerConfig, submodules: MLPSubmodules):
        super().__init__(config=config)
        self.config: TransformerConfig = config

        gg.assert_grouped_gemm_is_available()
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
        assert config.add_bias_linear == False, \
            "bias in the expert layer is not supported in Grouped GEMM yet."
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
                self.config.hidden_size,
                output_size_per_partition,
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
                partition_dim=1,
                expert_parallel=self.expert_parallel,
            )
        setattr(self.weight1, 'allreduce', not self.expert_parallel)
        setattr(self.weight2, 'allreduce', not self.expert_parallel)

    def scale_grad(self, w):
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)

    def forward(self, hidden_states):
        global_hidden_states, global_indices = self.token_permutation(hidden_states)

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
        w2 = w2.view(self.num_local_experts, self.config.hidden_size, -1)

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
            trans_b=True)
        # Un-permutation of tokens
        output_total = fc2_output[reverse_indices]

        output_total, _ = self.token_unpermutation(output_total)
        return output_total, None