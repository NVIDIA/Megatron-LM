# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch
from torch.nn.parameter import Parameter

from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer import grouped_gemm_util as gg
from megatron.core.transformer.transformer_config import TransformerConfig

from .base_moe_layer import BaseMoELayer


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

    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)
        self.config: TransformerConfig = config

        gg.assert_grouped_gemm_is_available()
        assert (
            config.add_bias_linear == False
        ), "bias in the expert layer is not supported in Grouped GEMM yet."

        self.expert_parallel = config.expert_model_parallel_size > 1
        self.gradient_scale = 1 / parallel_state.get_tensor_and_expert_parallel_world_size()
        if self.config.gated_linear_unit:

            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return self.config.activation_func(x[0]) * x[1]

            self.activation_func = glu
        else:
            self.activation_func = self.config.activation_func

        # How many feature each rank holds for fc1 and fc2, respectively.
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        fc1_output_size = self.config.ffn_hidden_size * self.num_local_experts
        if config.gated_linear_unit:
            # Project to 4h. If using swiglu double the output width,
            # see https://arxiv.org/pdf/2002.05202.pdf
            fc1_output_size *= 2
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        fc2_input_size = self.config.ffn_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp_size)

        # Initialize weight.
        if config.use_cpu_initialization:
            self.weight1 = Parameter(
                torch.empty(
                    fc1_output_size_per_partition,
                    self.config.hidden_size,
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc2_input_size_per_partition,
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight1,
                    fc1_output_size,
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    partition_dim=0,
                    init_method=config.init_method,
                    params_dtype=config.params_dtype,
                )
                _initialize_affine_weight_cpu(
                    self.weight2,
                    self.config.hidden_size,
                    fc2_input_size,
                    fc2_input_size_per_partition,
                    partition_dim=1,
                    init_method=config.output_layer_init_method,
                    params_dtype=config.params_dtype,
                )
        else:
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
                    fc2_input_size_per_partition,
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
            sorted_indices = torch.argsort(global_indices)
            # Permutation of tokens to each expert group.
            sorted_global_hidden_states = global_hidden_states[sorted_indices]
            # GroupedGEMM requires tokens_per_expert is on cpu.
            tokens_per_expert = torch.histc(
                global_indices,
                bins=self.config.num_moe_experts,
                min=0,
                max=self.config.num_moe_experts-1).cpu()

        w1, w2 = (self.scale_grad(self.weight1), self.scale_grad(self.weight2))
        # Reshape the weights for the grouped GEMMs.
        w1 = w1.view(self.num_local_experts, -1, self.config.hidden_size)
        w2 = w2.view(self.num_local_experts, self.config.hidden_size, -1)

        fc1_output = gg.ops.gmm(sorted_global_hidden_states, w1, tokens_per_expert, trans_b=True)

        intermediate_parallel = self.activation_func(fc1_output)

        fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=True)
        # Un-permutation of tokens
        original_order_ghs = torch.empty_like(fc2_output)
        original_order_ghs[sorted_indices] = fc2_output
        output_total, _ = self.token_unpermutation(original_order_ghs)

        return output_total, None
