# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch.nn.functional as F

from megatron.core import tensor_parallel
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class ParallelMLP(MegatronModule):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.

    We use the following notation: 
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(self, config: TransformerConfig):
        super(ParallelMLP, self).__init__(config=config)

        self.config = config
        self.hidden_size = config.hidden_size
        self.ffn_hidden_size = config.ffn_hidden_size
        self.init_method = config.init_method
        self.output_layer_init_method = config.output_layer_init_method
        self.use_cpu_initialization = config.use_cpu_initialization
        self.perform_initialization = config.perform_initialization
        self.bias_gelu_fusion = config.bias_gelu_fusion
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
        self.sequence_parallel_enabled = config.sequence_parallel_enabled
        self.params_dtype = config.params_dtype
        self.async_tensor_model_parallel_allreduce = config.async_tensor_model_parallel_allreduce

        # Project to 4h.
        # @jcasper should we change the name dense_h_to_4h here?
        self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
            self.hidden_size,
            self.ffn_hidden_size,
            gather_output=False,
            init_method=self.init_method,
            skip_bias_add=True,
            async_tensor_model_parallel_allreduce=self.async_tensor_model_parallel_allreduce,
            params_dtype=self.params_dtype,
            use_cpu_initialization=self.use_cpu_initialization,
            perform_initialization=self.perform_initialization,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
        )

        self.activation_func = F.gelu

        # @jcasper should we remove openai_gelu?
        # if args.openai_gelu:
        #     self.activation_func = openai_gelu
        # @jcasper should we remove onnx_safe?
        # elif args.onnx_safe:
        #     self.activation_func = erf_gelu

        # Project back to h.
        # @jcasper should we change the name here?
        self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
            self.ffn_hidden_size,
            self.hidden_size,
            input_is_parallel=True,
            init_method=self.output_layer_init_method,
            skip_bias_add=True,
            params_dtype=self.params_dtype,
            use_cpu_initialization=self.use_cpu_initialization,
            perform_initialization=self.perform_initialization,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
        )

    def forward(self, hidden_states):

        # [s, b, 4 * h/p]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.bias_gelu_fusion:
            intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            intermediate_parallel = self.activation_func(intermediate_parallel + bias_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias
