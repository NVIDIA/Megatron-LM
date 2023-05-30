# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch.nn.functional as F

from megatron.core import tensor_parallel
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.custom_layers.transformer_engine import \
        TERowParallelLinear, TEColumnParallelLinear

class MLP(MegatronModule):
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
        super().__init__(config=config)

        self.config: TransformerConfig = config

        self.linear_fc1 = TEColumnParallelLinear(
            self.config.hidden_size,
            self.config.ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=True,
            return_bias=True,
        )

        self.activation_func = F.gelu

        # @jcasper should we remove openai_gelu?
        # if args.openai_gelu:
        #     self.activation_func = openai_gelu
        # @jcasper should we remove onnx_safe?
        # elif args.onnx_safe:
        #     self.activation_func = erf_gelu

        self.linear_fc2 = TERowParallelLinear(
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=True,
            return_bias=True,
        )

    def forward(self, hidden_states):

        # [s, b, 4 * h/p]
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)

        if self.config.bias_gelu_fusion:
            intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            intermediate_parallel = self.activation_func(intermediate_parallel + bias_parallel)

        # [s, b, h]
        output, output_bias = self.linear_fc2(intermediate_parallel)
        return output, output_bias
