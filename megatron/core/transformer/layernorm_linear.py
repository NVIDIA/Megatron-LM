# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch.nn.functional as F

from megatron.core import tensor_parallel
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class LayernormLinear(MegatronModule):
    """
    LayernormLinear is just a composite module composed of `Layernorm` and
    `Linear` layers
    """

    def __init__(self, input_size: int, output_size: int, config: TransformerConfig, **kwargs):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        self.layernorm = FusedLayerNorm(
            hidden_size=self.config.hidden_size, eps=self.config.layernorm_epsilon
        )

        self.linear = ColumnParallelLinear(
            input_size,
            output_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
        )

    def forward(self, hidden_states):
        hidden_states = self.layernorm(hidden_states)
        output, output_bias = self.linear(hidden_states)
        return output, output_bias
