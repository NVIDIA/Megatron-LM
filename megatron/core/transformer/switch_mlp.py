# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import numpy as np
import torch

from megatron.core.transformer.transformer_config import TransformerConfig

from .base_moe_layer import BaseMoELayer
from .mlp import MLP, MLPSubmodules


class SwitchMLP(BaseMoELayer):
    """
    Top-1 Mixture of Experts Layer. Routes input to one of N MLP "experts"
    Curently supports Sinkhorn based expert routing.
    """

    def __init__(self, config: TransformerConfig, submodules: MLPSubmodules):
        super().__init__(config=config)

        self.local_experts = torch.nn.ModuleList()
        for _ in range(self.num_local_experts):
            expert = MLP(self.config, submodules, is_expert=True)
            self.local_experts.append(expert)

    def forward(self, hidden_states):
        global_hidden_states, global_indices = self.token_permutation(hidden_states)

        output_total = torch.zeros_like(global_hidden_states)
        output_bias_total = None
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

        output_total, output_bias_total = self.token_unpermutation(output_total, output_bias_total)

        return output_total, output_bias_total