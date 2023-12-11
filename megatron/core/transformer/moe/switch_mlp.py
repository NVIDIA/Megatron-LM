# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import numpy as np
import torch

from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig

from .base_moe_layer import BaseMoELayer


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
        (
            permuted_local_hidden_states,
            tokens_per_expert,
            indices,
            global_local_map,
        ) = self.token_permutation(hidden_states)

        output_local = torch.zeros_like(permuted_local_hidden_states)
        output_bias_local = None
        if self.add_bias:
            output_bias_local = torch.zeros_like(permuted_local_hidden_states)

        cumsum_num_tokens = torch.cumsum(tokens_per_expert, dim=0)
        # Insert zero at the begining for offset index's convenience
        zero_tensor = torch.zeros(1, dtype=torch.long)
        cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))
        for expert_num, expert in enumerate(self.local_experts):
            start = cumsum_num_tokens[expert_num]
            end = cumsum_num_tokens[expert_num + 1]
            hidden = permuted_local_hidden_states[start:end]
            output, output_bias = expert(hidden)

            output_local[start:end] = output
            if self.add_bias:
                output_bias = output_bias.expand_as(output)
                output_bias_local[start:end, :] = output_bias

        # Un-permutation of tokens.
        output_total, output_bias_total = self.token_unpermutation(
            output_local, indices, global_local_map, output_bias_local
        )

        return output_total, output_bias_total
