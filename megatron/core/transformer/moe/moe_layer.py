# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod

import torch

from megatron.core import parallel_state
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.base_moe_layer import DroplessSinkhornRouter, DroplessTopKRouter
from megatron.core.transformer.moe.grouped_mlp import GroupedMLP
from megatron.core.transformer.moe.switch_mlp import SwitchMLP
from megatron.core.transformer.transformer_config import TransformerConfig


class BaseMoELayer(MegatronModule, ABC):
    def __init__(self, config: TransformerConfig):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.router = None
        self.experts = None

    @abstractmethod
    def initialize_experts(self):
        pass

    @abstractmethod
    def initialize_router(self):
        pass

    @abstractmethod
    def forward(self, hidden_states):
        pass


class SwitchMLPLayer(BaseMoELayer):
    """
    Top-K Mixture of Experts Layer Without Token Dropping.
    Currently supports Sinkhorn-based expert routing (Top-1 only) and a generalized Top-k routing with Z loss and auxiliary loss.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(self, config: TransformerConfig, submodules: MLPSubmodules = None):
        self.submodules = submodules
        super(SwitchMLPLayer, self).__init__(config=config)
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )
        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        self.router = self.initialize_router()
        self.experts = self.initialize_experts()
        assert config.moe_token_dropping is False

    def forward(self, hidden_states):
        # process MoE
        scores, indices = self.router(hidden_states)
        (
            dispatched_input,
            tokens_per_expert,
            scores,
            indices,
            global_local_map,
        ) = self.router.token_dispatcher.dispatch(hidden_states, scores, indices)
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
        output, mlp_bias = self.router.token_dispatcher.restore(
            expert_output, scores, indices, global_local_map, mlp_bias
        )

        if mlp_bias is None:
            mlp_bias = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        return output, mlp_bias

    def initialize_experts(self):
        if self.config.moe_grouped_gemm:
            experts = GroupedMLP(self.num_local_experts, self.config)
        else:
            assert isinstance(self.submodules, MLPSubmodules)
            experts = SwitchMLP(self.num_local_experts, self.config, self.submodules)
        return experts

    def initialize_router(self):
        if self.config.moe_router_type.lower().startswith("top"):
            router = DroplessTopKRouter(
                self.num_local_experts, self.local_expert_indices, self.config
            )
        elif self.config.moe_router_type.lower() == "sinkhorn":
            router = DroplessSinkhornRouter(
                self.num_local_experts, self.local_expert_indices, self.config
            )
        else:
            raise NotImplementedError(f"Routing method {self.config.moe_router_type} not supported")
        return router
