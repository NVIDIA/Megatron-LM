# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import (  # type: ignore
    MoEAlltoAllSEQTokenDispatcher,
)
from megatron.core.transformer.moe.moe_utils import get_default_model_comm_pgs
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from megatron.core.extensions.transformer_engine import te_checkpoint

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: Optional[int] = None,
        model_comm_pgs: Optional[ModelCommProcessGroups] = None,
    ):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.ep_group = model_comm_pgs.ep
        # use model_comm_pgs.expt_tp_group as tensor parallel group in this module.
        self.tp_group = model_comm_pgs.expt_tp
        ep_size = self.ep_group.size()
        ep_rank = self.ep_group.rank()
        assert ep_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % ep_size == 0
        self.num_local_experts = self.config.num_moe_experts // ep_size
        local_expert_indices_offset = ep_rank * self.num_local_experts

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router: TopKRouter = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher: Optional[MoETokenDispatcher] = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        model_comm_pgs: Optional[ModelCommProcessGroups] = None,
    ):
        self.submodules = submodules
        # TODO(Hepteract): delete the usage of the global parallel_state.
        # Initialize process groups with the global parallel_state.
        if model_comm_pgs is None:
            model_comm_pgs = get_default_model_comm_pgs()
        super(MoELayer, self).__init__(
            config=config, layer_number=layer_number, model_comm_pgs=model_comm_pgs
        )
        self.moe_layer_recompute = (
            config.recompute_granularity == 'selective' and "moe" in config.recompute_modules
        )

        # Initialize router
        self.router = TopKRouter(config=self.config, model_comm_pgs=model_comm_pgs)

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                model_comm_pgs=model_comm_pgs,
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                model_comm_pgs=model_comm_pgs,
            )
        elif config.moe_token_dispatcher_type == "alltoall_seq":
            self.token_dispatcher = MoEAlltoAllSEQTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                model_comm_pgs=model_comm_pgs,
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                model_comm_pgs=model_comm_pgs,
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize experts
        self.experts = build_module(
            self.submodules.experts,
            self.num_local_experts,
            self.config,
            model_comm_pgs=model_comm_pgs,
        )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(
                self.submodules.shared_experts, config=self.config, model_comm_pgs=model_comm_pgs
            )
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

    def forward(self, hidden_states: torch.Tensor):
        if self.training and self.tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # process MoE
        def custom_forward(hidden_states):
            probs, routing_map = self.router(hidden_states)
            (dispatched_input, tokens_per_expert, permuted_probs) = (
                self.token_dispatcher.token_permutation(hidden_states, probs, routing_map)
            )
            expert_output, mlp_bias = self.experts(
                dispatched_input, tokens_per_expert, permuted_probs
            )
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            if self.use_shared_expert and not self.shared_expert_overlap:
                # if shared_expert_overlap is True, the expert calculation happens in
                # the token_dispatcher to overlap communications and computations
                output = output + self.shared_experts(hidden_states)
            return output, mlp_bias

        if self.moe_layer_recompute:
            if self.config.fp8:
                output, mlp_bias = te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_device_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                )
            else:
                output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias
