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

    def router_and_preprocess(self, hidden_states: torch.Tensor):
        """
        Determines token-to-expert routing and preprocesses tokens for dispatch.

        - Saves the input `hidden_states` as a residual.
        - Uses the MoE router to calculate routing probabilities (`probs`) and the
          binary routing map (`routing_map`) indicating expert assignments for each token.
        - Calls the token dispatcher's `dispatch_preprocess` method,
          which typically involves permuting and reshaping `hidden_states` and `probs`
          to prepare them for the communication phase (dispatch).

        Args:
            hidden_states (torch.Tensor): The input tensor to the MoE layer.
            Shape: (sequence_length, batch_size, hidden_dim) or (num_tokens, hidden_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - hidden_states (torch.Tensor): The input hidden_states, preprocessed
                  for dispatch.
                - probs (torch.Tensor): Routing probabilities preprocessed for dispatch.
                - residual (torch.Tensor): The original input `hidden_states`, preserved
                  for shared expert connection.
        """
        residual = hidden_states
        probs, routing_map = self.router(hidden_states)
        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
            hidden_states, routing_map, probs
        )
        return hidden_states, probs, residual

    def dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """
        Dispatches tokens and their probabilities to the assigned expert ranks.

        This method performs the actual communication (e.g., All-to-All) to distribute
        tokens and their associated probabilities to the devices hosting their assigned
        experts.

        Args:
            hidden_states (torch.Tensor): Tokens preprocessed for dispatch.
            probs (torch.Tensor): Routing probs preprocessed for dispatch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - hidden_states (torch.Tensor): Tokens dispatched to local experts.
                - probs (torch.Tensor): Routing probs dispatched to local experts.
        """
        return self.token_dispatcher.token_dispatch(hidden_states, probs)

    def experts_compute(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, residual: torch.Tensor
    ):
        """
        Processes dispatched tokens through local experts and prepares for combine.

        This method performs the following steps:
        1. Calls `token_dispatcher.dispatch_postprocess` to organize the dispatched
           `hidden_states` and `probs` per expert for efficient local computation.
        2. Passes the processed tokens and probabilities to the local experts.
        3. Calls `token_dispatcher.combine_preprocess` to prepare the output for the
           subsequent combine/communication phase (e.g., restoring token order before
           dispatch).
        4. Computes the shared_expert_output if enabled.

        Note: For optimal performance, especially with All-to-All overlapping, this
        function should ideally not contain inter-device communication.

        Args:
            hidden_states (torch.Tensor): Hidden states dispatched to local experts.
            probs (torch.Tensor): Routing probabilities dispatched to local experts.
            residual (torch.Tensor): The original input to the MoE layer, used for
                                     the shared expert if applicable.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
                - output (torch.Tensor): The output from local experts, preprocessed by
                  `combine_preprocess` and ready for the combine phase.
                - shared_expert_output (Optional[torch.Tensor]): The output from the
                  shared expert, if applicable and computed. Otherwise, None.
                - mlp_bias (Optional[torch.Tensor]): Bias from the MLP experts, if any.
                  Currently asserted to be None for this dispatcher type.
        """
        shared_expert_output = None
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
        )
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert, permuted_probs)
        assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
        output = self.token_dispatcher.combine_preprocess(expert_output)
        if self.use_shared_expert and not self.shared_expert_overlap:
            # if shared_expert_overlap is True, the expert calculation happens in
            # the token_dispatcher to overlap communications and computations
            shared_expert_output = self.shared_experts(residual)

        return output, shared_expert_output, mlp_bias

    def combine(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
        """
        Combines outputs from experts and finalizes the MoE layer output.

        This method performs the following steps:
        1. Calls `token_dispatcher.token_combine` to gather processed expert outputs
           from different expert ranks (e.g., All-to-All, Reduce-Scatter, etc.).
        2. Calls `token_dispatcher.combine_postprocess` to perform final processing
           on the combined output, such as unpermuting tokens to their original
           order and reshaping the tensor to its original input shape.
        3. If `shared_expert_output` is provided, it's added to the combined expert
           outputs.

        Args:
            output (torch.Tensor): The expert outputs, preprocessed and ready for combination
                                   (from `experts_compute`).
            shared_expert_output (Optional[torch.Tensor]): The output from the shared expert,
                                                          if applicable.

        Returns:
            torch.Tensor: The final output tensor of the MoE layer.
        """
        output = self.token_dispatcher.token_combine(output)
        output = self.token_dispatcher.combine_postprocess(output)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output

    def forward(self, hidden_states: torch.Tensor, start="None", end=None):
        if self.training and self.tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # process MoE
        # TODO: refactor all token dispatcher to use the same pre/postprocess interface
        def custom_forward(hidden_states):
            hidden_states, probs, residual = self.router_and_preprocess(hidden_states)
            dispatched_input, probs = self.dispatch(hidden_states, probs)
            output, shared_expert_output, mlp_bias = self.experts_compute(
                dispatched_input, probs, residual
            )
            output = self.combine(output, shared_expert_output)
            return output, mlp_bias

        if self.moe_layer_recompute:
            if self.config.fp8:
                output, mlp_bias = te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                )
            else:
                output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias

    def backward_dw(self):
        """Performs backward pass for weight gradients in MoELayer.

        This method executes the backward pass for weight gradients by calling
        backward_dw() on both the experts and shared_experts components.
        This ensures that gradients are properly computed for all expert weights
        in the mixture of experts layer.
        """
        self.experts.backward_dw()
        if self.use_shared_expert and not self.shared_expert_overlap:
            self.shared_experts.backward_dw()
