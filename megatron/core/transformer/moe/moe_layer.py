# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.transformer.cuda_graphs import is_graph_capturing
from megatron.core.transformer.module import MegatronModule
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
    from megatron.core.extensions.transformer_engine import te_checkpoint, TELinear

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
        self.attn_tp_group = model_comm_pgs.tp
        ep_size = self.ep_group.size()
        ep_rank = self.ep_group.rank()
        assert ep_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % ep_size == 0
        self.num_local_experts = self.config.num_moe_experts // ep_size
        local_expert_indices_offset = ep_rank * self.num_local_experts

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap
        self.shared_expert_compute_before_router = (
            self.config.moe_shared_expert_compute_before_router
        )

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
    """Mixture of Experts layer.

    This layer implements a Mixture of Experts model, where each token is routed to a
    subset of experts. This implementation supports different token dispatching
    strategies such as All-to-All and All-Gather.
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

        # Initialize latent projections
        if self.config.moe_latent_size:
            assert HAVE_TE
            self.fc1_latent_proj = TELinear(
                self.config.hidden_size,
                self.config.moe_latent_size,
                parallel_mode="duplicated",
                config=self.config,
                init_method=self.config.init_method,
                bias=self.config.add_bias_linear,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                is_expert=False,
            )
            self.fc2_latent_proj = TELinear(
                self.config.moe_latent_size,
                self.config.hidden_size,
                parallel_mode="duplicated",
                config=self.config,
                init_method=self.config.output_layer_init_method,
                bias=self.config.add_bias_linear,
                skip_bias_add=True,
                skip_weight_param_allocation=False,
                is_expert=False,
            )

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

    def preprocess(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ):
        """Preprocess the hidden states for dispatch.

        This method preprocesses the hidden states and probabilities for the token dispatcher.
        The original hidden states are returned as a residual connection.
        """
        residual = hidden_states
        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
            hidden_states, routing_map, probs
        )
        return hidden_states, probs, residual

    def dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Dispatches tokens to assigned expert ranks via communication.

        This method performs the actual communication (e.g., All-to-All) to distribute
        tokens and their associated probabilities to the devices hosting their assigned
        experts.
        """
        return self.token_dispatcher.token_dispatch(hidden_states, probs)

    def _shared_experts_compute(self, hidden_states: torch.Tensor):
        """Computes the output of the shared experts."""
        shared_expert_output = None
        if self.use_shared_expert and not self.shared_expert_overlap:
            # Compute the shared expert separately when not overlapped with communication.
            shared_expert_output = self.shared_experts(hidden_states)
        return shared_expert_output

    def experts_compute(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        residual: torch.Tensor,
        shared_expert_output: Optional[torch.Tensor] = None,
    ):
        """Computes the output of the experts on the dispatched tokens.

        This method first post-processes the dispatched input to get permuted tokens
        for each expert. It then passes the tokens through the local experts.
        If a shared expert is configured and not overlapped with communication,
        it is also applied. The output from the experts is preprocessed for the
        combine step.
        """
        if shared_expert_output is None:
            shared_expert_output = self._shared_experts_compute(residual)
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
        )
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert, permuted_probs)
        assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
        output = self.token_dispatcher.combine_preprocess(expert_output)

        return output, shared_expert_output, mlp_bias

    def combine(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
        """Combines expert outputs via communication and adds shared expert output.

        This method uses the token dispatcher to combine the outputs from different
        experts (e.g., via an All-to-All communication). It then adds the output
        from the shared expert if it exists.
        """
        output = self.token_dispatcher.token_combine(output)
        output = self.token_dispatcher.combine_postprocess(output)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output

    def router_and_preprocess(self, hidden_states: torch.Tensor):
        """This is a combined method of router and preprocess."""
        probs, routing_map = self.router(hidden_states)
        return self.preprocess(hidden_states, probs, routing_map)

    def dispatch_compute_combine(self, hidden_states, probs, residual, shared_expert_output=None):
        """This is a combined method of dispatch, compute, and combine."""
        # Project the hidden_states from hidden dimension down to latent dimenion.
        # Shared expert computation is still performed in hidden dimension with the 'residual' tensor.
        if self.config.moe_latent_size:
            assert not self.shared_expert_overlap, "Shared expert overlap not supported when MoE latent projections are used."
            experts_input, _ = self.fc1_latent_proj(hidden_states)
        else:
            experts_input = hidden_states

        dispatched_input, probs = self.dispatch(experts_input, probs)
        output, shared_expert_output, mlp_bias = self.experts_compute(
            dispatched_input, probs, residual, shared_expert_output
        )

        # Project the output back from latent dimension to hidden dimension
        if self.config.moe_latent_size:
            if mlp_bias is not None:
                output = output + mlp_bias
            output, mlp_bias = self.fc2_latent_proj(output)

        output = self.combine(output, shared_expert_output)
        return output, mlp_bias

    def forward(self, hidden_states: torch.Tensor):
        """Forward pass for the MoE layer.

        The forward pass comprises four main steps:
        1. Routing & Preprocessing: Route tokens to the assigned experts and prepare for dispatch.
        2. Dispatch: Tokens are sent to the expert devices using communication collectives.
        3. Expert Computation: Experts process the dispatched tokens.
        4. Combine: The outputs from the experts are combined and returned.

        Args:
            hidden_states (torch.Tensor): The input tensor to the MoE layer.

        Returns:
            A tuple containing the output tensor and the MLP bias, if any.
        """
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        def cuda_graph_early_returns(return_condition):
            if not (is_graph_capturing() and self.training):
                return False

            if self.config.external_cuda_graph or self.config.enable_cuda_graph:
                if return_condition == "router":
                    if (
                        'moe_router' in self.config.cuda_graph_scope
                        and 'moe_preprocess' not in self.config.cuda_graph_scope
                    ):
                        return True
                elif return_condition == "preprocess":
                    if 'moe_preprocess' in self.config.cuda_graph_scope:
                        return True
            return False

        # MoE forward: route -> dispatch -> compute -> combine
        def custom_forward(hidden_states):
            if self.shared_expert_compute_before_router:
                shared_expert_output = self._shared_experts_compute(hidden_states)
            else:
                shared_expert_output = None

            probs, routing_map = self.router(hidden_states)

            # Return right here if we are capturing the MoE router.
            if cuda_graph_early_returns("router"):
                outputs = [hidden_states, probs, routing_map]
                if shared_expert_output is not None:
                    outputs.append(shared_expert_output)
                return outputs

            hidden_states, probs, residual = self.preprocess(hidden_states, probs, routing_map)

            # Return right here if we are capturing the MoE preprocess.
            if cuda_graph_early_returns("preprocess"):
                outputs = [hidden_states, probs, residual]
                if shared_expert_output is not None:
                    outputs.append(shared_expert_output)
                valid_cudagraph_attrs = []
                for attr_name in self.token_dispatcher.cudagraph_attrs:
                    hier_attr_name = attr_name.split('.')
                    attr = self.token_dispatcher
                    for name in hier_attr_name:
                        attr = getattr(attr, name, None)
                        if attr is None:
                            break
                    if isinstance(attr, torch.Tensor):
                        outputs.append(attr)
                        valid_cudagraph_attrs.append(attr_name)
                if self.token_dispatcher.valid_cudagraph_attrs is None:
                    self.token_dispatcher.valid_cudagraph_attrs = valid_cudagraph_attrs
                else:
                    assert self.token_dispatcher.valid_cudagraph_attrs == valid_cudagraph_attrs, (
                        "valid_cudagraph_attrs mismatch: "
                        f"{self.token_dispatcher.valid_cudagraph_attrs} != {valid_cudagraph_attrs}"
                    )
                return outputs

            return self.dispatch_compute_combine(
                hidden_states, probs, residual, shared_expert_output
            )

        if self.moe_layer_recompute:
            if self.config.fp8:
                outputs = te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                )
            else:
                outputs = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            outputs = custom_forward(hidden_states)

        return outputs

    def backward_dw(self):
        """Compute weight gradients for experts and shared experts."""
        self.experts.backward_dw()
        if self.use_shared_expert and not self.shared_expert_overlap:
            self.shared_experts.backward_dw()
