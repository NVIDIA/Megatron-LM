# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import torch

from megatron.core import parallel_state, tensor_parallel, utils
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import is_graph_capturing
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
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
    import transformer_engine as te  # pylint: disable=unused-import

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
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.ep_group = pg_collection.ep
        # use pg_collection.expt_tp_group as tensor parallel group in this module.
        self.attn_tp_group = pg_collection.tp
        ep_size = utils.get_pg_size(self.ep_group)
        ep_rank = utils.get_pg_rank(self.ep_group)
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
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        self.submodules = submodules
        # TODO(Hepteract): delete the usage of the global parallel_state.
        # Initialize process groups with the global parallel_state.
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        super(MoELayer, self).__init__(
            config=config, layer_number=layer_number, pg_collection=pg_collection
        )
        self.moe_layer_recompute = (
            config.recompute_granularity == 'selective' and "moe" in config.recompute_modules
        )
        self.shared_experts_recompute = (
            config.recompute_granularity == 'selective'
            and "shared_experts" in config.recompute_modules
        )

        # Initialize router
        self.router = TopKRouter(config=self.config, pg_collection=pg_collection)

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
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
            pg_collection=pg_collection,
        )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(
                self.submodules.shared_experts, config=self.config, pg_collection=pg_collection
            )
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

    def router_and_preprocess(
        self,
        hidden_states: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
        routing_map: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
    ):
        """Compute and preprocess token routing for dispatch.

        This method uses the router to determine which experts to send each token to,
        producing routing probabilities and a mapping. It then preprocesses the
        hidden states and probabilities for the token dispatcher. The original
        hidden states are returned as a residual connection.
        """
        # If probs is given, the router should have been called. Skip it.
        if probs is None:
            assert (
                routing_map is None and residual is None
            ), "both routing_map and residual must be None if probs is None"
            probs, routing_map = self.router(hidden_states)
        else:
            assert (
                routing_map is not None or residual is not None
            ), "either routing_map or residual must be given if probs is given"

        # Return right here if we are capturing the MoE router.
        if self.cuda_graph_early_return("router"):
            self.early_returned = True
            outputs = [hidden_states, probs, routing_map]
            return outputs

        # If residual is given, the token dispatcher preprocess should have been called.
        # Skip it.
        if residual is None:
            residual = hidden_states
            hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
                hidden_states, routing_map, probs
            )
        else:
            assert probs is not None, "probs must not be None if residual is not None"
            assert routing_map is None, "routing_map must be None if residual is not None"

        # Return right here if we are capturing the MoE preprocess.
        if self.cuda_graph_early_return("preprocess"):
            self.early_returned = True
            outputs = [hidden_states, probs, residual]
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

        return hidden_states, probs, residual

    def dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Dispatches tokens to assigned expert ranks via communication.

        This method performs the actual communication (e.g., All-to-All) to distribute
        tokens and their associated probabilities to the devices hosting their assigned
        experts.
        """
        return self.token_dispatcher.token_dispatch(hidden_states, probs)

    def shared_experts_compute(self, hidden_states: torch.Tensor):
        """Computes the output of the shared experts.

        If a shared expert is configured and not overlapped with communication,
        it is computed here.
        """
        shared_expert_output = None
        if self.use_shared_expert and not self.shared_expert_overlap:
            # Compute the shared expert separately when not overlapped with communication.
            if self.shared_experts_recompute:
                if self.config.fp8:
                    shared_expert_output = te_checkpoint(
                        self.shared_experts,
                        False,
                        tensor_parallel.random.get_cuda_rng_tracker,
                        parallel_state.get_tensor_model_parallel_group(),
                        hidden_states,
                    )
                else:
                    shared_expert_output = tensor_parallel.checkpoint(
                        self.shared_experts, False, hidden_states
                    )
            else:
                shared_expert_output = self.shared_experts(hidden_states)

        return shared_expert_output

    def routed_experts_compute(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, residual: torch.Tensor
    ):
        """Computes the output of the routed experts on the dispatched tokens.

        This method first post-processes the dispatched input to get permuted tokens
        for each expert. It then passes the tokens through the local experts.
        The output from the experts is preprocessed for the combine step.
        """
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
        )
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert, permuted_probs)
        assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
        output = self.token_dispatcher.combine_preprocess(expert_output)

        return output, mlp_bias

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
        routing_map: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        shared_expert_output: Optional[torch.Tensor] = None,
    ):
        """Forward pass for the MoE layer.

        The forward pass comprises four main steps:
        1. Routing & Preprocessing: Route tokens to the assigned experts and prepare for dispatch.
        2. Dispatch: Tokens are sent to the expert devices using communication collectives.
        3. Expert Computation: Experts process the dispatched tokens.
        4. Combine: The outputs from the experts are combined and returned.

        Args:
            hidden_states (torch.Tensor): The input tensor to the MoE layer.
            probs (Optional[torch.Tensor]): The probabilities of the tokens to the experts. Router
                will be skipped if this is given.
            routing_map (Optional[torch.Tensor]): The routing map.
            residual (Optional[torch.Tensor]): The residual connection from router_and_preprocess.
                router_and_preprocess will be skipped if this is given.
            shared_expert_output (Optional[torch.Tensor]): The output of the shared experts. Shared
                expert computation will be skipped if this is given.

        Returns:
            A tuple containing the output tensor and the MLP bias, if any.
        """
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # MoE forward: route -> dispatch -> compute -> combine
        def custom_forward(
            hidden_states,
            probs: Optional[torch.Tensor] = None,
            routing_map: Optional[torch.Tensor] = None,
            residual: Optional[torch.Tensor] = None,
            shared_expert_output: Optional[torch.Tensor] = None,
        ):
            if shared_expert_output is None:
                shared_expert_output = self.shared_experts_compute(hidden_states)

            self.early_returned = False
            outputs = self.router_and_preprocess(
                hidden_states, probs=probs, routing_map=routing_map, residual=residual
            )
            if self.early_returned:
                if shared_expert_output is not None:
                    outputs.append(shared_expert_output)
                return outputs

            hidden_states, probs, residual = outputs
            dispatched_input, probs = self.dispatch(hidden_states, probs)
            output, mlp_bias = self.routed_experts_compute(dispatched_input, probs, residual)
            output = self.combine(output, shared_expert_output)
            return output, mlp_bias

        if self.moe_layer_recompute:
            if self.config.fp8:
                outputs = te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    probs,
                    routing_map,
                    residual,
                    shared_expert_output,
                )
            else:
                outputs = tensor_parallel.checkpoint(
                    custom_forward,
                    False,
                    hidden_states,
                    probs,
                    routing_map,
                    residual,
                    shared_expert_output,
                )
        else:
            outputs = custom_forward(
                hidden_states, probs, routing_map, residual, shared_expert_output
            )

        return outputs

    def backward_dw(self):
        """Compute weight gradients for experts and shared experts."""
        self.experts.backward_dw()
        if self.use_shared_expert and not self.shared_expert_overlap:
            self.shared_experts.backward_dw()

    def set_for_recompute_pre_mlp_layernorm(self):
        """Set the MoE layer for recompute pre_mlp_layernorm. Only needed for fp8."""
        # If shared_experts_recompute is used, nothing needs to be done because the checkpoint
        # function will save the original input tensors.
        if self.shared_experts is not None and not self.shared_experts_recompute:
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.shared_experts.linear_fc1)

    def cuda_graph_early_return(self, return_condition):
        """Check if the MoE layer should early return for CUDA graph capture."""
        if (
            self.config.cuda_graph_impl == "transformer_engine"
            and self.training
            and is_graph_capturing()
        ):
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
