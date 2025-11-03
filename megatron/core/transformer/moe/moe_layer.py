# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from math import e
from typing import Optional, Union

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import get_default_model_comm_pgs
from megatron.core.transformer.moe.offloading_planner import gen_offloading_plan, gen_random_offloading_plan
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEElasticExpertDispatcher,
    MoESyncFreeElasticExpertDispatcher,
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
        if self.config.moe_enable_echo:
            self.num_local_total_experts = (
                self.config.num_moe_experts + self.config.moe_num_echo_experts
            ) // ep_size
        else:
            self.num_local_total_experts = self.config.num_moe_experts // ep_size
        self.num_home_experts = self.config.num_moe_experts // ep_size
        local_expert_indices_offset = ep_rank * self.num_local_total_experts

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_total_experts)
        ]
        if self.config.moe_enable_echo:
            assert all(
                map(
                    lambda x: x < self.config.num_moe_experts + self.config.moe_num_echo_experts,
                    self.local_expert_indices,
                )
            )
        else:
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

    expert_dispatch_stream: Optional[torch.cuda.Stream] = None
    fc1_expert_dispatch_event: Optional[torch.cuda.Event] = None
    fc2_expert_dispatch_event: Optional[torch.cuda.Event] = None

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
                self.num_local_total_experts,
                self.local_expert_indices,
                config=self.config,
                model_comm_pgs=model_comm_pgs,
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_total_experts,
                self.local_expert_indices,
                config=self.config,
                model_comm_pgs=model_comm_pgs,
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_total_experts,
                self.local_expert_indices,
                config=self.config,
                model_comm_pgs=model_comm_pgs,
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        if config.moe_enable_echo:
            MoELayer.expert_dispatch_stream = torch.cuda.Stream()
            MoELayer.fc1_expert_dispatch_event = torch.cuda.Event()
            MoELayer.fc2_expert_dispatch_event = torch.cuda.Event()

            if config.moe_echo_expert_dispatcher_type == "hybridep":
                self.expert_dispatcher = MoESyncFreeElasticExpertDispatcher(
                    config=self.config, model_comm_pgs=model_comm_pgs
                )
            elif config.moe_echo_expert_dispatcher_type == "alltoall":
                self.expert_dispatcher = MoEElasticExpertDispatcher(
                    config=self.config, model_comm_pgs=model_comm_pgs
                )
            else:
                raise ValueError(f"Unsupported expert dispatcher type: {config.moe_echo_expert_dispatcher_type}")
            num_echo_local_experts = self.config.moe_num_echo_experts // self.ep_group.size()
            wgrad_accumulation_mask = [True] * self.num_home_experts + [False] * num_echo_local_experts
            wgrad_accumulation_mask = torch.tensor(wgrad_accumulation_mask)
            echo_config = self.config
            kargs = {}
            if self.config.moe_use_device_initiated_grouped_gemm:
                kargs["wgrad_accumulation_mask"] = wgrad_accumulation_mask
            else:
                echo_config = dataclasses.replace(
                    self.config, gradient_accumulation_fusion=False
                )
            self.experts = build_module(
                self.submodules.experts,
                num_echo_local_experts+self.num_home_experts,
                config=echo_config,
                model_comm_pgs=model_comm_pgs,
                **kargs,
            )
            self.echo_expert_indices = list(
                range(self.num_home_experts, num_echo_local_experts + self.num_home_experts)
            )
            self.home_expert_indices = list(range(self.num_home_experts))
            self.experts.free_expert_parameters(self.echo_expert_indices)
        else:
            self.experts = build_module(
                self.submodules.experts,
                self.num_home_experts,
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
        """Compute and preprocess token routing for dispatch.

        This method uses the router to determine which experts to send each token to,
        producing routing probabilities and a mapping. It then preprocesses the
        hidden states and probabilities for the token dispatcher. The original
        hidden states are returned as a residual connection.
        """
        probs, routing_map = self.router(hidden_states)
        metadata = self.token_dispatcher.preprocess(routing_map)
        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
            hidden_states, probs, metadata
        )
        return hidden_states, probs, metadata

    def dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor, metadata):
        """Dispatches tokens to assigned expert ranks via communication.
        This method performs the actual communication (e.g., All-to-All) to distribute
        tokens and their associated probabilities to the devices hosting their assigned
        experts.
        """
        return self.token_dispatcher.token_dispatch(hidden_states, probs, metadata)

    def experts_compute(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, residual: torch.Tensor, metadata
    ):
        """Computes the output of the experts on the dispatched tokens.

        This method first post-processes the dispatched input to get permuted tokens
        for each expert. It then passes the tokens through the local experts.
        If a shared expert is configured and not overlapped with communication,
        it is also applied. The output from the experts is preprocessed for the
        combine step.
        """
        shared_expert_output = None
        if self.use_shared_expert and not self.shared_expert_overlap:
            # Compute the shared expert separately when not overlapped with communication.
            shared_expert_output = self.shared_experts(residual)
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.token_dispatcher.dispatch_postprocess(hidden_states, probs, metadata)
        )
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert, permuted_probs)
        assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
        output = self.token_dispatcher.combine_preprocess(expert_output, metadata)

        return output, shared_expert_output, mlp_bias

    def combine(self, output: torch.Tensor, metadata, shared_expert_output: Optional[torch.Tensor]):
        """Combines expert outputs via communication and adds shared expert output.

        This method uses the token dispatcher to combine the outputs from different
        experts (e.g., via an All-to-All communication). It then adds the output
        from the shared expert if it exists.
        """
        output = self.token_dispatcher.token_combine(output, metadata)
        output = self.token_dispatcher.combine_postprocess(output, metadata)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output

    def echo_forward(self, hidden_states: torch.Tensor):
        """Forward pass for the MoE layer with echo experts.

        This implements the echo expert logic where overflow tokens are offloaded
        to spare/echo experts for better load balancing.

        Args:
            hidden_states (torch.Tensor): The input tensor to the MoE layer.

        Returns:
            A tuple containing the output tensor and the MLP bias, if any.
        """
        residual = hidden_states
        router = self.router

        # Step 1: Routing and offloading planning
        with torch.cuda.nvtx.range("router"):
            probs, routing_map = router(hidden_states)

        with torch.cuda.nvtx.range("rerouting"):
            tokens_per_expert_current_ep_rank = routing_map.sum(dim=0)
            tokens_per_expert_per_ep_rank = gather_from_sequence_parallel_region(
                tokens_per_expert_current_ep_rank, group=self.ep_group
            ).reshape(self.ep_group.size(), self.config.num_moe_experts)

            # Generate offloading plan to redistribute tokens to echo experts
            if self.config.moe_echo_enable_random_offloading:
                rerouting_map, rerouted_probs, expert_offloading_map = gen_random_offloading_plan(
                    routing_map,
                    probs,
                    tokens_per_expert_per_ep_rank,
                    self.ep_group.rank(),
                    ep=self.ep_group.size(),
                    spare_expert_per_ep_rank=self.config.moe_num_echo_experts // self.ep_group.size(),
                )
            else:
                rerouting_map, rerouted_probs, expert_offloading_map = gen_offloading_plan(
                    routing_map,
                    probs,
                    tokens_per_expert_per_ep_rank,
                    self.ep_group.rank(),
                    num_ep_ranks=self.ep_group.size(),
                    num_spare_experts_per_ep_rank=self.config.moe_num_echo_experts // self.ep_group.size(),
                )
        # Step 2: Expert weight dispatch for echo experts
        # Create checkpoints for gradient computation
        with torch.cuda.nvtx.range("expert_dispatch"):
            fc1_expert_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            fc2_expert_checkpoint = tensor_parallel.CheckpointWithoutOutput()

            # Dispatch fc1 expert weights
            fc1_expert_weights = self.experts.get_expert_weights(
                "fc1", self.home_expert_indices
            )

            expert_dispatch_metadata = self.expert_dispatcher.preprocess(expert_offloading_map)
            if self.config.moe_echo_expert_dispatch_overlap:
                MoELayer.expert_dispatch_stream.wait_stream(torch.cuda.current_stream())
                expert_dispatch_context = torch.cuda.stream(MoELayer.expert_dispatch_stream)
            else:
                expert_dispatch_context = nullcontext()
            default_stream = torch.cuda.current_stream()
            with expert_dispatch_context:
                if self.config.moe_echo_recompute_expert_dispatch:
                    dispatched_fc1_weights = fc1_expert_checkpoint.checkpoint(
                        partial(
                            self.expert_dispatcher.expert_dispatch, 
                            expert_dispatch_metadata, 
                        ),
                        *fc1_expert_weights,
                    )
                else:
                    dispatched_fc1_weights = self.expert_dispatcher.expert_dispatch(
                        expert_dispatch_metadata,
                        *fc1_expert_weights,
                    )
                MoELayer.fc1_expert_dispatch_event.record()
                for expert_weight in dispatched_fc1_weights:
                    expert_weight.record_stream(default_stream)
                self.experts.set_expert_weights(
                    "fc1",
                    dispatched_fc1_weights,
                    self.echo_expert_indices,
                )

                # Dispatch fc2 expert weights
                fc2_expert_weights = self.experts.get_expert_weights(
                    "fc2", self.home_expert_indices
                )
                if self.config.moe_echo_recompute_expert_dispatch:
                    dispatched_fc2_weights = fc2_expert_checkpoint.checkpoint(
                        partial(
                            self.expert_dispatcher.expert_dispatch, 
                            expert_dispatch_metadata, 
                        ),
                        *fc2_expert_weights,
                    )
                else:
                    dispatched_fc2_weights = self.expert_dispatcher.expert_dispatch(
                        expert_dispatch_metadata,
                        *fc2_expert_weights,
                    )
                for expert_weight in dispatched_fc2_weights:
                    expert_weight.record_stream(default_stream)
                self.experts.set_expert_weights(
                    "fc2",
                    dispatched_fc2_weights,
                    self.echo_expert_indices,
                )
                MoELayer.fc2_expert_dispatch_event.record()

        # Step 3: Token dispatch preprocess
        with torch.cuda.nvtx.range("token_dispatch_preprocess"):
            metadata = self.token_dispatcher.preprocess(rerouting_map)
            hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
                hidden_states, rerouted_probs, metadata
            )

        def dispatch_and_compute(hidden_states, probs, metadata):
            with torch.cuda.nvtx.range("token_dispatch"):
                dispatched_input, probs = self.token_dispatcher.token_dispatch(
                    hidden_states, probs, metadata
                )
                dispatched_input, tokens_per_expert, permuted_probs = (
                    self.token_dispatcher.dispatch_postprocess(dispatched_input, probs, metadata)
                )

            # Step 5: Expert computation
            with torch.cuda.nvtx.range("expert_compute"):
                if self.config.moe_echo_expert_dispatch_overlap:
                    expert_output, mlp_bias = self.experts(
                        dispatched_input, tokens_per_expert, permuted_probs, MoELayer.fc1_expert_dispatch_event, MoELayer.fc2_expert_dispatch_event
                    )
                else:
                    expert_output, mlp_bias = self.experts(
                        dispatched_input, tokens_per_expert, permuted_probs
                    )

            with torch.cuda.nvtx.range("token_combine"):
                # Step 6: Token combine
                # expert_output = torch.cat([expert_output, padded_tokens], dim=0) # TODO: remove this with device-inited grouped gemm
                output = self.token_dispatcher.combine_preprocess(expert_output, metadata)
                output = self.token_dispatcher.token_combine(output, metadata)
                output = self.token_dispatcher.combine_postprocess(output, metadata)
            return output, mlp_bias

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(
                partial(dispatch_and_compute, metadata=metadata), False, hidden_states, probs
            )
        else:
            output, mlp_bias = dispatch_and_compute(hidden_states, probs, metadata)

        # Register for gradient computation
        if self.config.moe_echo_recompute_expert_dispatch:
            fc1_expert_checkpoint.discard_output_and_register_recompute(output)
            fc2_expert_checkpoint.discard_output_and_register_recompute(output)

        # Handle shared expert if configured
        if self.use_shared_expert and not self.shared_expert_overlap:
            shared_expert_output = self.shared_experts(residual)
            output = output + shared_expert_output

        return output, mlp_bias

    def forward(self, hidden_states: torch.Tensor):
        """Forward pass for the MoE layer.

        The forward pass comprises four main steps:
        1. Routing & Preprocessing: Route tokens to the assigned experts and prepare for dispatch.
        2. Dispatch: Tokens are sent to the expert devices using communication collectives.
        3. Expert Computation: Experts process the dispatched tokens.
        4. Combine: The outputs from the experts are combined and returned.

        When echo mode is enabled (moe_enable_echo=True), the forward pass uses an alternative
        implementation that offloads overflow tokens to echo experts for better load balancing.

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

        # Use echo forward if echo mode is enabled
        if self.config.moe_enable_echo:
            return self.echo_forward(hidden_states)

        residual = hidden_states
        hidden_states, probs, metadata = self.router_and_preprocess(hidden_states)

        # MoE forward: dispatch -> compute -> combine
        def dispatch_and_compute(hidden_states, probs, residual, metadata):
            dispatched_input, probs = self.dispatch(hidden_states, probs, metadata)
            output, shared_expert_output, mlp_bias = self.experts_compute(
                dispatched_input, probs, residual, metadata
            )
            output = self.combine(output, metadata, shared_expert_output)
            return output, mlp_bias

        if self.moe_layer_recompute:
            if self.config.fp8:
                output, mlp_bias = te_checkpoint(
                    dispatch_and_compute,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    probs,
                    residual,
                    metadata,
                )
            else:
                output, mlp_bias = tensor_parallel.checkpoint(
                    partial(dispatch_and_compute, metadata=metadata),
                    False,
                    hidden_states,
                    probs,
                    residual,
                )
        else:
            output, mlp_bias = dispatch_and_compute(hidden_states, probs, residual, metadata)

        return output, mlp_bias

    def backward_dw(self):
        """Compute weight gradients for experts and shared experts."""
        self.experts.backward_dw()
        if self.use_shared_expert and not self.shared_expert_overlap:
            self.shared_experts.backward_dw()
