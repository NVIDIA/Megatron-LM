# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Inference-optimized MoE Layer with AlltoAll Token Dispatcher.

This implementation inherits from MoELayer to ensure checkpoint compatibility,
while providing a simplified forward pass optimized for inference:
1. Strips out training-specific code (aux losses, recomputation, backward)
2. Uses a simple, linear forward flow
3. Is designed to be CUDA graph compatible (future work)

The forward pass follows this flow:
    Input [S, B, H]
        ↓ Route (router gate → topk selection)
    probs, routing_map
        ↓ Permute (group tokens by expert)
    permuted_tokens [num_selected_tokens, H]
        ↓ EP AlltoAll (distribute to expert owners)
    global_tokens [tokens_on_this_rank, H]
        ↓ TP AllGather (if tp_size > 1)
    gathered_tokens
        ↓ Sort by local expert (if num_local_experts > 1)
    sorted_tokens
        ↓ Expert FFN (GroupedGEMM)
    expert_output
        ↓ Unsort by local expert
    unsorted_output
        ↓ TP ReduceScatter (if tp_size > 1)
    scattered_output
        ↓ EP AlltoAll (return to original ranks)
    combined_output
        ↓ Unpermute (restore original order)
    Output [S, B, H]

Usage:
    # Load a trained MoELayer checkpoint directly:
    inference_layer = InferenceMoELayer(config, submodules, layer_number, pg_collection)
    inference_layer.load_state_dict(trained_moe_layer.state_dict())

TODO: Add unit test to verify that InferenceMoELayer.forward() and MoELayer.forward()
      have aligned argument signatures (use inspect.signature to compare).
"""

from typing import Optional

import torch
import torch.nn.functional as F

from megatron.core import utils
from megatron.core.activations import squared_relu
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.moe.token_dispatcher_inference import InferenceAllGatherTokenDispatcher
import flashinfer.fused_moe as fused_moe
from flashinfer.fused_moe.core import ActivationType

import logging

class InferenceMoELayer(MoELayer):
    """
    Inference-optimized MoE layer that inherits from MoELayer for checkpoint compatibility.
    
    This implementation:
    - Inherits all weights/submodules from MoELayer (router, experts, token_dispatcher)
    - Provides a simplified forward() optimized for inference
    - Removes training overhead (aux losses, recomputation, gradient computation)
    - Only supports AlltoAll dispatcher (most common for inference)
    
    Checkpoints trained with MoELayer can be loaded directly.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        """
        Initialize the inference MoE layer.
        
        Args are identical to MoELayer for checkpoint compatibility.
        """
        # Initialize parent MoELayer (creates router, experts, token_dispatcher)
        if pg_collection is None:
            pg_collection = get_default_pg_collection()

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            pg_collection=pg_collection,
        )
        
        # Validate dispatcher type
        # todo: move this assert to arguments.py or transformer_config.py
        if config.moe_token_dispatcher_type != "alltoall":
            raise ValueError(
                f"InferenceMoELayer only supports 'alltoall' dispatcher, "
                f"got '{config.moe_token_dispatcher_type}'"
            )
        
        self.is_cuda_graphed_iteration = False
        self.inference_token_dispatcher = InferenceAllGatherTokenDispatcher( 
                                                                            self.num_local_experts,
                                                                            self.local_expert_indices,
                                                                            config=self.config,
                                                                            pg_collection=pg_collection,
                                                                        )  
    def set_is_cuda_graphed_iteration(self, set_to):
        self.is_cuda_graphed_iteration = set_to
        self.router.set_is_cuda_graphed_iteration(set_to)

    def activate_inference_token_dispatcher(self):
        # replace the token dispatcher with the inference-optimized version
        self.old_token_dispatcher = self.token_dispatcher
        self.token_dispatcher = self.inference_token_dispatcher

        # disable shared expert overlap during inference as it is not 
        # supported in InferenceAllGatherTokenDispatcher
        self.old_expert_overlap = self.shared_expert_overlap
        self.shared_expert_overlap = False 
        
    def deactivate_inference_token_dispatcher(self):
        # restore the original token dispatcher 
        # and shared expert overlap setting
        self.token_dispatcher = self.old_token_dispatcher
        self.shared_expert_overlap = self.old_expert_overlap
        
        
    # ==================== Simplified Forward Pass ====================
    def forward(self, hidden_states: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        """      
        """
        if not self.is_cuda_graphed_iteration:
            # Note: this will still call InferenceGroupedMLP.forward()
            # and therefore optimized cutlass grouped gemms. 
            return super().forward(hidden_states, padding_mask)

        self.activate_inference_token_dispatcher()
        assert self.token_dispatcher is self.inference_token_dispatcher 
        #logging.info("activated inference token dispatcher")

        forward_pass_output = super().forward(hidden_states, padding_mask)

        self.deactivate_inference_token_dispatcher()
        assert self.token_dispatcher is not self.inference_token_dispatcher
        #logging.info("deactivated inference token dispatcher")

        return forward_pass_output

    def routed_experts_compute(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Computes the output of the routed experts on the dispatched tokens.

        This method first post-processes the dispatched input to get permuted tokens
        for each expert. It then passes the tokens through the local experts.
        The output from the experts is preprocessed for the combine step.
        """
        if not self.is_cuda_graphed_iteration:
            # todo: can we go down the flashinfer path even if not cuda graphed? 
            return super().routed_experts_compute(hidden_states, probs)

        # Currently only squared_relu (non-gated) is supported with FlashInfer
        assert not self.config.gated_linear_unit, (
            "FlashInfer MoE kernel currently only supports non-gated activations. "
            f"Got gated_linear_unit={self.config.gated_linear_unit}"
        )
        assert self.config.activation_func == squared_relu, (
            "FlashInfer MoE kernel currently only supports squared_relu activation. "
            f"Got activation_func={self.config.activation_func}"
        )

        # Get dtype from input
        output_dtype = hidden_states.dtype
        
        # Get expert weights from self.experts (GroupedMLP)
        w1 = self.experts._fc1_weight
        w2 = self.experts._fc2_weight

        # Get routing information (stored from route() step)
        selected_experts = self.token_dispatcher.routing_map
        routing_weights = probs
        
        # Get EP attributes
        ep_size = utils.get_pg_size(self.ep_group)
        ep_rank = utils.get_pg_rank(self.ep_group)
    
        # Call FlashInfer fused MoE kernel with Relu2 (squared ReLU)
        output = fused_moe.cutlass_fused_moe(
            hidden_states,
            selected_experts.to(torch.int),
            routing_weights.float(),
            w1,
            w2,
            output_dtype,
            quant_scales=None,
            activation_type=ActivationType.Relu2,
            ep_size=ep_size,
            ep_rank=ep_rank,
        )[0]
        
        return output, None

      
