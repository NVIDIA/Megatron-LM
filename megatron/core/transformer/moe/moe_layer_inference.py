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
"""

from typing import Optional

import torch

from megatron.core import utils
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.transformer_config import TransformerConfig


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
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            pg_collection=pg_collection,
        )
        
        # Validate dispatcher type
        if config.moe_token_dispatcher_type != "alltoall":
            raise ValueError(
                f"InferenceMoELayer only supports 'alltoall' dispatcher, "
                f"got '{config.moe_token_dispatcher_type}'"
            )
        
        # Cache frequently used values
        self.hidden_size = config.hidden_size
        self.topk = config.moe_router_topk
        
        # Get process group info from token_dispatcher
        self.ep_size = self.token_dispatcher.ep_size
        self.ep_rank = utils.get_pg_rank(self.token_dispatcher.ep_group)
        self.tp_size = self.token_dispatcher.tp_size
        self.tp_rank = self.token_dispatcher.tp_rank
        
        # Precompute sort indices for multi-expert case
        if self.num_local_experts > 1:
            input_chunk_idxs = torch.arange(
                self.config.num_moe_experts * self.tp_size, device="cuda"
            )
            self.sort_input_by_local_experts = input_chunk_idxs.reshape(
                -1, self.num_local_experts
            ).T.ravel()
            self.restore_output_by_local_experts = input_chunk_idxs.reshape(
                self.num_local_experts, -1
            ).T.ravel()

    # ==================== Simplified Forward Pass ====================
    def forward(self, hidden_states: torch.Tensor):
        """
        Simplified forward pass optimized for inference.
        
        This overrides MoELayer.forward() with a streamlined version that:
        - Removes training overhead (aux losses, recomputation)
        - Uses a linear, easy-to-follow flow
        - Reuses inherited router, token_dispatcher, and experts
        
        Args:
            hidden_states: [S, B, H] input tensor
            
        Returns:
            Tuple of (output, None) for compatibility with MoELayer interface
        """
        print("USED INFERENCE MOE LAYER....")
        # Store original shape for restoration
        hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_shape[-1])
        num_tokens = hidden_states.shape[0]
        
        # ===== Step 1: Routing (using inherited router) =====
        # The router returns probs and routing_map
        probs, routing_map = self.router(hidden_states)
        
        # ===== Step 2: Dispatch Preprocess =====
        # Compute metadata and permute tokens by expert assignment
        permuted_tokens, permuted_probs = self.token_dispatcher.dispatch_preprocess(
            hidden_states, routing_map, probs
        )
        
        # ===== Step 3: Token Dispatch (EP AlltoAll) =====
        dispatched_tokens, dispatched_probs = self.token_dispatcher.token_dispatch(
            permuted_tokens, permuted_probs
        )
        
        # ===== Step 4: Dispatch Postprocess (TP AllGather + sort by expert) =====
        expert_input, tokens_per_expert, expert_probs = self.token_dispatcher.dispatch_postprocess(
            dispatched_tokens, dispatched_probs
        )
        
        # ===== Step 5: Expert Computation (using inherited experts) =====
        expert_output, mlp_bias = self.experts(expert_input, tokens_per_expert, expert_probs)
        
        # ===== Step 6: Combine Preprocess (unsort + TP ReduceScatter) =====
        combine_input = self.token_dispatcher.combine_preprocess(expert_output)
        
        # ===== Step 7: Token Combine (EP AlltoAll reverse) =====
        combined_output = self.token_dispatcher.token_combine(combine_input)
        
        # ===== Step 8: Combine Postprocess (unpermute to original order) =====
        output = self.token_dispatcher.combine_postprocess(combined_output)
        
        # Restore original shape
        output = output.view(hidden_shape)
        
        # Handle shared experts (if configured, computed separately)
        if self.use_shared_expert and not self.shared_expert_overlap:
            shared_output = self.shared_experts(hidden_states.view(hidden_shape))
            output = output + shared_output
        
        return output, mlp_bias

