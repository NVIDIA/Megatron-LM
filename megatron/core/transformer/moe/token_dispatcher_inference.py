# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Inference-optimized AlltoAll Token Dispatcher with GPU-resident metadata.

This implementation keeps tokens_per_expert GPU-resident to enable use of
torch._grouped_mm without host synchronization.
"""

import torch
from typing import List, Optional

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.moe.inference_kernels import (
    launch_fused_permute_and_probs,
    launch_unpermute_kernel,
)    

import logging

class InferenceAllGatherTokenDispatcher(MoEAllGatherTokenDispatcher):
    """
    Inference-optimized AllGather token dispatcher.

    This dispatcher uses AllGather instead of AlltoAll for token exchange,
    which can be simpler and more efficient for certain configurations.

    Key features:
    - Simpler communication pattern (AllGather vs AlltoAll)
    - GPU-resident metadata for CUDA graph compatibility
    - Assumes tp_size == 1 (no tensor parallelism within experts)
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """
        Initialize the inference AllGather token dispatcher.

        Args:
            num_local_experts: Number of experts on this rank.
            local_expert_indices: Global indices of experts on this rank.
            config: Transformer configuration.
            pg_collection: Process group collection for distributed ops.
        """
        super().__init__(
            num_local_experts=num_local_experts,
            local_expert_indices=local_expert_indices,
            config=config,
            pg_collection=pg_collection,
        )
        self.topk = config.moe_router_topk

    def token_dispatch(self, hidden_states, probs):
        """Gathers tokens from all TP*EP ranks using AllGather."""

        # Permute the tokens across the expert parallel devices.
        if self.tp_size > 1 or self.ep_size > 1:
            ## local_indices calculation
            with torch.no_grad():
                # [num_local_tokens, num_experts] -> [num_global_tokens, num_experts], where:
                #     num_local_tokens=(S/TP)*B, num_global_tokens=S*B*EP
                self.routing_map = gather_from_sequence_parallel_region(
                    self.routing_map, group=self.tp_ep_group
                )

            ## local_probs calculation
            # max_prob: [S/TP*B, num_experts] -> global_probs: [S*B*EP, num_experts]
            probs = gather_from_sequence_parallel_region(probs, group=self.tp_ep_group)
            # Note that this allgather spans the communication domain of TP*EP.
            #  [(S/TP)*B, H] -> [((S/TP)*B)*(TP*EP), H] = [S*B*EP, H]
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, group=self.tp_ep_group
            )

        return hidden_states, probs

    def test_permute_output(self, hidden_states, permute_output, mask):
        # Verification of Grouped-by-Expert layout
        E = self.local_map.size(1)
        T = hidden_states.size(0)
        mask = self.local_map
        buffer_idx = 0
        for e_idx in range(E):
            for t_idx in range(T):
                if mask[t_idx, e_idx]:
                    assert torch.allclose(permute_output[buffer_idx], hidden_states[t_idx])
                    buffer_idx += 1
        
        #assert static_buffer[buffer_idx:].sum() == 0, "Stale data found in buffer tail"

    def test_permute_probs_output(self, local_probs, probs_workspace, mask):
        """
        Verification of Grouped-by-Expert layout for probabilities.
        local_probs: [Tokens, Experts]
        probs_workspace: [MAX_OUT, 1] (or [MAX_OUT])
        mask: [Tokens, Experts] boolean mask
        """
        T = local_probs.size(0)
        E = local_probs.size(1)
        
        buffer_idx = 0
        # Expert-major traversal (Outer loop: Experts, Inner loop: Tokens)
        for e_idx in range(E):
            for t_idx in range(T):
                if mask[t_idx, e_idx]:
                    # Extract the expected probability from the source [Tokens, Experts]
                    expected_prob = local_probs[t_idx, e_idx]
                                        # Using a slightly relaxed atol for BF16 if necessary
                    actual_prob = probs_workspace[buffer_idx]
                    assert torch.allclose(
                        actual_prob,
                        expected_prob
                    ), f"Prob mismatch at buffer index {buffer_idx} (Expert {e_idx}, Token {t_idx})"
                    
                    buffer_idx += 1
        
    def dispatch_postprocess(self, hidden_states, probs):
        """After gathering in token_dispatch, this method identifies tokens for local experts and
        permutes them for expert processing.

        Optimized to:
        1. Fuse slice + transpose for mask (single kernel instead of two)
        2. Use stride-based probs access in kernel (avoids probs transpose entirely)
        3. Permute hidden states AND extract probs in a single kernel launch
        """
        self.hidden_shape_before_permute = hidden_states.shape

        # Fuse slice + transpose for mask: [T, num_experts] -> [num_local_experts, T]
        # This produces mask_T directly, avoiding a separate transpose kernel
        self._cached_mask_T = self.routing_map[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].t().contiguous()  # [E, T] layout

        # Probs: just slice, no transpose needed (kernel uses stride-based access)
        local_probs = probs[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()  # [T, E] layout

        # tokens_per_expert from transposed mask: sum over tokens (dim=1) for each expert
        tokens_per_expert = self._cached_mask_T.sum(dim=1)

        # Pre-allocate workspaces
        max_out = hidden_states.size(0) * min(self.topk, self.num_local_experts)
        tokens_workspace = torch.empty(
            max_out, hidden_states.size(1),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        probs_workspace = torch.empty(
            max_out,
            dtype=probs.dtype,
            device=probs.device,
        )

        # Fused kernel launch: permute hidden states + extract probs in one pass
        # Pass mask_T directly (already transposed), probs as [T, E] (kernel uses strides)
        self._cached_dest_indices = launch_fused_permute_and_probs(
            hidden_states, local_probs, self._cached_mask_T,
            tokens_workspace, probs_workspace
        )

        self.routing_map = None
        self.local_probs = probs_workspace
        return tokens_workspace, tokens_per_expert, probs_workspace

    def combine_preprocess(self, permuted_expert_outputs):
        """
        Reverses token permutation to restore original ordering.
        Handles Top-K summation into original hidden state positions.

        Uses cached mask_T and dest_indices from dispatch_postprocess to avoid
        recomputing them (saves 2 kernel launches).
        """
        # 1. Pre-allocate static output buffer (zeros for atomic accumulation)
        unpermuted_hidden = torch.zeros(
            self.hidden_shape_before_permute,
            dtype=permuted_expert_outputs.dtype,
            device=permuted_expert_outputs.device
        )

        # 2. Launch the Un-permute kernel with cached intermediates
        # It handles the Expert-grouped -> Token-major transition.
        launch_unpermute_kernel(
            unpermuted_hidden,        # The [T, H] destination
            permuted_expert_outputs,  # The [max_out, H] source
            self._cached_mask_T,      # Cached [E, T] mask
            self._cached_dest_indices # Cached cumsum indices
        )

        return unpermuted_hidden

