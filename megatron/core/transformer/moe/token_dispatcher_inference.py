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
from megatron.core.transformer.moe.moe_utils import permute
from megatron.core.transformer.moe.inference_kernels import launch_moe_kernels, launch_extract_probs    

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
        """
        self.hidden_shape_before_permute = hidden_states.shape

        # The routing map and probs that for local experts.
        self.local_map = self.routing_map[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()
        # probs of global token assignment to local experts.
        self.local_probs = probs[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()
        # logging.info(f"Routing map shapre: {self.routing_map.shape}, local_map shape: {self.local_map.shape}, hidden_states shape: {hidden_states.shape}, local_probs shape: {self.local_probs.shape}")
        # logging.info(f"Routing map: {self.routing_map}")
        # exit()

        # Change 1: Keep tokens_per_expert on GPU for CUDA graph compatibility.
        tokens_per_expert = self.local_map.sum(dim=0).long() #.cpu()
        #hidden_states = torch.randn_like(hidden_states)  # Dummy init for exit()
        if False:
            (permuted_local_hidden_states, permuted_local_probs, self.reversed_local_input_permutation_mapping) = permute(
                hidden_states,
                self.local_map,
                probs=probs, # Change 2: permute probs as well
                num_out_tokens=hidden_states.size(0) * self.topk, # Change 3: accounting for worst case
                fused=self.config.moe_permute_fusion,
            )
            self.test_permute_output(hidden_states, permuted_local_hidden_states, self.local_map)
            self.test_permute_probs_output(self.local_probs, permuted_local_probs, self.local_map)
            logging.info("TE: After permute verification for both tokens and probs")
        else:
            # shape of static_buffer is [hidden_states.shape(0) * min(topk, num_local_experts), hidden_states.shape(1)]
            tokens_workspace = torch.zeros(
                hidden_states.size(0) * min(self.topk, self.num_local_experts),
                hidden_states.size(1),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            launch_moe_kernels(hidden_states, self.local_map, tokens_workspace, unpermute=False)    
            #self.test_permute_output(hidden_states, tokens_workspace, self.local_map)
            #logging.info("Triton: After permute verification in token_dispatcher_inference for tokens")

            probs_workspace = torch.zeros(
                self.local_probs.size(0) * min(self.topk, self.num_local_experts),
                1,
                dtype=probs.dtype,
                device=probs.device,
            )
            launch_extract_probs(self.local_probs, self.local_map, probs_workspace)
            #self.test_permute_probs_output(self.local_probs, probs_workspace, self.local_map)
            #logging.info("Triton: After permute verification in token_dispatcher_inference for probs")

            permuted_local_hidden_states = tokens_workspace
            permuted_local_probs = probs_workspace.squeeze(-1)
            # probs_workspace = torch.zeros(
            #     local_probs.size(0) * min(self.topk, self.num_local_experts),
            #     1,
            #     dtype=probs.dtype,
            #     device=probs.device,
            # )

            # print(probs.shape)
            # launch_moe_kernels(probs.unsqueeze(-1), self.local_map, probs_workspace, unpermute=False)
            # self.test_permute_output(probs.unsqueeze(-1), probs_workspace, self.local_map)


        self.local_probs = permuted_local_probs 
        self.routing_map = None
        return permuted_local_hidden_states, tokens_per_expert, self.local_probs

    def combine_preprocess(self, permuted_expert_outputs):
        """
        Reverses token permutation to restore original ordering.
        Handles Top-K summation into original hidden state positions.
        """
        # 1. Pre-allocate/Fetch static output buffer
        # In a real CUDA Graph, this should be a pre-allocated buffer attribute
        # to ensure the data_ptr() remains constant.
        unpermuted_hidden = torch.empty(
            self.hidden_shape_before_permute,
            dtype=permuted_expert_outputs.dtype,
            device=permuted_expert_outputs.device
        ).zero_()

        # 2. Launch the Un-permute kernel
        # This kernel uses 'atomic_add' to gather expert outputs.
        # It handles the Expert-grouped -> Token-major transition.
        # We use the same self.local_map and self.local_probs we cached during dispatch.
        launch_moe_kernels(
            unpermuted_hidden,      # The [Tokens, Hidden] destination
            self.local_map,         # The boolean mask [Tokens, Experts]
            permuted_expert_outputs, # The [MAX_OUT, Hidden] source
            unpermute=True
        )

        return unpermuted_hidden

