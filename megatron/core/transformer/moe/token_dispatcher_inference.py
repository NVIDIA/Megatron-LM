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
                hidden_states, group=self.tp_ep_group, use_global_buffer=True
            )

        logging.info("Completed token dispatch AllGather.")
        exit()

        return hidden_states, probs

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

        tokens_per_expert = self.local_map.sum(dim=0).long().cpu()

        (permuted_local_hidden_states, _, self.reversed_local_input_permutation_mapping) = permute(
            hidden_states,
            self.local_map,
            num_out_tokens=tokens_per_expert.sum(),
            fused=self.config.moe_permute_fusion,
        )

        self.local_probs = self.local_probs.T.contiguous().masked_select(
            self.local_map.T.contiguous()
        )
        self.routing_map = None
        return permuted_local_hidden_states, tokens_per_expert, self.local_probs

   

