# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Inference-optimized AllGather Token Dispatcher with GPU-resident metadata.

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

from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)

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
        super().__init__(
            num_local_experts=num_local_experts,
            local_expert_indices=local_expert_indices,
            config=config,
            pg_collection=pg_collection,
        )
        self.topk = config.moe_router_topk

    def token_dispatch(self, hidden_states, probs):
        """
        Gathers tokens from all EP ranks using NCCL AllGather.
        """
        if self.ep_size == 1:
            return hidden_states, probs
    
        self.routing_map = gather_from_sequence_parallel_region(
            self.routing_map, group=self.tp_ep_group
        )
        probs = gather_from_sequence_parallel_region(probs, group=self.tp_ep_group)
        hidden_states = gather_from_sequence_parallel_region(
            hidden_states, group=self.tp_ep_group
        )

        return hidden_states, probs

    def dispatch_postprocess(self, hidden_states, probs):
        """No op for flashinfer."""
        raise NotImplementedError

    def combine_preprocess(self, permuted_expert_outputs):
        """No op for flashinfer."""
        raise NotImplementedError

    def token_combine(self, hidden_states):
        """
        Combines expert outputs using NCCL Reduce-Scatter.
        """
        if self.ep_size == 1:
            return hidden_states

        return reduce_scatter_to_sequence_parallel_region(
            hidden_states, group=self.tp_ep_group
        )

