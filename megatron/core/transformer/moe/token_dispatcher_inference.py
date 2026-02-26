# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Inference-optimized AlltoAll Token Dispatcher with GPU-resident metadata.

This implementation keeps tokens_per_expert GPU-resident to enable use of
torch._grouped_mm without host synchronization.

Supports latency-optimized NVLS collectives (multimem all-gather/reduce-scatter)
on Hopper+ GPUs with BF16, with automatic fallback to NCCL via superclass methods.
"""

from megatron.core.tensor_parallel.mappings import reduce_scatter_to_sequence_parallel_region
import torch
from typing import List, Optional

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.core.tensor_parallel import gather_from_sequence_parallel_region, reduce_scatter_to_sequence_parallel_region
from megatron.core.parallel_state import get_global_symmetric_memory_buffer_ep
from megatron.core.inference.communication.torch_symm_triton import (
    multimem_all_gather_fused,
    multimem_reduce_scatter,
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

        self.triton_nvls_kernels_allowed = not self.config.inference_disable_triton_nvls_kernels

    def _check_nvls_eligibility(self, x: torch.Tensor) -> bool:
        """
        Check if we can use NVLS (latency-optimized) collectives.
        Requirements: BF16 dtype, Hopper+ GPU (SM >= 9).
        """
        is_hopper_or_newer = torch.cuda.get_device_properties(x.device).major >= 9
        num_bytes = x.element_size() * x.numel()
        return self.triton_nvls_kernels_allowed and is_hopper_or_newer and num_bytes % 16 == 0

    def _maybe_allocate_ag_buffers(
        self,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> dict:
        """
        Allocate a single symmetric memory buffer for all-gather outputs of
        routing_map, probs and hidden_states. Returns sliced views for each.

        Returns dict with:
        - "handle": symmetric memory handle (or None if unavailable)
        - "routing_map" / "routing_map_offset": raw byte view and byte offset
        - "probs" / "probs_offset": raw byte view and byte offset
        - "hidden_states" / "hidden_states_offset": raw byte view and byte offset
        """
        _NONE = {
            "handle": None,
            "routing_map": None, "routing_map_offset": 0,
            "probs": None, "probs_offset": 0,
            "hidden_states": None, "hidden_states_offset": 0,
        }

        local_tokens = probs.size(0)
        global_tokens = local_tokens * self.ep_size
        topk = probs.size(-1)
        hidden_dim = hidden_states.size(-1)

        result = get_global_symmetric_memory_buffer_ep().maybe_get_tensors([
            (global_tokens * topk, routing_map.dtype),
            (global_tokens * topk, probs.dtype),
            (global_tokens * hidden_dim, hidden_states.dtype),
        ])

        if result["handle"] is None:
            return _NONE

        (rm_buf, rm_off), (p_buf, p_off), (hs_buf, hs_off) = result["tensors"]
        return {
            "handle": result["handle"],
            "routing_map": rm_buf, "routing_map_offset": rm_off,
            "probs": p_buf, "probs_offset": p_off,
            "hidden_states": hs_buf, "hidden_states_offset": hs_off,
        }

    def _maybe_allocate_rs_buffer(self, x: torch.Tensor) -> dict:
        """
        Allocate symmetric memory buffer for reduce-scatter input.
        Input shape matches x (the unpermuted hidden states).
        """
        symm_mem_buffer = get_global_symmetric_memory_buffer_ep().maybe_get_tensor(
            list(x.size()), dtype=x.dtype
        )
        return symm_mem_buffer

    def token_dispatch(self, hidden_states, probs):
        """
        Gathers tokens from all EP ranks using AllGather.

        Uses latency-optimized NVLS multimem_all_gather for routing_map, probs and hidden_states
        on Hopper+ GPUs with BF16. Falls back to NCCL otherwise.
        """
        if self.ep_size == 1:
            return hidden_states, probs
        
        # Check NVLS eligibility
        nvls_eligible = self._check_nvls_eligibility(hidden_states) and self._check_nvls_eligibility(probs) and self._check_nvls_eligibility(self.routing_map)
        ag_buffers = None

        if nvls_eligible:
            ag_buffers = self._maybe_allocate_ag_buffers(self.routing_map, probs, hidden_states)

        can_use_nvls = nvls_eligible and ag_buffers["handle"] is not None

        if can_use_nvls:
            # Capture shapes for reshaping after all-gather
            # Output shape: [local_tokens * ep_size, dim]
            local_tokens = probs.size(0)
            global_tokens = local_tokens * self.ep_size
            topk = probs.size(1)
            hidden_dim = hidden_states.size(1)
            routing_map_dtype = self.routing_map.dtype
            probs_dtype = probs.dtype
            hidden_dtype = hidden_states.dtype

            # Fused NVLS all-gather: single kernel launch + single barrier for all 3 tensors
            multimem_all_gather_fused(
                ag_buffers["routing_map"].view(torch.bfloat16),
                self.routing_map.view(torch.bfloat16),
                ag_buffers["routing_map_offset"],
                ag_buffers["probs"].view(torch.bfloat16),
                probs.view(torch.bfloat16),
                ag_buffers["probs_offset"],
                ag_buffers["hidden_states"].view(torch.bfloat16),
                hidden_states.view(torch.bfloat16),
                ag_buffers["hidden_states_offset"],
                ag_buffers["handle"],
            )
            self.routing_map = ag_buffers["routing_map"].view(routing_map_dtype).view(global_tokens, topk)
            probs = ag_buffers["probs"].view(probs_dtype).view(global_tokens, topk)
            hidden_states = ag_buffers["hidden_states"].view(hidden_dtype).view(global_tokens, hidden_dim)
        else:
            # Fallback to NCCL for all tensors
            with torch.no_grad():
                self.routing_map = gather_from_sequence_parallel_region(
                    self.routing_map, group=self.tp_ep_group
                )
            probs = gather_from_sequence_parallel_region(probs, group=self.tp_ep_group)
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, group=self.tp_ep_group
            )

        return hidden_states, probs

        
    def dispatch_postprocess(self, hidden_states, probs):
        """Pass-through: returns unpermuted inputs and routing_map for InferenceGroupedMLP."""
        return hidden_states, self.routing_map, probs

    def combine_preprocess(self, expert_output):
        """Pass-through: InferenceGroupedMLP already produces unpermuted output."""
        return expert_output

    def token_combine(self, hidden_states):
        """
        Combines expert outputs using Reduce-Scatter.

        Uses latency-optimized NVLS multimem_reduce_scatter on Hopper+ GPUs with BF16
        when symmetric memory is available. Falls back to NCCL via superclass otherwise.

        Args:
            hidden_states: [global_tokens, hidden_dim] tensor to reduce-scatter

        Returns:
            [local_tokens, hidden_dim] tensor after reduce-scatter
        """
        if self.ep_size == 1:
            return hidden_states

        # Check NVLS eligibility and try to allocate symmetric memory
        nvls_eligible = self._check_nvls_eligibility(hidden_states)
        rs_buffer = None

        if nvls_eligible:
            rs_buffer = self._maybe_allocate_rs_buffer(hidden_states)

        can_use_nvls = nvls_eligible and rs_buffer["handle"] is not None

        if can_use_nvls:
            # Copy input to symmetric memory for reduce-scatter
            rs_buffer["tensor"].copy_(hidden_states)

            # Allocate output tensor
            output_shape = list(hidden_states.size())
            output_shape[0] = hidden_states.size(0) // self.ep_size
            output = torch.empty(
                output_shape, dtype=hidden_states.dtype, device=hidden_states.device
            )

            # Use latency-optimized NVLS reduce-scatter
            multimem_reduce_scatter(output, rs_buffer["tensor"], rs_buffer["handle"])
            return output
        else:
            # Fallback to NCCL            
            hidden_states = reduce_scatter_to_sequence_parallel_region(
                hidden_states, group=self.tp_ep_group
            )
            return hidden_states

