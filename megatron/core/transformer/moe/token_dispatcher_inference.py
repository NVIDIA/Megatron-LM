# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Inference-optimized AlltoAll Token Dispatcher with GPU-resident metadata.

This implementation keeps tokens_per_expert GPU-resident to enable use of
torch._grouped_mm without host synchronization.

Supports latency-optimized NVLS collectives (multimem all-gather/reduce-scatter)
on Hopper+ GPUs with BF16, with automatic fallback to NCCL via superclass methods.
"""

import torch
from typing import List, Optional

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.core.transformer.moe.inference_kernels import (
    launch_fused_permute_and_probs,
    launch_unpermute_kernel,
)
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.parallel_state import get_global_symmetric_memory_buffer_ep
from megatron.core.inference.communication.torch_symm_triton import (
    multimem_all_gather,
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

        # Cache for NVLS eligibility
        self._nvls_eligible = None

    def _check_nvls_eligibility(self, x: torch.Tensor) -> bool:
        """
        Check if we can use NVLS (latency-optimized) collectives.
        Requirements: BF16 dtype, Hopper+ GPU (SM >= 9).
        """
        is_bf16 = x.dtype == torch.bfloat16
        is_hopper_or_newer = torch.cuda.get_device_properties(x.device).major >= 9
        return is_bf16 and is_hopper_or_newer

    def _maybe_allocate_ag_buffers(
        self,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> dict:
        """
        Allocate a single symmetric memory buffer for all-gather outputs of
        routing_map, probs and hidden_states. Returns sliced views for each.

        All tensors are gathered from ep_size ranks, so output shapes are:
        - routing_map: [local_tokens * ep_size, num_experts]
        - probs: [local_tokens * ep_size, num_experts]
        - hidden_states: [local_tokens * ep_size, hidden_dim]

        Returns dict with:
        - "handle": symmetric memory handle (or None if unavailable)
        - "routing_map": view for routing_map output
        - "routing_map_offset": byte offset of routing_map in the symmetric buffer
        - "probs": view for probs output
        - "probs_offset": byte offset of probs in the symmetric buffer
        - "hidden_states": view for hidden_states output
        - "hidden_states_offset": byte offset of hidden_states in the symmetric buffer
        """
        symm_buffer_mgr = get_global_symmetric_memory_buffer_ep()
        if symm_buffer_mgr.symm_mem_hdl is None:
            return {
                "handle": None,
                "routing_map": None, "routing_map_offset": 0,
                "probs": None, "probs_offset": 0,
                "hidden_states": None, "hidden_states_offset": 0,
            }

        # Calculate output shapes after all-gather
        local_tokens = probs.size(0)
        global_tokens = local_tokens * self.ep_size
        num_experts = probs.size(1)
        hidden_dim = hidden_states.size(1)

        # Calculate bytes needed for each tensor (with 16-byte alignment)
        def aligned_bytes(numel, dtype):
            elem_size = torch.tensor([], dtype=dtype).element_size()
            raw_bytes = numel * elem_size
            # Align to 16 bytes for 128-bit access
            return ((raw_bytes + 15) // 16) * 16

        routing_map_bytes = aligned_bytes(global_tokens * num_experts, routing_map.dtype)
        probs_bytes = aligned_bytes(global_tokens * num_experts, probs.dtype)
        hidden_states_bytes = aligned_bytes(global_tokens * hidden_dim, hidden_states.dtype)
        total_bytes = routing_map_bytes + probs_bytes + hidden_states_bytes

        # Check if buffer has enough space
        if total_bytes > symm_buffer_mgr.symm_buffer.numel():
            return {
                "handle": None,
                "routing_map": None, "routing_map_offset": 0,
                "probs": None, "probs_offset": 0,
                "hidden_states": None, "hidden_states_offset": 0,
            }

        # Slice the raw buffer and create views, tracking byte offsets
        # [routing_map_bytes | probs_bytes | hidden_states_bytes]
        #  offset=0            offset=rm       offset=rm+probs

        raw_buffer = symm_buffer_mgr.symm_buffer

        routing_map_offset = 0
        routing_map_buffer = raw_buffer[routing_map_offset : routing_map_offset + routing_map_bytes]

        probs_offset = routing_map_bytes
        probs_buffer = raw_buffer[probs_offset : probs_offset + probs_bytes]

        hidden_states_offset = probs_offset + probs_bytes
        hidden_states_buffer = raw_buffer[hidden_states_offset : hidden_states_offset + hidden_states_bytes]

        return {
            "handle": symm_buffer_mgr.symm_mem_hdl,
            "routing_map": routing_map_buffer,
            "routing_map_offset": routing_map_offset,
            "probs": probs_buffer,
            "probs_offset": probs_offset,
            "hidden_states": hidden_states_buffer,
            "hidden_states_offset": hidden_states_offset,
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
        on Hopper+ GPUs with BF16. Falls back to NCCL via superclass otherwise.
        """
        if self.ep_size == 1:
            return hidden_states, probs

        # Check NVLS eligibility
        nvls_eligible = self._check_nvls_eligibility(hidden_states)
        ag_buffers = None

        if nvls_eligible:
            ag_buffers = self._maybe_allocate_ag_buffers(self.routing_map, probs, hidden_states)

        can_use_nvls = nvls_eligible and ag_buffers["handle"] is not None

        if can_use_nvls:
            # Capture shapes for reshaping after all-gather
            # Output shape: [local_tokens * ep_size, dim]
            local_tokens = probs.size(0)
            global_tokens = local_tokens * self.ep_size
            num_experts = probs.size(1)
            hidden_dim = hidden_states.size(1)
            routing_map_dtype = self.routing_map.dtype
            probs_dtype = probs.dtype
            hidden_dtype = hidden_states.dtype

            # Use latency-optimized NVLS all-gather for routing_map, probs and hidden_states
            # Pass byte_offset so kernel writes to correct location in multicast buffer
            multimem_all_gather(
                ag_buffers["routing_map"].view(torch.bfloat16),
                self.routing_map.view(torch.bfloat16),
                ag_buffers["handle"],
                byte_offset=ag_buffers["routing_map_offset"],
            )
            self.routing_map = ag_buffers["routing_map"].view(routing_map_dtype).view(global_tokens, num_experts)

            multimem_all_gather(
                ag_buffers["probs"].view(torch.bfloat16),
                probs.view(torch.bfloat16),
                ag_buffers["handle"],
                byte_offset=ag_buffers["probs_offset"],
            )
            probs = ag_buffers["probs"].view(probs_dtype).view(global_tokens, num_experts)

            multimem_all_gather(
                ag_buffers["hidden_states"].view(torch.bfloat16),
                hidden_states.view(torch.bfloat16),
                ag_buffers["handle"],
                byte_offset=ag_buffers["hidden_states_offset"],
            )
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
        # 1. Pre-allocate output buffer w/ zeros.
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
            # Fallback to NCCL via superclass
            return super().token_combine(hidden_states)

