# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
CUDA-graph-compatible token dispatcher for inference.

This dispatcher is only used during CUDA-graphed inference iterations. It replaces
AlltoAll with AllGather/ReduceScatter for token exchange, keeping all metadata
GPU-resident to avoid host synchronizations that would break CUDA graph capture.

Supports latency-optimized NVLS collectives (multimem all-gather/reduce-scatter)
on Hopper+ GPUs with BF16, with automatic fallback to NCCL.
"""

from typing import List, Optional

import torch
import torch.distributed as dist

from megatron.core.inference.communication.torch_symm_triton import (
    are_tensors_nvls_eligible,
    multimem_all_gather_fused,
    multimem_reduce_scatter,
    multimem_all_gather_v,
    multimem_reduce_scatter_v,
)
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.moe.token_dispatcher import MoEAllGatherTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig


class InferenceCUDAGraphTokenDispatcher(MoEAllGatherTokenDispatcher):
    """
    CUDA-graph-compatible AllGather token dispatcher for inference.

    Only used during CUDA-graphed inference iterations. Swapped in by
    MoELayer.set_inference_dispatcher() before graph capture
    and swapped out by MoELayer.unset_inference_dispatcher() after.

    Key features:
    - AllGather/ReduceScatter instead of AlltoAll for CUDA graph compatibility
    - GPU-resident metadata (no host synchronization)
    - NVLS collectives on Hopper+ with automatic NCCL fallback
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """
        Initialize the InferenceCUDAGraphTokenDispatcher.

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

    def _maybe_allocate_ag_buffers(
        self, routing_map: torch.Tensor, probs: torch.Tensor, hidden_states: torch.Tensor
    ) -> dict:
        """Allocate a single symmetric memory output buffer for fused all-gather.

        Creates one contiguous symmetric memory buffer sized for the gathered
        (global) routing_map, probs, and hidden_states, then returns sliced views
        into it. This allows a single fused NVLS all-gather kernel to write all
        three outputs in one launch.

        Args:
            routing_map (torch.Tensor): Local routing map, shape [local_tokens, topk].
                Boolean or integer tensor mapping each token to its selected experts.
            probs (torch.Tensor): Local routing probabilities, shape [local_tokens, topk].
                Normalized weights for each token's selected experts.
            hidden_states (torch.Tensor): Local hidden states, shape [local_tokens, hidden_dim].

        Returns:
            dict: A dictionary with the following keys:
                - "handle": Symmetric memory handle for NVLS ops, or None if
                  symmetric memory is unavailable.
                - "routing_map": Raw byte view for the gathered routing map output.
                - "routing_map_offset": Byte offset of routing_map within the buffer.
                - "probs": Raw byte view for the gathered probs output.
                - "probs_offset": Byte offset of probs within the buffer.
                - "hidden_states": Raw byte view for the gathered hidden states output.
                - "hidden_states_offset": Byte offset of hidden_states within the buffer.
                When allocation fails, all tensor views are None and offsets are 0.
        """
        _NONE = {
            "handle": None,
            "routing_map": None,
            "routing_map_offset": 0,
            "probs": None,
            "probs_offset": 0,
            "hidden_states": None,
            "hidden_states_offset": 0,
        }

        local_tokens = probs.size(0)
        global_tokens = local_tokens * self.ep_size
        topk = probs.size(-1)
        hidden_dim = hidden_states.size(-1)

        result = SymmetricMemoryManager.get_buffer(
            "ep", process_group=self.ep_group
        ).maybe_get_tensors(
            [
                (global_tokens * topk, routing_map.dtype),
                (global_tokens * topk, probs.dtype),
                (global_tokens * hidden_dim, hidden_states.dtype),
            ]
        )

        if result["handle"] is None:
            return _NONE

        (rm_buf, rm_off), (p_buf, p_off), (hs_buf, hs_off) = result["tensors"]
        return {
            "handle": result["handle"],
            "routing_map": rm_buf,
            "routing_map_offset": rm_off,
            "probs": p_buf,
            "probs_offset": p_off,
            "hidden_states": hs_buf,
            "hidden_states_offset": hs_off,
        }

    def _maybe_allocate_rs_buffer(self, x: torch.Tensor) -> dict:
        """Allocate a symmetric memory buffer for reduce-scatter input.

        The buffer has the same shape and dtype as x so that x can be copied
        into it before the NVLS reduce-scatter kernel.

        Args:
            x (torch.Tensor): The global hidden states to be reduce-scattered,
                shape [global_tokens, hidden_dim].

        Returns:
            dict: A dictionary with keys "handle" (symmetric memory handle, or
                None if unavailable) and "tensor" (the allocated buffer, or None).
        """
        symm_mem_buffer = SymmetricMemoryManager.get_buffer(
            "ep", process_group=self.ep_group
        ).maybe_get_tensor(list(x.size()), dtype=x.dtype)
        return symm_mem_buffer

    def token_dispatch(self, hidden_states, probs):
        """Gathers tokens from all EP ranks using AllGather.

        Performs all-gather on routing_map (stored in self.routing_map), probs,
        and hidden_states so that every rank holds the full global view.
        Uses latency-optimized fused NVLS multimem_all_gather on Hopper+ GPUs
        with BF16 when symmetric memory is available. Falls back to NCCL otherwise.

        Args:
            hidden_states (torch.Tensor): Local hidden states,
                shape [local_tokens, hidden_dim].
            probs (torch.Tensor): Local routing probabilities,
                shape [local_tokens, topk]. Normalized weights for each token's
                selected experts.

        Returns:
            tuple: (hidden_states, probs) gathered across all EP ranks.
                - hidden_states (torch.Tensor): Shape [global_tokens, hidden_dim].
                - probs (torch.Tensor): Shape [global_tokens, topk].
                Also updates self.routing_map in-place to the gathered
                shape [global_tokens, topk].
        """
        if self.ep_size == 1:
            return hidden_states, probs

        # 1. Check inputs only: if inputs are 16-byte divisible,
        #  outputs (world_size * input) are too.
        nvls_eligible = self.triton_nvls_kernels_allowed and are_tensors_nvls_eligible(
            hidden_states, probs, self.routing_map
        )
        ag_buffers = None

        if nvls_eligible:
            # 2. Now attempt to allocate symmetric memory buffers for
            # all-gather outputs. If allocation fails, fallback to NCCL.
            ag_buffers = self._maybe_allocate_ag_buffers(self.routing_map, probs, hidden_states)

        # 3. Can use NVLS if eligible and buffers allocated successfully (handle is not None)
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
                ag_buffers["routing_map"].view(
                    torch.bfloat16
                ),  # .view does not change the underlying data
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
            self.routing_map = (
                ag_buffers["routing_map"].view(routing_map_dtype).view(global_tokens, topk)
            )
            probs = ag_buffers["probs"].view(probs_dtype).view(global_tokens, topk)
            hidden_states = (
                ag_buffers["hidden_states"].view(hidden_dtype).view(global_tokens, hidden_dim)
            )
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
        """Pass-through: returns inputs directly without permutation.

        Unlike the training dispatcher, this does not permute tokens or compute
        tokens_per_expert. The downstream InferenceGroupedMLP (FlashInfer /
        CUTLASS fused MoE kernel) operates directly on the routing map stored
        in self.routing_map.

        Args:
            hidden_states (torch.Tensor): Gathered hidden states,
                shape [global_tokens, hidden_dim].
            probs (torch.Tensor): Gathered routing probabilities,
                shape [global_tokens, topk].

        Returns:
            tuple: (hidden_states, tokens_per_expert, probs) where
                tokens_per_expert is always None.
        """
        return hidden_states, None, probs

    def combine_preprocess(self, expert_output):
        """Pass-through: InferenceGroupedMLP already produces unpermuted output.

        No unpermutation is needed because dispatch_postprocess did not permute
        the tokens in the first place.

        Args:
            expert_output (torch.Tensor): Output from InferenceGroupedMLP,
                shape [global_tokens, hidden_dim].

        Returns:
            torch.Tensor: The input tensor unchanged.
        """
        return expert_output

    def token_combine(self, hidden_states):
        """Combines expert outputs across EP ranks using Reduce-Scatter.

        Reduces the global expert output (summing contributions from each rank)
        and scatters the result so each rank receives its local token slice.
        Uses latency-optimized NVLS multimem_reduce_scatter on Hopper+ GPUs
        with BF16 when symmetric memory is available. Falls back to NCCL otherwise.

        Args:
            hidden_states (torch.Tensor): Combined expert output after routing
                weights have been applied, shape [global_tokens, hidden_dim].

        Returns:
            torch.Tensor: Local slice of the reduced output,
                shape [local_tokens, hidden_dim] where
                local_tokens = global_tokens // ep_size.
        """
        if self.ep_size == 1:
            return hidden_states

        # Compute output shape first — check NVLS eligibility on the output,
        # since if the smaller output is 16-byte divisible, the input is too.
        output_shape = list(hidden_states.size())
        output_shape[0] = hidden_states.size(0) // self.ep_size
        output = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)

        # Check output only: if output is 16-byte divisible, input (world_size * output) is too.
        nvls_eligible = (
            self.triton_nvls_kernels_allowed
            and output.dtype in (torch.bfloat16, torch.float32)
            and are_tensors_nvls_eligible(output)
        )
        rs_buffer = None

        if nvls_eligible:
            rs_buffer = self._maybe_allocate_rs_buffer(hidden_states)

        can_use_nvls = nvls_eligible and rs_buffer["handle"] is not None

        if can_use_nvls:
            # Copy input to symmetric memory for reduce-scatter
            rs_buffer["tensor"].copy_(hidden_states)

            # Use latency-optimized NVLS reduce-scatter
            multimem_reduce_scatter(output, rs_buffer["tensor"], rs_buffer["handle"])
            return output.to(torch.bfloat16)
        else:
            # Fallback to NCCL
            hidden_states = reduce_scatter_to_sequence_parallel_region(
                hidden_states, group=self.tp_ep_group
            )
            return hidden_states.to(torch.bfloat16)


class MoEAllGatherVTokenDispatcher(MoEAllGatherTokenDispatcher):
    """Variable-count AllGather-V / ReduceScatter-V dispatcher for inference CUDA graphs.

    Replaces the fixed AllGather/ReduceScatter of InferenceCUDAGraphTokenDispatcher
    with variable-count NVLS collectives so ranks can hold different token counts
    per step. All metadata lives on-device; no host sync is needed between steps.

    Engine responsibilities (call once at model init):
        MoEAllGatherVTokenDispatcher.set_engine_max_tokens(n)
        MoEAllGatherVTokenDispatcher.allocate_step_metadata(device)

    Engine responsibilities (call before each step):
        MoEAllGatherVTokenDispatcher.set_step_metadata(metadata)  # [3] int32 tensor
    """

    # ── Class-level step metadata (shared across all layers, set by engine) ──────
    # Single [3] int32 tensor: [valid_tokens, rank_token_offset, ep_max_tokens].
    # Sliced views are passed to AGV/RSV kernels as scalar pointer arguments.
    _step_metadata: Optional[torch.Tensor] = None  # [3] int32
    _ep_rank: int = 0
    _engine_max_tokens: int = 2048

    # Cached RSV symmetric buffer tensor — set on first _alloc_rsv_buffer call.
    # Exposed via _get_rsv_tensor() so mcore_fused_moe can write into it directly,
    # eliminating the copy before multimem_reduce_scatter_v.
    _rsv_symm_tensor: Optional[torch.Tensor] = None

    @classmethod
    def _get_rsv_tensor(cls) -> Optional[torch.Tensor]:
        """Return the RSV symmetric buffer tensor, or None if not yet allocated."""
        return cls._rsv_symm_tensor

    # Convenience views into _step_metadata — valid after allocate_step_metadata().
    @classmethod
    def _valid_tokens(cls) -> torch.Tensor:
        return cls._step_metadata[0:1]

    @classmethod
    def _rank_token_offset(cls) -> torch.Tensor:
        return cls._step_metadata[1:2]

    @classmethod
    def _ep_max_tokens(cls) -> torch.Tensor:
        return cls._step_metadata[2:3]

    @staticmethod
    def set_engine_max_tokens(n: int) -> None:
        """Set at model init. Determines AGV/RSV CTA grid size (fixed for CG)."""
        MoEAllGatherVTokenDispatcher._engine_max_tokens = n

    @staticmethod
    def allocate_step_metadata(device: torch.device, ep_rank: int) -> None:
        """Allocate a [3] int32 CUDA tensor for per-step metadata at model init.

        Stores [valid_tokens, rank_token_offset, ep_max_tokens] in a single
        contiguous buffer at a fixed address. The engine updates values in-place
        each step via set_step_metadata so CUDA graph replay sees stable addresses.

        Args:
            device: CUDA device on which to allocate (e.g. torch.device("cuda")).
            ep_rank: This rank's position in the EP group. Used by set_step_metadata
                to extract rank_token_offset from the allgathered counts.
        """
        MoEAllGatherVTokenDispatcher._ep_rank = ep_rank
        MoEAllGatherVTokenDispatcher._step_metadata = torch.zeros(3, dtype=torch.int32, device=device)

    @staticmethod
    def set_step_metadata(
        local_tokens_per_rank: torch.Tensor,
        ep_group: torch.distributed.ProcessGroup,
    ) -> None:
        """Compute and store step metadata from allgathered per-rank token counts.

        Computes valid_tokens (sum), rank_token_offset (exclusive prefix sum at
        ep_rank), and ep_max_tokens (max), then writes all three into the
        pre-allocated [3] buffer in a single copy so addresses remain stable
        for CUDA graph replay.

        Lazily calls allocate_step_metadata on the first invocation if the buffer
        has not been explicitly pre-allocated.

        Args:
            local_tokens_per_rank: [ep_size] int32 tensor of padded token counts
                per EP rank, produced by an all-gather before each step.
            ep_group: The expert parallel process group, used to determine ep_rank
                on the first call when lazy allocation is needed.
        """
        if MoEAllGatherVTokenDispatcher._step_metadata is None:
            MoEAllGatherVTokenDispatcher.allocate_step_metadata(
                device=local_tokens_per_rank.device,
                ep_rank=dist.get_rank(group=ep_group),
            )
        # Barrier ensures all EP ranks have completed initialize_attention_state
        # before any rank enters the model forward and the AGV kernel barrier.
        dist.barrier(group=ep_group)
        ep_rank = MoEAllGatherVTokenDispatcher._ep_rank
        valid_tokens = local_tokens_per_rank.sum()
        rank_token_offset = local_tokens_per_rank[:ep_rank].sum()
        ep_max_tokens = local_tokens_per_rank.max()
        MoEAllGatherVTokenDispatcher._step_metadata.copy_(
            torch.stack([valid_tokens, rank_token_offset, ep_max_tokens])
        )

    # ── Init ─────────────────────────────────────────────────────────────────────

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
        # Set in dispatch_preprocess; consumed by token_dispatch and token_combine.
        self._local_tokens: int = 0
        self.routing_map_dtype = None

    # ── Buffer allocation helpers ─────────────────────────────────────────────────

    def _alloc_agv_buffers(
        self, hidden_states: torch.Tensor, probs: torch.Tensor
    ) -> tuple:
        """Allocate separate symmetric buffers for the three AGV outputs.

        Buffers are sized at engine_max_tokens * ep_size (fixed across steps)
        so the same allocations can be reused every CUDA graph replay.

        Returns:
            (hidden_r, routing_r, probs_r) — each a dict with "handle" and "tensor",
            or "handle"=None if symmetric memory is unavailable.
        """
        global_max = self._engine_max_tokens * self.ep_size
        topk = probs.shape[1]
        hidden_dim = hidden_states.shape[1]

        hidden_r = SymmetricMemoryManager.get_buffer(
            "ep_agv_h", process_group=self.ep_group
        ).maybe_get_tensor([global_max, hidden_dim], dtype=hidden_states.dtype)

        routing_r = SymmetricMemoryManager.get_buffer(
            "ep_agv_r", process_group=self.ep_group
        ).maybe_get_tensor([global_max, topk], dtype=self.routing_map.dtype)

        probs_r = SymmetricMemoryManager.get_buffer(
            "ep_agv_p", process_group=self.ep_group
        ).maybe_get_tensor([global_max, topk], dtype=probs.dtype)

        return hidden_r, routing_r, probs_r

    def _alloc_rsv_buffer(self, hidden_states: torch.Tensor) -> dict:
        """Allocate a symmetric buffer for the RSV input (expert outputs).

        Sized at [engine_max_tokens * ep_size, hidden_dim] — fixed for CG.
        Caches the tensor as a class attribute so mcore_fused_moe can write
        into it directly, avoiding a copy before multimem_reduce_scatter_v.

        Returns:
            dict with "handle" and "tensor", or "handle"=None if unavailable.
        """
        global_max = self._engine_max_tokens * self.ep_size
        hidden_dim = hidden_states.shape[1]
        buf = SymmetricMemoryManager.get_buffer(
            "ep_rsv", process_group=self.ep_group
        ).maybe_get_tensor([global_max, hidden_dim], dtype=hidden_states.dtype)
        MoEAllGatherVTokenDispatcher._rsv_symm_tensor = buf["tensor"]
        return buf

    # ── Dispatch path ─────────────────────────────────────────────────────────────

    def dispatch_preprocess(self, hidden_states, routing_map, probs):
        """Store routing map and local token count; no communication."""
        self.hidden_shape = hidden_states.shape
        # [S/TP, B, H] -> [S*B/TP, H]
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        self._local_tokens = hidden_states.shape[0]
        self.routing_map = routing_map
        self.routing_map_dtype = routing_map.dtype
        return hidden_states, probs

    def token_dispatch(self, hidden_states, probs):
        """AllGather-V: gather hidden_states, probs, and routing_map from all EP ranks.

        Uses NVLS multimem_all_gather_v on Hopper+ when symmetric memory is
        available; falls back to NCCL otherwise. Global buffers are sized at
        engine_max_tokens * ep_size so allocations are fixed across steps.

        Args:
            hidden_states: [local_tokens, hidden_dim] local input.
            probs: [local_tokens, topk] local routing probabilities.

        Returns:
            (hidden_states, probs) gathered to [global_max, *] shape.
            Also updates self.routing_map to [global_max, topk].
        """
        if self.ep_size == 1:
            return hidden_states, probs

        hidden_r, routing_r, probs_r = self._alloc_agv_buffers(hidden_states, probs)
        assert all(r["handle"] is not None for r in (hidden_r, routing_r, probs_r)), (
            "MoEAllGatherVTokenDispatcher requires NVLS symmetric memory for AGV. "
            "Ensure the device is Hopper+ with NVLink and symmetric memory is available."
        )

        engine_max = self._engine_max_tokens
        global_max = engine_max * self.ep_size
        topk = probs.shape[1]
        hidden_dim = hidden_states.shape[1]

        rank_token_offset = self._rank_token_offset()
        ep_max_tokens = self._ep_max_tokens()
        multimem_all_gather_v(
            hidden_r["tensor"], hidden_states, hidden_r["handle"],
            rank_token_offset=rank_token_offset,
            ep_max_tokens=ep_max_tokens,
            engine_max_tokens=engine_max,
        )
        multimem_all_gather_v(
            routing_r["tensor"], self.routing_map, routing_r["handle"],
            rank_token_offset=rank_token_offset,
            ep_max_tokens=ep_max_tokens,
            engine_max_tokens=engine_max,
        )
        multimem_all_gather_v(
            probs_r["tensor"], probs, probs_r["handle"],
            rank_token_offset=rank_token_offset,
            ep_max_tokens=ep_max_tokens,
            engine_max_tokens=engine_max,
        )

        self.routing_map = (
            routing_r["tensor"].view(self.routing_map_dtype).view(global_max, topk)
        )
        probs = probs_r["tensor"].view(global_max, topk)
        hidden_states = hidden_r["tensor"].view(global_max, hidden_dim)
        return hidden_states, probs

    def dispatch_postprocess(self, hidden_states, probs):
        """Pass-through: mcore_fused_moe operates directly on the gathered tensors."""
        return hidden_states, None, probs

    # ── Combine path ──────────────────────────────────────────────────────────────

    def combine_preprocess(self, expert_output):
        """Pass-through: unpermute is handled inside mcore_fused_moe."""
        return expert_output

    def token_combine(self, hidden_states):
        """ReduceScatter-V: sum expert outputs across EP ranks, scatter to local tokens.

        hidden_states is the fp32 output of mcore_fused_moe [global_max, hidden_dim].

        Args:
            hidden_states: [global_max, hidden_dim] fp32 expert outputs.

        Returns:
            [local_tokens, hidden_dim] bf16 local token outputs.
        """
        if self.ep_size == 1:
            return hidden_states.to(torch.bfloat16)

        rs_buffer = self._alloc_rsv_buffer(hidden_states)
        assert rs_buffer["handle"] is not None, (
            "MoEAllGatherVTokenDispatcher requires NVLS symmetric memory for RSV. "
            "Ensure the device is Hopper+ with NVLink and symmetric memory is available."
        )

        # If mcore_fused_moe wrote unpermute output directly into the symm buffer
        # (via _get_rsv_tensor()), skip the copy. Otherwise copy now (first-step fallback).
        if hidden_states is not rs_buffer["tensor"]:
            rs_buffer["tensor"].copy_(hidden_states)
        output = torch.empty(
            self._local_tokens, hidden_states.shape[1],
            dtype=hidden_states.dtype, device=hidden_states.device,
        )
        multimem_reduce_scatter_v(
            output, rs_buffer["tensor"], rs_buffer["handle"],
            rank_token_offset=self._rank_token_offset(),
            ep_max_tokens=self._ep_max_tokens(),
            engine_max_tokens=self._engine_max_tokens,
        )
        return output.to(torch.bfloat16)

    def combine_postprocess(self, hidden_states):
        """Restore original input shape (e.g. [S/TP, B, H] from [S*B/TP, H])."""
        return hidden_states.view(self.hidden_shape)
