# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
CUDA-graph-compatible AllGather-V token dispatcher for inference.

This dispatcher replaces the FlashInfer-based InferenceCUDAGraphTokenDispatcher
with a Triton-based permutation pipeline that works with standard
torch.nn.functional.grouped_mm. It avoids any host-device synchronization:

  1. AllGather routing_map, probs, and hidden_states across EP ranks.
  2. Triton kernels compute per-expert token counts and aligned prefix-sum
     offsets entirely on-device (no cudaMemcpy D->H for token counts).
  3. Triton permute kernel groups tokens by local expert with alignment padding.
  4. Expert compute uses grouped_mm with GPU-resident offsets.
  5. Triton unpermute kernel scatters weighted expert outputs back.
  6. ReduceScatter combines contributions across EP ranks.

When ``inference_moe_max_tokens`` is set (automatically derived from
``--inference-dynamic-batching-max-tokens``), AllGather/ReduceScatter buffers
are pinned to a fixed maximum size.  This makes the NCCL collectives identical
across every CUDA graph, so different EP ranks can independently select
different graphs without any cross-rank synchronization -- the all-reduce in
``adjust_batch_dims_for_expert_parallelism`` is no longer required.

Selected via ``--moe-token-dispatcher-type allgather_v``.

The "V" (variable) refers to the fact that each expert receives a data-dependent
number of tokens, computed on-device without cross-rank synchronization.
"""

from typing import List, Optional

import torch

from megatron.core.inference.communication.torch_symm_triton import (
    are_tensors_nvls_eligible,
    multimem_all_gather_fused,
    multimem_reduce_scatter,
)
from megatron.core.inference.moe.permute import permute_tokens, unpermute_tokens
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.moe.token_dispatcher import MoEAllGatherTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig


class InferenceAllGatherVTokenDispatcher(MoEAllGatherTokenDispatcher):
    """CUDA-graph-compatible AllGather-V token dispatcher with Triton permutation.

    Unlike the FlashInfer-based InferenceCUDAGraphTokenDispatcher, this dispatcher
    explicitly permutes tokens into expert-grouped order using Triton kernels,
    making it compatible with torch.nn.functional.grouped_mm / scaled_grouped_mm.

    Token counts per expert are computed on-device by Triton kernels -- no
    host-device synchronization is needed, so the full dispatch/combine pipeline
    is CUDA-graphable.

    When ``config.inference_moe_max_tokens`` is set and EP > 1, all
    AllGather/ReduceScatter buffers are pinned to a fixed maximum size so that
    the embedded NCCL collectives are identical across every CUDA graph.  This
    allows different EP ranks to capture and replay **different** graphs
    independently -- no cross-rank batch-dimension synchronization is required.

    Key features:
    - AllGather/ReduceScatter for EP communication (CUDA-graph safe)
    - NVLS collectives on Hopper+ with automatic NCCL fallback
    - Triton-based permute/unpermute (no FlashInfer dependency)
    - GPU-resident token counts and expert offsets (no D->H sync)
    - Fixed-max-buffer mode eliminates EP rank synchronization
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
        self.local_expert_start = local_expert_indices[0]
        self.triton_nvls_kernels_allowed = not self.config.inference_disable_triton_nvls_kernels

        # Alignment for grouped_mm / scaled_grouped_mm.
        # MXFP8 swizzle requires 128; BF16 grouped_mm requires 16.
        self._expert_alignment = 128 if config.fp8_recipe == "mxfp8" else 16

        # Fixed-max-buffer mode: pin AllGather/ReduceScatter to this per-rank
        # token count so every CUDA graph embeds the same collective.
        self._max_tokens_per_rank: Optional[int] = config.inference_moe_max_tokens
        if self.ep_size > 1 and self._max_tokens_per_rank is None:
            raise ValueError(
                "inference_moe_max_tokens must be set when using the 'allgather_v' "
                "dispatcher with EP > 1.  It is automatically derived from "
                "--inference-dynamic-batching-max-tokens (or max_requests * "
                "(num_speculative_tokens + 1)).  Make sure one of these is set."
            )

        # Cached between dispatch_postprocess and combine_preprocess.
        self.expert_offsets: Optional[torch.Tensor] = None
        self.permutation_map: Optional[torch.Tensor] = None
        self._permuted_probs: Optional[torch.Tensor] = None
        self._num_global_tokens: int = 0
        self._actual_local_tokens: int = 0

    # ------------------------------------------------------------------
    # Padding helpers for fixed-max-buffer mode
    # ------------------------------------------------------------------

    def _pad_to_max(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
    ):
        """Embed actual tokens into fixed-max-size buffers.

        Creates tensors of size [max_tokens_per_rank, ...] and copies actual
        data into the leading rows.  Padding rows get zeros for hidden_states
        and probs, and -1 for routing_map (so Triton permute skips them).

        All output sizes are fixed across CUDA graph replays.
        """
        max_tokens = self._max_tokens_per_rank
        actual = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        topk = probs.shape[1]
        device = hidden_states.device

        padded_hidden = torch.zeros(
            max_tokens, hidden_dim, dtype=hidden_states.dtype, device=device
        )
        padded_hidden[:actual] = hidden_states

        padded_routing_map = torch.full(
            (max_tokens, topk), -1, dtype=routing_map.dtype, device=device
        )
        padded_routing_map[:actual] = routing_map

        padded_probs = torch.zeros(
            max_tokens, topk, dtype=probs.dtype, device=device
        )
        padded_probs[:actual] = probs

        return padded_hidden, padded_routing_map, padded_probs

    # ------------------------------------------------------------------
    # AllGather helpers (symmetric memory / NVLS)
    # ------------------------------------------------------------------

    def _maybe_allocate_ag_buffers(
        self, routing_map: torch.Tensor, probs: torch.Tensor, hidden_states: torch.Tensor
    ) -> dict:
        """Allocate a single symmetric memory output buffer for fused all-gather.

        Returns sliced views for routing_map, probs, and hidden_states, or all-None
        when symmetric memory is unavailable.
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
        """Allocate a symmetric memory buffer for reduce-scatter input."""
        return SymmetricMemoryManager.get_buffer(
            "ep", process_group=self.ep_group
        ).maybe_get_tensor(list(x.size()), dtype=x.dtype)

    # ------------------------------------------------------------------
    # Dispatch: pad -> AllGather -> Triton permute
    # ------------------------------------------------------------------

    def dispatch_preprocess(self, hidden_states, routing_map, probs):
        """Cache routing_map, reshape, and optionally pad to fixed max size.

        When ``_max_tokens_per_rank`` is set and EP > 1, embeds the actual
        tokens into fixed-max-size buffers so that every CUDA graph produces
        the same AllGather buffer size.

        Overrides the base class to insert the padding step.
        """
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        self._actual_local_tokens = hidden_states.shape[0]

        if self._max_tokens_per_rank is not None and self.ep_size > 1:
            hidden_states, routing_map, probs = self._pad_to_max(
                hidden_states, routing_map, probs
            )

        self.routing_map = routing_map
        return hidden_states, probs

    def token_dispatch(self, hidden_states, probs):
        """Gather tokens from all EP ranks using AllGather.

        After ``dispatch_preprocess`` padding, every rank sends exactly
        ``max_tokens_per_rank`` tokens (if configured).  The resulting
        AllGather is the same size across all CUDA graphs, allowing
        different EP ranks to replay different graphs independently.

        Uses fused NVLS multimem_all_gather on Hopper+ GPUs when available,
        with NCCL fallback.

        Args:
            hidden_states: [tokens_per_rank, hidden_dim] (may be padded).
            probs: [tokens_per_rank, topk] (may be padded).

        Returns:
            (hidden_states, probs) gathered across all EP ranks.
            Also updates self.routing_map in-place to the gathered shape.
        """
        if self.ep_size == 1:
            return hidden_states, probs

        nvls_eligible = self.triton_nvls_kernels_allowed and are_tensors_nvls_eligible(
            hidden_states, probs, self.routing_map
        )
        ag_buffers = None

        if nvls_eligible:
            ag_buffers = self._maybe_allocate_ag_buffers(self.routing_map, probs, hidden_states)

        can_use_nvls = nvls_eligible and ag_buffers["handle"] is not None

        if can_use_nvls:
            local_tokens = probs.size(0)
            global_tokens = local_tokens * self.ep_size
            topk = probs.size(1)
            hidden_dim = hidden_states.size(1)
            routing_map_dtype = self.routing_map.dtype
            probs_dtype = probs.dtype
            hidden_dtype = hidden_states.dtype

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
            self.routing_map = (
                ag_buffers["routing_map"].view(routing_map_dtype).view(global_tokens, topk)
            )
            probs = ag_buffers["probs"].view(probs_dtype).view(global_tokens, topk)
            hidden_states = (
                ag_buffers["hidden_states"].view(hidden_dtype).view(global_tokens, hidden_dim)
            )
        else:
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
        """Permute gathered tokens into expert-grouped order using Triton.

        Uses on-device Triton kernels to:
          1. Count tokens per local expert (atomic histogram).
          2. Compute aligned prefix-sum offsets for grouped_mm.
          3. Permute tokens + probs into expert-contiguous layout with
             alignment padding.

        No host-device synchronization occurs -- all metadata stays GPU-resident.
        Padding tokens (routing_map == -1) are automatically skipped by the
        Triton permute kernel.

        Args:
            hidden_states: [global_tokens, hidden_dim] gathered hidden states.
            probs: [global_tokens, topk] gathered routing probabilities.

        Returns:
            (permuted_hidden, tokens_per_expert, permuted_probs)
            - permuted_hidden: [output_size, hidden_dim] expert-grouped tokens.
            - tokens_per_expert: None (offsets are stored on self.expert_offsets
              instead -- grouped_mm uses those directly).
            - permuted_probs: [output_size] flat routing probabilities matching
              the permuted token order.
        """
        self._num_global_tokens = hidden_states.shape[0]

        permuted_hidden, permuted_probs, permutation_map, inclusive_offsets = permute_tokens(
            hidden_states,
            probs,
            self.routing_map,
            self.local_expert_start,
            self.num_local_experts,
            alignment=self._expert_alignment,
        )

        # Cache for combine_preprocess (Triton unpermute) and expert compute.
        self.expert_offsets = inclusive_offsets
        self.permutation_map = permutation_map
        self._permuted_probs = permuted_probs

        # tokens_per_expert = None: the expert uses self.expert_offsets directly.
        return permuted_hidden, None, permuted_probs

    # ------------------------------------------------------------------
    # Combine: Triton unpermute -> ReduceScatter -> unpad
    # ------------------------------------------------------------------

    def combine_preprocess(self, expert_output):
        """Scatter weighted expert outputs back to original token positions.

        Uses the Triton unpermute kernel which performs weighted (by routing
        probability) atomic scatter-add in fp32, then casts to bf16 for
        the subsequent ReduceScatter.

        Args:
            expert_output: [output_size, hidden_dim] raw FC2 output in
                expert-grouped order (no probability weighting applied yet).

        Returns:
            [global_tokens, hidden_dim] bf16 tensor with each token's output
            equal to the sum of its weighted expert contributions on this rank.
        """
        output = unpermute_tokens(
            expert_output,
            self._permuted_probs,
            self.permutation_map,
            self._num_global_tokens,
        )
        return output.to(torch.bfloat16)

    def token_combine(self, hidden_states):
        """Reduce-scatter expert outputs back to local token slices.

        Sums contributions across EP ranks (each rank contributes non-zero
        values only for tokens routed to its local experts) and scatters
        the result so each rank receives its local portion.

        Uses NVLS multimem_reduce_scatter on Hopper+ when available.

        Args:
            hidden_states: [global_tokens, hidden_dim] combined expert output.

        Returns:
            [tokens_per_rank, hidden_dim] bf16 (may include max-buffer padding).
        """
        if self.ep_size == 1:
            return hidden_states

        output_shape = list(hidden_states.size())
        output_shape[0] = hidden_states.size(0) // self.ep_size
        output = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)

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
            rs_buffer["tensor"].copy_(hidden_states)
            multimem_reduce_scatter(output, rs_buffer["tensor"], rs_buffer["handle"])
            return output.to(torch.bfloat16)
        else:
            hidden_states = reduce_scatter_to_sequence_parallel_region(
                hidden_states, group=self.tp_ep_group
            )
            return hidden_states.to(torch.bfloat16)

    def combine_postprocess(self, hidden_states):
        """Strip max-buffer padding and restore original tensor shape.

        When fixed-max-buffer mode is active, the ReduceScatter output has
        ``max_tokens_per_rank`` rows.  This method slices back to the actual
        local token count, then reshapes to the original ``[S/TP, B, H]``.
        """
        if self._max_tokens_per_rank is not None and self.ep_size > 1:
            hidden_states = hidden_states[: self._actual_local_tokens]
        return hidden_states.view(self.hidden_shape)
