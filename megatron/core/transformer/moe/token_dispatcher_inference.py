# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Inference token dispatchers for MoE expert parallelism.

Two dispatchers are provided, selected via config.inference_moe_token_dispatcher_type:

  NCCLAllGatherDispatcher ('nccl', default)
    Standard NCCL AllGather/ReduceScatter. All EP ranks must contribute the same
    token count per step; decode-only CUDA graphs are forced automatically.

  NVLSAllGatherVDispatcher ('nvls')
    Variable-count NVLS AllGather-V/ReduceScatter-V via multimem kernels. Supports
    different token counts per rank per step. Requires Hopper+ GPUs with NVLink and
    symmetric memory. Opt-in.

InferenceAllGatherDispatcherBase is a minimal base used solely for isinstance checks
and to hold _valid_tokens_tensor — the shared interface that mcore_fused_moe reads to
gate kernel work to the valid token prefix. Each dispatcher defines its own
set_step_metadata; the inference context calls the right one based on its dispatcher
type flag.
"""

from typing import List, Optional

import torch
import torch.distributed as dist

from megatron.core.inference.communication.torch_symm_triton import (
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


class InferenceAllGatherDispatcherBase(MoEAllGatherTokenDispatcher):
    """Minimal base for inference AllGather token dispatchers.

    Exists for isinstance checks and to expose _valid_tokens_tensor — the single
    class-level value that mcore_fused_moe reads (via experts.py) to gate kernel
    work to the valid token prefix. Each concrete subclass owns its own metadata
    and defines set_step_metadata independently.
    """

    # [1] int32: total valid tokens across all EP ranks this step.
    # Written in-place each step so CUDA graph replay sees a stable address.
    # NVLSAllGatherVDispatcher points this at _step_metadata[0:1] on first init
    # so that experts.py can always call _valid_tokens() on this base class.
    _valid_tokens_tensor: Optional[torch.Tensor] = None

    @classmethod
    def _valid_tokens(cls) -> torch.Tensor:
        return cls._valid_tokens_tensor


class NCCLAllGatherDispatcher(InferenceAllGatherDispatcherBase):
    """AllGather token dispatcher for inference using NCCL.

    Two modes selected per-step via set_step_metadata:

    CG path (use_allgather_v=False): all EP ranks contribute the same token count,
    guaranteed by decode-only CUDA graphs. Standard AllGather/ReduceScatter.

    Non-CG path (use_allgather_v=True): ranks may have different token counts
    (prefill). Each rank pads its tensors to max_tokens, runs a standard AllGather,
    then compacts by stripping per-rank padding. Combine is the reverse: expand
    compact output to padded layout, ReduceScatter, truncate to local token count.
    """

    _use_allgather_v: bool = False
    _local_tokens_per_rank: Optional[List[int]] = None

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

    @classmethod
    def set_step_metadata(
        cls,
        local_tokens_per_rank: torch.Tensor,
        ep_group: torch.distributed.ProcessGroup,
        use_allgather_v: bool = False,
    ) -> None:
        """Set per-step metadata.

        Args:
            local_tokens_per_rank: [ep_size] int32 tensor of token counts per rank.
            ep_group: Expert parallel process group.
            use_allgather_v: True on non-CG steps where ranks have variable token counts.
        """
        cls._use_allgather_v = use_allgather_v
        cls._local_tokens_per_rank = local_tokens_per_rank.tolist()
        if cls._valid_tokens_tensor is None:
            cls._valid_tokens_tensor = torch.zeros(
                1, dtype=torch.int32, device=local_tokens_per_rank.device
            )
        cls._valid_tokens_tensor.copy_(local_tokens_per_rank.sum())

    def token_dispatch(self, hidden_states, probs):
        """Gather hidden_states, probs, and routing_map from all EP ranks.

        CG path: standard AllGather (equal token counts guaranteed).
        Non-CG path: pad to max_tokens, AllGather, compact (strip per-rank padding).

        Args:
            hidden_states: [local_tokens, hidden_dim] local input.
            probs: [local_tokens, topk] local routing probabilities.

        Returns:
            (hidden_states, probs) gathered to [total_tokens, *] shape.
            Also updates self.routing_map to [total_tokens, topk].
        """
        if self.ep_size == 1:
            return hidden_states, probs

        if not self.__class__._use_allgather_v:
            # CG path: equal token counts, standard gather.
            with torch.no_grad():
                self.routing_map = gather_from_sequence_parallel_region(
                    self.routing_map, group=self.tp_ep_group
                )
            probs = gather_from_sequence_parallel_region(probs, group=self.tp_ep_group)
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, group=self.tp_ep_group
            )
            return hidden_states, probs

        # Non-CG path: pad → AllGather → compact.
        tokens_per_rank = self.__class__._local_tokens_per_rank
        max_tokens = max(tokens_per_rank)

        def pad_to_max(tensor):
            deficit = max_tokens - tensor.shape[0]
            if deficit == 0:
                return tensor
            return torch.cat([tensor, tensor.new_empty((deficit,) + tensor.shape[1:])], dim=0)

        def allgather(padded_tensor):
            gathered = padded_tensor.new_empty(
                (self.ep_size * max_tokens,) + padded_tensor.shape[1:]
            )
            dist.all_gather_into_tensor(gathered, padded_tensor, group=self.ep_group)
            return gathered

        hidden_gathered = allgather(pad_to_max(hidden_states))
        probs_gathered = allgather(pad_to_max(probs))
        with torch.no_grad():
            routing_gathered = allgather(pad_to_max(self.routing_map))

        def compact(gathered_tensor):
            return torch.cat(
                [
                    gathered_tensor[src_rank * max_tokens : src_rank * max_tokens + n_tokens]
                    for src_rank, n_tokens in enumerate(tokens_per_rank)
                ],
                dim=0,
            )

        hidden_states = compact(hidden_gathered)
        probs = compact(probs_gathered)
        self.routing_map = compact(routing_gathered)
        return hidden_states, probs

    def dispatch_postprocess(self, hidden_states, probs):
        """Pass-through: mcore_fused_moe operates directly on the gathered tensors."""
        return hidden_states, None, probs

    def combine_preprocess(self, expert_output):
        """Pass-through: unpermute is handled inside mcore_fused_moe."""
        return expert_output

    def token_combine(self, hidden_states):
        """Scatter-reduce expert outputs back to each EP rank.

        CG path: standard ReduceScatter (equal token counts guaranteed).
        Non-CG path: expand compact output to padded layout, ReduceScatter, truncate.

        Args:
            hidden_states: [total_tokens, hidden_dim] expert outputs.

        Returns:
            [local_tokens, hidden_dim] bf16 local token outputs.
        """
        if self.ep_size == 1:
            return hidden_states.to(torch.bfloat16)

        if not self.__class__._use_allgather_v:
            # CG path: equal token counts, standard reduce-scatter.
            hidden_states = reduce_scatter_to_sequence_parallel_region(
                hidden_states, group=self.tp_ep_group
            )
            return hidden_states.to(torch.bfloat16)

        # Non-CG path: expand compact → padded, ReduceScatter, truncate.
        tokens_per_rank = self.__class__._local_tokens_per_rank
        max_tokens = max(tokens_per_rank)
        ep_rank = self.ep_group.rank()

        # Expand [total_tokens, H] → [ep_size * max_tokens, H], zeros in padding slots.
        padded_output = hidden_states.new_zeros(self.ep_size * max_tokens, hidden_states.shape[1])
        offset = 0
        for dst_rank, n_tokens in enumerate(tokens_per_rank):
            padded_output[dst_rank * max_tokens : dst_rank * max_tokens + n_tokens] = (
                hidden_states[offset : offset + n_tokens]
            )
            offset += n_tokens

        # ReduceScatter: [ep_size * max_tokens, H] → [max_tokens, H].
        scattered = padded_output.new_empty(max_tokens, hidden_states.shape[1])
        dist.reduce_scatter_tensor(scattered, padded_output, group=self.ep_group)

        # Truncate padding and cast.
        return scattered[: tokens_per_rank[ep_rank]].to(torch.bfloat16)


class NVLSAllGatherVDispatcher(InferenceAllGatherDispatcherBase):
    """Variable-count AllGather-V / ReduceScatter-V dispatcher for inference CUDA graphs.

    Replaces the fixed AllGather/ReduceScatter of NCCLAllGatherDispatcher with
    variable-count NVLS collectives so ranks can hold different token counts per step.
    All metadata lives on-device; no host sync is needed between steps.

    Requires Hopper+ GPUs with NVLink and symmetric memory.
    """

    # ── Class-level NVLS step metadata ───────────────────────────────────────────
    # Packed [3] int32: [valid_tokens, rank_token_offset, ep_max_tokens].
    # Written in-place each step for stable CUDA graph addresses.
    # _valid_tokens_tensor on the base is pointed at _step_metadata[0:1] on first
    # init, so experts.py can read valid_tokens via the base class interface.
    _step_metadata: Optional[torch.Tensor] = None  # [3] int32
    _ep_rank: int = 0
    _engine_max_tokens: int = 2048

    # RSV symmetric buffer — written here by _alloc_rsv_buffer so mcore_fused_moe
    # can write unpermute output directly into it, avoiding a copy before RSV.
    _rsv_symm_tensor: Optional[torch.Tensor] = None

    @classmethod
    def _get_rsv_tensor(cls) -> Optional[torch.Tensor]:
        return cls._rsv_symm_tensor

    @classmethod
    def _rank_token_offset(cls) -> torch.Tensor:
        return cls._step_metadata[1:2]

    @classmethod
    def _ep_max_tokens(cls) -> torch.Tensor:
        return cls._step_metadata[2:3]

    @classmethod
    def set_engine_max_tokens(cls, n: int) -> None:
        """Set at model init. Determines AGV/RSV CTA grid size (fixed for CG)."""
        cls._engine_max_tokens = n

    @classmethod
    def set_step_metadata(
        cls,
        local_tokens_per_rank: torch.Tensor,
        ep_group: torch.distributed.ProcessGroup,
    ) -> None:
        """Set all three NVLS metadata fields from per-rank token counts.

        Allocates _step_metadata on the first call and wires the base class's
        _valid_tokens_tensor to _step_metadata[0:1] so experts.py can always
        call InferenceAllGatherDispatcherBase._valid_tokens() regardless of
        which dispatcher is active. Subsequent calls copy in-place.
        """
        if cls._step_metadata is None:
            cls._step_metadata = torch.zeros(
                3, dtype=torch.int32, device=local_tokens_per_rank.device
            )
            InferenceAllGatherDispatcherBase._valid_tokens_tensor = cls._step_metadata[0:1]
            cls._ep_rank = dist.get_rank(group=ep_group)
        ep_rank = cls._ep_rank
        cls._step_metadata.copy_(
            torch.stack([
                local_tokens_per_rank.sum(),
                local_tokens_per_rank[:ep_rank].sum(),
                local_tokens_per_rank.max(),
            ])
        )

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

    def _alloc_agv_buffers(self, hidden_states: torch.Tensor, probs: torch.Tensor) -> tuple:
        """Allocate separate symmetric buffers for the three AGV outputs.

        Buffers are sized at engine_max_tokens * ep_size (fixed across steps)
        so the same allocations can be reused every CUDA graph replay.
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
        Caches the tensor so mcore_fused_moe can write into it directly,
        avoiding a copy before multimem_reduce_scatter_v.
        """
        global_max = self._engine_max_tokens * self.ep_size
        hidden_dim = hidden_states.shape[1]
        buf = SymmetricMemoryManager.get_buffer(
            "ep_rsv", process_group=self.ep_group
        ).maybe_get_tensor([global_max, hidden_dim], dtype=hidden_states.dtype)
        NVLSAllGatherVDispatcher._rsv_symm_tensor = buf["tensor"]
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
            "NVLSAllGatherVDispatcher requires NVLS symmetric memory for AGV. "
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

        Args:
            hidden_states: [global_max, hidden_dim] fp32 expert outputs.

        Returns:
            [local_tokens, hidden_dim] bf16 local token outputs.
        """
        if self.ep_size == 1:
            return hidden_states.to(torch.bfloat16)

        rs_buffer = self._alloc_rsv_buffer(hidden_states)
        assert rs_buffer["handle"] is not None, (
            "NVLSAllGatherVDispatcher requires NVLS symmetric memory for RSV. "
            "Ensure the device is Hopper+ with NVLink and symmetric memory is available."
        )

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