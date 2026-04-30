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
update_metadata method, invoked from the first instance's token_dispatch so the
per-step metadata kernel is captured inside the CUDA graph.
"""

from typing import List, Optional

import torch
import torch.distributed as dist

from megatron.core.inference.communication.torch_symm_triton import (
    multimem_all_gatherv_3tensor,
    multimem_reduce_scatter_v,
)
from megatron.core.inference.moe import InferenceGroupedGemmBackend
from megatron.core.inference.moe.metadata import fused_metadata_update
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
    and defines update_metadata independently.
    """

    # [1] int32: total valid tokens across all EP ranks this step.
    # Written in-place each step so CUDA graph replay sees a stable address.
    # NVLSAllGatherVDispatcher points this at _step_metadata[0:1] on first init
    # so that experts.py can always call _valid_tokens() on this base class.
    _valid_tokens_tensor: Optional[torch.Tensor] = None

    def __init__(self, *args, runs_metadata_sync: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._runs_metadata_sync = runs_metadata_sync

    @classmethod
    def _valid_tokens(cls) -> torch.Tensor:
        return cls._valid_tokens_tensor

    def update_metadata(self, local_tokens: int) -> None:
        """Per-step metadata refresh fired from the first instance's token_dispatch.

        Must be idempotent across a step (only called once) and safe to capture
        into a CUDA graph on the decode path.
        """
        raise NotImplementedError


class NCCLAllGatherDispatcher(InferenceAllGatherDispatcherBase):
    """AllGather token dispatcher for inference using NCCL.

    Two modes, selected by _use_allgather_v (set from the context each step):

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
        runs_metadata_sync: bool = False,
    ) -> None:
        super().__init__(
            num_local_experts=num_local_experts,
            local_expert_indices=local_expert_indices,
            config=config,
            pg_collection=pg_collection,
            runs_metadata_sync=runs_metadata_sync,
        )
        self.topk = config.moe_router_topk

    @classmethod
    def allocate_buffers(cls) -> None:
        """Allocate the per-step valid-tokens tensor read by mcore_fused_moe.

        Called once at model init from the dynamic context. Must run outside any
        CUDA graph capture so update_metadata can write to a stable address during
        replay without triggering allocations inside the graph.
        """
        device = torch.cuda.current_device()
        InferenceAllGatherDispatcherBase._valid_tokens_tensor = torch.zeros(
            1, dtype=torch.int32, device=device
        )

    def update_metadata(self, local_tokens: int) -> None:
        """Per-step metadata update; invoked from the first instance's token_dispatch.

        CG path (_use_allgather_v=False): ranks have equal counts by construction, so
        we only refresh _valid_tokens_tensor — a single .fill_ that is safe to capture.

        Non-CG path (_use_allgather_v=True): ranks may differ, so we all-gather the
        per-rank counts and host-sync via .tolist() for the pad/compact logic below.
        This path never runs under graph capture.
        """
        cls = NCCLAllGatherDispatcher
        ep_size = self.ep_size
        device = torch.cuda.current_device()

        if cls._use_allgather_v:
            local_count = torch.tensor([local_tokens], dtype=torch.int32, device=device)
            local_tokens_per_rank = torch.empty(ep_size, dtype=torch.int32, device=device)
            dist.all_gather_into_tensor(local_tokens_per_rank, local_count, group=self.ep_group)
            cls._local_tokens_per_rank = local_tokens_per_rank.tolist()
            InferenceAllGatherDispatcherBase._valid_tokens_tensor.copy_(local_tokens_per_rank.sum())
        else:
            InferenceAllGatherDispatcherBase._valid_tokens_tensor.fill_(ep_size * local_tokens)

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

        if self._runs_metadata_sync:
            self.update_metadata(hidden_states.shape[0])

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
            padded_output[dst_rank * max_tokens : dst_rank * max_tokens + n_tokens] = hidden_states[
                offset : offset + n_tokens
            ]
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

    # ── Class-level symmetric buffer handles (allocated once at model init) ───────
    # Dtypes: hidden=bf16, routing=int64, probs=fp32, rsv=bf16.
    _symm_agv_hidden: Optional[dict] = None  # {"tensor": ..., "handle": ...}
    _symm_agv_routing: Optional[dict] = None
    _symm_agv_probs: Optional[dict] = None
    _symm_rsv: Optional[dict] = None

    @classmethod
    def _get_rsv_tensor(cls) -> Optional[torch.Tensor]:
        """Return the RSV symmetric buffer tensor so mcore_fused_moe can write
        unpermute output directly into it, avoiding a copy before RSV."""
        return cls._symm_rsv["tensor"] if cls._symm_rsv is not None else None

    @classmethod
    def _rank_token_offset(cls) -> torch.Tensor:
        return cls._step_metadata[1:2]

    @classmethod
    def _ep_max_tokens(cls) -> torch.Tensor:
        return cls._step_metadata[2:3]

    @classmethod
    def allocate_buffers(
        cls,
        engine_max_tokens: int,
        topk: int,
        hidden_size: int,
        ep_group: torch.distributed.ProcessGroup,
    ) -> None:
        """Allocate all symmetric buffers and initialize class-level metadata.

        Called once at model init. Allocates fixed-size AGV and RSV symmetric
        memory buffers so dispatch/combine can proceed without any allocation on
        the hot path.

        Args:
            engine_max_tokens: Maximum tokens per EP rank (engine-level cap).
            topk: MoE router top-k value.
            hidden_size: Model hidden dimension.
            ep_group: Expert parallel process group.
        """
        cls._engine_max_tokens = engine_max_tokens
        ep_size = dist.get_world_size(group=ep_group)
        global_max = engine_max_tokens * ep_size
        device = torch.cuda.current_device()

        cls._symm_agv_hidden = SymmetricMemoryManager.get_buffer(
            "ep_agv_h", process_group=ep_group
        ).maybe_get_tensor([global_max, hidden_size], dtype=torch.bfloat16)

        cls._symm_agv_routing = SymmetricMemoryManager.get_buffer(
            "ep_agv_r", process_group=ep_group
        ).maybe_get_tensor([global_max, topk], dtype=torch.int64)

        cls._symm_agv_probs = SymmetricMemoryManager.get_buffer(
            "ep_agv_p", process_group=ep_group
        ).maybe_get_tensor([global_max, topk], dtype=torch.float32)

        cls._symm_rsv = SymmetricMemoryManager.get_buffer(
            "ep_rsv", process_group=ep_group
        ).maybe_get_tensor([global_max, hidden_size], dtype=torch.bfloat16)

        # Small scratch buffer for fused metadata allgather (WORLD_SIZE int32s).
        cls._symm_metadata = SymmetricMemoryManager.get_buffer(
            "ep_meta", process_group=ep_group
        ).maybe_get_tensor([ep_size], dtype=torch.int32)

        failed = [
            name
            for name, buf in (
                ("ep_agv_h", cls._symm_agv_hidden),
                ("ep_agv_r", cls._symm_agv_routing),
                ("ep_agv_p", cls._symm_agv_probs),
                ("ep_rsv", cls._symm_rsv),
                ("ep_meta", cls._symm_metadata),
            )
            if buf["handle"] is None
        ]
        if failed:
            raise RuntimeError(
                f"NVLSAllGatherVDispatcher: symmetric memory allocation failed for "
                f"{failed}. This dispatcher requires Hopper+ GPUs with NVLink. "
                f"Use inference_moe_token_dispatcher_type='nccl' on non-NVLS systems."
            )

        # Initialise step-metadata tensor and wire base class valid_tokens pointer.
        cls._step_metadata = torch.zeros(3, dtype=torch.int32, device=device)
        InferenceAllGatherDispatcherBase._valid_tokens_tensor = cls._step_metadata[0:1]
        cls._ep_rank = dist.get_rank(group=ep_group)

    def update_metadata(self, local_tokens: int) -> None:
        """Per-step metadata update; invoked from the first instance's token_dispatch.

        Fires the fused NVLS allgather+reduce to publish
        [valid_tokens, rank_token_offset, ep_max_tokens] into _step_metadata, then
        (for FlashInfer) pre-masks the routing buffer with -1 so rows beyond
        valid_tokens are ignored by the GEMM; the AGV below overwrites
        [0, valid_tokens) in-place.
        """
        cls = NVLSAllGatherVDispatcher
        fused_metadata_update(
            local_tokens=local_tokens,
            local_buf=cls._symm_metadata["tensor"],
            symm_mem_hdl=cls._symm_metadata["handle"],
            step_metadata=cls._step_metadata,
        )
        if self.config.inference_grouped_gemm_backend == InferenceGroupedGemmBackend.FLASHINFER:
            cls._symm_agv_routing["tensor"].fill_(-1)

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        runs_metadata_sync: bool = False,
    ) -> None:
        super().__init__(
            num_local_experts=num_local_experts,
            local_expert_indices=local_expert_indices,
            config=config,
            pg_collection=pg_collection,
            runs_metadata_sync=runs_metadata_sync,
        )
        self.topk = config.moe_router_topk
        # Set in dispatch_preprocess; consumed by token_dispatch and token_combine.
        self._local_tokens: int = 0

    # ── Dispatch path ─────────────────────────────────────────────────────────────

    def dispatch_preprocess(self, hidden_states, routing_map, probs):
        """Store routing map and local token count; no communication."""
        self.hidden_shape = hidden_states.shape
        # [S/TP, B, H] -> [S*B/TP, H]
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        self._local_tokens = hidden_states.shape[0]
        self.routing_map = routing_map
        return hidden_states, probs

    def token_dispatch(self, hidden_states, probs):
        """AllGather-V: gather hidden_states, probs, and routing_map from all EP ranks.

        Args:
            hidden_states: [local_tokens, hidden_size] bf16 local input.
            probs: [local_tokens, topk] fp32 local routing probabilities.

        Returns:
            (hidden_states, probs) gathered to [global_max, *] shape.
            Also updates self.routing_map to [global_max, topk] int64.
        """
        if self.ep_size == 1:
            return hidden_states, probs

        if self._runs_metadata_sync:
            self.update_metadata(hidden_states.shape[0])

        agv_h = self.__class__._symm_agv_hidden
        agv_r = self.__class__._symm_agv_routing
        agv_p = self.__class__._symm_agv_probs

        engine_max = self._engine_max_tokens
        global_max = engine_max * self.ep_size
        rank_token_offset = self._rank_token_offset()
        ep_max_tokens = self._ep_max_tokens()

        multimem_all_gatherv_3tensor(
            agv_h["tensor"],
            agv_r["tensor"],
            agv_p["tensor"],
            hidden_states,
            self.routing_map,
            probs,
            agv_h["handle"],
            agv_r["handle"],
            agv_p["handle"],
            rank_token_offset=rank_token_offset,
            ep_max_tokens=ep_max_tokens,
            engine_max_tokens=engine_max,
        )

        topk = probs.shape[1]
        hidden_dim = hidden_states.shape[1]
        self.routing_map = agv_r["tensor"].view(global_max, topk)
        probs = agv_p["tensor"].view(global_max, topk)
        hidden_states = agv_h["tensor"].view(global_max, hidden_dim)
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
            hidden_states: [global_max, hidden_size] bf16 expert outputs.

        Returns:
            [local_tokens, hidden_size] bf16 local token outputs.
        """
        if self.ep_size == 1:
            return hidden_states.to(torch.bfloat16)

        rsv = self.__class__._symm_rsv

        if hidden_states is not rsv["tensor"]:
            rsv["tensor"].copy_(hidden_states)
        output = torch.empty(
            self._local_tokens,
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        multimem_reduce_scatter_v(
            output,
            rsv["tensor"],
            rsv["handle"],
            rank_token_offset=self._rank_token_offset(),
            ep_max_tokens=self._ep_max_tokens(),
            engine_max_tokens=self._engine_max_tokens,
        )
        return output.to(torch.bfloat16)

    def combine_postprocess(self, hidden_states):
        """Restore original input shape (e.g. [S/TP, B, H] from [S*B/TP, H])."""
        return hidden_states.view(self.hidden_shape)
