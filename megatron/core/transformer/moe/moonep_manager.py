# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoonEP runtime ownership and per-layer buffer registration for MCore MoE."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from megatron.core import utils

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from moonep import Buffer, MoonEPCommPlan
    from moonep._C import get_vmm_granularity, nvl_dist_alloc, nvl_dist_map, nvl_release_mem_handle
    from moonep.buffer import _exchange_ipc_fds
    from moonep.grad_reduce import launch_grad_reduce
    from moonep.prefetch import launch_prefetch

    HAVE_MOONEP = True
except ImportError:
    Buffer = None
    MoonEPCommPlan = None
    get_vmm_granularity = None
    nvl_dist_alloc = None
    nvl_dist_map = None
    nvl_release_mem_handle = None
    _exchange_ipc_fds = None
    launch_grad_reduce = None
    launch_prefetch = None
    HAVE_MOONEP = False

# MoonEP's prefetch and grad-reduce kernels tile both trailing weight dimensions.
MOONEP_WEIGHT_TILE = 128


@dataclass
class MoonEPProjectionBuffers:
    """MoonEP storage for one expert projection of one layer.

    ``weight`` and ``grad`` are the contiguous [E+B, H, H'] VMM ranges MoonEP's
    kernels expect: rows [0, E) map every rank's expert chunk (only this rank's
    E/R rows are physically local) and rows [E, E+B) are the process-global
    prefetch pool and gradient reduce chunk shared by all layers.
    """

    weight: torch.Tensor
    grad: torch.Tensor
    reduce_buffer: torch.Tensor
    num_global_experts: int
    master_parameters: List[torch.nn.Parameter] = field(default_factory=list)

    @property
    def source_weights(self) -> torch.Tensor:
        """Rows [0, E): every rank's expert weights for this projection."""
        return self.weight[: self.num_global_experts]

    @property
    def prefetch_slots(self) -> torch.Tensor:
        """Rows [E, E+B): the local prefetch slots the duplicated experts land in."""
        return self.weight[self.num_global_experts :]

    @property
    def source_grads(self) -> torch.Tensor:
        """Rows [0, E): every rank's expert gradients for this projection."""
        return self.grad[: self.num_global_experts]

    @property
    def prefetch_slot_grads(self) -> torch.Tensor:
        """Rows [E, E+B): backed by this rank's chunk of the shared reduce buffer."""
        return self.grad[self.num_global_experts :]


@dataclass
class MoonEPLayerBuffers:
    """Per-layer MoonEP storage for the two TEGroupedMLP projections."""

    fc1: MoonEPProjectionBuffers
    fc2: MoonEPProjectionBuffers
    parameters_bound: bool = False

    def projections(self) -> Tuple[MoonEPProjectionBuffers, MoonEPProjectionBuffers]:
        """Iterate projections in TEGroupedMLP order."""
        return (self.fc1, self.fc2)


@dataclass
class _SharedChunk:
    """A process-global chunk reused as the [E, E+B) tail of every layer's range."""

    keepalive: torch.Tensor
    fd: int
    reduce_view: Optional[torch.Tensor] = None


class MoonEPManager:
    """Own one MoonEP runtime per expert-parallel process group.

    The runtime holds the communication ``Buffer`` plus the process-global prefetch
    pool and gradient reduce buffer that every MoE layer shares, and maps each
    layer its own [E+B, H, H'] weight and gradient range.

    Args:
        config: Transformer configuration for the MoE model.
        ep_group: Expert-parallel process group owned by this runtime.
    """

    def __init__(self, config: TransformerConfig, ep_group: torch.distributed.ProcessGroup) -> None:
        if not HAVE_MOONEP:
            raise ImportError(
                "moe_token_dispatcher_type='moonep' requires the MoonEP package. Install "
                "https://github.com/MoonshotAI/MoonEP in the training container."
            )
        if config.num_moe_experts is None or config.moe_ffn_hidden_size is None:
            raise ValueError("MoonEP requires num_moe_experts and moe_ffn_hidden_size.")

        self.config = config
        self.group = ep_group
        self.rank = utils.get_pg_rank(ep_group)
        self.num_ranks = utils.get_pg_size(ep_group)
        self.num_global_experts = config.num_moe_experts
        if self.num_global_experts % self.num_ranks != 0:
            raise ValueError(
                f"MoonEP requires num_moe_experts ({self.num_global_experts}) to be divisible "
                f"by the expert-parallel size ({self.num_ranks})."
            )
        self.num_local_master_experts = self.num_global_experts // self.num_ranks
        # Training must reserve one prefetch slot per local expert: the planner can
        # duplicate a whole remote home group onto this rank.
        self.num_prefetch_slots = self.num_local_master_experts
        self.num_local_physical_experts = self.num_local_master_experts + self.num_prefetch_slots
        self.router_topk = config.moe_router_topk

        self.fc1_shape, self.fc2_shape = expert_weight_shapes(config)
        for name, shape in (("linear_fc1", self.fc1_shape), ("linear_fc2", self.fc2_shape)):
            for dim in shape:
                if dim % MOONEP_WEIGHT_TILE != 0:
                    raise ValueError(
                        f"MoonEP requires both {name} weight dims to be multiples of "
                        f"{MOONEP_WEIGHT_TILE}, got {shape}."
                    )

        self.num_sms = config.moe_moonep_num_sms
        self.token_padding = config.moe_moonep_token_padding
        self._buffer: Optional[Buffer] = None
        self._tokens_per_rank: Optional[int] = None
        self._layers: Dict[int, MoonEPLayerBuffers] = {}
        self._shared_chunks: Dict[str, _SharedChunk] = {}
        self._closed = False

    @property
    def signature(self) -> tuple:
        """Configuration fields that must match for layers sharing the runtime."""
        return _config_signature(self.config, self.group)

    @property
    def local_master_slice(self) -> slice:
        """Rows of the global [E] expert axis physically owned by this rank."""
        start = self.rank * self.num_local_master_experts
        return slice(start, start + self.num_local_master_experts)

    # ------------------------------------------------------------------
    # Runtime and storage
    # ------------------------------------------------------------------

    def ensure_buffer(self, tokens_per_rank: int) -> Buffer:
        """Create the communication buffer on first use and pin its token count."""
        if self._buffer is None:
            self._buffer = Buffer(
                S=tokens_per_rank,
                H=self.config.hidden_size,
                K=self.router_topk,
                E=self.num_global_experts,
                num_ep_ranks=self.num_ranks,
                num_sms=self.num_sms,
                token_padding=self.token_padding,
                B=self.num_prefetch_slots,
                group=self.group,
                explicitly_destroy=True,
            )
            self._tokens_per_rank = tokens_per_rank
        elif self._tokens_per_rank != tokens_per_rank:
            raise ValueError(
                f"MoonEP buffers are allocated for {self._tokens_per_rank} tokens per rank but "
                f"got {tokens_per_rank}; variable sequence or micro-batch size is unsupported."
            )
        return self._buffer

    @property
    def buffer(self) -> Buffer:
        """Return the communication buffer, which the first dispatch must have allocated."""
        if self._buffer is None:
            raise RuntimeError("MoonEP buffer is used before the first dispatch allocated it.")
        return self._buffer

    def _exchange_chunk_fds(self, local_fd: int) -> List[int]:
        """Swap one chunk's POSIX handle with every rank in the expert-parallel group."""
        received = _exchange_ipc_fds(
            local_fd, list(range(self.num_ranks)), self.rank, self.num_ranks, self.group
        )
        return [received[peer] for peer in range(self.num_ranks)]

    def _alloc_shared_chunk(
        self, name: str, chunk_shape: List[int], dtype: torch.dtype, symmetric: bool
    ) -> _SharedChunk:
        """Allocate (once) the chunk every layer reuses as its [E, E+B) tail rows.

        ``symmetric`` additionally maps all ranks' chunks as the [R, B, H, H']
        reduce-buffer view that ``reduce_grad`` reads over NVLink.
        """
        if name in self._shared_chunks:
            return self._shared_chunks[name]

        keepalive, local_fd, owned_handle = nvl_dist_alloc(shape=chunk_shape, dtype=dtype)
        tail_fd = os.dup(local_fd)
        reduce_view = None
        try:
            if symmetric:
                peer_fds = self._exchange_chunk_fds(local_fd)
                try:
                    reduce_view = nvl_dist_map(
                        chunk_shape=chunk_shape,
                        dtype=dtype,
                        fds=peer_fds,
                        local_rank=self.rank,
                        world_size=self.num_ranks,
                    ).view(self.num_ranks, *chunk_shape)
                finally:
                    for fd in peer_fds:
                        os.close(fd)
            os.close(local_fd)
        finally:
            nvl_release_mem_handle(owned_handle)

        keepalive.zero_()
        chunk = _SharedChunk(keepalive=keepalive, fd=tail_fd, reduce_view=reduce_view)
        self._shared_chunks[name] = chunk
        return chunk

    def _map_expert_range(
        self, chunk_shape: List[int], dtype: torch.dtype, tail_fd: int
    ) -> torch.Tensor:
        """Map [E+B, H, H'] as one VMM range: the R ranks' chunks then the shared tail."""
        keepalive, local_fd, owned_handle = nvl_dist_alloc(shape=chunk_shape, dtype=dtype)
        try:
            peer_fds = self._exchange_chunk_fds(local_fd)
            os.close(local_fd)
            try:
                full = nvl_dist_map(
                    chunk_shape=chunk_shape,
                    dtype=dtype,
                    fds=peer_fds + [tail_fd],
                    local_rank=self.rank,
                    world_size=self.num_ranks + 1,
                )
            finally:
                for fd in peer_fds:
                    os.close(fd)
        finally:
            nvl_release_mem_handle(owned_handle)
        full._keepalive = keepalive
        return full

    def _check_chunk_alignment(self, name: str, chunk_shape: List[int]) -> None:
        """MoonEP maps chunks with cuMemMap, so each must be an exact VMM granularity multiple."""
        granularity = get_vmm_granularity()
        for dtype, size in ((torch.bfloat16, 2), (torch.float32, 4)):
            nbytes = chunk_shape[0] * chunk_shape[1] * chunk_shape[2] * size
            if nbytes % granularity != 0:
                raise ValueError(
                    f"MoonEP {name} {dtype} chunk of {nbytes} bytes (shape {chunk_shape}) is not a "
                    f"multiple of the {granularity}-byte VMM granularity; adjust the expert "
                    f"count per rank or the expert FFN size."
                )

    def _make_projection(self, name: str, shape: Tuple[int, int]) -> MoonEPProjectionBuffers:
        """Allocate one layer's [E+B] weight and gradient ranges for one projection."""
        chunk_shape = [self.num_prefetch_slots, shape[0], shape[1]]
        self._check_chunk_alignment(name, chunk_shape)
        pool = self._alloc_shared_chunk(f"{name}_pool", chunk_shape, torch.bfloat16, False)
        reduce = self._alloc_shared_chunk(f"{name}_reduce", chunk_shape, torch.float32, True)
        weight = self._map_expert_range(chunk_shape, torch.bfloat16, pool.fd)
        grad = self._map_expert_range(chunk_shape, torch.float32, reduce.fd)
        grad[self.local_master_slice].zero_()
        return MoonEPProjectionBuffers(
            weight=weight,
            grad=grad,
            reduce_buffer=reduce.reduce_view,
            num_global_experts=self.num_global_experts,
        )

    def register_layer(self, layer_number: int) -> MoonEPLayerBuffers:
        """Allocate (once) and return the MoonEP storage for one MoE layer."""
        if layer_number not in self._layers:
            self._layers[layer_number] = MoonEPLayerBuffers(
                fc1=self._make_projection("fc1", self.fc1_shape),
                fc2=self._make_projection("fc2", self.fc2_shape),
            )
        return self._layers[layer_number]

    # ------------------------------------------------------------------
    # Parameter binding
    # ------------------------------------------------------------------

    def bind_replica_parameters(
        self, layer_buffers: MoonEPLayerBuffers, linear_modules: Tuple[torch.nn.Module, ...]
    ) -> None:
        """Point the prefetch-slot parameters at MoonEP's pool and reduce rows.

        Runs before DDP so the marked replicas are skipped when its buffers are built
        and the marked masters have their gradient release deferred past reduce_grad.
        """
        for projection, linear in zip(layer_buffers.projections(), linear_modules):
            slots = projection.prefetch_slots
            slot_grads = projection.prefetch_slot_grads
            for index in range(self.num_local_master_experts):
                getattr(linear, f"weight{index}")._moonep_is_master = True
            for slot in range(self.num_prefetch_slots):
                name = f"weight{self.num_local_master_experts + slot}"
                param = getattr(linear, name, None)
                if param is None:
                    raise AttributeError(f"MoonEP expects per-expert parameter {name} in {linear}.")
                if tuple(param.shape) != tuple(slots.shape[1:]):
                    raise ValueError(
                        f"MoonEP expert weight shape mismatch for {name}: expected "
                        f"{tuple(slots.shape[1:])}, got {tuple(param.shape)}."
                    )
                param._moonep_is_replica = True
                param.data = slots[slot]
                param.main_grad = slot_grads[slot]

    def bind_master_parameters(
        self, layer_buffers: MoonEPLayerBuffers, linear_modules: Tuple[torch.nn.Module, ...]
    ) -> None:
        """Record master weight and DDP main-grad pointers after DDP has built its buffers."""
        if layer_buffers.parameters_bound:
            return
        for projection, linear in zip(layer_buffers.projections(), linear_modules):
            projection.master_parameters.clear()
            for index in range(self.num_local_master_experts):
                param = getattr(linear, f"weight{index}")
                main_grad = getattr(param, "main_grad", None)
                if main_grad is None:
                    raise RuntimeError(
                        "MoonEP master registration must run after MCore DDP initialization."
                    )
                if main_grad.dtype != torch.float32:
                    raise TypeError(
                        "MoonEP requires FP32 main gradients; enable "
                        "--accumulate-allreduce-grads-in-fp32."
                    )
                projection.master_parameters.append(param)
        layer_buffers.parameters_bound = True

    # ------------------------------------------------------------------
    # Per-microbatch operations
    # ------------------------------------------------------------------

    def stage_master_weights(self, layer_buffers: MoonEPLayerBuffers) -> None:
        """Publish this rank's master expert weights into its symmetric slice."""
        if not layer_buffers.parameters_bound:
            raise RuntimeError("MoonEP master parameters are staged before they were bound.")
        for projection in layer_buffers.projections():
            local_rows = projection.source_weights[self.local_master_slice]
            # Read param.data live: the distributed optimizer re-points it on every
            # overlapped parameter all-gather.
            for index, param in enumerate(projection.master_parameters):
                local_rows[index].copy_(param.data)

    def prefetch(self, layer_buffers: MoonEPLayerBuffers, plan: MoonEPCommPlan) -> None:
        """Copy the duplicated experts' weights from their home ranks into local slots."""
        experts_to_copy = plan.experts_to_copy[self.rank]
        for projection in layer_buffers.projections():
            launch_prefetch(
                projection.source_weights, projection.prefetch_slots, experts_to_copy, self.num_sms
            )

    def reset_grad_accumulators(self, layer_buffers: MoonEPLayerBuffers) -> None:
        """Clear the rows that ``reduce_grad`` accumulates duplicated-expert wgrad into."""
        for projection in layer_buffers.projections():
            projection.source_grads[self.local_master_slice].zero_()

    def reduce_grad(self, layer_buffers: MoonEPLayerBuffers, plan: MoonEPCommPlan) -> None:
        """Merge duplicated experts' wgrad from every rank into the owning DDP gradients."""
        ctx = self.buffer._ctx  # noqa: SLF001 - MoonEP exposes no public accessor yet
        for projection in layer_buffers.projections():
            launch_grad_reduce(
                projection.source_grads,
                projection.reduce_buffer,
                plan.experts_to_copy,
                rank=self.rank,
                num_sms=self.num_sms,
                meta_buf=ctx['meta_buf'],
                meta_stride=int(ctx['meta_chunk_padded']),
                barrier_off=int(ctx['BARRIER_OFF']),
                grid_sync_bar=ctx['grid_sync_bar'],
            )
            local_rows = projection.source_grads[self.local_master_slice]
            for index, param in enumerate(projection.master_parameters):
                param.main_grad.add_(local_rows[index])
        self.release_master_grads(layer_buffers)

    def release_master_grads(self, layer_buffers: MoonEPLayerBuffers) -> None:
        """Hand the completed master gradients to DDP's overlapping reduction.

        DDP defers this for MoonEP masters so the duplicated-expert contribution is
        already accumulated when the bucket reduce starts.
        """
        for projection in layer_buffers.projections():
            for param in projection.master_parameters:
                bucket_group = getattr(param, "_moonep_ddp_bucket_group", None)
                if bucket_group is None or not bucket_group.ddp_config.overlap_grad_reduce:
                    continue
                bucket_group.register_grad_ready(
                    param, getattr(param, "_moonep_force_all_reduce", False)
                )

    def close(self) -> None:
        """Release the MoonEP runtime, its NVLink mappings and the retained chunk fds."""
        if self._closed:
            return
        if self._buffer is not None:
            self._buffer.destroy()
            self._buffer = None
        self._layers.clear()
        for chunk in self._shared_chunks.values():
            os.close(chunk.fd)
        self._shared_chunks.clear()
        self._closed = True

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def local_tokens_per_expert(self, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """Turn MoonEP's [E+B] cumulative offsets into this rank's physical group sizes.

        Rows outside this rank's home group are empty, so its master experts and then
        its prefetch slots are already contiguous at the front of the dispatched buffer.
        """
        counts = cu_seqlens - torch.cat((cu_seqlens.new_zeros(1), cu_seqlens[:-1]))
        masters = counts[self.local_master_slice]
        slots = counts[self.num_global_experts :]
        return torch.cat((masters, slots))


def expert_weight_shapes(config: TransformerConfig) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return the per-expert (fc1, fc2) weight shapes a TEGroupedMLP allocates."""
    hidden = config.hidden_size
    ffn = config.moe_ffn_hidden_size
    fc1_out = 2 * ffn if config.gated_linear_unit else ffn
    return (fc1_out, hidden), (hidden, ffn)


def _config_signature(config: TransformerConfig, ep_group: torch.distributed.ProcessGroup) -> tuple:
    """Signature of the MoonEP settings every layer on a process group must agree on."""
    num_ranks = utils.get_pg_size(ep_group)
    fc1_shape, fc2_shape = expert_weight_shapes(config)
    return (
        config.num_moe_experts,
        config.num_moe_experts // num_ranks,
        config.moe_router_topk,
        fc1_shape,
        fc2_shape,
        config.moe_moonep_num_sms,
        config.moe_moonep_token_padding,
    )


_MOONEP_MANAGER_REGISTRY: Dict[int, MoonEPManager] = {}


def get_or_create_moonep_manager(
    config: TransformerConfig, ep_group: torch.distributed.ProcessGroup
) -> MoonEPManager:
    """Get the shared MoonEP runtime for an expert-parallel process group."""
    key = id(ep_group)
    manager = _MOONEP_MANAGER_REGISTRY.get(key)
    if manager is None:
        manager = MoonEPManager(config, ep_group)
        _MOONEP_MANAGER_REGISTRY[key] = manager
    elif manager.signature != _config_signature(config, ep_group):
        raise ValueError(
            "MoE layers sharing an EP process group must use the same MoonEP configuration."
        )
    return manager


def clear_moonep_manager_registry() -> None:
    """Drop Python references to cached MoonEP managers (primarily for tests)."""
    _MOONEP_MANAGER_REGISTRY.clear()


def destroy_moonep_managers() -> None:
    """Destroy all MoonEP runtimes before the process group is torn down."""
    for manager in _MOONEP_MANAGER_REGISTRY.values():
        manager.close()
    _MOONEP_MANAGER_REGISTRY.clear()
