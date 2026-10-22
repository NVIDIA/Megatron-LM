# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed NCCL owner-to-replica weights and replica-to-owner gradients."""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from megatron.core.transformer.moe.replica_weight_transport import (
    HomeExpertPlacement,
    HomePreparedPlan,
    HomeTransferHandle,
    ReplicaGradDestination,
    ReplicaPlacement,
    ReplicaPreparedPlan,
    ReplicaTransferHandle,
    ReplicaTransportCapabilities,
    ReplicaTransportConfig,
    ReplicaWeightLayout,
    ReplicaWeightSource,
    ReplicaWeightTransport,
)

if TYPE_CHECKING:
    from megatron.core.transformer.moe.home_expert_state import (
        HomeStateComponent,
        ShardedHomeExpertState,
    )


@dataclass(frozen=True, slots=True)
class _PeerSlots:
    """One peer's entries, ordered by destination replica slot."""

    peer: int
    home_indices: tuple[int, ...]
    replica_slots: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _NcclSchedule:
    """Weight-direction routes; gradients traverse the same routes backwards."""

    sends: tuple[_PeerSlots, ...]
    receives: tuple[_PeerSlots, ...]
    local: _PeerSlots
    max_transfer_slots: int


def _compile_schedule(table: list[list[int]], home_experts: int, rank: int) -> _NcclSchedule:
    """Compile an identical global slot table without retaining tensor addresses."""
    sends: dict[int, list[tuple[int, int]]] = {}
    receives: dict[int, list[tuple[int, int]]] = {}
    local = []
    num_experts = len(table) * home_experts
    send_counts, receive_counts = [0] * len(table), [0] * len(table)
    for destination, row in enumerate(table):
        for slot, expert in enumerate(row):
            if expert == -1:
                continue
            if not 0 <= expert < num_experts:
                raise ValueError(f"Replica expert id {expert} is outside [0, {num_experts}).")
            owner, index = divmod(expert, home_experts)
            if owner != destination:
                send_counts[owner] += 1
                receive_counts[destination] += 1
            if owner == destination == rank:
                local.append((index, slot))
            elif owner == rank:
                sends.setdefault(destination, []).append((index, slot))
            elif destination == rank:
                receives.setdefault(owner, []).append((index, slot))

    def entries(peer, pairs):
        return _PeerSlots(
            peer, tuple(index for index, _ in pairs), tuple(slot for _, slot in pairs)
        )

    return _NcclSchedule(
        tuple(entries(peer, pairs) for peer, pairs in sorted(sends.items())),
        tuple(entries(peer, pairs) for peer, pairs in sorted(receives.items())),
        entries(rank, local),
        max(1, *send_counts, *receive_counts),
    )


class _ReplicaChunkWorkspace:
    """Bounded scratch and a shared stream, independent of layer/expert shape."""

    def __init__(self, device, size_bytes):
        third = size_bytes // 3 // 8 * 8
        if third < 8:
            raise ValueError("NCCL replica scratch budget must be at least 24 bytes.")
        self.send = torch.empty(third, dtype=torch.uint8, device=device)
        self.receive = torch.empty_like(self.send)
        self.accumulate = torch.empty(third // 4, dtype=torch.float32, device=device)
        self.stream = torch.cuda.Stream(device=device)


_replica_chunk_workspaces = weakref.WeakValueDictionary()


def _get_replica_chunk_workspace(config):
    key = (config.group, config.device, config.replica_exchange_buffer_bytes)
    workspace = _replica_chunk_workspaces.get(key)
    if workspace is None:
        workspace = _ReplicaChunkWorkspace(config.device, config.replica_exchange_buffer_bytes)
        _replica_chunk_workspaces[key] = workspace
    return workspace


class _NcclWorkspace:
    """Stable storage and one communication stream for compatible runtimes."""

    def __init__(self, config):
        self._mxfp8 = config.weight_format == "mxfp8"
        self._numels = tuple(math.prod(shape) for shape in config.member_shapes)
        if config.num_local_replica_slots == 0:
            # Home-only transfers allocate typed staging only at a migration
            # boundary. In particular, do not allocate native wgrad staging:
            # native experts already accumulate through their normal path.
            self._weights = self._columnwise_weights = self._replica_grads = ()
            self._scales = {}
            self._stream = torch.cuda.Stream(device=config.device)
            self._inflight = []
            return
        self._weights = tuple(
            torch.empty(
                (config.num_local_replica_slots, *shape),
                dtype=torch.uint8 if self._mxfp8 else torch.bfloat16,
                device=config.device,
            )
            for shape in config.member_shapes
        )
        # Directional aliases are opt-in storage policy. The runtime fences
        # consumers before reuse; separate mode supports persistent caches.
        self._columnwise_weights = (
            (
                tuple(torch.empty_like(weight) for weight in self._weights)
                if config.replica_mxfp8_storage == "separate"
                else self._weights
            )
            if self._mxfp8
            else ()
        )
        self._scales = {}
        if self._mxfp8:
            row_scales, col_scales = [], []
            for row_shape, col_shape in zip(
                config.rowwise_scale_shapes, config.columnwise_scale_shapes
            ):
                if config.replica_mxfp8_storage == "reuse":
                    size = max(math.prod(row_shape), math.prod(col_shape))
                    storage = torch.empty(
                        (config.num_local_replica_slots, size),
                        dtype=torch.uint8,
                        device=config.device,
                    )
                    row_scales.append(
                        storage[:, : math.prod(row_shape)].view(
                            config.num_local_replica_slots, *row_shape
                        )
                    )
                    col_scales.append(
                        storage[:, : math.prod(col_shape)].view(
                            config.num_local_replica_slots, *col_shape
                        )
                    )
                else:
                    row_scales.append(
                        torch.empty(
                            (config.num_local_replica_slots, *row_shape),
                            dtype=torch.uint8,
                            device=config.device,
                        )
                    )
                    col_scales.append(
                        torch.empty(
                            (config.num_local_replica_slots, *col_shape),
                            dtype=torch.uint8,
                            device=config.device,
                        )
                    )
            self._scales[ReplicaWeightLayout.ROWWISE] = tuple(row_scales)
            self._scales[ReplicaWeightLayout.COLUMNWISE] = tuple(col_scales)
        self._replica_grads = tuple(
            torch.empty_like(weight, dtype=config.grad_dtype) for weight in self._weights
        )
        self.chunks = _get_replica_chunk_workspace(config)
        self._stream = self.chunks.stream
        self._inflight: list[tuple[torch.cuda.Event, tuple]] = []


_nccl_workspaces = weakref.WeakValueDictionary()


def _get_nccl_workspace(config):
    if config.replica_storage == "layerwise":
        return _NcclWorkspace(config)
    workspace = _nccl_workspaces.get(config)
    if workspace is None:
        workspace = _NcclWorkspace(config)
        _nccl_workspaces[config] = workspace
    return workspace


class _HomeChunkWorkspace:
    """One bounded send/receive pair shared by all serialized layers on a group."""

    def __init__(self, device: torch.device, size_bytes: int) -> None:
        # Round down for typed views, never up beyond the requested budget.
        half = (size_bytes // 2 // 8) * 8
        self.send = torch.empty(half, dtype=torch.uint8, device=device)
        self.receive = torch.empty_like(self.send)


_home_chunk_workspaces = weakref.WeakValueDictionary()


class NcclP2PTransport(ReplicaWeightTransport):
    """Transport BF16 or MXFP8 expert storage over a borrowed NCCL EP process group.

    Construction is collective over ``config.group`` and must occur in the
    same order on every member. A small all-reduce initializes the group before
    sparse batched P2P (whose first use otherwise requires every rank). All
    ranks must supply identical placements and issue transport operations in
    matching order, including relative to other operations on the same group.

    Planning synchronously copies the slot table to the host once per plan.
    Dynamic host schedules are not CUDA-graph safe, including when prepared
    before capture. Replica storage and one communication stream follow the
    configured storage policy. Serialized runtimes share replica storage by
    default and write native gradients directly into caller-owned main_grad.
    Callers must finish consumers before overwriting reusable replica storage.
    ``num_sms`` is not a portable NCCL P2P launch control and is unused here.
    """

    transport_name = "replica_nccl"
    capabilities = ReplicaTransportCapabilities(
        weight_formats=("bf16", "mxfp8"),
        grad_dtypes=(torch.bfloat16, torch.float32),
        home_exchange=True,
    )

    def __init__(self, config: ReplicaTransportConfig) -> None:
        super().__init__(config)
        if config.device.type != "cuda" or config.device.index is None:
            raise ValueError("replica_nccl requires an indexed CUDA device.")
        if config.group is None or not dist.is_initialized():
            raise ValueError("replica_nccl requires an initialized explicit NCCL process group.")
        if dist.get_backend(config.group) != "nccl":
            raise ValueError("replica_nccl requires a NCCL process group.")
        if dist.get_world_size(config.group) != config.world_size:
            raise ValueError("replica_nccl world_size does not match its process group.")
        if (
            config.num_local_home_experts <= 0
            or config.num_local_replica_slots < 0
            or len(config.member_shapes) != 2
            or any(len(shape) != 2 or min(shape) <= 0 for shape in config.member_shapes)
        ):
            raise ValueError("replica_nccl requires valid expert counts and two matrix shapes.")
        self._mxfp8 = config.weight_format == "mxfp8"
        if self._mxfp8:
            for shapes in (config.rowwise_scale_shapes, config.columnwise_scale_shapes):
                if (
                    shapes is None
                    or len(shapes) != 2
                    or any(not shape or any(dim <= 0 for dim in shape) for shape in shapes)
                ):
                    raise ValueError("replica_nccl MXFP8 requires both projections' scale shapes.")
        self._home_chunk_workspace = None
        self._destroyed = False
        self._check_available()
        self.rank = dist.get_rank(config.group)
        self._global_ranks = tuple(
            dist.get_global_rank(config.group, rank) for rank in range(config.world_size)
        )
        self.workspace = _get_nccl_workspace(config)
        self._numels = self.workspace._numels
        self._weights = self.workspace._weights
        self._columnwise_weights = self.workspace._columnwise_weights
        self._scales = self.workspace._scales
        self._replica_grads = self.workspace._replica_grads
        self._replica_chunks = getattr(self.workspace, "chunks", None)
        self._stream = self.workspace._stream
        self._inflight = self.workspace._inflight
        # All members participate even when this layer's eventual plan has no
        # traffic on some ranks. This group is borrowed, never destroyed here.
        ready = torch.zeros(1, device=config.device)
        dist.all_reduce(ready, group=config.group)

    def _check_available(self) -> None:
        if self._destroyed:
            raise RuntimeError("replica_nccl transport was destroyed.")
        if torch.cuda.current_device() != self.config.device.index:
            raise ValueError("Set the current CUDA device to the replica_nccl transport device.")
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("replica_nccl does not support CUDA capture with host schedules.")

    def prepare_home_exchange(self, placement: HomeExpertPlacement) -> HomePreparedPlan:
        """Snapshot and validate a physical permutation, without capturing weights."""
        self._check_available()
        table = placement.source_slots
        h = self.config.num_local_home_experts
        if table.dtype not in (torch.int32, torch.int64) or tuple(table.shape) != (
            self.config.world_size,
            h,
        ):
            raise ValueError("Home placement must be an integer [EP, H] source-slot table.")
        host = table.detach().cpu().clone()
        if sorted(host.flatten().tolist()) != list(range(host.numel())):
            raise ValueError("Home placement must be a permutation of all physical home slots.")
        if placement.version <= 0:
            raise ValueError("Home proposal version must be positive.")
        snapshot = HomeExpertPlacement(host, placement.version)
        return HomePreparedPlan(snapshot, self, _compile_schedule(host.tolist(), h, self.rank))

    @torch.no_grad()
    def exchange_home_chunks(
        self,
        *,
        state: ShardedHomeExpertState,
        components: tuple[HomeStateComponent, ...],
        plan: HomePreparedPlan,
        buffer_size_bytes: int,
    ) -> None:
        """Pack all old tile values before any write, including local permutation cycles.

        No full-layer snapshot, concatenation or per-peer device allocation is
        used. Stream ordering protects tile reuse; one final fence protects
        cross-layer reuse and publication of the new ownership mapping.
        """
        self._check_available()
        if plan.transport is not self:
            raise ValueError("Home plan belongs to a different transport.")
        h = self.config.num_local_home_experts
        if not components or any(len(c.shards) != h for c in components):
            raise ValueError("Home components must contain every local expert.")
        # Validate every tile size before the first in-place write.
        half = (buffer_size_bytes // 2 // 8) * 8
        widths = [half // (h * c.dtype.itemsize) for c in components]
        if min(widths) < 1:
            raise ValueError("Home exchange budget cannot hold one element per expert per buffer.")
        key = (self.config.group, self.config.device, buffer_size_bytes)
        workspace = _home_chunk_workspaces.get(key)
        if workspace is None:
            workspace = _HomeChunkWorkspace(self.config.device, buffer_size_bytes)
            _home_chunk_workspaces[key] = workspace
        self._home_chunk_workspace = workspace
        schedule = plan.metadata
        # Remote routes are contiguous per peer. Local fixed points need no copy.
        local = [
            (a, b)
            for a, b in zip(schedule.local.home_indices, schedule.local.replica_slots)
            if a != b
        ]
        send_slots = tuple(i for route in schedule.sends for i in route.home_indices)
        send_slots += tuple(a for a, _ in local)
        recv_slots = tuple(i for route in schedule.receives for i in route.replica_slots)
        recv_slots += tuple(b for _, b in local)
        if len(send_slots) != len(recv_slots):
            raise ValueError("Home permutation must preserve local expert capacity.")
        stream = torch.cuda.current_stream(self.config.device)
        # Prior training and transport streams were drained by the caller.
        # Copy, NCCL wait, install and reuse are ordered on this stream.
        for component, width in zip(components, widths):
            for offset in range(0, component.numel, width):
                count = min(width, component.numel - offset)
                nbytes = len(send_slots) * count * component.dtype.itemsize
                send = workspace.send[:nbytes].view(component.dtype).view(len(send_slots), count)
                receive = (
                    workspace.receive[:nbytes].view(component.dtype).view(len(recv_slots), count)
                )
                state.read_chunk(component, send_slots, offset, send)
                edges = []
                start = 0
                for route in schedule.sends:
                    end = start + len(route.home_indices)
                    edges.append((self.rank, route.peer, dist.isend, send[start:end]))
                    start = end
                recv_start = 0
                for route in schedule.receives:
                    end = recv_start + len(route.replica_slots)
                    edges.append((route.peer, self.rank, dist.irecv, receive[recv_start:end]))
                    recv_start = end
                if local:
                    receive[recv_start:].copy_(send[start:])
                ops = [
                    dist.P2POp(
                        op,
                        tensor,
                        self._global_ranks[destination if source == self.rank else source],
                        group=self.config.group,
                    )
                    for source, destination, op, tensor in sorted(edges, key=lambda e: e[:2])
                ]
                works = dist.batch_isend_irecv(ops) if ops else ()
                for work in works:
                    work.wait()
                state.write_chunk(component, recv_slots, offset, receive)
        # Every EP member finishes before a caller can publish the new mapping.
        dist.barrier(group=self.config.group)
        stream.synchronize()

    @torch.no_grad()
    def start_home_exchange(self, *, sources, plan) -> HomeTransferHandle:
        """Receive before install: cycles, local swaps, mixed dtypes and shapes are safe."""
        self._check_available()
        if plan.transport is not self:
            raise ValueError("Home plan belongs to a different transport.")
        if not sources:
            raise ValueError("Home exchange requires at least one state component.")
        h = self.config.num_local_home_experts
        for source in sources:
            if (
                source.device != self.config.device
                or source.ndim < 2
                or source.shape[0] != h
                or not source.is_contiguous()
            ):
                raise ValueError("Home components must be contiguous device tensors [H, ...].")
        schedule = plan.metadata
        producer = torch.cuda.current_stream(self.config.device)
        self._stream.wait_stream(producer)
        self._inflight[:] = [(event, refs) for event, refs in self._inflight if not event.query()]
        with torch.cuda.stream(self._stream):
            received = tuple(torch.empty_like(source) for source in sources)
            buffers = []
            for source, destination in zip(sources, received):
                source.record_stream(self._stream)
                numel = source[0].numel()
                sends = {
                    route.peer: torch.cat([source[i].reshape(-1) for i in route.home_indices])
                    for route in schedule.sends
                }
                remote, keepalive = self._exchange(
                    sends,
                    {route.peer: len(route.replica_slots) * numel for route in schedule.receives},
                    source.dtype,
                )
                buffers.append(keepalive)
                for index, slot in zip(schedule.local.home_indices, schedule.local.replica_slots):
                    destination[slot].copy_(source[index])
                for route in schedule.receives:
                    for i, slot in enumerate(route.replica_slots):
                        destination[slot].view(-1).copy_(
                            remote[route.peer][i * numel : (i + 1) * numel]
                        )
            return HomeTransferHandle(self._finish(plan, sources, received, buffers), received)

    def wait_home_exchange(self, handle: HomeTransferHandle) -> tuple[torch.Tensor, ...]:
        """Finish a rare home exchange and release its large transport keepalives.

        Home migrations run at a drained optimizer boundary. Unlike microbatch
        replica prefetch, this wait also blocks the host so completed snapshots
        do not accumulate across every layer until the next forward pass.
        Installation remains on the caller's stream, protected by record_stream.
        """
        self.wait_weight_sync(handle.transfer)
        handle.transfer.completion.synchronize()
        self._inflight[:] = [(event, refs) for event, refs in self._inflight if not event.query()]
        for tensor in handle.received:
            tensor.record_stream(torch.cuda.current_stream(self.config.device))
        return handle.received

    def _require_replicas(self) -> None:
        if self.config.num_local_replica_slots == 0:
            raise RuntimeError("Home-only NCCL transport has no replica storage or operations.")

    def _projection_index(self, projection_index: int) -> None:
        self._require_replicas()
        if projection_index not in (0, 1):
            raise ValueError("Replica projection index must be 0 (FC1) or 1 (FC2).")
        if self._destroyed:
            raise RuntimeError("replica_nccl transport was destroyed.")

    @property
    def grad_dtype(self) -> torch.dtype:
        """Return the replica gradient storage and wire dtype."""
        return self.config.grad_dtype

    def projection_views(
        self, projection_index: int
    ) -> tuple[tuple[torch.Tensor | tuple[torch.Tensor, ...], ...], torch.Tensor]:
        """Return stable BF16 weights or MXFP8 row/column data and scale views."""
        self._projection_index(projection_index)
        if self._mxfp8:
            return (
                tuple(
                    zip(
                        self._weights[projection_index],
                        self._scales[ReplicaWeightLayout.ROWWISE][projection_index],
                        self._columnwise_weights[projection_index],
                        self._scales[ReplicaWeightLayout.COLUMNWISE][projection_index],
                    )
                ),
                self._replica_grads[projection_index],
            )
        return tuple(self._weights[projection_index]), self._replica_grads[projection_index]

    def prepare_plan(self, placement: ReplicaPlacement) -> ReplicaPreparedPlan:
        """Read device placement synchronously and compile deterministic peer lists."""
        self._check_available()
        self._require_replicas()
        super().prepare_plan(placement)
        schedule = _compile_schedule(
            placement.slot_to_expert.cpu().tolist(), placement.ownership.home_experts, self.rank
        )
        return ReplicaPreparedPlan(placement, self, schedule)

    def _schedule(self, plan: ReplicaPreparedPlan) -> _NcclSchedule:
        self._check_available()
        self._require_replicas()
        self.validate_plan(plan)
        if not isinstance(plan.metadata, _NcclSchedule):
            raise ValueError("replica_nccl requires a compiled NCCL schedule.")
        # Keep inputs alive even if the caller drops a handle before GPU completion.
        self._inflight[:] = [(event, refs) for event, refs in self._inflight if not event.query()]
        return plan.metadata

    def _validate_tensors(self, tensors, projection: int, dtype: torch.dtype, shape=None) -> None:
        if len(tensors) != self.config.num_local_home_experts:
            raise ValueError("Replica tensors must contain every canonical local expert.")
        for tensor in tensors:
            if (
                tensor.device != self.config.device
                or tensor.dtype != dtype
                or tensor.numel()
                != (math.prod(shape) if shape is not None else self._numels[projection])
                or (shape is not None and tuple(tensor.shape) != tuple(shape))
                or not tensor.is_contiguous()
            ):
                raise ValueError("Replica tensor device, dtype, size or contiguity is invalid.")

    def _exchange(self, sends, receive_sizes=None, dtype=None, *, receives=None):
        """Enqueue one packed message per directed peer pair on the current stream."""
        if receives is None:
            receives = {
                peer: torch.empty(size, dtype=dtype, device=self.config.device)
                for peer, size in receive_sizes.items()
            }
        # Sorting directed edges gives every rank a consistent subset of one
        # global ordering. NCCL has no tags to distinguish mismatched messages.
        edges = [(self.rank, peer, dist.isend, tensor) for peer, tensor in sends.items()]
        edges.extend((peer, self.rank, dist.irecv, tensor) for peer, tensor in receives.items())
        ops = [
            dist.P2POp(
                op,
                tensor,
                self._global_ranks[destination if source == self.rank else source],
                group=self.config.group,
            )
            for source, destination, op, tensor in sorted(edges, key=lambda edge: edge[:2])
        ]
        works = tuple(dist.batch_isend_irecv(ops)) if ops else ()
        for work in works:
            # Connect NCCL's internal stream to this stream before unpack/add.
            # Under NCCL blocking-wait settings this may also block the host.
            work.wait()
        return receives, (tuple(sends.values()), tuple(receives.values()), works)

    def _chunk_width(self, schedule, dtype, *, gradients=False):
        # The bound is global, so all ranks use identical message/tile boundaries,
        # including ranks with no local traffic and duplicate hot-expert slots.
        chunks = self._replica_chunks
        width = chunks.send.numel() // (schedule.max_transfer_slots * dtype.itemsize)
        if gradients:
            width = min(width, chunks.accumulate.numel() // self.config.num_local_home_experts)
        if width < 1:
            raise ValueError("NCCL replica budget cannot hold one element per transfer slot.")
        return width

    def _chunk_exchange(self, schedule, tensors, offset, count, dtype, *, reverse=False):
        send_routes = schedule.receives if reverse else schedule.sends
        recv_routes = schedule.sends if reverse else schedule.receives
        sends, receives = {}, {}
        cursor = 0
        for route in send_routes:
            indices = route.replica_slots if reverse else route.home_indices
            size = len(indices) * count * dtype.itemsize
            view = self._replica_chunks.send[cursor : cursor + size].view(dtype).view(-1, count)
            for row, index in zip(view, indices):
                row.copy_(tensors[index].view(-1)[offset : offset + count])
            sends[route.peer] = view
            cursor += size
        cursor = 0
        for route in recv_routes:
            size = len(route.home_indices) * count * dtype.itemsize
            receives[route.peer] = (
                self._replica_chunks.receive[cursor : cursor + size].view(dtype).view(-1, count)
            )
            cursor += size
        self._exchange(sends, receives=receives)
        return receives

    def _finish(self, *keepalive) -> ReplicaTransferHandle:
        done = torch.cuda.Event()
        done.record(self._stream)
        self._inflight.append((done, keepalive))
        # The workspace owns asynchronous inputs until completion. A runtime
        # retains its completed handle for repeated consumer waits; attaching
        # packed messages there would retain a full temporary copy per layer.
        return ReplicaTransferHandle(self, done)

    @torch.no_grad()
    def start_weight_sync(
        self, *, sources: tuple[ReplicaWeightSource, ...], plan: ReplicaPreparedPlan
    ) -> ReplicaTransferHandle:
        """Pack FC1/FC2 by peer and complete only after local copies and unpack."""
        schedule = self._schedule(plan)
        if len(sources) != 2:
            raise ValueError("replica_nccl requires FC1 and FC2 weight sources.")
        dtype = torch.uint8 if self._mxfp8 else torch.bfloat16
        parts = []
        layout = sources[0].layout
        for projection, source in enumerate(sources):
            if self._mxfp8:
                if (
                    layout not in (ReplicaWeightLayout.ROWWISE, ReplicaWeightLayout.COLUMNWISE)
                    or source.layout is not layout
                    or source.scales is None
                ):
                    raise ValueError(
                        "replica_nccl MXFP8 requires matching directional data and scales."
                    )
                weights = (
                    self._columnwise_weights
                    if layout is ReplicaWeightLayout.COLUMNWISE
                    else self._weights
                )
                scales = self._scales[layout][projection]
                self._validate_tensors(source.scales, projection, torch.uint8, scales.shape[1:])
            else:
                if source.layout is not ReplicaWeightLayout.PLAIN or source.scales is not None:
                    raise ValueError("replica_nccl supports plain BF16 weights without scales.")
                weights = self._weights
            self._validate_tensors(source.data, projection, dtype)
            parts.append((source.data, weights[projection], self._numels[projection]))
            if self._mxfp8:
                parts.append((source.scales, scales, math.prod(scales.shape[1:])))
        self._stream.wait_stream(torch.cuda.current_stream(self.config.device))
        with torch.cuda.stream(self._stream):
            for tensors, storage, _ in parts:
                for tensor in tensors:
                    tensor.record_stream(self._stream)
                storage.record_stream(self._stream)
            width = self._chunk_width(schedule, dtype)
            for tensors, storage, numel in parts:
                for index, slot in zip(schedule.local.home_indices, schedule.local.replica_slots):
                    storage[slot].view(-1).copy_(tensors[index].view(-1))
                for offset in range(0, numel, width):
                    count = min(width, numel - offset)
                    receives = self._chunk_exchange(schedule, tensors, offset, count, dtype)
                    for route in schedule.receives:
                        for row, slot in zip(receives[route.peer], route.replica_slots):
                            storage[slot].view(-1)[offset : offset + count].copy_(row)
            return self._finish(plan, sources)

    def wait_weight_sync(self, handle: ReplicaTransferHandle) -> None:
        """Order every calling stream after transfer and postprocessing."""
        self._check_available()
        if handle.transport is not self:
            raise ValueError("Replica completion belongs to a different transport.")
        torch.cuda.current_stream(self.config.device).wait_event(handle.completion)

    @torch.no_grad()
    def start_grad_reduce(
        self,
        *,
        native_grads: tuple[ReplicaGradDestination, ...],
        plan: ReplicaPreparedPlan,
        projections: tuple[int, ...],
    ) -> ReplicaTransferHandle:
        """Return selected replica gradients and add in FP32 with one final cast."""
        schedule = self._schedule(plan)
        if (
            not projections
            or len(set(projections)) != len(projections)
            or any(projection not in (0, 1) for projection in projections)
            or len(native_grads) != 2
        ):
            raise ValueError("Select distinct FC1/FC2 projections and supply both destinations.")
        projections = tuple(sorted(projections))
        for projection in projections:
            self._validate_tensors(native_grads[projection].tensors, projection, self.grad_dtype)
        self._stream.wait_stream(torch.cuda.current_stream(self.config.device))
        with torch.cuda.stream(self._stream):
            for projection in projections:
                self._replica_grads[projection].record_stream(self._stream)
                for tensor in native_grads[projection].tensors:
                    tensor.record_stream(self._stream)
            width = self._chunk_width(schedule, self.grad_dtype, gradients=True)
            # Only touched homes are written. FP32 scratch preserves the old
            # one-final-cast BF16 reduction semantics without a full-expert copy.
            touched = sorted(
                set(schedule.local.home_indices).union(
                    index for route in schedule.sends for index in route.home_indices
                )
            )
            for projection in projections:
                native = native_grads[projection].tensors
                replicas = self._replica_grads[projection]
                for offset in range(0, self._numels[projection], width):
                    count = min(width, self._numels[projection] - offset)
                    accum = self._replica_chunks.accumulate[
                        : self.config.num_local_home_experts * count
                    ].view(self.config.num_local_home_experts, count)
                    for index in touched:
                        accum[index].copy_(native[index].view(-1)[offset : offset + count])
                    receives = self._chunk_exchange(
                        schedule, replicas, offset, count, self.grad_dtype, reverse=True
                    )
                    for index, slot in zip(
                        schedule.local.home_indices, schedule.local.replica_slots
                    ):
                        accum[index].add_(replicas[slot].view(-1)[offset : offset + count])
                    for route in schedule.sends:
                        for row, index in zip(receives[route.peer], route.home_indices):
                            accum[index].add_(row)
                    for index in touched:
                        native[index].view(-1)[offset : offset + count].copy_(accum[index])
            return self._finish(plan, native_grads)

    def wait_grad_reduce(self, handle: ReplicaTransferHandle) -> None:
        """Order the caller after communication, accumulation and final cast."""
        self.wait_weight_sync(handle)

    def destroy(self) -> None:
        """Drain this layer's operations and release storage, retaining the borrowed group."""
        if self._destroyed:
            return
        self._check_available()
        self._stream.synchronize()
        self._inflight.clear()
        self._weights = self._columnwise_weights = self._replica_grads = ()
        self._scales = {}
        self.workspace = None
        self._home_chunk_workspace = None
        self._replica_chunks = None
        self._destroyed = True
