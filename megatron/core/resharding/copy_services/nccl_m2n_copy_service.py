# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, TypeVar

import torch
import torch.distributed as dist

from .base import CopyService, RecvOp, SendOp

logger = logging.getLogger(__name__)

_TransferOpT = TypeVar("_TransferOpT", SendOp, RecvOp)

# nccl-extensions' public M2N header requires NCCL 2.30.5 or newer.
# _validate_nccl_version checks the loaded libnccl before M2N is initialized.
_MINIMUM_NCCL_VERSION = (2, 30, 5)

# _operation_layout uses these constants for a deterministic, FNV-inspired
# fingerprint of each peer's ordered operations. The mask keeps the result in
# the non-negative torch.int64 range used by _exchange_pair_layouts; the offset
# is the initial seed, and the prime mixes each operation field and dtype byte.
_DIGEST_MASK = (1 << 63) - 1
_DIGEST_OFFSET = 1469598103934665603
_DIGEST_PRIME = 1099511628211


@dataclass(frozen=True)
class _M2NTopology:
    """Contiguous source/destination rank intervals required by NCCL M2N."""

    src_ranks: tuple[int, ...]
    dst_ranks: tuple[int, ...]


def _validate_role_roster(roles: list[tuple[bool, bool]]) -> _M2NTopology:
    """Validate NCCL M2N's disjoint, source-first rank topology."""
    overlapping = [rank for rank, (is_src, is_dst) in enumerate(roles) if is_src and is_dst]
    idle = [rank for rank, (is_src, is_dst) in enumerate(roles) if not is_src and not is_dst]
    if overlapping:
        raise RuntimeError(
            "NCCL M2N refit requires non-collocated source and destination ranks; "
            f"ranks {overlapping} participate on both sides"
        )
    if idle:
        raise RuntimeError(
            "NCCL M2N refit requires the process group to contain exactly the source and "
            f"destination meshes; idle ranks are not supported (idle ranks: {idle})"
        )

    src_ranks = tuple(rank for rank, (is_src, _is_dst) in enumerate(roles) if is_src)
    dst_ranks = tuple(rank for rank, (_is_src, is_dst) in enumerate(roles) if is_dst)
    if not src_ranks or not dst_ranks:
        raise RuntimeError("NCCL M2N refit requires at least one source and one destination rank")

    expected_src = tuple(range(len(src_ranks)))
    expected_dst = tuple(range(len(src_ranks), len(roles)))
    if src_ranks != expected_src or dst_ranks != expected_dst:
        raise RuntimeError(
            "NCCL M2N refit requires one source-first contiguous rank interval followed by "
            f"one destination interval; got source ranks {src_ranks} and destination ranks "
            f"{dst_ranks}"
        )
    return _M2NTopology(src_ranks=src_ranks, dst_ranks=dst_ranks)


def _ordered_ops_by_peer(
    ops: list[_TransferOpT], *, is_send: bool
) -> dict[int, list[_TransferOpT]]:
    """Group operations by peer and order them deterministically by task ID."""
    indexed_by_peer: dict[int, list[tuple[int, _TransferOpT]]] = defaultdict(list)
    for submission_index, op in enumerate(ops):
        if op.task_id is None:
            raise RuntimeError("NCCL M2N refit requires a task_id for every transfer")
        peer = getattr(op, "dest_rank" if is_send else "src_rank")
        indexed_by_peer[peer].append((submission_index, op))

    ordered = {}
    for peer, indexed_ops in indexed_by_peer.items():
        # Transforms may submit multiple tensors for one logical transfer, so
        # duplicate task IDs are intentional. Submission order is the stable
        # tie-breaker and is identical on the matching sender and receiver.
        indexed_ops.sort(key=lambda item: (item[1].task_id, item[0]))
        ordered[peer] = [op for _index, op in indexed_ops]
    return ordered


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _byte_view(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().reshape(-1).view(torch.uint8)


def _operation_layout(ops: list[_TransferOpT]) -> tuple[int, int, int]:
    """Return the byte count, op count, and digest for one peer's ordered operations.

    ``_exchange_pair_layouts`` gathers this tuple from every rank, then
    ``_validate_pair_layouts`` compares each sender with its receiver before
    any payload is submitted to M2N.
    """
    if not ops:
        return 0, 0, 0

    total_bytes = 0
    digest = _DIGEST_OFFSET
    for op in ops:
        if op.task_id is None:
            raise RuntimeError("NCCL M2N refit requires a task_id for every transfer")
        size = _tensor_nbytes(op.tensor)
        total_bytes += size
        fields = (op.task_id, size, op.tensor.element_size())
        for value in fields:
            digest = ((digest ^ (int(value) & _DIGEST_MASK)) * _DIGEST_PRIME) & _DIGEST_MASK
        for value in str(op.tensor.dtype).encode("ascii"):
            digest = ((digest ^ value) * _DIGEST_PRIME) & _DIGEST_MASK
        digest = ((digest ^ 0xFF) * _DIGEST_PRIME) & _DIGEST_MASK
    return total_bytes, len(ops), digest


def _validate_pair_layouts(topology: _M2NTopology, layouts: list[list[list[int]]]) -> int:
    """Validate source/destination pair layouts and return the uniform slot size."""
    slot_bytes = 0
    for src_index, src_rank in enumerate(topology.src_ranks):
        for dst_index, dst_rank in enumerate(topology.dst_ranks):
            send_layout = tuple(layouts[src_rank][dst_index])
            recv_layout = tuple(layouts[dst_rank][src_index])
            if send_layout != recv_layout:
                if send_layout[:2] == recv_layout[:2]:
                    mismatch = "ordered task/dtype layouts differ"
                else:
                    mismatch = (
                        f"source submitted {send_layout[0]} bytes across {send_layout[1]} "
                        f"tensors, but destination expects {recv_layout[0]} bytes across "
                        f"{recv_layout[1]} tensors"
                    )
                raise RuntimeError(
                    "NCCL M2N transfer layout mismatch for source rank "
                    f"{src_rank} and destination rank {dst_rank}: {mismatch}"
                )
            slot_bytes = max(slot_bytes, send_layout[0])
    return slot_bytes


def _validate_nccl_version(nccl: Any) -> None:
    """Ensure the loaded NCCL library supports the current M2N API."""
    try:
        version = nccl.get_version().libnccl.version
        release = tuple(version.release)
    except AttributeError as exc:
        raise RuntimeError("NCCL M2N requires the current NCCL4Py package") from exc

    if release < _MINIMUM_NCCL_VERSION:
        required = ".".join(str(value) for value in _MINIMUM_NCCL_VERSION)
        raise RuntimeError(f"NCCL M2N requires NCCL >= {required}, found {version}")


def _load_backend() -> tuple[Any, Any]:
    """Load the optional official M2N and NCCL4Py modules."""
    try:
        import nccl.core as nccl
        import nccl.m2n as m2n
    except ImportError as exc:
        raise RuntimeError("NCCL M2N refit requires NVIDIA/nccl-extensions and NCCL4Py") from exc
    _validate_nccl_version(nccl)
    return m2n, nccl


class NCCLM2NCopyService(CopyService):
    """Hierarchical non-collocated ReFIT transport backed by NCCL M2N.

    The generic ReFIT planner emits point-to-point slices. This service packs
    those slices into a dense ``[source, destination, bytes]`` logical tensor:
    source ranks shard dimension 0 and destination ranks shard dimension 1.
    One official ``nccl.m2n.reshard`` call moves the complete batch, after
    which receive slices are unpacked into their original tensors.

    NCCL M2N requires disjoint, contiguous source and destination meshes, so
    this backend supports non-collocated ReFIT only. The process group must
    contain source ranks first, destination ranks second, and no idle ranks.

    Args:
        group: NCCL process group containing exactly the source and destination ranks.
    """

    requires_process_group_barrier = False
    supports_idle_ranks = False

    def __init__(self, group=None):
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before NCCLM2NCopyService()")
        if not torch.cuda.is_available():
            raise RuntimeError("NCCLM2NCopyService requires CUDA")
        super().__init__(group=group)

        self._device = torch.device("cuda", torch.cuda.current_device())
        if dist.get_backend(group) != "nccl":
            raise RuntimeError("NCCLM2NCopyService requires an NCCL process group")

        self._m2n, self._nccl = _load_backend()
        self._handle = self._m2n.init()
        self._is_source: bool | None = None
        self._is_destination: bool | None = None
        self._topology: _M2NTopology | None = None
        self._comm: Any | None = None
        self._active_plan: object | None = None
        self._active_transform: object | None = None
        self._slot_bytes: int | None = None
        self._closed = False
        self._poisoned = False
        self.send_ops: list[SendOp] = []
        self.recv_ops: list[RecvOp] = []
        logger.info("NCCLM2NCopyService initialized on rank %d/%d", self.rank, self.world_size)

    def set_model_roles(self, *, is_source: bool, is_destination: bool) -> None:
        """Set this rank's fixed source/destination participation."""
        current = (self._is_source, self._is_destination)
        requested = (is_source, is_destination)
        if current != (None, None) and current != requested:
            raise RuntimeError(
                "NCCL M2N model roles cannot change during the service lifetime; construct the "
                "service for exactly one source mesh and one destination mesh"
            )
        self._is_source = is_source
        self._is_destination = is_destination

    def set_plan(self, plan: object, *, transform: object | None = None) -> None:
        """Select the immutable plan whose M2N layout should be reused."""
        if plan is not self._active_plan or transform is not self._active_transform:
            self._active_plan = plan
            self._active_transform = transform
            self._slot_bytes = None

    def submit_send(
        self, src_tensor: torch.Tensor, dest_rank: int, task_id: int | None = None
    ) -> None:
        self.send_ops.append(SendOp(task_id=task_id, tensor=src_tensor, dest_rank=dest_rank))

    def submit_recv(
        self, dest_tensor: torch.Tensor, src_rank: int, task_id: int | None = None
    ) -> None:
        self.recv_ops.append(RecvOp(task_id=task_id, tensor=dest_tensor, src_rank=src_rank))

    def _validate_tensor(self, tensor: torch.Tensor, operation: str) -> None:
        if not tensor.is_cuda:
            raise RuntimeError(f"NCCL M2N refit {operation} tensors must be CUDA tensors")
        if tensor.device != self._device:
            raise RuntimeError(
                f"NCCL M2N refit {operation} tensor is on {tensor.device}, expected {self._device}"
            )
        if not tensor.is_contiguous():
            raise RuntimeError(f"NCCL M2N refit {operation} tensors must be contiguous")

    def _prepare_ops(self) -> tuple[dict[int, list[SendOp]], dict[int, list[RecvOp]]]:
        sends = _ordered_ops_by_peer(self.send_ops, is_send=True)
        recvs = _ordered_ops_by_peer(self.recv_ops, is_send=False)
        for ops in sends.values():
            for op in ops:
                self._validate_tensor(op.tensor, "send")
        for ops in recvs.values():
            for op in ops:
                self._validate_tensor(op.tensor, "receive")
        return sends, recvs

    def _get_topology(self) -> _M2NTopology:
        if self._is_source is None or self._is_destination is None:
            raise RuntimeError(
                "NCCLM2NCopyService model roles were not configured; call set_model_roles() "
                "or use swap_model_weights()"
            )
        if self._topology is None:
            # The source and destination meshes are fixed for the service lifetime.
            roles = torch.tensor(
                [int(self._is_source), int(self._is_destination)],
                dtype=torch.int64,
                device=self._device,
            )
            gathered = [torch.empty_like(roles) for _ in range(self.world_size)]
            dist.all_gather(gathered, roles, group=self.group)
            host_roles = [tuple(bool(value) for value in item.cpu().tolist()) for item in gathered]
            self._topology = _validate_role_roster(host_roles)
        return self._topology

    def _validate_peers(
        self, topology: _M2NTopology, sends: dict[int, list[SendOp]], recvs: dict[int, list[RecvOp]]
    ) -> None:
        if self.rank in topology.src_ranks:
            if recvs:
                raise RuntimeError("NCCL M2N source ranks cannot submit receive operations")
            invalid = sorted(set(sends) - set(topology.dst_ranks))
            if invalid:
                raise RuntimeError(f"NCCL M2N sends target non-destination ranks: {invalid}")
        else:
            if sends:
                raise RuntimeError("NCCL M2N destination ranks cannot submit send operations")
            invalid = sorted(set(recvs) - set(topology.src_ranks))
            if invalid:
                raise RuntimeError(f"NCCL M2N receives name non-source ranks: {invalid}")

    def _exchange_pair_layouts(
        self, topology: _M2NTopology, sends: dict[int, list[SendOp]], recvs: dict[int, list[RecvOp]]
    ) -> int:
        """Agree on every pair's ordered byte layout before a plan's first M2N run."""
        peer_count = max(len(topology.src_ranks), len(topology.dst_ranks))
        local_layouts = [[0, 0, 0] for _ in range(peer_count)]
        if self.rank in topology.src_ranks:
            peer_indices = {rank: index for index, rank in enumerate(topology.dst_ranks)}
            local_ops = sends
        else:
            peer_indices = {rank: index for index, rank in enumerate(topology.src_ranks)}
            local_ops = recvs
        for peer, ops in local_ops.items():
            local_layouts[peer_indices[peer]] = list(_operation_layout(ops))

        local_tensor = torch.tensor(local_layouts, dtype=torch.int64, device=self._device)
        gathered = torch.empty(
            (self.world_size * peer_count, 3), dtype=torch.int64, device=self._device
        )
        dist.all_gather_into_tensor(gathered, local_tensor, group=self.group)
        host_layouts = gathered.view(self.world_size, peer_count, 3).cpu().tolist()
        return _validate_pair_layouts(topology, host_layouts)

    def _get_slot_bytes(
        self, topology: _M2NTopology, sends: dict[int, list[SendOp]], recvs: dict[int, list[RecvOp]]
    ) -> int:
        """Return a plan's agreed slot size, collecting it only on first use."""
        if self._active_plan is None:
            # Direct CopyService users have not promised an immutable plan, so
            # retain the conservative per-run validation for that lower-level API.
            return self._exchange_pair_layouts(topology, sends, recvs)

        if self._slot_bytes is None:
            self._slot_bytes = self._exchange_pair_layouts(topology, sends, recvs)
        return self._slot_bytes

    def _get_comm(self) -> Any:
        if self._comm is not None:
            return self._comm

        unique_id = bytes(self._nccl.get_unique_id(empty=self.rank != 0))
        if not unique_id:
            raise RuntimeError("NCCL4Py returned an empty NCCL unique ID")
        unique_id_tensor = torch.tensor(list(unique_id), dtype=torch.uint8, device=self._device)
        src_rank = 0 if self.group is None else dist.get_global_rank(self.group, 0)
        dist.broadcast(unique_id_tensor, src=src_rank, group=self.group)
        unique_id = bytes(unique_id_tensor.cpu().tolist())
        self._comm = self._nccl.Communicator.init(
            self.world_size, self.rank, self._nccl.UniqueId.from_bytes(unique_id)
        )
        return self._comm

    def _pack_sends(
        self,
        buffer: torch.Tensor,
        topology: _M2NTopology,
        sends: dict[int, list[SendOp]],
        slot_bytes: int,
    ) -> None:
        dst_index = {rank: index for index, rank in enumerate(topology.dst_ranks)}
        destinations = []
        sources = []
        for peer, ops in sends.items():
            offset = dst_index[peer] * slot_bytes
            for op in ops:
                size = _tensor_nbytes(op.tensor)
                destinations.append(buffer[offset : offset + size])
                sources.append(_byte_view(op.tensor))
                offset += size
        with torch.no_grad():
            if destinations:
                torch._foreach_copy_(destinations, sources)

    def _unpack_recvs(
        self,
        buffer: torch.Tensor,
        topology: _M2NTopology,
        recvs: dict[int, list[RecvOp]],
        slot_bytes: int,
    ) -> None:
        src_index = {rank: index for index, rank in enumerate(topology.src_ranks)}
        destinations = []
        sources = []
        for peer, ops in recvs.items():
            offset = src_index[peer] * slot_bytes
            for op in ops:
                size = _tensor_nbytes(op.tensor)
                destinations.append(_byte_view(op.tensor))
                sources.append(buffer[offset : offset + size])
                offset += size
        with torch.no_grad():
            if destinations:
                torch._foreach_copy_(destinations, sources)

    def _local_m2n_buffers(
        self, buffer: torch.Tensor, topology: _M2NTopology, slot_bytes: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        src = None
        dst = None
        if self.rank in topology.src_ranks:
            src_size = len(topology.dst_ranks) * slot_bytes
            src = buffer[:src_size].view(1, len(topology.dst_ranks), slot_bytes)
        if self.rank in topology.dst_ranks:
            dst_size = len(topology.src_ranks) * slot_bytes
            dst = buffer[:dst_size].view(len(topology.src_ranks), 1, slot_bytes)
        return src, dst

    def run(self) -> None:
        """Pack and execute all queued transfers in one hierarchical M2N reshard."""
        if self._closed:
            raise RuntimeError("NCCLM2NCopyService is closed")
        if self._poisoned:
            raise RuntimeError(
                "NCCLM2NCopyService is unusable after an M2N submission failure; "
                "close it and initialize a new service"
            )
        if torch.cuda.current_device() != self._device.index:
            raise RuntimeError(
                f"NCCL M2N service was created on {self._device}, but the current CUDA device is "
                f"cuda:{torch.cuda.current_device()}"
            )

        try:
            sends, recvs = self._prepare_ops()
            topology = self._get_topology()
            self._validate_peers(topology, sends, recvs)
            slot_bytes = self._get_slot_bytes(topology, sends, recvs)
            if slot_bytes == 0:
                return

            comm = self._get_comm()
            peer_count = (
                len(topology.dst_ranks)
                if self.rank in topology.src_ranks
                else len(topology.src_ranks)
            )
            buffer = torch.empty(peer_count * slot_bytes, dtype=torch.uint8, device=self._device)
            if self.rank in topology.src_ranks:
                self._pack_sends(buffer, topology, sends, slot_bytes)

            src, dst = self._local_m2n_buffers(buffer, topology, slot_bytes)
            stream = torch.cuda.current_stream(self._device)
            try:
                src_count = len(topology.src_ranks)
                dst_count = len(topology.dst_ranks)
                self._m2n.reshard(
                    src=src,
                    dst=dst,
                    comm=comm,
                    stream=stream,
                    src_mesh=self._m2n.Mesh((src_count,), start_rank=topology.src_ranks[0]),
                    src_placements=(self._m2n.Shard(0),),
                    src_local_shape=(1, dst_count, slot_bytes),
                    src_dtype=torch.uint8,
                    dst_mesh=self._m2n.Mesh((dst_count,), start_rank=topology.dst_ranks[0]),
                    dst_placements=(self._m2n.Shard(1),),
                    dst_local_shape=(src_count, 1, slot_bytes),
                    dst_dtype=torch.uint8,
                    handle=self._handle,
                )
            except BaseException:
                self._poisoned = True
                raise
            finally:
                # nccl.m2n enqueues work outside PyTorch's dispatcher. Teach
                # the caching allocator that the staging tensor is in use.
                buffer.record_stream(stream)

            if self.rank in topology.dst_ranks:
                self._unpack_recvs(buffer, topology, recvs, slot_bytes)
        finally:
            self.send_ops.clear()
            self.recv_ops.clear()

    def close(self) -> None:
        """Wait for local work and release M2N resources; this is not collective."""
        if self._closed:
            return
        self._closed = True
        self._active_plan = None
        self._active_transform = None
        self._slot_bytes = None
        handle, self._handle = self._handle, None
        comm, self._comm = self._comm, None
        try:
            torch.cuda.synchronize(self._device)
        finally:
            try:
                if handle is not None:
                    handle.destroy()
            finally:
                if comm is not None:
                    comm.destroy()
