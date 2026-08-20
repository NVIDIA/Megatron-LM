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

_MINIMUM_NCCL_VERSION = (2, 30, 5)


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
        self._slot_bytes_tensor = torch.empty((), dtype=torch.int64, device=self._device)
        self._buffer: torch.Tensor | None = None
        self._comm: Any | None = None
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
                "NCCL M2N model roles cannot change while reusing a copy service; "
                "create a new service or clear the service cache before changing meshes"
            )
        self._is_source = is_source
        self._is_destination = is_destination

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

    def _prepare_ops(self) -> tuple[dict[int, list[SendOp]], dict[int, list[RecvOp]], int]:
        sends = _ordered_ops_by_peer(self.send_ops, is_send=True)
        recvs = _ordered_ops_by_peer(self.recv_ops, is_send=False)
        pair_bytes = []
        for ops in sends.values():
            for op in ops:
                self._validate_tensor(op.tensor, "send")
            pair_bytes.append(sum(_tensor_nbytes(op.tensor) for op in ops))
        for ops in recvs.values():
            for op in ops:
                self._validate_tensor(op.tensor, "receive")
            pair_bytes.append(sum(_tensor_nbytes(op.tensor) for op in ops))
        return sends, recvs, max(pair_bytes, default=0)

    def _get_topology_and_slot_bytes(self, local_max_pair_bytes: int) -> tuple[_M2NTopology, int]:
        if self._is_source is None or self._is_destination is None:
            raise RuntimeError(
                "NCCLM2NCopyService model roles were not configured; call set_model_roles() "
                "or use swap_model_weights()"
            )
        topology = self._topology
        if topology is None:
            # The source and destination meshes are fixed for the service lifetime.
            state = torch.tensor(
                [int(self._is_source), int(self._is_destination), local_max_pair_bytes],
                dtype=torch.int64,
                device=self._device,
            )
            gathered = [torch.empty_like(state) for _ in range(self.world_size)]
            dist.all_gather(gathered, state, group=self.group)
            host_states = [tuple(int(value) for value in item.cpu().tolist()) for item in gathered]
            topology = _validate_role_roster(
                [(bool(is_src), bool(is_dst)) for is_src, is_dst, _size in host_states]
            )
            self._topology = topology
            slot_bytes = max(size for _is_src, _is_dst, size in host_states)
        else:
            # Payload sizes may vary, so only their scalar maximum remains per-run.
            self._slot_bytes_tensor.fill_(local_max_pair_bytes)
            dist.all_reduce(self._slot_bytes_tensor, op=dist.ReduceOp.MAX, group=self.group)
            slot_bytes = int(self._slot_bytes_tensor.item())
        return topology, slot_bytes

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

    def _ensure_buffer(self, required_size: int) -> None:
        if self._buffer is not None and self._buffer.numel() >= required_size:
            return
        self._buffer = torch.empty(required_size, dtype=torch.uint8, device=self._device)

    def _pack_sends(
        self, topology: _M2NTopology, sends: dict[int, list[SendOp]], slot_bytes: int
    ) -> None:
        if self._buffer is None:
            raise RuntimeError("NCCL M2N refit buffer is not allocated")
        self._buffer[: len(topology.dst_ranks) * slot_bytes].zero_()
        dst_index = {rank: index for index, rank in enumerate(topology.dst_ranks)}
        with torch.no_grad():
            for peer, ops in sends.items():
                offset = dst_index[peer] * slot_bytes
                for op in ops:
                    size = _tensor_nbytes(op.tensor)
                    self._buffer[offset : offset + size].copy_(_byte_view(op.tensor))
                    offset += size

    def _unpack_recvs(
        self, topology: _M2NTopology, recvs: dict[int, list[RecvOp]], slot_bytes: int
    ) -> None:
        if self._buffer is None:
            raise RuntimeError("NCCL M2N refit buffer is not allocated")
        src_index = {rank: index for index, rank in enumerate(topology.src_ranks)}
        with torch.no_grad():
            for peer, ops in recvs.items():
                offset = src_index[peer] * slot_bytes
                for op in ops:
                    size = _tensor_nbytes(op.tensor)
                    _byte_view(op.tensor).copy_(self._buffer[offset : offset + size])
                    offset += size

    def _local_m2n_buffers(
        self, topology: _M2NTopology, slot_bytes: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self._buffer is None:
            raise RuntimeError("NCCL M2N refit buffer is not allocated")
        src = None
        dst = None
        if self.rank in topology.src_ranks:
            src_size = len(topology.dst_ranks) * slot_bytes
            src = self._buffer[:src_size].view(1, len(topology.dst_ranks), slot_bytes)
        if self.rank in topology.dst_ranks:
            dst_size = len(topology.src_ranks) * slot_bytes
            dst = self._buffer[:dst_size].view(len(topology.src_ranks), 1, slot_bytes)
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
            sends, recvs, local_max_pair_bytes = self._prepare_ops()
            topology, slot_bytes = self._get_topology_and_slot_bytes(local_max_pair_bytes)
            self._validate_peers(topology, sends, recvs)
            if slot_bytes == 0:
                return

            comm = self._get_comm()
            peer_count = (
                len(topology.dst_ranks)
                if self.rank in topology.src_ranks
                else len(topology.src_ranks)
            )
            required_size = peer_count * slot_bytes
            self._ensure_buffer(required_size)
            if self.rank in topology.src_ranks:
                self._pack_sends(topology, sends, slot_bytes)

            src, dst = self._local_m2n_buffers(topology, slot_bytes)
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
                if self._buffer is not None:
                    self._buffer.record_stream(stream)

            if self.rank in topology.dst_ranks:
                self._unpack_recvs(topology, recvs, slot_bytes)
        finally:
            self.send_ops.clear()
            self.recv_ops.clear()

    def close(self) -> None:
        """Collectively drain work and release the official M2N runtime."""
        if self._closed:
            return
        torch.cuda.synchronize(self._device)
        self._buffer = None
        self._handle.destroy()
        self._handle = None
        if self._comm is not None:
            self._comm.destroy()
        self._comm = None
        self._closed = True
