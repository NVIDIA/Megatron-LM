# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import torch
import torch.distributed as dist

from .base import CopyService, RecvOp, SendOp

logger = logging.getLogger(__name__)

_TransferOpT = TypeVar("_TransferOpT", SendOp, RecvOp)

_NCCL_SUCCESS = 0
_NCCL_UINT8 = 1
_NCCL_WIN_COLL_SYMMETRIC = 0x01
_NCCL_RESHARD_REPLICATE = -1
_NCCL_RESHARD_MAX_TENSOR_DIMS = 3
_NCCL_RESHARD_ELEMENTS_PER_CHUNK = 32
_NCCL_RESHARD_RING_MAX_SOURCES = 16
_NCCL_RESHARD_RING_MAX_TARGETS = 64
_NCCL_RESHARD_DIRECT_MAX_SOURCES = 32
_NCCL_RESHARD_DIRECT_MAX_TARGETS = 64


class _NcclMesh(ctypes.Structure):
    _fields_ = [
        ("dims", ctypes.c_int * 2),
        ("startRank", ctypes.c_int),
        ("placement", ctypes.c_int * 2),
    ]


class _NcclDistTensor(ctypes.Structure):
    _fields_ = [
        ("dataPtr", ctypes.c_void_p),
        ("localShape", ctypes.c_size_t * _NCCL_RESHARD_MAX_TENSOR_DIMS),
        ("ndims", ctypes.c_int),
        ("dtype", ctypes.c_int),
        ("mesh", ctypes.POINTER(_NcclMesh)),
    ]


class _NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


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


def _validate_mesh_limits(topology: _M2NTopology, algorithm: str) -> None:
    """Reject meshes that would trip NCCL M2N v0.1's fatal static bounds."""
    algorithm = algorithm.upper()
    if algorithm == "DIRECT":
        max_sources = _NCCL_RESHARD_DIRECT_MAX_SOURCES
        max_targets = _NCCL_RESHARD_DIRECT_MAX_TARGETS
    else:
        # AUTO aliases to RING in v0.1; invalid values also fall back to the
        # library default, so every non-DIRECT value uses the safer RING caps.
        max_sources = _NCCL_RESHARD_RING_MAX_SOURCES
        max_targets = _NCCL_RESHARD_RING_MAX_TARGETS

    src_count = len(topology.src_ranks)
    dst_count = len(topology.dst_ranks)
    if src_count > max_sources or dst_count > max_targets:
        raise RuntimeError(
            f"NCCL M2N {algorithm or 'AUTO'} mesh exceeds the v0.1 static limits: "
            f"{src_count} source ranks (max {max_sources}) and {dst_count} destination ranks "
            f"(max {max_targets}). Use a supported mesh; if the library was rebuilt with larger "
            "reshard_limits.h bounds, construct the service with enforce_mesh_limits=False."
        )


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


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _byte_view(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().reshape(-1).view(torch.uint8)


def _m2n_library_candidates(explicit_path: str | os.PathLike | None) -> list[str]:
    candidates: list[str] = []

    def add_path(path: str | os.PathLike | None) -> None:
        if path is None:
            return
        candidate = Path(path).expanduser()
        if candidate.is_dir():
            candidate = candidate / "libnccl_m2n.so"
        candidates.append(str(candidate))

    add_path(explicit_path)
    add_path(os.getenv("NCCL_M2N_LIBRARY"))
    home = os.getenv("NCCL_M2N_HOME")
    if home:
        add_path(Path(home) / "lib" / "libnccl_m2n.so")
    discovered = ctypes.util.find_library("nccl_m2n")
    if discovered:
        candidates.append(discovered)
    candidates.append("libnccl_m2n.so")
    return list(dict.fromkeys(candidates))


def _load_m2n_library(explicit_path: str | os.PathLike | None) -> tuple[ctypes.CDLL, str]:
    errors = []
    for candidate in _m2n_library_candidates(explicit_path):
        try:
            return ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL), candidate
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    details = "; ".join(errors)
    raise RuntimeError(
        "NCCL M2N refit requires libnccl_m2n.so. Build/install NVIDIA/nccl's "
        "contrib/nccl_m2n library and set NCCL_M2N_LIBRARY to the shared-library path "
        f"(load attempts: {details})"
    )


class _NCCLM2NBindings:
    """Typed ctypes bindings for NCCL memory windows and NCCL M2N."""

    def __init__(self, library_path: str | os.PathLike | None = None):
        nccl_path = ctypes.util.find_library("nccl") or "libnccl.so.2"
        try:
            self._nccl = ctypes.CDLL(nccl_path, mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            raise RuntimeError(f"NCCL M2N refit could not load NCCL ({nccl_path}): {exc}") from exc
        self._m2n, self.library_path = _load_m2n_library(library_path)

        self._nccl.ncclGetErrorString.argtypes = [ctypes.c_int]
        self._nccl.ncclGetErrorString.restype = ctypes.c_char_p
        self._nccl.ncclGetUniqueId.argtypes = [ctypes.POINTER(_NcclUniqueId)]
        self._nccl.ncclGetUniqueId.restype = ctypes.c_int
        self._nccl.ncclCommInitRank.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            _NcclUniqueId,
            ctypes.c_int,
        ]
        self._nccl.ncclCommInitRank.restype = ctypes.c_int
        self._nccl.ncclCommDestroy.argtypes = [ctypes.c_void_p]
        self._nccl.ncclCommDestroy.restype = ctypes.c_int
        self._nccl.ncclMemAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._nccl.ncclMemAlloc.restype = ctypes.c_int
        self._nccl.ncclMemFree.argtypes = [ctypes.c_void_p]
        self._nccl.ncclMemFree.restype = ctypes.c_int
        self._nccl.ncclCommWindowRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
        ]
        self._nccl.ncclCommWindowRegister.restype = ctypes.c_int
        self._nccl.ncclCommWindowDeregister.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._nccl.ncclCommWindowDeregister.restype = ctypes.c_int

        self._m2n.ncclM2nInit.argtypes = [ctypes.c_void_p]
        self._m2n.ncclM2nInit.restype = ctypes.c_int
        self._m2n.ncclM2nFinalize.argtypes = []
        self._m2n.ncclM2nFinalize.restype = ctypes.c_int
        self._m2n.ncclReshardWithWindow.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(_NcclDistTensor),
            ctypes.POINTER(_NcclDistTensor),
            ctypes.c_void_p,
        ]
        self._m2n.ncclReshardWithWindow.restype = ctypes.c_int
        self._check(self._m2n.ncclM2nInit(None), "ncclM2nInit")

    def _check(self, result: int, operation: str) -> None:
        if result == _NCCL_SUCCESS:
            return
        error = self._nccl.ncclGetErrorString(result)
        error_text = error.decode("utf-8", errors="replace") if error else f"error {result}"
        raise RuntimeError(f"NCCL M2N refit: {operation} failed: {error_text}")

    def allocate(self, size: int) -> int:
        """Allocate symmetric NCCL device memory and return its address."""
        ptr = ctypes.c_void_p()
        self._check(self._nccl.ncclMemAlloc(ctypes.byref(ptr), size), "ncclMemAlloc")
        if not ptr.value:
            raise RuntimeError("NCCL M2N refit: ncclMemAlloc returned a null pointer")
        return ptr.value

    def get_unique_id(self) -> bytes:
        """Create a serialized NCCL communicator ID."""
        unique_id = _NcclUniqueId()
        self._check(self._nccl.ncclGetUniqueId(ctypes.byref(unique_id)), "ncclGetUniqueId")
        return ctypes.string_at(ctypes.byref(unique_id), ctypes.sizeof(unique_id))

    def init_comm(self, world_size: int, rank: int, unique_id_bytes: bytes) -> int:
        """Create a blocking NCCL communicator from a serialized ID."""
        if len(unique_id_bytes) != ctypes.sizeof(_NcclUniqueId):
            raise RuntimeError(
                f"NCCL M2N refit received a {len(unique_id_bytes)}-byte NCCL unique ID; "
                f"expected {ctypes.sizeof(_NcclUniqueId)} bytes"
            )
        unique_id = _NcclUniqueId.from_buffer_copy(unique_id_bytes)
        comm = ctypes.c_void_p()
        self._check(
            self._nccl.ncclCommInitRank(ctypes.byref(comm), world_size, unique_id, rank),
            "ncclCommInitRank",
        )
        if not comm.value:
            raise RuntimeError("NCCL M2N refit: ncclCommInitRank returned a null communicator")
        return comm.value

    def destroy_comm(self, comm: int) -> None:
        """Destroy a communicator created by :meth:`init_comm`."""
        self._check(self._nccl.ncclCommDestroy(ctypes.c_void_p(comm)), "ncclCommDestroy")

    def free(self, ptr: int) -> None:
        """Free symmetric NCCL device memory."""
        self._check(self._nccl.ncclMemFree(ctypes.c_void_p(ptr)), "ncclMemFree")

    def register_window(self, comm: int, ptr: int, size: int) -> int:
        """Collectively register a symmetric-memory window."""
        window = ctypes.c_void_p()
        self._check(
            self._nccl.ncclCommWindowRegister(
                ctypes.c_void_p(comm),
                ctypes.c_void_p(ptr),
                size,
                ctypes.byref(window),
                _NCCL_WIN_COLL_SYMMETRIC,
            ),
            "ncclCommWindowRegister",
        )
        if not window.value:
            raise RuntimeError("NCCL M2N refit: ncclCommWindowRegister returned a null window")
        return window.value

    def deregister_window(self, comm: int, window: int) -> None:
        """Collectively deregister a symmetric-memory window."""
        self._check(
            self._nccl.ncclCommWindowDeregister(ctypes.c_void_p(comm), ctypes.c_void_p(window)),
            "ncclCommWindowDeregister",
        )

    def reshard(
        self, comm: int, window: int, src: _NcclDistTensor, dst: _NcclDistTensor, stream: int
    ) -> None:
        """Enqueue one window-backed M2N reshard on a CUDA stream."""
        self._check(
            self._m2n.ncclReshardWithWindow(
                ctypes.c_void_p(comm),
                ctypes.c_void_p(window),
                ctypes.byref(src),
                ctypes.byref(dst),
                ctypes.c_void_p(stream),
            ),
            "ncclReshardWithWindow",
        )

    def finalize(self) -> None:
        """Release process-global NCCL M2N resources."""
        self._check(self._m2n.ncclM2nFinalize(), "ncclM2nFinalize")


_m2n_lock = threading.Lock()
_shared_bindings: _NCCLM2NBindings | None = None
_shared_bindings_users = 0
_pending_comms: list[int] = []


def _acquire_shared_bindings(library_path: str | os.PathLike | None) -> _NCCLM2NBindings:
    global _shared_bindings, _shared_bindings_users
    # NCCL M2N v0.1 requires process-wide serialization of lifecycle and
    # reshard calls, including calls that use different communicators.
    with _m2n_lock:
        if _shared_bindings is None:
            _shared_bindings = _NCCLM2NBindings(library_path)
        elif library_path is not None:
            requested = str(Path(library_path).expanduser())
            loaded = str(Path(_shared_bindings.library_path).expanduser())
            if requested != loaded:
                raise RuntimeError(
                    "NCCL M2N is process-global and was already loaded from "
                    f"'{loaded}', so it cannot be reloaded from '{requested}'"
                )
        _shared_bindings_users += 1
        return _shared_bindings


def _release_shared_bindings(bindings: _NCCLM2NBindings, comm: int | None) -> None:
    global _shared_bindings, _shared_bindings_users
    with _m2n_lock:
        if bindings is not _shared_bindings:
            return
        if comm is not None:
            _pending_comms.append(comm)
        _shared_bindings_users -= 1
        if _shared_bindings_users == 0:
            bindings.finalize()
            for pending_comm in _pending_comms:
                bindings.destroy_comm(pending_comm)
            _pending_comms.clear()
            _shared_bindings = None


class NCCLM2NCopyService(CopyService):
    """Hierarchical non-collocated ReFIT transport backed by NCCL M2N.

    The generic ReFIT planner emits point-to-point slices. This service packs
    those slices into a dense ``[source, destination, bytes]`` logical tensor:
    source ranks shard dimension 0 and destination ranks shard dimension 1.
    One cross-dimension ``ncclReshardWithWindow`` call then moves the complete
    batch through NCCL M2N's hierarchical ring, after which receive slices are
    unpacked into their original tensors.

    NCCL M2N v0.1 requires disjoint, contiguous source and destination meshes,
    so this backend supports non-collocated ReFIT only. The process group must
    contain source ranks first, destination ranks second, and no idle ranks.

    Args:
        group: NCCL process group containing exactly the source and destination ranks.
        library_path: Optional explicit path to ``libnccl_m2n.so``.
        enforce_mesh_limits: Enforce the stock v0.1 compile-time mesh caps. Disable only
            when using a library rebuilt with larger ``reshard_limits.h`` bounds.
    """

    requires_process_group_barrier = False

    def __init__(
        self,
        group=None,
        library_path: str | os.PathLike | None = None,
        *,
        enforce_mesh_limits: bool = True,
        _bindings: _NCCLM2NBindings | None = None,
    ):
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before NCCLM2NCopyService()")
        if not torch.cuda.is_available():
            raise RuntimeError("NCCLM2NCopyService requires CUDA")
        super().__init__(group=group)

        self._device = torch.device("cuda", torch.cuda.current_device())
        if dist.get_backend(group) != "nccl":
            raise RuntimeError("NCCLM2NCopyService requires an NCCL process group")
        self._process_group = group if group is not None else dist.group.WORLD

        self._uses_shared_bindings = _bindings is None
        self._bindings = _acquire_shared_bindings(library_path) if _bindings is None else _bindings
        self._enforce_mesh_limits = enforce_mesh_limits
        self._is_source: bool | None = None
        self._is_destination: bool | None = None
        self._buffer_ptr: int | None = None
        self._buffer_storage = None
        self._buffer: torch.Tensor | None = None
        self._buffer_size = 0
        self._window: int | None = None
        self._comm: int | None = None
        self._closed = False
        self.send_ops: list[SendOp] = []
        self.recv_ops: list[RecvOp] = []
        logger.info(
            "NCCLM2NCopyService initialized on rank %d/%d with %s",
            self.rank,
            self.world_size,
            getattr(self._bindings, "library_path", "injected bindings"),
        )

    def set_model_roles(self, *, is_source: bool, is_destination: bool) -> None:
        """Set this rank's source/destination participation for the next run."""
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

    def _collect_topology_and_slot_bytes(
        self, local_max_pair_bytes: int
    ) -> tuple[_M2NTopology, int]:
        if self._is_source is None or self._is_destination is None:
            raise RuntimeError(
                "NCCLM2NCopyService model roles were not configured; call set_model_roles() "
                "or use swap_model_weights()"
            )
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
        if self._enforce_mesh_limits:
            _validate_mesh_limits(topology, os.getenv("NCCL_RESHARD_ALGORITHM", "AUTO"))
        max_pair_bytes = max(size for _is_src, _is_dst, size in host_states)
        # M2N's RING kernel advances in DEFAULT_ELEMENTS_PER_CHUNK elements.
        # Padding the uint8 slot keeps its innermost dimension chunk-aligned;
        # older preview builds can otherwise stall on a partial final chunk.
        slot_bytes = _align_up(max_pair_bytes, _NCCL_RESHARD_ELEMENTS_PER_CHUNK)
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

    def _get_comm(self) -> int:
        if self._comm is not None:
            return self._comm

        if self.rank == 0:
            unique_id = self._bindings.get_unique_id()
            unique_id_tensor = torch.tensor(list(unique_id), dtype=torch.uint8, device=self._device)
        else:
            unique_id_tensor = torch.empty(
                ctypes.sizeof(_NcclUniqueId), dtype=torch.uint8, device=self._device
            )
        src_rank = 0 if self.group is None else dist.get_global_rank(self._process_group, 0)
        dist.broadcast(unique_id_tensor, src=src_rank, group=self.group)
        unique_id = bytes(unique_id_tensor.cpu().tolist())
        torch.cuda.synchronize(self._device)
        self._comm = self._bindings.init_comm(self.world_size, self.rank, unique_id)
        return self._comm

    def _release_window(self) -> None:
        if self._window is None:
            return
        if self._comm is None or self._buffer_ptr is None:
            raise RuntimeError("NCCL M2N refit window state is inconsistent")
        torch.cuda.synchronize(self._device)
        self._bindings.deregister_window(self._comm, self._window)
        ptr = self._buffer_ptr
        self._window = None
        self._buffer = None
        self._buffer_storage = None
        self._buffer_ptr = None
        self._buffer_size = 0
        self._bindings.free(ptr)

    def _ensure_window(self, comm: int, required_size: int) -> None:
        if self._buffer_size >= required_size:
            return
        if self._window is not None:
            self._release_window()

        ptr = self._bindings.allocate(required_size)
        try:
            storage = torch._C._construct_storage_from_data_pointer(
                ptr, self._device, required_size
            )
            buffer = torch.empty(0, dtype=torch.uint8, device=self._device).set_(
                storage, 0, (required_size,), (1,)
            )
            window = self._bindings.register_window(comm, ptr, required_size)
        except Exception:
            self._bindings.free(ptr)
            raise

        self._buffer_ptr = ptr
        self._buffer_storage = storage
        self._buffer = buffer
        self._buffer_size = required_size
        self._window = window

    def _pack_sends(
        self, topology: _M2NTopology, sends: dict[int, list[SendOp]], slot_bytes: int
    ) -> None:
        if self._buffer is None:
            raise RuntimeError("NCCL M2N refit buffer is not allocated")
        self._buffer.zero_()
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

    def _build_descriptors(
        self, topology: _M2NTopology, slot_bytes: int
    ) -> tuple[_NcclMesh, _NcclMesh, _NcclDistTensor, _NcclDistTensor]:
        src_count = len(topology.src_ranks)
        dst_count = len(topology.dst_ranks)

        src_mesh = _NcclMesh()
        src_mesh.dims[:] = (1, src_count)
        src_mesh.startRank = topology.src_ranks[0]
        src_mesh.placement[:] = (_NCCL_RESHARD_REPLICATE, 0)

        dst_mesh = _NcclMesh()
        dst_mesh.dims[:] = (1, dst_count)
        dst_mesh.startRank = topology.dst_ranks[0]
        dst_mesh.placement[:] = (_NCCL_RESHARD_REPLICATE, 1)

        src = _NcclDistTensor()
        src.dataPtr = self._buffer_ptr if self.rank in topology.src_ranks else None
        src.localShape[:] = (1, dst_count, slot_bytes)
        src.ndims = 3
        src.dtype = _NCCL_UINT8
        src.mesh = ctypes.pointer(src_mesh)

        dst = _NcclDistTensor()
        dst.dataPtr = self._buffer_ptr if self.rank in topology.dst_ranks else None
        dst.localShape[:] = (src_count, 1, slot_bytes)
        dst.ndims = 3
        dst.dtype = _NCCL_UINT8
        dst.mesh = ctypes.pointer(dst_mesh)
        return src_mesh, dst_mesh, src, dst

    def run(self) -> None:
        """Pack and execute all queued transfers in one hierarchical M2N reshard."""
        if self._closed:
            raise RuntimeError("NCCLM2NCopyService is closed")
        if torch.cuda.current_device() != self._device.index:
            raise RuntimeError(
                f"NCCL M2N service was created on {self._device}, but the current CUDA device is "
                f"cuda:{torch.cuda.current_device()}"
            )

        try:
            sends, recvs, local_max_pair_bytes = self._prepare_ops()
            topology, slot_bytes = self._collect_topology_and_slot_bytes(local_max_pair_bytes)
            self._validate_peers(topology, sends, recvs)
            if slot_bytes == 0:
                return

            comm = self._get_comm()
            required_size = max(len(topology.src_ranks), len(topology.dst_ranks)) * slot_bytes
            self._ensure_window(comm, required_size)
            if self._window is None:
                raise RuntimeError("NCCL M2N refit window registration did not produce a handle")
            if self.rank in topology.src_ranks:
                self._pack_sends(topology, sends, slot_bytes)

            # Keep mesh objects alive until the ctypes call returns; the tensor
            # descriptors contain raw pointers to them.
            src_mesh, dst_mesh, src, dst = self._build_descriptors(topology, slot_bytes)
            stream = torch.cuda.current_stream(self._device).cuda_stream
            with _m2n_lock:
                self._bindings.reshard(comm, self._window, src, dst, stream)
            del src_mesh, dst_mesh

            if self.rank in topology.dst_ranks:
                self._unpack_recvs(topology, recvs, slot_bytes)
        finally:
            self.send_ops.clear()
            self.recv_ops.clear()

    def close(self) -> None:
        """Collectively release the NCCL window and process-global M2N state."""
        if self._closed:
            return
        self._release_window()
        if self._uses_shared_bindings:
            _release_shared_bindings(self._bindings, self._comm)
        elif self._comm is not None:
            self._bindings.destroy_comm(self._comm)
        self._comm = None
        self._closed = True
