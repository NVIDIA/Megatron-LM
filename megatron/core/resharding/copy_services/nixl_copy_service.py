# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from .base import CopyService, RecvOp, SendOp, match_local_ops_by_task_id

logger = logging.getLogger(__name__)

# Transports that let UCX read/write CUDA memory. Without one of these UCX sees
# a GPU pointer as host memory and segfaults mid-transfer.
_CUDA_UCX_TRANSPORTS = ("cuda_copy", "cuda_ipc")

# (addr, len_bytes, device_id) for a registered region. Exchanged between ranks
# so a sender can WRITE straight into a receiver's destination buffer.
_MemDesc = Tuple[int, int, int]


def _ensure_cuda_ucx_transports() -> None:
    """Add cuda transports to UCX_TLS if it's pinned to a host-only allowlist.

    UCX reads UCX_TLS once at agent init. Deployments often set it to e.g. "tcp",
    which can't touch GPU memory. Only augment a plain inclusion list; leave an
    unset value (defaults to "all") or an exclusion list ("^...") alone.
    """
    tls = os.environ.get("UCX_TLS")
    if not tls:
        return
    tokens = [t.strip() for t in tls.split(",") if t.strip()]
    if not tokens or tokens[0].startswith("^") or any("cuda" in t for t in tokens):
        return
    os.environ["UCX_TLS"] = tls + "," + ",".join(_CUDA_UCX_TRANSPORTS)
    logger.warning("UCX_TLS=%r has no cuda transport; using %r", tls, os.environ["UCX_TLS"])


class NixlCopyService(CopyService):
    """Refit transport over NIXL (UCX/RDMA), for cross-cluster non-collocated refit.

    Each rank runs a NIXL agent. To WRITE into a peer it needs that peer's agent
    metadata (connection info) plus a {task_id: (addr, len, dev)} map of the peer's
    registered recv buffers. A one-time torch all-gather builds and caches this
    peer table. Buffers are registered locally, not exchanged.

    Every refit after that is pure NIXL and sender-driven: a receiver signals each
    source that its buffers are free ("ready"); the source waits, syncs its weights,
    and issues one WRITE per receiver, each carrying a "data" notification. Those two
    notifications order producer and consumer per refit, so there's no barrier and no
    per-refit collective. Notifications are tagged with a per-refit sequence, so stale
    ones are ignored. Same-rank transfers skip NIXL and copy directly.

    Registered buffers are assumed address-stable across refits; if a recv address
    changes after setup, call clear_service_cache() to rebuild.
    """

    requires_process_group_barrier = False

    def __init__(self, group=None, agent_name: Optional[str] = None):
        super().__init__(group=group)
        # Object collectives (the one-time handshake) run on this group to support
        # cross-world PGs.
        self.group = group

        try:
            from nixl._api import nixl_agent, nixl_agent_config
        except ImportError as e:
            raise ImportError(
                "NixlCopyService requires the 'nixl' package; install it or use "
                "another refit backend (nccl/gloo/nvshmem)."
            ) from e

        _ensure_cuda_ucx_transports()

        # Name by group rank, which is unique across the (possibly cross-world)
        # group. dist.get_rank() would collide: separate worlds each have a rank 0.
        self.agent_name = agent_name or f"refit-nixl-rank-{self.rank}"
        # This backend is intentionally UCX-only: refit moves CPU/CUDA tensors
        # directly between ranks and does not use NIXL's storage-oriented plugins.
        self.agent = nixl_agent(self.agent_name, nixl_agent_config(backends=["UCX"]))

        self.send_ops: List[SendOp] = []
        self.recv_ops: List[RecvOp] = []
        self._copy_stream = torch.cuda.Stream()

        # Peer table {group rank -> (agent_name, agent_metadata, recv_descs)},
        # populated by the first run's handshake.
        self._remote_agent_names: Dict[int, str] = {}  # group rank -> agent name
        self._gathered: Optional[Dict[int, tuple]] = None
        self._recv_descs: Optional[Dict[int, _MemDesc]] = None
        # RDMA registrations, kept across refits; send and recv tracked separately
        # so a changing side doesn't re-pin the other.
        self._reg: Dict[str, tuple] = {}  # 'send'/'recv' -> (handle, signature)
        # Notifications are tagged (kind, seq). _future_notifs buffers tags for
        # later refits that we're not draining yet.
        self._seq = 0
        self._future_notifs: Dict[Tuple[str, int], int] = {}

    def submit_send(self, src_tensor: torch.Tensor, dest_rank: int, task_id: Optional[int] = None):
        self.send_ops.append(SendOp(task_id=task_id, tensor=src_tensor, dest_rank=dest_rank))

    def submit_recv(self, dest_tensor: torch.Tensor, src_rank: int, task_id: Optional[int] = None):
        self.recv_ops.append(RecvOp(task_id=task_id, tensor=dest_tensor, src_rank=src_rank))

    @staticmethod
    def _mem_desc(tensor: torch.Tensor) -> _MemDesc:
        if not tensor.is_contiguous():
            raise RuntimeError("NixlCopyService requires contiguous tensors")
        dev = tensor.get_device()  # -1 for host tensors; NIXL addresses DRAM as device 0
        return (tensor.data_ptr(), tensor.numel() * tensor.element_size(), dev if dev >= 0 else 0)

    def _register(self, which: str, tensors: List[torch.Tensor]) -> None:
        # Re-register only when the set of regions changes.
        sig = tuple((t.data_ptr(), t.numel() * t.element_size()) for t in tensors)
        cached = self._reg.get(which)
        if cached is not None and cached[1] == sig:
            return
        if cached is not None and cached[0] is not None:
            self.agent.deregister_memory(cached[0])
        handle = self.agent.register_memory(tensors) if tensors else None
        self._reg[which] = (handle, sig)

    def _handshake(self, recv_descs: Dict[int, _MemDesc]) -> None:
        # Initial bootstrap over the torch group: share agent metadata and recv
        # descriptors, connect to every peer, and cache the peer table. This is
        # NIXL's one torch collective.
        payload = (self.agent_name, self.agent.get_agent_metadata(), recv_descs)
        gathered: List[Optional[tuple]] = [None] * self.world_size
        dist.all_gather_object(gathered, payload, group=self.group)
        self._gathered = {rank: entry for rank, entry in enumerate(gathered) if entry is not None}
        for rank, (name, metadata, _descs) in self._gathered.items():
            if rank != self.rank:
                self._remote_agent_names[rank] = self.agent.add_remote_agent(metadata) or name
        self._recv_descs = recv_descs

    def _do_local_copies(self) -> None:
        # Collocated (same-rank) transfers never hit the network.
        local_sends = [op for op in self.send_ops if op.dest_rank == self.rank]
        local_recvs = [op for op in self.recv_ops if op.src_rank == self.rank]
        if not local_sends and not local_recvs:
            return
        pairs = match_local_ops_by_task_id(local_sends, local_recvs, "NixlCopyService", self.rank)
        with torch.no_grad(), torch.cuda.stream(self._copy_stream):
            for send_op, recv_op in pairs:
                recv_op.tensor.copy_(send_op.tensor)

    def _plan_writes(self, remote_sends: List[SendOp]):
        # Group writes by destination agent: one WRITE per receiver pushes all its
        # regions at once, with local[i] landing in remote[i].
        if self._gathered is None:
            raise RuntimeError("NixlCopyService: handshake has not completed")
        by_dst: Dict[int, Tuple[List[torch.Tensor], List[_MemDesc]]] = {}
        for op in remote_sends:
            dst_entry = self._gathered.get(op.dest_rank)
            if dst_entry is None:
                raise RuntimeError(f"NixlCopyService: no metadata from dst rank {op.dest_rank}")
            remote_desc = dst_entry[2].get(op.task_id)
            if remote_desc is None:
                raise RuntimeError(
                    f"NixlCopyService: dst rank {op.dest_rank} missing task_id {op.task_id}"
                )
            local_list, remote_list = by_dst.setdefault(op.dest_rank, ([], []))
            local_list.append(op.tensor)
            remote_list.append(remote_desc)
        return by_dst

    @staticmethod
    def _notif(kind: str, seq: int) -> bytes:
        return f"{kind}{seq}".encode()

    @staticmethod
    def _parse_notif(m: bytes) -> Tuple[str, int]:
        return chr(m[0]), int(m[1:])

    def _await_notifs(self, kind: str, expected: int, seq: int) -> None:
        # Wait for `expected` notifications of this kind ('R' ready / 'D' data)
        # tagged with this refit. Buffer any tagged otherwise — e.g. a data notif
        # arriving while we're still collecting ready signals.
        if expected == 0:
            return
        want = (kind, seq)
        got = self._future_notifs.pop(want, 0)
        while got < expected:
            for _agent, msgs in self.agent.get_new_notifs().items():
                for m in msgs:
                    tag = self._parse_notif(m)
                    if tag == want:
                        got += 1
                    else:
                        self._future_notifs[tag] = self._future_notifs.get(tag, 0) + 1
            if got < expected:
                time.sleep(0)  # NIXL delivers notifs on its own thread

    def run(self):
        remote_sends = [op for op in self.send_ops if op.dest_rank != self.rank]
        remote_recvs = [op for op in self.recv_ops if op.src_rank != self.rank]
        seq = self._seq

        # Overlaps with the writes below.
        self._do_local_copies()

        # Both sides of a WRITE must be registered: our send tensors (we read them)
        # and our recv buffers (peers write into them).
        self._register("send", [op.tensor for op in remote_sends])
        self._register("recv", [op.tensor for op in remote_recvs])

        recv_descs = {op.task_id: self._mem_desc(op.tensor) for op in remote_recvs}
        if self._gathered is None:
            self._handshake(recv_descs)
        elif recv_descs != self._recv_descs:
            raise RuntimeError(
                "NixlCopyService: recv tensor addresses changed after setup; "
                "call clear_service_cache() to rebuild the handshake."
            )

        # Tell each source our destination buffers are free for this refit; this is
        # what orders a source's write after we've consumed the previous one.
        ready = self._notif("R", seq)
        for src_rank in {op.src_rank for op in remote_recvs}:
            self.agent.send_notif(self._remote_agent_names[src_rank], ready)

        # As a source: wait for every receiver's ready, then push once the weights
        # are finished (a peer must not read a half-written buffer).
        self._await_notifs("R", len({op.dest_rank for op in remote_sends}), seq)
        if remote_sends:
            torch.cuda.current_stream().synchronize()

        data = self._notif("D", seq)
        handles = []
        for dst_rank, (local_tensors, remote_descs) in self._plan_writes(remote_sends).items():
            # local and remote are the same memory class (GPU->GPU for refit); the
            # remote tuples carry no type, so take it from the send tensor.
            mem_type = "cuda" if local_tensors[0].is_cuda else "cpu"
            local_xfer = self.agent.get_xfer_descs(local_tensors)
            remote_xfer = self.agent.get_xfer_descs(remote_descs, mem_type=mem_type)
            handle = self.agent.initialize_xfer(
                "WRITE", local_xfer, remote_xfer, self._remote_agent_names[dst_rank], notif_msg=data
            )
            if self.agent.transfer(handle) == "ERR":
                raise RuntimeError(f"NixlCopyService: WRITE to dst {dst_rank} failed to start")
            handles.append((dst_rank, handle))

        # NIXL's asynchronous Python API exposes transfer completion through
        # check_xfer_state(); the official examples poll it until DONE. Transfers
        # progress in the backend, so yield the Python thread between checks.
        for dst_rank, handle in handles:
            state = self.agent.check_xfer_state(handle)
            while state not in ("DONE", "ERR"):
                time.sleep(0)
                state = self.agent.check_xfer_state(handle)
            if state == "ERR":
                raise RuntimeError(f"NixlCopyService: WRITE to dst {dst_rank} errored")
            self.agent.release_xfer_handle(handle)
        self._await_notifs("D", len({op.src_rank for op in remote_recvs}), seq)

        torch.cuda.current_stream().wait_stream(self._copy_stream)
        self._seq += 1
        self.send_ops.clear()
        self.recv_ops.clear()

    def close(self) -> None:
        for handle, _sig in self._reg.values():
            if handle is not None:
                self.agent.deregister_memory(handle)
        self._reg.clear()
        for name in self._remote_agent_names.values():
            try:
                self.agent.remove_remote_agent(name)
            except Exception:
                pass
        self._remote_agent_names.clear()
        self._gathered = None
        self._recv_descs = None
