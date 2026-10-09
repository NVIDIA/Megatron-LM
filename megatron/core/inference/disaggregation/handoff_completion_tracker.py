# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU control-plane aggregation for model-parallel handoff completion."""

import socket
import struct
from typing import Any

import torch.distributed as dist

try:
    import zmq
except ImportError:
    from unittest.mock import MagicMock

    zmq = MagicMock()
    HAVE_ZMQ = False
else:
    HAVE_ZMQ = True


class HandoffCompletionTracker:
    """Aggregate per-rank transfer results at the model-parallel coordinator."""

    _REPORT_FORMAT = "!qi??"

    def __init__(self, zmq_context, process_group: dist.ProcessGroup, hostname: str | None = None):
        if not HAVE_ZMQ:
            raise ImportError("pyzmq is required for disaggregated KV handoff")

        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.is_coordinator = self.rank == 0
        self._reports: dict[int, dict[int, tuple[bool, bool]]] = {}  # req -> rank -> (failed, safe)
        self._failure_notified: set[int] = set()
        self._socket: Any = None
        self.sockets = []

        if self.world_size == 1:
            return

        coordinator_rank = dist.get_process_group_ranks(process_group)[0]
        if self.is_coordinator:
            self._socket = zmq_context.socket(zmq.PULL)
            self._socket.bind_to_random_port(f"tcp://{hostname or socket.gethostname()}")
            address = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)
            dist.broadcast_object_list([address], src=coordinator_rank, group=process_group)
        else:
            address_holder = [None]
            dist.broadcast_object_list(address_holder, src=coordinator_rank, group=process_group)
            self._socket = zmq_context.socket(zmq.PUSH)
            self._socket.connect(address_holder[0])
        self.sockets.append(self._socket)

    def report(self, request_id: int, failed: bool, source_safe: bool) -> None:
        """Report this rank's transfer result and source-storage safety."""

        if self.is_coordinator:
            self._record(request_id, self.rank, failed, source_safe)
            return
        self._socket.send(
            struct.pack(self._REPORT_FORMAT, request_id, self.rank, failed, source_safe)
        )

    def drain_completed(self) -> list[tuple[int, bool, bool]]:
        """Return requests that failed or completed on every model-parallel rank."""

        if not self.is_coordinator:
            return []
        if self._socket is not None:
            while True:
                try:
                    request_id, rank, failed, source_safe = struct.unpack(
                        self._REPORT_FORMAT, self._socket.recv(flags=zmq.NOBLOCK)
                    )
                    self._record(request_id, rank, failed, source_safe)
                except zmq.Again:
                    break

        completed = []
        for request_id, reports in list(self._reports.items()):
            failed = any(report[0] for report in reports.values())
            source_safe = len(reports) == self.world_size and all(
                report[1] for report in reports.values()
            )
            if failed and request_id not in self._failure_notified:
                completed.append((request_id, True, source_safe))
                self._failure_notified.add(request_id)
            elif failed and source_safe:
                completed.append((request_id, True, True))
            elif not failed and source_safe:
                completed.append((request_id, False, True))
            if source_safe:
                del self._reports[request_id]
                self._failure_notified.discard(request_id)
        return completed

    def _record(self, request_id: int, rank: int, failed: bool, source_safe: bool) -> None:
        reports = self._reports.setdefault(request_id, {})
        previous = reports.get(rank)
        if previous is None:
            reports[rank] = (failed, source_safe)
            return
        if previous[0] != failed:
            raise RuntimeError(
                f"Conflicting KV handoff results for request {request_id} from rank {rank}"
            )
        if previous[1] and not source_safe:
            raise RuntimeError(
                f"KV handoff source safety regressed for request {request_id} from rank {rank}"
            )
        reports[rank] = (failed, previous[1] or source_safe)
