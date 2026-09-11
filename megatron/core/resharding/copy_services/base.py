# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.distributed as dist


@dataclass
class SendOp:
    """Single send operation pending in a CopyService queue."""

    task_id: int | None
    tensor: torch.Tensor
    dest_rank: int


@dataclass
class RecvOp:
    """Single receive operation pending in a CopyService queue."""

    task_id: int | None
    tensor: torch.Tensor
    src_rank: int


class CopyService(ABC):
    """Abstract interface for submitting and executing batched P2P copy operations.

    All backends accept an optional *task_id* on submit calls.  The task_id is
    a globally unique identifier shared between the matching send and recv for
    the same transfer.  It is required for local (same-rank) copy matching and
    for the NVSHMEM backend's scheduling.  Backends that do not need it for
    remote transfers simply ignore it.
    """

    # Most torch.distributed backends retain the executor's historical
    # process-group rendezvous after run(). Backends whose run() protocol already
    # establishes completion across every participating peer can opt out.
    requires_process_group_barrier = True
    # Point-to-point transports tolerate ranks with no local operations. Collective
    # transports whose mesh must cover the whole process group can opt out.
    supports_idle_ranks = True
    # Multiple runs per plan are opt-in so existing third-party services retain
    # the historical model-wide run(). Supporting them requires run() to release
    # submitted tensor references and establish current-stream ordering for all
    # submitted sends and recvs before returning.
    supports_multiple_runs_per_plan = False

    def __init__(self, group=None):
        self.group = group
        # group.rank()/size() supports cross-cluster ProcessGroups where members
        # have independent default PGs.
        self.rank = group.rank() if group is not None else dist.get_rank()
        self.world_size = group.size() if group is not None else dist.get_world_size()

    @abstractmethod
    def submit_send(self, src_tensor: torch.Tensor, dest_rank: int, task_id: Optional[int] = None):
        """Register a tensor send from the current rank to ``dest_rank``."""
        ...

    @abstractmethod
    def submit_recv(self, dest_tensor: torch.Tensor, src_rank: int, task_id: Optional[int] = None):
        """Register a tensor receive into ``dest_tensor`` from ``src_rank``."""
        ...

    @abstractmethod
    def run(self):
        """Execute all previously submitted send/recv operations as a single batch."""
        ...

    def close(self) -> None:
        """Release backend-owned resources. The default implementation is a no-op."""

    def set_model_roles(self, *, is_source: bool, is_destination: bool) -> None:
        """Provide this rank's model participation to topology-aware backends.

        Point-to-point backends infer everything they need from submitted
        operations and keep this default no-op. Collective cross-group
        backends may override it to construct their source/destination meshes.
        """

    def set_plan(self, plan: object, *, transform: object | None = None) -> None:
        """Associate subsequent submissions with an immutable reshard plan.

        Backends may use this identity to cache collective setup that is valid
        for the lifetime of the plan and transform. Point-to-point backends do
        not require per-plan setup and keep this default no-op.
        """

    def execute_plan(
        self,
        plan: object,
        src_tensors: Mapping[str, torch.Tensor],
        dst_tensors: Mapping[str, torch.Tensor],
        *,
        transform: object | None = None,
    ) -> bool:
        """Execute a plan natively, returning whether it was handled.

        The default keeps the established submit/run path. Backends whose
        native primitive needs whole tensors and mesh metadata can override
        this hook without changing the public ReFIT API or the slice-copy
        contract used by other transports.
        """
        return False


def match_local_ops_by_task_id(
    local_sends: list, local_recvs: list, backend_name: str, rank: int
) -> list[tuple]:
    """Pair same-rank send/recv ops by task_id, raising on any mismatch.

    Returns a list of ``(send_op, recv_op)`` tuples for the caller to apply
    backend-specific local-copy logic.  Either op type may be a backend-local
    wrapper as long as it exposes ``.task_id``.
    """
    sends_by_id = {op.task_id: op for op in local_sends}
    recvs_by_id = {op.task_id: op for op in local_recvs}
    if None in sends_by_id or None in recvs_by_id:
        raise RuntimeError(
            f"{backend_name}: local (same-rank) transfer requires a task_id "
            "to match sends with recvs"
        )
    # Count mismatch catches both imbalanced send/recv lists (which would
    # otherwise silently drop the longer side) and duplicate task_ids (which
    # collapse to fewer dict entries than list entries).
    if (
        len(local_sends) != len(local_recvs)
        or len(sends_by_id) != len(local_sends)
        or len(recvs_by_id) != len(local_recvs)
    ):
        raise RuntimeError(
            f"{backend_name}: unmatched local ops on rank {rank}: "
            f"{len(local_sends)} local sends vs {len(local_recvs)} local recvs"
        )
    pairs = []
    for task_id, recv_op in recvs_by_id.items():
        send_op = sends_by_id.get(task_id)
        if send_op is None:
            raise RuntimeError(
                f"{backend_name}: missing local send for task_id={task_id} on rank {rank}"
            )
        pairs.append((send_op, recv_op))
    return pairs
