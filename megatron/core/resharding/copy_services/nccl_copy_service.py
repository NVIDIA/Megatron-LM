# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import torch
import torch.distributed as dist

from .base import CopyService

logger = logging.getLogger(__name__)


@dataclass
class SendOp:
    """Simple container describing a single NCCL send operation."""

    task_id: int | None
    tensor: torch.Tensor
    dest_rank: int


@dataclass
class RecvOp:
    """Simple container describing a single NCCL receive operation."""

    task_id: int | None
    tensor: torch.Tensor
    src_rank: int


class NCCLCopyService(CopyService):
    """
    Thin wrapper around torch.distributed batch_isend_irecv to submit and execute
    a batch of point-to-point sends and recvs.
    """

    def __init__(self):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.send_ops: List[SendOp] = []
        self.recv_ops: List[RecvOp] = []
        # Dedicated stream for local (same-rank) copies to avoid unnecessary
        # serialization with work on the default stream.
        self._copy_stream = torch.cuda.Stream()
        if self.rank == 0:
            logger.info(f"NCCLCopyService initialized with {self.world_size} ranks")

    def submit_send(self, src_tensor: torch.Tensor, dest_rank: int):
        self.send_ops.append(SendOp(task_id=None, tensor=src_tensor, dest_rank=dest_rank))

    def submit_send_with_id(self, task_id: int, src_tensor: torch.Tensor, dest_rank: int):
        """Submit a send operation with a unique task identifier."""
        self.send_ops.append(SendOp(task_id=task_id, tensor=src_tensor, dest_rank=dest_rank))

    def submit_recv(self, dest_tensor: torch.Tensor, src_rank: int):
        """Submit a receive operation."""
        self.recv_ops.append(RecvOp(task_id=None, tensor=dest_tensor, src_rank=src_rank))

    def submit_recv_with_id(self, task_id: int, dest_tensor: torch.Tensor, src_rank: int):
        """Submit a receive operation with a unique task identifier."""
        self.recv_ops.append(RecvOp(task_id=task_id, tensor=dest_tensor, src_rank=src_rank))

    def run(self):
        total_ops = len(self.send_ops) + len(self.recv_ops)
        if self.rank == 0:
            logger.info(
                "Executing batched communication: %d sends + %d recvs = %d ops",
                len(self.send_ops),
                len(self.recv_ops),
                total_ops,
            )

        local_sends = [op for op in self.send_ops if op.dest_rank == self.rank]
        remote_sends = [op for op in self.send_ops if op.dest_rank != self.rank]
        local_recvs = [op for op in self.recv_ops if op.src_rank == self.rank]
        remote_recvs = [op for op in self.recv_ops if op.src_rank != self.rank]

        if local_sends or local_recvs:
            local_sends_by_id = {op.task_id: op for op in local_sends}
            if None in local_sends_by_id:
                raise RuntimeError(
                    "NCCLCopyService: local send missing task_id; "
                    "use submit_send_with_id/submit_recv_with_id for local copies"
                )
            local_recvs_by_id = {op.task_id: op for op in local_recvs}
            if None in local_recvs_by_id:
                raise RuntimeError(
                    "NCCLCopyService: local recv missing task_id; "
                    "use submit_send_with_id/submit_recv_with_id for local copies"
                )
            if len(local_sends_by_id) != len(local_sends) or len(local_recvs_by_id) != len(
                local_recvs
            ):
                raise RuntimeError(
                    f"NCCLCopyService: unmatched local ops on rank {self.rank}: "
                    f"{len(local_sends)} local sends vs {len(local_recvs)} local recvs"
                )
            for task_id, recv_op in local_recvs_by_id.items():
                send_op = local_sends_by_id.get(task_id)
                if send_op is None:
                    raise RuntimeError(
                        f"NCCLCopyService: missing local send for task_id={task_id} "
                        f"on rank {self.rank}"
                    )
                with torch.no_grad():
                    with torch.cuda.stream(self._copy_stream):
                        recv_op.tensor.copy_(send_op.tensor)

        p2p_ops = []
        for op in remote_sends:
            p2p_ops.append(dist.P2POp(dist.isend, op.tensor, op.dest_rank))
        for op in remote_recvs:
            p2p_ops.append(dist.P2POp(dist.irecv, op.tensor, op.src_rank))

        if p2p_ops:
            reqs = dist.batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()

        # Make sure the copy stream is finished
        torch.cuda.current_stream().wait_stream(self._copy_stream)

        if self.rank == 0:
            logger.info("Batched communication completed")
        self.send_ops.clear()
        self.recv_ops.clear()
