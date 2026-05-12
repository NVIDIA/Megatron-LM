# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.distributed as dist

from .base import CopyService, RecvOp, SendOp, match_local_ops_by_task_id

logger = logging.getLogger(__name__)


class NCCLCopyService(CopyService):
    """
    Thin wrapper around torch.distributed batch_isend_irecv to submit and execute
    a batch of point-to-point sends and recvs.
    """

    def __init__(self, group=None):
        super().__init__(group=group)
        self.send_ops: List[SendOp] = []
        self.recv_ops: List[RecvOp] = []
        # Dedicated stream for local (same-rank) copies to avoid unnecessary
        # serialization with work on the default stream.
        self._copy_stream = torch.cuda.Stream()
        if self.rank == 0:
            logger.info(f"NCCLCopyService initialized with {self.world_size} ranks")

    def submit_send(self, src_tensor: torch.Tensor, dest_rank: int, task_id: Optional[int] = None):
        self.send_ops.append(SendOp(task_id=task_id, tensor=src_tensor, dest_rank=dest_rank))

    def submit_recv(self, dest_tensor: torch.Tensor, src_rank: int, task_id: Optional[int] = None):
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
            pairs = match_local_ops_by_task_id(
                local_sends, local_recvs, "NCCLCopyService", self.rank
            )
            with torch.no_grad(), torch.cuda.stream(self._copy_stream):
                for send_op, recv_op in pairs:
                    recv_op.tensor.copy_(send_op.tensor)

        p2p_ops = []
        for op in remote_sends:
            p2p_ops.append(dist.P2POp(dist.isend, op.tensor, op.dest_rank, group=self.group))
        for op in remote_recvs:
            p2p_ops.append(dist.P2POp(dist.irecv, op.tensor, op.src_rank, group=self.group))

        if p2p_ops:
            reqs = dist.batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()

        torch.cuda.current_stream().wait_stream(self._copy_stream)

        if self.rank == 0:
            logger.info("Batched communication completed")
        self.send_ops.clear()
        self.recv_ops.clear()
