from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


@dataclass
class SendOp:
    tensor: torch.Tensor
    dest_rank: int


@dataclass
class RecvOp:
    tensor: torch.Tensor
    src_rank: int


class NCCLCopyService:
    """
    Thin wrapper around torch.distributed batch_isend_irecv to submit and execute
    a batch of point-to-point sends and recvs.
    """

    def __init__(self):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.send_ops: List[SendOp] = []
        self.recv_ops: List[RecvOp] = []
        logger.info(f"NCCLCopyService initialized with {self.world_size} ranks")

    def submit_send(self, src_tensor: torch.Tensor, dest_rank: int):
        self.send_ops.append(SendOp(tensor=src_tensor, dest_rank=dest_rank))

    def submit_recv(self, dest_tensor: torch.Tensor, src_rank: int):
        self.recv_ops.append(RecvOp(tensor=dest_tensor, src_rank=src_rank))

    def run(self):
        total_ops = len(self.send_ops) + len(self.recv_ops)
        logger.info(f"Executing batched communication: {len(self.send_ops)} sends + {len(self.recv_ops)} recvs = {total_ops} ops")

        p2p_ops = []
        for op in self.send_ops:
            p2p_ops.append(dist.P2POp(dist.isend, op.tensor, op.dest_rank))
        for op in self.recv_ops:
            p2p_ops.append(dist.P2POp(dist.irecv, op.tensor, op.src_rank))

        if p2p_ops:
            reqs = dist.batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()

        logger.info("Batched communication completed")
        self.send_ops.clear()
        self.recv_ops.clear()


