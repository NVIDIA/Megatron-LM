from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.distributed as dist

from .base import CopyService

logger = logging.getLogger(__name__)


@dataclass
class SendOp:
    """Simple container describing a single send operation."""

    tensor: torch.Tensor
    dest_rank: int


@dataclass
class RecvOp:
    """Simple container describing a single receive operation."""

    tensor: torch.Tensor
    src_rank: int


class GlooCopyService(CopyService):
    """
    CopyService implementation that routes refit traffic over a CPU/Gloo
    process group instead of NCCL.
    """

    def __init__(self):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.gloo_pg = dist.new_group(backend="gloo")
        self.send_ops: List[SendOp] = []
        self.recv_ops: List[Tuple[RecvOp, torch.Tensor]] = []
        logger.info(f"GlooCopyService initialized on rank {self.rank} with {self.world_size} ranks")

    def submit_send(self, src_tensor: torch.Tensor, dest_rank: int):
        self.send_ops.append(SendOp(tensor=src_tensor, dest_rank=dest_rank))

    def submit_recv(self, dest_tensor: torch.Tensor, src_rank: int):
        # Allocate a CPU buffer that matches the destination view; we'll
        # copy into dest_tensor after the Gloo recv completes.
        cpu_buffer = torch.empty_like(dest_tensor, device="cpu").contiguous()
        self.recv_ops.append((RecvOp(tensor=cpu_buffer, src_rank=src_rank), dest_tensor))

    def run(self):
        total_ops = len(self.send_ops) + len(self.recv_ops)
        logger.info(
            f"GlooCopyService rank {self.rank}: executing batched communication: "
            f"{len(self.send_ops)} sends + {len(self.recv_ops)} recvs = {total_ops} ops"
        )

        p2p_ops: List[dist.P2POp] = []

        # Build Gloo P2P ops over CPU tensors. For sends we clone to CPU;
        # for recvs we use the preallocated CPU buffers.
        for op in self.send_ops:
            cpu_tensor = op.tensor.detach().to("cpu").contiguous()
            p2p_ops.append(dist.P2POp(dist.isend, cpu_tensor, op.dest_rank, group=self.gloo_pg))
        for recv, _dst_tensor in self.recv_ops:
            p2p_ops.append(dist.P2POp(dist.irecv, recv.tensor, recv.src_rank, group=self.gloo_pg))

        if p2p_ops:
            reqs = dist.batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()

        # Copy received CPU buffers back into the original destination tensors.
        for recv, dst_tensor in self.recv_ops:
            if dst_tensor.is_cuda:
                dst_tensor.copy_(recv.tensor.to(dst_tensor.device))
            else:
                dst_tensor.copy_(recv.tensor)

        logger.info("GlooCopyService: batched communication completed")
        self.send_ops.clear()
        self.recv_ops.clear()
