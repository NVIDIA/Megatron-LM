# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

    task_id: int | None
    tensor: torch.Tensor
    dest_rank: int


@dataclass
class RecvOp:
    """Simple container describing a single receive operation."""

    task_id: int | None
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
        self._copy_stream = torch.cuda.Stream()
        logger.info(f"GlooCopyService initialized on rank {self.rank} with {self.world_size} ranks")

    def submit_send(self, src_tensor: torch.Tensor, dest_rank: int):
        self.send_ops.append(SendOp(task_id=None, tensor=src_tensor, dest_rank=dest_rank))

    def submit_send_with_id(self, task_id: int, src_tensor: torch.Tensor, dest_rank: int):
        """Submit a send operation with a unique task identifier."""
        self.send_ops.append(SendOp(task_id=task_id, tensor=src_tensor, dest_rank=dest_rank))

    def submit_recv(self, dest_tensor: torch.Tensor, src_rank: int):
        """Submit a receive operation."""
        # Allocate a CPU buffer that matches the destination view; we'll
        # copy into dest_tensor after the Gloo recv completes.
        cpu_buffer = torch.empty_like(dest_tensor, device="cpu").contiguous()
        self.recv_ops.append(
            (RecvOp(task_id=None, tensor=cpu_buffer, src_rank=src_rank), dest_tensor)
        )

    def submit_recv_with_id(self, task_id: int, dest_tensor: torch.Tensor, src_rank: int):
        """Submit a receive operation with a unique task identifier."""
        cpu_buffer = torch.empty_like(dest_tensor, device="cpu").contiguous()
        self.recv_ops.append(
            (RecvOp(task_id=task_id, tensor=cpu_buffer, src_rank=src_rank), dest_tensor)
        )

    def run(self):
        total_ops = len(self.send_ops) + len(self.recv_ops)
        logger.info(
            f"GlooCopyService rank {self.rank}: executing batched communication: "
            f"{len(self.send_ops)} sends + {len(self.recv_ops)} recvs = {total_ops} ops"
        )

        p2p_ops: List[dist.P2POp] = []

        # Short-circuit self transfers into local device copies.
        local_sends = [op for op in self.send_ops if op.dest_rank == self.rank]
        remote_sends = [op for op in self.send_ops if op.dest_rank != self.rank]
        local_recvs = [(recv, dst) for (recv, dst) in self.recv_ops if recv.src_rank == self.rank]
        remote_recvs = [(recv, dst) for (recv, dst) in self.recv_ops if recv.src_rank != self.rank]

        if local_sends or local_recvs:
            local_sends_by_id = {op.task_id: op for op in local_sends}
            if None in local_sends_by_id:
                raise RuntimeError(
                    "GlooCopyService: local send missing task_id; "
                    "use submit_send_with_id/submit_recv_with_id for local copies"
                )
            local_recvs_by_id = {recv.task_id: (recv, dst) for (recv, dst) in local_recvs}
            if None in local_recvs_by_id:
                raise RuntimeError(
                    "GlooCopyService: local recv missing task_id; "
                    "use submit_send_with_id/submit_recv_with_id for local copies"
                )
            if len(local_sends_by_id) != len(local_sends) or len(local_recvs_by_id) != len(
                local_recvs
            ):
                raise RuntimeError(
                    f"GlooCopyService: unmatched local ops on rank {self.rank}: "
                    f"{len(local_sends)} local sends vs {len(local_recvs)} local recvs"
                )
            for task_id, (recv_op, dst_tensor) in local_recvs_by_id.items():
                send_op = local_sends_by_id.get(task_id)
                if send_op is None:
                    raise RuntimeError(
                        f"GlooCopyService: missing local send for task_id={task_id} "
                        f"on rank {self.rank}"
                    )
                with torch.no_grad():
                    src_tensor = send_op.tensor
                    if dst_tensor.device != src_tensor.device:
                        dst_tensor.copy_(src_tensor.to(dst_tensor.device))
                    else:
                        dst_tensor.copy_(src_tensor)

        # Build Gloo P2P ops over CPU tensors. For sends we clone to CPU;
        # for recvs we use the preallocated CPU buffers.
        for op in remote_sends:
            cpu_tensor = op.tensor.detach().to("cpu").contiguous()
            p2p_ops.append(dist.P2POp(dist.isend, cpu_tensor, op.dest_rank, group=self.gloo_pg))
        for recv, _dst_tensor in remote_recvs:
            p2p_ops.append(dist.P2POp(dist.irecv, recv.tensor, recv.src_rank, group=self.gloo_pg))

        if p2p_ops:
            reqs = dist.batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()

        # Copy received CPU buffers back into the original destination tensors.
        for recv, dst_tensor in remote_recvs:
            if dst_tensor.is_cuda:
                dst_tensor.copy_(recv.tensor.to(dst_tensor.device))
            else:
                dst_tensor.copy_(recv.tensor)

        if self._copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self._copy_stream)

        logger.info("GlooCopyService: batched communication completed")
        self.send_ops.clear()
        self.recv_ops.clear()
