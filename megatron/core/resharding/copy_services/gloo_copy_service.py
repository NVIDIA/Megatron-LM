# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from .base import CopyService, RecvOp, SendOp, match_local_ops_by_task_id

logger = logging.getLogger(__name__)


class GlooCopyService(CopyService):
    """
    CopyService implementation that routes refit traffic over a CPU/Gloo
    process group instead of NCCL.
    """

    def __init__(self, group=None):
        super().__init__(group=group)
        if group is not None:
            self.gloo_pg = group
        else:
            self.gloo_pg = dist.new_group(backend="gloo")
        self.send_ops: List[SendOp] = []
        # Each recv op is paired with its GPU destination tensor; the SendOp/RecvOp
        # itself carries a pinned-CPU staging buffer for Gloo's CPU PG.
        self.recv_ops: List[Tuple[RecvOp, torch.Tensor]] = []
        if self.rank == 0:
            logger.info(
                f"GlooCopyService initialized on rank {self.rank} with {self.world_size} ranks"
            )

    def submit_send(self, src_tensor: torch.Tensor, dest_rank: int, task_id: Optional[int] = None):
        self.send_ops.append(SendOp(task_id=task_id, tensor=src_tensor, dest_rank=dest_rank))

    def submit_recv(self, dest_tensor: torch.Tensor, src_rank: int, task_id: Optional[int] = None):
        # Pinned CPU staging buffer for the Gloo recv; copied back to dest_tensor in run().
        cpu_buffer = torch.empty(
            dest_tensor.shape, dtype=dest_tensor.dtype, device="cpu", pin_memory=True
        )
        self.recv_ops.append(
            (RecvOp(task_id=task_id, tensor=cpu_buffer, src_rank=src_rank), dest_tensor)
        )

    def run(self):
        total_ops = len(self.send_ops) + len(self.recv_ops)
        if self.rank == 0:
            logger.info(
                f"GlooCopyService rank {self.rank}: executing batched communication: "
                f"{len(self.send_ops)} sends + {len(self.recv_ops)} recvs = {total_ops} ops"
            )

        local_sends = [op for op in self.send_ops if op.dest_rank == self.rank]
        remote_sends = [op for op in self.send_ops if op.dest_rank != self.rank]
        local_recvs = [(recv, dst) for (recv, dst) in self.recv_ops if recv.src_rank == self.rank]
        remote_recvs = [(recv, dst) for (recv, dst) in self.recv_ops if recv.src_rank != self.rank]

        if local_sends or local_recvs:
            local_recv_objs = [recv for recv, _ in local_recvs]
            dst_by_task_id = {recv.task_id: dst for recv, dst in local_recvs}
            pairs = match_local_ops_by_task_id(
                local_sends, local_recv_objs, "GlooCopyService", self.rank
            )
            with torch.no_grad():
                for send_op, recv_op in pairs:
                    src_tensor = send_op.tensor
                    dst_tensor = dst_by_task_id[recv_op.task_id]
                    if dst_tensor.device != src_tensor.device:
                        dst_tensor.copy_(src_tensor.to(dst_tensor.device))
                    else:
                        dst_tensor.copy_(src_tensor)

        # Build Gloo P2P ops over CPU tensors. For sends we stage all
        # GPU→CPU copies with non_blocking, sync once, then build P2P ops.
        # Use group_peer (not peer) to pass ranks directly in group space,
        # avoiding the global-to-group rank conversion in P2POp which doesn't
        # work for cross-world ProcessGroups.
        cpu_send_bufs: List[torch.Tensor] = []
        for op in remote_sends:
            cpu_tensor = torch.empty(
                op.tensor.shape, dtype=op.tensor.dtype, device="cpu", pin_memory=True
            )
            cpu_tensor.copy_(op.tensor.detach(), non_blocking=True)
            cpu_send_bufs.append(cpu_tensor)
        if cpu_send_bufs:
            torch.cuda.synchronize()

        for op in remote_sends:
            # Drop the GPU reference now that staging is complete.
            op.tensor = None

        p2p_ops: List[dist.P2POp] = []
        for cpu_tensor, op in zip(cpu_send_bufs, remote_sends):
            p2p_ops.append(
                dist.P2POp(dist.isend, cpu_tensor, group=self.gloo_pg, group_peer=op.dest_rank)
            )
        for recv, _dst_tensor in remote_recvs:
            p2p_ops.append(
                dist.P2POp(dist.irecv, recv.tensor, group=self.gloo_pg, group_peer=recv.src_rank)
            )

        if p2p_ops:
            reqs = dist.batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()

        for recv, dst_tensor in remote_recvs:
            if dst_tensor.is_cuda:
                dst_tensor.copy_(recv.tensor, non_blocking=True)
            else:
                dst_tensor.copy_(recv.tensor)

        # Ensure all async CPU→GPU copies are complete.
        torch.cuda.synchronize()

        if self.rank == 0:
            logger.info("GlooCopyService: batched communication completed")
        self.send_ops.clear()
        self.recv_ops.clear()
