# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import logging
from typing import Dict

import torch
import torch.distributed as dist

from ..nvshmem_copy_service import RemoteCopyService
from .base import CopyService

logger = logging.getLogger(__name__)


class NVSHMEMCopyService(CopyService):
    """CopyService implementation backed by NVSHMEM RemoteCopyService."""

    def __init__(self):
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before NVSHMEMCopyService()")

        self.rank = dist.get_rank()
        self._remote = RemoteCopyService()
        # Lazily initialized on first use to avoid side effects at import time
        self._initialized = False

        # NOTE: keep the original typed tensors here (not uint8 views) so local copies
        # preserve shape/strides semantics and avoid byte-offset pitfalls.
        self._local_send_ops: Dict[int, torch.Tensor] = {}
        self._local_recv_ops: Dict[int, torch.Tensor] = {}
        self._local_copy_stream = torch.cuda.Stream()

        logger.info("NVSHMEMCopyService constructed")

    def _ensure_initialized(self):
        if not self._initialized:
            self._remote.init(log_level="INFO")
            self._initialized = True
            logger.info(
                "NVSHMEMCopyService initialized: PE %d / %d", self._remote.my_pe, self._remote.n_pes
            )

    def submit_send(self, src_tensor: torch.Tensor, dest_rank: int):
        """
        Basic CopyService API is not rich enough to drive the NVSHMEM planner
        (it lacks a globally shared task identifier), so this method is kept
        only for interface compatibility and should not be used directly.

        The resharding path calls into NVSHMEMCopyService via the
        submit_send_with_id/submit_recv_with_id helpers instead.
        """
        raise RuntimeError(
            "NVSHMEMCopyService.submit_send() is not supported; "
            "use submit_send_with_id(...) from execute_reshard_plan."
        )

    def submit_recv(self, dest_tensor: torch.Tensor, src_rank: int):
        raise RuntimeError(
            "NVSHMEMCopyService.submit_recv() is not supported; "
            "use submit_recv_with_id(...) from execute_reshard_plan."
        )

    #
    # New helper API used from execute_reshard_plan via monkey-patching:
    # we avoid changing the existing execute_reshard_plan signature by adding
    # a small adapter layer that batches up matched send/recv slices.
    #

    def submit_send_with_id(self, task_id: int, src_tensor: torch.Tensor, dest_rank: int):
        """Register a send with an explicit, globally shared task_id."""
        self._ensure_initialized()

        if not src_tensor.is_contiguous():
            src_tensor = src_tensor.contiguous()

        # Local transfers: keep them out of RemoteCopyService entirely.
        if dest_rank == self.rank:
            self._local_send_ops[task_id] = src_tensor
            return

        num_bytes = src_tensor.numel() * src_tensor.element_size()
        src_bytes = src_tensor.view(torch.uint8)

        logger.debug(
            "NVSHMEMCopyService: register_send task_id=%d, %d bytes (%d → %d)",
            task_id,
            num_bytes,
            self.rank,
            dest_rank,
        )

        # Use public API on RemoteCopyService
        self._remote.register_send(
            task_id=task_id, src_tensor=src_bytes, src_pos=0, size=num_bytes, dest_pe=dest_rank
        )

    def submit_recv_with_id(self, task_id: int, dest_tensor: torch.Tensor, src_rank: int):
        """Register a recv with an explicit, globally shared task_id."""
        self._ensure_initialized()

        if not dest_tensor.is_contiguous():
            dest_tensor = dest_tensor.contiguous()

        # Local transfers: keep them out of RemoteCopyService entirely.
        if src_rank == self.rank:
            self._local_recv_ops[task_id] = dest_tensor
            return

        num_bytes = dest_tensor.numel() * dest_tensor.element_size()
        dst_bytes = dest_tensor.view(torch.uint8)

        logger.debug(
            "NVSHMEMCopyService: register_recv task_id=%d, %d bytes (%d ← %d)",
            task_id,
            num_bytes,
            self.rank,
            src_rank,
        )

        self._remote.register_receive(
            task_id=task_id, dest_tensor=dst_bytes, dest_pos=0, size=num_bytes, src_pe=src_rank
        )

    def run(self):
        """
        Execute all registered transfer pairs via NVSHMEM.

        This converts the registered pairs into RemoteCopyService send/receive
        requests, builds a schedule, runs the pipelined NVSHMEM transfer, and
        then clears internal state.
        """
        self._ensure_initialized()

        # 1) Run same-rank copies (match by task_id), like NCCL backend.
        if self._local_send_ops or self._local_recv_ops:
            missing_sends = set(self._local_recv_ops.keys()) - set(self._local_send_ops.keys())
            missing_recvs = set(self._local_send_ops.keys()) - set(self._local_recv_ops.keys())
            if missing_sends or missing_recvs:
                raise RuntimeError(
                    "NVSHMEMCopyService: unmatched local ops on rank "
                    f"{self.rank}: missing_sends={sorted(list(missing_sends))[:10]} "
                    f"missing_recvs={sorted(list(missing_recvs))[:10]}"
                )

            with torch.no_grad():
                with torch.cuda.stream(self._local_copy_stream):
                    for task_id, dst in self._local_recv_ops.items():
                        src = self._local_send_ops[task_id]
                        if src.numel() != dst.numel() or src.element_size() != dst.element_size():
                            raise RuntimeError(
                                "NVSHMEMCopyService: local copy size mismatch on rank "
                                f"{self.rank} task_id={task_id}: "
                                f"src=({tuple(src.shape)}, {src.dtype}) "
                                f"dst=({tuple(dst.shape)}, {dst.dtype})"
                            )
                        dst.copy_(src, non_blocking=True)

            torch.cuda.current_stream().wait_stream(self._local_copy_stream)
            self._local_send_ops.clear()
            self._local_recv_ops.clear()

        # 2) Execute remote schedule (if any remote sends/recvs were registered).
        if not self._remote.send_requests and not self._remote.receive_requests:
            logger.info("NVSHMEMCopyService: no remote requests; local copies complete")
            return

        logger.info("NVSHMEMCopyService: building NVSHMEM schedule and executing")
        self._remote.schedule()
        self._remote.run()
        self._remote.clear_requests()
        logger.info("NVSHMEMCopyService: NVSHMEM transfers complete")
