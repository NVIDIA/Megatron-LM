from __future__ import annotations

"""
NVSHMEM-based implementation of the CopyService interface.

This wraps the higher-level RemoteCopyService so it can be used anywhere a
CopyService is expected (e.g., refit/reshard execution).

NOTE: This is a first, minimal wiring. It currently mirrors the point-to-point
semantics of execute_reshard_plan by treating each send/recv pair as an
independent NVSHMEM "task" defined over contiguous slices.
"""

import logging
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist

from ..nvshmem_copy_service import RemoteCopyService
from .base import CopyService

logger = logging.getLogger(__name__)


class NVSHMEMCopyService(CopyService):
    """CopyService implementation backed by NVSHMEM RemoteCopyService."""

    def __init__(self):
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before NVSHMEMCopyService()"
            )

        self._remote = RemoteCopyService()
        # Lazily initialized on first use to avoid side effects at import time
        self._initialized = False

        # Internal bookkeeping of registration calls before schedule/run
        self._next_task_id: int = 0
        self._registered_pairs: List[Tuple[int, torch.Tensor, torch.Tensor, int]] = []

        logger.info("NVSHMEMCopyService constructed")

    def _ensure_initialized(self):
        if not self._initialized:
            self._remote.init(log_level="INFO")
            self._initialized = True
            logger.info(
                "NVSHMEMCopyService initialized: PE %d / %d",
                self._remote.my_pe,
                self._remote.n_pes,
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

    def submit_send_with_id(
        self,
        task_id: int,
        src_tensor: torch.Tensor,
        dest_rank: int,
    ):
        """Register a send with an explicit, globally shared task_id."""
        self._ensure_initialized()

        if not src_tensor.is_contiguous():
            src_tensor = src_tensor.contiguous()

        num_bytes = src_tensor.numel() * src_tensor.element_size()
        src_bytes = src_tensor.view(torch.uint8)

        logger.debug(
            "NVSHMEMCopyService: register_send task_id=%d, %d bytes (%d → %d)",
            task_id,
            num_bytes,
            dist.get_rank(),
            dest_rank,
        )

        # Use public API on RemoteCopyService
        self._remote.register_send(
            task_id=task_id,
            src_tensor=src_bytes,
            src_pos=0,
            size=num_bytes,
            dest_pe=dest_rank,
        )

    def submit_recv_with_id(
        self,
        task_id: int,
        dest_tensor: torch.Tensor,
        src_rank: int,
    ):
        """Register a recv with an explicit, globally shared task_id."""
        self._ensure_initialized()

        if not dest_tensor.is_contiguous():
            dest_tensor = dest_tensor.contiguous()

        num_bytes = dest_tensor.numel() * dest_tensor.element_size()
        dst_bytes = dest_tensor.view(torch.uint8)

        logger.debug(
            "NVSHMEMCopyService: register_recv task_id=%d, %d bytes (%d ← %d)",
            task_id,
            num_bytes,
            dist.get_rank(),
            src_rank,
        )

        self._remote.register_receive(
            task_id=task_id,
            dest_tensor=dst_bytes,
            dest_pos=0,
            size=num_bytes,
            src_pe=src_rank,
        )

    def run(self):
        """
        Execute all registered transfer pairs via NVSHMEM.

        This converts the registered pairs into RemoteCopyService send/receive
        requests, builds a schedule, runs the pipelined NVSHMEM transfer, and
        then clears internal state.
        """
        # Execute schedule built from submit_send_with_id/submit_recv_with_id
        self._ensure_initialized()
        logger.info("NVSHMEMCopyService: building NVSHMEM schedule and executing")
        self._remote.schedule()
        self._remote.run()
        self._remote.clear_requests()
        logger.info("NVSHMEMCopyService: NVSHMEM transfers complete")


