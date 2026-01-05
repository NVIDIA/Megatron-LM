# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Double buffer management for NVSHMEM symmetric memory.

Manages send and receive buffers with double-buffering for pipelined communication.
"""

try:
    import nvshmem.core.interop.torch

    HAVE_NVSHMEM = True
except ImportError:
    HAVE_NVSHMEM = False

import torch

from ..nvshmem_types import MAX_SEGMENT_SIZE


class DoubleBufferManager:
    """Manages double-buffered NVSHMEM symmetric buffers for send/receive operations."""

    def __init__(self, slot_size: int = MAX_SEGMENT_SIZE):
        """
        Initialize buffer manager.

        Args:
            slot_size: Size of each buffer slot in bytes (default: 256MB)
        """
        self.slot_size = slot_size
        self.send_slots = [None, None]
        self.recv_slots = [None, None]

    def allocate(self) -> None:
        """Allocate NVSHMEM symmetric buffers for double-buffering."""
        if not HAVE_NVSHMEM:
            raise RuntimeError(
                "nvshmem.core.interop.torch is not available. "
                "Please install nvshmem to use DoubleBufferManager."
            )

        for i in range(2):
            self.send_slots[i] = nvshmem.core.interop.torch.bytetensor(
                (self.slot_size,), dtype=torch.uint8
            )
            self.recv_slots[i] = nvshmem.core.interop.torch.bytetensor(
                (self.slot_size,), dtype=torch.uint8
            )
            # Zero out buffers
            self.send_slots[i].zero_()
            self.recv_slots[i].zero_()

    def get_send_slot(self, iteration: int):
        """
        Get send buffer for given iteration.

        Args:
            iteration: Iteration number

        Returns:
            NVSHMEM tensor for sending
        """
        return self.send_slots[iteration % 2]

    def get_recv_slot(self, iteration: int):
        """
        Get receive buffer for given iteration.

        Args:
            iteration: Iteration number

        Returns:
            NVSHMEM tensor for receiving
        """
        return self.recv_slots[iteration % 2]

    def free(self) -> None:
        """Free NVSHMEM symmetric buffers."""
        for i in range(2):
            if self.send_slots[i] is not None:
                nvshmem.core.interop.torch.free_tensor(self.send_slots[i])
                self.send_slots[i] = None
            if self.recv_slots[i] is not None:
                nvshmem.core.interop.torch.free_tensor(self.recv_slots[i])
                self.recv_slots[i] = None
