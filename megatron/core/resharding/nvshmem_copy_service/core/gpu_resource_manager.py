# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
GPU resource management for NVSHMEM operations.

Handles NVSHMEM initialization, CUDA device setup, stream management,
and event lifecycle.
"""

import logging
from typing import Dict, Optional

try:
    import nvshmem.core
    from cuda.core.experimental import Device

    HAVE_NVSHMEM = True
except ImportError:
    HAVE_NVSHMEM = False

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class GPUResourceManager:
    """Manages GPU resources including NVSHMEM, streams, and events."""

    def __init__(self):
        self.device = None
        self.my_pe: int = -1
        self.n_pes: int = -1
        self.initialized: bool = False

        # CUDA streams (cuda.core.experimental)
        self.pack_stream = None
        self.unpack_stream = None
        self.send_stream = None
        self.copy_stream = None

        # PyTorch stream wrappers
        self.torch_pack_stream = None
        self.torch_unpack_stream = None
        self.torch_send_stream = None
        self.torch_copy_stream = None

        # Stream name to PyTorch stream mapping
        self._torch_streams: Dict[str, torch.cuda.ExternalStream] = {}

    def init(self) -> None:
        """
        Initialize NVSHMEM, CUDA device, and streams.

        Expects torch.distributed to be already initialized.
        """
        if self.initialized:
            return

        if not HAVE_NVSHMEM:
            raise RuntimeError(
                "nvshmem.core is not available. Please install nvshmem to use GPUResourceManager."
            )

        # torch.distributed must be initialized before calling this
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before " "GPUResourceManager.init()"
            )

        # Get current CUDA device (already set by caller based on LOCAL_RANK)
        local_rank = torch.cuda.current_device()

        # nvshmem4py requires a cuda.core Device at init time
        self.device = Device(local_rank)
        self.device.set_current()

        # Extract rank, nranks from the default process group
        num_ranks = dist.get_world_size()
        rank_id = dist.get_rank()

        # Create/Broadcast UniqueID using broadcast_object_list
        uniqueid = nvshmem.core.get_unique_id(empty=True)
        if rank_id == 0:
            uniqueid = nvshmem.core.get_unique_id()
            broadcast_objects = [uniqueid]
        else:
            broadcast_objects = [None]

        # Broadcast ID to all ranks using the default group
        dist.broadcast_object_list(broadcast_objects, src=0)

        # Barrier to ensure everyone has the ID before NVSHMEM init
        dist.barrier()

        # Initialize NVSHMEM with the broadcasted UID
        nvshmem.core.init(
            device=self.device,
            uid=broadcast_objects[0],
            rank=rank_id,
            nranks=num_ranks,
            initializer_method="uid",
        )

        logger.info("NVSHMEM initialized")

        self.my_pe = nvshmem.core.my_pe()
        self.n_pes = nvshmem.core.n_pes()

        # Create CUDA streams
        self.pack_stream = self.device.create_stream()
        self.unpack_stream = self.device.create_stream()
        self.send_stream = self.device.create_stream()
        self.copy_stream = self.device.create_stream()

        # Get stream pointers and create PyTorch wrappers
        _, pack_stream_ptr = self.pack_stream.__cuda_stream__()
        _, unpack_stream_ptr = self.unpack_stream.__cuda_stream__()
        _, send_stream_ptr = self.send_stream.__cuda_stream__()
        _, copy_stream_ptr = self.copy_stream.__cuda_stream__()

        self.torch_pack_stream = torch.cuda.ExternalStream(pack_stream_ptr)
        self.torch_unpack_stream = torch.cuda.ExternalStream(unpack_stream_ptr)
        self.torch_send_stream = torch.cuda.ExternalStream(send_stream_ptr)
        self.torch_copy_stream = torch.cuda.ExternalStream(copy_stream_ptr)

        # Build stream mapping
        self._torch_streams = {
            "pack": self.torch_pack_stream,
            "unpack": self.torch_unpack_stream,
            "send": self.torch_send_stream,
            "copy": self.torch_copy_stream,
        }

        logger.info("Stream mapping built")

        self.initialized = True

        # Initial barrier to ensure all PEs are ready
        nvshmem.core.barrier_all(stream=self.send_stream)

    def get_stream(self, name: str):
        """
        Get CUDA stream by name.

        Args:
            name: Stream name ('pack', 'unpack', 'send', 'copy')

        Returns:
            CUDA stream object
        """
        streams = {
            "pack": self.pack_stream,
            "unpack": self.unpack_stream,
            "send": self.send_stream,
            "copy": self.copy_stream,
        }
        return streams.get(name)

    def get_torch_stream(self, name: str) -> Optional[torch.cuda.ExternalStream]:
        """
        Get PyTorch ExternalStream by name.

        Args:
            name: Stream name ('pack', 'unpack', 'send', 'copy')

        Returns:
            PyTorch ExternalStream
        """
        return self._torch_streams.get(name)

    def create_events(self, num_events: int = 2):
        """
        Create double-buffered CUDA events for pack and unpack operations.

        Args:
            num_events: Number of events to create for each type
                (default: 2 for double buffering)

        Returns:
            tuple: (pack_events, unpack_events) lists of torch.cuda.Event
        """
        pack_events = [torch.cuda.Event(enable_timing=False) for _ in range(num_events)]
        unpack_events = [torch.cuda.Event(enable_timing=False) for _ in range(num_events)]
        return pack_events, unpack_events

    def finalize(self) -> None:
        """Cleanup resources (streams are automatically managed by CUDA)."""
        self.initialized = False
        self.my_pe = -1
        self.n_pes = -1
        # Streams are automatically cleaned up when objects are deleted
