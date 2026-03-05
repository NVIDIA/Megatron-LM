# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
CUDA kernel management and launching for pack/unpack operations.

Handles kernel compilation, launching, and stream coordination.
"""

import os
from typing import Any, Tuple

try:
    import cupy as cp

    HAVE_CUPY = True
except ImportError:
    HAVE_CUPY = False

import torch
import torch.cuda.nvtx as nvtx


class KernelLauncher:
    """Manages CUDA kernel loading and launching for data pack/unpack operations."""

    def __init__(self):
        self.chunked_copy_kernel = None
        # Cached CuPy stream wrappers for efficient kernel launching
        self.cp_pack_stream = None
        self.cp_unpack_stream = None

    def load_kernels(self) -> None:
        """Load and compile CUDA kernels from source."""
        if not HAVE_CUPY:
            raise RuntimeError("cupy is not available. Please install cupy to use KernelLauncher.")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        kernel_path = os.path.join(current_dir, "..", "kernels", "chunked_kernel.cu")

        with open(kernel_path, "r") as f:
            kernel_source = f.read()

        self.chunked_copy_kernel = cp.RawKernel(
            kernel_source, "chunked_batched_copy_kernel", options=("-std=c++11",)
        )

    def set_streams(self, pack_stream, unpack_stream) -> None:
        """
        Cache CuPy stream wrappers for kernel launching.

        This eliminates per-launch overhead of stream pointer extraction
        and CuPy ExternalStream creation.

        Args:
            pack_stream: CUDA stream for pack operations
            unpack_stream: CUDA stream for unpack operations
        """
        _, pack_stream_ptr = pack_stream.__cuda_stream__()
        _, unpack_stream_ptr = unpack_stream.__cuda_stream__()
        self.cp_pack_stream = cp.cuda.ExternalStream(pack_stream_ptr)
        self.cp_unpack_stream = cp.cuda.ExternalStream(unpack_stream_ptr)

    def launch_pack(
        self,
        gpu_plan: Tuple[Any, Any, Any, int],
        pack_stream,
        torch_pack_stream: torch.cuda.ExternalStream,
        pack_event: torch.cuda.Event,
    ) -> None:
        """
        Launch pack kernel to copy data from user tensors to send buffer.

        Args:
            gpu_plan: Tuple of (cp_src_addrs, cp_dst_addrs, cp_sizes, num_chunks)
                as CuPy arrays
            pack_stream: CUDA stream (cuda.core.experimental.Stream) - unused,
                kept for compatibility
            torch_pack_stream: PyTorch external stream wrapper
            pack_event: CUDA event to record after kernel launch
        """
        nvtx.range_push("Launch Pack Kernel")
        if not gpu_plan:
            nvtx.range_pop()
            return

        # Unpack cached CuPy arrays from gpu_plan
        cp_src, cp_dst, cp_sizes, num_chunks = gpu_plan

        # Grid/Block configuration
        THREADS_PER_BLOCK = 1024
        NUM_BLOCKS = 75

        # Launch kernel using cached CuPy stream
        assert self.chunked_copy_kernel is not None
        assert self.cp_pack_stream is not None
        self.chunked_copy_kernel(
            (NUM_BLOCKS,),
            (THREADS_PER_BLOCK,),
            (cp_src, cp_dst, cp_sizes, num_chunks),
            stream=self.cp_pack_stream,
        )
        nvtx.range_pop()
        # Record event on PyTorch stream
        pack_event.record(stream=torch_pack_stream)

    def launch_unpack(
        self,
        gpu_plan: Tuple[Any, Any, Any, int],
        unpack_stream,
        torch_unpack_stream: torch.cuda.ExternalStream,
        unpack_event: torch.cuda.Event,
    ) -> None:
        """
        Launch unpack kernel to copy data from receive buffer to user tensors.

        Args:
            gpu_plan: Tuple of (cp_src_addrs, cp_dst_addrs, cp_sizes, num_chunks)
                as CuPy arrays
            unpack_stream: CUDA stream (cuda.core.experimental.Stream) - unused,
            kept for compatibility
            torch_unpack_stream: PyTorch external stream wrapper
            unpack_event: CUDA event to record after kernel launch
        """
        nvtx.range_push("Launch Unpack Kernel")
        if not gpu_plan:
            nvtx.range_pop()
            return

        # Unpack cached CuPy arrays from gpu_plan
        cp_src, cp_dst, cp_sizes, num_chunks = gpu_plan

        # Grid/Block configuration
        THREADS_PER_BLOCK = 1024
        NUM_BLOCKS = 75

        # Launch kernel using cached CuPy stream
        assert self.chunked_copy_kernel is not None
        assert self.cp_unpack_stream is not None
        self.chunked_copy_kernel(
            (NUM_BLOCKS,),
            (THREADS_PER_BLOCK,),
            (cp_src, cp_dst, cp_sizes, num_chunks),
            stream=self.cp_unpack_stream,
        )
        nvtx.range_pop()
        # Record event on PyTorch stream
        unpack_event.record(stream=torch_unpack_stream)
