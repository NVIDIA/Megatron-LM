# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Lazy-initialized symmetric memory manager for inference.

Provides a registry of SymmetricMemoryBuffer instances keyed by a
user-supplied identifier (e.g. "tp", "ep").  Buffers are created on first
access so that callers never need to worry about initialization ordering
relative to the inference context.
"""

from __future__ import annotations

import operator
from functools import reduce
from typing import Optional

import torch

try:
    import torch.distributed._symmetric_memory as symm_mem

    HAVE_TORCH_SYMM_MEM = True
except ImportError:
    HAVE_TORCH_SYMM_MEM = False

try:
    import triton  # pylint: disable=unused-import

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


class SymmetricMemoryBuffer:
    """
     symmetric memory buffer used in inference.
    This buffer is used by mcore-inference's low-latency
    NVLS all-gather and reduce-scatter collectives.
    """

    def __init__(self, size_in_mb, process_group):
        if not HAVE_TORCH_SYMM_MEM or not HAVE_TRITON:
            # This should be hit if the user is running an older
            # version of torch, or if they do not have triton
            # installed.
            self.symm_buffer = None
            self.symm_mem_hdl = None
        else:
            numel = int(size_in_mb * 1024 * 1024)  # size in bytes
            try:
                symm_mem.enable_symm_mem_for_group(process_group.group_name)
                self.symm_buffer = symm_mem.empty(numel, dtype=torch.uint8, device='cuda')
                self.symm_mem_hdl = symm_mem.rendezvous(self.symm_buffer, process_group)
            except RuntimeError as e:
                # If symmetric memory initialization fails, set buffer and handle to None
                # This should happen if the process group is not contained within NVlink
                self.symm_buffer = None
                self.symm_mem_hdl = None

    def _can_allocate(self, numel, dtype) -> bool:
        """
        Returns whether enough symmetric memory is available
        for the given tensor shape and dtype.
        """
        if self.symm_mem_hdl is None:
            return False
        size_of_dtype = torch.tensor([], dtype=dtype).element_size()
        required_len = numel * size_of_dtype
        return required_len <= self.symm_buffer.numel()

    def _allocate(self, numel, dtype) -> torch.Tensor:
        """
        Allocates a sub-tensor from the self.symm_buffer for the given numel and dtype"""
        required_bytes = numel * torch.tensor([], dtype=dtype).element_size()
        return self.symm_buffer[0:required_bytes].view(dtype).view(numel)

    def maybe_get_tensors(self, tensor_specs, alignment=16):
        """
        Pack multiple tensors contiguously in the symmetric buffer with alignment.

        Each tensor's starting offset is aligned to `alignment` bytes (default 16
        for 128-bit multimem access).

        Args:
            tensor_specs: list of (numel, dtype) tuples.
            alignment: byte alignment for each tensor's start offset (default 16).

        Returns:
            {"handle": None, "tensors": None} if unavailable or insufficient space.
            {"handle": symm_mem_hdl, "tensors": [(raw_byte_view, byte_offset), ...]}
            on success, where raw_byte_view is a uint8 slice of the buffer.
        """
        _NONE_RESULT = {"handle": None, "tensors": None}
        if self.symm_mem_hdl is None:
            return _NONE_RESULT

        # Compute aligned byte sizes and running offsets
        slices = []
        current_offset = 0
        for numel, dtype in tensor_specs:
            nbytes = numel * torch.tensor([], dtype=dtype).element_size()
            aligned_nbytes = ((nbytes + alignment - 1) // alignment) * alignment
            slices.append((current_offset, nbytes))
            current_offset += aligned_nbytes

        if not self._can_allocate(current_offset, torch.uint8):
            return _NONE_RESULT

        tensors = []
        for offset, nbytes in slices:
            tensors.append((self.symm_buffer[offset : offset + nbytes], offset))

        return {"handle": self.symm_mem_hdl, "tensors": tensors}

    def maybe_get_tensor(self, tensor_shape, dtype):
        """
        Returns (potentially) a sub-tensor from the self.symm_buffer for the given shape.
        If enough symmetric memory is not available, returns None.
        """
        if self.symm_mem_hdl is None:
            return {"tensor": None, "handle": None}
        numel = reduce(operator.mul, tensor_shape, 1)
        if not self._can_allocate(numel, dtype):
            return {"tensor": None, "handle": None}
        return {
            "tensor": self._allocate(numel, dtype).view(*tensor_shape),
            "handle": self.symm_mem_hdl,
        }


class SymmetricMemoryManager:
    """Registry of lazily-initialized symmetric memory buffers.

    Usage::

        buf = SymmetricMemoryManager.get_buffer("tp", process_group=tp_group)
        result = buf.maybe_get_tensor(shape, dtype)
    """

    _buffers: dict[str, SymmetricMemoryBuffer] = {}
    _default_size_mb: int = 512

    @classmethod
    def get_buffer(
        cls,
        key: str,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        size_mb: Optional[int] = None,
    ) -> SymmetricMemoryBuffer:
        """Return the buffer for *key*, creating it on first call.

        Args:
            key: Unique identifier (e.g. "tp", "ep").
            process_group: Required on the first call for a given key.
                Subsequent calls may omit it.
            size_mb: Buffer size in MiB (default 256).
        """
        if key not in cls._buffers:
            assert (
                process_group is not None
            ), f"SymmetricMemoryManager: process_group is required on first access for key='{key}'"
            cls._buffers[key] = SymmetricMemoryBuffer(
                size_in_mb=size_mb or cls._default_size_mb, process_group=process_group
            )
        return cls._buffers[key]

    @classmethod
    def destroy(cls, key: Optional[str] = None) -> None:
        """Destroy one or all buffers.

        Args:
            key: If provided, destroy only that buffer. Otherwise destroy all.
        """
        if key is not None:
            cls._buffers.pop(key, None)
        else:
            cls._buffers.clear()

    @classmethod
    def is_initialized(cls, key: str) -> bool:
        """Check whether a buffer has been created for *key*."""
        return key in cls._buffers
