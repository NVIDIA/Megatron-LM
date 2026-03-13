# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Lazy-initialized symmetric memory manager for inference.

Provides a registry of GlobalSymmetricMemoryBuffer instances keyed by a
user-supplied identifier (e.g. "tp", "ep").  Buffers are created on first
access so that callers never need to worry about initialization ordering
relative to the inference context.
"""

from __future__ import annotations

from typing import Optional

import torch

from megatron.core.utils import GlobalSymmetricMemoryBuffer


class SymmetricMemoryManager:
    """Registry of lazily-initialized symmetric memory buffers.

    Usage::

        buf = SymmetricMemoryManager.get_buffer("tp", process_group=tp_group)
        result = buf.maybe_get_tensor(shape, dtype)
    """

    _buffers: dict[str, GlobalSymmetricMemoryBuffer] = {}
    _default_size_mb: int = 256

    @classmethod
    def get_buffer(
        cls,
        key: str,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        size_mb: Optional[int] = None,
    ) -> GlobalSymmetricMemoryBuffer:
        """Return the buffer for *key*, creating it on first call.

        Args:
            key: Unique identifier (e.g. "tp", "ep").
            process_group: Required on the first call for a given key.
                Subsequent calls may omit it.
            size_mb: Buffer size in MiB (default 256).
        """
        if key not in cls._buffers:
            assert process_group is not None, (
                f"SymmetricMemoryManager: process_group is required on first access for key='{key}'"
            )
            cls._buffers[key] = GlobalSymmetricMemoryBuffer(
                size_in_mb=size_mb or cls._default_size_mb,
                process_group=process_group,
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
