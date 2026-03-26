# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

from .base import CopyService
from .nccl_copy_service import NCCLCopyService

# NVSHMEMCopyService lazy-imported to avoid heavy CUDA/nvshmem init at import time
def __getattr__(name):
    if name == "NVSHMEMCopyService":
        from .nvshmem_copy_service import NVSHMEMCopyService
        import sys
        setattr(sys.modules[__name__], name, NVSHMEMCopyService)
        return NVSHMEMCopyService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["CopyService", "NCCLCopyService", "NVSHMEMCopyService"]
