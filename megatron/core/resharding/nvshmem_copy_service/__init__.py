# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
NVSHMEM-based remote copy service and supporting components.

This package is an in-tree integration of the standalone
`nvshmem_copy_service/python` implementation so that Megatron
can use it without relying on an external library.
"""

# Lazy imports: eagerly importing these modules triggers heavy CUDA/nvshmem
# initialization that deadlocks when multiple workers start simultaneously.
# All symbols are importable on demand via their submodules.

def __getattr__(name):
    import sys
    _mod = sys.modules[__name__]
    if name == "ensure_nvshmem_compat":
        from .compat import ensure_nvshmem_compat
        setattr(_mod, name, ensure_nvshmem_compat)
        return ensure_nvshmem_compat
    if name == "RemoteCopyService":
        from .compat import ensure_nvshmem_compat
        ensure_nvshmem_compat()
        from .service import RemoteCopyService
        setattr(_mod, name, RemoteCopyService)
        return RemoteCopyService
    if name == "nvshmem_types":
        from . import nvshmem_types
        setattr(_mod, name, nvshmem_types)
        return nvshmem_types
    if name in ("GPUResourceManager", "KernelLauncher", "PipelineExecutor"):
        from .compat import ensure_nvshmem_compat
        ensure_nvshmem_compat()
        from . import core
        val = getattr(core, name)
        setattr(_mod, name, val)
        return val
    if name in ("DoubleBufferManager", "TensorPointerExtractor"):
        from . import memory
        val = getattr(memory, name)
        setattr(_mod, name, val)
        return val
    if name in ("CommunicationScheduler", "GPUExecutionPlanner", "TaskSegmenter", "WorkloadPacker"):
        from . import planning
        val = getattr(planning, name)
        setattr(_mod, name, val)
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "RemoteCopyService",
    "nvshmem_types",
    "GPUResourceManager",
    "KernelLauncher",
    "PipelineExecutor",
    "DoubleBufferManager",
    "TensorPointerExtractor",
    "CommunicationScheduler",
    "GPUExecutionPlanner",
    "TaskSegmenter",
    "WorkloadPacker",
]
