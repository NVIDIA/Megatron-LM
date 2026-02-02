# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
NVSHMEM-based remote copy service and supporting components.

This package is an in-tree integration of the standalone
`nvshmem_copy_service/python` implementation so that Megatron
can use it without relying on an external library.
"""

from . import nvshmem_types
from .core import GPUResourceManager, KernelLauncher, PipelineExecutor
from .memory import DoubleBufferManager, TensorPointerExtractor
from .planning import CommunicationScheduler, GPUExecutionPlanner, TaskSegmenter, WorkloadPacker
from .service import RemoteCopyService

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
