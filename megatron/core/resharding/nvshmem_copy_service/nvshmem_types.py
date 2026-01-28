# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass, field
from typing import Any, List

# Constants
# Reduced from 256MB to 64MB to test NVSHMEM symmetric memory allocation with 128 PEs
# Total symmetric memory per PE: 4 buffers × 64MB = 256MB
# Total across 128 PEs: 128 × 256MB = 32GB (vs 128GB with 256MB buffers)
MAX_SEGMENT_SIZE = 64 * 1024 * 1024  # 64MB (reduced for debugging)
MAX_TASKS_PER_BATCH = 10000


@dataclass
class SendRequest:
    """Container for a send operation request."""

    task_id: int
    src_tensor: Any  # cupy.ndarray or pointer
    src_pos: int
    size: int
    dest_pe: int


@dataclass
class ReceiveRequest:
    """Container for a receive operation request."""

    task_id: int
    dest_tensor: Any  # cupy.ndarray or pointer
    dest_pos: int
    size: int
    src_pe: int


@dataclass
class WorkloadGroup:
    """Container for a group of send requests to a specific destination PE."""

    dest_pe: int
    tasks: List[SendRequest] = field(default_factory=list)
    total_size: int = 0


@dataclass
class ScheduledBatch:
    """Metadata for a scheduled communication batch."""

    src_pe: int
    dest_pe: int
    batch_index: int
    iteration: int
    # Metadata for GPU execution
    gpu_plan: Any = None  # Placeholder for GPU-resident plan
    tasks: List[SendRequest] = field(default_factory=list)
    total_size: int = 0
    tasks_summary: Any = None  # WorkloadSummary


@dataclass
class WorkloadSummary:
    """Summary of a workload group for communication with other PEs."""

    total_size: int
    task_ids: List[int]
    task_sizes: List[int]


@dataclass
class TransferMetadata:
    """GPU-resident metadata for communication tasks."""

    ptrs: Any  # cupy array of uint64 (pointers)
    sizes: Any  # cupy array of uint64 (sizes)
    num_tasks: int
    total_size: int
