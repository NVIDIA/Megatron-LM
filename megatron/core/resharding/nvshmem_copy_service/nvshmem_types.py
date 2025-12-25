from dataclasses import dataclass, field
from typing import List, Any

# Constants
MAX_SEGMENT_SIZE = 256 * 1024 * 1024  # 256MB
MAX_TASKS_PER_BATCH = 10000


@dataclass
class SendRequest:
    task_id: int
    src_tensor: Any  # cupy.ndarray or pointer
    src_pos: int
    size: int
    dest_pe: int


@dataclass
class ReceiveRequest:
    task_id: int
    dest_tensor: Any  # cupy.ndarray or pointer
    dest_pos: int
    size: int
    src_pe: int


@dataclass
class WorkloadGroup:
    dest_pe: int
    tasks: List[SendRequest] = field(default_factory=list)
    total_size: int = 0


@dataclass
class ScheduledBatch:
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
    total_size: int
    task_ids: List[int]
    task_sizes: List[int]


@dataclass
class TransferMetadata:
    ptrs: Any  # cupy array of uint64 (pointers)
    sizes: Any  # cupy array of uint64 (sizes)
    num_tasks: int
    total_size: int


