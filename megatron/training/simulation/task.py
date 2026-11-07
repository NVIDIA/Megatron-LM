# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Task abstraction for VPP training simulation framework."""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, List


class TaskType(Enum):
    """Enumeration of task types in the simulation framework.

    Attributes:
        FORWARD: Forward pass computation task
        BACKWARD: Backward pass computation task
        SEND: Communication send task
        RECV: Communication receive task
        GBS_FINISH: Global batch step completion marker
    """

    FORWARD = "forward"
    BACKWARD = "backward"
    SEND = "send"
    RECV = "recv"
    GBS_FINISH = "gbs_finish"


@dataclass
class Task:
    """Represents a single task in the VPP simulation framework.

    A task encapsulates information about a pipeline operation including its type,
    position in the pipeline, dependencies, and execution metrics.

    Attributes:
        task_type: Type of the task (FORWARD, BACKWARD, etc.)
        pp_rank: Pipeline parallel rank executing this task
        microbatch_id: Microbatch index
        model_chunk_id: Model chunk index for virtual pipeline parallelism
        task_id: Unique identifier for the task (auto-generated)
        dependencies: List of task IDs this task depends on
        finished: Whether the task has completed execution
        duration: Execution time in seconds (if finished)
    """
    # Required fields
    task_type: TaskType
    pp_rank: int
    microbatch_id: int
    model_chunk_id: int  # Local VPP chunk ID

    # Optional fields with defaults
    task_id: str = None  # Auto-generated unique identifier
    dependencies: List[str] = None  # List of task IDs this task depends on
    finished: bool = False
    duration: float = None  # Execution time in seconds
    
    def __post_init__(self):
        """Initialize optional fields and generate task_id."""
        if self.dependencies is None:
            self.dependencies = []
        if self.task_type == TaskType.GBS_FINISH:
            self.task_id = "finish_task"
        else:
            self.task_id = (
                f"pp_{self.pp_rank}-mbs_{self.microbatch_id}-"
                f"model_chunk_{self.model_chunk_id}-task_type_{self.task_type.value}"
            )

    def to_dict(self) -> Dict:
        """Convert Task to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the task with enum values converted to strings.
        """
        data = asdict(self)
        data['task_type'] = self.task_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'Task':
        """Create Task instance from dictionary.

        Args:
            data: Dictionary containing task data. The 'task_type' field should be
                either a TaskType enum or a string value.

        Returns:
            New Task instance constructed from the dictionary.
        """
        if isinstance(data['task_type'], str):
            data['task_type'] = TaskType(data['task_type'])
        return cls(**data)
