from typing import Any, Dict, List, Optional, Callable, Type
from dataclasses import dataclass, asdict
from enum import Enum


class TaskType(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"  
    SEND = "send"
    RECV = "recv"
    GBS_FINISH = "gbs_finish"


@dataclass
class Task:
    """Base task class for pipeline operations"""
    # Required fields (no default values)
    task_type: TaskType
    pp_rank: int
    microbatch_id: int
    model_chunk_id: int # local vpp id
    
    # Optional fields (with default values)
    task_id: str = None # '{pp_rank}_{microbatch_id}_{model_chunk_id}_{forward/backward}'
    dependencies: List[str] = None # [task_id]

    finished: bool = False
    duration: float = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.task_type == TaskType.GBS_FINISH:
            self.task_id = f"finish_task"
        else:
            self.task_id = f"pp_{self.pp_rank}-mbs_{self.microbatch_id}-model_chunk_{self.model_chunk_id}-task_type_{self.task_type.value}"
    
    def to_dict(self) -> Dict:
        """Convert Task to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert enum to string
        data['task_type'] = self.task_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Task':
        """Create Task from dictionary (JSON deserialization)"""
        # Convert string back to enum
        if isinstance(data['task_type'], str):
            data['task_type'] = TaskType(data['task_type'])
        return cls(**data)
    


