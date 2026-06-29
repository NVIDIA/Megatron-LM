# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Data parallel inference coordinator package.

The coordinator class itself lives in :mod:`coordinator`; message handlers are
in :mod:`handlers` and the control-signal state machine in :mod:`state`. This
module re-exports the public names so existing imports of
``megatron.core.inference.data_parallel_inference_coordinator`` keep working.
"""

from .coordinator import DataParallelInferenceCoordinator
from .handlers import HANDLERS, message_handler
from .state import CONTROL_TRANSITIONS, ControlTransition, CoordinatorState

__all__ = [
    "DataParallelInferenceCoordinator",
    "CoordinatorState",
    "ControlTransition",
    "CONTROL_TRANSITIONS",
    "HANDLERS",
    "message_handler",
]
