# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""VPP Training Simulation Framework.

This module provides tools for simulating Virtual Pipeline Parallelism (VPP) training
to analyze performance characteristics without running full training jobs.
"""

from megatron.training.simulation.task import Task, TaskType

__all__ = ['Task', 'TaskType']
