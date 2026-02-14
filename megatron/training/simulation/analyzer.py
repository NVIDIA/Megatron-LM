# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Analysis tools for VPP Training Simulation

This module provides comprehensive analysis capabilities for VPP training simulation:
- Timeline visualization (Chrome Trace format)
- Throughput statistics (TFLOPS, tokens/day)
- Memory usage analysis and visualization

Functions are adapted from ATorch centralized scheduler implementation.
"""

import json
import logging
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple
from megatron.training.utils import print_rank_0

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def parse_task_id(task_id: str) -> Dict[str, Any]:
    """
    Parse task_id string to extract task information

    Expected format: pp_{pp_rank}-mbs_{microbatch_id}-model_chunk_{model_chunk_id}-task_type_{task_type}
    Example: pp_0-mbs_0-model_chunk_0-task_type_forward

    Args:
        task_id: Task ID string to parse

    Returns:
        Dict containing:
            - task_type: str (e.g., "forward", "backward")
            - pp_rank: int (pipeline parallel rank)
            - vpp_rank: int (virtual pipeline parallel rank, same as model_chunk_id)
            - microbatch_id: int (microbatch ID)
            - model_chunk_id: int (model chunk ID, same as vpp_rank)

    Raises:
        ValueError: If task_id format is invalid
    """
    # Regular expression pattern to match the task_id format
    pattern = r'^pp_(\d+)-mbs_(\d+)-model_chunk_(\d+)-task_type_(.+)$'

    match = re.match(pattern, task_id)
    if not match:
        raise ValueError(
            f"Invalid task_id format: {task_id}. "
            f"Expected format: pp_{{pp_rank}}-mbs_{{microbatch_id}}-model_chunk_{{model_chunk_id}}-task_type_{{task_type}}"
        )

    pp_rank = int(match.group(1))
    microbatch_id = int(match.group(2))
    model_chunk_id = int(match.group(3))
    task_type = match.group(4)

    return {
        'task_type': task_type,
        'pp_rank': pp_rank,
        'vpp_rank': model_chunk_id,  # vpp_rank is the same as model_chunk_id
        'microbatch_id': microbatch_id,
        'model_chunk_id': model_chunk_id
    }



def token_per_day(gbs: int, gbs_time: float, seq_length: int, world_size: int) -> float:
    """Calculate tokens per day per GPU

    Args:
        gbs: Global batch size
        gbs_time: Time for one global batch step (seconds)
        seq_length: Sequence length
        world_size: Total number of GPUs

    Returns:
        Tokens per day per GPU (in billions)
    """
    seconds_per_day = 24 * 60 * 60
    steps_per_day = seconds_per_day / gbs_time
    tokens_per_day_value = gbs * seq_length * steps_per_day / world_size
    # Return in billions
    return tokens_per_day_value / 1e9


# =============================================================================
# Throughput Statistics
# =============================================================================

def display_gbs_statistics(pp_finished_task_queue_dict: Dict, pipeline_parallel_size: int,
                          num_microbatches: int, num_model_chunks: int,
                          args, task_type_enum):
    """
    Display global batch step statistics including throughput and timing

    Args:
        pp_finished_task_queue_dict: Dict of finished tasks per PP rank {pp_rank: [tasks]}
        pipeline_parallel_size: Number of pipeline parallel stages
        num_microbatches: Number of microbatches
        num_model_chunks: Number of model chunks
        args: Training arguments (containing batch sizes, world size, etc.)
        task_type_enum: TaskType enum class for type checking
    """
    # Collect all finished tasks
    all_finished_tasks = []
    t_id_task_dict = {}
    for pp_rank in range(pipeline_parallel_size):
        if pp_rank in pp_finished_task_queue_dict:
            all_finished_tasks.extend(pp_finished_task_queue_dict[pp_rank])
    for task in all_finished_tasks:
        t_id_task_dict[task.task_id] = task

    if not all_finished_tasks:
        logger.warning("No finished tasks found, cannot calculate GBS statistics")
        return

    # Filter tasks based on execute mode
    execute_mode = getattr(args, 'execute_mode', 'router_balanced')
    if execute_mode == 'load_router_map':
        # For load_router_map mode, only use tasks from microbatch_id=0
        filtered_tasks = [task for task in all_finished_tasks if task.microbatch_id == 0]
        logger.info(
            f"load_router_map mode: using {len(filtered_tasks)} tasks from mbs=0 "
            f"out of {len(all_finished_tasks)} total tasks"
        )
    else:
        # For router_balanced mode, use all tasks
        filtered_tasks = all_finished_tasks

    if not filtered_tasks:
        logger.warning("No filtered tasks found for statistics calculation")
        return

    # Calculate each task's start and end time
    task_timing = {}  # {task_id: {'start_time': float, 'end_time': float}}

    def calculate_task_start_time(task_id: str) -> float:
        """Recursively calculate task start time (in milliseconds)"""
        if task_id in task_timing:
            return task_timing[task_id]['start_time']

        if task_id not in t_id_task_dict:
            logger.error(f"[TIMELINE] ❌ Task {task_id} not found in t_id_task_dict")
            return 0.0

        task = t_id_task_dict[task_id]

        # If no dependencies, start time is 0
        if not task.dependencies:
            start_time = 0.0
            logger.debug(f"[TIMELINE] Task {task_id} has no dependencies, start_time=0.0")
        else:
            # Start time = max end time of all dependency tasks (milliseconds)
            logger.debug(f"[TIMELINE] Task {task_id} has {len(task.dependencies)} dependencies: {task.dependencies}")
            max_dep_end_time = 0.0
            for dep_task_id in task.dependencies:
                # Check if dependency task exists (some tasks may not have been executed)
                if dep_task_id not in t_id_task_dict:
                    logger.error(f"[TIMELINE] ❌ MISSING DEPENDENCY: Task {task_id} depends on {dep_task_id}, but it's not in t_id_task_dict!")
                    logger.error(f"[TIMELINE]    This will cause incorrect start_time calculation!")
                    continue
                dep_end_time = calculate_task_start_time(dep_task_id) + t_id_task_dict[dep_task_id].duration
                logger.debug(f"[TIMELINE]   Dependency {dep_task_id} ends at {dep_end_time:.3f}ms")
                max_dep_end_time = max(max_dep_end_time, dep_end_time)
            start_time = max_dep_end_time

        # Record timing info (milliseconds)
        task_timing[task_id] = {
            'start_time': start_time,
            'end_time': start_time + task.duration
        }

        logger.debug(f"[TIMELINE] Task {task_id}: start={start_time:.3f}ms, duration={task.duration:.3f}ms, end={start_time + task.duration:.3f}ms")

        return start_time

    # Calculate timing for filtered_tasks
    for task in filtered_tasks:
        calculate_task_start_time(task.task_id)

    if not task_timing:
        logger.warning("No task timing information available")
        return

    # Calculate single mbs time (based on filtered_tasks)
    earliest_start_time = min(timing['start_time'] for timing in task_timing.values())
    latest_end_time = max(timing['end_time'] for timing in task_timing.values())
    single_mbs_time_ms = latest_end_time - earliest_start_time

    # For load_router_map mode, multiply by num_microbatches to get total GBS time
    if execute_mode == 'load_router_map':
        total_gbs_time_ms = single_mbs_time_ms * num_microbatches
        total_gbs_time_seconds = total_gbs_time_ms / 1000.0
        logger.info(
            f"load_router_map mode: single_mbs_time = {single_mbs_time_ms:.3f}ms, "
            f"total_gbs_time = {total_gbs_time_ms:.3f}ms (x{num_microbatches} mbs)"
        )
    else:
        total_gbs_time_ms = single_mbs_time_ms
        total_gbs_time_seconds = total_gbs_time_ms / 1000.0

    # Calculate TFLOPS
    from megatron.training.training import num_floating_point_operations

    global_batch_size = getattr(args, 'global_batch_size')
    flops = num_floating_point_operations(args, global_batch_size)

    # VTrainer world size = args.world_size * pipeline_parallel_size
    # because we're simulating PP ranks on a single GPU
    vtrainer_world_size = args.world_size * pipeline_parallel_size

    throughput = flops / (total_gbs_time_seconds * 10**12 * vtrainer_world_size)

    # Count task types
    forward_tasks = [task for task in all_finished_tasks if task.task_type == task_type_enum.FORWARD]
    backward_tasks = [task for task in all_finished_tasks if task.task_type == task_type_enum.BACKWARD]

    # Calculate average execution time
    avg_forward_time = sum(task.duration for task in forward_tasks) / len(forward_tasks) if forward_tasks else 0
    avg_backward_time = sum(task.duration for task in backward_tasks) / len(backward_tasks) if backward_tasks else 0

    # Output statistics
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("GBS Execution Statistics")
    output_lines.append("=" * 80)
    output_lines.append(f"Configuration:")
    output_lines.append(f"  VTrainer World Size: {vtrainer_world_size}")
    output_lines.append(f"  World Size: {args.world_size}")
    output_lines.append(f"  Pipeline Parallel Size: {pipeline_parallel_size}")
    output_lines.append(f"  Num Microbatches: {num_microbatches}")
    output_lines.append(f"  Num Model Chunks: {num_model_chunks}")
    output_lines.append(f"  Global Batch Size: {global_batch_size}")
    output_lines.append(f"  Micro Batch Size: {getattr(args, 'micro_batch_size')}")
    output_lines.append("-" * 80)

    output_lines.append(f"Timing Statistics:")
    output_lines.append(f"  First task start time: {earliest_start_time:.3f}ms")
    output_lines.append(f"  Last task end time: {latest_end_time:.3f}ms")
    output_lines.append("-" * 80)

    output_lines.append(f"Performance Metrics:")
    output_lines.append(f"  GBS time: {total_gbs_time_seconds:.3f}s")
    flops_as_tera = flops / 10**12
    output_lines.append(f"  FLOPs for one GBS: {flops_as_tera:.3f} TFLOPs")
    output_lines.append(f"  Throughput: {throughput:.3f} TFLOPS per GPU")

    # Calculate tokens per day per GPU
    seq_length = getattr(args, 'seq_length')
    tokens_per_day_per_gpu = token_per_day(
        gbs=global_batch_size,
        gbs_time=total_gbs_time_seconds,
        seq_length=seq_length,
        world_size=vtrainer_world_size
    )
    output_lines.append(f"  Tokens per day per GPU: {tokens_per_day_per_gpu:.3f} B tokens/day")

    output_lines.append("=" * 80)

    # Print all info
    print_rank_0('\n'.join(output_lines))
