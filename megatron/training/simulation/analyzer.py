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


def format_memory_size(size_bytes: int) -> str:
    """Format memory size in human-readable format"""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes/(1024*1024):.1f}MB"
    else:
        return f"{size_bytes/(1024*1024*1024):.1f}GB"


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


def get_snapshot_path(profile_dir: str, global_rank: int, pp_rank: int,
                     vpp_rank: int, microbatch_id: int, task_type: str) -> str:
    """Get memory snapshot file path

    Args:
        profile_dir: Directory containing snapshots
        global_rank: Global rank
        pp_rank: Pipeline parallel rank
        vpp_rank: Virtual pipeline parallel rank
        microbatch_id: Microbatch ID
        task_type: Task type ("forward" or "backward")

    Returns:
        Full path to snapshot file
    """
    # Convert task_type to the actual format used in snapshot filenames
    if task_type == "forward":
        task_type_str = "TaskType.FORWARD"
    elif task_type == "backward":
        task_type_str = "TaskType.BACKWARD"
    else:
        # Keep original task_type if it's already in the correct format
        task_type_str = task_type

    snapshot_file_path = os.path.join(
        profile_dir,
        f"memory_snapshot-global_rank{global_rank}-pp_rank_{pp_rank}-vpp_rank_{vpp_rank}-microbatch_id_{microbatch_id}-task_type_{task_type_str}.pickle"
    )
    return snapshot_file_path


# =============================================================================
# Memory Analysis Functions
# =============================================================================

def classify_memory_phase(frames: List[Dict]) -> str:
    """
    Classify memory allocation phase based on call stack frames
    Adapted from memory_viz.py classify_call_stack function
    """
    if not frames:
        return "other"

    found_backward = False

    # Traverse call stack in reverse order (from outer to inner)
    for frame in reversed(frames):
        name = frame.get("name", "").lower()

        if "forward_backward" in name:
            continue

        # First phase: look for first backward
        if not found_backward:
            if "backward" in name:
                found_backward = True
            elif "forward" in name:
                return "forward"
        # Second phase: look for forward after backward (recompute)
        else:
            if "forward" in name:
                return "backward_recompute"

    return "backward" if found_backward else "other"


def cluster_events_by_address(device_trace: List[Dict]) -> Dict[int, Dict[str, Any]]:
    """
    Cluster memory events by address to track allocation lifecycle

    Args:
        device_trace: List of memory events from PyTorch CUDA memory snapshot

    Returns:
        Dict mapping address to event cluster info
    """
    addr_clusters = {}

    for event in device_trace:
        addr = event.get('addr')
        action = event.get('action')
        size = event.get('size', 0)
        stream = event.get('stream')
        timestamp = event.get('time_us', 0)

        if addr is None:
            continue

        if addr not in addr_clusters:
            addr_clusters[addr] = {
                'actions': [],
                'addr': addr,
                'size': size,
                'stream': stream,
                'time_us': []
            }

        # Add action and timestamp
        addr_clusters[addr]['actions'].append(action)
        addr_clusters[addr]['time_us'].append(timestamp)

        # Update size if it's an alloc action
        if action == 'alloc':
            addr_clusters[addr]['size'] = size

    return addr_clusters


def filter_active_addresses(addr_clusters: Dict[int, Dict[str, Any]]) -> set:
    """
    Filter addresses to find those that represent active memory allocations

    Active addresses are those that only have 'alloc' action without any free-related actions

    Args:
        addr_clusters: Address clusters from cluster_events_by_address

    Returns:
        Set of addresses that represent active memory
    """
    active_addresses = set()

    for addr, cluster_info in addr_clusters.items():
        actions = cluster_info['actions']

        # Only alloc, no free-related actions
        if 'alloc' in actions:
            has_free = any(action in ['free', 'free_requested', 'free_completed'] for action in actions)
            if not has_free:
                active_addresses.add(addr)

    return active_addresses


def filter_dynamic_addresses_for_backward(addr_clusters: Dict[int, Dict[str, Any]]) -> set:
    """
    Filter addresses to find those that represent dynamic memory allocations for backward tasks

    Dynamic addresses for backward tasks are those that have 'alloc' or 'free' related actions,
    excluding 'segment_map' only addresses.

    Args:
        addr_clusters: Address clusters from cluster_events_by_address

    Returns:
        Set of addresses that represent dynamic memory for backward tasks
    """
    dynamic_addresses = set()

    for addr, cluster_info in addr_clusters.items():
        actions = cluster_info['actions']

        # Skip addresses that only have segment_map (non-dynamic allocations)
        if len(set(actions)) == 1 and 'segment_map' in actions:
            continue
        if all(action == 'segment_map' for action in actions):
            continue

        # Include addresses with actual dynamic memory operations
        has_dynamic_ops = any(action in ['alloc', 'free', 'free_requested', 'free_completed']
                             for action in actions)
        if has_dynamic_ops:
            dynamic_addresses.add(addr)

    return dynamic_addresses


def analyze_vpp_forward_task_memory_usage(snapshot_file_path: str, device: int = 0) -> Dict[str, Any]:
    """
    Analyze VPP forward task memory usage, focusing on active memory during task execution

    Args:
        snapshot_file_path: Path to memory snapshot pickle file
        device: CUDA device index (default: 0)

    Returns:
        Dict containing:
            - active_memory: Memory that was active throughout task execution (bytes)
            - peak_memory: Peak memory in bytes
    """
    with open(snapshot_file_path, 'rb') as f:
        snapshot = pickle.load(f)

    device_trace = snapshot['device_traces'][device]

    # Use address-based clustering approach to identify active events
    addr_clusters = cluster_events_by_address(device_trace)
    active_addresses = filter_active_addresses(addr_clusters)

    # Filter events to only include those from active addresses
    valid_events = []

    for event in device_trace:
        addr = event.get('addr')
        if addr in active_addresses:
            valid_events.append(event)

    # Track active memory throughout the task
    active_memory = 0

    # Process events to track memory usage over time
    for event in valid_events:
        action = event.get('action')
        addr = event.get('addr')
        size = event.get('size', 0)
        active_memory += size

    # Save filtered snapshot with only active allocations
    filtered_snapshot_path = snapshot_file_path.replace('.pickle', '_active_only.pickle')

    filtered_snapshot = {
        'device_traces': [valid_events],
        'segments': [],
        'allocator_settings': {},
        'metadata': {
            'original_file': snapshot_file_path,
            'filtered_for': 'forward_task_active_memory',
            'active_memory_bytes': active_memory
        }
    }

    with open(filtered_snapshot_path, 'wb') as f:
        pickle.dump(filtered_snapshot, f)

    result = {
        'active_memory': active_memory,
        'peak_memory': active_memory,
    }

    return result


def analyze_vpp_backward_task_memory_usage(snapshot_file_path: str, device: int = 0) -> Dict[str, Any]:
    """
    Analyze VPP backward task memory usage, focusing on peak dynamic memory during task execution

    Args:
        snapshot_file_path: Path to memory snapshot pickle file
        device: CUDA device index (default: 0)

    Returns:
        Dict containing:
            - active_memory: 0.0 (not applicable for backward)
            - peak_memory: Peak dynamic memory usage during task execution (bytes)
    """
    with open(snapshot_file_path, 'rb') as f:
        snapshot = pickle.load(f)

    device_trace = snapshot['device_traces'][device]

    # Use address-based clustering to identify dynamic memory allocations
    addr_clusters = cluster_events_by_address(device_trace)
    dynamic_addresses = filter_dynamic_addresses_for_backward(addr_clusters)

    # Filter events to only include those from dynamic addresses
    dynamic_events = []

    for event in device_trace:
        addr = event.get('addr')
        if addr in dynamic_addresses:
            dynamic_events.append(event)

    allocated_blocks: Dict[int, int] = {}

    current_memory = 0
    peak_memory = 0

    for event in dynamic_events:
        action = event.get('action')
        addr = event.get('addr')
        size = event.get('size', 0)

        if action == 'alloc':
            allocated_blocks[addr] = size
            current_memory += size

            peak_memory = max(peak_memory, current_memory)

        elif action == 'free_completed':
            if addr in allocated_blocks:
                current_memory -= allocated_blocks[addr]
                del allocated_blocks[addr]

    filtered_snapshot = {
        'device_traces': [dynamic_events],
        'segments': [],
        'allocator_settings': {},
        'metadata': {
            'original_file': snapshot_file_path,
            'filtered_for': 'backward_task_active_memory',
        }
    }

    filtered_snapshot_path = snapshot_file_path.replace('.pickle', '_backward_only.pickle')

    with open(filtered_snapshot_path, 'wb') as f:
        pickle.dump(filtered_snapshot, f)

    return {
        'active_memory': 0.0,
        'peak_memory': peak_memory,
    }


def get_vpp_task_memory_from_snapshot_dir(profile_dir: str, global_rank: int, pp_rank: int,
                                         vpp_rank: int, microbatch_id: int, task_type: str = "forward") -> Dict[str, Any]:
    """
    Get VPP task memory usage from snapshot directory

    Args:
        profile_dir: Directory containing memory snapshots
        global_rank: Global rank
        pp_rank: Pipeline parallel rank
        vpp_rank: Virtual pipeline parallel rank (model_chunk_id)
        microbatch_id: Microbatch ID
        task_type: Task type ("forward" or "backward")

    Returns:
        Memory usage analysis results
    """
    # Construct snapshot filename
    snapshot_path = get_snapshot_path(
        profile_dir=profile_dir,
        global_rank=global_rank,
        pp_rank=pp_rank,
        vpp_rank=vpp_rank,
        microbatch_id=microbatch_id,
        task_type=task_type
    )

    # Use different analysis functions based on task type
    if task_type.upper() == "FORWARD" or task_type == "TaskType.FORWARD":
        return analyze_vpp_forward_task_memory_usage(snapshot_path)
    elif task_type.upper() == "BACKWARD" or task_type == "TaskType.BACKWARD":
        return analyze_vpp_backward_task_memory_usage(snapshot_path)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def get_recored_task_dynamic_memory_info_from_snapshots(profile_dir: str) -> Dict:
    """Get memory usage data for recorded VPP tasks from snapshot files offline"""
    recored_task_memory_dict = {}  # {(pp_rank, vpp_rank, microbatch_id, task_type): memory_info}

    if not os.path.exists(profile_dir):
        logger.warning(f"Profile directory does not exist: {profile_dir}")
        return recored_task_memory_dict

    # Get available snapshot files
    pattern = r'memory_snapshot-global_rank(\d+)-pp_rank_(\d+)-vpp_rank_(\d+)-microbatch_id_(\d+)-task_type_(.+)\.pickle'

    # Only analyze global_rank=0 snapshots
    global_rank = 0

    # Scan for available snapshot files
    for filename in os.listdir(profile_dir):
        match = re.match(pattern, filename)
        if match:
            file_global_rank = int(match.group(1))
            pp_rank = int(match.group(2))
            vpp_rank = int(match.group(3))
            microbatch_id = int(match.group(4))
            task_type_str = match.group(5)

            # Extract task type (remove TaskType. prefix if present)
            if task_type_str.startswith('TaskType.'):
                task_type = task_type_str.replace('TaskType.', '').lower()
            else:
                task_type = task_type_str.lower()

            if task_type in ['forward', 'backward']:
                memory_info = get_vpp_task_memory_from_snapshot_dir(
                    profile_dir=profile_dir,
                    global_rank=global_rank,
                    pp_rank=pp_rank,
                    vpp_rank=vpp_rank,
                    microbatch_id=microbatch_id,
                    task_type=task_type
                )

                if memory_info:
                    recored_task_memory_dict[(pp_rank, vpp_rank, microbatch_id, task_type)] = memory_info

    return recored_task_memory_dict


def load_pp_task_orders(profile_dir: str) -> Dict[int, List]:
    """Load finished tasks data from pp_task_orders.json file"""
    pp_task_orders_path = os.path.join(profile_dir, 'pp_task_orders.json')
    with open(pp_task_orders_path, 'r') as f:
        pp_task_orders_raw = json.load(f)
    # Convert string keys back to integers
    pp_task_orders = {int(k): v for k, v in pp_task_orders_raw.items()}
    return pp_task_orders


def load_all_vpp_layout(profile_dir: str) -> Dict[Tuple[int, int], Any]:
    """Load all VPP layouts from all_pp_vpp_layout.json file"""
    vpp_layouts_path = os.path.join(profile_dir, 'all_pp_vpp_layout.json')
    with open(vpp_layouts_path, 'r') as f:
        vpp_layouts_raw = json.load(f)
    # Convert string keys back to tuples
    vpp_layouts = {}
    for pp_rank, vpp_layout_dict in vpp_layouts_raw.items():
        for vpp_rank, layout in vpp_layout_dict.items():
            vpp_layouts[(int(pp_rank), int(vpp_rank))] = layout
    return vpp_layouts


def get_all_task_dynamic_memory_info_from_snapshots(profile_dir: str, recored_task_memory_dict: Dict) -> Dict:
    """
    Expand recorded task memory info to all tasks in pp_tasks_order

    Args:
        profile_dir: profile directory path
        recored_task_memory_dict: recorded memory info with format {(pp_rank, vpp_rank, microbatch_id, task_type): memory_info}

    Returns:
        all_task_dynamic_memory_dict: memory info for all tasks with format {(pp_rank, vpp_rank, task_type, microbatch_id): memory_info}
    """
    all_task_dynamic_memory_dict = {}

    # Load task execution order
    pp_tasks_order = load_pp_task_orders(profile_dir)

    # Load all vpp layouts: {(pp, vpp): layout}
    all_vpp_layout_dict = load_all_vpp_layout(profile_dir)

    # Build mapping for layout matching
    layout_memory_map = {}  # {(pp_rank, vpp_rank, task_type, microbatch_id): memory_info}

    # Build mapping relationships from recorded memory dict
    for key, memory_info in recored_task_memory_dict.items():
        pp_rank, vpp_rank, microbatch_id, task_type = key

        # Build mappings
        layout_memory_map[(pp_rank, vpp_rank, task_type, microbatch_id)] = memory_info
        all_task_dynamic_memory_dict[(pp_rank, vpp_rank, task_type, microbatch_id)] = memory_info

    # Process all tasks for matching
    for pp_rank, tasks in pp_tasks_order.items():
        for task_id in tasks:
            task_info = parse_task_id(task_id)
            task_type = task_info['task_type']
            vpp_rank = task_info['vpp_rank']
            microbatch_id = task_info['microbatch_id']

            now_vpp_task = (pp_rank, vpp_rank, task_type, microbatch_id)
            if now_vpp_task not in all_task_dynamic_memory_dict:
                # Try to find same layout task memory info with microbatch id 0
                same_layout_mid_0 = (pp_rank, vpp_rank, task_type, 0)
                if same_layout_mid_0 in all_task_dynamic_memory_dict:
                    all_task_dynamic_memory_dict[now_vpp_task] = all_task_dynamic_memory_dict[same_layout_mid_0]
                else:
                    # Try to find different pp vpp but same layout with microbatch id 0
                    now_layout = all_vpp_layout_dict[(pp_rank, vpp_rank)]
                    for p_key, p_layout in all_vpp_layout_dict.items():
                        if p_layout == now_layout:
                            if (p_key[0], p_key[1], task_type, 0) in all_task_dynamic_memory_dict:
                                all_task_dynamic_memory_dict[now_vpp_task] = all_task_dynamic_memory_dict[
                                    (p_key[0], p_key[1], task_type, 0)]
                                break

    return all_task_dynamic_memory_dict


def build_pp_memory_timeline(profile_dir: str, static_memory_data: Dict,
                            task_dynamic_memory_dict: Dict) -> Dict[int, Dict]:
    """Build memory timeline data for each PP rank from offline data

    Args:
        profile_dir: profile directory path
        static_memory_data: static memory data with format: {executor_id: {pp_rank: {'static': {...}, 'dynamic': {...}}}}
        task_dynamic_memory_dict: dynamic memory data for all tasks with format {(pp_rank, vpp_rank, task_type, microbatch_id): memory_info}

    Returns:
        Dict[pp_rank, {
            'static_memory_gb': float,
            'peak_memory_gb': float,
            'timeline': List[{
                'task_id': str,
                'current_memory_gb': float,
                'task_type': str,
                'vpp_rank': int,
                'microbatch_id': int
            }]
        }]
    """
    pp_timeline_memory_dict = {}

    # Load finished task data
    pp_tasks_order = load_pp_task_orders(profile_dir)

    executor_0_pp_memory_dict = static_memory_data.get('executor_0', static_memory_data)

    pp_ranks = [int(rank) for rank in executor_0_pp_memory_dict.keys() if isinstance(rank, (str, int))]

    pipeline_parallel_size = len(pp_ranks)

    # Build timeline for each PP rank
    for pp_rank in range(pipeline_parallel_size):
        # Get static memory for this PP rank
        pp_rank_key = str(pp_rank)
        current_pp_static_memory_dict = executor_0_pp_memory_dict[pp_rank_key]['static']

        params_gb = current_pp_static_memory_dict['params_memory_gb']
        grads_gb = current_pp_static_memory_dict['grads_memory_gb']
        static_memory_gb = params_gb + grads_gb

        # Get finished tasks for this PP rank
        current_pp_finished_task_ids = pp_tasks_order.get(pp_rank, [])

        # Initialize timeline
        timeline = []
        overall_peak_gb = static_memory_gb

        # Track memory state
        current_memory_gb = static_memory_gb

        vpp_mid_active_mem_dict = {}  # {(vpp, microbatch_id): active_memory_gb}

        # Process each finished task in execution order
        for task_id in current_pp_finished_task_ids:
            task_info = parse_task_id(task_id)
            task_type = task_info['task_type']
            vpp_rank = task_info['model_chunk_id']
            microbatch_id = task_info['microbatch_id']

            memory_key = (pp_rank, vpp_rank, task_type.lower(), microbatch_id)

            memory_info = task_dynamic_memory_dict[memory_key]
            active_memory_gb = memory_info['active_memory'] / 1024.0 / 1024.0 / 1024.0
            peak_memory_gb = memory_info['peak_memory'] / 1024.0 / 1024.0 / 1024.0

            if task_type.lower() == 'forward':
                # Forward: first calculate peak during execution, then update retained memory
                overall_peak_gb = max(overall_peak_gb, current_memory_gb + peak_memory_gb)
                current_memory_gb += active_memory_gb
                vpp_mid_active_mem_dict[(vpp_rank, microbatch_id)] = active_memory_gb

            elif task_type.lower() == 'backward':
                # Backward: account for peak during backward, then release forward memory
                overall_peak_gb = max(current_memory_gb + peak_memory_gb, overall_peak_gb)
                prev_forward_memory_gb = vpp_mid_active_mem_dict[(vpp_rank, microbatch_id)]
                current_memory_gb -= prev_forward_memory_gb

            # Add to timeline
            timeline.append({
                'task_id': task_id,
                'current_memory_gb': current_memory_gb,
                'task_type': task_type,
                'vpp_rank': vpp_rank,
                'microbatch_id': microbatch_id,
            })

        pp_timeline_memory_dict[pp_rank] = {
            'static_memory_gb': static_memory_gb,
            'peak_memory_gb': overall_peak_gb,
            'timeline': timeline
        }

    return pp_timeline_memory_dict


def analyze_gbs_memory_statistics(profile_dir: str, execute_mode: str = None) -> Dict[str, Any]:
    """
    Analyze GBS memory statistics from saved profile directory

    Args:
        profile_dir: Path to profile directory containing memory snapshots and static info
        execute_mode: Execution mode ('router_balanced' or 'load_router_map'), optional

    Returns:
        Dict containing memory timeline data per PP rank
    """
    # Load static memory data
    static_memory_path = os.path.join(profile_dir, 'static_memory_info.json')

    with open(static_memory_path, 'r') as f:
        static_memory_data = json.load(f)

    # Load dynamic memory data
    # Get dynamic memory data from snapshots
    recored_task_dynamic_memory_dict = get_recored_task_dynamic_memory_info_from_snapshots(profile_dir)

    # Get all task dynamic memory info
    all_task_dynamic_memory_dict = get_all_task_dynamic_memory_info_from_snapshots(
        profile_dir, recored_task_dynamic_memory_dict
    )

    # Generate timeline charts
    pp_memory_timeline = build_pp_memory_timeline(
        profile_dir, static_memory_data, all_task_dynamic_memory_dict
    )

    return pp_memory_timeline


def create_pp_memory_timeline_chart(pp_memory_info_dict: Dict, profile_dir: str):
    """
    Create memory timeline visualization for each PP rank

    Args:
        pp_memory_info_dict: Dictionary with PP rank timeline data
        profile_dir: Directory to save chart files
    """
    import matplotlib.pyplot as plt

    pp_memory_charts_dir = os.path.join(profile_dir, 'pp_memory_charts')
    os.makedirs(pp_memory_charts_dir, exist_ok=True)

    for pp_rank, memory_info in pp_memory_info_dict.items():
        # Extract data for visualization
        plt.figure(figsize=(10, 6))

        task_lists = []
        memory_lists = []

        for i, mem_info in enumerate(memory_info['timeline']):
            task_id = mem_info['task_id']
            mem = mem_info['current_memory_gb']

            task_lists.append(i)
            memory_lists.append(mem)

        plt.plot(task_lists, memory_lists, marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.title(f'PP Rank {pp_rank} Memory Timeline', fontsize=16)
        plt.xlabel('Task Index', fontsize=12)
        plt.ylabel('Memory (GB)', fontsize=12)

        plt.tight_layout()
        chart_path = os.path.join(pp_memory_charts_dir, f'pp_rank_{pp_rank}_memory_timeline.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')

        # Add grid lines
        plt.grid(True, alpha=0.3)
        plt.close()


# =============================================================================
# Timeline Analysis
# =============================================================================

def save_gbs_trace_timeline(pp_finished_task_queue_dict: Dict, pipeline_parallel_size: int,
                           num_microbatches: int, num_model_chunks: int,
                           profile_dir: str, task_type_enum):
    """
    Generate Chrome Trace format timeline for global batch step

    Args:
        pp_finished_task_queue_dict: Dict of finished tasks per PP rank {pp_rank: [tasks]}
        pipeline_parallel_size: Number of pipeline parallel stages
        num_microbatches: Number of microbatches
        num_model_chunks: Number of model chunks (virtual pipeline stages)
        profile_dir: Directory to save trace file
        task_type_enum: TaskType enum class for type checking
    """
    # Ensure profile_dir exists
    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir, exist_ok=True)

    # Collect all finished tasks
    all_finished_tasks = []
    t_id_task_dict = {}
    for pp_rank in range(pipeline_parallel_size):
        if pp_rank in pp_finished_task_queue_dict:
            all_finished_tasks.extend(pp_finished_task_queue_dict[pp_rank])

    for task in all_finished_tasks:
        t_id_task_dict[task.task_id] = task

    if not all_finished_tasks:
        logger.warning("No finished tasks found, cannot create timeline")
        return

    # Verify dependency completeness
    logger.info(f"[TIMELINE] Total tasks collected: {len(all_finished_tasks)}")
    missing_deps_count = 0
    for task in all_finished_tasks:
        for dep_id in task.dependencies:
            if dep_id not in t_id_task_dict:
                missing_deps_count += 1
                logger.error(f"[TIMELINE] ❌ Task {task.task_id} has missing dependency: {dep_id}")

    if missing_deps_count > 0:
        logger.error(f"[TIMELINE] ❌ CRITICAL: Found {missing_deps_count} missing dependencies! Timeline will be incorrect!")
    else:
        logger.info(f"[TIMELINE] ✅ All dependencies are satisfied")

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

    # Calculate timing for all tasks
    for task in all_finished_tasks:
        calculate_task_start_time(task.task_id)

    # Generate Chrome Trace format events
    trace_events = []

    for task in all_finished_tasks:
        if task.task_id not in task_timing:
            continue

        timing = task_timing[task.task_id]

        # Format task name
        task_type_short = "F" if task.task_type == task_type_enum.FORWARD else "B"
        task_name = f"{task_type_short}(mb{task.microbatch_id},mc{task.model_chunk_id})"

        # Task category (for color coding)
        category = "forward" if task.task_type == task_type_enum.FORWARD else "backward"

        # Chrome Trace event
        event = {
            "name": task_name,
            "cat": category,
            "ph": "X",  # Complete Event
            "ts": int(timing['start_time'] * 1000),  # Milliseconds to microseconds
            "dur": int(task.duration * 1000),  # Milliseconds to microseconds
            "pid": 0,  # Process ID
            "tid": task.pp_rank,  # Thread ID, use pp_rank as track
            "args": {
                "microbatch_id": task.microbatch_id,
                "model_chunk_id": task.model_chunk_id,
                "task_id": task.task_id,
                "task_type": task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type),
                "duration_ms": task.duration,
                "start_time_ms": timing['start_time'],
                "end_time_ms": timing['end_time']
            }
        }
        trace_events.append(event)

    # Add thread name events (track names for each pp_rank)
    for pp_rank in range(pipeline_parallel_size):
        thread_name_event = {
            "name": "thread_name",
            "ph": "M",  # Metadata Event
            "pid": 0,
            "tid": pp_rank,
            "args": {
                "name": f"PP Rank {pp_rank}"
            }
        }
        trace_events.append(thread_name_event)

    # Add process name event
    process_name_event = {
        "name": "process_name",
        "ph": "M",
        "pid": 0,
        "args": {
            "name": "VPP Scheduler Timeline"
        }
    }
    trace_events.append(process_name_event)

    # Generate final trace data
    trace_data = {
        "traceEvents": trace_events,
        "displayTimeUnit": "ms",
        "systemTraceEvents": "SystemTraceData",
        "otherData": {
            "version": "1.0",
            "pipeline_parallel_size": pipeline_parallel_size,
            "num_microbatches": num_microbatches,
            "num_model_chunks": num_model_chunks,
            "total_tasks": len(all_finished_tasks)
        }
    }

    pp_merge_timeline_path = os.path.join(profile_dir, "pp_merge_timeline.trace")

    try:
        with open(pp_merge_timeline_path, 'w') as f:
            json.dump(trace_data, f, indent=2)

        logger.info(f"Chrome Trace timeline saved to {pp_merge_timeline_path}")
        logger.info(f"Total tasks: {len(all_finished_tasks)}")
        max_end_time_ms = max(timing['end_time'] for timing in task_timing.values())
        logger.info(f"Timeline duration: {max_end_time_ms:.3f}ms ({max_end_time_ms/1000:.3f}s)")
        logger.info("Open chrome://tracing and load the file to visualize the timeline")

    except Exception as e:
        logger.error(f"Failed to write timeline trace file: {e}")


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
    logger.info('\n'.join(output_lines))


# =============================================================================
# Memory Statistics
# =============================================================================

def display_memory_statistics(profile_dir: str, enable_memory_snapshot: bool,
                             execute_mode: str, pipeline_parallel_size: int):
    """
    Display memory statistics and create visualization charts

    Args:
        profile_dir: Profile directory containing memory snapshots
        enable_memory_snapshot: Whether memory snapshot is enabled
        execute_mode: Execution mode ('router_balanced' or 'load_router_map')
        pipeline_parallel_size: Number of pipeline parallel stages
    """
    if not enable_memory_snapshot:
        logger.info("Memory snapshot not enabled, skipping memory analysis")
        return

    logger.info(f"Using offline memory analysis from profile_dir: {profile_dir}")

    # Analyze GBS memory statistics
    pp_memory_info_dict = analyze_gbs_memory_statistics(profile_dir, execute_mode)

    # Create memory timeline charts
    create_pp_memory_timeline_chart(pp_memory_info_dict, profile_dir)

    # Display results summary
    output_lines = []
    output_lines.append(" Memory analysis completed successfully!")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append("=== Memory Analysis Results Summary:")

    # Add mode information
    mode_str = "load_router_map" if execute_mode == 'load_router_map' else "router_balanced"
    output_lines.append(f"Execute Mode: {mode_str}")
    output_lines.append("")

    for pp_rank, memory_info in pp_memory_info_dict.items():
        pp_static_mem = memory_info['static_memory_gb']
        peak_mem = memory_info['peak_memory_gb']
        output_lines.append(
            f'PP Rank {pp_rank}: Static Memory = {pp_static_mem:.2f} GB, '
            f'Peak Memory = {peak_mem:.2f} GB'
        )

    logger.info('\n'.join(output_lines))
