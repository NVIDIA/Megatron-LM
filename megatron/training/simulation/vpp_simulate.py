# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""VPP Training Simulation for Performance Profiling"""
import itertools
import os
import time
import json
from typing import Dict, List, Tuple

import torch
import torch.distributed

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_schedule_table, get_pp_rank_microbatches
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training.global_vars import get_args
from megatron.training.utils import print_rank_0
from megatron.training.simulation.task import Task, TaskType
from megatron.core.transformer.enums import LayerType
from megatron.core.pipeline_parallel.schedules import forward_step
from megatron.training.simulation.model_executor import (
    create_model,
    clear_model_mem,
    prepare_input_tensor,
    execute_forward_with_timing,
    execute_backward_with_timing,
)


def calculate_model_static_memory(model):
    """Calculate static memory (parameters + gradients) for a model

    Args:
        model: Model or list of model chunks (for VPP)

    Returns:
        tuple: (params_memory_gb, grads_memory_gb)
    """
    # Handle list of model chunks (VPP case)
    model_chunks = model if isinstance(model, list) else [model]

    total_params_bytes = 0
    total_grads_bytes = 0

    for model_chunk in model_chunks:
        # Access DDP buffers (_ParamAndGradBuffer instances)
        # Megatron uses buffers to manage parameters and gradients
        all_buffers = []

        if hasattr(model_chunk, 'buffers'):
            all_buffers = model_chunk.buffers
        if hasattr(model_chunk, 'expert_parallel_buffers'):
            all_buffers += model_chunk.expert_parallel_buffers

        for buffer in all_buffers:
            # Parameter memory from buffer
            if hasattr(buffer, 'param_data') and buffer.param_data is not None:
                total_params_bytes += buffer.param_data.numel() * buffer.param_data.element_size()

            # Gradient memory from buffer
            if hasattr(buffer, 'grad_data') and buffer.grad_data is not None:
                total_grads_bytes += buffer.grad_data.numel() * buffer.grad_data.element_size()

    # Convert to GB
    params_memory_gb = total_params_bytes / (1024 ** 3)
    grads_memory_gb = total_grads_bytes / (1024 ** 3)

    return params_memory_gb, grads_memory_gb


class VppSimulator(object):
    """
    Simulate VPP training for performance profiling.

    This function executes a complete global batch step across all pipeline ranks,
    measures execution time for each forward/backward pass, and generates:
    - Timeline trace (Chrome trace format)
    - Memory usage profile
    - Performance statistics

    Args:
        train_data_iterator: training data iterator
        model_provider: function to create model

    Execution flow:
        1. Iterate through each PP rank and create corresponding model chunks
        2. Execute forward/backward passes based on execution granularity
        3. Collect timeline and memory information
        4. Rank 0 merges all results and generates reports
    """
    def __init__(self, train_data_iterator, model_provider, forward_step_func):
        self.train_data_iterator = train_data_iterator
        self.model_provider = model_provider
        self.forward_step_func = forward_step_func
        self._initialize()

    def _initialize(self):
        args = get_args()
        self.pipeline_parallel_size = args.pipeline_model_parallel_size

        self.num_microbatches = get_num_microbatches()
        self.num_model_chunks = parallel_state.get_virtual_pipeline_model_parallel_world_size()
        if self.num_model_chunks is None:
            self.num_model_chunks = 1
        self.total_num_microbatches = self.num_microbatches * self.num_model_chunks

        # Get schedule table
        # Use 'or' to handle None case: getattr returns None when arg exists but is None
        self.microbatch_group_size_per_vp_stage = getattr(args, 'microbatch_group_size_per_vp_stage', None) or self.pipeline_parallel_size
        self.schedule_table = get_schedule_table(
            self.num_microbatches,
            self.num_model_chunks,
            self.microbatch_group_size_per_vp_stage
        )

        self.microbatch_id_table, self.model_chunk_id_table = zip(*self.schedule_table)

        # create tasks
        self.pp_mbid_mcid_fb_task_dict = {}
        self.task_num_count = 0


        # Create forward tasks
        for pp_rank, model_chunk_id, micro_batch_id in self.forward_task_iter():
            self.pp_mbid_mcid_fb_task_dict[(pp_rank, micro_batch_id, model_chunk_id, TaskType.FORWARD)] = Task(
                task_type=TaskType.FORWARD,
                pp_rank=pp_rank,
                microbatch_id=micro_batch_id,
                model_chunk_id=model_chunk_id,
            )
            self.task_num_count += 1

        # Create backward tasks
        for pp_rank, model_chunk_id, micro_batch_id in self.backward_task_iter():
            self.pp_mbid_mcid_fb_task_dict[(pp_rank, micro_batch_id, model_chunk_id, TaskType.BACKWARD)] = Task(
                task_type=TaskType.BACKWARD,
                pp_rank=pp_rank,
                microbatch_id=micro_batch_id,
                model_chunk_id=model_chunk_id,
            )
            self.task_num_count += 1

        # Task execution order for each PP rank {pp_rank: [task_id1, task_id2, ...]}
        # Must be initialized BEFORE _create_task_dependencies() is called
        self.pp_rank_execution_order = {}

        self._create_task_dependencies()

        # create task execution time recorder
        self.task_recoreder = {}

        # Model caching for reuse across tasks
        self.current_model = None
        self.current_model_pp_rank = None
        self.current_model_vpp_rank = None

        # Task output tensor storage
        self.task_output_tensor_dict = {}

        # Task input gradient tensor storage for backward
        self.task_input_tensor_grad_dict = {}

        # Static memory info for each PP rank {pp_rank: (params_gb, grads_gb)}
        self.pp_static_memory_dict = {}

        # DEBUG: Print schedule_table for debugging
        print_rank_0("\n" + "="*80)
        print_rank_0("DEBUG: Schedule Table Information")
        print_rank_0("="*80)
        print_rank_0(f"num_microbatches: {self.num_microbatches}")
        print_rank_0(f"num_model_chunks: {self.num_model_chunks}")
        print_rank_0(f"total_num_microbatches: {self.total_num_microbatches}")
        print_rank_0(f"microbatch_group_size_per_vp_stage: {self.microbatch_group_size_per_vp_stage}")
        print_rank_0(f"\nSchedule table (first 10 entries):")
        for i, (mb_id, mc_id) in enumerate(self.schedule_table[:10]):
            print_rank_0(f"  virtual_mb_id {i}: (microbatch_id={mb_id}, model_chunk_id={mc_id})")
        print_rank_0("="*80 + "\n")

    def _create_task_dependencies(self):
        for pp_rank in range(self.pipeline_parallel_size):
            for micro_batch_id in range(self.num_microbatches):
                for model_chunk_id in range(self.num_model_chunks):
                    backward_task = self.pp_mbid_mcid_fb_task_dict[(pp_rank, micro_batch_id, model_chunk_id, TaskType.BACKWARD)]
                    forward_task = self.pp_mbid_mcid_fb_task_dict[(pp_rank, micro_batch_id, model_chunk_id, TaskType.FORWARD)]
                    backward_task.dependencies.append(forward_task.task_id)

        self._create_cross_pp_rank_dependencies()

        for pp_rank in range(self.pipeline_parallel_size):
            self._create_pp_rank_schedule_dependencies(pp_rank)

    def _create_cross_pp_rank_dependencies(self):
        """创建跨PP rank的pipeline依赖关系
        
        Forward pipeline: PP rank i 依赖于 PP rank i-1 的同一microbatch
        Backward pipeline: PP rank i 依赖于 PP rank i+1 的同一microbatch  
        """
        for micro_batch_id in range(self.num_microbatches):
            for model_chunk_id in range(self.num_model_chunks):
                # Forward pipeline依赖：pp_rank > 0 的forward task依赖于前一个pp_rank的forward task
                for pp_rank in range(1, self.pipeline_parallel_size):
                    current_forward_task = self.pp_mbid_mcid_fb_task_dict[(pp_rank, micro_batch_id, model_chunk_id, TaskType.FORWARD)]
                    prev_forward_task = self.pp_mbid_mcid_fb_task_dict[(pp_rank-1, micro_batch_id, model_chunk_id, TaskType.FORWARD)]
                    current_forward_task.dependencies.append(prev_forward_task.task_id)
                
                # Backward pipeline依赖：pp_rank < pipeline_parallel_size-1 的backward task依赖于后一个pp_rank的backward task
                for pp_rank in range(self.pipeline_parallel_size-1):
                    current_backward_task = self.pp_mbid_mcid_fb_task_dict[(pp_rank, micro_batch_id, model_chunk_id, TaskType.BACKWARD)]
                    next_backward_task = self.pp_mbid_mcid_fb_task_dict[(pp_rank+1, micro_batch_id, model_chunk_id, TaskType.BACKWARD)]
                    current_backward_task.dependencies.append(next_backward_task.task_id)

    def _get_pp_rank_microbatches(self, pp_rank: int):
        num_warmup_microbatches = (self.pipeline_parallel_size - pp_rank - 1) * 2
        num_warmup_microbatches += (self.num_model_chunks - 1) * self.microbatch_group_size_per_vp_stage

        if num_warmup_microbatches >= self.total_num_microbatches:
            num_warmup_microbatches = self.total_num_microbatches
            are_all_microbatches_in_warmup = True
        else:
            are_all_microbatches_in_warmup = False

        num_microbatches_remaining = self.total_num_microbatches - num_warmup_microbatches

        return (
            self.total_num_microbatches,
            are_all_microbatches_in_warmup,
            num_warmup_microbatches,
            num_microbatches_remaining,
        )

    def _get_microbatch_id_in_model_chunk(self, virtual_microbatch_id: int, forward: bool):
        if forward:
            microbatch_id = self.microbatch_id_table[virtual_microbatch_id % self.total_num_microbatches]
        else:
            # 反向传播时也使用同样的microbatch_id
            microbatch_id = self.microbatch_id_table[virtual_microbatch_id % self.total_num_microbatches]
        return microbatch_id

    def _get_model_chunk_id(self, virtual_microbatch_id: int, forward: bool):
        model_chunk_id = self.model_chunk_id_table[virtual_microbatch_id % self.total_num_microbatches]
        original_chunk_id = model_chunk_id
        if not forward:
            model_chunk_id = self.num_model_chunks - model_chunk_id - 1
            # DEBUG: Print reversal logic for backward at virtual_mb_id=0
            if virtual_microbatch_id == 0:
                print_rank_0(f"      [_get_model_chunk_id] virtual_mb_id={virtual_microbatch_id}, forward={forward}")
                print_rank_0(f"      [_get_model_chunk_id] original_chunk_id={original_chunk_id}, reversed_chunk_id={model_chunk_id}")
        return model_chunk_id

    def _create_pp_rank_schedule_dependencies(self, pp_rank: int):
        # Initialize execution order list for this PP rank
        self.pp_rank_execution_order[pp_rank] = []

        # 获取该PP rank的microbatch分布
        (
            total_num_microbatches,
            are_all_microbatches_in_warmup,
            num_warmup_microbatches,
            num_microbatches_remaining,
        ) = self._get_pp_rank_microbatches(pp_rank)

        # 为每个virtual microbatch创建调度依赖
        prev_task_id = None

        # Phase 1: Warmup阶段 - 只有前向任务
        for k in range(num_warmup_microbatches):
            model_chunk_id = self._get_model_chunk_id(k, forward=True)
            microbatch_id = self._get_microbatch_id_in_model_chunk(k, forward=True)

            forward_task = self.pp_mbid_mcid_fb_task_dict[(pp_rank, microbatch_id, model_chunk_id, TaskType.FORWARD)]

            # Record execution order
            self.pp_rank_execution_order[pp_rank].append(forward_task.task_id)

            # 同一PP rank内的顺序依赖
            if prev_task_id is not None:
                forward_task.dependencies.append(prev_task_id)

            prev_task_id = forward_task.task_id
        
        # Phase 2: Steady State阶段 - 1F1B
        for k in range(num_microbatches_remaining):
            # 前向任务
            forward_k = k + num_warmup_microbatches
            f_model_chunk_id = self._get_model_chunk_id(forward_k, forward=True)
            f_microbatch_id = self._get_microbatch_id_in_model_chunk(forward_k, forward=True)

            forward_task = self.pp_mbid_mcid_fb_task_dict[(pp_rank, f_microbatch_id, f_model_chunk_id, TaskType.FORWARD)]

            # 反向任务
            backward_k = k

            # DEBUG: Print backward task calculation for first iteration and specific pp_rank
            if k == 0 and pp_rank in [0, 3]:
                print_rank_0(f"\n>>> DEBUG 1F1B Phase (pp_rank={pp_rank}, k={k}):")
                print_rank_0(f"    backward_k = {backward_k}")
                # Get raw values from schedule table
                raw_mb_id = self.microbatch_id_table[backward_k % self.total_num_microbatches]
                raw_mc_id = self.model_chunk_id_table[backward_k % self.total_num_microbatches]
                print_rank_0(f"    schedule_table[{backward_k}] = (mb_id={raw_mb_id}, mc_id={raw_mc_id})")

            b_model_chunk_id = self._get_model_chunk_id(backward_k, forward=False)
            b_microbatch_id = self._get_microbatch_id_in_model_chunk(backward_k, forward=False)

            # DEBUG: Print results after get functions
            if k == 0 and pp_rank in [0, 3]:
                print_rank_0(f"    After _get_model_chunk_id(backward_k={backward_k}, forward=False):")
                print_rank_0(f"      b_model_chunk_id = {b_model_chunk_id} (should be reversed!)")
                print_rank_0(f"      b_microbatch_id = {b_microbatch_id}")
                print_rank_0(f"    num_model_chunks = {self.num_model_chunks}")

            backward_task = self.pp_mbid_mcid_fb_task_dict[(pp_rank, b_microbatch_id, b_model_chunk_id, TaskType.BACKWARD)]

            # DEBUG: Print task_id
            if k == 0 and pp_rank in [0, 3]:
                print_rank_0(f"    backward_task.task_id = {backward_task.task_id}\n")

            # 1F1B阶段的正确依赖关系：forward先执行，然后backward
            # 1. forward依赖于前一个任务
            if prev_task_id is not None:
                forward_task.dependencies.append(prev_task_id)

            # Record execution order: forward first
            self.pp_rank_execution_order[pp_rank].append(forward_task.task_id)

            # 更新 prev_task_id 为当前 forward
            prev_task_id = forward_task.task_id

            # 2. backward依赖于前一个任务 (刚执行完的 forward)
            if prev_task_id is not None:
                backward_task.dependencies.append(prev_task_id)

            # Record execution order: backward second
            self.pp_rank_execution_order[pp_rank].append(backward_task.task_id)

            # 3. 下一轮的任务会依赖于当前的backward
            prev_task_id = backward_task.task_id
        
        # Phase 3: Cooldown阶段 - 只有反向任务
        if not are_all_microbatches_in_warmup:
            for k in range(num_microbatches_remaining, total_num_microbatches):
                model_chunk_id = self._get_model_chunk_id(k, forward=False)
                microbatch_id = self._get_microbatch_id_in_model_chunk(k, forward=False)

                backward_task = self.pp_mbid_mcid_fb_task_dict[(pp_rank, microbatch_id, model_chunk_id, TaskType.BACKWARD)]

                # Record execution order
                self.pp_rank_execution_order[pp_rank].append(backward_task.task_id)

                # 同一PP rank内的顺序依赖
                if prev_task_id is not None:
                    backward_task.dependencies.append(prev_task_id)

                prev_task_id = backward_task.task_id

    def forward_task_iter(self):
        """Iterate all forward tasks in order: microbatch -> vpp_rank -> pp_rank"""
        for micro_batch_id in range(self.num_microbatches):
            for model_chunk_id in range(self.num_model_chunks):
                for pp_rank in range(self.pipeline_parallel_size):
                    yield (pp_rank, model_chunk_id, micro_batch_id)

    def backward_task_iter(self):
        """Iterate all backward tasks in reverse order: microbatch -> vpp_rank(reversed) -> pp_rank(reversed)"""
        for micro_batch_id in range(self.num_microbatches):
            for model_chunk_id in range(self.num_model_chunks - 1, -1, -1):
                for pp_rank in range(self.pipeline_parallel_size - 1, -1, -1):
                    yield (pp_rank, model_chunk_id, micro_batch_id)
    
    def _get_moe_pattern_for_chunk(self, pp_rank, vpp_rank, base_layout):
        """
        Determine MoE/Dense pattern for decoder layers in the current chunk
        
        Returns:
            list: pattern like ['dense', 'moe', 'moe'] for each decoder layer in chunk
        """

        args = get_args()
        
        layer_offset = 0
        for _vp_rank in range(vpp_rank + 1):
            for _pp_rank in range(
                args.pipeline_model_parallel_size if _vp_rank < vpp_rank else pp_rank
            ):
                layer_offset += args.pipeline_model_parallel_layout.layout[_pp_rank][_vp_rank].count(LayerType.decoder)

        num_decoder_layers = base_layout.count(LayerType.decoder)
        
        if num_decoder_layers == 0:
            return []
        
        # Get MoE frequency configuration
        moe_layer_freq = getattr(args, 'moe_layer_freq', None)
        total_layers = getattr(args, 'num_layers', 0)

        assert isinstance(moe_layer_freq, list)

        pattern = []
        for i in range(num_decoder_layers):
            global_layer_idx = layer_offset + i
            if global_layer_idx < len(moe_layer_freq):
                is_moe = moe_layer_freq[global_layer_idx] == 1
                pattern.append('moe' if is_moe else 'dense')
            else:
                raise ValueError(f"Global layer index {global_layer_idx} out of range for moe_layer_freq at pp_rank {pp_rank}, vpp_rank {vpp_rank}")

        return pattern 

    def _get_enhanced_layout_key(self, pp_rank, model_chunk_id):
        args = get_args()

        current_model_chunk_layout = tuple(args.pipeline_model_parallel_layout.layout[pp_rank][model_chunk_id])

        # Get MoE pattern for decoder layers in this chunk
        moe_pattern_list = self._get_moe_pattern_for_chunk(pp_rank, model_chunk_id, current_model_chunk_layout)
        
        # Get attention type (sigmoid/softmax)
        attn_pattern_list = ["softmax" for _ in moe_pattern_list]

        # Build simplified key
        enhanced_layout = []
        decoder_idx = 0  # Separate index for decoder layers only

        for layer_idx, layer_type in enumerate(current_model_chunk_layout):
            # Handle LayerType enum cases
            if layer_type == LayerType.embedding:
                enhanced_layout.append('embedding')
            elif layer_type == LayerType.loss:
                enhanced_layout.append('loss')
            elif layer_type == LayerType.encoder:
                enhanced_layout.append('encoder')
            elif layer_type == LayerType.mtp:
                enhanced_layout.append('mtp')
            elif layer_type == LayerType.decoder:
                # Use decoder_idx to access decoder-specific patterns
                if decoder_idx < len(attn_pattern_list) and decoder_idx < len(moe_pattern_list):
                    attn_type = f"attn_{attn_pattern_list[decoder_idx]}"
                    mlp_type = f"mlp_{moe_pattern_list[decoder_idx]}"            
                    enhanced_layout.append(('decoder_layer', attn_type, mlp_type))
                    decoder_idx += 1  # Increment decoder index
                else:
                    enhanced_layout.append(('decoder_layer', 'attn_softmax', 'mlp_dense'))  # Default fallback
                    decoder_idx += 1
            else:
                raise ValueError(f"Unknown layer type {layer_type} at pp_rank {pp_rank}, vpp_rank {model_chunk_id}")
            
        return tuple(enhanced_layout)

    def does_need_to_execute_task(self, now_task):
        args = get_args()
        now_enhanced_layout = self._get_enhanced_layout_key(now_task.pp_rank, now_task.model_chunk_id)
        if args.execute_mode == 'router_balanced':
            task_key = (now_enhanced_layout, now_task.task_type)
        elif args.execute_mode == 'load_router_map':
            task_key = (now_enhanced_layout, now_task.microbatch_id, now_task.task_type)
        else:
            raise ValueError(f'Unknown execute_mode: {args.execute_mode}')        
        if task_key in self.task_recoreder:
            return False
        else:
            return True

    def record_task(self, task):
        args = get_args()
        now_enhanced_layout = self._get_enhanced_layout_key(task.pp_rank, task.model_chunk_id)
        if args.execute_mode == 'router_balanced':
            task_key = (now_enhanced_layout, task.task_type)
        elif args.execute_mode == 'load_router_map':
            task_key = (now_enhanced_layout, task.microbatch_id, task.task_type)
        else:
            raise ValueError(f'Unknown execute_mode: {args.execute_mode}')
        self.task_recoreder[task_key] = task.duration

    def _assign_task_duration(self, task):
        """
        Assign duration to task from task_recorder and mark as finished.

        This is used when a task doesn't need actual execution (same layout already executed).

        Args:
            task: Task object to assign duration to

        Returns:
            task: Updated task with duration and finished=True
        """
        args = get_args()
        now_enhanced_layout = self._get_enhanced_layout_key(task.pp_rank, task.model_chunk_id)

        if args.execute_mode == 'router_balanced':
            task_key = (now_enhanced_layout, task.task_type)
        elif args.execute_mode == 'load_router_map':
            task_key = (now_enhanced_layout, task.microbatch_id, task.task_type)
        else:
            raise ValueError(f'Unknown execute_mode: {args.execute_mode}')

        if task_key in self.task_recoreder:
            task.duration = self.task_recoreder[task_key]
            task.finished = True
            print_rank_0(f'\tAssigned duration from recorded task: {task.duration:.2f}ms')
        else:
            raise RuntimeError(
                f'Task key {task_key} not found in task_recorder. '
                f'This should not happen - make sure a task with the same layout was executed first.'
            )

        return task

    def run_global_step(self):
        args = get_args()

        # Check if we should skip execution and load from previous results
        if args.skip_execute:
            print_rank_0("\n" + "="*80)
            print_rank_0("SKIP EXECUTE MODE: Loading Previous Simulation Results")
            print_rank_0("="*80)
            print_rank_0(f"Loading results from: {args.load_result_dir}")

            # Skip Phase 1 and Phase 2, directly go to Phase 3 with loaded data
            # Only rank 0 performs analysis to avoid file I/O conflicts
            if torch.distributed.get_rank() == 0:
                print_rank_0("\n" + "="*80)
                print_rank_0("PHASE 3: Analyzing Global Batch Results (from loaded data)")
                print_rank_0("="*80)
                self.analyze_global_batch(load_from_dir=args.load_result_dir)
            return

        # Normal execution path
        # Phase 1: Execute all forward tasks
        print_rank_0("\n" + "="*80)
        print_rank_0("PHASE 1: Executing Forward Tasks")
        print_rank_0("="*80)
        for pp_rank, model_chunk_id, micro_batch_id in self.forward_task_iter():
            now_task = self.pp_mbid_mcid_fb_task_dict[(pp_rank, micro_batch_id, model_chunk_id, TaskType.FORWARD)]
            print_rank_0("'During run_global_step. Prepare to run FORWARD task. \n"
                         f"\tpp_rank: {pp_rank}; model_chunk_id: {model_chunk_id}; micro_batch_id: {micro_batch_id}")

            # Determine execution requirements
            need_execute_task = self.does_need_to_execute_task(now_task)
            need_create_model = (pp_rank not in self.pp_static_memory_dict)

            # Execute task if needed (for timing or static memory collection)
            if need_execute_task or need_create_model:
                if need_execute_task:
                    print_rank_0(f'\tExecuting task for timing')
                else:
                    print_rank_0(f'\tCreating model for static memory collection only')

                # Execute task (may skip computation if only collecting static memory)
                now_task = self._execute_forward_task(
                    now_task,
                    self.train_data_iterator,
                    execute_task=need_execute_task
                )

                # Handle task completion
                if need_execute_task:
                    # Task was actually executed - record duration
                    self.record_task(now_task)
                else:
                    # Model was created but task not executed - assign duration from recorder
                    now_task = self._assign_task_duration(now_task)
            else:
                # No execution or model creation needed - assign duration from recorder
                print_rank_0(f'\tSkipping task execution')
                now_task = self._assign_task_duration(now_task)

        # Phase 2: Execute all backward tasks
        print_rank_0("\n" + "="*80)
        print_rank_0("PHASE 2: Executing Backward Tasks")
        print_rank_0("="*80)
        for pp_rank, model_chunk_id, micro_batch_id in self.backward_task_iter():
            now_task = self.pp_mbid_mcid_fb_task_dict[(pp_rank, micro_batch_id, model_chunk_id, TaskType.BACKWARD)]
            print_rank_0("'During run_global_step. Prepare to run BACKWARD task. \n"
                         f"\tpp_rank: {pp_rank}; model_chunk_id: {model_chunk_id}; micro_batch_id: {micro_batch_id}")

            # Check if backward task needs execution
            need_execute_task = self.does_need_to_execute_task(now_task)

            # Ensure corresponding forward task is finished (dependency requirement)
            if need_execute_task:
                forward_task = self.pp_mbid_mcid_fb_task_dict[(pp_rank, micro_batch_id, model_chunk_id, TaskType.FORWARD)]
                if not forward_task.finished:
                    print_rank_0(f'\tCorresponding forward task not finished, executing it first...')
                    forward_task = self._execute_forward_task(forward_task, self.train_data_iterator, execute_task=True)
                    self.record_task(forward_task)
                    print_rank_0(f'\tForced forward execution completed for backward dependency')

            # Execute or assign duration
            if need_execute_task:
                print_rank_0(f'\tExecuting backward task for timing')
                now_task = self._execute_backward_task(now_task, self.train_data_iterator)
                self.record_task(now_task)
            else:
                print_rank_0(f'\tSkipping backward task execution')
                now_task = self._assign_task_duration(now_task)

        # Phase 3: Collect and save results
        # Only rank 0 performs file I/O operations to avoid conflicts
        if torch.distributed.get_rank() == 0:
            print_rank_0("\n" + "="*80)
            print_rank_0("PHASE 3: Collecting and Saving Results")
            print_rank_0("="*80)
            self._collect_and_save_results()

            # Phase 4: Analyze and generate reports
            print_rank_0("\n" + "="*80)
            print_rank_0("PHASE 4: Analyzing Global Batch Results")
            print_rank_0("="*80)
            self.analyze_global_batch(load_from_dir=args.simulate_result_dir)

    def _collect_and_save_results(self):
        """Collect finished tasks from execution and save auxiliary files

        This method is called after Phase 2 (backward execution) in execute mode.
        It collects all finished tasks from the execution and saves them along with
        auxiliary files (task orders, VPP layouts, memory info, etc.) to simulate_result_dir.
        """
        args = get_args()

        # Ensure simulate_result_dir is specified
        simulate_result_dir = getattr(args, 'simulate_result_dir', None)
        if simulate_result_dir is None:
            raise ValueError("--simulate-result-dir must be specified in execute mode")

        print_rank_0(f"Collecting results and saving to: {simulate_result_dir}")

        # Collect finished tasks from current execution
        # IMPORTANT: Collect tasks in execution order (not dict iteration order)
        from megatron.training.simulation.analyzer import parse_task_id

        pp_finished_task_queue_dict = {}
        for pp_rank in range(self.pipeline_parallel_size):
            pp_finished_task_queue_dict[pp_rank] = []

            # Use execution_order to collect tasks in correct order
            for task_id in self.pp_rank_execution_order[pp_rank]:
                # Parse task_id to get task information
                task_info = parse_task_id(task_id)

                # Build task_key to lookup task object
                task_type_enum = TaskType.FORWARD if task_info['task_type'] == 'forward' else TaskType.BACKWARD
                task_key = (
                    task_info['pp_rank'],
                    task_info['microbatch_id'],
                    task_info['model_chunk_id'],
                    task_type_enum
                )

                # Get task object from dict
                if task_key in self.pp_mbid_mcid_fb_task_dict:
                    task = self.pp_mbid_mcid_fb_task_dict[task_key]
                    if task.finished:
                        pp_finished_task_queue_dict[pp_rank].append(task)
                else:
                    print_rank_0(f"Warning: task {task_id} not found in task dict")

        # Count total finished tasks
        total_finished_tasks = sum(len(tasks) for tasks in pp_finished_task_queue_dict.values())
        print_rank_0(f"Collected {total_finished_tasks} finished tasks across {self.pipeline_parallel_size} PP ranks")

        # Verification: Print task collection order for debugging
        print_rank_0("\n" + "=" * 80)
        print_rank_0("Task Collection Order Verification:")
        print_rank_0("=" * 80)
        for pp_rank in range(min(2, self.pipeline_parallel_size)):  # Print first 2 PP ranks
            print_rank_0(f"PP Rank {pp_rank}: {len(pp_finished_task_queue_dict[pp_rank])} tasks collected")
            for i, task in enumerate(pp_finished_task_queue_dict[pp_rank][:5]):  # Print first 5 tasks
                print_rank_0(f"  [{i}] {task.task_id}")
        print_rank_0("=" * 80 + "\n")

        # Save auxiliary files
        self._save_auxiliary_files(pp_finished_task_queue_dict, simulate_result_dir)

        print_rank_0("✅ Results collection and saving complete!")

    def analyze_global_batch(self, load_from_dir):
        """Analyze complete global batch execution including timeline, throughput, and memory

        This function performs comprehensive analysis of the completed global batch step:
        1. Load finished tasks from load_from_dir
        2. Generate timeline visualization (Chrome Trace format)
        3. Calculate throughput statistics (TFLOPS, tokens/day)
        4. Analyze memory usage and create visualization charts

        Args:
            load_from_dir: Directory to load finished tasks and auxiliary files from.
                          In execute mode: load from simulate_result_dir
                          In skip-execute mode: load from load_result_dir
        """
        args = get_args()

        print_rank_0("=" * 80)
        print_rank_0("Starting Global Batch Analysis")
        print_rank_0("=" * 80)

        # Step 1: Load finished tasks from directory
        print_rank_0(f"Loading finished tasks from: {load_from_dir}")
        pp_finished_task_queue_dict = self._load_finished_tasks(load_from_dir)

        # Use load_from_dir as the analysis directory
        analysis_dir = load_from_dir

        # Step 2: Generate timeline visualization
        print_rank_0("\n" + "-" * 80)
        print_rank_0("Generating timeline visualization...")
        print_rank_0("-" * 80)
        from megatron.training.simulation import analyzer
        from megatron.training.simulation.task import TaskType

        analyzer.save_gbs_trace_timeline(
            pp_finished_task_queue_dict=pp_finished_task_queue_dict,
            pipeline_parallel_size=self.pipeline_parallel_size,
            num_microbatches=self.num_microbatches,
            num_model_chunks=self.num_model_chunks,
            profile_dir=analysis_dir,
            task_type_enum=TaskType
        )

        # Step 3: Display throughput statistics
        print_rank_0("\n" + "-" * 80)
        print_rank_0("Calculating throughput statistics...")
        print_rank_0("-" * 80)

        analyzer.display_gbs_statistics(
            pp_finished_task_queue_dict=pp_finished_task_queue_dict,
            pipeline_parallel_size=self.pipeline_parallel_size,
            num_microbatches=self.num_microbatches,
            num_model_chunks=self.num_model_chunks,
            args=args,
            task_type_enum=TaskType
        )

        # Step 4: Display memory statistics
        print_rank_0("\n" + "-" * 80)
        print_rank_0("Analyzing memory usage...")
        print_rank_0("-" * 80)

        # Always enable memory analysis in simulation mode
        enable_memory_snapshot = True
        execute_mode = getattr(args, 'execute_mode', 'router_balanced')

        analyzer.display_memory_statistics(
            profile_dir=analysis_dir,
            enable_memory_snapshot=enable_memory_snapshot,
            execute_mode=execute_mode,
            pipeline_parallel_size=self.pipeline_parallel_size
        )

        print_rank_0("\n" + "=" * 80)
        print_rank_0("✅ Global Batch Analysis Complete!")
        print_rank_0("=" * 80)
        print_rank_0(f"Analysis results from: {analysis_dir}")

    def _load_finished_tasks(self, load_result_dir):
        """Load finished tasks from a previous simulation run

        Args:
            load_result_dir: Directory containing saved simulation results

        Returns:
            pp_finished_task_queue_dict: Dictionary of finished tasks per PP rank
        """
        finished_tasks_path = os.path.join(load_result_dir, 'finished_tasks.json')

        if not os.path.exists(finished_tasks_path):
            raise FileNotFoundError(
                f"Cannot find finished_tasks.json in {load_result_dir}. "
                "Please ensure the directory contains results from a previous simulation run."
            )

        print_rank_0(f"Loading finished tasks from: {finished_tasks_path}")
        with open(finished_tasks_path, 'r') as f:
            finished_tasks_json = json.load(f)

        # Reconstruct Task objects from JSON
        pp_finished_task_queue_dict = {}
        for pp_rank_str, task_list in finished_tasks_json.items():
            pp_rank = int(pp_rank_str)
            pp_finished_task_queue_dict[pp_rank] = [Task.from_dict(task_dict) for task_dict in task_list]

        # Count total tasks loaded
        total_tasks = sum(len(tasks) for tasks in pp_finished_task_queue_dict.values())
        print_rank_0(f"Loaded {total_tasks} finished tasks from {len(pp_finished_task_queue_dict)} PP ranks")

        return pp_finished_task_queue_dict

    def _save_auxiliary_files(self, pp_finished_task_queue_dict, simulate_result_dir):
        """Save auxiliary files needed for analysis

        Args:
            pp_finished_task_queue_dict: Dictionary of finished tasks per PP rank
            simulate_result_dir: Directory to save files
        """
        # Ensure directory exists
        os.makedirs(simulate_result_dir, exist_ok=True)

        # 1. Save pp_task_orders.json using recorded execution order
        pp_task_orders = {}
        for pp_rank in range(self.pipeline_parallel_size):
            # Use the execution order recorded during schedule creation
            pp_task_orders[pp_rank] = self.pp_rank_execution_order[pp_rank]

        pp_task_orders_path = os.path.join(simulate_result_dir, 'pp_task_orders.json')
        with open(pp_task_orders_path, 'w') as f:
            json.dump(pp_task_orders, f, indent=2)
        print_rank_0(f"Saved task orders to: {pp_task_orders_path}")

        # 2. Save all_pp_vpp_layout.json
        all_pp_vpp_layout = {}
        for pp_rank in range(self.pipeline_parallel_size):
            vpp_layout = {}
            for vpp_rank in range(self.num_model_chunks):
                # Get enhanced layout key for this pp_rank and vpp_rank
                enhanced_layout = self._get_enhanced_layout_key(pp_rank, vpp_rank)
                # Convert tuple to list for JSON serialization
                vpp_layout[vpp_rank] = list(enhanced_layout) if isinstance(enhanced_layout, tuple) else enhanced_layout
            all_pp_vpp_layout[pp_rank] = vpp_layout

        vpp_layout_path = os.path.join(simulate_result_dir, 'all_pp_vpp_layout.json')
        with open(vpp_layout_path, 'w') as f:
            json.dump(all_pp_vpp_layout, f, indent=2)
        print_rank_0(f"Saved VPP layouts to: {vpp_layout_path}")

        # 3. Save static_memory_info.json with actual model memory info
        static_memory_info = {
            "executor_0": {}
        }

        for pp_rank in range(self.pipeline_parallel_size):
            # Use actual static memory if available, otherwise use 0.0
            if pp_rank in self.pp_static_memory_dict:
                params_gb = self.pp_static_memory_dict[pp_rank]['params_memory_gb']
                grads_gb = self.pp_static_memory_dict[pp_rank]['grads_memory_gb']
            else:
                # This should not happen in normal simulation, but provide fallback
                params_gb = 0.0
                grads_gb = 0.0
                print_rank_0(f"Warning: PP rank {pp_rank} static memory not collected, using 0.0")

            static_memory_info["executor_0"][str(pp_rank)] = {
                "static": {
                    "params_memory_gb": params_gb,
                    "grads_memory_gb": grads_gb
                },
                "dynamic": {}
            }

        static_memory_path = os.path.join(simulate_result_dir, 'static_memory_info.json')
        with open(static_memory_path, 'w') as f:
            json.dump(static_memory_info, f, indent=2)
        print_rank_0(f"Saved static memory info to: {static_memory_path}")

        # 4. Save finished_tasks.json - complete task objects for re-analysis
        finished_tasks_json = {}
        for pp_rank in range(self.pipeline_parallel_size):
            finished_tasks_json[str(pp_rank)] = [task.to_dict() for task in pp_finished_task_queue_dict[pp_rank]]

        finished_tasks_path = os.path.join(simulate_result_dir, 'finished_tasks.json')
        with open(finished_tasks_path, 'w') as f:
            json.dump(finished_tasks_json, f, indent=2)
        print_rank_0(f"Saved finished tasks to: {finished_tasks_path}")

        # 5. Save task_durations.json - human-readable task duration records
        task_durations = {}
        for pp_rank in range(self.pipeline_parallel_size):
            for task in pp_finished_task_queue_dict[pp_rank]:
                task_durations[task.task_id] = {
                    'duration_ms': task.duration,
                    'pp_rank': task.pp_rank,
                    'microbatch_id': task.microbatch_id,
                    'model_chunk_id': task.model_chunk_id,
                    'task_type': task.task_type.value
                }

        task_durations_path = os.path.join(simulate_result_dir, 'task_durations.json')
        with open(task_durations_path, 'w') as f:
            json.dump(task_durations, f, indent=2)
        print_rank_0(f"Saved task durations to: {task_durations_path}")

    def _execute_forward_task(self, task: Task, data_iterator, execute_task: bool = True):
        """Execute forward task with actual computation

        Args:
            task: Task object containing pp_rank, vpp_rank (model_chunk_id), microbatch_id
            data_iterator: training data iterator
            execute_task: If True, execute actual forward computation for timing.
                         If False, only create model and collect static memory.

        Returns:
            task: Updated task with duration and finished status
        """
        args = get_args()
        pp_rank = task.pp_rank
        vpp_rank = task.model_chunk_id
        microbatch_id = task.microbatch_id

        global_rank = torch.distributed.get_rank()

        # Determine if model needs to be recreated
        need_recreate = (
            self.current_model is None or
            self.current_model_pp_rank != pp_rank
        )

        if need_recreate:
            # Clear old model if exists
            if self.current_model is not None:
                if global_rank == 0:
                    print_rank_0(f"Clearing old model: pp_rank={self.current_model_pp_rank}, "
                                f"vpp_rank={self.current_model_vpp_rank}")
                clear_model_mem(self.current_model)
                self.current_model = None

            # Create new model
            if global_rank == 0:
                print_rank_0(f"Creating new model for pp_rank={pp_rank}, vpp_rank={vpp_rank}")
            self.current_model = create_model(pp_rank, self.model_provider)
            self.current_model_pp_rank = pp_rank
            self.current_model_vpp_rank = vpp_rank

            # Calculate and store static memory for this PP rank (only once per PP rank)
            if pp_rank not in self.pp_static_memory_dict:
                params_gb, grads_gb = calculate_model_static_memory(self.current_model)
                self.pp_static_memory_dict[pp_rank] = {
                    'params_memory_gb': params_gb,
                    'grads_memory_gb': grads_gb
                }
                if global_rank == 0:
                    print_rank_0(f"PP rank {pp_rank} static memory: params={params_gb:.4f} GB, grads={grads_gb:.4f} GB")
        else:
            if global_rank == 0:
                print_rank_0(f"Reusing existing model for pp_rank={pp_rank}, vpp_rank={vpp_rank}")

        # If we don't need to execute the task (only collecting static memory), return early
        if not execute_task:
            if global_rank == 0:
                print_rank_0(f"Skipping forward execution (static memory collected)")
            # Mark task as finished with 0 duration (will be filled from task_recorder later)
            task.duration = 0.0
            task.finished = False  # Not actually executed
            return task

        # Prepare input tensor
        input_tensor = prepare_input_tensor(
            pp_rank, vpp_rank, microbatch_id, self.task_output_tensor_dict
        )

        input_info = f"shape: {input_tensor.shape}" if input_tensor is not None else "None"
        print_rank_0(f"Forward task Before execute. - pp_rank: {pp_rank}, vpp_rank: {vpp_rank}, "
                    f"microbatch_id: {microbatch_id}, input_tensor: {input_info}")

        # Execute forward with timing
        output_tensor, avg_forward_time = execute_forward_with_timing(
            model=self.current_model,
            vpp_rank=vpp_rank,
            input_tensor=input_tensor,
            data_iterator=data_iterator,
            forward_step_func=self.forward_step_func,
            args=args,
            microbatch_id=microbatch_id,
            warmup_times=15,
            measure_times=10,
        )

        # Cache output tensor for downstream stages
        print_rank_0(f"Forward task After execute. - pp_rank: {pp_rank}, vpp_rank: {vpp_rank}, "
                    f"microbatch_id: {microbatch_id}, output_tensor: {output_tensor}")
        cache_key = (pp_rank, vpp_rank, microbatch_id)
        self.task_output_tensor_dict[cache_key] = output_tensor.detach().clone()

        if global_rank == 0:
            output_info = f"shape: {output_tensor.shape}" if output_tensor is not None else "None"
            print_rank_0(f"Forward task completed - output_tensor: {output_info}, "
                        f"duration: {avg_forward_time:.2f}ms")

        # Update task
        task.duration = avg_forward_time
        task.finished = True

        # Clean up output tensor reference (keep cached version only)
        del output_tensor

        return task

    def _execute_backward_task(self, task: Task, data_iterator):
        """Execute backward task with actual computation

        Args:
            task: Task object containing pp_rank, vpp_rank (model_chunk_id), microbatch_id
            data_iterator: training data iterator

        Returns:
            task: Updated task with duration and finished status
        """
        args = get_args()
        pp_rank = task.pp_rank
        vpp_rank = task.model_chunk_id
        microbatch_id = task.microbatch_id

        global_rank = torch.distributed.get_rank()

        # Determine if model needs to be recreated
        need_recreate = (
            self.current_model is None or
            self.current_model_pp_rank != pp_rank
        )

        if need_recreate:
            # Clear old model if exists
            if self.current_model is not None:
                if global_rank == 0:
                    print_rank_0(f"Clearing old model: pp_rank={self.current_model_pp_rank}, "
                                f"vpp_rank={self.current_model_vpp_rank}")
                clear_model_mem(self.current_model)
                self.current_model = None

            # Create new model
            if global_rank == 0:
                print_rank_0(f"Creating new model for pp_rank={pp_rank}, vpp_rank={vpp_rank}")
            self.current_model = create_model(pp_rank, self.model_provider)
            self.current_model_pp_rank = pp_rank
            self.current_model_vpp_rank = vpp_rank
            # Note: Static memory is already collected in forward task, no need to recalculate here
        else:
            if global_rank == 0:
                print_rank_0(f"Reusing existing model for pp_rank={pp_rank}, vpp_rank={vpp_rank}")

        # Prepare input tensor from forward pass output
        input_tensor = prepare_input_tensor(
            pp_rank, vpp_rank, microbatch_id, self.task_output_tensor_dict
        )

        # Clone and enable gradient for backward
        if input_tensor is not None:
            input_tensor = input_tensor.clone().detach().requires_grad_(True)

        if global_rank == 0:
            input_info = f"shape: {input_tensor.shape}" if input_tensor is not None else "None"
            print_rank_0(f"Backward task - pp_rank: {pp_rank}, vpp_rank: {vpp_rank}, "
                        f"microbatch_id: {microbatch_id}, input_tensor: {input_info}")

        # Execute backward with timing (output_tensor_grad will be prepared inside)
        input_tensor_grad, avg_backward_time = execute_backward_with_timing(
            model=self.current_model,
            vpp_rank=vpp_rank,
            input_tensor=input_tensor,
            data_iterator=data_iterator,
            forward_step_func=self.forward_step_func,
            args=args,
            microbatch_id=microbatch_id,
            num_model_chunks=self.num_model_chunks,
            task_input_tensor_grad_dict=self.task_input_tensor_grad_dict,
            warmup_times=15,
            measure_times=10,
        )

        # Cache input gradient tensor for upstream stages
        if input_tensor_grad is not None:
            cache_key = (pp_rank, vpp_rank, microbatch_id)
            self.task_input_tensor_grad_dict[cache_key] = input_tensor_grad.detach().clone()

        if global_rank == 0:
            input_grad_info = f"shape: {input_tensor_grad.shape}" if input_tensor_grad is not None else "None"
            print_rank_0(f"Backward task completed - input_tensor_grad: {input_grad_info}, "
                        f"duration: {avg_backward_time:.2f}ms")

        # Update task
        task.duration = avg_backward_time
        task.finished = True

        # Clean up gradient reference (keep cached version only)
        del input_tensor_grad

        return task

