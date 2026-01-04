# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Model Executor Utilities for VPP Training Simulation"""

import os
import time
import torch
import torch.distributed

from contextlib import nullcontext
from functools import partial

from megatron.core import parallel_state
from megatron.training.global_vars import get_args
from megatron.core.pipeline_parallel.schedules import forward_step
from megatron.core.utils import get_model_config, get_attr_wrapped_model
from megatron.training.utils import print_rank_0


def create_model(pp_rank, model_provider):
    """Create model for specified pipeline rank

    Args:
        pp_rank: pipeline parallel rank
        model_provider: function to create model

    Returns:
        model: created model (list of model chunks for VPP)
    """
    # Reset MTP embedding for second MTP build
    from megatron.training import get_model
    #from megatron.core.transformer.multi_token_prediction import MTPEmbeddingHelper
    #MTPEmbeddingHelper.set_embedding(None)

    # Set pipeline rank and world size for correct is_last_stage calculation
    args = get_args()
    parallel_state.set_pipeline_model_parallel_rank(pp_rank)
    parallel_state.set_pipeline_model_parallel_world_size(args.pipeline_model_parallel_size)
    model = get_model(model_provider)

    return model


def clear_model_mem(model):
    """Clear model memory including DDP buffers

    Args:
        model: model to clear (can be a list for VPP)
    """
    from megatron.core import DistributedDataParallel
    import gc

    def _clear_single_model(single_model):
        """Clear a single model chunk"""
        if isinstance(single_model, DistributedDataParallel):
            def _clear_buffer(buffer):
                # Clear param_data
                if buffer.param_data is not None:
                    buffer.param_data.untyped_storage().resize_(0)
                    buffer.param_data = None

                # Clear grad_data
                if buffer.grad_data is not None:
                    buffer.grad_data.untyped_storage().resize_(0)
                    buffer.grad_data = None

            # Clear standard buffers
            if hasattr(single_model, 'buffers'):
                for buffer in single_model.buffers:
                    _clear_buffer(buffer)

            # Clear expert parallel buffers
            if hasattr(single_model, 'expert_parallel_buffers'):
                for buffer in single_model.expert_parallel_buffers:
                    _clear_buffer(buffer)

    # Handle list of model chunks (VPP)
    if isinstance(model, list):
        for model_chunk in model:
            _clear_single_model(model_chunk)
    else:
        _clear_single_model(model)

    # Delete model reference
    del model

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Force garbage collection
    gc.collect()


def _create_timeline_profiler(pp_rank, vpp_rank, microbatch_id, profile_dir, global_rank, task_type='FORWARD'):
    """Create timeline profiler for performance analysis

    Args:
        pp_rank: pipeline parallel rank
        vpp_rank: virtual pipeline rank (model_chunk_id)
        microbatch_id: microbatch id
        profile_dir: directory to save profiler results
        global_rank: global rank for multi-GPU setup
        task_type: type of task ('FORWARD' or 'BACKWARD')

    Returns:
        torch.profiler.profile: configured profiler instance
    """
    current_profile_sub_dir = f'pp_{pp_rank}-vpp_{vpp_rank}-microbatch_id_{microbatch_id}-task_type_{task_type}'

    def trace_handler():
        # Only save for rank 0 to reduce overhead
        if global_rank == 0:
            return torch.profiler.tensorboard_trace_handler(
                os.path.join(profile_dir, current_profile_sub_dir),
                worker_name=f"torch_profiler-global_rank_{global_rank}",
                use_gzip=True,
            )
        else:
            def _dummy_writer(p):
                # Do nothing for non-zero ranks
                pass
            return _dummy_writer

    default_profiler_config = dict(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_stack=True,
        record_shapes=True,
        profile_memory=False,
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=1,
            repeat=1,
            skip_first=0,
        ),
        on_trace_ready=trace_handler(),
    )

    return torch.profiler.profile(**default_profiler_config)


def _stop_memory_tracking():
    """Stop memory tracking and clear history"""
    torch.cuda.memory._record_memory_history(enabled=None)


def _start_memory_tracking():
    """Start memory tracking for snapshot generation"""
    # Stop any previous tracking to ensure clean state
    _stop_memory_tracking()
    torch.cuda.memory._record_memory_history(
        enabled='all',  # Keep information for currently allocated memory
        max_entries=10000000,
    )


def _save_memory_snapshot(pp_rank, vpp_rank, microbatch_id, profile_dir, global_rank, task_type='FORWARD'):
    """Save memory snapshot to file

    Args:
        pp_rank: pipeline parallel rank
        vpp_rank: virtual pipeline rank (model_chunk_id)
        microbatch_id: microbatch id
        profile_dir: directory to save snapshot
        global_rank: global rank for multi-GPU setup
        task_type: type of task ('FORWARD' or 'BACKWARD')
    """
    # Only save for rank 0 to reduce overhead
    if global_rank == 0:
        os.makedirs(profile_dir, exist_ok=True)

        # Convert task_type to standard format
        if task_type == 'FORWARD' or task_type.upper() == 'FORWARD':
            task_type_str = 'TaskType.FORWARD'
        elif task_type == 'BACKWARD' or task_type.upper() == 'BACKWARD':
            task_type_str = 'TaskType.BACKWARD'
        else:
            # Keep original task_type if it's already in the correct format
            task_type_str = task_type

        snapshot_file = os.path.join(
            profile_dir,
            f'memory_snapshot-global_rank{global_rank}-pp_rank_{pp_rank}-vpp_rank_{vpp_rank}-microbatch_id_{microbatch_id}-task_type_{task_type_str}.pickle'
        )
        torch.cuda.memory._dump_snapshot(snapshot_file)
        print(f"Memory snapshot saved: {snapshot_file}")


def prepare_input_tensor(pp_rank, vpp_rank, microbatch_id, task_output_tensor_dict):
    """Prepare input tensor for forward pass

    Args:
        pp_rank: pipeline parallel rank
        vpp_rank: virtual pipeline rank (model_chunk_id)
        microbatch_id: microbatch id
        task_output_tensor_dict: dictionary storing task output tensors

    Returns:
        input_tensor: input tensor for forward pass (None for first stage)
    """
    # First stage gets data from data_iterator
    if pp_rank == 0 and vpp_rank == 0 and microbatch_id == 0:
        return None

    # Other stages get input from previous stage output
    cache_key = (0, 0, 0)  # Always use output from pp0_vpp0

    if cache_key in task_output_tensor_dict:
        input_tensor = task_output_tensor_dict[cache_key]
        return input_tensor
    else:
        raise KeyError(f"Input tensor not found for pp_rank={pp_rank}, vpp_rank={vpp_rank}, "
                      f"microbatch_id={microbatch_id}. Cache key: {cache_key} not in task_output_tensor_dict")


def execute_forward_with_timing(
    model,
    vpp_rank,
    input_tensor,
    data_iterator,
    forward_step_func,
    args,
    microbatch_id,
    warmup_times=15,
    measure_times=10
):
    """Execute forward pass with timing measurement

    Args:
        model: model to execute (list of model chunks)
        vpp_rank: virtual pipeline rank (model_chunk_id)
        input_tensor: input tensor
        data_iterator: training data iterator
        forward_step_func: forward step function to use
        args: training arguments
        microbatch_id: microbatch id
        warmup_times: number of warmup iterations (default 15)
        measure_times: number of measurement iterations (default 10)

    Returns:
        output_tensor: output tensor from forward pass
        avg_forward_time: average forward time in milliseconds
    """
    # Get ranks for profiling
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    # Get profiling configuration
    simulate_result_dir = args.simulate_result_dir
    # Prepare forward step parameters
    num_microbatches = args.global_batch_size // parallel_state.get_data_parallel_world_size()
    forward_data_store = []
    collect_non_loss_data = False
    config = get_model_config(model[0])
    checkpoint_activations_microbatch = None
    cp_group_size = parallel_state.get_context_parallel_world_size()

    # Determine if this is the last pipeline stage (both PP and VPP)
    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    is_last_stage = (pp_rank == pipeline_parallel_size - 1 and vpp_rank == len(model) - 1)

    # Create timeline profiler
    timeline_profiler = _create_timeline_profiler(
        pp_rank, vpp_rank, microbatch_id, simulate_result_dir, global_rank, task_type='FORWARD'
    )

    # Warmup phase with profiling
    with timeline_profiler as p:
        for i in range(warmup_times):
            output_tensor, num_tokens = forward_step(
                forward_step_func,
                data_iterator,
                model[vpp_rank],
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                cp_group_size,
                collect_non_loss_data=collect_non_loss_data,
                checkpoint_activations_microbatch=checkpoint_activations_microbatch,
                is_first_microbatch=False,
                current_microbatch=microbatch_id,
                vp_stage=vpp_rank,
                is_last_stage=is_last_stage,
            )
            torch.cuda.synchronize()
            torch.distributed.barrier()
            forward_data_store = []

            # Enable memory tracking after first warmup iteration
            if i == 1:
                _start_memory_tracking()

            # Call profiler step
            p.step()

            # Save memory snapshot after first warmup iteration
            if i == 1 and simulate_result_dir is not None:
                _save_memory_snapshot(pp_rank, vpp_rank, microbatch_id, simulate_result_dir, global_rank, task_type='FORWARD')

    # Measurement phase - measure time with CUDA events
    total_forward_time = 0
    measure_skip_times = 5  # Skip first few measurements

    for i in range(measure_times):
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)

        forward_start.record()
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model[vpp_rank],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=False,
            current_microbatch=microbatch_id,
            vp_stage=vpp_rank,
            is_last_stage=is_last_stage,
        )
        forward_end.record()
        forward_data_store = []

        torch.cuda.synchronize()
        torch.distributed.barrier()

        forward_time = forward_start.elapsed_time(forward_end)
        if i >= measure_skip_times:
            total_forward_time += forward_time

    # Calculate average time
    torch.cuda.synchronize()
    measured_iterations = measure_times - measure_skip_times
    avg_forward_time = total_forward_time / measured_iterations

    return output_tensor, avg_forward_time


def prepare_output_grad_tensor(pp_rank, vpp_rank, microbatch_id, task_input_tensor_grad_dict, num_model_chunks, forward_output_tensor=None):
    """Prepare output gradient tensor for backward pass

    Args:
        pp_rank: pipeline parallel rank
        vpp_rank: virtual pipeline rank (model_chunk_id)
        microbatch_id: microbatch id
        task_input_tensor_grad_dict: dictionary storing task input gradient tensors
        num_model_chunks: number of model chunks (for VPP)
        forward_output_tensor: forward output tensor for generating mock gradient if needed

    Returns:
        output_tensor_grad: output gradient tensor for backward pass (None for last stage)
    """
    # Debug: Log pipeline stage information
    vpp_world_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()
    print_rank_0(f"[DEBUG prepare_output_grad_tensor] pp_rank={pp_rank}, vpp_rank={vpp_rank}, "
                f"microbatch_id={microbatch_id}, vpp_world_size={vpp_world_size}")

    # Last stage has no output gradient
    is_last = parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vpp_rank)
    print_rank_0(f"[DEBUG prepare_output_grad_tensor] is_pipeline_last_stage={is_last}")

    if is_last:
        return None

    # Other stages get gradient from next stage
    # Assuming the gradient comes from the last pp stage's output
    # Use pp=-1, vpp=-1 to get the last stage's output
    last_pp_rank = parallel_state.get_pipeline_model_parallel_world_size() - 1
    last_vpp_rank = num_model_chunks - 1
    cache_key = (last_pp_rank, last_vpp_rank, microbatch_id)

    if cache_key in task_input_tensor_grad_dict:
        output_tensor_grad = task_input_tensor_grad_dict[cache_key]
        return output_tensor_grad
    else:
        # If not found, generate mock gradient for simulation
        # This is needed for performance profiling when tasks are executed out of dependency order
        if forward_output_tensor is not None:
            print_rank_0(f"Warning: Gradient not found for pp_rank={pp_rank}, vpp_rank={vpp_rank}, "
                        f"microbatch_id={microbatch_id}. Generating mock gradient.")
            return torch.ones_like(forward_output_tensor)
        else:
            return None


def execute_backward_with_timing(
    model,
    vpp_rank,
    input_tensor,
    data_iterator,
    forward_step_func,
    args,
    microbatch_id,
    num_model_chunks,
    task_input_tensor_grad_dict,
    warmup_times=15,
    measure_times=10
):
    """Execute backward pass with timing measurement

    Args:
        model: model to execute (list of model chunks)
        vpp_rank: virtual pipeline rank (model_chunk_id)
        input_tensor: input tensor (requires_grad=True for gradient computation)
        data_iterator: training data iterator
        forward_step_func: forward step function to use
        args: training arguments
        microbatch_id: microbatch id
        num_model_chunks: number of model chunks (for VPP)
        task_input_tensor_grad_dict: dictionary storing task input gradient tensors
        warmup_times: number of warmup iterations (default 15)
        measure_times: number of measurement iterations (default 10)

    Returns:
        input_tensor_grad: input gradient tensor for upstream stage
        avg_backward_time: average backward time in milliseconds
    """
    from megatron.core.pipeline_parallel.schedules import backward_step
    from megatron.core.utils import get_model_type

    # Get ranks for profiling
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    # Get profiling configuration
    simulate_result_dir = getattr(args, 'simulate_result_dir', None)

    # Prepare forward step parameters
    num_microbatches = args.global_batch_size // parallel_state.get_data_parallel_world_size()
    forward_data_store = []
    collect_non_loss_data = False
    config = get_model_config(model[0])
    checkpoint_activations_microbatch = None
    model_type = get_model_type(model[0])
    cp_group_size = parallel_state.get_context_parallel_world_size()

    # Determine if this is the last pipeline stage (both PP and VPP)
    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    is_last_stage = (pp_rank == pipeline_parallel_size - 1 and vpp_rank == len(model) - 1)

    # Save original config setting and temporarily disable deallocate
    original_deallocate = config.deallocate_pipeline_outputs
    config.deallocate_pipeline_outputs = False

    # Create timeline profiler
    timeline_profiler = _create_timeline_profiler(
        pp_rank, vpp_rank, microbatch_id, simulate_result_dir, global_rank, task_type='BACKWARD'
    )

    # Warmup phase with profiling
    with timeline_profiler as p:
        for i in range(warmup_times):
            # Forward pass to build autograd graph
            output_tensor, num_tokens = forward_step(
                forward_step_func,
                data_iterator,
                model[vpp_rank],
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                cp_group_size,
                collect_non_loss_data=collect_non_loss_data,
                checkpoint_activations_microbatch=checkpoint_activations_microbatch,
                is_first_microbatch=False,
                current_microbatch=microbatch_id,
                vp_stage=vpp_rank,
                is_last_stage=is_last_stage,
            )
            forward_data_store = []
            torch.cuda.synchronize()
            torch.distributed.barrier()

            # Prepare output gradient tensor (with mock gradient generation if needed)
            output_tensor_grad = prepare_output_grad_tensor(
                pp_rank, vpp_rank, microbatch_id, task_input_tensor_grad_dict, num_model_chunks,
                forward_output_tensor=output_tensor
            )

            # Enable memory tracking after first warmup iteration
            if i == 1:
                _start_memory_tracking()

            # Backward pass
            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )
            torch.cuda.synchronize()
            torch.distributed.barrier()

            # Call profiler step
            p.step()

            # Save memory snapshot after first warmup iteration
            if i == 1 and simulate_result_dir is not None:
                _save_memory_snapshot(pp_rank, vpp_rank, microbatch_id, simulate_result_dir, global_rank, task_type='BACKWARD')

    # Measurement phase - measure backward time with CUDA events
    total_backward_time = 0
    measure_skip_times = 5  # Skip first few measurements

    for i in range(measure_times):
        backward_start = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)

        # Forward pass (not measured)
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model[vpp_rank],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=False,
            current_microbatch=microbatch_id,
            vp_stage=vpp_rank,
            is_last_stage=is_last_stage,
        )
        forward_data_store = []
        torch.cuda.synchronize()
        torch.distributed.barrier()

        # Prepare output gradient tensor (with mock gradient generation if needed)
        output_tensor_grad = prepare_output_grad_tensor(
            pp_rank, vpp_rank, microbatch_id, task_input_tensor_grad_dict, num_model_chunks,
            forward_output_tensor=output_tensor
        )

        # Backward pass (measured)
        backward_start.record()
        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )
        backward_end.record()

        torch.cuda.synchronize()
        torch.distributed.barrier()

        backward_time = backward_start.elapsed_time(backward_end)
        if i >= measure_skip_times:
            total_backward_time += backward_time

    # Restore original config setting
    config.deallocate_pipeline_outputs = original_deallocate

    # Calculate average time
    torch.cuda.synchronize()
    measured_iterations = measure_times - measure_skip_times
    avg_backward_time = total_backward_time / measured_iterations

    return input_tensor_grad, avg_backward_time
