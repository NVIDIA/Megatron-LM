# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Model Executor Utilities for VPP Training Simulation"""

import logging
import gc
import os
import time
import torch
import torch.distributed

from contextlib import contextmanager, nullcontext
from functools import partial

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core import parallel_state
from megatron.core.rerun_state_machine import RerunMode, get_rerun_state_machine
from megatron.core.utils import get_model_config, get_attr_wrapped_model, log_single_rank
from megatron.training.global_vars import get_args
from megatron.core.pipeline_parallel.schedules import forward_step
from megatron.training.utils import print_rank_0
from megatron.training.simulation.utils import MockPipelineProcessGroup

logger = logging.getLogger(__name__)


@contextmanager
def _maybe_profile_simulation_task(
    args,
    *,
    task_type,
    pp_rank,
    vpp_rank,
    microbatch_id,
    measure_index,
):
    """Optionally export a PyTorch chrome trace for one sampled simulator task."""
    if os.getenv("MCORE_SIM_TRACE", "0") != "1":
        yield
        return

    global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    target_rank = int(os.getenv("MCORE_SIM_TRACE_RANK", "0"))
    target_pp = int(os.getenv("MCORE_SIM_TRACE_PP", "0"))
    target_vpp = int(os.getenv("MCORE_SIM_TRACE_VPP", "0"))
    target_microbatch = int(os.getenv("MCORE_SIM_TRACE_MICROBATCH", "0"))
    target_measure_index = int(os.getenv("MCORE_SIM_TRACE_MEASURE_INDEX", "0"))
    target_types = {
        item.strip()
        for item in os.getenv("MCORE_SIM_TRACE_TYPES", "forward,backward").split(",")
        if item.strip()
    }

    should_profile = (
        global_rank == target_rank
        and pp_rank == target_pp
        and vpp_rank == target_vpp
        and microbatch_id == target_microbatch
        and measure_index == target_measure_index
        and task_type in target_types
    )
    if not should_profile:
        yield
        return

    trace_dir = os.getenv(
        "MCORE_SIM_TRACE_DIR",
        os.path.join(args.simulate_result_dir, "torch_profile"),
    )
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(
        trace_dir,
        (
            f"rank-{global_rank}-pp-{pp_rank}-vpp-{vpp_rank}-mb-{microbatch_id}-"
            f"{task_type}-measure-{measure_index}.json"
        ),
    )
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
    ) as profiler:
        yield
        profiler.step()

    profiler.export_chrome_trace(trace_path)
    print_rank_0(f"[SIM_TRACE] exported {task_type} trace to {trace_path}")


@contextmanager
def _simulation_forward_backward_state():
    """Enter the rerun state expected by loss/gradient validation helpers.

    Simulation times individual forward/backward tasks directly instead of using
    Megatron's full forward_backward_func wrapper. Validation hooks in loss and
    grad code still expect the rerun state machine to be inside a
    forward-backward pass, so provide that state while disabling rerun/replay
    behavior that is not meaningful for sampled simulator tasks.
    """
    rerun_state_machine = get_rerun_state_machine()
    original_mode = rerun_state_machine.get_mode()
    rerun_state_machine.set_mode(RerunMode.DISABLED)
    entered = rerun_state_machine.should_run_forward_backward(None)
    if not entered:
        rerun_state_machine.set_mode(original_mode)
        raise RuntimeError("Unable to enter rerun state for simulation task execution")

    try:
        yield
    finally:
        rerun_state_machine.should_run_forward_backward(None)
        rerun_state_machine.set_mode(original_mode)


def get_pg_collection_for_simulation(pp_rank: int, pp_size: int):
    """Create ProcessGroupCollection for simulation mode.

    In simulation mode, we use fewer physical GPUs (EP GPUs) to simulate
    a larger configuration (EP × PP GPUs). The PP process group is mocked
    to return virtual size and rank, while all other groups use real
    process groups from parallel_state.

    Args:
        pp_rank: Virtual pipeline parallel rank (0 to pp_size-1)
        pp_size: Virtual pipeline parallel world size (e.g., 4)

    Returns:
        ProcessGroupCollection with mocked PP group and real other groups

    """
    # Start with real process groups from parallel_state
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    # Replace PP group with mock
    pg_collection.pp = MockPipelineProcessGroup(size=pp_size, rank=pp_rank)

    return pg_collection


def create_model(pp_rank, model_provider):
    """Create model for specified pipeline rank in simulation mode.

    Args:
        pp_rank: Virtual pipeline parallel rank (0 to pp_size-1)
        model_provider: Function to create model

    Returns:
        model: Created model (list of model chunks for VPP)
    """
    # Reset MTP embedding for second MTP build
    from megatron.training import get_model
    from megatron.core.process_groups_config import ProcessGroupCollection
    #from megatron.core.transformer.multi_token_prediction import MTPEmbeddingHelper
    #MTPEmbeddingHelper.set_embedding(None)

    # Set pipeline rank and world size for correct is_last_stage calculation
    args = get_args()
    parallel_state.set_pipeline_model_parallel_rank(pp_rank)
    parallel_state.set_pipeline_model_parallel_world_size(args.pipeline_model_parallel_size)

    # Create simulation-specific ProcessGroupCollection with mocked PP group
    # This allows build_model() to see virtual PP size/rank without code changes
    pg_collection = get_pg_collection_for_simulation(
        pp_rank=pp_rank,
        pp_size=args.pipeline_model_parallel_size
    )
    # Pass the simulation pg_collection to get_model
    model = get_model(model_provider, pg_collection=pg_collection)

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


def clear_model_grad_state(model):
    """Clear model gradient state at simulator model-chunk boundaries."""
    model_chunks = model if isinstance(model, list) else [model]

    for model_chunk in model_chunks:
        if hasattr(model_chunk, "zero_grad_buffer"):
            model_chunk.zero_grad_buffer()
        if hasattr(model_chunk, "zero_grad"):
            try:
                model_chunk.zero_grad(set_to_none=False)
            except TypeError:
                model_chunk.zero_grad()

        for param in model_chunk.parameters():
            if param.grad is not None:
                param.grad.zero_()
            if hasattr(param, "grad_added_to_main_grad"):
                param.grad_added_to_main_grad = False


def _release_timing_iteration(model, input_tensor):
    """Release per-repeat autograd state before the next simulator timing repeat."""
    _clear_tensor_grad(input_tensor)
    gc.collect()


def _clear_tensor_grad(tensor_or_tensors):
    """Drop retained input gradients from tensors reused across timing repeats."""
    if tensor_or_tensors is None:
        return
    if isinstance(tensor_or_tensors, torch.Tensor):
        tensor_or_tensors.grad = None
        return
    if isinstance(tensor_or_tensors, dict):
        for tensor in tensor_or_tensors.values():
            _clear_tensor_grad(tensor)
        return
    if isinstance(tensor_or_tensors, (list, tuple)):
        for tensor in tensor_or_tensors:
            _clear_tensor_grad(tensor)


def _detach_tensor_tree(tensor_or_tensors):
    """Detach tensors while preserving container structure."""
    if tensor_or_tensors is None:
        return None
    if isinstance(tensor_or_tensors, torch.Tensor):
        return tensor_or_tensors.detach()
    if isinstance(tensor_or_tensors, dict):
        return {key: _detach_tensor_tree(value) for key, value in tensor_or_tensors.items()}
    if isinstance(tensor_or_tensors, tuple):
        return tuple(_detach_tensor_tree(tensor) for tensor in tensor_or_tensors)
    if isinstance(tensor_or_tensors, list):
        return [_detach_tensor_tree(tensor) for tensor in tensor_or_tensors]
    return tensor_or_tensors


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
    measure_times=10,
    measure_skip_times=5,
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
        measure_skip_times: number of measurement iterations to skip in average (default 5)

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

    # Warmup phase with profiling
    for i in range(warmup_times):
        _clear_tensor_grad(input_tensor)
        with _simulation_forward_backward_state():
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
        del output_tensor
        _release_timing_iteration(model, input_tensor)
        torch.distributed.barrier()

    # Measurement phase - measure only the sampled forward task. Cleanup stays
    # outside the timed region and a barrier before timing prevents prior
    # cleanup skew from being charged to the next collective.
    total_forward_time = 0
    measured_output_tensor = None
    for i in range(measure_times):
        _clear_tensor_grad(input_tensor)
        torch.distributed.barrier()
        with _simulation_forward_backward_state():
            torch.cuda.synchronize()
            forward_start = time.perf_counter()
            with _maybe_profile_simulation_task(
                args,
                task_type="forward",
                pp_rank=pp_rank,
                vpp_rank=vpp_rank,
                microbatch_id=microbatch_id,
                measure_index=i,
            ):
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
            forward_time = (time.perf_counter() - forward_start) * 1000.0
        forward_data_store = []

        torch.distributed.barrier()
        if i == measure_times - 1:
            measured_output_tensor = _detach_tensor_tree(output_tensor)
        del output_tensor
        _release_timing_iteration(model, input_tensor)
        torch.distributed.barrier()
        if i >= measure_skip_times:
            total_forward_time += forward_time

    # Calculate average time
    torch.cuda.synchronize()
    measured_iterations = measure_times - measure_skip_times
    avg_forward_time = total_forward_time / measured_iterations

    return measured_output_tensor, avg_forward_time


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
    measure_times=10,
    measure_skip_times=5,
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
        measure_skip_times: number of measurement iterations to skip in average (default 5)

    Returns:
        input_tensor_grad: input gradient tensor for upstream stage
        avg_backward_time: average backward time in milliseconds
    """
    from megatron.core.pipeline_parallel.schedules import backward_step

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
    cp_group_size = parallel_state.get_context_parallel_world_size()

    # Determine if this is the last pipeline stage (both PP and VPP)
    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    is_last_stage = (pp_rank == pipeline_parallel_size - 1 and vpp_rank == len(model) - 1)

    # Save original config setting and temporarily disable deallocate
    original_deallocate = config.deallocate_pipeline_outputs
    config.deallocate_pipeline_outputs = False

    # Warmup phase with profiling
    for i in range(warmup_times):
        _clear_tensor_grad(input_tensor)
        with _simulation_forward_backward_state():
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

            # Backward pass
            input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad, config)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        del output_tensor, output_tensor_grad, input_tensor_grad
        _release_timing_iteration(model, input_tensor)
        torch.distributed.barrier()

    # Measurement phase - measure only the sampled backward task. The setup
    # forward and cleanup are excluded from the timed region.
    total_backward_time = 0
    measured_input_tensor_grad = None
    for i in range(measure_times):
        _clear_tensor_grad(input_tensor)
        with _simulation_forward_backward_state():
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
            torch.distributed.barrier()
            torch.cuda.synchronize()
            backward_start = time.perf_counter()
            with _maybe_profile_simulation_task(
                args,
                task_type="backward",
                pp_rank=pp_rank,
                vpp_rank=vpp_rank,
                microbatch_id=microbatch_id,
                measure_index=i,
            ):
                input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad, config)
            torch.cuda.synchronize()
            backward_time = (time.perf_counter() - backward_start) * 1000.0

        torch.distributed.barrier()
        if i == measure_times - 1:
            measured_input_tensor_grad = _detach_tensor_tree(input_tensor_grad)
        del output_tensor, output_tensor_grad, input_tensor_grad
        _release_timing_iteration(model, input_tensor)
        torch.distributed.barrier()
        if i >= measure_skip_times:
            total_backward_time += backward_time

    # Restore original config setting
    config.deallocate_pipeline_outputs = original_deallocate

    # Calculate average time
    torch.cuda.synchronize()
    measured_iterations = measure_times - measure_skip_times
    avg_backward_time = total_backward_time / measured_iterations

    return measured_input_tensor_grad, avg_backward_time
