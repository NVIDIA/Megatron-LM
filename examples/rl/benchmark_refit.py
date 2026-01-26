#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import sys
import time
from functools import partial

import torch

from megatron.core.enums import ModelType
from megatron.core.resharding.refit import swap_model_weights
from megatron.training import get_args, print_rank_0
from megatron.training.initialize import initialize_megatron
from megatron.training.arguments import core_transformer_config_from_args
from megatron.rl.parallel_utils import build_inference_pg_collection
from gpt_builders import gpt_builder


def add_benchmark_args(parser):
    """Add benchmark-specific arguments."""
    group = parser.add_argument_group(title='refit benchmark')

    group.add_argument(
        '--refit-mode',
        type=str,
        required=True,
        choices=['collocated', 'non-collocated'],
        help='Collocated: both models share GPUs. Non-collocated: separate GPU sets.'
    )
    group.add_argument(
        '--num-benchmark-warmup',
        type=int,
        default=2,
        help='Number of warmup iterations before benchmarking.'
    )
    group.add_argument(
        '--num-benchmark-iterations',
        type=int,
        default=10,
        help='Number of benchmark iterations to measure.'
    )
    group.add_argument(
        '--no-load-checkpoint',
        action='store_true',
        help='Skip checkpoint loading (benchmark uses random weights).'
    )

    return parser


def model_provider(pre_process=True, post_process=True, parallel_output=False, pg_collection=None, config=None):
    """Build the model."""
    args = get_args()
    if config is None:
        config = core_transformer_config_from_args(args)

    print_rank_0('Building models...')
    model = gpt_builder(
        args=args,
        pre_process=pre_process,
        post_process=post_process,
        config=config,
        pg_collection=pg_collection,
    )

    return [model]


# Note: swap_model_weights now natively supports None models for idle ranks
# in non-collocated mode. Idle ranks participate in collective operations
# (all_gather, broadcast, barriers) but don't perform actual send/recv operations.


def get_model_size_mb(model):
    """Calculate model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size / 1024 / 1024


def benchmark_refit_collocated():
    """Benchmark refit in collocated mode (both models on same GPUs)."""
    args = get_args()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # Calculate required sizes
    src_tp = args.tensor_model_parallel_size
    src_pp = args.pipeline_model_parallel_size
    src_ep = args.expert_model_parallel_size
    src_world_size = src_tp * src_pp * src_ep

    # Get inference parallelism (default to training parallelism if not specified)
    dst_tp = args.rl_inference_tensor_model_parallel_size if args.rl_inference_tensor_model_parallel_size is not None else src_tp
    dst_pp = args.rl_inference_pipeline_model_parallel_size if args.rl_inference_pipeline_model_parallel_size is not None else src_pp
    dst_ep = args.rl_inference_expert_model_parallel_size if args.rl_inference_expert_model_parallel_size is not None else src_ep
    dst_world_size = dst_tp * dst_pp * dst_ep

    src_dp = world_size // src_world_size
    dst_dp = world_size // dst_world_size

    if src_dp < 1 or dst_dp < 1:
        raise ValueError(
            f"Invalid parallelism for world_size={world_size}: "
            f"src needs {src_world_size}, dst needs {dst_world_size}"
        )

    print_rank_0(f"\n{'='*80}")
    print_rank_0("COLLOCATED MODE REFIT BENCHMARK")
    print_rank_0(f"{'='*80}")
    print_rank_0(f"World size: {world_size}")
    print_rank_0(f"Source (training): TP={src_tp}, PP={src_pp}, EP={src_ep}, DP={src_dp}")
    print_rank_0(f"Destination (inference): TP={dst_tp}, PP={dst_pp}, EP={dst_ep}, DP={dst_dp}")
    print_rank_0(f"Model: {args.num_layers} layers, {args.hidden_size} hidden, {args.num_attention_heads} heads")
    if args.num_experts:
        print_rank_0(f"MoE: {args.num_experts} experts")
    print_rank_0(f"Refit backend: {args.refit_method}")
    print_rank_0(f"{'='*80}\n")

    # Build training model (uses default parallel state from initialize_megatron)
    src_model = model_provider(parallel_output=False)
    # Move to GPU
    src_model[0] = src_model[0].cuda()

    # Build inference model with custom parallelism (like in RL training loop)
    # In collocated mode, ALL ranks participate in creating the PG
    dst_pg_collection = build_inference_pg_collection(
        world_size,  # FULL world size, not subset
        tp_size=dst_tp,
        pp_size=dst_pp,
        ep_size=dst_ep,
        expt_tp_size=args.rl_inference_expert_tensor_model_parallel_size,
        use_tp_pp_dp_mapping=args.use_tp_pp_dp_mapping,
        # NO rank_offset in collocated mode
    )

    # Build inference config
    dst_config = core_transformer_config_from_args(args)
    if args.num_experts:
        dst_config.expert_model_parallel_size = dst_ep
    dst_config.tensor_model_parallel_size = dst_tp
    if args.rl_inference_expert_tensor_model_parallel_size is not None:
        dst_config.expert_tensor_parallel_size = args.rl_inference_expert_tensor_model_parallel_size

    # Build inference model (matches RL training loop)
    dst_model = model_provider(
        pre_process=True,
        post_process=True,
        pg_collection=dst_pg_collection,
        config=dst_config,
    )
    # Move to GPU
    dst_model[0] = dst_model[0].cuda()

    # Print model sizes
    if rank == 0:
        src_size = get_model_size_mb(src_model[0])
        dst_size = get_model_size_mb(dst_model[0])
        print(f"Source model size on rank 0: {src_size:.2f} MB")
        print(f"Destination model size on rank 0: {dst_size:.2f} MB")

    torch.distributed.barrier()

    # Pre-initialize NVSHMEM if using nvshmem backend
    # NVSHMEM init requires ALL ranks to participate in collective operations
    if args.refit_method == 'nvshmem':
        print_rank_0("Pre-initializing NVSHMEM (collective operation requires all ranks)...")
        from megatron.core.resharding.copy_services.nvshmem_copy_service import NVSHMEMCopyService
        _init_service = NVSHMEMCopyService()
        _init_service._ensure_initialized()
        torch.distributed.barrier()
        print_rank_0("NVSHMEM initialized on all ranks.")

    # Warmup iterations (to build and cache the refit plan)
    print_rank_0(f"\nWarmup: {args.num_benchmark_warmup} iterations...")
    print_rank_0("  (First iteration builds refit plan, subsequent iterations reuse cached plan)")
    for i in range(args.num_benchmark_warmup):
        torch.cuda.synchronize()
        torch.distributed.barrier()
        swap_model_weights(src_model, dst_model, refit_method=args.refit_method)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        if rank == 0:
            print(f"  Warmup iteration {i+1}/{args.num_benchmark_warmup} complete")

    print_rank_0("  Plan building complete, now benchmarking execution only...")
    torch.cuda.synchronize()
    torch.distributed.barrier()

    # Benchmark iterations (only measures execution, plan is cached)
    print_rank_0(f"\nBenchmarking: {args.num_benchmark_iterations} iterations...")
    timings = []

    for i in range(args.num_benchmark_iterations):
        # Ensure no pending GPU work and all ranks are ready
        torch.cuda.synchronize()
        torch.distributed.barrier()

        # Start timing
        start_time = time.perf_counter()

        # Execute refit (uses cached plan)
        swap_model_weights(src_model, dst_model, refit_method=args.refit_method)

        # Wait for all GPU operations to complete
        torch.cuda.synchronize()

        # Stop timing (before barrier to exclude synchronization overhead)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        timings.append(elapsed)

        # Keep ranks synchronized (not included in timing)
        torch.distributed.barrier()

        if rank == 0:
            print(f"  Iteration {i+1}/{args.num_benchmark_iterations}: {elapsed*1000:.2f} ms")

    # Calculate statistics
    if rank == 0:
        mean_time = sum(timings) / len(timings)
        min_time = min(timings)
        max_time = max(timings)

        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(f"Mean refit time: {mean_time*1000:.2f} ms")
        print(f"Min refit time:  {min_time*1000:.2f} ms")
        print(f"Max refit time:  {max_time*1000:.2f} ms")
        print(f"{'='*80}\n")


def benchmark_refit_non_collocated():
    """Benchmark refit in non-collocated mode (separate GPU sets)."""
    args = get_args()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # Calculate sizes for each model
    src_tp = args.tensor_model_parallel_size
    src_pp = args.pipeline_model_parallel_size
    src_ep = args.expert_model_parallel_size
    src_world_size = src_tp * src_pp * src_ep

    # Get inference parallelism (default to training parallelism if not specified)
    dst_tp = args.rl_inference_tensor_model_parallel_size if args.rl_inference_tensor_model_parallel_size is not None else src_tp
    dst_pp = args.rl_inference_pipeline_model_parallel_size if args.rl_inference_pipeline_model_parallel_size is not None else src_pp
    dst_ep = args.rl_inference_expert_model_parallel_size if args.rl_inference_expert_model_parallel_size is not None else src_ep
    dst_world_size = dst_tp * dst_pp * dst_ep

    required_size = src_world_size + dst_world_size
    if world_size < required_size:
        raise ValueError(
            f"Non-collocated mode requires at least {required_size} GPUs "
            f"({src_world_size} for src + {dst_world_size} for dst), "
            f"but only {world_size} available"
        )
    elif world_size > required_size:
        print_rank_0(f"Note: Using {required_size} of {world_size} available GPUs "
                     f"(ranks {required_size}-{world_size-1} will be idle)")

    # Determine which model this rank belongs to
    is_src_rank = rank < src_world_size
    is_dst_rank = src_world_size <= rank < required_size
    is_idle_rank = rank >= required_size

    print_rank_0(f"\n{'='*80}")
    print_rank_0("NON-COLLOCATED MODE REFIT BENCHMARK")
    print_rank_0(f"{'='*80}")
    print_rank_0(f"World size: {world_size} (using {required_size} GPUs)")
    print_rank_0(f"Source ranks: 0-{src_world_size-1} (TP={src_tp}, PP={src_pp}, EP={src_ep})")
    print_rank_0(f"Destination ranks: {src_world_size}-{required_size-1} (TP={dst_tp}, PP={dst_pp}, EP={dst_ep})")
    if world_size > required_size:
        print_rank_0(f"Idle ranks: {required_size}-{world_size-1} (not participating)")
    print_rank_0(f"Model: {args.num_layers} layers, {args.hidden_size} hidden, {args.num_attention_heads} heads")
    if args.num_experts:
        print_rank_0(f"MoE: {args.num_experts} experts")
    print_rank_0(f"Refit backend: {args.refit_method}")
    print_rank_0(f"{'='*80}\n")

    # ALL ranks must participate in creating ALL process groups (collective operation)
    # Create destination process groups for all ranks
    print_rank_0("Creating process groups for destination model...")
    dst_pg_collection = build_inference_pg_collection(
        world_size=dst_world_size,
        tp_size=dst_tp,
        pp_size=dst_pp,
        ep_size=dst_ep,
        expt_tp_size=args.rl_inference_expert_tensor_model_parallel_size,
        use_tp_pp_dp_mapping=args.use_tp_pp_dp_mapping,
        rank_offset=src_world_size,
    )
    torch.distributed.barrier()
    print_rank_0("Process groups created.")

    # Idle ranks don't build models - they'll pass None to swap_model_weights_safe
    if is_idle_rank:
        if rank == required_size:
            print(f"Ranks {required_size}-{world_size-1} are idle (no models, will sync via barriers)...")

        torch.distributed.barrier()

        # Set models to None - the wrapper will handle this gracefully
        src_model = None
        dst_model = None

        torch.distributed.barrier()

        # Continue to pre-init and warmup sections (don't return early)

    # Build model for this rank's group
    if is_src_rank:
        # Source rank: build training model with default PG
        model_type = "source"
        my_model = model_provider(parallel_output=False)
        my_model[0] = my_model[0].cuda()
        local_rank = rank
    elif is_dst_rank:
        # Destination rank: build inference model with custom PG
        model_type = "destination"
        local_rank = rank - src_world_size

        # Build config for destination
        dst_config = core_transformer_config_from_args(args)
        if args.num_experts:
            dst_config.expert_model_parallel_size = dst_ep
        dst_config.tensor_model_parallel_size = dst_tp
        if args.rl_inference_expert_tensor_model_parallel_size is not None:
            dst_config.expert_tensor_parallel_size = args.rl_inference_expert_tensor_model_parallel_size

        my_model = model_provider(
            pre_process=True,
            post_process=True,
            pg_collection=dst_pg_collection,
            config=dst_config,
        )
        my_model[0] = my_model[0].cuda()

    if rank == 0 or rank == src_world_size:
        model_size = get_model_size_mb(my_model[0])
        print(f"Rank {rank} ({model_type}, local {local_rank}): Model size = {model_size:.2f} MB")

    torch.distributed.barrier()

    # For swap_model_weights, we need both src and dst models on every active rank
    # The refit code accesses properties from both models, so we create dummy models
    # for whichever model this rank doesn't have.
    # Note: Dummy models use default parallel state (not custom pg_collection)
    if not is_idle_rank:
        if is_src_rank:
            src_model = my_model
            # Need a dummy dst model - use default PG (source rank is not part of dst PGs)
            dst_model = model_provider(parallel_output=False)
            dst_model[0] = dst_model[0].cuda()
        elif is_dst_rank:
            # Need a dummy src model - use default PG (dest rank is not part of src PGs)
            src_model = model_provider(parallel_output=False)
            src_model[0] = src_model[0].cuda()
            dst_model = my_model

    torch.distributed.barrier()

    # Pre-initialize NVSHMEM if using nvshmem backend
    # NVSHMEM init requires ALL ranks to participate in collective operations
    if args.refit_method == 'nvshmem':
        print_rank_0("Pre-initializing NVSHMEM (collective operation requires all ranks)...")
        from megatron.core.resharding.copy_services.nvshmem_copy_service import NVSHMEMCopyService
        _init_service = NVSHMEMCopyService()
        _init_service._ensure_initialized()
        torch.distributed.barrier()
        print_rank_0("NVSHMEM initialized on all ranks.")

    # Warmup (to build and cache the refit plan)
    # Note: swap_model_weights now handles None models natively for idle ranks
    print_rank_0(f"\nWarmup: {args.num_benchmark_warmup} iterations...")
    print_rank_0("  (First iteration builds refit plan, subsequent iterations reuse cached plan)")
    for i in range(args.num_benchmark_warmup):
        torch.cuda.synchronize()
        torch.distributed.barrier()
        swap_model_weights(src_model, dst_model, refit_method=args.refit_method)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        if rank == 0:
            print(f"  Warmup iteration {i+1}/{args.num_benchmark_warmup} complete")

    print_rank_0("  Plan building complete, now benchmarking execution only...")
    torch.cuda.synchronize()
    torch.distributed.barrier()

    # Benchmark (only measures execution, plan is cached)
    print_rank_0(f"\nBenchmarking: {args.num_benchmark_iterations} iterations...")
    timings = []

    for i in range(args.num_benchmark_iterations):
        # Ensure no pending GPU work and all ranks are ready
        torch.cuda.synchronize()
        torch.distributed.barrier()

        # Start timing
        start_time = time.perf_counter()

        # Execute refit (uses cached plan)
        swap_model_weights(src_model, dst_model, refit_method=args.refit_method)

        # Wait for all GPU operations to complete
        torch.cuda.synchronize()

        # Stop timing (before barrier to exclude synchronization overhead)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        timings.append(elapsed)

        # Keep ranks synchronized (not included in timing)
        torch.distributed.barrier()

        if rank == 0:
            print(f"  Iteration {i+1}/{args.num_benchmark_iterations}: {elapsed*1000:.2f} ms")

    # Calculate statistics
    if rank == 0:
        mean_time = sum(timings) / len(timings)
        min_time = min(timings)
        max_time = max(timings)

        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(f"Mean refit time: {mean_time*1000:.2f} ms")
        print(f"Min refit time:  {min_time*1000:.2f} ms")
        print(f"Max refit time:  {max_time*1000:.2f} ms")
        print(f"{'='*80}\n")


def main():
    """Main benchmark function."""
    # Initialize Megatron (handles all arg parsing, distributed setup, etc.)
    # Skip tokenizer since we don't need it for benchmarking
    initialize_megatron(
        extra_args_provider=add_benchmark_args,
        args_defaults={
            'tokenizer_type': 'NullTokenizer',
            'vocab_size': 1024,  # Dummy vocab size for NullTokenizer
            'no_load_optim': True,
            'no_load_rng': True,
            'no_save_optim': True,
            'no_save_rng': True,
        },
        ignore_unknown_args=False,
    )

    args = get_args()

    # Run appropriate benchmark
    if args.refit_mode == 'collocated':
        benchmark_refit_collocated()
    else:  # non-collocated
        benchmark_refit_non_collocated()


if __name__ == "__main__":
    main()
