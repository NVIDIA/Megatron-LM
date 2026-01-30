#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import sys
import time
from functools import partial

import torch

from megatron.core.enums import ModelType
from megatron.core.resharding.refit import swap_model_weights
from megatron.training import get_args, get_model as get_training_model, print_rank_0
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
    group.add_argument(
        '--nvshmem-scheduling-algorithm',
        type=str,
        default='dsatur',
        choices=['dsatur', 'greedy'],
        help='NVSHMEM scheduling algorithm: dsatur (near-optimal, default) or greedy (baseline)'
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

    return model




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

    # Build training model WITHOUT DDP wrapping (no gradient buffers needed)
    src_model = get_training_model(
        lambda pre_process, post_process, **kwargs: model_provider(pre_process, post_process, parallel_output=False, **kwargs),
        wrap_with_ddp=False
    )
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

    # Build inference model WITHOUT DDP wrapping (no gradient buffers needed)
    dst_model = get_training_model(
        lambda pre_process, post_process, **kwargs: model_provider(
            pre_process, post_process, pg_collection=dst_pg_collection, config=dst_config
        ),
        wrap_with_ddp=False
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

    # Create refit service once and reuse it across all iterations
    # This avoids repeated NVSHMEM buffer allocations
    refit_service = None
    if args.refit_method == 'nvshmem':
        print_rank_0(f"Creating NVSHMEM service with '{args.nvshmem_scheduling_algorithm}' scheduler...")
        from megatron.core.resharding.copy_services.nvshmem_copy_service import NVSHMEMCopyService
        refit_service = NVSHMEMCopyService(scheduling_algorithm=args.nvshmem_scheduling_algorithm)
        # Service will be lazily initialized on first use

        # CRITICAL: Ensure all CUDA work from NVSHMEM init is complete
        # NVSHMEM uses custom streams that need to be fully synchronized
        # before torch.distributed (NCCL) operations can safely proceed
        torch.cuda.synchronize()
        torch.distributed.barrier()
        print_rank_0(f"NVSHMEM service created (algorithm: {args.nvshmem_scheduling_algorithm}).")
    elif args.refit_method == 'nccl':
        print_rank_0("Creating NCCL service (will be reused across all iterations)...")
        from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService
        refit_service = NCCLCopyService()
    elif args.refit_method == 'gloo':
        print_rank_0("Creating Gloo service (will be reused across all iterations)...")
        from megatron.core.resharding.copy_services.gloo_copy_service import GlooCopyService
        refit_service = GlooCopyService()
    else:
        # Use string method if unknown (will create service each time - legacy behavior)
        refit_service = args.refit_method

    # Warmup iterations (to build and cache the refit plan)
    print_rank_0(f"\nWarmup: {args.num_benchmark_warmup} iterations...")
    print_rank_0("  (First iteration builds refit plan, subsequent iterations reuse cached plan)")
    for i in range(args.num_benchmark_warmup):
        torch.cuda.synchronize()
        torch.distributed.barrier()
        swap_model_weights(src_model, dst_model, refit_method=refit_service)
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

        # Execute refit (uses cached plan and reuses service instance)
        swap_model_weights(src_model, dst_model, refit_method=refit_service)

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
    print_rank_0(f"  -> {args.num_experts} experts / {src_ep} EP groups = {args.num_experts // src_ep} experts per EP group")
    print_rank_0(f"Destination ranks: {src_world_size}-{required_size-1} (TP={dst_tp}, PP={dst_pp}, EP={dst_ep})")
    print_rank_0(f"  -> {args.num_experts} experts / {dst_ep} EP groups = {args.num_experts // dst_ep} experts per EP group")
    if world_size > required_size:
        print_rank_0(f"Idle ranks: {required_size}-{world_size-1} (not participating)")
    print_rank_0(f"Model: {args.num_layers} layers, {args.hidden_size} hidden, {args.num_attention_heads} heads")
    if args.num_experts:
        print_rank_0(f"MoE: {args.num_experts} experts, topk={args.moe_router_topk}")
    print_rank_0(f"Data type: {args.params_dtype}")
    print_rank_0(f"Refit backend: {args.refit_method}")
    print_rank_0(f"{'='*80}\n")

    # Check GPU memory BEFORE creating process groups
    if rank < 5:
        torch.cuda.empty_cache()
        mem_start = torch.cuda.memory_allocated() / 1024**3
        print(f"[Rank {rank}] GPU memory at start of non-collocated: {mem_start:.2f} GB")

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

    if rank < 5:
        mem_after_pg = torch.cuda.memory_allocated() / 1024**3
        print(f"[Rank {rank}] GPU memory after PG creation: {mem_after_pg:.2f} GB")

    # Idle ranks don't build models - they'll pass None to swap_model_weights
    if is_idle_rank:
        if rank == required_size:
            print(f"Ranks {required_size}-{world_size-1} are idle (no models, will sync via barriers)...")

        # Set models to None - idle ranks participate in collectives but have no transfers
        src_model = None
        dst_model = None

        # Continue to pre-init and warmup sections (will sync at global barriers with other ranks)

    # Build model for this rank's group
    if is_src_rank:
        # Source rank: build training model with default PG
        from megatron.core import parallel_state
        model_type = "source"
        print(f"[Rank {rank}] Building SOURCE model...")
        print(f"[Rank {rank}] Parallel state: TP={parallel_state.get_tensor_model_parallel_world_size()}, "
              f"EP={parallel_state.get_expert_model_parallel_world_size()}, "
              f"PP={parallel_state.get_pipeline_model_parallel_world_size()}")
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated() / 1024**3
        print(f"[Rank {rank}] GPU memory before model build: {mem_before:.2f} GB")

        # Build SOURCE model WITHOUT DDP wrapping (no gradient buffers needed)
        my_model = get_training_model(
            lambda pre_process, post_process, **kwargs: model_provider(pre_process, post_process, parallel_output=False, **kwargs),
            wrap_with_ddp=False
        )

        mem_after_cpu = torch.cuda.memory_allocated() / 1024**3
        print(f"[Rank {rank}] GPU memory after model build (CPU): {mem_after_cpu:.2f} GB")

        # Count parameters
        total_params = sum(p.numel() for p in my_model[0].parameters())
        param_bytes = sum(p.numel() * p.element_size() for p in my_model[0].parameters())
        print(f"[Rank {rank}] Total parameters: {total_params:,} ({param_bytes/1024**3:.2f} GB)")

        my_model[0] = my_model[0].cuda()

        mem_after_cuda = torch.cuda.memory_allocated() / 1024**3
        print(f"[Rank {rank}] GPU memory after .cuda(): {mem_after_cuda:.2f} GB")
        print(f"[Rank {rank}] Memory increase: {mem_after_cuda - mem_before:.2f} GB")

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

        # Build DESTINATION model WITHOUT DDP wrapping (no gradient buffers needed)
        my_model = get_training_model(
            lambda pre_process, post_process, **kwargs: model_provider(
                pre_process, post_process, pg_collection=dst_pg_collection, config=dst_config
            ),
            wrap_with_ddp=False
        )
        my_model[0] = my_model[0].cuda()

    if rank == 0 or rank == src_world_size:
        model_size = get_model_size_mb(my_model[0])
        print(f"Rank {rank} ({model_type}, local {local_rank}): Model size = {model_size:.2f} MB")

    torch.distributed.barrier()

    # Set up models for swap_model_weights in non-collocated mode
    #
    # Each rank passes only the model it owns:
    # - Source ranks: (src_model, None) - send data only
    # - Destination ranks: (None, dst_model) - receive data only
    #
    # The metadata now includes local rank positions within parallel groups
    # (tensor_parallel_local_rank, expert_parallel_local_rank), which provides
    # sufficient information for the planner to correctly map between source and
    # destination ranks even when they use different process group configurations.
    #
    if is_src_rank:
        src_model = my_model
        dst_model = None
    elif is_dst_rank:
        src_model = None
        dst_model = my_model

    torch.distributed.barrier()

    # Create refit service once and reuse it across all iterations
    # This avoids repeated NVSHMEM buffer allocations
    refit_service = None
    if args.refit_method == 'nvshmem':
        print_rank_0(f"Creating NVSHMEM service with '{args.nvshmem_scheduling_algorithm}' scheduler...")
        from megatron.core.resharding.copy_services.nvshmem_copy_service import NVSHMEMCopyService
        refit_service = NVSHMEMCopyService(scheduling_algorithm=args.nvshmem_scheduling_algorithm)
        # Service will be lazily initialized on first use

        # CRITICAL: Ensure all CUDA work from NVSHMEM init is complete
        # NVSHMEM uses custom streams that need to be fully synchronized
        # before torch.distributed (NCCL) operations can safely proceed
        torch.cuda.synchronize()
        torch.distributed.barrier()
        print_rank_0(f"NVSHMEM service created (algorithm: {args.nvshmem_scheduling_algorithm}).")
    elif args.refit_method == 'nccl':
        print_rank_0("Creating NCCL service (will be reused across all iterations)...")
        from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService
        refit_service = NCCLCopyService()
    elif args.refit_method == 'gloo':
        print_rank_0("Creating Gloo service (will be reused across all iterations)...")
        from megatron.core.resharding.copy_services.gloo_copy_service import GlooCopyService
        refit_service = GlooCopyService()
    else:
        # Use string method if unknown (will create service each time - legacy behavior)
        refit_service = args.refit_method

    # Warmup (to build and cache the refit plan)
    # Note: Each rank passes only the model it owns (None for models it doesn't have)
    print_rank_0(f"\nWarmup: {args.num_benchmark_warmup} iterations...")
    print_rank_0("  (First iteration builds refit plan, subsequent iterations reuse cached plan)")
    for i in range(args.num_benchmark_warmup):
        torch.cuda.synchronize()
        torch.distributed.barrier()
        swap_model_weights(src_model, dst_model, refit_method=refit_service)
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

        # Execute refit (uses cached plan and reuses service instance)
        swap_model_weights(src_model, dst_model, refit_method=refit_service)

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
    rank = torch.distributed.get_rank()

    # CHECK MEMORY RIGHT AFTER INITIALIZATION
    if rank < 5:
        torch.cuda.empty_cache()
        mem_after_init = torch.cuda.memory_allocated() / 1024**3
        print(f"[Rank {rank}] GPU memory after initialize_megatron: {mem_after_init:.2f} GB")
        if mem_after_init > 1.0:
            print(f"[Rank {rank}] WARNING: {mem_after_init:.2f} GB allocated after init! Something is wrong!")

    # Run appropriate benchmark
    if args.refit_mode == 'collocated':
        benchmark_refit_collocated()
    else:  # non-collocated
        benchmark_refit_non_collocated()


if __name__ == "__main__":
    main()
