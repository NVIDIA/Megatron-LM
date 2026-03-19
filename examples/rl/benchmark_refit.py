#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""
Benchmark script for model refit performance.

Measures the time to transfer model weights between different parallelism configurations.
Supports both collocated (models share GPUs) and non-collocated (separate GPU sets) modes.
"""
import time

import torch

from megatron.core.resharding.refit import swap_model_weights
from megatron.training import get_args, get_model as get_training_model, print_rank_0
from megatron.training.initialize import initialize_megatron
from megatron.training.arguments import core_transformer_config_from_args
from megatron.rl.parallel_utils import build_inference_pg_collection
from gpt_builders import gpt_builder
from megatron.core.resharding.copy_services.nvshmem_copy_service import NVSHMEMCopyService
from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService
from megatron.core.resharding.copy_services.gloo_copy_service import GlooCopyService


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
        help='Number of warmup iterations (first builds refit plan).'
    )
    group.add_argument(
        '--num-benchmark-iterations',
        type=int,
        default=10,
        help='Number of timed benchmark iterations.'
    )

    return parser


def model_provider(pre_process=True, post_process=True, parallel_output=False,
                   pg_collection=None, config=None):
    """Build the model."""
    args = get_args()
    if config is None:
        config = core_transformer_config_from_args(args)

    return gpt_builder(
        args=args,
        pre_process=pre_process,
        post_process=post_process,
        config=config,
        pg_collection=pg_collection,
    )


def create_refit_service(method):
    """Create and return a refit service instance."""
    if method == 'nvshmem':
        return NVSHMEMCopyService()
    elif method == 'nccl':
        return NCCLCopyService()
    elif method == 'gloo':
        return GlooCopyService()
    else:
        return method


def print_config_summary(args, src_config, dst_config, world_size, mode):
    """Print benchmark configuration."""
    print_rank_0(f"\n{'='*80}")
    print_rank_0(f"REFIT BENCHMARK - {mode.upper()} MODE")
    print_rank_0(f"{'='*80}")
    print_rank_0(f"World size: {world_size}")
    print_rank_0(f"Source:      TP={src_config['tp']}, PP={src_config['pp']}, EP={src_config['ep']}, DP={src_config['dp']}")
    print_rank_0(f"Destination: TP={dst_config['tp']}, PP={dst_config['pp']}, EP={dst_config['ep']}, DP={dst_config['dp']}")
    print_rank_0(f"Model: {args.num_layers}L, {args.hidden_size}H, {args.num_attention_heads} heads, vocab={args.vocab_size}")
    if args.num_experts:
        print_rank_0(f"MoE: {args.num_experts} experts, top-{args.moe_router_topk}")
    print_rank_0(f"Backend: {args.refit_method}")
    print_rank_0(f"{'='*80}\n")


def run_benchmark(src_model, dst_model, refit_service, num_warmup, num_iterations):
    """Run warmup and benchmark iterations, return timings."""
    rank = torch.distributed.get_rank()

    # Warmup (builds refit plan on first iteration)
    print_rank_0(f"Warmup: {num_warmup} iterations...")
    for i in range(num_warmup):
        torch.cuda.synchronize()
        torch.distributed.barrier()
        swap_model_weights(src_model, dst_model, refit_method=refit_service)
        torch.cuda.synchronize()
        torch.distributed.barrier()

    print_rank_0("Warmup complete. Starting benchmark...\n")

    # Benchmark iterations
    print_rank_0(f"Benchmark: {num_iterations} iterations...")
    timings = []

    for i in range(num_iterations):
        torch.cuda.synchronize()
        torch.distributed.barrier()

        start_time = time.perf_counter()
        swap_model_weights(src_model, dst_model, refit_method=refit_service)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        elapsed = end_time - start_time
        timings.append(elapsed)
        torch.distributed.barrier()

    return timings


def print_results(timings):
    """Print benchmark results."""
    if torch.distributed.get_rank() == 0:
        mean_time = sum(timings) / len(timings)
        min_time = min(timings)
        max_time = max(timings)

        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(f"Mean: {mean_time*1000:.2f} ms")
        print(f"Min:  {min_time*1000:.2f} ms")
        print(f"Max:  {max_time*1000:.2f} ms")
        print(f"{'='*80}\n")


def benchmark_collocated():
    """Benchmark refit in collocated mode (both models on same GPUs)."""
    args = get_args()
    world_size = torch.distributed.get_world_size()

    # Calculate parallelism
    src_tp = args.tensor_model_parallel_size
    src_pp = args.pipeline_model_parallel_size
    src_ep = args.expert_model_parallel_size
    src_world = src_tp * src_pp * src_ep
    src_dp = world_size // src_world

    dst_tp = args.rl_inference_tensor_model_parallel_size or src_tp
    dst_pp = args.rl_inference_pipeline_model_parallel_size or src_pp
    dst_ep = args.rl_inference_expert_model_parallel_size or src_ep
    dst_world = dst_tp * dst_pp * dst_ep
    dst_dp = world_size // dst_world

    # Print config
    src_config = {'tp': src_tp, 'pp': src_pp, 'ep': src_ep, 'dp': src_dp}
    dst_config = {'tp': dst_tp, 'pp': dst_pp, 'ep': dst_ep, 'dp': dst_dp}
    print_config_summary(args, src_config, dst_config, world_size, 'collocated')

    # Build source model
    print_rank_0("Building source model...")
    src_model = get_training_model(
        lambda pre_process, post_process, **kwargs: model_provider(
            pre_process=pre_process, post_process=post_process, parallel_output=False
        ),
        wrap_with_ddp=False
    )
    src_model[0] = src_model[0].cuda()

    # Build destination model with custom parallelism
    print_rank_0("Building destination model...")
    dst_pg_collection = build_inference_pg_collection(
        world_size,
        tp_size=dst_tp,
        pp_size=dst_pp,
        ep_size=dst_ep,
        expt_tp_size=args.rl_inference_expert_tensor_model_parallel_size,
        use_tp_pp_dp_mapping=args.use_tp_pp_dp_mapping,
    )

    dst_config = core_transformer_config_from_args(args)
    if args.num_experts:
        dst_config.expert_model_parallel_size = dst_ep
    dst_config.tensor_model_parallel_size = dst_tp
    if args.rl_inference_expert_tensor_model_parallel_size:
        dst_config.expert_tensor_parallel_size = args.rl_inference_expert_tensor_model_parallel_size

    dst_model = get_training_model(
        lambda pre_process, post_process, **kwargs: model_provider(
            pre_process=pre_process, post_process=post_process,
            pg_collection=dst_pg_collection, config=dst_config
        ),
        wrap_with_ddp=False
    )
    dst_model[0] = dst_model[0].cuda()

    torch.distributed.barrier()

    # Create refit service
    print_rank_0(f"Creating {args.refit_method} service...")
    refit_service = create_refit_service(args.refit_method)
    print_rank_0("Service created.\n")

    # Run benchmark
    timings = run_benchmark(src_model, dst_model, refit_service,
                           args.num_benchmark_warmup, args.num_benchmark_iterations)

    # Print results
    print_results(timings)


def benchmark_non_collocated():
    """Benchmark refit in non-collocated mode (separate GPU sets)."""
    args = get_args()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # Calculate parallelism
    src_tp = args.tensor_model_parallel_size
    src_pp = args.pipeline_model_parallel_size
    src_ep = args.expert_model_parallel_size
    src_world = src_tp * src_pp * src_ep

    dst_tp = args.rl_inference_tensor_model_parallel_size or src_tp
    dst_pp = args.rl_inference_pipeline_model_parallel_size or src_pp
    dst_ep = args.rl_inference_expert_model_parallel_size or src_ep
    dst_world = dst_tp * dst_pp * dst_ep

    required_size = src_world + dst_world
    if world_size < required_size:
        raise ValueError(f"Non-collocated requires {required_size} GPUs, got {world_size}")

    # Determine rank roles
    is_src_rank = rank < src_world
    is_dst_rank = src_world <= rank < required_size
    is_idle_rank = rank >= required_size

    # Print config
    src_config = {'tp': src_tp, 'pp': src_pp, 'ep': src_ep, 'dp': 1}
    dst_config = {'tp': dst_tp, 'pp': dst_pp, 'ep': dst_ep, 'dp': 1}
    print_config_summary(args, src_config, dst_config, world_size, 'non-collocated')
    if world_size > required_size:
        print_rank_0(f"Note: Ranks {required_size}-{world_size-1} are idle\n")

    # Create destination process groups (all ranks participate)
    print_rank_0("Creating process groups...")
    dst_pg_collection = build_inference_pg_collection(
        world_size=dst_world,
        tp_size=dst_tp,
        pp_size=dst_pp,
        ep_size=dst_ep,
        expt_tp_size=args.rl_inference_expert_tensor_model_parallel_size,
        use_tp_pp_dp_mapping=args.use_tp_pp_dp_mapping,
        rank_offset=src_world,
    )
    torch.distributed.barrier()

    # Idle ranks participate in collectives but have no models
    if is_idle_rank:
        src_model = None
        dst_model = None
    elif is_src_rank:
        # Build source model
        print_rank_0("Building source model...")
        src_model = get_training_model(
            lambda pre_process, post_process, **kwargs: model_provider(
                pre_process=pre_process, post_process=post_process, parallel_output=False
            ),
            wrap_with_ddp=False
        )
        src_model[0] = src_model[0].cuda()
        dst_model = None
    else:  # is_dst_rank
        # Build destination model
        print_rank_0("Building destination model...")
        dst_config = core_transformer_config_from_args(args)
        if args.num_experts:
            dst_config.expert_model_parallel_size = dst_ep
        dst_config.tensor_model_parallel_size = dst_tp
        if args.rl_inference_expert_tensor_model_parallel_size:
            dst_config.expert_tensor_parallel_size = args.rl_inference_expert_tensor_model_parallel_size

        dst_model = get_training_model(
            lambda pre_process, post_process, **kwargs: model_provider(
                pre_process=pre_process, post_process=post_process,
                pg_collection=dst_pg_collection, config=dst_config
            ),
            wrap_with_ddp=False
        )
        dst_model[0] = dst_model[0].cuda()
        src_model = None

    torch.distributed.barrier()

    # Create refit service
    print_rank_0(f"Creating {args.refit_method} service...")
    refit_service = create_refit_service(args.refit_method)
    print_rank_0("Service created.\n")

    # Run benchmark
    timings = run_benchmark(src_model, dst_model, refit_service,
                           args.num_benchmark_warmup, args.num_benchmark_iterations)

    # Print results
    print_results(timings)


def main():
    """Main benchmark function."""
    initialize_megatron(
        extra_args_provider=add_benchmark_args,
        args_defaults={
            'tokenizer_type': 'NullTokenizer',
            'no_load_optim': True,
            'no_load_rng': True,
            'no_save_optim': True,
            'no_save_rng': True,
        },
        ignore_unknown_args=False,
    )

    args = get_args()

    # Set default vocab size if not provided
    if args.vocab_size is None:
        args.vocab_size = 50257
        print_rank_0("Using default vocab_size=50257")

    # Run benchmark
    if args.refit_mode == 'collocated':
        benchmark_collocated()
    else:
        benchmark_non_collocated()


if __name__ == "__main__":
    main()
