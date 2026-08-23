"""Benchmark vLLM-aligned DeepEP and HybridEP dispatch/combine."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from megatron.lite.model.deepseek_v4.vllm.primitive.moe.communication import (
    VLLMAlignedHybridEPDispatcher,
    VLLMAlignedNormalDeepEPDispatcher,
)
from megatron.lite.primitive.parallel import ParallelState


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("deepep", "hybridep"))
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--local-experts", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", local_rank)
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.new_group(list(range(world_size)), backend="nccl")
    parallel_state = ParallelState(
        ep_size=world_size,
        ep_rank=rank,
        ep_group=group,
        tp_ep_group=group,
    )
    dispatcher_type = (
        VLLMAlignedNormalDeepEPDispatcher
        if args.backend == "deepep"
        else VLLMAlignedHybridEPDispatcher
    )
    dispatcher = dispatcher_type(
        world_size * args.local_experts,
        args.hidden,
        parallel_state,
        **(
            {"hybridep_max_tokens_per_rank": args.tokens * args.topk}
            if args.backend == "hybridep"
            else {}
        ),
    )
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260823 + rank)
    hidden = torch.randn(
        args.tokens,
        args.hidden,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    rows = torch.arange(args.tokens, device="cuda").unsqueeze(1)
    slots = torch.arange(args.topk, device="cuda").unsqueeze(0)
    indices = (
        rank * args.local_experts + rows + slots * 3
    ).remainder(world_size * args.local_experts)
    weights = torch.arange(
        1, args.topk + 1, dtype=torch.float32, device="cuda"
    )
    weights = (weights / weights.sum()).unsqueeze(0).expand(
        args.tokens, -1
    ).contiguous()

    def operation() -> torch.Tensor:
        dispatched, _, _ = dispatcher.dispatch(hidden, weights, indices)
        return dispatcher.combine(dispatched)

    for _ in range(args.warmup):
        output = operation()
    torch.cuda.synchronize()
    dist.barrier()
    started = time.perf_counter()
    for _ in range(args.iterations):
        output = operation()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000 / args.iterations
    elapsed = torch.tensor(elapsed_ms, device="cuda", dtype=torch.float64)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    if not torch.isfinite(output).all():
        raise AssertionError(f"{args.backend} produced non-finite output")
    if rank == 0:
        print(
            json.dumps(
                {
                    "backend": args.backend,
                    "world_size": world_size,
                    "tokens_per_rank": args.tokens,
                    "hidden": args.hidden,
                    "topk": args.topk,
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                    "dispatch_combine_ms_max_rank": float(elapsed.item()),
                    "nvlink_domain_ranks": int(
                        os.environ.get(
                            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN",
                            "0",
                        )
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
