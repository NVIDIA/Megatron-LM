# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Probe same-process-group NCCL stream behavior for synchronous collectives.

With ``async_op=False``, ProcessGroupNCCL launches each NCCL call on the current
user CUDA stream. This script issues a reduce-scatter and an all-gather
sequentially on separate user streams. Profile it with:

    nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none \
        --cpuctxsw=none --cuda-memory-usage=false \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        --export=sqlite --output=/tmp/nccl_same_pg_two_streams_sync \
        uv run torchrun --nproc-per-node=2 \
        megatron/core/distributed/fsdp/src/docs/scripts/nccl_same_pg_two_streams_sync.py

Then inspect the NCCL kernels with:

    nsys stats --report cuda_gpu_trace --force-export=true \
        /tmp/nccl_same_pg_two_streams_sync.nsys-rep | grep -i nccl

The trace shows the reduce-scatter kernel completing before the all-gather
kernel starts, even though the operations use different user streams.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import torch
import torch.distributed as dist

DEFAULT_ELEMENTS = 64 * 1024 * 1024
logger = logging.getLogger(__name__)


def _reduce_scatter_sync(
    stream: torch.cuda.Stream, input_tensor: torch.Tensor, output_tensor: torch.Tensor
) -> None:
    """Launch a synchronous reduce-scatter on the given user CUDA stream."""
    with torch.cuda.stream(stream):
        torch.cuda.nvtx.range_push("sync_reduce_scatter_on_rs_stream")
        dist.reduce_scatter_tensor(output_tensor, input_tensor, async_op=False)
        torch.cuda.nvtx.range_pop()


def _all_gather_sync(
    stream: torch.cuda.Stream, input_tensor: torch.Tensor, output_tensor: torch.Tensor
) -> None:
    """Launch a synchronous all-gather on the given user CUDA stream."""
    with torch.cuda.stream(stream):
        torch.cuda.nvtx.range_push("sync_all_gather_on_ag_stream")
        dist.all_gather_into_tensor(output_tensor, input_tensor, async_op=False)
        torch.cuda.nvtx.range_pop()


def launch_collectives(
    *,
    rs_stream: torch.cuda.Stream,
    ag_stream: torch.cuda.Stream,
    rs_in: torch.Tensor,
    rs_out: torch.Tensor,
    ag_in: torch.Tensor,
    ag_out: torch.Tensor,
) -> None:
    """Launch reduce-scatter and all-gather on separate user CUDA streams."""
    _reduce_scatter_sync(rs_stream, rs_in, rs_out)
    _all_gather_sync(ag_stream, ag_in, ag_out)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Probe async_op=False NCCL stream behavior on one process group."
    )
    parser.add_argument(
        "--elements",
        type=int,
        default=DEFAULT_ELEMENTS,
        help="Number of bf16 elements contributed per rank.",
    )
    return parser.parse_args()


def main() -> None:
    """Run and profile the synchronous collective probe."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise RuntimeError(f"Expected at least 2 ranks, got {world_size}.")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")

    rs_stream = torch.cuda.Stream(device=device)
    ag_stream = torch.cuda.Stream(device=device)

    ag_in = torch.full((args.elements,), rank + 1, dtype=torch.bfloat16, device=device)
    ag_out = torch.empty((world_size * args.elements,), dtype=torch.bfloat16, device=device)
    rs_in = torch.full((world_size * args.elements,), rank + 1, dtype=torch.bfloat16, device=device)
    rs_out = torch.empty((args.elements,), dtype=torch.bfloat16, device=device)

    launch_kwargs = {
        "rs_stream": rs_stream,
        "ag_stream": ag_stream,
        "rs_in": rs_in,
        "rs_out": rs_out,
        "ag_in": ag_in,
        "ag_out": ag_out,
    }

    # Warm up communicator initialization and one steady-state iteration outside
    # the profiler range.
    launch_collectives(**launch_kwargs)
    torch.cuda.synchronize(device)
    dist.barrier()

    if rank == 0:
        logger.info("rank0 torch rs_stream.cuda_stream=%s", rs_stream.cuda_stream)
        logger.info("rank0 torch ag_stream.cuda_stream=%s", ag_stream.cuda_stream)

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("profile_sync_same_pg")
    launch_collectives(**launch_kwargs)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(device)
    torch.cuda.cudart().cudaProfilerStop()

    dist.barrier()
    if rank == 0:
        logger.info("done")
    time.sleep(0.2)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
