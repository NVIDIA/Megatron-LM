# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Check multi-node symmetric all-gather and reduce-scatter with separate buffers.

Reduce-scatter uses an independent output allocation outside symmetric memory.
"""

import argparse
import json
import os
import socket
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.autograd import DeviceType
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Partial, Replicate

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.dbuffer import DBuffer
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import RowAtomic


def main() -> None:
    """Check collective values, output allocation, and NCCL symmetric kernels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # Cross-node GIN all-gather requires both windows registered, unlike all-LSA kernels.
    assert os.environ.get("NCCL_MNNVL_ENABLE") == "0", "Run with NCCL_MNNVL_ENABLE=0."
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    symm_mem.set_backend("NCCL")
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=5))
    rank, world_size = dist.get_rank(), dist.get_world_size()
    hosts = [None] * world_size
    # This also initializes the communicator before symmetric-window registration.
    dist.all_gather_object(hosts, socket.gethostname())
    assert len(set(hosts)) >= 2, "This functional test requires multiple nodes."
    mesh = DeviceMesh.from_group(dist.group.WORLD, device_type=device.type)

    allocator = symm_mem.get_mempool_allocator(device)
    # Follow PyTorch's no-split policy to avoid concurrent operations sharing
    # a segment's signal pad.
    # https://github.com/pytorch/pytorch/blob/v2.15.0/torch/distributed/_symmetric_memory/__init__.py#L2415-L2420
    pool = torch.cuda.MemPool(allocator, no_split=True)
    with torch.cuda.use_mem_pool(pool):
        source = DBuffer.empty(
            mesh, [RowAtomic()], [(world_size * 1024, 1024)], torch.bfloat16, device
        )
        output = DBuffer(mesh, [Replicate()], source.layout, source.dtype, device)
    assert source.is_symmetric_memory and output.is_symmetric_memory
    source.local_buffer.fill_(rank + 1)

    # Reuse the gathered allocation as RS input; its output uses the ordinary allocator.
    rs_input = output.view([Partial("sum")])
    rs_output = DBuffer(mesh, [RowAtomic()], source.layout, source.dtype, device)
    rs_input_symmetric = rs_input.is_symmetric_memory
    rs_output_symmetric = rs_output.is_symmetric_memory
    rs_buffers_independent = (
        rs_input.local_buffer.untyped_storage().data_ptr()
        != rs_output.local_buffer.untyped_storage().data_ptr()
    )
    assert rs_input_symmetric, "Reduce-scatter input must use symmetric memory."
    assert not rs_output_symmetric, "Reduce-scatter output must use ordinary CUDA memory."
    assert rs_buffers_independent, "Reduce-scatter input and output must not share storage."

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        assert source.allgather(0, out=output) is output
        torch.cuda.synchronize()
        actual = output.local_buffer.view(world_size, -1).cpu()

        rs_input.local_buffer.fill_(rank + 1)
        assert rs_input.reduce_scatter(0, RowAtomic(), out=rs_output) is rs_output
        torch.cuda.synchronize()

    rs_actual = rs_output.local_buffer.cpu()
    # Release registered allocations before destroying their communicator.
    del source, output, rs_input, rs_output, pool
    kernels = {
        event.name
        for event in prof.events()
        if event.device_type == DeviceType.CUDA and "nccl" in event.name.lower()
    }
    ag_kernels = sorted(kernel for kernel in kernels if "allgather" in kernel.lower())
    rs_kernels = sorted(kernel for kernel in kernels if "reducescatter" in kernel.lower())
    expected = torch.arange(1, world_size + 1, dtype=actual.dtype)[:, None].expand_as(actual)
    result = {
        "rank": rank,
        "hosts": sorted(set(hosts)),
        "world_size": world_size,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "allgather_kernels": ag_kernels,
        "values_match": torch.equal(actual, expected),
        "symmetric_allgather": bool(ag_kernels)
        and all("ncclSymk" in kernel for kernel in ag_kernels),
        "reduce_scatter_kernels": rs_kernels,
        "reduce_scatter_values_match": torch.equal(
            rs_actual, torch.full_like(rs_actual, world_size * (world_size + 1) // 2)
        ),
        "reduce_scatter_input_symmetric": rs_input_symmetric,
        "reduce_scatter_output_symmetric": rs_output_symmetric,
        "reduce_scatter_buffers_independent": rs_buffers_independent,
        "symmetric_reduce_scatter": bool(rs_kernels)
        and all("ncclSymkDevKernel_ReduceScatter_RailA2A_" in kernel for kernel in rs_kernels),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    dist.barrier(device_ids=[device.index])
    dist.destroy_process_group()
    print(json.dumps(result), flush=True)
    assert result["values_match"], "All-gather returned incorrect rank-dependent values."
    assert result[
        "symmetric_allgather"
    ], f"Expected NCCL symmetric all-gather, observed {ag_kernels}."
    assert result["reduce_scatter_values_match"], "Reduce-scatter returned incorrect values."
    assert result[
        "symmetric_reduce_scatter"
    ], f"Expected GIN symmetric reduce-scatter, observed {rs_kernels}."


if __name__ == "__main__":
    main()
