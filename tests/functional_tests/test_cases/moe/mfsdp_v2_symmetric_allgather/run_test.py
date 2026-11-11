# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Check symmetric all-gather registration with independent buffers across nodes."""

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
from torch.distributed.tensor import Replicate

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.dbuffer import DBuffer
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import RowAtomic


def main() -> None:
    """Gather rank-dependent values and verify NCCL selected symmetric kernels."""
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
    # Match get_mem_pool()'s no-split policy: a segment owns a shared signal pad.
    input_pool = torch.cuda.MemPool(allocator, no_split=True)
    output_pool = torch.cuda.MemPool(allocator, no_split=True)
    # Separate pools keep output registration from also registering the input segment.
    with torch.cuda.use_mem_pool(input_pool):
        source = DBuffer.empty(
            mesh, [RowAtomic()], [(world_size * 1024, 1024)], torch.bfloat16, device
        )
    with torch.cuda.use_mem_pool(output_pool):
        output = DBuffer(mesh, [Replicate()], source.layout, source.dtype, device)
    assert source.is_symmetric_memory and output.is_symmetric_memory
    source.local_buffer.fill_(rank + 1)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        assert source.allgather(0, out=output) is output
        torch.cuda.synchronize()

    actual = output.local_buffer.view(world_size, -1).cpu()
    # Release registered allocations before destroying their communicator.
    del source, output, input_pool, output_pool
    kernels = sorted(
        {
            event.name
            for event in prof.events()
            if event.device_type == DeviceType.CUDA
            and "nccl" in event.name.lower()
            and "allgather" in event.name.lower()
        }
    )
    expected = torch.arange(1, world_size + 1, dtype=actual.dtype)[:, None].expand_as(actual)
    result = {
        "rank": rank,
        "hosts": sorted(set(hosts)),
        "world_size": world_size,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "allgather_kernels": kernels,
        "values_match": torch.equal(actual, expected),
        "symmetric_allgather": bool(kernels) and all("ncclSymk" in kernel for kernel in kernels),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    dist.barrier(device_ids=[device.index])
    dist.destroy_process_group()
    print(json.dumps(result), flush=True)
    assert result["values_match"], "All-gather returned incorrect rank-dependent values."
    assert result["symmetric_allgather"], f"Expected NCCL symmetric all-gather, observed {kernels}."


if __name__ == "__main__":
    main()
