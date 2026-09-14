# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Profile local MFSDP v2 optimizer tensors on a small CPU-bound model.

Run with torchrun --standalone --nproc-per-node=2 tools/benchmark_mfsdp_local_tensors.py.
This isolates wrapper overhead; it is not the DeepSeek proxy from issue #7264.
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)


class Branches(nn.Module):
    """Many independent small FSDP units to expose per-parameter CPU overhead."""

    def __init__(self, units: int, hidden: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(units)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average scalar losses across the independent units."""
        return torch.stack([layer(x).square().mean() for layer in self.layers]).mean()


def main() -> None:
    """Report synchronized step latency and DTensor-wrap counts for the runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--units', type=int, default=32)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--microbatches', type=int, default=2)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    mesh = init_device_mesh('cuda', (dist.get_world_size(),))
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    results = []
    torch.manual_seed(1234)
    model = Branches(args.units, args.hidden).cuda()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with fully_shard_context(device=torch.device('cuda', torch.cuda.current_device())):
        for layer in model.layers:
            fully_shard(layer, mesh, placements)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, foreach=True)
    fully_shard_optimizer(optimizer)
    torch.cuda.synchronize()
    init_ms = (time.perf_counter() - start) * 1000
    inputs = torch.randn(args.microbatches, 4, args.hidden, device='cuda')

    def step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        for x in inputs:
            with torch.profiler.record_function('benchmark_forward'):
                loss = model(x) / args.microbatches
            with torch.profiler.record_function('benchmark_backward'):
                loss.backward()
        with torch.profiler.record_function('benchmark_optimizer'):
            optimizer.step()
        return loss.detach()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    times = []
    for _ in range(args.steps):
        start = time.perf_counter()
        loss = step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        step()
        torch.cuda.synchronize()
    result = {
        'rank': dist.get_rank(),
        'init_ms': init_ms,
        'step_median_ms': statistics.median(times),
        'last_loss': loss.item(),
        'dtensor_wraps': sum(e.name == '_FromTorchTensor' for e in prof.events()),
        'profile_host_ms': {
            name: sum(e.cpu_time_total for e in prof.events() if e.name == name) / 1000
            for name in (
                'benchmark_forward',
                'benchmark_backward',
                'benchmark_optimizer',
                '_FromTorchTensor',
            )
        },
    }
    results.append(result)
    del optimizer, model
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, results)
    if dist.get_rank() == 0:
        sys.stdout.write(json.dumps(gathered, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
