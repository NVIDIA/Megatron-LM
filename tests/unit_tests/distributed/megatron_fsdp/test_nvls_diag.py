# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""THROWAWAY diagnostic (DO NOT MERGE): does NCCL_NVLS_ENABLE=1 engage the multicast
symmetric kernels (STMC/LDMC) at moderate sizes on the CI runner, and at what memory cost?

CI's run_ci_test.sh exports NCCL_NVLS_ENABLE=0; this file overrides it to 1 (module level,
before the process group is created) to see whether the multicast symmetric path engages at
hidden=1024/2048/4096 (vs the non-multicast ST/LD path, vs ring) and its peak memory. Prints
[NVLSDIAG] lines (pyproject sets pytest -s) and never fails.
"""
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.profiler import ProfilerActivity, profile

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
)

# NCCL reads this at communicator init (created in the distributed_setup fixture), so
# setting it here at collection time overrides the shell default of 0 for this process.
os.environ["NCCL_NVLS_ENABLE"] = "1"


def _run(hidden, device, mesh):
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats(device)
    model = torch.nn.Linear(hidden, hidden, bias=False).to(device=device, dtype=torch.bfloat16)
    placements = Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])
    fully_shard(model, mesh=mesh, placements=placements, use_symm_mem=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.0, foreach=False)
    x = torch.randn(64, hidden, device=device, dtype=torch.bfloat16)

    def step():
        opt.zero_grad()
        model(x).sum().backward()
        opt.step()
        torch.cuda.synchronize()

    step()  # warm
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step()
    names = [e.name for e in prof.events()]
    symk = sorted({n for n in names if "ncclSymk" in n})
    ring = sorted({n for n in names if "RING_LL" in n})
    peak_mib = torch.cuda.max_memory_allocated(device) / 2**20
    return symk, ring, peak_mib


def test_nvls_diagnostic(distributed_setup):
    """Informational only; always passes."""
    device = distributed_setup.device
    world_size = distributed_setup.world_size
    rank = dist.get_rank()
    if rank == 0:
        print(
            f"[NVLSDIAG] NCCL_NVLS_ENABLE={os.environ.get('NCCL_NVLS_ENABLE')} "
            f"world_size={world_size} device={torch.cuda.get_device_name(device)}"
        )
    if world_size < 2:
        return

    mesh = init_device_mesh(device.type, (world_size,))
    for hidden in (1024, 2048, 4096):
        try:
            symk, ring, peak_mib = _run(hidden, device, mesh)
        except Exception as e:  # noqa: BLE001 -- diagnostic: report whatever it raises
            if rank == 0:
                print(f"[NVLSDIAG] hidden={hidden} FAILED: {type(e).__name__}: {e}")
            continue
        if rank == 0:
            multicast = any("MC" in k for k in symk)  # STMC / LDMC == multicast variants
            print(
                f"[NVLSDIAG] hidden={hidden} multicast={multicast} peak_MiB={peak_mib:.0f} "
                f"symk={symk} ring={ring}"
            )
