# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""THROWAWAY diagnostic (DO NOT MERGE): why does symmetric memory fall back to ring on CI?

Dumps CUDA multicast support + NVLink topology on the CI runner and checks whether an
8-rank symmetric-memory all-gather actually selects ncclSymk* kernels. Prints [SYMMDIAG]
lines (pyproject sets pytest -s, so they reach the CI job log) and never fails.
"""
import ctypes
import subprocess

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.profiler import ProfilerActivity, profile

import torch.distributed as dist
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
)


def _multicast_supported(dev_index: int):
    # CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
        libcuda.cuInit(0)
        dev = ctypes.c_int()
        libcuda.cuDeviceGet(ctypes.byref(dev), dev_index)
        val = ctypes.c_int(-1)
        rc = libcuda.cuDeviceGetAttribute(ctypes.byref(val), 132, dev)
        return f"rc={rc} value={val.value}"
    except Exception as e:  # noqa: BLE001
        return f"query-failed: {type(e).__name__}: {e}"


def test_symm_multicast_diagnostic(distributed_setup):
    """Informational only; always passes."""
    device = distributed_setup.device
    world_size = distributed_setup.world_size
    rank = torch.distributed.get_rank()

    if rank == 0:
        print(f"[SYMMDIAG] world_size={world_size} device={torch.cuda.get_device_name(device)}")
        print(f"[SYMMDIAG] visible_device_count={torch.cuda.device_count()}")
        print(f"[SYMMDIAG] multicast_supported(dev0)={_multicast_supported(0)}")
        try:
            topo = subprocess.run(
                ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=60
            ).stdout
            print("[SYMMDIAG] nvidia-smi topo -m:\n" + topo)
        except Exception as e:  # noqa: BLE001
            print(f"[SYMMDIAG] topo query failed: {type(e).__name__}: {e}")
        try:
            mig = subprocess.run(
                ["nvidia-smi", "--query-gpu=mig.mode.current", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=60,
            ).stdout.strip()
            print(f"[SYMMDIAG] mig.mode.current: {mig!r}")
        except Exception as e:  # noqa: BLE001
            print(f"[SYMMDIAG] mig query failed: {type(e).__name__}: {e}")

    if world_size < 2:
        return

    # Does an 8-rank symmetric all-gather actually select ncclSymk* kernels here?
    mesh = init_device_mesh(device.type, (world_size,))
    torch.manual_seed(0)
    model = torch.nn.Linear(1024, 1024, bias=False).to(device=device, dtype=torch.bfloat16)
    placements = Placements(
        dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()]
    )
    fully_shard(model, mesh=mesh, placements=placements, use_symm_mem=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.0, foreach=False)
    x = torch.randn(64, 1024, device=device, dtype=torch.bfloat16)

    def step():
        opt.zero_grad()
        model(x).sum().backward()
        opt.step()
        torch.cuda.synchronize()

    step()  # warm
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step()
    names = [e.name for e in prof.events()]
    symk = [n for n in names if "ncclSymk" in n]
    ring = [n for n in names if "RING_LL" in n]
    if rank == 0:
        print(f"[SYMMDIAG] ncclSymk kernels: {sorted(set(symk))}")
        print(f"[SYMMDIAG] RING kernels: {sorted(set(ring))}")
        print(f"[SYMMDIAG] symm_engaged={len(symk) > 0}")
