# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in check that inference threads are bound to GPU-local CPUs."""

import logging
import os
import threading
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch

logger = logging.getLogger(__name__)


def _format_cpus(cpus: Iterable[int]) -> str:
    """Format CPU IDs as ranges, e.g. "0-69,140"."""
    ranges = []
    for cpu in sorted(cpus):
        if ranges and cpu == ranges[-1][1] + 1:
            ranges[-1][1] = cpu
        else:
            ranges.append([cpu, cpu])
    return ",".join(str(lo) if lo == hi else f"{lo}-{hi}" for lo, hi in ranges)


def _gpu_local_cpus() -> Optional[set[int]]:
    """Read the CPUs of the current CUDA device's NUMA node from sysfs, or None if unknown."""
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    pci = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
    try:
        node = int(Path(f"/sys/bus/pci/devices/{pci}/numa_node").read_text())
        if node < 0:  # Usual on single-NUMA machines.
            return None
        cpulist = Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip()
    except OSError:  # No sysfs, e.g. non-Linux or a masked /sys.
        return None
    cpus = set()
    for part in cpulist.split(","):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def _check_threads(
    tids: Iterable[int],
    caller: int,
    local_cpus: set[int],
    get_affinity: Callable[[int], set[int]],
    warn_for_helper_threads: bool,
) -> None:
    violations = {}
    for tid in tids:
        try:
            allowed = get_affinity(tid)
        except ProcessLookupError:
            if tid == caller:
                raise
            continue  # A helper thread exited after enumeration.
        if not allowed <= local_cpus:
            violations[tid] = allowed
    if not violations:
        return
    details = "; ".join(
        f"tid={tid}{' (caller)' if tid == caller else ''} allowed=[{_format_cpus(cpus)}]"
        for tid, cpus in sorted(violations.items())
    )
    message = (
        f"CPU affinity check failed (rank={os.environ.get('RANK', '?')}): {details}; "
        f"GPU-local CPUs=[{_format_cpus(local_cpus)}]. "
        "Bind this rank to GPU-local CPUs with your launcher (e.g. torchrun --numa-binding=node), "
        "or disable this check with --no-require-gpu-local-cpu-affinity."
    )
    if warn_for_helper_threads and caller not in violations:
        logger.warning(message)
    else:
        raise RuntimeError(message)


def validate_inference_cpu_affinity(
    *, all_threads: bool = False, warn_for_helper_threads: bool = False
) -> None:
    """Raise if the calling thread, or any thread with all_threads, may run on CPUs that are
    not local to the current CUDA device. With warn_for_helper_threads, threads other than
    the caller only log a warning. Skipped if the GPU's NUMA node is unknown. Never changes
    affinity.
    """
    local_cpus = _gpu_local_cpus() if hasattr(os, "sched_getaffinity") else None
    if local_cpus is None:
        return
    caller = threading.get_native_id()
    tids = sorted(int(t) for t in os.listdir("/proc/self/task")) if all_threads else [caller]
    _check_threads(tids, caller, local_cpus, os.sched_getaffinity, warn_for_helper_threads)
