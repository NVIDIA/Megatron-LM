# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in CPU placement validation for inference workers."""

import ctypes
import logging
import os
import threading
from pathlib import Path
from typing import Callable, Iterable

import torch

logger = logging.getLogger(__name__)

_NVML_UUID_PREFIXES = ("GPU-", "MIG-")


def _nvml_uuid(torch_uuid: object) -> str:
    """Convert a torch device UUID to the prefixed form NVML expects."""
    uuid = str(torch_uuid)
    if uuid.startswith(_NVML_UUID_PREFIXES):
        return uuid
    return f"GPU-{uuid}"


def _possible_cpu_count(possible: str) -> int:
    """Number of CPU IDs covered by a /sys/devices/system/cpu/possible list, e.g. "0-3,128-131"."""
    return max(int(part.split("-")[-1]) for part in possible.strip().split(",")) + 1


def _mask_to_cpus(words: Iterable[int], word_bits: int) -> set[int]:
    """Decode an NVML CPU affinity bitmask (array of words) into CPU IDs."""
    return {
        word_index * word_bits + bit
        for word_index, word in enumerate(words)
        for bit in range(word_bits)
        if word & (1 << bit)
    }


def _gpu_local_cpus(gpu_uuid: str) -> set[int]:
    """Query NVML for the CPUs on the NUMA node(s) local to the given GPU."""
    try:
        import pynvml
    except ImportError as exc:
        raise RuntimeError("CPU affinity validation requires nvidia-ml-py (pynvml).") from exc

    # CPU IDs can be sparse; size the NVML bitmask using the highest possible ID.
    num_cpus = _possible_cpu_count(Path("/sys/devices/system/cpu/possible").read_text())
    word_bits = ctypes.sizeof(ctypes.c_ulong) * 8
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid)
        words = pynvml.nvmlDeviceGetCpuAffinity(handle, (num_cpus + word_bits - 1) // word_bits)
    finally:
        pynvml.nvmlShutdown()
    cpus = _mask_to_cpus(words, word_bits)
    if not cpus:
        raise RuntimeError(f"NVML returned no GPU-local CPUs for {gpu_uuid}.")
    return cpus


def _format_cpus(cpus: Iterable[int]) -> str:
    """Format CPU IDs as a compact list of ranges, e.g. "0-69,140"."""
    ranges = []
    for cpu in sorted(cpus):
        if ranges and cpu == ranges[-1][1] + 1:
            ranges[-1][1] = cpu
        else:
            ranges.append([cpu, cpu])
    return ",".join(str(lo) if lo == hi else f"{lo}-{hi}" for lo, hi in ranges)


def _find_violations(
    tids: Iterable[int], caller: int, local_cpus: set[int], get_affinity: Callable[[int], set[int]]
) -> list[str]:
    """Return a description of each thread whose allowed CPUs are not all GPU-local."""
    violations = []
    for tid in tids:
        try:
            allowed = set(get_affinity(tid))
        except ProcessLookupError:
            if tid == caller:
                raise
            continue  # A helper thread may exit between enumeration and inspection.
        if not allowed or not allowed <= local_cpus:
            violations.append(f"tid={tid} allowed=[{_format_cpus(allowed)}]")
    return violations


def validate_inference_cpu_affinity(*, all_threads: bool = False, warn_only: bool = False) -> None:
    """Check that threads may only run on CPUs local to the current CUDA device.

    Checks the calling thread, or every thread in the process when all_threads is set.
    Never changes affinity. With warn_only, violations are logged instead of raised;
    failure to determine the topology always raises.

    Raises:
        RuntimeError: Topology is unavailable, or (unless warn_only) a checked thread
            may run on non-local CPUs.
    """
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("Inference CPU affinity validation requires Linux sched_getaffinity.")
    identity = f"rank={os.environ.get('RANK', '?')} pid={os.getpid()}"
    try:
        gpu_uuid = _nvml_uuid(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
        local_cpus = _gpu_local_cpus(gpu_uuid)
    except Exception as exc:
        raise RuntimeError(f"Cannot determine GPU-local CPU affinity ({identity}): {exc}") from exc

    caller = threading.get_native_id()
    try:
        tids = {caller}
        if all_threads:
            tids.update(int(path.name) for path in Path("/proc/self/task").iterdir())
        violations = _find_violations(sorted(tids), caller, local_cpus, os.sched_getaffinity)
    except OSError as exc:
        raise RuntimeError(f"Cannot inspect thread CPU affinity ({identity}): {exc}") from exc
    if not violations:
        return

    message = (
        f"Inference CPU affinity check failed ({identity}, GPU={gpu_uuid}): "
        f"{'; '.join(violations)}; GPU-local CPUs=[{_format_cpus(local_cpus)}]. "
        "Bind this rank to GPU-local CPUs with your launcher before CUDA initialization."
    )
    if warn_only:
        logger.warning(message)
    else:
        raise RuntimeError(message)
