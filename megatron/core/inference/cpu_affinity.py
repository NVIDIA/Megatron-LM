# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in CPU placement validation for inference workers."""

import ctypes
import os
import sys
import threading
from pathlib import Path


def _gpu_local_cpus() -> tuple[str, set[int]]:
    import torch

    try:
        import pynvml
    except ImportError as exc:
        raise RuntimeError("CPU affinity validation requires nvidia-ml-py (pynvml).") from exc

    # CUDA ordinals need not match NVML ordinals, especially with CUDA_VISIBLE_DEVICES.
    device = torch.cuda.current_device()
    gpu_uuid = str(torch.cuda.get_device_properties(device).uuid)
    # CPU IDs can be sparse; size the NVML bitmask using the highest possible ID.
    possible = Path("/sys/devices/system/cpu/possible").read_text().strip()
    num_cpus = max(int(part.split("-")[-1]) for part in possible.split(",")) + 1
    word_bits = ctypes.sizeof(ctypes.c_ulong) * 8
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid)
        words = pynvml.nvmlDeviceGetCpuAffinity(handle, (num_cpus + word_bits - 1) // word_bits)
    finally:
        pynvml.nvmlShutdown()
    cpus = {
        word_index * word_bits + bit
        for word_index, word in enumerate(words)
        for bit in range(word_bits)
        if word & (1 << bit)
    }
    if not cpus:
        raise RuntimeError(f"NVML returned no GPU-local CPUs for {gpu_uuid}.")
    return gpu_uuid, cpus


def validate_inference_cpu_affinity(*, all_threads: bool = False) -> None:
    """Require GPU-local allowed CPUs without changing thread affinity.

    Checks the calling thread, or every existing thread when ``all_threads`` is true.
    The current CUDA device must already be selected. Uses its UUID to query NVML's
    NUMA-node CPU affinity. This is a snapshot: it cannot certify earlier binding,
    memory placement, future threads, or performance. Call again after application
    warmup if it creates threads or changes affinity.

    Raises:
        RuntimeError: Locality cannot be determined, or a checked thread can run on
            CPUs outside the GPU-local set. Requires Linux and nvidia-ml-py.
    """
    if sys.platform != "linux" or not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("Inference CPU affinity validation requires Linux sched_getaffinity.")
    identity = f"rank={os.environ.get('RANK', '?')} pid={os.getpid()}"
    try:
        gpu_uuid, local_cpus = _gpu_local_cpus()
    except Exception as exc:
        raise RuntimeError(f"Cannot determine GPU-local CPU affinity ({identity}): {exc}") from exc

    caller = threading.get_native_id()
    try:
        tids = {caller}
        if all_threads:
            tids.update(int(path.name) for path in Path("/proc/self/task").iterdir())
        violations = []
        for tid in sorted(tids):
            try:
                allowed = set(os.sched_getaffinity(tid))
            except ProcessLookupError:
                if tid == caller:
                    raise
                continue  # A helper may exit between enumeration and inspection.
            if not allowed or not allowed <= local_cpus:
                violations.append(f"tid={tid} allowed={sorted(allowed)}")
    except OSError as exc:
        raise RuntimeError(f"Cannot inspect thread CPU affinity ({identity}): {exc}") from exc
    if violations:
        raise RuntimeError(
            f"Inference CPU affinity check failed ({identity}, GPU={gpu_uuid}): "
            f"{'; '.join(violations)}; GPU-local CPUs={sorted(local_cpus)}. "
            "Bind this rank to GPU-local CPUs with your launcher before CUDA initialization "
            "and keep helper threads within that set. No affinity was changed."
        )
