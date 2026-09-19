# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optional telemetry and post-timing profiles, ineligible for performance gates."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path

CACHE_ENV = ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")
CACHE_SUFFIXES = (".py", ".json", ".cubin", ".ptx", ".ttir", ".ttgir", ".so")


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cache_snapshot() -> dict:
    """Hash selected compiler files only in explicitly configured cache roots."""
    snapshot = {}
    for name in CACHE_ENV:
        root = os.environ.get(name)
        row: dict = {"root": root, "files": {}, "status": "unavailable"}
        snapshot[name] = row
        if not root:
            row["reason"] = "Cache environment variable is unset; effective root is unknown"
            continue
        try:
            directory = Path(root)
            if not directory.is_dir():
                raise ValueError("Cache root is not a directory")
            for path in sorted(directory.rglob("*")):
                if path.suffix not in CACHE_SUFFIXES or not path.is_file():
                    continue
                if path.is_symlink():
                    raise ValueError("Cache inventory contains a symlink")
                before = path.stat()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("Cache file changed while being hashed")
                row["files"][path.relative_to(directory).as_posix()] = {
                    "sha256": digest,
                    "bytes": after.st_size,
                }
            row["status"] = "observed" if row["files"] else "empty"
        except (OSError, ValueError) as error:
            row["reason"] = str(error)
    return snapshot


def compare_caches(before: dict, after: dict) -> dict:
    """Missing or empty inventories cannot establish unchanged compiler caches."""
    complete = all(
        before[name]["status"] == after[name]["status"] == "observed"
        and before[name]["root"] == after[name]["root"]
        for name in CACHE_ENV
    )
    changed = {
        name: sorted(
            key
            for key in before[name]["files"].keys() | after[name]["files"].keys()
            if before[name]["files"].get(key) != after[name]["files"].get(key)
        )
        for name in CACHE_ENV
    }
    return {
        "compiler_cache_unchanged": not any(changed.values()) if complete else None,
        "changed_files": changed,
    }


def profile_calls(torch, forward, inputs, phase: str, directory: Path) -> list[dict]:
    """Profile three calls after timing; backward graph setup stays outside each trace."""
    gradient = torch.ones_like(forward()) if phase == "backward" else None
    calls = []
    for index in range(3):
        output = forward() if phase == "backward" else None
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profiler:
            if phase == "backward":
                result = torch.autograd.grad(output, inputs, grad_outputs=gradient)
            else:
                result = forward()
            torch.cuda.synchronize()
        del result, output
        path = directory / f"profile-{index}.json"
        profiler.export_chrome_trace(str(path))
        events = json.loads(path.read_text()).get("traceEvents", [])
        kernels = [event for event in events if event.get("cat") == "kernel"]
        if not kernels:
            raise ValueError(f"No GPU kernels captured in {path.name}")
        calls.append({"trace": path.name, "kernels": kernels})
    return calls


def measure_with_diagnostics(
    measure, torch, forward, inputs, phase: str, warmup: int, steps: int, directory: Path
) -> tuple[list[float], dict]:
    """Retain partial diagnostics on error and always stop the external sampler."""
    directory.mkdir(parents=True, exist_ok=False)
    uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
    device = str(uuid) if uuid else None
    report: dict = {
        "schema_version": 1,
        "kind": "kernel_performance_diagnostic",
        "diagnostic_only": True,
        "status": "incomplete",
        "mode": os.environ.get("DETERMINISM_PERF_MODE"),
        "phase": phase,
        "warmup": warmup,
        "steps": steps,
        "device_uuid": device,
        "process": {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "cpu_affinity": (
                sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
            ),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "environment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
            },
        },
        "scope": "External telemetry during the original event-timing loop, then three profiled calls. Diagnostic only; not eligible for performance approval or baseline publication.",
        "telemetry_scope": "Sparse samples include compilation, warmup, timing, cache hashing and post-timing profiling. Wall times do not isolate measured CUDA-event samples.",
        "cache_scope": {
            "suffixes": CACHE_SUFFIXES,
            "limitation": "Snapshots bracket post-timing profiling, not the earlier timed dispatch. Unchanged files do not prove that the profiled and timed kernels are identical. Shared caches may also change through other processes.",
        },
    }
    path = directory / "diagnostics.json"
    telemetry = directory / "telemetry.csv"
    errors = directory / "telemetry.stderr"
    sampler = None
    _write(path, report)
    try:
        with telemetry.open("w") as output, errors.open("w") as error_output:
            try:
                if not device:
                    raise ValueError("GPU UUID unavailable; cannot target telemetry")
                command = [
                    "nvidia-smi",
                    "--id=" + (device if device.startswith("GPU-") else "GPU-" + device),
                    "--query-gpu=timestamp,uuid,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,utilization.gpu",
                    "--format=csv,noheader,nounits",
                    "--loop-ms=20",
                ]
                report["telemetry_command"] = command
                sampler = subprocess.Popen(command, stdout=output, stderr=error_output)
            except (OSError, ValueError) as error:
                report["telemetry_error"] = str(error)
            try:
                report["timing_started_ns"] = time.time_ns()
                samples = measure(torch, forward, inputs, phase, warmup, steps)
                report["timing_finished_ns"] = time.time_ns()
                _write(directory / "timing.json", {"diagnostic_only": True, "samples_ms": samples})
                report["cache_before_profile"] = cache_snapshot()
                report["profile_started_ns"] = time.time_ns()
                report["calls"] = profile_calls(torch, forward, inputs, phase, directory)
                report["profile_finished_ns"] = time.time_ns()
                report["cache_after_profile"] = cache_snapshot()
                report.update(
                    compare_caches(report["cache_before_profile"], report["cache_after_profile"])
                )
                report["status"] = "observed"
            finally:
                if sampler is not None:
                    if sampler.poll() is None:
                        sampler.terminate()
                    try:
                        sampler.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        sampler.kill()
                        sampler.wait(timeout=5)
                    report["telemetry_exit_code"] = sampler.returncode
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["telemetry_bytes"] = telemetry.stat().st_size if telemetry.exists() else 0
        report["telemetry_available"] = bool(
            report["telemetry_bytes"]
            and report.get("telemetry_exit_code") in (0, -15)
            and errors.exists()
            and not errors.stat().st_size
        )
        _write(path, report)
    return samples, {
        "status": report["status"],
        "report": "diagnostics/diagnostics.json",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
