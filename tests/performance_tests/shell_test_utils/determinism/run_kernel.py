# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Single-GPU operator timings; called in a fresh process for each policy arm."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def read_result(path: Path, measurement: dict, mode: str) -> dict:
    """Reject partial, nonfinite, or differently configured timing results."""
    result = json.loads(path.read_text())
    validate_result(result, measurement, mode)
    return result


def validate_result(result: dict, measurement: dict, mode: str) -> None:
    """Apply the same timing contract to embedded and separately saved results."""
    keys = ("kernel_case", "phase", "tokens", "hidden_size", "dtype", "warmup", "steps")
    expected = {key: measurement[key] for key in keys}
    if "diagnostic_only" in measurement:
        raise ValueError("Profiled timings are not eligible for performance evidence")
    if (
        not isinstance(result, dict)
        or result.get("measurement") != expected
        or result.get("mode") != mode
        or result.get("deterministic_algorithms") is not (mode == "det")
        or "diagnostics" in result
    ):
        raise ValueError("Kernel timing configuration does not match the requested arm")
    samples = result.get("samples_ms")
    if (
        not isinstance(samples, list)
        or len(samples) != measurement["steps"]
        or any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in samples
        )
    ):
        raise ValueError("Kernel timings must contain every requested finite, positive sample")


def measure(torch, forward, inputs, phase: str, warmup: int, steps: int) -> list[float]:
    """Time only the requested phase with CUDA events, excluding setup and warmup.

    Backward builds a fresh graph before each sample. autograd.grad avoids leaf
    gradient accumulation; graph construction and upstream-gradient allocation
    finish before the start event. The interval includes all operator launches,
    not just one generated device kernel. No CUDA graph capture is used.
    """
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    gradient = torch.ones_like(forward()) if phase == "backward" else None
    for index in range(warmup + steps):
        output = forward() if phase == "backward" else None
        torch.cuda.synchronize()
        start.record()
        if phase == "backward":
            result = torch.autograd.grad(output, inputs, grad_outputs=gradient)
        else:
            result = forward()
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end)
        if index >= warmup:
            samples.append(elapsed)
        del result, output
    return samples


def main(argv: list[str] | None = None) -> int:
    """Measure one arm after the parent has installed its environment policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-case", required=True)
    parser.add_argument("--phase", choices=("forward", "backward"), required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args(argv)
    measurement = vars(args)
    import torch

    mode = os.environ["DETERMINISM_PERF_MODE"]
    if (
        mode not in ("det", "default")
        or min(args.tokens, args.hidden_size, args.steps) < 1
        or args.warmup < 1
    ):
        parser.error("Invalid policy, shape, or sample count")
    if not torch.cuda.is_available():
        raise RuntimeError("Kernel timing requires a CUDA GPU")
    torch.cuda.set_device(0)
    from kernel_case import case_signature, kernel_policy, make_case

    with kernel_policy(torch, mode == "det"):
        function, arguments, op_id = make_case(
            torch, args.kernel_case, args.tokens, args.hidden_size, getattr(torch, args.dtype)
        )
        inputs = tuple(
            value for value in arguments if isinstance(value, torch.Tensor) and value.requires_grad
        )
        signature = case_signature(torch, args.kernel_case, arguments)
        device_uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
        result = {
            "measurement": measurement,
            "mode": mode,
            "op_id": op_id,
            "implementation": "torch.compile:" + args.kernel_case,
            "inputs": [
                {
                    "shape": list(value.shape),
                    "stride": list(value.stride()),
                    "dtype": str(value.dtype),
                }
                for value in inputs
            ],
            "case_signature": signature,
            "device_uuid": str(device_uuid) if device_uuid else None,
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
            "seed": signature["input_seed"],
        }
        result["samples_ms"] = measure(
            torch, lambda: function(*arguments), inputs, args.phase, args.warmup, args.steps
        )
    path = Path(os.environ["DETERMINISM_PERF_LOG_DIR"]) / "kernel.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
