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
    keys = ("kernel_case", "phase", "tokens", "hidden_size", "dtype", "warmup", "steps")
    if (
        not isinstance(result, dict)
        or result.get("measurement") != {key: measurement[key] for key in keys}
        or result.get("mode") != mode
        or result.get("deterministic_algorithms") is not (mode == "det")
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
    return result


def make_case(torch, name: str, tokens: int, hidden: int, dtype):
    """Create fixed inputs and the production operator; imports follow mode setup."""
    if name in ("bias_swiglu", "weighted_swiglu"):
        from megatron.core.fusions.fused_bias_swiglu import (
            bias_swiglu_impl,
            weighted_bias_swiglu_impl,
        )

        x = torch.randn(tokens, 2 * hidden, device="cuda", dtype=dtype, requires_grad=True)
        if name == "bias_swiglu":
            bias = torch.randn(2 * hidden, device="cuda", dtype=dtype, requires_grad=True)
            return lambda: bias_swiglu_impl(x, bias), (x, bias), "fused_bias_swiglu"
        weights = torch.rand(tokens, 1, device="cuda", dtype=torch.float32, requires_grad=True)
        return (
            lambda: weighted_bias_swiglu_impl(x, None, weights),
            (x, weights),
            "fused_bias_swiglu",
        )
    if name == "weighted_squared_relu":
        from megatron.core.fusions.fused_weighted_squared_relu import weighted_squared_relu_impl

        x = torch.randn(tokens, hidden, device="cuda", dtype=dtype, requires_grad=True)
        weights = torch.rand(tokens, 1, device="cuda", dtype=torch.float32, requires_grad=True)
        return (
            lambda: weighted_squared_relu_impl(x, weights),
            (x, weights),
            "fused_weighted_squared_relu",
        )
    raise ValueError(f"Unknown kernel case: {name}")


def measure(torch, forward, inputs, phase: str, warmup: int, steps: int) -> list[float]:
    """Time only the requested phase with CUDA events, excluding setup and warmup.

    Backward builds a fresh graph before each sample. autograd.grad avoids leaf
    gradient accumulation; graph construction and upstream-gradient allocation
    finish before the start event. The interval includes all operator launches,
    not just one generated device kernel. No CUDA graph capture is used.
    """
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    gradient = torch.randn_like(forward()) if phase == "backward" else None
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
    torch.use_deterministic_algorithms(mode == "det", warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1234)
    forward, inputs, op_id = make_case(
        torch, args.kernel_case, args.tokens, args.hidden_size, getattr(torch, args.dtype)
    )
    result = {
        "measurement": vars(args),
        "mode": mode,
        "op_id": op_id,
        "implementation": "torch.compile:" + args.kernel_case,
        "inputs": [
            {"shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype)}
            for value in inputs
        ],
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
        "seed": 1234,
        "samples_ms": measure(torch, forward, inputs, args.phase, args.warmup, args.steps),
    }
    path = Path(os.environ["DETERMINISM_PERF_LOG_DIR"]) / "kernel.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
