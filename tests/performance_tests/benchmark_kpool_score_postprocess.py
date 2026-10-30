# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Measure isolated detached KPool postprocessing; no training-step speedup is inferred.

Run from the repository root with ``python -m tests.performance_tests.benchmark_kpool_score_postprocess``.
Compilation/first-call wall time is reported separately from warm CUDA-event medians.
The torch backend mutates its score buffer, so every measured invocation is preceded
by the same input reset. Resets and numerical checks are outside the event interval.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from megatron.core.ops.attention.dsa.backends import select_kpool_score_postprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=257)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--keys", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--no-relu", action="store_true")
    args = parser.parse_args()
    if min(args.queries, args.batch, args.keys, args.warmup, args.repeats) <= 0:
        parser.error("shape, warmup and repeat values must be positive")
    if not torch.cuda.is_available():
        parser.error("requires CUDA")
    torch.cuda.set_device(0)
    shape = (args.queries, args.batch, 32, args.keys)
    payload_bytes = args.queries * args.batch * 32 * args.keys * 4
    if payload_bytes * 3 > torch.cuda.mem_get_info()[0] // 2:
        parser.error("benchmark requires three score buffers below half of free device memory")
    torch.manual_seed(41)
    original = torch.randn(shape, device="cuda")
    scores = original.clone()
    weights = torch.randn(shape[:3], device="cuda")
    use_relu = not args.no_relu
    operations = {name: select_kpool_score_postprocess(name, 32) for name in ("torch", "triton")}
    torch.cuda.synchronize()
    first_call, outputs = {}, {}
    for name, operation in operations.items():
        scores.copy_(original)
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs[name] = operation(scores, weights, use_relu)
        torch.cuda.synchronize()
        first_call[name] = (time.perf_counter() - start) * 1000
    torch.testing.assert_close(outputs["triton"], outputs["torch"], rtol=0, atol=0)
    del outputs
    times = {name: [] for name in operations}
    for repeat in range(args.warmup + args.repeats):
        names = ("torch", "triton") if repeat % 2 == 0 else ("triton", "torch")
        for name in names:
            scores.copy_(original)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = operations[name](scores, weights, use_relu)
            end.record()
            end.synchronize()
            if repeat >= args.warmup:
                times[name].append(start.elapsed_time(end))
            del result
    module = (
        Path(__file__).resolve().parents[2]
        / "megatron/core/ops/attention/dsa/kpool_score_kernels.py"
    )
    medians = {name: statistics.median(values) for name, values in times.items()}
    print(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                shape=shape,
                payload_bytes=payload_bytes,
                use_relu=use_relu,
                kernel_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
                first_call_including_compile_ms=first_call,
                warm_samples_ms=times,
                warm_median_ms=medians,
                warm_speedup=medians["torch"] / medians["triton"],
                exact_reference_match=True,
                limitations=[
                    "Isolated kernel, no preceding GEMM, no PG or training interference",
                    "Fresh allocator and timer state; no end-to-end claim",
                    "First-call compilation is excluded from warm medians",
                ],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
