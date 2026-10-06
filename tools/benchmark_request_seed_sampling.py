# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare request-local sampling implementations, including CPU launch overhead.

Run from the repository root with PYTHONPATH=. and an environment with CUDA:
    python tools/benchmark_request_seed_sampling.py --baseline 0001daeb

This is a sampler microbenchmark, not a model-server throughput measurement.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from functools import partial
from types import SimpleNamespace

import torch
import triton

from megatron.core.inference.sampling.torch_sampling import TorchSampling


def emit_result(result):
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()


def profile_noise_calls(call):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        call()
        torch.cuda.synchronize()
    return {
        "torch_exponential_calls": sum(
            event.count for event in profile.key_averages() if event.key == "aten::exponential_"
        ),
        "batched_noise_kernels": sum(
            event.name.startswith("_request_seed_noise_kernel")
            and event.device_type == torch.autograd.DeviceType.CUDA
            for event in profile.events()
        ),
    }


def measure(call, iterations):
    for _ in range(10):
        call()
    torch.cuda.synchronize()
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="0001daeb")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 16, 64, 256, 1024])
    parser.add_argument("--vocab", type=int, default=131072)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--filters", nargs="+", choices=["none", "top_k", "top_p"], default=["none"]
    )
    args = parser.parse_args()
    source = subprocess.check_output(
        ["git", "show", f"{args.baseline}:megatron/core/inference/sampling/torch_sampling.py"],
        text=True,
    )
    namespace = {}
    exec(compile(source, "<baseline-torch-sampling>", "exec"), namespace)
    baseline = namespace["TorchSampling"]
    emit_result(
        {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "triton": triton.__version__,
            **vars(args),
        }
    )
    torch.manual_seed(123)
    profiles = []
    with torch.inference_mode():
        for batch in args.batches:
            logits = torch.randn(batch, args.vocab, device="cuda")
            for filtering in args.filters:
                k, p = {"none": (0, 0.0), "top_k": (50, 0.0), "top_p": (0, 0.9)}[filtering]
                for mode in ("unseeded", "seeded", "mixed"):
                    seeds = torch.arange(batch, dtype=torch.int64) + 100
                    if mode == "unseeded":
                        seeds.fill_(-1)
                    elif mode == "mixed":
                        seeds[1::2] = -1
                    ctx = SimpleNamespace(
                        total_request_count=batch,
                        paused_request_count=0,
                        config=SimpleNamespace(num_speculative_tokens=0),
                        active_request_metadata={
                            "seed": seeds,
                            "temperature": torch.ones(batch),
                            "top_k": torch.full((batch,), k, dtype=torch.int32),
                            "top_p": torch.full((batch,), p),
                        },
                        get_active_sequence_lengths=lambda: torch.full((batch,), 128),
                    )
                    result = {"batch": batch, "filter": filtering, "mode": mode}
                    for label, implementation in (
                        ("baseline_ms", baseline),
                        ("current_ms", TorchSampling),
                    ):
                        sampler = implementation(
                            torch.Generator(device="cuda").manual_seed(11), args.vocab
                        )

                        sample = partial(
                            sampler.sample_kernel,
                            logits,
                            batch,
                            ctx,
                            no_top_k=k == 0,
                            no_top_p=p == 0,
                        )

                        result[label] = measure(sample, args.iterations)
                        if args.profile:
                            profiles.append((batch, filtering, mode, label, sample))
                    result["speedup"] = result["baseline_ms"] / result["current_ms"]
                    emit_result(result)
        # Initialize CUPTI only after every timing measurement; profiler startup
        # can change dispatch overhead for subsequent "unprofiled" calls.
        for batch, filtering, mode, label, sample in profiles:
            emit_result(
                {
                    "kind": "profile",
                    "batch": batch,
                    "filter": filtering,
                    "mode": mode,
                    "implementation": label,
                    **profile_noise_calls(sample),
                }
            )


if __name__ == "__main__":
    main()
