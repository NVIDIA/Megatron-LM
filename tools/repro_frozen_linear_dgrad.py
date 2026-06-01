# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Minimal repro for high-rank frozen-linear dgrad matmul dispatch.

This script compares the old high-rank backward expression:

    grad_output.matmul(weight)

against the flattened 2D GEMM form used by LinearWithFrozenWeight:

    grad_output.reshape(-1, hidden).matmul(weight).reshape(...)

It does not run training. It initializes one frozen linear operation, runs a
forward + backward pass, and times the old and flattened backward paths.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


class FrozenLinear(torch.autograd.Function):
    """Frozen-weight linear with selectable dgrad implementation."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, weight: torch.Tensor, mode: str) -> torch.Tensor:
        ctx.save_for_backward(weight)
        ctx.mode = mode
        return x.matmul(weight.t())

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (weight,) = ctx.saved_tensors
        if ctx.mode == "flat" and grad_output.dim() > 2:
            grad_input = grad_output.reshape(-1, grad_output.size(-1)).matmul(weight)
            grad_input = grad_input.reshape(*grad_output.shape[:-1], weight.size(1))
        else:
            grad_input = grad_output.matmul(weight)
        return grad_input, None, None


@dataclass(frozen=True)
class Problem:
    leading_0: int
    leading_1: int
    hidden: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.leading_0, self.leading_1, self.hidden)

    @classmethod
    def parse(cls, value: str) -> "Problem":
        fields = value.lower().replace(",", "x").split("x")
        if len(fields) != 3:
            raise argparse.ArgumentTypeError("problem must be formatted as AxBxH, e.g. 384x8x4096")
        return cls(*(int(field) for field in fields))

    def __str__(self) -> str:
        return f"{self.leading_0}x{self.leading_1}x{self.hidden}"


def make_grad_output(problem: Problem, layout: str, dtype: torch.dtype) -> torch.Tensor:
    """Create a dgrad output tensor with the requested logical shape and layout."""
    a, b, h = problem.shape
    if layout == "contiguous":
        return torch.randn((a, b, h), device="cuda", dtype=dtype)
    if layout == "transpose-leading":
        return torch.randn((b, a, h), device="cuda", dtype=dtype).transpose(0, 1)
    if layout == "expanded-middle":
        return torch.randn((a, 1, h), device="cuda", dtype=dtype).expand(a, b, h)
    raise ValueError(f"unknown layout: {layout}")


def run_backward(
    mode: str,
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
) -> None:
    x.grad = None
    y = FrozenLinear.apply(x, weight, mode)
    y.backward(grad_output)


def time_ms(fn: Callable[[], None], warmup: int, reps: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    values = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end))
    return values


def profile_ops(fn: Callable[[], None]) -> list[str]:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        fn()
    ops = []
    for event in prof.key_averages(group_by_input_shape=True):
        if event.key in {
            "aten::matmul",
            "aten::mm",
            "aten::bmm",
            "aten::reshape",
            "aten::clone",
            "aten::copy_",
        }:
            ops.append(f"{event.key} {event.input_shapes}")
    return ops


def benchmark(
    problem: Problem,
    layout: str,
    dtype: torch.dtype,
    warmup: int,
    reps: int,
    profile: bool,
) -> dict[str, Any]:
    x = torch.randn(problem.shape, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn((problem.hidden, problem.hidden), device="cuda", dtype=dtype)
    grad_output = make_grad_output(problem, layout, dtype)

    run_backward("old", x, weight, grad_output)
    old_grad = x.grad.detach().clone()
    run_backward("flat", x, weight, grad_output)
    flat_grad = x.grad.detach().clone()
    max_abs_diff = (old_grad.float() - flat_grad.float()).abs().max().item()

    old_times = time_ms(lambda: run_backward("old", x, weight, grad_output), warmup, reps)
    flat_times = time_ms(lambda: run_backward("flat", x, weight, grad_output), warmup, reps)
    old_median = statistics.median(old_times)
    flat_median = statistics.median(flat_times)

    row: dict[str, Any] = {
        "problem": str(problem),
        "layout": layout,
        "grad_output_shape": tuple(grad_output.shape),
        "grad_output_stride": tuple(grad_output.stride()),
        "old_median_ms": old_median,
        "flat_median_ms": flat_median,
        "speedup": old_median / flat_median,
        "max_abs_diff": max_abs_diff,
    }
    if profile:
        row["old_ops"] = profile_ops(lambda: run_backward("old", x, weight, grad_output))
        row["flat_ops"] = profile_ops(lambda: run_backward("flat", x, weight, grad_output))
    return row


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problems",
        nargs="+",
        type=Problem.parse,
        default=[Problem.parse("384x8x4096")],
    )
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=["contiguous", "transpose-leading", "expanded-middle"],
        default=["contiguous", "transpose-leading", "expanded-middle"],
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSONL rows instead of a text table",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = dtype_from_name(args.dtype)
    metadata = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "dtype": args.dtype,
        "warmup": args.warmup,
        "reps": args.reps,
    }
    if args.json:
        print(json.dumps({"metadata": metadata}, sort_keys=True))
    else:
        print(
            f"device={metadata['device']} torch={metadata['torch']} cuda={metadata['cuda']} "
            f"dtype={metadata['dtype']} warmup={args.warmup} reps={args.reps}"
        )
        print(
            f"{'problem':>14}  {'layout':>18}  {'stride':>24}  "
            f"{'old ms':>10}  {'flat ms':>10}  {'speedup':>9}"
        )

    for problem in args.problems:
        for layout in args.layouts:
            row = benchmark(problem, layout, dtype, args.warmup, args.reps, args.profile)
            if args.json:
                print(json.dumps(row, sort_keys=True))
            else:
                print(
                    f"{row['problem']:>14}  {row['layout']:>18}  "
                    f"{str(row['grad_output_stride']):>24}  "
                    f"{row['old_median_ms']:10.3f}  {row['flat_median_ms']:10.3f}  "
                    f"{row['speedup']:9.2f}x"
                )
                if args.profile:
                    print("  old ops:")
                    for op in row["old_ops"]:
                        print(f"    {op}")
                    print("  flat ops:")
                    for op in row["flat_ops"]:
                        print(f"    {op}")


if __name__ == "__main__":
    main()
