#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Benchmark Mamba-2 computation kernels comparing (B, L) vs (2*B, L/2).

This script isolates and measures the pure computational kernel throughput
of Mamba-2 (SSD Scan, Causal Conv1d, and Combined Conv+SSD) between:
  - Baseline contiguous shape: (Batch = B, SeqLen = L)
  - Virtual CP shape:          (Batch = 2*B, SeqLen = L/2)

Both configurations process the EXACT same total token count (B * L tokens)
and require identical theoretical FLOPs.

Usage:
    # Run standalone on a single GPU:
    python examples/mamba_state_passing_context_parallel/benchmark_mamba2_virtual_shape.py

    # Custom sweep:
    python examples/mamba_state_passing_context_parallel/benchmark_mamba2_virtual_shape.py \\
        --batch-sizes 1 2 4 8 \\
        --seq-lens 2048 4096 8192 16384 \\
        --iters 30 --warmup 10
"""

import argparse
import statistics
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

try:
    import causal_conv1d_cuda
    from causal_conv1d import causal_conv1d_fn
    from mamba_ssm.ops.triton.ssd_combined import (
        mamba_chunk_scan_combined,
        mamba_split_conv1d_scan_combined,
    )
    HAVE_MAMBA_KERNELS = True
except ImportError:
    HAVE_MAMBA_KERNELS = False


@dataclass(frozen=True)
class MambaModelShape:
    """Mamba-2 model configuration."""
    hidden_size: int = 2688
    nheads: int = 64
    headdim: int = 64
    ngroups: int = 8
    d_state: int = 128
    d_conv: int = 4
    chunk_size: int = 128

    @property
    def d_inner(self) -> int:
        return self.nheads * self.head_dim if hasattr(self, "head_dim") else self.nheads * self.headdim

    @property
    def conv_dim(self) -> int:
        return self.d_inner + 2 * self.ngroups * self.d_state

    @property
    def projected_width(self) -> int:
        return 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads


@dataclass
class TimingResult:
    fwd_mean_ms: float
    fwd_p50_ms: float
    bwd_mean_ms: float
    bwd_p50_ms: float
    total_mean_ms: float
    peak_mem_mb: float


def benchmark_cuda_fn(
    fwd_fn,
    bwd_fn=None,
    warmup: int = 10,
    iters: int = 30,
) -> TimingResult:
    """Accurately benchmark a forward (and optional backward) function with CUDA Events."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Warmup
    for _ in range(warmup):
        out = fwd_fn()
        if bwd_fn is not None:
            bwd_fn(out)

    torch.cuda.synchronize()

    # Measure Forward
    fwd_times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(iters):
        start_event.record()
        out = fwd_fn()
        end_event.record()
        torch.cuda.synchronize()
        fwd_times.append(start_event.elapsed_time(end_event))

    # Measure Backward
    bwd_times = []
    if bwd_fn is not None:
        for _ in range(iters):
            out = fwd_fn()
            start_event.record()
            bwd_fn(out)
            end_event.record()
            torch.cuda.synchronize()
            bwd_times.append(start_event.elapsed_time(end_event))
    else:
        bwd_times = [0.0] * iters

    peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

    fwd_mean = statistics.mean(fwd_times)
    fwd_p50 = statistics.median(fwd_times)
    bwd_mean = statistics.mean(bwd_times) if bwd_fn else 0.0
    bwd_p50 = statistics.median(bwd_times) if bwd_fn else 0.0

    return TimingResult(
        fwd_mean_ms=fwd_mean,
        fwd_p50_ms=fwd_p50,
        bwd_mean_ms=bwd_mean,
        bwd_p50_ms=bwd_p50,
        total_mean_ms=fwd_mean + bwd_mean,
        peak_mem_mb=peak_mem,
    )


def run_ssd_scan_benchmark(
    shape: MambaModelShape,
    batch: int,
    seqlen: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = 10,
    iters: int = 30,
) -> TimingResult:
    """Benchmark mamba_chunk_scan_combined kernel."""
    x = torch.randn(batch, seqlen, shape.nheads, shape.headdim, device=device, dtype=dtype, requires_grad=True)
    dt = torch.randn(batch, seqlen, shape.nheads, device=device, dtype=dtype, requires_grad=True)
    A = -torch.exp(torch.rand(shape.nheads, device=device, dtype=torch.float32)).requires_grad_(True)
    B = torch.randn(batch, seqlen, shape.ngroups, shape.d_state, device=device, dtype=dtype, requires_grad=True)
    C = torch.randn(batch, seqlen, shape.ngroups, shape.d_state, device=device, dtype=dtype, requires_grad=True)
    D = torch.randn(shape.nheads, shape.headdim, device=device, dtype=torch.float32, requires_grad=True)
    z = torch.randn(batch, seqlen, shape.nheads, shape.headdim, device=device, dtype=dtype, requires_grad=True)
    dt_bias = torch.rand(shape.nheads, device=device, dtype=torch.float32, requires_grad=True)

    grad_out = torch.randn_like(x)

    def fwd():
        return mamba_chunk_scan_combined(
            x, dt, A, B, C,
            chunk_size=shape.chunk_size,
            D=D, z=z, dt_bias=dt_bias,
            dt_softplus=True,
        )

    def bwd(out):
        x.grad = None
        dt.grad = None
        A.grad = None
        B.grad = None
        C.grad = None
        D.grad = None
        z.grad = None
        dt_bias.grad = None
        out.backward(grad_out, retain_graph=True)

    return benchmark_cuda_fn(fwd, bwd, warmup=warmup, iters=iters)


def run_conv1d_benchmark(
    shape: MambaModelShape,
    batch: int,
    seqlen: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = 10,
    iters: int = 30,
) -> TimingResult:
    """Benchmark causal_conv1d_fn kernel."""
    x = torch.randn(batch, shape.conv_dim, seqlen, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(shape.conv_dim, shape.d_conv, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(shape.conv_dim, device=device, dtype=dtype, requires_grad=True)

    grad_out = torch.randn_like(x)

    def fwd():
        return causal_conv1d_fn(x, weight, bias, activation="silu")

    def bwd(out):
        x.grad = None
        weight.grad = None
        bias.grad = None
        out.backward(grad_out, retain_graph=True)

    return benchmark_cuda_fn(fwd, bwd, warmup=warmup, iters=iters)


def run_combined_mamba_benchmark(
    shape: MambaModelShape,
    batch: int,
    seqlen: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = 10,
    iters: int = 30,
) -> TimingResult:
    """Benchmark mamba_split_conv1d_scan_combined (Fused Conv + SSD scan)."""
    zxbcdt = torch.randn(
        batch, seqlen, shape.projected_width, device=device, dtype=dtype, requires_grad=True
    )
    conv1d_weight = torch.randn(shape.conv_dim, shape.d_conv, device=device, dtype=dtype, requires_grad=True)
    conv1d_bias = torch.randn(shape.conv_dim, device=device, dtype=dtype, requires_grad=True)
    dt_bias = torch.rand(shape.nheads, device=device, dtype=torch.float32, requires_grad=True)
    A = -torch.exp(torch.rand(shape.nheads, device=device, dtype=torch.float32, requires_grad=True))
    D = torch.randn(shape.nheads, shape.headdim, device=device, dtype=torch.float32, requires_grad=True)

    grad_out = torch.randn(batch, seqlen, shape.d_inner, device=device, dtype=dtype)

    def fwd():
        return mamba_split_conv1d_scan_combined(
            zxbcdt,
            conv1d_weight,
            conv1d_bias,
            dt_bias,
            A,
            D=D,
            chunk_size=shape.chunk_size,
            activation="silu",
            headdim=shape.headdim,
            ngroups=shape.ngroups,
        )

    def bwd(out):
        zxbcdt.grad = None
        conv1d_weight.grad = None
        conv1d_bias.grad = None
        dt_bias.grad = None
        A.grad = None
        D.grad = None
        out.backward(grad_out, retain_graph=True)

    return benchmark_cuda_fn(fwd, bwd, warmup=warmup, iters=iters)


def format_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    data_lines = [
        "| " + " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + data_lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Mamba2 (B, L) vs (2B, L/2) Kernel Performance")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8], help="Base batch sizes (B)")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 4096, 8192, 16384], help="Base sequence lengths (L)")
    parser.add_argument("--hidden-size", type=int, default=2688, help="Hidden size")
    parser.add_argument("--nheads", type=int, default=64, help="Number of heads")
    parser.add_argument("--headdim", type=int, default=64, help="Head dimension")
    parser.add_argument("--ngroups", type=int, default=8, help="Number of groups")
    parser.add_argument("--d-state", type=int, default=128, help="State dimension")
    parser.add_argument("--chunk-size", type=int, default=128, help="SSD chunk size")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=30, help="Benchmark iterations")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires an NVIDIA GPU.")
        sys.exit(1)

    if not HAVE_MAMBA_KERNELS:
        print("ERROR: mamba_ssm or causal_conv1d is not installed. Please run inside the container.")
        sys.exit(1)

    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)

    shape = MambaModelShape(
        hidden_size=args.hidden_size,
        nheads=args.nheads,
        headdim=args.headdim,
        ngroups=args.ngroups,
        d_state=args.d_state,
        d_conv=4,
        chunk_size=args.chunk_size,
    )

    print("=" * 80)
    print(f" Mamba-2 Kernel Benchmark: (B, L) vs (2*B, L/2)")
    print(f" Device: {device_name}")
    print(f" Config: Heads={shape.nheads}, HeadDim={shape.headdim}, Groups={shape.ngroups}, DState={shape.d_state}, ChunkSize={shape.chunk_size}")
    print("=" * 80)

    benchmarks = [
        ("Fused Conv1d + SSD Scan", run_combined_mamba_benchmark),
        ("SSD Chunk Scan Only", run_ssd_scan_benchmark),
        ("Causal Conv1d Only", run_conv1d_benchmark),
    ]

    for bench_name, bench_fn in benchmarks:
        print(f"\n### Benchmark: {bench_name}")
        headers = [
            "Base (B, L)",
            "Virtual (2B, L/2)",
            "Total Tokens",
            "Base Fwd (ms)",
            "Virt Fwd (ms)",
            "Fwd Speedup",
            "Base Total (ms)",
            "Virt Total (ms)",
            "Total Speedup",
        ]
        rows = []

        for b in args.batch_sizes:
            for l in args.seq_lens:
                if (l // 2) % shape.chunk_size != 0:
                    continue  # L/2 must be divisible by chunk_size

                # 1. Base shape: (B, L)
                res_base = bench_fn(shape, b, l, device, warmup=args.warmup, iters=args.iters)

                # 2. Virtual shape: (2*B, L/2)
                res_virt = bench_fn(shape, 2 * b, l // 2, device, warmup=args.warmup, iters=args.iters)

                fwd_speedup = res_base.fwd_mean_ms / res_virt.fwd_mean_ms
                total_speedup = res_base.total_mean_ms / res_virt.total_mean_ms

                rows.append([
                    f"({b}, {l})",
                    f"({2 * b}, {l // 2})",
                    f"{b * l:,}",
                    f"{res_base.fwd_mean_ms:.3f}",
                    f"{res_virt.fwd_mean_ms:.3f}",
                    f"{fwd_speedup:.2f}x" if fwd_speedup >= 1.0 else f"{fwd_speedup:.2f}x (slower)",
                    f"{res_base.total_mean_ms:.3f}",
                    f"{res_virt.total_mean_ms:.3f}",
                    f"{total_speedup:.2f}x" if total_speedup >= 1.0 else f"{total_speedup:.2f}x (slower)",
                ])

        print(format_markdown_table(headers, rows))


if __name__ == "__main__":
    main()

