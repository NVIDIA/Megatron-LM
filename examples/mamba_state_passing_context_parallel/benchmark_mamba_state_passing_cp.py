#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Benchmark the Mamba2 context-parallel paths against each other.

Measures the post-projection part of ``MambaMixer`` -- the causal convolution
and the SSD scan, which is where the CP paths differ -- for the existing
all-to-all path and for each state-passing load-balancing mode. The projection
layers are excluded on purpose: they are identical across paths and would dilute
the comparison.

Run from the repository root, one process per GPU:

    PYTHONPATH=. torchrun --standalone --nproc_per_node=4 \\
      examples/mamba_state_passing_context_parallel/benchmark_mamba_state_passing_cp.py \\
      --L 32768 --iters 20

Sweep several shapes in one run:

    PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \\
      examples/mamba_state_passing_context_parallel/benchmark_mamba_state_passing_cp.py \\
      --sequence-lengths 32768 131072 --batch-sizes 1 3

The world size is the CP size; no tensor-parallel group is created. ``--tp-size``
only divides the head and group counts so a TP-local shard shape can be measured
on a single rank per CP position.
"""

import argparse
import statistics
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict

import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

from megatron.core.ssm.mamba_context_parallel import MambaContextParallel
from megatron.core.ssm.ops.ssd_state_passing_cp import MambaStatePassingCPAdapter

CP_PATH_NAMES = ("a2a", "permute-p2p", "permute-a2a", "virtual")


@dataclass(frozen=True)
class MambaCoreShape:
    """TP-local Mamba mixer shape driving the measured kernels."""

    nheads: int = 64
    headdim: int = 64
    ngroups: int = 8
    d_state: int = 128
    d_conv: int = 4
    chunk_size: int = 128

    @property
    def d_inner(self) -> int:
        """Inner (gated) width of the mixer."""
        return self.nheads * self.headdim

    @property
    def projected_width(self) -> int:
        """Width of the ``in_proj`` output (z, x, B, C, dt)."""
        return 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads

    @property
    def conv_dim(self) -> int:
        """Number of channels the depthwise causal convolution operates on."""
        return self.d_inner + 2 * self.ngroups * self.d_state


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Per-iteration latency statistics and peak memory for one CP path."""

    mean_ms: float
    stdev_ms: float
    p50_ms: float
    peak_gib: float


@dataclass
class MambaCPInputs:
    """Rank-local activation and output gradient for one measured configuration."""

    projected_input: torch.Tensor
    grad_output: torch.Tensor


class MambaCPWorkload(nn.Module):
    """Minimal post-projection mixer satisfying the contract both CP paths expect.

    ``MambaContextParallel`` and ``MambaStatePassingCPAdapter`` read a specific
    set of attributes off the mixer. Reproducing just those keeps the benchmark
    independent of the surrounding model, spec, and process-group plumbing.
    """

    def __init__(self, shape: MambaCoreShape, device: torch.device, cp_group: dist.ProcessGroup):
        super().__init__()
        self.shape = shape
        dtype = torch.bfloat16
        generator = torch.Generator(device=device).manual_seed(4321)

        self.nheads_local_tp = shape.nheads
        self.d_inner_local_tp = shape.d_inner
        self.ngroups_local_tp = shape.ngroups
        self.d_state = shape.d_state
        self.d_conv = shape.d_conv
        self.headdim = shape.headdim
        self.chunk_size = shape.chunk_size
        self.activation = "silu"
        self.act = nn.SiLU()
        # RMSNorm and the output projection are outside the measured region.
        self.rmsnorm = False
        self.norm_before_gate = False
        self.D_has_hdim = False
        self.config = SimpleNamespace(mamba_state_passing_cp_load_balancing="permute_p2p")

        def randn(*sizes, dtype=dtype, scale=1.0):
            return torch.randn(*sizes, device=device, dtype=dtype, generator=generator) * scale

        self.conv1d_weight = nn.Parameter(randn(shape.conv_dim, 1, self.d_conv, scale=0.02))
        self.conv1d_bias = nn.Parameter(randn(shape.conv_dim, scale=0.02))
        self.dt_bias = nn.Parameter(randn(self.nheads_local_tp))
        self.A_log = nn.Parameter(randn(self.nheads_local_tp, dtype=torch.float32))
        self.D = nn.Parameter(randn(self.nheads_local_tp, dtype=torch.float32))

        self.cp = MambaContextParallel(
            cp_group=cp_group,
            d_inner_local_tp=self.d_inner_local_tp,
            nheads_local_tp=self.nheads_local_tp,
            ngroups_local_tp=self.ngroups_local_tp,
            d_state=self.d_state,
            conv1d_weight_cp1=self.conv1d_weight,
            conv1d_bias_cp1=self.conv1d_bias,
            conv1d_padding=self.d_conv - 1,
            dt_bias_cp1=self.dt_bias,
            A_log_cp1=self.A_log,
            D_cp1=self.D,
            D_has_hdim=self.D_has_hdim,
        )
        self.state_passing_cp_adapter = MambaStatePassingCPAdapter(self)


def parse_args():
    """Parse benchmark shape, sweep, and iteration-count arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", dest="sequence_length", type=int, default=32768)
    parser.add_argument("--batch", dest="batch_size", type=int, default=1)
    parser.add_argument("--sequence-lengths", type=int, nargs="+")
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument("--nheads", type=int, default=64)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--ngroups", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=128)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help=(
            "emulate a tensor-parallel local Mamba shard by dividing --nheads and "
            "--ngroups; no TP process group or TP communication is created"
        ),
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--paths", nargs="+", choices=CP_PATH_NAMES, default=CP_PATH_NAMES)
    return parser.parse_args()


def shape_from_args(args) -> MambaCoreShape:
    """Build the TP-local shape the benchmark measures."""
    assert args.tp_size > 0
    assert args.nheads % args.tp_size == 0, "global nheads must divide TP size"
    assert args.ngroups % args.tp_size == 0, "global ngroups must divide TP size"
    return MambaCoreShape(
        nheads=args.nheads // args.tp_size,
        headdim=args.headdim,
        ngroups=args.ngroups // args.tp_size,
        d_state=args.d_state,
        d_conv=args.d_conv,
        chunk_size=args.chunk,
    )


def validate_shape(shape: MambaCoreShape, sequence_length: int, cp_size: int) -> int:
    """Check the CP and SSD alignment constraints and return the local length."""
    assert shape.nheads % shape.ngroups == 0
    assert sequence_length % cp_size == 0, "global sequence length must divide CP size"
    local_length = sequence_length // cp_size
    assert local_length % 2 == 0, "load-balanced CP requires two local chunks"
    assert (
        local_length // 2
    ) % shape.chunk_size == 0, "each local load-balanced chunk must align with SSD chunk size"
    return local_length


def create_inputs(
    workload: MambaCPWorkload,
    sequence_length: int,
    batch_size: int,
    rank: int,
    device: torch.device,
    cp_size: int,
) -> MambaCPInputs:
    """Allocate the rank-local projected activation and output gradient."""
    local_length = validate_shape(workload.shape, sequence_length, cp_size)
    generator = torch.Generator(device=device).manual_seed(1234 + rank)
    projected_input = torch.randn(
        local_length,
        batch_size,
        workload.shape.projected_width,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    grad_output = torch.randn(
        local_length,
        batch_size,
        workload.d_inner_local_tp,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    return MambaCPInputs(projected_input=projected_input, grad_output=grad_output)


def build_cp_path_runners(
    workload: MambaCPWorkload, inputs: MambaCPInputs
) -> Dict[str, Callable[[], torch.Tensor]]:
    """Return one output-producing closure per measured CP path."""

    def a2a():
        projected = workload.cp.pre_conv_ssm(inputs.projected_input)
        projected = rearrange(projected, "l b d -> b l d").contiguous()
        output = mamba_split_conv1d_scan_combined(
            projected,
            rearrange(workload.cp.get_conv1d_weight(), "d 1 w -> d w"),
            workload.cp.get_conv1d_bias(),
            workload.cp.get_dt_bias().float(),
            -torch.exp(workload.cp.get_A_log().float()),
            D=workload.cp.get_D(),
            chunk_size=workload.chunk_size,
            activation=workload.activation,
            headdim=workload.headdim,
            ngroups=workload.cp.ngroups_local_tpcp,
            norm_before_gate=workload.norm_before_gate,
        )
        output = rearrange(output, "b l d -> l b d").contiguous()
        return workload.cp.post_conv_ssm(output)

    def state_passing(load_balancing):
        workload.config.mamba_state_passing_cp_load_balancing = load_balancing
        return workload.state_passing_cp_adapter.forward(inputs.projected_input)

    return {
        "a2a": a2a,
        "permute-p2p": lambda: state_passing("permute_p2p"),
        "permute-a2a": lambda: state_passing("permute_a2a"),
        "virtual": lambda: state_passing("virtual"),
    }


def benchmark_cp_path(
    name: str,
    runner: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
    forward_only: bool,
    workload: MambaCPWorkload,
    inputs: MambaCPInputs,
    group: dist.ProcessGroup,
    device: torch.device,
    rank: int,
) -> BenchmarkMetrics:
    """Time one CP path, reporting the per-iteration maximum across ranks.

    A CP path is only as fast as its slowest rank, so each iteration is reduced
    with MAX rather than averaged. Rank alignment happens on a barrier outside
    the timed interval so that waiting for stragglers is not counted twice.
    """
    times = []
    peaks = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    for iteration in range(warmup + iterations):
        inputs.projected_input.grad = None
        workload.zero_grad(set_to_none=True)
        dist.barrier(group=group)
        torch.cuda.reset_peak_memory_stats(device)
        start.record()
        output = runner()
        if not forward_only:
            torch.autograd.backward(output, inputs.grad_output)
        end.record()
        end.synchronize()

        metrics = torch.tensor(
            [start.elapsed_time(end), torch.cuda.max_memory_allocated(device) / 1024**3],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX, group=group)
        if iteration >= warmup:
            times.append(metrics[0].item())
            peaks.append(metrics[1].item())
        del output
    torch.cuda.synchronize(device)
    dist.barrier(group=group)

    result = BenchmarkMetrics(
        mean_ms=statistics.mean(times),
        stdev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        p50_ms=statistics.median(times),
        peak_gib=max(peaks),
    )
    if rank == 0:
        print(
            f"{name:>18}: {result.mean_ms:8.2f} ms +/- {result.stdev_ms:6.2f} "
            f"p50={result.p50_ms:8.2f}  peak={result.peak_gib:6.2f} GiB",
            flush=True,
        )
    return result


def run_configuration(args, workload, context, sequence_length, batch_size):
    """Benchmark every requested path for one (sequence length, batch size)."""
    rank, cp_size, device, group = context
    local_length = validate_shape(workload.shape, sequence_length, cp_size)
    inputs = create_inputs(workload, sequence_length, batch_size, rank, device, cp_size)
    runners = build_cp_path_runners(workload, inputs)

    if rank == 0:
        mode = "fwd" if args.forward_only else "fwd+bwd"
        print(
            f"Mamba CP paths ({mode}) L={sequence_length} batch={batch_size} "
            f"local_L={local_length} cp={cp_size} tp={args.tp_size}(local-shape-only) "
            f"d_inner={workload.d_inner_local_tp} nheads={workload.nheads_local_tp} "
            f"ngroups={workload.ngroups_local_tp}",
            flush=True,
        )

    results = {
        name: benchmark_cp_path(
            name,
            runners[name],
            warmup=args.warmup,
            iterations=args.iters,
            forward_only=args.forward_only,
            workload=workload,
            inputs=inputs,
            group=group,
            device=device,
            rank=rank,
        )
        for name in args.paths
    }
    if rank != 0:
        return

    if "a2a" in results:
        baseline = results["a2a"].p50_ms
        speedups = ", ".join(
            f"{name}={baseline / metrics.p50_ms:.3f}x"
            for name, metrics in results.items()
            if name != "a2a"
        )
        print(f"p50 speedup vs a2a: {speedups}", flush=True)

    for name, metrics in results.items():
        print(
            f"RESULT,cp={cp_size},L={sequence_length},batch={batch_size},"
            f"tp_size={args.tp_size},path={name},mean_ms={metrics.mean_ms:.6f},"
            f"stdev_ms={metrics.stdev_ms:.6f},p50_ms={metrics.p50_ms:.6f},"
            f"peak_gib={metrics.peak_gib:.6f}",
            flush=True,
        )
    print("Times are per-iteration max latency across ranks; lower is better.", flush=True)


def main():
    """Initialize one process per GPU and sweep the requested configurations."""
    args = parse_args()
    assert args.iters > 0 and args.warmup >= 0

    local_rank = dist.get_node_local_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    group = dist.group.WORLD
    context = (dist.get_rank(), dist.get_world_size(), device, group)

    workload = MambaCPWorkload(shape_from_args(args), device, group)
    for sequence_length in args.sequence_lengths or [args.sequence_length]:
        for batch_size in args.batch_sizes or [args.batch_size]:
            run_configuration(args, workload, context, sequence_length, batch_size)
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            dist.barrier(group=group)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
