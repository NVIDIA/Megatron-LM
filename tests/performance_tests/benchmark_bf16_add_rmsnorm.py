# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare fused BF16 add/RMSNorm with add + TE; emit JSON lines per GPU.

Run through torch.distributed.run. Timings include an identical input-reset
copy in every path. This measures an individual boundary, not a model.
"""

import argparse
import gc
import hashlib
import json
import os
import statistics
from importlib.metadata import version
from pathlib import Path

import torch
import torch.distributed as dist
import transformer_engine as te  # type: ignore[import-untyped]
import transformer_engine.pytorch as te_pytorch  # type: ignore[import-untyped]
import triton  # type: ignore[import-untyped]

from megatron.core.ops.normalization.bf16_add_rmsnorm import fused_bf16_add_rmsnorm


def timing(fn):
    """Amortize replay launch overhead over 64 boundaries inside each graph."""
    for _ in range(3):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(64):
            result = fn()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(7):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / (20 * 64))
    # Keep graph outputs alive throughout replay, including the allocated sum.
    del result
    return dict(
        median_us=statistics.median(times),
        min_us=min(times),
        max_us=max(times),
        boundaries_per_graph=64,
        replays_per_batch=20,
        batches=7,
    )


@torch.no_grad()
def boundary(rows, rank, emit):
    """Measure latency and PyTorch live/peak allocation for both storage modes."""
    torch.manual_seed(42 + rank)
    source = torch.randn(rows, 2688, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(source)
    x = source.clone()
    norm = te_pytorch.RMSNorm(2688, device="cuda", dtype=torch.bfloat16)
    norm.weight.copy_(torch.randn_like(norm.weight))
    for inplace in (False, True):
        for fused in (False, True):

            def run():
                # Both in-place implementations overwrite x; include the same
                # reset copy in every measured path for a fair comparison.
                x.copy_(source)
                if fused:
                    return fused_bf16_add_rmsnorm(x, residual, norm.weight, inplace=inplace)
                summed = x.add_(residual) if inplace else x + residual
                return norm(summed), summed

            for _ in range(3):
                run()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            result = run()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - before
            live = torch.cuda.memory_allocated() - before
            expected_sum = source + residual
            expected_y = norm(expected_sum)
            assert torch.equal(result[0].view(torch.int16), expected_y.view(torch.int16))
            assert torch.equal(result[1].view(torch.int16), expected_sum.view(torch.int16))
            del result, expected_sum, expected_y
            emit(
                "boundary",
                rows=rows,
                inplace=inplace,
                fused=fused,
                peak_extra_bytes=peak,
                live_extra_bytes=live,
                residual_payload_bytes=source.numel() * source.element_size(),
                **timing(run),
            )


@torch.no_grad()
def feedback_chain(rank, emit):
    """Check 52 boundaries with feedback through a synthetic branch function."""
    torch.manual_seed(100 + rank)
    h = torch.randn(32, 2688, device="cuda", dtype=torch.bfloat16)
    norm = te_pytorch.RMSNorm(2688, device="cuda", dtype=torch.bfloat16)
    norm.weight.copy_(torch.randn_like(norm.weight) * 0.1 + 1)
    expected_h, actual_h = h.clone(), h.clone()
    expected_y, actual_y = norm(h), norm(h)
    for _ in range(52):
        # This exercises feedback, not the attention/SSM/MoE computations.
        expected_branch = torch.tanh(expected_y.float()).bfloat16() * 0.125
        actual_branch = torch.tanh(actual_y.float()).bfloat16() * 0.125
        expected_h = expected_h + expected_branch
        expected_y = norm(expected_h)
        actual_y, actual_h = fused_bf16_add_rmsnorm(
            actual_branch, actual_h, norm.weight, inplace=True
        )
        assert torch.equal(actual_h.view(torch.int16), expected_h.view(torch.int16))
        assert torch.equal(actual_y.view(torch.int16), expected_y.view(torch.int16))
    emit("feedback_chain", boundaries=52, rows=32, mismatches=0)


def errors(actual, reference):
    """Measure arithmetic error in FP64, retaining small differences."""
    delta = actual.double() - reference.double()
    return dict(
        rmse=delta.square().mean().sqrt().item(),
        max_abs=delta.abs().max().item(),
        relative_l2=(delta.norm() / reference.double().norm().clamp_min(1e-30)).item(),
    )


@torch.no_grad()
def compare_approaches(hidden, rows, rank, emit):
    """Compare two rounding contracts and two standalone RMSNorm implementations.

    Every timed path resets the same two preallocated BF16 working tensors.
    FlashInfer fused add/RMSNorm owns and overwrites both; the other paths
    preserve the working skip input and allocate the normalized output.
    """
    from flashinfer.norm import fused_add_rmsnorm, rmsnorm

    torch.manual_seed(2026 + rank)
    source = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16) * 0.125
    skip_source = torch.randn_like(source)
    x, skip = source.clone(), skip_source.clone()
    norm = te_pytorch.RMSNorm(hidden, device="cuda", dtype=torch.bfloat16)
    norm.weight.copy_(1 + 0.1 * torch.randn_like(norm.weight))

    def run(method):
        x.copy_(source)
        skip.copy_(skip_source)
        if method == "te_bf16_add":
            summed = x.add_(skip)
            return norm(summed), summed
        if method == "training_fused":
            return fused_bf16_add_rmsnorm(x, skip, norm.weight, inplace=True)
        if method == "flashinfer_bf16_add":
            summed = x.add_(skip)
            return rmsnorm(summed, norm.weight, eps=norm.eps), summed
        fused_add_rmsnorm(x, skip, norm.weight, eps=norm.eps)
        return x, skip

    methods = ("te_bf16_add", "training_fused", "flashinfer_bf16_add", "flashinfer_fused_add")
    for method in methods:
        fn = lambda: run(method)
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - before
        live = torch.cuda.memory_allocated() - before
        del result
        emit(
            "comparison_latency",
            hidden=hidden,
            rows=rows,
            method=method,
            live_extra_bytes=live,
            peak_extra_bytes=peak,
            reset_copies=2,
            working_payload_bytes=2 * source.numel() * source.element_size(),
            **timing(fn),
        )

    for case in ("random", "cancellation", "rounding_ties", "large_offset"):
        if case == "cancellation":
            source.normal_()
            skip_source.copy_(-source)
            skip_source[:, ::2] *= 1.0078125
        elif case == "rounding_ties":
            source.fill_(1.0)
            source[:, 1::2] = 1.0078125
            skip_source.fill_(1.0 / 256)
        elif case == "large_offset":
            source.normal_().add_(1024)
            skip_source.normal_().mul_(0.01)
        summed = source + skip_source
        with torch.enable_grad():
            training = norm(summed.detach().requires_grad_()).detach()
        exact_sum = source.double() + skip_source.double()
        ideal = exact_sum * torch.rsqrt(exact_sum.square().mean(-1, keepdim=True) + norm.eps)
        ideal *= norm.weight.double()
        rounded = summed.double()
        rounded_ideal = rounded * torch.rsqrt(rounded.square().mean(-1, keepdim=True) + norm.eps)
        rounded_ideal *= norm.weight.double()
        for method in methods:
            output, next_residual = run(method)
            emit(
                "comparison_accuracy",
                hidden=hidden,
                rows=rows,
                method=method,
                case=case,
                elements=output.numel(),
                training_mismatches=(output.view(torch.int16) != training.view(torch.int16))
                .sum()
                .item(),
                residual_mismatches=(next_residual.view(torch.int16) != summed.view(torch.int16))
                .sum()
                .item(),
                versus_training=errors(output, training),
                versus_fp64_sum=errors(output, ideal),
                versus_fp64_rounded_sum=errors(output, rounded_ideal),
            )


def main():
    """Run independently on each GPU and save reproducible measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compare-approaches", action="store_true")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[2688, 4096, 8192])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 8, 32, 128, 2048])
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = args.output_dir / f"rank{dist.get_rank()}.jsonl"
    with log.open("w", encoding="utf-8") as stream:

        def emit(kind, **data):
            record = dict(kind=kind, rank=dist.get_rank(), **data)
            line = json.dumps(record)
            stream.write(line + "\n")
            stream.flush()
            print(line, flush=True)

        emit(
            "environment",
            torch=torch.__version__,
            te=te.__version__,
            triton=triton.__version__,
            gpu=torch.cuda.get_device_name(),
            cuda=torch.version.cuda,
            env={k: v for k, v in os.environ.items() if k.startswith("NVTE_")},
        )
        if args.compare_approaches:
            emit("comparison_environment", flashinfer=version("flashinfer-python"))
            for hidden in args.hidden_sizes:
                for rows in args.rows:
                    compare_approaches(hidden, rows, rank, emit)
                    dist.barrier()
        else:
            feedback_chain(rank, emit)
            for rows in (1, 32, 2120, 4096):
                boundary(rows, rank, emit)
                dist.barrier()
        source = Path(__file__).resolve().parents[2] / (
            "megatron/core/ops/normalization/bf16_add_rmsnorm.py"
        )
        emit("source", sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
