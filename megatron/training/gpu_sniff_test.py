# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPU performance sniff tests.

Five micro-benchmarks are run and metrics are gathered across all ranks:
  1. GEMMs           – standard shapes + optional FFN shapes, reports TFLOP/s/GPU.
  2. All-reduce      – over the global PG, reports bus bandwidth.
  3. Reduce-scatter  – over the TP PG, reports bus bandwidth.
  4. All-to-all      – over the EP PG, reports bus bandwidth.
  5. Send/recv       – pairwise within the DP PG, reports bus bandwidth.

For each metric, any rank whose value differs from the mean by more than one
standard deviation is flagged as an outlier.

Can be run in two ways:

  1. Integrated into training (called from the training loop):
       Controlled by --gpu-sniff-test-interval.

  2. Standalone (no Megatron machinery required):
       torchrun --nproc_per_node=NUM_GPUS megatron/training/gpu_sniff_test.py [OPTIONS]
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


WARMUP_ITERS = 5
BENCH_ITERS = 20

STANDARD_GEMM_SHAPES = [
    (8192, 8192, 8192),
    (4096, 4096, 16384),
]

MSG_BYTES_LARGE = 256 * 1024 * 1024  # 256 MiB.
MSG_BYTES_SMALL = 1 * 1024 * 1024    # 1 MiB.
MSG_SIZES = [MSG_BYTES_LARGE, MSG_BYTES_SMALL]

OUTLIER_MIN_DEVIATION_FRAC = 0.10  # Only flag if deviation from mean exceeds 10% of mean.


# ---------------------------------------------------------------------------
# Timing / reporting helpers (no Megatron dependency).
# ---------------------------------------------------------------------------

def _time_cuda_op(fn, warmup, iters):
    """Time *fn* using CUDA events, return average seconds per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / (iters * 1000.0)


def _gather_hostnames():
    """Gather hostnames from all ranks. Returns list of hostnames on rank 0, None elsewhere."""
    hostname = os.uname().nodename
    hostnames = [None] * dist.get_world_size()
    dist.all_gather_object(hostnames, hostname)
    return hostnames if dist.get_rank() == 0 else None


def _gather_and_check(name, local_value, hostnames=None):
    """Gather a scalar metric from all ranks, print report on rank 0.

    Args:
        name: metric name for logging.
        local_value: scalar value from this rank.
        hostnames: list of hostnames indexed by rank (from _gather_hostnames).

    Returns True if outliers were detected.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    val_tensor = torch.tensor([local_value], dtype=torch.float64, device="cuda")
    gathered = [torch.zeros(1, dtype=torch.float64, device="cuda") for _ in range(world_size)]
    dist.all_gather(gathered, val_tensor)

    has_outliers = False
    if rank == 0:
        all_vals = np.array([t.item() for t in gathered])
        # Filter out NaN (used by ranks that didn't participate, e.g., unpaired send/recv).
        participating_mask = ~np.isnan(all_vals)
        if not np.any(participating_mask):
            return False
        participating_indices = np.where(participating_mask)[0]
        vals = all_vals[participating_mask]

        median = np.median(vals)
        mean = np.mean(vals)
        mad = np.median(np.abs(vals - median))
        min_val, max_val = np.min(vals), np.max(vals)
        min_deviation = median * OUTLIER_MIN_DEVIATION_FRAC
        threshold = max(mad, min_deviation)
        outlier_mask = np.abs(vals - median) > threshold
        outlier_ranks = participating_indices[outlier_mask]
        has_outliers = len(outlier_ranks) > 0

        logger.info(f"  {name}: median={median:.2f}, mean={mean:.2f}, min={min_val:.2f}, max={max_val:.2f}")
        if has_outliers:
            for i in outlier_ranks:
                node = f" ({hostnames[i]})" if hostnames else ""
                logger.warning(f"    OUTLIER rank {i}{node}: {all_vals[i]:.2f} ({(all_vals[i] - median) / median:+.1%})")

    return has_outliers


# ---------------------------------------------------------------------------
# GEMM benchmark.
# ---------------------------------------------------------------------------

def bench_gemms(extra_shapes=None):
    """Run GEMM benchmarks.

    Args:
        extra_shapes: optional list of (M, N, K, label) tuples to benchmark
            in addition to the standard shapes.
    """
    results = []

    shapes = [(M, N, K, None) for M, N, K in STANDARD_GEMM_SHAPES]
    if extra_shapes:
        shapes.extend(extra_shapes)

    for M, N, K, label in shapes:
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        avg = _time_cuda_op(lambda A=A, B=B: torch.mm(A, B), WARMUP_ITERS, BENCH_ITERS)
        tflops = 2.0 * M * N * K / avg / 1e12
        name = f"GEMM throughput ({M}x{N}x{K}, bf16"
        if label:
            name += f", {label}"
        name += ") [TFLOP/s/GPU]"
        results.append((name, tflops))
        del A, B
    return results


# ---------------------------------------------------------------------------
# All-reduce benchmark.
# ---------------------------------------------------------------------------

def bench_all_reduce(group):
    """Run all-reduce benchmark over *group*.

    Args:
        group: a torch.distributed ProcessGroup (e.g., the world group).
            If None or size <= 1, returns [].
    """
    if group is None or group.size() <= 1:
        return []

    group_size = group.size()
    results = []
    for msg_bytes in MSG_SIZES:
        numel = msg_bytes // 2  # BF16.
        buf = torch.randn(numel, device="cuda", dtype=torch.bfloat16)

        def _run(buf=buf):
            dist.all_reduce(buf, group=group)

        avg = _time_cuda_op(_run, WARMUP_ITERS, BENCH_ITERS)
        nbytes = numel * 2
        busbw = 2 * nbytes * (group_size - 1) / group_size / avg / 1e9
        results.append((f"All-reduce busbw (global PG, size={group_size}, {nbytes / 1e6:.0f} MB) [GB/s]", busbw))
        del buf
    return results


# ---------------------------------------------------------------------------
# Reduce-scatter benchmark.
# ---------------------------------------------------------------------------

def bench_reduce_scatter(group):
    """Run reduce-scatter benchmark over *group*.

    Args:
        group: a torch.distributed ProcessGroup (e.g., a TP group).
            If None or size <= 1, returns [].
    """
    if group is None or group.size() <= 1:
        return []

    group_size = group.size()
    results = []
    for msg_bytes in MSG_SIZES:
        numel = msg_bytes // 2  # BF16.
        numel = (numel // group_size) * group_size
        sendbuf = torch.randn(numel, device="cuda", dtype=torch.bfloat16)
        recvbuf = torch.empty(numel // group_size, device="cuda", dtype=torch.bfloat16)

        def _run(sendbuf=sendbuf, recvbuf=recvbuf):
            dist.reduce_scatter_tensor(recvbuf, sendbuf, group=group)

        avg = _time_cuda_op(_run, WARMUP_ITERS, BENCH_ITERS)
        nbytes = numel * 2
        busbw = nbytes * (group_size - 1) / group_size / avg / 1e9
        results.append((f"Reduce-scatter busbw (TP PG, size={group_size}, {nbytes / 1e6:.0f} MB) [GB/s]", busbw))
        del sendbuf, recvbuf
    return results


# ---------------------------------------------------------------------------
# All-to-all benchmark.
# ---------------------------------------------------------------------------

def bench_all_to_all(group):
    """Run all-to-all benchmark over *group*.

    Args:
        group: a torch.distributed ProcessGroup (e.g., an EP group).
            If None or size <= 1, returns [].
    """
    if group is None or group.size() <= 1:
        return []

    group_size = group.size()
    results = []
    for msg_bytes in MSG_SIZES:
        numel = msg_bytes // 2  # BF16.
        numel = (numel // group_size) * group_size
        sendbuf = torch.randn(numel, device="cuda", dtype=torch.bfloat16)
        recvbuf = torch.empty_like(sendbuf)

        def _run(sendbuf=sendbuf, recvbuf=recvbuf):
            dist.all_to_all_single(recvbuf, sendbuf, group=group)

        avg = _time_cuda_op(_run, WARMUP_ITERS, BENCH_ITERS)
        nbytes = numel * 2
        busbw = nbytes * (group_size - 1) / group_size / avg / 1e9
        results.append((f"All-to-all busbw (EP PG, size={group_size}, {nbytes / 1e6:.0f} MB) [GB/s]", busbw))
        del sendbuf, recvbuf
    return results


# ---------------------------------------------------------------------------
# Send/recv benchmark.
# ---------------------------------------------------------------------------

def bench_sendrecv(group):
    """Pairwise send/recv within *group* at multiple strides.

    Tests up to 3 exponentially spaced strides (powers of 2) within the group.
    XOR pairing at each stride: stride 1 pairs 0<->1, stride 2 pairs 0<->2, etc.

    Args:
        group: a torch.distributed ProcessGroup (e.g., a DP group).
            If None or size <= 1, returns [].
    """
    if group is None or group.size() <= 1:
        return []

    group_size = group.size()
    my_rank_in_group = group.rank()
    global_ranks = dist.get_process_group_ranks(group)

    # Pick up to 3 exponentially spaced strides (powers of 2).
    max_stride = group_size // 2
    all_strides = []
    s = 1
    while s <= max_stride:
        all_strides.append(s)
        s *= 2
    if len(all_strides) <= 3:
        strides = all_strides
    else:
        strides = [all_strides[0], all_strides[len(all_strides) // 2], all_strides[-1]]

    results = []
    for msg_bytes in MSG_SIZES:
        numel = msg_bytes // 2
        sendbuf = torch.randn(numel, device="cuda", dtype=torch.bfloat16)
        recvbuf = torch.empty_like(sendbuf)

        for stride in strides:
            partner_rank_in_group = my_rank_in_group ^ stride
            have_partner = partner_rank_in_group < group_size
            partner_global = global_ranks[partner_rank_in_group] if have_partner else -1

            def _run(pg=partner_global, hp=have_partner, sb=sendbuf, rb=recvbuf):
                if not hp:
                    return
                ops = [
                    dist.P2POp(dist.isend, sb, pg, group=group),
                    dist.P2POp(dist.irecv, rb, pg, group=group),
                ]
                reqs = dist.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()

            if have_partner:
                avg = _time_cuda_op(_run, WARMUP_ITERS, BENCH_ITERS)
                busbw = msg_bytes / avg / 1e9
            else:
                busbw = float('nan')

            rank0_partner = global_ranks[stride] if stride < group_size else -1
            results.append((
                f"Send/recv busbw (DP PG, {msg_bytes / 1e6:.0f} MB, e.g., rank {global_ranks[0]} <-> rank {rank0_partner}) [GB/s]",
                busbw,
            ))

        del sendbuf, recvbuf
    return results


# ---------------------------------------------------------------------------
# Orchestration: run all benchmarks and report
# ---------------------------------------------------------------------------

def run_sniff_tests(
    ep_group, dp_group, ar_group=None, tp_group=None,
    extra_gemm_shapes=None, tag="",
):
    """Run all sniff tests and report.

    Args:
        ep_group: ProcessGroup for all-to-all (or None to skip).
        dp_group: ProcessGroup for send/recv (or None to skip).
        ar_group: ProcessGroup for all-reduce (or None to skip).
        tp_group: ProcessGroup for reduce-scatter (or None to skip).
        extra_gemm_shapes: optional list of (M, N, K, label) for additional GEMMs.
        tag: string included in the header (e.g. iteration number).
    """
    rank = dist.get_rank()
    hostnames = _gather_hostnames()
    if rank == 0:
        logger.info(f"{'=' * 60}")
        logger.info(f"  GPU sniff test{f' -- {tag}' if tag else ''}")
        logger.info(f"{'=' * 60}")

    any_outliers = False
    for name, value in bench_gemms(extra_shapes=extra_gemm_shapes):
        if _gather_and_check(name, value, hostnames):
            any_outliers = True

    for name, value in bench_all_reduce(ar_group):
        if _gather_and_check(name, value, hostnames):
            any_outliers = True

    for name, value in bench_reduce_scatter(tp_group):
        if _gather_and_check(name, value, hostnames):
            any_outliers = True

    for name, value in bench_all_to_all(ep_group):
        if _gather_and_check(name, value, hostnames):
            any_outliers = True

    for name, value in bench_sendrecv(dp_group):
        if _gather_and_check(name, value, hostnames):
            any_outliers = True

    if rank == 0:
        status = "OUTLIERS DETECTED" if any_outliers else "ALL RANKS OK"
        log_fn = logger.warning if any_outliers else logger.info
        log_fn(f"  Result: {status}")
        logger.info(f"{'=' * 60}")

    dist.barrier()


# ---------------------------------------------------------------------------
# Entry point for the Megatron training loop
# ---------------------------------------------------------------------------

def _get_ffn_gemm_shapes():
    """Derive up-proj and down-proj GEMM shapes from current training args."""
    from megatron.training.global_vars import get_args

    args = get_args()
    h = args.hidden_size
    ffn_h = args.ffn_hidden_size
    if h is None or ffn_h is None:
        return []

    tp = args.tensor_model_parallel_size
    gated = getattr(args, 'gated_linear_unit', False) or getattr(args, 'swiglu', False)

    tokens = args.micro_batch_size * args.seq_length
    glu_mult = 2 if gated else 1

    # fc1 / up-proj: (tokens, ffn_hidden_size * glu_mult / TP) with K = hidden_size
    fc1_M = tokens
    fc1_N = ffn_h * glu_mult // tp
    fc1_K = h

    # fc2 / down-proj: (tokens, hidden_size) with K = ffn_hidden_size / TP
    fc2_M = tokens
    fc2_N = h
    fc2_K = ffn_h // tp

    return [
        (fc1_M, fc1_N, fc1_K, "up-proj/fc1"),
        (fc2_M, fc2_N, fc2_K, "down-proj/fc2"),
    ]


def run_gpu_sniff_test(tag="", pg_collection=None):
    """Called from the Megatron training loop.

    Args:
        tag: string included in the header (e.g. "iteration 100").
        pg_collection: a ProcessGroupCollection. If None, one is built from mpu.
    """
    if pg_collection is None:
        from megatron.core.process_groups_config import ProcessGroupCollection

        pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['ep', 'dp', 'tp'],
        )

    ffn_shapes = _get_ffn_gemm_shapes()

    run_sniff_tests(
        ep_group=pg_collection.ep,
        dp_group=pg_collection.dp,
        ar_group=dist.group.WORLD,
        tp_group=pg_collection.tp,
        extra_gemm_shapes=ffn_shapes,
        tag=tag,
    )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def _create_ep_groups(ep_size):
    """Create EP-style groups: consecutive blocks of *ep_size* ranks."""
    world = dist.get_world_size()
    rank = dist.get_rank()
    assert world % ep_size == 0, f"world_size ({world}) not divisible by ep_size ({ep_size})"
    my_group = None
    for start in range(0, world, ep_size):
        ranks = list(range(start, start + ep_size))
        g = dist.new_group(ranks)
        if rank in ranks:
            my_group = g
    return my_group


def _create_dp_groups(ep_size):
    """Create DP-style groups: ranks sharing the same offset within EP blocks."""
    world = dist.get_world_size()
    rank = dist.get_rank()
    assert world % ep_size == 0
    my_group = None
    for offset in range(ep_size):
        ranks = list(range(offset, world, ep_size))
        g = dist.new_group(ranks)
        if rank in ranks:
            my_group = g
    return my_group


def _create_tp_groups(tp_size):
    """Create TP-style groups: consecutive blocks of *tp_size* ranks."""
    world = dist.get_world_size()
    rank = dist.get_rank()
    assert world % tp_size == 0, f"world_size ({world}) not divisible by tp_size ({tp_size})"
    my_group = None
    for start in range(0, world, tp_size):
        ranks = list(range(start, start + tp_size))
        g = dist.new_group(ranks)
        if rank in ranks:
            my_group = g
    return my_group


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--ep-size", type=int, default=None,
        help="EP group size. Default: LOCAL_WORLD_SIZE (one node).",
    )
    p.add_argument(
        "--tp-size", type=int, default=None,
        help="TP group size for reduce-scatter. Default: LOCAL_WORLD_SIZE (one node).",
    )
    p.add_argument(
        "--gemm-shapes", type=str, nargs="*", default=None,
        help="Extra GEMM shapes as MxNxK strings, e.g. 2048x8192x4096.",
    )
    p.add_argument("--skip-gemm", action="store_true")
    p.add_argument("--skip-allreduce", action="store_true")
    p.add_argument("--skip-reducescatter", action="store_true")
    p.add_argument("--skip-alltoall", action="store_true")
    p.add_argument("--skip-sendrecv", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    ep_size = args.ep_size or int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    tp_size = args.tp_size or int(os.environ.get("LOCAL_WORLD_SIZE", world_size))

    if rank == 0:
        logger.info(f"GPU sniff test (standalone): world_size={world_size}, ep_size={ep_size}, tp_size={tp_size}")
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")

    ep_group = _create_ep_groups(ep_size) if not args.skip_alltoall else None
    dp_group = _create_dp_groups(ep_size) if not args.skip_sendrecv else None
    tp_group = _create_tp_groups(tp_size) if not args.skip_reducescatter else None

    ar_group = dist.group.WORLD if not args.skip_allreduce else None

    extra_shapes = None
    if args.gemm_shapes:
        extra_shapes = []
        for s in args.gemm_shapes:
            parts = s.split("x")
            assert len(parts) == 3, f"Expected MxNxK, got {s}"
            M, N, K = int(parts[0]), int(parts[1]), int(parts[2])
            extra_shapes.append((M, N, K, "custom"))

    run_sniff_tests(
        ep_group=ep_group if not args.skip_alltoall else None,
        dp_group=dp_group if not args.skip_sendrecv else None,
        ar_group=ar_group if not args.skip_allreduce else None,
        tp_group=tp_group if not args.skip_reducescatter else None,
        extra_gemm_shapes=extra_shapes if not args.skip_gemm else None,
        tag="standalone",
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
