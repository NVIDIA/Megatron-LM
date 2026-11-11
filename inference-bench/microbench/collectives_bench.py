"""Our NVLS all-gather-v / reduce-scatter-v vs NCCL all-gather / reduce-scatter, under CUDA graphs.

The MoE dispatch collectives, called as the NVLS dispatcher calls them (token_dispatcher_inference):
  agv_nvls   multimem_all_gather_v into a symmetric buffer
  ag_nccl    torch.distributed.all_gather_into_tensor
  rsv_nvls   multimem_reduce_scatter_v out of a symmetric buffer
  rs_nccl    torch.distributed.reduce_scatter_tensor
Every rank contributes the same number of local tokens T (rank offset = rank * T, EP max = T), as
in a balanced step. The variable-count kernels take the offset and max as device tensors and a
static per-rank token maximum that sets their grid (the largest T swept, as the server's
worst case does).

Each op is captured REPS_PER_GRAPH times into one CUDA graph; the graph is replayed after warmup
and timed with CUDA events, so the time per op excludes launch overhead, as in the engine's graphed
steps. Each op is checked once against NCCL before it is timed. bf16 throughout.

Run one process per GPU (run_collectives_bench.sh does this under Slurm):

    python inference-bench/microbench/collectives_bench.py --hidden 1024 2048 4096 \\
        --tokens 1 2 4 8 16 32 64 128 256 512 1024 2048 --csv out.csv
"""

import argparse
import csv
import os
import statistics

import torch
import torch.distributed as dist

from megatron.core.inference.communication.torch_symm_triton import (
    multimem_all_gather_v,
    multimem_reduce_scatter_v,
)
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

REPS_PER_GRAPH = 10


def time_graph(fn, replays, warmup):
    """Mean microseconds per fn() call: fn captured REPS_PER_GRAPH times in one CUDA graph."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):  # eager warmup: Triton compile, NCCL setup
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(REPS_PER_GRAPH):
            fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    samples = []
    for _ in range(replays):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / REPS_PER_GRAPH)
    # The slowest rank sets the collective's time.
    t = torch.tensor([statistics.median(samples)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--tokens", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
                    help="local tokens per GPU")
    ap.add_argument("--replays", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    group = dist.group.WORLD
    max_tokens = max(args.tokens)
    dtype = torch.bfloat16

    # One symmetric buffer sized for the largest gathered tensor (world x max tokens x hidden).
    size_mb = world * max_tokens * max(args.hidden) * 2 // 2**20 + 16
    buf = SymmetricMemoryManager.get_buffer("bench", process_group=group, size_mb=size_mb)
    if buf.symm_mem_hdl is None:
        raise RuntimeError(f"symmetric memory init failed: {buf.init_failure_reason}")

    rows = []
    for hidden in args.hidden:
        for tokens in args.tokens:
            symm = buf.maybe_get_tensor([world * tokens, hidden], dtype=dtype)
            gathered, handle = symm["tensor"], symm["handle"]
            local = torch.randn(tokens, hidden, device="cuda", dtype=dtype)
            out_local = torch.empty(tokens, hidden, device="cuda", dtype=dtype)
            out_global = torch.empty(world * tokens, hidden, device="cuda", dtype=dtype)
            offset = torch.tensor([rank * tokens], device="cuda", dtype=torch.int32)
            ep_max = torch.tensor([tokens], device="cuda", dtype=torch.int32)

            ops = {
                "agv_nvls": lambda: multimem_all_gather_v(
                    gathered, local, handle, offset, ep_max, max_tokens),
                "ag_nccl": lambda: dist.all_gather_into_tensor(out_global, local, group=group),
                "rsv_nvls": lambda: multimem_reduce_scatter_v(
                    out_local, gathered, handle, offset, ep_max, max_tokens),
                "rs_nccl": lambda: dist.reduce_scatter_tensor(out_local, out_global, group=group),
            }

            # Check ours against NCCL once, eagerly.
            ops["agv_nvls"]()
            ops["ag_nccl"]()
            torch.cuda.synchronize()
            dist.barrier()
            assert torch.equal(gathered, out_global), "all-gather-v disagrees with NCCL"
            gathered.copy_(torch.randn_like(gathered))
            out_global.copy_(gathered)
            dist.barrier()
            ops["rsv_nvls"]()
            ours = out_local.clone()
            ops["rs_nccl"]()
            # bf16 sums of `world` terms round differently in the switch and in NCCL, so compare
            # both with an fp32 sum: ours must be no worse than twice NCCL's error (a real bug,
            # e.g. a wrong offset, puts whole rows off by O(1)).
            exact = torch.empty(tokens, hidden, device="cuda", dtype=torch.float32)
            dist.reduce_scatter_tensor(exact, out_global.float(), group=group)
            torch.cuda.synchronize()
            err_ours = (ours.float() - exact).abs().max().item()
            err_nccl = (out_local.float() - exact).abs().max().item()
            assert err_ours <= 2 * err_nccl + 1e-2, (
                f"reduce-scatter-v error {err_ours:.4g} vs NCCL {err_nccl:.4g} (fp32 reference)")

            nbytes = world * tokens * hidden * 2  # gathered tensor
            for name, fn in ops.items():
                us = time_graph(fn, args.replays, args.warmup)
                rows.append({"op": name, "gpus": world, "hidden": hidden, "local_tokens": tokens,
                             "gathered_mib": round(nbytes / 2**20, 3), "us": round(us, 2),
                             "busbw_gbs": round(nbytes * (world - 1) / world / (us * 1e-6) / 1e9, 1)})
            if rank == 0:
                r = {row["op"]: row["us"] for row in rows[-4:]}
                print(f"gpus {world:3d} hidden {hidden:5d} T {tokens:5d} | "
                      f"AG nvls {r['agv_nvls']:8.2f} nccl {r['ag_nccl']:8.2f} us "
                      f"({r['ag_nccl'] / r['agv_nvls']:.2f}x) | "
                      f"RS nvls {r['rsv_nvls']:8.2f} nccl {r['rs_nccl']:8.2f} us "
                      f"({r['rs_nccl'] / r['rsv_nvls']:.2f}x)", flush=True)

    if rank == 0 and args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"-> {args.csv}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
