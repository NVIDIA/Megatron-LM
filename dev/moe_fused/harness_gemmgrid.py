#!/usr/bin/env python3
"""Per-GEMM tile sweep for the decode expert GEMMs, bf16 vs weight-only fp8.

`harness_fp8` found that halving the expert-weight bytes (302 -> 151 MB/layer)
bought nothing: 0.995x at 256 tokens. If the kernel were bandwidth bound that is
impossible, so either it is not bandwidth bound or the fp8 load pattern is
request-bound rather than byte-bound — an fp8 tile at BLOCK_SIZE_K=64 moves 64
bytes per row where bf16 moves 128, so the same number of requests carry half
the payload and nothing improves.

This sweeps each GEMM separately over the tile grid for both dtypes. If fp8 wins
at a larger BLOCK_SIZE_K the request-bound reading is right and the win is
recoverable; if no fp8 config beats the best bf16 config, the GEMM is not
bandwidth bound at this shape and fp8 expert weights are dead as a lever.

Usage (inside the container):
  python dev/moe_fused/harness_gemmgrid.py --valid 256 --iters 100
"""
import argparse
import importlib
import itertools

import torch

from megatron.core.inference.moe.fp8_experts import quantize_activations, quantize_expert_weights

vfm = importlib.import_module("megatron.core.inference.moe.vllm_fused_moe")
assert hasattr(vfm, "_invoke_fused_moe_kernel"), "vfm is not the module"

H = 2048
NF = 768
TOPK = 8
NUM_GLOBAL_EXPERTS = 128
NUM_LOCAL_EXPERTS = 32


def build(valid, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    hidden = (torch.randn(valid, H, generator=g, device=dev) * 0.1).to(torch.bfloat16)
    fc1 = (torch.randn(NUM_LOCAL_EXPERTS, 2 * NF, H, generator=g, device=dev) * 0.02).to(
        torch.bfloat16
    )
    fc2 = (torch.randn(NUM_LOCAL_EXPERTS, H, NF, generator=g, device=dev) * 0.02).to(torch.bfloat16)
    routing = torch.full((valid, TOPK), -1, device=dev, dtype=torch.int64)
    for t in range(valid):
        routing[t] = torch.randperm(NUM_GLOBAL_EXPERTS, generator=g, device=dev)[:TOPK]
    probs = torch.softmax(
        torch.randn(valid, TOPK, generator=g, device=dev, dtype=torch.float32), dim=-1
    )
    valid_t = torch.tensor([valid], device=dev, dtype=torch.int32)
    return hidden, probs, fc1, fc2, routing, valid_t


def time_gemm(fn, iters):
    """Device time for one launch, measured under graph replay.

    A plain launch loop reports max(host, device), and the host side of one
    Triton launch is tens of microseconds — the same order as these kernels. That
    is how a first pass concluded fp8 weights changed nothing: every arm was
    pinned at ~153 us of Python, with the GEMM invisible underneath. Capture the
    launch in a graph and replay it so the number is the kernel.
    """
    for _ in range(10):  # let Triton JIT and autotune settle outside capture
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / iters  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", type=int, default=256)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    hidden, probs, fc1, fc2, routing, valid_t = build(args.valid)
    fc1_fp8 = quantize_expert_weights(fc1)
    fc2_fp8 = quantize_expert_weights(fc2)
    probs_flat = probs.reshape(-1).contiguous()

    num_valid = args.valid * TOPK
    inter1 = torch.empty(num_valid, NF, dtype=torch.bfloat16, device="cuda")
    inter2 = torch.empty(num_valid, H, dtype=torch.bfloat16, device="cuda")

    # w8a8 inputs. Quantized outside the timed region: this measures the ceiling
    # on the w8a8 GEMM, and the two quantize kernels it would need in production
    # (~2 us each on these shapes) have to be subtracted from any win.
    hidden_q, hidden_s = quantize_activations(hidden)
    inter1.normal_(0, 0.1)
    inter1_q, inter1_s = quantize_activations(inter1)

    # The indirection table depends on BLOCK_SIZE_M, so build one per M-tile.
    tables = {}
    for block_m in (16, 32, 64):
        tables[block_m] = vfm._moe_align_block_size_single(
            routing, block_m, NUM_LOCAL_EXPERTS, 0, valid_t
        )

    def run(pass_name, block_m, block_n, block_k, warps, stages, mode):
        cfg = dict(
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            GROUP_SIZE_M=1,
            num_warps=warps,
            num_stages=stages,
        )
        sorted_ids, expert_ids, num_post = tables[block_m]
        em_hint = args.valid * TOPK + block_m * NUM_LOCAL_EXPERTS
        num_pid_m = -(-em_hint // block_m)
        if pass_name == "fc1":
            grid = num_pid_m * -(-NF // block_n)
            act, act_scale = (hidden, None) if mode != "w8a8" else (hidden_q, hidden_s)
            weight = fc1 if mode == "bf16" else fc1_fp8.weight
            scale = None if mode == "bf16" else fc1_fp8.scale
            args_ = (act, weight, inter1)
            kw = dict(top_k=TOPK, fuse_swiglu=True)
        else:
            grid = num_pid_m * -(-H // block_n)
            act, act_scale = (inter1, None) if mode != "w8a8" else (inter1_q, inter1_s)
            weight = fc2 if mode == "bf16" else fc2_fp8.weight
            scale = None if mode == "bf16" else fc2_fp8.scale
            args_ = (act, weight, inter2)
            kw = dict(top_k=1, fuse_swiglu=False)
        return lambda: vfm._invoke_fused_moe_kernel(
            *args_,
            probs_flat,
            sorted_ids,
            expert_ids,
            num_post,
            mul_routed_weight=False,
            config=cfg,
            grid_size=grid,
            b_scale=scale,
            a_scale=act_scale,
            **kw,
        )

    grids = {
        "fc1": list(itertools.product((16, 32), (32, 64, 128), (64, 128, 256), ((4, 3), (8, 4)))),
        "fc2": list(itertools.product((16, 32), (128, 256), (64, 128, 256), ((4, 3), (8, 4)))),
    }
    modes = ("bf16", "w8a16", "w8a8")

    for pass_name in ("fc1", "fc2"):
        print(f"\n===== {pass_name.upper()} (valid={args.valid}) =====")
        head = " ".join(f"{m:>9}" for m in modes)
        print(f"{'M':>4} {'N':>4} {'K':>4} {'w':>2} {'s':>2} {head}  w8a8/bf16")
        best = {}
        for block_m, block_n, block_k, (warps, stages) in grids[pass_name]:
            row = []
            for mode in modes:
                try:
                    t = time_gemm(
                        run(pass_name, block_m, block_n, block_k, warps, stages, mode), args.iters
                    )
                except Exception as exc:  # a tile can be rejected by the compiler
                    t = float("nan")
                    print(
                        f"  skip M{block_m} N{block_n} K{block_k} {mode}: "
                        f"{type(exc).__name__} {str(exc)[:60]}"
                    )
                row.append(t)
                if t == t and (mode not in best or t < best[mode][0]):
                    best[mode] = (t, (block_m, block_n, block_k, warps, stages))
            ratio = row[2] / row[0] if row[0] == row[0] and row[0] else float("nan")
            cells = " ".join(f"{t:>9.2f}" for t in row)
            print(f"{block_m:>4} {block_n:>4} {block_k:>4} {warps:>2} {stages:>2} {cells} {ratio:>10.3f}")
        for mode in modes:
            if mode in best:
                t, cfg = best[mode]
                print(f"  BEST {mode:>5}: {t:.2f} us at M{cfg[0]} N{cfg[1]} K{cfg[2]} w{cfg[3]} s{cfg[4]}")


if __name__ == "__main__":
    main()
