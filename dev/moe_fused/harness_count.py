#!/usr/bin/env python3
"""Correctness + A/B timing harness for the MoE local-token-count kernel.

Compares the reference per-pair-atomic count kernel
(`_count_local_tokens_kernel_persistent`) against the new tl.histogram variant
(`_count_local_tokens_kernel_histogram`) on Qwen3-30B-A3B decode routing shapes.

- Correctness: tokens_per_expert must match EXACTLY (integer counts).
- Timing: CUDA-event median over many iters, warmup first.

Run in-session (single GPU is enough):
    python dev/moe_fused/harness_count.py
"""
import torch

from megatron.core.inference.moe.permute import compute_local_tokens_per_expert

torch.manual_seed(0)
dev = "cuda"

# Qwen3-30B-A3B: 128 experts, top-8, EP4 -> 32 local experts. Decode BS256.
NUM_EXPERTS = 128
TOPK = 8
NUM_LOCAL = 32
LOCAL_START = 0  # rank 0
MAX_TOKENS = 4096  # buffer sized for prefill; decode uses valid_tokens
VALID = 256  # decode batch


def make_routing(valid, max_tokens, topk, num_experts):
    rm = torch.full((max_tokens, topk), -1, dtype=torch.int32, device=dev)
    # each valid token picks topk distinct experts
    for t in range(valid):
        rm[t] = torch.randperm(num_experts, device=dev)[:topk].to(torch.int32)
    return rm


routing_map = make_routing(VALID, MAX_TOKENS, TOPK, NUM_EXPERTS)
valid_tokens = torch.tensor([VALID], dtype=torch.int32, device=dev)


def run(use_hist):
    return compute_local_tokens_per_expert(
        routing_map, LOCAL_START, NUM_LOCAL, valid_tokens,
        persistent=True, use_histogram=use_hist,
    )


ref = run(False)
cand = run(True)
torch.cuda.synchronize()

exact = torch.equal(ref, cand)
print(f"counts sum ref={int(ref.sum())} cand={int(cand.sum())} (expected {VALID*TOPK} pairs, "
      f"minus non-local)")
print(f"EXACT MATCH: {exact}")
if not exact:
    diff = (ref - cand).abs()
    print("max abs diff:", int(diff.max()), "nonzero bins:", int((diff > 0).sum()))
    print("ref :", ref.tolist())
    print("cand:", cand.tolist())


def bench(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us


t_ref = bench(lambda: run(False))
t_cand = bench(lambda: run(True))
print(f"\nreference per-pair-atomic : {t_ref:8.2f} us")
print(f"histogram variant         : {t_cand:8.2f} us")
print(f"speedup                   : {t_ref / t_cand:6.3f}x")
