#!/usr/bin/env python3
"""Standalone correctness + timing harness for the fused MoE decode kernel.

Reference = the production `vllm_fused_moe` (4-kernel path). Candidate =
`fused_moe_decode` (single fused kernel). Shapes mirror Qwen3-30B-A3B decode on
EP4: H=2048, moe_ffn=768, 128 experts / 4 EP = 32 local, top-8.

Usage (on a Blackwell GPU inside the container):
  python dev/moe_fused/harness.py --valid 256 --iters 50
"""
import argparse
import torch

from megatron.core.inference.moe.fused_moe import ActivationType
from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe
from megatron.core.inference.moe.fused_moe_decode import fused_moe_decode


def build_inputs(valid, max_tokens, H, Nf, num_local_experts, num_global_experts, topk,
                 local_expert_start, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    hidden = (torch.randn(max_tokens, H, generator=g, device=dev, dtype=torch.float32) * 0.1
              ).to(torch.bfloat16)
    fc1 = (torch.randn(num_local_experts, 2 * Nf, H, generator=g, device=dev,
                       dtype=torch.float32) * 0.02).to(torch.bfloat16)
    fc2 = (torch.randn(num_local_experts, H, Nf, generator=g, device=dev,
                       dtype=torch.float32) * 0.02).to(torch.bfloat16)
    # routing: each valid token picks `topk` distinct global experts.
    routing = torch.full((max_tokens, topk), -1, device=dev, dtype=torch.int64)
    for t in range(valid):
        perm = torch.randperm(num_global_experts, generator=g, device=dev)[:topk]
        routing[t] = perm
    probs = torch.zeros(max_tokens, topk, device=dev, dtype=torch.float32)
    probs[:valid] = torch.softmax(
        torch.randn(valid, topk, generator=g, device=dev, dtype=torch.float32), dim=-1)
    valid_t = torch.tensor([valid], device=dev, dtype=torch.int32)
    return hidden, probs, fc1, fc2, routing, valid_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="buffer rows (default = valid)")
    ap.add_argument("--iters", type=int, default=0, help="timing iters (0 = correctness only)")
    ap.add_argument("--block-m", type=int, default=16)
    ap.add_argument("--block-n2", type=int, default=128)
    ap.add_argument("--block-k1", type=int, default=64)
    ap.add_argument("--num-warps", type=int, default=8)
    ap.add_argument("--num-stages", type=int, default=3)
    args = ap.parse_args()

    H, Nf, topk = 2048, 768, 8
    num_local_experts, num_global_experts = 32, 128
    local_expert_start = 0
    max_tokens = args.max_tokens or args.valid

    hidden, probs, fc1, fc2, routing, valid_t = build_inputs(
        args.valid, max_tokens, H, Nf, num_local_experts, num_global_experts, topk,
        local_expert_start)

    common = dict(activation_type=ActivationType.SWIGLU, num_local_experts=num_local_experts,
                  local_expert_start=local_expert_start, valid_tokens=valid_t,
                  routing_map=routing, num_tokens_hint=args.valid)

    ref = vllm_fused_moe(hidden, probs, fc1, fc2, **common)
    cand = fused_moe_decode(hidden, probs, fc1, fc2, block_m=args.block_m,
                            block_n2=args.block_n2, block_k1=args.block_k1,
                            num_warps=args.num_warps, num_stages=args.num_stages, **common)
    torch.cuda.synchronize()

    r = ref[:args.valid].float()
    c = cand[:args.valid].float()
    abs_diff = (r - c).abs()
    denom = r.abs().clamp_min(1e-4)
    rel = (abs_diff / denom)
    print(f"ref norm={r.norm().item():.4f} cand norm={c.norm().item():.4f}")
    print(f"max_abs_diff={abs_diff.max().item():.6e}  mean_abs={abs_diff.mean().item():.6e}")
    print(f"max_rel_diff={rel.max().item():.6e}  mean_rel={rel.mean().item():.6e}")
    ok = torch.allclose(r, c, rtol=2e-2, atol=2e-2)
    print(f"ALLCLOSE(rtol=2e-2,atol=2e-2): {ok}")

    if args.iters:
        def bench(fn):
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            s = torch.cuda.Event(True); e = torch.cuda.Event(True)
            s.record()
            for _ in range(args.iters):
                fn()
            e.record(); torch.cuda.synchronize()
            return s.elapsed_time(e) / args.iters
        t_ref = bench(lambda: vllm_fused_moe(hidden, probs, fc1, fc2, **common))
        t_cand = bench(lambda: fused_moe_decode(hidden, probs, fc1, fc2, block_m=args.block_m,
                                                block_n2=args.block_n2, block_k1=args.block_k1,
                                                num_warps=args.num_warps,
                                                num_stages=args.num_stages, **common))
        print(f"\nreference (4-kernel vllm) : {t_ref*1e3:.2f} us/call")
        print(f"candidate (fused)         : {t_cand*1e3:.2f} us/call")
        print(f"speedup                   : {t_ref/t_cand:.3f}x")


if __name__ == "__main__":
    main()
