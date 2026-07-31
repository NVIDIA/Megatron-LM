#!/usr/bin/env python3
"""Correctness + timing A/B for FC1+SwiGLU epilogue fusion in vllm_fused_moe.

reference  = vllm_fused_moe(..., fuse_fc1_activation=False)  # FC1(2N)+bounded_silu_mul+FC2+sum
candidate  = vllm_fused_moe(..., fuse_fc1_activation=True)   # fused FC1+SwiGLU + FC2 + sum

Shapes mirror Qwen3-30B-A3B decode on EP4: H=2048, moe_ffn=768, 128/4=32 local
experts, top-8. Usage (inside the container):
  python dev/moe_fused/harness_fc1.py --valid 256 --iters 100
"""
import argparse
import torch

from megatron.core.inference.moe.fused_moe import ActivationType
from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe


def build_inputs(valid, max_tokens, H, Nf, num_local_experts, num_global_experts, topk, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    hidden = (torch.randn(max_tokens, H, generator=g, device=dev, dtype=torch.float32) * 0.1
              ).to(torch.bfloat16)
    fc1 = (torch.randn(num_local_experts, 2 * Nf, H, generator=g, device=dev,
                       dtype=torch.float32) * 0.02).to(torch.bfloat16)
    fc2 = (torch.randn(num_local_experts, H, Nf, generator=g, device=dev,
                       dtype=torch.float32) * 0.02).to(torch.bfloat16)
    routing = torch.full((max_tokens, topk), -1, device=dev, dtype=torch.int64)
    for t in range(valid):
        routing[t] = torch.randperm(num_global_experts, generator=g, device=dev)[:topk]
    probs = torch.zeros(max_tokens, topk, device=dev, dtype=torch.float32)
    probs[:valid] = torch.softmax(
        torch.randn(valid, topk, generator=g, device=dev, dtype=torch.float32), dim=-1)
    valid_t = torch.tensor([valid], device=dev, dtype=torch.int32)
    return hidden, probs, fc1, fc2, routing, valid_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--iters", type=int, default=0)
    args = ap.parse_args()

    H, Nf, topk = 2048, 768, 8
    num_local_experts, num_global_experts = 32, 128
    max_tokens = args.max_tokens or args.valid

    hidden, probs, fc1, fc2, routing, valid_t = build_inputs(
        args.valid, max_tokens, H, Nf, num_local_experts, num_global_experts, topk)

    common = dict(activation_type=ActivationType.SWIGLU, num_local_experts=num_local_experts,
                  local_expert_start=0, valid_tokens=valid_t, routing_map=routing,
                  num_tokens_hint=args.valid)

    ref = vllm_fused_moe(hidden, probs, fc1, fc2, fuse_fc1_activation=False, **common)
    cand = vllm_fused_moe(hidden, probs, fc1, fc2, fuse_fc1_activation=True, **common)
    torch.cuda.synchronize()

    r = ref[:args.valid].float()
    c = cand[:args.valid].float()
    abs_diff = (r - c).abs()
    rel = abs_diff / r.abs().clamp_min(1e-4)
    print(f"ref norm={r.norm().item():.4f} cand norm={c.norm().item():.4f}")
    print(f"max_abs_diff={abs_diff.max().item():.6e}  mean_abs={abs_diff.mean().item():.6e}")
    print(f"max_rel_diff={rel.max().item():.6e}  mean_rel={rel.mean().item():.6e}")
    print(f"ALLCLOSE(rtol=2e-2,atol=2e-2): {torch.allclose(r, c, rtol=2e-2, atol=2e-2)}")

    if args.iters:
        def bench(fuse):
            fn = lambda: vllm_fused_moe(hidden, probs, fc1, fc2, fuse_fc1_activation=fuse, **common)
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            s = torch.cuda.Event(True); e = torch.cuda.Event(True)
            s.record()
            for _ in range(args.iters):
                fn()
            e.record(); torch.cuda.synchronize()
            return s.elapsed_time(e) / args.iters
        t_ref = bench(False)
        t_cand = bench(True)
        print(f"\nreference (FC1+silu_mul+FC2+sum) : {t_ref*1e3:.2f} us/call")
        print(f"candidate (fused FC1-SwiGLU)     : {t_cand*1e3:.2f} us/call")
        print(f"speedup                          : {t_ref/t_cand:.3f}x")


if __name__ == "__main__":
    main()
