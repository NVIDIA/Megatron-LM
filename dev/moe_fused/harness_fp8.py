#!/usr/bin/env python3
"""Correctness + timing A/B for weight-only fp8 expert weights in vllm_fused_moe.

reference  = vllm_fused_moe(...)                       # bf16 weights
candidate  = vllm_fused_moe(..., fc1_fp8=, fc2_fp8=)   # e4m3 weights, per-channel scale

The kernel is memory bound on the expert weights, so halving their bytes is the
only lever left on it; this harness answers both questions that gates the e2e
run — how much time it actually buys, and how far the output moves.

Shapes mirror Qwen3-30B-A3B decode on EP4: H=2048, moe_ffn=768, 128/4=32 local
experts, top-8. Usage (inside the container):
  python dev/moe_fused/harness_fp8.py --valid 256 --iters 200
"""
import argparse

import torch

from megatron.core.inference.moe.fp8_experts import quantize_expert_weights
from megatron.core.inference.moe.fused_moe import ActivationType
from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

from harness_fc1 import build_inputs


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
        args.valid, max_tokens, H, Nf, num_local_experts, num_global_experts, topk
    )
    fc1_fp8 = quantize_expert_weights(fc1)
    fc2_fp8 = quantize_expert_weights(fc2)

    bytes_bf16 = fc1.numel() * 2 + fc2.numel() * 2
    bytes_fp8 = fc1.numel() + fc2.numel() + fc1_fp8.scale.numel() * 4 + fc2_fp8.scale.numel() * 4
    print(f"expert weight bytes/layer: bf16 {bytes_bf16/1e6:.1f} MB -> fp8 {bytes_fp8/1e6:.1f} MB")

    common = dict(
        activation_type=ActivationType.SWIGLU,
        num_local_experts=num_local_experts,
        local_expert_start=0,
        valid_tokens=valid_t,
        routing_map=routing,
        num_tokens_hint=args.valid,
        fuse_fc1_activation=True,
    )

    ref = vllm_fused_moe(hidden, probs, fc1, fc2, **common)
    cand = vllm_fused_moe(hidden, probs, fc1, fc2, fc1_fp8=fc1_fp8, fc2_fp8=fc2_fp8, **common)
    torch.cuda.synchronize()

    r = ref[: args.valid].float()
    c = cand[: args.valid].float()
    abs_diff = (r - c).abs()
    rel = abs_diff / r.abs().clamp_min(1e-4)
    print(f"ref norm={r.norm().item():.4f} cand norm={c.norm().item():.4f}")
    print(f"max_abs_diff={abs_diff.max().item():.6e}  mean_abs={abs_diff.mean().item():.6e}")
    print(f"max_rel_diff={rel.max().item():.6e}  mean_rel={rel.mean().item():.6e}")
    # cosine similarity is the honest metric for a quantization: per-element rel
    # error is dominated by near-zero outputs that carry no information.
    cos = torch.nn.functional.cosine_similarity(r.flatten(), c.flatten(), dim=0)
    print(f"cosine_similarity={cos.item():.6f}")

    if args.iters:

        def bench(**kw):
            fn = lambda: vllm_fused_moe(hidden, probs, fc1, fc2, **kw, **common)
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            s = torch.cuda.Event(True)
            e = torch.cuda.Event(True)
            s.record()
            for _ in range(args.iters):
                fn()
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) / args.iters

        t_ref = bench()
        t_cand = bench(fc1_fp8=fc1_fp8, fc2_fp8=fc2_fp8)
        print(f"\nreference (bf16 weights) : {t_ref*1e3:.2f} us/call")
        print(f"candidate (fp8 weights)  : {t_cand*1e3:.2f} us/call")
        print(f"speedup                  : {t_ref/t_cand:.3f}x")


if __name__ == "__main__":
    main()
