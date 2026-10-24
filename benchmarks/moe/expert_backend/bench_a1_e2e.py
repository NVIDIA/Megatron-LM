#!/usr/bin/env python3
"""A1 end-to-end: a full MoE layer (router + expert GEMMs + combine) forward and backward.

The kernel-level harness (bench_a1_backend.py) measures the expert GEMMs alone. This one
measures what a training step would actually see: routing, expert computation, gating, and
the backward pass, with the two expert backends swapped underneath.

  dense   : per-expert masked loop, the shape of work Megatron does without an expert kernel
  scatter : ScatterMoE's fused scatter2scatter kernels (scattermoe.mlp.MLP)

Both consume identical activations and weights and must agree on the output, so a faster
number cannot come from computing something different.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn


def build(H, I, E, k, tokens, seed, device, dtype, scattermoe_root):
    if scattermoe_root not in sys.path:
        sys.path.insert(0, scattermoe_root)
    from scattermoe.mlp import MLP

    g = torch.Generator(device="cpu").manual_seed(seed)
    mlp = MLP(H, I, E, k, activation=nn.SiLU()).to(device).to(dtype)
    with torch.no_grad():
        for p in mlp.parameters():
            p.copy_((torch.randn(p.shape, generator=g) / H ** 0.5).to(dtype))
    x = torch.randn(tokens, H, generator=g).to(dtype).to(device)
    w_router = (torch.randn(E, H, generator=g) / H ** 0.5).to(dtype).to(device)
    return mlp, x, w_router


def router(x, w_router, k):
    logits = x @ w_router.T
    probs = logits.softmax(-1)
    gates, idx = probs.topk(k, dim=-1)
    return gates / gates.sum(-1, keepdim=True), idx


def forward_dense(mlp, x, gates, idx):
    """Per-expert masked loop over the layer's own weights (verified layouts)."""
    T, H = x.shape
    W1 = mlp.experts.weight            # [E, I, H]
    W2 = mlp.output_experts.weight     # [E, H, I]
    out = torch.zeros(T, H, device=x.device, dtype=x.dtype)
    for e in range(W1.shape[0]):
        mask = idx == e
        if not mask.any():
            continue
        rows = mask.any(dim=1).nonzero(as_tuple=True)[0]
        he = torch.nn.functional.silu(x[rows] @ W1[e].T)
        ye = he @ W2[e].T
        g = (gates * mask.to(gates.dtype)).sum(dim=1)[rows].unsqueeze(-1)
        out[rows] = out[rows] + ye * g
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("dense", "scatter"), required=True)
    ap.add_argument("--scattermoe-root", required=True)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--intermediate", type=int, default=2048)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--jsonl", required=True)
    a = ap.parse_args()

    dev = torch.device("cuda")
    dtype = getattr(torch, a.dtype)
    mlp, x0, w_router = build(a.hidden, a.intermediate, a.experts, a.topk,
                              a.tokens, a.seed, dev, dtype, a.scattermoe_root)

    def step():
        """One full training step: route -> experts -> gate -> loss -> backward."""
        x = x0.detach().clone().requires_grad_(True)
        wr = w_router.detach().clone().requires_grad_(True)
        gates, idx = router(x, wr, a.topk)
        if a.variant == "dense":
            y = forward_dense(mlp, x, gates, idx)
        else:
            y = mlp(x, gates.to(dtype), idx)
        loss = y.float().pow(2).mean()
        loss.backward()
        return y.detach(), loss.detach()

    y, loss = step()
    with torch.no_grad():
        gates, idx = router(x0, w_router, a.topk)
        ref = forward_dense(mlp, x0, gates, idx)
    rel = (y.float() - ref.float()).abs().max().item() / max(1e-6, ref.float().abs().max().item())
    agree = rel < 2e-2

    for _ in range(a.warmup):
        step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(a.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    peak = torch.cuda.max_memory_allocated()

    rec = dict(variant=a.variant, scope="moe_layer_fwd_bwd", hidden=a.hidden,
               intermediate=a.intermediate, experts=a.experts, topk=a.topk,
               tokens=a.tokens, seed=a.seed, dtype=a.dtype, iters=a.iters,
               median_ms=statistics.median(ts), min_ms=min(ts),
               mean_ms=statistics.fmean(ts),
               p95_ms=sorted(ts)[min(len(ts) - 1, int(0.95 * len(ts)))],
               peak_allocated_bytes=peak, final_loss=float(loss),
               agree_dense_ref=agree, rel_err=rel,
               device=torch.cuda.get_device_name(0),
               cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
    Path(a.jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(a.jsonl, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[a1e2e] {a.variant:7s} median={rec['median_ms']:.3f}ms "
          f"peak={peak / 2 ** 20:.1f}MiB loss={float(loss):.4f} rel_err={rel:.2e} agree={agree}")


if __name__ == "__main__":
    main()
