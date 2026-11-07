#!/usr/bin/env python3
"""A1 decision harness: ScatterMoE's Triton scatter2scatter vs the dense per-expert path.

The question A1 must answer before any Megatron integration is whether the Triton expert
backend is actually faster here -- not whether it runs. Both variants receive the SAME
dispatcher output (a gathered [assignments, H] activation plus per-assignment expert ids) and
must produce the SAME output tensor, so a "fast" result cannot come from skipping work.

  dense   : for e in experts: x[mask_e] @ W1[e].T -> act -> @ W2[e].T, scatter back
            (this is the shape of work Megatron does without a grouped/expert kernel)
  scatter : parallel_linear twice with flatten_sort_count, fused scatter/gather

Layout note (verified against the library source, not guessed): ParallelExperts stores its
weight as [E, OUT, IN] and ParallelExperts.forward() passes self.weight.permute(0, 2, 1) --
i.e. [E, IN, OUT] -- to parallel_linear, which computes X @ Wk[e] with IN == K. The w2 pass
uses k=1 and RETURNS ASSIGNMENT ROWS (k*T), so the caller folds top-k back into tokens.
Both variants therefore consume the identical activation/weight values and are directly
comparable.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

ACT = os.environ.get("A1_ACT", "silu")


def build_inputs(H, I, E, k, tokens, seed, device, dtype, scattermoe_root):
    """Build inputs AND the weights from the library's own MLP.

    Taking the weights from ``scattermoe.mlp.MLP`` removes any chance that the two variants
    disagree because the harness laid the weights out differently than the library expects:
    the dense path reads the very tensors the library module owns.
    """
    if scattermoe_root not in sys.path:
        sys.path.insert(0, scattermoe_root)
    from torch import nn
    from scattermoe.mlp import MLP

    g = torch.Generator(device="cpu").manual_seed(seed)
    mlp = MLP(H, I, E, k, activation=nn.SiLU()).to(device).to(dtype)
    with torch.no_grad():
        for p_ in mlp.parameters():
            p_.copy_((torch.randn(p_.shape, generator=g) / H ** 0.5).to(dtype))
    logits = torch.randn(tokens, E, generator=g)
    gates, idx = logits.softmax(-1).topk(k, dim=-1)
    gates = gates / gates.sum(-1, keepdim=True)
    x = torch.randn(tokens, H, generator=g).to(dtype).to(device)
    return mlp, x, gates.to(device), idx.to(device)


def _act(t):
    return torch.nn.functional.silu(t) if ACT == "silu" else torch.nn.functional.gelu(t)


def reference(mlp, x, gates, idx):
    """Dense per-expert loop over the library's own weights.

    Verified shapes (printed, not inferred):
      mlp.experts.weight        = [E, I, H]   (ParallelExperts stores [E, OUT, IN])
      mlp.output_experts.weight = [E, H, I]
    The module passes weight.permute(0, 2, 1) to the kernel, so the kernel computes
    x @ W1perm[e] with W1perm = [E, H, I], i.e. the same contraction as x @ W1[e].T.
    The dense path therefore mirrors it exactly with an explicit .T.
    """
    T, H = x.shape
    W1 = mlp.experts.weight            # [E, I, H]
    W2 = mlp.output_experts.weight     # [E, H, I]
    out = torch.zeros(T, H, device=x.device, dtype=torch.float32)
    for e in range(W1.shape[0]):
        mask = idx == e                # [T, k]
        if not mask.any():
            continue
        rows = mask.any(dim=1).nonzero(as_tuple=True)[0]
        xe = x[rows].float()
        he = torch.nn.functional.silu(xe @ W1[e].float().T)
        ye = he @ W2[e].float().T
        g = (gates.float() * mask.to(gates.dtype).float()).sum(dim=1)[rows].unsqueeze(-1)
        out[rows] += ye * g
    return out.to(x.dtype)


def scatter_forward(mlp, x, gates, idx):
    """The library's own fused module -- no harness-side layout decisions at all."""
    return mlp(x, gates.to(x.dtype), idx)


def timed(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("dense", "scatter", "both"), required=True)
    ap.add_argument("--solo-mem", action="store_true",
                    help="measure peak memory for this variant alone (run one variant per process; the cached allocator makes in-process peaks meaningless)")
    ap.add_argument("--scattermoe-root", default=None)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--intermediate", type=int, default=1024)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=2048, help="per-rank source tokens")
    ap.add_argument("--ep", type=int, default=2, help="gathered width multiplier")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    dev = torch.device("cuda")
    dtype = getattr(torch, a.dtype)
    assert a.scattermoe_root, "--scattermoe-root required"
    total = a.tokens * a.ep            # dispatcher output width
    mlp, x, gates, idx = build_inputs(
        a.hidden, a.intermediate, a.experts, a.topk, total, a.seed, dev, dtype,
        a.scattermoe_root)

    def fwd_dense():
        return reference(mlp, x, gates, idx)

    def fwd_scatter():
        return scatter_forward(mlp, x, gates, idx)

    variants = (("dense", fwd_dense), ("scatter", fwd_scatter)) if a.variant == "both" \
        else ((a.variant, fwd_dense if a.variant == "dense" else fwd_scatter),)

    # correctness: both variants must agree with the dense reference
    ref = reference(mlp, x, gates, idx)
    stats = {}
    for name, fn in variants:
        got = fn()
        maxdiff = (got.float() - ref.float()).abs().max().item()
        rel = maxdiff / max(1e-6, ref.float().abs().max().item())
        agree = rel < 2e-2
        if a.check and not agree:
            print(f"[a1] DISAGREE variant={name} rel={rel:.3e} maxabs={maxdiff:.3e}")
            raise SystemExit(7)
        stats[name] = (rel, maxdiff, agree)

    # Paired interleaved timing: warm both variants, then alternate dense/scatter inside one
    # process. A drifting co-tenant GPU load therefore hits both variants equally, which
    # separate processes cannot guarantee (the unconditioned runs here were bimodal).
    for _ in range(a.warmup):
        for _n, fn in variants:
            fn()
    torch.cuda.synchronize()
    per = {n: [] for n, _ in variants}
    for _ in range(a.iters):
        for n, fn in variants:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            per[n].append((time.perf_counter() - t0) * 1e3)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # Peak memory is only interpretable when a single variant has run in this process: the
    # caching allocator retains the other variant's blocks, which is why the interleaved mode
    # reports identical peaks. Use --solo-mem (one variant per process) for memory claims.
    for _ in range(3):
        for _n, fn in variants:
            fn()
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_res = torch.cuda.max_memory_reserved()

    for name, fn in variants:
        ts = per[name]
        rel, maxdiff, agree = stats[name]
        rec = dict(variant=name, hidden=a.hidden, intermediate=a.intermediate,
                   experts=a.experts, topk=a.topk, tokens=a.tokens, ep=a.ep,
                   total_assignments=total, seed=a.seed, dtype=a.dtype, act=ACT,
                   iters=a.iters, warmup=a.warmup, paired=a.variant == "both",
                   median_ms=statistics.median(ts), mean_ms=statistics.fmean(ts),
                   min_ms=min(ts), p95_ms=sorted(ts)[min(len(ts) - 1, int(0.95 * len(ts)))],
                   peak_allocated_bytes=peak_alloc, peak_reserved_bytes=peak_res,
                   agree_dense_ref=agree, rel_err=rel, max_abs_err=maxdiff,
                   cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
                   device=torch.cuda.get_device_name(0))
        with open(a.jsonl, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[a1] {name:7s} median={rec['median_ms']:.3f}ms "
              f"peak_alloc={peak_alloc / 2 ** 20:.1f}MiB peak_res={peak_res / 2 ** 20:.1f}MiB "
              f"rel_err={rel:.2e} agree={agree}")


if __name__ == "__main__":
    main()
