# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Benchmark / correctness harness for the fused CuTe-DSL CSA Compressor kernels.

Not collected by pytest (see ``test_csa_fused_compressor.py`` for the unit tests); this
is the standalone harness behind the numbers in
https://github.com/NVIDIA/Megatron-LM/issues/5968. The reference implementation is the
verbatim eager region of ``Compressor._forward_thd`` (non-pre-grouped THD path),
restricted to exactly the region the fused kernels replace: from the ``linear_wkv`` /
``linear_wgate`` outputs to the pre-RMSNorm pooled output. Projection GEMMs, RMSNorm and
RoPE are outside the region for both implementations (they are unchanged).
``cu_seqlens_compressed`` / ``total_comp`` are computed outside the timed region for both
implementations (both need them; upstream derives them once per layer).

Modes:
  --check           correctness fwd+bwd (bit-exact tests + tolerances + fp64 oracle)
  --bench           CUDA-event wall-clock timing (fwd, bwd separately)
  --nsys IMPL       loop for nsys pure-kernel capture (cudaProfilerApi gated, NVTX
                    FWD/BWD ranges), IMPL in {ref,fused}

Shapes: --dim, --ratio, --lens (comma list of per-segment lengths), e.g.::

  python tests/unit_tests/transformer/experimental_attention_variant/\
bench_csa_fused_compressor.py --check --lens 8191,8192,4093 --dim 128 --ratio 4

  nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o rep_fused python ...bench_csa_fused_compressor.py --nsys fused --lens 8192 --dim 128
  nsys stats --report nvtx_kern_sum --format csv,column --force-export=true rep_fused.nsys-rep
"""

import argparse
from types import SimpleNamespace

import torch

from megatron.core.transformer.experimental_attention_variant.csa import Compressor, batch_of_row
from megatron.core.transformer.experimental_attention_variant.csa_fused_compressor import (
    compress_thd_fused,
)

# ---------------------------------------------------------------------------
# Upstream eager reference (the exact region the fused kernels replace)
# ---------------------------------------------------------------------------


def ref_pool(
    kv, score, ape, cu_seqlens, cu_seqlens_comp, total_comp, ratio, d, coff, mode="upstream"
):
    """Eager reference for the pooled region; ``mode`` in {"upstream", "fp32", "fp64"}.

    "upstream" reproduces the current eager numerics exactly (softmax weights rounded to
    bf16, bf16 multiply, fp32-accumulated sum); "fp32" keeps all intermediates fp32 with
    a single final bf16 rounding (the fused kernels' numerics); "fp64" is an oracle.
    """
    device = kv.device
    row_idx = torch.arange(total_comp, device=device, dtype=cu_seqlens_comp.dtype)
    batch_ids = batch_of_row(cu_seqlens_comp, total_q=total_comp)
    valid_comp = row_idx < cu_seqlens_comp[-1]
    local_pos = row_idx - cu_seqlens_comp[batch_ids]
    local_pos = torch.where(valid_comp, local_pos, torch.zeros_like(local_pos))
    base = cu_seqlens[batch_ids].unsqueeze(1) + local_pos.unsqueeze(1) * ratio
    base = torch.where(valid_comp.unsqueeze(1), base, torch.zeros_like(base))
    offsets = torch.arange(ratio, device=device, dtype=base.dtype).unsqueeze(0)
    gather_idx = base + offsets  # (total_comp, ratio)

    if mode == "fp32":
        kv = kv.float()
        score = score.float()
    elif mode == "fp64":
        kv = kv.double()
        score = score.double()
        ape = ape.double()

    kv_grouped = kv[gather_idx]  # (total_comp, ratio, 1, coff * d)
    score_grouped = score[gather_idx]
    score_grouped = score_grouped + ape.view(1, ratio, 1, -1)

    if coff == 2:
        is_first = local_pos == 0
        stub = SimpleNamespace(head_dim=d)
        kv_grouped = Compressor._overlap_transform_thd(stub, kv_grouped, is_first, fill_value=0)
        score_grouped = Compressor._overlap_transform_thd(
            stub, score_grouped, is_first, fill_value=float("-inf")
        )

    if mode == "upstream":
        weights = torch.softmax(score_grouped, dim=1, dtype=torch.float32).to(kv_grouped.dtype)
        out = (kv_grouped * weights).sum(dim=1)
    elif mode == "fp32":
        weights = torch.softmax(score_grouped, dim=1, dtype=torch.float32)
        out = (kv_grouped * weights).sum(dim=1).to(torch.bfloat16)
    else:  # fp64 oracle
        weights = torch.softmax(score_grouped, dim=1, dtype=torch.float64)
        out = (kv_grouped * weights).sum(dim=1)
    return out


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def make_inputs(lens, d, ratio, coff, seed=1234, device="cuda"):
    """Build a THD pack of bf16 kv/score, fp32 ape, cu_seqlens(+compressed) and grads."""
    total = sum(lens)
    w = coff * d
    gen = torch.Generator(device="cpu").manual_seed(seed)
    kv = torch.randn(total, 1, w, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    score = (torch.randn(total, 1, w, generator=gen, dtype=torch.float32).mul_(1.5)).to(
        torch.bfloat16
    )
    ape = torch.randn(ratio, w, generator=gen, dtype=torch.float32).mul_(0.25)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=device)
    seg_comp = torch.tensor([seg_len // ratio for seg_len in lens])
    cuc = torch.tensor([0] + list(seg_comp.cumsum(0)), dtype=torch.int32, device=device)
    total_comp = int(cuc[-1].item())
    go = torch.randn(total_comp, 1, d, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    return (kv.to(device), score.to(device), ape.to(device), cu, cuc, total_comp, go.to(device))


def run_ref(kv, score, ape, cu, cuc, total_comp, ratio, d, coff):
    """Eager reference over the replaced region (verbatim upstream numerics)."""
    return ref_pool(kv, score, ape, cu, cuc, total_comp, ratio, d, coff)


def run_fused(kv, score, ape, cu, cuc, total_comp, ratio, d, coff):
    """Fused CuTe-DSL kernels over the replaced region."""
    out = compress_thd_fused(
        kv.view(kv.shape[0], -1),
        score.view(score.shape[0], -1),
        ape,
        cu,
        cuc,
        ratio,
        d,
        coff,
        total_comp=total_comp,
    )
    return out.view(total_comp, 1, d)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


def check(lens, d, ratio, coff, seed):
    """Full fwd+bwd comparison: fused vs upstream eager, fp32 eager, fp64 oracle."""
    kv, score, ape, cu, cuc, total_comp, go = make_inputs(lens, d, ratio, coff, seed)

    def fwd_bwd(fn, mode=None, dtype=None):
        kv_l = (kv.to(dtype) if dtype else kv.clone()).requires_grad_(True)
        score_l = (score.to(dtype) if dtype else score.clone()).requires_grad_(True)
        ape_l = (ape.to(dtype) if dtype else ape.clone()).requires_grad_(True)
        if mode is not None:
            out = ref_pool(kv_l, score_l, ape_l, cu, cuc, total_comp, ratio, d, coff, mode=mode)
        else:
            out = fn(kv_l, score_l, ape_l, cu, cuc, total_comp, ratio, d, coff)
        out.backward(go.to(out.dtype))
        torch.cuda.synchronize()
        return (out.detach(), kv_l.grad.detach(), score_l.grad.detach(), ape_l.grad.detach())

    r_up = fwd_bwd(None, mode="upstream")
    r_32 = fwd_bwd(None, mode="fp32")
    r_64 = fwd_bwd(None, mode="fp64", dtype=torch.float64)
    r_fu = fwd_bwd(run_fused)

    names = ("fwd", "dKV", "dScore", "dAPE")
    print(f"[check] lens={lens} d={d} ratio={ratio} coff={coff} total_comp={total_comp}")

    def cmp(a, b, label):
        exact = torch.equal(a, b)
        af, bf = a.float(), b.float()
        max_abs = (af - bf).abs().max().item()
        n_diff = (af != bf).sum().item()
        print(
            f"  {label:26s} bit_exact={exact}  n_diff={n_diff}/{a.numel()}"
            f"  max_abs={max_abs:.3e}"
        )
        return exact

    print(" fused vs upstream-eager (verbatim numerics):")
    for i, name in enumerate(names):
        cmp(r_fu[i], r_up[i], name)
    print(" fused vs fp32-eager (same math, fp32 intermediates):")
    for i, name in enumerate(names):
        cmp(r_fu[i], r_32[i], name)
    print(" error vs fp64 oracle (max abs):")
    for i, name in enumerate(names):
        e_up = (r_up[i].double() - r_64[i].double()).abs().max().item()
        e_32 = (r_32[i].double() - r_64[i].double()).abs().max().item()
        e_fu = (r_fu[i].double() - r_64[i].double()).abs().max().item()
        print(f"  {name:8s} upstream-eager={e_up:.3e}  fp32-eager={e_32:.3e}  fused={e_fu:.3e}")

    # Run-to-run determinism of the fused kernels (dAPE uses fp32 atomics).
    r_fu2 = fwd_bwd(run_fused)
    det = [torch.equal(a, b) for a, b in zip(r_fu, r_fu2)]
    print(f" fused replay determinism (fwd,dKV,dScore,dAPE): {det}")


# ---------------------------------------------------------------------------
# Wall-clock bench (CUDA events)
# ---------------------------------------------------------------------------


def bench_impl(fn, kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go, warmup=20, iters=100):
    """Median wall-clock (ms) of forward-only and backward-only over the region."""
    with torch.no_grad():
        for _ in range(warmup):
            fn(kv, score, ape, cu, cuc, total_comp, ratio, d, coff)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)
            start.record()
            fn(kv, score, ape, cu, cuc, total_comp, ratio, d, coff)
            end.record()
            torch.cuda.synchronize()
            ts.append(start.elapsed_time(end))
    fwd_ms = sorted(ts)[len(ts) // 2]

    kv_l = kv.clone().requires_grad_(True)
    score_l = score.clone().requires_grad_(True)
    ape_l = ape.clone().requires_grad_(True)
    for _ in range(warmup):
        out = fn(kv_l, score_l, ape_l, cu, cuc, total_comp, ratio, d, coff)
        out.backward(go)
        kv_l.grad = score_l.grad = ape_l.grad = None
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        out = fn(kv_l, score_l, ape_l, cu, cuc, total_comp, ratio, d, coff)
        torch.cuda.synchronize()
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        start.record()
        out.backward(go)
        end.record()
        torch.cuda.synchronize()
        ts.append(start.elapsed_time(end))
        kv_l.grad = score_l.grad = ape_l.grad = None
    bwd_ms = sorted(ts)[len(ts) // 2]
    return fwd_ms, bwd_ms


def bench(lens, d, ratio, coff, seed):
    """Wall-clock comparison of eager vs fused over the replaced region."""
    kv, score, ape, cu, cuc, total_comp, go = make_inputs(lens, d, ratio, coff, seed)
    rf, rb = bench_impl(run_ref, kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go)
    ff, fb = bench_impl(run_fused, kv, score, ape, cu, cuc, total_comp, ratio, d, coff, go)
    print(f"[bench-wall] lens={lens} d={d} ratio={ratio} coff={coff}")
    print(f"  ref   fwd {rf*1e3:9.1f} us   bwd {rb*1e3:9.1f} us")
    print(f"  fused fwd {ff*1e3:9.1f} us   bwd {fb*1e3:9.1f} us")
    print(f"  speedup fwd {rf/ff:6.2f}x   bwd {rb/fb:6.2f}x")


# ---------------------------------------------------------------------------
# nsys capture loop
# ---------------------------------------------------------------------------


def nsys_loop(impl, lens, d, ratio, coff, seed, iters=50):
    """cudaProfilerApi-gated fwd+bwd loop with NVTX FWD/BWD ranges for nsys."""
    fn = run_ref if impl == "ref" else run_fused
    kv, score, ape, cu, cuc, total_comp, go = make_inputs(lens, d, ratio, coff, seed)
    kv_l = kv.clone().requires_grad_(True)
    score_l = score.clone().requires_grad_(True)
    ape_l = ape.clone().requires_grad_(True)
    for _ in range(20):  # warmup (also outside profiler: JIT compile)
        out = fn(kv_l, score_l, ape_l, cu, cuc, total_comp, ratio, d, coff)
        out.backward(go)
        kv_l.grad = score_l.grad = ape_l.grad = None
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for _ in range(iters):
        torch.cuda.nvtx.range_push("FWD")
        out = fn(kv_l, score_l, ape_l, cu, cuc, total_comp, ratio, d, coff)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("BWD")
        out.backward(go)
        torch.cuda.nvtx.range_pop()
        kv_l.grad = score_l.grad = ape_l.grad = None
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(f"[nsys-loop] impl={impl} iters={iters} done")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--nsys", choices=["ref", "fused"])
    parser.add_argument("--lens", default="8192")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--coff", type=int, default=2)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    lens = [int(x) for x in args.lens.split(",")]
    coff = args.coff if args.ratio == 4 else 1
    if args.check:
        check(lens, args.dim, args.ratio, coff, args.seed)
    if args.bench:
        bench(lens, args.dim, args.ratio, coff, args.seed)
    if args.nsys:
        nsys_loop(args.nsys, lens, args.dim, args.ratio, coff, args.seed, args.iters)


if __name__ == "__main__":
    main()
