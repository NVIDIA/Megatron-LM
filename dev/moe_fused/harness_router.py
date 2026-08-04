#!/usr/bin/env python3
"""Can one kernel replace the router's whole four-launch chain?

The last trace attributed the largest remaining mcore-only work category: 48 of the 82
`splitKreduce` launches per step belong to the router GEMM. That is not a mis-tuned
heuristic -- at M=256, K=2048, N=128 the output is a couple of tiles, so splitting K is
the only way cuBLASLt gets parallelism, and the reduce is the price. The way out is not to
stop the split but to remove the GEMM boundary altogether:

    router GEMM -> splitK reduce -> softmax+topk -> padding mask      (4 launches/layer)
    one fused kernel                                                  (1 launch/layer)

The catch, and the reason this is a microbenchmark and not a patch: a fused kernel cannot
split K, because softmax needs a token's whole logit row. So its parallelism is capped at
M/BLOCK_M CTAs -- 8 CTAs at BLOCK_M=32, on 148 SMs -- and each CTA reads the entire
512 KB weight. Whether that beats a splitting GEMM plus three small kernels is a question
about this specific shape, and the honest answer might be no.

Numerics: production runs `te_general_gemm(bf16 in, fp32 out)`, so bf16 multiply with fp32
accumulate. The fused kernel does the same via `tl.dot`, differing only in K-reduction
order, so the bar is that every token selects the same set of experts -- not bitwise
equality of the probabilities.
"""

import argparse
import itertools

import torch
import triton
import triton.language as tl

from megatron.core.inference.moe.router_topk import fused_softmax_topk
from megatron.core.transformer.moe.inference_routing_mask_kernel import mask_routing_padding

M, K, N, TOPK = 256, 2048, 128, 8


@triton.jit
def _fused_router_kernel(
    x_ptr, w_ptr, probs_ptr, idx_ptr, real_cnt_ptr,
    M, K, x_rs, w_rs, row_offset,
    N: tl.constexpr, TOPK: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """GEMM + softmax + top-k + padding mask for one block of tokens, all experts.

    The expert axis is a compile-time constant and never tiled: the whole logit row has to
    be live for the softmax and the selection anyway.
    """
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    ns = tl.arange(0, N)
    row_live = rows < M

    acc = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        k_live = ks < K
        x = tl.load(
            x_ptr + rows[:, None] * x_rs + ks[None, :],
            mask=row_live[:, None] & k_live[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptr + ns[:, None] * w_rs + ks[None, :],
            mask=k_live[None, :],
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(w))

    # Pre-softmax routing: softmax over all experts, then take the top-k of the result.
    p = tl.exp(acc - tl.max(acc, axis=1)[:, None])
    p = p / tl.sum(p, axis=1)[:, None]

    # Padding rows route nowhere. Folding this in removes the separate mask launch, and
    # the count is read from device memory so the boundary can move across graph replays.
    pad = (rows + row_offset) >= tl.load(real_cnt_ptr).to(tl.int32)

    cur = p
    for t in tl.static_range(TOPK):
        best = tl.max(cur, axis=1)
        best_idx = tl.min(tl.where(cur == best[:, None], ns[None, :], N), axis=1)
        tl.store(probs_ptr + rows * TOPK + t, best, mask=row_live)
        tl.store(
            idx_ptr + rows * TOPK + t,
            tl.where(pad, -1, best_idx).to(tl.int64),
            mask=row_live,
        )
        cur = tl.where(ns[None, :] == best_idx[:, None], -float("inf"), cur)


def fused_router(x, w, real_cnt, block_m, block_k, warps, row_offset=0):
    probs = torch.empty(M, TOPK, dtype=torch.float32, device=x.device)
    idx = torch.empty(M, TOPK, dtype=torch.int64, device=x.device)
    _fused_router_kernel[(triton.cdiv(M, block_m),)](
        x, w, probs, idx, real_cnt,
        M, K, x.stride(0), w.stride(0), row_offset,
        N=N, TOPK=TOPK, BLOCK_M=block_m, BLOCK_K=block_k, num_warps=warps,
    )
    return probs, idx


def reference(x, w, real_cnt):
    """The production chain: fp32-output GEMM, then fused softmax+topk, then the mask."""
    logits = torch.mm(x.to(torch.float32), w.to(torch.float32).t())
    probs, idx = fused_softmax_topk(logits, TOPK)
    mask_routing_padding(idx, real_cnt, tp_rank=0)
    return probs, idx


def graph_time(fn, iters=300):
    for _ in range(10):
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
    return s.elapsed_time(e) * 1e3 / iters


def agreement(ref_idx, got_idx, n_real):
    """Fraction of real tokens whose selected expert *set* matches the reference."""
    r = ref_idx[:n_real].sort(dim=1).values
    g = got_idx[:n_real].sort(dim=1).values
    return (r == g).all(dim=1).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=int, default=M, help="unpadded token count")
    args = ap.parse_args()

    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.05
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.05
    real_cnt = torch.tensor([args.real], dtype=torch.int32, device="cuda")

    ref_p, ref_i = reference(x, w, real_cnt)
    t_ref = graph_time(lambda: reference(x, w, real_cnt))
    print(f"reference chain (GEMM + splitK reduce + softmax/topk + mask): {t_ref:6.2f} us")
    print(f"  real tokens {args.real}, padded rows get -1: "
          f"{bool((ref_i[args.real:] == -1).all()) if args.real < M else 'n/a'}\n")

    print(f"{'BLOCK_M':>8}{'BLOCK_K':>8}{'warps':>7}{'us':>8}{'vs ref':>8}"
          f"{'CTAs':>6}{'set match':>11}  probs max|d|")
    best = None
    for bm, bk, wp in itertools.product((16, 32, 64, 128), (32, 64, 128, 256), (4, 8)):
        try:
            p, i = fused_router(x, w, real_cnt, bm, bk, wp)
            t = graph_time(lambda: fused_router(x, w, real_cnt, bm, bk, wp))
        except Exception as ex:
            print(f"{bm:>8}{bk:>8}{wp:>7}  failed: {str(ex)[:44]}")
            continue
        match = agreement(ref_i, i, args.real)
        dp = (p[: args.real].float() - ref_p[: args.real].float()).abs().max().item()
        pad_ok = bool((i[args.real:] == -1).all()) if args.real < M else True
        print(f"{bm:>8}{bk:>8}{wp:>7}{t:>8.2f}{t_ref / t:>7.2f}x"
              f"{triton.cdiv(M, bm):>6}{match:>11.3f}  {dp:.2e}"
              f"{'' if pad_ok else '  PAD WRONG'}")
        if match == 1.0 and pad_ok and (best is None or t < best[0]):
            best = (t, bm, bk, wp)

    if best:
        t, bm, bk, wp = best
        print(f"\nbest agreeing config: BLOCK_M={bm} BLOCK_K={bk} warps={wp}  {t:.2f} us"
              f"  {t_ref / t:.2f}x vs reference")
        print(f"per step (48 layers): {48 * (t_ref - t) / 1000:+.3f} ms")
    else:
        print("\nno config both agreed with the reference and ran; fused router rejected")


if __name__ == "__main__":
    main()
