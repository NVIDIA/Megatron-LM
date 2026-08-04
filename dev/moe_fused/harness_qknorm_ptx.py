#!/usr/bin/env python3
"""Why is the shipped q/k norm 6.15 us where a harness kernel with the same math is 4.12?

Three source-level explanations were tested and rejected (runtime output stride, clamped
store addressing, split load address arrays), each measuring 6.15 us unchanged. One oddity
in the tuning sweep suggests the difference is not source-level at all: the harness times
quantized to exactly three values (8.21 / 6.15 / 4.12 us) and BLOCK=8 hit 4.12 us at
*every* warp count, which is not how a bandwidth-bound kernel responds to warp count.

So stop reading the source and read what the two actually compile to: register count,
spills, and how wide the generated global loads and stores are. A kernel moving 128-wide
bf16 rows should be issuing vectorized (v2/v4) accesses; if one of them is issuing scalar
ones, that is the whole factor.
"""

import re

import torch
import triton
import triton.language as tl

from megatron.core.inference.attention import fused_qk_norm as fqn

SQ, B, NG, NPG, HN = 1, 256, 4, 8, 128
HEADS = NG * NPG
EPS = 1e-6


@triton.jit
def _fast_kernel(
    q_ptr, k_ptr, qo_ptr, ko_ptr, wq_ptr, wk_ptr,
    n_q_rows, n_rows, k_in_rs, q_grp_rs, eps,
    HN: tl.constexpr, NPG: tl.constexpr, HEADS: tl.constexpr, ROWS: tl.constexpr,
):
    """The harness kernel that measured 4.12 us, copied here verbatim for comparison."""
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, HN)
    live = rows < n_rows
    is_q = rows < n_q_rows

    q_row = tl.where(is_q, rows, 0)
    head = q_row % HEADS
    q_off = ((q_row // HEADS) * (HEADS // NPG) + head // NPG) * q_grp_rs + (head % NPG) * HN
    k_off = tl.where(is_q, 0, rows - n_q_rows) * k_in_rs

    src = tl.where(is_q, q_off, k_off)
    base = tl.where(is_q, 0, 1)
    offs = src[:, None] + cols[None, :]
    m = live[:, None]

    xq = tl.load(q_ptr + offs, mask=m & (base == 0)[:, None], other=0.0).to(tl.float32)
    xk = tl.load(k_ptr + offs, mask=m & (base == 1)[:, None], other=0.0).to(tl.float32)
    x = tl.where(is_q[:, None], xq, xk)

    var = tl.sum(x * x, axis=1) / HN
    xn = x * (1.0 / tl.sqrt(var + eps))[:, None]

    wq = tl.load(wq_ptr + cols).to(tl.float32)
    wk = tl.load(wk_ptr + cols).to(tl.float32)
    w = tl.where(is_q[:, None], wq[None, :], wk[None, :])
    y = xn * w

    out_q = rows[:, None] * HN + cols[None, :]
    out_k = (rows - n_q_rows)[:, None] * HN + cols[None, :]
    tl.store(qo_ptr + out_q, y.to(qo_ptr.dtype.element_ty), mask=m & is_q[:, None])
    tl.store(ko_ptr + out_k, y.to(ko_ptr.dtype.element_ty), mask=m & (~is_q)[:, None])


def run_fast(gq, k, wq, wk, rows, warps):
    n_q_rows = SQ * B * HEADS
    n_rows = n_q_rows + SQ * B * NG
    qo = torch.empty(SQ, B, HEADS, HN, dtype=gq.dtype, device=gq.device)
    ko = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    h = _fast_kernel[(triton.cdiv(n_rows, rows),)](
        gq, k, qo.view(-1, HN), ko.view(-1, HN), wq, wk,
        n_q_rows, n_rows, k.reshape(-1, HN).stride(0), gq.stride(2), float(EPS),
        HN=HN, NPG=NPG, HEADS=HEADS, ROWS=rows, num_warps=warps,
    )
    return (qo, ko), h


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


def describe(handle, label):
    regs = getattr(handle, "n_regs", None)
    spills = getattr(handle, "n_spills", None)
    print(f"\n=== {label}")
    print(f"  registers/thread {regs}   spills {spills}")
    asm = getattr(handle, "asm", {}) or {}
    ptx = asm.get("ptx", "")
    if not ptx:
        print("  (no PTX available on this Triton build)")
        return
    for kind in ("ld.global", "st.global"):
        widths = {}
        for mm in re.finditer(rf"{re.escape(kind)}(\.v[24])?", ptx):
            widths[mm.group(1) or ".scalar"] = widths.get(mm.group(1) or ".scalar", 0) + 1
        print(f"  {kind:<10} " + "  ".join(f"{k}={v}" for k, v in sorted(widths.items())))
    for kind in ("ld.global.nc", "cvt.rn", "mul.f32", "bar.sync"):
        print(f"  {kind:<12} {len(re.findall(re.escape(kind), ptx))}")


def main():
    torch.manual_seed(0)
    mixed = torch.randn(SQ, B, NG, (NPG + 2) * HN, dtype=torch.bfloat16, device="cuda")
    gq, k, v = torch.split(mixed, [NPG * HN, HN, HN], dim=3)
    wq = torch.randn(HN, dtype=torch.bfloat16, device="cuda")
    wk = torch.randn(HN, dtype=torch.bfloat16, device="cuda")

    (qf, kf), h_fast = run_fast(gq, k, wq, wk, 8, 8)
    qs, ks = fqn.fused_qk_rmsnorm_grouped(gq, k, wq, wk, EPS, False)
    print(f"same answer: q {torch.equal(qf, qs)}  k {torch.equal(kf, ks)}")

    t_fast = graph_time(lambda: run_fast(gq, k, wq, wk, 8, 8))
    t_ship = graph_time(lambda: fqn.fused_qk_rmsnorm_grouped(gq, k, wq, wk, EPS, False))
    print(f"harness kernel {t_fast:6.2f} us     shipped {t_ship:6.2f} us"
          f"     ratio {t_ship / t_fast:.2f}x")

    describe(h_fast, f"harness kernel (ROWS=8, warps=8)")
    # Reach the shipped kernel's compiled handle through the JIT cache.
    ship = fqn._fused_qk_rmsnorm_kernel
    cache = getattr(ship, "cache", {})
    handles = [h for per_dev in cache.values() for h in per_dev.values()]
    if handles:
        describe(handles[-1], f"shipped kernel ({len(handles)} cache entries)")
    else:
        print("\n(could not reach the shipped kernel's compiled handle)")


if __name__ == "__main__":
    main()
