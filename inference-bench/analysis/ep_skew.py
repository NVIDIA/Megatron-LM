"""Line up EP dispatch/combine calls on two ranks of one nsys run and measure arrival skew.

The ranks' traces use different nodes' clocks, but an EP collective ends at (nearly) the same
instant on every rank, so the per-call end-time difference is the clock offset (a rolling median
absorbs drift). With it removed, the start-time difference is the arrival skew. Decode steps only.
Needs trace_rank<r>.sqlite (nsys export / nsys stats --force-export) for both ranks.

    python3 inference-bench/analysis/ep_skew.py \
        nvl_a2a=inference-bench/experiments/<a2a trace run> nvls=inference-bench/experiments/<nvls trace run>
"""

import sqlite3
import sys

import numpy as np

KERNELS = {
    "nvl_a2a": ("_a2a_dispatch_v_kernel", "_a2a_combine_v_kernel"),
    "nvls": ("_multimem_all_gatherv_3tensor_kernel", "_multimem_reduce_scatter_v_kernel"),
}
RANKS = (4, 44)
runs = {}
for spec in sys.argv[1:]:
    label, d = spec.split("=", 1)
    runs[label] = (d, *KERNELS[label])
def load(d, r, name):
    c = sqlite3.connect(f"{d}/trace_rank{r}.sqlite")
    q = """select k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id
           where s.value=? order by k.start"""
    return np.array(c.execute(q, (name,)).fetchall(), dtype=np.int64)
pct = lambda a: "p5 {:6.1f}  p50 {:6.1f}  p95 {:6.1f}".format(*np.percentile(a, [5, 50, 95]))
for lab, (d, disp, comb) in runs.items():
    moe = load(d, RANKS[0], "_fused_moe_kernel"); moe_dur = (moe[:, 1] - moe[:, 0]) / 1000
    step_max = moe_dur.reshape(-1, 96).max(axis=1)           # 96 fused_moe per step
    decode = step_max < 100                                     # prefill steps have long expert GEMMs
    print(f"\n=== {lab}: {decode.sum()} decode steps of {len(decode)}")
    for kind, name in (("dispatch", disp), ("combine", comb)):
        a, b = load(d, RANKS[0], name), load(d, RANKS[1], name)
        keep = np.repeat(decode, 48)
        a, b = a[keep], b[keep]
        dend = (b[:, 1] - a[:, 1]).astype(float)
        # rolling median of end differences = clock offset (+ drift)
        w = 201
        off = np.array([np.median(dend[max(0, i - w // 2): i + w // 2 + 1]) for i in range(len(dend))])
        resid_end = (dend - off) / 1000
        arrive = ((b[:, 0] - a[:, 0]) - off) / 1000          # + = rank 44 started later
        d4, d44 = (a[:, 1] - a[:, 0]) / 1000, (b[:, 1] - b[:, 0]) / 1000
        print(f"  {kind:8s} end-time residual (should be ~0): {pct(resid_end)} us   |resid|<2us: {100*np.mean(abs(resid_end)<2):.0f}%")
        print(f"  {kind:8s} arrival skew r44-r4:              {pct(arrive)} us   |skew| p50 {np.median(abs(arrive)):.1f} us, r44 later in {100*np.mean(arrive>0):.0f}% of calls")
        print(f"  {kind:8s} duration r4:                      {pct(d4)} us")
        print(f"  {kind:8s} duration r44:                     {pct(d44)} us")
        print(f"  {kind:8s} min(r4, r44) (later arriver):     {pct(np.minimum(d4, d44))} us")
        print(f"  {kind:8s} max - min (= skew absorbed):      {pct(np.abs(d4 - d44))} us")
