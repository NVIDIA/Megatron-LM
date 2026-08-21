"""Matched per-category decode budget for two engines, over steady-state windows.

The point of this script is to answer "where does mcore spend device time that vLLM
does not" without any of the three traps that made earlier attempts wrong:

  1. **Windowing.** Whole-trace sums include model load, capture, warmup and prefill,
     where the collectives move 100x the bytes of a decode step. Both traces are cut to
     the last N whole steps located from a one-per-layer-per-step anchor kernel.
  2. **Matched taxonomy.** The two engines share no kernel names, so a name-by-name diff
     is unreadable. Every kernel is mapped into one of a dozen functional buckets that
     mean the same thing on both sides, and anything unmatched is surfaced rather than
     silently dropped.
  3. **Device time only.** Both traces were captured with `--cuda-graph-trace=node`,
     which inflates host-side gaps and therefore wall time. Per-kernel *durations* are
     unaffected, so only sum-of-durations per bucket is compared -- never wall or idle.

One caveat that no windowing can remove: a device-side collective blocks until its peers
arrive, so its duration includes skew. Collective buckets are exposure, not cost.

Usage:
  python compare_budget.py A.sqlite '<anchor A>' B.sqlite '<anchor B>' [steps] [layers]
"""

import re
import sqlite3
import sys

# Ordered: first match wins, so put specific patterns before general ones.
TAXONOMY = [
    ("expert GEMM", r"_fused_moe_kernel|^bmm_Bfloat16"),
    ("attention", r"flash_?attn|fmhaSm100|FlashAttentionForward|splitkv|splitKV"),
    ("dense GEMM", r"^nvjet|cutlass.*gemm|^sm100_xmma"),
    ("collective", r"multimem|ncclDevKernel|nccl.*Kernel"),
    ("MoE routing", r"routingIndices|_align_single|_mask_routing|softmax_topk|topk"),
    ("MoE finalize", r"_moe_sum|finalizeKernel"),
    ("norm", r"rmsnorm|layernorm|_fused_add_rmsnorm|rsqrt"),
    ("rotary / KV", r"rotary|append_kv|reshape_and_cache|cache_kernel"),
    ("elementwise / copy", r"elementwise_kernel|CatArrayBatched|copy|vectorized_elementwise"),
    ("splitK reduce", r"splitKreduce|splitk"),
    ("metadata / bookkeep", r"_fused_metadata|delayStream|memset|fill"),
    ("sampling", r"sampl|argmax|multinomial|gumbel"),
]


def bucket(name):
    for label, pat in TAXONOMY:
        if re.search(pat, name, re.I):
            return label
    return "OTHER"


def window(con, anchor, layers, want):
    ts = [
        r[0]
        for r in con.execute(
            """SELECT k.start FROM CUPTI_ACTIVITY_KIND_KERNEL k
               JOIN StringIds s ON k.demangledName=s.id
               WHERE k.deviceId=0 AND s.value LIKE ? ORDER BY k.start""",
            (anchor,),
        )
    ]
    if not ts:
        sys.exit(f"anchor {anchor!r} not found")
    need = want * layers
    sel = ts[-need:] if len(ts) > need else ts
    lo, hi = sel[0], sel[-1]
    steps = len(sel) / layers
    gaps = [b - a for a, b in zip(sel, sel[1:])]
    big = [g for g in gaps if g > 50_000_000]
    return lo, hi, steps, big


def budget(db, anchor, layers, want):
    con = sqlite3.connect(db)
    lo, hi, steps, big = window(con, anchor, layers, want)
    rows = con.execute(
        """SELECT s.value, SUM(k.end-k.start), COUNT(*)
           FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
           WHERE k.deviceId=0 AND k.start>=? AND k.start<=? GROUP BY s.value""",
        (lo, hi),
    ).fetchall()
    per = {}
    for name, ns, n in rows:
        b = bucket(name)
        ms, cnt = per.get(b, (0.0, 0.0))
        per[b] = (ms + ns / 1e6 / steps, cnt + n / steps)
    other = sorted(
        ((ns / 1e6 / steps, n / steps, name) for name, ns, n in rows if bucket(name) == "OTHER"),
        reverse=True,
    )
    return {
        "db": db,
        "steps": steps,
        "wall": (hi - lo) / 1e6 / steps,
        "big": big,
        "per": per,
        "other": other[:8],
        "total": sum(v[0] for v in per.values()),
        "launches": sum(v[1] for v in per.values()),
    }


def main():
    a_db, a_anchor, b_db, b_anchor = sys.argv[1:5]
    want = int(sys.argv[5]) if len(sys.argv) > 5 else 60
    layers = int(sys.argv[6]) if len(sys.argv) > 6 else 48
    A = budget(a_db, a_anchor, layers, want)
    B = budget(b_db, b_anchor, layers, want)

    for X in (A, B):
        warn = f"  WARNING {len(X['big'])} inter-iteration gaps in window" if X["big"] else ""
        print(
            f"{X['db']}: steps={X['steps']:.0f} wall/step={X['wall']:.3f} ms "
            f"device={X['total']:.3f} ms launches/step={X['launches']:.0f}{warn}"
        )

    print(
        f"\n{'bucket':<22}{'A ms':>8}{'A n':>7}{'B ms':>9}{'B n':>7}"
        f"{'B-A ms':>9}{'B-A n':>8}"
    )
    labels = sorted(
        set(A["per"]) | set(B["per"]),
        key=lambda l: -(B["per"].get(l, (0, 0))[0] - A["per"].get(l, (0, 0))[0]),
    )
    for label in labels:
        am, an = A["per"].get(label, (0.0, 0.0))
        bm, bn = B["per"].get(label, (0.0, 0.0))
        print(
            f"{label:<22}{am:>8.3f}{an:>7.0f}{bm:>9.3f}{bn:>7.0f}"
            f"{bm-am:>+9.3f}{bn-an:>+8.0f}"
        )
    print(
        f"{'TOTAL':<22}{A['total']:>8.3f}{A['launches']:>7.0f}"
        f"{B['total']:>9.3f}{B['launches']:>7.0f}"
        f"{B['total']-A['total']:>+9.3f}{B['launches']-A['launches']:>+8.0f}"
    )

    for X, tag in ((A, "A"), (B, "B")):
        if X["other"]:
            print(f"\nunclassified in {tag} ({X['db']}):")
            for ms, n, name in X["other"]:
                print(f"  {ms:7.3f} ms/step {n:6.1f}/step  {name[:78]}")


if __name__ == "__main__":
    main()
