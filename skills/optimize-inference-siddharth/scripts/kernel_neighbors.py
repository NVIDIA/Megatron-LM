"""Identify an unattributed kernel from its trace neighbours.

A profile shows `elementwise_kernel` costing 0.16 ms/step and says nothing about which
line launched it. The instinct is to instrument the framework -- a `TorchDispatchMode`
hook, a monkeypatched `copy_`, a stack capture. Under CUDA-graph replay that instinct
is usually wrong: the hook fires at *capture* time, not replay, so an arming predicate
based on per-step state either never becomes true or becomes true during prefill, and
each attempt costs a full job launch to find out.

The trace already knows. A kernel's immediate predecessor and successor on the same
device are stable under graph replay, and the pair usually names the call site outright:
a copy between a QKV GEMM and a QK-norm is the split; a cast between an expert GEMM and
a reduce-scatter is the collective's input buffer dtype.

Reads the same steady-state window convention as the other scripts here.

Worked example -- Qwen3-30B-A3B, 76.3 unattributed `elementwise_kernel` launches/step,
after four failed instrumentation attempts. Two neighbour pairs at exactly 23.9/step
(the 48-layer count halved, i.e. one per layer per boundary) named both sources:

  23.9  _multimem_reduce_scatter_v_kernel -> triton_poi_fused_add_copy__0
  23.9  nvjet_sm100_tst_64x8_... (QKV GEMM) -> transformer_engine::...rmsnorm_fwd

The first is the reduce-scatter output cast -- fixed by making the collective's buffer
bf16, worth +2.5%. The second is the QKV split before QK-norm.

Usage:
  python kernel_neighbors.py trace.sqlite '<target kernel LIKE pattern>' \
      '<anchor kernel LIKE pattern>' [steps] [layers]
"""

import sqlite3
import sys
from collections import Counter


def main():
    db, target, anchor = sys.argv[1], sys.argv[2], sys.argv[3]
    want = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    layers = int(sys.argv[5]) if len(sys.argv) > 5 else 48

    con = sqlite3.connect(db)
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
    sel = ts[-want * layers :] if len(ts) > want * layers else ts
    lo, hi = sel[0], sel[-1]
    steps = len(sel) / layers

    # Whole ordered kernel stream for the window; neighbours are positional, and a
    # stream-aware version would need the correlation ids, which decode rarely needs
    # because the per-layer chain is serial anyway.
    stream = con.execute(
        """SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
           JOIN StringIds s ON k.demangledName=s.id
           WHERE k.deviceId=0 AND k.start>=? AND k.start<=? ORDER BY k.start""",
        (lo, hi),
    ).fetchall()

    tgt = target.replace("%", "").lower()
    hits = [i for i, (_, _, name) in enumerate(stream) if tgt in name.lower()]
    if not hits:
        sys.exit(f"target {target!r} not found in window")

    pairs = Counter()
    dur = 0
    for i in hits:
        dur += stream[i][1] - stream[i][0]
        prev = stream[i - 1][2] if i else "<window start>"
        nxt = stream[i + 1][2] if i + 1 < len(stream) else "<window end>"
        pairs[(prev[:52], nxt[:52])] += 1

    print(f"{db}")
    print(
        f"  target {target!r}: {len(hits)/steps:.1f} launches/step, "
        f"{dur/1e6/steps:.4f} ms/step over {steps:.0f} steps"
    )
    print(f"\n  {'n/step':>7}  predecessor -> successor")
    for (prev, nxt), n in pairs.most_common(12):
        print(f"  {n/steps:>7.1f}  {prev}\n           -> {nxt}")


if __name__ == "__main__":
    main()
