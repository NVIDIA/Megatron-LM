# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Summarise a PyTorch-profiler chrome trace of one training step per category.

    python tools/dsv41/summarize_profile.py rank-0.json.gz [--top 40]

Reports, for the GPU streams: total kernel time, time per category (GEMM, attention, MoE
permutation / grouped GEMM, NCCL, elementwise / norm, memory copies, indexing, other), the
busiest kernels, and the idle fraction (time between the first and the last kernel that no
kernel on any stream covers). Categories are matched on kernel names; unknown names fall
into "other" and are listed so the table can be extended.
"""

import argparse
import gzip
import json
import re
from collections import defaultdict

_CATEGORIES = [
    ("nccl", re.compile(r"nccl|ncclDevKernel|AllToAll|AllGather|ReduceScatter|AllReduce", re.I)),
    ("gemm", re.compile(r"gemm|cutlass|nvjet|cublas|Gemm|xmma|sm90_|sm100_|wgmma|matmul", re.I)),
    (
        "attention",
        re.compile(r"flash|fmha|attn|attention|cudnn.*(sdpa|mha)|FlashMLA|sparse_attn", re.I),
    ),
    (
        "moe",
        re.compile(
            r"permut|unpermut|moe|router|topk|sort|scatter|gather|index_select|bincount", re.I
        ),
    ),
    ("norm", re.compile(r"norm|rms|layer_norm|LayerNorm|RMSNorm", re.I)),
    (
        "elementwise",
        re.compile(
            r"elementwise|vectorized|unrolled|fill|add|mul|silu|swiglu|sigmoid|softmax|exp|cast|copy_kernel|Functor|reduce_kernel|where|clamp|masked",
            re.I,
        ),
    ),
    ("memcpy", re.compile(r"Memcpy|Memset|memcpy|memset", re.I)),
    ("optimizer", re.compile(r"adam|multi_tensor|l2norm|clip", re.I)),
    ("rope", re.compile(r"rope|rotary", re.I)),
    ("embedding", re.compile(r"embedding|Embedding", re.I)),
    ("cross_entropy", re.compile(r"cross_entropy|log_softmax|nll", re.I)),
]


def categorize(name: str) -> str:
    for cat, rx in _CATEGORIES:
        if rx.search(name):
            return cat
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()
    opener = gzip.open if args.trace.endswith(".gz") else open
    with opener(args.trace, "rt") as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    kernels = [
        e
        for e in events
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"
    ]
    if not kernels:
        raise SystemExit("no GPU kernel events in trace")
    kernels.sort(key=lambda e: e["ts"])
    t0, t1 = kernels[0]["ts"], max(e["ts"] + e["dur"] for e in kernels)
    span = (t1 - t0) / 1e3  # ms
    per_cat = defaultdict(float)
    per_name = defaultdict(lambda: [0.0, 0])
    per_stream = defaultdict(float)
    busy_intervals = []
    for e in kernels:
        name = e["name"]
        cat = categorize(name) if e.get("cat") == "kernel" else "memcpy"
        per_cat[cat] += e["dur"] / 1e3
        per_name[name][0] += e["dur"] / 1e3
        per_name[name][1] += 1
        per_stream[e.get("tid")] += e["dur"] / 1e3
        busy_intervals.append((e["ts"], e["ts"] + e["dur"]))
    # union of busy intervals across all streams -> fraction of the span with any GPU work
    busy_intervals.sort()
    covered, cur_s, cur_e = 0.0, None, None
    for s, e in busy_intervals:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                covered += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        covered += cur_e - cur_s
    covered /= 1e3
    total = sum(per_cat.values())
    print(
        f"trace span {span:.1f} ms, GPU busy (any stream) {covered:.1f} ms ({100 * covered / span:.1f}%), "
        f"summed kernel time {total:.1f} ms over {len(kernels)} kernels"
    )
    print("-- per category (ms, % of summed kernel time):")
    for cat, ms in sorted(per_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:14s} {ms:9.1f}  {100 * ms / total:5.1f}%")
    print("-- per stream (ms):")
    for tid, ms in sorted(per_stream.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  stream {tid}: {ms:.1f}")
    print(f"-- top {args.top} kernels (ms, count, category):")
    for name, (ms, n) in sorted(per_name.items(), key=lambda kv: -kv[1][0])[: args.top]:
        print(f"  {ms:8.1f} {n:6d}  {categorize(name):12s} {name[:110]}")
    others = [(n, v[0]) for n, v in per_name.items() if categorize(n) == "other"]
    if others:
        print("-- uncategorised kernels (top 15):")
        for name, ms in sorted(others, key=lambda kv: -kv[1])[:15]:
            print(f"  {ms:8.1f}  {name[:120]}")


if __name__ == "__main__":
    main()
