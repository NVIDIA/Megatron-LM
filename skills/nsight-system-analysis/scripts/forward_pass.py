# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-forward-pass (decode-step) analysis for nsys traces.

Everything in an inference trace except the steady-state decode loop is noise:
model load, warmup, prefill, and long idle stretches while the server waits for
the client. This script throws all of that away and reports insights about
*one* forward pass (one decode step), for one or two profiles.

What it does, fully automatically:
  1. Picks the busiest GPU/rank (one deviceId) so ranks don't interleave.
  2. Finds the steady-state decode region (densest contiguous kernel window).
  3. Finds a per-step "anchor" kernel: one that fires exactly once per forward
     pass (embedding / LM-head / sampling / metadata). Its inter-arrival period
     IS the forward-pass wall time (and matches TPOT).
  4. Extracts exactly one representative step (a typical anchor-to-anchor
     interval) and reports:
       - forward-pass wall time, GPU-busy (interval union), idle gaps
       - total kernel launches in the step
       - per-category breakdown: launches, GPU-time (Σ durations), µs/kernel

With two profiles it prints a side-by-side comparison and, per category, the
delta in launches and time, so you can answer the two questions that matter:
  (a) is any individual kernel slower in one engine, or
  (b) does one engine just launch MORE kernels for the same work?

Usage:
  # one profile
  python scripts/forward_pass.py mcore.sqlite

  # compare two (label order is preserved in the output)
  python scripts/forward_pass.py mcore.sqlite vllm.sqlite \
      --label-a mcore --label-b vllm

  # override auto-detection (seconds match the Nsight GUI timeline; raw_ns/1e9)
  python scripts/forward_pass.py mcore.sqlite \
      --anchor _fused_metadata_kernel
  python scripts/forward_pass.py vllm.sqlite --step-window 156.9498,156.9543

  # machine-readable
  python scripts/forward_pass.py mcore.sqlite vllm.sqlite --json

Notes:
  - "GPU-time (Σ durations)" per category is the sum of kernel durations; it
    intentionally over-counts wall time when kernels overlap across streams
    (that overlap is the point of comparing composition). Use the reported
    GPU-busy (interval union) for the true wall figure.
  - The default categories are a sensible inference taxonomy. Pass --yaml
    <taxonomy.yml> to override with the skill's regex taxonomy instead.
"""

from __future__ import annotations

import argparse
import statistics
import sys

from _lib import classify, clip, err, load_taxonomy, open_sqlite, union_total_ns, write_json

# Ordered, first-match-wins default taxonomy (case-insensitive substring match).
# Tuned for MoE decode inference (mcore + vLLM). Override with --yaml.
_DEFAULT_CATEGORIES: list[tuple[str, tuple[str, ...]]] = [
    ("comm (EP dispatch/combine)", ("multimem", "nccl", "all_gather", "allgather",
                                    "reduce_scatter", "reducescatter", "alltoall", "sendrecv")),
    ("attention", ("flash", "fmha", "attention", "attn", "paged", "mha")),
    ("MoE expert GEMM", ("_fused_moe_kernel", "bmm_")),
    ("MoE routing/permute", ("routing", "topk", "gathertopk", "moe_sum", "count_local",
                             "metadata", "finalize", "scatter", "permute", "sort",
                             "cumsum", "histogram", "softmax")),
    ("activation (SwiGLU)", ("silu", "act_and_mul", "swiglu", "glu", "gelu")),
    ("norm (RMSNorm)", ("rmsnorm", "rms_norm", "layernorm", "layer_norm", "norm_fwd", "rsqrt")),
    ("dense GEMM (qkv/o/router/lm-head)", ("nvjet", "cutlass", "sgemm", "gemm",
                                           "sm100_tst", "sm90", "cublas", "ampere")),
    ("sampling/logits", ("reduce_kernel", "argmax", "multinomial", "sample", "catarray", "cat_")),
    ("embedding", ("embedding",)),
    ("elementwise/copy/cast", ("elementwise", "vectorized", "copy", "index_", "triton_poi",
                               "triton_red", "fused", "cast", "_add")),
]


def _categorize(name: str, taxonomy) -> str:
    if taxonomy is not None:
        return classify(name, taxonomy) or "other/misc"
    low = name.lower()
    for cat, pats in _DEFAULT_CATEGORIES:
        if any(p in low for p in pats):
            return cat
    return "other/misc"


def _kernels(con, dev=None):
    """(start, end, deviceId, name) ordered by start, optionally one device."""
    where = "1=1" if dev is None else f"k.deviceId={int(dev)}"
    cur = con.execute(
        f"""SELECT k.start, k.end, k.deviceId,
                   COALESCE((SELECT value FROM StringIds WHERE id=k.demangledName), '')
            FROM CUPTI_ACTIVITY_KIND_KERNEL k WHERE {where} ORDER BY k.start"""
    )
    return cur.fetchall()


def _busiest_device(con) -> int:
    row = con.execute(
        "SELECT deviceId, COUNT(*) c FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "GROUP BY deviceId ORDER BY c DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise SystemExit("ERROR: no kernels in trace")
    return row[0]


def _decode_window(starts: list[int], bin_ns: int = 1_000_000_000) -> tuple[int, int]:
    """Densest contiguous region of kernel launches = steady-state decode."""
    mn, mx = starts[0], starts[-1]
    nb = (mx - mn) // bin_ns + 1
    cnt = [0] * nb
    for s in starts:
        cnt[(s - mn) // bin_ns] += 1
    pk = max(range(nb), key=lambda i: cnt[i])
    thr = 0.3 * cnt[pk]
    lo = hi = pk
    while lo > 0 and cnt[lo - 1] >= thr:
        lo -= 1
    while hi < nb - 1 and cnt[hi + 1] >= thr:
        hi += 1
    return mn + lo * bin_ns, mn + (hi + 1) * bin_ns


# Kernels that fire ~once per forward pass (step boundaries), across engines.
_STEP_HINTS = ("metadata", "to_copy_embedding", "embedding", "lm_head", "logits",
               "argmax", "multinomial", "sample")


def _gap_stats(ss):
    ss = sorted(ss)
    gaps = [b - a for a, b in zip(ss, ss[1:])]
    m = statistics.median(gaps)
    cv = (statistics.pstdev(gaps) / m) if m else 9e9
    return ss, m, cv


def _find_anchor(win_kernels, window_ns, forced=None):
    """Return (anchor_name, sorted_starts_on_one_device, period_ns).

    A per-step anchor fires ~once per forward pass. Per-layer kernels fire
    n_layers× more often, so among *low-jitter* recurring kernels the per-step
    marker has the *longest* inter-arrival period. We therefore:
      1. prefer a kernel whose name matches a known step-boundary marker
         (embedding / lm-head / sampling / metadata), else
      2. pick the longest-period, low-jitter recurring kernel (a sane [0.5 ms,
         window/3] band, so it spans several steps but isn't per-layer).

    win_kernels spans all devices (the marker may live on a different rank than
    the busiest compute device), but the *period* is always measured on a
    SINGLE device — otherwise N ranks firing the same marker once per step,
    staggered, would collapse the apparent period by ~N×.
    """
    from collections import defaultdict

    dev_starts = defaultdict(lambda: defaultdict(list))  # name -> device -> [starts]
    for s, _e, d, name in win_kernels:
        dev_starts[name][d].append(s)

    # For each name, evaluate it on the single device where it fires most.
    stats = {}  # name -> (starts_on_best_dev, period_ns, cv)
    for name, per_dev in dev_starts.items():
        best_dev = max(per_dev, key=lambda d: len(per_dev[d]))
        ss = per_dev[best_dev]
        if len(ss) < 15:
            continue
        s, m, cv = _gap_stats(ss)
        stats[name] = (s, m, cv)
    if not stats:
        raise SystemExit("ERROR: no recurring kernel found in decode window; pass --anchor/--step-window")

    if forced is not None:
        cands = {n: v for n, v in stats.items() if forced.lower() in n.lower()}
        if not cands:
            raise SystemExit(f"ERROR: --anchor '{forced}' matched no recurring kernel")
        name = min(cands, key=lambda n: cands[n][2])  # lowest jitter
        s, m, _ = cands[name]
        return name, s, m

    lo, hi = 0.5e6, window_ns / 3  # plausible forward-pass period band (ns)
    periodic = {n: v for n, v in stats.items() if v[2] < 0.5 and lo <= v[1] <= hi}
    if not periodic:  # relax the band if nothing qualifies
        periodic = {n: v for n, v in stats.items() if v[1] > 0}

    hinted = {n: v for n, v in periodic.items() if any(h in n.lower() for h in _STEP_HINTS)}
    pool = hinted or periodic
    # longest low-jitter period == fires least often == once per step
    name = max(pool, key=lambda n: pool[n][1])
    s, m, _ = pool[name]
    return name, s, m


def analyze(path: str, taxonomy, forced_anchor=None, forced_dev=None, forced_window=None):
    with open_sqlite(path) as con:
        if forced_window is not None:
            w0, w1 = forced_window
            dev = forced_dev
            if dev is None:
                row = con.execute(
                    "SELECT deviceId, COUNT(*) c FROM CUPTI_ACTIVITY_KIND_KERNEL "
                    "WHERE start>=? AND start<? GROUP BY deviceId ORDER BY c DESC LIMIT 1",
                    (w0, w1),
                ).fetchone()
                dev = row[0] if row else _busiest_device(con)
        else:
            dev = forced_dev if forced_dev is not None else _busiest_device(con)
            all_starts = [k[0] for k in _kernels(con, dev)]
            if not all_starts:
                raise SystemExit(f"ERROR: no kernels on device {dev}")
            w0, w1 = _decode_window(all_starts)

        # anchor detection spans ALL devices (the per-step marker may live on a
        # different rank than the busiest compute device); the step window is
        # then applied to `dev` for kernel extraction.
        win_all = [k for k in _kernels(con, None) if w0 <= k[0] < w1]
        anchor, astarts, period = _find_anchor(win_all, w1 - w0, forced_anchor)

        # representative step = anchor interval whose duration is closest to the
        # median period, biased toward the middle of the decode window.
        pairs = list(zip(astarts, astarts[1:]))
        mid = astarts[len(astarts) // 2]
        t0, t1 = min(pairs, key=lambda p: (abs((p[1] - p[0]) - period), abs(p[0] - mid)))

        step = [(s, e, name) for s, e, _d, name in _kernels(con, dev) if t0 <= s < t1]
        n_steps = max(1, round((w1 - w0) / period))

    ivals = [(s, e) for s, e, _ in step]
    wall = t1 - t0
    busy = union_total_ns(ivals)
    from collections import defaultdict
    cats: dict[str, list] = defaultdict(lambda: [0, 0])  # cat -> [count, dur_ns]
    for s, e, name in step:
        c = _categorize(name, taxonomy)
        cats[c][0] += 1
        cats[c][1] += e - s

    return {
        "profile": path,
        "device": dev,
        "decode_window_s": [w0 / 1e9, w1 / 1e9],
        "steps_in_window": n_steps,
        "anchor": anchor,
        "forward_pass_ms": period / 1e6,
        "step_wall_ms": wall / 1e6,
        "gpu_busy_ms": busy / 1e6,
        "gpu_idle_ms": (wall - busy) / 1e6,
        "n_kernels": len(step),
        "categories": {
            c: {"launches": v[0], "gpu_time_us": v[1] / 1e3,
                "us_per_kernel": (v[1] / 1e3 / v[0]) if v[0] else 0.0}
            for c, v in cats.items()
        },
    }


def _print_one(a: dict, label: str):
    print(f"\n=== {label}  ({a['profile'].split('/')[-1]}) ===")
    print(f"  device={a['device']}  decode window {a['decode_window_s'][0]:.1f}-{a['decode_window_s'][1]:.1f}s"
          f"  (~{a['steps_in_window']} steps)   anchor: {a['anchor'][:48]}")
    print(f"  forward pass = {a['forward_pass_ms']:.3f} ms   step wall = {a['step_wall_ms']:.3f} ms"
          f"   GPU-busy = {a['gpu_busy_ms']:.3f} ms   idle = {a['gpu_idle_ms']:.3f} ms")
    print(f"  kernels in one forward pass = {a['n_kernels']}")
    rows = sorted(a["categories"].items(), key=lambda kv: -kv[1]["gpu_time_us"])
    print(f"  {'category':34} {'#':>5} {'GPU us':>9} {'us/kern':>8}")
    for c, v in rows:
        print(f"  {c:34} {v['launches']:>5} {v['gpu_time_us']:>9.1f} {v['us_per_kernel']:>8.1f}")


def _print_compare(a: dict, b: dict, la: str, lb: str):
    print(f"\n=== COMPARISON: {la} vs {lb} (one forward pass) ===")
    print(f"  forward pass : {a['forward_pass_ms']:.3f} ms ({la})  vs  {b['forward_pass_ms']:.3f} ms ({lb})"
          f"   -> {la} is {a['forward_pass_ms']/b['forward_pass_ms']:.2f}x")
    print(f"  kernels/step : {a['n_kernels']} ({la})  vs  {b['n_kernels']} ({lb})"
          f"   -> {la} launches {a['n_kernels']/max(b['n_kernels'],1):.2f}x")
    allc = list(dict.fromkeys(list(a["categories"]) + list(b["categories"])))
    def g(d, c): return d["categories"].get(c, {"launches": 0, "gpu_time_us": 0.0, "us_per_kernel": 0.0})
    allc.sort(key=lambda c: -(g(a, c)["gpu_time_us"] - g(b, c)["gpu_time_us"]))
    print(f"\n  {'category':34} | {lb+' #':>7} {lb+' us':>9} {'us/k':>6} | {la+' #':>7} {la+' us':>9} {'us/k':>6} | {'Δus':>8}")
    print("  " + "-" * 104)
    for c in allc:
        av, bv = g(a, c), g(b, c)
        print(f"  {c:34} | {bv['launches']:>7} {bv['gpu_time_us']:>9.1f} {bv['us_per_kernel']:>6.1f} |"
              f" {av['launches']:>7} {av['gpu_time_us']:>9.1f} {av['us_per_kernel']:>6.1f} |"
              f" {av['gpu_time_us']-bv['gpu_time_us']:>8.1f}")
    print("\n  Read: 'Δus' > 0 = category costs more in %s. Compare 'us/k' to see if a kernel is\n"
          "  individually slower, vs '#' to see if it's just more launches for the same work." % la)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Single forward-pass (decode step) analysis for nsys sqlite traces.")
    ap.add_argument("profiles", nargs="+", help="1 or 2 .sqlite profiles (A then optional B).")
    ap.add_argument("--anchor", default=None, help="Substring of a once-per-step kernel to force the anchor.")
    ap.add_argument("--device", type=int, default=None, help="Force GPU deviceId (default: busiest).")
    ap.add_argument("--step-window", default=None,
                    help="Force window 't0,t1' in seconds (Nsight GUI time = raw_ns/1e9).")
    ap.add_argument("--yaml", default=None, help="Override default taxonomy with a regex YAML.")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of tables.")
    args = ap.parse_args(argv)

    if len(args.profiles) > 2:
        err("WARNING: more than 2 profiles given; only the first two are used.")
    taxonomy = load_taxonomy(args.yaml) if args.yaml else None
    fw = None
    if args.step_window:
        t0, t1 = (float(x) for x in args.step_window.split(","))
        fw = (int(t0 * 1e9), int(t1 * 1e9))

    a = analyze(args.profiles[0], taxonomy, args.anchor, args.device, fw)
    b = analyze(args.profiles[1], taxonomy, args.anchor, args.device, fw) if len(args.profiles) >= 2 else None

    if args.json:
        write_json({"a": a, "b": b})
        return
    _print_one(a, args.label_a)
    if b is not None:
        _print_one(b, args.label_b)
        _print_compare(a, b, args.label_a, args.label_b)


if __name__ == "__main__":
    sys.exit(main())
