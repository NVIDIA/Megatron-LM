"""Engine step time vs. global active tokens, from per-step server logs (LOG_INTERVAL=1).

Each data-parallel group's lead rank logs its own real steps (dummy steps are not logged):
    * rank R | step N | HH:MM:SS ... time: X ms [decode + real config [T]: P P + D D + ...]
Step numbers are not aligned across groups, but all groups step in lockstep (every MoE layer
synchronizes the whole EP group), so steps are joined on their wall-clock second:

    global_tokens(second) = sum over groups of that group's median T in that second
                            (a group with no logged step that second was idle: 0 tokens)
    step_ms(second)       = median step time over all logged steps in that second

Only seconds where every logged step is decode-only (no prefill) are kept. As a sanity check on
the logged time, steps/second of the busiest group is compared with 1000 / step_ms.

    python3 inference-bench/analysis/step_time_vs_tokens.py \
        nvl_a2a=inference-bench/experiments/<run_a> [nvls=inference-bench/experiments/<run_b>] \
        --out inference-bench/analysis/step_time_vs_tokens.png
"""

import argparse
import csv
import glob
import math
import os
import re
import statistics as st
from collections import defaultdict

STEP_RE = re.compile(
    r"\* rank (\d+) \| step (\d+) \| (\d\d:\d\d:\d\d) \.\.\. time: ([\d.]+) ms "
    r"\[(\S+) \+ real config \[(\d+)\]: (\d+) P \+ (\d+) D"
)

# Okabe-Ito, colorblind-safe; fixed per series in argument order (color follows the run).
STYLES = [("#0072B2", "o"), ("#D55E00", "^"), ("#009E73", "s"), ("#CC79A7", "D")]


def parse_run(exp_dir):
    """{second: [(rank, ms, T, P, D)]} over all ranks' logs."""
    per_second = defaultdict(list)
    # Every rank logs its group's steps; one lead per TP group (ranks 0, tp, 2*tp, ...) suffices.
    tp = int(re.search(r"\bTP=(\d+)", open(os.path.join(exp_dir, "config.env")).read()).group(1))
    paths = glob.glob(os.path.join(exp_dir, "server_rank*.log"))
    for path in [q for q in paths if int(re.search(r"rank(\d+)", q).group(1)) % tp == 0]:
        with open(path, errors="replace") as f:
            for line in f:
                if "| step " not in line:
                    continue
                m = STEP_RE.search(line)
                if not m:
                    continue
                rank, _, sec, ms, _, total, prefill, decode = m.groups()
                per_second[sec].append((int(rank), float(ms), int(total), int(prefill), int(decode)))
    return per_second


def aggregate(per_second):
    rows = []
    for sec, steps in sorted(per_second.items()):
        if any(p > 0 for _, _, _, p, _ in steps):
            continue  # prefill (or mixed) work happened this second
        by_rank = defaultdict(list)
        for rank, ms, total, _, _ in steps:
            by_rank[rank].append((ms, total))
        global_tokens = sum(st.median(t for _, t in v) for v in by_rank.values())
        if global_tokens == 0:
            continue
        step_ms = st.median(ms for _, ms, _, _, _ in steps)
        max_ms = max(ms for _, ms, _, _, _ in steps)
        busiest = max(len(v) for v in by_rank.values())
        rows.append(
            {
                "second": sec,
                "global_tokens": global_tokens,
                "active_groups": len(by_rank),
                "step_ms": step_ms,
                "max_step_ms": max_ms,
                "steps_in_second": busiest,
                "ms_from_step_rate": 1000.0 / busiest,
            }
        )
    return rows


def drop_stalls(rows):
    """Drop seconds containing a host-side stall: any step > max(200 ms, 4x its bucket median)."""
    bins = defaultdict(list)
    for r in rows:
        bins[bucket(r["global_tokens"])].append(r["step_ms"])
    med = {b: st.median(v) for b, v in bins.items()}
    return [r for r in rows if r["max_step_ms"] <= max(200, 4 * med[bucket(r["global_tokens"])])]


def bucket(tokens):
    return 2 ** round(math.log2(tokens))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="label=experiment_dir")
    parser.add_argument("--out", default="inference-bench/analysis/step_time_vs_tokens.png")
    parser.add_argument("--min-tokens", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1 << 30)
    parser.add_argument("--logy", action="store_true", help="log-scale step-time axis")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import logging

    import matplotlib.pyplot as plt
    import matplotlib.ticker

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    plt.rcParams["font.family"] = [
        "Gill Sans", "Gill Sans MT", "URW Gothic", "Open Sans", "DejaVu Sans"
    ]
    # With two runs, a second panel shows the first run's step time relative to the second's.
    if len(args.runs) == 2:
        fig, (ax, dax) = plt.subplots(
            2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]}
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        dax = None
    medians = []
    for i, spec in enumerate(args.runs):
        label, exp_dir = spec.split("=", 1)
        all_rows = aggregate(parse_run(exp_dir))
        rows = [r for r in drop_stalls(all_rows)
                if args.min_tokens <= bucket(r["global_tokens"]) <= args.max_tokens]
        csv_path = os.path.splitext(args.out)[0] + f"_{label}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        color, marker = STYLES[i % len(STYLES)]
        xs = [r["global_tokens"] for r in rows]
        ys = [r["step_ms"] for r in rows]
        ax.scatter(xs, ys, s=14, alpha=0.35, color=color, marker=marker, linewidths=0)
        # Median per power-of-2 bin of global tokens, drawn as the series' line.
        bins = defaultdict(list)
        for x, y in zip(xs, ys):
            bins[bucket(x)].append(y)
        bx = sorted(bins)
        medians.append((label, {b: st.median(bins[b]) for b in bx}, bins))
        ax.plot(bx, [st.median(bins[b]) for b in bx], color=color, marker=marker,
                markersize=6, linewidth=2, label=label,
                linestyle="-" if i == 0 else "--")
        rate_ratio = st.median(r["ms_from_step_rate"] / r["step_ms"] for r in rows)
        print(f"{label}: dropped {len(all_rows) - len(drop_stalls(all_rows))} stall seconds; "
              f"{len(rows)} steady decode seconds; tokens {min(xs):.0f}-{max(xs):.0f}; "
              f"step-rate/logged-time ratio (median) {rate_ratio:.2f}; csv -> {csv_path}")
        for b in bx:
            print(f"  ~{b:6d} tokens: median step {st.median(bins[b]):7.2f} ms  (n={len(bins[b])})")

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    xlabel = "Global active tokens per step (sum over DP groups)"
    if dax is None:
        ax.set_xlabel(xlabel)
    else:
        (la, ma, ba), (lb, mb, bb) = medians
        common = sorted(set(ma) & set(mb))
        diff = [100 * (ma[b] / mb[b] - 1) for b in common]
        # Spread: half the gap between the 25th and 75th percentile ratios of the two bins.
        print(f"{la} vs {lb} (median step, + = {la} slower):")
        for b, d in zip(common, diff):
            print(f"  ~{b:6d} tokens: {ma[b]:7.2f} vs {mb[b]:7.2f} ms  {ma[b] - mb[b]:+6.2f} ms  {d:+5.1f}%")
        dax.axhline(0, color="#999999", linewidth=1)
        dax.plot(common, diff, color="#555555", marker="o", markersize=6, linewidth=2)
        for b, d in zip(common, diff):
            dax.annotate(f"{d:+.1f}%", (b, d), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=8, color="#333333")
        lim = max(3, max(abs(d) for d in diff) * 1.6)
        dax.set_ylim(-lim, lim)
        dax.set_ylabel(f"{la} vs {lb} (%)")
        dax.set_xlabel(xlabel)
        dax.grid(True, which="major", color="#dddddd", linewidth=0.8)
        for side in ("top", "right"):
            dax.spines[side].set_visible(False)
    ax.set_ylabel("Engine step time (ms)")
    ax.set_title("Ultra v3, TP4 x EP64 (16 nodes): decode step time vs global batch")
    ax.grid(True, which="major", color="#dddddd", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if args.logy:
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128]))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    else:
        ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"plot -> {args.out}")


if __name__ == "__main__":
    main()
