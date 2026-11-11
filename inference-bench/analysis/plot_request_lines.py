"""Per-request context timelines from steady_client's engine samples (--sample-interval S): one
thin line per request (its exact context, prompt + generated, at every sample it was active), with
the max, p90, p50, and mean over all active requests at each sample overlaid; throughput per window
above, on the same time axis. --per-engine splits the context panel by DP rank (engine).

    python3 inference-bench/analysis/plot_request_lines.py <exp dir> --out <png> [--per-engine]
"""

import argparse
import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-engine", action="store_true", help="one context panel per DP rank (engine)")
    args = ap.parse_args()
    d = args.exp_dir
    read = lambda f: [json.loads(line) for line in open(os.path.join(d, f))]  # noqa: E731
    windows, samples = read("steady.jsonl"), read("steady.engines.jsonl")
    samples = [s for s in samples if any("requests" in e for e in s["engines"].values())]
    t0 = min(r["start"] for r in read("steady.requests.jsonl"))
    gpus = 64 if "wide" in d else 8

    engines = sorted({e for s in samples for e in s["engines"]})
    groups = [(e, [e]) for e in engines] if args.per_engine else [("all DP ranks", engines)]

    fig, axes = plt.subplots(2 + len(groups), 1, figsize=(8, 5 + 3.2 * len(groups)), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1] + [2] * len(groups)})
    ax_t, ax_c = axes[0], axes[1]
    axes = [ax_t] + list(axes[2:])
    xs = [(w["window_start_s"] + w["window_s"] / 2) / 60 for w in windows]
    ax_t.plot(xs, [w["output_tok_s"] / gpus for w in windows], color="#0072B2", marker="o", lw=2)
    ax_t.set_title("Throughput (tokens/s/GPU), one point per 5-minute window"
                   + ("" if windows else ": no window finished yet"), fontsize=9, loc="left")
    clean = [w["step_time_ms"].get("all", {}).get("decode_clean", {}).get("mean_ms", float("nan"))
             for w in windows]
    ax_c.plot(xs, clean, color="#CC79A7", marker="o", lw=2)
    ax_c.set_title("Clean decode step (ms): no prefill on any EP rank, mean per 5-minute window",
                   fontsize=9, loc="left")
    ax_c.set_ylim(bottom=0)
    ax_c.yaxis.grid(ls="dotted")
    ax_c.spines[["top", "right"]].set_visible(False)
    for ax_r, (name, members) in zip(axes[1:], groups):
        # (engine, request id) -> [(minutes, K tokens)]
        lines, ts, stats = {}, [], {"max": [], "p90": [], "p50": [], "mean": []}
        for s in samples:
            t = (s["t"] - t0) / 60
            ctx = []
            for engine in members:
                for rid, tokens in s["engines"].get(engine, {}).get("requests", {}).items():
                    lines.setdefault((engine, rid), []).append((t, tokens / 1e3))
                    ctx.append(tokens / 1e3)
            if not ctx:
                continue
            ctx.sort()
            ts.append(t)
            stats["max"].append(ctx[-1])
            stats["p90"].append(ctx[int(0.9 * (len(ctx) - 1))])
            stats["p50"].append(ctx[(len(ctx) - 1) // 2])
            stats["mean"].append(statistics.mean(ctx))
        ax_r.add_collection(LineCollection([pts for pts in lines.values() if len(pts) > 1],
                                           colors="0.55", linewidths=0.5, alpha=0.35))
        for key, style in (("max", dict(color="#D55E00", lw=2.2)), ("p90", dict(color="#E69F00", lw=1.6)),
                           ("p50", dict(color="#009E73", lw=1.6)),
                           ("mean", dict(color="black", lw=1.5, ls="--"))):
            ax_r.plot(ts, stats[key], label=key, **style)
        ax_r.plot([], [], color="0.55", lw=0.8, label="each request")
        ax_r.set_title(f"Context per active request (tokens), {name}: {len(lines)} requests",
                       fontsize=9, loc="left")
        ax_r.autoscale_view()
        ax_r.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}k"))
    axes[1].legend(frameon=False, fontsize=8, ncol=5, loc="upper left")
    axes[-1].set_xlabel("Minutes since the level started")
    ymax = max(ax.get_ylim()[1] for ax in axes[1:])
    for ax in axes:
        ax.set_ylim(0, ymax if ax is not ax_t else None)
        ax.set_xlim(left=0)
        ax.yaxis.grid(ls="dotted")
        ax.spines[["top", "right"]].set_visible(False)
    ax_t.set_ylim(bottom=0)
    fig.suptitle(os.path.basename(os.path.normpath(d)), fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(args.out)


if __name__ == "__main__":
    main()
