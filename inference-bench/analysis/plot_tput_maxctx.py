"""Throughput per window and the distribution of active requests' context length (min, p50,
p90, p99, max; mean) over one long steady
level: two panels on a shared time axis.

    python3 inference-bench/analysis/plot_tput_maxctx.py <exp dir> --out <png>
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = args.exp_dir
    read = lambda f: [json.loads(line) for line in open(os.path.join(d, f))]  # noqa: E731
    windows, samples = read("steady.jsonl"), read("steady.engines.jsonl")
    t0 = min(r["start"] for r in read("steady.requests.jsonl"))
    gpus = 64 if "wide" in d else 8

    fig, (ax_t, ax_m) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    xs = [(w["window_start_s"] + w["window_s"] / 2) / 60 for w in windows]
    ax_t.plot(xs, [w["output_tok_s"] / gpus for w in windows], color="#0072B2", marker="o", lw=2)
    ax_t.set_title("Throughput (tokens/s/GPU), one point per 5-minute window", fontsize=9, loc="left")
    ts = [(s["t"] - t0) / 60 for s in samples]
    # Per sample, across DP ranks: the max of each rank's percentile (the slowest rank's view).
    def stat(s, key):
        vals = [e[key] for e in s["engines"].values() if key in e]
        return max(vals) / 1e3 if vals else float("nan")

    def mean_ctx(s):
        e = s["engines"].values()
        return sum(x["kv_tokens"] for x in e) / max(sum(x["requests_running"] for x in e), 1) / 1e3

    for key, label, style in (("max_context_tokens", "max", dict(color="#D55E00", lw=2)),
                              ("p99_context_tokens", "p99", dict(color="#CC79A7", lw=1.5)),
                              ("p90_context_tokens", "p90", dict(color="#E69F00", lw=1.5)),
                              ("p50_context_tokens", "p50", dict(color="#009E73", lw=1.5)),
                              ("min_context_tokens", "min", dict(color="0.5", lw=1.2))):
        ax_m.plot(ts, [stat(s, key) for s in samples], label=label, **style)
    ax_m.plot(ts, [mean_ctx(s) for s in samples], color="black", lw=1.5, ls="--", label="mean")
    ax_m.legend(frameon=False, fontsize=8, ncol=6)
    ax_m.set_title("Active requests' context length (K tokens), sampled every minute;\n"
                   "each line: the highest value across DP ranks (mean: over all requests)",
                   fontsize=9, loc="left")
    ax_m.set_xlabel("Minutes since the level started")
    for ax in (ax_t, ax_m):
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        ax.yaxis.grid(ls="dotted")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(os.path.basename(os.path.normpath(d)), fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(args.out)


if __name__ == "__main__":
    main()
