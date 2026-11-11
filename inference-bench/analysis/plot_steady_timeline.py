"""One long steady level over time (steady_client --windows N --sample-interval S), several runs
overlaid, as a 4 x 2 grid of small panels sharing one legend (one color per run):

    throughput per window             | average step time (all timed steps)
    clean decode step                 | interfered decode step (another EP rank prefills)
    prefill step (this rank prefills) | share of steps with a prefill on any EP rank
    mean context per request per rank | output length CDF of finished requests
    (line: mean over ranks; band: min to max over ranks)

    python3 inference-bench/analysis/plot_steady_timeline.py <exp dir> [<exp dir> ...] [--out PNG]
Writes --out, or <first exp dir>/steady_timeline.png.
"""

import argparse
import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]


def load(exp_dir):
    name = os.path.basename(os.path.normpath(exp_dir))
    gpus = 64 if "wide" in name else 8
    label = ("WildChat" if "wildchat" in name else "math") + (
        ", wide EP (16 DP ranks)" if gpus == 64 else ", narrow EP (2 DP ranks)")
    label = ("vLLM, " if "vllm" in name else "MInf, ") + label
    if "osl1k" in name:
        label = label.replace("WildChat", "WildChat, fixed 1K OSL")
    read = lambda f: [json.loads(line) for line in open(os.path.join(exp_dir, f))]  # noqa: E731
    requests = read("steady.requests.jsonl")
    return {"gpus": gpus, "label": label, "windows": read("steady.jsonl"),
            "samples": read("steady.engines.jsonl"), "requests": requests,
            "t0": min(r["start"] for r in requests)}


def step_ms(window, cls=None):
    """Mean step time of one class in a window, or over all timed steps (cls=None)."""
    h = window["step_time_ms"].get("all", {})
    if cls is not None:
        return h.get(cls, {}).get("mean_ms", float("nan"))
    n = sum(c["count"] for c in h.values())
    return sum(c["count"] * c["mean_ms"] for c in h.values()) / n if n else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dirs", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fig, axes = plt.subplots(4, 2, figsize=(11, 12.5))
    (ax_tput, ax_avg), (ax_clean, ax_intf), (ax_pre, ax_share), (ax_ctx, ax_osl) = axes
    time_axes = (ax_tput, ax_avg, ax_clean, ax_intf, ax_pre, ax_share, ax_ctx)
    handles = []
    for i, d in enumerate(args.exp_dirs):
        run = load(d)
        color = COLORS[i % len(COLORS)]
        w = run["windows"]
        xs = [(x["window_start_s"] + x["window_s"] / 2) / 60 for x in w]
        style = dict(color=color, marker="o", lw=2, ms=4)
        (h,) = ax_tput.plot(xs, [x["output_tok_s"] / run["gpus"] for x in w], **style)
        handles.append((h, run["label"]))
        ax_avg.plot(xs, [step_ms(x) for x in w], **style)
        ax_clean.plot(xs, [step_ms(x, "decode_clean") for x in w], **style)
        ax_intf.plot(xs, [step_ms(x, "decode_interfered") for x in w], **style)
        ax_pre.plot(xs, [step_ms(x, "prefill") for x in w], **style)
        # Share of engine steps that are interfered or prefill (a prefill anywhere in the EP group).
        def prefill_share(x):
            h = x["step_time_ms"].get("all", {})
            n = sum(c["count"] for c in h.values())
            k = sum(h.get(c, {}).get("count", 0) for c in ("decode_interfered", "prefill"))
            return 100 * k / n if n else float("nan")

        ax_share.plot(xs, [prefill_share(x) for x in w], **style)

        # Per-engine KV gauges (MInf only; vLLM reports none, so its samples are empty).
        samples = [s for s in run["samples"]
                   if s["engines"] and all("kv_tokens" in e for e in s["engines"].values())]
        if samples:
            ts = [(s["t"] - run["t0"]) / 60 for s in samples]
            ctx = [[e["kv_tokens"] / 1e3 / max(e["requests_running"], 1) for e in s["engines"].values()]
                   for s in samples]
            ax_ctx.fill_between(ts, [min(c) for c in ctx], [max(c) for c in ctx], color=color,
                                alpha=0.2, lw=0)
            ax_ctx.plot(ts, [statistics.mean(c) for c in ctx], color=color, lw=2)
            # Longest active request across ranks, where the engine reports it (dashed).
            mx = [(s["t"] - run["t0"]) / 60 for s in samples
                  if all("max_context_tokens" in e for e in s["engines"].values())]
            if mx:
                ax_ctx.plot(mx, [max(e["max_context_tokens"] for e in s["engines"].values()) / 1e3
                                 for s in samples
                                 if all("max_context_tokens" in e for e in s["engines"].values())],
                            color=color, lw=1.5, ls="--")

        osl = sorted(r["output_tokens"] for r in run["requests"])
        n = len(osl)
        ax_osl.plot([o / 1e3 for o in osl], [(k + 1) / n for k in range(n)], color=color, lw=2)
        for q in (0.5, 0.9):
            ax_osl.plot([osl[int(q * n)] / 1e3], [q], marker="o", color=color, ms=5)

    titles = {
        ax_tput: "Throughput (tokens/s/GPU)",
        ax_avg: "Average step, all timed steps (ms): what throughput follows",
        ax_clean: "Clean decode step (ms): no prefill on any EP rank",
        ax_intf: "Interfered decode step (ms): another EP rank prefills",
        ax_pre: "Prefill step (ms): this rank prefills",
        ax_share: "Steps with a prefill on any EP rank (% of steps)",
        ax_ctx: "Context per request per DP rank (K tokens)\nsolid: mean over ranks, band: min to max; dashed: longest request",
        ax_osl: "Output length of finished requests (CDF; dots: p50, p90)",
    }
    for ax, title in titles.items():
        ax.set_title(title, fontsize=9, loc="left")
        ax.yaxis.grid(ls="dotted")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    for ax in time_axes:
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
    for ax in (ax_pre, ax_share, ax_ctx):
        ax.set_xlabel("Minutes since the level started", fontsize=8)
    ax_osl.set_xlabel("Output tokens (K)", fontsize=8)
    ax_osl.set_ylim(0, 1)
    ax_osl.set_xlim(left=0)
    for q in (0.5, 0.9):
        ax_osl.axhline(q, color="0.6", lw=0.8, ls="--", zorder=0)
    fig.suptitle("Super, 32 requests per GPU, one level for an hour", y=0.995, fontsize=10)
    fig.legend([h for h, _ in handles], [lbl for _, lbl in handles], loc="upper center",
               bbox_to_anchor=(0.5, 0.975), ncol=len(handles), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = args.out or os.path.join(args.exp_dirs[0], "steady_timeline.png")
    fig.savefig(path, dpi=150)
    print(path)


if __name__ == "__main__":
    main()
