"""Prefill-decode interference: aggregated vs disaggregated (PD) serving, narrow vs wide EP.

Reads the steady-state client's raw output of each run (steady.jsonl windows, steady.requests.jsonl
per-request times, config.env GPU counts) and writes, for runs labeled {narrow,wide}_{agg,pd} (plus
{narrow,wide}_agg_nopc, an aggregated control without prefix caching):

  pareto.png    total (input + output) tok/s/GPU vs p90 interactivity (tok/s/user of the slowest
                10% of requests: 1000 / p90 of end-to-end latency per output token)
  step_mix.png  share of engine time by step class per run and concurrency (decode pool for PD)
  latency.png   p90 TTFT and p90 decode TPOT (after the first token) vs concurrency
  summary.csv / summary.json   every number behind the plots, per run and level

Each level pools the requests that finished inside its measurement windows (all windows of the
level), so percentiles are not taken over a handful of requests per window. Requests submitted before the server's first
token (its cold start) are left out of the latency percentiles but not the throughput.

GPU accounting: aggregated runs count their GPUs. A PD run counts its decode GPUs plus its share of
the prefill pool: --prefill-gpus-per-64 prefill GPUs serve 64 decode GPUs (default: the run's own
prefill pool size), so an 8-GPU narrow replica is charged 8/64 of it, as one prefill pool would
serve all eight narrow replicas of a 64-GPU deployment. The as-run count (decode + whole prefill
pool) is in the summary too. Concurrency is reported per decode GPU.

    python3 inference-bench/analysis/plot_pd_interference.py \\
        --run narrow_agg=inference-bench/experiments/<dir> --run narrow_pd=... \\
        --run wide_agg=... --run wide_pd=... --out inference-bench/analysis/pareto/pd_interference
"""

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference categorical palette, slots 1-4 (validated all-pairs for the first three).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WIDTH_COLOR = {"wide": BLUE, "narrow": ORANGE}
MODE_STYLE = {
    "agg": ("o", "-", "aggregated"),
    "agg_nopc": ("D", ":", "aggregated, no prefix caching"),
    "pd": ("s", "--", "disaggregated (PD)"),
}
WIDTH_NAME = {"wide": "Wide EP (EP64)", "narrow": "Narrow EP (EP8)"}
STEP_CLASSES = [  # stacking order, bottom to top; main has no per-step EP split, so plain "decode"
    ("decode", "decode", BLUE),
    ("decode_clean", "decode, no prefill in EP group", BLUE),
    ("decode_interfered", "decode, another EP rank prefills", ORANGE),
    ("mixed", "prefill + decode in batch", AQUA),
    ("prefill_only", "prefill only", YELLOW),
]


def read_env(path):
    """KEY=value pairs of a config.env (several per line, space separated)."""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        for token in line.split():
            if "=" in token:
                key, _, value = token.partition("=")
                out.setdefault(key, value)
    return out


def percentile(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def summarize(label, run_dir, prefill_gpus_per_64):
    cfg = read_env(os.path.join(run_dir, "config.env"))
    gpus = int(cfg.get("GPUS") or 4 * int(cfg["NODES"]))
    # Native PD (one job, one config.env) records PREFILL_GPUS; the earlier two-world PD runs
    # wrote the prefill pool's own config.env under prefill/.
    prefill_cfg = read_env(os.path.join(run_dir, "prefill", "config.env"))
    prefill_gpus = int(cfg.get("PREFILL_GPUS") or 0) or (
        int(prefill_cfg.get("GPUS") or 0) if prefill_cfg else 0
    )
    per64 = prefill_gpus if prefill_gpus_per_64 is None else prefill_gpus_per_64
    charged = gpus + (per64 * gpus / 64 if prefill_gpus else 0)

    windows = [json.loads(line) for line in open(os.path.join(run_dir, "steady.jsonl"))]
    requests = [json.loads(line) for line in open(os.path.join(run_dir, "steady.requests.jsonl"))]
    # Requests submitted before the server's first token anywhere saw its cold start (first-use
    # compilation and warm-up, up to minutes on 64 GPUs), not steady-state serving: keep them out
    # of the latency percentiles. Throughput still counts their tokens.
    first_ready = min((r["first_token"] for r in requests if r.get("first_token")), default=0.0)
    rows = []
    for c in sorted({w["concurrency"] for w in windows}):
        level = [w for w in windows if w["concurrency"] == c]
        span = sum(w["window_s"] for w in level)
        t0, t1 = min(w["window_t0"] for w in level), max(w["window_t1"] for w in level)
        out_tok_s = sum(w["output_tok_s"] * w["window_s"] for w in level) / span
        in_tok_s = sum(w["prompt_tok_s"] * w["window_s"] for w in level) / span
        finished = [r for r in requests if t0 <= r["end"] <= t1 and r["output_tokens"] > 0]
        done = [r for r in finished if r["start"] >= first_ready]
        tpot = [1000 * (r["end"] - r["start"]) / r["output_tokens"] for r in done]
        ttft = [1000 * (r["first_token"] - r["start"]) for r in done if r.get("first_token")]
        decode_tpot = [1000 * (r["end"] - r["first_token"]) / (r["output_tokens"] - 1)
                       for r in done if r.get("first_token") and r["output_tokens"] > 1]
        mix = {}
        for w in level:
            for scope, classes in w.get("step_mix", {}).items():
                for step_class, d in classes.items():
                    acc = mix.setdefault(scope, {}).setdefault(step_class, {"steps": 0, "wall_ms": 0.0})
                    acc["steps"] += d["steps"]
                    acc["wall_ms"] += d["wall_ms"]
        scope = "decode" if "decode" in mix else "all"
        wall = sum(d["wall_ms"] for d in mix.get(scope, {}).values()) or float("nan")
        steps = sum(d["steps"] for d in mix.get(scope, {}).values()) or float("nan")
        row = {
            "run": label, "dataset": cfg.get("STEADY_DATASET", "math"), "dir": run_dir, "concurrency": c, "concurrency_per_64": c * 64 // gpus, "requests_per_gpu": c / gpus,
            "decode_gpus": gpus, "prefill_gpus": prefill_gpus, "charged_gpus": charged,
            "as_run_gpus": gpus + prefill_gpus, "window_s": span, "requests": len(done),
            "cold_start_requests_excluded": len(finished) - len(done),
            "output_tok_s": out_tok_s, "prompt_tok_s": in_tok_s,
            "total_tok_s_per_gpu": (out_tok_s + in_tok_s) / charged,
            "output_tok_s_per_gpu": out_tok_s / charged,
            "total_tok_s_per_gpu_as_run": (out_tok_s + in_tok_s) / (gpus + prefill_gpus),
            "tpot_ms_p50": percentile(tpot, 0.5), "tpot_ms_p90": percentile(tpot, 0.9),
            "user_tok_s_p50": 1000 / percentile(tpot, 0.5) if tpot else float("nan"),
            "user_tok_s_p90": 1000 / percentile(tpot, 0.9) if tpot else float("nan"),
            "ttft_ms_p50": percentile(ttft, 0.5), "ttft_ms_p90": percentile(ttft, 0.9),
            "decode_tpot_ms_p50": percentile(decode_tpot, 0.5),
            "decode_tpot_ms_p90": percentile(decode_tpot, 0.9),
            "step_mix_scope": scope, "step_mix": mix,
        }
        for step_class, _, _ in STEP_CLASSES:
            d = mix.get(scope, {}).get(step_class, {"steps": 0, "wall_ms": 0.0})
            row[f"{step_class}_time_share"] = d["wall_ms"] / wall
            row[f"{step_class}_step_share"] = d["steps"] / steps
            row[f"{step_class}_mean_ms"] = d["wall_ms"] / d["steps"] if d["steps"] else float("nan")
        if "prefill" in mix:  # the prefill pool's busy share is how loaded it is
            row["prefill_pool_steps"] = sum(d["steps"] for d in mix["prefill"].values())
        rows.append(row)
    return rows


def marker_colors(color, mode):
    """Filled markers with a surface ring; the control is hollow so its twin shows through."""
    if mode == "agg_nopc":
        return {"markerfacecolor": "white", "markeredgecolor": color, "markeredgewidth": 1.5}
    return {"markeredgecolor": "white", "markeredgewidth": 1.5}


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_pareto(rows_by_run, path):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label, rows in rows_by_run.items():
        width, _, mode = label.partition("_")
        marker, line, mode_name = MODE_STYLE[mode]
        xs = [r["user_tok_s_p90"] for r in rows]
        ys = [r["total_tok_s_per_gpu"] for r in rows]
        ax.plot(xs, ys, line, color=WIDTH_COLOR[width], marker=marker, markersize=8, linewidth=2,
                **marker_colors(WIDTH_COLOR[width], mode),
                label=f"{WIDTH_NAME[width]}, {mode_name}")
        # Aggregated labels above-right, PD below-left, so paired points don't collide.
        offset, ha = ((6, 4), "left") if mode == "agg" else ((-6, -12), "right")
        for r, x, y in zip(rows, xs, ys):
            if mode == "agg_nopc":
                continue  # sits on its aggregated twin; its levels match
            ax.annotate(f"{r['requests_per_gpu']:g}/GPU", (x, y), textcoords="offset points",
                        xytext=offset, ha=ha, fontsize=8, color=MUTED)
    style_axes(ax)
    ax.set_xlabel("p90 interactivity (tok/s/user, slowest 10% of requests)", color=TEXT)
    ax.set_ylabel("Throughput, input + output (tok/s/GPU)", color=TEXT)
    dataset = next(iter(rows_by_run.values()))[0]["dataset"]
    ax.set_title(f"Super v3 on {dataset}: aggregated vs disaggregated, narrow vs wide EP\n"
                 "labels: concurrent requests per decode GPU", fontsize=10, color=TEXT)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_step_mix(rows_by_run, path):
    """One panel per run (shared y): stacked share of engine time by step class per level."""
    fig, axes = plt.subplots(1, len(rows_by_run), figsize=(3 * len(rows_by_run), 4.2), sharey=True)
    axes = list(axes) if len(rows_by_run) > 1 else [axes]
    present = [sc for sc in STEP_CLASSES
               if any(r[f"{sc[0]}_step_share"] > 0 for rows in rows_by_run.values() for r in rows)]
    for ax, (label, rows) in zip(axes, rows_by_run.items()):
        width, _, mode = label.partition("_")
        xs = range(len(rows))
        bottom = [0.0] * len(rows)
        for step_class, name, color in present:
            values = [100 * r[f"{step_class}_time_share"] for r in rows]
            values = [0.0 if v != v else v for v in values]  # NaN -> 0
            ax.bar(xs, values, bottom=bottom, color=color, width=0.7, edgecolor="white",
                   linewidth=2, label=name)
            bottom = [b + v for b, v in zip(bottom, values)]
        style_axes(ax)
        ax.set_xticks(list(xs), [f"{r['requests_per_gpu']:g}" for r in rows])
        ax.set_xlabel("Requests per decode GPU", color=TEXT, fontsize=9)
        ax.set_title(f"{WIDTH_NAME[width]}\n{MODE_STYLE[mode][2]}", fontsize=9, color=TEXT)
    axes[0].set_ylabel("Share of engine time (%)", color=TEXT)
    axes[0].set_ylim(0, 100)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=8, ncol=len(present), loc="lower center")
    fig.suptitle("Engine time by step class (PD runs: decode pool)", fontsize=10, color=TEXT)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_latency(rows_by_run, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in ((axes[0], "ttft_ms_p90", "p90 time to first token (ms)"),
                           (axes[1], "decode_tpot_ms_p90", "p90 time per output token after the first (ms)")):
        for label, rows in rows_by_run.items():
            width, _, mode = label.partition("_")
            marker, line, mode_name = MODE_STYLE[mode]
            ax.plot([r["requests_per_gpu"] for r in rows], [r[key] for r in rows], line,
                    color=WIDTH_COLOR[width], marker=marker, markersize=8, linewidth=2,
                    **marker_colors(WIDTH_COLOR[width], mode),
                    label=f"{WIDTH_NAME[width]}, {mode_name}")
        style_axes(ax)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Concurrent requests per decode GPU", color=TEXT)
        ax.set_title(title, fontsize=10, color=TEXT)
    axes[0].set_yscale("log")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True,
                    help="{narrow,wide}_{agg,agg_nopc,pd}=<experiment dir>; repeat per run")
    ap.add_argument("--prefill-gpus-per-64", type=float, default=None,
                    help="prefill GPUs charged per 64 decode GPUs (default: the run's prefill pool)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "pareto", "pd_interference"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows_by_run = {}
    for spec in args.run:
        label, _, run_dir = spec.partition("=")
        rows_by_run[label] = summarize(label, run_dir, args.prefill_gpus_per_64)
    rows = [r for rs in rows_by_run.values() for r in rs]
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    flat = [{k: v for k, v in r.items() if k != "step_mix"} for r in rows]
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        fieldnames = list(dict.fromkeys(k for r in flat for k in r))  # PD rows add pool fields
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)
    plot_pareto(rows_by_run, os.path.join(args.out, "pareto.png"))
    plot_step_mix(rows_by_run, os.path.join(args.out, "step_mix.png"))
    plot_latency(rows_by_run, os.path.join(args.out, "latency.png"))
    for r in rows:
        print(f"{r['run']:16s} c={r['concurrency']:5d} ({r['requests_per_gpu']:g}/GPU)  "
              f"{r['total_tok_s_per_gpu']:8.1f} tok/s/GPU  p90 {r['user_tok_s_p90']:6.1f} tok/s/user  "
              f"TTFT p90 {r['ttft_ms_p90']:8.0f} ms  interfered {100 * r['decode_interfered_time_share']:5.1f}% "
              f"mixed {100 * r['mixed_time_share']:5.1f}%  ({r['requests']} requests)")


if __name__ == "__main__":
    main()
