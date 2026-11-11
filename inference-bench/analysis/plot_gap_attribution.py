"""Super, wide vs narrow EP (MInf, NVLS): split the difference in average step time into clean
decode steps and prefill interference, from the step-time histograms of the interference runs.

Per engine, average step time T = C + I: C is the mean clean decode step, I the excess over it from
steps that have a prefill anywhere in the EP group (prefill and interfered decode steps). At equal
requests per GPU each step yields the same tokens per GPU, so throughput is proportional to 1 / T;
T_narrow / T_wide reproduces the measured throughput ratio to within ~1%. Then
    T_wide - T_narrow = (C_wide - C_narrow) + (I_wide - I_narrow).

    python3 inference-bench/analysis/plot_gap_attribution.py
Writes inference-bench/analysis/pareto/interference/super_wide_vs_narrow_gap.png.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.join(HERE, "..", "experiments")
WIDE = ("20261102_032910_math_superv3_wide_ep_nvls_interference", 64)
NARROW = ("20261102_031636_math_superv3_narrow_ep_nvls_interference", 8)


def steps(run, gpus):
    """{requests per GPU: (T, C, I, tok/s/GPU)} from the per-level step-time histograms."""
    out = {}
    for line in open(os.path.join(EXPERIMENTS, run, "steady.jsonl")):
        r = json.loads(line)
        h = r["step_time_ms"].get("all", {})
        n = {k: h.get(k, {}).get("count", 0) for k in ("decode_clean", "decode_interfered", "prefill")}
        m = {k: h.get(k, {}).get("mean_ms", 0.0) for k in n}
        t = sum(n[k] * m[k] for k in n) / sum(n.values())
        out[r["concurrency"] // gpus] = (t, m["decode_clean"], t - m["decode_clean"],
                                         r["output_tok_s"] / gpus)
    return out


def main():
    wide, narrow = steps(*WIDE), steps(*NARROW)
    levels = sorted(set(wide) & set(narrow))
    rows = []
    print("req/GPU  T wide/narrow (ms)  clean part  interference part  "
          "predicted vs measured wide/narrow throughput")
    for c in levels:
        (tw, cw, iw, xw), (tn, cn, in_, xn) = wide[c], narrow[c]
        rows.append((c, 100 * (cw - cn) / tn, 100 * (iw - in_) / tn, 100 * (tw - tn) / tn))
        print(f"  {c:3d}    {tw:6.2f} / {tn:6.2f}      {cw - cn:+6.2f}      {iw - in_:+6.2f}"
              f"            {tn / tw:.3f} vs {xw / xn:.3f}")
    os.makedirs(os.path.join(HERE, "pareto", "interference"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    x = range(len(rows))
    clean = [r[1] for r in rows]
    interf = [r[2] for r in rows]
    # Stack each component on its own side of zero so opposite signs do not hide each other.
    ax.bar(x, clean, width=0.6, color="#0072B2", label="Clean decode steps")
    base = [c if (c > 0) == (i > 0) else 0 for c, i in zip(clean, interf)]
    ax.bar(x, interf, width=0.6, bottom=base, color="#D55E00", label="Prefill interference")
    ax.plot(x, [r[3] for r in rows], color="black", marker="D", ms=5, ls="none",
            label="Net (wide - narrow)")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(r[0]) for r in rows])
    ax.set_xlabel("Requests per GPU")
    ax.set_ylabel("Wide minus narrow avg step time\n(% of narrow's step time)")
    ax.set_title("Nemotron-3-Super-120B-A12B, MInf (NVLS): where wide EP gains and loses\n"
                 "below 0: wide EP faster; above 0: slower", fontsize=8.5)
    ax.yaxis.grid(ls="dotted")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    path = os.path.join(HERE, "pareto", "interference", "super_wide_vs_narrow_gap.png")
    fig.savefig(path, dpi=160)
    print(path)


if __name__ == "__main__":
    main()
