"""Paper figures: MInf vs vLLM throughput-interactivity Pareto plots, as pgfplots .tex files.

Each figure is one column wide: a groupplot with one panel per model, a shared legend above the
panels (so no legend covers data) and one shared x label. Plots use the paper's styles from
tex/tikz_styles.tex: `mplot` axes, `sysline` for \\sys{}, `vllmline` for vLLM. Dashed lines mark
interactivity SLOs, with \\sys{}'s throughput over vLLM's at each SLO (linearly interpolated
between measured levels, ringed). The 1-request-per-GPU level is omitted (few completions in the
window; first level after startup).

    python3 inference-bench/analysis/paper_figures.py --tex inference-bench/paper/overleaf/tex/inference-perf.tex

The figures are embedded in the section: each lives between `% BEGIN <name>` and `% END <name>`
marker lines, which this script rewrites in place (everything else in the file is untouched).
"""

import argparse
import json
import math
import os
import re

EXPERIMENTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments")

# Interactivity SLOs (p90 tokens/s/user): a median 18.9K-token answer in ~3 min at 100 (Super)
# and ~6 min at 50 (Ultra); Ultra wide also marks a stricter 60 (~5 min; vLLM's p90 never
# reaches 70 there), and Super wide a looser 80 (~4 min), which vLLM reaches there while it
# never reaches 100.
SUPER_SLO, ULTRA_SLO = [100], [50]

# Interactivity is the p90-slowest request's tokens/s (InferenceX style); throughput and
# interactivity are each averaged over the repeated runs of a configuration, level by level.
INTERACTIVITY = "user_tok_s_p90_slow"

# Panels per figure: (title lines, MInf runs, vLLM runs, SLO lines); each runs entry is the
# three independent trials of one configuration.
FIGURES = {
    "e2e_wide": [
        (("Nemotron-3-Ultra-550B-A55B", r"TP8$\times$DP8, EP64"),
         ("20261024_184036_math_ultrav3_wide_ep_nvls", "20261029_121740_math_ultrav3_wide_ep_nvls_rep2", "20261029_190451_math_ultrav3_wide_ep_nvls_rep3"),
         ("20261027_165152_ultrav3_wide_ep_vllm", "20261029_125620_ultrav3_wide_ep_vllm_rep2", "20261029_190451_ultrav3_wide_ep_vllm_rep3"),
         ULTRA_SLO + [60]),
        (("Nemotron-3-Super-120B-A12B", r"TP4$\times$DP16, EP64"),
         ("20261024_160510_math_superv3_wide_ep_nvls", "20261029_133805_math_superv3_wide_ep_nvls_rep2", "20261029_170943_math_superv3_wide_ep_nvls_rep3"),
         ("20261028_141706_superv3_wide_ep_vllm", "20261029_121420_superv3_wide_ep_vllm_rep2", "20261029_181520_superv3_wide_ep_vllm_rep3"),
         [(80, "left")] + SUPER_SLO),
    ],
    "e2e_narrow": [
        (("Nemotron-3-Ultra-550B-A55B", "TP8, EP8"),
         ("20261024_171149_math_ultrav3_narrow_ep_nvls", "20261029_113536_math_ultrav3_narrow_ep_nvls_rep2", "20261029_162257_math_ultrav3_narrow_ep_nvls_rep3"),
         ("20261027_154043_ultrav3_narrow_ep_vllm", "20261029_125840_ultrav3_narrow_ep_vllm_rep2", "20261029_162257_ultrav3_narrow_ep_vllm_rep3"),
         ULTRA_SLO),
        (("Nemotron-3-Super-120B-A12B", r"TP4$\times$DP2, EP8"),
         ("20261024_175142_math_superv3_narrow_ep_nvls", "20261029_105211_math_superv3_narrow_ep_nvls_rep2", "20261029_153823_math_superv3_narrow_ep_nvls_rep3"),
         ("20261027_162013_superv3_narrow_ep_vllm_fi_one_sided", "20261029_105212_superv3_narrow_ep_vllm_rep2", "20261029_153823_superv3_narrow_ep_vllm_rep3"),
         SUPER_SLO),
    ],
}


def gpus_of(run):
    """GPUs of a run: NODES in its config.env, 4 GPUs per node."""
    m = re.search(r"\bNODES=(\d+)", open(os.path.join(EXPERIMENTS, run, "config.env")).read())
    return 4 * int(m.group(1))


def levels(run):
    """{concurrency: (tokens/s/user, tokens/s/GPU)} of one run, without the 1-per-GPU level."""
    gpus = gpus_of(run)
    out = {}
    for line in open(os.path.join(EXPERIMENTS, run, "steady.jsonl")):
        r = json.loads(line)
        if r["concurrency"] == gpus or r[INTERACTIVITY] != r[INTERACTIVITY]:
            continue
        out[r["concurrency"]] = (r[INTERACTIVITY], r["output_tok_s"] / gpus)
    return out


def points(runs, spread=False):
    """(tokens/s/user, tokens/s/GPU) per concurrency level, each the mean over the repeated runs
    (levels every run measured); with spread, also each axis's half range (max - min) / 2."""
    per_run = [levels(r) for r in runs]
    common = sorted(set.intersection(*(set(lv) for lv in per_run)))
    out = []
    for c in common:
        vals = [[lv[c][k] for lv in per_run] for k in (0, 1)]
        mean = tuple(sum(v) / len(v) for v in vals)
        out.append(mean + tuple((max(v) - min(v)) / 2 for v in vals) if spread else mean)
    return out


def at_interactivity(pts, x):
    """Throughput at interactivity x, linearly interpolated between consecutive concurrency
    levels (pts in concurrency order); the highest if the curve crosses x more than once, None
    if it never reaches x."""
    ys = [y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 != x0 else max(y0, y1)
          for (x0, y0), (x1, y1) in zip(pts, pts[1:]) if min(x0, x1) <= x <= max(x0, x1)]
    return max(ys) if ys else None


def nice_max(v):
    """A round axis maximum a little above v."""
    step = 10 ** math.floor(math.log10(v)) / 2
    return math.ceil(v * 1.08 / step) * step


def coords(pts):
    return " ".join(f"({x:.1f},{y:.1f})" for x, y in pts)


def panel(title, sys_run, vllm_run, slos, first, legend_name, ylabel=None):
    """One \\nextgroupplot block; the first panel carries the legend, and ylabel (default: the
    first panel's) the y label."""
    sys_pts = points(sys_run)
    vllm_pts = points(vllm_run)
    xmax = nice_max(max(x for x, _ in sys_pts + vllm_pts))
    ymax = nice_max(max(y for _, y in sys_pts + vllm_pts))
    opts = [f"title={{{title[0]}\\\\{title[1]}}}", f"xmax={xmax:g}", f"ymax={ymax:g}"]
    if first if ylabel is None else ylabel:
        opts.append("ylabel={Throughput (tokens/s/GPU)}")
    if first:
        opts.append(f"legend to name={legend_name}")
    out = ["\\nextgroupplot[" + ", ".join(opts) + "]"]
    if first:
        # Legend entries independent of the data, so a pending series still gets its entry.
        out += ["  \\addlegendimage{sysline}\\addlegendentry{\\sys{}}",
                "  \\addlegendimage{vllmline}\\addlegendentry{vLLM}"]
    # Run-to-run spread (points(..., spread=True)) is below the marker size, so it is stated in
    # the caption rather than drawn as error bars.
    out.append(f"  \\addplot[sysline, forget plot] coordinates {{{coords(sys_pts)}}};")
    out.append(f"  \\addplot[vllmline, forget plot] coordinates {{{coords(vllm_pts)}}};")
    for i, slo in enumerate(slos):
        # An SLO is a number, or (number, "left") to put its ratio label left of the line
        # (clear of data points sitting on the line's foot).
        slo, side = slo if isinstance(slo, tuple) else (slo, "center")
        label = f"SLO {slo}" if len(slos) == 1 else f"{slo}"
        # On the background layer, behind the curves and the ratio label's white box.
        out.append(f"  \\addplot[sloline, forget plot, on layer=axis background] "
                   f"coordinates {{({slo},0) ({slo},{ymax:g})}};")
        out.append(f"  \\node[slolabel] at (axis cs:{slo},{ymax:g}) {{{label}}};")
        y_sys, y_vllm = at_interactivity(sys_pts, slo), at_interactivity(vllm_pts, slo)
        if y_sys is None or y_vllm is None:
            continue
        out.append(f"  \\addplot[ring, draw=figbluedraw, forget plot] coordinates {{({slo},{y_sys:.1f})}};")
        out.append(f"  \\addplot[ring, draw=figorange, forget plot] coordinates {{({slo},{y_vllm:.1f})}};")
        # Just above the x axis, on a white box over the dashes; later SLOs stack higher.
        anchor = ", anchor=south east" if side == "left" else ""
        out.append(f"  \\node[sloratio{anchor}] at (axis cs:{slo},{ymax * (0.03 + 0.12 * i):.1f}) "
                   f"{{{y_sys / y_vllm:.2f}$\\times$}};")
    return "\n".join(out)


def figure(name, panels):
    legend_name = name.replace("_", "") + "legend"
    blocks = [panel(*p, first=(i == 0), legend_name=legend_name) for i, p in enumerate(panels)]
    runs = ", ".join(r for p in panels for rs in p[1:3] for r in rs)
    return f"""% Generated by inference-bench/analysis/paper_figures.py; edit the script, not this block.
% Runs: {runs}
\\begin{{tikzpicture}}[
    sloline/.style={{draw=figgrayline, densely dashed, line width=0.6pt, mark=none}},
    slolabel/.style={{anchor=north west, inner sep=1pt, font=\\figfont\\tiny, text=black!60}},
    sloratio/.style={{anchor=south, fill=white, inner sep=1pt, font=\\figfont\\bfseries\\tiny}},
    ring/.style={{only marks, mark=o, mark size=2.2pt, mark options={{solid, line width=0.7pt}}}}]
\\begin{{groupplot}}[mplot, set layers, scale only axis, width=29mm, height=26mm,
    group style={{group size={len(panels)} by 1, horizontal sep=10mm}},
    xmin=0, ymin=0,
    title style={{font=\\figfont\\scriptsize, align=center, inner sep=1pt}},
    legend columns=2, legend style={{draw=none, fill=none, column sep=6pt}}]
{chr(10).join(blocks)}
\\end{{groupplot}}
\\node[anchor=north, font=\\figfont\\footnotesize]
    at ($(group c1r1.south)!0.5!(group c{len(panels)}r1.south) + (0,-5mm)$)
    {{P90 interactivity (tokens/s/user)}};
\\node[anchor=south]
    at ($(group c1r1.north)!0.5!(group c{len(panels)}r1.north) + (0,7mm)$)
    {{\\pgfplotslegendfromname{{{legend_name}}}}};
\\end{{tikzpicture}}
"""


def e2e_figure():
    """One 2x2 figure: (a) wide EP on the top row, (b) narrow EP on the bottom row, a shared
    legend above and one shared x label."""
    rows = [("(a) Wide EP on 64 GPUs", FIGURES["e2e_wide"]),
            ("(b) Narrow EP on 8 GPUs", FIGURES["e2e_narrow"])]
    blocks, runs = [], []
    for r, (label, panels) in enumerate(rows):
        blocks.append(f"% ---- {label.split(')')[0]}) {label.split(') ')[1].split(' on')[0]} ----")
        for i, p in enumerate(panels):
            blocks.append(panel(*p, first=(r == 0 and i == 0), legend_name="e2elegend",
                                ylabel=(i == 0)))
        runs.append(f"% Runs ({'wide' if r == 0 else 'narrow'}): "
                    + ", ".join(x for p in panels for rs in p[1:3] for x in rs))
    return f"""% Generated by inference-bench/analysis/paper_figures.py; edit the script, not this block.
{chr(10).join(runs)}
\\begin{{tikzpicture}}[
    sloline/.style={{draw=figgrayline, densely dashed, line width=0.6pt, mark=none}},
    slolabel/.style={{anchor=north west, inner sep=1pt, font=\\figfont\\tiny, text=black!60}},
    sloratio/.style={{anchor=south, fill=white, inner sep=1pt, font=\\figfont\\bfseries\\tiny}},
    ring/.style={{only marks, mark=o, mark size=2.2pt, mark options={{solid, line width=0.7pt}}}},
    rowlbl/.style={{anchor=south, font=\\figfont\\footnotesize}}]
\\begin{{groupplot}}[mplot, set layers, scale only axis, width=29mm, height=24mm,
    group style={{group size=2 by 2, horizontal sep=10mm, vertical sep=21mm}},
    xmin=0, ymin=0,
    title style={{font=\\figfont\\scriptsize, align=center, inner sep=1pt}},
    legend columns=2, legend style={{draw=none, fill=none, column sep=6pt}}]
{chr(10).join(blocks)}
\\end{{groupplot}}
\\node[rowlbl] at ($(group c1r1.north)!0.5!(group c2r1.north) + (0,8mm)$)
    {{{rows[0][0]}}};
\\node[rowlbl] at ($(group c1r2.north)!0.5!(group c2r2.north) + (0,8mm)$)
    {{{rows[1][0]}}};
\\node[anchor=north, font=\\figfont\\footnotesize]
    at ($(group c1r2.south)!0.5!(group c2r2.south) + (0,-5mm)$)
    {{P90 interactivity (tokens/s/user)}};
\\node[anchor=south]
    at ($(group c1r1.north)!0.5!(group c2r1.north) + (0,12.5mm)$)
    {{\\pgfplotslegendfromname{{e2elegend}}}};
\\end{{tikzpicture}}
"""


# Collective microbenchmark (microbench/collectives_bench.py): latency vs local tokens per GPU at
# one hidden size, ours vs NCCL, one curve per GPU count whose CSV exists.
COLLECTIVES_HIDDEN = 4096
COLLECTIVES_CSVS = {4: "collectives_4gpu_8422239.csv", 16: "collectives_16gpu_8422826.csv"}
MICROBENCH_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "microbench",
                                  "results")


def collectives_figure():
    """pgfplots groupplot: all-gather and reduce-scatter latency, NVLS (ours) vs NCCL."""
    import csv

    series = {}  # (op, gpus) -> [(tokens, us)]
    for gpus, name in COLLECTIVES_CSVS.items():
        path = os.path.join(MICROBENCH_RESULTS, name)
        if not os.path.exists(path):
            continue
        for row in csv.DictReader(open(path)):
            if int(row["hidden"]) == COLLECTIVES_HIDDEN:
                series.setdefault((row["op"], gpus), []).append((int(row["local_tokens"]),
                                                                  float(row["us"])))
    gpu_counts = sorted({g for _, g in series})
    panels = []
    # Both move a [gpus x tokens, hidden] tensor: the all-gather assembles it from each GPU's
    # tokens (its input), the reduce-scatter reduces it into each GPU's tokens (its output).
    for i, (title, xlabel, ours, nccl) in enumerate((
            ("All-gather", "Input tokens per GPU", "agv_nvls", "ag_nccl"),
            ("Reduce-scatter", "Output tokens per GPU", "rsv_nvls", "rs_nccl"))):
        opts = [f"title={{{title}}}", f"xlabel={{{xlabel}}}"]
        if i == 0:
            opts += ["ylabel={Latency ($\\mu$s)}", "legend to name=microcollectiveslegend"]
        lines = ["\\nextgroupplot[" + ", ".join(opts) + "]"]
        for g in gpu_counts:
            dashed = ", densely dashed" if g != gpu_counts[0] else ""
            for op, style, label in ((ours, "sysline", f"Ours, {g} GPUs"),
                                     (nccl, "vllmline", f"\\texttt{{torch.dist}}, {g} GPUs")):
                pts = sorted(series[(op, g)])
                coords_ = " ".join(f"({t},{us:.2f})" for t, us in pts)
                lines.append(f"  \\addplot[{style}{dashed}] coordinates {{{coords_}}};")
                if i == 0:
                    lines.append(f"  \\addlegendentry{{{label}}}")
        panels.append("\n".join(lines))
    sources = ", ".join(COLLECTIVES_CSVS[g] for g in gpu_counts)
    return f"""% Generated by inference-bench/analysis/paper_figures.py; edit the script, not this block.
% Data: inference-bench/microbench/results/{{{sources}}} (hidden {COLLECTIVES_HIDDEN}, bf16)
\\begin{{tikzpicture}}
\\begin{{groupplot}}[mplot, scale only axis, width=29mm, height=26mm,
    group style={{group size=2 by 1, horizontal sep=8mm}},
    xmode=log, log basis x=2, ymode=log, xmin=1, xmax=2048,
    xtick={{1,8,64,512}}, xticklabels={{1,8,64,512}},
    title style={{font=\\figfont\\scriptsize, inner sep=1pt}},
    legend columns=2, legend style={{draw=none, fill=none, column sep=6pt}}]
{chr(10).join(panels)}
\\end{{groupplot}}
\\node[anchor=south]
    at ($(group c1r1.north)!0.5!(group c2r1.north) + (0,5mm)$)
    {{\\pgfplotslegendfromname{{microcollectiveslegend}}}};
\\end{{tikzpicture}}
"""


def replace_block(path, name, body):
    """Rewrite the lines between '% BEGIN <name>' and '% END <name>' in path."""
    text = open(path).read()
    begin, end = f"% BEGIN {name}\n", f"% END {name}\n"
    i, j = text.find(begin), text.find(end)
    assert 0 <= i < j, f"{path} has no '% BEGIN {name}' ... '% END {name}' block"
    open(path, "w").write(text[: i + len(begin)] + body + text[j:])
    print(f"-> {name} in {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", required=True,
                    help="inference-perf section, with a % BEGIN/% END e2e block")
    ap.add_argument("--parallelism-tex", default=None,
                    help="tex/dp_ep_tp.tex (DP / TP / EP subsections), with a "
                         "% BEGIN/% END micro_collectives block")
    args = ap.parse_args()
    replace_block(args.tex, "e2e", e2e_figure())
    if args.parallelism_tex:
        replace_block(args.parallelism_tex, "micro_collectives", collectives_figure())


if __name__ == "__main__":
    main()
