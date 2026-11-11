"""Per-decode-step kernel breakdown from `nsys stats` kernel summaries, side by side for runs.

Reads <run>/rank<r>_cuda_gpu_kern_sum.csv (nsys stats --report cuda_gpu_kern_sum --format csv
--timeunit usec). Per kernel: median duration x calls per step, where steps = _fused_moe_kernel
calls / (2 x MoE layers). Medians drop prefill steps and step-boundary waits that inflate means.

    python3 inference-bench/analysis/kernel_breakdown.py \
        tp4=inference-bench/experiments/<run_a> tp8=inference-bench/experiments/<run_b> [--ranks 4 44]
"""

import argparse
import csv
import os
import re
from collections import defaultdict

MOE_LAYERS = 48  # Ultra v3

# (category, regex on the short kernel name); first match wins.
CATEGORIES = [
    # vLLM Triton fused MoE; FlashInfer TRT-LLM BF16 (bmm_* GEMMs, routing, our finalize).
    ("Routed expert GEMMs", r"^_fused_moe_kernel$|^bmm_Bfloat16|routingIndices|^_finalize_kernel$"),
    ("EP communication", r"all_gatherv_[23]tensor|reduce_scatter_v_kernel|_a2a_dispatch|_a2a_combine|_fused_metadata"),
    ("TP communication", r"_multimem_all_gather_kernel|_multimem_reduce_scatter_kernel|ncclDevKernel"),
    ("Dense GEMMs", r"nvjet|splitKreduce|cutlass(?!.*flash)|gemm|sm100_xmma"),
    ("Routing + MoE bookkeeping", r"gatherTopK|_moe_sum|_scatter_token_indices|_count_local_tokens|_init_sorted_ids|_mask_routing_padding|_fill_expert_block_ids|_prefix_sum|FillFunctor|sigmoid|_sigmoid_topk|_parallel_align"),
    ("Norms", r"rmsnorm|RMSNorm|norm_kernel|_layer_norm_fwd|NormKernel"),
    ("Mamba recurrence", r"_selective_scan|causal_conv1d"),
    ("Attention", r"flash|_append_kv_cache|CatArrayBatchedCopy"),
]


def short(name):
    return name.replace("void ", "").split("<")[0].split("(")[0].strip()


def load(run, rank):
    rows = list(csv.DictReader(open(os.path.join(run, f"rank{rank}_cuda_gpu_kern_sum.csv"))))
    kernels = {}
    for r in rows:
        n = r["Name"]
        kernels[n] = (int(r["Instances"]), float(r["Med (us)"]))
    moe_calls = sum(c for n, (c, _) in kernels.items() if short(n) == "_fused_moe_kernel")
    steps = moe_calls / (2 * MOE_LAYERS)
    per_kernel = {n: (c / steps, med, c / steps * med) for n, (c, med) in kernels.items()}
    return steps, per_kernel


def categorize(name):
    s = short(name)
    for cat, pattern in CATEGORIES:
        if re.search(pattern, s) or re.search(pattern, name):
            return cat
    return "Elementwise / other"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="label=run_dir")
    p.add_argument("--ranks", type=int, nargs="+", default=[4, 44])
    p.add_argument("--top", type=int, default=12, help="largest kernels to list per run")
    args = p.parse_args()

    runs = [s.split("=", 1) for s in args.runs]
    cats = defaultdict(dict)   # cat -> label -> us/step (mean over ranks)
    kern = defaultdict(dict)   # short name -> label -> (calls/step, med us, us/step)
    totals = {}
    for label, run in runs:
        acc_c, acc_k = defaultdict(float), defaultdict(lambda: [0.0, 0.0, 0.0])
        for rank in args.ranks:
            steps, pk = load(run, rank)
            print(f"{label} rank {rank}: {steps:.1f} step-equivalents")
            for n, (cps, med, us) in pk.items():
                acc_c[categorize(n)] += us / len(args.ranks)
                k = acc_k[short(n)]
                k[0] += cps / len(args.ranks); k[1] += med / len(args.ranks); k[2] += us / len(args.ranks)
        for c, v in acc_c.items():
            cats[c][label] = v
        for n, v in acc_k.items():
            kern[n][label] = tuple(v)
        totals[label] = sum(acc_c.values())

    labels = [l for l, _ in runs]
    order = [c for c, _ in CATEGORIES] + ["Elementwise / other"]
    print("\nPer decode step, sum of median x calls (us), mean of ranks " + ", ".join(map(str, args.ranks)))
    head = f"| Category | " + " | ".join(labels) + (" | Delta |" if len(labels) == 2 else " |")
    print(head)
    print("|" + "---|" * (len(labels) + 1 + (len(labels) == 2)))
    for c in order:
        vals = [cats[c].get(l, 0.0) for l in labels]
        cells = " | ".join(f"{v:7.0f}" for v in vals)
        delta = f" | {vals[1] - vals[0]:+7.0f} |" if len(labels) == 2 else " |"
        print(f"| {c} | {cells}{delta}")
    vals = [totals[l] for l in labels]
    print(f"| **Total** | " + " | ".join(f"{v:7.0f}" for v in vals)
          + (f" | {vals[1] - vals[0]:+7.0f} |" if len(labels) == 2 else " |"))

    print("\nLargest kernels: calls/step, median us, us/step")
    names = sorted(kern, key=lambda n: -max(v[2] for v in kern[n].values()))[: args.top * 2]
    for n in names:
        cells = "  ".join(
            f"{l}: {kern[n][l][0]:6.1f} x {kern[n][l][1]:7.2f} = {kern[n][l][2]:7.0f}" if l in kern[n] else f"{l}: {'-':>28}"
            for l in labels
        )
        print(f"  {n[:52]:52s} [{categorize(n)[:12]:12s}]  {cells}")


if __name__ == "__main__":
    main()
