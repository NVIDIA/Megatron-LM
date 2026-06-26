# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Analysis: For each consecutive MoE-layer pair (L_prev, L), applies L's actual router weights to
L_prev's hidden states and compares the resulting predicted per-expert token-count
distribution to what L actually routed.

High cosine/Spearman here means a one-layer-ahead predictor can accurately anticipate
load distribution at the next MoE layer.

Requires traces collected with --moe-routing-trace-capture-hidden-states and
--moe-routing-trace-dump-weights.  These sidecars are produced only by the
forward-hook trace path, not the in-pipeline sink, so collect them with the
hook path (remove --moe-enable-routing-replay, add --cuda-graph-impl none to trigger hooks).

Usage:
    python analyze_routing_predictability.py /path/to/trace_dir
    python analyze_routing_predictability.py /path/to/trace_dir --output-dir plots/
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import torch


def load_router_state(trace_dir):
    pattern = os.path.join(trace_dir, "router_state_rank*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No router_state_rank*.pt in {trace_dir}. "
            "Router weights are captured only by the forward-hook trace path, not the "
            "in-pipeline sink (RouterReplay/RoutingMetadata holds top-K indices only). "
            "Re-run with the hook path (omit --moe-enable-routing-replay, add "
            "--cuda-graph-impl none) and --moe-routing-trace-dump-weights."
        )
    merged = {}
    for p in paths:
        state = torch.load(p, map_location="cpu", weights_only=False)
        for layer, info in state.items():
            # Older traces keyed router state by a (block, mtp_idx, layer) tuple;
            # normalize to the integer layer to match the JSONL records' "layer".
            if isinstance(layer, (tuple, list)):
                layer = layer[-1]
            merged.setdefault(layer, info)
    print(f"Loaded router state for {len(merged)} layers from {len(paths)} rank files.")
    return merged


def load_trace(trace_dir):
    from megatron.core.transformer.moe.router_trace import load_hidden_states_for_record

    pattern = os.path.join(trace_dir, "router_trace_rank*.jsonl")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No router_trace_rank*.jsonl in {trace_dir}")
    data = defaultdict(lambda: defaultdict(dict))
    n_with = n_total = 0
    for path in paths:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                n_total += 1
                if "hs_offset" not in r:
                    continue
                hs = load_hidden_states_for_record(r, trace_dir)
                data[r["rank"]][r["step"]][r["layer"]] = (r, hs)
                n_with += 1
    print(f"Loaded {n_with}/{n_total} records with hidden states.")
    return data


def apply_router(hidden_state, layer_state, top_k):
    weight = layer_state["weight"]
    expert_bias = layer_state.get("expert_bias")
    score_fn = layer_state.get("score_function", "sigmoid")
    h = hidden_state.float()
    if h.shape[-1] != weight.shape[-1]:
        return None
    logits = h @ weight.T
    if score_fn == "sigmoid":
        scores = torch.sigmoid(logits)
    elif score_fn == "softmax":
        scores = torch.softmax(logits, dim=-1)
    else:
        scores = logits
    if expert_bias is not None:
        scores = scores + expert_bias.float()
    return scores.topk(top_k, dim=-1).indices


def _pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va == 0 or vb == 0:
        return float("nan")
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return cov / math.sqrt(va * vb)


def _ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(a, b):
    return _pearson(_ranks(a), _ranks(b))


def _cosine(a, b):
    dot = sum(a[i] * b[i] for i in range(len(a)))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else float("nan")


def _hot_overlap(a, b, m):
    top_a = set(sorted(range(len(a)), key=lambda e: a[e], reverse=True)[:m])
    top_b = set(sorted(range(len(b)), key=lambda e: b[e], reverse=True)[:m])
    return len(top_a & top_b) / m if m else float("nan")


def iter_aligned_samples(data, L_prev, L):
    for per_step in data.values():
        for layers in per_step.values():
            if L_prev not in layers or L not in layers:
                continue
            prev_rec, prev_hs = layers[L_prev]
            dst_rec, _ = layers[L]
            if prev_rec["num_tokens"] != dst_rec["num_tokens"]:
                continue
            yield prev_hs, dst_rec


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trace_dir", help="Trace dir with hidden states + router state.")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Experts per token. Default: inferred from trace.")
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--layers", default=None,
                        help="Comma-separated MoE layer numbers to analyze (default: all pairs).")
    parser.add_argument("--router-state-dir", default=None,
                        help="Load router_state_rank*.pt from here instead of trace_dir.")
    parser.add_argument("--output-dir", default=None, help="Write CSV and plots here.")
    args = parser.parse_args()

    weights_dir = args.router_state_dir or args.trace_dir
    router_state = load_router_state(weights_dir)
    data = load_trace(args.trace_dir)

    trace_layers = set()
    inferred_topk = None
    for per_step in data.values():
        for layers in per_step.values():
            trace_layers.update(layers.keys())
            for rec, _hs in layers.values():
                inferred_topk = inferred_topk or rec.get("topk")
    common = sorted(trace_layers & set(router_state.keys()))
    if len(common) < 2:
        raise SystemExit("Need >=2 MoE layers with both router state and trace records.")

    top_k = args.top_k or inferred_topk
    E = router_state[common[0]]["weight"].shape[0]
    if args.num_experts != E:
        print(f"WARNING: --num-experts {args.num_experts} != router weight dim {E}. Using {E}.")
    hot_m = max(1, E // 8)  # top-12.5% as the "hot set" for overlap

    print(f"\nLayers: {len(common)} ({common[0]}..{common[-1]})  |  "
          f"top_k: {top_k}  |  num_experts: {E}\n")

    if args.layers:
        chosen = sorted({int(x) for x in args.layers.split(",")})
        layer_pairs = [
            (common[common.index(L) - 1], L)
            for L in chosen if L in common and common.index(L) > 0
        ]
    else:
        layer_pairs = [(common[i], common[i + 1]) for i in range(len(common) - 1)]

    print("DISTRIBUTION PREDICTABILITY  (L's router applied to L_prev's hidden states)")
    print(f"  {'src':>4} -> {'dst':>4} | {'cos':>6} | {'spearman':>8} | hot-{hot_m} overlap")
    print("  " + "-" * 52)

    results = []
    for L_prev, L in layer_pairs:
        L_state = router_state[L]
        c_pred = torch.zeros(E)
        c_act = torch.zeros(E)
        n = 0
        for prev_hs, dst_rec in iter_aligned_samples(data, L_prev, L):
            predicted = apply_router(prev_hs, L_state, top_k)
            if predicted is None:
                continue
            act = torch.tensor(dst_rec["top_indices"], dtype=torch.long).flatten()
            c_pred += torch.bincount(predicted.flatten(), minlength=E).float()
            c_act += torch.bincount(act, minlength=E).float()
            n += 1
        if c_act.sum() == 0:
            print(f"  {L_prev:>4} -> {L:>4} | (no data)")
            continue
        a, b = c_pred.tolist(), c_act.tolist()
        cos = _cosine(a, b)
        spear = _spearman(a, b)
        hot = _hot_overlap(a, b, hot_m)
        print(f"  {L_prev:>4} -> {L:>4} | {cos:>6.3f} | {spear:>8.3f} | {hot:>6.3f}")
        results.append((L_prev, L, cos, spear, hot, n))

    if results:
        mean_cos = sum(r[2] for r in results) / len(results)
        mean_spear = sum(r[3] for r in results) / len(results)
        print(f"\nMean cosine: {mean_cos:.3f}  |  Mean Spearman: {mean_spear:.3f}")
        print(
            "\nInterpretation:"
            "\n  cos/Spearman ≥ 0.90 / 0.70 : strong distributional signal — L_prev's hidden"
            "\n    states are sufficient to predict L's aggregate expert load with high fidelity."
            "\n  Values near zero            : weak cross-layer signal for this layer pair."
        )

    if args.output_dir and results:
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, "predictability_per_layer.csv")
        with open(csv_path, "w") as f:
            f.write("src,dst,count_cosine,count_spearman,hot_overlap,samples\n")
            for r in results:
                f.write(",".join(str(x) for x in r) + "\n")
        print(f"\nWrote {csv_path}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib unavailable; skipping plot.")
        else:
            fig, ax = plt.subplots(figsize=(max(8, len(results) * 0.3), 4))
            x = list(range(len(results)))
            labels = [f"{p}→{d}" for p, d, *_ in results]
            ax.plot(x, [r[2] for r in results], marker="o", label="cosine similarity")
            ax.plot(x, [r[3] for r in results], marker="s", label="Spearman correlation")
            ax.axhline(0.9, color="green", linestyle=":", linewidth=1, label="cos ≥ 0.90 threshold")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score")
            ax.set_xlabel("Consecutive MoE layer pair (L_prev → L)")
            ax.legend()
            ax.set_title("Distribution predictability: L's router on L_prev's hidden states")
            out = os.path.join(args.output_dir, "predictability_per_layer.png")
            fig.tight_layout()
            fig.savefig(out, dpi=120)
            plt.close(fig)
            print(f"Wrote {out}")


if __name__ == "__main__":
    main()
