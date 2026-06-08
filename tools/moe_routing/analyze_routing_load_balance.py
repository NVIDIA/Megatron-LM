# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Can one-layer-ahead routing prediction balance expert load across EP GPUs?

That script asks a per-token question (does L_prev's hidden state let us recover
L's exact top-K?). Load balancing is a distributional question: we never need
to know which expert an individual token picks — we still run L's real router and
dispatch tokens to their true experts. The only thing decided one MoE layer ahead
is the expert -> GPU placement, and that depends solely on the per-expert token
*count histogram*, summed over the batch.

Aggregate counts are far more predictable than individual assignments because
per-token errors cancel in the sum. So a mediocre per-token Jaccard does not
doom load-aware placement — this script measures the aggregate directly.

For each consecutive MoE-layer pair (L_prev, L):
  1. Build the predicted per-expert count vector by applying L's actual router to
     L_prev's hidden state, then taking top-K per token (no learning, just aggregated to histogram).
  2. Build the actual per-expert count vector from L's real top-K decisions.
  3. Characterize predictability of the *distribution* (cosine / Spearman /
     hot-expert overlap).
  4. Run a placement simulation across ``--ep-size`` GPUs (round-robin layout
     assumed for the baseline) under three policies:
       - baseline   : today's static round-robin placement,
       - oracle     : balance using the ACTUAL counts (best achievable),
       - predicted  : balance using the PREDICTED counts, then score the
                      resulting placement on the ACTUAL counts (realistic gain).
     Imbalance is max_gpu_load / mean_gpu_load — the slowest GPU sets MoE step
     time, so this factor bounds the load-imbalance slowdown directly.

The headline number is the *recovery fraction*:
    (baseline_imbalance - predicted_imbalance) / (baseline_imbalance - oracle_imbalance)
~1.0 means one-layer-ahead prediction closes essentially all of the achievable
load-imbalance gap (i.e., recovers the ~10% E2E lost to imbalance).

Usage:
    python analyze_routing_load_balance.py /path/to/trace_dir --ep-size 8
    python analyze_routing_load_balance.py /path/to/trace_dir --ep-size 8 \
        --top-k 6 --num-experts 128 --min-tokens 64 --output-dir out/
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import torch


# --------------------------------------------------------------------------- #
# Trace / router-state loading (self-contained, mirrors the sibling scripts).
# --------------------------------------------------------------------------- #
def load_router_state(trace_dir):
    """Load saved per-layer router state (weight, expert_bias, score_function)."""
    pattern = os.path.join(trace_dir, "router_state_rank*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No router_state_rank*.pt in {trace_dir}. "
            "Re-run inference with --moe-routing-trace-dump-router-weights."
        )
    merged = {}
    for p in paths:
        state = torch.load(p, map_location="cpu", weights_only=False)
        for layer, info in state.items():
            merged.setdefault(layer, info)
    print(f"Loaded router state for {len(merged)} layers from {len(paths)} rank files.")
    return merged


def load_trace(trace_dir):
    """Load JSONL records that carry both hidden state and top_indices.

    Returns: {rank: {step: {layer: (record, hidden_state_tensor)}}}
    """
    from megatron.core.transformer.moe.router_trace import load_hidden_states_for_record

    pattern = os.path.join(trace_dir, "router_trace_rank*.jsonl")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No router_trace_rank*.jsonl in {trace_dir}")
    data = defaultdict(lambda: defaultdict(dict))
    n_with = 0
    n_total = 0
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
    """Apply layer L's router to a [num_tokens, hidden_dim] hidden state.

    Mimics TopKRouter.routing after gating: score_function(h @ weight.T), add
    expert_bias if present, take top-K. Returns [num_tokens, top_k] or None on
    a hidden-dim mismatch (e.g. unexpected latent compression).
    """
    weight = layer_state["weight"]  # [num_experts, hidden_dim]
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


# --------------------------------------------------------------------------- #
# Small stats helpers (no scipy dependency).
# --------------------------------------------------------------------------- #
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
    """Average ranks (1-based), ties shared."""
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
    """Fraction of the m heaviest experts shared between count vectors a and b."""
    top_a = set(sorted(range(len(a)), key=lambda e: a[e], reverse=True)[:m])
    top_b = set(sorted(range(len(b)), key=lambda e: b[e], reverse=True)[:m])
    return len(top_a & top_b) / m if m else float("nan")


# --------------------------------------------------------------------------- #
# Placement simulation.
# --------------------------------------------------------------------------- #
def round_robin_placement(num_experts, ep_size):
    """Static baseline: expert e lives on GPU (e % ep_size)."""
    return [e % ep_size for e in range(num_experts)]


def balanced_placement(counts, ep_size, num_experts):
    """Cardinality-constrained greedy balance (LPT).

    Real EP holds ceil(num_experts / ep_size) experts per GPU, so this is
    balanced number-partitioning with an equal-cardinality cap, not free
    bin-packing. Assign heaviest experts first to the least-loaded GPU that
    still has capacity. This is the 'materialize experts for balance' policy
    the one-layer lead time would enable.
    """
    cap = math.ceil(num_experts / ep_size)
    order = sorted(range(num_experts), key=lambda e: counts[e], reverse=True)
    loads = [0.0] * ep_size
    fill = [0] * ep_size
    placement = [0] * num_experts
    for e in order:
        best = -1
        for g in range(ep_size):
            if fill[g] < cap and (best == -1 or loads[g] < loads[best]):
                best = g
        placement[e] = best
        loads[best] += counts[e]
        fill[best] += 1
    return placement


def imbalance_factor(counts, placement, ep_size):
    """max_gpu_load / mean_gpu_load for a given placement and count vector."""
    loads = [0.0] * ep_size
    for e, g in enumerate(placement):
        loads[g] += counts[e]
    total = sum(loads)
    if total == 0:
        return float("nan")
    return max(loads) / (total / ep_size)


def replicated_imbalance(actual_counts, hot_set, replicas, ep_size, num_experts):
    """Imbalance when the `hot_set` experts are replicated `replicas`x each.

    Models the mechanism that beats the re-placement floor: a hot expert with
    load c and R replicas contributes c/R to each of its R home GPUs, so a single
    expert no longer pins the busiest GPU. Replicas of one expert must sit on
    distinct GPUs (else the load doesn't actually split). Cold experts stay
    single-copy. Total expert *instances* = E + |hot|*(R-1), so each GPU holds
    ceil(instances / ep_size) of them -- that extra is the memory cost.

    Returns (imbalance, mem_overhead_fraction). mem_overhead = |hot|*(R-1)/E.
    """
    replicas = min(replicas, ep_size)  # >ep_size replicas can't avoid colocation
    instances = []  # (expert_id, load_per_instance)
    for e in range(num_experts):
        if e in hot_set:
            for _ in range(replicas):
                instances.append((e, actual_counts[e] / replicas))
        else:
            instances.append((e, actual_counts[e]))
    cap = math.ceil(len(instances) / ep_size)
    order = sorted(range(len(instances)), key=lambda i: instances[i][1], reverse=True)
    loads = [0.0] * ep_size
    fill = [0] * ep_size
    held = [set() for _ in range(ep_size)]  # experts already on each GPU
    for i in order:
        e, load = instances[i]
        best = -1
        for g in range(ep_size):
            if fill[g] < cap and e not in held[g] and (best == -1 or loads[g] < loads[best]):
                best = g
        if best == -1:  # no non-colocating slot; relax (rare, R close to ep_size)
            for g in range(ep_size):
                if fill[g] < cap and (best == -1 or loads[g] < loads[best]):
                    best = g
        loads[best] += load
        fill[best] += 1
        held[best].add(e)
    total = sum(loads)
    imb = max(loads) / (total / ep_size) if total > 0 else float("nan")
    mem_overhead = len(hot_set) * (replicas - 1) / num_experts
    return imb, mem_overhead


# --------------------------------------------------------------------------- #
# Per-layer accumulation.
# --------------------------------------------------------------------------- #
def iter_aligned_samples(data, L_prev, L):
    """Yield (predicted_indices_or_None, actual_top_indices) for aligned pairs."""
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
    parser.add_argument(
        "--ep-size", type=int, required=True,
        help="Expert-parallel degree: number of GPUs the experts are sharded across.",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Experts activated per token. Default: inferred from the trace's "
        "actual topk field (avoids a predicted/actual set-size mismatch).",
    )
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument(
        "--min-tokens", type=int, default=0,
        help="Skip per-step samples with fewer than this many tokens (decode steps "
        "can be tiny and give degenerate, noisy per-step imbalance).",
    )
    parser.add_argument(
        "--layers", default=None,
        help="Comma-separated MoE layer numbers to probe (default: all consecutive pairs).",
    )
    parser.add_argument(
        "--router-state-dir", default=None,
        help="Load router_state_rank*.pt from here instead of trace_dir (cross-trace).",
    )
    parser.add_argument("--output-dir", default=None, help="Write CSV + plot here.")
    parser.add_argument(
        "--mode", choices=["placement", "replicate"], default="placement",
        help="placement: re-place experts to balance (default). replicate: also "
        "sweep hot-expert replication, which beats the re-placement floor at "
        "high EP / few experts-per-GPU.",
    )
    parser.add_argument(
        "--hot-n", default="4,8,16",
        help="replicate mode: comma-separated counts of hottest experts to replicate.",
    )
    parser.add_argument(
        "--replicas", default="2,3,4",
        help="replicate mode: comma-separated replica counts per hot expert.",
    )
    args = parser.parse_args()

    weights_dir = args.router_state_dir or args.trace_dir
    router_state = load_router_state(weights_dir)
    data = load_trace(args.trace_dir)

    trace_layers = set()
    inferred_topk = None
    for per_step in data.values():
        for layers in per_step.values():
            trace_layers.update(layers.keys())
            for _rec, _hs in layers.values():
                inferred_topk = inferred_topk or _rec.get("topk")
    common = sorted(trace_layers & set(router_state.keys()))
    if len(common) < 2:
        raise SystemExit("Need >=2 MoE layers with both router state and trace records.")

    top_k = args.top_k or inferred_topk
    if args.top_k and inferred_topk and args.top_k != inferred_topk:
        print(
            f"WARNING: --top-k {args.top_k} != trace topk {inferred_topk}. Predicted "
            f"and actual histograms will have different total mass; cosine/Spearman "
            f"are scale-free but absolute counts will differ. The trace's real "
            f"activated count is {inferred_topk}; pass --top-k {inferred_topk} or omit it."
        )

    # Expert count is ground truth in the router weight, not the flag. Deriving it
    # avoids sizing histograms wrong (bincount would overflow a too-small E).
    E = router_state[common[0]]["weight"].shape[0]
    if args.num_experts != E:
        print(
            f"WARNING: --num-experts {args.num_experts} != router weight expert dim "
            f"{E}. Using {E} (the model's actual expert count). If you expected "
            f"{args.num_experts}, this trace is from a different model than you think."
        )
    G = args.ep_size
    hot_m = math.ceil(E / G)  # experts per GPU = the set that matters for placement
    print(
        f"\nLayers with router state + trace: {len(common)} ({common[0]}..{common[-1]})\n"
        f"EP size: {G} | num_experts: {E} (from router weight) | top_k: {top_k} "
        f"(trace topk: {inferred_topk}) | experts/GPU: {hot_m}\n"
    )

    # Build layer pairs.
    if args.layers:
        chosen = sorted({int(x) for x in args.layers.split(",")})
        layer_pairs = [
            (common[common.index(L) - 1], L)
            for L in chosen if L in common and common.index(L) > 0
        ]
    else:
        layer_pairs = [(common[i], common[i + 1]) for i in range(len(common) - 1)]

    rr = round_robin_placement(E, G)
    rand_cos = None  # filled lazily for reference

    print("=" * 100)
    print("DISTRIBUTION PREDICTABILITY (aggregate per-expert counts, L's router on L_prev's hs)")
    print(f"  {'src':>4} -> {'dst':>4} | {'cos':>6} | {'spearman':>8} | "
          f"hot-{hot_m} overlap")
    print("  " + "-" * 56)

    results = []  # (L_prev, L, cos, spear, hot, imb_base, imb_oracle, imb_pred, recov, n)
    agg = []  # (L_prev, L, c_pred, c_act) aggregate vectors, for replicate mode
    for L_prev, L in layer_pairs:
        L_state = router_state[L]
        c_pred = torch.zeros(E)
        c_act = torch.zeros(E)
        per_sample = []  # (imb_base, imb_oracle, imb_pred)
        n_samples = 0
        for prev_hs, dst_rec in iter_aligned_samples(data, L_prev, L):
            predicted = apply_router(prev_hs, L_state, top_k)
            if predicted is None:
                continue
            act = torch.tensor(dst_rec["top_indices"], dtype=torch.long).flatten()
            if dst_rec["num_tokens"] < args.min_tokens:
                # still accumulate to aggregate, but skip per-step sim
                c_pred += torch.bincount(predicted.flatten(), minlength=E).float()
                c_act += torch.bincount(act, minlength=E).float()
                continue
            sp = torch.bincount(predicted.flatten(), minlength=E).float()
            sa = torch.bincount(act, minlength=E).float()
            c_pred += sp
            c_act += sa
            sa_l = sa.tolist()
            imb_base = imbalance_factor(sa_l, rr, G)
            imb_oracle = imbalance_factor(sa_l, balanced_placement(sa_l, G, E), G)
            imb_pred = imbalance_factor(sa_l, balanced_placement(sp.tolist(), G, E), G)
            per_sample.append((imb_base, imb_oracle, imb_pred))
            n_samples += 1

        if c_act.sum() == 0:
            print(f"  {L_prev:>4} -> {L:>4} | (no aligned data)")
            continue
        agg.append((L_prev, L, c_pred.clone(), c_act.clone()))
        a, b = c_pred.tolist(), c_act.tolist()
        cos = _cosine(a, b)
        spear = _spearman(a, b)
        hot = _hot_overlap(a, b, hot_m)
        print(f"  {L_prev:>4} -> {L:>4} | {cos:>6.3f} | {spear:>8.3f} | {hot:>6.3f}")

        if per_sample:
            mean_base = sum(s[0] for s in per_sample) / len(per_sample)
            mean_oracle = sum(s[1] for s in per_sample) / len(per_sample)
            mean_pred = sum(s[2] for s in per_sample) / len(per_sample)
            denom = mean_base - mean_oracle
            recov = (mean_base - mean_pred) / denom if abs(denom) > 1e-9 else float("nan")
        else:
            mean_base = mean_oracle = mean_pred = recov = float("nan")
        results.append((L_prev, L, cos, spear, hot, mean_base, mean_oracle,
                        mean_pred, recov, n_samples))

    # ---- Imbalance / recovery table ----
    print()
    print("=" * 100)
    print(f"LOAD-IMBALANCE SIMULATION across {G} EP GPUs (max_gpu_load / mean_gpu_load; "
          f"lower is better)")
    print(f"  {'src':>4} -> {'dst':>4} | {'baseline':>8} | {'oracle':>7} | "
          f"{'predicted':>9} | {'recovery':>8} | samples")
    print("  " + "-" * 70)
    for (L_prev, L, _c, _s, _h, mb, mo, mp, rec, n) in results:
        print(f"  {L_prev:>4} -> {L:>4} | {mb:>8.3f} | {mo:>7.3f} | {mp:>9.3f} | "
              f"{rec:>7.1%} | {n}")

    if results:
        ok = [r for r in results if not math.isnan(r[8])]
        if ok:
            mb = sum(r[5] for r in ok) / len(ok)
            mo = sum(r[6] for r in ok) / len(ok)
            mp = sum(r[7] for r in ok) / len(ok)
            rec = (mb - mp) / (mb - mo) if abs(mb - mo) > 1e-9 else float("nan")
            print()
            print(f"OVERALL (mean over {len(ok)} layer pairs):")
            print(f"  baseline imbalance : {mb:.3f}  (≈ {100*(mb-1):.1f}% slower than perfectly balanced)")
            print(f"  oracle imbalance   : {mo:.3f}  (best achievable by re-placement)")
            print(f"  predicted imbalance: {mp:.3f}  (one-layer-ahead prediction)")
            print(f"  recovery fraction  : {rec:.1%}  of the achievable imbalance gap")
            mean_cos = sum(r[2] for r in results) / len(results)
            mean_spear = sum(r[3] for r in results) / len(results)
            print(f"  mean count cosine  : {mean_cos:.3f} | mean Spearman: {mean_spear:.3f}")

    # ---- Replicate mode: hot-expert replication vs the re-placement floor ----
    replicate_grid = []  # (hot_n, R, mean_imb, mem_overhead)
    if args.mode == "replicate" and agg:
        hot_ns = sorted({int(x) for x in args.hot_n.split(",")})
        rep_list = sorted({int(x) for x in args.replicas.split(",")})
        # Reference points on the AGGREGATE per-pair counts (apples-to-apples with
        # the grid below, which is also aggregate). Hot set chosen by PREDICTED
        # counts (realistic); imbalance scored on ACTUAL counts.
        base_agg, oracle_agg, floor_agg = [], [], []
        for (_p, _d, cp, ca) in agg:
            ca_l = ca.tolist()
            base_agg.append(imbalance_factor(ca_l, rr, G))
            oracle_agg.append(imbalance_factor(ca_l, balanced_placement(ca_l, G, E), G))
            floor_agg.append(imbalance_factor(ca_l, balanced_placement(cp.tolist(), G, E), G))
        mean_base = sum(base_agg) / len(base_agg)
        mean_oracle = sum(oracle_agg) / len(oracle_agg)
        mean_floor = sum(floor_agg) / len(floor_agg)

        print()
        print("=" * 100)
        print(f"REPLICATION SWEEP across {G} EP GPUs (hot set chosen by PREDICTED counts, "
              f"scored on ACTUAL; aggregate)")
        print(f"  reference: round-robin {mean_base:.3f} | re-placement floor "
              f"(predicted) {mean_floor:.3f} | oracle re-placement {mean_oracle:.3f}")
        print(f"  {'hot-N':>6} {'R':>3} | {'imbalance':>9} | {'mem +%':>7} | "
              f"vs floor")
        print("  " + "-" * 50)
        for hn in hot_ns:
            for R in rep_list:
                imbs, mems = [], []
                for (_p, _d, cp, ca) in agg:
                    hot = set(sorted(range(E), key=lambda e: cp[e].item(), reverse=True)[:hn])
                    imb, mem = replicated_imbalance(ca.tolist(), hot, R, G, E)
                    imbs.append(imb)
                    mems.append(mem)
                mi = sum(imbs) / len(imbs)
                mem = mems[0]  # same for all pairs (= hn*(R-1)/E)
                delta = mean_floor - mi
                replicate_grid.append((hn, R, mi, mem))
                print(f"  {hn:>6} {R:>3} | {mi:>9.3f} | {100*mem:>6.2f}% | "
                      f"{'-' if delta<=0 else f'{delta:.3f} lower'}")
        print()
        print("Interpretation: replication splits each hot expert's load across R GPUs, so")
        print("a single expert no longer pins the busiest GPU -- this is what beats the")
        print("re-placement floor at few-experts-per-GPU. Read the smallest (hot-N, R) that")
        print("brings imbalance near oracle for an acceptable memory %, and recall hot-N is")
        print("chosen from the PREDICTED counts, so this also tests predicted hot-set quality.")

    # ---- Optional CSV + plot ----
    if args.output_dir and results:
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, "load_balance_per_layer.csv")
        with open(csv_path, "w") as f:
            f.write("src,dst,count_cosine,count_spearman,hot_overlap,"
                    "imbalance_baseline,imbalance_oracle,imbalance_predicted,"
                    "recovery_fraction,samples\n")
            for r in results:
                f.write(",".join(str(x) for x in r) + "\n")
        print(f"\nWrote {csv_path}")

        if replicate_grid:
            rpath = os.path.join(args.output_dir, "replication_sweep.csv")
            with open(rpath, "w") as f:
                f.write("hot_n,replicas,mean_imbalance,mem_overhead_fraction\n")
                for (hn, R, mi, mem) in replicate_grid:
                    f.write(f"{hn},{R},{mi:.6f},{mem:.6f}\n")
            print(f"Wrote {rpath}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib unavailable; skipping plot.")
        else:
            fig, ax = plt.subplots(figsize=(max(8, len(results) * 0.3), 5))
            x = list(range(len(results)))
            labels = [f"{p}→{d}" for (p, d, *_rest) in results]
            ax.plot(x, [r[5] for r in results], marker="o", label="baseline (round-robin)")
            ax.plot(x, [r[7] for r in results], marker="s", label="predicted-planned")
            ax.plot(x, [r[6] for r in results], marker="^", label="oracle (actual counts)")
            ax.axhline(1.0, color="k", linestyle=":", linewidth=1, label="perfect balance")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_ylabel(f"Imbalance (max/mean load) across {G} EP GPUs")
            ax.set_xlabel("Consecutive MoE layer pair")
            ax.legend()
            ax.set_title("One-layer-ahead load-aware placement vs static round-robin and oracle")
            out = os.path.join(args.output_dir, "load_balance_imbalance.png")
            fig.tight_layout()
            fig.savefig(out, dpi=120)
            plt.close(fig)
            print(f"Wrote {out}")

    print()
    print("Interpretation:")
    print("  recovery ~100%  -> predicted counts plan placement as well as an oracle;")
    print("                     one-layer-ahead prediction recovers ~all of the imbalance loss.")
    print("  recovery ~0%    -> predicted placement is no better than static round-robin;")
    print("                     aggregate counts are too noisy even though per-token signal exists.")
    print("  baseline≈oracle -> little imbalance to begin with at this EP size (re-placement can't help).")


if __name__ == "__main__":
    main()
