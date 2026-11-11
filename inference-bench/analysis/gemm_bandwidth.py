"""Achieved HBM bandwidth of every GEMM in Ultra v3 decode steps, from an nsys SQLite export.

At decode sizes (a few rows per GPU) every GEMM is a weight read, so its speed of light is
weight bytes / HBM bandwidth. Each GEMM in a decode step (split as in decode_steps.py) gets a
role from its neighbours in time order (the graph replay alternates layers' kernels between two
CUDA streams, so same-stream neighbours are not reliable):

  mamba_in_proj   nvjet followed by causal_conv1d_update
  mamba_out_proj  nvjet after the gated _layer_norm_fwd_1pass_kernel
  attn_qkv        nvjet followed by the qkv all-gather / flash attention
  attn_proj       nvjet after the flash-attention combine kernel
  router          the fp32 nvjet (tss) followed by the router's sigmoid kernel or sigmoid top-k
  shared_fc1      nvjet after the shared expert's TP all-gather, before its squared-ReLU (pow_relu)
  shared_fc2      nvjet after that squared-ReLU
  latent_fc1      nvjet followed by the MoE dispatch (AGV / a2a dispatch / mask)
  latent_fc2      nvjet after the MoE combine (RSV / a2a combine)
  routed_fc1/fc2  the two _fused_moe_kernel launches of a layer
  lm_head         nvjet after the last-token gather (final norm -> hidden all-gather -> gather)
A split-K reduce kernel is charged to the GEMM it follows on the same stream.

Weight bytes per GPU come from Ultra v3's shapes and the run's TP (config.env). Routed experts
assume every local expert is hit (true at decode batch >= ~64 for top-22 of 512).

    python3 inference-bench/analysis/gemm_bandwidth.py inference-bench/experiments/<trace run> [--rank 4]
"""

import argparse
import os
import re
import sqlite3
import statistics as st
from collections import defaultdict

PEAK_TBS = 8.0       # GB200 HBM3e spec
ACHIEVABLE_TBS = 7.0  # realistic sustained read bandwidth

# Ultra v3 shapes.
H, VOCAB = 8192, 131072
MAMBA_HEADS, MAMBA_HEAD_DIM, MAMBA_GROUPS, MAMBA_STATE = 256, 64, 8, 128
ATTN_HEADS, KV_HEADS, HEAD_DIM = 64, 2, 128
EXPERTS, EP, LATENT, EXPERT_FFN, SHARED_FFN = 512, 64, 2048, 5120, 10240
MOE_LAYERS = 48
MAMBA_LAYERS, ATTN_LAYERS = 48, 12
EXPECTED_CALLS = {
    "mamba_in_proj": MAMBA_LAYERS, "mamba_out_proj": MAMBA_LAYERS,
    "attn_qkv": ATTN_LAYERS, "attn_proj": ATTN_LAYERS,
    "router": MOE_LAYERS, "latent_fc1": MOE_LAYERS, "latent_fc2": MOE_LAYERS,
    "shared_fc1": MOE_LAYERS, "shared_fc2": MOE_LAYERS,
    "routed_fc1": MOE_LAYERS, "routed_fc2": MOE_LAYERS, "lm_head": 1,
}


def weight_bytes(tp):
    d_inner = MAMBA_HEADS * MAMBA_HEAD_DIM
    local_experts = EXPERTS // EP
    return {
        "mamba_in_proj": H * (2 * d_inner + 2 * MAMBA_GROUPS * MAMBA_STATE + MAMBA_HEADS) // tp * 2,
        "mamba_out_proj": d_inner // tp * H * 2,
        "attn_qkv": H * ((ATTN_HEADS + 2 * KV_HEADS) * HEAD_DIM) // tp * 2,
        "attn_proj": ATTN_HEADS * HEAD_DIM // tp * H * 2,
        "router": EXPERTS * H * 4,
        "latent_fc1": H * LATENT * 2,
        "latent_fc2": LATENT * H * 2,
        "shared_fc1": H * SHARED_FFN // tp * 2,
        "shared_fc2": SHARED_FFN // tp * H * 2,
        "routed_fc1": local_experts * LATENT * EXPERT_FFN * 2,
        "routed_fc2": local_experts * EXPERT_FFN * LATENT * 2,
        "lm_head": VOCAB // tp * H * 2,
    }


def is_gemm(n):
    return "nvjet" in n or ("gemm" in n.lower() and "cutlass" in n)


def is_reduce(n):
    return "splitKreduce" in n


def label(ks, i):
    """Role of the GEMM at index i of a step's kernels (time order, all streams)."""
    name = ks[i][0]
    nxt = [k[0] for k in ks[i + 1 : i + 6] if not is_reduce(k[0])]
    prv = [k[0] for k in ks[max(0, i - 4) : i] if not is_reduce(k[0])][::-1]
    has = lambda names, *subs: any(x in n for n in names for x in subs)
    if "tss" in name:
        return "router"
    if has(nxt[:3], "causal_conv1d"):
        return "mamba_in_proj"
    if has(prv[:2], "_layer_norm_fwd"):
        return "mamba_out_proj"
    if has(prv[:2], "vectorized_gather"):
        return "lm_head"  # after the final norm, hidden-state all-gather and last-token gather
    if has(nxt, "CatArray", "_append_kv_cache", "flash_fwd_sm100F"):
        return "attn_qkv"  # its output is all-gathered across TP, then attention runs
    if has(prv[:2], "fwd_combine", "fwd_comb"):
        return "attn_proj"
    if has(prv[:2], "pow_relu"):
        return "shared_fc2"
    if has(prv[:3], "_multimem_all_gather_kernel") and has(nxt, "pow_relu"):
        return "shared_fc1"
    if has(nxt[:4], "all_gatherv_3tensor", "_a2a_dispatch", "_mask_routing"):
        return "latent_fc1"
    if has(prv[:2], "reduce_scatter_v", "_a2a_combine"):
        return "latent_fc2"
    return "other"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--rank", type=int, default=4)
    args = p.parse_args()

    tp = int(re.search(r"\bTP=(\d+)", open(os.path.join(args.run, "config.env")).read()).group(1))
    wb = weight_bytes(tp)
    con = sqlite3.connect(os.path.join(args.run, f"trace_rank{args.rank}.sqlite"))
    rows = con.execute(
        "select s.value, k.start, k.end, k.streamId from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds s on k.shortName = s.id order by k.start"
    ).fetchall()
    bounds = [i for i, r in enumerate(rows) if r[0] == "_fused_metadata_kernel"]

    gemm_us = defaultdict(list)    # role -> [gemm us] over all decode steps
    reduce_us = defaultdict(list)  # role -> [reduce us]
    per_step_calls = defaultdict(list)
    n_steps = 0
    for a, b in zip(bounds, bounds[1:]):
        ks = rows[a:b]
        moe = [(e - s) / 1000 for n, s, e, _ in ks if n == "_fused_moe_kernel"]
        if len(moe) != 2 * MOE_LAYERS or max(moe) > 100 or st.median(moe) < 10:
            continue  # partial, prefill or near-empty step
        n_steps += 1
        calls = defaultdict(int)
        for j, d in enumerate(moe):
            role = "routed_fc1" if j % 2 == 0 else "routed_fc2"
            gemm_us[role].append(d)
            calls[role] += 1
        unlabeled = []
        for i, (n, s, e, stream) in enumerate(ks):
            if not is_gemm(n):
                continue
            role = label(ks, i)
            # Split-K reduce: the next kernel on this GEMM's stream, if it is one.
            red = 0.0
            for k in ks[i + 1 : i + 6]:
                if k[3] == stream:
                    if is_reduce(k[0]):
                        red = (k[2] - k[1]) / 1000
                    break
            if role == "other":
                unlabeled.append(((e - s) / 1000, red, n))
                continue
            gemm_us[role].append((e - s) / 1000)
            reduce_us[role].append(red)
            calls[role] += 1
        for g, r, n in unlabeled:
            gemm_us["other"].append(g)
            reduce_us["other"].append(r)
            calls["other"] += 1
        for role, c in calls.items():
            per_step_calls[role].append(c)

    print(f"{args.run} rank {args.rank}: TP{tp}, {n_steps} decode steps. Medians per call; "
          f"peak {PEAK_TBS} TB/s, speed of light at {ACHIEVABLE_TBS} TB/s.\n")
    hdr = ("| GEMM | Calls/step (measured) | Weight/GPU (MB) | GEMM (us) | Split-K reduce (us) | Total (us) | "
           "Achieved (TB/s) | % of peak | SOL (us) | Per step (us) | Gap to SOL per step (us) |")
    print(hdr)
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    order = ["mamba_in_proj", "mamba_out_proj", "attn_qkv", "attn_proj", "router", "latent_fc1",
             "shared_fc1", "shared_fc2", "routed_fc1", "routed_fc2", "latent_fc2", "lm_head", "other"]
    tot_step = tot_sol = tot_bytes = 0.0
    for role in order:
        if role not in gemm_us:
            continue
        measured = st.median(per_step_calls[role]) if per_step_calls[role] else 0
        calls = EXPECTED_CALLS.get(role, measured)
        g = st.median(gemm_us[role])
        r = st.median(reduce_us[role]) if reduce_us.get(role) else 0.0
        t = g + r
        if role in wb:
            mb = wb[role] / 1e6
            tbs = wb[role] / (t * 1e-6) / 1e12
            sol = wb[role] / (ACHIEVABLE_TBS * 1e12) * 1e6
            pct = f"{100 * tbs / PEAK_TBS:.0f}%"
            sol_s = f"{sol:.1f}"
            tot_sol += sol * calls
            tot_bytes += wb[role] * calls
            tbs_s, mb_s = f"{tbs:.2f}", f"{mb:.1f}"
            gap = (t - sol) * calls
            gap_s = f"{gap:.0f}"
        else:
            mb_s = tbs_s = pct = sol_s = gap_s = "-"
        per_step = t * calls
        tot_step += per_step
        print(f"| {role} | {calls:.0f} ({measured:.0f}) | {mb_s} | {g:.2f} | {r:.2f} | {t:.2f} | {tbs_s} | {pct} | "
              f"{sol_s} | {per_step:.0f} | {gap_s} |")
    print(f"| **Total** | | {tot_bytes / 1e6:.0f} MB/step | | | | "
          f"{tot_bytes / (tot_step * 1e-6) / 1e12:.2f} avg | {100 * tot_bytes / (tot_step * 1e-6) / 1e12 / PEAK_TBS:.0f}% | "
          f"{tot_sol:.0f} | {tot_step:.0f} | {tot_step - tot_sol:.0f} |")


if __name__ == "__main__":
    main()
