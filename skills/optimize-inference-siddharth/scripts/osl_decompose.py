"""Three-way decomposition: prefill, decode step vs context, for both engines.

The client reports tpot = wall / num_output_tokens, which folds prefill into every
token. With a fixed prompt set and batch, prefill is the *same constant* in every
wall time, so it cancels in a difference between two OSLs:

    wall(N) = prefill + sum_{i=1..N-1} step(c0 + i)
    (wall(N2) - wall(N1)) / (N2 - N1) = mean step over contexts [c0+N1, c0+N2-1]

Each consecutive pair therefore yields one clean measurement of the decode step at a
known mean context, with no profiler and no prefill contamination. Fitting
step(c) = a + b*c to those points gives the context-scaling slope directly, and
back-solving prefill from every OSL independently checks the linear model: if the
model is wrong, the recovered prefill will not be constant.

The slope is also checked against the KV-cache bandwidth floor, since re-reading the
KV cache is decode's only meaningful per-context-token cost. A slope already at that
floor has no headroom regardless of how it compares to a competitor.

Input: one directory per engine containing `osl_<N>.log` files, each the stdout of a
benchmark client that prints `wall=<ms>` per iteration. Generate them by running the
same client against the same server at several OSLs, warming up once at the longest
OSL first so graph capture and any autotuning land outside every timed point.

Usage:
  python osl_decompose.py <dir_a> <dir_b> [--label-a A] [--label-b B]
      [--prompt-tokens F] [--seqs-per-gpu-a N] [--seqs-per-gpu-b N]
      [--bandwidth-tbs F] [--kv-bytes-per-ctx-token-per-seq F]

Defaults describe Qwen3-30B-A3B on 4xGB200 (mcore EP4/TP1 vs vLLM DP4); override
them for any other model or sharding.
"""

import argparse
import re
import statistics

OSLS = [64, 128, 256, 512, 768, 1024, 1152]

ap = argparse.ArgumentParser()
ap.add_argument("dir_a")
ap.add_argument("dir_b")
ap.add_argument("--label-a", default="engine A")
ap.add_argument("--label-b", default="engine B")
ap.add_argument("--prompt-tokens", type=float, default=60.86328125,
                help="mean prompt length; must be identical across all OSL points")
ap.add_argument("--seqs-per-gpu-a", type=int, default=64,
                help="sequences whose KV this GPU reads each step (A)")
ap.add_argument("--seqs-per-gpu-b", type=int, default=72)
ap.add_argument("--bandwidth-tbs", type=float, default=7.0,
                help="achievable HBM streaming ceiling, TB/s")
ap.add_argument("--kv-bytes-per-ctx-token-per-seq", type=float, default=96.0 * 1024,
                help="2*kv_heads*head_dim*dtype_bytes*layers")
ap.add_argument("--osls", type=int, nargs="+", default=OSLS)
args = ap.parse_args()

C0 = args.prompt_tokens
BW_TBS = args.bandwidth_tbs
OSLS = args.osls


def load(run_dir, osls=OSLS):
    out = {}
    for osl in osls:
        try:
            txt = open(f"{run_dir}/osl_{osl}.log").read()
        except FileNotFoundError:
            continue
        walls = [float(m) for m in re.findall(r"wall=(\d+(?:\.\d+)?) ms", txt)]
        if walls:
            out[osl] = walls
    return out


def fit(points):
    """Least squares a + b*c, with the standard error on the slope.

    The slope is the whole argument, so it needs an error bar: the finite
    differences are individually noisy and a slope difference inside the error
    bars is not a finding.
    """
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    den = n * sxx - sx * sx
    b = (n * sxy - sx * sy) / den
    a = (sy - b * sx) / n
    resid = [p[1] - (a + b * p[0]) for p in points]
    s2 = sum(r * r for r in resid) / (n - 2)
    se_b = (s2 * n / den) ** 0.5
    ybar = sy / n
    sstot = sum((p[1] - ybar) ** 2 for p in points)
    r2 = 1 - sum(r * r for r in resid) / sstot if sstot else float("nan")
    return a, b, se_b, r2


def analyse(label, run_dir, seqs_per_gpu):
    data = load(run_dir)
    if not data:
        print(f"\n### {label}: no data at {run_dir}")
        return None
    print(f"\n### {label}")
    print(f"{'OSL':>6} {'mean wall ms':>13} {'sd':>7} {'spread':>8} {'n':>3} {'tpot ms':>9}")
    means = {}
    for osl, walls in sorted(data.items()):
        m = statistics.mean(walls)
        sd = statistics.stdev(walls) if len(walls) > 1 else 0.0
        means[osl] = m
        print(f"{osl:6d} {m:13.1f} {sd:7.1f} "
              f"{(max(walls)-min(walls))/m*100:7.2f}% {len(walls):3d} {m/osl:9.4f}")

    print(f"\n  finite differences (prefill cancels):")
    print(f"  {'interval':>14} {'mean context':>13} {'decode step ms':>15}")
    pts = []
    ks = sorted(means)
    for n1, n2 in zip(ks, ks[1:]):
        step = (means[n2] - means[n1]) / (n2 - n1)
        ctx = C0 + (n1 + n2 - 1) / 2.0
        pts.append((ctx, step))
        print(f"  {f'{n1}->{n2}':>14} {ctx:13.1f} {step:15.4f}")

    a, b, se_b, r2 = fit(pts)
    print(f"\n  fit: step(c) = {a:.4f} + {b:.6f} * c  ms   (R2 {r2:.3f})")
    print(f"       context-free term {a:.4f} ms")
    print(f"       context slope {b*1000:.3f} +/- {se_b*1000:.3f} us per context token")
    # The decode KV read is the only per-context-token cost of consequence:
    # 2(K,V) * 4 kv-heads * 128 dim * 2 bytes * 48 layers = 96 KiB per context
    # token per sequence, streamed once per step by whichever sequences this GPU
    # owns. That sets a hard floor on the slope.
    gb_per_ctx_tok = (args.kv_bytes_per_ctx_token_per_seq * seqs_per_gpu
                      / 1024 ** 3)
    floor_us = gb_per_ctx_tok / BW_TBS * 1e3
    print(f"       KV floor at {seqs_per_gpu} seqs/GPU: {gb_per_ctx_tok*1024:.2f} MB "
          f"per context token => {floor_us:.3f} us at {BW_TBS} TB/s")
    print(f"       measured / floor = {b*1000/floor_us:.2f}x")

    def mean_step(n):
        return a + b * (C0 + n / 2.0)

    print(f"\n  {'OSL':>6} {'mean-context step':>18} {'recovered prefill ms':>21}")
    prefills = []
    for osl in ks:
        pf = means[osl] - sum(a + b * (C0 + i) for i in range(1, osl))
        prefills.append(pf)
        print(f"  {osl:6d} {mean_step(osl):18.4f} {pf:21.1f}")
    print(f"  recovered prefill: mean {statistics.mean(prefills):.1f} ms, "
          f"sd {statistics.stdev(prefills):.1f} ms "
          f"(constant => the linear step model holds)")
    return {
        "a": a, "b": b, "se_b": se_b, "means": means,
        "step128": mean_step(128), "step1024": mean_step(1024),
        "prefill": statistics.mean(prefills),
        "prefill_sd": statistics.stdev(prefills),
    }


# seqs-per-gpu is the count whose KV this rank reads each step, and it is easy to get
# wrong: read it off a per-step kernel's grid dimension (e.g. the KV-write kernel)
# rather than assuming batch/world_size.
m = analyse(args.label_a, args.dir_a, args.seqs_per_gpu_a)
v = analyse(args.label_b, args.dir_b, args.seqs_per_gpu_b)

if m and v:
    print("\n\n########## THREE-WAY DECOMPOSITION ##########")
    print(f"{'quantity':<34} {'mcore':>11} {'vLLM':>11} {'ratio':>8}")
    print("-" * 68)
    rows = [
        ("prefill (ms, BS256)", m["prefill"], v["prefill"]),
        ("pure decode step @OSL128 (ms)", m["step128"], v["step128"]),
        ("pure decode step @OSL1024 (ms)", m["step1024"], v["step1024"]),
        ("context slope (us/context token)", m["b"] * 1000, v["b"] * 1000),
        ("context-free term (ms)", m["a"], v["a"]),
        ("client tpot @OSL1024 (ms)", m["means"].get(1024, 0) / 1024,
         v["means"].get(1024, 0) / 1024),
    ]
    for name, mv, vv in rows:
        print(f"{name:<34} {mv:11.4f} {vv:11.4f} {mv/vv:8.3f}x")

    print("\n--- is the context-scaling lever real? ---")
    gm = m["step1024"] / m["step128"]
    gv = v["step1024"] / v["step128"]
    print(f"  pure-decode growth OSL128 -> OSL1024:  mcore {gm:.4f}x, vLLM {gv:.4f}x")
    if_v = m["step128"] * gv
    print(f"  mcore @OSL1024 if it scaled at vLLM's rate: {if_v:.4f} ms")
    print(f"  excess attributable to context scaling:     {m['step1024']-if_v:+.4f} ms/step")
    deficit = m["step1024"] - v["step1024"]
    print(f"  pure-decode deficit at OSL1024:             {deficit:+.4f} ms/step")
    if deficit:
        print(f"  context scaling is {(m['step1024']-if_v)/deficit*100:.1f}% of that deficit")
    print("\n--- where the OSL1024 deficit actually lives ---")
    cm = C0 + 1024 / 2.0
    print(f"  context-free term:     mcore {m['a']:8.4f}  vLLM {v['a']:8.4f}  "
          f"delta {m['a']-v['a']:+8.4f} ms")
    print(f"  context term @c={cm:.0f}:  mcore {m['b']*cm:8.4f}  vLLM {v['b']*cm:8.4f}  "
          f"delta {(m['b']-v['b'])*cm:+8.4f} ms")
    tot = (m['a']-v['a']) + (m['b']-v['b'])*cm
    print(f"  total                                                delta {tot:+8.4f} ms")
    print(f"  => {(m['a']-v['a'])/tot*100:.1f}% of the deficit is context-FREE per-step work")

    print(f"\n  parity at the OSL1024 pure decode step needs "
          f"{(1 - v['step1024']/m['step1024'])*100:.1f}% off mcore's step")
    print(f"  parity at the client throughput metric needs "
          f"{(m['means'][1024]/v['means'][1024] - 1)*100:.1f}% more throughput")
