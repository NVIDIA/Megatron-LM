# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Side-by-side leaderboard from ``nsys stats nvtx_sum`` CSVs (det vs nondet).

Usage: ``python print_nsys_leaderboard.py LEADERBOARD_DIR [LOG_DIR]``.
If LOG_DIR is given, also check det/nondet step-time ratio < MAX_DET_NONDET_RATIO.
"""

import csv
import glob
import os
import re
import sys
from pathlib import Path

MAX_DET_NONDET_RATIO = 1.35
MEASUREMENT_ITER = 5  # steady-state; iter 7 is noisy under nsys profile teardown
LEADERBOARD_TOP_N = int(os.environ.get("NSYS_LEADERBOARD_TOP_N", "20"))  # 0 = all rows
# nvtx_sum column to compare. "Med" is robust to a one-off compile inside a range.
COLUMN = os.environ.get("NSYS_LEADERBOARD_COLUMN", "Total Time")
# Strip per-call-site ``, op_id = N`` and autograd-engine ``, seq = N`` so
# identical op kinds aggregate across det/nondet.
OP_ID_SUFFIX_RE = re.compile(r",\s*(op_id|seq)\s*=\s*\d+")


def load_nsys_csv(path):
    """Return {range_name: ms} (the ``COLUMN`` value) from one nsys nvtx_sum CSV."""
    if not path.exists():
        return {}
    with path.open() as f:
        rows = list(csv.reader(f))
    h = next((r for r in rows if "Range" in r and any(COLUMN in c for c in r)), None)
    if h is None:
        return {}
    ti = next(i for i, c in enumerate(h) if COLUMN in c)
    ri = h.index("Range")
    out, merged = {}, set()
    for r in rows[rows.index(h) + 1 :]:
        if len(r) <= max(ti, ri) or not r[ri].strip() or r[ri] == "Range":
            continue
        name = OP_ID_SUFFIX_RE.sub("", r[ri])
        try:
            value = float(r[ti].replace(",", "")) / 1e6
        except ValueError:
            continue
        if name in out:
            merged.add(name)
        out[name] = out.get(name, 0.0) + value
    if COLUMN != "Total Time":
        # Totals add up when per-call-site names merge; medians and averages do not.
        for name in merged:
            out[name] = None
    return {k: v for k, v in out.items() if v is not None}


def load_nsys_spread(path):
    """Return {range_name: (instances, min_ms, max_ms)} from one nsys nvtx_sum CSV."""
    if not path.exists():
        return {}
    with path.open() as f:
        rows = list(csv.reader(f))
    h = next((r for r in rows if "Range" in r and "Instances" in r), None)
    if h is None:
        return {}
    ni, ri = h.index("Instances"), h.index("Range")
    lo = next(i for i, c in enumerate(h) if c.startswith("Min"))
    hi = next(i for i, c in enumerate(h) if c.startswith("Max"))
    out = {}
    for r in rows[rows.index(h) + 1 :]:
        if len(r) <= max(ni, ri, lo, hi) or not r[ri].strip() or r[ri] == "Range":
            continue
        name = OP_ID_SUFFIX_RE.sub("", r[ri])
        try:
            n, a, b = int(r[ni]), float(r[lo]) / 1e6, float(r[hi]) / 1e6
        except ValueError:
            continue
        n0, a0, b0 = out.get(name, (0, a, b))
        out[name] = (n0 + n, min(a0, a), max(b0, b))
    return out


def _verdict(d, n):
    """Per-call range separation within ONE run: ``det slower``/``det faster`` only when the
    det and nondet [min, max] per-call ranges do not overlap. With a few calls per side this
    is not a performance conclusion on its own: rerun with the other pass order
    (``DETERMINISM_PERF_ORDER``) and keep the rows that agree. Ranges with one call per side
    get ``-``."""
    if d is None or n is None or d[0] < 2 or n[0] < 2:
        return "-"
    if d[1] > n[2]:
        return "det slower"
    if d[2] < n[1]:
        return "det faster"
    return "overlap"


def _print_table(title, ranges, det, non, top_n, det_spread, non_spread):
    shown = f"top {top_n}" if top_n else "all"
    print(f"\n=== {title} ({shown} by |det - nondet|) ===")
    ranked = sorted(ranges, key=lambda k: -abs(det.get(k, 0) - non.get(k, 0)))
    ranked = ranked[:top_n] if top_n else ranked
    if not ranked:
        print("  (no ranges in this bucket)")
        return
    name_w = max(len(k) for k in ranked)
    header = (
        f"{'Range':<{name_w}}  {'det ms':>10}  {'nondet ms':>10}  {'delta ms':>10}  {'delta %':>9}"
        f"  {'det n [min-max]':>24}  {'nondet n [min-max]':>24}  per-call range (1 run)"
    )
    print(header)
    print("-" * len(header))
    for k in ranked:
        a, b = det.get(k), non.get(k)
        delta_ms = (a if a is not None else 0) - (b if b is not None else 0)
        pct = f"{(a - b) / b * 100:+.2f}" if (a is not None and b is not None and b > 0) else "-"
        a_str = "-" if a is None else f"{a:.3f}"
        b_str = "-" if b is None else f"{b:.3f}"
        ds, ns = det_spread.get(k), non_spread.get(k)
        ds_str = "-" if ds is None else f"{ds[0]} [{ds[1]:.3f}-{ds[2]:.3f}]"
        ns_str = "-" if ns is None else f"{ns[0]} [{ns[1]:.3f}-{ns[2]:.3f}]"
        print(
            f"{k:<{name_w}}  {a_str:>10}  {b_str:>10}  {delta_ms:>+10.3f}  {pct:>9}"
            f"  {ds_str:>24}  {ns_str:>24}  {_verdict(ds, ns)}"
        )


def write_rows(path, det, non, det_spread, non_spread):
    """Every range of both passes, one CSV row each, with spread and per-call range separation."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "range",
                "det_ms",
                "nondet_ms",
                "det_n",
                "det_min_ms",
                "det_max_ms",
                "nondet_n",
                "nondet_min_ms",
                "nondet_max_ms",
                "per_call_range_1run",
            ]
        )
        for k in sorted(set(det) | set(non)):
            ds, ns = det_spread.get(k), non_spread.get(k)
            w.writerow(
                [
                    k,
                    det.get(k, ""),
                    non.get(k, ""),
                    *(ds or ("", "", "")),
                    *(ns or ("", "", "")),
                    _verdict(ds, ns),
                ]
            )


def _phase(name):
    """forward = dotted mcore path, backward = ``Backward`` substring, op = rest."""
    if "Backward" in name:
        return "backward"
    if "." in name and "::" not in name:
        return "forward"
    return "op"


def print_leaderboard(det, non, det_spread, non_spread, top_n=LEADERBOARD_TOP_N):
    buckets = {"forward": set(), "backward": set(), "op": set()}
    for k in set(det) | set(non):
        buckets[_phase(k)].add(k)
    spread = (det_spread, non_spread)
    _print_table("forward — mcore module ranges", buckets["forward"], det, non, top_n, *spread)
    _print_table("backward — autograd engine ranges", buckets["backward"], det, non, top_n, *spread)
    _print_table("op-level — aten / NCCL / kernels", buckets["op"], det, non, top_n, *spread)


def step_time_from_log_dir(log_dir, mode, iteration):
    """Read ``elapsed time per iteration (ms)`` for ``iteration`` from torchrun stdout."""
    pat = re.compile(r"iteration\s+(\d+)/\s*\d+.*elapsed time per iteration \(ms\):\s*([\d.]+)")
    pattern = f"{glob.escape(log_dir)}/torchrun-{mode}/**/stdout.log"
    for path in glob.glob(pattern, recursive=True):
        with open(path) as f:
            for line in f:
                m = pat.search(line)
                if m and int(m.group(1)) == iteration:
                    return float(m.group(2))
    return None


def check_step_time_ratio(log_dir):
    det_ms = step_time_from_log_dir(log_dir, "det", MEASUREMENT_ITER)
    non_ms = step_time_from_log_dir(log_dir, "nondet", MEASUREMENT_ITER)
    if det_ms is None or non_ms is None:
        return f"missing step time for iter {MEASUREMENT_ITER} (det={det_ms}, nondet={non_ms})"
    ratio = det_ms / non_ms
    print(
        f"\nstep_time iter={MEASUREMENT_ITER}: det={det_ms:.2f}ms nondet={non_ms:.2f}ms "
        f"ratio={ratio:.2f}x (threshold {MAX_DET_NONDET_RATIO:.2f}x)"
    )
    if ratio > MAX_DET_NONDET_RATIO:
        return f"det {ratio:.2f}x slower than nondet (> {MAX_DET_NONDET_RATIO:.2f}x)"
    return None


def main():
    leaderboard_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/perf-leaderboards")
    det = load_nsys_csv(leaderboard_dir / "nsys-det.csv")
    non = load_nsys_csv(leaderboard_dir / "nsys-nondet.csv")
    if not (det and non):
        sys.exit(f"need both CSVs: det={len(det)}, nondet={len(non)} rows")
    det_spread = load_nsys_spread(leaderboard_dir / "nsys-det.csv")
    non_spread = load_nsys_spread(leaderboard_dir / "nsys-nondet.csv")
    print_leaderboard(det, non, det_spread, non_spread)
    write_rows(leaderboard_dir / "leaderboard.csv", det, non, det_spread, non_spread)
    print(f"\nall rows: {leaderboard_dir / 'leaderboard.csv'}")
    if len(sys.argv) > 2 and sys.argv[2]:
        failure = check_step_time_ratio(sys.argv[2])
        if failure:
            sys.exit(f"FAIL: {failure}")


if __name__ == "__main__":
    main()
