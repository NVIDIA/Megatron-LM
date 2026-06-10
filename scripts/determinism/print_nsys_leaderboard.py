# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Side-by-side leaderboard from ``nsys stats nvtx_sum`` CSVs (det vs nondet).

Usage: ``python print_nsys_leaderboard.py LEADERBOARD_DIR [LOG_DIR]``.
If LOG_DIR is given, also check det/nondet step-time ratio < MAX_DET_NONDET_RATIO.
"""
import csv
import glob
import re
import sys
from pathlib import Path

MAX_DET_NONDET_RATIO = 1.25
MEASUREMENT_ITER = 7  # clean iter inside --profile-step-start=5/end=7 window
LEADERBOARD_TOP_N = 20
# Strip per-call-site ``, op_id = N`` and autograd-engine ``, seq = N`` so
# identical op kinds aggregate across det/nondet.
OP_ID_SUFFIX_RE = re.compile(r",\s*(op_id|seq)\s*=\s*\d+")


def load_nsys_csv(path):
    """Return {range_name: total_ms} from one nsys nvtx_sum CSV."""
    if not path.exists():
        return {}
    with path.open() as f:
        rows = list(csv.reader(f))
    h = next((r for r in rows if "Range" in r and any("Total Time" in c for c in r)), None)
    if h is None:
        return {}
    ti = next(i for i, c in enumerate(h) if "Total Time" in c)
    ri = h.index("Range")
    out = {}
    for r in rows[rows.index(h) + 1 :]:
        if len(r) <= max(ti, ri) or not r[ri].strip() or r[ri] == "Range":
            continue
        name = OP_ID_SUFFIX_RE.sub("", r[ri])
        try:
            out[name] = out.get(name, 0.0) + float(r[ti].replace(",", "")) / 1e6
        except ValueError:
            pass
    return out


def _print_table(title, ranges, det, non, top_n):
    print(f"\n=== {title} (top {top_n} by |det - nondet|) ===")
    ranked = sorted(ranges, key=lambda k: -abs(det.get(k, 0) - non.get(k, 0)))[:top_n]
    if not ranked:
        print("  (no ranges in this bucket)")
        return
    name_w = min(max((len(k) for k in ranked), default=8), 80)
    header = (
        f"{'Range':<{name_w}}  {'det ms':>10}  {'nondet ms':>10}  {'delta ms':>10}  {'delta %':>9}"
    )
    print(header)
    print("-" * len(header))
    for k in ranked:
        a, b = det.get(k), non.get(k)
        delta_ms = (a if a is not None else 0) - (b if b is not None else 0)
        pct = f"{(a - b) / b * 100:+.2f}" if (a is not None and b is not None and b > 0) else "-"
        name = k if len(k) <= name_w else k[: name_w - 1] + "…"
        a_str = "-" if a is None else f"{a:.3f}"
        b_str = "-" if b is None else f"{b:.3f}"
        print(f"{name:<{name_w}}  {a_str:>10}  {b_str:>10}  {delta_ms:>+10.3f}  {pct:>9}")


def _phase(name):
    """forward = dotted mcore path, backward = ``Backward`` substring, op = rest."""
    if "Backward" in name:
        return "backward"
    if "." in name and "::" not in name:
        return "forward"
    return "op"


def print_leaderboard(det, non, top_n=LEADERBOARD_TOP_N):
    buckets = {"forward": set(), "backward": set(), "op": set()}
    for k in set(det) | set(non):
        buckets[_phase(k)].add(k)
    _print_table("forward — mcore module ranges", buckets["forward"], det, non, top_n)
    _print_table("backward — autograd engine ranges", buckets["backward"], det, non, top_n)
    _print_table("op-level — aten / NCCL / kernels", buckets["op"], det, non, top_n)


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
    print_leaderboard(det, non)
    if len(sys.argv) > 2 and sys.argv[2]:
        failure = check_step_time_ratio(sys.argv[2])
        if failure:
            sys.exit(f"FAIL: {failure}")


if __name__ == "__main__":
    main()
