#!/usr/bin/env python3
"""Compare global gradient norm logs produced by Megatron-LM."""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _load_grad_norms(path: Path) -> Dict[int, float]:
    """Return a mapping from iteration index to global grad norm."""
    entries: Dict[int, float] = {}
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            iteration = int(record['iteration'])
            entries[iteration] = float(record['grad_norm'])
    return entries


def _percentile(sorted_values: List[float], quantile: float) -> float:
    if not sorted_values:
        return float('nan')
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * quantile))))
    return sorted_values[index]


def compare(baseline: Path, candidate: Path, atol: float, rtol: float, top_k: int) -> None:
    base_map = _load_grad_norms(baseline)
    cand_map = _load_grad_norms(candidate)

    all_iters = sorted(set(base_map) | set(cand_map))

    missing_in_baseline = [iteration for iteration in all_iters if iteration not in base_map]
    missing_in_candidate = [iteration for iteration in all_iters if iteration not in cand_map]

    matched = []
    for iteration in all_iters:
        if iteration not in base_map or iteration not in cand_map:
            continue
        base_value = base_map[iteration]
        cand_value = cand_map[iteration]
        abs_diff = abs(base_value - cand_value)
        denom = max(abs(base_value), abs(cand_value), 1e-12)
        tol = atol + rtol * denom
        rel_diff = abs_diff / denom if denom > 0 else float('inf')
        within_tol = abs_diff <= tol
        matched.append((iteration, base_value, cand_value, abs_diff, rel_diff, within_tol))

    failing = [item for item in matched if not item[5]]
    abs_diffs = [item[3] for item in matched]
    rel_diffs = [item[4] for item in matched]

    print(f"Baseline file : {baseline}")
    print(f"Candidate file: {candidate}\n")
    print(f"Iterations compared      : {len(matched)}")
    print(f"Missing in baseline      : {len(missing_in_baseline)}")
    print(f"Missing in candidate     : {len(missing_in_candidate)}")
    print(f"Failing tolerance (atol={atol}, rtol={rtol}) : {len(failing)}\n")

    if abs_diffs:
        sorted_abs = sorted(abs_diffs)
        sorted_rel = sorted(rel_diffs)
        print("Absolute difference stats:")
        print(f"  max   : {max(sorted_abs):.6e}")
        print(f"  p99   : {_percentile(sorted_abs, 0.99):.6e}")
        print(f"  p95   : {_percentile(sorted_abs, 0.95):.6e}")
        print(f"  mean  : {sum(sorted_abs) / len(sorted_abs):.6e}")
        print(f"  median: {sorted_abs[len(sorted_abs)//2]:.6e}\n")

        print("Relative difference stats:")
        print(f"  max   : {max(sorted_rel):.6e}")
        print(f"  p99   : {_percentile(sorted_rel, 0.99):.6e}")
        print(f"  p95   : {_percentile(sorted_rel, 0.95):.6e}")
        print(f"  mean  : {sum(sorted_rel) / len(sorted_rel):.6e}")
        print(f"  median: {sorted_rel[len(sorted_rel)//2]:.6e}\n")

    if missing_in_baseline:
        print("Iterations missing from baseline:")
        for iteration in missing_in_baseline[:top_k]:
            print(f"  iteration {iteration}")
        if len(missing_in_baseline) > top_k:
            print(f"  ... {len(missing_in_baseline) - top_k} more")
        print()

    if missing_in_candidate:
        print("Iterations missing from candidate:")
        for iteration in missing_in_candidate[:top_k]:
            print(f"  iteration {iteration}")
        if len(missing_in_candidate) > top_k:
            print(f"  ... {len(missing_in_candidate) - top_k} more")
        print()

    if failing:
        print("Iterations exceeding tolerance:")
        for entry in sorted(failing, key=lambda item: item[3], reverse=True)[:top_k]:
            iteration, base_value, cand_value, abs_diff, rel_diff, _ = entry
            print(
                f"  iter {iteration} | base={base_value:.6e}, candidate={cand_value:.6e}, "
                f"abs diff={abs_diff:.6e}, rel diff={rel_diff:.6e}"
            )
        if len(failing) > top_k:
            print(f"  ... {len(failing) - top_k} more iterations exceed tolerance")
    else:
        print("All shared iterations are within tolerance.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path, help='Reference JSONL log file')
    parser.add_argument('--candidate', required=True, type=Path, help='JSONL log file to compare against baseline')
    parser.add_argument('--atol', type=float, default=1e-6, help='Absolute tolerance for differences')
    parser.add_argument('--rtol', type=float, default=1e-4, help='Relative tolerance for differences')
    parser.add_argument('--top-k', type=int, default=20, help='Number of detailed entries to print per category')
    args = parser.parse_args()

    compare(args.baseline, args.candidate, args.atol, args.rtol, args.top_k)


if __name__ == '__main__':
    main()
