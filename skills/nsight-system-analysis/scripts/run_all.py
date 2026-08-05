#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the full Steps 1–5 pipeline on one or two profiles and write all JSON
intermediates to an output directory.

Usage:
  # Comparative mode
  python run_all.py --profile-a flat.sqlite --profile-b nonflat.sqlite \
                    --yaml taxonomy.yml --out /tmp/analysis/

  # Single-profile mode
  python run_all.py --profile-a profile.sqlite --yaml taxonomy.yml --out /tmp/analysis/

The script:
  1. Runs iter_anchor.py on each profile → windows_<a|b>.json
  2. Runs busy_idle.py on each → busy_<a|b>.json
  3. Runs categorize.py on each → cat_<a|b>.json (full taxonomy + Step 3 view)
  4. Decides Step 4 mode (op-group vs module-slicing) from the Step 3 output
     `module_slicing_recommended` flag (uses OR across both profiles).
  5. If module-slicing: runs module_slice.py on each → mod_<a|b>.json
     If op-group: re-runs categorize with --residual-only → opgroup_<a|b>.json
  6. Runs exposed_comm.py on each → comm_<a|b>.json (skips cleanly if no NCCL).
  7. Writes a top-level summary.json with all key numbers plus the Step 4 decision.

This is a convenience wrapper. For more control, invoke individual scripts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def run(cmd: list[str], capture: bool = True) -> dict | str:
    """Run a script, return parsed JSON from stdout (or text)."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(f"ERROR running: {' '.join(cmd)}\nstderr:\n{result.stderr}\n")
        sys.exit(result.returncode)
    if result.stderr.strip():
        # Forward warnings to the user, but don't fail.
        sys.stderr.write(result.stderr)
    if not capture:
        return result.stdout
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        sys.stderr.write(
            f"Could not parse JSON from {cmd[0]}:\n{result.stdout[:1000]}\n"
        )
        sys.exit(2)


def analyze_profile(
    profile: Path, yaml: Path, out_dir: Path, tag: str, n_iters: int | None
) -> dict:
    """Run Steps 1–5 on a single profile. Returns the assembled summary."""
    summary: dict = {"profile": str(profile), "tag": tag}
    py = sys.executable

    # Step 1: anchor + windows
    cmd = [py, str(SCRIPT_DIR / "iter_anchor.py"), str(profile)]
    if n_iters:
        cmd += ["--n-iters", str(n_iters)]
    anchor = run(cmd)
    windows_path = out_dir / f"windows_{tag}.json"
    windows_path.write_text(json.dumps(anchor))
    summary["step1"] = {
        "anchor": anchor["anchor"]["name"],
        "iter_count_used": anchor["iter_count_used"],
        "median_ms": anchor["median_ms"],
        "min_ms": anchor["min_ms"],
        "max_ms": anchor["max_ms"],
        "cross_check": anchor.get("cross_check"),
    }

    # Step 2: busy/idle
    busy = run(
        [
            py,
            str(SCRIPT_DIR / "busy_idle.py"),
            str(profile),
            "--windows",
            str(windows_path),
            "--yaml",
            str(yaml),
        ]
    )
    (out_dir / f"busy_{tag}.json").write_text(json.dumps(busy))
    summary["step2"] = {
        "median": busy["median"],
        "longest_single_stream_union_ms_median": busy[
            "longest_single_stream_union_ms_median"
        ],
    }

    # Step 3 view: gemm/conv/mha
    cat3 = run(
        [
            py,
            str(SCRIPT_DIR / "categorize.py"),
            str(profile),
            "--yaml",
            str(yaml),
            "--windows",
            str(windows_path),
            "--report-categories",
            "gemm,conv,mha",
        ]
    )
    (out_dir / f"cat_step3_{tag}.json").write_text(json.dumps(cat3))
    summary["step3"] = {
        "per_category": cat3["per_category"],
        "fused_share_of_residual_pct": cat3["fused_share_of_residual_pct"],
        "module_slicing_recommended": cat3["module_slicing_recommended"],
    }

    # Full categorize for uncategorized inspection
    cat_full = run(
        [
            py,
            str(SCRIPT_DIR / "categorize.py"),
            str(profile),
            "--yaml",
            str(yaml),
            "--windows",
            str(windows_path),
        ]
    )
    (out_dir / f"cat_full_{tag}.json").write_text(json.dumps(cat_full))
    summary["uncategorized_above_1pct"] = cat_full["uncategorized_above_threshold"][:10]

    # Step 5: comm
    comm = run(
        [
            py,
            str(SCRIPT_DIR / "exposed_comm.py"),
            str(profile),
            "--yaml",
            str(yaml),
            "--windows",
            str(windows_path),
        ]
    )
    (out_dir / f"comm_{tag}.json").write_text(json.dumps(comm))
    summary["step5"] = comm

    return summary


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--profile-a", required=True, help="Path to first .sqlite")
    p.add_argument("--profile-b", help="Path to second .sqlite (comparative mode)")
    p.add_argument("--yaml", required=True, help="Taxonomy YAML")
    p.add_argument("--out", required=True, help="Output directory for intermediates")
    p.add_argument("--n-iters", type=int, help="Expected iteration count (optional)")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = Path(args.yaml)

    summary: dict = {"mode": "comparative" if args.profile_b else "single"}
    summary["a"] = analyze_profile(
        Path(args.profile_a), yaml_path, out_dir, "a", args.n_iters
    )
    if args.profile_b:
        summary["b"] = analyze_profile(
            Path(args.profile_b), yaml_path, out_dir, "b", args.n_iters
        )

    # Step 4 decision
    fused_a = summary["a"]["step3"]["fused_share_of_residual_pct"]
    fused_b = (
        summary.get("b", {}).get("step3", {}).get("fused_share_of_residual_pct", 0.0)
    )
    use_module_slicing = fused_a > 10.0 or fused_b > 10.0
    summary["step4_mode"] = "module-slicing" if use_module_slicing else "op-group"
    summary["step4_decision"] = {
        "fused_share_a_pct": fused_a,
        "fused_share_b_pct": fused_b if args.profile_b else None,
        "threshold_pct": 10.0,
        "use_module_slicing": use_module_slicing,
    }

    py = sys.executable
    for tag, prof in [("a", args.profile_a)] + (
        [("b", args.profile_b)] if args.profile_b else []
    ):
        windows_path = out_dir / f"windows_{tag}.json"
        if use_module_slicing:
            mod = run(
                [
                    py,
                    str(SCRIPT_DIR / "module_slice.py"),
                    str(prof),
                    "--yaml",
                    str(yaml_path),
                    "--windows",
                    str(windows_path),
                    "--signature-mode",
                    "shape",
                ]
            )
            mod_path = out_dir / f"mod_{tag}.json"
            mod_path.write_text(json.dumps(mod))
            summary[tag]["step4_module_slice"] = {
                "anchor_count_per_iter": mod["anchor_count_per_iter_first"],
                "anchor_count_constant": mod["anchor_count_constant_across_iters"],
                "anchor_overlap_pct": mod["anchor_overlap_pct"],
                "iter_total_anchor_ms": mod["iter_total_anchor_ms_median"],
                "iter_total_window_union_ms": mod["iter_total_window_union_ms_median"],
                "iter_total_window_sum_ms": mod["iter_total_window_sum_ms_median"],
                "top_windows": mod["grouped_windows"][:10],
            }
        else:
            opgroup = run(
                [
                    py,
                    str(SCRIPT_DIR / "categorize.py"),
                    str(prof),
                    "--yaml",
                    str(yaml_path),
                    "--windows",
                    str(windows_path),
                    "--residual-only",
                ]
            )
            (out_dir / f"opgroup_{tag}.json").write_text(json.dumps(opgroup))
            summary[tag]["step4_op_group"] = opgroup["per_category"]

    # Arithmetic invariants (top-level sanity)
    invariants = {}
    for tag in ["a"] + (["b"] if args.profile_b else []):
        s = summary[tag]
        med = s["step2"]["median"]
        invariants[tag] = {
            "iter_eq_busy_plus_idle": round(
                med["iter_ms"] - med["busy_ms"] - med["idle_ms"], 3
            ),
        }
        if "step5" in s and "totals" in s["step5"]:
            recon = s["step5"]["totals"]
            exposed = recon["exposed_ms_median_per_iter"]
            non_nccl = s["step5"]["reconciliation"]["non_nccl_union_ms_median_per_iter"]
            invariants[tag]["exposed_plus_non_nccl_minus_busy_ms"] = round(
                exposed + non_nccl - med["busy_ms"], 3
            )
    summary["invariants"] = invariants

    # Comparative Δ
    if args.profile_b:
        med_a = summary["a"]["step2"]["median"]
        med_b = summary["b"]["step2"]["median"]
        summary["delta"] = {
            "iter_ms": round(med_b["iter_ms"] - med_a["iter_ms"], 3),
            "busy_ms": round(med_b["busy_ms"] - med_a["busy_ms"], 3),
            "idle_ms": round(med_b["idle_ms"] - med_a["idle_ms"], 3),
        }

    # Comparative module-slice diff if both profiles + module-slicing mode.
    if args.profile_b and use_module_slicing:
        diff = run(
            [
                py,
                str(SCRIPT_DIR / "module_diff.py"),
                str(out_dir / "mod_a.json"),
                str(out_dir / "mod_b.json"),
            ]
        )
        (out_dir / "mod_diff.json").write_text(json.dumps(diff))
        summary["step4_module_diff_top"] = diff["signatures"]

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Compact stdout summary — agent reads summary.json for the rest.
    print(f"\nFull summary JSON: {summary_path}")
    print(f"Intermediates in: {out_dir}/")
    print()
    print(f"Mode: {summary['mode']}")
    a = summary["a"]
    print(
        f"  A ({Path(a['profile']).name}): "
        f"anchor={a['step1']['anchor']}, iters_used={a['step1']['iter_count_used']} "
        f"(after dropping warmup+cooldown), "
        f"per_iter_median={a['step1']['median_ms']:.2f} ms"
    )
    if args.profile_b:
        b = summary["b"]
        print(
            f"  B ({Path(b['profile']).name}): "
            f"anchor={b['step1']['anchor']}, "
            f"iters_used={b['step1']['iter_count_used']} "
            f"(after dropping warmup+cooldown), "
            f"per_iter_median={b['step1']['median_ms']:.2f} ms"
        )
        d = summary["delta"]
        print(
            f"  Δ iter={d['iter_ms']:+.2f} ms, "
            f"busy={d['busy_ms']:+.2f}, idle={d['idle_ms']:+.2f}"
        )
    print()
    print(
        f"Step 4 mode: {summary['step4_mode']} "
        f"(fused_share A={fused_a:.1f}%"
        + (f", B={fused_b:.1f}%" if args.profile_b else "")
        + ", threshold=10%)"
    )
    print()
    print("Invariants (should be near zero):")
    for tag, inv in summary["invariants"].items():
        print(f"  {tag}: {inv}")


if __name__ == "__main__":
    main()
