# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare verified, unbudgeted baselines without pooling samples across runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
from pathlib import Path

import baseline


def _identity(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _source_context(report: dict) -> dict:
    return {
        label: {key: source[key] for key in ("revision", "dirty")}
        for label, source in report["sources"].items()
    }


def _context(report: dict, compare_cache_locations: bool = False) -> dict:
    """Keep every recorded case/runtime setting; omit only allocation identities."""
    signatures: dict[str, dict] = {}
    for run in report["runs"]:
        signatures.setdefault(run["mode"], copy.deepcopy(run["kernel"]["case_signature"]))
    if compare_cache_locations:
        for signature in signatures.values():
            environment = signature["runtime"]["environment"]
            location = environment["TRITON_CACHE_DIR"]
            if location is not None:
                if not isinstance(location, str) or not location:
                    raise ValueError("Explicit cache locations must be nonempty strings")
                environment["TRITON_CACHE_DIR"] = "<explicit cache location>"
    return {
        "measurement": report["measurement"],
        "sources": _source_context(report),
        "signatures": signatures,
        "machine_runtime": {
            key: value
            for key, value in report["machine"].items()
            if key not in ("host", "gpus", "cuda_visible_devices")
        },
        "visible_gpu_count": len(report["machine"]["gpus"]),
    }


def compare(inputs: list[tuple[Path, str]], *, compare_cache_locations: bool = False) -> dict:
    """Reverify pinned bundles, then summarize matching per-run paired estimates.

    Each bundle remains one observation. Device counts describe retained records,
    not independent random samples. No confidence interval across allocations or
    performance budget is inferred from these observed ranges.
    """
    if not inputs:
        raise ValueError("At least one pinned baseline is required")
    groups: dict[str, dict] = {}
    receipts: list[dict] = []
    identities: set[str] = set()
    measurements: set[str] = set()
    for directory, expected_id in inputs:
        if (
            not isinstance(expected_id, str)
            or len(expected_id) != 64
            or any(character not in "0123456789abcdef" for character in expected_id)
        ):
            raise ValueError("A complete baseline SHA-256 is required")
        if expected_id in identities:
            raise ValueError("Repeated baseline identifier; copies are not new measurements")
        identities.add(expected_id)
        receipt = baseline.verify(directory, expected_id)
        manifest_raw = baseline._bytes(directory / baseline.MANIFEST)
        if hashlib.sha256(manifest_raw).hexdigest() != expected_id:
            raise ValueError("Baseline manifest changed after verification")
        manifest = baseline._json(manifest_raw)
        name = "performance/leaderboard.json"
        raw = baseline._bytes(directory / name)
        if manifest["files"][name] != {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }:
            raise ValueError("Leaderboard changed after verification")
        reports = baseline._json(raw)
        receipts.append(receipt)
        for report in reports:
            if any(value["limit"] is not None for value in report["comparisons"].values()):
                raise ValueError(
                    "Calibration requires unbudgeted measurements, not selected gate passes"
                )
            ordered = sorted(
                report["runs"], key=lambda run: (run["pair"], run["revision_label"], run["mode"])
            )
            measurement_id = _identity(
                {
                    "context": _context(report, compare_cache_locations=True),
                    "runs": [
                        {
                            **{key: run[key] for key in ("pair", "revision_label", "mode")},
                            "device_uuid": run["kernel"]["device_uuid"],
                            "samples_ms": run["kernel"]["samples_ms"],
                        }
                        for run in ordered
                    ],
                }
            )
            if measurement_id in measurements:
                raise ValueError("Repeated raw measurement under different baseline metadata")
            measurements.add(measurement_id)
            context = _context(report, compare_cache_locations)
            cohort = _identity(context)
            group = groups.setdefault(cohort, {"cohort_id": cohort, "context": context, "runs": []})
            group["runs"].append(
                {
                    "baseline_id": expected_id,
                    "origin": receipt["origin"],
                    "measurement_id": measurement_id,
                    "device_uuid": ordered[0]["kernel"]["device_uuid"],
                    "machine": report["machine"],
                    "sources": report["sources"],
                    "source_status": receipt["status"],
                    "cache_locations": {
                        run["mode"]: run["kernel"]["case_signature"]["runtime"]["environment"][
                            "TRITON_CACHE_DIR"
                        ]
                        for run in ordered
                    },
                    "comparisons": report["comparisons"],
                }
            )
    for group in groups.values():
        runs = sorted(group["runs"], key=lambda run: run["baseline_id"])
        group.update(
            runs=runs,
            run_count=len(runs),
            distinct_timing_gpus=len({run["device_uuid"] for run in runs}),
            repeat_status="repeated_measurements" if len(runs) > 1 else "single_measurement",
            comparisons={},
        )
        for name in sorted(runs[0]["comparisons"]):
            values = [run["comparisons"][name]["median_ratio"] for run in runs]
            group["comparisons"][name] = {
                "run_medians": values,
                "median_ratio": statistics.median(values),
                "minimum_ratio": min(values),
                "maximum_ratio": max(values),
                "observed_range_percentage_points": 100 * (max(values) - min(values)),
            }
    return {
        "schema_version": 1,
        "kind": "determinism_performance_calibration",
        "status": "report_only",
        "performance_gate": "not_gated",
        "compare_cache_locations": compare_cache_locations,
        "baselines": sorted(receipts, key=lambda receipt: receipt["baseline_id"]),
        "groups": [groups[key] for key in sorted(groups)],
        "scope": (
            "Selected verified unbudgeted local-activation baselines; exact recorded case, "
            "source, runtime and measurement cohorts. Cache paths match unless location-only "
            "comparison is explicitly requested; recorded paths remain visible. Neither path "
            "equality nor grouping proves cache-content or compiled-dispatch equality. "
            "Per-run paired intervals are retained, "
            "not pooled across allocations. Observed ranges do not establish a population "
            "bound, causal explanation, reviewed budget, durable promotion or production performance."
        ),
    }


def markdown_report(report: dict) -> str:
    """Show all cohorts and their run provenance without hiding large observed costs."""
    lines = [
        "# Determinism performance calibration",
        "",
        "Report only; observed ranges are neither performance budgets nor confidence bounds.",
        "Each bundle contributes one run estimate. A GPU count does not prove statistical independence.",
        f"Compare explicit cache locations: {report['compare_cache_locations']}.",
        "",
        "| Cohort | GPU | Case | Dtype | Phase | Comparison | Runs | GPUs | Min ratio | Median ratio | Max ratio | Range (pp) |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    for group in report["groups"]:
        context, cohort = group["context"], group["cohort_id"]
        measurement = context["measurement"]
        identity = (
            f"`{cohort[:12]}` | {cell(context['signatures']['det']['runtime']['gpu'])} | "
            f"{cell(measurement['kernel_case'])} | {cell(measurement['dtype'])} | "
            f"{cell(measurement['phase'])}"
        )
        for name, values in group["comparisons"].items():
            lines.append(
                f"| {identity} | {name} | {group['run_count']} | {group['distinct_timing_gpus']} | "
                f"{values['minimum_ratio']:.6f} | {values['median_ratio']:.6f} | "
                f"{values['maximum_ratio']:.6f} | {values['observed_range_percentage_points']:.4f} |"
            )
    lines.extend(["", "## Input baselines", ""])
    for receipt in report["baselines"]:
        lines.append(
            f"- `{receipt['baseline_id']}`: {cell(receipt['origin'])}; "
            f"revision `{receipt['revision']}`, {receipt['files_verified']} verified files, "
            f"{receipt['author_cases']} author cases, `{receipt['status']}`."
        )
    lines.extend(["", report["scope"], ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Compare explicitly pinned bundles and retain JSON plus a reviewable table."""
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--baseline", nargs=2, action="append", metavar=("PATH", "SHA256"))
    selection.add_argument(
        "--collective-baseline", nargs=2, action="append", metavar=("PATH", "SHA256")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compare-cache-locations",
        action="store_true",
        help="Group different explicit cache paths while retaining them; does not prove cache contents match",
    )
    args = parser.parse_args(argv)
    try:
        if (
            args.output.suffix != ".json"
            or args.output.exists()
            or args.output.with_suffix(".md").exists()
        ):
            raise ValueError("Use a new .json output path and preserve earlier reports")
        compare_runs, render = compare, markdown_report
        if args.collective_baseline:
            import collective_calibration

            compare_runs, render = (
                collective_calibration.compare,
                collective_calibration.markdown_report,
            )
        report = compare_runs(
            [
                (Path(path), identity)
                for path, identity in (args.collective_baseline or args.baseline)
            ],
            compare_cache_locations=args.compare_cache_locations,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        args.output.with_suffix(".md").write_text(render(report))
    except (OSError, AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"status": "invalid", "error": str(error)}))
        return 1
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
