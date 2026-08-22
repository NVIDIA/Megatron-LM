#!/usr/bin/env python3

"""Merge and summarize rank-local DSA diagnostic JSONL shards."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable, List


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _expand_globs(patterns: Iterable[str]) -> List[Path]:
    paths = []
    for pattern in patterns:
        paths.extend(Path(path) for path in glob.glob(pattern))
    return sorted(set(paths))


def _read_records(paths: Iterable[Path]) -> List[dict]:
    records = []
    seen = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("schema_version") != 1:
                    raise ValueError(
                        f"Unsupported schema in {path}:{line_number}: "
                        f"{record.get('schema_version')}"
                    )
                record["source_file"] = str(path)
                identity = (
                    record.get("dp_rank"),
                    record.get("request_id"),
                    record.get("layer"),
                    record.get("query_position"),
                )
                if identity in seen:
                    raise ValueError(
                        f"Duplicate diagnostic record {identity} in {path}:{line_number}; "
                        f"first seen in {seen[identity]}. Use a fresh output directory per run."
                    )
                seen[identity] = f"{path}:{line_number}"
                records.append(record)
    return records


def _flatten_support_records(records: Iterable[dict]) -> List[dict]:
    flattened = []
    metadata_fields = (
        "dp_rank",
        "pp_rank",
        "request_id",
        "layer",
        "phase",
        "offset",
        "query_position",
        "prompt_length",
        "context_length",
        "model_topk",
        "indexer_mode",
        "valid_key_count",
    )
    metric_fields = (
        "captured_mass_mean",
        "captured_mass_min",
        "output_relative_l2_mean",
        "output_relative_l2_max",
        "output_cosine_mean",
        "actual_output_relative_l2_mean",
        "actual_output_relative_l2_max",
        "actual_output_cosine_mean",
        "sum_oracle_recall",
        "max_oracle_recall",
        "head_support_union_size",
        "support_fraction_first_16",
        "support_fraction_last_128",
        "support_fraction_last_1024",
        "support_fraction_last_4096",
        "support_fraction_last_16384",
    )
    for record in records:
        metadata = {field: record.get(field) for field in metadata_fields}
        model_metrics = record.get("model_support", {})
        flattened.append(
            {
                **metadata,
                "topk": record.get("model_topk"),
                "support_type": "model_support",
                **{field: model_metrics.get(field) for field in metric_fields},
            }
        )
        for topk, support_types in record.get("supports", {}).items():
            for support_type, metrics in support_types.items():
                if support_type == "support_indices":
                    continue
                flattened.append(
                    {
                        **metadata,
                        "topk": int(topk),
                        "support_type": support_type,
                        **{field: metrics.get(field) for field in metric_fields},
                    }
                )
    return flattened


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: Iterable[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["layer"], row["phase"], row["topk"], row["support_type"])].append(row)

    metrics = (
        "captured_mass_mean",
        "captured_mass_min",
        "output_relative_l2_mean",
        "output_relative_l2_max",
        "output_cosine_mean",
        "actual_output_relative_l2_mean",
        "actual_output_relative_l2_max",
        "actual_output_cosine_mean",
        "sum_oracle_recall",
        "max_oracle_recall",
        "head_support_union_size",
        "support_fraction_first_16",
        "support_fraction_last_128",
        "support_fraction_last_1024",
        "support_fraction_last_4096",
        "support_fraction_last_16384",
    )
    summaries = []
    for (layer, phase, topk, support_type), group in sorted(grouped.items()):
        summary = {
            "layer": layer,
            "phase": phase,
            "topk": topk,
            "support_type": support_type,
            "count": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            summary[f"{metric}_mean"] = mean(values) if values else None
            summary[f"{metric}_p50"] = _percentile(values, 0.50) if values else None
            summary[f"{metric}_p90"] = _percentile(values, 0.90) if values else None
            summary[f"{metric}_p99"] = _percentile(values, 0.99) if values else None
        summaries.append(summary)
    return summaries


def _flatten_distribution_width(records: Iterable[dict]) -> List[dict]:
    rows = []
    for record in records:
        widths = record.get("distribution_width", {})
        per_head_metrics = {
            name.removesuffix("_per_head"): values
            for name, values in widths.items()
            if name.endswith("_per_head")
        }
        head_count = max((len(values) for values in per_head_metrics.values()), default=0)
        for head in range(head_count):
            row = {
                "dp_rank": record.get("dp_rank"),
                "request_id": record.get("request_id"),
                "layer": record.get("layer"),
                "phase": record.get("phase"),
                "offset": record.get("offset"),
                "query_position": record.get("query_position"),
                "context_length": record.get("context_length"),
                "max_measured_topk": widths.get("max_measured_topk"),
                "head": head,
            }
            for name, values in per_head_metrics.items():
                row[name] = values[head] if head < len(values) else None
            rows.append(row)
    return rows


def _summarize_distribution_width(rows: Iterable[dict]) -> List[dict]:
    grouped = defaultdict(lambda: {"values": [], "unresolved_caps": [], "total_count": 0})
    metadata = {
        "dp_rank",
        "request_id",
        "layer",
        "phase",
        "offset",
        "query_position",
        "context_length",
        "max_measured_topk",
        "head",
    }
    for row in rows:
        for metric, value in row.items():
            if metric in metadata or value is None:
                continue
            group = grouped[(row["layer"], row["phase"], metric)]
            group["total_count"] += 1
            if metric.startswith("k") and value < 0:
                measurement_cap = row.get("max_measured_topk")
                if measurement_cap is not None:
                    group["unresolved_caps"].append(float(measurement_cap))
                continue
            group["values"].append(float(value))

    summary = []
    for (layer, phase, metric), group in sorted(grouped.items()):
        values = group["values"]
        unresolved_caps = group["unresolved_caps"]
        total_count = group["total_count"]
        resolved_count = len(values)
        unresolved_count = total_count - resolved_count
        summary.append(
            {
                "layer": layer,
                "phase": phase,
                "metric": metric,
                "count": total_count,
                "resolved_count": resolved_count,
                "unresolved_count": unresolved_count,
                "unresolved_fraction": unresolved_count / total_count,
                "mean": mean(values) if values else None,
                "p50": _percentile(values, 0.50) if values else None,
                "p90": _percentile(values, 0.90) if values else None,
                "p99": _percentile(values, 0.99) if values else None,
                "max": max(values) if values else None,
                "unresolved_measurement_cap_min": (
                    min(unresolved_caps) if unresolved_caps else None
                ),
                "unresolved_measurement_cap_mean": (
                    mean(unresolved_caps) if unresolved_caps else None
                ),
                "unresolved_measurement_cap_max": (
                    max(unresolved_caps) if unresolved_caps else None
                ),
            }
        )
    return summary


def _flatten_aggregate_distribution_width(records: Iterable[dict]) -> List[dict]:
    rows = []
    for record in records:
        for teacher, metrics in record.get("aggregate_distribution_width", {}).items():
            rows.append(
                {
                    "dp_rank": record.get("dp_rank"),
                    "request_id": record.get("request_id"),
                    "layer": record.get("layer"),
                    "phase": record.get("phase"),
                    "offset": record.get("offset"),
                    "query_position": record.get("query_position"),
                    "context_length": record.get("context_length"),
                    "teacher": teacher,
                    **metrics,
                }
            )
    return rows


def _gap_rows(rows: Iterable[dict]) -> List[dict]:
    grouped = defaultdict(dict)
    identity_fields = (
        "dp_rank",
        "request_id",
        "layer",
        "phase",
        "query_position",
        "topk",
    )
    for row in rows:
        if row["support_type"] == "model_support":
            continue
        identity = tuple(row[field] for field in identity_fields)
        grouped[identity][row["support_type"]] = row

    gaps = []
    for identity, supports in grouped.items():
        if not {"indexer", "sum_head_oracle", "max_head_oracle", "per_head_oracle"}.issubset(
            supports
        ):
            continue
        base = dict(zip(identity_fields, identity))
        for metric in ("captured_mass_mean", "captured_mass_min", "output_relative_l2_mean"):
            indexer = supports["indexer"].get(metric)
            sum_oracle = supports["sum_head_oracle"].get(metric)
            max_oracle = supports["max_head_oracle"].get(metric)
            per_head = supports["per_head_oracle"].get(metric)
            if None in (indexer, sum_oracle, max_oracle, per_head):
                continue
            if metric.startswith("output_"):
                router_improvement = indexer - sum_oracle
                aggregation_improvement = sum_oracle - max_oracle
                shared_support_improvement = max_oracle - per_head
            else:
                router_improvement = sum_oracle - indexer
                aggregation_improvement = max_oracle - sum_oracle
                shared_support_improvement = per_head - max_oracle
            gaps.append(
                {
                    **base,
                    "metric": metric,
                    "router_improvement": router_improvement,
                    "aggregation_improvement": aggregation_improvement,
                    "shared_support_improvement": shared_support_improvement,
                }
            )
    return gaps


def _summarize_gaps(rows: Iterable[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["layer"], row["phase"], row["topk"], row["metric"])].append(row)
    summary = []
    for (layer, phase, topk, metric), group in sorted(grouped.items()):
        result = {
            "layer": layer,
            "phase": phase,
            "topk": topk,
            "metric": metric,
            "count": len(group),
        }
        for gap in (
            "router_improvement",
            "aggregation_improvement",
            "shared_support_improvement",
        ):
            values = [float(row[gap]) for row in group]
            result[f"{gap}_mean"] = mean(values)
            result[f"{gap}_p90"] = _percentile(values, 0.90)
            result[f"{gap}_p99"] = _percentile(values, 0.99)
        summary.append(result)
    return summary


def _plot_summary(summary: List[dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    for metric, filename, ylabel in (
        ("captured_mass_mean_mean", "captured_mass_vs_topk.png", "Captured mass"),
        ("output_relative_l2_mean_mean", "output_error_vs_topk.png", "Relative L2 error"),
    ):
        figure, axis = plt.subplots(figsize=(9, 5))
        plotted = False
        groups = defaultdict(list)
        for row in summary:
            if row["support_type"] == "model_support" or row.get(metric) is None:
                continue
            groups[(row["layer"], row["phase"], row["support_type"])].append(row)
        for (layer, phase, support), rows in sorted(groups.items()):
            rows.sort(key=lambda row: row["topk"])
            axis.plot(
                [row["topk"] for row in rows],
                [row[metric] for row in rows],
                marker="o",
                label=f"L{layer} {phase} {support}",
            )
            plotted = True
        if plotted:
            axis.set_xscale("log", base=2)
            axis.set_xlabel("Top-K")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize=7)
            figure.tight_layout()
            figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = _expand_globs(args.input_glob)
    if not paths:
        parser.error("No diagnostic shards matched --input-glob.")
    records = _read_records(paths)
    rows = _flatten_support_records(records)
    width_rows = _flatten_distribution_width(records)
    width_summary = _summarize_distribution_width(width_rows)
    aggregate_width_rows = _flatten_aggregate_distribution_width(records)
    summary = _summarize(rows)
    gaps = _gap_rows(rows)
    gaps_summary = _summarize_gaps(gaps)
    worst_requests = sorted(
        (
            row
            for row in rows
            if row["support_type"] == "model_support"
            and row.get("actual_output_relative_l2_max") is not None
        ),
        key=lambda row: float(row["actual_output_relative_l2_max"]),
        reverse=True,
    )[:100]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "per_request.csv", rows)
    _write_csv(args.output_dir / "distribution_width.csv", width_rows)
    _write_csv(args.output_dir / "distribution_width_summary.csv", width_summary)
    _write_csv(
        args.output_dir / "aggregate_distribution_width.csv", aggregate_width_rows
    )
    _write_csv(args.output_dir / "summary.csv", summary)
    _write_csv(args.output_dir / "gaps.csv", gaps)
    _write_csv(args.output_dir / "gaps_summary.csv", gaps_summary)
    _write_csv(args.output_dir / "worst_requests.csv", worst_requests)
    _plot_summary(summary, args.output_dir)
    print(
        f"Read {len(records)} query records from {len(paths)} shards; "
        f"wrote summaries to {args.output_dir}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
