# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Review pinned collective baselines without pooling ranks, groups or allocations."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import statistics
from pathlib import Path

import baseline
import collective_baseline


def _identity(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _read(directory: Path, manifest: dict, name: str) -> dict:
    raw = baseline._bytes(directory / name)
    if manifest["files"][name] != {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}:
        raise ValueError("Collective metadata changed after verification: " + name)
    return baseline._json(raw)


def _cache_locations(value: object, normalize: bool = False) -> set[str | None]:
    """Visit only recorded environment keys; keep unset and explicit paths distinct."""
    locations: set[str | None] = set()
    if isinstance(value, dict):
        if "TRITON_CACHE_DIR" in value:
            path = value["TRITON_CACHE_DIR"]
            if path is not None and (not isinstance(path, str) or not path):
                raise ValueError("Explicit cache locations must be nonempty strings")
            locations.add(path)
            if normalize and path is not None:
                value["TRITON_CACHE_DIR"] = "<explicit cache location>"
        for child in value.values():
            locations.update(_cache_locations(child, normalize))
    elif isinstance(value, list):
        for child in value:
            locations.update(_cache_locations(child, normalize))
    return locations


def _workload(captures: list[dict], compare_cache_locations: bool) -> list[dict]:
    """Preserve the entire rank schedule; omit only physical device identities."""
    result = copy.deepcopy(captures)
    for capture in result:
        for event in capture["events"]:
            del event["signature"]["configuration"]["collective"]["device_uuid"]
    _cache_locations(result, compare_cache_locations)
    return result


def _machine(machine: dict, captures: list[dict]) -> dict:
    """Preserve recorded hardware and rank placement, excluding UUIDs and host names."""
    context = captures[0]["context"]
    if any(
        not context.get(key)
        for key in ("python", "versions", "cuda", "driver", "gpu", "capability", "environment")
    ):
        raise ValueError("Collective calibration requires complete recorded runtime metadata")
    hardware = machine.get("gpus")
    if not isinstance(hardware, list) or not hardware:
        raise ValueError("Collective calibration requires a recorded GPU inventory")
    normalized = copy.deepcopy(machine)
    normalized.pop("host", None)
    normalized.pop("cuda_visible_devices", None)
    inventory, indices, identifiers = [], set(), {}
    for line in hardware:
        if not isinstance(line, str):
            raise ValueError("Unsupported GPU inventory row")
        try:
            fields = next(csv.reader([line], skipinitialspace=True, strict=True))
        except (csv.Error, StopIteration) as error:
            raise ValueError("Unsupported GPU inventory row") from error
        if len(fields) != 4 or any(not field for field in fields) or not fields[0].isdigit():
            raise ValueError("Unsupported GPU inventory row")
        index, identifier, name, driver = fields
        identifier = identifier.removeprefix("GPU-").lower()
        if index in indices or identifier in identifiers:
            raise ValueError("Duplicate GPU inventory entry")
        indices.add(index)
        identifiers[identifier] = index
        inventory.append({"index": index, "name": name, "driver": driver})
    placement = []
    for capture in captures:
        devices = {
            event["signature"]["configuration"]["collective"]["device_uuid"]
            for event in capture["events"]
        }
        if len(devices) != 1:
            raise ValueError("A captured rank changes physical devices within its schedule")
        identifier = next(iter(devices)).removeprefix("GPU-").lower()
        if identifier not in identifiers:
            raise ValueError("Captured rank device is absent from the hardware inventory")
        placement.append(identifiers[identifier])
    if len(set(placement)) != len(captures):
        raise ValueError("Collective calibration requires distinct physical devices per rank")
    normalized.update(gpus=inventory, rank_device_indices=placement)
    return normalized


def compare(inputs: list[tuple[Path, str]], *, compare_cache_locations: bool = False) -> dict:
    """Reverify complete bundles and summarize one estimate per event/group per run.

    All ranks and events remain part of the workload identity. A different call
    schedule, input byte hash, group, source, runtime or protocol forms a separate
    cohort. Allocation identities remain visible without implying independence.
    """
    if not inputs:
        raise ValueError("At least one pinned collective baseline is required")
    groups: dict[str, dict] = {}
    workloads: dict[str, list[dict]] = {}
    receipts: list[dict] = []
    identities: set[str] = set()
    previous_arms: set[str] = set()
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
        receipt = collective_baseline.verify(directory, expected_id)
        raw = baseline._bytes(directory / baseline.MANIFEST)
        if hashlib.sha256(raw).hexdigest() != expected_id:
            raise ValueError("Baseline manifest changed after verification")
        manifest = baseline._json(raw)
        report = _read(directory, manifest, "timing/benchmark.json")
        if any(
            estimate["limit"] is not None
            for comparison in report["comparisons"].values()
            for estimate in comparison.values()
        ):
            raise ValueError(
                "Calibration requires unbudgeted measurements, not selected gate passes"
            )
        captures = [
            _read(directory, manifest, f"capture/rank-{rank}/manifest.json")
            for rank in range(report["measurement"]["world_size"])
        ]
        workload = _workload(captures, compare_cache_locations)
        workload_id = _identity(workload)
        workloads[workload_id] = workload
        machine = _machine(report["machine"], captures)
        cache_locations = sorted(
            _cache_locations(captures), key=lambda value: (value is not None, value or "")
        )
        protocol = {
            key: value for key, value in report["measurement"].items() if key != "manifest_sha256"
        }
        sources = {
            label: {key: value for key, value in source.items() if key != "checkout"}
            for label, source in report["sources"].items()
        }
        arm_ids = []
        for arm in report["runs"]:
            pair, label, mode = arm["pair"], arm["revision_label"], arm["mode"]
            payloads = []
            for rank in range(len(captures)):
                name = f"timing/pair-{pair}/{label}-{mode}/rank-{rank}.json"
                record = _read(directory, manifest, name)
                # Ignore run labels, transport paths and cache locations when
                # detecting reused measured traces. Preserve bytes/UUIDs/samples.
                rows = copy.deepcopy(record["rows"])
                _cache_locations(rows, True)
                payloads.append({"rank": rank, "rows": rows})
            identity = _identity({"source": sources[label], "mode": mode, "ranks": payloads})
            if identity in previous_arms:
                raise ValueError("Repeated raw timing arm under different baseline metadata")
            arm_ids.append(identity)
        previous_arms.update(arm_ids)
        measurement_id = _identity(sorted(arm_ids))
        receipts.append({**receipt, "measurement_id": measurement_id})
        for key, first in report["runs"][0]["rows"].items():
            index, members = first["event_index"], first["group_ranks"]
            context = {
                "workload_id": workload_id,
                "recipe_id": report["capture"]["recipe_id"],
                "event_index": index,
                "call_id": captures[0]["events"][index]["call_id"],
                "group_ranks": members,
                "case": first["case"],
                "phase": first["phase"],
                "dtypes": [
                    captures[rank]["events"][index]["signature"]["inputs"][0]["dtype"]
                    for rank in members
                ],
                "measurement": protocol,
                "sources": sources,
                "machine_runtime": machine,
            }
            cohort = _identity(context)
            group = groups.setdefault(cohort, {"cohort_id": cohort, "context": context, "runs": []})
            devices = [
                captures[rank]["events"][index]["signature"]["configuration"]["collective"][
                    "device_uuid"
                ]
                for rank in members
            ]
            group["runs"].append(
                {
                    "baseline_id": expected_id,
                    "origin": receipt["origin"],
                    "measurement_id": measurement_id,
                    "event_group": key,
                    "group_device_uuids": devices,
                    "machine": report["machine"],
                    "sources": report["sources"],
                    "capture_context": captures[0]["context"],
                    "capture_manifest_sha256": report["measurement"]["manifest_sha256"],
                    "source_status": receipt["status"],
                    "cache_locations": cache_locations,
                    "comparisons": report["comparisons"][key],
                    "arm_medians": [
                        {
                            **{field: arm[field] for field in ("pair", "revision_label", "mode")},
                            "median_ms": arm["rows"][key]["median_ms"],
                        }
                        for arm in report["runs"]
                    ],
                }
            )
    for group in groups.values():
        runs = sorted(group["runs"], key=lambda run: run["baseline_id"])
        if len({run["baseline_id"] for run in runs}) != len(runs):
            raise ValueError("Ambiguous event/group alignment within a baseline")
        group.update(
            runs=runs,
            run_count=len(runs),
            distinct_rank_device_assignments=len(
                {tuple(run["group_device_uuids"]) for run in runs}
            ),
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
        "kind": "determinism_collective_calibration",
        "status": "report_only",
        "performance_gate": "not_gated",
        "compare_cache_locations": compare_cache_locations,
        "baselines": sorted(receipts, key=lambda receipt: receipt["baseline_id"]),
        "workloads": {key: workloads[key] for key in sorted(workloads)},
        "groups": [groups[key] for key in sorted(groups)],
        "scope": (
            "Selected verified unbudgeted collective baselines. Exact complete captured rank "
            "schedules, input/gradient bytes, layouts, groups/options, source, runtime and timing "
            "protocols define cohorts. UUIDs and host names remain provenance; physical fabric "
            "equivalence and allocation independence are not established. Explicit cache locations "
            "may be grouped only when requested; neither equal nor grouped paths prove equal cache "
            "contents or dispatch. Per-run paired intervals are retained without pooling ranks, "
            "groups or allocations. Observed ranges are not population bounds, causal explanations, "
            "reviewed budgets, promotion approval or production-model performance acceptance."
        ),
    }


def markdown_report(report: dict) -> str:
    """Show every measured event/group, including singleton and incompatible cohorts."""

    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    lines = [
        "# Collective performance calibration",
        "",
        "Report only. Each baseline contributes one estimate per event/group; no samples are pooled.",
        "A singleton's zero observed range does not estimate run-to-run variation.",
        "Device-assignment counts do not prove independent allocations or equivalent physical fabrics.",
        f"Compare explicit cache locations: {report['compare_cache_locations']}.",
        "",
        "| Cohort | GPU | Head | Recipe | Case | Dtypes | Phase | Event | Ranks | Comparison | Runs | Assignments | Min ratio | Median ratio | Max ratio | Range (pp) |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in report["groups"]:
        context = group["context"]
        gpu = report["workloads"][context["workload_id"]][0]["context"]["gpu"]
        for name, values in group["comparisons"].items():
            lines.append(
                f"| `{group['cohort_id'][:12]}` | {cell(gpu)} | `{context['sources']['head']['revision'][:12]}` | "
                f"{cell(context['recipe_id'])} | {cell(context['case'])} | "
                f"{cell(','.join(sorted(set(context['dtypes']))))} | {cell(context['phase'])} | "
                f"{context['event_index']} | {cell(context['group_ranks'])} | {name} | {group['run_count']} | "
                f"{group['distinct_rank_device_assignments']} | {values['minimum_ratio']:.6f} | "
                f"{values['median_ratio']:.6f} | {values['maximum_ratio']:.6f} | "
                f"{values['observed_range_percentage_points']:.4f} |"
            )
    lines.extend(["", "## Input baselines", ""])
    for receipt in report["baselines"]:
        lines.append(
            f"- `{receipt['baseline_id']}`: {cell(receipt['origin'])}; revision `{receipt['revision']}`, "
            f"{receipt['files_verified']} verified files, {receipt['rows']} event/group rows, "
            f"{receipt['rank_samples']} rank samples, `{receipt['status']}`."
        )
    lines.extend(["", report["scope"], ""])
    return "\n".join(lines)
