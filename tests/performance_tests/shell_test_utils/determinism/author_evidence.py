# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Join scoped kernel author checks to matched, uninstrumented phase timings.

Consumes the coverage producer's aggregate report and this driver's benchmark
JSON. A complete evidence bundle is distinct from passing configured budgets.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from pathlib import Path

import benchmark
from kernel_case import ADAPTER, OP_IDS, POLICY_ENV, SEED
from run_kernel import validate_result

VERIFIED = "verified_deterministic"
UNKNOWN = "not_verified"


def _digest(value, lengths=(64,)) -> bool:
    return (
        isinstance(value, str)
        and len(value) in lengths
        and all(character in "0123456789abcdef" for character in value)
    )


def _contract(case: dict, ranks: list[int]) -> dict:
    observations = case.get("observations", [])
    if not observations or {row["rank"] for row in observations} != set(ranks):
        raise ValueError("Missing replay observations or required ranks")
    contracts = []
    for row in observations:
        signature = row["signature"]
        contract = signature.get("configuration", {}).get("kernel_case")
        if not isinstance(contract, dict) or contract.get("adapter") != ADAPTER:
            raise ValueError("Replay has no supported local-activation adapter")
        name = contract.get("case")
        if (
            not isinstance(name, str)
            or name not in OP_IDS
            or contract.get("distribution") != "local_without_collectives"
            or contract.get("op_id") != OP_IDS[name]
            or case["op_id"] != OP_IDS[name]
            or signature.get("op_id") != OP_IDS[name]
            or contract.get("implementation") != "torch.compile:" + name
            or signature.get("implementation") != contract["implementation"]
            or signature.get("phase") != "forward_backward"
            or signature.get("deterministic_algorithms") is not True
            or contract.get("upstream_gradient") != "ones_like_output"
            or contract.get("input_seed") != SEED
        ):
            raise ValueError("Replay adapter, operation, backward phase or input protocol differs")
        arguments = contract.get("inputs", [])
        if not arguments or any(
            value is not None
            and (not _digest(value.get("sha256")) or not value.get("requires_grad"))
            for value in arguments
        ):
            raise ValueError("Replay input fingerprints or gradient requirements are missing")
        metadata = [
            (
                {key: value[key] for key in ("shape", "stride", "dtype", "requires_grad")}
                if value is not None
                else None
            )
            for value in arguments
        ]
        if signature.get("inputs") != metadata:
            raise ValueError("Adapter inputs differ from the actual replay dispatch")
        runtime = contract["runtime"]
        required_runtime = {
            "python",
            "packages",
            "torch",
            "cuda",
            "gpu",
            "capability",
            "driver",
            "deterministic_algorithms",
            "warn_only",
            "fill_uninitialized_memory",
            "autocast",
            "autocast_dtype",
            "float32_matmul_precision",
            "matmul_allow_tf32",
            "cudnn_allow_tf32",
            "cudnn_deterministic",
            "cudnn_benchmark",
            "environment",
        }
        if (
            not required_runtime <= runtime.keys()
            or not (
                set(POLICY_ENV) | {"CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_PROTO", "TRITON_CACHE_DIR"}
            )
            <= runtime.get("environment", {}).keys()
            or runtime.get("deterministic_algorithms") is not True
            or runtime.get("warn_only") is not False
            or not runtime.get("gpu")
            or not runtime.get("cuda")
            or not runtime.get("driver")
            or not runtime.get("capability")
            or not runtime.get("packages", {}).get("torch")
            or not _digest(contract.get("adapter_sha256"))
        ):
            raise ValueError("Strict policy and GPU/software/adapter provenance are required")
        contracts.append(contract)
    if any(contract != contracts[0] for contract in contracts):
        raise ValueError("Replay ranks or calls used different local case contracts")
    return contracts[0]


def _without_mode(signature: dict) -> dict:
    common = copy.deepcopy(signature)
    runtime = common["runtime"]
    runtime.pop("deterministic_algorithms")
    for key in POLICY_ENV:
        runtime["environment"].pop(key, None)
    return common


def _limit(report: dict, name: str) -> float | None:
    value = report["comparisons"][name]["limit"]
    if value is not None and (
        type(value) not in (int, float) or not math.isfinite(value) or value <= 0
    ):
        raise ValueError("Invalid configured performance limit")
    return value


def replay_contention_supported(protocol: dict, context: dict) -> bool:
    """Accept serialized replay only when the recorded launch policy requires it."""
    return protocol.get("contention") is True or (
        protocol.get("contention") is False
        and protocol.get("contention_requested") is True
        and context.get("environment", {}).get("CUDA_DEVICE_MAX_CONNECTIONS") == "1"
    )


def _numerically_verified(case: dict, context: dict) -> bool:
    """Reject summaries contradicted by missing or failing raw observations/checks."""
    if case["status"] != VERIFIED or any(
        case.get("check_status", {}).get(kind) != "passed" for kind in ("reference", "sensitivity")
    ):
        return False
    observations = case.get("observations", [])
    if any(
        row.get("kind") not in ("reference", "sensitivity")
        or not any(
            row.get("rank") == observation["rank"]
            and row.get("signature") == observation["signature"]
            for observation in observations
        )
        for row in case.get("checks", [])
    ):
        return False
    for observation in observations:
        if (
            observation.get("status") != VERIFIED
            or observation.get("protocol", {}).get("replays", 0) < 2
            or observation.get("protocol", {}).get("warn_only") is not False
            or not replay_contention_supported(observation.get("protocol", {}), context)
            or any(
                type(observation.get(key)) is not int or observation[key] < 1
                for key in ("compared_outputs", "compared_gradients")
            )
        ):
            return False
        for kind in ("reference", "sensitivity"):
            matching = [
                row
                for row in case.get("checks", [])
                if row.get("kind") == kind
                and row.get("rank") == observation["rank"]
                and row.get("signature") == observation["signature"]
            ]
            required = sum(
                other["rank"] == observation["rank"]
                and other["signature"] == observation["signature"]
                for other in observations
            )
            if len(matching) < required or any(
                row.get("status") != "passed"
                or any(
                    type(row.get(key)) is not int or row[key] != observation[key]
                    for key in ("compared_outputs", "compared_gradients")
                )
                or (
                    kind == "sensitivity"
                    and row.get("detected_perturbations")
                    != observation["compared_outputs"] + observation["compared_gradients"]
                )
                for row in matching
            ):
                return False
    return bool(observations)


def _timing(report: dict, contract: dict, revision: str, phase: str) -> dict:
    if report.get("schema_version") != 1 or report.get("kind") != "determinism_kernel_performance":
        raise ValueError("Only versioned operator timing reports can supply phase evidence")
    measurement = report["measurement"]
    if "diagnostic_only" in measurement:
        raise ValueError(
            "Diagnostic timings cannot supply author performance evidence or baselines"
        )
    arguments = contract["inputs"]
    expected_width = arguments[0]["shape"][-1]
    if contract["case"] != "weighted_squared_relu":
        expected_width //= 2
    if (
        measurement.get("phase") != phase
        or measurement.get("kernel_case") != contract["case"]
        or measurement.get("dtype") != arguments[0]["dtype"].removeprefix("torch.")
        or measurement.get("tokens") != arguments[0]["shape"][0]
        or measurement.get("hidden_size") != expected_width
        or measurement.get("gpus") != 1
        or measurement.get("timing") != "cuda_event_ms"
        or any(
            type(measurement.get(key)) is not int or measurement[key] < minimum
            for key, minimum in (("pairs", 3), ("warmup", 1), ("steps", 1))
        )
    ):
        raise ValueError("Timing phase, dimensions, precision, warmup or pair count differs")
    sources = report["sources"]
    if (
        sources.get("head", {}).get("revision") != revision
        or set(sources) not in ({"head"}, {"head", "base"})
        or any(
            source.get("dirty") is not False or not _digest(source.get("revision"), (40, 64))
            for source in sources.values()
        )
        or not report.get("machine", {}).get("gpus")
    ):
        raise ValueError("Timing source is stale/dirty or GPU provenance is missing")
    values, devices = [], set()
    for run in report["runs"]:
        mode = run["mode"]
        if mode not in ("det", "default") or run.get("status") != "complete":
            raise ValueError("Unknown policy or incomplete timing arm")
        kernel = run["kernel"]
        validate_result(kernel, measurement, mode)
        actual = kernel.get("case_signature")
        expected_policy = benchmark.DET_ENV if mode == "det" else benchmark.DEFAULT_ENV
        if (
            not isinstance(actual, dict)
            or actual.get("runtime", {}).get("deterministic_algorithms") is not (mode == "det")
            or _without_mode(actual) != _without_mode(contract)
            or (mode == "det" and actual != contract)
            or any(
                actual["runtime"]["environment"].get(key) != expected_policy.get(key)
                for key in POLICY_ENV
            )
        ):
            raise ValueError("Timing inputs, adapter, GPU, software or runtime settings differ")
        if not kernel.get("device_uuid") or kernel["device_uuid"] == "None":
            raise ValueError("Timing allocation UUID is missing")
        devices.add(kernel["device_uuid"])
        samples = kernel["samples_ms"]
        if run.get("samples_ms") != samples or run.get("median_ms") != statistics.median(samples):
            raise ValueError("Cached timing samples or median differ from raw measurements")
        values.append({**run, "median_ms": statistics.median(samples)})
    if len(devices) != 1:
        raise ValueError("Paired timing arms ran on different GPU allocations")
    has_base = "base" in sources
    overhead = _limit(report, "head_overhead")
    regression = _limit(report, "det_regression") if has_base else None
    if has_base and _limit(report, "default_regression") != regression:
        raise ValueError("Revision modes use inconsistent limits")
    comparisons = benchmark.summarize(values, measurement["pairs"], has_base, overhead, regression)
    if report.get("comparisons") != comparisons:
        raise ValueError("Cached comparisons differ from recomputed paired results")
    gated = [
        row["status"]
        for name, row in comparisons.items()
        if name != "base_overhead" and row["status"] != "not_gated"
    ]
    status = (
        "fail"
        if "fail" in gated
        else "inconclusive" if "inconclusive" in gated else "pass" if gated else "reported"
    )
    if report.get("status") != status:
        raise ValueError("Cached report status differs from recomputed performance gates")
    performance_status = status
    if status in ("pass", "reported") and (not has_base or overhead is None or regression is None):
        performance_status = "not_gated"
    return {
        "evidence_status": "complete",
        "performance_status": performance_status,
        "measurement_status": status,
        "comparisons": comparisons,
        "base_revision": sources.get("base", {}).get("revision"),
        "device_uuid": next(iter(devices)),
    }


def join(coverage: dict, reports: list[dict], revision: str) -> dict:
    """Keep every manifest requirement, including absent or mismatched evidence."""
    if not _digest(revision, (40, 64)):
        raise ValueError("A complete target git revision is required")
    if coverage.get("schema_version") != 1 or coverage.get("kind") != "determinism_coverage":
        raise ValueError("Expected a versioned operator coverage report, not a model report")
    context = coverage["context"]
    world_size = context.get("world_size")
    fresh = (
        context.get("revision") == revision
        and context.get("dirty") is False
        and type(world_size) is int
        and world_size > 0
        and coverage.get("ranks_present") == list(range(world_size))
    )
    rows = []
    for requirement in coverage.get("author_requirements", []):
        row = {
            "test": requirement["test"],
            "op_id": requirement["op_id"],
            "evidence_status": UNKNOWN,
            "performance_status": UNKNOWN,
            "phases": {},
        }
        selected = [
            case
            for case in coverage["cases"]
            if case["case_id"] == requirement["test"] and case["op_id"] == requirement["op_id"]
        ]
        rows.append(row)
        if not fresh or len(selected) != 1:
            row["reason"] = (
                "Coverage is stale/dirty/incomplete or the required case is missing/ambiguous"
            )
            continue
        case = selected[0]
        row.update(replay=case["status"], checks=case.get("check_status", {}))
        try:
            contract = _contract(case, coverage["ranks_present"])
            row["case_signature"] = contract
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
            row["reason"] = str(error)
            continue
        for phase in ("forward", "backward"):
            candidates = [
                report
                for report in reports
                if report.get("measurement", {}).get("kernel_case") == contract["case"]
                and report["measurement"].get("phase") == phase
                and report["measurement"].get("dtype")
                == contract["inputs"][0]["dtype"].removeprefix("torch.")
            ]
            try:
                if len(candidates) != 1:
                    raise ValueError(
                        "Missing or ambiguous phase report; do not select the best retry"
                    )
                row["phases"][phase] = _timing(candidates[0], contract, revision, phase)
            except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
                row["phases"][phase] = {"evidence_status": UNKNOWN, "reason": str(error)}
        numeric = _numerically_verified(case, context) and requirement.get("status") == "passed"
        if numeric and all(
            value["evidence_status"] == "complete" for value in row["phases"].values()
        ):
            if len({value["base_revision"] for value in row["phases"].values()}) != 1:
                row["reason"] = "Forward/backward timing reports use different base revisions"
                continue
            row["evidence_status"] = "complete"
            statuses = [value["performance_status"] for value in row["phases"].values()]
            row["performance_status"] = (
                "fail"
                if "fail" in statuses
                else (
                    "inconclusive"
                    if "inconclusive" in statuses
                    else "pass" if all(value == "pass" for value in statuses) else "not_gated"
                )
            )
        elif (
            case["status"] == "verified_nondeterministic"
            or "failed" in case.get("check_status", {}).values()
        ):
            row["evidence_status"] = "failed"
        else:
            row["reason"] = (
                "Replay, reference or sensitivity evidence is not verified"
                if not numeric
                else "Forward/backward timing evidence is missing or incompatible"
            )
    complete = bool(rows) and all(row["evidence_status"] == "complete" for row in rows)
    failed = any(
        row["evidence_status"] == "failed" or row["performance_status"] == "fail" for row in rows
    )
    status = (
        "failed"
        if failed
        else (
            "passed"
            if complete and all(row["performance_status"] == "pass" for row in rows)
            else (
                "not_gated"
                if complete
                and all(row["performance_status"] in ("pass", "not_gated") for row in rows)
                else UNKNOWN
            )
        )
    )
    return {
        "schema_version": 1,
        "kind": "kernel_author_evidence",
        "revision": revision,
        "coverage_run_id": coverage.get("run_id"),
        "evidence_complete": complete,
        "status": status,
        "requirements": rows,
        "scope": "Selected local activation cases only; configured limits are not proof of baseline calibration or full-model behavior",
    }


def markdown_report(report: dict) -> str:
    """Show missing evidence and budget state without merging their meanings."""
    lines = [
        "# Kernel author evidence",
        "",
        f"Status: **{report['status']}**.",
        "",
        "| Required case | Evidence | Performance |",
        "| --- | --- | --- |",
    ]
    diagnostics = []
    for row in report.get("requirements", []):
        name = row["test"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{name}` | {row['evidence_status']} | {row['performance_status']} |")
        for phase, value in row["phases"].items():
            if value.get("reason"):
                diagnostics.append(f"- `{name}`, {phase}: {value['reason']}")
        if row.get("reason"):
            diagnostics.append(f"- `{name}`: {row['reason']}")
    if diagnostics:
        lines.extend(["", *diagnostics])
    if report.get("error"):
        lines.append(f"\n{report['error']}")
    lines.append(
        "\nConfigured performance limits need independently reviewed hardware calibration."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write diagnostics before failing missing evidence or a requested budget gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--performance", type=Path, action="append", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-performance-pass", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.with_suffix(".md").exists():
        parser.error("Use a new report path; preserve previous attempts")
    provenance = []
    try:

        def reject_constant(value):
            raise ValueError(f"Nonfinite JSON number: {value}")

        def read(path):
            raw = path.read_bytes()
            provenance.append(
                {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
            )
            return json.loads(raw, parse_constant=reject_constant)

        coverage = read(args.coverage)
        reports = []
        for path in args.performance:
            data = read(path)
            reports.extend(data if isinstance(data, list) else [data])
        report = join(coverage, reports, args.revision)
    except (OSError, AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
        report = {
            "schema_version": 1,
            "kind": "kernel_author_evidence",
            "status": UNKNOWN,
            "evidence_complete": False,
            "revision": args.revision,
            "error": str(error),
        }
    report["inputs"] = provenance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".md").write_text(markdown_report(report))
    print(markdown_report(report))
    if report["status"] == "failed":
        return 1
    if (
        report["status"] == UNKNOWN
        or not report["evidence_complete"]
        or (args.require_performance_pass and report["status"] != "passed")
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
