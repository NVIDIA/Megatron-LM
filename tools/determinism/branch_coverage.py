# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Python source branches associated with numerically verified replay cases.

Coverage contexts describe execution, not numerical correctness of each branch.
Compiled/device branches are outside this view. Only public coverage.py APIs are
used, including when sharing the collector that produces ordinary CI coverage.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path


class BranchRecorder:
    """Measure an explicit source scope, including files never executed."""

    def __init__(self, root: Path, sources: list[str], output: Path):
        import coverage

        self.root = root.resolve()
        self.output = output
        paths: set[Path] = set()
        for source in sources:
            path = (self.root / source).resolve()
            if not path.is_relative_to(self.root) or not path.exists():
                raise ValueError(f"Branch source must exist inside the repository: {source}")
            paths.update(path.rglob("*.py") if path.is_dir() else [path])
        if not paths or any(
            p.suffix != ".py" or not p.resolve().is_relative_to(self.root) for p in paths
        ):
            raise ValueError("Branch scope must contain repository Python source files")
        self.files = {str(p.resolve()): p.relative_to(self.root).as_posix() for p in sorted(paths)}
        self.hashes = self._hashes()
        self.contexts: dict[str, str] = {}
        collector = coverage.Coverage.current()
        self.owned = collector is None
        if collector is None:
            collector = coverage.Coverage(
                config_file=False,
                branch=True,
                include=list(self.files),
                data_file=str(output.with_suffix(".coverage")),
            )
        self.collector = collector
        if not self.collector.get_option("run:branch"):
            raise ValueError(
                "Determinism branches require coverage run --branch (or run.branch=true)"
            )
        if self.collector.get_option("run:dynamic_context"):
            raise ValueError(
                "Determinism branch contexts cannot share another dynamic_context policy"
            )
        self.static_context = self.collector.get_option("run:context")
        self.version = coverage.__version__
        if self.owned:
            self.collector.start()

    def _hashes(self):
        return {
            relative: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path, relative in self.files.items()
        }

    @contextlib.contextmanager
    def case(self, case_id: str):
        """Exclude fixture setup, teardown, and unmarked calls from case contexts."""
        context = "determinism:" + case_id
        self.contexts[case_id] = (
            self.static_context + "|" + context if self.static_context else context
        )
        self.collector.switch_context(context)
        try:
            yield
        finally:
            # Do not depend on switch_context's return value (added in 7.16).
            self.collector.switch_context("")

    def finish(self) -> tuple[dict, dict]:
        """Return the branch catalog and case arcs without stopping an outer collector."""
        if self.owned:
            self.collector.stop()
        data = self.collector.get_data()
        data.set_query_contexts(None)
        # Empty arc sets retain unexecuted files without inventing execution.
        data.add_arcs({path: [] for path in self.files})
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.collector.save()
        self.collector.json_report(morfs=list(self.files), outfile=str(self.output), contexts=None)
        raw = json.loads(self.output.read_text())
        catalog = {}
        for filename, details in raw["files"].items():
            relative = self.files[str(Path(filename).resolve())]
            branches = sorted(
                {tuple(arc) for arc in details["executed_branches"] + details["missing_branches"]}
            )
            if len(branches) != details["summary"]["num_branches"]:
                raise ValueError(f"Inconsistent branch denominator for {relative}")
            catalog[relative] = {
                "sha256": self.hashes[relative],
                "arcs": [list(arc) for arc in branches],
            }
        if set(catalog) != set(self.files.values()) or self._hashes() != self.hashes:
            raise ValueError("Branch source scope is missing or changed during the run")
        observations = {}
        try:
            for case_id, context in self.contexts.items():
                data.set_query_context(context)
                arcs = {}
                for path, relative in self.files.items():
                    possible = {tuple(arc) for arc in catalog[relative]["arcs"]}
                    executed = sorted(possible.intersection(data.arcs(path) or ()))
                    if executed:
                        arcs[relative] = [list(arc) for arc in executed]
                observations[case_id] = arcs
        finally:
            data.set_query_contexts(None)
        return {"complete": True, "coverage_version": self.version, "files": catalog}, observations


def branch_report(shards: list[dict], cases: list[dict], provenance_valid: bool) -> dict | None:
    """Join source execution to cases that passed replay on every required rank."""
    if not any("branches" in shard for shard in shards):
        return None
    catalogs = [shard.get("branches", {}) for shard in shards]
    available = next((catalog for catalog in catalogs if catalog.get("files") is not None), None)
    reasons = sorted(
        {
            catalog.get("reason", "Branch collection incomplete")
            for catalog in catalogs
            if not catalog.get("complete")
        }
    )
    world_size = shards[0]["context"]["world_size"]
    complete = (
        provenance_valid
        and len(shards) == world_size
        and all(
            shard["complete"] and catalog.get("complete")
            for shard, catalog in zip(shards, catalogs)
        )
    )
    if not provenance_valid:
        reasons.append("Source revision is stale or has uncommitted changes")
    if len(shards) != world_size or not all(shard["complete"] for shard in shards):
        reasons.append("Missing rank or incomplete session")
    if available is None:
        return {
            "complete": False,
            "reasons": reasons,
            "counts": None,
            "passing_replay_percent": None,
            "files": {},
        }
    for catalog in catalogs:
        if catalog.get("complete") and (
            catalog.get("files") != available["files"]
            or catalog.get("coverage_version") != available["coverage_version"]
        ):
            raise ValueError("Branch catalogs or coverage.py versions differ across ranks")
    case_statuses = {case["case_id"]: case["status"] for case in cases}
    observed: dict[tuple, dict[str, list[int]]] = {}
    for shard in shards:
        for case_id, case in shard["cases"].items():
            for filename, arcs in case.get("branch_arcs", {}).items():
                possible = available["files"].get(filename, {}).get("arcs", [])
                for arc in arcs:
                    if arc not in possible:
                        raise ValueError("Observed branch is outside the declared catalog")
                    observed.setdefault((filename, *arc), {}).setdefault(case_id, []).append(
                        shard["rank"]
                    )
    files = {}
    counts = {"total": 0, "observed": 0, "passing_replay": 0}
    for filename, catalog in available["files"].items():
        arcs = []
        for source, destination in catalog["arcs"]:
            support = observed.get((filename, source, destination), {})
            passing = complete and any(
                case_statuses[case_id] == "verified_deterministic" for case_id in support
            )
            counts["total"] += 1
            counts["observed"] += bool(support)
            counts["passing_replay"] += bool(passing)
            arcs.append(
                {
                    "arc": [source, destination],
                    "passing_replay": bool(passing),
                    "observed_by": [
                        {
                            "case_id": case_id,
                            "ranks": sorted(ranks),
                            "replay_status": case_statuses[case_id],
                        }
                        for case_id, ranks in sorted(support.items())
                    ],
                }
            )
        files[filename] = {"sha256": catalog["sha256"], "arcs": arcs}
    counts["uncovered_by_passing_replay"] = counts["total"] - counts["passing_replay"]
    counts["never_observed"] = counts["total"] - counts["observed"]
    return {
        "scope": (
            "Python source branches during declared test calls; "
            "excludes fixtures and compiled/device branches"
        ),
        "complete": bool(complete),
        "reasons": reasons,
        "coverage_version": available["coverage_version"],
        "counts": counts,
        "passing_replay_percent": (
            100 * counts["passing_replay"] / counts["total"]
            if complete and counts["total"]
            else None
        ),
        "files": files,
    }
