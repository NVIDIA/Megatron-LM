#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Summarize opt-in bugfix snapshots without importing Megatron or GPU packages."""

import argparse
import json
from pathlib import Path


def summarize(directory):
    """Read cumulative snapshots once per process; report observed activity only."""
    processes = {}
    skipped_files = 0
    for path in sorted(Path(directory).glob("bugfix-*.json")):
        try:
            with path.open(encoding="utf-8") as stream:
                record = json.load(stream)
            if (
                record["schema_version"] != 1
                or not isinstance(record["process_id"], str)
                or not record["process_id"]
                or type(record["sequence"]) is not int
                or record["sequence"] < 1
                or not isinstance(record["counters"], dict)
                or any(
                    not isinstance(name, str) or type(count) is not int or count < 0
                    for name, count in record["counters"].items()
                )
                or any(
                    record.get(key) is not None and not isinstance(record[key], str)
                    for key in ("cluster", "job_id", "step_id", "username", "rank", "world_size")
                )
            ):
                raise ValueError("Invalid snapshot")
            previous = processes.get(record["process_id"])
            if previous is None or record["sequence"] > previous["sequence"]:
                processes[record["process_id"]] = record
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            skipped_files += 1

    fixes = {}
    for record in processes.values():
        for name, count in record["counters"].items():
            if not count:
                continue
            totals = fixes.setdefault(name, {"process_hits": 0, "jobs": set(), "users": set()})
            totals["process_hits"] += count
            if record.get("job_id"):
                totals["jobs"].add((record.get("cluster"), record["job_id"]))
            if record.get("username"):
                totals["users"].add(record["username"])

    return {
        "fixes": [
            {
                "name": name,
                "process_hits": totals["process_hits"],
                "affected_jobs": len(totals["jobs"]),
                "affected_users": len(totals["users"]),
            }
            for name, totals in sorted(fixes.items())
        ],
        "processes": sorted(processes.values(), key=lambda record: record["process_id"]),
        "skipped_files": skipped_files,
    }


def main():
    """Print observed counts, optionally including individual process metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing bugfix-*.json snapshots")
    parser.add_argument("--json", action="store_true", help="Print machine-readable results")
    parser.add_argument("--details", action="store_true", help="Include process/rank metadata")
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error("Statistics directory is not accessible")
    result = summarize(args.directory)
    processes = result.pop("processes")
    result["observed_processes"] = len(processes)
    if args.details:
        result["processes"] = processes
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print("Bugfix | Process hits | Affected jobs | Affected users")
    for row in result["fixes"]:
        print(
            f"{row['name']} | {row['process_hits']} | "
            f"{row['affected_jobs']} | {row['affected_users']}"
        )
    print(
        f"Observed processes: {len(processes)}; unreadable/invalid files: {result['skipped_files']}"
    )
    print("Process hits include TP/PP replication. Missing files do not establish zero activity.")
    if args.details:
        print(json.dumps(processes, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
