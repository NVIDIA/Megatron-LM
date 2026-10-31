# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Measure functional-test overlap and propose bounded-loss subsets of a complete baseline."""

import argparse
import collections
import json
import math
from pathlib import Path


def load_baseline(directory: Path) -> dict:
    """Require both platform matrices and successful coverage for every expected test."""
    expected = set()
    for platform in ("h100", "gb200"):
        matrices = list(directory.rglob(f"integration-tests-{platform}.json"))
        if len(matrices) != 1:
            raise ValueError(f"Expected one {platform} matrix, found {len(matrices)}")
        for row in json.loads(matrices[0].read_text()):
            expected.add((f"dgx_{platform}", row["model"], row["test_case"]))

    tests = {}
    identities = set()
    for path in sorted(directory.rglob("attempt-*/metadata.json")):
        metadata = json.loads(path.read_text())
        key = (metadata["platform"], metadata["model"], metadata["test_case"])
        if key not in expected:
            raise ValueError(f"Unexpected test {key}")
        if metadata["collection_status"] != "collected" or metadata["workload_exit_code"] != 0:
            continue
        if key in tests:
            raise ValueError(f"Multiple successful attempts for {key}; select one run attempt")
        report = json.loads(path.with_name("coverage.json").read_text())
        if not report["meta"]["branch_coverage"]:
            raise ValueError(f"Branch coverage missing for {key}")
        points: dict[str, set[tuple]] = {"lines": set(), "branches": set()}
        for filename, file in report["files"].items():
            if not filename.startswith("megatron/"):
                raise ValueError(f"Expected repository-relative megatron path: {filename}")
            points["lines"].update((filename, line) for line in file["executed_lines"])
            points["branches"].update((filename, *arc) for arc in file["executed_branches"])
        if not points["lines"]:
            raise ValueError(f"Empty coverage for {key}")
        tests[key] = {"metadata": metadata, **points}
        identities.add((metadata["source_sha"], metadata["run_id"], metadata["run_attempt"]))
    missing = expected - tests.keys()
    if missing:
        raise ValueError(
            "Missing successful coverage: " + ", ".join("/".join(key) for key in sorted(missing))
        )
    if len(identities) != 1:
        raise ValueError(f"Mixed source/run identities: {identities}")
    return tests


def plan_reduction(tests: dict, fraction: float, max_loss: float) -> dict:
    """Greedily remove tests while bounding line AND branch loss on EACH platform."""
    counts: dict[tuple[str, str], collections.Counter] = {}
    for key, test in tests.items():
        for kind in ("lines", "branches"):
            counts.setdefault((key[0], kind), collections.Counter()).update(test[kind])
    totals = {key: len(counter) for key, counter in counts.items()}
    lost = {key: 0 for key in counts}
    retained = set(tests)
    removed: list[str] = []
    target = math.ceil(len(tests) * fraction)
    while len(removed) < target:
        candidates = []
        for key in sorted(retained):
            marginal = {}
            ratios = []
            for kind in ("lines", "branches"):
                group = (key[0], kind)
                marginal[group] = sum(counts[group][point] == 1 for point in tests[key][kind])
                ratios.append((lost[group] + marginal[group]) / max(totals[group], 1))
            if max(ratios) < max_loss:
                candidates.append(
                    (max(ratios), -tests[key]["metadata"]["elapsed_seconds"], key, marginal)
                )
        if not candidates:
            break
        _, _, key, marginal = min(candidates)
        for group, amount in marginal.items():
            lost[group] += amount
            counts[group].subtract(tests[key][group[1]])
        retained.remove(key)
        removed.append("/".join(key))
    return {
        "target_removed": target,
        "removed_count": len(removed),
        "removed_fraction": len(removed) / len(tests),
        "target_met": len(removed) == target,
        "removed": removed,
        "retained": ["/".join(key) for key in sorted(retained)],
        "coverage_loss": {
            f"{platform}/{kind}": {
                "lost": lost[(platform, kind)],
                "baseline": total,
                "fraction": lost[(platform, kind)] / max(total, 1),
            }
            for (platform, kind), total in sorted(totals.items())
        },
    }


def analyze(directory: Path, max_loss: float = 0.05) -> dict:
    """Return per-test counts, exclusive coverage, and candidate 10%/20% reductions."""
    tests = load_baseline(directory)
    counts: dict[tuple[str, str], collections.Counter] = {}
    for key, test in tests.items():
        for kind in ("lines", "branches"):
            counts.setdefault((key[0], kind), collections.Counter()).update(test[kind])
    inventory = []
    for key, test in sorted(tests.items()):
        inventory.append(
            {
                "test": "/".join(key),
                "observed_ranks": test["metadata"]["observed_ranks"],
                "requested_gpus": test["metadata"]["nodes"] * test["metadata"]["gpus_per_node"],
                "instrumented_seconds": test["metadata"]["elapsed_seconds"],
                **{kind: len(test[kind]) for kind in ("lines", "branches")},
                **{
                    f"exclusive_{kind}": sum(
                        counts[(key[0], kind)][point] == 1 for point in test[kind]
                    )
                    for kind in ("lines", "branches")
                },
            }
        )
    return {
        "test_count": len(tests),
        "max_loss_exclusive": max_loss,
        "limitation": "Python line/branch overlap is not numerical, native-kernel, topology, or assertion equivalence. Review candidates before changing CI.",
        "inventory": inventory,
        "plans": {
            str(fraction): plan_reduction(tests, fraction, max_loss) for fraction in (0.1, 0.2)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Downloaded coverage-functional-* artifacts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.directory), indent=2) + "\n")


if __name__ == "__main__":
    main()
