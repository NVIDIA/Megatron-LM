# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Throwaway PR #7824 validation: private baseline, control, and mandatory execution."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUCKET = "tests/unit_tests/distributed/mfsdp_v2/**/*.py"
PHASES = ("prod", "experimental")
PROBE_SOURCE = "megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/owner_planning.py"


def require(condition: bool, message: str) -> None:
    """Fail the validation immediately when a required condition is absent."""
    if not condition:
        raise ValueError(message)


def baseline_hashes(cache: Path) -> dict[str, str]:
    """Fingerprint the original databases and metadata for both phases."""
    return {
        f"{phase}/{name}": hashlib.sha256((cache / phase / name).read_bytes()).hexdigest()
        for phase in PHASES
        for name in (".testmondata", "metadata.json")
    }


def run_stage(name: str, mode: str, cache: Path, evidence: Path, report: dict) -> None:
    """Run the real GPU runner and preserve its results before the next stage."""
    destination = evidence / name
    destination.mkdir()
    print(f"Testmon private probe: running {name}; logs: {destination}", flush=True)
    command = [
        "bash",
        "tests/unit_tests/run_ci_test.sh",
        "--tag",
        "latest",
        "--environment",
        "dev",
        "--bucket",
        BUCKET,
        "--platform",
        "h100",
        "--unit-test-repeat",
        "1",
        "--log-dir",
        str(destination / "ranks"),
    ]
    with (destination / "runner.log").open("w") as stream:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "UNIT_TESTMON_MODE": mode, "UNIT_TESTMON_CACHE_DIR": str(cache)},
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    state = {"returncode": result.returncode, "phases": {}}
    report["stages"][name] = state
    summary = (cache / "summary.md").read_text() if (cache / "summary.md").exists() else ""
    (destination / "summary.md").write_text(summary)
    for phase in PHASES:
        work = cache / ".testmon-work" / phase
        if not work.exists():
            continue
        saved = destination / phase
        saved.mkdir()
        raw = set()
        for selection in sorted(work.glob("rank-*/selected-tests")):
            shutil.copyfile(selection, saved / f"{selection.parent.name}-selected-tests")
            raw.update(selection.read_text().splitlines())
        (saved / "raw-selected-tests").write_text("".join(f"{item}\n" for item in sorted(raw)))
        state["phases"][phase] = {"raw_selected_count": len(raw)}
        for filename in ("selected-tests", "mandatory-tests"):
            source = work / filename
            if source.exists():
                shutil.copyfile(source, saved / filename)
                state["phases"][phase][filename] = source.read_text().splitlines()
    require(
        result.returncode == 0, f"{name} exited {result.returncode}; see {destination}/runner.log"
    )
    expected_result = "baseline produced" if mode == "baseline" else "selective tests passed"
    require(f"- Mode: `{mode}`" in summary, f"{name}: missing expected {mode} mode in summary")
    require(
        f"- Result: {expected_result}" in summary, f"{name}: unexpected result or full fallback"
    )


def run_probe(args: argparse.Namespace, report: dict) -> None:
    """Compare empty and mapped PR changes against one private baseline."""
    os.chdir(ROOT)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata = json.loads(args.metadata.read_text())
    require(
        isinstance(metadata, dict) and metadata.get("tested_sha") == sha,
        "PR artifact tested_sha does not match the checked-out commit",
    )
    for field, limit in (("changed_files", 3000), ("changed_paths", 6000)):
        count = metadata.get(field)
        require(type(count) is int and 0 <= count <= limit, f"invalid artifact {field}: {count!r}")
    contents = args.changed_files.read_bytes().decode("utf-8")
    require(
        contents.endswith("\n") and "\r" not in contents and "\0" not in contents,
        "PR changed paths must be complete newline-terminated records",
    )
    paths = contents.splitlines()
    require(
        all(paths) and len(paths) == len(set(paths)) == metadata["changed_paths"],
        "PR changed path count or uniqueness does not match metadata",
    )
    require(
        0 < metadata["changed_files"] <= len(paths) <= 2 * metadata["changed_files"],
        "PR record and expanded path counts are inconsistent",
    )
    require(
        PROBE_SOURCE in paths and (ROOT / PROBE_SOURCE).is_file(),
        f"real PR artifact must contain the existing probe source: {PROBE_SOURCE}",
    )
    expected = sorted(
        str(path.relative_to(ROOT))
        for path in (ROOT / "tests/unit_tests/distributed/mfsdp_v2").rglob("test_*.py")
    )
    require(len(expected) == 16, f"expected 16 MFSDP v2 test files, found {len(expected)}")
    report.update(tested_sha=sha, artifact_metadata=metadata, expected_mandatory_files=expected)
    shutil.copyfile(args.metadata, args.evidence_dir / "input-metadata.json")
    shutil.copyfile(args.changed_files, args.evidence_dir / "input-changed-files")
    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=args.evidence_dir))
    report["attempt"] = str(attempt.relative_to(args.evidence_dir))
    # Private phase databases never receive a shared-cache manifest or cache-save call.
    with tempfile.TemporaryDirectory(prefix="testmon-private-", dir=ROOT / "assets_dir") as private:
        cache = Path(private)
        run_stage("baseline", "baseline", cache, attempt, report)
        original = baseline_hashes(cache)
        report["baseline_hashes"] = original
        for stage, changed in (("control", ""), ("mapped", contents)):
            (cache / "changed-files").write_text(changed)
            run_stage(stage, "enforce", cache, attempt, report)
            current = baseline_hashes(cache)
            report["stages"][stage]["baseline_hashes"] = current
            require(current == original, f"{stage}: enforcement changed the private baseline")
            for phase in PHASES:
                state = report["stages"][stage]["phases"].get(phase, {})
                mandatory = state.get("mandatory-tests")
                require(
                    mandatory == ([] if stage == "control" else expected),
                    f"{stage}/{phase}: unexpected mandatory files: {mandatory!r}",
                )
                if stage == "mapped":
                    require(
                        set(expected) <= set(state.get("selected-tests", [])),
                        f"{stage}/{phase}: mandatory whole files missing from selection",
                    )
                    raw = (attempt / stage / phase / "raw-selected-tests").read_text().splitlines()
                    selected_files = {nodeid.split("::", 1)[0] for nodeid in raw}
                    state["files_added_beyond_testmon"] = sorted(set(expected) - selected_files)
                    if phase == "prod":
                        require(
                            state["files_added_beyond_testmon"],
                            "mapped/prod: Testmon already selected every file; override not proven",
                        )
    report["status"] = "passed"


def main() -> int:
    """Write a persistent report and return a failing status unless all checks pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changed-files", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("changed_files", "metadata", "evidence_dir"):
        setattr(args, name, getattr(args, name).resolve())
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "bucket": BUCKET, "stages": {}}
    try:
        run_probe(args, report)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report["error"] = str(error)
        print(f"Testmon private probe failed: {error}", file=sys.stderr)
    finally:
        (args.evidence_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        lines = [
            "### Testmon private probe",
            "",
            f"- Result: **{report['status']}**",
            f"- Bucket: `{BUCKET}`",
            f"- Evidence: `{report.get('attempt', 'report.json')}`",
        ]
        for name, stage in report["stages"].items():
            for phase, state in stage["phases"].items():
                lines.append(
                    f"- {name}/{phase}: {state['raw_selected_count']} raw Testmon node IDs; "
                    f"{len(state.get('mandatory-tests', []))} mandatory files"
                )
        if "error" in report:
            lines.append(f"- Error: {report['error']}")
        (args.evidence_dir / "summary.md").write_text("\n".join(lines) + "\n")
    if report["status"] == "passed":
        print("TESTMON_PRIVATE_PROBE=passed", flush=True)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
