# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Discover tests for all buckets on one GPU runner and publish execution plans."""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _actions(operation: str, *args: str) -> None:
    # The workflow supplies the small adapter for GitHub's Node-only SDKs.
    subprocess.run(
        [os.environ["TESTMON_NODE"], "-e", os.environ["TESTMON_ACTIONS"], operation, *args],
        check=True,
    )


def _python(script: str, args: list[str]) -> dict[str, str]:
    output = subprocess.check_output(
        [sys.executable, f"tests/unit_tests/{script}", *args], text=True
    )
    logger.info(output.strip())
    return dict(line.split("=", 1) for line in output.strip().splitlines())


def _clear_cache(cache_dir: Path) -> None:
    # Native collection writes rank-private files as the container user.
    subprocess.run(["sudo", "rm", "-rf", "--", str(cache_dir)], check=True)


def _launch_gpu_selection(bucket: str, selection_gpus: str = "1") -> None:
    subprocess.run(
        [
            sys.executable,
            "tests/test_utils/python_scripts/launch_nemo_run_workload.py",
            "--scope",
            "unit-tests",
            "--model",
            "unit-tests",
            "--test-case",
            bucket,
            "--environment",
            "dev",
            "--platform",
            os.environ["RECIPE_PLATFORM"],
            "--tag",
            "latest",
            "--unit-testmon-mode",
            "select-only",
            "--unit-testmon-selection-gpus",
            selection_gpus,
            "--container-image",
            os.environ["CONTAINER_IMAGE"],
            "--hf-home",
            "/mnt/datadrive/TestData/nemo-fw/TestData/HF_HOME",
        ],
        env={**os.environ, "PYTHONPATH": str(Path.cwd()), "NEMORUN_HOME": str(Path.cwd())},
        check=True,
    )


def _collect_gpu_selection(bucket: str, cache_dir: Path, selection_gpus: str) -> dict:
    _launch_gpu_selection(bucket, selection_gpus)
    phases = {}
    mandatory = set()
    for phase in ("prod", "experimental"):
        directory = cache_dir / ".testmon-work" / phase
        phases[phase] = (directory / "selected-tests").read_text().splitlines()
        mandatory.update((directory / "mandatory-tests").read_text().splitlines())
    return {
        "schema": 1,
        "source_sha": os.environ["SOURCE_SHA"],
        "bucket": bucket,
        "platform": os.environ["RECIPE_PLATFORM"],
        "mode": "selected",
        "reason": "native pytest/Testmon discovery on GPU workers",
        "selection_gpus": selection_gpus,
        "phases": phases,
        "mandatory_files": sorted(mandatory),
    }


def _collect_for_bucket(bucket: str, cache_dir: Path) -> dict:
    compare = os.environ.get("UNIT_TESTMON_COMPARE_SELECTIONS", "true")
    if compare not in {"true", "false"}:
        raise ValueError("UNIT_TESTMON_COMPARE_SELECTIONS must be 'true' or 'false'")
    exceptions = json.loads(os.environ.get("UNIT_TESTMON_MULTI_RANK_BUCKETS", "{}"))
    if not isinstance(exceptions, dict):
        raise ValueError("UNIT_TESTMON_MULTI_RANK_BUCKETS must map platforms to bucket lists")
    platform_exceptions = exceptions.get(os.environ["RECIPE_PLATFORM"], [])
    if not isinstance(platform_exceptions, list) or any(
        not isinstance(item, str) for item in platform_exceptions
    ):
        raise ValueError("Multi-rank selection exceptions must be a list of bucket names")
    selection_gpus = "recipe" if bucket in platform_exceptions else "1"
    try:
        selection = _collect_gpu_selection(bucket, cache_dir, selection_gpus)
    except (OSError, subprocess.SubprocessError) as error:
        if selection_gpus == "recipe":
            raise
        logger.warning(
            "Single-worker selection failed for %s; retrying recipe ranks: %s", bucket, error
        )
        return _collect_gpu_selection(bucket, cache_dir, "recipe")

    if selection_gpus == "1" and compare == "true":
        reference = _collect_gpu_selection(bucket, cache_dir, "recipe")
        differences = {
            phase: {
                "single_only": sorted(
                    set(selection["phases"][phase]) - set(reference["phases"][phase])
                ),
                "recipe_only": sorted(
                    set(reference["phases"][phase]) - set(selection["phases"][phase])
                ),
            }
            for phase in ("prod", "experimental")
        }
        differences["mandatory_files"] = {
            "single_only": sorted(
                set(selection["mandatory_files"]) - set(reference["mandatory_files"])
            ),
            "recipe_only": sorted(
                set(reference["mandatory_files"]) - set(selection["mandatory_files"])
            ),
        }
        equivalent = not any(points for phase in differences.values() for points in phase.values())
        if not equivalent:
            logger.warning("Single-worker selection differs for %s; using recipe ranks", bucket)
            selection = reference
        selection["rank_comparison"] = {"equivalent": equivalent, "differences": differences}
    return selection


def _select_on_gpu(bucket: str, cache_dir: Path, identity_path: Path, matched_key: str) -> dict:
    subprocess.run(
        [
            sys.executable,
            "tests/unit_tests/testmon_cache.py",
            "validate",
            "--cache-dir",
            str(cache_dir),
            "--identity",
            str(identity_path),
            "--matched-key",
            matched_key,
        ],
        check=True,
    )
    pr_files = Path(os.environ["PR_FILES_DIR"])
    metadata = json.loads((pr_files / "metadata.json").read_text())
    changed_text = (pr_files / "changed-files").read_text()
    if (
        metadata["tested_sha"] != os.environ["SOURCE_SHA"]
        or type(metadata["changed_files"]) is not int
        or not 0 <= metadata["changed_files"] <= 3000
        or type(metadata["changed_paths"]) is not int
        or not 0 <= metadata["changed_paths"] <= 6000
        or metadata["changed_paths"] != len(changed_text.splitlines())
        or metadata["changed_paths"] != changed_text.count("\n")
    ):
        raise ValueError("PR changed-file evidence is incomplete or belongs to another commit")
    shutil.copyfile(pr_files / "changed-files", cache_dir / "changed-files")
    return _collect_for_bucket(bucket, cache_dir)


def main() -> None:
    """Restore each baseline, collect on GPU workers, and upload execution plans."""
    if os.environ["TARGET_BRANCH"] != "main":
        raise ValueError("Testmon selection requires main")
    platform = os.environ["RECIPE_PLATFORM"]
    generation = f"{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
    temporary = Path(os.environ["RUNNER_TEMP"])
    cache_dir = Path("assets_dir/testmon")
    plan = Path("assets_dir/testmon-plan/plan.json")
    plan.parent.mkdir(parents=True, exist_ok=True)
    matrix = []
    summary = [
        "### Unit tests selection",
        "",
        "| Bucket | Mode | Execution required |",
        "| --- | --- | --- |",
    ]

    # Restore sequentially: the path is part of the existing cache version.
    for entry in json.loads(os.environ["BUCKETS"]):
        bucket = entry["bucket"]
        logger.info("Selecting %s", bucket)
        # These bespoke recipes intentionally always execute in full.
        if bucket == "tests/unit_tests/inference/test_batch_invariant_fp8.py" or (
            bucket.startswith("tests/unit_tests/inference/") and "/test_disagg_" in bucket
        ):
            matrix.append({**entry, "mode": "full", "plan_artifact": ""})
            summary.append(f"| `{bucket}` | full | true |")
            continue
        identity = _python(
            "testmon_cache.py",
            [
                "identity",
                "--bucket",
                bucket,
                "--platform",
                platform,
                "--output",
                str(temporary / "unit-testmon-identity.json"),
            ],
        )
        _clear_cache(cache_dir)
        plan.unlink(missing_ok=True)
        matched_key = ""
        key_file = temporary / "unit-testmon-matched-key"
        prefix = identity["cache_prefix"]
        try:
            _actions(
                "restore", str(cache_dir), f"{prefix}lookup-{generation}", prefix, str(key_file)
            )
            matched_key = key_file.read_text()
        except (OSError, subprocess.SubprocessError):
            logger.warning("Testmon cache restore failed for %s", bucket)
        try:
            result = _select_on_gpu(
                bucket, cache_dir, temporary / "unit-testmon-identity.json", matched_key
            )
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
            logger.warning(
                "GPU selection unavailable for %s; scheduling its full bucket: %s", bucket, error
            )
            result = {"mode": "full"}
        mode = result["mode"]
        has_tests = mode == "full" or any(result["phases"].values())
        artifact = ""
        if mode == "selected":
            bucket_hash = hashlib.sha256(bucket.encode()).hexdigest()[:16]
            artifact = f"unit-test-plan-{platform}-{bucket_hash}-{generation}"
            # Upload empty plans too: the coverage gate uses them as skip evidence.
            plan.write_text(json.dumps(result, indent=2) + "\n")
            _actions("upload", artifact, str(plan), "assets_dir/testmon-plan")
        if has_tests:
            matrix.append(
                {
                    **entry,
                    "mode": "preselected" if mode == "selected" else "full",
                    "plan_artifact": artifact,
                }
            )
        summary.append(f"| `{bucket}` | {mode} | {str(has_tests).lower()} |")

    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
        stream.write("\n".join(summary) + "\n")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        stream.write(f"matrix={json.dumps(matrix, separators=(',', ':'))}\n")
        stream.write(f"has_tests={str(bool(matrix)).lower()}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
