# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Select all unit-test buckets on one CPU runner and publish the GPU matrix."""

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


def main() -> None:
    """Restore each baseline, run the existing selector, and upload its plan."""
    if os.environ["TARGET_BRANCH"] != "main":
        raise ValueError("Testmon selection requires main")
    platform = os.environ["RECIPE_PLATFORM"]
    generation = f"{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
    temporary = Path(os.environ["RUNNER_TEMP"])
    cache_dir = Path("assets_dir/testmon")
    plan = "assets_dir/testmon-plan/plan.json"
    matrix = []
    summary = [
        "### Unit tests selection",
        "",
        "| Bucket | Mode | GPU required |",
        "| --- | --- | --- |",
    ]

    # Restore sequentially: the path is part of the existing cache version.
    for entry in json.loads(os.environ["BUCKETS"]):
        bucket = entry["bucket"]
        logger.info("Selecting %s", bucket)
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
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        matched_key = ""
        key_file = temporary / "unit-testmon-matched-key"
        prefix = identity["cache_prefix"]
        try:
            _actions(
                "restore", str(cache_dir), f"{prefix}lookup-{generation}", prefix, str(key_file)
            )
            matched_key = key_file.read_text()
        except subprocess.CalledProcessError:
            logger.warning("Testmon cache restore failed for %s", bucket)
        result = _python(
            "testmon_selector.py",
            [
                "select",
                "--cache-dir",
                str(cache_dir),
                "--bucket",
                bucket,
                "--platform",
                platform,
                "--source-sha",
                os.environ["SOURCE_SHA"],
                "--matched-key",
                matched_key,
                "--pr-files-dir",
                os.environ["PR_FILES_DIR"],
                "--output",
                plan,
            ],
        )
        mode, has_tests = result["mode"], result["has_tests"]
        if (
            mode not in {"selected", "full"}
            or has_tests not in {"true", "false"}
            or (mode == "full" and has_tests != "true")
        ):
            raise ValueError(f"Invalid Testmon selection result for {bucket}")
        artifact = ""
        if mode == "selected":
            bucket_hash = hashlib.sha256(bucket.encode()).hexdigest()[:16]
            artifact = f"unit-test-plan-{platform}-{bucket_hash}-{generation}"
            # Upload empty plans too: the coverage gate uses them as skip evidence.
            _actions("upload", artifact, plan, "assets_dir/testmon-plan")
        if has_tests == "true":
            matrix.append(
                {
                    **entry,
                    "mode": "preselected" if mode == "selected" else "full",
                    "plan_artifact": artifact,
                }
            )
        summary.append(f"| `{bucket}` | {mode} | {has_tests} |")

    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
        stream.write("\n".join(summary) + "\n")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        stream.write(f"matrix={json.dumps(matrix, separators=(',', ':'))}\n")
        stream.write(f"has_tests={str(bool(matrix)).lower()}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
