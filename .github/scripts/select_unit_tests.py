# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Select all unit-test buckets on one CPU runner and publish the GPU matrix."""

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Artifact baselines come only from this workflow's push or dispatch runs on the
# PR's target branch. PR runs can neither create such runs nor add artifacts to them.
PRODUCER_WORKFLOW = ".github/workflows/populate-build-cache.yml"
PRODUCER_EVENTS = {"push", "workflow_dispatch"}
PRODUCER_RUNS = 5
SHA = re.compile(r"[0-9a-f]{40}")


def _actions(operation: str, *args: str) -> None:
    # The workflow supplies the small adapter for GitHub's Node-only SDKs.
    subprocess.run(
        [os.environ["TESTMON_NODE"], "-e", os.environ["TESTMON_ACTIONS"], operation, *args],
        check=True,
    )


def _gh(*args: str) -> str:
    return subprocess.check_output(["gh", *args], text=True)


def _gh_records(endpoint: str, query: str, paginate: bool = True) -> list[dict]:
    pages = ["--paginate"] if paginate else []
    output = _gh("api", *pages, endpoint, "--jq", f"{query} | @json")
    return [json.loads(line) for line in output.splitlines() if line]


def _python(script: str, args: list[str]) -> dict[str, str]:
    output = subprocess.check_output(
        [sys.executable, f"tests/unit_tests/{script}", *args], text=True
    )
    logger.info(output.strip())
    return dict(line.split("=", 1) for line in output.strip().splitlines())


def _artifact_baselines(
    repository: str, branch: str, run_id: int | None = None, attempt: int | None = None
) -> dict[str, dict]:
    """Map lookup prefixes to the newest artifact baseline of a trusted producer run.

    ``run_id`` and ``attempt`` restrict the search to the generation pinned by a
    prepared checkout, whose run must pass the same trust checks.
    """
    tip = _gh("api", f"repos/{repository}/git/ref/heads/{branch}", "--jq", ".object.sha").strip()
    if not SHA.fullmatch(tip):
        raise ValueError(f"unable to resolve {branch}")
    if run_id is None:
        candidates = _gh_records(
            f"repos/{repository}/actions/workflows/{Path(PRODUCER_WORKFLOW).name}/runs"
            f"?branch={branch}&exclude_pull_requests=true&per_page=20",
            ".workflow_runs[]",
            paginate=False,
        )
    else:
        candidates = _gh_records(f"repos/{repository}/actions/runs/{run_id}", ".", paginate=False)
    runs = [
        run
        for run in candidates
        if run.get("event") in PRODUCER_EVENTS
        and run.get("head_branch") == branch
        and run.get("path") == PRODUCER_WORKFLOW
        and (run.get("repository") or {}).get("full_name") == repository
        and (run.get("head_repository") or {}).get("full_name") == repository
        and isinstance(run.get("id"), int)
        and SHA.fullmatch(str(run.get("head_sha")))
    ]
    baselines: dict[str, dict] = {}
    # A newer producer run's generation wins for each bucket it published.
    for run in sorted(runs, key=lambda run: run["id"], reverse=True)[:PRODUCER_RUNS]:
        # A tag or other ref can share the branch name; require the branch's history.
        status = _gh(
            "api",
            f"repos/{repository}/compare/{run['head_sha']}...{tip}?per_page=1",
            "--jq",
            ".status",
        ).strip()
        if status not in {"ahead", "identical"}:
            logger.warning("Ignoring producer run %s outside %s", run["id"], branch)
            continue
        name = re.compile(
            rf"(unit-testmon-v[0-9]+-{re.escape(branch)}-.+-){run['id']}-([1-9][0-9]*)"
        )
        found: dict[str, dict] = {}
        for artifact in _gh_records(
            f"repos/{repository}/actions/runs/{run['id']}/artifacts?per_page=100", ".artifacts[]"
        ):
            match = name.fullmatch(str(artifact.get("name")))
            source = artifact.get("workflow_run") or {}
            if (
                not match
                or artifact.get("expired") is not False
                or not isinstance(artifact.get("id"), int)
                or source.get("id") != run["id"]
                or source.get("head_sha") != run["head_sha"]
            ):
                continue
            prefix, generation_attempt = match.group(1), int(match.group(2))
            if attempt is not None and generation_attempt != attempt:
                continue
            if generation_attempt > found.get(prefix, {}).get("attempt", 0):
                found[prefix] = {
                    "attempt": generation_attempt,
                    "artifact_id": artifact["id"],
                    "digest": str(artifact.get("digest") or ""),
                    "key": artifact["name"],
                    "run_id": run["id"],
                    "source_sha": run["head_sha"],
                }
        for prefix, baseline in found.items():
            baselines.setdefault(prefix, baseline)
    return baselines


def _cell(text: str) -> str:
    return " ".join(str(text).replace("|", "\\|").split())


def main() -> None:
    """Restore each baseline, run the existing selector, and upload its plan."""
    target_branch = os.environ["TARGET_BRANCH"]
    platform = os.environ["RECIPE_PLATFORM"]
    repository = os.environ["GITHUB_REPOSITORY"]
    generation = f"{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
    baseline_generation = os.environ.get("TESTMON_GENERATION", "")
    target_arguments = [] if target_branch == "main" else ["--target-branch", target_branch]
    temporary = Path(os.environ["RUNNER_TEMP"])
    cache_dir = Path("assets_dir/testmon")
    plan = "assets_dir/testmon-plan/plan.json"
    matrix = []
    artifacts = None
    summary = [
        "### Unit tests selection",
        "",
        f"Target branch: `{target_branch}`. Configure only requests selection. Each row is "
        "the CPU decision for one bucket; every allocated GPU job reselects with Testmon and "
        "runs its full bucket when the baseline or selection is unusable.",
        "",
        "| Bucket | Baseline | CPU decision | GPU job |",
        "| --- | --- | --- | --- |",
    ]

    # Restore sequentially: the path is part of the existing cache version.
    for entry in json.loads(os.environ["BUCKETS"]):
        bucket = entry["bucket"]
        logger.info("Selecting %s", bucket)
        # Rejects target branches without a trusted baseline producer.
        identity = _python(
            "testmon_cache.py",
            [
                "identity",
                "--bucket",
                bucket,
                "--platform",
                platform,
                *target_arguments,
                "--output",
                str(temporary / "unit-testmon-identity.json"),
            ],
        )
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        matched_key = ""
        baseline = None
        prefix = identity["cache_prefix"]
        if identity.get("baseline_transport", "cache") == "cache":
            key_file = temporary / "unit-testmon-matched-key"
            requested_key = prefix + (baseline_generation or f"lookup-{generation}")
            try:
                _actions(
                    "restore",
                    str(cache_dir),
                    requested_key,
                    "" if baseline_generation else prefix,
                    str(key_file),
                )
                matched_key = key_file.read_text()
                if baseline_generation and matched_key:
                    manifest = json.loads((cache_dir / "manifest.json").read_text())
                    if (
                        matched_key != requested_key
                        or not isinstance(manifest, dict)
                        or manifest.get("source_sha") != os.environ.get("TESTMON_BASE_SHA")
                    ):
                        raise ValueError("cache does not match the pinned baseline")
                source = f"`{matched_key}`" if matched_key else "cache miss"
            except (OSError, ValueError, subprocess.CalledProcessError) as error:
                logger.warning("Testmon cache restore failed for %s: %s", bucket, error)
                matched_key = ""
                source = "cache restore failed"
        else:
            if artifacts is None:
                try:
                    # A prepared checkout pins one producer generation; otherwise use the newest.
                    pinned = (
                        [int(part) for part in baseline_generation.split("-")]
                        if baseline_generation
                        else [None, None]
                    )
                    artifacts = _artifact_baselines(repository, target_branch, *pinned)
                except (OSError, subprocess.CalledProcessError, ValueError, KeyError, TypeError):
                    logger.exception("Unable to list trusted %s baselines", target_branch)
                    artifacts = {}
            baseline = artifacts.get(prefix)
            if (
                baseline_generation
                and baseline
                and (
                    baseline["key"] != prefix + baseline_generation
                    or baseline["source_sha"] != os.environ.get("TESTMON_BASE_SHA")
                )
            ):
                logger.warning("Testmon artifact for %s does not match the pinned baseline", bucket)
                baseline = None
            source = f"no trusted `{target_branch}` artifact"
            if baseline:
                source = f"`{baseline['key']}` (`{baseline['source_sha'][:12]}`)"
                try:
                    _actions(
                        "download",
                        str(baseline["artifact_id"]),
                        str(cache_dir),
                        str(baseline["run_id"]),
                        repository,
                        baseline["digest"],
                    )
                    matched_key = baseline["key"]
                except subprocess.CalledProcessError:
                    logger.warning("Testmon artifact download failed for %s", bucket)
                    source += ", download failed"
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
                *target_arguments,
                *(["--baseline-source-sha", baseline["source_sha"]] if baseline else []),
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
        details = json.loads(Path(plan).read_text())
        artifact = ""
        if mode == "selected":
            bucket_hash = hashlib.sha256(bucket.encode()).hexdigest()[:16]
            artifact = f"unit-test-plan-{platform}-{bucket_hash}-{generation}"
            # Upload empty plans too: the coverage gate uses them as skip evidence.
            _actions("upload", artifact, plan, "assets_dir/testmon-plan")
            files = set().union(*details["phases"].values())
            decision = (
                f"{len(files)} affected file(s), {len(details['mandatory_files'])} mandatory"
                if files
                else "no affected tests"
            )
        else:
            decision = "full bucket: " + (details["reason"] if matched_key else "no baseline")
        if has_tests == "true":
            matrix.append(
                {
                    **entry,
                    "mode": "preselected" if mode == "selected" else "full",
                    "plan_artifact": artifact,
                    # GPU jobs download the same immutable producer artifact.
                    **(
                        {
                            "baseline_run_id": str(baseline["run_id"]),
                            "baseline_artifact_id": str(baseline["artifact_id"]),
                            "baseline_key": baseline["key"],
                            "baseline_source_sha": baseline["source_sha"],
                        }
                        if baseline
                        else {}
                    ),
                }
            )
        gpu = "Testmon reselects" if has_tests == "true" else "not allocated"
        summary.append(f"| `{bucket}` | {source} | {_cell(decision)} | {gpu} |")

    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
        stream.write("\n".join(summary) + "\n")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        stream.write(f"matrix={json.dumps(matrix, separators=(',', ':'))}\n")
        stream.write(f"has_tests={str(bool(matrix)).lower()}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
