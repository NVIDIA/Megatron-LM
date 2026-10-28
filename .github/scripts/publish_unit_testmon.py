# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Promote successful merge-group Testmon artifacts from an identical main tree."""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/unit_tests"))
from testmon_cache import PHASES, PLATFORMS, cache_identity, finalize, validate_cache

logger = logging.getLogger(__name__)
REPOSITORY = "NVIDIA/Megatron-LM"
WORKFLOW = "cicd-main.yml"


def api(endpoint: str) -> dict:
    """Read repository metadata with the job's read-only GitHub token."""
    return json.loads(
        subprocess.check_output(["gh", "api", f"repos/{REPOSITORY}/{endpoint}"], text=True)
    )


def actions(operation: str, *args: str) -> None:
    """Use the workflow adapter for artifact and cache transport."""
    subprocess.run(
        [os.environ["TESTMON_NODE"], "-e", os.environ["TESTMON_ACTIONS"], operation, *args],
        check=True,
    )


def find_producer(source_sha: str) -> dict | None:
    """Accept only successful main merge groups whose complete Git tree matches."""
    tree = api(f"git/commits/{source_sha}")["tree"]["sha"]
    runs = api(f"actions/workflows/{WORKFLOW}/runs?event=merge_group&per_page=100")["workflow_runs"]
    for run in runs:
        if (
            run["event"] != "merge_group"
            or run["repository"]["full_name"] != REPOSITORY
            or run["head_repository"]["full_name"] != REPOSITORY
            or not run["head_branch"].startswith("gh-readonly-queue/main/")
        ):
            continue
        if run["head_sha"] != source_sha:
            if api(f"git/commits/{run['head_sha']}")["tree"]["sha"] != tree:
                continue
        # The required gate may finish before the overall workflow completes.
        for _ in range(20):
            if run["status"] == "completed":
                break
            time.sleep(30)
            run = api(f"actions/runs/{run['id']}")
        if run["status"] == "completed" and run["conclusion"] == "success":
            return run
    return None


def artifact_matrix(run: dict) -> list[dict]:
    """Select only named recipe buckets from this exact successful run attempt."""
    artifacts = []
    page = 1
    while True:
        batch = api(f"actions/runs/{run['id']}/artifacts?per_page=100&page={page}")["artifacts"]
        artifacts.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    generation = f"{run['id']}-{run['run_attempt']}"
    expected = {}
    for platform, config in PLATFORMS.items():
        recipe = yaml.safe_load(Path(config["recipe"]).read_text())
        for product in recipe["products"]:
            for bucket in product["test_case"]:
                identity = cache_identity(Path.cwd(), bucket, platform)
                expected[identity["cache_prefix"] + generation] = identity
    selected = []
    seen = set()
    for artifact in artifacts:
        name = artifact["name"]
        if artifact["expired"] or name not in expected:
            continue
        if name in seen:
            raise ValueError(f"Duplicate baseline artifact: {name}")
        seen.add(name)
        selected.append({"id": artifact["id"], "name": name, "identity": expected[name]})
    return selected


def promote(directory: Path, artifact: dict, run: dict, source_sha: str, generation: str) -> str:
    """Validate producer provenance and checksums before writing a main manifest."""
    identity = artifact["identity"]
    if any(path.is_symlink() for path in directory.rglob("*")):
        raise ValueError("Symlinks are not allowed in baseline artifacts")
    manifest = validate_cache(
        directory, identity, artifact["name"], source_ref=f"refs/heads/{run['head_branch']}"
    )
    if manifest["source_sha"] != run["head_sha"]:
        raise ValueError("Baseline source does not match the successful merge group")
    # Keep the observed GPU image for diagnostics; CPU hosts do not resolve it.
    identity = {**identity, "image_id": manifest["image_id"]}
    # The cache path and gzip compression must match existing PR restores.
    destination = Path("assets_dir/testmon")
    if destination.exists():
        shutil.rmtree(destination)
    for phase in PHASES:
        (destination / phase).mkdir(parents=True)
        for name in (".testmondata", "metadata.json"):
            shutil.copyfile(directory / phase / name, destination / phase / name)
    manifest = finalize(destination, identity, source_sha, generation)
    manifest["producer_sha"] = run["head_sha"]
    manifest["producer_run_id"] = run["id"]
    manifest["producer_run_attempt"] = run["run_attempt"]
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    key = identity["cache_prefix"] + generation
    validate_cache(destination, identity, key)
    return key


def main() -> None:
    """Publish validated artifacts only while this push is the current main head."""
    if (
        os.environ["GITHUB_REPOSITORY"] != REPOSITORY
        or os.environ["GITHUB_REF"] != "refs/heads/main"
        or os.environ["GITHUB_EVENT_NAME"] != "push"
    ):
        raise ValueError("Baseline promotion requires a push to NVIDIA/Megatron-LM main")
    source_sha = os.environ["GITHUB_SHA"]
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != source_sha:
        raise ValueError("Publisher checkout does not match the main push")
    if api("git/ref/heads/main")["object"]["sha"] != source_sha:
        logger.info("A newer main push superseded this publication")
        return
    run = find_producer(source_sha)
    if run is None:
        logger.info("No successful merge-group run matches this main tree; retaining old caches")
        return
    artifacts = artifact_matrix(run)
    if not artifacts:
        logger.info(
            "Matching run has no baseline artifacts; retain caches or use manual regeneration"
        )
        return
    generation = f"{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
    for artifact in artifacts:
        if api("git/ref/heads/main")["object"]["sha"] != source_sha:
            logger.info("A newer main push superseded this publication")
            return
        directory = Path(os.environ["RUNNER_TEMP"]) / f"testmon-{artifact['id']}"
        actions("download", str(artifact["id"]), str(run["id"]), str(directory))
        key = promote(directory, artifact, run, source_sha, generation)
        actions("save", "assets_dir/testmon", key)
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as summary:
            summary.write(f"- Published `{key}` from merge-group run {run['id']}\n")
        logger.info("Published %s", key)
    logger.info("Published %s Testmon baselines", len(artifacts))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
