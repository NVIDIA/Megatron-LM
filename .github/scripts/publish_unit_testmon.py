# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Publish Testmon data from a successful merge group with the same tree as main."""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/unit_tests"))
from testmon_cache import PHASES, PLATFORMS, cache_identity, finalize

logger = logging.getLogger(__name__)
REPOSITORY = "NVIDIA/Megatron-LM"
CACHE_DIR = Path("assets_dir/testmon")


def api(path: str) -> dict:
    return json.loads(subprocess.check_output(["gh", "api", f"repos/{REPOSITORY}/{path}"]))


def find_producer(source_sha: str) -> dict | None:
    """Match the landed tree, including squash merges with a new commit ID."""
    tree = api(f"git/commits/{source_sha}")["tree"]["sha"]
    # On a main push, search the 100 most recent merge groups for its producer.
    runs = api("actions/workflows/cicd-main.yml/runs?event=merge_group&per_page=100")
    matches = [
        run
        for run in runs["workflow_runs"]
        if run["event"] == "merge_group"
        and (run["head_repository"] or {}).get("full_name") == REPOSITORY
        and (run["head_branch"] or "").startswith("gh-readonly-queue/main/")
        and run["head_commit"]
        and run["head_commit"]["id"] == run["head_sha"]
        and run["head_commit"]["tree_id"] == tree
    ]
    # The queue can land the group before the last workflow jobs finish.
    while matches:
        pending = []
        for run in matches:
            if run["status"] == "completed":
                if run["conclusion"] == "success":
                    return run
            else:
                pending.append(run["id"])
        if not pending:
            break
        logger.info("Waiting for matching merge-group runs %s", pending)
        time.sleep(60)
        matches = [api(f"actions/runs/{run_id}") for run_id in pending]
    return None


def artifact_matrix(run: dict) -> list[dict]:
    """Choose the latest recorded attempt for each enabled bucket of each active platform."""
    expected = {}
    for platform, config in PLATFORMS.items():
        recipe = yaml.safe_load(Path(config["recipe"]).read_text())
        for product in recipe["products"]:
            if product.get("testmon") is False:
                continue
            for bucket in product["test_case"]:
                identity = cache_identity(Path.cwd(), bucket, platform)
                expected[identity["cache_prefix"] + str(run["id"])] = (platform, bucket)

    selected: dict[str, dict] = {}
    active = set()
    page = 1
    while True:
        artifacts = api(f"actions/runs/{run['id']}/artifacts?per_page=100&page={page}")["artifacts"]
        for artifact in artifacts:
            name = artifact["name"]
            key, _, number = name.rpartition("-")
            if key not in expected or not number.isdecimal():
                continue
            attempt = int(number)
            if not 1 <= attempt <= run["run_attempt"]:
                continue
            platform, bucket = expected[key]
            active.add(platform)
            if artifact["expired"] or (key in selected and attempt <= selected[key]["attempt"]):
                continue
            selected[key] = dict(
                name=name,
                platform=platform,
                bucket=bucket,
                generation=f"{run['id']}-{attempt}",
                attempt=attempt,
            )
        if len(artifacts) < 100:
            break
        page += 1

    missing = [
        f"{platform}/{bucket}"
        for key, (platform, bucket) in expected.items()
        if platform in active and key not in selected
    ]
    if missing:
        raise ValueError(f"Missing merge-group Testmon artifacts: {', '.join(missing)}")
    return list(selected.values())


def promote(
    directory: Path,
    bucket: str,
    platform: str,
    producer_sha: str,
    producer_generation: str,
    source_sha: str,
    generation: str,
) -> str:
    """Validate recorded phases and write the cache manifest for the landed commit."""
    if any(path.is_symlink() for path in directory.rglob("*")):
        raise ValueError("Unexpected symlink in Testmon artifact")
    producer = json.loads((directory / "producer.json").read_text())
    identity = cache_identity(Path.cwd(), bucket, platform, producer["identity"]["image_id"])
    if (
        producer["source_sha"] != producer_sha
        or producer["generation"] != producer_generation
        or producer["identity"] != identity
    ):
        raise ValueError("Testmon producer SHA, generation, or bucket identity does not match")
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    for phase in PHASES:
        (CACHE_DIR / phase).mkdir(parents=True)
        for name in (".testmondata", "metadata.json"):
            shutil.copyfile(directory / phase / name, CACHE_DIR / phase / name)
    finalize(CACHE_DIR, identity, source_sha, generation)
    return identity["cache_prefix"] + generation


def main() -> None:
    source_sha = os.environ["GITHUB_SHA"]
    run = find_producer(source_sha)
    if run is None:
        logger.info("No successful main merge group matches %s", source_sha)
        return
    records = artifact_matrix(run)
    if not records:
        logger.info("Merge group %s supplied no Testmon artifacts", run["id"])
        return

    generation = f"{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        command = ["gh", "run", "download", str(run["id"]), "-R", REPOSITORY, "-D", temporary]
        for record in records:
            command.extend(["--name", record["name"]])
        subprocess.run(command, check=True)
        for record in records:
            # gh extracts a single explicitly named artifact directly into the destination.
            source = directory / record["name"] if len(records) > 1 else directory
            key = promote(
                source,
                record["bucket"],
                record["platform"],
                run["head_sha"],
                record["generation"],
                source_sha,
                generation,
            )
            subprocess.run(
                [os.environ["TESTMON_NODE"], "-e", os.environ["TESTMON_SAVE"], str(CACHE_DIR), key],
                check=True,
            )
    logger.info(
        "Published %d Testmon caches for %s from merge group %s",
        len(records),
        source_sha,
        run["id"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
