# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "publish_unit_testmon", ROOT / ".github/scripts/publish_unit_testmon.py"
)
assert SPEC is not None and SPEC.loader is not None
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)
cache = sys.modules["testmon_cache"]
MAIN_SHA = "a" * 40
MERGE_SHA = "b" * 40


@pytest.fixture
def publication(tmp_path, monkeypatch):
    DB = pytest.importorskip("testmon.db").DB
    monkeypatch.chdir(tmp_path)
    for filename in cache.COMPATIBILITY_FILES:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    bucket = "tests/unit_tests/test_example.py"
    for platform in cache.PLATFORMS.values():
        path = Path(platform["recipe"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"products": [{"test_case": [bucket]}]}))
    identity = cache.cache_identity(tmp_path, bucket, "dgx_h100", "sha256:" + "c" * 64)
    directory = tmp_path / "artifact"
    for phase in cache.PHASES:
        database = directory / phase / ".testmondata"
        database.parent.mkdir(parents=True)
        DB(str(database)).con.close()
        cache.record_phase(directory, phase)
    (directory / "producer.json").write_text(
        json.dumps({"identity": identity, "source_sha": MERGE_SHA, "generation": "123-2"})
    )
    responses = {
        f"git/commits/{MAIN_SHA}": {"tree": {"sha": "same-tree"}},
        f"git/commits/{MERGE_SHA}": {"tree": {"sha": "same-tree"}},
        "git/ref/heads/main": {"object": {"sha": MAIN_SHA}},
        "actions/workflows/cicd-main.yml/runs?event=merge_group&per_page=100": {
            "workflow_runs": [
                {
                    "id": 123,
                    "run_attempt": 2,
                    "event": "merge_group",
                    "repository": {"full_name": publisher.REPOSITORY},
                    "head_repository": {"full_name": publisher.REPOSITORY},
                    "head_branch": "gh-readonly-queue/main/pr-1",
                    "head_sha": MERGE_SHA,
                    "status": "completed",
                    "conclusion": "success",
                }
            ]
        },
        "actions/runs/123/artifacts?per_page=100&page=1": {
            "artifacts": [{"id": 42, "name": identity["cache_prefix"] + "123-2", "expired": False}]
        },
    }
    monkeypatch.setattr(publisher, "api", responses.__getitem__)
    for key, value in {
        "GITHUB_SHA": MAIN_SHA,
        "GITHUB_RUN_ID": "456",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "RUNNER_TEMP": str(tmp_path),
    }.items():
        monkeypatch.setenv(key, value)
    saved_keys = []

    def actions(operation, *args):
        if operation == "download":
            assert args[:2] == ("42", "123")
            shutil.copytree(directory, args[2])
        else:
            assert operation == "save"
            shutil.copytree(args[0], tmp_path / "published")
            saved_keys.append(args[1])

    monkeypatch.setattr(publisher, "actions", actions)
    return directory, identity, saved_keys


def test_publishes_matching_tree_as_restorable_main_cache(publication, tmp_path):
    directory, identity, saved_keys = publication

    publisher.main()

    assert saved_keys == [identity["cache_prefix"] + "456-1"]
    published = tmp_path / "published"
    manifest = cache.validate_cache(published, identity, saved_keys[0])
    assert manifest["source_sha"] == MAIN_SHA
    assert manifest["generation"] == "456-1"
    assert manifest["image_id"] == "sha256:" + "c" * 64
    for phase in ("prod", "experimental"):
        assert (published / phase / ".testmondata").read_bytes() == (
            directory / phase / ".testmondata"
        ).read_bytes()


def test_rejects_wrong_attempt_without_replacing_existing_cache(publication, tmp_path):
    directory, _, saved_keys = publication
    producer_file = directory / "producer.json"
    producer = json.loads(producer_file.read_text())
    producer["generation"] = "123-1"
    producer_file.write_text(json.dumps(producer))
    destination = Path("assets_dir/testmon")
    destination.mkdir(parents=True)
    previous = destination / "existing-cache"
    previous.write_bytes(b"previous baseline")

    with pytest.raises(ValueError, match="Baseline does not match the successful merge group"):
        publisher.main()

    assert saved_keys == []
    assert list(destination.iterdir()) == [previous]
    assert previous.read_bytes() == b"previous baseline"
    assert not (tmp_path / "published").exists()
    assert not (tmp_path / "summary").exists()
