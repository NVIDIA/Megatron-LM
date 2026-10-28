# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import importlib.util
import json
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


def run_metadata(**overrides):
    return {
        "id": 123,
        "run_attempt": 2,
        "event": "merge_group",
        "repository": {"full_name": publisher.REPOSITORY},
        "head_repository": {"full_name": publisher.REPOSITORY},
        "head_branch": "gh-readonly-queue/main/pr-1",
        "head_sha": MERGE_SHA,
        "status": "completed",
        "conclusion": "success",
        **overrides,
    }


@pytest.mark.parametrize(
    "scenario,accepted",
    [
        ("identical-tree", True),
        ("exact-sha", True),
        ("different-tree", False),
        ("failed", False),
        ("cancelled", False),
        ("wrong-event", False),
        ("fork", False),
        ("dev", False),
        ("pending", True),
        ("timeout", False),
    ],
)
def test_producer_requires_successful_trusted_matching_tree(monkeypatch, scenario, accepted):
    run = run_metadata()
    if scenario in {"failed", "cancelled"}:
        run["conclusion"] = "failure" if scenario == "failed" else "cancelled"
    if scenario == "exact-sha":
        run["head_sha"] = MAIN_SHA
    if scenario == "wrong-event":
        run["event"] = "pull_request"
    if scenario == "fork":
        run["head_repository"]["full_name"] = "someone/Megatron-LM"
    if scenario == "dev":
        run["head_branch"] = "gh-readonly-queue/dev/pr-1"
    if scenario in {"pending", "timeout"}:
        run.update(status="in_progress", conclusion=None)
    sleeps = []
    monkeypatch.setattr(publisher.time, "sleep", sleeps.append)

    def api(endpoint):
        if endpoint == f"git/commits/{MAIN_SHA}":
            return {"tree": {"sha": "tree"}}
        if endpoint == f"git/commits/{MERGE_SHA}":
            return {"tree": {"sha": "other" if scenario == "different-tree" else "tree"}}
        if endpoint.startswith("actions/workflows/cicd-main.yml/runs?"):
            return {"workflow_runs": [copy.deepcopy(run)]}
        assert endpoint == "actions/runs/123"
        return run_metadata() if scenario == "pending" else copy.deepcopy(run)

    monkeypatch.setattr(publisher, "api", api)
    assert (publisher.find_producer(MAIN_SHA) is not None) == accepted
    assert len(sleeps) == (20 if scenario == "timeout" else 1 if scenario == "pending" else 0)


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    DB = pytest.importorskip("testmon.db").DB

    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / next(iter(cache.COMPATIBILITY_FILES))
    dataset.parent.mkdir(parents=True)
    dataset.write_text("fixture")
    identity = cache.cache_identity(tmp_path, "tests/unit_tests/test_example.py", "dgx_h100")
    directory = tmp_path / "download"
    for phase in cache.PHASES:
        database = directory / phase / ".testmondata"
        database.parent.mkdir(parents=True)
        DB(str(database)).con.close()
        cache.record_phase(directory, phase)
    run = run_metadata()
    (directory / "producer.json").write_text(
        json.dumps({"identity": identity, "source_sha": MERGE_SHA, "generation": "123-2"})
    )
    artifact = {"id": 42, "name": identity["cache_prefix"] + "123-2", "identity": identity}
    return directory, artifact, run


@pytest.mark.parametrize("problem", [None, "sha", "attempt", "identity", "incomplete"])
def test_promotion_rejects_invalid_artifacts(baseline, problem):
    directory, artifact, run = baseline
    producer_file = directory / "producer.json"
    producer = json.loads(producer_file.read_text())
    if problem == "sha":
        producer["source_sha"] = "c" * 40
    elif problem == "attempt":
        producer["generation"] = "123-1"
    elif problem == "identity":
        producer["identity"]["compatibility"]["platform"] = "dgx_gb200"
    elif problem == "incomplete":
        (directory / "experimental/metadata.json").unlink()
    producer_file.write_text(json.dumps(producer))
    if problem:
        with pytest.raises((ValueError, OSError)):
            publisher.promote(directory, artifact, run, MAIN_SHA, "456-1")
        assert not Path("assets_dir/testmon/manifest.json").exists()
    else:
        key = publisher.promote(directory, artifact, run, MAIN_SHA, "456-1")
        promoted = cache.validate_cache(Path("assets_dir/testmon"), artifact["identity"], key)
        assert promoted["source_sha"] == MAIN_SHA
        assert publisher.promote(directory, artifact, run, MAIN_SHA, "456-1") == key


def test_artifact_discovery_paginates_and_excludes_old_attempts(baseline, monkeypatch):
    _, artifact, run = baseline
    for platform in cache.PLATFORMS.values():
        path = Path(platform["recipe"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(
                {"products": [{"test_case": [artifact["identity"]["compatibility"]["bucket"]]}]}
            )
        )
    wrong_attempt = {"id": 1, "name": artifact["name"].removesuffix("2") + "1", "expired": False}
    calls = []

    def api(endpoint):
        calls.append(endpoint)
        return {
            "artifacts": (
                [wrong_attempt] * 100
                if "page=1" in endpoint.split("&")[-1]
                else [{"id": 42, "name": artifact["name"], "expired": False}]
            )
        }

    monkeypatch.setattr(publisher, "api", api)
    assert publisher.artifact_matrix(run) == [artifact]
    assert len(calls) == 2


@pytest.mark.parametrize(
    "scenario", ["success", "superseded", "no-producer", "download-failure", "save-failure"]
)
def test_main_publishes_only_validated_artifacts(baseline, monkeypatch, tmp_path, scenario):
    directory, artifact, run = baseline
    for key, value in {
        "GITHUB_SHA": MAIN_SHA,
        "GITHUB_RUN_ID": "456",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "RUNNER_TEMP": str(tmp_path),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        publisher, "find_producer", lambda sha: None if scenario == "no-producer" else run
    )
    monkeypatch.setattr(publisher, "artifact_matrix", lambda value: [artifact])
    monkeypatch.setattr(
        publisher,
        "api",
        lambda endpoint: {"object": {"sha": "c" * 40 if scenario == "superseded" else MAIN_SHA}},
    )
    operations = []

    def actions(operation, *args):
        operations.append(operation)
        if scenario == operation + "-failure":
            raise RuntimeError("transport failed")
        if operation == "download":
            import shutil

            assert args[:2] == ("42", "123")
            shutil.copytree(directory, args[2])
        else:
            assert operation == "save"
            manifest = cache.validate_cache(Path(args[0]), artifact["identity"], args[1])
            assert manifest["source_sha"] == MAIN_SHA

    monkeypatch.setattr(publisher, "actions", actions)
    if scenario.endswith("-failure"):
        with pytest.raises(RuntimeError, match="transport failed"):
            publisher.main()
        assert not (tmp_path / "summary").exists()
    else:
        publisher.main()
    assert operations == (
        []
        if scenario in {"superseded", "no-producer"}
        else ["download"] if scenario == "download-failure" else ["download", "save"]
    )
    if scenario == "success":
        assert "Published" in (tmp_path / "summary").read_text()
