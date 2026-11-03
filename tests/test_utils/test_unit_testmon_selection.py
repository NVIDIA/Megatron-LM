# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Validate native selection transport independently of GPU test dependencies."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPTS = ROOT / "tests/unit_tests"
for name in ("testmon_cache", "testmon_selection"):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
cache = sys.modules["testmon_cache"]
selection = sys.modules["testmon_selection"]
IMAGE = "sha256:" + "a" * 64
SHA = "b" * 40
BUCKET = "tests/unit_tests/example/**/*.py"
PLATFORM = "dgx_h100"
TARGET = "tests/unit_tests/example/test_case.py::test_one"
MANDATORY = "tests/unit_tests/example/test_required.py"


def _write(path: Path, value: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{item}\n" for item in value))


def _snapshot(directory: Path) -> dict:
    return {
        str(p.relative_to(directory)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in directory.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def native(tmp_path, monkeypatch):
    DB = pytest.importorskip("testmon.db").DB
    root = tmp_path / "source"
    for name in cache.COMPATIBILITY_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    (root / "tests/unit_tests").mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(root)
    monkeypatch.delenv("UNIT_TESTMON_CACHE_KEY", raising=False)
    directory = tmp_path / "native"
    for phase in cache.PHASES:
        database = directory / phase / ".testmondata"
        database.parent.mkdir(parents=True)
        DB(str(database)).con.close()
        cache.record_phase(directory, phase)
        output = directory / ".testmon-work" / phase
        for rank in range(2):
            _write(output / f"rank-{rank}/selected-tests", [TARGET] if phase == "prod" else [])
        _write(output / "mandatory-tests", [MANDATORY] if phase == "prod" else [])
        _write(output / "selected-tests", sorted([TARGET, MANDATORY]) if phase == "prod" else [])
    identity = cache.cache_identity(root, BUCKET, PLATFORM, IMAGE)
    cache.finalize(directory, identity, "c" * 40, "123-1")
    (directory / "changed-files").write_text("megatron/core/example.py\n")
    args = (directory, BUCKET, PLATFORM, SHA, IMAGE, 2)
    return directory, args


def _export(native) -> dict:
    return selection.export_selection(*native[1])


def test_native_export_and_validation_do_not_modify_baseline(native):
    directory, args = native
    before = _snapshot(directory)
    result = _export(native)
    assert result["phases"]["prod"]["selected"] == sorted([TARGET, MANDATORY])
    assert result["phases"]["experimental"]["selected"] == []
    assert result["cache_key"].endswith("123-1")
    assert selection.validate_selection(*args, check_runtime=True) == result
    after = _snapshot(directory)
    assert {name: after[name] for name in before} == before
    assert set(after) - before.keys() == {"selection.json"}


@pytest.mark.parametrize("missing", ["rank-1/selected-tests", "selected-tests", "mandatory-tests"])
def test_incomplete_native_collection_cannot_be_exported(native, missing):
    directory, _ = native
    (directory / ".testmon-work/prod" / missing).unlink()
    with pytest.raises((ValueError, OSError)):
        _export(native)
    assert not (directory / "selection.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha", "d" * 40),
        ("bucket", "tests/unit_tests/other.py"),
        ("platform", "dgx_gb200"),
        ("image_id", "sha256:" + "d" * 64),
        ("world_size", 4),
        ("complete", False),
        ("schema", 99),
        ("schema", True),
        ("cache_key", "wrong"),
    ],
)
def test_identity_corruption_is_rejected(native, field, value):
    directory, args = native
    exported = _export(native)
    exported[field] = value
    (directory / "selection.json").write_text(json.dumps(exported))
    with pytest.raises(ValueError):
        selection.validate_selection(*args)


@pytest.mark.parametrize("filename", selection.SNAPSHOT_FILES)
def test_snapshot_corruption_is_rejected(native, filename):
    directory, args = native
    _export(native)
    path = directory / filename
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="snapshot"):
        selection.validate_selection(*args)


@pytest.mark.parametrize("problem", ["missing-rank", "union", "unsafe-target", "missing-phase"])
def test_malformed_native_ids_are_rejected(native, problem):
    directory, args = native
    exported = _export(native)
    prod = exported["phases"]["prod"]
    if problem == "missing-rank":
        del prod["ranks"]["1"]
    elif problem == "missing-phase":
        del exported["phases"]["experimental"]
    elif problem == "unsafe-target":
        prod["ranks"]["0"] = ["-x"]
    else:
        prod["selected"] = []
    (directory / "selection.json").write_text(json.dumps(exported))
    with pytest.raises(ValueError):
        selection.validate_selection(*args)


def test_prepare_exposes_exact_validated_ids_and_clears_stale_work(native):
    directory, args = native
    _export(native)
    validated = selection.validate_selection(*args, check_runtime=True)
    selection.prepare_selection(directory, validated)
    work = directory / ".testmon-work"
    assert not list(work.glob("*/rank-*"))
    for phase in cache.PHASES:
        assert (work / phase / "selected-tests").read_text().splitlines() == validated["phases"][
            phase
        ]["selected"]
        assert (work / phase / "mandatory-tests").read_text().splitlines() == validated["phases"][
            phase
        ]["mandatory"]


def test_runtime_drift_rejected_inside_execution_container(native, monkeypatch):
    directory, args = native
    _export(native)
    monkeypatch.setattr(cache, "runtime_identity", lambda: {})
    with pytest.raises(ValueError, match="runtime"):
        selection.validate_selection(*args, check_runtime=True)
    # Host-side validation does not compare host packages against the GPU image.
    selection.validate_selection(*args)


def test_cli_reports_native_empty_selection_and_baseline_key(native, capsys):
    directory, _ = native
    for phase in cache.PHASES:
        work = directory / ".testmon-work" / phase
        for name in (
            "rank-0/selected-tests",
            "rank-1/selected-tests",
            "selected-tests",
            "mandatory-tests",
        ):
            _write(work / name, [])
    args = [
        "export",
        "--cache-dir",
        str(directory),
        "--bucket",
        BUCKET,
        "--platform",
        PLATFORM,
        "--source-sha",
        SHA,
        "--image-id",
        IMAGE,
        "--world-size",
        "2",
    ]
    assert selection.main(args) == 0
    output = capsys.readouterr().out
    assert "has_tests=false" in output
    assert "cache_key=unit-testmon-v1-main-dgx_h100-" in output
    args[0] = "validate"
    assert selection.main(args[:-2]) == 0
    args[0] = "prepare"
    assert selection.main(args[:-2]) == 2
