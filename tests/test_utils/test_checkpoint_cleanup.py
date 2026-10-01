# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
DAY_SECONDS = 24 * 60 * 60


def _run_cleanup(directory: Path, paths: str | None) -> None:
    recipe = yaml.safe_load((ROOT / "tests/test_utils/recipes/_cleanup.yaml").read_text())
    environment = os.environ.copy()
    environment.pop("CLEANUP_PATHS", None)
    if paths is not None:
        environment["CLEANUP_PATHS"] = paths
    result = subprocess.run(
        ["bash"],
        input=recipe["spec"]["script"].format(),
        cwd=directory,
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _set_age(path: Path, days: float, now: float) -> None:
    timestamp = now - days * DAY_SECONDS
    os.utime(path, (timestamp, timestamp), follow_symlinks=False)


def test_cleanup_weekly_boundary_preserves_release_retention(tmp_path: Path) -> None:
    base = tmp_path / "checkpoint assets"
    base.mkdir()
    now = time.time()
    cases = [
        ("weekly-13d23h", 13 + 23 / 24, True),
        ("weekly-14d", 14, False),
        ("weekly-14d12h", 14.5, False),
        ("weekly-15d", 15, False),
        ("release-14d", 14, True),
        ("release-28d12h", 28.5, True),
        ("release-29d", 29, False),
    ]
    for name, age, _ in cases:
        checkpoint = base / name
        checkpoint.mkdir()
        (checkpoint / "weights.pt").write_text("checkpoint")
        _set_age(checkpoint, age, now)

    _run_cleanup(tmp_path, str(base))

    assert {path.name for path in base.iterdir()} == {
        name for name, _, retained in cases if retained
    }


def test_cleanup_limits_deletion_to_matching_child_directories(tmp_path: Path) -> None:
    bases = [tmp_path / "checkpoint assets", tmp_path / "checkpoint artifacts"]
    now = time.time()
    for base in bases:
        base.mkdir()
        expired = base / "weekly-expired with spaces"
        expired.mkdir()
        (expired / "weights.pt").write_text("checkpoint")
        _set_age(expired, 30, now)

    unrelated = bases[0] / "other-run"
    nested = unrelated / "weekly-nested"
    nested.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "weights.pt").write_text("keep")
    preserved = [unrelated, nested, outside]
    for prefix in ("weekly", "release"):
        link = bases[0] / f"{prefix}-symlink"
        link.symlink_to(outside, target_is_directory=True)
        regular_file = bases[0] / f"{prefix}-file"
        regular_file.write_text("keep")
        preserved.extend([link, regular_file])
    for path in preserved:
        _set_age(path, 60, now)

    paths = "\n".join([str(bases[0]), "", str(tmp_path / "missing"), str(bases[1])])
    _run_cleanup(tmp_path, paths)

    assert all(path.exists() for path in preserved)
    assert all((bases[0] / f"{prefix}-symlink").is_symlink() for prefix in ("weekly", "release"))
    assert (outside / "weights.pt").read_text() == "keep"
    assert all(not (base / "weekly-expired with spaces").exists() for base in bases)


def test_cleanup_without_configured_paths_is_a_noop(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weekly-expired"
    checkpoint.mkdir()
    _set_age(checkpoint, 60, time.time())

    for paths in (None, ""):
        _run_cleanup(tmp_path, paths)
        assert checkpoint.is_dir()


def test_cleanup_jobs_are_independent_of_trigger_and_functional_configuration() -> None:
    workflow = yaml.safe_load((ROOT / ".gitlab/stages/04.functional-tests.yml").read_text())

    assert workflow[".functional_cleanup_rules"]["rules"] == [
        {"if": '$BUILD == "no"', "when": "never"},
        {"when": "on_success"},
    ]
    cleanup = workflow[".functional_cleanup"]
    assert cleanup["extends"] == [".functional_cleanup_rules"]
    assert cleanup["needs"] == ["test:build_image"]
    for platform in ("dgx_a100", "dgx_h100", "dgx_gb200"):
        job = workflow[f"functional:cleanup_{platform}"]
        assert job["extends"] == [".functional_cleanup"]
        assert "rules" not in job
        assert "needs" not in job
