# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
SCRIPT_PATH = ROOT / "tests/unit_tests/testmon_mandatory.py"
CONFIG_PATH = ROOT / "tests/unit_tests/testmon_mandatory_tests.yaml"

for name, filename in (
    ("find_test_cases", "find_test_cases.py"),
    ("testmon_mandatory", "testmon_mandatory.py"),
):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH.with_name(filename))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
mandatory = sys.modules["testmon_mandatory"]

MAPPINGS = [
    {
        "source_dirs": ["megatron/core/foo", "megatron/core/bar/*.py"],
        "test_buckets": {
            "dgx_h100": ["tests/unit_tests/foo/**/*.py", "tests/unit_tests/test_bar.py"],
            "dgx_gb200": ["tests/unit_tests/test_bar.py"],
        },
    },
    {"source_dirs": ["megatron/core/baz"], "test_buckets": {"dgx_h100": ["tests/unit_tests/baz"]}},
]


@pytest.fixture
def project(tmp_path, monkeypatch):
    for relative in (
        "tests/unit_tests/foo/test_a.py",
        "tests/unit_tests/foo/nested/test_b.py",
        "tests/unit_tests/foo/conftest.py",
        "tests/unit_tests/foo/helpers.py",
        "tests/unit_tests/test_bar.py",
        "tests/unit_tests/test_other.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def test_case():\n    assert True\n")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"mappings": MAPPINGS}))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_repository_config_is_valid():
    mappings = mandatory.load_mappings(CONFIG_PATH)
    assert mappings
    for mapping in mappings:
        for source in mapping["source_dirs"]:
            assert (ROOT / source).exists(), source
        for patterns in mapping["test_buckets"].values():
            for pattern in patterns:
                base = pattern.split("/**")[0].rsplit("/", 1)[0] if "*" in pattern else pattern
                assert (ROOT / base).exists(), pattern


@pytest.mark.parametrize(
    "document",
    [
        "[]",
        "mappings: {}",
        "mappings: [{source_dirs: [a]}]",
        "mappings: [{source_dirs: a, test_buckets: {dgx_h100: [t]}}]",
        "mappings: [{source_dirs: [a], test_buckets: {dgx_unknown: [t]}}]",
        "mappings: [{source_dirs: [a], test_buckets: {dgx_h100: t}}]",
        "mappings: [{source_dirs: [''], test_buckets: {dgx_h100: [t]}}]",
    ],
)
def test_invalid_config_is_rejected(tmp_path, document):
    config = tmp_path / "config.yaml"
    config.write_text(document)
    with pytest.raises(ValueError):
        mandatory.load_mappings(config)


@pytest.mark.parametrize(
    ("changed", "platform", "expected_sources", "expected_patterns"),
    [
        ([], "dgx_h100", [], []),
        (["megatron/core/foobar/x.py"], "dgx_h100", [], []),
        (["megatron/core/other.py", "docs/a.md"], "dgx_h100", [], []),
        (
            ["megatron/core/foo/x.py"],
            "dgx_h100",
            ["megatron/core/foo"],
            ["tests/unit_tests/foo/**/*.py", "tests/unit_tests/test_bar.py"],
        ),
        (
            ["megatron/core/foo"],
            "dgx_h100",
            ["megatron/core/foo"],
            MAPPINGS[0]["test_buckets"]["dgx_h100"],
        ),
        (
            ["megatron/core/bar/x.py"],
            "dgx_gb200",
            ["megatron/core/bar/*.py"],
            ["tests/unit_tests/test_bar.py"],
        ),
        # Glob entries use fnmatch semantics, where `*` also spans directories.
        (
            ["megatron/core/bar/sub/x.py"],
            "dgx_gb200",
            ["megatron/core/bar/*.py"],
            ["tests/unit_tests/test_bar.py"],
        ),
        (["megatron/core/baz/x.py"], "dgx_gb200", [], []),
        (
            ["megatron/core/baz/x.py", "megatron/core/foo/y.py"],
            "dgx_h100",
            ["megatron/core/foo", "megatron/core/baz"],
            [
                "tests/unit_tests/foo/**/*.py",
                "tests/unit_tests/test_bar.py",
                "tests/unit_tests/baz",
            ],
        ),
    ],
)
def test_triggered_patterns(changed, platform, expected_sources, expected_patterns):
    assert mandatory.triggered_patterns(MAPPINGS, changed, platform) == (
        expected_sources,
        expected_patterns,
    )


def test_mandatory_files_are_restricted_to_the_bucket_and_test_modules(project):
    patterns = ["tests/unit_tests/foo/**/*.py", "tests/unit_tests/test_bar.py"]
    assert mandatory.mandatory_files(patterns, "tests/unit_tests/foo/**/*.py", set()) == [
        "tests/unit_tests/foo/nested/test_b.py",
        "tests/unit_tests/foo/test_a.py",
    ]
    assert mandatory.mandatory_files(patterns, "tests/unit_tests/**/*.py", set()) == [
        "tests/unit_tests/foo/nested/test_b.py",
        "tests/unit_tests/foo/test_a.py",
        "tests/unit_tests/test_bar.py",
    ]
    ignored = {"tests/unit_tests/foo/test_a.py", "tests/unit_tests/foo/nested/test_b.py"}
    assert mandatory.mandatory_files(patterns, "tests/unit_tests/**/*.py", ignored) == [
        "tests/unit_tests/test_bar.py"
    ]
    assert mandatory.mandatory_files(patterns, "tests/unit_tests/test_other.py", set()) == []


def test_merge_selection_replaces_node_ids_of_mandatory_files(tmp_path):
    selection = tmp_path / "selected-tests"
    selection.write_text(
        "tests/unit_tests/foo/test_a.py::test_case\n"
        "tests/unit_tests/test_other.py::TestOther::test_case[x]\n"
        "\n"
    )
    merged = mandatory.merge_selection(selection, ["tests/unit_tests/foo/test_a.py"])
    expected = [
        "tests/unit_tests/foo/test_a.py",
        "tests/unit_tests/test_other.py::TestOther::test_case[x]",
    ]
    assert merged == expected
    assert selection.read_text().splitlines() == expected


def _invoke(project, changed, bucket, platform="dgx_h100", *extra):
    work = project / "work"
    work.mkdir(exist_ok=True)
    (project / "changed-files").write_text("".join(f"{path}\n" for path in changed))
    selection = work / "selected-tests"
    selection.write_text("tests/unit_tests/foo/test_a.py::test_case\n")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--config",
            str(project / "config.yaml"),
            "--changed-files",
            str(project / "changed-files"),
            "--bucket",
            bucket,
            "--platform",
            platform,
            "--selection",
            str(selection),
            *extra,
        ],
        cwd=project,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        timeout=30,
    )
    return result, selection, work / "mandatory-tests"


def test_cli_adds_mandatory_files_and_records_them(project):
    result, selection, record = _invoke(
        project,
        ["megatron/core/foo/x.py"],
        "tests/unit_tests/**/*.py",
        "dgx_h100",
        "--ignore=tests/unit_tests/foo/nested/test_b.py",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "changed megatron/core/foo; added 2 file(s)" in result.stdout
    expected = ["tests/unit_tests/foo/test_a.py", "tests/unit_tests/test_bar.py"]
    assert selection.read_text().splitlines() == expected
    assert record.read_text().splitlines() == expected


def test_cli_keeps_selection_when_nothing_mapped_changed(project):
    result, selection, record = _invoke(
        project, ["megatron/core/other.py"], "tests/unit_tests/**/*.py"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no mapped source changes" in result.stdout
    assert selection.read_text().splitlines() == ["tests/unit_tests/foo/test_a.py::test_case"]
    assert record.read_text() == ""


def test_cli_fails_without_changed_files_or_with_invalid_config(project):
    result, _, _ = _invoke(project, [], "tests/unit_tests/**/*.py")
    assert result.returncode == 0
    (project / "changed-files").unlink()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--changed-files",
            str(project / "changed-files"),
            "--bucket",
            "tests/unit_tests/**/*.py",
            "--platform",
            "dgx_h100",
            "--selection",
            str(project / "work/selected-tests"),
        ],
        cwd=project,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "Testmon mandatory tests:" in result.stderr
    assert "Traceback" not in result.stderr


def test_runner_applies_mandatory_tests_in_enforce_path():
    runner = (ROOT / "tests/unit_tests/run_ci_test.sh").read_text()
    enforce = runner.split("run_enforced_tests() {", 1)[1].split("\n}\n", 1)[0]
    assert enforce.count("apply_mandatory_tests") == 2
    assert enforce.index("merge_rank_selections prod") < enforce.index("apply_mandatory_tests prod")
    assert "tests/unit_tests/testmon_mandatory.py" in runner
    assert '--platform "dgx_$PLATFORM"' in runner
    action = (ROOT / ".github/actions/action.yml").read_text()
    assert "assets_dir/testmon/changed-files" in action
