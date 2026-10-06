# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent / "python_scripts" / "changed_functional_tests.py"
SPEC = importlib.util.spec_from_file_location("changed_functional_tests", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
changed_tests = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(changed_tests)

TESTED_SHA = "a" * 40
PREFIX = "tests/functional_tests/test_cases"


def write_artifact(directory, entries):
    directory.mkdir(exist_ok=True)
    (directory / "metadata.json").write_text(
        json.dumps({"tested_sha": TESTED_SHA, "changed_files": len(entries)}), encoding="utf-8"
    )
    (directory / "changed-files").write_text(
        "".join(f"{entry['filename']}\n" for entry in entries), encoding="utf-8"
    )
    (directory / "files.json").write_text(json.dumps(entries), encoding="utf-8")
    return directory


@pytest.fixture
def artifact(tmp_path):
    return write_artifact(
        tmp_path / "artifact",
        [{"filename": f"{PREFIX}/gpt/example/model_config.yaml", "status": "modified"}],
    )


def invoke(artifact):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--artifact-dir",
            str(artifact),
            "--tested-sha",
            TESTED_SHA,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_selects_acmr_cases_and_preserves_base_file_list(tmp_path):
    entries = [
        {"filename": f"{PREFIX}/gpt/zeta/model_config.yaml", "status": "added"},
        {"filename": f"{PREFIX}/gpt/alpha/nested/golden.json", "status": "modified"},
        {"filename": f"{PREFIX}/gpt/alpha/model_config.yaml", "status": "modified"},
        {"filename": f"{PREFIX}/other/alpha/golden.json", "status": "modified"},
        {"filename": f"{PREFIX}/other/copied/golden.json", "status": "copied"},
        {
            "filename": f"{PREFIX}/gpt/new_case/model_config.yaml",
            "previous_filename": f"{PREFIX}/gpt/old_case/model_config.yaml",
            "status": "renamed",
        },
        {"filename": f"{PREFIX}/gpt/removed/model_config.yaml", "status": "removed"},
        {"filename": f"{PREFIX}/gpt/type_changed/model_config.yaml", "status": "changed"},
        {"filename": f"{PREFIX}/gpt/untouched/model_config.yaml", "status": "unchanged"},
        {"filename": "docs/readme.md", "status": "modified"},
        {"filename": f"{PREFIX}/gpt/no_file", "status": "modified"},
        {"filename": "other/" + f"{PREFIX}/gpt/not_a_case/config.yaml", "status": "added"},
    ]
    artifact = write_artifact(tmp_path, entries)
    original = (artifact / "changed-files").read_bytes()
    assert changed_tests.load_changed_test_cases(artifact, TESTED_SHA) == [
        "alpha",
        "copied",
        "new_case",
        "zeta",
    ]
    assert (artifact / "changed-files").read_bytes() == original
    result = invoke(artifact)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "alpha,copied,new_case,zeta\n"
    assert result.stderr == ""


def test_empty_artifact_is_distinct_from_unavailable(tmp_path):
    artifact = write_artifact(tmp_path / "empty", [])
    assert changed_tests.load_changed_test_cases(artifact, TESTED_SHA) == []
    result = invoke(artifact)
    assert (result.returncode, result.stdout, result.stderr) == (0, "\n", "")
    missing = invoke(tmp_path / "unavailable")
    assert missing.returncode == 2
    assert missing.stdout == ""
    assert "metadata.json" in missing.stderr


@pytest.mark.parametrize("name", ["metadata.json", "changed-files", "files.json"])
def test_missing_artifact_files_are_rejected(artifact, name):
    (artifact / name).unlink()
    result = invoke(artifact)
    assert result.returncode == 2
    assert result.stdout == ""
    assert name in result.stderr


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        None,
        {},
        {"tested_sha": "b" * 40, "changed_files": 1},
        *[
            {"tested_sha": TESTED_SHA, "changed_files": count}
            for count in (None, "1", True, False, 1.0, -1, 3001, 0, 2)
        ],
    ],
)
def test_invalid_metadata_is_rejected(artifact, metadata):
    (artifact / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata.json"):
        changed_tests.load_changed_test_cases(artifact, TESTED_SHA)


@pytest.mark.parametrize("name", ["metadata.json", "files.json"])
def test_invalid_json_is_rejected(artifact, name):
    (artifact / name).write_text("{", encoding="utf-8")
    result = invoke(artifact)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "Changed functional tests:" in result.stderr


@pytest.mark.parametrize(
    "entries",
    [
        {},
        None,
        [],
        [None],
        [{}],
        [{"filename": 3, "status": "modified"}],
        [{"filename": "different/file.py", "status": "modified"}],
        [{"filename": "a.py", "status": "added"}, {"filename": "b.py", "status": "added"}],
    ],
)
def test_file_manifest_must_match_base_list(artifact, entries):
    (artifact / "files.json").write_text(json.dumps(entries), encoding="utf-8")
    with pytest.raises(ValueError, match="files.json"):
        changed_tests.load_changed_test_cases(artifact, TESTED_SHA)


@pytest.mark.parametrize("status", [None, "unknown", "M", 1, {}, []])
def test_unsupported_status_is_rejected(artifact, status):
    (artifact / "files.json").write_text(
        json.dumps([{"filename": f"{PREFIX}/gpt/example/model_config.yaml", "status": status}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported file status"):
        changed_tests.load_changed_test_cases(artifact, TESTED_SHA)


def test_file_manifest_order_must_match_base_list(tmp_path):
    entries = [
        {"filename": "source/a.py", "status": "modified"},
        {"filename": "source/b.py", "status": "modified"},
    ]
    artifact = write_artifact(tmp_path, entries)
    (artifact / "files.json").write_text(json.dumps(entries[::-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="same order"):
        changed_tests.load_changed_test_cases(artifact, TESTED_SHA)


def test_duplicate_paths_are_rejected(tmp_path):
    entry = {"filename": f"{PREFIX}/gpt/example/model_config.yaml", "status": "modified"}
    artifact = write_artifact(tmp_path, [entry, entry])
    with pytest.raises(ValueError, match="duplicate paths"):
        changed_tests.load_changed_test_cases(artifact, TESTED_SHA)


def test_maximum_complete_file_count_is_valid(tmp_path):
    artifact = write_artifact(
        tmp_path,
        [{"filename": f"source/{index}.py", "status": "modified"} for index in range(3000)],
    )
    assert changed_tests.load_changed_test_cases(artifact, TESTED_SHA) == []


@pytest.mark.parametrize(
    "filename",
    [
        "",
        "/absolute/file",
        "./file",
        "../file",
        "a/../file",
        "a/./file",
        "a//file",
        "a/",
        "a\rfile",
        "a\0file",
        "a\nfile",
    ],
)
def test_invalid_paths_are_rejected(tmp_path, filename):
    artifact = write_artifact(tmp_path, [{"filename": filename, "status": "modified"}])
    with pytest.raises(ValueError):
        changed_tests.load_changed_test_cases(artifact, TESTED_SHA)


def test_case_names_cannot_contain_csv_separator(tmp_path):
    artifact = write_artifact(
        tmp_path, [{"filename": f"{PREFIX}/gpt/a,b/config.yaml", "status": "added"}]
    )
    with pytest.raises(ValueError, match="comma"):
        changed_tests.load_changed_test_cases(artifact, TESTED_SHA)


@pytest.mark.parametrize("content", [b"source/file.py", b"source/file.py\r\n", b"\xff\n"])
def test_invalid_base_list_encoding_and_records_are_rejected(artifact, content):
    (artifact / "changed-files").write_bytes(content)
    result = invoke(artifact)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "Changed functional tests:" in result.stderr


@pytest.mark.parametrize("tested_sha", ["", "a" * 39, "A" * 40, "z" * 40])
def test_invalid_tested_sha_is_rejected(artifact, tested_sha):
    with pytest.raises(ValueError, match="full lowercase Git commit SHA"):
        changed_tests.load_changed_test_cases(artifact, tested_sha)
