# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU checks for unit-test recipe selection and legacy-suite isolation."""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTOR_PATH = REPO_ROOT / "tests/unit_tests/find_test_cases.py"


@pytest.fixture
def selector():
    # Import the standalone script without loading the GPU unit-test package.
    spec = importlib.util.spec_from_file_location("find_test_cases", SELECTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_recipe(tmp_path, products, platform="h100"):
    path = tmp_path / f"tests/test_utils/recipes/{platform}/unit-tests.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"products": products}))
    return path


def product(bucket, tags):
    return {"test_case": [bucket], "products": [{"tag": tags}]}


def create_file(tmp_path, relative_path, content=""):
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def ignored_files(selector, bucket, platform, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(SELECTOR_PATH), bucket, platform])
    selector.main()
    return {line.removeprefix("--ignore=") for line in capsys.readouterr().out.splitlines() if line}


def test_recipe_buckets_filter_tags_and_deduplicate(selector, tmp_path):
    path = write_recipe(
        tmp_path,
        [
            product("tests/unit_tests/**/*.py", ["latest", "legacy"]),
            product("tests/unit_tests/core/models/**/*.py", ["latest"]),
            product("tests/unit_tests/models/**/*.py", ["legacy"]),
            product("tests/unit_tests/core/models/**/*.py", ["latest"]),
        ],
    )

    assert selector.get_test_cases(path) == [
        "tests/unit_tests/**/*.py",
        "tests/unit_tests/core/models/**/*.py",
    ]
    assert selector.get_test_cases(path, "legacy") == [
        "tests/unit_tests/**/*.py",
        "tests/unit_tests/models/**/*.py",
    ]


@pytest.mark.parametrize("tag", ["latest", "legacy"])
def test_catchall_only_excludes_active_suite_buckets(selector, tmp_path, monkeypatch, capsys, tag):
    latest = "tests/unit_tests/core/models/test_gpt.py"
    legacy = "tests/unit_tests/models/test_gpt.py"
    remaining = "tests/unit_tests/test_external_tool.py"
    bucket = "tests/unit_tests/**/*.py"
    for path in (latest, legacy, remaining):
        create_file(tmp_path, path)
    write_recipe(
        tmp_path,
        [
            product(bucket, ["latest", "legacy"]),
            product(latest, ["latest"]),
            product(legacy, ["legacy"]),
        ],
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UNIT_TEST_TAG", tag)

    assert ignored_files(selector, bucket, "h100", monkeypatch, capsys) == {
        latest if tag == "latest" else legacy
    }


def test_gb200_marker_filter_preserves_helpers_and_conftest(
    selector, tmp_path, monkeypatch, capsys
):
    bucket = "tests/unit_tests/**/*.py"
    marked = "tests/unit_tests/core/models/test_marked.py"
    unmarked = "tests/unit_tests/core/models/test_unmarked.py"
    create_file(tmp_path, marked, "@pytest.mark.launch_on_gb200\ndef test_model(): pass\n")
    create_file(tmp_path, unmarked)
    create_file(tmp_path, "tests/unit_tests/core/models/conftest.py")
    create_file(tmp_path, "tests/unit_tests/core/models/helpers.py")
    write_recipe(tmp_path, [product(bucket, ["latest"])], platform="gb200")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UNIT_TEST_TAG", raising=False)

    assert ignored_files(selector, bucket, "gb200", monkeypatch, capsys) == {unmarked}


@pytest.mark.parametrize("platform", ["h100", "gb200"])
def test_latest_recipes_select_each_test_file_once(selector, monkeypatch, capsys, platform):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv("UNIT_TEST_TAG", "latest")
    recipe = REPO_ROOT / f"tests/test_utils/recipes/{platform}/unit-tests.yaml"
    buckets = selector.get_test_cases(recipe)
    counts = Counter()
    for bucket in buckets:
        ignored = ignored_files(selector, bucket, platform, monkeypatch, capsys)
        selected = {
            path
            for path in selector.expand_pattern(bucket)
            if Path(path).name.startswith("test_") and path not in ignored
        }
        assert selected, f"Empty latest {platform} bucket: {bucket}"
        counts.update(selected)

    expected = {
        str(path)
        for path in Path("tests/unit_tests").rglob("test_*.py")
        if platform != "gb200" or selector.file_has_marker(path, "launch_on_gb200")
    }
    assert set(counts) == expected
    assert all(count == 1 for count in counts.values()), counts
