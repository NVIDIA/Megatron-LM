# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dedicated runtime-CP tests must be scheduled once on both CI GPU platforms."""

import sys
from pathlib import Path

import pytest
import yaml

from tests.unit_tests import find_test_cases

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_BUCKET = "tests/unit_tests/context_parallel/runtime/**/*.py"


@pytest.mark.parametrize("platform,gpus", [("h100", 8), ("gb200", 4)])
def test_runtime_cp_bucket_has_exclusive_ownership(monkeypatch, capsys, platform, gpus):
    """Exercise real bucket expansion, child exclusions, and GB200 marker filtering."""
    monkeypatch.chdir(REPO_ROOT)
    recipe_path = Path(f"tests/test_utils/recipes/{platform}/unit-tests.yaml")
    recipe = yaml.safe_load(recipe_path.read_text())
    buckets = [name for product in recipe["products"] for name in product["test_case"]]
    assert buckets.count(RUNTIME_BUCKET) == 1
    assert recipe["spec"]["gpus"] == gpus
    runtime = next(p for p in recipe["products"] if p["test_case"] == [RUNTIME_BUCKET])
    assert any(
        "dev" in p["environment"] and "latest" in p["tag"] and "unit-tests" in p["scope"]
        for p in runtime["products"]
    )
    # Keep the selector's actual recipe data while avoiding a yq binary dependency
    # for this CPU infrastructure test. CI's yq expression reads this same list.
    monkeypatch.setattr(find_test_cases, "get_test_cases", lambda path: buckets)
    expected = {
        str(path) for path in Path("tests/unit_tests/context_parallel/runtime").glob("test_*.py")
    }
    assert expected
    owners = {path: [] for path in expected}
    for bucket in buckets:
        monkeypatch.setattr(sys, "argv", ["find_test_cases.py", bucket, platform])
        find_test_cases.main()
        ignored = {
            line.removeprefix("--ignore=")
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("--ignore=")
        }
        selected = set(find_test_cases.expand_pattern(bucket)) - ignored
        for path in expected & selected:
            owners[path].append(bucket)
    assert all(value == [RUNTIME_BUCKET] for value in owners.values()), owners
