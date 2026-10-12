# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Validate recipe scheduling without importing GPU test modules."""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
GDN_BUCKETS = {
    "tests/unit_tests/ssm/test_gated_delta_net.py": "gb200",
    "tests/unit_tests/ssm/test_gated_delta_net2.py": "h100",
}


@pytest.fixture
def finder(monkeypatch):
    """Load the CI file selector without importing the GPU test package."""
    monkeypatch.chdir(ROOT)
    spec = importlib.util.spec_from_file_location(
        "gdn_bucket_finder", ROOT / "tests/unit_tests/find_test_cases.py"
    )
    assert spec is not None and spec.loader is not None
    finder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(finder)
    return finder


@pytest.mark.parametrize("platform", ["h100", "gb200"])
def test_gdn_variants_have_exclusive_platform_buckets(platform, finder, monkeypatch, capsys):
    """Schedule each variant once on its intended platform, outside catch-alls."""
    recipe = yaml.safe_load(
        (ROOT / f"tests/test_utils/recipes/{platform}/unit-tests.yaml").read_text()
    )
    buckets = [case for product in recipe["products"] for case in product["test_case"]]
    monkeypatch.setattr(finder, "get_test_cases", lambda _: buckets)

    owners = {path: [] for path in GDN_BUCKETS}
    for bucket in buckets:
        candidates = set(finder.expand_pattern(bucket)) & owners.keys()
        if not candidates:
            continue
        monkeypatch.setattr(sys, "argv", ["find_test_cases.py", bucket, platform])
        finder.main()
        ignored = {line.removeprefix("--ignore=") for line in capsys.readouterr().out.splitlines()}
        for path in candidates - ignored:
            owners[path].append(bucket)

    assert owners == {
        path: [path] if platform == target else [] for path, target in GDN_BUCKETS.items()
    }
    products = {
        case: product["products"] for product in recipe["products"] for case in product["test_case"]
    }
    for path, target in GDN_BUCKETS.items():
        if platform != target:
            assert path not in products
            continue
        assert products[path] == [
            {
                "environment": ["lts", "dev"] if platform == "h100" else ["dev"],
                "tag": ["latest"],
                "scope": ["unit-tests"],
                "n_repeat": [1],
                "time_limit": [1800],
            }
        ]
    assert recipe["spec"]["gpus"] == (8 if platform == "h100" else 4)


def test_gdn_exclusion_preserves_other_gb200_marked_tests_on_h100(finder, monkeypatch, capsys):
    """GB200 opt-in remains additive for files other than the GDN-only bucket."""
    other = "tests/unit_tests/ssm/test_gdp_chunkwise_context_parallel.py"
    assert finder.file_has_marker(other, "launch_on_gb200")
    recipe = yaml.safe_load((ROOT / "tests/test_utils/recipes/h100/unit-tests.yaml").read_text())
    buckets = [case for product in recipe["products"] for case in product["test_case"]]
    monkeypatch.setattr(finder, "get_test_cases", lambda _: buckets)
    bucket = "tests/unit_tests/ssm/**/*.py"
    assert other in finder.expand_pattern(bucket)
    monkeypatch.setattr(sys, "argv", ["find_test_cases.py", bucket, "h100"])
    finder.main()
    ignored = {line.removeprefix("--ignore=") for line in capsys.readouterr().out.splitlines()}
    assert "tests/unit_tests/ssm/test_gated_delta_net.py" in ignored
    assert other not in ignored
