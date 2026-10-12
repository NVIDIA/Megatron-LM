# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Validate recipe scheduling without importing GPU test modules."""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
GDN_BUCKETS = tuple(
    f"tests/unit_tests/ssm/{filename}"
    for filename in (
        "test_gated_delta_net.py",
        "test_gated_delta_net2.py",
        "test_gated_delta_net_fla_correctness.py",
        "test_gated_delta_net_te_correctness.py",
        "test_gated_delta_net2_correctness.py",
    )
)


@pytest.mark.parametrize("platform", ["h100", "gb200"])
def test_gdn_shards_have_exclusive_h100_buckets(platform, monkeypatch, capsys):
    """Keep split files out of overlapping SSM/global buckets and GB200 jobs."""
    monkeypatch.chdir(ROOT)
    spec = importlib.util.spec_from_file_location(
        "gdn_bucket_finder", ROOT / "tests/unit_tests/find_test_cases.py"
    )
    assert spec is not None and spec.loader is not None
    finder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(finder)
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

    assert owners == {path: [path] if platform == "h100" else [] for path in GDN_BUCKETS}
    if platform == "h100":
        products = {
            case: product["products"]
            for product in recipe["products"]
            for case in product["test_case"]
        }
        for path in GDN_BUCKETS[1:]:
            assert products[path] == products[GDN_BUCKETS[0]]
