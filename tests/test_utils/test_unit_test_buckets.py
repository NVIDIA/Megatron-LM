# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Directory moves must preserve CI bucket ownership and collection order."""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "find_test_cases", REPO_ROOT / "tests/unit_tests/find_test_cases.py"
)
find_test_cases = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(find_test_cases)
RUNTIME_BUCKET = "tests/unit_tests/context_parallel/runtime/**/*.py"
ROOT_BUCKET = "tests/unit_tests/**/*.py"


@pytest.mark.parametrize("platform", ["h100", "gb200"])
def test_recipe_buckets_partition_all_eligible_test_files(platform):
    """Every eligible file has exactly one owner, with no duplicate manifest entries."""
    recipe = REPO_ROOT / f"tests/test_utils/recipes/{platform}/unit-tests.yaml"
    buckets = find_test_cases.get_test_cases(recipe, "latest")
    manifest = yaml.safe_load((REPO_ROOT / find_test_cases.BUCKET_MANIFEST).read_text())[platform]
    assert set(manifest) == set(buckets)
    explicit = Counter(path for paths in manifest.values() for path in paths)
    assert all(count == 1 for count in explicit.values()), explicit

    exempt = {
        path
        for bucket in find_test_cases.PLATFORM_MARKER_EXEMPTIONS.get(platform, ())
        for path in manifest[bucket]
    }
    marker = find_test_cases.PLATFORM_MARKERS.get(platform)
    excluded = find_test_cases.PLATFORM_EXCLUSIONS.get(platform, ())
    eligible = {
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "tests/unit_tests").rglob("test_*.py")
        if path.is_file()
        and str(path.relative_to(REPO_ROOT)) not in excluded
        and (
            marker is None
            or marker in path.read_text()
            or str(path.relative_to(REPO_ROOT)) in exempt
        )
    }
    resolved = Counter(
        path
        for bucket in buckets
        for path in find_test_cases.resolve_bucket(bucket, platform, REPO_ROOT)
    )
    assert resolved == Counter({path: 1 for path in eligible})


@pytest.mark.parametrize("platform,gpus", [("h100", 8), ("gb200", 4)])
def test_runtime_cp_bucket_has_exclusive_ownership(monkeypatch, capsys, platform, gpus):
    """Relocated runtime-CP tests keep their dedicated job on both platforms."""
    monkeypatch.chdir(REPO_ROOT)
    recipe_path = Path(f"tests/test_utils/recipes/{platform}/unit-tests.yaml")
    recipe = yaml.safe_load(recipe_path.read_text())
    buckets = find_test_cases.get_test_cases(recipe_path, "latest")
    assert buckets.count(RUNTIME_BUCKET) == 1
    assert recipe["spec"]["gpus"] == gpus
    runtime = next(p for p in recipe["products"] if p["test_case"] == [RUNTIME_BUCKET])
    assert any(
        "dev" in p["environment"] and "latest" in p["tag"] and "unit-tests" in p["scope"]
        for p in runtime["products"]
    )
    expected = {
        str(path)
        for path in Path("tests/unit_tests/core/context_parallel").glob("test_*.py")
        if path.name != "test_layout.py"
    }
    assert expected
    owners = {path: [] for path in expected}
    for bucket in buckets:
        selected = set(find_test_cases.resolve_bucket(bucket, platform))
        for path in expected & selected:
            owners[path].append(bucket)
    assert all(value == [RUNTIME_BUCKET] for value in owners.values()), owners
    monkeypatch.setattr(sys, "argv", ["find_test_cases.py", RUNTIME_BUCKET, platform, "--targets"])
    find_test_cases.main()
    assert set(capsys.readouterr().out.splitlines()) == expected


def test_entrypoint_imports_stay_in_the_data_bucket():
    """Collecting entrypoints in the root job changes rank-dependent logging state."""
    path = "tests/unit_tests/test_packed_pretrain_process_groups.py"
    assert path in find_test_cases.resolve_bucket(
        "tests/unit_tests/data/**/*.py", "h100", REPO_ROOT
    )
    assert path not in find_test_cases.resolve_bucket(ROOT_BUCKET, "h100", REPO_ROOT)


def test_gtp_fp32_tests_preserve_original_catch_all_isolation():
    """The FP32 tests retain their original catch-all job and its eight-rank allocation."""
    path = "tests/unit_tests/core/tensor_parallel/test_gtp_fp32_accumulation_reduce_scatter.py"
    gtp = find_test_cases.resolve_bucket(
        "tests/unit_tests/generalized_tensor_parallel/**/*.py", "h100", REPO_ROOT
    )
    assert path not in gtp
    assert path in find_test_cases.resolve_bucket(ROOT_BUCKET, "h100", REPO_ROOT)


def test_gb200_fp8_profile_bucket_does_not_require_the_generic_marker():
    """The specialized FP8 launcher selects its own profiles without the generic marker."""
    assert find_test_cases.resolve_bucket(
        "tests/unit_tests/inference/test_batch_invariant_fp8.py", "gb200", REPO_ROOT
    ) == ["tests/unit_tests/core/inference/test_batch_invariant_fp8.py"]


def test_catch_all_preserves_order_and_discovers_unassigned_files(tmp_path):
    """New files join the catch-all without duplicating explicitly owned targets."""
    first = "tests/unit_tests/training/test_first.py"
    second = "tests/unit_tests/core/test_second.py"
    dedicated = "tests/unit_tests/core/test_dedicated.py"
    new = "tests/unit_tests/test_new.py"
    for relative in (first, second, dedicated, new):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def test_example():\n    pass\n")
    manifest = {
        "h100": {ROOT_BUCKET: [first, second], "tests/unit_tests/old_location/**/*.py": [dedicated]}
    }
    (tmp_path / find_test_cases.BUCKET_MANIFEST).write_text(yaml.safe_dump(manifest))
    assert find_test_cases.resolve_bucket(ROOT_BUCKET, "h100", tmp_path) == [first, second, new]
    assert find_test_cases.resolve_bucket(
        "tests/unit_tests/old_location/**/*.py", "h100", tmp_path
    ) == [dedicated]


def test_missing_explicit_target_fails_instead_of_losing_coverage(tmp_path):
    manifest_path = tmp_path / find_test_cases.BUCKET_MANIFEST
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        yaml.safe_dump({"h100": {ROOT_BUCKET: ["tests/unit_tests/test_missing.py"]}})
    )
    with pytest.raises(ValueError, match="missing or invalid unit-test target"):
        find_test_cases.resolve_bucket(ROOT_BUCKET, "h100", tmp_path)
