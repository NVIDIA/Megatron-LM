# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
HELPER = ROOT / "tests/unit_tests/testmon_source_mapping.py"
SPEC = importlib.util.spec_from_file_location("unit_testmon_source_mapping", HELPER)
assert SPEC is not None and SPEC.loader is not None
mapping = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mapping)

MAPPED_BUCKET = "tests/unit_tests/distributed/mfsdp_v2/**/*.py"
UNMAPPED_BUCKET = "tests/unit_tests/models/**/*.py"


@pytest.fixture
def source_tree(tmp_path):
    root = tmp_path / "source"
    for name in ("tests/test_utils/recipes/h100/unit-tests.yaml",
                 "tests/test_utils/recipes/gb200/unit-tests.yaml"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        platform = "dgx_h100" if "/h100/" in name else "dgx_gb200"
        path.write_text(f"spec:\n  platforms: {platform}\nproducts: []\n")
    (root / mapping.SOURCE_MAPPING_FILE).parent.mkdir(parents=True, exist_ok=True)
    (root / mapping.SOURCE_MAPPING_FILE).write_text("mappings:\n")
    return root


def _write_mapping(root, mappings):
    (root / mapping.SOURCE_MAPPING_FILE).write_text(
        yaml.dump({"mappings": mappings}, default_flow_style=False)
    )


def _make_mapped_tree(source_tree, *, bucket=MAPPED_BUCKET, platform="dgx_h100"):
    _write_mapping(
        source_tree,
        [
            {
                "source_dirs": ["megatron/core/mapped"],
                "test_buckets": {platform: [bucket]},
            }
        ],
    )
    return source_tree


# ---------------------------------------------------------------------------
# Core: forced_full_buckets
# ---------------------------------------------------------------------------


def test_changed_file_under_mapped_dir_forces_bucket(source_tree):
    _make_mapped_tree(source_tree)
    result = mapping.forced_full_buckets(
        source_tree,
        ["megatron/core/mapped/module.py"],
        "dgx_h100",
    )
    assert result == [MAPPED_BUCKET]


def test_changed_file_in_subdir_forces_bucket(source_tree):
    _make_mapped_tree(source_tree)
    result = mapping.forced_full_buckets(
        source_tree,
        ["megatron/core/mapped/sub/nested.c"],
        "dgx_h100",
    )
    assert result == [MAPPED_BUCKET]


def test_no_changed_files_returns_empty(source_tree):
    _make_mapped_tree(source_tree)
    assert mapping.forced_full_buckets(source_tree, [], "dgx_h100") == []


def test_unrelated_change_returns_empty(source_tree):
    _make_mapped_tree(source_tree)
    result = mapping.forced_full_buckets(
        source_tree,
        ["megatron/core/other/file.py"],
        "dgx_h100",
    )
    assert result == []


def test_prefix_match_requires_slash_boundary(source_tree):
    _make_mapped_tree(source_tree)
    result = mapping.forced_full_buckets(
        source_tree,
        ["megatron/core/mapped_extra/file.py"],
        "dgx_h100",
    )
    assert result == []


def test_wrong_platform_returns_empty(source_tree):
    _make_mapped_tree(source_tree, platform="dgx_h100")
    result = mapping.forced_full_buckets(
        source_tree,
        ["megatron/core/mapped/module.py"],
        "dgx_gb200",
    )
    assert result == []


def test_unknown_platform_returns_empty(source_tree):
    _make_mapped_tree(source_tree)
    assert mapping.forced_full_buckets(
        source_tree,
        ["megatron/core/mapped/module.py"],
        "nonexistent_platform",
    ) == []


def test_no_mapping_file_returns_empty(source_tree):
    (source_tree / mapping.SOURCE_MAPPING_FILE).unlink()
    result = mapping.forced_full_buckets(
        source_tree,
        ["megatron/core/mapped/module.py"],
        "dgx_h100",
    )
    assert result == []


def test_empty_mapping_returns_empty(source_tree):
    result = mapping.forced_full_buckets(
        source_tree,
        ["megatron/core/mapped/module.py"],
        "dgx_h100",
    )
    assert result == []


def test_multiple_sources_and_platforms(source_tree):
    _write_mapping(
        source_tree,
        [
            {
                "source_dirs": ["src/a", "src/b"],
                "test_buckets": {
                    "dgx_h100": [MAPPED_BUCKET],
                    "dgx_gb200": [UNMAPPED_BUCKET],
                },
            }
        ],
    )
    assert mapping.forced_full_buckets(
        source_tree, ["src/a/x.py"], "dgx_h100"
    ) == [MAPPED_BUCKET]
    assert mapping.forced_full_buckets(
        source_tree, ["src/b/y.py"], "dgx_gb200"
    ) == [UNMAPPED_BUCKET]
    assert mapping.forced_full_buckets(
        source_tree, ["src/a/x.py"], "dgx_gb200"
    ) == [UNMAPPED_BUCKET]


def test_additive_duplicate_rules(source_tree):
    _write_mapping(
        source_tree,
        [
            {
                "source_dirs": ["src/a"],
                "test_buckets": {"dgx_h100": [MAPPED_BUCKET]},
            },
            {
                "source_dirs": ["src/b"],
                "test_buckets": {"dgx_h100": [UNMAPPED_BUCKET]},
            },
        ],
    )
    result = mapping.forced_full_buckets(
        source_tree,
        ["src/a/x.py", "src/b/y.py"],
        "dgx_h100",
    )
    assert result == sorted([MAPPED_BUCKET, UNMAPPED_BUCKET])


def test_multiple_changed_files_single_bucket(source_tree):
    _make_mapped_tree(source_tree)
    result = mapping.forced_full_buckets(
        source_tree,
        ["megatron/core/mapped/a.py", "megatron/core/mapped/b.py", "unrelated/c.py"],
        "dgx_h100",
    )
    assert result == [MAPPED_BUCKET]


def test_empty_bucket_list_returns_empty(source_tree):
    _write_mapping(
        source_tree,
        [
            {
                "source_dirs": ["megatron/core"],
                "test_buckets": {"dgx_h100": []},
            }
        ],
    )
    result = mapping.forced_full_buckets(
        source_tree, ["megatron/core/file.py"], "dgx_h100"
    )
    assert result == []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content,error_fragment",
    [
        ("not_a_mapping", "source mapping must be a YAML mapping"),
        ('mappings: "invalid"', "'mappings' must be a sequence"),
        ("mappings:\n  - not_a_dict", "must be a mapping"),
        ("mappings:\n  - source_dirs: []\n    test_buckets: {}", "non-empty list"),
        (
            "mappings:\n  - source_dirs: [src]\n    test_buckets:\n      fake_platform: []",
            "unknown platform",
        ),
    ],
)
def test_invalid_config(source_tree, content, error_fragment):
    (source_tree / mapping.SOURCE_MAPPING_FILE).write_text(content)
    with pytest.raises(ValueError, match=error_fragment):
        mapping.forced_full_buckets(
            source_tree, ["src/file.py"], "dgx_h100"
        )


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../escape", "src/../../escape", "src/\x00bad"],
)
def test_unsafe_paths(source_tree, path):
    _write_mapping(
        source_tree,
        [
            {
                "source_dirs": [path],
                "test_buckets": {"dgx_h100": [MAPPED_BUCKET]},
            }
        ],
    )
    with pytest.raises(ValueError, match="unsafe"):
        mapping.forced_full_buckets(
            source_tree, ["anything"], "dgx_h100"
        )


# ---------------------------------------------------------------------------
# Platform discovery
# ---------------------------------------------------------------------------


def test_discover_recipe_platforms(source_tree):
    platforms = mapping._discover_recipe_platforms(source_tree)
    assert platforms == frozenset({"dgx_h100", "dgx_gb200"})


def test_discover_recipe_platforms_empty(tmp_path):
    with pytest.raises(ValueError, match="no unit-test recipe platforms"):
        mapping._discover_recipe_platforms(tmp_path)


# ---------------------------------------------------------------------------
# YAML fallback parser
# ---------------------------------------------------------------------------


def test_parse_mapping_yaml_roundtrip():
    text = (ROOT / mapping.SOURCE_MAPPING_FILE).read_text()
    from_yaml = yaml.safe_load(text)
    from_fallback = mapping._parse_mapping_yaml(text)
    assert from_fallback == from_yaml
