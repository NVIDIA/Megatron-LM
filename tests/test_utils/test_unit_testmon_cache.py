# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

testmon_db = pytest.importorskip("testmon.db", reason="requires the testmon dependency group")
DB = testmon_db.DB

ROOT = Path(__file__).parents[2]
HELPER = ROOT / "tests/unit_tests/testmon_cache.py"
SPEC = importlib.util.spec_from_file_location("unit_testmon_cache", HELPER)
assert SPEC is not None and SPEC.loader is not None
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)
IMAGE_ID = "sha256:" + "a" * 64
BUCKET = "tests/unit_tests/pipeline_parallel/**/*.py"
MAPPED_BUCKET = "tests/unit_tests/distributed/mfsdp_v2/**/*.py"
MAPPED_SOURCE = "megatron/core/distributed/fsdp/src/megatron_fsdp/experimental"


@pytest.fixture
def source_tree(tmp_path):
    root = tmp_path / "source"
    for name in (
        *cache.COMPATIBILITY_FILES,
        "docker/.ngc_version.dev",
        "docker/Dockerfile.ci.dev",
        ".dockerignore",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    (root / cache.SOURCE_MAPPING_FILE).write_bytes((ROOT / cache.SOURCE_MAPPING_FILE).read_bytes())
    return root


@pytest.fixture
def generation(tmp_path, source_tree):
    identity = cache.cache_identity(source_tree, BUCKET, "dgx_h100", IMAGE_ID)
    directory = tmp_path / "assets_dir/testmon"
    _create_generation(directory, identity)
    return directory, identity


def _create_generation(directory, identity):
    for phase in cache.PHASES:
        path = directory / phase / ".testmondata"
        path.parent.mkdir(parents=True, exist_ok=True)
        database = DB(str(path))
        database.con.close()
        cache.record_phase(directory, phase)
    cache.finalize(directory, identity, "b" * 40, "123-1")


@pytest.fixture
def mapped_source(source_tree):
    directory = source_tree / MAPPED_SOURCE
    directory.mkdir(parents=True)
    (directory / "module.py").write_text("def hook():\n    return True\n")
    (directory / "nested").mkdir()
    (directory / "nested/hooks.py").write_text("def nested_hook():\n    return False\n")
    return directory


@pytest.fixture
def mapped_generation(tmp_path, source_tree, mapped_source):
    identity = cache.cache_identity(source_tree, MAPPED_BUCKET, "dgx_h100", IMAGE_ID)
    directory = tmp_path / "assets_dir/testmon"
    _create_generation(directory, identity)
    return directory, identity


def _snapshot(directory):
    return {
        str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.rglob("*")
        if path.is_file()
    }


def _mapping_document(test_buckets, sources=(MAPPED_SOURCE,)):
    return {"mappings": [{"source_dirs": list(sources), "test_buckets": test_buckets}]}


def test_source_edits_preserve_identity(source_tree):
    before = cache.cache_identity(source_tree, BUCKET, "dgx_h100", IMAGE_ID)
    (source_tree / "megatron/core/ordinary.py").write_text("changed source")
    assert cache.cache_identity(source_tree, BUCKET, "dgx_h100", IMAGE_ID) == before


def test_configured_source_mappings_target_existing_recipe_buckets():
    document = yaml.safe_load((ROOT / cache.SOURCE_MAPPING_FILE).read_text())
    recipe_buckets = {}
    for platform in ("h100", "gb200"):
        recipe = yaml.safe_load(
            (ROOT / "tests/test_utils/recipes" / platform / "unit-tests.yaml").read_text()
        )
        recipe_buckets[f"dgx_{platform}"] = {
            bucket for product in recipe["products"] for bucket in product["test_case"]
        }
    assert document["mappings"]
    for mapping in document["mappings"]:
        sources = mapping["source_dirs"]
        platforms = mapping["test_buckets"]
        for source in sources:
            assert (ROOT / source).is_dir(), source
        assert set(platforms) <= recipe_buckets.keys(), sources
        for platform, buckets in platforms.items():
            assert set(buckets) <= recipe_buckets[platform], (sources, platform)


def test_unchanged_mapped_sources_accept_the_recorded_baseline(source_tree, mapped_generation):
    directory, producer = mapped_generation
    consumer = cache.cache_identity(source_tree, MAPPED_BUCKET, "dgx_h100", IMAGE_ID)
    expected_paths = [f"{MAPPED_SOURCE}/module.py", f"{MAPPED_SOURCE}/nested/hooks.py"]
    assert consumer["compatibility"]["source_inputs"] == {
        name: hashlib.sha256((source_tree / name).read_bytes()).hexdigest()
        for name in expected_paths
    }
    assert consumer == producer
    before = _snapshot(directory)
    cache.validate_cache(directory, consumer, producer["cache_prefix"] + "123-1")
    assert _snapshot(directory) == before


@pytest.mark.parametrize("change", ["modified", "added", "deleted", "renamed"])
def test_mapped_source_changes_reject_baseline_without_affecting_other_buckets(
    source_tree, mapped_source, mapped_generation, change
):
    directory, producer = mapped_generation
    unrelated_before = cache.cache_identity(source_tree, BUCKET, "dgx_h100", IMAGE_ID)
    source = mapped_source / "module.py"
    if change == "modified":
        source.write_text("def hook():\n    return False\n")
    elif change == "added":
        (mapped_source / "nested/added.py").write_text("def added_hook():\n    return True\n")
    elif change == "deleted":
        source.unlink()
    else:
        source.rename(mapped_source / "renamed.py")
    consumer = cache.cache_identity(source_tree, MAPPED_BUCKET, "dgx_h100", IMAGE_ID)
    assert consumer["cache_prefix"] == producer["cache_prefix"]
    assert consumer["compatibility"]["source_inputs"] != producer["compatibility"]["source_inputs"]
    before = _snapshot(directory)
    with pytest.raises(ValueError, match="mapped source"):
        cache.validate_cache(directory, consumer, producer["cache_prefix"] + "123-1")
    assert _snapshot(directory) == before
    assert cache.cache_identity(source_tree, BUCKET, "dgx_h100", IMAGE_ID) == unrelated_before


def test_unmapped_source_changes_preserve_mapped_bucket_identity(source_tree, mapped_generation):
    directory, producer = mapped_generation
    (source_tree / "megatron/core/ordinary.py").write_text("changed unrelated source")
    # A prefix match must not include a sibling directory with a similar name.
    sibling = source_tree / f"{MAPPED_SOURCE}_other"
    sibling.mkdir()
    (sibling / "module.py").write_text("changed neighboring source")
    consumer = cache.cache_identity(source_tree, MAPPED_BUCKET, "dgx_h100", IMAGE_ID)
    assert consumer == producer
    cache.validate_cache(directory, consumer, producer["cache_prefix"] + "123-1")


@pytest.mark.parametrize(
    "platform,bucket,invalidated",
    [
        ("dgx_h100", MAPPED_BUCKET, True),
        ("dgx_gb200", "tests/unit_tests/**/*.py", False),
        ("dgx_h100", "tests/unit_tests/**/*.py", False),
        ("dgx_gb200", "tests/unit_tests/generalized_tensor_parallel/**/*.py", False),
    ],
)
def test_mapped_source_change_only_invalidates_platform_bucket_owning_tests(
    tmp_path, source_tree, mapped_source, platform, bucket, invalidated
):
    producer = cache.cache_identity(source_tree, bucket, platform, IMAGE_ID)
    directory = tmp_path / "assets_dir/testmon"
    _create_generation(directory, producer)
    (mapped_source / "module.py").write_text("def hook():\n    return False\n")
    consumer = cache.cache_identity(source_tree, bucket, platform, IMAGE_ID)
    assert consumer["cache_prefix"] == producer["cache_prefix"]
    before = _snapshot(directory)
    if invalidated:
        assert producer["compatibility"]["source_inputs"]
        with pytest.raises(ValueError, match="mapped source"):
            cache.validate_cache(directory, consumer, producer["cache_prefix"] + "123-1")
    else:
        assert consumer == producer
        assert consumer["compatibility"]["source_inputs"] == {}
        cache.validate_cache(directory, consumer, producer["cache_prefix"] + "123-1")
    assert _snapshot(directory) == before


def test_gb200_source_mapping_can_be_enabled_explicitly(tmp_path, source_tree, mapped_source):
    bucket = "tests/unit_tests/**/*.py"
    (source_tree / cache.SOURCE_MAPPING_FILE).write_text(
        yaml.safe_dump(_mapping_document({"dgx_h100": [MAPPED_BUCKET], "dgx_gb200": [bucket]}))
    )
    producer = cache.cache_identity(source_tree, bucket, "dgx_gb200", IMAGE_ID)
    assert producer["compatibility"]["source_inputs"]
    directory = tmp_path / "assets_dir/testmon"
    _create_generation(directory, producer)
    (mapped_source / "module.py").write_text("def hook():\n    return False\n")
    consumer = cache.cache_identity(source_tree, bucket, "dgx_gb200", IMAGE_ID)
    with pytest.raises(ValueError, match="mapped source"):
        cache.validate_cache(directory, consumer, producer["cache_prefix"] + "123-1")


@pytest.mark.parametrize("changed_source", [0, 1], ids=["first-source", "second-source"])
def test_each_source_change_invalidates_all_configured_platform_buckets(
    tmp_path, source_tree, mapped_source, changed_source
):
    second_source = source_tree / "megatron/core/second_source"
    second_source.mkdir()
    (second_source / "hook.py").write_text("def second_hook():\n    return True\n")
    sources = [MAPPED_SOURCE, str(second_source.relative_to(source_tree))]
    platforms = {
        "dgx_h100": [MAPPED_BUCKET, BUCKET],
        "dgx_gb200": [
            "tests/unit_tests/**/*.py",
            "tests/unit_tests/generalized_tensor_parallel/**/*.py",
        ],
    }
    (source_tree / cache.SOURCE_MAPPING_FILE).write_text(
        yaml.safe_dump(_mapping_document(platforms, sources))
    )
    expected_paths = {
        f"{MAPPED_SOURCE}/module.py",
        f"{MAPPED_SOURCE}/nested/hooks.py",
        f"{sources[1]}/hook.py",
    }
    baselines = []
    for platform, buckets in platforms.items():
        for bucket in buckets:
            identity = cache.cache_identity(source_tree, bucket, platform, IMAGE_ID)
            assert set(identity["compatibility"]["source_inputs"]) == expected_paths
            directory = tmp_path / f"baseline-{len(baselines)}"
            _create_generation(directory, identity)
            baselines.append((platform, bucket, directory, identity))
    unrelated_bucket = "tests/unit_tests/tensor_parallel/**/*.py"
    unrelated_before = cache.cache_identity(source_tree, unrelated_bucket, "dgx_h100", IMAGE_ID)
    changed_file = [mapped_source / "module.py", second_source / "hook.py"][changed_source]
    changed_file.write_text("def changed_hook():\n    return False\n")
    for platform, bucket, directory, producer in baselines:
        consumer = cache.cache_identity(source_tree, bucket, platform, IMAGE_ID)
        assert consumer["cache_prefix"] == producer["cache_prefix"]
        before = _snapshot(directory)
        with pytest.raises(ValueError, match="mapped source"):
            cache.validate_cache(directory, consumer, producer["cache_prefix"] + "123-1")
        assert _snapshot(directory) == before
    assert (
        cache.cache_identity(source_tree, unrelated_bucket, "dgx_h100", IMAGE_ID)
        == unrelated_before
    )


def test_empty_platform_bucket_list_ignores_mapped_source_changes(source_tree, mapped_source):
    (source_tree / cache.SOURCE_MAPPING_FILE).write_text(
        yaml.safe_dump(_mapping_document({"dgx_h100": []}))
    )
    before = cache.cache_identity(source_tree, MAPPED_BUCKET, "dgx_h100", IMAGE_ID)
    (mapped_source / "module.py").write_text("def hook():\n    return False\n")
    after = cache.cache_identity(source_tree, MAPPED_BUCKET, "dgx_h100", IMAGE_ID)
    assert before == after
    assert after["compatibility"]["source_inputs"] == {}


@pytest.mark.parametrize("present", [False, True], ids=["missing-directory", "empty-directory"])
def test_new_source_after_empty_baseline_requires_full_bucket(tmp_path, source_tree, present):
    source = source_tree / MAPPED_SOURCE
    if present:
        source.mkdir(parents=True)
    producer = cache.cache_identity(source_tree, MAPPED_BUCKET, "dgx_h100", IMAGE_ID)
    assert producer["compatibility"]["source_inputs"] == {}
    directory = tmp_path / "assets_dir/testmon"
    _create_generation(directory, producer)
    cache.validate_cache(directory, producer, producer["cache_prefix"] + "123-1")
    source.mkdir(parents=True, exist_ok=True)
    (source / "new.py").write_text("def new_hook():\n    return True\n")
    consumer = cache.cache_identity(source_tree, MAPPED_BUCKET, "dgx_h100", IMAGE_ID)
    with pytest.raises(ValueError, match="mapped source"):
        cache.validate_cache(directory, consumer, producer["cache_prefix"] + "123-1")


def test_removing_entire_mapped_directory_rejects_baseline(
    source_tree, mapped_source, mapped_generation
):
    directory, producer = mapped_generation
    shutil.rmtree(mapped_source)
    consumer = cache.cache_identity(source_tree, MAPPED_BUCKET, "dgx_h100", IMAGE_ID)
    assert consumer["compatibility"]["source_inputs"] == {}
    with pytest.raises(ValueError, match="mapped source"):
        cache.validate_cache(directory, consumer, producer["cache_prefix"] + "123-1")


@pytest.mark.parametrize(
    "generated",
    ["__pycache__/module.cpython-312.pyc", ".pytest_cache/state", "module.pyc", "module.pyo"],
)
def test_generated_files_do_not_invalidate_mapped_source_baseline(
    source_tree, mapped_source, mapped_generation, generated
):
    directory, producer = mapped_generation
    path = mapped_source / generated
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"generated runtime artifact")
    consumer = cache.cache_identity(source_tree, MAPPED_BUCKET, "dgx_h100", IMAGE_ID)
    assert consumer == producer
    cache.validate_cache(directory, consumer, producer["cache_prefix"] + "123-1")


@pytest.mark.parametrize(
    "mapping",
    [
        None,
        [],
        {},
        {"mappings": None},
        {"mappings": {}},
        {"mappings": [], "unexpected": True},
        {"mappings": [None]},
        {"mappings": [{"source_dirs": [MAPPED_SOURCE]}]},
        {"mappings": [{"source_dirs": [MAPPED_SOURCE], "test_buckets": {}, "unexpected": True}]},
        {
            "mappings": [
                {"source_dir": MAPPED_SOURCE, "test_buckets": {"dgx_h100": [MAPPED_BUCKET]}}
            ]
        },
        {
            "mappings": [
                {"source_dirs": MAPPED_SOURCE, "test_buckets": {"dgx_h100": [MAPPED_BUCKET]}}
            ]
        },
        {"mappings": [{"source_dirs": None, "test_buckets": {"dgx_h100": [MAPPED_BUCKET]}}]},
        _mapping_document([]),
        _mapping_document(None),
        _mapping_document({}),
        _mapping_document({"unsupported_platform": [MAPPED_BUCKET]}),
        _mapping_document({"dgx_h100": None}),
        _mapping_document({"dgx_h100": MAPPED_BUCKET}),
        _mapping_document({"dgx_h100": [None]}),
        _mapping_document({"dgx_h100": ["outside/tests.py"]}),
        _mapping_document({"dgx_h100": [MAPPED_BUCKET]}, sources=[]),
        _mapping_document({"dgx_h100": [MAPPED_BUCKET]}, sources=[None]),
        _mapping_document({"dgx_h100": [MAPPED_BUCKET]}, sources=[MAPPED_SOURCE, 1]),
        _mapping_document({"dgx_h100": [MAPPED_BUCKET]}, sources=["/absolute/source"]),
        _mapping_document({"dgx_h100": [MAPPED_BUCKET]}, sources=["../outside"]),
        _mapping_document({"dgx_h100": [MAPPED_BUCKET]}, sources=["source/../outside"]),
        _mapping_document({"dgx_h100": [MAPPED_BUCKET]}, sources=["./source"]),
        _mapping_document({"dgx_h100": [MAPPED_BUCKET]}, sources=["source//nested"]),
        _mapping_document({"dgx_h100": [MAPPED_BUCKET]}, sources=["source/**/*.py"]),
    ],
)
def test_invalid_source_mapping_rejects_identity(source_tree, mapping):
    (source_tree / cache.SOURCE_MAPPING_FILE).write_text(yaml.safe_dump(mapping))
    with pytest.raises(ValueError):
        cache.cache_identity(source_tree, MAPPED_BUCKET, "dgx_h100", IMAGE_ID)


@pytest.mark.parametrize(
    "mapping",
    ["mappings: [", "mappings: !unsupported []", "!!python/object/apply:builtins.eval ['1 + 1']"],
)
def test_invalid_or_unsafe_yaml_rejects_identity(source_tree, mapping):
    (source_tree / cache.SOURCE_MAPPING_FILE).write_text(mapping)
    with pytest.raises(ValueError):
        cache.cache_identity(source_tree, MAPPED_BUCKET, "dgx_h100", IMAGE_ID)


def test_repeated_sources_merge_targets_without_duplicates_or_cross_source_leaks(
    source_tree, mapped_source
):
    second_source = source_tree / "megatron/core/second_source"
    second_source.mkdir()
    (second_source / "hook.py").write_text("def second_hook():\n    return True\n")
    second_source_name = str(second_source.relative_to(source_tree))
    gb200_bucket = "tests/unit_tests/**/*.py"
    document = _mapping_document(
        {"dgx_h100": [MAPPED_BUCKET, MAPPED_BUCKET]},
        sources=[MAPPED_SOURCE, second_source_name, MAPPED_SOURCE],
    )
    document["mappings"].extend(
        [
            _mapping_document(
                {"dgx_h100": [BUCKET, MAPPED_BUCKET], "dgx_gb200": [gb200_bucket, gb200_bucket]}
            )["mappings"][0],
            _mapping_document({"dgx_h100": []})["mappings"][0],
        ]
    )
    (source_tree / cache.SOURCE_MAPPING_FILE).write_text(yaml.safe_dump(document))
    mapping = cache._source_mapping(source_tree)
    assert set(mapping) == {MAPPED_SOURCE, second_source_name}
    assert set(mapping[MAPPED_SOURCE]["dgx_h100"]) == {MAPPED_BUCKET, BUCKET}
    assert len(mapping[MAPPED_SOURCE]["dgx_h100"]) == 2
    assert mapping[MAPPED_SOURCE]["dgx_gb200"] == [gb200_bucket]
    assert mapping[second_source_name] == {"dgx_h100": [MAPPED_BUCKET]}
    targets = [("dgx_h100", MAPPED_BUCKET), ("dgx_h100", BUCKET), ("dgx_gb200", gb200_bucket)]
    before = {
        target: cache.cache_identity(source_tree, target[1], target[0], IMAGE_ID)
        for target in targets
    }
    first_source_paths = {f"{MAPPED_SOURCE}/module.py", f"{MAPPED_SOURCE}/nested/hooks.py"}
    assert set(before[targets[0]]["compatibility"]["source_inputs"]) == first_source_paths | {
        f"{second_source_name}/hook.py"
    }
    for target in targets[1:]:
        assert set(before[target]["compatibility"]["source_inputs"]) == first_source_paths
    (second_source / "hook.py").write_text("def second_hook():\n    return False\n")
    after_second = {
        target: cache.cache_identity(source_tree, target[1], target[0], IMAGE_ID)
        for target in targets
    }
    assert after_second[targets[0]] != before[targets[0]]
    for target in targets[1:]:
        assert after_second[target] == before[target]
    (mapped_source / "module.py").write_text("def first_hook():\n    return False\n")
    for platform, bucket in targets:
        assert (
            cache.cache_identity(source_tree, bucket, platform, IMAGE_ID)
            != after_second[(platform, bucket)]
        )


def test_missing_yaml_dependency_rejects_identity(source_tree, monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(ValueError):
        cache.cache_identity(source_tree, MAPPED_BUCKET, "dgx_h100", IMAGE_ID)


def test_baseline_validation_and_finalization_do_not_require_yaml(mapped_generation, monkeypatch):
    directory, identity = mapped_generation
    monkeypatch.setitem(sys.modules, "yaml", None)
    before = _snapshot(directory)
    cache.validate_cache(directory, identity, identity["cache_prefix"] + "123-1")
    assert _snapshot(directory) == before
    manifest = cache.finalize(directory, identity, "b" * 40, "456-1")
    assert manifest["generation"] == "456-1"
    cache.validate_cache(directory, identity, identity["cache_prefix"] + "456-1")


def test_mapping_can_target_multiple_buckets_including_single_file(source_tree, mapped_source):
    single_file_bucket = "tests/unit_tests/distributed/mfsdp_v2/test_hooks.py"
    (source_tree / cache.SOURCE_MAPPING_FILE).write_text(
        yaml.safe_dump(_mapping_document({"dgx_h100": [MAPPED_BUCKET, single_file_bucket]}))
    )
    wildcard = cache.cache_identity(source_tree, MAPPED_BUCKET, "dgx_h100", IMAGE_ID)
    single_file = cache.cache_identity(source_tree, single_file_bucket, "dgx_h100", IMAGE_ID)
    assert (
        single_file["compatibility"]["source_inputs"] == wildcard["compatibility"]["source_inputs"]
    )
    assert single_file["compatibility"]["source_inputs"]


@pytest.mark.parametrize("kind", ["root", "ancestor", "file", "directory"])
def test_symlinks_in_mapped_sources_reject_identity(source_tree, mapped_source, tmp_path, kind):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "hook.py").write_text("def hook():\n    return True\n")
    if kind in {"root", "ancestor"}:
        link = mapped_source if kind == "root" else mapped_source.parent
        shutil.rmtree(link)
        link.symlink_to(outside, target_is_directory=True)
    elif kind == "file":
        (mapped_source / "linked.py").symlink_to(outside / "hook.py")
    else:
        (mapped_source / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises((OSError, ValueError)):
        cache.cache_identity(source_tree, MAPPED_BUCKET, "dgx_h100", IMAGE_ID)


def test_unreadable_mapped_source_rejects_identity(source_tree, mapped_source, monkeypatch):
    unreadable = mapped_source / "module.py"
    original_read_bytes = Path.read_bytes

    def read_bytes(path):
        if path == unreadable:
            raise PermissionError("mapped source cannot be read")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    with pytest.raises((OSError, ValueError)):
        cache.cache_identity(source_tree, MAPPED_BUCKET, "dgx_h100", IMAGE_ID)


@pytest.mark.parametrize(
    "changed",
    ["uv.lock", "docker/.ngc_version.dev", ".dockerignore", "tests/unit_tests/find_test_cases.py"],
)
def test_build_inputs_preserve_lookup_prefix_but_reject_restored_generation(
    source_tree, generation, changed
):
    directory, producer = generation
    (source_tree / changed).write_text("changed")
    consumer = cache.cache_identity(source_tree, BUCKET, "dgx_h100", IMAGE_ID)
    assert consumer["cache_prefix"] == producer["cache_prefix"]
    assert consumer["compatibility"] != producer["compatibility"]
    before = _snapshot(directory)
    with pytest.raises(ValueError, match="compatibility"):
        cache.validate_cache(directory, consumer, producer["cache_prefix"] + "123-1")
    assert _snapshot(directory) == before


def test_platform_and_bucket_are_isolated(source_tree):
    identities = [
        cache.cache_identity(source_tree, BUCKET, "dgx_h100", IMAGE_ID),
        cache.cache_identity(source_tree, BUCKET, "dgx_gb200", IMAGE_ID),
        cache.cache_identity(source_tree, "tests/unit_tests/other.py", "dgx_h100", IMAGE_ID),
    ]
    assert len({identity["cache_prefix"] for identity in identities}) == 3
    assert all(
        identity["cache_prefix"].startswith("unit-testmon-v1-main-") for identity in identities
    )


def test_runtime_tracks_normalized_exact_versions_and_duplicate_distributions(monkeypatch):
    monkeypatch.setattr(
        cache,
        "distributions",
        lambda: [
            SimpleNamespace(metadata={"Name": name}, version=value)
            for name, value in [
                ("torch", "2.10.0"),
                ("torch", "2.11.0"),
                ("Transformer_Engine_CU12", "2.5.0"),
                ("megatron-core", "dev"),
            ]
        ],
    )
    identity = cache.runtime_identity()
    assert identity["packages"] == [
        ["torch", "2.10.0"],
        ["torch", "2.11.0"],
        ["transformer-engine-cu12", "2.5.0"],
    ]
    assert identity["testmon"] == "2.2.0"
    assert identity["python"]


@pytest.mark.parametrize("consumer_image", [IMAGE_ID, "sha256:" + "c" * 64, "image:latest", None])
def test_valid_generation_accepts_optional_image_diagnostics_and_is_read_only(
    generation, source_tree, consumer_image
):
    directory, identity = generation
    if consumer_image is None:
        consumer = cache.cache_identity(source_tree, BUCKET, "dgx_h100")
        assert consumer["image_id"] == "unknown"
    else:
        consumer = cache.cache_identity(source_tree, BUCKET, "dgx_h100", consumer_image)
        assert consumer["image_id"] == consumer_image
    assert consumer["cache_prefix"] == identity["cache_prefix"]
    before = _snapshot(directory)
    manifest = cache.validate_cache(directory, consumer, identity["cache_prefix"] + "123-1")
    assert manifest["source_sha"] == "b" * 40
    assert manifest["image_id"] == IMAGE_ID
    assert manifest["identity"] == consumer["compatibility"]
    assert "image_id" not in manifest["identity"]
    for phase in cache.PHASES:
        cache.validate_phase(directory, phase)
    assert _snapshot(directory) == before


def test_run_and_attempt_generations_share_prefix_and_require_matching_key(generation):
    directory, identity = generation
    keys = []
    for generation_id in ("123-1", "123-2", "456-1"):
        cache.finalize(directory, identity, "b" * 40, generation_id)
        key = identity["cache_prefix"] + generation_id
        manifest = cache.validate_cache(directory, identity, key)
        assert manifest["generation"] == generation_id
        keys.append(key)
    assert len(set(keys)) == 3
    with pytest.raises(ValueError, match="generation"):
        cache.validate_cache(directory, identity, keys[0])


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "corrupt",
        "schema",
        "wal",
        "metadata",
        "unsupported-metadata-schema",
        "runtime",
        "checksum",
    ],
)
def test_invalid_phase_is_rejected_without_repair(generation, mutation):
    directory, _ = generation
    database = directory / "prod/.testmondata"
    metadata_path = directory / "prod/metadata.json"
    if mutation == "missing":
        database.unlink()
    elif mutation == "corrupt":
        database.write_bytes(b"not SQLite")
    elif mutation == "schema":
        connection = sqlite3.connect(database)
        connection.execute("PRAGMA user_version=13")
        connection.close()
    elif mutation == "wal":
        database.with_name(database.name + "-wal").write_bytes(b"unfinished transaction")
    elif mutation == "metadata":
        metadata_path.write_text("[]")
    else:
        metadata = json.loads(metadata_path.read_text())
        if mutation == "unsupported-metadata-schema":
            metadata["schema"] = 2
        elif mutation == "runtime":
            metadata["runtime"]["python"] = "0.0.0"
        else:
            metadata["database_sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata))
    before = _snapshot(directory)
    with pytest.raises((ValueError, OSError)):
        cache.validate_phase(directory, "prod")
    assert _snapshot(directory) == before


def test_manifest_rejects_wrong_key_and_incomplete_phase(generation):
    directory, identity = generation
    with pytest.raises(ValueError, match="generation"):
        cache.validate_cache(directory, identity, identity["cache_prefix"] + "999-1")
    (directory / "manifest.json").unlink()
    (directory / "experimental/metadata.json").unlink()
    with pytest.raises(OSError):
        cache.finalize(directory, identity, "c" * 40, "456-1")
    assert not (directory / "manifest.json").exists()


def test_manifest_rejects_unsupported_cache_schema(generation):
    directory, identity = generation
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["schema"] = 2
    path.write_text(json.dumps(manifest))
    before = _snapshot(directory)
    with pytest.raises(ValueError, match="compatibility"):
        cache.validate_cache(directory, identity, identity["cache_prefix"] + "123-1")
    assert _snapshot(directory) == before


def test_manifest_requires_timezone_for_age_reporting(generation):
    directory, identity = generation
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["created_at"] = "2026-09-14T00:00:00"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="timezone"):
        cache.validate_cache(directory, identity, identity["cache_prefix"] + "123-1")


def _action_script(name):
    action = yaml.safe_load((ROOT / ".github/actions/action.yml").read_text())
    return next(step["run"] for step in action["runs"]["steps"] if step["name"] == name)


@pytest.mark.parametrize("diagnostic", ["failed-inspection", "omitted-cli-option"])
def test_identity_without_image_diagnostics_preserves_usable_cache(
    generation, source_tree, tmp_path, diagnostic
):
    directory, producer = generation
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    output = tmp_path / "output"
    before = _snapshot(directory)
    script = (
        _action_script("Compute unit Testmon cache identity")
        if diagnostic == "failed-inspection"
        else 'python tests/unit_tests/testmon_cache.py identity --bucket "$BUCKET" '
        '--platform "$RECIPE_PLATFORM" --output "$RUNNER_TEMP/unit-testmon-identity.json" '
        '| tee -a "$GITHUB_OUTPUT"'
    )
    result = subprocess.run(
        ["bash", "-e", "-u", "-o", "pipefail"],
        input="\n".join(
            (
                # Do not inspect Docker or delete anything; use only the temporary source tree.
                'docker() { [[ "$1 $2" == "image inspect" ]]; return 1; }',
                'sudo() { [[ "$*" == "rm -rf -- assets_dir/testmon" ]]; }',
                'python() { [[ "$1" == "tests/unit_tests/testmon_cache.py" ]]; '
                'shift; "$TEST_PYTHON" "$TESTMON_HELPER" "$@"; }',
                'uv() { [[ "$1 $2 $3 $4 $5" == '
                '"run --no-project --with pyyaml==6.0.3 python" ]] || return; '
                'shift 5; python "$@"; }',
                script,
            )
        ),
        cwd=source_tree,
        env={
            **os.environ,
            "TEST_PYTHON": sys.executable,
            "TESTMON_HELPER": str(HELPER),
            "TARGET_BRANCH": "main",
            "SUITE_TAG": "latest",
            "BUCKET": BUCKET,
            "RECIPE_PLATFORM": "dgx_h100",
            "CONTAINER_IMAGE": "image:latest",
            "RUNNER_TEMP": str(runtime_dir),
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    consumer = json.loads((runtime_dir / "unit-testmon-identity.json").read_text())
    assert consumer["image_id"] == "unknown"
    assert consumer["cache_prefix"] == producer["cache_prefix"]
    assert consumer["compatibility"] == producer["compatibility"]
    assert output.read_text().strip() == f"cache_prefix={producer['cache_prefix']}"
    manifest = cache.validate_cache(directory, consumer, producer["cache_prefix"] + "123-1")
    assert manifest["source_sha"] == "b" * 40
    assert _snapshot(directory) == before


@pytest.mark.parametrize(
    "mode,publication,expected",
    [
        ("baseline", "failure", "false"),
        ("baseline", "skipped", "false"),
        ("baseline", "success", "true"),
        ("enforce", "skipped", "true"),
    ],
)
def test_producer_result_requires_cache_publication(mode, publication, expected):
    script = _action_script("Check result")
    script = script[script.index('EXIT_CODE="${MAIN_EXIT_CODE') :]
    script = script.split('if [[ "$IS_SUCCESS" == "false"', 1)[0]
    result = subprocess.run(
        ["bash", "-e", "-u", "-o", "pipefail", "-c", script + 'printf "%s" "$IS_SUCCESS"'],
        env={
            **os.environ,
            "MAIN_EXIT_CODE": "0",
            "MAIN_CONCLUSION": "success",
            "TESTMON_MODE": mode,
            "CACHE_PUBLICATION": publication,
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == expected


@pytest.mark.parametrize(
    "restore",
    [
        "valid",
        "different-image",
        "missing-image",
        "changed-config",
        "mapped-source-change",
        "miss",
        "error",
        "invalid",
        "identity-error",
    ],
)
def test_action_resolver_uses_prefix_restores_and_never_bootstraps(
    generation, source_tree, mapped_source, tmp_path, restore
):
    directory, identity = generation
    if restore == "different-image":
        identity = cache.cache_identity(source_tree, BUCKET, "dgx_h100", "sha256:" + "c" * 64)
    elif restore == "missing-image":
        identity = cache.cache_identity(source_tree, BUCKET, "dgx_h100")
    elif restore == "changed-config":
        (source_tree / "tests/unit_tests/find_test_cases.py").write_text("changed")
        identity = cache.cache_identity(source_tree, BUCKET, "dgx_h100", IMAGE_ID)
    elif restore == "mapped-source-change":
        identity = cache.cache_identity(source_tree, MAPPED_BUCKET, "dgx_h100", IMAGE_ID)
        _create_generation(directory, identity)
        (mapped_source / "module.py").write_text("def hook():\n    return False\n")
        identity = cache.cache_identity(source_tree, MAPPED_BUCKET, "dgx_h100", IMAGE_ID)
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    identity_file = runtime_dir / "unit-testmon-identity.json"
    identity_file.write_text(json.dumps(identity))
    helper = tmp_path / "tests/unit_tests/testmon_cache.py"
    helper.parent.mkdir(parents=True)
    shutil.copy2(HELPER, helper)
    if restore == "miss":
        shutil.rmtree(directory)
    elif restore == "invalid":
        (directory / "manifest.json").write_text("[]")
    before = _snapshot(directory) if directory.exists() else {}
    output = tmp_path / "output"
    summary = tmp_path / "summary"
    result = subprocess.run(
        ["bash", "-e", "-u", "-o", "pipefail", "-c", _action_script("Resolve unit Testmon mode")],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
            "REQUESTED_MODE": "enforce",
            "IDENTITY_OUTCOME": "failure" if restore == "identity-error" else "success",
            "RESTORE_OUTCOME": "failure" if restore == "error" else "success",
            "MATCHED_KEY": "" if restore == "miss" else identity["cache_prefix"] + "123-1",
            "CACHE_HIT": "false",
            "RUNNER_TEMP": str(runtime_dir),
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    valid = restore in {"valid", "different-image", "missing-image"}
    assert output.read_text().strip() == ("mode=enforce" if valid else "mode=full")
    after = _snapshot(directory)
    after.pop("summary.md", None)
    assert after == before
    if valid:
        assert "b" * 40 in summary.read_text()
    else:
        assert "without recording or saving" in summary.read_text()


@pytest.mark.parametrize(
    "override,allowed",
    [
        ({}, True),
        ({"SOURCE_REF": "refs/heads/pull-request/6934"}, False),
        ({"SOURCE_REPOSITORY": "fork/Megatron-LM"}, False),
        ({"SOURCE_EVENT": "pull_request"}, False),
        ({"REQUESTED_SHA": "c" * 40}, False),
    ],
)
def test_action_baseline_guard_rejects_untrusted_producers(tmp_path, override, allowed):
    fake_git = tmp_path / "git"
    fake_git.write_text("#!/bin/sh\nprintf '%s\\n' \"$SOURCE_SHA\"\n")
    fake_git.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-e",
            "-u",
            "-o",
            "pipefail",
            "-c",
            _action_script("Validate unit Testmon baseline producer"),
        ],
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "SOURCE_REPOSITORY": "NVIDIA/Megatron-LM",
            "SOURCE_REF": "refs/heads/main",
            "SOURCE_EVENT": "schedule",
            "SOURCE_SHA": "b" * 40,
            "REQUESTED_SHA": "b" * 40,
            "TARGET_BRANCH": "main",
            "SUITE_TAG": "latest",
            **override,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is allowed
