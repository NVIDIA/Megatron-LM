# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Versioned, read-only validation of main's unit Testmon cache generations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distributions, version
from pathlib import Path, PurePosixPath

SCHEMA = 1
TESTMON_VERSION = "2.2.0"
PHASES = ("prod", "experimental")
RECIPE_PLATFORMS = frozenset({"dgx_h100", "dgx_gb200"})
TRACKED_ENVIRONMENT_PACKAGES = frozenset(
    {"numpy", "pytest", "torch", "transformer-engine", "triton"}
)
TRACKED_ENVIRONMENT_PACKAGE_PREFIXES = ("transformer-engine-",)
SOURCE_MAPPING_FILE = "tests/unit_tests/testmon_source_mapping.yml"
COMPATIBILITY_FILES = (
    ".github/actions/action.yml",
    ".github/workflows/_build_ci_container.yml",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "megatron/core/__init__.py",
    "megatron/core/package_info.py",
    "tests/unit_tests/run_ci_test.sh",
    "tests/unit_tests/find_test_cases.py",
    "tests/unit_tests/testmon_selector.py",
    "tests/unit_tests/testmon_cache.py",
    SOURCE_MAPPING_FILE,
    "tests/test_utils/python_scripts/launch_nemo_run_workload.py",
    "tests/test_utils/python_scripts/recipe_parser.py",
    "tests/test_utils/python_scripts/download_unit_tests_dataset.py",
    "tests/test_utils/recipes/h100/unit-tests.yaml",
    "tests/test_utils/recipes/gb200/unit-tests.yaml",
)
COMPATIBILITY_GLOBS = ("docker/**/*", ".dockerignore", "tests/unit_tests/**/conftest.py")
DATABASE_TABLES = {
    "metadata",
    "environment",
    "test_execution",
    "file_fp",
    "test_execution_file_fp",
    "suite_execution_file_fsha",
}


def _normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def is_tracked_package(name: str) -> bool:
    """Return whether Testmon tracks this distribution as an environment dependency."""
    name = _normalized_name(name)
    return name in TRACKED_ENVIRONMENT_PACKAGES or name.startswith(
        TRACKED_ENVIRONMENT_PACKAGE_PREFIXES
    )


def runtime_identity() -> dict:
    """Describe the exact interpreter and tracked packages inside the test container."""
    packages = sorted(
        [_normalized_name(name), distribution.version]
        for distribution in distributions()
        if (name := distribution.metadata["Name"]) and is_tracked_package(name)
    )
    try:
        testmon_version = version("pytest-testmon")
    except PackageNotFoundError as error:
        raise ValueError("pytest-testmon is not installed") from error
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "testmon": testmon_version,
        "packages": packages,
    }


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return data


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_mapping_path(path: str) -> None:
    if (
        not isinstance(path, str)
        or not path
        or PurePosixPath(path).is_absolute()
        or str(PurePosixPath(path)) != path
        or ".." in PurePosixPath(path).parts
        or "\\" in path
        or any(character in path for character in "\r\n\0")
    ):
        raise ValueError(f"invalid Testmon mapping path: {path!r}")


def _source_mapping(root: Path) -> dict[str, dict[str, list[str]]]:
    # Only identity calculation needs YAML; restored-cache validation stays stdlib-only.
    try:
        import yaml
    except ImportError as error:
        raise ValueError("PyYAML is required to read Testmon source mappings") from error
    try:
        document = yaml.safe_load((root / SOURCE_MAPPING_FILE).read_text())
    except yaml.YAMLError as error:
        raise ValueError(f"invalid Testmon source mapping YAML: {error}") from error
    if (
        not isinstance(document, dict)
        or set(document) != {"mappings"}
        or not isinstance(document["mappings"], list)
    ):
        raise ValueError("expected a Testmon mapping document with a mappings list")
    mapping = {}
    for entry in document["mappings"]:
        if not isinstance(entry, dict) or set(entry) != {"source_dir", "test_buckets"}:
            raise ValueError("expected source_dir and test_buckets in each Testmon mapping")
        source = entry["source_dir"]
        platforms = entry["test_buckets"]
        _validate_mapping_path(source)
        if source == "." or any(character in source for character in "*?[]"):
            raise ValueError(f"expected a source directory in Testmon mapping: {source!r}")
        if source in mapping:
            raise ValueError(f"duplicate Testmon source directory: {source!r}")
        if not isinstance(platforms, dict) or not platforms:
            raise ValueError(f"expected recipe platforms for mapped source: {source!r}")
        for recipe_platform, buckets in platforms.items():
            if recipe_platform not in RECIPE_PLATFORMS:
                raise ValueError(f"unsupported mapped Testmon platform: {recipe_platform!r}")
            if not isinstance(buckets, list):
                raise ValueError(f"expected unit-test buckets for mapped source: {source!r}")
            for bucket in buckets:
                if not isinstance(bucket, str):
                    raise ValueError(f"invalid mapped unit-test bucket: {bucket!r}")
                _validate_mapping_path(bucket)
                if not bucket.startswith("tests/unit_tests/") or not bucket.endswith(".py"):
                    raise ValueError(f"invalid mapped unit-test bucket: {bucket!r}")
        mapping[source] = platforms
    return mapping


def _raise_walk_error(error: OSError) -> None:
    raise error


def _mapped_source_inputs(root: Path, bucket: str, recipe_platform: str) -> dict[str, str]:
    """Fingerprint mapped source trees independently of traced test dependencies."""
    inputs = {}
    for source, platforms in _source_mapping(root).items():
        if bucket not in platforms.get(recipe_platform, []):
            continue
        directory = root / source
        parent = root
        for part in PurePosixPath(source).parts:
            parent /= part
            if parent.is_symlink():
                raise ValueError(f"mapped source directory contains a symlink: {source}")
        try:
            directory.stat()
        except FileNotFoundError:
            # Removing a source tree must compare against its recorded files.
            continue
        if not directory.is_dir():
            raise ValueError(f"mapped source is not a directory: {source}")
        for current, directories, filenames in os.walk(directory, onerror=_raise_walk_error):
            directories[:] = sorted(
                name
                for name in directories
                if name not in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
            )
            for name in [*directories, *filenames]:
                path = Path(current) / name
                if path.suffix in {".pyc", ".pyo"}:
                    continue
                if path.is_symlink():
                    raise ValueError(f"mapped source contains a symlink: {path}")
                if name in filenames:
                    inputs[str(path.relative_to(root))] = _digest(path)
    return dict(sorted(inputs.items()))


def cache_identity(
    root: Path, bucket: str, recipe_platform: str, image_id: str = "unknown"
) -> dict:
    """Separate cache lookup from compatibility checks and diagnostic image identity."""
    if recipe_platform not in RECIPE_PLATFORMS:
        raise ValueError(f"unsupported Testmon platform: {recipe_platform}")
    if not bucket.startswith("tests/unit_tests/") or "\n" in bucket:
        raise ValueError("invalid unit-test bucket")
    paths = {root / path for path in COMPATIBILITY_FILES}
    paths.update(
        path for pattern in COMPATIBILITY_GLOBS for path in root.glob(pattern) if path.is_file()
    )
    inputs = {str(path.relative_to(root)): _digest(path) for path in sorted(paths)}
    contract = {
        "schema": SCHEMA,
        "testmon": TESTMON_VERSION,
        "platform": recipe_platform,
        "bucket": bucket,
        "environment": "dev",
        "tag": "latest",
        "inputs": inputs,
        "source_inputs": _mapped_source_inputs(root, bucket, recipe_platform),
    }
    bucket_hash = hashlib.sha256(bucket.encode()).hexdigest()[:16]
    return {
        "compatibility": contract,
        "image_id": image_id,
        "cache_prefix": f"unit-testmon-v{SCHEMA}-main-{recipe_platform}-{bucket_hash}-",
    }


def _database(cache_dir: Path, phase: str) -> Path:
    if phase not in PHASES:
        raise ValueError(f"unsupported Testmon phase: {phase}")
    return cache_dir / phase / ".testmondata"


def _validate_database(database: Path) -> None:
    # Immutable read-only connections cannot create journals or repair a restored database.
    wal = database.with_name(database.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError(f"uncheckpointed Testmon database: {database}")
    try:
        with closing(
            sqlite3.connect(database.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
        ) as connection:
            if connection.execute("PRAGMA user_version").fetchone() != (14,):
                raise ValueError(f"unsupported Testmon database schema: {database}")
            if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise ValueError(f"corrupt Testmon database: {database}")
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if not DATABASE_TABLES <= tables:
                raise ValueError(f"incomplete Testmon database schema: {database}")
    except sqlite3.Error as error:
        raise ValueError(f"invalid Testmon database {database}: {error}") from error


def record_phase(cache_dir: Path, phase: str) -> None:
    """Checkpoint and mark a successfully completed rank-zero baseline phase."""
    database = _database(cache_dir, phase)
    if not database.is_file():
        raise ValueError(f"missing Testmon database: {database}")
    try:
        with closing(sqlite3.connect(database)) as connection:
            if connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] != 0:
                raise ValueError(f"unable to checkpoint Testmon database: {database}")
    except sqlite3.Error as error:
        raise ValueError(f"unable to checkpoint Testmon database {database}: {error}") from error
    _validate_database(database)
    identity = runtime_identity()
    if identity["testmon"] != TESTMON_VERSION:
        raise ValueError("unexpected pytest-testmon version")
    _write_json(
        database.parent / "metadata.json",
        {
            "schema": SCHEMA,
            "phase": phase,
            "complete": True,
            "runtime": identity,
            "database_sha256": _digest(database),
        },
    )


def validate_phase(cache_dir: Path, phase: str, *, check_runtime: bool = True) -> None:
    """Validate a restored phase without changing its database or metadata."""
    database = _database(cache_dir, phase)
    metadata = _read_json(database.parent / "metadata.json")
    if (
        metadata.get("schema") != SCHEMA
        or metadata.get("phase") != phase
        or metadata.get("complete") is not True
    ):
        raise ValueError(f"incomplete Testmon phase: {phase}")
    runtime = metadata.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("testmon") != TESTMON_VERSION:
        raise ValueError(f"invalid Testmon runtime metadata: {phase}")
    if check_runtime and runtime != runtime_identity():
        raise ValueError(f"Testmon runtime environment changed: {phase}")
    _validate_database(database)
    if metadata.get("database_sha256") != _digest(database):
        raise ValueError(f"Testmon database checksum changed: {phase}")


def finalize(cache_dir: Path, identity: dict, source_sha: str, generation: str) -> dict:
    """Write the generation manifest only after both baseline phases succeeded."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha) or not re.fullmatch(
        r"[0-9]+-[0-9]+", generation
    ):
        raise ValueError("invalid baseline source or generation")
    for phase in PHASES:
        validate_phase(cache_dir, phase, check_runtime=False)
    manifest = {
        "schema": SCHEMA,
        "source_ref": "refs/heads/main",
        "source_sha": source_sha,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generation": generation,
        "identity": identity["compatibility"],
        "image_id": identity["image_id"],
    }
    _write_json(cache_dir / "manifest.json", manifest)
    return manifest


def validate_cache(cache_dir: Path, identity: dict, matched_key: str) -> dict:
    """Require a compatible completed main generation, including a prefix cache match."""
    manifest = _read_json(cache_dir / "manifest.json")
    generation = manifest.get("generation", "")
    if not isinstance(generation, str) or not re.fullmatch(r"[0-9]+-[0-9]+", generation):
        raise ValueError("invalid Testmon cache generation")
    if matched_key != identity["cache_prefix"] + generation:
        raise ValueError("restored Testmon key does not match its generation")
    if manifest.get("schema") != SCHEMA or manifest.get("identity") != identity["compatibility"]:
        recorded = manifest.get("identity")
        if isinstance(recorded, dict) and recorded.get("source_inputs", {}) != identity[
            "compatibility"
        ].get("source_inputs", {}):
            raise ValueError("Testmon mapped source files changed; full unit-test bucket required")
        raise ValueError("Testmon cache compatibility changed")
    if manifest.get("source_ref") != "refs/heads/main" or not re.fullmatch(
        r"[0-9a-f]{40}", str(manifest.get("source_sha", ""))
    ):
        raise ValueError("invalid Testmon baseline source")
    if not isinstance(manifest.get("created_at"), str):
        raise ValueError("missing Testmon baseline creation time")
    created_at = datetime.fromisoformat(manifest["created_at"])
    if created_at.tzinfo is None:
        raise ValueError("Testmon baseline creation time must include a timezone")
    for phase in PHASES:
        validate_phase(cache_dir, phase, check_runtime=False)
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Expose cache checks to the host; only identity calculation requires PyYAML."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    identity_parser = subparsers.add_parser("identity")
    identity_parser.add_argument("--root", type=Path, default=Path("."))
    identity_parser.add_argument("--bucket", required=True)
    identity_parser.add_argument("--platform", required=True)
    identity_parser.add_argument("--image-id", default="unknown")
    identity_parser.add_argument("--output", type=Path, required=True)
    for command in ("validate", "finalize"):
        child = subparsers.add_parser(command)
        child.add_argument("--cache-dir", type=Path, required=True)
        child.add_argument("--identity", type=Path, required=True)
        if command == "validate":
            child.add_argument("--matched-key", required=True)
        else:
            child.add_argument("--source-sha", required=True)
            child.add_argument("--generation", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "identity":
            identity = cache_identity(args.root, args.bucket, args.platform, args.image_id)
            _write_json(args.output, identity)
            print(f"cache_prefix={identity['cache_prefix']}")
        elif args.command == "finalize":
            finalize(args.cache_dir, _read_json(args.identity), args.source_sha, args.generation)
        else:
            identity = _read_json(args.identity)
            manifest = validate_cache(args.cache_dir, identity, args.matched_key)
            created_at = datetime.fromisoformat(manifest["created_at"])
            age_hours = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600
            print(
                f"Baseline source: {manifest['source_sha']} "
                f"(created {manifest['created_at']}; age {age_hours:.1f} hours)"
            )
            print(
                f"Baseline image: {manifest.get('image_id', 'unknown')}; "
                f"current image: {identity['image_id']}"
            )
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f"Testmon cache: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
