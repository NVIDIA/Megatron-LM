# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Build and consume distributed pytest-testmon baseline metadata.

The module intentionally depends only on the Python standard library so it can
run on bare GitHub setup-python hosts.  All policy failures raise
``TestmonValidationError`` so callers can fail closed to the exhaustive suite.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import fnmatch
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
TESTMON_VERSION = "2.2.0"
DEFAULT_PHASES = ("prod", "experimental")
DEFAULT_MAX_AGE_HOURS = 72.0
UNIT_TEST_ROOT = "tests/unit_tests"

_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_HEX256_RE = re.compile(r"^[0-9a-f]{64}$")
_SYNTHETIC_PR_REF_RE = re.compile(r"^refs/heads/pull-request/[0-9]+$")
_NATIVE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hh", ".hpp", ".so"}

# Any change here can alter dependency collection, CI filters, or the execution
# environment.  Such changes invalidate selection before Testmon is consulted.
_FULL_EXACT_PATHS = {
    "pyproject.toml",
    "uv.lock",
    "tests/unit_tests/conftest.py",
    "tests/unit_tests/find_test_cases.py",
    "tests/unit_tests/run_ci_test.sh",
    "tests/unit_tests/testmon_selected_plugin.py",
    "tests/test_utils/python_scripts/launch_nemo_run_workload.py",
}
_FULL_PREFIXES = (
    ".github/actions/",
    ".github/workflows/",
    "docker/",
    "tests/test_utils/recipes/",
    "tests/unit_tests/testmon/",
)


class TestmonValidationError(ValueError):
    """A condition for which selective testing must fail closed."""


@dataclasses.dataclass(frozen=True)
class Change:
    """One normalized git diff entry."""

    status: str
    path: str
    old_path: str | None = None

    def __post_init__(self) -> None:
        status = self.status.upper()
        if not status or status[0] not in {"A", "C", "D", "M", "R", "T", "U"}:
            raise TestmonValidationError(f"unsupported git status: {self.status!r}")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "path", normalize_repo_path(self.path))
        if self.old_path is not None:
            object.__setattr__(self, "old_path", normalize_repo_path(self.old_path))
        if status[0] in {"R", "C"} and self.old_path is None:
            raise TestmonValidationError(f"{status} change is missing its old path")


def _write_json(path: str | os.PathLike[str], value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _read_json(path: str | os.PathLike[str]) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TestmonValidationError(f"cannot read JSON {path}: {exc}") from exc


def normalize_repo_path(path: str | os.PathLike[str]) -> str:
    """Return a safe, normalized repository-relative POSIX path."""

    raw = str(path).replace("\\", "/")
    candidate = PurePosixPath(raw)
    if not raw or candidate.is_absolute() or raw.startswith("/"):
        raise TestmonValidationError(f"path must be repository-relative: {raw!r}")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise TestmonValidationError(f"path contains traversal or empty components: {raw!r}")
    normalized = candidate.as_posix()
    if normalized.startswith("../"):
        raise TestmonValidationError(f"path escapes repository: {raw!r}")
    return normalized


def _safe_file(root: Path, relative_path: str, *, must_exist: bool = True) -> Path:
    relative_path = normalize_repo_path(relative_path)
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise TestmonValidationError(f"path escapes {root}: {relative_path}") from exc
    if must_exist and not candidate.is_file():
        raise TestmonValidationError(f"required file is missing: {relative_path}")
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise TestmonValidationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def bucket_hash(bucket: str) -> str:
    """Hash a normalized recipe bucket for use in immutable cache keys."""

    normalized = normalize_repo_path(bucket)
    return hashlib.sha256(("unit-testmon-bucket-v1\0" + normalized).encode()).hexdigest()


def expected_topology_identity(platform: str, world_size: int) -> str:
    """Return the canonical distributed selector topology identity."""

    if platform not in {"h100", "gb200"} or world_size <= 0:
        raise TestmonValidationError(f"invalid topology: {platform}/r{world_size}")
    value = f"{platform}:r{world_size}:prod,experimental:torchrun"
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _default_environment_paths(platform: str) -> list[str]:
    if platform not in {"h100", "gb200"}:
        raise TestmonValidationError(f"unsupported platform: {platform}")
    return [
        "pyproject.toml",
        "uv.lock",
        "docker/.ngc_version.dev",
        "docker/Dockerfile.ci.dev",
        ".github/actions/action.yml",
        ".github/workflows/cicd-main.yml",
        ".github/workflows/unit-testmon-baseline.yml",
        f"tests/test_utils/recipes/{platform}/unit-tests.yaml",
        "tests/test_utils/python_scripts/launch_nemo_run_workload.py",
        "tests/unit_tests/run_ci_test.sh",
        "tests/unit_tests/conftest.py",
        "tests/unit_tests/find_test_cases.py",
        "tests/unit_tests/testmon_selected_plugin.py",
    ]


def compute_environment_hash(
    repo_root: str | os.PathLike[str],
    platform: str,
    *,
    testmon_version: str = TESTMON_VERSION,
    extra_paths: Iterable[str] = (),
    identities: Mapping[str, str] | None = None,
) -> str:
    """Hash every input which can affect Testmon collection or bucketing."""

    root = Path(repo_root)
    paths = set(_default_environment_paths(platform))
    paths.update(normalize_repo_path(path) for path in extra_paths)
    selector_root = root / "tests/unit_tests/testmon"
    if not selector_root.is_dir():
        raise TestmonValidationError("selector implementation directory is missing")
    for selector_file in selector_root.rglob("*.py"):
        if selector_file.is_file():
            paths.add(selector_file.relative_to(root).as_posix())
    for conftest in (root / UNIT_TEST_ROOT).rglob("conftest.py"):
        if conftest.is_file():
            paths.add(conftest.relative_to(root).as_posix())

    digest = hashlib.sha256()
    digest.update(b"unit-testmon-environment-v1\0")
    digest.update(f"testmon={testmon_version}\0platform={platform}\0".encode())
    for relative_path in sorted(paths):
        source = _safe_file(root, relative_path)
        digest.update(relative_path.encode() + b"\0")
        digest.update(bytes.fromhex(_sha256_file(source)))
    for name, value in sorted((identities or {}).items()):
        digest.update(f"identity:{name}={value}\0".encode())
    return digest.hexdigest()


def decide_pr_eligibility(
    *,
    enabled: bool,
    event_name: str,
    ref: str,
    github_sha: str,
    pr_head_sha: str,
    metadata_valid: bool,
    labels: Sequence[str] = (),
    force_run_all: bool = False,
    container: str = "dev",
) -> dict[str, Any]:
    """Apply the complete event-level gate for PR-only selective testing."""

    reasons: list[str] = []
    if not enabled:
        reasons.append("ENABLE_PR_UNIT_TESTMON is not true")
    if event_name != "push":
        reasons.append(f"event {event_name!r} is not a synthetic PR push")
    if not _SYNTHETIC_PR_REF_RE.fullmatch(ref):
        reasons.append(f"ref {ref!r} is not a synthetic numeric PR ref")
    if not metadata_valid:
        reasons.append("PR metadata is invalid")
    try:
        tested_sha = _validate_sha(github_sha, "github_sha")
        head_sha = _validate_sha(pr_head_sha, "pr_head_sha")
        if tested_sha != head_sha:
            reasons.append("PR metadata head SHA does not match the tested SHA")
    except TestmonValidationError as exc:
        reasons.append(str(exc))
    full_labels = sorted({"Run tests", "Run functional tests"} & set(labels))
    if full_labels:
        reasons.append(f"full-test label attached: {', '.join(full_labels)}")
    if force_run_all:
        reasons.append("force_run_all is true")
    if container != "dev":
        reasons.append(f"container {container!r} is not the default dev container")
    return {
        "mode": "enforce" if not reasons else "full",
        "eligible": not reasons,
        "reason": "; ".join(reasons) if reasons else "eligible synthetic PR push",
        "reasons": reasons,
    }


def _parse_timestamp(value: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TestmonValidationError(f"invalid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise TestmonValidationError("timestamp must include a timezone")
    return parsed.astimezone(dt.timezone.utc)


def _format_timestamp(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_sha(value: str, field: str) -> str:
    if not _SHA_RE.fullmatch(value):
        raise TestmonValidationError(f"{field} must be a full 40-character commit SHA")
    return value.lower()


def _load_nodeids(path: Path) -> list[str]:
    payload = _read_json(path)
    if isinstance(payload, list):
        nodeids = payload
    elif isinstance(payload, dict):
        nodeids = None
        for key in ("nodeids", "selected_node_ids", "selected", "tests"):
            if key in payload:
                nodeids = payload[key]
                break
        if nodeids is None:
            raise TestmonValidationError(f"node JSON {path} has no recognized node-id key")
    else:
        raise TestmonValidationError(f"node JSON {path} must be a list or object")
    if not isinstance(nodeids, list) or any(not isinstance(item, str) for item in nodeids):
        raise TestmonValidationError(f"node JSON {path} must contain a list of strings")
    if len(nodeids) != len(set(nodeids)):
        raise TestmonValidationError(f"node JSON {path} contains duplicate node IDs")
    for nodeid in nodeids:
        _test_file_from_nodeid(nodeid)
    return sorted(nodeids)


def _test_file_from_nodeid(nodeid: str) -> str:
    path = nodeid.split("::", 1)[0]
    normalized = normalize_repo_path(path)
    if not normalized.startswith(UNIT_TEST_ROOT + "/"):
        raise TestmonValidationError(f"selected node is outside unit tests: {nodeid!r}")
    if not normalized.endswith(".py"):
        raise TestmonValidationError(f"selected node is not in a Python file: {nodeid!r}")
    if not PurePosixPath(normalized).name.startswith("test_"):
        raise TestmonValidationError(f"selected node is not in a pytest test file: {nodeid!r}")
    return normalized


def _sqlite_integrity(path: Path, *, checkpoint: bool) -> None:
    if not path.is_file():
        raise TestmonValidationError(f"Testmon database is missing: {path}")
    try:
        if checkpoint:
            connection = sqlite3.connect(path)
        else:
            connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        try:
            if checkpoint:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
            result = connection.execute("PRAGMA integrity_check").fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise TestmonValidationError(f"invalid SQLite database {path}: {exc}") from exc
    if result != [("ok",)]:
        raise TestmonValidationError(f"SQLite integrity check failed for {path}: {result!r}")


def _testmon_attributed_nodeids(database: Path, collected_nodeids: set[str]) -> set[str]:
    """Find collected tests with an in-process dependency under ``megatron/``."""

    try:
        connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
        try:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            required = {"test_execution", "test_execution_file_fp", "file_fp"}
            if not required.issubset(tables):
                raise TestmonValidationError(
                    f"{database} is SQLite but not a Testmon dependency database; "
                    f"missing tables {sorted(required - tables)}"
                )
            rows = connection.execute("""
                SELECT te.test_name, f.filename
                FROM test_execution AS te
                LEFT JOIN test_execution_file_fp AS link
                  ON te.id = link.test_execution_id
                LEFT JOIN file_fp AS f
                  ON link.fingerprint_id = f.id
                """).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise TestmonValidationError(
            f"cannot inspect Testmon dependencies in {database}: {exc}"
        ) from exc

    production_dependencies: dict[str, bool] = {}
    for test_name, filename in rows:
        if not isinstance(test_name, str):
            continue
        production_dependencies.setdefault(test_name, False)
        if isinstance(filename, str) and filename.replace("\\", "/").startswith("megatron/"):
            production_dependencies[test_name] = True
    return {nodeid for nodeid in collected_nodeids if production_dependencies.get(nodeid, False)}


def _default_tracked_python_files(repo_root: Path) -> list[str]:
    tracked: list[str] = []
    for prefix in ("megatron", UNIT_TEST_ROOT):
        base = repo_root / prefix
        if base.is_dir():
            tracked.extend(
                path.relative_to(repo_root).as_posix()
                for path in base.rglob("*.py")
                if path.is_file() and "__pycache__" not in path.parts
            )
    return sorted(set(tracked))


def _validate_test_file(path: str, repo_root: Path, *, must_exist: bool = True) -> str:
    normalized = normalize_repo_path(path)
    if not normalized.startswith(UNIT_TEST_ROOT + "/"):
        raise TestmonValidationError(f"test path is outside {UNIT_TEST_ROOT}: {path!r}")
    if not normalized.endswith(".py") or not PurePosixPath(normalized).name.startswith("test_"):
        raise TestmonValidationError(f"not a pytest unit-test file: {path!r}")
    _safe_file(repo_root, normalized, must_exist=must_exist)
    return normalized


def _cache_key(
    platform: str, world_size: int, bucket: str, environment_hash: str, producer_sha: str
) -> str:
    return (
        f"unit-testmon-v1-main-{platform}-r{world_size}-dev-"
        f"{bucket_hash(bucket)}-{environment_hash}-{producer_sha}"
    )


def build_bucket_manifest(
    cache_dir: str | os.PathLike[str],
    *,
    repo_root: str | os.PathLike[str],
    platform: str,
    world_size: int,
    bucket: str,
    producer_sha: str,
    producer_time: str,
    environment_hash: str,
    container_identity: str,
    dependency_identity: str,
    topology_identity: str,
    phases: Sequence[str] = DEFAULT_PHASES,
    tracked_python_files: Sequence[str] | None = None,
    always_run_files: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate a completed baseline bucket and create its immutable manifest."""

    root = Path(repo_root).resolve()
    cache_root = Path(cache_dir).resolve()
    bucket = normalize_repo_path(bucket)
    producer_sha = _validate_sha(producer_sha, "producer_sha")
    producer_time = _format_timestamp(_parse_timestamp(producer_time))
    if platform not in {"h100", "gb200"}:
        raise TestmonValidationError(f"unsupported platform: {platform}")
    if world_size <= 0:
        raise TestmonValidationError("world_size must be positive")
    if not _HEX256_RE.fullmatch(environment_hash):
        raise TestmonValidationError("environment_hash must be a lowercase SHA-256 digest")
    if not phases or len(phases) != len(set(phases)):
        raise TestmonValidationError("phases must be non-empty and unique")
    for identity_name, identity in (
        ("container_identity", container_identity),
        ("dependency_identity", dependency_identity),
        ("topology_identity", topology_identity),
    ):
        if not identity:
            raise TestmonValidationError(f"{identity_name} must not be empty")

    databases: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    collected_node_counts: dict[str, int] = {}
    all_collected_nodeids: set[str] = set()
    attributed_nodeids: set[str] = set()
    for phase in phases:
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", phase):
            raise TestmonValidationError(f"invalid phase name: {phase!r}")
        expected_nodes: set[str] | None = None
        for rank in range(world_size):
            database = cache_root / phase / f"rank-{rank}.testmondata"
            selection = cache_root / "collected" / phase / f"rank-{rank}.json"
            _sqlite_integrity(database, checkpoint=True)
            nodeids = set(_load_nodeids(selection))
            all_collected_nodeids.update(nodeids)
            attributed_nodeids.update(_testmon_attributed_nodeids(database, nodeids))
            if expected_nodes is None:
                expected_nodes = nodeids
            elif nodeids != expected_nodes:
                missing = sorted(expected_nodes - nodeids)[:3]
                extra = sorted(nodeids - expected_nodes)[:3]
                raise TestmonValidationError(
                    f"collected node sets disagree in {phase} rank {rank}; "
                    f"missing={missing}, extra={extra}"
                )
            databases.append(
                {
                    "phase": phase,
                    "rank": rank,
                    "path": database.relative_to(cache_root).as_posix(),
                    "sha256": _sha256_file(database),
                    "size": database.stat().st_size,
                }
            )
            selections.append(
                {
                    "phase": phase,
                    "rank": rank,
                    "path": selection.relative_to(cache_root).as_posix(),
                    "sha256": _sha256_file(selection),
                    "node_count": len(nodeids),
                }
            )
        collected_node_counts[phase] = len(expected_nodes or set())

    tracked = tracked_python_files
    if tracked is None:
        tracked = _default_tracked_python_files(root)
    normalized_tracked = sorted(
        {
            normalize_repo_path(path)
            for path in tracked
            if str(path).replace("\\", "/").endswith(".py")
        }
    )
    for path in normalized_tracked:
        _safe_file(root, path)

    # A dependency observed on any distributed rank is attributable because PR
    # selection unions every rank.  Tests with no such dependency anywhere are
    # the subprocess/untraced conservative floor.
    unattributed_tests = {
        _validate_test_file(_test_file_from_nodeid(nodeid), root)
        for nodeid in all_collected_nodeids - attributed_nodeids
    }
    normalized_always = sorted(
        {_validate_test_file(path, root) for path in always_run_files}
        | unattributed_tests
        | (
            {"tests/unit_tests/test_basic.py"}
            if (root / "tests/unit_tests/test_basic.py").is_file()
            else set()
        )
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "testmon_version": TESTMON_VERSION,
        "producer_sha": producer_sha,
        "producer_time": producer_time,
        "platform": platform,
        "world_size": world_size,
        "environment": "dev",
        "bucket": bucket,
        "bucket_hash": bucket_hash(bucket),
        "environment_hash": environment_hash,
        "container_identity": container_identity,
        "dependency_identity": dependency_identity,
        "topology_identity": topology_identity,
        "phases": list(phases),
        "cache_key": _cache_key(platform, world_size, bucket, environment_hash, producer_sha),
        "collected_node_counts": collected_node_counts,
        "databases": databases,
        "collected_files": selections,
        "tracked_python_files": normalized_tracked,
        "always_run_files": normalized_always,
        "unattributed_tests": sorted(unattributed_tests),
    }


def _validate_manifest_shape(
    manifest: Mapping[str, Any],
    *,
    expected_platform: str | None = None,
    expected_world_size: int | None = None,
    expected_bucket: str | None = None,
    expected_environment_hash: str | None = None,
) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise TestmonValidationError("unsupported bucket-manifest schema version")
    if manifest.get("testmon_version") != TESTMON_VERSION:
        raise TestmonValidationError("bucket manifest uses an incompatible Testmon version")
    producer_sha = _validate_sha(str(manifest.get("producer_sha", "")), "producer_sha")
    _parse_timestamp(str(manifest.get("producer_time", "")))
    platform = manifest.get("platform")
    world_size = manifest.get("world_size")
    bucket = normalize_repo_path(str(manifest.get("bucket", "")))
    environment_hash = manifest.get("environment_hash")
    if platform not in {"h100", "gb200"}:
        raise TestmonValidationError("bucket manifest has an unsupported platform")
    if not isinstance(world_size, int) or world_size <= 0:
        raise TestmonValidationError("bucket manifest has an invalid world size")
    if not isinstance(environment_hash, str) or not _HEX256_RE.fullmatch(environment_hash):
        raise TestmonValidationError("bucket manifest has an invalid environment hash")
    if manifest.get("bucket_hash") != bucket_hash(bucket):
        raise TestmonValidationError("bucket hash does not match the bucket")
    if manifest.get("cache_key") != _cache_key(
        str(platform), world_size, bucket, environment_hash, producer_sha
    ):
        raise TestmonValidationError("bucket cache key is not canonical")
    if expected_platform is not None and platform != expected_platform:
        raise TestmonValidationError(
            f"platform mismatch: expected {expected_platform}, found {platform}"
        )
    if expected_world_size is not None and world_size != expected_world_size:
        raise TestmonValidationError(
            f"world-size mismatch: expected {expected_world_size}, found {world_size}"
        )
    if expected_bucket is not None and bucket != normalize_repo_path(expected_bucket):
        raise TestmonValidationError(f"bucket mismatch: expected {expected_bucket}, found {bucket}")
    if expected_environment_hash is not None and environment_hash != expected_environment_hash:
        raise TestmonValidationError("environment hash does not match the current checkout")
    phases = manifest.get("phases")
    if not isinstance(phases, list) or not phases or len(phases) != len(set(phases)):
        raise TestmonValidationError("bucket manifest phases are invalid")
    expected_pairs = {(phase, rank) for phase in phases for rank in range(world_size)}
    for collection_name in ("databases", "collected_files"):
        collection = manifest.get(collection_name)
        if not isinstance(collection, list):
            raise TestmonValidationError(f"bucket manifest {collection_name} must be a list")
        actual_pairs: set[tuple[str, int]] = set()
        for record in collection:
            if not isinstance(record, dict):
                raise TestmonValidationError(f"invalid {collection_name} record")
            pair = (record.get("phase"), record.get("rank"))
            if pair in actual_pairs:
                raise TestmonValidationError(f"duplicate {collection_name} record for {pair}")
            actual_pairs.add(pair)
            normalize_repo_path(str(record.get("path", "")))
            checksum = record.get("sha256")
            if not isinstance(checksum, str) or not _HEX256_RE.fullmatch(checksum):
                raise TestmonValidationError(f"invalid checksum in {collection_name}")
        if actual_pairs != expected_pairs:
            raise TestmonValidationError(
                f"incomplete {collection_name}; expected {sorted(expected_pairs)}, "
                f"found {sorted(actual_pairs)}"
            )
    counts = manifest.get("collected_node_counts")
    if not isinstance(counts, dict) or set(counts) != set(phases):
        raise TestmonValidationError("collected-node counts do not cover every phase")
    if any(not isinstance(value, int) or value < 0 for value in counts.values()):
        raise TestmonValidationError("collected-node counts must be non-negative integers")
    for field in ("container_identity", "dependency_identity", "topology_identity"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise TestmonValidationError(f"bucket manifest is missing {field}")
    for field in ("tracked_python_files", "always_run_files", "unattributed_tests"):
        paths = manifest.get(field)
        if not isinstance(paths, list) or paths != sorted(set(paths)):
            raise TestmonValidationError(f"bucket manifest {field} must be sorted and unique")
        for path in paths:
            normalized = normalize_repo_path(path)
            if not normalized.endswith(".py"):
                raise TestmonValidationError(f"non-Python path in {field}: {path}")


def validate_bucket_manifest(
    manifest: Mapping[str, Any],
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    manifest_path: str | os.PathLike[str] | None = None,
    trusted_index_record: Mapping[str, Any] | None = None,
    expected_platform: str | None = None,
    expected_world_size: int | None = None,
    expected_bucket: str | None = None,
    expected_environment_hash: str | None = None,
) -> None:
    """Validate manifest metadata, checksums, and all restored SQLite files."""

    _validate_manifest_shape(
        manifest,
        expected_platform=expected_platform,
        expected_world_size=expected_world_size,
        expected_bucket=expected_bucket,
        expected_environment_hash=expected_environment_hash,
    )
    if trusted_index_record is not None:
        if manifest_path is None:
            raise TestmonValidationError(
                "manifest_path is required when validating against a trusted index record"
            )
        expected_manifest_checksum = trusted_index_record.get("manifest_sha256")
        if not isinstance(expected_manifest_checksum, str) or not _HEX256_RE.fullmatch(
            expected_manifest_checksum
        ):
            raise TestmonValidationError("trusted index record has an invalid manifest checksum")
        actual_manifest_checksum = _sha256_file(Path(manifest_path))
        if actual_manifest_checksum != expected_manifest_checksum:
            raise TestmonValidationError("restored manifest checksum does not match platform index")

        expected_database_checksums = trusted_index_record.get("database_checksums")
        actual_database_checksums = [record["sha256"] for record in manifest["databases"]]
        if expected_database_checksums != actual_database_checksums:
            raise TestmonValidationError(
                "restored database checksum list does not match platform index"
            )
    if cache_dir is None:
        return
    root = Path(cache_dir).resolve()
    for collection_name in ("databases", "collected_files"):
        for record in manifest[collection_name]:
            path = _safe_file(root, record["path"])
            if _sha256_file(path) != record["sha256"]:
                raise TestmonValidationError(f"checksum mismatch for {record['path']}")
            if collection_name == "databases":
                _sqlite_integrity(path, checkpoint=False)


def recipe_buckets(recipe: str | os.PathLike[str]) -> list[str]:
    """Read top-level unit-test buckets from the repository recipe format.

    Publish jobs run on bare setup-python hosts, so this deliberately avoids a
    YAML dependency.  The unit recipes use a constrained one-line form at two
    spaces of indentation: ``- test_case: [path, ...]``.  Any other structure
    fails closed instead of guessing.
    """

    try:
        lines = Path(recipe).read_text().splitlines()
    except OSError as exc:
        raise TestmonValidationError(f"cannot read recipe {recipe}: {exc}") from exc
    pattern = re.compile(r"^  - test_case:\s*\[(.*)\]\s*(?:#.*)?$")
    raw: list[str] = []
    for line in lines:
        match = pattern.fullmatch(line)
        if not match:
            continue
        try:
            values = next(csv.reader([match.group(1)], skipinitialspace=True))
        except csv.Error as exc:
            raise TestmonValidationError(f"invalid test_case list in {recipe}: {line}") from exc
        raw.extend(value.strip().strip("'\"") for value in values if value.strip())
    if not raw:
        raise TestmonValidationError(
            f"recipe {recipe} has no supported top-level inline test_case entries"
        )
    buckets = [normalize_repo_path(bucket) for bucket in raw]
    if len(buckets) != len(set(buckets)):
        raise TestmonValidationError(f"recipe {recipe} contains duplicate buckets")
    return sorted(buckets)


def build_platform_index(
    manifests: Sequence[tuple[Mapping[str, Any], Path]],
    *,
    expected_buckets: Sequence[str],
    platform: str,
    world_size: int,
    environment_hash: str,
    producer_time_override: str | None = None,
) -> dict[str, Any]:
    """Create an index only when a platform baseline is complete and coherent."""

    expected = {normalize_repo_path(bucket) for bucket in expected_buckets}
    if not expected:
        raise TestmonValidationError("the platform recipe contains no buckets")
    by_bucket: dict[str, tuple[Mapping[str, Any], Path]] = {}
    for manifest, manifest_path in manifests:
        validate_bucket_manifest(
            manifest,
            expected_platform=platform,
            expected_world_size=world_size,
            expected_environment_hash=environment_hash,
        )
        bucket = manifest["bucket"]
        if bucket in by_bucket:
            raise TestmonValidationError(f"duplicate manifest for bucket {bucket}")
        by_bucket[bucket] = (manifest, manifest_path)
    if set(by_bucket) != expected:
        missing = sorted(expected - set(by_bucket))
        extra = sorted(set(by_bucket) - expected)
        raise TestmonValidationError(
            f"platform baseline is incomplete; missing={missing}, unexpected={extra}"
        )

    first = next(iter(by_bucket.values()))[0]
    if first["topology_identity"] != expected_topology_identity(platform, world_size):
        raise TestmonValidationError("baseline topology identity is not canonical")
    coherence_fields = [
        "schema_version",
        "testmon_version",
        "producer_sha",
        "platform",
        "world_size",
        "environment",
        "environment_hash",
        "container_identity",
        "dependency_identity",
        "topology_identity",
        "phases",
    ]
    if producer_time_override is None:
        coherence_fields.append("producer_time")
        producer_time = first["producer_time"]
    else:
        producer_time = _format_timestamp(_parse_timestamp(producer_time_override))
    for bucket, (manifest, _) in by_bucket.items():
        for field in coherence_fields:
            if manifest[field] != first[field]:
                raise TestmonValidationError(
                    f"bucket {bucket} has inconsistent baseline field {field}"
                )

    bucket_records: dict[str, dict[str, Any]] = {}
    tracked: set[str] = set()
    always_run: set[str] = set()
    unattributed: set[str] = set()
    for bucket in sorted(by_bucket):
        manifest, manifest_path = by_bucket[bucket]
        tracked.update(manifest["tracked_python_files"])
        always_run.update(manifest["always_run_files"])
        unattributed.update(manifest["unattributed_tests"])
        bucket_records[bucket] = {
            "bucket_hash": manifest["bucket_hash"],
            "cache_key": manifest["cache_key"],
            "manifest_sha256": _sha256_file(manifest_path),
            "collected_node_counts": manifest["collected_node_counts"],
            "database_checksums": [record["sha256"] for record in manifest["databases"]],
            "always_run_files": manifest["always_run_files"],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "testmon_version": TESTMON_VERSION,
        "producer_sha": first["producer_sha"],
        "producer_time": producer_time,
        "platform": platform,
        "world_size": world_size,
        "environment": "dev",
        "environment_hash": environment_hash,
        "container_identity": first["container_identity"],
        "dependency_identity": first["dependency_identity"],
        "topology_identity": first["topology_identity"],
        "phases": first["phases"],
        "buckets": bucket_records,
        "tracked_python_files": sorted(tracked),
        "always_run_files": sorted(always_run),
        "unattributed_tests": sorted(unattributed),
    }


def _git_is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo_root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def validate_platform_index(
    index: Mapping[str, Any],
    *,
    repo_root: str | os.PathLike[str],
    base_sha: str,
    platform: str,
    world_size: int,
    environment_hash: str,
    expected_buckets: Sequence[str] | None = None,
    now: dt.datetime | None = None,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
) -> None:
    """Validate index compatibility, freshness, completeness, and ancestry."""

    if index.get("schema_version") != SCHEMA_VERSION:
        raise TestmonValidationError("unsupported platform-index schema version")
    if index.get("testmon_version") != TESTMON_VERSION:
        raise TestmonValidationError("platform index uses an incompatible Testmon version")
    if index.get("platform") != platform or index.get("world_size") != world_size:
        raise TestmonValidationError("platform index topology does not match this job")
    if index.get("environment") != "dev":
        raise TestmonValidationError("only a dev baseline may be used for selective testing")
    if index.get("environment_hash") != environment_hash:
        raise TestmonValidationError("platform index environment hash is incompatible")
    if index.get("topology_identity") != expected_topology_identity(platform, world_size):
        raise TestmonValidationError("platform index topology identity is incompatible")
    producer_sha = _validate_sha(str(index.get("producer_sha", "")), "producer_sha")
    base_sha = _validate_sha(base_sha, "base_sha")
    produced = _parse_timestamp(str(index.get("producer_time", "")))
    current = now or dt.datetime.now(dt.timezone.utc)
    if current.tzinfo is None:
        raise TestmonValidationError("current time must include a timezone")
    age = current.astimezone(dt.timezone.utc) - produced
    if age < dt.timedelta(minutes=-5):
        raise TestmonValidationError("platform index timestamp is in the future")
    if age > dt.timedelta(hours=max_age_hours):
        raise TestmonValidationError(
            f"platform index is stale ({age.total_seconds() / 3600:.1f} hours old)"
        )
    if not _git_is_ancestor(Path(repo_root), producer_sha, base_sha):
        raise TestmonValidationError("baseline producer is not an ancestor of the PR base")

    buckets = index.get("buckets")
    if not isinstance(buckets, dict) or not buckets:
        raise TestmonValidationError("platform index has no bucket records")
    if expected_buckets is not None:
        expected = {normalize_repo_path(bucket) for bucket in expected_buckets}
        if set(buckets) != expected:
            raise TestmonValidationError("platform index does not exactly cover the current recipe")
    for bucket, record in buckets.items():
        bucket = normalize_repo_path(bucket)
        if not isinstance(record, dict):
            raise TestmonValidationError(f"invalid index record for bucket {bucket}")
        if record.get("bucket_hash") != bucket_hash(bucket):
            raise TestmonValidationError(f"invalid bucket hash for {bucket}")
        if record.get("cache_key") != _cache_key(
            platform, world_size, bucket, environment_hash, producer_sha
        ):
            raise TestmonValidationError(f"invalid cache key for {bucket}")
        if not _HEX256_RE.fullmatch(str(record.get("manifest_sha256", ""))):
            raise TestmonValidationError(f"invalid manifest checksum for {bucket}")
        checksums = record.get("database_checksums")
        phases = index.get("phases")
        if not isinstance(phases, list) or not phases:
            raise TestmonValidationError("platform index phases are invalid")
        if not isinstance(checksums, list) or len(checksums) != world_size * len(phases):
            raise TestmonValidationError(f"incomplete database checksum metadata for {bucket}")
        if any(
            not isinstance(value, str) or not _HEX256_RE.fullmatch(value) for value in checksums
        ):
            raise TestmonValidationError(f"invalid database checksum metadata for {bucket}")
    for field in ("tracked_python_files", "always_run_files", "unattributed_tests"):
        paths = index.get(field)
        if not isinstance(paths, list) or paths != sorted(set(paths)):
            raise TestmonValidationError(f"platform index {field} must be sorted and unique")
        for path in paths:
            if not normalize_repo_path(path).endswith(".py"):
                raise TestmonValidationError(f"non-Python path in platform index {field}")


def _get_base_path(pattern: str) -> str:
    if "**" in pattern:
        return pattern.split("/**", 1)[0]
    if "*" in pattern:
        return pattern.rsplit("/", 1)[0]
    return pattern.rstrip("/")


def _is_child_bucket(candidate: str, parent: str) -> bool:
    candidate_base = _get_base_path(candidate)
    parent_base = _get_base_path(parent)
    return candidate_base.startswith(parent_base + "/")


def _bucket_matches(path: str, bucket: str) -> bool:
    path = normalize_repo_path(path)
    bucket = normalize_repo_path(bucket)
    if "**" in bucket:
        base, suffix = bucket.split("/**", 1)
        if not path.startswith(base.rstrip("/") + "/"):
            return False
        suffix = suffix.lstrip("/")
        if not suffix:
            return True
        return fnmatch.fnmatch(PurePosixPath(path).name, suffix)
    return fnmatch.fnmatchcase(path, bucket)


def owning_bucket(path: str, buckets: Sequence[str]) -> str:
    """Resolve the same parent/child precedence used by find_test_cases.py."""

    path = normalize_repo_path(path)
    normalized_buckets = [normalize_repo_path(bucket) for bucket in buckets]
    matching = [bucket for bucket in normalized_buckets if _bucket_matches(path, bucket)]
    effective = [
        bucket
        for bucket in matching
        if not any(
            other != bucket and _is_child_bucket(other, bucket) and _bucket_matches(path, other)
            for other in normalized_buckets
        )
    ]
    if len(effective) != 1:
        raise TestmonValidationError(
            f"test path {path} has {len(effective)} effective recipe owners: {sorted(effective)}"
        )
    return effective[0]


def _is_direct_test(path: str) -> bool:
    candidate = PurePosixPath(path)
    return (
        path.startswith(UNIT_TEST_ROOT + "/")
        and candidate.suffix == ".py"
        and candidate.name.startswith("test_")
    )


def classify_changes(
    changes: Sequence[Change], *, tracked_python_files: Sequence[str], buckets: Sequence[str]
) -> dict[str, Any]:
    """Classify a diff, returning ``full`` for every uncertain path."""

    tracked = {normalize_repo_path(path) for path in tracked_python_files}
    direct_tests: set[str] = set()
    changed_python: set[str] = set()
    deleted_tests: set[str] = set()
    ignored_paths: set[str] = set()
    reasons: list[str] = []

    def full(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    for change in changes:
        path = change.path
        kind = change.status[0]
        considered_paths = [path]
        if change.old_path is not None:
            considered_paths.append(change.old_path)
        for candidate in considered_paths:
            if candidate in _FULL_EXACT_PATHS or candidate.startswith(_FULL_PREFIXES):
                full(f"selection input changed: {candidate}")
            if (
                candidate.startswith(UNIT_TEST_ROOT + "/")
                and PurePosixPath(candidate).name == "conftest.py"
            ):
                full(f"pytest collection hook changed: {candidate}")
            if PurePosixPath(candidate).suffix.lower() in _NATIVE_SUFFIXES:
                full(f"native or compiled source changed: {candidate}")

        if path.startswith("docs/") or (
            "/" not in path and PurePosixPath(path).suffix.lower() in {".md", ".rst"}
        ):
            ignored_paths.add(path)
            continue

        if _is_direct_test(path):
            if kind == "D":
                deleted_tests.add(path)
            else:
                direct_tests.add(path)
                try:
                    owning_bucket(path, buckets)
                except TestmonValidationError as exc:
                    full(str(exc))
            continue

        suffix = PurePosixPath(path).suffix.lower()
        is_dependency_python = path.startswith("megatron/") or path.startswith(UNIT_TEST_ROOT + "/")
        if is_dependency_python and suffix == ".py":
            changed_python.add(path)
            # Added/copied/renamed dependency files are not represented by an
            # older Testmon baseline.  Existing tracked files may be modified
            # or deleted and Testmon can compare those states safely.
            if kind in {"A", "C", "R"} and path not in tracked:
                full(f"new Python dependency is absent from the baseline: {path}")
            elif kind in {"M", "D", "T"} and path not in tracked:
                full(f"changed Python dependency is absent from the baseline: {path}")
            continue

        # Explicitly harmless documentation inside unit-test directories.
        if path.startswith(UNIT_TEST_ROOT + "/") and suffix in {".md", ".rst"}:
            ignored_paths.add(path)
            continue

        # Deleting a test was handled above.  Every other file type or scope is
        # unknown and therefore forces the exhaustive path.
        full(f"changed path is not safely classified: {path}")

    bucket_tests: dict[str, list[str]] = {bucket: [] for bucket in buckets}
    for test_path in sorted(direct_tests):
        try:
            bucket_tests[owning_bucket(test_path, buckets)].append(test_path)
        except TestmonValidationError:
            pass
    return {
        "mode": "full" if reasons else "select",
        "reason": "; ".join(reasons) if reasons else "all changed paths are understood",
        "reasons": reasons,
        "direct_tests": sorted(direct_tests),
        "direct_tests_by_bucket": bucket_tests,
        "changed_python_files": sorted(changed_python),
        "deleted_tests": sorted(deleted_tests),
        "ignored_paths": sorted(ignored_paths),
    }


def _parse_git_diff_z(payload: bytes) -> list[Change]:
    fields = payload.decode("utf-8", errors="strict").split("\0")
    if fields and fields[-1] == "":
        fields.pop()
    changes: list[Change] = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        if not status:
            raise TestmonValidationError("git diff contains an empty status")
        kind = status[0]
        if kind in {"R", "C"}:
            if index + 1 >= len(fields):
                raise TestmonValidationError("truncated rename/copy entry in git diff")
            old_path, new_path = fields[index], fields[index + 1]
            index += 2
            changes.append(Change(status=status, path=new_path, old_path=old_path))
        else:
            if index >= len(fields):
                raise TestmonValidationError("truncated entry in git diff")
            changes.append(Change(status=status, path=fields[index]))
            index += 1
    return changes


def git_diff_changes(repo_root: str | os.PathLike[str], start: str, end: str) -> list[Change]:
    start = _validate_sha(start, "diff start SHA")
    end = _validate_sha(end, "diff end SHA")
    result = subprocess.run(
        ["git", "diff", "--name-status", "-z", "--find-renames", start, end],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        message = result.stderr.decode(errors="replace").strip()
        raise TestmonValidationError(f"cannot compute git diff: {message}")
    return _parse_git_diff_z(result.stdout)


def union_rank_selections(
    cache_dir: str | os.PathLike[str],
    *,
    repo_root: str | os.PathLike[str],
    bucket: str,
    platform: str,
    world_size: int,
    buckets: Sequence[str],
    phases: Sequence[str] = DEFAULT_PHASES,
    direct_tests: Sequence[str] = (),
    always_run_files: Sequence[str] = (),
) -> dict[str, Any]:
    """Union rank/phase node selections and emit one validated file list."""

    cache_root = Path(cache_dir).resolve()
    root = Path(repo_root).resolve()
    bucket = normalize_repo_path(bucket)
    if bucket not in {normalize_repo_path(value) for value in buckets}:
        raise TestmonValidationError(f"bucket is not present in the {platform} recipe: {bucket}")
    if world_size <= 0:
        raise TestmonValidationError("world_size must be positive")
    if not phases or len(phases) != len(set(phases)):
        raise TestmonValidationError("phases must be non-empty and unique")

    # Rank collections were proved equal while publishing the baseline, so
    # rank zero is sufficient to measure the current file-level test universe.
    # Missing current files are deleted tests and intentionally do not count.
    baseline_files: set[str] = set()
    for phase in phases:
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", phase):
            raise TestmonValidationError(f"invalid phase name: {phase!r}")
        baseline_source = cache_root / "collected" / phase / "rank-0.json"
        for nodeid in _load_nodeids(baseline_source):
            path = _validate_test_file(_test_file_from_nodeid(nodeid), root, must_exist=False)
            if not (root / path).is_file():
                continue
            if owning_bucket(path, buckets) == bucket:
                baseline_files.add(path)

    selected_nodes: set[str] = set()
    selection_sources: list[dict[str, Any]] = []
    for phase in phases:
        for rank in range(world_size):
            source = cache_root / "selection" / phase / f"rank-{rank}.json"
            nodeids = _load_nodeids(source)
            selected_nodes.update(nodeids)
            selection_sources.append({"phase": phase, "rank": rank, "node_count": len(nodeids)})

    selected_files: set[str] = set()
    for nodeid in selected_nodes:
        path = _validate_test_file(_test_file_from_nodeid(nodeid), root)
        owner = owning_bucket(path, buckets)
        if owner != bucket:
            raise TestmonValidationError(
                f"Testmon selected {path} outside effective bucket {bucket}; owner is {owner}"
            )
        selected_files.add(path)

    # Direct and always-run lists can be global. Include only paths belonging
    # to this runner's bucket, while validating every supplied path first.
    direct_files: set[str] = set()
    for path in direct_tests:
        normalized = _validate_test_file(path, root)
        if owning_bucket(normalized, buckets) == bucket:
            direct_files.add(normalized)
            selected_files.add(normalized)

    for path in always_run_files:
        normalized = _validate_test_file(path, root, must_exist=False)
        if not (root / normalized).is_file():
            continue
        if owning_bucket(normalized, buckets) == bucket:
            selected_files.add(normalized)

    eligible_files = baseline_files | direct_files
    unexpected_selected = selected_files - eligible_files
    if unexpected_selected:
        raise TestmonValidationError(
            "selected files are absent from the baseline and direct-test set: "
            f"{sorted(unexpected_selected)}"
        )
    selected_file_count = len(selected_files)
    eligible_file_count = len(eligible_files)
    selection_ratio = selected_file_count / eligible_file_count if eligible_file_count else 0.0

    return {
        "schema_version": SCHEMA_VERSION,
        "platform": platform,
        "world_size": world_size,
        "bucket": bucket,
        "selected_test_files": sorted(selected_files),
        "selected_test_file_count": selected_file_count,
        "eligible_test_file_count": eligible_file_count,
        "selection_ratio": selection_ratio,
        "selected_node_count": len(selected_nodes),
        "selection_sources": selection_sources,
    }


def prepare_matrix(
    index: Mapping[str, Any],
    *,
    repo_root: str | os.PathLike[str],
    platform: str,
    world_size: int,
    base_sha: str,
    current_sha: str,
    recipe: str | os.PathLike[str],
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Validate an index and classify its producer-to-PR diff fail-safely."""

    root = Path(repo_root)
    buckets = recipe_buckets(recipe)
    raw_index_buckets = index.get("buckets")
    index_buckets = raw_index_buckets if isinstance(raw_index_buckets, Mapping) else {}
    bucket_output = {
        bucket: {
            "cache_key": record.get("cache_key"),
            "manifest_sha256": record.get("manifest_sha256"),
            "database_checksums": record.get("database_checksums"),
            "direct_tests": [],
        }
        for bucket, record in sorted(index_buckets.items())
        if isinstance(record, dict)
    }
    reasons: list[str] = []
    classification: dict[str, Any] | None = None
    environment_hash: str | None = None
    cache_age_hours: float | None = None
    try:
        current_sha = _validate_sha(current_sha, "current_sha")
        environment_hash = compute_environment_hash(root, platform)
        validation_now = now or dt.datetime.now(dt.timezone.utc)
        validate_platform_index(
            index,
            repo_root=root,
            base_sha=base_sha,
            platform=platform,
            world_size=world_size,
            environment_hash=environment_hash,
            expected_buckets=buckets,
            now=validation_now,
        )
        producer_time = _parse_timestamp(str(index["producer_time"]))
        cache_age_hours = max(
            0.0, (validation_now.astimezone(dt.timezone.utc) - producer_time).total_seconds() / 3600
        )
        if not _git_is_ancestor(root, _validate_sha(base_sha, "base_sha"), current_sha):
            raise TestmonValidationError("PR base is not an ancestor of the tested commit")
        changes = git_diff_changes(root, index["producer_sha"], current_sha)
        classification = classify_changes(
            changes, tracked_python_files=index["tracked_python_files"], buckets=buckets
        )
        reasons.extend(classification["reasons"])
        for bucket, tests in classification["direct_tests_by_bucket"].items():
            bucket_output[bucket]["direct_tests"] = tests
    except TestmonValidationError as exc:
        reasons.append(str(exc))
    mode = "full" if reasons else "enforce"
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "reason": "; ".join(reasons) if reasons else "valid baseline and safely classified diff",
        "reasons": reasons,
        "platform": platform,
        "world_size": world_size,
        "producer_sha": index.get("producer_sha"),
        "base_sha": base_sha,
        "current_sha": current_sha,
        "environment_hash": environment_hash,
        "cache_age_hours": cache_age_hours,
        "buckets": bucket_output,
        "classification": classification,
    }


def _load_tracked_files(path: str | None) -> list[str] | None:
    if path is None:
        return None
    payload = _read_json(path)
    if isinstance(payload, dict):
        payload = payload.get("tracked_python_files")
    if not isinstance(payload, list) or any(not isinstance(item, str) for item in payload):
        raise TestmonValidationError("tracked-files JSON must be a list of strings")
    return payload


def _add_common_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", help="write JSON to this path instead of stdout")


def _emit(payload: Any, output: str | None) -> None:
    if output:
        _write_json(output, payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    environment = subparsers.add_parser("environment-hash")
    environment.add_argument("--repo-root", default=".")
    environment.add_argument("--platform", choices=("h100", "gb200"), required=True)
    environment.add_argument("--testmon-version", default=TESTMON_VERSION)
    environment.add_argument("--path", action="append", default=[], dest="extra_paths")
    environment.add_argument(
        "--identity", action="append", default=[], metavar="NAME=VALUE", dest="identities"
    )

    bucket = subparsers.add_parser("bucket-hash")
    bucket.add_argument("--bucket", required=True)

    manifest = subparsers.add_parser("build-manifest")
    manifest.add_argument("--cache-dir", required=True)
    manifest.add_argument("--repo-root", default=".")
    manifest.add_argument("--platform", choices=("h100", "gb200"), required=True)
    manifest.add_argument("--world-size", type=int, required=True)
    manifest.add_argument("--bucket", required=True)
    manifest.add_argument("--producer-sha", required=True)
    manifest.add_argument("--producer-time", required=True)
    manifest.add_argument("--environment-hash", required=True)
    manifest.add_argument("--container-identity", required=True)
    manifest.add_argument("--dependency-identity", required=True)
    manifest.add_argument("--topology-identity", required=True)
    manifest.add_argument("--phase", action="append", dest="phases")
    manifest.add_argument("--tracked-files-json")
    manifest.add_argument("--always-run", action="append", default=[])
    _add_common_output(manifest)

    validate_manifest = subparsers.add_parser("validate-manifest")
    validate_manifest.add_argument("--manifest", required=True)
    validate_manifest.add_argument("--cache-dir", required=True)
    validate_manifest.add_argument(
        "--index-record",
        "--trusted-index-record",
        dest="trusted_index_record",
        help="JSON index record used to bind this restored manifest and its databases",
    )
    validate_manifest.add_argument("--platform", choices=("h100", "gb200"), required=True)
    validate_manifest.add_argument("--world-size", type=int, required=True)
    validate_manifest.add_argument("--bucket", required=True)
    validate_manifest.add_argument("--environment-hash")
    _add_common_output(validate_manifest)

    index = subparsers.add_parser("build-index")
    index.add_argument("--manifest-root", required=True)
    index.add_argument("--recipe", required=True)
    index.add_argument("--platform", choices=("h100", "gb200"), required=True)
    index.add_argument("--world-size", type=int, required=True)
    index.add_argument("--environment-hash", required=True)
    index.add_argument(
        "--producer-time",
        help="trusted platform-index validation time; permits reused bucket manifest times",
    )
    _add_common_output(index)

    validate_index = subparsers.add_parser("validate-index")
    validate_index.add_argument("--index", required=True)
    validate_index.add_argument("--repo-root", default=".")
    validate_index.add_argument("--base-sha", required=True)
    validate_index.add_argument("--platform", choices=("h100", "gb200"), required=True)
    validate_index.add_argument("--world-size", type=int, required=True)
    validate_index.add_argument("--environment-hash", required=True)
    validate_index.add_argument("--recipe")
    validate_index.add_argument("--max-age-hours", type=float, default=DEFAULT_MAX_AGE_HOURS)
    _add_common_output(validate_index)

    classify = subparsers.add_parser("classify-diff")
    classify.add_argument("--index", required=True)
    classify.add_argument("--repo-root", default=".")
    classify.add_argument("--base-sha", required=True)
    classify.add_argument("--head-sha", required=True)
    classify.add_argument("--recipe", required=True)
    _add_common_output(classify)

    eligibility = subparsers.add_parser("decide-eligibility")
    eligibility.add_argument("--enabled", action="store_true")
    eligibility.add_argument("--event-name", required=True)
    eligibility.add_argument("--ref", required=True)
    eligibility.add_argument("--github-sha", required=True)
    eligibility.add_argument("--pr-head-sha", required=True)
    eligibility.add_argument("--metadata-valid", action="store_true")
    eligibility.add_argument("--label", action="append", default=[])
    eligibility.add_argument("--force-run-all", action="store_true")
    eligibility.add_argument("--container", default="dev")
    _add_common_output(eligibility)

    union = subparsers.add_parser("union-selection")
    union.add_argument("--cache-dir", required=True)
    union.add_argument("--repo-root", default=".")
    union.add_argument("--bucket", required=True)
    union.add_argument("--platform", choices=("h100", "gb200"), required=True)
    union.add_argument("--world-size", type=int, required=True)
    union.add_argument("--recipe")
    union.add_argument("--phase", action="append", dest="phases")
    union.add_argument("--direct-test", action="append", default=[])
    union.add_argument("--direct-tests-json")
    union.add_argument("--always-run", action="append", default=[])
    union.add_argument("--manifest")
    _add_common_output(union)

    prepare = subparsers.add_parser("prepare-matrix")
    prepare.add_argument("--index", required=True)
    prepare.add_argument("--repo-root", default=".")
    prepare.add_argument("--platform", choices=("h100", "gb200"), required=True)
    prepare.add_argument("--world-size", type=int, required=True)
    prepare.add_argument("--base-sha", required=True)
    prepare.add_argument("--current-sha", required=True)
    prepare.add_argument("--recipe", required=True)
    _add_common_output(prepare)
    return parser


def _parse_identities(values: Sequence[str]) -> dict[str, str]:
    identities: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise TestmonValidationError(f"identity must use NAME=VALUE syntax: {value!r}")
        name, identity = value.split("=", 1)
        if not name or not identity or name in identities:
            raise TestmonValidationError(f"invalid or duplicate identity: {value!r}")
        identities[name] = identity
    return identities


def _main(args: argparse.Namespace) -> None:
    if args.command == "environment-hash":
        print(
            compute_environment_hash(
                args.repo_root,
                args.platform,
                testmon_version=args.testmon_version,
                extra_paths=args.extra_paths,
                identities=_parse_identities(args.identities),
            )
        )
    elif args.command == "bucket-hash":
        print(bucket_hash(args.bucket))
    elif args.command == "build-manifest":
        payload = build_bucket_manifest(
            args.cache_dir,
            repo_root=args.repo_root,
            platform=args.platform,
            world_size=args.world_size,
            bucket=args.bucket,
            producer_sha=args.producer_sha,
            producer_time=args.producer_time,
            environment_hash=args.environment_hash,
            container_identity=args.container_identity,
            dependency_identity=args.dependency_identity,
            topology_identity=args.topology_identity,
            phases=args.phases or DEFAULT_PHASES,
            tracked_python_files=_load_tracked_files(args.tracked_files_json),
            always_run_files=args.always_run,
        )
        _emit(payload, args.output)
    elif args.command == "validate-manifest":
        payload = _read_json(args.manifest)
        trusted_index_record = (
            _read_json(args.trusted_index_record) if args.trusted_index_record else None
        )
        if trusted_index_record is not None and not isinstance(trusted_index_record, dict):
            raise TestmonValidationError("trusted index record JSON must be an object")
        validate_bucket_manifest(
            payload,
            cache_dir=args.cache_dir,
            manifest_path=args.manifest,
            trusted_index_record=trusted_index_record,
            expected_platform=args.platform,
            expected_world_size=args.world_size,
            expected_bucket=args.bucket,
            expected_environment_hash=args.environment_hash,
        )
        _emit({"valid": True, "cache_key": payload["cache_key"]}, args.output)
    elif args.command == "build-index":
        root = Path(args.manifest_root)
        paths = sorted(root.rglob("manifest.json"))
        if not paths:
            raise TestmonValidationError(f"no manifest.json files found under {root}")
        manifests = [(_read_json(path), path) for path in paths]
        payload = build_platform_index(
            manifests,
            expected_buckets=recipe_buckets(args.recipe),
            platform=args.platform,
            world_size=args.world_size,
            environment_hash=args.environment_hash,
            producer_time_override=args.producer_time,
        )
        _emit(payload, args.output)
    elif args.command == "validate-index":
        payload = _read_json(args.index)
        expected = recipe_buckets(args.recipe) if args.recipe else None
        validate_platform_index(
            payload,
            repo_root=args.repo_root,
            base_sha=args.base_sha,
            platform=args.platform,
            world_size=args.world_size,
            environment_hash=args.environment_hash,
            expected_buckets=expected,
            max_age_hours=args.max_age_hours,
        )
        _emit({"valid": True, "producer_sha": payload["producer_sha"]}, args.output)
    elif args.command == "classify-diff":
        index = _read_json(args.index)
        payload = classify_changes(
            git_diff_changes(args.repo_root, args.base_sha, args.head_sha),
            tracked_python_files=index["tracked_python_files"],
            buckets=recipe_buckets(args.recipe),
        )
        _emit(payload, args.output)
    elif args.command == "decide-eligibility":
        payload = decide_pr_eligibility(
            enabled=args.enabled,
            event_name=args.event_name,
            ref=args.ref,
            github_sha=args.github_sha,
            pr_head_sha=args.pr_head_sha,
            metadata_valid=args.metadata_valid,
            labels=args.label,
            force_run_all=args.force_run_all,
            container=args.container,
        )
        _emit(payload, args.output)
    elif args.command == "union-selection":
        direct_tests = list(args.direct_test)
        if args.direct_tests_json:
            direct_payload = _read_json(args.direct_tests_json)
            if isinstance(direct_payload, dict):
                direct_payload = direct_payload.get("direct_tests", direct_payload.get("tests"))
            if not isinstance(direct_payload, list) or any(
                not isinstance(item, str) for item in direct_payload
            ):
                raise TestmonValidationError("direct-tests JSON must be a list of strings")
            direct_tests.extend(direct_payload)
        always_run = list(args.always_run)
        if args.manifest:
            manifest = _read_json(args.manifest)
            always_run.extend(manifest.get("always_run_files", []))
        recipe = args.recipe or f"tests/test_utils/recipes/{args.platform}/unit-tests.yaml"
        payload = union_rank_selections(
            args.cache_dir,
            repo_root=args.repo_root,
            bucket=args.bucket,
            platform=args.platform,
            world_size=args.world_size,
            buckets=recipe_buckets(recipe),
            phases=args.phases or DEFAULT_PHASES,
            direct_tests=direct_tests,
            always_run_files=always_run,
        )
        _emit(payload, args.output)
    elif args.command == "prepare-matrix":
        payload = prepare_matrix(
            _read_json(args.index),
            repo_root=args.repo_root,
            platform=args.platform,
            world_size=args.world_size,
            base_sha=args.base_sha,
            current_sha=args.current_sha,
            recipe=args.recipe,
        )
        _emit(payload, args.output)
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(args.command)


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _main(args)
    except TestmonValidationError as exc:
        print(f"unit-testmon: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
