# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Small, fail-closed pytest-testmon adapter for unit-test CI."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = 1
TESTMON_VERSION = "2.2.0"
PHASES = ("prod", "experimental")
MAX_AGE_HOURS = 72.0
UNIT_ROOT = "tests/unit_tests"
SHA_RE = re.compile(r"[0-9a-f]{40}")
HASH_RE = re.compile(r"[0-9a-f]{64}")


class SelectionError(ValueError):
    """An uncertainty which requires the caller to run the full bucket."""


def _safe_path(value: str) -> str:
    raw = value.replace("\\", "/")
    path = PurePosixPath(raw)
    if (
        not value
        or any(character in value for character in "\0\r\n")
        or path.is_absolute()
        or raw != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SelectionError(f"unsafe repository path: {value!r}")
    return path.as_posix()


def _safe_file(root: Path, value: str) -> Path:
    relative = _safe_path(value)
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise SelectionError(f"path escapes repository: {value!r}") from error
    if not candidate.is_file():
        raise SelectionError(f"file is missing: {relative}")
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SelectionError(f"cannot read {path}: {error}") from error


def _sha(value: str, label: str) -> str:
    if not SHA_RE.fullmatch(value):
        raise SelectionError(f"{label} is not a full lowercase git SHA")
    return value


def _timestamp(value: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SelectionError(f"invalid producer timestamp: {value!r}") from error
    if parsed.tzinfo is None:
        raise SelectionError("producer timestamp has no timezone")
    return parsed.astimezone(dt.timezone.utc)


def bucket_hash(bucket: str) -> str:
    return hashlib.sha256(("unit-testmon-bucket-v2\0" + _safe_path(bucket)).encode()).hexdigest()[
        :16
    ]


def _config_paths(root: Path, platform: str) -> list[str]:
    if platform not in {"h100", "gb200"}:
        raise SelectionError(f"unsupported platform: {platform}")
    paths = {
        "pyproject.toml",
        "uv.lock",
        "docker/.ngc_version.dev",
        "docker/Dockerfile.ci.dev",
        "docker/common/install.sh",
        ".github/actions/action.yml",
        ".github/workflows/cicd-main.yml",
        ".github/workflows/unit-testmon-baseline.yml",
        f"tests/test_utils/recipes/{platform}/unit-tests.yaml",
        "tests/test_utils/python_scripts/launch_nemo_run_workload.py",
        "tests/unit_tests/run_ci_test.sh",
        "tests/unit_tests/conftest.py",
        "tests/unit_tests/find_test_cases.py",
        "tests/unit_tests/testmon_selector.py",
    }
    unit_root = root / UNIT_ROOT
    if unit_root.is_dir():
        paths.update(path.relative_to(root).as_posix() for path in unit_root.rglob("conftest.py"))
    return sorted(paths)


def config_hash(
    root: Path,
    platform: str,
    world_size: int,
    bucket: str,
    identities: tuple[str, ...] | list[str] = (),
) -> str:
    if world_size < 1:
        raise SelectionError("world size must be positive")
    digest = hashlib.sha256()
    digest.update(
        f"unit-testmon-config-v2\0testmon={TESTMON_VERSION}\0{platform}\0r{world_size}\0"
        f"{_safe_path(bucket)}\0".encode()
    )
    for relative in _config_paths(root, platform):
        digest.update(relative.encode() + b"\0")
        digest.update(bytes.fromhex(_sha256(_safe_file(root, relative))))
    for identity in sorted(identities):
        if "=" not in identity:
            raise SelectionError("identities must use name=value")
        digest.update(b"identity:" + identity.encode() + b"\0")
    return digest.hexdigest()


def _sqlite_ok(path: Path, *, checkpoint: bool) -> None:
    if not path.is_file():
        raise SelectionError(f"missing Testmon database: {path}")
    try:
        target = str(path) if checkpoint else f"file:{path.resolve()}?mode=ro"
        with sqlite3.connect(target, uri=not checkpoint) as connection:
            if checkpoint:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
            result = connection.execute("PRAGMA integrity_check").fetchall()
    except sqlite3.Error as error:
        raise SelectionError(f"invalid Testmon database {path}: {error}") from error
    if result != [("ok",)]:
        raise SelectionError(f"SQLite integrity check failed for {path}")


def _node_file(nodeid: str, root: Path, *, must_exist: bool = True) -> str:
    if not isinstance(nodeid, str):
        raise SelectionError("pytest node IDs must be strings")
    relative = _safe_path(nodeid.split("::", 1)[0])
    path = PurePosixPath(relative)
    if (
        path.parts[:2] != ("tests", "unit_tests")
        or path.suffix != ".py"
        or not path.name.startswith("test_")
    ):
        raise SelectionError(f"node ID is outside unit tests: {nodeid!r}")
    if must_exist:
        _safe_file(root, relative)
    return relative


def _load_nodes(path: Path, phase: str) -> list[str]:
    value = _read_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise SelectionError(f"unsupported node-list schema: {path}")
    nodes = value.get("nodeids")
    if value.get("phase") != phase or not isinstance(nodes, list):
        raise SelectionError(f"invalid node-list identity: {path}")
    if nodes != sorted(set(nodes)) or not all(isinstance(item, str) for item in nodes):
        raise SelectionError(f"node IDs must be sorted unique strings: {path}")
    return nodes


def _unattributed(database: Path, collected: set[str]) -> set[str]:
    try:
        with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            required = {"test_execution", "test_execution_file_fp", "file_fp"}
            if not required <= tables:
                raise SelectionError(f"{database} is missing Testmon dependency tables")
            rows = connection.execute(
                "SELECT te.test_name, f.filename FROM test_execution te "
                "LEFT JOIN test_execution_file_fp l ON te.id=l.test_execution_id "
                "LEFT JOIN file_fp f ON l.fingerprint_id=f.id"
            ).fetchall()
    except sqlite3.Error as error:
        raise SelectionError(f"cannot inspect dependencies in {database}: {error}") from error
    attributed = {
        test
        for test, filename in rows
        if isinstance(test, str)
        and isinstance(filename, str)
        and filename.replace("\\", "/").startswith("megatron/")
    }
    return collected - attributed


def _subprocess_tests(root: Path, collected: set[str]) -> set[str]:
    """Return files whose child-process execution Testmon cannot trace."""
    files = {_node_file(node, root) for node in collected}
    pattern = re.compile(
        r"^\s*(?:import|from)\s+(?:subprocess|(?:torch\.)?multiprocessing)\b", re.MULTILINE
    )
    return {path for path in files if pattern.search((root / path).read_text(errors="replace"))}


def finalize(args: argparse.Namespace) -> Path:
    root, cache = args.repo_root.resolve(), args.cache_dir.resolve()
    if args.world_size < 1:
        raise SelectionError("world size must be positive")
    if not HASH_RE.fullmatch(args.config_hash):
        raise SelectionError("config hash is not a lowercase SHA-256 digest")
    all_nodes: set[str] = set()
    attributed_somewhere: set[str] = set()
    databases, collections = {}, {}
    owned = _bucket_files(root, args.platform, args.bucket)
    for phase in PHASES:
        database = cache / f"{phase}.testmondata"
        collection = cache / "collected" / f"{phase}.json"
        _sqlite_ok(database, checkpoint=True)
        nodes = set(_load_nodes(collection, phase))
        outside = {_node_file(node, root) for node in nodes} - owned
        if outside:
            raise SelectionError(f"baseline collection escapes effective bucket: {sorted(outside)}")
        all_nodes.update(nodes)
        attributed_somewhere.update(nodes - _unattributed(database, nodes))
        databases[phase] = {"path": database.name, "sha256": _sha256(database)}
        collections[phase] = {
            "path": collection.relative_to(cache).as_posix(),
            "sha256": _sha256(collection),
            "node_count": len(nodes),
        }
    always = (
        {_node_file(node, root) for node in all_nodes - attributed_somewhere}
        | _subprocess_tests(root, all_nodes)
    ) & owned
    basic = root / "tests/unit_tests/test_basic.py"
    if basic.is_file() and "tests/unit_tests/test_basic.py" in owned:
        always.add("tests/unit_tests/test_basic.py")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "testmon_version": TESTMON_VERSION,
        "producer_sha": _sha(args.producer_sha, "producer SHA"),
        "producer_time": _timestamp(args.producer_time).isoformat().replace("+00:00", "Z"),
        "platform": args.platform,
        "world_size": args.world_size,
        "bucket": _safe_path(args.bucket),
        "config_hash": args.config_hash,
        "databases": databases,
        "collections": collections,
        "always_run_files": sorted(always),
    }
    output = (args.output or cache / "metadata.json").resolve()
    _write_json(output, metadata)
    return output


def _recipe_buckets(root: Path, platform: str) -> list[str]:
    recipe = _safe_file(root, f"tests/test_utils/recipes/{platform}/unit-tests.yaml").read_text()
    values = re.findall(r"^\s*-\s+test_case:\s*\[([^\]]+)]\s*$", recipe, re.MULTILINE)
    buckets = [value.strip().strip("'\"") for value in values]
    if not buckets:
        raise SelectionError(f"unit-test recipe contains no buckets for {platform}")
    return buckets


def _expand(root: Path, pattern: str) -> set[str]:
    pattern = _safe_path(pattern)
    if "/**/" in pattern:
        base, leaf = pattern.split("/**/", 1)
        paths = (root / base).rglob(leaf)
    elif "*" in PurePosixPath(pattern).name:
        base, leaf = pattern.rsplit("/", 1)
        paths = (root / base).glob(leaf)
    else:
        paths = [root / pattern]
    return {path.relative_to(root).as_posix() for path in paths if path.is_file()}


def _base(pattern: str) -> str:
    if "/**" in pattern:
        return pattern.split("/**", 1)[0]
    if "*" in pattern:
        return pattern.rsplit("/", 1)[0]
    return pattern.rstrip("/")


def _bucket_files(root: Path, platform: str, bucket: str) -> set[str]:
    buckets = _recipe_buckets(root, platform)
    if bucket not in buckets:
        raise SelectionError(f"bucket is absent from the {platform} recipe: {bucket}")
    files = _expand(root, bucket)
    parent = _base(bucket)
    for child in buckets:
        if child != bucket and _base(child).startswith(parent + "/"):
            files.difference_update(_expand(root, child))
    if platform == "gb200":
        files = {
            path
            for path in files
            if not PurePosixPath(path).name.startswith("test_")
            or "launch_on_gb200" in (root / path).read_text(errors="replace")
        }
    return {
        path
        for path in files
        if PurePosixPath(path).name.startswith("test_") and path.endswith(".py")
    }


def _git(root: Path, *arguments: str, capture: bool = False) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise SelectionError(result.stderr.strip() or f"git {' '.join(arguments)} failed")
    return result.stdout if capture else ""


def _changed_paths(root: Path, producer: str, head: str) -> list[str]:
    output = _git(root, "diff", "--name-status", "-z", producer, head, capture=True)
    fields = output.split("\0")
    if fields[-1:] == [""]:
        fields.pop()
    if len(fields) % 2:
        raise SelectionError("malformed git name-status output")
    paths = []
    for status, raw_path in zip(fields[::2], fields[1::2]):
        path = _safe_path(raw_path)
        if status != "M":
            raise SelectionError(f"{status} change requires full testing: {path}")
        if not path.endswith(".py") or not (
            path.startswith("megatron/") or path.startswith(UNIT_ROOT + "/")
        ):
            raise SelectionError(f"changed path requires full testing: {path}")
        if path.startswith(UNIT_ROOT + "/") and not PurePosixPath(path).name.startswith("test_"):
            raise SelectionError(f"modified unit-test helper requires full testing: {path}")
        if not (root / path).is_file():
            raise SelectionError(f"changed Python file is missing: {path}")
        paths.append(path)
    return paths


def _validate(args: argparse.Namespace) -> tuple[dict, set[str], list[str]]:
    root, cache = args.repo_root.resolve(), args.cache_dir.resolve()
    metadata_path = (args.metadata or cache / "metadata.json").resolve()
    value = _read_json(metadata_path)
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise SelectionError("unsupported baseline metadata schema")
    if not HASH_RE.fullmatch(args.config_hash) or args.max_age_hours <= 0:
        raise SelectionError("invalid selector hash or maximum age")
    expected = {
        "testmon_version": TESTMON_VERSION,
        "platform": args.platform,
        "world_size": args.world_size,
        "bucket": _safe_path(args.bucket),
        "config_hash": args.config_hash,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise SelectionError(f"baseline {key} mismatch")
    age = dt.datetime.now(dt.timezone.utc) - _timestamp(str(value.get("producer_time", "")))
    if age.total_seconds() < -300 or age > dt.timedelta(hours=args.max_age_hours):
        raise SelectionError("baseline is stale or dated in the future")
    producer = _sha(str(value.get("producer_sha", "")), "producer SHA")
    base, head = _sha(args.base_sha, "base SHA"), _sha(args.head_sha, "head SHA")
    _git(root, "merge-base", "--is-ancestor", producer, base)
    databases, collections = value.get("databases"), value.get("collections")
    if not isinstance(databases, dict) or not isinstance(collections, dict):
        raise SelectionError("baseline artifact records are missing")
    for phase in PHASES:
        database, collection = cache / f"{phase}.testmondata", cache / "collected" / f"{phase}.json"
        for records, path in ((databases, database), (collections, collection)):
            record = records.get(phase)
            if not isinstance(record, dict) or record.get("sha256") != _sha256(path):
                raise SelectionError(f"baseline checksum mismatch: {path}")
        _sqlite_ok(database, checkpoint=False)
        _load_nodes(collection, phase)
    changed = _changed_paths(root, producer, head)
    return value, _bucket_files(root, args.platform, args.bucket), changed


def select(args: argparse.Namespace) -> list[str]:
    metadata, owned, changed = _validate(args)
    if args.validate_only:
        return []
    root, cache = args.repo_root.resolve(), args.cache_dir.resolve()
    selected_nodes = {
        node
        for phase in PHASES
        for node in _load_nodes(cache / "selection" / f"{phase}.json", phase)
    }
    selected = {_node_file(node, root) for node in selected_nodes}
    direct = {
        path
        for path in changed
        if path.startswith(UNIT_ROOT + "/")
        and PurePosixPath(path).name.startswith("test_")
        and path in owned
    }
    selected.update(direct)
    always = metadata.get("always_run_files")
    if not isinstance(always, list) or not all(isinstance(path, str) for path in always):
        raise SelectionError("baseline always-run list is invalid")
    selected.update(
        _node_file(path, root) for path in always if (root / _safe_path(path)).is_file()
    )
    outside = selected - owned
    if outside:
        raise SelectionError(f"selected files escape effective bucket: {sorted(outside)}")
    eligible = (
        {
            _node_file(node, root, must_exist=False)
            for phase in PHASES
            for node in _load_nodes(cache / "collected" / f"{phase}.json", phase)
            if (root / _node_file(node, root, must_exist=False)).is_file()
        }
        & owned
    ) | direct
    result = sorted(selected)
    output = args.output or cache / "selected.json"
    _write_json(
        output,
        {
            "schema_version": SCHEMA_VERSION,
            "selected_files": result,
            "eligible_file_count": len(eligible),
            "selection_ratio": len(result) / len(eligible) if eligible else 0.0,
        },
    )
    return result


class _Recorder:
    def __init__(self, output: Path, phase: str):
        self.output, self.phase = output, phase

    def pytest_collection_finish(self, session) -> None:
        _write_json(
            self.output,
            {
                "schema_version": SCHEMA_VERSION,
                "phase": self.phase,
                "nodeids": sorted(item.nodeid for item in session.items),
            },
        )


def _run(args: argparse.Namespace) -> int:
    try:
        rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise SelectionError("run must be launched by torchrun") from error
    if rank < 0 or world_size < 1 or rank >= world_size:
        raise SelectionError("invalid torchrun rank or world size")
    cache, pytest_args = args.cache_dir.resolve(), list(args.pytest_args)
    if pytest_args[:1] == ["--"]:
        pytest_args.pop(0)
    if not pytest_args:
        raise SelectionError("pytest arguments are required after --")
    recorder = None
    if rank == 0:
        database = cache / f"{args.phase}.testmondata"
        output = (
            cache / ("collected" if args.mode == "baseline" else "selection") / f"{args.phase}.json"
        )
        if args.mode == "baseline":
            for suffix in ("", "-wal", "-shm"):
                Path(f"{database}{suffix}").unlink(missing_ok=True)
            database.parent.mkdir(parents=True, exist_ok=True)
            pytest_args.extend(("--testmon", "--testmon-noselect"))
        else:
            if not database.is_file():
                raise SelectionError(f"missing Testmon database: {database}")
            disposable = cache / ".selection-work" / f"{args.phase}.testmondata"
            disposable.parent.mkdir(parents=True, exist_ok=True)
            for suffix in ("", "-wal", "-shm"):
                Path(f"{disposable}{suffix}").unlink(missing_ok=True)
            shutil.copy2(database, disposable)
            database = disposable
            pytest_args.extend(
                ("--collect-only", "--testmon", "--testmon-nocollect", "--testmon-forceselect")
            )
        os.environ["TESTMON_DATAFILE"] = str(database)
        output.unlink(missing_ok=True)
        recorder = _Recorder(output, args.phase)
    else:
        os.environ.pop("TESTMON_DATAFILE", None)
        pytest_args.extend(("-p", "no:testmon", "-p", "no:pytest-testmon"))
        if args.mode == "select":
            pytest_args.append("--collect-only")
    script_dir = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if Path(entry or Path.cwd()).resolve() != script_dir]
    sys.path.insert(0, str(script_dir.parents[1]))
    import pytest

    result = int(pytest.main(pytest_args, plugins=[recorder] if recorder else []))
    return 0 if result == 5 else result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser("config-hash")
    config.add_argument("--repo-root", type=Path, default=Path.cwd())
    config.add_argument("--platform", choices=("h100", "gb200"), required=True)
    config.add_argument("--world-size", type=int, required=True)
    config.add_argument("--bucket", required=True)
    config.add_argument("--identity", action="append", default=[])
    bucket = commands.add_parser("bucket-hash")
    bucket.add_argument("--bucket", required=True)
    run = commands.add_parser("run")
    run.add_argument("--mode", choices=("baseline", "select"), required=True)
    run.add_argument("--cache-dir", type=Path, required=True)
    run.add_argument("--phase", choices=PHASES, required=True)
    run.add_argument("pytest_args", nargs=argparse.REMAINDER)
    final = commands.add_parser("finalize")
    for target in (final,):
        target.add_argument("--repo-root", type=Path, default=Path.cwd())
        target.add_argument("--cache-dir", type=Path, required=True)
        target.add_argument("--platform", choices=("h100", "gb200"), required=True)
        target.add_argument("--world-size", type=int, required=True)
        target.add_argument("--bucket", required=True)
        target.add_argument("--config-hash", required=True)
    final.add_argument("--producer-sha", required=True)
    final.add_argument("--producer-time", required=True)
    final.add_argument("--output", type=Path)
    selection = commands.add_parser("select")
    selection.add_argument("--repo-root", type=Path, default=Path.cwd())
    selection.add_argument("--cache-dir", type=Path, required=True)
    selection.add_argument("--metadata", type=Path)
    selection.add_argument("--platform", choices=("h100", "gb200"), required=True)
    selection.add_argument("--world-size", type=int, required=True)
    selection.add_argument("--bucket", required=True)
    selection.add_argument("--config-hash", required=True)
    selection.add_argument("--base-sha", required=True)
    selection.add_argument("--head-sha", required=True)
    selection.add_argument("--max-age-hours", type=float, default=MAX_AGE_HOURS)
    selection.add_argument("--validate-only", action="store_true")
    selection.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "config-hash":
            print(
                config_hash(
                    args.repo_root.resolve(),
                    args.platform,
                    args.world_size,
                    args.bucket,
                    args.identity,
                )
            )
        elif args.command == "bucket-hash":
            print(bucket_hash(args.bucket))
        elif args.command == "run":
            return _run(args)
        elif args.command == "finalize":
            print(finalize(args))
        else:
            for path in select(args):
                print(path)
        return 0
    except (OSError, SelectionError) as error:
        print(f"testmon selector: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
