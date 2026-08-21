# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Per-rank pytest launcher and collection recorder for Testmon CI runs.

This module is launched by ``torch.distributed.run``.  Resolving the database
path here (rather than in the parent shell) is important: only the child
process knows its global distributed rank, and Testmon does not provide a
command-line option for its data file.  Its supported ``TESTMON_DATAFILE``
environment variable must be set before pytest initializes the plugin.

The ordinary unit-test path does not import this module.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Sequence

SCHEMA_VERSION = 1
PHASES = ("prod", "experimental")


def _distributed_identity() -> tuple[int, int]:
    """Return the global rank and world size supplied by torchrun."""
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise RuntimeError("Testmon collection must be launched by torchrun") from error
    if rank < 0 or world_size < 1 or rank >= world_size:
        raise RuntimeError(f"Invalid torchrun topology: rank={rank}, world_size={world_size}")
    return rank, world_size


def _database_path(cache_dir: Path, phase: str, rank: int) -> Path:
    return cache_dir / phase / f"rank-{rank}.testmondata"


def _node_output_path(cache_dir: Path, phase: str, rank: int, mode: str) -> Path:
    root = "collected" if mode == "baseline" else "selection"
    return cache_dir / root / phase / f"rank-{rank}.json"


def _remove_sqlite_files(database: Path) -> None:
    """Remove one rank's database and SQLite sidecars, if present."""
    for suffix in ("", "-wal", "-shm"):
        path = Path(f"{database}{suffix}")
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _prepare_database(cache_dir: Path, phase: str, rank: int, mode: str) -> Path:
    baseline_database = _database_path(cache_dir, phase, rank)
    if mode == "baseline":
        baseline_database.parent.mkdir(parents=True, exist_ok=True)
        _remove_sqlite_files(baseline_database)
        return baseline_database

    if not baseline_database.is_file():
        raise FileNotFoundError(f"Missing Testmon database: {baseline_database}")

    disposable = cache_dir / ".selection-work" / phase / baseline_database.name
    disposable.parent.mkdir(parents=True, exist_ok=True)
    _remove_sqlite_files(disposable)
    shutil.copy2(baseline_database, disposable)
    disposable.chmod(disposable.stat().st_mode | stat.S_IWUSR)
    return disposable


def _atomic_json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _prepare_pytest_import_path() -> None:
    """Keep the selector package from shadowing the installed Testmon plugin.

    Executing this file directly puts ``tests/unit_tests`` first on
    ``sys.path``. That directory contains Megatron's selector directory named
    ``testmon``, while pytest-testmon exposes a top-level ``testmon`` package.
    Remove the script directory and prepend the repository root before pytest
    loads third-party entry points.
    """
    script_directory = Path(__file__).resolve().parent
    repo_root = script_directory.parents[1]
    sys.path[:] = [
        entry for entry in sys.path if Path(entry or Path.cwd()).resolve() != script_directory
    ]
    sys.path.insert(0, str(repo_root))


class SelectedNodesPlugin:
    """Record the final pytest collection after Testmon has deselected tests."""

    def __init__(self, output: Path, phase: str, rank: int, world_size: int):
        self.output = output
        self.phase = phase
        self.rank = rank
        self.world_size = world_size

    def pytest_collection_finish(self, session) -> None:
        _atomic_json_dump(
            self.output,
            {
                "schema_version": SCHEMA_VERSION,
                "phase": self.phase,
                "rank": self.rank,
                "world_size": self.world_size,
                "nodeids": sorted(item.nodeid for item in session.items),
            },
        )


def _valid_repo_test_file(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("selected test paths must be non-empty strings")
    if any(character in value for character in ("\0", "\n", "\r")):
        raise ValueError("selected test paths cannot contain control characters")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"selected test path must be repository-relative: {value!r}")
    if path.suffix != ".py" or path.parts[:2] != ("tests", "unit_tests"):
        raise ValueError(f"selected path is not a unit-test Python file: {value!r}")
    return path.as_posix()


def load_selected_manifest(manifest: Path, repo_root: Path) -> tuple[list[str], int, float]:
    """Load and defensively validate the selector's complete output manifest."""
    value = json.loads(manifest.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("selected-test manifest has an unsupported schema")

    files = value.get("selected_test_files")
    if not isinstance(files, list):
        raise ValueError("selected-test manifest must contain a selected_test_files list")

    validated = [_valid_repo_test_file(item) for item in files]
    if validated != sorted(set(validated)):
        raise ValueError("selected-test files must be unique and sorted")
    for relative in validated:
        resolved = (repo_root / relative).resolve()
        try:
            resolved.relative_to(repo_root.resolve())
        except ValueError as error:
            raise ValueError(f"selected test escapes the repository: {relative!r}") from error
        if not resolved.is_file():
            raise ValueError(f"selected test does not exist: {relative!r}")

    selected_count = value.get("selected_test_file_count")
    if isinstance(selected_count, bool) or not isinstance(selected_count, int):
        raise ValueError("selected_test_file_count must be an integer")
    if selected_count != len(validated):
        raise ValueError("selected_test_file_count does not match selected_test_files")

    eligible_count = value.get("eligible_test_file_count")
    if isinstance(eligible_count, bool) or not isinstance(eligible_count, int):
        raise ValueError("eligible_test_file_count must be an integer")
    if eligible_count < len(validated):
        raise ValueError("eligible_test_file_count cannot be smaller than the selection")

    selection_ratio = value.get("selection_ratio")
    if isinstance(selection_ratio, bool) or not isinstance(selection_ratio, (int, float)):
        raise ValueError("selection_ratio must be a number")
    selection_ratio = float(selection_ratio)
    if not math.isfinite(selection_ratio) or not 0.0 <= selection_ratio <= 1.0:
        raise ValueError("selection_ratio must be finite and between zero and one")
    expected_ratio = len(validated) / eligible_count if eligible_count else 0.0
    if not math.isclose(selection_ratio, expected_ratio, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("selection_ratio does not match selected and eligible file counts")

    return validated, eligible_count, selection_ratio


def load_selected_test_files(manifest: Path, repo_root: Path) -> list[str]:
    """Load the file list from a defensively validated selector manifest."""
    return load_selected_manifest(manifest, repo_root)[0]


def verify_baseline_artifacts(cache_dir: Path, world_size: int) -> None:
    """Checkpoint and validate every baseline DB and per-rank collection."""
    if world_size < 1:
        raise ValueError("world size must be positive")

    for phase in PHASES:
        expected_nodeids: list[str] | None = None
        for rank in range(world_size):
            node_file = _node_output_path(cache_dir, phase, rank, "baseline")
            value = json.loads(node_file.read_text())
            if not isinstance(value, dict):
                raise ValueError(f"invalid collection document: {node_file}")
            if value.get("phase") != phase or value.get("rank") != rank:
                raise ValueError(f"collection identity mismatch: {node_file}")
            if value.get("world_size") != world_size:
                raise ValueError(f"collection topology mismatch: {node_file}")
            nodeids = value.get("nodeids")
            if not isinstance(nodeids, list) or not all(isinstance(item, str) for item in nodeids):
                raise ValueError(f"invalid collected node IDs: {node_file}")
            if nodeids != sorted(set(nodeids)):
                raise ValueError(f"collected node IDs must be unique and sorted: {node_file}")
            if expected_nodeids is None:
                expected_nodeids = nodeids
            elif nodeids != expected_nodeids:
                raise ValueError(f"collected node IDs differ across ranks in phase {phase}")

            database = _database_path(cache_dir, phase, rank)
            if not database.is_file():
                raise FileNotFoundError(f"missing baseline database: {database}")
            with sqlite3.connect(database) as connection:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
                integrity = connection.execute("PRAGMA integrity_check").fetchall()
            if integrity != [("ok",)]:
                raise ValueError(f"SQLite integrity check failed for {database}: {integrity}")


def _run_pytest(mode: str, cache_dir: Path, phase: str, pytest_args: Sequence[str]) -> int:
    rank, world_size = _distributed_identity()
    database = _prepare_database(cache_dir, phase, rank, mode)
    output = _node_output_path(cache_dir, phase, rank, mode)
    try:
        output.unlink()
    except FileNotFoundError:
        pass

    os.environ["TESTMON_DATAFILE"] = str(database.resolve())

    # Import pytest only in the dedicated Testmon child process.  In
    # particular, the runner's ordinary/full path never imports Testmon.
    _prepare_pytest_import_path()
    import pytest

    plugin = SelectedNodesPlugin(output, phase, rank, world_size)
    args = list(pytest_args)
    if mode == "baseline":
        args.extend(("--testmon", "--testmon-noselect"))
    else:
        args.extend(("--collect-only", "--testmon", "--testmon-nocollect", "--testmon-forceselect"))
    return int(pytest.main(args, plugins=[plugin]))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="launch one per-rank pytest phase")
    run_parser.add_argument("--mode", choices=("baseline", "select"), required=True)
    run_parser.add_argument("--cache-dir", type=Path, required=True)
    run_parser.add_argument("--phase", choices=PHASES, required=True)
    run_parser.add_argument("pytest_args", nargs=argparse.REMAINDER)

    manifest_parser = commands.add_parser(
        "selected-files", help="validate a selected-file manifest and print one path per line"
    )
    manifest_parser.add_argument("--manifest", type=Path, required=True)
    manifest_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    manifest_parser.add_argument(
        "--include-summary-metrics",
        action="store_true",
        help="print eligible file count and formatted selection percentage before file paths",
    )

    verify_parser = commands.add_parser(
        "verify-baseline", help="validate per-rank baseline databases and collections"
    )
    verify_parser.add_argument("--cache-dir", type=Path, required=True)
    verify_parser.add_argument("--world-size", type=int, required=True)

    args = parser.parse_args(argv)
    if args.command == "run":
        if args.pytest_args[:1] == ["--"]:
            args.pytest_args = args.pytest_args[1:]
        if not args.pytest_args:
            parser.error("pytest arguments are required after '--'")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "selected-files":
        test_files, eligible_count, selection_ratio = load_selected_manifest(
            args.manifest, args.repo_root
        )
        if args.include_summary_metrics:
            print(eligible_count)
            print(f"{selection_ratio:.2%}")
        for test_file in test_files:
            print(test_file)
        return 0
    if args.command == "verify-baseline":
        verify_baseline_artifacts(args.cache_dir, args.world_size)
        return 0
    return _run_pytest(args.mode, args.cache_dir.resolve(), args.phase, args.pytest_args)


if __name__ == "__main__":
    sys.exit(main())
