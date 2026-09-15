# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run pytest with a rank-zero baseline or a per-rank Testmon copy."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

PHASES = ("prod", "experimental")


class _SelectedCount:
    def __init__(self, output: Path):
        self.output = output

    def pytest_collection_finish(self, session) -> None:
        self.output.write_text(f"{len(session.items)}\n")


def _database(cache_dir: Path, phase: str) -> Path:
    return cache_dir.resolve() / phase / ".testmondata"


def _clear_database_files(database: Path) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    for path in database.parent.glob(f"{database.name}*"):
        if path.is_file():
            path.unlink()


def _copy_database(cache_dir: Path, phase: str, rank: int) -> Path:
    source = _database(cache_dir, phase)
    if not source.is_file():
        raise RuntimeError(f"missing Testmon baseline: {source}")

    destination = cache_dir.resolve() / ".testmon-work" / phase / f"rank-{rank}" / source.name
    _clear_database_files(destination)
    for path in source.parent.glob(f"{source.name}*"):
        if path.is_file():
            shutil.copy2(path, destination.parent / path.name)
    return destination


def _run(args: argparse.Namespace) -> int:
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise RuntimeError("the Testmon wrapper must be launched by torchrun") from error
    if rank < 0 or world_size < 1 or rank >= world_size:
        raise RuntimeError("invalid torchrun rank or world size")

    pytest_args = list(args.pytest_args)
    if pytest_args[:1] == ["--"]:
        pytest_args.pop(0)
    if not pytest_args:
        raise RuntimeError("pytest arguments are required after --")

    count_plugin = None
    exit_code_file = None
    if args.mode == "baseline":
        if rank == 0:
            database = _database(args.cache_dir, args.phase)
            _clear_database_files(database)
            os.environ["TESTMON_DATAFILE"] = str(database)
            pytest_args.extend(("--testmon", "--testmon-noselect"))
        else:
            os.environ.pop("TESTMON_DATAFILE", None)
            pytest_args.extend(("-p", "no:testmon", "-p", "no:pytest-testmon"))
    else:
        database = _copy_database(args.cache_dir, args.phase, rank)
        os.environ["TESTMON_DATAFILE"] = str(database)
        count_file = database.parent / "selected-count"
        count_file.unlink(missing_ok=True)
        exit_code_file = database.parent / "pytest-exit-code"
        exit_code_file.unlink(missing_ok=True)
        count_plugin = _SelectedCount(count_file)
        pytest_args.extend(("--testmon", "--testmon-nocollect", "--testmon-forceselect"))

    # Avoid importing modules from tests/unit_tests in place of dependencies.
    script_dir = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if Path(entry or Path.cwd()).resolve() != script_dir]
    sys.path.insert(0, str(script_dir.parents[1]))

    import pytest

    result = int(pytest.main(pytest_args, plugins=[count_plugin] if count_plugin else []))
    if exit_code_file is not None:
        exit_code_file.write_text(f"{result}\n")
    return 0 if result == 5 else result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "enforce"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(_parser().parse_args(argv))
    except (OSError, RuntimeError) as error:
        print(f"Testmon wrapper: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
