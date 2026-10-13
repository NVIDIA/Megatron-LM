# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Generate a rank-zero Testmon baseline or select tests from a private copy."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from importlib.metadata import distributions
from pathlib import Path

PHASES = ("prod", "experimental")


class _SelectionOutput:
    def __init__(self, output: Path):
        self.output = output

    def pytest_collection_finish(self, session) -> None:
        selected_tests = sorted({item.nodeid for item in session.items})
        self.output.write_text("".join(f"{nodeid}\n" for nodeid in selected_tests))


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
            shutil.copyfile(path, destination.parent / path.name)
    return destination


def _testmon_dependency_override() -> str:
    from testmon_cache import is_tracked_package

    installed_packages = {
        name for distribution in distributions() if (name := distribution.metadata["Name"])
    }
    ignored_packages = sorted(name for name in installed_packages if not is_tracked_package(name))
    return f"testmon_ignore_dependencies={' '.join(ignored_packages)}"


def _run(args: argparse.Namespace) -> int:
    # Spawned workers reload this script after its directory leaves sys.path.
    from testmon_cache import record_phase, validate_phase

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

    selection_plugin = None
    if args.mode == "baseline":
        if rank == 0:
            database = _database(args.cache_dir, args.phase)
            _clear_database_files(database)
            (database.parent / "metadata.json").unlink(missing_ok=True)
            os.environ["TESTMON_DATAFILE"] = str(database)
            pytest_args.extend(
                ("-o", _testmon_dependency_override(), "--testmon", "--testmon-noselect")
            )
        else:
            os.environ.pop("TESTMON_DATAFILE", None)
            pytest_args.extend(("-p", "no:testmon", "-p", "no:pytest-testmon"))
    else:
        validate_phase(args.cache_dir, args.phase)
        database = _copy_database(args.cache_dir, args.phase, rank)
        selection_file = database.parent / "selected-tests"
        selection_file.unlink(missing_ok=True)
        selection_plugin = _SelectionOutput(selection_file)
        pytest_args.extend(
            (
                "-o",
                _testmon_dependency_override(),
                "--collect-only",
                "--testmon",
                "--testmon-nocollect",
                "--testmon-forceselect",
            )
        )
        os.environ["TESTMON_DATAFILE"] = str(database)

    # Avoid importing modules from tests/unit_tests in place of dependencies.
    script_dir = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if Path(entry or Path.cwd()).resolve() != script_dir]
    sys.path.insert(0, str(script_dir.parents[1]))

    import pytest

    plugins = [selection_plugin] if selection_plugin else []
    result = int(pytest.main(pytest_args, plugins=plugins))
    if args.mode == "baseline" and rank == 0 and result in (0, 5):
        record_phase(args.cache_dir, args.phase)
    return 0 if result == 5 else result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "select"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Testmon wrapper ({args.mode}/{args.phase}): {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
