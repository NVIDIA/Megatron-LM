# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Generate a rank-zero Testmon baseline or select tests from a private copy."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from importlib.metadata import distributions
from pathlib import Path

from packaging.utils import canonicalize_name

PHASES = ("prod", "experimental")
TRACKED_ENVIRONMENT_PACKAGES = frozenset(
    {"numpy", "pytest", "torch", "transformer-engine", "triton"}
)
TRACKED_ENVIRONMENT_PACKAGE_PREFIXES = ("transformer-engine-",)


class _SelectionOutput:
    def __init__(self, output: Path):
        self.output = output

    def pytest_collection_finish(self, session) -> None:
        selected_files = sorted({item.nodeid.split("::", 1)[0] for item in session.items})
        self.output.write_text("".join(f"{path}\n" for path in selected_files))


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


def _testmon_dependency_override() -> str:
    installed_packages = {
        name for distribution in distributions() if (name := distribution.metadata["Name"])
    }
    ignored_packages = sorted(
        name
        for name in installed_packages
        if canonicalize_name(name) not in TRACKED_ENVIRONMENT_PACKAGES
        and not canonicalize_name(name).startswith(TRACKED_ENVIRONMENT_PACKAGE_PREFIXES)
    )
    return f"testmon_ignore_dependencies={' '.join(ignored_packages)}"


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

    selection_plugin = None
    if args.mode == "baseline":
        if rank == 0:
            database = _database(args.cache_dir, args.phase)
            _clear_database_files(database)
            os.environ["TESTMON_DATAFILE"] = str(database)
            pytest_args.extend(
                ("-o", _testmon_dependency_override(), "--testmon", "--testmon-noselect")
            )
        else:
            os.environ.pop("TESTMON_DATAFILE", None)
            pytest_args.extend(("-p", "no:testmon", "-p", "no:pytest-testmon"))
    else:
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
    return 0 if result == 5 else result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "select"), required=True)
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
