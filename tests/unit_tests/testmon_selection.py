# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Transport native pytest/Testmon selections without deciding dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

from testmon_cache import PHASES, cache_identity, validate_cache, validate_phase

SCHEMA = 1
SELECTION = "selection.json"
SNAPSHOT_FILES = (
    "manifest.json",
    "changed-files",
    *(f"{phase}/{filename}" for phase in PHASES for filename in (".testmondata", "metadata.json")),
)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _targets(value: object) -> list[str]:
    """Reject malformed transport entries before using them as pytest arguments."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("Selected tests must be a list of strings")
    if value != sorted(set(value)):
        raise ValueError("Selected tests must be sorted and unique")
    for nodeid in value:
        filename = nodeid.split("::", 1)[0]
        path = Path(filename)
        if (
            not filename.startswith("tests/unit_tests/")
            or path.suffix != ".py"
            or ".." in path.parts
            or any(char in nodeid for char in "\r\n\0")
        ):
            raise ValueError(f"Invalid selected test: {nodeid!r}")
    return value


def _read_targets(path: Path) -> list[str]:
    return _targets(path.read_text().splitlines())


def _merged(ranks: dict[str, list[str]], mandatory: list[str]) -> list[str]:
    # The existing mandatory helper replaces individual IDs with whole files.
    return sorted(
        {
            nodeid
            for ids in ranks.values()
            for nodeid in ids
            if nodeid.split("::", 1)[0] not in mandatory
        }
        | set(mandatory)
    )


def _phase_outputs(cache_dir: Path, world_size: int) -> dict:
    phases = {}
    for phase in PHASES:
        directory = cache_dir / ".testmon-work" / phase
        paths = {path.parent.name: path for path in directory.glob("rank-*/selected-tests")}
        expected = {f"rank-{rank}" for rank in range(world_size)}
        if paths.keys() != expected:
            raise ValueError(f"Incomplete native {phase} rank selections: {sorted(paths)}")
        ranks = {str(rank): _read_targets(paths[f"rank-{rank}"]) for rank in range(world_size)}
        mandatory = _read_targets(directory / "mandatory-tests")
        if any("::" in filename for filename in mandatory):
            raise ValueError("Mandatory entries must be whole test files")
        selected = _read_targets(directory / "selected-tests")
        if selected != _merged(ranks, mandatory):
            raise ValueError(f"Native {phase} union does not match rank and mandatory outputs")
        phases[phase] = {"selected": selected, "mandatory": mandatory, "ranks": ranks}
    return phases


def _baseline(cache_dir: Path, bucket: str, platform: str, check_runtime: bool) -> str:
    identity = cache_identity(Path.cwd(), bucket, platform)
    manifest = _read_json(cache_dir / "manifest.json")
    key = identity["cache_prefix"] + manifest["generation"]
    if os.environ.get("UNIT_TESTMON_CACHE_KEY", key) != key:
        raise ValueError("Native selection does not match the restored cache key")
    validate_cache(cache_dir, identity, key)
    if check_runtime:
        for phase in PHASES:
            validate_phase(cache_dir, phase, check_runtime=True)
    return key


def _identity(source_sha: str, image_id: str, world_size: int) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("Selection source SHA must be a full commit ID")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("Selection requires an immutable Docker image ID")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("Selection world size must be a positive integer")


def export_selection(
    cache_dir: Path, bucket: str, platform: str, source_sha: str, image_id: str, world_size: int
) -> dict:
    """Export successful native results; callers must check every collection process first."""
    _identity(source_sha, image_id, world_size)
    key = _baseline(cache_dir, bucket, platform, check_runtime=True)
    selection = {
        "schema": SCHEMA,
        "complete": True,
        "source_sha": source_sha,
        "bucket": bucket,
        "platform": platform,
        "image_id": image_id,
        "world_size": world_size,
        "cache_key": key,
        "snapshot": {name: _digest(cache_dir / name) for name in SNAPSHOT_FILES},
        "phases": _phase_outputs(cache_dir, world_size),
    }
    (cache_dir / SELECTION).write_text(json.dumps(selection, indent=2) + "\n")
    return selection


def validate_selection(
    cache_dir: Path,
    bucket: str,
    platform: str,
    source_sha: str,
    image_id: str,
    world_size: int | None = None,
    *,
    check_runtime: bool = False,
) -> dict:
    """Validate identity, baseline snapshot, and completeness without selecting any tests."""
    selection = _read_json(cache_dir / SELECTION)
    size = selection["world_size"]
    _identity(source_sha, image_id, size)
    if (
        type(selection["schema"]) is not int
        or selection["schema"] != SCHEMA
        or selection["complete"] is not True
    ):
        raise ValueError("Incomplete or unsupported native selection artifact")
    if any(
        selection[name] != expected
        for name, expected in {
            "source_sha": source_sha,
            "bucket": bucket,
            "platform": platform,
            "image_id": image_id,
        }.items()
    ):
        raise ValueError("Native selection does not match this checkout, image, or bucket")
    if world_size is not None and size != world_size:
        raise ValueError("Native selection recipe world size changed")
    if selection["snapshot"] != {name: _digest(cache_dir / name) for name in SNAPSHOT_FILES}:
        raise ValueError("Native selection baseline snapshot changed")
    if selection["cache_key"] != _baseline(cache_dir, bucket, platform, check_runtime):
        raise ValueError("Native selection cache generation changed")
    phases = selection["phases"]
    if not isinstance(phases, dict):
        raise ValueError("Native selection phases must be an object")
    if set(phases) != set(PHASES):
        raise ValueError("Native selection is missing a phase")
    for phase in PHASES:
        output = phases[phase]
        if not isinstance(output, dict) or not isinstance(output.get("ranks"), dict):
            raise ValueError("Native phase and rank selections must be objects")
        ranks = output["ranks"]
        if set(ranks) != {str(rank) for rank in range(size)}:
            raise ValueError(f"Native {phase} selection is missing ranks")
        for ids in ranks.values():
            _targets(ids)
        mandatory = _targets(output["mandatory"])
        if any("::" in filename for filename in mandatory):
            raise ValueError("Mandatory entries must be whole test files")
        if _targets(output["selected"]) != _merged(ranks, mandatory):
            raise ValueError(f"Native {phase} selection union changed")
    return selection


def prepare_selection(cache_dir: Path, selection: dict) -> None:
    """Expose previously validated native IDs to the existing GPU execution harness."""
    work = cache_dir / ".testmon-work"
    if work.exists():
        shutil.rmtree(work)
    for phase in PHASES:
        directory = work / phase
        directory.mkdir(parents=True)
        for filename, field in (("selected-tests", "selected"), ("mandatory-tests", "mandatory")):
            directory.joinpath(filename).write_text(
                "".join(f"{nodeid}\n" for nodeid in selection["phases"][phase][field])
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "validate", "prepare"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--check-runtime", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command != "validate" and args.world_size is None:
            raise ValueError("Container collection/execution requires the actual recipe world size")
        values = (
            args.cache_dir,
            args.bucket,
            args.platform,
            args.source_sha,
            args.image_id,
            args.world_size,
        )
        if args.command == "export":
            selection = export_selection(*values)
        else:
            selection = validate_selection(*values, check_runtime=args.check_runtime)
            if args.command == "prepare":
                prepare_selection(args.cache_dir, selection)
        print(f"has_tests={str(any(p['selected'] for p in selection['phases'].values())).lower()}")
        print(f"cache_key={selection['cache_key']}")
        return 0
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
        print(f"Native Testmon selection unavailable: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
