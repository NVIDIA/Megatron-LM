# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Select unit tests on CPU without importing or collecting the GPU test suite."""

from __future__ import annotations

import argparse
import json
import platform
import re
import sqlite3
import sys
from contextlib import closing
from importlib.metadata import version
from pathlib import Path, PurePosixPath

from testmon_cache import (
    PHASES,
    PLATFORMS,
    TESTMON_VERSION,
    cache_identity,
    collection_inventory,
    validate_cache,
)


def _source_path(root: Path, filename: str) -> Path:
    relative = PurePosixPath(filename)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != filename:
        raise ValueError(f"invalid Testmon source path: {filename!r}")
    path = root / filename
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Testmon source escapes checkout: {filename!r}")
    return path


def _nodeid(root: Path, value: str) -> str:
    if not isinstance(value, str) or not value or any(char in value for char in "\r\n\0"):
        raise ValueError("invalid Testmon collection node ID")
    filename = value.split("::", 1)[0]
    if not filename.startswith("tests/unit_tests/") or not filename.endswith(".py"):
        raise ValueError(f"invalid unit-test node ID: {value!r}")
    _source_path(root, filename)
    return value


def _collection(root: Path, metadata: dict, files: dict[str, str]) -> dict:
    collection = metadata.get("collection")
    if not isinstance(collection, dict):
        raise ValueError("missing complete rank collection inventory")
    world_size = collection.get("world_size")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("invalid rank collection world size")
    nodeids = collection.get("nodeids")
    if not isinstance(nodeids, list) or len(set(nodeids)) != len(nodeids):
        raise ValueError("invalid rank collection node IDs")
    for nodeid in nodeids:
        _nodeid(root, nodeid)
    if collection.get("files") != files:
        # Collection can execute arbitrary Python, generate parameters, and vary by rank.
        # Any new or changed test/helper needs collection in its original GPU environment.
        raise ValueError("unit-test sources changed; GPU collection is required")
    return collection


def _changed_files(directory: Path, source_sha: str) -> list[str]:
    metadata = json.loads((directory / "metadata.json").read_text())
    if not isinstance(metadata, dict) or metadata.get("tested_sha") != source_sha:
        raise ValueError("PR changed-file artifact does not match the tested commit")
    for field, maximum in (("changed_files", 3000), ("changed_paths", 6000)):
        count = metadata.get(field)
        if type(count) is not int or not 0 <= count <= maximum:
            raise ValueError(f"invalid PR changed-file artifact {field}")
    data = (directory / "changed-files").read_bytes()
    paths = data.decode("utf-8").splitlines()
    if (
        data.count(b"\n") != metadata["changed_paths"]
        or len(paths) != metadata["changed_paths"]
        or any(not path for path in paths)
    ):
        raise ValueError("incomplete PR changed-file artifact")
    return paths


def _mandatory_files(
    root: Path, bucket: str, recipe_platform: str, changed: list[str]
) -> list[str]:
    import yaml
    from find_test_cases import (
        PLATFORM_MARKERS,
        expand_pattern,
        file_has_marker,
        is_child_of_bucket,
    )
    from testmon_mandatory import load_mappings, mandatory_files, triggered_patterns

    recipe_path = Path(PLATFORMS[recipe_platform]["recipe"])
    try:
        recipe = yaml.safe_load((root / recipe_path).read_text())
        mappings = load_mappings(root / "tests/unit_tests/testmon_mandatory_tests.yaml")
    except yaml.YAMLError as error:
        raise ValueError(f"invalid mandatory-test configuration: {error}") from error
    if not isinstance(recipe, dict) or not isinstance(recipe.get("products"), list):
        raise ValueError("invalid unit-test recipe")
    all_buckets = []
    for product in recipe["products"]:
        cases = product.get("test_case") if isinstance(product, dict) else None
        if not isinstance(cases, list) or not all(isinstance(case, str) for case in cases):
            raise ValueError("invalid unit-test recipe buckets")
        all_buckets.extend(cases)
    bucket_files = set(expand_pattern(str(root / bucket)))
    ignored = set()
    for test_case in all_buckets:
        if test_case != bucket and is_child_of_bucket(test_case, bucket):
            ignored.update(expand_pattern(str(root / test_case)))
    marker = PLATFORM_MARKERS.get(recipe_path.parent.name)
    if marker:
        ignored.update(
            path
            for path in bucket_files
            if Path(path).name.startswith("test_") and not file_has_marker(path, marker)
        )
    _, patterns = triggered_patterns(mappings, changed, recipe_platform)
    return [
        _nodeid(root, str(Path(path).relative_to(root)))
        for path in mandatory_files(
            [str(root / pattern) for pattern in patterns], str(root / bucket), ignored
        )
    ]


def _select_phase(root: Path, database: Path, collected: list[str]) -> list[str]:
    from testmon.process_code import Module, blob_to_checksums, read_source_sha

    with closing(
        sqlite3.connect(database.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    ) as connection:
        environments = connection.execute("SELECT id FROM environment").fetchall()
        if len(environments) != 1:
            raise ValueError("Testmon baseline must have exactly one environment")
        recorded = connection.execute("SELECT test_name, failed FROM test_execution").fetchall()
        fingerprints = connection.execute(
            "SELECT te.test_name, f.filename, f.fsha, f.method_checksums "
            "FROM test_execution te "
            "JOIN test_execution_file_fp tf ON tf.test_execution_id = te.id "
            "JOIN file_fp f ON f.id = tf.fingerprint_id"
        ).fetchall()

    inventory = set(collected)
    # Testmon records placeholders before pytest applies marker deselection.
    known = {_nodeid(root, nodeid) for nodeid, _ in recorded} & inventory
    # Tests collected only on other ranks have no rank-zero dependency evidence.
    selected = inventory - known
    selected.update(nodeid for nodeid, failed in recorded if nodeid in inventory and failed != 0)
    sources = {}
    covered = set()
    for nodeid, filename, recorded_sha, fingerprint in fingerprints:
        if nodeid not in inventory:
            continue
        covered.add(nodeid)
        if filename not in sources:
            path = _source_path(root, filename)
            source, source_sha = read_source_sha(path)
            module = (
                Module(source_code=source, ext=path.suffix.lstrip("."))
                if source is not None
                else None
            )
            sources[filename] = (source_sha, module)
        source_sha, module = sources[filename]
        if module is None:
            selected.add(nodeid)
        elif recorded_sha is None or recorded_sha != source_sha:
            current = set(module.method_checksums)
            if not current or not set(blob_to_checksums(fingerprint)) <= current:
                selected.add(nodeid)
    selected.update(known - covered)
    return sorted(selected)


def select(
    root: Path,
    cache_dir: Path,
    bucket: str,
    recipe_platform: str,
    image_id: str,
    source_sha: str,
    matched_key: str,
    pr_files_dir: Path,
) -> dict:
    """Return a CPU selection, or a full-bucket plan if any evidence is unavailable."""
    plan = {
        "schema": 1,
        "source_sha": source_sha,
        "bucket": bucket,
        "platform": recipe_platform,
        "image_id": image_id,
        "mode": "full",
        "reason": "",
        "phases": {},
    }
    try:
        if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
            raise ValueError("invalid selection source commit")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise ValueError("immutable container image identity is unavailable")
        identity = cache_identity(root, bucket, recipe_platform, image_id)
        manifest = validate_cache(cache_dir, identity, matched_key)
        if manifest.get("image_id") != image_id:
            # The CPU host cannot measure the packages in a different GPU image.
            # Equal image IDs prove the baseline runtime is still the execution runtime.
            raise ValueError("container image changed; GPU runtime validation is required")
        if not any(char in bucket for char in "*?[") and source_sha != manifest["source_sha"]:
            # Explicit-file pytest arguments bypass Testmon's stable-file collection skip.
            # They may discover new IDs from collection-only production dependencies.
            raise ValueError(
                "explicit test-file bucket requires GPU collection after source changes"
            )
        if version("pytest-testmon") != TESTMON_VERSION:
            raise ValueError("unexpected pytest-testmon version")
        changed_files = _changed_files(pr_files_dir, source_sha)
        mandatory = _mandatory_files(root, bucket, recipe_platform, changed_files)
        files = collection_inventory(root)
        metadata = {}
        collections = {}
        for phase in PHASES:
            metadata[phase] = json.loads((cache_dir / phase / "metadata.json").read_text())
            runtime = metadata[phase]["runtime"]
            python = runtime.get("python", "")
            if (
                not isinstance(python, str)
                or python.split(".")[:2] != platform.python_version().split(".")[:2]
                or runtime.get("implementation") != platform.python_implementation()
            ):
                raise ValueError("CPU Python differs from the Testmon fingerprint interpreter")
            collections[phase] = _collection(root, metadata[phase], files)
        if len({collection["world_size"] for collection in collections.values()}) != 1:
            raise ValueError("phase collection world sizes differ")
        phases = {
            phase: _select_phase(
                root, cache_dir / phase / ".testmondata", collections[phase]["nodeids"]
            )
            for phase in PHASES
        }
        # Recollect affected files in both modes: production dependencies can change
        # parameter IDs or experimental markers without changing the test source.
        affected_files = sorted(
            {nodeid.split("::", 1)[0] for nodes in phases.values() for nodeid in nodes}
            | set(mandatory)
        )
        phases = {phase: affected_files for phase in PHASES}
        plan.update(
            mode="selected",
            reason="selected on CPU from the compatible main baseline",
            phases=phases,
            runtime={phase: metadata[phase]["runtime"] for phase in PHASES},
            world_size=collections["prod"]["world_size"],
            mandatory_files=mandatory,
        )
    except (
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        KeyError,
        sqlite3.Error,
    ) as error:
        plan["reason"] = str(error)
    return plan


def prepare(
    plan: dict,
    cache_dir: Path,
    bucket: str,
    recipe_platform: str,
    source_sha: str,
    *,
    image_id: str | None = None,
    check_runtime: bool = False,
    world_size: int | None = None,
) -> None:
    """Validate a selected plan before exposing its IDs to the distributed runner."""
    from testmon_cache import runtime_identity

    if not isinstance(plan, dict) or plan.get("schema") != 1 or plan.get("mode") != "selected":
        raise ValueError("no valid CPU-selected Testmon plan")
    if (
        plan.get("source_sha") != source_sha
        or plan.get("bucket") != bucket
        or plan.get("platform") != recipe_platform
    ):
        raise ValueError("Testmon plan does not match this checkout and bucket")
    planned_image = plan.get("image_id")
    if not isinstance(planned_image, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", planned_image
    ):
        raise ValueError("Testmon plan has no immutable container image identity")
    if image_id is not None and planned_image != image_id:
        raise ValueError("Testmon plan container image changed")
    planned_world_size = plan.get("world_size")
    if type(planned_world_size) is not int or planned_world_size < 1:
        raise ValueError("invalid Testmon plan world size")
    if world_size is not None and planned_world_size != world_size:
        raise ValueError("Testmon plan distributed world size changed")
    phases = plan.get("phases")
    if not isinstance(phases, dict) or set(phases) != set(PHASES):
        raise ValueError("incomplete Testmon plan phases")
    for nodeids in phases.values():
        if not isinstance(nodeids, list):
            raise ValueError("invalid Testmon plan node IDs")
        for nodeid in nodeids:
            _nodeid(Path.cwd(), nodeid)
    mandatory = plan.get("mandatory_files")
    if not isinstance(mandatory, list):
        raise ValueError("invalid mandatory test files in Testmon plan")
    for filename in mandatory:
        _nodeid(Path.cwd(), filename)
        if "::" in filename or any(filename not in nodeids for nodeids in phases.values()):
            raise ValueError("mandatory test file missing from Testmon selection")
    if check_runtime:
        runtime = runtime_identity()
        planned_runtime = plan.get("runtime")
        if not isinstance(planned_runtime, dict) or any(
            planned_runtime.get(phase) != runtime for phase in PHASES
        ):
            raise ValueError("Testmon plan runtime environment changed")
    for phase, nodeids in phases.items():
        output = cache_dir / ".testmon-work" / phase / "selected-tests"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(f"{nodeid}\n" for nodeid in sorted(set(nodeids))))
        output.with_name("mandatory-tests").write_text(
            "".join(f"{filename}\n" for filename in sorted(set(mandatory)))
        )


def main(argv: list[str] | None = None) -> int:
    """Write a selection artifact and Actions outputs; uncertainty always schedules GPUs."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select_parser = commands.add_parser("select")
    select_parser.add_argument("--matched-key", required=True)
    select_parser.add_argument("--pr-files-dir", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--plan", type=Path, required=True)
    prepare_parser.add_argument("--check-runtime", action="store_true")
    prepare_parser.add_argument("--world-size", type=int)
    for child in (select_parser, prepare_parser):
        child.add_argument("--cache-dir", type=Path, required=True)
        child.add_argument("--bucket", required=True)
        child.add_argument("--platform", required=True)
        child.add_argument("--image-id", required=child is select_parser)
        child.add_argument("--source-sha", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        try:
            prepare(
                json.loads(args.plan.read_text()),
                args.cache_dir,
                args.bucket,
                args.platform,
                args.source_sha,
                image_id=args.image_id,
                check_runtime=args.check_runtime,
                world_size=args.world_size,
            )
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
            print(f"Unit Testmon plan unavailable: {error}", file=sys.stderr)
            return 2
        return 0
    plan = select(
        Path.cwd(),
        args.cache_dir,
        args.bucket,
        args.platform,
        args.image_id,
        args.source_sha,
        args.matched_key,
        args.pr_files_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")
    has_tests = plan["mode"] == "full" or any(plan["phases"].values())
    print(f"mode={plan['mode']}")
    print(f"has_tests={str(has_tests).lower()}")
    print(f"Unit Testmon CPU selection: {plan['reason']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
