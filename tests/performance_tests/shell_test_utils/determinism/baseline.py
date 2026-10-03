# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Publish and verify portable, content-addressed kernel performance baselines.

Raw samples and the author/timing join are checked again after transport.
Publication preserves report-only status and does not calibrate performance limits.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable

import author_evidence

MANIFEST = "baseline.json"


def _json(raw: bytes):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(raw, parse_constant=invalid)


def _bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular artifact file: {path}")
    return path.read_bytes()


def _key(report: dict) -> tuple[str, str, str]:
    measurement = report["measurement"]
    values = tuple(measurement[key] for key in ("kernel_case", "dtype", "phase"))
    if any(
        not isinstance(value, str)
        or not value
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in value)
        for value in values
    ):
        raise ValueError("Invalid leaderboard row identity")
    return values


def snapshot(coverage_path: Path, leaderboard_path: Path, revision: str, origin: str):
    """Validate every declared author case, phase report and separate raw timing file."""
    if not isinstance(origin, str) or not origin.strip():
        raise ValueError("An explicit source reference is required")
    coverage_raw, leaderboard_raw = _bytes(coverage_path), _bytes(leaderboard_path)
    coverage, reports = _json(coverage_raw), _json(leaderboard_raw)
    if not isinstance(reports, list) or not reports:
        raise ValueError("Expected a nonempty leaderboard report list")
    joined = author_evidence.join(coverage, reports, revision)
    if not joined["evidence_complete"] or joined["status"] not in ("passed", "not_gated"):
        raise ValueError("Baseline requires complete, nonfailing author and timing evidence")

    expected = {
        (
            row["case_signature"]["case"],
            row["case_signature"]["inputs"][0]["dtype"].removeprefix("torch."),
            phase,
        )
        for row in joined["requirements"]
        for phase in ("forward", "backward")
    }
    keys = [_key(report) for report in reports]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Leaderboard has duplicate, missing or undeclared phase reports")
    if any(
        report["machine"] != reports[0]["machine"]
        or report["sources"] != reports[0]["sources"]
        or any(
            report["measurement"][key] != reports[0]["measurement"][key]
            for key in ("pairs", "warmup", "steps", "tokens", "hidden_size")
        )
        for report in reports
    ):
        raise ValueError("One baseline must retain one source, allocation and measurement protocol")
    devices = {run["kernel"]["device_uuid"] for report in reports for run in report["runs"]}
    if len(devices) != 1:
        raise ValueError("Leaderboard rows use different timing GPUs")

    root = leaderboard_path.parent.resolve()
    locations = {}
    for path in root.rglob("benchmark.json"):
        if not path.resolve().is_relative_to(root):
            raise ValueError("Benchmark artifact escapes its source directory")
        report = _json(_bytes(path))
        key = _key(report)
        if key in locations:
            raise ValueError("Multiple attempts found; do not select a favorable retry")
        locations[key] = (path, report)
    if set(locations) != expected:
        raise ValueError("Separate benchmark reports are missing or do not match the leaderboard")

    files = {"coverage.json": coverage_raw, "performance/leaderboard.json": leaderboard_raw}
    markdown = leaderboard_path.with_suffix(".md")
    if markdown.exists():
        files["performance/leaderboard.md"] = _bytes(markdown)
    for key, report in zip(keys, reports):
        path, separate = locations[key]
        if separate != report:
            raise ValueError("Leaderboard differs from its separate benchmark report")
        for run in report["runs"]:
            if type(run["pair"]) is not int:
                raise ValueError("Invalid timing pair number")
            arm = path.parent / f"pair-{run['pair']}" / f"{run['revision_label']}-{run['mode']}"
            raw = _json(_bytes(arm / "kernel.json"))
            if raw != run["kernel"]:
                raise ValueError("Separate raw timing file differs from embedded measurements")
            _bytes(arm / "launcher.log")
        prefix = "performance/" + "-".join(key)
        for item in path.parent.rglob("*"):
            if item.is_symlink():
                raise ValueError(f"Artifact symlinks are not supported: {item}")
            if item.is_file():
                relative = item.relative_to(path.parent).as_posix()
                name = prefix + "/" + relative
                if name in files:
                    raise ValueError("Overlapping benchmark artifact directories")
                files[name] = _bytes(item)
    manifest = {
        "schema_version": 1,
        "kind": "determinism_performance_baseline",
        "revision": revision,
        "origin": origin,
        "author_evidence": joined,
        "files": {
            name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            for name, raw in sorted(files.items())
        },
    }
    return manifest, files


def _encode(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def verified_contents(directory: Path, expected_id: str | None = None) -> tuple[dict, str, int]:
    """Rehash every regular file before a format-specific evidence verification."""
    if directory.is_symlink():
        raise ValueError("Baseline directory must not be a symlink")
    raw = _bytes(directory / MANIFEST)
    identity = hashlib.sha256(raw).hexdigest()
    if expected_id is not None and identity != expected_id:
        raise ValueError("Baseline identifier differs from the requested immutable reference")
    manifest = _json(raw)
    actual = {}
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("Baseline contains a symlink")
        if path.is_file() and path != directory / MANIFEST:
            content = _bytes(path)
            actual[path.relative_to(directory).as_posix()] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
    if actual != manifest["files"]:
        raise ValueError("Baseline files are missing, changed or undeclared")
    return manifest, identity, len(actual)


def verify(directory: Path, expected_id: str | None = None) -> dict:
    """Rehash the complete bundle and independently recompute its evidence join."""
    manifest, identity, count = verified_contents(directory, expected_id)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "determinism_performance_baseline"
    ):
        raise ValueError("Unsupported baseline schema")
    rebuilt, _ = snapshot(
        directory / "coverage.json",
        directory / "performance/leaderboard.json",
        manifest["revision"],
        manifest["origin"],
    )
    if rebuilt != manifest:
        raise ValueError("Baseline summary differs from recomputed evidence")
    return {
        "baseline_id": identity,
        "revision": manifest["revision"],
        "origin": manifest["origin"],
        "status": rebuilt["author_evidence"]["status"],
        "author_cases": len(rebuilt["author_evidence"]["requirements"]),
        "files_verified": count,
    }


def publish(coverage: Path, leaderboard: Path, store: Path, revision: str, origin: str) -> dict:
    """Publish a new content identifier atomically, never replacing an existing baseline."""
    if store.resolve().is_relative_to(leaderboard.parent.resolve()):
        raise ValueError("Baseline store must be outside the source artifact directory")
    manifest, files = snapshot(coverage, leaderboard, revision, origin)
    return publish_snapshot(manifest, files, store, verify)


def publish_snapshot(
    manifest: dict, files: dict[str, bytes], store: Path, verifier: Callable
) -> dict:
    """Atomically publish verified immutable bytes for one explicit bundle format."""
    raw = _encode(manifest)
    identity = hashlib.sha256(raw).hexdigest()
    store.mkdir(parents=True, exist_ok=True)
    destination = store / identity
    if destination.exists():
        return {**verifier(destination, identity), "path": str(destination), "created": False}
    temporary = Path(tempfile.mkdtemp(prefix=".publishing-", dir=store))
    try:
        for name, content in files.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (temporary / MANIFEST).write_bytes(raw)
        verified = verifier(temporary, identity)
        try:
            temporary.rename(destination)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            verified = verifier(destination, identity)
            return {**verified, "path": str(destination), "created": False}
        return {**verified, "path": str(destination), "created": True}
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main(argv: list[str] | None = None) -> int:
    """Publish from downloaded artifacts, or verify an existing relocated bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("publish")
    create.add_argument("--coverage", type=Path, required=True)
    create.add_argument("--leaderboard", type=Path, required=True)
    create.add_argument("--store", type=Path, required=True)
    create.add_argument("--revision", required=True)
    create.add_argument(
        "--origin", required=True, help="Caller-supplied CI run or execution reference"
    )
    check = commands.add_parser("verify")
    check.add_argument("directory", type=Path)
    check.add_argument("--expected-id")
    args = parser.parse_args(argv)
    try:
        if args.command == "publish":
            result = publish(
                args.coverage, args.leaderboard, args.store, args.revision, args.origin
            )
        else:
            result = verify(args.directory, args.expected_id)
    except (OSError, AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"status": "invalid", "error": str(error)}))
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
