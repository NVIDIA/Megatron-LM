#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fingerprint the build environment, leaving individual wheel identity to uv.

Changing a runtime dependency must not discard every other package's compiled
wheel. uv already keys those wheels by their source; this namespace additionally
separates the native build environments that uv does not capture itself.
"""

import argparse
import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path

BASE_PACKAGES = frozenset({"torch", "torchvision", "triton"})
RECIPE_FILES = (
    "docker/Dockerfile.ci.dev",
    "docker/common/install_nccl.sh",
    "docker/common/uv_cache_key.py",
)
BUILD_SETTINGS = (
    "no-build-isolation",
    "no-build-isolation-package",
    "extra-build-dependencies",
    "extra-build-variables",
    "build-constraint-dependencies",
    "config-settings",
    "config-settings-package",
    "no-binary",
    "no-binary-package",
    "no-build",
    "no-build-package",
    "cache-keys",
)


def normalized_name(name: str) -> str:
    """Normalize Python distribution names without importing packaging."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(requirement: str | dict) -> str:
    """Read the distribution name from a lock reference or PEP 508 requirement."""
    if isinstance(requirement, dict):
        requirement = requirement.get("name", requirement.get("requirement", ""))
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    if not match:
        raise ValueError(f"Cannot identify build requirement: {requirement!r}")
    return normalized_name(match.group(1))


def dependencies(package: dict):
    """Include every marker/extra alternative rather than emulate uv's resolver."""
    for field in ("dependencies", "optional-dependencies", "build-dependencies"):
        references = package.get(field, [])
        if isinstance(references, dict):
            for group in references.values():
                yield from group
        else:
            yield from references


def build_environment(project: dict, lock: dict) -> dict:
    """Select the resolved native build tools and settings affecting their builds."""
    project_name = normalized_name(project["project"]["name"])
    packages_by_name = {}
    for package in lock["package"]:
        packages_by_name.setdefault(normalized_name(package["name"]), []).append(package)
    roots = packages_by_name.get(project_name, [])
    root = next(
        (
            package
            for package in roots
            if package.get("source", {}).get("editable") == "."
            or package.get("source", {}).get("virtual") == "."
        ),
        None,
    )
    if root is None:
        raise ValueError(f"Root package {project_name!r} is missing from uv.lock")
    groups = root.get("dev-dependencies", root.get("dependency-groups", {}))
    if "build" not in groups:
        raise ValueError("uv.lock has no resolved build dependency group")

    uv_settings = project.get("tool", {}).get("uv", {})
    pending = [requirement_name(reference) for reference in groups["build"]]
    for requirements in uv_settings.get("extra-build-dependencies", {}).values():
        pending.extend(requirement_name(requirement) for requirement in requirements)
    selected = set()
    while pending:
        name = pending.pop()
        if name in selected or name in BASE_PACKAGES:
            continue
        candidates = packages_by_name.get(name)
        if not candidates:
            raise ValueError(f"Build dependency {name!r} is missing from uv.lock")
        selected.add(name)
        for package in candidates:
            pending.extend(requirement_name(reference) for reference in dependencies(package))

    settings = {key: uv_settings[key] for key in BUILD_SETTINGS if key in uv_settings}
    settings["sources"] = {
        name: value
        for name, value in uv_settings.get("sources", {}).items()
        if normalized_name(name) in selected
    }
    settings["dependency-metadata"] = [
        metadata
        for metadata in uv_settings.get("dependency-metadata", [])
        if normalized_name(metadata["name"]) in selected
    ]
    for key in ("override-dependencies", "constraint-dependencies"):
        settings[key] = [
            requirement
            for requirement in uv_settings.get(key, [])
            if requirement_name(requirement) in selected | BASE_PACKAGES
        ]

    return {
        "build-group": project["dependency-groups"]["build"],
        "build-system": project.get("build-system", {}),
        "requires-python": lock.get("requires-python"),
        "uv-settings": settings,
        "packages": sorted(
            (package for name in selected for package in packages_by_name[name]),
            key=lambda package: json.dumps(package, sort_keys=True),
        ),
    }


def cache_key(repository: Path, base_image: str, architecture: str, cuda_archs: str) -> str:
    """Return a content hash scoped to an immutable base and CPU architecture."""
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", base_image):
        raise ValueError("--base-image must be an image reference pinned by sha256 digest")
    if architecture not in ("amd64", "arm64"):
        raise ValueError("--architecture must be amd64 or arm64")
    if not re.fullmatch(r"[0-9]+[af]?(;[0-9]+[af]?)*", cuda_archs):
        raise ValueError("--cuda-archs must contain explicit semicolon-separated GPU architectures")
    with (repository / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    with (repository / "uv.lock").open("rb") as stream:
        lock = tomllib.load(stream)
    fingerprint = {
        "schema": 1,
        "base-image": base_image,
        "architecture": architecture,
        "cuda-archs": cuda_archs,
        "recipes": {
            name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
            for name in RECIPE_FILES
        },
        "build-environment": build_environment(project, lock),
    }
    return hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:32]


def main() -> None:
    """Print only the digest so callers can safely use it as a registry tag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--architecture", required=True, choices=("amd64", "arm64"))
    parser.add_argument("--cuda-archs", required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        key = cache_key(args.repository, args.base_image, args.architecture, args.cuda_archs)
        sys.stdout.write(f"{key}\n")
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
