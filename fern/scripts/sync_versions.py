#!/usr/bin/env python3
"""Synchronize canonical docs into Fern's versioned page layout."""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
FERN_ROOT = REPO_ROOT / "fern"
VERSIONS_ROOT = FERN_ROOT / "versions"
GENERATED_HEADER = """\
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""


def copy_pages(source: Path, destination: Path) -> None:
    """Copy MDX pages and images while preserving paths relative to docs/."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    for page in source.rglob("*.mdx"):
        target = destination / page.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(page, target)

    images = source / "images"
    if images.exists():
        shutil.copytree(images, destination / "images")


def write_navigation(source: Path, destination: Path, version: str) -> None:
    navigation = (source / "index.yml").read_text(encoding="utf-8")
    navigation = re.sub(
        r"(?m)^(\s*path:\s*)\./",
        rf"\1./{version}/pages/",
        navigation,
    )
    destination.write_text(GENERATED_HEADER + navigation, encoding="utf-8")


def sync_assets(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source / "images", destination)


def generate(source: Path, fern_root: Path, version: str, latest: bool = False) -> None:
    versions = fern_root / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    copy_pages(source, versions / version / "pages")
    write_navigation(source, versions / f"{version}.yml", version)
    sync_assets(source, fern_root / "assets" / "images")
    if latest:
        shutil.copy2(versions / f"{version}.yml", versions / "latest.yml")


def trees_match(expected: Path, actual: Path) -> bool:
    expected_files = {
        path.relative_to(expected) for path in expected.rglob("*") if path.is_file()
    }
    actual_files = {
        path.relative_to(actual) for path in actual.rglob("*") if path.is_file()
    }
    return expected_files == actual_files and all(
        (expected / path).read_bytes() == (actual / path).read_bytes()
        for path in expected_files
    )


def check_nightly() -> int:
    with tempfile.TemporaryDirectory() as temp:
        generated = Path(temp) / "fern"
        generate(DOCS_ROOT, generated, "nightly")
        expected_paths = (
            Path("versions/nightly.yml"),
            Path("versions/nightly/pages"),
            Path("assets/images"),
        )
        stale = [
            str(path)
            for path in expected_paths
            if not (FERN_ROOT / path).exists()
            or (
                (generated / path).is_dir()
                and not trees_match(generated / path, FERN_ROOT / path)
            )
            or (
                (generated / path).is_file()
                and (generated / path).read_bytes() != (FERN_ROOT / path).read_bytes()
            )
        ]
    if stale:
        print("Fern nightly content is stale:")
        for path in stale:
            print(f"  {path}")
        print("Run: python3 fern/scripts/sync_versions.py")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release",
        help="Create a frozen release snapshot and make it the latest version.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DOCS_ROOT,
        help="Documentation source directory (defaults to docs/).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if committed nightly pages, navigation, or assets are stale.",
    )
    args = parser.parse_args()

    if args.check:
        return check_nightly()

    if args.release:
        generate(args.source.resolve(), FERN_ROOT, args.release, latest=True)
    else:
        generate(args.source.resolve(), FERN_ROOT, "nightly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
