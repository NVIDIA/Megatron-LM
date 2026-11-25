# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify an uploaded release against its build artifacts and signed provenance."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename

INDEXES = {"pypi": "https://pypi.org", "testpypi": "https://test.pypi.org"}


def release_files(directory: Path, package: str) -> tuple[str, dict[str, str]]:
    """Return one release version and SHA-256 digests for its distributions."""
    files = {}
    versions = set()
    for path in sorted(directory.iterdir()):
        if path.name.endswith(".whl"):
            name, version, *_ = parse_wheel_filename(path.name)
        elif path.name.endswith(".tar.gz"):
            name, version = parse_sdist_filename(path.name)
        else:
            raise ValueError(f"Unexpected distribution: {path.name}")
        if canonicalize_name(name) != canonicalize_name(package):
            raise ValueError(f"Unexpected package: {name}")
        versions.add(str(version))
        with path.open("rb") as stream:
            files[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    if len(versions) != 1 or not any(name.endswith(".whl") for name in files):
        raise ValueError("Expected wheels for exactly one release version")
    if sum(name.endswith(".tar.gz") for name in files) != 1:
        raise ValueError("Expected exactly one source distribution")
    return versions.pop(), files


def fetch_json(url: str) -> dict:
    """Fetch index metadata, allowing bounded retries for index propagation."""
    for attempt in range(12):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in (404, 429, 500, 502, 503, 504) or attempt == 11:
                raise
        time.sleep(10)
    raise RuntimeError(f"Index metadata did not become available: {url}")


def check_release(metadata: dict, expected: dict[str, str]) -> None:
    """Require exactly the built files, matching hashes, and no yanked files."""
    actual = {entry["filename"]: entry for entry in metadata["urls"]}
    if actual.keys() != expected.keys():
        raise ValueError(
            f"File set mismatch: expected {sorted(expected)}, received {sorted(actual)}"
        )
    for name, digest in expected.items():
        if actual[name]["digests"]["sha256"] != digest or actual[name].get("yanked", False):
            raise ValueError(f"Published file differs from the build or is yanked: {name}")


def verify_attestations(
    index: str, package: str, version: str, directory: Path, files: dict
) -> None:
    """Verify each artifact against provenance served by the selected index."""
    with tempfile.TemporaryDirectory() as temporary:
        provenance_path = Path(temporary) / "provenance.json"
        for name in files:
            provenance = fetch_json(f"{index}/integrity/{package}/{version}/{name}/provenance")
            bundles = provenance.get("attestation_bundles", [])
            if not bundles or any(not bundle.get("attestations") for bundle in bundles):
                raise ValueError(f"Missing attestations: {name}")
            for bundle in bundles:
                publisher = bundle["publisher"]
                if publisher.get("kind") != "GitHub" or publisher.get("workflow") != "release.yaml":
                    raise ValueError(f"Unexpected publisher: {publisher}")
            provenance_path.write_text(json.dumps(provenance))
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pypi_attestations",
                    "verify",
                    "pypi",
                    "--repository",
                    "https://github.com/NVIDIA/Megatron-LM",
                    "--provenance-file",
                    str(provenance_path),
                    str(directory / name),
                ],
                check=True,
                timeout=180,
            )


def main() -> None:
    """Validate an index release and emit its version for installation checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", choices=INDEXES, required=True)
    parser.add_argument("--package", choices=("megatron-core", "megatron-fsdp"), required=True)
    parser.add_argument("--dist-dir", type=Path, required=True)
    args = parser.parse_args()
    version, files = release_files(args.dist_dir, args.package)
    index = INDEXES[args.index]
    check_release(fetch_json(f"{index}/pypi/{args.package}/{version}/json"), files)
    verify_attestations(index, args.package, version, args.dist_dir, files)
    sys.stdout.write(
        f"Verified {len(files)} distributions for {args.package}=={version} on {args.index}\n"
    )
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as stream:
            stream.write(f"version={version}\n")


if __name__ == "__main__":
    main()
