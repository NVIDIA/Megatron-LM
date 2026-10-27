# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Manage released versions of the Fern docs site.

Subcommands:
  bump VERSION      Add a released version: create docs/fern/versions/vVERSION.yml from the
                    nightly navigation and update the versions list in docs/fern/docs.yml.
  extract-content   For every docs/fern/versions/vX.Y.Z.yml, extract the docs/ tree from the
                    core_vX.Y.Z tag into docs/fern/versions/vX.Y.Z-content/ so each released
                    version renders the docs as they were at release time.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = REPO_ROOT / "docs"
FERN_DIR = DOCS_DIR / "fern"
VERSIONS_DIR = FERN_DIR / "versions"
DOCS_YML = FERN_DIR / "docs.yml"
NIGHTLY_NAV = VERSIONS_DIR / "nightly.yml"

TAG_PREFIX = "core_v"
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
RELEASE_NAV_RE = re.compile(r"^versions/v(\d+\.\d+\.\d+)\.yml$")
IMG_RE = re.compile(r"<img ([^>]*[^/])>")


def warn(message):
    print(f"::warning::{message}", file=sys.stderr)


def git(*args, check=True):
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=check
    )


def tag_exists(tag):
    return git("rev-parse", "--verify", "--quiet", f"refs/tags/{tag}", check=False).returncode == 0


def version_key(version):
    return tuple(int(part) for part in version.split("."))


def content_dir_name(version):
    return f"v{version}-content"


def freeze_nav(nav_text, version):
    """Point a navigation file at the frozen content directory for `version`."""
    nav_text = re.sub(r"\A(?:#[^\n]*\n)+", "", nav_text)
    nav_text = nav_text.replace("path: ../../", f"path: ./{content_dir_name(version)}/")
    return nav_text.replace("/developer-guide/nightly/", f"/developer-guide/{version}/")


def read_nightly_nav(ref):
    result = git("show", f"{ref}:docs/fern/versions/nightly.yml", check=False)
    if result.returncode == 0:
        return result.stdout
    warn(f"Could not read nightly.yml at {ref}; using the working tree copy instead.")
    return NIGHTLY_NAV.read_text()


def split_versions_block(text):
    """Return (text before the versions block, list of version entries).

    The versions block must be the last top-level key in docs.yml.
    """
    marker = "\nversions:\n"
    idx = text.find(marker)
    if idx == -1:
        raise SystemExit(f"No top-level 'versions:' key found in {DOCS_YML}")
    head = text[: idx + 1]
    entries = []
    for line in text[idx + len(marker) :].splitlines():
        if not line.strip():
            continue
        match = re.match(r"^\s+(-\s+)?([\w-]+):\s*(.*?)\s*$", line)
        if not match:
            raise SystemExit(f"'versions:' must be the last key in {DOCS_YML}; found: {line!r}")
        is_new_entry, key, value = match.groups()
        if is_new_entry:
            entries.append({})
        elif not entries:
            raise SystemExit(f"Unexpected line in versions block: {line!r}")
        entries[-1][key] = value.strip("\"'")
    return head, entries


def render_versions_block(entries):
    lines = ["versions:"]
    for entry in entries:
        for i, (key, value) in enumerate(entry.items()):
            prefix = "  - " if i == 0 else "    "
            if key == "display-name":
                value = f'"{value}"'
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines) + "\n"


def update_docs_yml(new_version):
    head, entries = split_versions_block(DOCS_YML.read_text())

    releases = {}
    others = []
    for entry in entries:
        match = RELEASE_NAV_RE.match(entry.get("path", ""))
        if match:
            releases[match.group(1)] = entry
        else:
            others.append(entry)
    releases.setdefault(new_version, {"path": f"versions/v{new_version}.yml"})

    ordered = sorted(releases, key=version_key, reverse=True)
    release_entries = []
    for i, version in enumerate(ordered):
        entry = {
            "display-name": f"{version} (latest)" if i == 0 else version,
            "path": f"versions/v{version}.yml",
            "slug": "latest" if i == 0 else version.replace(".", "-"),
        }
        extra = {k: v for k, v in releases[version].items() if k not in entry}
        release_entries.append({**entry, **extra})

    nightly = [e for e in others if e.get("slug") == "nightly"]
    rest = [e for e in others if e.get("slug") != "nightly"]
    new_entries = release_entries[:1] + nightly + release_entries[1:] + rest

    DOCS_YML.write_text(head + render_versions_block(new_entries))


def cmd_bump(args):
    version = args.version.removeprefix(TAG_PREFIX).removeprefix("v")
    if not VERSION_RE.match(version):
        raise SystemExit(f"Expected a version like 0.21.0, got {args.version!r}")

    nav_path = VERSIONS_DIR / f"v{version}.yml"
    if nav_path.exists():
        print(f"{nav_path.relative_to(REPO_ROOT)} already exists; leaving it unchanged.")
    else:
        ref = args.ref or f"{TAG_PREFIX}{version}"
        nav_path.write_text(freeze_nav(read_nightly_nav(ref), version))
        print(f"Created {nav_path.relative_to(REPO_ROOT)} from nightly.yml at {ref}.")

    update_docs_yml(version)
    print(f"Updated {DOCS_YML.relative_to(REPO_ROOT)}.")


def self_close_img_tags(content_dir):
    for path in content_dir.rglob("*"):
        if path.suffix in (".md", ".mdx") and path.is_file():
            text = path.read_text()
            fixed = IMG_RE.sub(r"<img \1 />", text)
            if fixed != text:
                path.write_text(fixed)


def extract_from_tag(tag, dest):
    archive = subprocess.Popen(
        ["git", "archive", tag, "--", "docs/"], cwd=REPO_ROOT, stdout=subprocess.PIPE
    )
    subprocess.run(
        ["tar", "-x", "--strip-components=1", "--exclude=docs/fern", "-C", str(dest)],
        stdin=archive.stdout,
        check=True,
    )
    archive.stdout.close()
    if archive.wait() != 0:
        raise SystemExit(f"git archive failed for {tag}")


def cmd_extract_content(args):
    for nav_path in sorted(VERSIONS_DIR.glob("v*.yml")):
        version = nav_path.stem[1:]
        if not VERSION_RE.match(version):
            continue
        dest = VERSIONS_DIR / content_dir_name(version)
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)

        tag = f"{TAG_PREFIX}{version}"
        if tag_exists(tag):
            extract_from_tag(tag, dest)
            print(f"Extracted docs for {version} from {tag}.")
        else:
            warn(f"Tag {tag} not found; using the current docs/ tree for {version}.")
            shutil.copytree(
                DOCS_DIR,
                dest,
                ignore=lambda d, names: ["fern"] if Path(d) == DOCS_DIR else [],
                dirs_exist_ok=True,
            )
        self_close_img_tags(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    bump = subparsers.add_parser("bump", help="Add a released version to the docs site.")
    bump.add_argument("version", help="Release version, for example 0.21.0 or core_v0.21.0.")
    bump.add_argument(
        "--ref", help="Git ref to read nightly.yml from. Defaults to the core_vVERSION tag."
    )
    bump.set_defaults(func=cmd_bump)

    extract = subparsers.add_parser(
        "extract-content", help="Extract frozen docs content for every released version."
    )
    extract.set_defaults(func=cmd_extract_content)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
