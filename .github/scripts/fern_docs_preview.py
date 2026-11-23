# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve synchronized PR revisions and publish Fern preview comments from trusted CI."""

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

REQUEST_WORKFLOW = "fern-docs-preview-request.yml"
MARKER = "<!-- megatron-fern-preview -->"
SHA = re.compile(r"[0-9a-f]{40}")
RELEVANT = re.compile(
    r"^(docs/|fern/|\.github/actions/(setup-fern|preview-fern)/|"
    r"\.github/scripts/(fern_docs_preview|test_fern_docs_preview)|"
    r"\.github/workflows/fern-docs-preview)"
)


class GitHub:
    """Use the runner's authenticated GitHub CLI, without additional Python dependencies."""

    def __init__(self, repository: str):
        self.prefix = f"repos/{repository}"

    def get(self, endpoint: str):
        result = subprocess.run(
            ["gh", "api", f"{self.prefix}/{endpoint}"], check=True, capture_output=True, text=True
        )
        return json.loads(result.stdout)

    def paginate(self, endpoint: str):
        result = subprocess.run(
            ["gh", "api", "--paginate", "--slurp", f"{self.prefix}/{endpoint}?per_page=100"],
            check=True,
            capture_output=True,
            text=True,
        )
        return [item for page in json.loads(result.stdout) for item in page]

    def write(self, method: str, endpoint: str, body: str):
        subprocess.run(
            ["gh", "api", "--method", method, f"{self.prefix}/{endpoint}", "--input", "-"],
            input=json.dumps({"body": body}),
            check=True,
            capture_output=True,
            text=True,
        )


@dataclass(frozen=True)
class PreviewRequest:
    """The original fork head and the synchronized revision to preview."""

    number: int
    source_sha: str
    head_sha: str


def resolve(github: GitHub, repository: str, run_id: int) -> PreviewRequest | None:
    """Resolve metadata directly from GitHub; never trust requester artifacts or outputs."""
    run = github.get(f"actions/runs/{run_id}")
    workflow = github.get(f"actions/workflows/{REQUEST_WORKFLOW}")
    branch = re.fullmatch(r"pull-request/([1-9][0-9]*)", run.get("head_branch") or "")
    if (
        run["workflow_id"] != workflow["id"]
        or run["event"] != "push"
        or run["status"] != "completed"
        or run["conclusion"] != "success"
        or (run.get("head_repository") or {}).get("full_name") != repository
        or not branch
        or not SHA.fullmatch(run["head_sha"])
    ):
        return None
    number = int(branch[1])
    pr = github.get(f"pulls/{number}")
    if (
        pr["state"] != "open"
        or pr["draft"]
        or pr["base"]["repo"]["full_name"] != repository
        or not SHA.fullmatch(pr["head"]["sha"])
    ):
        return None
    ref = github.get(f"git/ref/heads/{run['head_branch']}")
    if ref["object"]["sha"] != run["head_sha"]:
        return None
    # The synchronized revision may merge the fork head with its base.
    comparison = github.get(f"compare/{pr['head']['sha']}...{run['head_sha']}")
    if (
        comparison["status"] not in ("identical", "ahead")
        or comparison["merge_base_commit"]["sha"] != pr["head"]["sha"]
    ):
        return None
    files = github.paginate(f"pulls/{number}/files")
    if not any(
        RELEVANT.match(item["filename"]) or RELEVANT.match(item.get("previous_filename", ""))
        for item in files
    ):
        return None
    return PreviewRequest(number, run["head_sha"], pr["head"]["sha"])


def prepare(root: Path = Path(".")) -> None:
    """Treat PR files as content and retain the trusted CLI pin and organization."""
    source = root / "preview-source"

    def check_tree(entry: Path) -> None:
        if entry.is_symlink():
            raise ValueError(f"Preview sources contain a symlink: {entry}")
        if entry.is_dir():
            for child in entry.iterdir():
                check_tree(child)

    for name in ("docs", "fern"):
        check_tree(source / name)
    shutil.copyfile(root / "fern/fern.config.json", source / "fern/fern.config.json")


def comment(github: GitHub, request: PreviewRequest, url: str) -> None:
    """Create or update only the Actions bot's preview comment."""
    body = (
        f"{MARKER}\nPreview the documentation: <{url}>\n\n"
        f"PR head: `{request.head_sha}`\nSynchronized revision: `{request.source_sha}`"
    )
    comments = github.paginate(f"issues/{request.number}/comments")
    existing = next(
        (
            item
            for item in comments
            if item["user"]["login"] == "github-actions[bot]" and MARKER in (item.get("body") or "")
        ),
        None,
    )
    if existing:
        github.write("PATCH", f"issues/comments/{existing['id']}", body)
    else:
        github.write("POST", f"issues/{request.number}/comments", body)


def preview_url(log: str) -> str:
    """Extract a Fern HTTPS preview URL before exposing it in comments or outputs."""
    match = re.search(r"Published docs to (https://\S+)", log)
    if not match:
        raise ValueError("Fern did not report a published preview URL")
    url = urlsplit(match[1])
    if (
        url.scheme != "https"
        or url.username
        or url.password
        or not (url.hostname or "").endswith(".buildwithfern.com")
    ):
        raise ValueError("Fern returned an unexpected preview URL")
    # Accessing port also rejects invalid port syntax, as the original URL parser did.
    _ = url.port
    return url.geturl()


def output(name: str, value) -> None:
    """Append a single-line value to this step's outputs."""
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        stream.write(f"{name}={value}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("resolve", "prepare", "current", "comment", "url"))
    parser.add_argument("log_path", nargs="?", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
        return
    if args.command == "url":
        if args.log_path is None:
            parser.error("url requires a log path")
        print(preview_url(args.log_path.read_text()))
        return

    repository = os.environ["GITHUB_REPOSITORY"]
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    github = GitHub(repository)
    request = resolve(github, repository, int(event["workflow_run"]["id"]))
    if args.command == "resolve":
        if request is None:
            print("Skipping a stale, ineligible, or non-documentation preview request.")
            return
        output("pr-number", request.number)
        output("source-sha", request.source_sha)
        output("head-sha", request.head_sha)
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
            stream.write(
                f"## Fern preview for PR #{request.number}\n\n"
                f"PR head: {request.head_sha}\n\nSynchronized revision: {request.source_sha}\n"
            )
    else:
        current = request is not None and request.head_sha == os.environ["EXPECTED_HEAD_SHA"]
        if args.command == "current":
            output("current", str(current).lower())
        elif current:
            comment(github, request, os.environ["PREVIEW_URL"])
        else:
            print("The PR changed during publishing; leaving its preview comment unchanged.")


if __name__ == "__main__":
    main()
