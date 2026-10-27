#!/usr/bin/env python3
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

"""Report the trusted DCO App verdict as the result of this repository's gate job."""

import json
import os
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

DCO_APP_SLUG = "dco"
DCO_CHECK_NAME = "DCO"
GATE_CHECK_NAME = "DCO gate"
VERIFY_TIMEOUT_SECONDS = 600
VERIFY_POLL_SECONDS = 15
_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class GateError(RuntimeError):
    """Raised when the GitHub response cannot be trusted."""


def _validated_sha(value: object, source: str) -> str:
    if not isinstance(value, str) or not _SHA_PATTERN.fullmatch(value):
        raise GateError(f"{source} has an invalid head SHA")
    return value


def _check_run_id(check_run: dict[str, object]) -> int:
    check_run_id = check_run.get("id")
    if not isinstance(check_run_id, int) or check_run_id <= 0:
        raise GateError("check run has an invalid ID")
    return check_run_id


def select_latest_dco(check_runs: list[dict[str, object]], head_sha: str) -> dict[str, object]:
    """Select the newest completed DCO App check for the exact head SHA.

    Only the DCO App may decide this gate, so a check of the same name published by any
    other app is ignored rather than trusted.
    """

    trusted = []
    for check_run in check_runs:
        app = check_run.get("app")
        app_slug = app.get("slug") if isinstance(app, dict) else None
        if (
            check_run.get("name") == DCO_CHECK_NAME
            and app_slug == DCO_APP_SLUG
            and check_run.get("head_sha") == head_sha
            and check_run.get("status") == "completed"
        ):
            trusted.append(check_run)

    if not trusted:
        raise GateError("no completed DCO App check exists for the requested SHA")
    return max(trusted, key=_check_run_id)


def gate_conclusion(source: dict[str, object]) -> str:
    """Map a DCO App verdict onto the fail-closed conclusion this gate must report."""

    return "success" if source.get("conclusion") == "success" else "failure"


def _request_json(url: str, token: str) -> dict[str, object]:
    request = Request(
        url,
        method="GET",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:  # nosec B310 - URL is fixed to GitHub API.
            result = json.load(response)
    except HTTPError as error:
        detail = error.read().decode(errors="replace")[:1000]
        raise GateError(f"GitHub API returned {error.code}: {detail}") from error
    except (URLError, TimeoutError) as error:
        raise GateError(f"GitHub API request failed: {error}") from error
    if not isinstance(result, dict):
        raise GateError("GitHub API returned a non-object response")
    return result


def _list_check_runs(
    api_url: str, repository: str, head_sha: str, name: str, token: str
) -> list[dict[str, object]]:
    check_runs: list[dict[str, object]] = []
    for page in range(1, 11):
        url = (
            f"{api_url}/repos/{repository}/commits/{head_sha}/check-runs"
            f"?check_name={quote(name)}&filter=all&per_page=100&page={page}"
        )
        response = _request_json(url, token)
        batch = response.get("check_runs")
        if not isinstance(batch, list) or not all(isinstance(item, dict) for item in batch):
            raise GateError("GitHub API returned invalid check-run data")
        check_runs.extend(batch)
        if len(batch) < 100:
            return check_runs
    raise GateError("check-run pagination exceeded the safety limit")


def await_dco_verdict(
    api_url: str,
    repository: str,
    head_sha: str,
    token: str,
    *,
    timeout: float = VERIFY_TIMEOUT_SECONDS,
    poll: float = VERIFY_POLL_SECONDS,
) -> dict[str, object]:
    """Wait for the trusted DCO App to publish a completed verdict for this head.

    The DCO App reports on pull-request events, which precede the mirror-branch push that
    runs this job, so the verdict is normally already there; the wait only covers the case
    where it is not. Running out of time is a hard error, never a pass.
    """

    deadline = time.monotonic() + timeout
    while True:
        check_runs = _list_check_runs(api_url, repository, head_sha, DCO_CHECK_NAME, token)
        try:
            return select_latest_dco(check_runs, head_sha)
        except GateError:
            pass
        if time.monotonic() >= deadline:
            raise GateError(f"no completed DCO App verdict for {head_sha} within {timeout:g}s")
        time.sleep(poll)


def verify(repository: str, api_url: str, token: str) -> int:
    """Report the trusted DCO verdict as this job's own result.

    A job reports into the check suite of the branch GitHub Actions ran it for, which is a
    real branch of this repository. That is why the gate is a job and not a check run
    published through the Checks API: such a run is pinned to the first check suite for its
    head SHA, which for a fork pull request is keyed to a branch this repository does not
    have. A merge box that fails to bind that suite reports the gate as "Expected — waiting
    for status to be reported" indefinitely, and because a check run cannot move between
    suites, neither updating it nor publishing a replacement can repair it.
    """

    head_sha = _validated_sha(os.environ.get("GITHUB_SHA"), "workflow head")
    source = await_dco_verdict(api_url, repository, head_sha, token)
    output = source.get("output")
    summary = output.get("summary") if isinstance(output, dict) else None
    sys.stdout.write(
        f"DCO App reported `{source.get('conclusion')}` for {head_sha} "
        f"({source.get('html_url')})\n{summary or ''}\n"
    )
    if gate_conclusion(source) != "success":
        sys.stderr.write(f"{GATE_CHECK_NAME} failed: commits are not signed off\n")
        return 1
    return 0


def main() -> int:
    """Verify the trusted DCO result for the commit this workflow is running against."""

    try:
        repository = os.environ["GITHUB_REPOSITORY"]
        token = os.environ["GITHUB_TOKEN"]
        api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
        return verify(repository, api_url, token)
    except (GateError, KeyError, OSError, json.JSONDecodeError) as error:
        sys.stderr.write(f"DCO gate failed: {error}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
