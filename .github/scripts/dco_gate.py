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

"""Mirror the trusted DCO App verdict to a repository-owned check."""

import json
import os
import re
import sys
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

DCO_APP_SLUG = "dco"
DCO_CHECK_NAME = "DCO"
GATE_CHECK_NAME = "DCO gate"
_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class GateError(RuntimeError):
    """Raised when the event or GitHub response cannot be trusted."""


def _validated_sha(value: object, source: str) -> str:
    if not isinstance(value, str) or not _SHA_PATTERN.fullmatch(value):
        raise GateError(f"{source} has an invalid head SHA")
    return value


def validate_trigger(payload: dict[str, object], requested_sha: str | None = None) -> str:
    """Return the target SHA after validating the check-run or manual trigger."""

    if requested_sha:
        return _validated_sha(requested_sha, "manual request")

    check_run = payload.get("check_run")
    if not isinstance(check_run, dict):
        raise GateError("trusted DCO check_run payload is missing")

    app = check_run.get("app")
    app_slug = app.get("slug") if isinstance(app, dict) else None
    if check_run.get("name") != DCO_CHECK_NAME or app_slug != DCO_APP_SLUG:
        raise GateError("event did not originate from the trusted DCO App check")
    if check_run.get("status") != "completed":
        raise GateError("DCO check run is not completed")
    return _validated_sha(check_run.get("head_sha"), "DCO check run")


def select_latest_dco(check_runs: list[dict[str, object]], head_sha: str) -> dict[str, object]:
    """Select the newest completed DCO App check for the exact head SHA."""

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


def select_existing_gate(
    check_runs: list[dict[str, object]], head_sha: str
) -> dict[str, object] | None:
    """Find the newest externally identified DCO gate for this SHA."""

    external_id = _gate_external_id(head_sha)
    matching = [
        check_run
        for check_run in check_runs
        if check_run.get("name") == GATE_CHECK_NAME
        and check_run.get("external_id") == external_id
        and check_run.get("head_sha") == head_sha
    ]
    return max(matching, key=_check_run_id) if matching else None


def gate_payload(
    source: dict[str, object], head_sha: str, *, include_head: bool
) -> dict[str, object]:
    """Build a fail-closed create or update request for the mirrored gate."""

    source_conclusion = source.get("conclusion")
    conclusion = "success" if source_conclusion == "success" else "failure"
    source_id = _check_run_id(source)
    source_url = source.get("html_url")
    source_output = source.get("output", {})
    source_summary = source_output.get("summary") if isinstance(source_output, dict) else None

    if isinstance(source_url, str) and source_url.startswith("https://"):
        source_reference = f"[DCO check run {source_id}]({source_url})"
    else:
        source_reference = f"DCO check run {source_id}"
    summary = f"Mirrored `{source_conclusion}` from trusted {source_reference} for `{head_sha}`."
    if isinstance(source_summary, str) and source_summary.strip():
        summary += f"\n\nDCO App result: {source_summary.strip()}"

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload: dict[str, object] = {
        "name": GATE_CHECK_NAME,
        "external_id": _gate_external_id(head_sha),
        "status": "completed",
        "conclusion": conclusion,
        "completed_at": now,
        "output": {"title": GATE_CHECK_NAME, "summary": summary[:65000]},
    }
    if isinstance(source_url, str) and source_url.startswith("https://"):
        payload["details_url"] = source_url
    if include_head:
        payload["head_sha"] = head_sha
    return payload


def _check_run_id(check_run: dict[str, object]) -> int:
    check_run_id = check_run.get("id")
    if not isinstance(check_run_id, int) or check_run_id <= 0:
        raise GateError("check run has an invalid ID")
    return check_run_id


def _gate_external_id(head_sha: str) -> str:
    return f"dco-gate:{head_sha}"


def _request_json(
    method: str, url: str, token: str, payload: dict[str, object] | None = None
) -> dict[str, object]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
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
        response = _request_json("GET", url, token)
        batch = response.get("check_runs")
        if not isinstance(batch, list) or not all(isinstance(item, dict) for item in batch):
            raise GateError("GitHub API returned invalid check-run data")
        check_runs.extend(batch)
        if len(batch) < 100:
            return check_runs
    raise GateError("check-run pagination exceeded the safety limit")


def publish_gate(
    payload: dict[str, object],
    repository: str,
    api_url: str,
    token: str,
    requested_sha: str | None = None,
) -> dict[str, object]:
    """Re-read the current DCO result and publish its repository gate."""

    head_sha = validate_trigger(payload, requested_sha)
    source_runs = _list_check_runs(api_url, repository, head_sha, DCO_CHECK_NAME, token)
    source = select_latest_dco(source_runs, head_sha)
    gate_runs = _list_check_runs(api_url, repository, head_sha, GATE_CHECK_NAME, token)
    existing_gate = select_existing_gate(gate_runs, head_sha)

    if existing_gate is None:
        url = f"{api_url}/repos/{repository}/check-runs"
        return _request_json("POST", url, token, gate_payload(source, head_sha, include_head=True))

    gate_id = _check_run_id(existing_gate)
    url = f"{api_url}/repos/{repository}/check-runs/{gate_id}"
    return _request_json("PATCH", url, token, gate_payload(source, head_sha, include_head=False))


def main() -> int:
    """Publish the mirrored DCO check for the current workflow event."""

    try:
        event_path = os.environ["GITHUB_EVENT_PATH"]
        repository = os.environ["GITHUB_REPOSITORY"]
        token = os.environ["GITHUB_TOKEN"]
        api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
        requested_sha = os.environ.get("DCO_GATE_SHA") or None
        with open(event_path, encoding="utf-8") as event_file:
            payload = json.load(event_file)
        if not isinstance(payload, dict):
            raise GateError("event payload is not an object")
        result = publish_gate(payload, repository, api_url, token, requested_sha)
        sys.stdout.write(f"Published {GATE_CHECK_NAME} check run {result.get('id')}\n")
    except (GateError, KeyError, OSError, json.JSONDecodeError) as error:
        sys.stderr.write(f"DCO gate failed: {error}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
