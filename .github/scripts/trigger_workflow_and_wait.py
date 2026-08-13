#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trigger or poll a GitHub workflow using a downscoped installation token."""

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Callable

API_VERSION = "2022-11-28"
JsonObject = dict[str, object]
Request = Callable[[str, str, str, JsonObject | None], JsonObject | None]


class GitHubApiError(RuntimeError):
    """A failed GitHub API request."""

    def __init__(self, status: int, body: str):
        super().__init__(f"GitHub API request failed with HTTP {status}: {body}")
        self.status = status


def request_json(
    token: str, method: str, url: str, payload: JsonObject | None = None
) -> JsonObject | None:
    """Send one authenticated GitHub API request and decode its JSON response."""
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": API_VERSION,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        raise GitHubApiError(error.code, error.read().decode(errors="replace")) from error
    return json.loads(body) if body else None


def append_output(name: str, value: str) -> None:
    """Write one GitHub Actions step output when running in Actions."""
    if output_path := os.getenv("GITHUB_OUTPUT"):
        with open(output_path, "a", encoding="utf-8") as output_file:
            output_file.write(f"{name}={value}\n")


def workflow_runs_url(
    api_url: str, owner: str, repo: str, workflow: str, ref: str, created_after: str
) -> str:
    """Build the URL used to discover a newly dispatched workflow run."""
    query = urllib.parse.urlencode(
        {
            "event": "workflow_dispatch",
            "branch": ref,
            "created": f">={created_after}",
            "per_page": 100,
        }
    )
    workflow = urllib.parse.quote(workflow, safe="")
    return f"{api_url}/repos/{owner}/{repo}/actions/workflows/{workflow}/runs?{query}"


def trigger_workflow(
    token: str,
    api_url: str,
    owner: str,
    repo: str,
    workflow: str,
    ref: str,
    client_payload: JsonObject,
    wait_interval: int,
    discovery_timeout: int,
    request: Request = request_json,
) -> int:
    """Dispatch a workflow and return the newly created run ID."""
    started_at = time.time()
    created_after = (
        datetime.fromtimestamp(started_at - 120).astimezone().isoformat(timespec="seconds")
    )
    runs_url = workflow_runs_url(api_url, owner, repo, workflow, ref, created_after)
    old_runs_response = request(token, "GET", runs_url, None)
    assert old_runs_response is not None
    old_runs = old_runs_response["workflow_runs"]
    assert isinstance(old_runs, list)
    old_run_ids = {run["id"] for run in old_runs}

    workflow_path = urllib.parse.quote(workflow, safe="")
    dispatch_url = f"{api_url}/repos/{owner}/{repo}/actions/workflows/{workflow_path}/dispatches"
    request(token, "POST", dispatch_url, {"ref": ref, "inputs": client_payload})

    deadline = time.monotonic() + discovery_timeout
    while time.monotonic() < deadline:
        runs_response = request(token, "GET", runs_url, None)
        assert runs_response is not None
        runs = runs_response["workflow_runs"]
        assert isinstance(runs, list)
        new_runs = [run for run in runs if run["id"] not in old_run_ids]
        if new_runs:
            run = min(new_runs, key=lambda candidate: candidate["created_at"])
            return int(run["id"])
        time.sleep(wait_interval)
    raise TimeoutError("Timed out waiting for the dispatched workflow run to appear")


def poll_workflow(
    token: str,
    api_url: str,
    owner: str,
    repo: str,
    run_id: int,
    wait_interval: int,
    poll_timeout: int,
    request: Request = request_json,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[bool, str | None]:
    """Poll one workflow for a bounded interval using one installation token."""
    run_url = f"{api_url}/repos/{owner}/{repo}/actions/runs/{run_id}"
    deadline = clock() + poll_timeout
    while True:
        run = request(token, "GET", run_url, None)
        assert run is not None
        status = str(run["status"])
        conclusion = run["conclusion"]
        print(f"Workflow status={status} conclusion={conclusion}")
        if status == "completed":
            return True, None if conclusion is None else str(conclusion)

        remaining = deadline - clock()
        if remaining <= 0:
            return False, None
        sleep(min(wait_interval, remaining))


def validate_workflow_result(completed: bool, conclusion: str | None, run_id: int) -> None:
    """Fail unless the downstream workflow completed successfully."""
    if not completed:
        raise TimeoutError(f"Timed out waiting for downstream workflow run {run_id} to complete")
    if conclusion != "success":
        raise RuntimeError(f"Downstream workflow concluded with {conclusion}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--wait-interval", type=int, default=60)
    subparsers = parser.add_subparsers(dest="command", required=True)

    trigger = subparsers.add_parser("trigger")
    trigger.add_argument("--workflow", required=True)
    trigger.add_argument("--ref", required=True)
    trigger.add_argument("--client-payload", default="{}")
    trigger.add_argument("--discovery-timeout", type=int, default=10 * 60)

    poll = subparsers.add_parser("poll")
    poll.add_argument("--run-id", type=int, required=True)
    poll.add_argument("--poll-timeout", type=int, default=35 * 60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ["GH_TOKEN"]
    api_url = os.getenv("GITHUB_API_URL", "https://api.github.com").rstrip("/")

    if args.command == "trigger":
        client_payload = json.loads(args.client_payload)
        if not isinstance(client_payload, dict):
            raise ValueError("--client-payload must be a JSON object")
        run_id = trigger_workflow(
            token,
            api_url,
            args.owner,
            args.repo,
            args.workflow,
            args.ref,
            client_payload,
            args.wait_interval,
            args.discovery_timeout,
        )
        append_output("workflow_id", str(run_id))
        print(f"Triggered workflow run {run_id}")
        return

    completed, conclusion = poll_workflow(
        token, api_url, args.owner, args.repo, args.run_id, args.wait_interval, args.poll_timeout
    )
    append_output("completed", str(completed).lower())
    if conclusion is not None:
        append_output("conclusion", conclusion)
    validate_workflow_result(completed, conclusion, args.run_id)


if __name__ == "__main__":
    main()
